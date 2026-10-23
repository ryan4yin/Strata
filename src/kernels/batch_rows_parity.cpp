// src/kernels/batch_rows_parity.cpp - the batch window's one-launch-for-all-rows kernels (#2234, STRATA_BATCH_ATTN_ROWS /
// STRATA_BATCH_GDN_GROUPS) against the per-row launches they replace, bitwise (GPU, synthetic, no model).
//
//   1. qsa_block_scores_rows   == qsa_block_scores, one call per row, each row with its OWN indexer keys;
//   2. qsa_decode_attn_batch_rows == qsa_decode_attn_batch, one call per row, each row over its OWN K/V pools
//      (int8 codes + scales, and fp16);
//   3. gdn_step_norm_groups    == gdn_step_norm_multi, one call per slot group, in the window (all outputs, the state
//      untouched) and in the 1-row commit (the state written after `keep` tokens).
//   4. STRATA_FUSE_BCOMMIT: the batch commit's per-slot gdn_commit_layers == capture_commit_batch's per-layer, per-slot
//      gdn_conv_commit + gdn_step_norm_multi (a 1-row group: the 0.1.39 kernel), and its per-slot
//      native_qsa_indexer_append_steps_layers == the per-layer tail copy_from_mapped + one native_qsa_indexer_append per row.
//   5. STRATA_FUSE_GDNCONV: gdn_conv_groups == gdn_conv_l2_multi per slot group (+ gdn_ab_multi over all rows), the
//      window (history read only) and the 1-row self-commit (history written).
//   6. STRATA_FUSE_IDXAPP: native_qsa_indexer_append_slots == per slot group the tail snapshot (copy_from_mapped) and
//      one native_qsa_indexer_append per row (positions at the window's kStepCount stride).
//   7. STRATA_FUSE_ATTNMERGE: per row qsa_decode_attn_batch_chunks, then one qsa_decode_attn_batch_merge over the rows ==
//      qsa_decode_attn_batch per row (in attn_case); STRATA_FUSE_KVROWS: kv_append_q8_rows == kv_append_q8_step per row,
//      each row its own pools, page table (non-resident pages) and host copy.
// Every case compares with memcmp, not a tolerance. Usage: batch_rows_parity
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int bad = 0;
void report(const char* what, bool same) {
    std::printf("  %-64s %s\n", what, same ? "bitwise" : "*** DIFFERS ***");
    if (!same) ++bad;
}

// ---- 1. the block scores ----
void scores_case(int rows) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    std::mt19937 rng(11 + rows);
    std::normal_distribution<float> nd(0.f, 1.f);
    k::QsaScoreRows sr;
    sr.n = rows;
    std::vector<const float*> pooled(rows), dead(rows);
    std::vector<float> q((size_t) rows * s.idx_n_head * s.idx_dim);
    for (auto& x : q) x = nd(rng);
    std::vector<int32_t> steps((size_t) rows * k::kStepCount);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> p((size_t) max_blocks * s.idx_dim), d((size_t) s.idx_dim);
        for (auto& x : p) x = nd(rng);
        for (auto& x : d) x = nd(rng);
        pooled[r] = up(p);
        dead[r] = up(d);
        sr.pooled[r] = pooled[r];
        sr.dead[r] = dead[r];
        const int64_t pos = 700 + 997 * r + (r % 3) * 5;   // different lengths, and an incomplete block at the tail
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) pos;
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) (pos + 1);
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *d_rows = nullptr, *d_ref = nullptr;
    const size_t n = (size_t) rows * max_blocks;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_block_scores_rows(sr, d_q, d_steps, rows, max_blocks, s, d_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_scores(pooled[r], dead[r], d_q + (size_t) r * s.idx_n_head * s.idx_dim, d_steps + (size_t) r * k::kStepCount, 1,
                            max_blocks, s, d_ref + (size_t) r * max_blocks, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_block_scores_rows, %d rows (own keys each)", rows);
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
    // STRATA_FUSE_QSASEL: the streamed window's one scores launch over the rows (qsa_block_scores_slots)
    {
        float* d_sl = nullptr;
        ck(cudaMalloc(&d_sl, n * 4), "malloc");
        ck(cudaMemset(d_sl, 0, n * 4), "memset");
        const bool ran = k::qsa_block_scores_slots(pooled.data(), dead.data(), d_q, d_steps, rows, max_blocks, s, d_sl, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        const std::vector<float> c = down(d_sl, n);
        std::snprintf(name, sizeof name, "qsa_block_scores_slots, %d rows (own keys each)%s", rows, ran ? "" : " (declined)");
        report(name, ran && std::memcmp(c.data(), b.data(), n * 4) == 0);
        cudaFree(d_sl);
    }
    // the window selects with one top-k launch over all rows (row-independent) where the per-row path ran one per row
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    int32_t *i_rows = nullptr, *i_ref = nullptr;
    ck(cudaMalloc(&i_rows, (size_t) rows * cap * 4), "malloc");
    ck(cudaMalloc(&i_ref, (size_t) rows * cap * 4), "malloc");
    ck(cudaMemset(i_rows, 0, (size_t) rows * cap * 4), "memset");
    ck(cudaMemset(i_ref, 0, (size_t) rows * cap * 4), "memset");
    k::qsa_block_topk(d_ref, d_steps, rows, max_blocks, cap, s, i_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_topk(d_ref + (size_t) r * max_blocks, d_steps + (size_t) r * k::kStepCount, 1, max_blocks, cap, s,
                          i_ref + (size_t) r * cap, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<int32_t> ta = down(i_rows, (size_t) rows * cap), tb = down(i_ref, (size_t) rows * cap);
    bool tsame = true;
    for (int r = 0; r < rows; ++r) {
        const int64_t w = steps[(size_t) r * k::kStepCount + k::kStepWidth];
        tsame = tsame && std::memcmp(ta.data() + (size_t) r * cap, tb.data() + (size_t) r * cap, (size_t) w * 4) == 0;
    }
    std::snprintf(name, sizeof name, "qsa_block_topk over %d rows == one call per row", rows);
    report(name, tsame);
}

// ---- 2. the decode attention ----
void attn_case(int rows, bool int8) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t ctx_max = 5000, pages = (ctx_max + PS - 1) / PS, prow = pages * NKV * PS;
    const int64_t cap = k::qsa_selection_width(ctx_max, s);
    std::mt19937 rng(23 + rows + int8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    k::QsaAttnPoolsRows pr;
    pr.n = rows;
    std::vector<float> q((size_t) rows * NH * HD);
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> ids((size_t) rows * cap, 0), steps((size_t) rows * k::kStepCount, 0);
    for (int r = 0; r < rows; ++r) {
        k::QsaAttnPools& pl = pr.p[r];
        if (int8) {
            std::vector<int8_t> kq((size_t) prow * HD), vq((size_t) prow * HD);
            std::vector<uint16_t> ks((size_t) prow * (HD / 64)), vs(ks.size());
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
            for (auto& x : vs) x = f2h(sc(rng));
            pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs);
        } else {
            std::vector<uint16_t> kh((size_t) prow * HD), vh(kh.size());
            for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
            for (auto& x : vh) x = f2h(nd(rng));
            pl.k_pool = up(kh); pl.v_pool = up(vh);
        }
        std::vector<int32_t> table(pages);
        for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
        std::shuffle(table.begin(), table.end(), rng);   // each row's pool has its own page order
        pl.page_table = up(table);
        const int64_t nkv = 900 + 641 * r + (r % 2) * 3;
        const int64_t w = k::qsa_selection_width(nkv, s);
        std::vector<int32_t> all((size_t) nkv);
        for (int64_t c = 0; c < nkv; ++c) all[c] = (int32_t) c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + w);
        std::copy(all.begin(), all.begin() + w, ids.begin() + (size_t) r * cap);
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) (nkv - 1);
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) nkv;
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) (nkv / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) w;
    }
    const float* d_q = up(q);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const int64_t scr = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float *scratch = nullptr, *d_rows = nullptr, *d_ref = nullptr;
    ck(cudaMalloc(&scratch, (size_t) rows * scr * 4), "malloc");
    const size_t n = (size_t) rows * NH * HD;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_decode_attn_batch_rows(d_q, pr, d_ids, d_steps, cap, s, scratch, d_rows, rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_decode_attn_batch(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap, d_steps + (size_t) r * k::kStepCount, cap,
                                 s, scratch + (size_t) r * scr, d_ref + (size_t) r * NH * HD, 1, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    {   // STRATA_FUSE_ATTNMERGE: each row's chunk pass alone, then one merge over every row's partials
        float *scratch2 = nullptr, *d_late = nullptr;
        ck(cudaMalloc(&scratch2, (size_t) rows * scr * 4), "malloc");
        ck(cudaMalloc(&d_late, n * 4), "malloc");
        ck(cudaMemset(d_late, 0, n * 4), "memset");
        for (int r = 0; r < rows; ++r)
            k::qsa_decode_attn_batch_chunks(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap,
                                            d_steps + (size_t) r * k::kStepCount, cap, s, scratch2 + (size_t) r * scr, 1,
                                            nullptr);
        k::qsa_decode_attn_batch_merge(d_steps, cap, s, scratch2, d_late, rows, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        const std::vector<float> a = down(d_late, n), b = down(d_ref, n);
        char nm[96];
        std::snprintf(nm, sizeof nm, "attn chunks per row + one merge, %d rows, %s pools", rows, int8 ? "int8" : "fp16");
        report(nm, std::memcmp(a.data(), b.data(), n * 4) == 0);
        // STRATA_FUSE_QSAWAVE: the chunk passes by waves of non-adjacent rows (even rows, then odd), one launch each
        ck(cudaMemset(scratch2, 0, (size_t) rows * scr * 4), "memset");
        ck(cudaMemset(d_late, 0, n * 4), "memset");
        bool ran = true;
        for (int par = 0; par < 2; ++par) {
            k::QsaAttnWave wv;
            for (int r = par; r < rows; r += 2) { wv.p[wv.n] = pr.p[r]; wv.row[wv.n++] = r; }
            ran = k::qsa_decode_attn_wave_chunks(d_q, wv, d_ids, d_steps, cap, s, scratch2, nullptr) && ran;
        }
        k::qsa_decode_attn_batch_merge(d_steps, cap, s, scratch2, d_late, rows, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        const std::vector<float> w = down(d_late, n);
        if (ran) {
            std::snprintf(nm, sizeof nm, "attn wave chunks (even, odd rows) + one merge, %d rows, %s", rows, int8 ? "int8" : "fp16");
            report(nm, std::memcmp(w.data(), b.data(), n * 4) == 0);
        } else
            std::printf("  attn wave chunks: declined (this card does not run attn_chunk_kernel_pre75)\n");
        cudaFree(scratch2);
        cudaFree(d_late);
    }
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_decode_attn_batch_rows, %d rows, %s pools (own pools each)", rows, int8 ? "int8" : "fp16");
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
}

// ---- 3. the GDN recurrence ----
void gdn_case(const std::vector<int>& ntok, bool commit) {
    const int S = 128, h_k = 16, h_v = 48;   // the artifact's geometry
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const float eps = 1e-6f;
    int T = 0;
    for (int n : ntok) T += n;
    const int G = (int) ntok.size();
    std::mt19937 rng(31 + T * 7 + G + commit);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd), gamma(S);
    for (auto& v : h) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.0f + 0.1f * nd(rng);
    const size_t st_n = (size_t) S * h_v * S;
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> st_g(G), st_r(G);
    std::vector<int32_t*> keep(G);
    k::GdnGroups gg;
    gg.n = G;
    int first = 0;
    for (int i = 0; i < G; ++i) {
        st0[i].resize(st_n);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        st_g[i] = up(st0[i]);
        st_r[i] = up(st0[i]);
        const int32_t kp = commit ? ntok[i] : 0;   // a commit group keeps all of its rows here
        keep[i] = up(std::vector<int32_t>{kp});
        gg.state[i] = st_g[i];
        gg.keep[i] = commit ? keep[i] : nullptr;
        gg.first[i] = first;
        gg.ntok[i] = ntok[i];
        first += ntok[i];
    }
    const float* d_h = up(h);
    const float* d_gate = up(gate);
    const float* d_beta = up(beta);
    const float* d_z = up(z);
    const float* d_gm = up(gamma);
    float *d_y = nullptr, *d_yr = nullptr;
    ck(cudaMalloc(&d_y, (size_t) T * vd * 4), "malloc");
    ck(cudaMalloc(&d_yr, (size_t) T * vd * 4), "malloc");
    ck(cudaMemset(d_y, 0, (size_t) T * vd * 4), "memset");
    ck(cudaMemset(d_yr, 0, (size_t) T * vd * 4), "memset");
    k::gdn_step_norm_groups(gg, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, commit, nullptr);
    for (int i = 0; i < G; ++i) {
        const int f = gg.first[i];
        // the engine's per-slot call: a window group runs its rows with all outputs; a commit group (1 row) keeps them
        k::gdn_step_norm_multi(st_r[i], d_h + (size_t) f * C, C, d_gate + (size_t) f * h_v, d_beta + (size_t) f * h_v,
                               d_z + (size_t) f * vd, d_gm, eps, d_yr + (size_t) f * vd, h_k, h_v, ntok[i],
                               commit ? keep[i] : nullptr, nullptr, 0);
    }
    ck(cudaDeviceSynchronize(), "sync");
    bool same = true;
    for (int i = 0; i < G; ++i) {
        const std::vector<float> a = down(st_g[i], st_n), b = down(st_r[i], st_n);
        same = same && std::memcmp(a.data(), b.data(), st_n * 4) == 0;
        if (!commit) same = same && std::memcmp(a.data(), st0[i].data(), st_n * 4) == 0;   // a window leaves the state alone
    }
    if (!commit) {
        const std::vector<float> a = down(d_y, (size_t) T * vd), b = down(d_yr, (size_t) T * vd);
        same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G && i < 8; ++i) std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_step_norm_groups %s, groups of %s rows", commit ? "commit" : "window", rows);
    report(name, same);
}
// ---- 4. the batch commit graph, fused per slot (STRATA_FUSE_BCOMMIT) ----
// slot groups of ntok[i] rows (rows contiguous, slot i's first row = the sum before it), each keeping keep[i] of them
void bcommit_gdn_case(const std::vector<int>& ntok, const std::vector<int>& keepn) {
    const int S = 128, h_k = 16, h_v = 48, L = 3, MT = k::kVerifyMaxT;
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const int64_t conv_off = (int64_t) S * h_v * S, stride = conv_off + (int64_t) C * 3;
    const int G = (int) ntok.size();
    std::mt19937 rng(71 + G * 13 + ntok[0] * 3 + keepn[0]);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qkv((size_t) L * MT * C), h((size_t) L * MT * C), gate((size_t) L * MT * h_v),
        beta((size_t) L * MT * h_v), gamma(S, 1.0f);
    for (auto& v : qkv) v = nd(rng);
    for (auto& v : h) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    const float *d_qkv = up(qkv), *d_h = up(h), *d_g = up(gate), *d_b = up(beta), *d_gm = up(gamma);
    float *d_z = nullptr, *d_y = nullptr;
    ck(cudaMalloc(&d_z, (size_t) MT * vd * 4), "malloc");
    ck(cudaMalloc(&d_y, (size_t) MT * vd * 4), "malloc");
    ck(cudaMemset(d_z, 0, (size_t) MT * vd * 4), "memset");
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> ref(G), fus(G);
    std::vector<int32_t*> keep(G);
    for (int i = 0; i < G; ++i) {
        st0[i].resize((size_t) stride * L);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        ref[i] = up(st0[i]);
        fus[i] = up(st0[i]);
        keep[i] = up(std::vector<int32_t>{keepn[i]});
    }
    // the reference: capture_commit_batch's loop, layer by layer, slot by slot
    for (int l = 0; l < L; ++l)
        for (int i = 0, first = 0; i < G; first += ntok[i], ++i) {
            float* stl = ref[i] + (size_t) l * stride;
            k::gdn_conv_commit(stl + conv_off, d_qkv + (size_t) l * MT * C + (size_t) first * C, C, keep[i], nullptr);
            k::gdn_step_norm_multi(stl, d_h + (size_t) l * MT * C + (size_t) first * C, C,
                                   d_g + (size_t) l * MT * h_v + (size_t) first * h_v,
                                   d_b + (size_t) l * MT * h_v + (size_t) first * h_v, d_z + (size_t) first * vd, d_gm,
                                   1e-6f, d_y + (size_t) first * vd, h_k, h_v, ntok[i], keep[i], nullptr,
                                   ntok[i] > 1 ? ntok[i] : 0);
        }
    bool launched = true;
    for (int i = 0, first = 0; i < G; first += ntok[i], ++i)
        launched = launched && k::gdn_commit_layers(fus[i], stride, conv_off, d_qkv + (size_t) first * C, (int64_t) MT * C,
                                                    d_h + (size_t) first * C, (int64_t) MT * C, d_g + (size_t) first * h_v,
                                                    d_b + (size_t) first * h_v, (int64_t) MT * h_v, C, h_k, h_v, ntok[i], L,
                                                    keep[i], nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    bool same = launched;
    for (int i = 0; i < G; ++i) {
        const std::vector<float> a = down(ref[i], st0[i].size()), b = down(fus[i], st0[i].size());
        same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        if (keepn[i] > 0) same = same && std::memcmp(a.data(), st0[i].data(), a.size() * 4) != 0;   // it did commit
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G; ++i)
        std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d/%d", i ? "," : "", keepn[i], ntok[i]);
    std::snprintf(name, sizeof name, "bcommit gdn_commit_layers per slot, keep/rows %s%s", rows, launched ? "" : " (declined)");
    report(name, same);
}

void bcommit_qsa_case(const std::vector<int>& ntok, const std::vector<std::vector<int32_t>>& pos) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int L = 3, MT = k::kVerifyMaxT, ID = (int) s.idx_dim, G = (int) ntok.size();
    const int64_t MC = 1024, TS = (s.idx_block - 1) * ID, PR = (MC / s.idx_block + 1) * ID;
    std::mt19937 rng(91 + G * 5 + ntok[0]);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    std::vector<float> raw((size_t) L * MT * ID), gam((size_t) L * ID);
    for (auto& v : raw) v = nd(rng) * 2.0f;
    for (auto& v : gam) v = 1.0f + 0.1f * nd(rng);
    const float *d_raw = up(raw), *d_gam = up(gam);
    struct Buf { float *tail, *dead, *pooled; int32_t* bpos; };
    auto mk = [&](const std::vector<float>& tail0, const std::vector<float>& pooled0, const std::vector<float>& dead0) {
        return Buf{up(tail0), up(dead0), up(pooled0), up(std::vector<int32_t>{7})};
    };
    std::vector<std::vector<Buf>> ref(G), fus(G);
    std::vector<std::vector<const float*>> snap(G);
    std::vector<const int32_t*> d_pos(G);
    for (int i = 0; i < G; ++i) {
        d_pos[i] = up(pos[i]);
        for (int l = 0; l < L; ++l) {
            std::vector<float> t0((size_t) TS), p0((size_t) PR), dd((size_t) ID), sn((size_t) TS);
            for (auto& v : t0) v = nd(rng);
            for (auto& v : p0) v = nd(rng);
            for (auto& v : dd) v = nd(rng);
            for (auto& v : sn) v = nd(rng);
            ref[i].push_back(mk(t0, p0, dd));
            fus[i].push_back(mk(t0, p0, dd));
            snap[i].push_back(up(sn));
        }
    }
    const k::RopeScaling sc;
    for (int l = 0; l < L; ++l)   // the reference: capture_commit_batch's loop
        for (int i = 0, first = 0; i < G; first += ntok[i], ++i) {
            const Buf& b = ref[i][(size_t) l];
            k::copy_from_mapped(b.tail, snap[i][(size_t) l], TS, cs);
            const k::QsaIndexerBuffers ib{b.tail, b.dead, b.pooled, b.bpos};
            for (int u = 0; u < ntok[i]; ++u)
                k::native_qsa_indexer_append(d_raw + (size_t) (l * MT + first + u) * ID, d_pos[i] + u, 0,
                                             d_gam + (size_t) l * ID, 1e-6f, ib, s, MC, sc, cs);
        }
    for (int i = 0, first = 0; i < G; first += ntok[i], ++i) {
        k::QsaIndexerLayers il;
        for (int l = 0; l < L; ++l) {
            const Buf& b = fus[i][(size_t) l];
            il.gamma[l] = d_gam + (size_t) l * ID;
            il.tail_src[l] = snap[i][(size_t) l];
            il.tail[l] = b.tail; il.dead[l] = b.dead; il.pooled[l] = b.pooled; il.block_pos[l] = b.bpos;
        }
        k::native_qsa_indexer_append_steps_layers(d_raw + (size_t) first * ID, (int64_t) MT * ID, d_pos[i], 1, ntok[i], 0,
                                                  1e-6f, il, L, s, MC, sc, cs);
    }
    ck(cudaStreamSynchronize(cs), "sync");
    bool same = true;
    for (int i = 0; i < G; ++i)
        for (int l = 0; l < L; ++l) {
            const Buf &a = ref[i][(size_t) l], &b = fus[i][(size_t) l];
            auto eq = [&](const float* x, const float* y, int64_t n) {
                const std::vector<float> hx = down(x, (size_t) n), hy = down(y, (size_t) n);
                return std::memcmp(hx.data(), hy.data(), (size_t) n * 4) == 0;
            };
            same = same && eq(a.tail, b.tail, TS) && eq(a.dead, b.dead, ID) && eq(a.pooled, b.pooled, PR) &&
                   down(a.bpos, 1)[0] == down(b.bpos, 1)[0];
        }
    ck(cudaStreamDestroy(cs), "stream");
    char name[96], rows[64] = "";
    for (int i = 0; i < G; ++i)
        std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d@%d", i ? "," : "", ntok[i], pos[i][0]);
    std::snprintf(name, sizeof name, "bcommit indexer append_steps_layers per slot, rows@pos %s", rows);
    report(name, same);
}
// ---- 5. the window's conv of every slot group in one launch (STRATA_FUSE_GDNCONV) ----
void conv_groups_case(const std::vector<int>& ntok, bool commit, bool ab) {
    const int S = 128, h_k = 16, h_v = 48, n_embd = 2048;
    const int C = 2 * S * h_k + S * h_v;
    const int G = (int) ntok.size();
    int T = 0;
    for (int n : ntok) T += n;
    std::mt19937 rng(131 + T * 3 + G + commit * 7 + ab);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bf = [&](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); };
    std::vector<float> qkv((size_t) T * C), w((size_t) C * 4), x((size_t) T * n_embd), dt(h_v), sa(h_v);
    std::vector<uint16_t> wa((size_t) h_v * n_embd), wb((size_t) h_v * n_embd);
    for (auto& v : qkv) v = nd(rng);
    for (auto& v : w) v = nd(rng) * 0.5f;
    for (auto& v : x) v = nd(rng);
    for (auto& v : dt) v = nd(rng);
    for (auto& v : sa) v = -std::fabs(nd(rng));
    for (auto& v : wa) v = bf(nd(rng) * 0.05f);
    for (auto& v : wb) v = bf(nd(rng) * 0.05f);
    const float *d_qkv = up(qkv), *d_w = up(w), *d_x = up(x), *d_dt = up(dt), *d_sa = up(sa);
    const uint16_t *d_wa = up(wa), *d_wb = up(wb);
    std::vector<std::vector<float>> h0(G);
    std::vector<float*> hr(G), hf(G);
    k::GdnConvGroups cg;
    for (int i = 0, first = 0; i < G; first += ntok[i], ++i) {
        h0[i].resize((size_t) C * 3);
        for (auto& v : h0[i]) v = nd(rng);
        hr[i] = up(h0[i]);
        hf[i] = up(h0[i]);
        cg.hist[i] = hf[i];
        cg.first[i] = first;
        cg.ntok[i] = ntok[i];
    }
    cg.n = G;
    auto zeros = [](size_t n) { float* d = nullptr; ck(cudaMalloc(&d, n * 4), "malloc"); ck(cudaMemset(d, 0, n * 4), "memset"); return d; };
    float *d_hr = zeros((size_t) T * C), *d_hf = zeros((size_t) T * C), *d_gr = zeros((size_t) T * h_v),
          *d_gf = zeros((size_t) T * h_v), *d_br = zeros((size_t) T * h_v), *d_bf = zeros((size_t) T * h_v);
    for (int i = 0; i < G; ++i)   // the reference: the window's per-group calls, then the a/b over all rows
        k::gdn_conv_l2_multi(hr[i], d_qkv + (size_t) cg.first[i] * C, d_w, d_hr + (size_t) cg.first[i] * C, C, 2 * h_k, 1e-6f,
                             ntok[i], nullptr, 0, commit);
    if (ab) k::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, d_gr, d_br, n_embd, h_v, T, nullptr);
    k::gdn_conv_groups(cg, d_qkv, d_w, d_hf, C, 2 * h_k, 1e-6f, commit, ab ? d_x : nullptr, d_wa, d_wb, d_dt, d_sa, d_gf, d_bf,
                       n_embd, h_v, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    auto eq = [](const float* a, const float* b, size_t n) {
        const std::vector<float> x = down(a, n), y = down(b, n);
        return std::memcmp(x.data(), y.data(), n * 4) == 0;
    };
    bool same = eq(d_hr, d_hf, (size_t) T * C);
    if (ab) same = same && eq(d_gr, d_gf, (size_t) T * h_v) && eq(d_br, d_bf, (size_t) T * h_v);
    for (int i = 0; i < G; ++i) {
        same = same && eq(hr[i], hf[i], (size_t) C * 3);
        const std::vector<float> a = down(hf[i], (size_t) C * 3);
        same = same && (std::memcmp(a.data(), h0[i].data(), a.size() * 4) == 0) == !commit;   // written only by a commit
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G; ++i) std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_conv_groups%s %s, groups of %s rows", ab ? " + a/b" : "", commit ? "commit" : "window", rows);
    report(name, same);
}
// ---- 6. the window's indexer snapshots + appends in one launch (STRATA_FUSE_IDXAPP) ----
void idx_slots_case(const std::vector<int>& ntok, const std::vector<int32_t>& pos, bool snap) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int ID = (int) s.idx_dim, G = (int) ntok.size(), KS = k::kStepCount;
    const int64_t MC = 1024, TS = (s.idx_block - 1) * ID, PR = (MC / s.idx_block + 1) * ID;
    int T = 0;
    for (int n : ntok) T += n;
    std::mt19937 rng(171 + T * 3 + G + snap);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    std::vector<float> raw((size_t) T * ID), gam((size_t) ID);
    for (auto& v : raw) v = nd(rng) * 2.0f;
    for (auto& v : gam) v = 1.0f + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) T * KS, -7);
    for (int t = 0; t < T; ++t) steps[(size_t) t * KS + k::kStepPos] = pos[(size_t) t];
    const float *d_raw = up(raw), *d_gam = up(gam);
    const int32_t* d_steps = up(steps);
    struct Buf { float *tail, *dead, *pooled, *snap; int32_t* bpos; };
    std::vector<Buf> ref, fus;
    for (int i = 0; i < G; ++i) {
        std::vector<float> t0((size_t) TS), p0((size_t) PR), dd((size_t) ID), sn((size_t) TS, -1.0f);
        for (auto& v : t0) v = nd(rng);
        for (auto& v : p0) v = nd(rng);
        for (auto& v : dd) v = nd(rng);
        ref.push_back(Buf{up(t0), up(dd), up(p0), up(sn), up(std::vector<int32_t>{5})});
        fus.push_back(Buf{up(t0), up(dd), up(p0), up(sn), up(std::vector<int32_t>{5})});
    }
    const k::RopeScaling sc;
    k::QsaIndexerSlots g;
    for (int i = 0, first = 0; i < G; first += ntok[i], ++i) {   // the reference: the window's per-slot snapshot, per-row appends
        const Buf& b = ref[(size_t) i];
        if (snap) k::copy_from_mapped(b.snap, b.tail, TS, cs);
        const k::QsaIndexerBuffers ib{b.tail, b.dead, b.pooled, b.bpos};
        for (int t = first; t < first + ntok[i]; ++t)
            k::native_qsa_indexer_append(d_raw + (size_t) t * ID, d_steps + (size_t) t * KS + k::kStepPos, 0, d_gam, 1e-6f, ib,
                                         s, MC, sc, cs);
        const Buf& f = fus[(size_t) i];
        g.raw[i] = d_raw + (size_t) first * ID;
        g.pos[i] = d_steps + (size_t) first * KS + k::kStepPos;
        g.n_steps[i] = ntok[i];
        g.snap[i] = snap ? f.snap : nullptr;
        g.tail[i] = f.tail; g.dead[i] = f.dead; g.pooled[i] = f.pooled; g.block_pos[i] = f.bpos;
        g.max_cells[i] = (int) MC;
    }
    g.n = G;
    k::native_qsa_indexer_append_slots(g, KS, 0, d_gam, 1e-6f, s, sc, cs);
    ck(cudaStreamSynchronize(cs), "sync");
    auto eq = [](const float* a, const float* b, int64_t n) {
        const std::vector<float> x = down(a, (size_t) n), y = down(b, (size_t) n);
        return std::memcmp(x.data(), y.data(), (size_t) n * 4) == 0;
    };
    bool same = true;
    for (int i = 0; i < G; ++i) {
        const Buf &a = ref[(size_t) i], &b = fus[(size_t) i];
        same = same && eq(a.tail, b.tail, TS) && eq(a.dead, b.dead, ID) && eq(a.pooled, b.pooled, PR) &&
               eq(a.snap, b.snap, TS) && down(a.bpos, 1)[0] == down(b.bpos, 1)[0];
    }
    ck(cudaStreamDestroy(cs), "stream");
    char name[96], rows[64] = "";
    for (int i = 0, first = 0; i < G; first += ntok[i], ++i)
        std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d@%d", i ? "," : "", ntok[i], pos[(size_t) first]);
    std::snprintf(name, sizeof name, "indexer append_slots%s, rows@pos %s", snap ? " + snapshot" : "", rows);
    report(name, same);
}
// ---- 7b. the window's int8 K/V cells of every row in one launch (STRATA_FUSE_KVROWS) ----
void kvrows_case(int rows, bool mirror) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, PS = s.page_size, G = HD / k::KV_Q8_GROUP;
    const int64_t blocks = 40, slots = 24, vrow = slots * NKV * PS, hrow = blocks * NKV * PS;   // fewer VRAM slots than blocks
    std::mt19937 rng(201 + rows * 2 + mirror);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> kc((size_t) rows * NKV * HD), vc(kc.size());
    for (auto& x : kc) x = nd(rng) * 3.0f;
    for (auto& x : vc) x = nd(rng);
    for (int64_t i = 0; i < NKV * HD; i += 97) kc[(size_t) i] = 0.0f;   // a few zeros, and an all-zero group in row 0
    for (int64_t i = 0; i < k::KV_Q8_GROUP; ++i) vc[(size_t) i] = 0.0f;
    std::vector<int32_t> steps((size_t) rows * k::kStepCount, -3);
    struct Pools { int8_t *kq, *vq; uint16_t *ks, *vs; int32_t* table; k::KvHostPools h; };
    auto alloc = [&](int64_t nrow) {
        Pools p{};
        std::vector<int8_t> z8((size_t) nrow * HD, 5);
        std::vector<uint16_t> z16((size_t) nrow * G, 7);
        p.kq = up(z8); p.vq = up(z8); p.ks = up(z16); p.vs = up(z16);
        return p;
    };
    std::vector<Pools> ref, fus;
    k::KvQ8Rows kr;
    kr.n = rows;
    for (int r = 0; r < rows; ++r) {
        std::vector<int32_t> table((size_t) blocks, -1);
        for (int64_t b = 0; b < blocks; ++b) if (rng() % 3 != 0) table[(size_t) b] = (int32_t) (rng() % slots);
        const int32_t pos = (int32_t) (rng() % (blocks * PS));
        if (r == 0) table[(size_t) (pos / PS)] = 3;            // row 0's block resident, row 1's (if any) not
        if (r == 1) table[(size_t) (pos / PS)] = -1;
        steps[(size_t) r * k::kStepCount + k::kStepPos] = pos;
        for (Pools* side : {&ref.emplace_back(alloc(vrow)), &fus.emplace_back(alloc(vrow))}) {
            side->table = up(table);
            if (mirror && r != 2) {   // row 2 has no host copy
                const Pools hp = alloc(hrow);
                side->h.k_q = hp.kq; side->h.v_q = hp.vq; side->h.k_scale = hp.ks; side->h.v_scale = hp.vs;
            }
        }
        kr.k_q[r] = fus.back().kq; kr.v_q[r] = fus.back().vq; kr.k_scale[r] = fus.back().ks; kr.v_scale[r] = fus.back().vs;
        kr.table[r] = fus.back().table; kr.host[r] = fus.back().h;
    }
    const int32_t* d_steps = up(steps);
    const float *d_k = up(kc), *d_v = up(vc);
    for (int r = 0; r < rows; ++r) {
        const Pools& p = ref[(size_t) r];
        k::kv_append_q8_step(p.kq, p.vq, p.ks, p.vs, p.table, d_steps + (size_t) r * k::kStepCount, d_k + (size_t) r * NKV * HD,
                             d_v + (size_t) r * NKV * HD, s, nullptr, p.h.present() ? &p.h : nullptr);
    }
    k::kv_append_q8_rows(kr, d_steps, (int) k::kStepCount, d_k, d_v, (int) (NKV * HD), s, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    auto eq8 = [](const int8_t* a, const int8_t* b, int64_t n) {
        const std::vector<int8_t> x = down(a, (size_t) n), y = down(b, (size_t) n);
        return std::memcmp(x.data(), y.data(), (size_t) n) == 0;
    };
    auto eq16 = [](const uint16_t* a, const uint16_t* b, int64_t n) {
        const std::vector<uint16_t> x = down(a, (size_t) n), y = down(b, (size_t) n);
        return std::memcmp(x.data(), y.data(), (size_t) n * 2) == 0;
    };
    bool same = true;
    for (int r = 0; r < rows; ++r) {
        const Pools &a = ref[(size_t) r], &b = fus[(size_t) r];
        same = same && eq8(a.kq, b.kq, vrow * HD) && eq8(a.vq, b.vq, vrow * HD) && eq16(a.ks, b.ks, vrow * G) &&
               eq16(a.vs, b.vs, vrow * G);
        if (a.h.present())
            same = same && eq8(a.h.k_q, b.h.k_q, hrow * HD) && eq8(a.h.v_q, b.h.v_q, hrow * HD) &&
                   eq16(a.h.k_scale, b.h.k_scale, hrow * G) && eq16(a.h.v_scale, b.h.v_scale, hrow * G);
    }
    char name[96];
    std::snprintf(name, sizeof name, "kv_append_q8_rows, %d rows%s", rows, mirror ? " (host copies, non-resident pages)" : "");
    report(name, same);
}
}  // namespace

int main() {
    std::printf("batch rows parity (#2234 kernels against the per-row launches)\n");
    for (int r : {1, 2, 4, 8}) scores_case(r);
    for (int r : {1, 3, 8}) {
        attn_case(r, true);
        attn_case(r, false);
    }
    gdn_case({1}, false);
    gdn_case({1, 1, 1, 1}, false);
    gdn_case({2, 2, 2, 2}, false);
    gdn_case({1, 3, 2, 1, 1}, false);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, false);
    gdn_case({1}, true);
    gdn_case({1, 1, 1, 1}, true);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, true);
    bcommit_gdn_case({2, 2}, {2, 2});
    bcommit_gdn_case({2, 2}, {1, 2});
    bcommit_gdn_case({2, 2}, {0, 1});
    bcommit_gdn_case({1, 1}, {1, 1});
    bcommit_gdn_case({1, 1, 1, 1}, {1, 0, 1, 1});
    bcommit_gdn_case({1, 3, 2}, {1, 3, 1});
    bcommit_gdn_case({4}, {3});
    // positions: a block completes at the second row (reads the restored tail), a row not kept (-1), the spare (0)
    bcommit_qsa_case({2, 2}, {{102, 103}, {57, 58}});
    bcommit_qsa_case({2, 2}, {{103, -1}, {58, 59}});
    bcommit_qsa_case({1, 1}, {{7}, {0}});
    bcommit_qsa_case({1, 3, 2}, {{11}, {29, 30, 31}, {14, -1}});
    for (const bool ab : {false, true}) {
        conv_groups_case({2, 2}, false, ab);
        conv_groups_case({1, 3, 2, 1, 1}, false, ab);
        conv_groups_case({4, 4}, false, ab);
        conv_groups_case({1, 1, 1, 1}, true, ab);
        conv_groups_case({1}, true, ab);
        conv_groups_case({1, 1, 1, 1, 1, 1, 1}, false, ab);
    }
    for (const bool snap : {true, false}) {
        idx_slots_case({2, 2}, {102, 103, 57, 58}, snap);
        idx_slots_case({1, 1, 1, 1}, {3, 0, 7, 11}, snap);
        idx_slots_case({1, 3, 2}, {11, 29, 30, 31, 14, -1}, snap);
        idx_slots_case({4, 4}, {200, 201, 202, 203, 33, 34, 35, 36}, snap);
    }
    for (const bool mirror : {false, true})
        for (int r : {1, 2, 4, 8}) kvrows_case(r, mirror);
    std::printf("%s\n", bad ? "batch_rows_parity: FAILED" : "batch_rows_parity: ok");
    return bad ? 1 : 0;
}
