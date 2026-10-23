// src/kernels/fuse_moe_parity.cpp - the verify window's MoE-path launch fusions (STRATA_FUSE_*) against the launch
// sequences they replace, BITWISE: every output byte of the fused launch is compared with the old sequence's, on
// random inputs (GPU, synthetic, no model).  Exit 0 when all match.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 64), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
// mapped host memory and its device alias
template <typename T> T* mapped(size_t n, T** d) {
    T* h = nullptr;
    check(cudaHostAlloc((void**) &h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    check(cudaHostGetDevicePointer((void**) d, h, 0), "devptr");
    std::memset(h, 0, n * sizeof(T) + 64);
    return h;
}
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// STRATA_FUSE_FETCH: wait_fetch_rebase vs wait_flag_ge(_or) + fetch_blobs + rebase_ptrs.  Blobs in device memory and in
// mapped host memory (the PCIe share's source), 0..cap of them, the skip word equal to the ring or not.
int test_fetch(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0, cases = 0;
    const int nblob_src = 24;
    for (const int64_t blob : {(int64_t) 16, (int64_t) 4096 + 16, (int64_t) 302080}) {
        const size_t srcb = (size_t) nblob_src * (size_t) blob;
        std::vector<uint8_t> hs(srcb);
        for (auto& v : hs) v = (uint8_t) rng();
        uint8_t* dsrc = dev<uint8_t>(srcb);
        up(dsrc, hs);
        uint8_t* hsrc_d = nullptr;
        uint8_t* hsrc = mapped<uint8_t>(srcb, &hsrc_d);
        std::memcpy(hsrc, hs.data(), srcb);
        uint32_t* flag_d = nullptr;
        uint32_t* flag = mapped<uint32_t>(4, &flag_d);
        uint32_t* skip = dev<uint32_t>(4);
        for (const int cap : {16, 8}) {
            for (const int n : {0, 1, 3, cap}) {
                for (const int src_kind : {0, 1}) {   // 0: device blobs, 1: mapped host blobs
                    for (const int sk : {0, 1}) {     // skip word: 1 = equal to the ring (device plan)
                        const uint32_t ring = 7 + (uint32_t) (rng() % 100);
                        *(volatile uint32_t*) flag = ring + (uint32_t) (rng() % 2);   // the flag has risen
                        const uint32_t skv = sk ? ring : ring - 1;
                        check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
                        // the plan: counts[2] = n, ptr2[0, cap) blob pointers (some repeated), the rest garbage
                        std::vector<int32_t> cnt = {5, 9, n, 0};
                        std::vector<unsigned long long> ptr(cap);
                        uint8_t* base = src_kind ? hsrc_d : dsrc;
                        for (int k = 0; k < cap; ++k) ptr[k] = (unsigned long long) (base + (size_t) (rng() % nblob_src) * blob);
                        int32_t* dcnt = dev<int32_t>(4);
                        up(dcnt, cnt);
                        unsigned long long* p_old = dev<unsigned long long>(cap);
                        unsigned long long* p_src = dev<unsigned long long>(cap);
                        unsigned long long* p_out = dev<unsigned long long>(cap);
                        up(p_old, ptr);
                        up(p_src, ptr);
                        std::vector<unsigned long long> junk(cap, 0xdeadbeefcafef00dull);
                        up(p_out, junk);
                        const size_t stb = (size_t) cap * (size_t) blob;
                        std::vector<uint8_t> fill(stb);
                        for (auto& v : fill) v = (uint8_t) rng();
                        uint8_t* st_old = dev<uint8_t>(stb);
                        uint8_t* st_new = dev<uint8_t>(stb);
                        up(st_old, fill);
                        up(st_new, fill);
                        // the old launches
                        if (sk) wait_flag_ge_or(flag_d, ring, skip, s);
                        else wait_flag_ge(flag_d, ring, s);
                        fetch_blobs(p_old, dcnt + 2, st_old, blob, cap, s);
                        rebase_ptrs(p_old, dcnt + 2, st_old, blob, s);
                        // the fused one
                        wait_fetch_rebase(flag_d, ring, sk ? skip : nullptr, p_src, dcnt + 2, st_new, blob, cap, p_out, s);
                        check(cudaStreamSynchronize(s), "sync");
                        const auto a = down(st_old, stb), b = down(st_new, stb);
                        const auto pa = down(p_old, cap), pb = down(p_out, cap), ps = down(p_src, cap);
                        bool ok = same(a, b) && same(ps, ptr);   // the staging bytes; src untouched
                        for (int k = 0; k < cap && ok; ++k) {
                            // the first n pointers: the rebased ones (relative to each staging buffer); the others untouched
                            if (k < n) ok = pa[k] - (unsigned long long) st_old == pb[k] - (unsigned long long) st_new;
                            else ok = pb[k] == junk[k] && pa[k] == ptr[k];
                        }
                        ++cases;
                        if (!ok) {
                            std::printf("    *** wait_fetch_rebase blob %lld cap %d n %d src %d skip %d differs ***\n",
                                        (long long) blob, cap, n, src_kind, sk);
                            ++bad;
                        }
                        cudaFree(dcnt); cudaFree(p_old); cudaFree(p_src); cudaFree(p_out); cudaFree(st_old); cudaFree(st_new);
                    }
                }
            }
        }
        // the wait itself: a flag below the ring holds the fused launch until the host raises it
        {
            const uint32_t ring = 1000;
            *(volatile uint32_t*) flag = ring - 1;
            const uint32_t skv = 0;
            check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
            int32_t* dcnt = dev<int32_t>(4);
            up(dcnt, std::vector<int32_t>{0, 0, 2, 0});
            std::vector<unsigned long long> ptr = {(unsigned long long) dsrc, (unsigned long long) (dsrc + blob)};
            unsigned long long* p_src = dev<unsigned long long>(2);
            unsigned long long* p_out = dev<unsigned long long>(2);
            up(p_src, ptr);
            uint8_t* st = dev<uint8_t>((size_t) 2 * blob);
            wait_fetch_rebase(flag_d, ring, skip, p_src, dcnt + 2, st, blob, 2, p_out, s);
            int polls = 0;
            while (cudaStreamQuery(s) == cudaErrorNotReady && polls < 50) { ++polls; struct timespec ts{0, 2000000}; nanosleep(&ts, nullptr); }
            const bool held = polls == 50;
            *(volatile uint32_t*) flag = ring;
            check(cudaStreamSynchronize(s), "sync");
            const auto b = down(st, (size_t) 2 * blob);
            const bool ok = held && std::memcmp(b.data(), hs.data(), (size_t) 2 * blob) == 0;
            ++cases;
            if (!ok) { std::printf("    *** wait_fetch_rebase: the wait (held %d) or the copy after it is wrong ***\n", (int) held); ++bad; }
            cudaFree(dcnt); cudaFree(p_src); cudaFree(p_out); cudaFree(st);
        }
        cudaFree(dsrc); cudaFreeHost(hsrc); cudaFreeHost(flag); cudaFree(skip);
    }
    std::printf("  %-60s %s (%d cases)\n", "STRATA_FUSE_FETCH wait_fetch_rebase == wait + fetch + rebase",
                bad ? "*** NO ***" : "bitwise", cases);
    return bad;
}

// strict: every byte; nan_cases: also NaN inputs, where two NaN operands meet in one add and the hardware keeps one
// of their sign / payload bits by operand order, which the compiler may commute: a NaN result must stay a NaN there
bool g_nan = false;
// mismatching floats: -1 if any differs other than NaN against NaN, else the count of NaN-vs-NaN bit differences
long diff_floats(const std::vector<float>& a, const std::vector<float>& b, size_t* first) {
    long nanx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
        if (std::isnan(a[i]) && std::isnan(b[i])) { ++nanx; continue; }
        *first = i;
        return -1;
    }
    return nanx;
}
// random floats with the awkward values in: -0.0, +0.0, denormals, large, inf (and NaN when g_nan)
float odd_float(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    switch (rng() % 64) {
        case 0: return -0.0f;
        case 1: return 0.0f;
        case 2: return 1e-40f * (rng() % 2 ? 1.0f : -1.0f);
        case 3: return 3e30f * nd(rng);
        case 4: return g_nan && rng() % 2 ? (rng() % 2 ? std::nanf("") : -std::nanf("")) : (rng() % 2 ? 1.0f : -1.0f) * std::numeric_limits<float>::infinity();
        default: return nd(rng);
    }
}

// STRATA_FUSE_COMBINE: native_moe_combine_multi_hits_mapped vs
//   copy_rows_from_mapped_after + native_moe_combine_multi_hits_added                  (skip == nullptr)
//   wait_flag_ge_or + copy_or_zero_from_mapped + native_moe_combine_multi_hits_added   (skip: the device plan)
int test_combine(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    native_moe_combine_set_enabled(true);
    int bad = 0, cases = 0;
    long nan_bits = 0;
    const int K = 10;
    for (const int N : {2560, 2048, 64}) {
        for (int n = 1; n <= 8; ++n) {
            for (const int dp : {0, 1}) {
                for (const int sk : {0, 1}) {
                    if (!dp && sk) continue;
                    for (const int hit_kind : {0, 1, 2}) {   // no hits, some, all
                        const size_t rows = (size_t) n * K, nf = rows * N;
                        std::vector<float> mp(nf), hits(nf), w(rows), sh((size_t) n * N);
                        for (auto& v : mp) v = odd_float(rng);
                        for (auto& v : hits) v = odd_float(rng);
                        for (auto& v : w) v = (rng() % 16 == 0) ? -0.0f : (float) (rng() % 1000) / 997.0f;
                        for (auto& v : sh) v = odd_float(rng);
                        std::vector<int32_t> ent(rows);
                        for (size_t i = 0; i < rows; ++i) ent[i] = (int32_t) i;
                        std::shuffle(ent.begin(), ent.end(), rng);
                        const int nh = hit_kind == 0 ? 0 : hit_kind == 2 ? (int) rows : (int) (rng() % rows);
                        std::vector<int32_t> dsth(rows, -1);
                        for (int i = 0; i < nh; ++i) dsth[i] = ent[i];
                        std::vector<int32_t> cnt = {3, nh, 0, 0};
                        float* m_d = nullptr;
                        float* m_h = mapped<float>(nf, &m_d);
                        std::memcpy(m_h, mp.data(), nf * 4);
                        uint32_t* flag_d = nullptr;
                        uint32_t* flag = mapped<uint32_t>(4, &flag_d);
                        const uint32_t ring = 5 + (uint32_t) (rng() % 50);
                        *(volatile uint32_t*) flag = ring;
                        uint32_t* skip = dev<uint32_t>(4);
                        const uint32_t skv = sk ? ring : ring + 1;
                        check(cudaMemcpy(skip, &skv, 4, cudaMemcpyHostToDevice), "skip");
                        float *parts = dev<float>(nf), *hd = dev<float>(nf), *wd = dev<float>(rows), *shd = dev<float>((size_t) n * N);
                        float *o1 = dev<float>((size_t) n * N), *o2 = dev<float>((size_t) n * N);
                        int32_t *dd = dev<int32_t>(rows), *cd = dev<int32_t>(4);
                        std::vector<float> junk(nf);
                        for (auto& v : junk) v = odd_float(rng);
                        up(parts, junk);
                        up(hd, hits); up(wd, w); up(shd, sh); up(dd, dsth); up(cd, cnt);
                        check(cudaMemset(o1, 0x7f, (size_t) n * N * 4), "memset");
                        check(cudaMemset(o2, 0x3c, (size_t) n * N * 4), "memset");
                        if (dp) {
                            wait_flag_ge_or(flag_d, ring, skip, s);
                            copy_or_zero_from_mapped(parts, m_d, (long long) nf, skip, ring, s);
                        } else {
                            copy_rows_from_mapped_after(flag_d, ring, parts, m_d, (int64_t) rows, N, dd, cd + 1, s);
                        }
                        native_moe_combine_multi_hits_added(parts, hd, dd, cd + 1, wd, shd, o1, N, K, n, s);
                        native_moe_combine_multi_hits_mapped(m_d, hd, dd, cd + 1, wd, shd, o2, N, K, n, flag_d, ring,
                                                             dp ? skip : nullptr, nullptr, s);
                        check(cudaStreamSynchronize(s), "sync");
                        const auto a = down(o1, (size_t) n * N), b = down(o2, (size_t) n * N);
                        ++cases;
                        size_t i = 0;
                        const long d = diff_floats(a, b, &i);
                        if (d < 0 || (d > 0 && !g_nan)) {
                            std::printf("    *** combine mapped N %d n %d dp %d skip %d hits %d differs at %zu: %a vs %a ***\n",
                                        N, n, dp, sk, nh, i, a[i], b[i]);
                            ++bad;
                        }
                        nan_bits += d > 0 ? d : 0;
                        cudaFreeHost(m_h); cudaFreeHost(flag); cudaFree(skip); cudaFree(parts); cudaFree(hd); cudaFree(wd);
                        cudaFree(shd); cudaFree(o1); cudaFree(o2); cudaFree(dd); cudaFree(cd);
                    }
                }
            }
        }
    }
    std::printf("  %-60s %s (%d cases%s)\n", "STRATA_FUSE_COMBINE combine_hits_mapped == copy + combine_hits",
                bad ? "*** NO ***" : g_nan ? "bitwise but NaN-vs-NaN" : "bitwise", cases,
                g_nan ? (", NaN inputs: " + std::to_string(nan_bits) + " NaN outputs differ in sign/payload only").c_str() : "");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) { std::printf("no GPU: skipped\n"); return 77; }
    const std::string only = argc > 1 ? argv[1] : "";
    std::mt19937 rng(1234);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    if (only.empty() || only == "fetch") bad += test_fetch(rng, s);
    if (only.empty() || only == "combine") {
        bad += test_combine(rng, s);
        g_nan = true;   // the same with NaN inputs: every non-NaN output bitwise, NaN outputs NaN
        bad += test_combine(rng, s);
        g_nan = false;
    }
    std::printf("fuse_moe_parity: %s\n", bad ? "FAILED" : "all bitwise");
    return bad ? 1 : 0;
}
