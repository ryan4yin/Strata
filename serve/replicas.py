"""Data-parallel replicas: N independent engines, each on its own group of cards, behind one server.

A model that fits two cards (IQ3_S on two R9700s) runs better as two 2-card engines than as one 4-card layer split:
the split runs one pipeline whose stages wait on each other, two replicas run two pipelines.  The server keeps its one
HTTP front, its one Service and its one set of endpoints; `ReplicaEngine` stands in for the engine and spreads the
requests over the replicas.

Opt-in, in the run config:

    "replicas": 2                                     the "gpu" list cut into 2 equal groups ([0,1] and [2,3])
    "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]  the groups named; an entry may add "env" and "args"

or `serve/server.py --replicas 2`.  Nothing changes without it.

Scheduling (`Scheduler`).  A request is not given a replica when it arrives: it joins one queue, and a replica that has
room takes the next request from it.  What each replica holds is tracked as hashes of 64-token blocks of the prompts
and answers it served (the engine itself checks the real prefix, as it always does), so a request's "home" is the
replica holding the longest start of its prompt; a new talk is at home everywhere.  A free replica first takes the
oldest request at home there.  It takes a request whose home is busy only once waiting costs more than moving:
moving reads the prompt tokens the free replica lacks (PREFILL_TOK_S); waiting is what the request has already
waited plus one measured service time for every other request queued for the same home (moving the oldest of them
shortens that queue).  It is the rent-or-buy rule: a lone request waits at most as long as moving would cost, and a
card does not idle while a queue it could serve grows.  Placing by the held prefix needs no client to name its
conversation.

Host RAM.  Every engine keeps its own copy of the experts in RAM (about 47 GiB for Qwen3.8-Flash-Next IQ3_S), so two
replicas need two copies and four need four.  "replica_shared_arena": true (or STRATA_REPLICA_SHARED_ARENA=1; opt-in)
backs the experts' arena of all replicas with ONE shared memory object (the engine's --shared-expert-arena, MAP_SHARED):
the experts are in RAM once, whatever the number of replicas.  On Linux the object is a memfd the server creates
(os.memfd_create): it is not a file on a mount, so the size of /dev/shm does not limit it; the engines open it as
/proc/<server pid>/fd/<n>.  On Windows it is a named section (CreateFileMapping) the engines open by name.  Without memfd (an old kernel, a sandbox that refuses it) a file in /dev/shm is used instead.
The pack hash in the header refuses another model; the first engine to start reads the experts into it and the others
wait for it and skip the read (a state word in the header: nobody / a pid reading / filled; an engine whose reader died
takes the read over); the server closes the memfd (or unlinks the file) when it ends - an engine that still maps it
keeps its pages, the kernel frees them with the last mapping - and a /dev/shm file whose server is gone is removed at
the next start.  See shared_arena_plan.

CPU cores.  On Linux the replicas get disjoint physical cores (serve/cpu_affinity.py, from PR #1436 by agorevski), so
the host threads of the engines (the expert workers) do not compete.  "replica_cpus": false turns it off.  It also
takes one explicit CPU list per replica, e.g. "replica_cpus": [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13,
14, 15]], for a hybrid P/E machine where the automatic contiguous split gives each replica a different mix of cores.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import sys
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # one request's time on a replica, in prompt tokens, until a service time is measured
PREFILL_TOK_S = 1000.0              # prompt tokens a replica reads per second (R9700 IQ3_S, 12K cold: 960-1,250)
SERVICE_EWMA = 0.2                  # weight of the newest request in the measured service time
WAIT_POLL_S = 0.25                  # longest sleep of a queued request: its cancel and the steal deadline are checked
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
CONV_MAX = 4096                     # conversation names remembered (a tie-break and the session map, see the module doc)
CONV_MAX_LEN = 128                  # characters kept of a conversation id
HOME_SLACK = 2 * BLOCK              # tokens a prompt may lose to its own tail (a template boundary) and still be at home
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often
SESSION_MAP = ".map"                # next to the session files: the router's conversation->replica bindings
SESSION_MAP_MAX = 256               # bindings kept in it: the most recently used conversations


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[int]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    cards = _cards(cfg.get("gpu"))
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"gpus": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec <= 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs the "gpu" list to cut into groups (e.g. "gpu": [0, 1, 2, 3])')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: name the '
                                     f'groups ("replicas": [{{"gpus": [0, 1]}}, ...]) or change the count')
        n = len(cards) // spec
        return [cards[i * n:(i + 1) * n] for i in range(spec)]
    groups = []
    for e in spec:
        g = e.get("gpus") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs its cards: {"gpus": [0, 1]}')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        return []
    return groups


def _cards(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    try:
        return [int(str(x).strip()) for x in items if str(x).strip() != ""]
    except ValueError:
        raise ReplicaConfigError(f"cards must be numbers, not {g!r}") from None


ARENA_PREFIX = "strata-replicas-"
_MEMFDS: dict = {}                   # the memfds this server created: path (/proc/<pid>/fd/<n>) -> fd
ARENA_MIN_RAM = 8 << 30              # a memfd arena: at least this much free RAM to try (the engine reserves the real size)
ARENA_MIN_FREE = 8 << 30             # /dev/shm must have at least this much free to try (the engine reserves the real size)
ARENA_INCOMPATIBLE = ("--mmap-experts", "--resident-experts", "--shared-expert-arena", "--resident-budget-gib")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_arenas(shm: str = "/dev/shm", uid: int | None = None) -> list[str]:
    """Remove the arena files of servers that no longer run (a killed server cannot unlink its own).  A file of a living
    server, or of another user, is never touched; a mapping an engine still holds stays valid either way."""
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else 0
    gone = []
    try:
        names = os.listdir(shm)
    except OSError:
        return gone
    head = f"{ARENA_PREFIX}{uid}-"
    for n in names:
        if not (n.startswith(head) and n.endswith(".bin")):
            continue
        try:
            pid = int(n[len(head):-4])
        except ValueError:
            continue
        if _pid_alive(pid):
            continue
        try:
            os.unlink(os.path.join(shm, n))
            gone.append(n)
        except OSError:
            pass
    return gone


def _free(path: str) -> int:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _mem_available() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def shared_arena_plan(cfg: dict, n_replicas: int, *, shm: str = "/dev/shm", platform: str | None = None,
                      environ=None, free_bytes=None, use_memfd: bool | None = None) -> tuple[str | None, str]:
    """(the file for --shared-expert-arena or None, a line for the log).  Opt-in: "replica_shared_arena": true (or
    STRATA_REPLICA_SHARED_ARENA=1).  Only on Linux (the engine's MAP_SHARED backing), for Qwen's arena (not DeepSeek,
    not the mapped or budgeted RAM modes, which have no arena), and with room in /dev/shm.  The engine checks the real
    size and reserves it before it loads (a tmpfs that is too small is an error at start, not a bus error later)."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    want = cfg.get("replica_shared_arena")
    if want is None:
        want = str(environ.get("STRATA_REPLICA_SHARED_ARENA", "")).strip().lower() in ("1", "true", "on", "yes")
    if not want or n_replicas < 2:
        return None, ""
    if platform.startswith("win"):
        # a named pagefile-backed section the engines open by name (src/core/pinned.cu); it lives while any engine holds it
        name = f"Local{chr(92)}strata-replicas-{os.getpid()}"
        return f"win:{name}", (f"one shared expert arena for {n_replicas} replicas (a Windows section, {name}): "
                               "the experts are in RAM once")
    if not platform.startswith("linux"):
        return None, "shared expert arena: Linux and Windows only; each replica keeps its own copy of the experts in RAM"
    args = [str(a) for a in cfg.get("args") or []]
    if isinstance(cfg.get("replicas"), list):
        for e in cfg["replicas"]:
            if isinstance(e, dict):
                args += [str(a) for a in e.get("args") or []]
    if cfg.get("arch") == "deepseek41" or any(a in args for a in ARENA_INCOMPATIBLE):
        return None, "shared expert arena: this model/config has no private expert arena to share; each replica as before"
    if use_memfd is None:
        use_memfd = hasattr(os, "memfd_create") and str(cfg.get("replica_shared_arena")).lower() != "devshm"
    if use_memfd:
        avail = free_bytes if free_bytes is not None else _mem_available()
        if avail < ARENA_MIN_RAM:
            return None, (f"shared expert arena: only {avail / 2**30:.1f} GiB of RAM is free; each replica keeps its own "
                          "copy of the experts in RAM")
        try:
            fd = os.memfd_create("strata-replicas-arena", os.MFD_CLOEXEC)
        except OSError:
            fd = -1
        if fd >= 0:
            path = f"/proc/{os.getpid()}/fd/{fd}"
            _MEMFDS[path] = fd
            warn = ""
            try:                 # the arena once plus about 7 GB of private RAM per engine (measured); a warning, never a refusal
                from strata_control import replica_profile as _rp
                q = _rp.quant_of(cfg)
                if q:
                    need = _rp.ARENA_GB[q] + n_replicas * _rp.PER_REPLICA_GB
                    if avail / 2**30 < need + 4:
                        warn = (f" WARNING: about {need:.0f} GB are needed for {n_replicas} engines and {avail / 2**30:.0f} GB "
                                "are free: the last engine may not fit (fewer replicas, or the layer split, is safer)")
            except Exception:    # noqa: BLE001
                pass
            return path, (f"one shared expert arena for {n_replicas} replicas (memory object of the server, not limited by "
                          f"the size of /dev/shm): the experts are in RAM once.{warn}")
    if not (os.path.isdir(shm) and os.access(shm, os.W_OK)):
        return None, f"shared expert arena: {shm} is not usable; each replica keeps its own copy of the experts in RAM"
    free = free_bytes if free_bytes is not None else _free(shm)
    if free < ARENA_MIN_FREE:
        return None, (f"shared expert arena: {shm} has only {free / 2**30:.1f} GiB free (a tmpfs is half the RAM by "
                      "default); each replica keeps its own copy of the experts in RAM")
    sweep_stale_arenas(shm)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = os.path.join(shm, f"{ARENA_PREFIX}{uid}-{os.getpid()}.bin")
    return path, f"one shared expert arena for {n_replicas} replicas ({path}): the experts are in RAM once"


def remove_arena(path: str | None) -> None:
    """Close the memfd / unlink the arena file (the server's end, or a start that fell back): engines that map it keep
    their pages."""
    if not path or path.startswith("win:"):      # (a Windows section goes with its last handle)
        return
    fd = _MEMFDS.pop(path, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[int]) -> dict:
    """The config of replica i: its own cards, log and learned-profile file; the entry's "env" and "args" added."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c["gpu"] = gpus if len(gpus) > 1 else gpus[0]
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if entry.get("env"):
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if entry.get("args"):
        c["args"] = list(c["args"]) + [str(x) for x in entry["args"]]
    return c


def _explicit_cpu_sets(n: int, spec) -> list[list[int]]:
    """Validate explicit per-replica CPU lists: exactly one non-empty list per replica, no CPU in two lists, and every
    CPU in the process's sched_getaffinity (an engine cannot use a CPU this process may not run on).  The order in each
    list is kept; the kernel accepts it as an affinity mask and the caller passes it unchanged to each engine."""
    if not isinstance(spec, (list, tuple)):
        raise ReplicaConfigError(f'"replica_cpus" must be true, false, or one CPU list per replica, not {spec!r}')
    if len(spec) != n:
        raise ReplicaConfigError(f'"replica_cpus" needs one CPU list for each of the {n} replicas, but got '
                                 f'{len(spec)} list(s)')
    if not hasattr(os, "sched_getaffinity"):
        raise ReplicaConfigError('"replica_cpus" CPU lists need sched_getaffinity (Linux) to validate the CPUs')
    allowed = set(os.sched_getaffinity(0))
    seen: dict[int, int] = {}
    result: list[list[int]] = []
    for i, cpus in enumerate(spec):
        if not isinstance(cpus, (list, tuple)):
            raise ReplicaConfigError(f'"replica_cpus"[{i}] must be a list of CPU numbers, not {cpus!r}')
        if not cpus:
            raise ReplicaConfigError(f'"replica_cpus"[{i}] is empty: each replica needs at least one CPU')
        group: list[int] = []
        for cpu in cpus:
            if isinstance(cpu, bool) or not isinstance(cpu, int):
                raise ReplicaConfigError(f'"replica_cpus"[{i}] has {cpu!r}; every entry must be a CPU number')
            if cpu not in allowed:
                raise ReplicaConfigError(f'"replica_cpus"[{i}] has CPU {cpu}, which is not in this process\'s '
                                         f'sched_getaffinity; allowed CPUs: {sorted(allowed)}')
            if cpu in seen:
                if seen[cpu] == i:
                    raise ReplicaConfigError(f'"replica_cpus"[{i}] lists CPU {cpu} more than once')
                raise ReplicaConfigError(f'CPU {cpu} is in both replica {seen[cpu]} and replica {i}; '
                                         'replica CPU lists must not overlap')
            seen[cpu] = i
            group.append(cpu)
        result.append(group)
    return result


def cpu_sets(n: int, enabled: bool | list[list[int]] | None = True):
    """n disjoint sets of CPUs (whole physical cores, SMT siblings together), or None (off, not Linux, or the topology
    cannot be read: the engines then share the cores as they would without replicas).  An explicit list of CPU lists
    (one per replica) is returned after validation; it is meant for hybrid P/E machines where the automatic split's
    contiguous slices would give replicas different P/E mixes."""
    if isinstance(enabled, (list, tuple)):
        return _explicit_cpu_sets(n, enabled)
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        print(f"[strata] replicas: no separate CPU cores ({e})", flush=True)
        return None


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX, conv_cap: int = CONV_MAX):
        self.n, self.cap, self.conv_cap = n, cap, conv_cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.convs: collections.OrderedDict[str, int] = collections.OrderedDict()   # conversation id -> replica
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        return self.matched_chain(self.chain(ids))

    def matched_chain(self, chain: list[int]) -> list[int]:
        """`matched` for a prompt already hashed with `chain` (a queued request is weighed again on every dispatch)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(chain):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def held(self, conv: str) -> int | None:
        """The replica that holds this conversation, or None (a return makes it the newest entry again)."""
        with self.lock:
            r = self.convs.get(conv)
            if r is not None:
                self.convs.move_to_end(conv)
            return r

    def bind(self, conv: str, r: int) -> None:
        """Record that replica r served this conversation last (the oldest entry goes past the cap)."""
        with self.lock:
            self.convs[conv] = r
            self.convs.move_to_end(conv)
            while len(self.convs) > self.conv_cap:
                self.convs.popitem(last=False)

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted), and with it its conversations."""
        with self.lock:
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]
            for conv in [c for c, k in self.convs.items() if k == r]:
                del self.convs[conv]


class Ticket:
    """One request, queued for a replica or running on one."""
    __slots__ = ("chain", "since", "pin", "hint", "replica", "started", "away")

    def __init__(self, ids, since: float, pin: int | None = None, hint: int | None = None):
        self.chain = Router.chain(ids)
        self.since, self.pin, self.hint = since, pin, hint
        self.replica: int | None = None
        self.started = 0.0
        self.away: tuple[int, float] | None = None     # taken off its busy home: (tokens read beyond it, seconds waited)


class Scheduler:
    """Which queued request runs on which replica (see "Scheduling" in the module doc).  It keeps no lock and no clock:
    the caller holds one lock around every call and passes `now`, so a simulated clock drives it the same way."""

    def __init__(self, router: Router, caps: list[int], prefill_tok_s: float = PREFILL_TOK_S,
                 load_tokens: int = LOAD_TOKENS):
        n = len(caps)
        self.router, self.caps, self.prefill_tok_s = router, list(caps), float(prefill_tok_s)
        self.service_s = load_tokens / self.prefill_tok_s   # one request's time on a replica: a prior until measured
        self.queue: list[Ticket] = []                        # oldest first
        self.running = [0] * n
        self.routed = [0] * n
        self.held = [0] * n                                  # a session save/restore owns the replica meanwhile

    def submit(self, t: Ticket) -> None:
        self.queue.append(t)

    def withdraw(self, t: Ticket) -> None:
        if t in self.queue:
            self.queue.remove(t)

    def free(self, alive: list[int]) -> list[int]:
        return [k for k in alive if not self.held[k] and self.running[k] < self.caps[k]]

    def dispatch(self, now: float, alive: list[int]) -> list[Ticket]:
        """Start queued requests on the replicas with room; returns the started ones (their `replica` set)."""
        started = []
        while self.queue:
            free = self.free(alive)
            if not free:
                break
            pick = self._pick(now, alive, free)
            if pick is None:
                break
            t, k = pick
            self.queue.remove(t)
            t.replica, t.started = k, now
            self.running[k] += 1
            self.routed[k] += 1
            started.append(t)
        return started

    def finish(self, t: Ticket, now: float, measured: bool = True) -> None:
        self.running[t.replica] -= 1
        if measured:
            self.service_s += SERVICE_EWMA * ((now - t.started) - self.service_s)

    def next_wake(self, now: float, alive: list[int]) -> float | None:
        """Seconds until a queued request may move to a replica that has room now, None when nothing can."""
        free = self.free(alive)
        if not free:
            return None
        soonest = None
        for t, _, extra, rivals in self._weigh(alive):
            if extra is None:
                continue
            due = t.since + min(extra[k] for k in free) / self.prefill_tok_s - rivals * self.service_s - now
            soonest = due if soonest is None else min(soonest, due)
        return None if soonest is None else max(0.0, soonest)

    def _weigh(self, alive: list[int]):
        """Per queued request: its home replicas, the tokens each replica would read beyond the best one (None when
        pinned), and how many other queued requests wait for the same home.  Moving any one of them shortens that
        queue by one service time, so the whole queue's wait counts toward moving its oldest."""
        rows = []
        for t in self.queue:
            if t.pin in alive:
                home, extra = [t.pin], None
            else:
                got = self.router.matched_chain(t.chain)
                best = max(got[k] for k in alive)
                home = [k for k in alive if got[k] >= best - HOME_SLACK]
                extra = {k: best - got[k] for k in alive}
            rows.append((t, home, extra))
        return [(t, home, extra, sum(1 for u, h, _ in rows if u is not t and set(h) & set(home)))
                for t, home, extra in rows]

    def _pick(self, now: float, alive: list[int], free: list[int]):
        weighed = self._weigh(alive)
        for t, home, _, _ in weighed:                       # the oldest request at home on a replica with room
            room = [k for k in home if k in free]
            if room:
                return t, min(room, key=lambda k: (k != t.hint, self.running[k], self.routed[k], k))
        best = None
        for t, _, extra, rivals in weighed:                 # else one whose wait has outgrown the cost of moving
            if extra is None:
                continue
            waited = now - t.since + rivals * self.service_s
            for k in free:
                cost = extra[k] / self.prefill_tok_s
                if waited >= cost and (best is None or (cost, t.since) < best[0]):
                    best = ((cost, t.since), t, k, extra[k], now - t.since)
        if best is None:
            return None
        _, t, k, tokens, waited = best
        t.away = (tokens, waited)
        return t, k


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    NOT_HERE = {"vram", "ctl", "slot_cv"}       # not offered: VRAM resize, the control line (session files are per replica)
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS, prefill_tok_s: float = PREFILL_TOK_S):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        d = self.__dict__
        d["engines"] = list(engines)
        d["router"] = Router(len(engines))
        d["sched"] = Scheduler(d["router"], [1] * len(engines), prefill_tok_s, load_tokens)
        d["cv"] = threading.Condition()        # guards sched; a queued request sleeps on it
        d["tl"] = threading.local()
        d["pin"] = threading.local()          # a dedicated port sets "index": that replica, whatever the cost
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

    @property
    def inflight(self) -> list[int]:
        """Requests running on each replica (a queued one is not on any yet)."""
        return self.sched.running

    @property
    def routed(self) -> list[int]:
        return self.sched.routed

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return self.current.last

    @last.setter
    def last(self, value):
        self.current.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def session_serial(self) -> bool:
        """Disk sessions are safe here: every replica runs one request at a time, so a save can wait for it
        (serve/server.py's slot_action asks this before its parallel-requests refusal)."""
        return all(not int(getattr(e, "batch", 0) or 0) for e in self.engines)

    @property
    def waiting(self) -> int:
        """Requests queued for a replica, plus those each engine holds for its control line."""
        return len(self.sched.queue) + sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if self.inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + s["slot"], "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far."""
        return [{"replica": k, "alive": e.alive(), "running": self.inflight[k], "routed": self.routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0)} for k, e in enumerate(self.engines)]

    # ---- disk sessions
    def session_file(self, action: str, path: str) -> dict:
        """Disk sessions for a replicas deployment: the engine's single-slot SAVE / RESTORE, once per replica.

        Each engine has its own slot 0, so replica 0 keeps PATH and replica k (k >= 1) PATH.r<k>, and the counts of
        all replicas are summed for the caller.  A replica that refuses (its own fingerprint, a missing file) does
        not stop the others: it only loses that card's cache, which a restart costs anyway.

        One file cannot name the conversation it holds, but the router can: on a save its conversation->replica
        bindings go to PATH.map, and a restore reads them back before the files, so a continuation is routed to the
        replica that holds it again and uses the restored state instead of whichever card placement picks.  A caller
        that deletes PATH after a restore should delete its PATH.r<k> siblings and PATH.map as well (serve/images.py
        saves and restores around its own unload job)."""
        if not self.session_serial:
            raise ValueError("session save/restore is not available with parallel requests (\"parallel\" / --batch)")
        files = [path] + [_suffixed(path, k) for k in range(1, len(self.engines))]
        if action == "restore":
            self._restore_bindings(path)
        total, refused, done = {"tokens": 0, "bytes": 0, "ms": 0.0}, None, 0
        for k, (engine, file) in enumerate(zip(self.engines, files)):
            if action == "restore" and k and not os.path.exists(file):
                continue                                # nothing was saved for this replica
            with self.cv:                               # a request running here ends first: one control line per engine
                self.sched.held[k] += 1                 # and none starts here until the file is done
                while self.sched.running[k]:
                    self.cv.wait(WAIT_POLL_S)
            try:
                result = engine.session_file(action, file)
            except Exception as exc:                    # noqa: BLE001 - one card's cache, not the whole call
                refused = exc
                print(f"[strata] replica {k}: session {action} of {os.path.basename(file)} failed: {exc}", flush=True)
                continue
            finally:
                with self.cv:
                    self.sched.held[k] -= 1
                    self.cv.notify_all()
            done += 1
            total["tokens"] += int(result.get("tokens") or 0)
            total["bytes"] += int(result.get("bytes") or 0)
            total["ms"] = max(total["ms"], float(result.get("ms") or 0.0))
        if not done and refused is not None:
            raise refused
        if action == "save":
            self._save_bindings(path)
        return total

    def _save_bindings(self, path: str) -> None:
        """The conversations each replica serves, for the next start (newest last, a few hundred at most)."""
        with self.router.lock:
            bindings = dict(self.router.convs)
        if not bindings:
            return
        if len(bindings) > SESSION_MAP_MAX:
            bindings = dict(list(bindings.items())[-SESSION_MAP_MAX:])
        target = path + SESSION_MAP
        temporary = target + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({"bindings": bindings}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            print(f"[strata] session bindings not written next to {path}: {exc}", flush=True)

    def _restore_bindings(self, path: str) -> None:
        """Put the saved conversations back on their replicas before a continuation of one is routed."""
        try:
            with open(path + SESSION_MAP, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return                                      # no map: placement decides, as it did before this feature
        bindings = saved.get("bindings") if isinstance(saved, dict) else None
        if not isinstance(bindings, dict):
            return
        placed = 0
        for conversation, replica in bindings.items():
            if isinstance(conversation, str) and isinstance(replica, int) and 0 <= replica < len(self.engines):
                self.router.bind(conversation, replica)
                placed += 1
        if placed:
            print(f"[strata] restored {placed} conversation bindings (each back on the replica that held it)",
                  flush=True)

    # ---- life cycle
    def restart(self, tries: int = 3):
        dead = [k for k, e in enumerate(self.engines) if not e.alive()]
        errors: dict[int, BaseException] = {}

        def one(k):
            try:
                self.engines[k].restart(tries)
                self.router.forget(k)
            except BaseException as e:           # noqa: BLE001
                errors[k] = e
        ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in dead]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errors and not self.alive():
            raise next(iter(errors.values()))
        for k, e in errors.items():
            print(f"[strata] replica {k} did not start ({e}); the others serve", flush=True)

    def unload(self):
        for e in self.engines:
            e.unload()
        for k in range(len(self.engines)):
            self.router.forget(k)

    def close(self):
        err = None
        for e in self.engines:
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        remove_arena(self.__dict__.get("arena_file"))   # (an engine that still maps it keeps its pages)
        if err:
            raise err

    # ---- scheduling
    def _alive(self) -> list[int]:
        return [k for k, e in enumerate(self.engines) if e.alive()]

    def _dispatch(self, alive: list[int]) -> None:
        """Start what can start (the caller holds cv) and wake the requests that were given a replica."""
        self.sched.caps = [max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines]
        started = self.sched.dispatch(time.monotonic(), alive)
        for t in started:
            if t.away is not None:
                # the one decision that gives up a warm cache: logged, as utilization alone cannot show it
                print(f"[strata] replica {t.replica} takes a request whose cache is busy elsewhere after "
                      f"{t.away[1]:.1f} s queued (reads {t.away[0]} more prompt tokens; running {self.sched.running}, "
                      f"queued {len(self.sched.queue)})", flush=True)
        if started:
            self.cv.notify_all()

    def _admit(self, ids, cancel) -> Ticket | None:
        """Queue this request until a replica takes it (see Scheduler); None when it was cancelled meanwhile.

        A dedicated port (STRATA_DEDICATED_PORTS in server.py, this module's "pin") names its replica: the client asked
        for that card, so the request waits for it and never moves."""
        from serve.server import EngineDied     # imported here: serve.server imports this module
        conv = getattr(cancel, "_strata_conversation", None)
        t = Ticket(ids, time.monotonic(), getattr(self.pin, "index", None), self.router.held(conv) if conv else None)
        with self.cv:
            self.sched.submit(t)
            try:
                while True:
                    alive = self._alive()
                    if not alive:
                        raise EngineDied("no replica is running; this request was not sent")
                    if len(alive) < len(self.engines):
                        self.heal()
                    self._dispatch(alive)
                    if t.replica is not None:
                        break
                    if cancel is not None and cancel.is_set():
                        return None
                    due = self.sched.next_wake(time.monotonic(), alive)
                    self.cv.wait(WAIT_POLL_S if due is None else min(WAIT_POLL_S, due))
            finally:
                if t.replica is None:
                    self.sched.withdraw(t)
        if conv:
            self.router.bind(conv, t.replica)
        return t

    def _release(self, t: Ticket, measured: bool) -> None:
        with self.cv:
            self.sched.finish(t, time.monotonic(), measured)
            self._dispatch(self._alive())
            self.cv.notify_all()                # a session save waits for this replica to empty

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        now = time.monotonic()
        for k, e in enumerate(self.engines):
            if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing.get(k):
                continue
            if now < self.heal_at[k]:
                continue
            self.heal_at[k] = now + HEAL_COOLDOWN_S
            self.healing[k] = True

            def run(k=k):
                try:
                    print(f"[strata] replica {k} had stopped; starting it again", flush=True)
                    self.engines[k].restart()
                    self.router.forget(k)
                    print(f"[strata] replica {k} is running again", flush=True)
                except BaseException as x:       # noqa: BLE001
                    print(f"[strata] replica {k} did not start again: {x}", flush=True)
                finally:
                    self.healing[k] = False
            threading.Thread(target=run, daemon=True).start()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        from serve.server import EngineNonFinite   # imported here: serve.server imports this module
        t = self._admit(ids, cancel)
        if t is None:
            return
        k = t.replica
        e = self.engines[k]
        self.tl.cur = e
        produced: list[int] = []
        nonfinite = False
        try:
            selected = getattr(cancel, "_strata_replica_selected", None)
            if selected is not None:
                selected(k)
            self.router.note(k, ids)
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for tok in gen:
                if tok is not None:
                    produced.append(tok)
                yield tok
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            if nonfinite:                       # #879: the engine dropped every cached conversation
                self.router.forget(k)
            elif produced:
                self.router.note(k, list(ids) + produced)
            self._release(t, measured=bool(produced))
