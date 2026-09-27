"""LRU-3 eviction (FREETOKEN_MOE_EVICT=lru3): kernel vs an exact sequential reference.

The reference mirrors ``_lru3_ensure_kernel`` semantics one-for-one: one step per call,
slots touched by this call never victims, misses installed in ascending global-id order
against the ascending-key victims, key = ``g3 > 0 ? g3 : -max(usage, 1)`` over the
per-EXPERT ghost history, empty slots coldest, ``usage >= 1<<40`` (VMM-unbacked) untouchable.
"""

import pytest
import torch

import freetoken.moe.offload_kernels as ok
from freetoken.moe.offload_cache import OffloadMoeCache

L, E = 3, 8
CAP = 12
BLOCKED = 1 << 40
EMPTY_KEY = -(1 << 41)
BIG = 1 << 41

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU slot cache")


def _cache(device: str = "cuda", cache_size: int = CAP) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=torch.device(device),
        quant_format="bf16",
    )
    srcs = {
        "gate_up": [torch.randn(E, 4, 8, dtype=torch.bfloat16) for _ in range(L)],
        "down": [torch.randn(E, 8, 4, dtype=torch.bfloat16) for _ in range(L)],
    }
    if device == "cuda":
        srcs = {k: [t.pin_memory() for t in v] for k, v in srcs.items()}
    cache.set_bank_sources(srcs)
    return cache


def lru3_ref(cache: OffloadMoeCache, layer_id: int, experts: list[int]) -> None:
    """Exact sequential mirror of the kernel; mutates ``cache`` like ``ensure_experts``."""
    base = layer_id * cache.num_experts
    ids_t, usage_t = cache.pool_state(layer_id)
    flat = cache.slot_for_id.view(-1)
    ghost = cache.ghost_hist
    nc = ids_t.numel()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)

    q = [base + e for e in experts]
    slots = [int(flat[g].item()) for g in q]
    out = list(slots)

    # hits: usage bump + one history shift per distinct expert (duplicate lanes in the
    # kernel load the same pre-state and store identical values -> idempotent).
    shifted = set()
    for g, s in zip(q, slots):
        if s >= 0 and g not in shifted:
            shifted.add(g)
            usage_t[s] = step
            g1, g2 = int(ghost[0, g].item()), int(ghost[1, g].item())
            ghost[2, g] = g2
            ghost[1, g] = g1
            ghost[0, g] = step
    first_seen = {}
    for i, g in enumerate(q):
        first_seen.setdefault(g, i)
    missing_ids = sorted({g for i, g in enumerate(q) if slots[i] == -1})
    rank = {g: r for r, g in enumerate(missing_ids)}
    num_missing = len(missing_ids)
    cache.num_indices.fill_(num_missing)

    if num_missing:
        packed = []
        for s in range(nc):
            oid, u = int(ids_t[s].item()), int(usage_t[s].item())
            if u >= BLOCKED or u == step:
                key = BIG
            elif oid < 0:
                key = EMPTY_KEY
            else:
                g3 = int(ghost[2, oid].item())
                key = g3 if g3 > 0 else -max(u, 1)
            packed.append(((key + (1 << 42)) << 12) | s)
        mx = (1 << 63) - 1
        for r, g in enumerate(missing_ids):
            v = min(range(nc), key=lambda s: packed[s])
            packed[v] = mx
            old = int(ids_t[v].item())
            if old >= 0:
                flat[old] = -1
            ids_t[v] = g
            flat[g] = v
            usage_t[v] = step
            g1, g2 = int(ghost[0, g].item()), int(ghost[1, g].item())
            ghost[2, g] = g2
            ghost[1, g] = g1
            ghost[0, g] = step
            cache.evict_slots[r] = v
            cache.src_indices[r] = g - base
            for i in range(len(q)):
                if q[i] == g:
                    out[i] = v
    if cache.collect_stats:
        cache.lru_stats[layer_id, 0] += len(set(q))  # ACTIVE = distinct queried ids
        cache.lru_stats[layer_id, 1] += num_missing
        cache.lru_stats[layer_id, 2] += 1
    return out, rank, num_missing


def _run_pair(workloads: list[list[int]], layers: list[int] | None = None, seed=0):
    """Feed identical request streams to a kernel cache and a reference cache."""
    torch.manual_seed(seed)
    kc, rc = _cache(), _cache()
    # identical random bank bytes are irrelevant (bookkeeping only); state starts cold
    layers = layers or [0] * len(workloads)
    for layer, experts in zip(layers, workloads):
        ids_k = torch.tensor(experts, dtype=torch.int32, device="cuda")
        out_ref, _, nmiss_ref = lru3_ref(rc, layer, experts)
        ok._ensure_experts_lru3(kc, layer, ids_k)
        torch.cuda.synchronize()
        assert ids_k.tolist() == out_ref, f"out rewrite mismatch at layer {layer}"
        assert int(kc.num_indices.item()) == nmiss_ref
        assert int(kc.step.item()) == int(rc.step.item())
        torch.testing.assert_close(kc.slot_for_id, rc.slot_for_id)
        torch.testing.assert_close(kc.id_of_slot, rc.id_of_slot)
        torch.testing.assert_close(kc.usage, rc.usage)
        torch.testing.assert_close(kc.ghost_hist, rc.ghost_hist)
        n = int(kc.num_indices.item())
        if n:
            torch.testing.assert_close(kc.evict_slots[:n], rc.evict_slots[:n])
            torch.testing.assert_close(kc.src_indices[:n], rc.src_indices[:n])
    return kc


@cuda
def test_lru3_matches_reference_random_workload():
    wl = []
    for i in range(60):
        n = int(torch.randint(1, E + 1, (1,)).item())
        wl.append(torch.randperm(E)[:n].sort().values.tolist())
    _run_pair(wl, layers=[i % L for i in range(60)], seed=1)


@cuda
def test_lru3_duplicates_collapse_to_one_copy():
    wl = [[3, 3, 5, 3, 5, 7]] * 8
    kc = _run_pair(wl, layers=[0] * 8, seed=2)
    assert int(kc.num_indices.item()) <= 4


@cuda
def test_lru3_never_evicts_slots_touched_this_call():
    """cap=3, all residents equally old: a hit on the would-be coldest slot survives
    the same call's installs (usage == step protection); the next-coldest die instead."""
    cache = _cache()
    ids_t, usage_t = cache.pool_state(0)
    ids_t[3:].fill_(-1)
    usage_t[3:].fill_(BLOCKED)  # live = 3
    t, dev = torch.int32, "cuda"
    ok._ensure_experts_lru3(cache, 0, torch.tensor([0, 1, 2], dtype=t, device=dev))
    torch.cuda.synchronize()
    ok._ensure_experts_lru3(cache, 0, torch.tensor([0, 3, 4], dtype=t, device=dev))
    torch.cuda.synchronize()
    assert int(cache.slot_for_id[0, 0].item()) >= 0, "hit this call was evicted"
    assert int(cache.slot_for_id[0, 1].item()) == -1
    assert int(cache.slot_for_id[0, 2].item()) == -1
    assert int(cache.slot_for_id[0, 3].item()) >= 0
    assert int(cache.slot_for_id[0, 4].item()) >= 0


@cuda
def test_lru3_blocked_slots_are_never_victims():
    """VMM-unbacked rows (usage = 1<<40, id -1) must never receive an install."""
    cache = _cache()
    ids_t, usage_t = cache.pool_state(0)
    live = 5
    ids_t[live:].fill_(-1)
    usage_t[live:].fill_(BLOCKED)
    for i in range(10):
        experts = torch.randperm(E)[:4].sort().values.to(torch.int32).cuda()
        ok._ensure_experts_lru3(cache, 0, experts)
        torch.cuda.synchronize()
        assert (ids_t[live:] == -1).all(), "install landed in an unbacked slot"
        assert (usage_t[live:] == BLOCKED).all()
    assert int((ids_t[:live] >= 0).sum().item()) == live


@cuda
def test_lru3_victim_is_third_most_recent_ref():
    """Hand-built ref-depth ladder: g3(e0)=3 > g3(e1)=2 > g3(e2)=1 -> the miss eats e2."""
    cache = _cache()
    ids_t, usage_t = cache.pool_state(0)
    ids_t[3:].fill_(-1)
    usage_t[3:].fill_(BLOCKED)  # live = 3
    t, dev = torch.int32, "cuda"

    def call(*experts):
        ok._ensure_experts_lru3(cache, 0, torch.tensor(list(experts), dtype=t, device=dev))
        torch.cuda.synchronize()

    call(0, 1, 2)  # step 1: install, g1=1
    call(0, 1, 2)  # step 2: g2=1, g1=2
    call(0, 1, 2)  # step 3: g3=1 for all
    call(0)  # step 4: g3(e0)=2
    call(0)  # step 5: g3(e0)=3
    call(1)  # step 6: g3(e1)=2
    g3 = cache.ghost_hist[2, 0:3].tolist()
    assert g3 == [3, 2, 1]
    call(5)  # step 7: one miss -> coldest g3 (e2) loses its slot
    assert int(cache.slot_for_id[0, 2].item()) == -1, "wrong victim evicted"
    assert int(cache.slot_for_id[0, 0].item()) >= 0
    assert int(cache.slot_for_id[0, 1].item()) >= 0
    assert int(cache.slot_for_id[0, 5].item()) >= 0


@cuda
def test_lru3_ghost_history_survives_eviction():
    """A re-missed expert re-enters with its pre-eviction depth (g3 keeps ordering it)."""
    cache = _cache()
    ids_t, usage_t = cache.pool_state(0)
    ids_t[4:].fill_(-1)
    usage_t[4:].fill_(BLOCKED)  # live = 4
    t, dev = torch.int32, "cuda"
    for _ in range(3):  # give experts 0..3 three refs each
        ok._ensure_experts_lru3(cache, 0, torch.tensor([0, 1, 2, 3], dtype=t, device=dev))
        torch.cuda.synchronize()
    ok._ensure_experts_lru3(cache, 0, torch.tensor([7], dtype=t, device=dev))  # evicts coldest g3
    torch.cuda.synchronize()
    victim = [e for e in range(4) if int(cache.slot_for_id[0, e].item()) == -1]
    assert len(victim) == 1
    v = victim[0]
    assert int(cache.ghost_hist[2, v].item()) > 0, "ghost g3 wiped on eviction"
    # re-miss v: installs with its history shifted, g3 stays > 0
    ok._ensure_experts_lru3(cache, 0, torch.tensor([v], dtype=t, device=dev))
    torch.cuda.synchronize()
    assert int(cache.ghost_hist[2, v].item()) > 0
    assert int(cache.slot_for_id[0, v].item()) >= 0


@cuda
def test_lru3_stats_match_reference():
    kc, rc = _cache(), _cache()
    kc.collect_stats = rc.collect_stats = True
    for i in range(12):
        experts = torch.randperm(E)[: 1 + i % E].sort().values.tolist()
        layer = i % L
        out_ref, _, _ = lru3_ref(rc, layer, experts)
        ids_k = torch.tensor(experts, dtype=torch.int32, device="cuda")
        ok._ensure_experts_lru3(kc, layer, ids_k)
        torch.cuda.synchronize()
    torch.testing.assert_close(kc.lru_stats, rc.lru_stats)


@cuda
def test_lru3_graph_capture_replay_matches_eager():
    eager = _cache()
    graphed = _cache()
    experts = torch.randperm(E).to(torch.int32).cuda()
    # warm up (JIT compile) on a side stream, then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ok._ensure_experts_lru3(graphed, 0, experts.clone())
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # reset graphed state so capture starts cold like eager
    graphed.reset()
    g = torch.cuda.CUDAGraph()
    ids_g = experts.clone()
    with torch.cuda.graph(g):
        ok._ensure_experts_lru3(graphed, 0, ids_g)
    for r in range(3):
        g.replay()  # capture does not execute: replay N == eager call N
        ids_e = experts.clone()
        ok._ensure_experts_lru3(eager, 0, ids_e)
        torch.cuda.synchronize()
        torch.testing.assert_close(graphed.slot_for_id, eager.slot_for_id)
        torch.testing.assert_close(graphed.id_of_slot, eager.id_of_slot)
        torch.testing.assert_close(graphed.usage, eager.usage)
        torch.testing.assert_close(graphed.ghost_hist, eager.ghost_hist)
        assert ids_g.tolist() == ids_e.tolist()


@cuda
def test_materialize_resets_lru3_history():
    cache = _cache()
    cache.ghost_hist.fill_(7)
    cache.materialize_layer(1)
    torch.cuda.synchronize()
    base = 1 * E
    assert (cache.ghost_hist[0, base : base + E] == cache.step).all()
    assert (cache.ghost_hist[1, base : base + E] == 0).all()
    assert (cache.ghost_hist[2, base : base + E] == 0).all()
    # other layers untouched
    assert int(cache.ghost_hist[0, 0].item()) == 7


@cuda
def test_reset_clears_ghost_history():
    cache = _cache()
    ok._ensure_experts_lru3(cache, 0, torch.arange(E, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    assert cache.ghost_hist.abs().sum() > 0
    cache.reset()
    torch.cuda.synchronize()
    assert int(cache.ghost_hist.abs().sum().item()) == 0
