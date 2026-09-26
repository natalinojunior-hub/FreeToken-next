"""Mixed-geometry expert cache: one byte budget split into per-geometry pools.

Each distinct per-layer row geometry owns an LRU slot range sized by its layer count, so a
one-layer bank at a wide quant (a GGUF MTP draft's own Q8_0 experts) no longer costs a row
for every resident slot of the target.
"""

import pytest
import torch

from freetoken.engine.cache_budget import (
    ExpertPool,
    expert_bytes_per_slot,
    expert_cache_bytes,
    expert_pools,
    expert_rows_bounds,
    max_expert_rows,
    pool_capacities,
    pool_layout,
    pool_staging_fits,
)
from freetoken.moe.offload_cache import OffloadMoeCache

# 3 "target" layers + 1 "draft" layer with 4x wider gate_up and down rows.
L, E, FLOOR = 4, 8, 2
GU = [(4, 64)] * 3 + [(4, 256)]  # row bytes 256 / 1024
DN = [(2, 64)] * 3 + [(2, 256)]  # row bytes 128 / 512


def _sources(pin: bool = False):
    def bank(shapes, tag):
        layers = []
        for l, shape in enumerate(shapes):
            t = torch.empty(E, *shape, dtype=torch.uint8)
            for e in range(E):
                t[e].fill_(tag + l * E + e)  # fingerprint: tag + flat expert id
            layers.append(t.pin_memory() if pin else t)
        return layers

    return {"gate_up": bank(GU, 0), "down": bank(DN, 100)}


def _cache(cache_size: int, device: str = "cpu") -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=torch.device(device),
        quant_format="gguf",
        gguf_expert_types=[(23, 20)] * 3 + [(8, 8)],
        min_pool_rows=FLOOR,
    )
    cache.set_bank_sources(_sources(pin=device == "cuda"))
    return cache


def test_pools_group_layers_by_geometry_largest_first():
    pools = expert_pools(_sources())
    assert pools == [ExpertPool((0, 1, 2), (256, 128)), ExpertPool((3,), (1024, 512))]


def test_capacities_share_rows_per_layer_and_clamp_each_pool():
    pools = expert_pools(_sources())
    # share 6/layer -> [18, 6], remainder 2 to the largest unsaturated pool
    assert pool_capacities(pools, E, 26, FLOOR) == [20, 6]
    # the draft pool never exceeds one full layer however many rows are offered
    caps = pool_capacities(pools, E, 1000, FLOOR)
    assert caps == [3 * E, E]
    # tiny budgets keep every pool at its decode floor
    assert pool_capacities(pools, E, 0, FLOOR) == [FLOOR, FLOOR]
    # ...but "no cache" prices at zero (the planner's residual ledger asks for 0 slots)
    assert expert_cache_bytes(pools, E, 0, FLOOR) == 0


def test_one_layer_bank_cannot_take_target_sized_capacity():
    """48 target layers + a one-layer draft: the draft's pool is bounded by one layer and by
    its per-layer share, and the whole cache costs far less than one row of every geometry
    per slot (the old layout)."""
    target = ExpertPool(tuple(range(48)), (1408000, 921600))
    draft = ExpertPool((48,), (3481600, 1740800))
    pools, E_, floor = [target, draft], 512, 80
    lo, hi = expert_rows_bounds(pools, E_, floor)
    budget = 6540 << 20
    rows = max_expert_rows(pools, E_, budget, floor, max(lo, E_), hi)
    caps = pool_capacities(pools, E_, rows, floor)
    assert caps[1] <= E_ and caps[1] < caps[0] // 20
    assert expert_cache_bytes(pools, E_, rows, floor) <= budget
    old_slots = budget // sum(target.row_bytes + draft.row_bytes)
    assert caps[0] > 2 * old_slots


def test_byte_accounting_is_exact_and_aligned():
    pools = expert_pools(_sources())
    caps = pool_capacities(pools, E, 26, FLOOR)
    offsets, ends = pool_layout(pools, caps)
    assert offsets == [[0, 0], [20 * 256, 20 * 128]]
    assert ends == [20 * 256 + 6 * 1024, 20 * 128 + 6 * 512]
    assert expert_cache_bytes(pools, E, 26, FLOOR) == sum(ends)
    # the draft pool (6 < E rows) stages a whole layer at the front of each arena
    assert pool_staging_fits(pools, caps, E, ends)
    assert not pool_staging_fits(pools, [2, 2], E, pool_layout(pools, [2, 2])[1])
    lo, hi = expert_rows_bounds(pools, E, FLOOR)
    assert hi == L * E
    small = pool_capacities(pools, E, lo, FLOOR)
    assert pool_staging_fits(pools, small, E, pool_layout(pools, small)[1])
    below = pool_capacities(pools, E, lo - 1, FLOOR)
    assert not pool_staging_fits(pools, below, E, pool_layout(pools, below)[1])


def test_budget_search_never_exceeds_budget():
    pools = expert_pools(_sources())
    lo, hi = expert_rows_bounds(pools, E, FLOOR)
    for budget in range(expert_cache_bytes(pools, E, lo, FLOOR), 40_000, 997):
        rows = max_expert_rows(pools, E, budget, FLOOR, lo, hi)
        assert expert_cache_bytes(pools, E, rows, FLOOR) <= budget


def test_cache_allocates_pool_views_and_matches_budget_bytes():
    cache = _cache(26)
    assert cache.pool_caps == [20, 6] and cache.cache_size == 26
    assert cache.pool_of_layer == [0, 0, 0, 1]
    v0, v3 = cache.bank_views(layer_id=0), cache.bank_views(layer_id=3)
    assert [tuple(v.shape) for v in v0] == [(20, 4, 64), (20, 2, 64)]
    assert [tuple(v.shape) for v in v3] == [(6, 4, 256), (6, 2, 256)]
    assert cache.expert_pool_bytes == expert_cache_bytes(cache.pools, E, 26, FLOOR)
    assert cache.expert_pool_bytes < 26 * expert_bytes_per_slot(cache.bank_sources)
    # a whole-layer view of the sub-layer pool is the staging window, not out-of-range rows
    s3 = cache.bank_views(E, layer_id=3)
    assert [tuple(v.shape) for v in s3] == [(E, 4, 256), (E, 2, 256)]
    assert cache.pool_state(3)[0].numel() == 6


def test_cache_rejects_sizes_whose_staging_window_does_not_fit():
    with pytest.raises(ValueError, match="cannot stage a prefill layer"):
        _cache(8)
    cache = _cache(26)
    with pytest.raises(ValueError, match="cannot stage a prefill layer"):
        cache.validate_rebuild(8)
    assert cache.pool_caps == [20, 6]  # rejected before any teardown


def test_hybrid_cpu_lru_stays_inside_the_layer_pool():
    from freetoken.moe.offload_kernels import _ensure_experts_hybrid_cpu

    cache = _cache(26)
    for step in range(3):
        ids = torch.tensor([step, step + 3], dtype=torch.int32)
        _ensure_experts_hybrid_cpu(cache, 3, ids, max_fetch=8, frac_q16=0)
        assert all(0 <= s < 6 for s in ids.tolist())
    pool_ids = cache.pool_state(3)[0]
    assert sorted(i for i in pool_ids.tolist() if i >= 0) == [3 * E + e for e in range(6)]
    assert (cache.pool_state(0)[0] == -1).all()  # the target pool is untouched


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU slot cache")


@cuda
def test_decode_eviction_prefill_staging_rebuild_and_reset():
    cache = _cache(26, "cuda")
    cache.reset()
    gu = lambda layer: cache.bank_views(layer_id=layer)[0]  # noqa: E731

    def resident(layer: int, experts: list[int]) -> list[int]:
        ids = torch.tensor(experts, dtype=torch.int32, device="cuda")
        cache.ensure_experts(layer, ids)
        cache.copy_missing()
        torch.cuda.synchronize()
        slots = ids.tolist()
        assert all(0 <= s < cache.pool_caps[cache.pool_of_layer[layer]] for s in slots)
        # the bytes in each slot are that exact (layer, expert)'s, in both banks
        for e, s in zip(experts, slots):
            assert int(gu(layer)[s, 0, 0]) == layer * E + e
            assert int(cache.bank_views(layer_id=layer)[1][s, 0, 0]) == 100 + layer * E + e
        return slots

    resident(0, [1, 2])
    resident(3, [0, 1, 2, 3, 4, 5])  # fills the 6-row draft pool
    resident(3, [6, 7])  # evicts the two least recently used draft experts
    assert cache.slot_for_id[3].tolist().count(-1) == 2
    resident(3, [2, 3, 4, 5, 6, 7])  # the survivors still hit with correct bytes
    resident(0, [1, 2])

    # prefill of the draft layer stages it at the front of each arena
    cache.materialize_layer(3)
    cache.copy_missing()
    torch.cuda.synchronize()
    staged = cache.bank_views(E, layer_id=3)
    assert [int(staged[0][e, 0, 0]) for e in range(E)] == [3 * E + e for e in range(E)]
    # every target row the window overlaid lost its mapping; no stale hit remains
    for s in cache._staging[1][1].tolist():
        assert int(cache.id_of_slot[s]) == -1
    ids = cache.id_of_slot.tolist()
    for layer in range(L):
        for e, slot in enumerate(cache.slot_for_id[layer].tolist()):
            if slot >= 0:
                start = cache._pool_starts[cache.pool_of_layer[layer]]
                assert ids[start + slot] == layer * E + e
    resident(0, [1, 2])  # reloads what the staging overlaid
    resident(3, [0, 7])

    # a full-layer pool materializes into its own first E slots and keeps them resident
    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert cache.slot_for_id[1].tolist() == list(range(E))
    resident(1, [5])

    cache.rebuild(30)
    assert cache.cache_size == sum(cache.pool_caps) == 30
    assert (cache.slot_for_id == -1).all()
    assert cache.expert_pool_bytes == expert_cache_bytes(cache.pools, E, 30, FLOOR)
    resident(3, [1, 2, 3])
    resident(2, [7])
    cache.reset()
    assert (cache.id_of_slot == -1).all() and (cache.slot_for_id == -1).all()
    resident(2, [7])


def _oc(override: str, cache_size: int, device: str = "cpu") -> OffloadMoeCache:
    """Build a cache with an explicit --moe-pool-caps override (bank sources attached)."""
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=torch.device(device),
        quant_format="gguf",
        gguf_expert_types=[(23, 20)] * 3 + [(8, 8)],
        min_pool_rows=FLOOR,
        pool_caps_override=override,
    )
    cache.set_bank_sources(_sources(pin=device == "cuda"))
    return cache


# --moe-pool-caps: the override names the SHAPE of the split; cache_size (the planner's byte
# budget) stays the authority, so caps always sum to cache_size and a rebuild keeps proportions
# instead of silently reverting to uniform (campaign23 A3/A4).


def test_override_caps_are_proportional_to_the_budget_authority():
    cache = _oc("20,6", 26)  # sum == cache_size: identity split, no overcommit
    assert cache.pool_caps == [20, 6] and cache.cache_size == 26
    pools = expert_pools(_sources())
    # sum(override) > cache_size: scaled DOWN to the budget, not allowed to override it
    assert cache._override_caps(pools, 20) == [15, 5]  # 20,6 -> sum 20, ratio preserved
    assert cache._override_caps(pools, 13) == [10, 3]  # pure math, staging checked elsewhere
    # deterministic
    assert cache._override_caps(pools, 20) == cache._override_caps(pools, 20)


def test_override_build_never_silently_overcommits_the_budget():
    # regression: sum(want)=32 but the planned budget is 26 -> must allocate 26, not 32
    cache = _oc("24,8", 26)
    assert cache.cache_size == 26 and sum(cache.pool_caps) == 26
    assert cache.pool_caps == [20, 6]


def test_override_rebuild_keeps_proportions_and_respects_the_target():
    cache = _oc("20,6", 26)
    cache.rebuild(20)  # a guard shrink must actually shrink, proportionally
    assert cache.cache_size == sum(cache.pool_caps) == 20
    assert cache.pool_caps == [15, 5]
    cache.rebuild(26)  # ...and regrow back to the operator's proportions
    assert cache.cache_size == sum(cache.pool_caps) == 26 and cache.pool_caps == [20, 6]


def test_override_validation_domain():
    pools = expert_pools(_sources())
    with pytest.raises(ValueError, match="entries but the banks have"):
        _oc("20,6,1", 27)  # wrong pool count
    with pytest.raises(ValueError, match="must be >= 0"):
        _oc("-1,6", 26)._override_caps(pools, 26)
    with pytest.raises(ValueError, match="exceeds its pool"):
        _oc("100,6", 26)._override_caps(pools, 26)  # 100 > pool0 max (3 layers * E = 24)
    with pytest.raises(ValueError, match="decode"):
        _oc("20,6", 26)._override_caps(pools, 3)  # below the sum of per-pool floors (2+2)


@cuda
def test_hybrid_gpu_kernel_matches_cpu_reference_on_a_sub_layer_pool():
    from freetoken.moe.offload_kernels import (
        _ensure_experts_hybrid_cpu,
        _ensure_experts_hybrid_gpu,
    )

    gpu, cpu = _cache(26, "cuda"), _cache(26)
    gpu.reset()
    for step in ([0, 3], [1, 4, 5], [6, 7], [0, 2, 3], [5, 1]):
        ids_g = torch.tensor(step, dtype=torch.int32, device="cuda")
        ids_c = torch.tensor(step, dtype=torch.int32)
        _ensure_experts_hybrid_gpu(gpu, 3, ids_g, 2, 0)
        _ensure_experts_hybrid_cpu(cpu, 3, ids_c, 2, 0)
        assert ids_g.tolist() == ids_c.tolist()
        assert all(-1 <= s < 6 for s in ids_g.tolist())
        assert gpu.pool_state(3)[0].tolist() == cpu.pool_state(3)[0].tolist()
        assert gpu.slot_for_id.tolist() == cpu.slot_for_id.tolist()
    assert (gpu.pool_state(0)[0] == -1).all()
