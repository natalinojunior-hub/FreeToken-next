"""In-place expert residency (CUDA VMM): the backed prefix of every pool moves without moving
an address, so resident experts stay valid across a grow and the kept prefix across a shrink."""

import pytest
import torch
from freetoken.moe import vmm
from freetoken.moe.offload_cache import OffloadMoeCache

L, E = 4, 8
GU = [(4, 64)] * 3 + [(4, 128)]  # target rows 256 B, draft 512 B
DN = [(2, 64)] * 3 + [(2, 128)]

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and vmm.supported(torch.device("cuda"))), reason="needs VMM"
)


def _cache(size: int, vmm_rows: int) -> OffloadMoeCache:
    def bank(shapes, tag):
        out = []
        for layer, shape in enumerate(shapes):
            t = torch.empty(E, *shape, dtype=torch.uint8)
            for e in range(E):
                t[e].fill_(tag + layer * E + e)
            out.append(t.pin_memory())
        return out

    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=size,
        device=torch.device("cuda"),
        quant_format="gguf",
        gguf_expert_types=[(23, 20)] * 3 + [(8, 8)],
        min_pool_rows=2,
        vmm_rows=vmm_rows,
    )
    cache.set_bank_sources({"gate_up": bank(GU, 0), "down": bank(DN, 100)})
    return cache


def _resident(cache, layer: int, experts: list[int]) -> list[int]:
    ids = torch.tensor(experts, dtype=torch.int32, device="cuda")
    cache.ensure_experts(layer, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    p = cache.pool_of_layer[layer]
    slots = ids.tolist()
    assert all(0 <= s < cache.live_caps[p] for s in slots), (slots, cache.live_caps)
    for e, s in zip(experts, slots):
        assert int(cache.bank_views(layer_id=layer)[0][s, 0, 0]) == layer * E + e
        assert int(cache.bank_views(layer_id=layer)[1][s, 0, 0]) == 100 + layer * E + e
    return slots


def _consistent(cache) -> None:
    ids = cache.id_of_slot.tolist()
    for layer in range(L):
        p = cache.pool_of_layer[layer]
        for e, slot in enumerate(cache.slot_for_id[layer].tolist()):
            if slot >= 0:
                assert slot < cache.live_caps[p]
                assert ids[cache._pool_starts[p] + slot] == layer * E + e


def test_grow_keeps_residents_and_shrink_keeps_the_prefix():
    cache = _cache(12, vmm_rows=30)
    assert cache.resident_rows < cache.cache_size
    cache.reset()  # must leave the unbacked tail blocked
    before = cache.expert_pool_bytes
    a = _resident(cache, 0, [1, 2, 3])
    grown = cache.set_live(30)
    assert grown > 12 and cache.expert_pool_bytes >= before  # tiny rows: one granule each
    assert _resident(cache, 0, [1, 2, 3]) == a  # warm: same slots, no reload
    _resident(cache, 0, [0, 4, 5, 6, 7])
    _resident(cache, 1, list(range(E)))
    _resident(cache, 2, list(range(E)))
    _consistent(cache)
    cache.set_live(12)
    assert cache.expert_pool_bytes == before
    _consistent(cache)
    _resident(cache, 2, [0, 1, 2, 3, 4, 5, 6, 7])  # evictions stay inside the live prefix
    _consistent(cache)


def test_staging_and_reset_after_resize():
    cache = _cache(12, vmm_rows=30)
    assert 1 in cache._staging  # the draft pool plans below one layer
    cache.set_live(30)
    cache.set_live(12)
    cache.materialize_layer(3)
    cache.copy_missing()
    torch.cuda.synchronize()
    staged = cache.bank_views(E, layer_id=3)
    assert [int(staged[0][e, 0, 0]) for e in range(E)] == [3 * E + e for e in range(E)]
    cache.reset()
    for (ids, usage), live in zip(cache._pool_state, cache.live_caps):
        assert (usage[live:] == cache._BLOCKED_USAGE).all() and (ids[live:] == -1).all()
    _resident(cache, 3, [0, 1])
    _consistent(cache)


def test_set_live_shrink_never_grows_a_pool(monkeypatch):
    """Apportionment flips must not turn a shrink into a per-pool grow: growing needs a
    cuMemCreate the pressure prompting the shrink may not have (128K cert worker death)."""
    cache = _cache(12, vmm_rows=30)
    cache.set_live(30)
    live_before = list(cache.live_caps)
    arena = cache._vmm_arenas[0]
    backed_p1_before = arena.backed[1]
    # A shrink split that hands pool 1 MORE rows than it holds (largest-remainder flip).
    flipped = [live_before[0] - 3, live_before[1] + 1]
    assert sum(flipped) < sum(live_before)
    monkeypatch.setattr(cache, "_live_caps_for", lambda pools, caps, size: list(flipped))
    grown = cache.set_live(sum(flipped))
    # clamped per pool, the shrink frees at least what was asked (never grows pool 1)
    assert grown == flipped[0] + live_before[1]
    assert cache.live_caps == [flipped[0], live_before[1]], "pool 1 grew during a shrink"
    assert arena.backed[1] == backed_p1_before, "pool 1 backing grew during a shrink"
    _consistent(cache)


def test_vmm_mtp_draft_bank_prefill_boundary():
    """Draft-only layers (prefill_moe_layers) must not force prefill staging reservations in pool 0."""
    def bank(shapes, tag):
        out = []
        for layer, shape in enumerate(shapes):
            t = torch.empty(E, *shape, dtype=torch.uint8)
            for e in range(E):
                t[e].fill_(tag + layer * E + e)
            out.append(t.pin_memory())
        return out

    # Pool 0 has 3 layers at 256 B / row; layer 3 is a draft layer at 1024 B / row.
    # Staging layer 3 in pool 0 would require E * 1024 // 256 = 32 rows, exceeding pool 0 cap.
    wide_gu = [(4, 64)] * 3 + [(4, 256)]
    wide_dn = [(2, 64)] * 3 + [(2, 256)]
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=12,
        device=torch.device("cuda"),
        quant_format="gguf",
        gguf_expert_types=[(23, 20)] * 3 + [(8, 8)],
        min_pool_rows=2,
        vmm_rows=20,
        prefill_moe_layers=3,  # layer 3 is draft-only
    )
    cache.set_bank_sources({"gate_up": bank(wide_gu, 0), "down": bank(wide_dn, 100)})
    assert cache._vmm_arenas, "VMM must succeed with draft bank boundary"
    assert 1 not in cache._staged, "draft pool must not be marked staged"
    grown = cache.set_live(20)
    assert grown >= 12
    cache.reset()
    _consistent(cache)
