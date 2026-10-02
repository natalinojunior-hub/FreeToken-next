"""Radix metadata allocation failures leave page ownership retryable."""

import pytest
import torch

from freetoken.kvcache.radix_cache import RadixCacheHandle
from freetoken.scheduler.cache import CacheManager
from tests.scheduler.test_commit_repoints_page_table import PROMPT, _admit, _pend


@pytest.mark.parametrize("lazy", [False, True])
def test_canonical_allocation_oom_commit_retry_preserves_unique_pages(monkeypatch, lazy):
    page_table = torch.zeros(4, 32, dtype=torch.int32)
    cache = CacheManager(32, 1, page_table, "radix")
    a = _admit(cache, page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    b = _admit(cache, page_table, 1, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    cache.cache_req(a, finished=False)
    before = cache.free_slots.clone()
    old_handle, refcount = b.cache_handle, b.cache_handle.node.ref_count
    original = RadixCacheHandle.get_matched_indices

    def fail(_):
        raise torch.OutOfMemoryError("canonical injection")

    monkeypatch.setattr(RadixCacheHandle, "get_matched_indices", fail)
    with pytest.raises(torch.OutOfMemoryError, match="canonical injection"):
        if lazy:
            with cache.lazy_free_region():
                cache.cache_req(b, finished=False)
        else:
            cache.cache_req(b, finished=False)
    assert b.cache_handle is old_handle and old_handle.node.ref_count == refcount
    assert torch.equal(cache.free_slots, before)
    assert set(page_table[1, :8].tolist()).isdisjoint(cache.free_slots.tolist())
    monkeypatch.setattr(RadixCacheHandle, "get_matched_indices", original)
    if lazy:
        with cache.lazy_free_region():
            cache.cache_req(b, finished=False)
    else:
        cache.cache_req(b, finished=False)
    assert cache.free_slots.numel() == torch.unique(cache.free_slots).numel()
    assert set(page_table[1, :8].tolist()).isdisjoint(cache.free_slots.tolist())
    assert torch.equal(page_table[1, :8], b.cache_handle.get_matched_indices())
    cache.prefix_cache.check_integrity()


def test_lazy_free_entry_oom_precedes_body_and_exit_never_allocates(monkeypatch):
    cache = CacheManager(8, 1, torch.zeros(1, 8, dtype=torch.int32), "radix")
    before = cache.free_slots
    original = torch.empty
    calls = []

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("region entry injection")

    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(torch.OutOfMemoryError, match="region entry injection"):
        with cache.lazy_free_region():
            calls.append("entered")
    assert not calls and cache.free_slots is before
    monkeypatch.setattr(torch, "empty", original)
    with cache.lazy_free_region():
        allocated = cache._allocate(2)
        monkeypatch.setattr(torch, "empty", fail)
        monkeypatch.setattr(torch, "cat", fail)
        cache._free(allocated)
    assert cache.free_slots.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]


@pytest.mark.parametrize("finished", [False, True])
def test_hybrid_staging_oom_before_frozen_and_live_donation_retries_exactly(monkeypatch, finished):
    from tests.scheduler.test_hybrid_cache_manager import _pool

    pool = _pool()
    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cache = CacheManager(32, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    req = _admit(cache, page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    before_free = cache.free_slots.clone()
    before_slots = pool._free_slots.copy()
    before_ping_pong = req.mamba_ping_pong
    old_handle = req.cache_handle
    original_clone = torch.Tensor.clone
    original_collect = cache.prefix_cache._collect_kv
    clones = []

    def fail_second_clone(tensor, *args, **kwargs):
        clones.append(tensor)
        if len(clones) == 2:
            raise torch.OutOfMemoryError("second donation image injection")
        return original_clone(tensor, *args, **kwargs)

    def fail_canonical(node):
        raise torch.OutOfMemoryError("hybrid canonical injection")

    if finished:
        monkeypatch.setattr(torch.Tensor, "clone", fail_second_clone)
    else:
        monkeypatch.setattr(cache.prefix_cache, "_collect_kv", fail_canonical)
    with pytest.raises(torch.OutOfMemoryError):
        cache.cache_req(req, finished=finished)
    assert req.cache_handle is old_handle
    assert req.mamba_ping_pong == before_ping_pong
    assert pool._free_slots == before_slots
    assert torch.equal(cache.free_slots, before_free)
    monkeypatch.setattr(torch.Tensor, "clone", original_clone)
    monkeypatch.setattr(cache.prefix_cache, "_collect_kv", original_collect)
    assert cache.prefix_cache.match_prefix(req.input_ids).cached_len == 0
    cache.cache_req(req, finished=finished)
    assert cache.free_slots.numel() == torch.unique(cache.free_slots).numel()
    assert len(pool._free_slots) == len(set(pool._free_slots))
    cache.prefix_cache.check_integrity()
    match = cache.prefix_cache.match_prefix(req.input_ids)
    assert match.cached_len == (8 if finished else 4)
    if finished:
        cache.check_integrity()  # idle-only: all request pages were finalized
        assert req.linear_slot_idx is None and req.mamba_ping_pong is None
    else:
        assert req.mamba_ping_pong[0] != before_ping_pong[0]
        assert req.mamba_last_track_seqlen is None


def test_hybrid_defers_donation_until_replacement_slot_is_available():
    from tests.scheduler.test_hybrid_cache_manager import _pool

    pool = _pool()
    cache = CacheManager(
        32, 1, torch.zeros(2, 32, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool
    )
    req = _admit(cache, cache.page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    held = pool.alloc(pool.num_free_slots)
    original = req.mamba_ping_pong
    cache.cache_req(req, finished=False)
    assert req.mamba_ping_pong == original and req.mamba_last_track_seqlen == 4
    assert cache.prefix_cache.match_prefix(req.input_ids).cached_len == 0
    pool.free(held)
    cache.cache_req(req, finished=False)
    assert req.mamba_ping_pong[0] != original[0]
    assert cache.prefix_cache.match_prefix(req.input_ids).cached_len == 4


def test_hybrid_partial_prefix_canonical_has_exact_matched_length():
    from tests.scheduler.test_hybrid_cache_manager import _pool

    pool = _pool()
    cache = CacheManager(
        32, 1, torch.zeros(2, 32, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool
    )
    reqs = []
    for index, ids in enumerate([PROMPT, [1, 2, 50, 51, 52, 53, 54, 55]]):
        req = _admit(cache, cache.page_table, index, ids, cache.match_req(_pend(ids)).cuda_handle)
        req.linear_slot_idx = pool.alloc(1)[0]
        req.mamba_ping_pong = tuple(pool.alloc(2))
        req.mamba_next_track_idx = 1
        req.mamba_last_track_seqlen = None if index == 0 else 4
        reqs.append(req)
    cache.cache_req(reqs[0], finished=True)
    own = cache.page_table[1, :8].clone()
    cache.cache_req(reqs[1], finished=False)
    match = cache.prefix_cache.match_prefix(reqs[1].input_ids)
    assert match.cached_len == match.kv_indices.numel() == 4
    assert torch.equal(match.kv_indices[:2], cache.page_table[0, :2])
    assert torch.equal(match.kv_indices[2:], own[2:4])
    assert torch.equal(cache.page_table[1, :4], match.kv_indices)
    assert set(own[:2].tolist()).issubset(cache.free_slots.tolist())
    assert set(cache.page_table[1, :8].tolist()).isdisjoint(cache.free_slots.tolist())
    cache.prefix_cache.check_integrity()


def test_hybrid_replacement_reservation_is_returned_if_insert_fails(monkeypatch):
    from tests.scheduler.test_hybrid_cache_manager import _pool

    pool = _pool()
    cache = CacheManager(
        32, 1, torch.zeros(2, 32, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool
    )
    req = _admit(cache, cache.page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    before = pool._free_slots.copy()

    def fail(*args, **kwargs):
        raise MemoryError("insert injection")

    monkeypatch.setattr(cache.prefix_cache, "insert", fail)
    with pytest.raises(MemoryError, match="insert injection"):
        cache.cache_req(req, finished=False)
    assert pool._free_slots == before
    assert req.mamba_last_track_seqlen == 4
