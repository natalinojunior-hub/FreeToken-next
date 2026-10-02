"""Eviction refuses allocation before relinquishing cached ownership."""

import pytest
import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.radix_cache import RadixPrefixCache
from freetoken.kvcache.swa_radix_cache import SWARadixCache
from freetoken.scheduler.cache import CacheManager
from tests.scheduler.test_commit_repoints_page_table import PROMPT, _admit, _pend


@pytest.mark.parametrize("mode", ["radix", "hybrid_full", "hybrid_mamba", "swa_full", "swa"])
def test_eviction_result_reservation_oom_keeps_tree_retryable(monkeypatch, mode):
    ids = torch.arange(4, dtype=torch.int32)
    values = torch.arange(4, dtype=torch.int32) + 10
    if mode == "radix":
        tree = RadixPrefixCache(torch.device("cpu"), page_size=1)
        tree.insert_prefix(ids, values)
        evict = lambda: tree.evict(4)
        match = lambda: tree.match_prefix(ids).cuda_handle.cached_len
    elif mode.startswith("hybrid"):
        tree = HybridRadixCache(torch.device("cpu"), page_size=1)
        tree.insert(ids, values, 7)
        evict = lambda: tree.evict_full(4) if mode == "hybrid_full" else tree.evict_mamba(1)
        match = lambda: tree.match_prefix(ids).cached_len
    else:
        tree = SWARadixCache(torch.device("cpu"), page_size=1, sliding_window_size=2)
        tree.insert(ids, values)
        evict = lambda: tree.evict_full(4) if mode == "swa_full" else tree.evict_swa(4)
        match = lambda: tree.match_prefix(ids).cached_len
    original_empty = torch.empty

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("eviction backing injection")

    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(torch.OutOfMemoryError, match="eviction backing injection"):
        evict()
    assert match() == 4
    monkeypatch.setattr(torch, "empty", original_empty)

    monkeypatch.setattr(torch, "cat", fail)
    result = evict()
    assert torch.equal(result if mode == "radix" else result.kv_indices, values)
    assert match() == 0


def test_lazy_region_entry_retries_only_after_accepted_recovery(monkeypatch):
    cache = CacheManager(8, 1, torch.zeros(1, 8, dtype=torch.int32), "radix")
    original = torch.empty
    attempts, recovered = [], []

    def allocate(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("metadata backing injection")
        return original(*args, **kwargs)

    cache.recover_oom = lambda error: recovered.append(str(error)) or True
    monkeypatch.setattr(torch, "empty", allocate)
    with cache.lazy_free_region():
        pass
    assert attempts == [1, 1] and recovered == ["metadata backing injection"]


def test_staged_radix_commit_retries_after_reclaimed_metadata_headroom(monkeypatch):
    from freetoken.kvcache.radix_cache import RadixCacheHandle

    cache = CacheManager(32, 1, torch.zeros(2, 32, dtype=torch.int32), "radix")
    a = _admit(cache, cache.page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    b = _admit(cache, cache.page_table, 1, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    cache.cache_req(a, finished=False)
    original = RadixCacheHandle.get_matched_indices
    attempts, recovered = [], []

    def canonical(handle):
        attempts.append(1)
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("canonical retry injection")
        return original(handle)

    monkeypatch.setattr(RadixCacheHandle, "get_matched_indices", canonical)
    cache.recover_oom = lambda error: recovered.append(str(error)) or True
    cache.cache_req(b, finished=False)
    assert attempts == [1, 1]
    assert recovered == ["canonical retry injection"]
    assert cache.free_slots.numel() == torch.unique(cache.free_slots).numel()
    assert set(cache.page_table[1, :8].tolist()).isdisjoint(cache.free_slots.tolist())


def test_manager_eviction_reserves_return_backing_before_tree_unlink(monkeypatch):
    cache = CacheManager(16, 1, torch.zeros(1, 16, dtype=torch.int32), "radix")
    req = _admit(cache, cache.page_table, 0, PROMPT, cache.match_req(_pend(PROMPT)).cuda_handle)
    cache.cache_req(req, finished=True)
    original = torch.empty

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("manager return reservation")

    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(torch.OutOfMemoryError, match="manager return reservation"):
        cache._allocate(12)
    assert cache.prefix_cache.match_prefix(req.input_ids).cuda_handle.cached_len == 8
    assert cache.free_slots.numel() == 8
    monkeypatch.setattr(torch, "empty", original)
    allocated = cache._allocate(12)
    assert allocated.numel() == torch.unique(allocated).numel() == 12
    assert set(allocated.tolist()).isdisjoint(cache.free_slots.tolist())
    assert cache.free_slots.numel() == 4
