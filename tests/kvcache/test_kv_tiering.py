import pytest
import torch

from freetoken.kvcache.kv_tiering import KVPage, KVPagePool, Residency, tiering_safe


def page(i):
    return KVPage(i, i * 4, host=torch.zeros(8), device=torch.zeros(8))


def test_versioned_ownership_and_duplicate_release():
    pool = KVPagePool([page(0)], 1)
    p = pool.prefetch(0, "r1")
    generation = p.generation
    assert p.residency is Residency.RESIDENT
    assert not pool.evict(0, "r2", generation)
    assert pool.evict(0, "r1", generation)
    assert p.generation > generation


def test_pool_backpressure_and_cancel_cleanup():
    pool = KVPagePool([page(0), page(1)], 1)
    pool.prefetch(0, "r1")
    with pytest.raises(MemoryError):
        pool.admit(1, "r2")
    pool.cancel("r1")
    pool.prefetch(1, "r2")
    assert pool.table.get(1).residency is Residency.RESIDENT


def test_wrong_position_or_backing_is_rejected():
    p = page(0)
    p.device = torch.zeros(4)
    with pytest.raises(ValueError):
        KVPagePool([p], 1).prefetch(0, "r")


def test_capability_gate_keeps_fallback():
    assert tiering_safe(cuda_graph=True, stable_indirection=True, pinned_host=True)
    assert not tiering_safe(cuda_graph=False, stable_indirection=True, pinned_host=True)
