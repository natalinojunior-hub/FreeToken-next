"""A failed free-list allocation retains the speculative ownership ceiling."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.cache import CacheManager


def test_spec_page_return_oom_retries_without_losing_or_double_freeing_pages():
    req = SimpleNamespace(table_idx=0, alloc_page_bound=5)
    free_slots = torch.tensor([8, 9], dtype=torch.int32)
    attempts = []

    def free(indices):
        nonlocal free_slots
        attempts.append(indices.tolist())
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("free-list allocation")
        free_slots = torch.cat([free_slots, indices])

    cache = SimpleNamespace(
        page_size=1,
        page_table=torch.arange(5, dtype=torch.int32).view(1, 5),
        swa_paged=False,
        _free=free,
    )
    with pytest.raises(torch.OutOfMemoryError, match="free-list allocation"):
        CacheManager.free_spec_reject(cache, req, keep_len=3, alloc_len=req.alloc_page_bound)
    assert req.alloc_page_bound == 5 and free_slots.tolist() == [8, 9]
    CacheManager.free_spec_reject(cache, req, keep_len=3, alloc_len=req.alloc_page_bound)
    assert req.alloc_page_bound == 3 and free_slots.tolist() == [8, 9, 3, 4]
    CacheManager.free_spec_reject(cache, req, keep_len=3, alloc_len=req.alloc_page_bound)
    assert attempts == [[3, 4], [3, 4]]
