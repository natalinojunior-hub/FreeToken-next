"""KV RAM tiering (engine/config.py kv_tiering="force"): the highest page ids are the
(slower) host RAM tier, so CacheManager._allocate must hand out the lowest ids (device
pages) first once host_pages > 0. Without a RAM tier, free_slots order must stay
byte-identical to the pre-tiering behavior (no sort)."""

from __future__ import annotations

import torch

from freetoken.scheduler.cache import CacheManager

WIDTH = 64
MAX_RUNNING = 4


def _page_table():
    return torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")


def test_allocate_prefers_device_pages_when_host_tier_present():
    cm = CacheManager(
        num_pages=8, page_size=1, page_table=_page_table(), type="radix", host_pages=3
    )
    # Simulate free_slots arriving out of ascending order, e.g. after eviction appended a
    # low device id after some high (RAM-tier) ids were already free.
    cm.free_slots = torch.tensor([6, 7, 1, 0, 5, 4, 3, 2], dtype=torch.int32)
    allocated = cm._allocate(5)
    assert allocated.tolist() == [0, 1, 2, 3, 4]


def test_allocate_does_not_sort_without_host_tier():
    cm = CacheManager(num_pages=8, page_size=1, page_table=_page_table(), type="radix")
    cm.free_slots = torch.tensor([6, 7, 1, 0, 5, 4, 3, 2], dtype=torch.int32)
    allocated = cm._allocate(5)
    assert allocated.tolist() == [6, 7, 1, 0, 5]
