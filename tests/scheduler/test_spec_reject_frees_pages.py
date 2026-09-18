"""CacheManager.free_spec_reject: return a rejected speculative window's unused whole pages,
without touching the prefix cache (a rejected window was never committed). CPU, real
CacheManager, no engine."""
from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.spec import _spec_mrope_positions

PROMPT = list(range(1, 9))  # 8 tokens


def _make(page_size, num_pages=8):
    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, page_table, "radix")
    return cm, page_table


def _req(table_idx, prompt_len):
    from types import SimpleNamespace

    req = Req(input_ids=torch.tensor(PROMPT[:prompt_len], dtype=torch.int32),
              table_idx=table_idx, cached_len=0, output_len=4, uid=table_idx,
              sampling_params=SamplingParams(), cache_handle=SimpleNamespace(cached_len=0))
    return req


def test_reject_inside_one_page_frees_nothing():
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=1)
    req.device_len = 1
    cm.allocate_paged([req])  # page [0]: page-ceil(1) = 1 page
    req.cached_len = req.device_len
    before = set(cm.free_slots.tolist())

    # verify window sped up device_len to 2 (draft position) inside the SAME page
    req.device_len = 2
    cm.allocate_paged([req])  # still page-ceil(2)=1 page, no new allocation
    after_alloc = set(cm.free_slots.tolist())
    assert after_alloc == before  # nothing new was allocated within the page

    cm.free_spec_reject(req, keep_len=1, alloc_len=2)
    assert set(cm.free_slots.tolist()) == after_alloc, "same page as keep_len -> no-op"


def test_reject_crossing_a_page_boundary_frees_the_speculative_page():
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.device_len = 4
    cm.allocate_paged([req])  # fills page 0 entirely: tokens [0,4)
    req.cached_len = req.device_len
    before = set(cm.free_slots.tolist())

    # verify window extends into a second page for the one draft token
    req.device_len = 5
    cm.allocate_paged([req])
    after_alloc = set(cm.free_slots.tolist())
    assert len(after_alloc) == len(before) - 1, "one new page (4 slots) was allocated"

    spec_page_base = page_table[0, 4].item()
    cm.free_spec_reject(req, keep_len=4, alloc_len=5)
    freed = set(cm.free_slots.tolist())
    assert spec_page_base in freed, "the speculative page must come back"
    assert freed == before, "rollback restores exactly the pre-verify free list"


def test_reject_never_touches_the_prefix_cache():
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.device_len = 5
    cm.allocate_paged([req])
    cm.free_spec_reject(req, keep_len=4, alloc_len=5)
    # a rejected window was never cache_req'd, so the tree stays empty -- nothing to unlock/evict
    assert cm.prefix_cache.size_info.total_size == 0


def test_spec_mrope_positions_use_three_axis_fallback():
    from types import SimpleNamespace

    req = SimpleNamespace(mrope_positions_full=None, mrope_delta=7)
    got = _spec_mrope_positions(req, cached_len=4, device_len=5, device=torch.device("cpu"))

    assert got.shape == (3, 1)
    assert got.dtype == torch.int32
    assert got.tolist() == [[11], [11], [11]]
