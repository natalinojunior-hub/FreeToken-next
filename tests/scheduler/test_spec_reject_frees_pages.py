"""CacheManager.free_spec_reject: return a rejected speculative window's unused whole pages,
without touching the prefix cache (a rejected window was never committed). CPU, real
CacheManager, no engine."""
from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.spec import SchedulerSpecMixin, _spec_mrope_positions
from freetoken.scheduler.table import TableManager

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


def test_finished_mid_verify_frees_the_surplus_page_before_recycle():
    """EXP-033: a verify window (spec_alloc_len, e.g. d+k) can allocate the page holding the
    correction token past where a mid-window finish truncates cached_len/device_len. cache_req's
    own finished-tail-free only ever sees the truncated device_len (cached_len < device_len is a
    standing invariant, not a spec-only signal -- see cache.py's _padded_tail comment), so the
    surplus [truncated device_len, spec_alloc_len) must be reclaimed via free_spec_reject BEFORE
    cache_req(finished=True) runs and the caller recycles table_idx (SchedulerSpecMixin does this
    ordering in _commit_spec_tokens/run_spec_step); doing it after is a no-op-by-corruption once
    table_idx == -1, since page_table[-1] silently frees a different row's pages instead."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.device_len = 4
    cm.allocate_paged([req])  # fills page 0: tokens [0,4)
    req.cached_len = req.device_len
    req.cache_handle = cm.prefix_cache.match_prefix(req.input_ids[:0]).cuda_handle

    # verify window allocates through a correction token in a second page (spec_alloc_len=5),
    # but the request finishes at the first drafted position (truncated device_len=4)
    req.device_len = 5
    cm.allocate_paged([req])
    req.device_len = 4

    # the fix: free the verify window's surplus BEFORE the finish tail-free / table_idx recycle
    cm.free_spec_reject(req, keep_len=req.cached_len, alloc_len=5)
    cm.cache_req(req, finished=True)
    cm.check_integrity()  # raises RuntimeError if the surplus page was leaked


def test_reject_keep_len_must_be_the_post_commit_device_len_not_cached_len():
    """Live crash (RTX 5080, real serve): a 16384-token prompt is exactly page-aligned at
    page_size=64. run_spec_step's own `keep_len = d - 1 + committed` matches req.cached_len's
    standing complete_one-style lag (cached_len is always ONE BEHIND the last real, committed
    token -- see cache.py's _padded_tail comment) -- but free_spec_reject's own contract (see
    test_reject_crossing_a_page_boundary_frees_the_speculative_page above) is the OPPOSITE: its
    keep_len must be a count where index keep_len itself is already purely speculative, i.e. the
    boundary AFTER the lag (== req.device_len). Passing the lag value straight through only
    differs from the correct one at an EXACT page boundary -- exactly a 16384-token prompt at
    page_size=64 -- where it wrongly hands back the page holding the just-committed, still-real
    token. This is what actually produced EXP-033's live 'free_pages+cache_pages != num_pages'
    crash: not an under-free, but this over-free corrupting a live page."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.device_len = 4
    cm.allocate_paged([req])  # page 0 full and page-aligned: tokens [0,4)
    req.cached_len = req.device_len

    # verify (k=1) extends into page 1 for the one draft position (index 4)
    req.device_len = 5
    cm.allocate_paged([req])
    real_page_base = page_table[0, 4].item()  # page 1: holds the correction token once accepted

    # k=1, draft rejected, correction accepted -> committed=1, matching _commit_spec_tokens'
    # convention: req.cached_len stays at d (4, the lag), req.device_len becomes d+1 (5)
    buggy_keep_len = 4  # what run_spec_step used to pass (== req.cached_len, the lag value)
    correct_keep_len = 5  # == req.device_len: the fix

    cm.free_spec_reject(req, keep_len=correct_keep_len, alloc_len=5)
    assert real_page_base not in set(cm.free_slots.tolist()), (
        "fixed keep_len must NOT free the page holding the just-committed real token"
    )

    cm.free_spec_reject(req, keep_len=buggy_keep_len, alloc_len=5)
    assert real_page_base in set(cm.free_slots.tolist()), (
        "demonstrates the bug: the unadjusted (lag) keep_len wrongly frees a live page "
        "at an exact page-size boundary -- this must never regress back to `d - 1 + committed`"
    )


def test_allocate_paged_twice_over_the_same_range_orphans_a_page():
    """Live crash (RTX 5080, real serve, page_size=64, 16384-token prompt): SchedulerSpecMixin's
    GDN-state replay rewinds cached_len/device_len to an already-verified window and calls
    _prepare_batch again to rebuild positions/attention metadata -- but _prepare_batch's
    allocate_paged has no memory of the verify step's own earlier call over that SAME range,
    so calling it twice on an identical (cached_len, device_len) pair that crosses a page
    boundary allocates a SECOND, different page and overwrites the page_table row that already
    held the first one -- orphaning it (never returned to free_slots, never reachable again).
    This is what actually produced free_pages+cache_pages != num_pages: not a spec_reject
    accounting error, but allocate_paged's own lack of idempotency across two calls for the
    same range. The real fix is scheduler.py's `_prepare_batch(..., skip_alloc=True)` on the
    replay call; this test pins the underlying mechanism at the CacheManager level."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.cached_len, req.device_len = 4, 4
    cm.allocate_paged([req])  # page 0 full and page-aligned: tokens [0,4)

    # verify (k=1): allocate_paged([4, 5)) crosses into page 1 for the correction token
    req.cached_len, req.device_len = 4, 5
    cm.allocate_paged([req])
    first_page_value = page_table[0, 4].item()

    # replay rewinds to the SAME (cached_len, device_len) pair to rebuild metadata -- calling
    # allocate_paged again must be skipped (this is what skip_alloc does); simulating the old,
    # unguarded behaviour here to pin the failure mode:
    cm.allocate_paged([req])
    second_page_value = page_table[0, 4].item()

    assert second_page_value != first_page_value, (
        "demonstrates the bug: a second allocate_paged call over an unchanged range grabs a "
        "different physical page"
    )
    assert first_page_value not in set(cm.free_slots.tolist()), (
        "the orphaned page is neither referenced by page_table nor back in free_slots -- lost"
    )


def test_free_req_resources_releases_the_spec_snapshot_slot_on_any_finish_path():
    """A request that stops being spec-eligible on its own last token (remain_len <= 1, see
    SchedulerSpecMixin._spec_eligible_req) finishes through the PLAIN decode path, which never
    runs run_spec_step's own `if finished: free_spec_snapshot_slot(req)` branch. Before this
    fix, that slot leaked on every single request, exhausting LinearStatePool by the 2nd
    request of a live benchmark (RTX 5080: 'LinearStatePool exhausted: need 1, have 0' on the
    2nd repeat, MTP=1 + Turbo4). _free_req_resources is the one cleanup path every finish route
    (abort, plain decode, spec) shares, so the release belongs there, not in spec.py alone."""
    from types import SimpleNamespace

    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(8, 4, page_table, "radix")
    tm = TableManager(max_running_reqs=2, page_table=page_table)
    freed_slots = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(free=freed_slots.extend)),
        _spec_snapshot_slots={},
    )
    stub.free_spec_snapshot_slot = lambda req: SchedulerSpecMixin.free_spec_snapshot_slot(stub, req)

    req = _req(0, prompt_len=4)
    req.device_len = 4
    cm.allocate_paged([req])
    req.cached_len = req.device_len
    req.cache_handle = cm.prefix_cache.match_prefix(req.input_ids[:0]).cuda_handle
    stub._spec_snapshot_slots[req.uid] = 7  # as if _spec_snapshot_slot allocated slot 7 earlier

    Scheduler._free_req_resources(stub, req)

    assert freed_slots == [7], "the GDN snapshot slot must be returned to the pool"
    assert req.uid not in stub._spec_snapshot_slots


def test_spec_mrope_positions_use_three_axis_fallback():
    from types import SimpleNamespace

    req = SimpleNamespace(mrope_positions_full=None, mrope_delta=7)
    got = _spec_mrope_positions(req, cached_len=4, device_len=5, device=torch.device("cpu"))

    assert got.shape == (3, 1)
    assert got.dtype == torch.int32
    assert got.tolist() == [[11], [11], [11]]
