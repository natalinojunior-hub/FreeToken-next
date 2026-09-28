"""CacheManager.free_spec_reject: return a rejected speculative window's unused whole pages,
without touching the prefix cache (a rejected window was never committed). CPU, real
CacheManager, no engine."""

from __future__ import annotations

from types import SimpleNamespace

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

    req = Req(
        input_ids=torch.tensor(PROMPT[:prompt_len], dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=4,
        uid=table_idx,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(cached_len=0),
    )
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


def test_allocate_paged_twice_over_the_same_range_is_idempotent():
    """Live crash (RTX 5080, real serve, page_size=64, 16384-token prompt): SchedulerSpecMixin's
    GDN-state replay rewinds cached_len/device_len to an already-verified window and calls
    _prepare_batch again to rebuild positions/attention metadata. allocate_paged used to have no
    memory of its own earlier call over that SAME range: div_ceil(cached_len, page_size) alone
    treated the already-owned page as unallocated, so a second call crossing the same page
    boundary allocated a SECOND, different page and overwrote the page_table row that already
    held the first one -- orphaning it (never returned to free_slots, never reachable again).
    The fix: req.alloc_page_bound tracks the real ownership high-water mark (set by
    allocate_paged, dropped by free_spec_reject), so a repeat call over an unchanged range is
    now a no-op instead of a double-allocation."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.cached_len, req.device_len = 4, 4
    cm.allocate_paged([req])  # page 0 full and page-aligned: tokens [0,4)

    # verify (k=1): allocate_paged([4, 5)) crosses into page 1 for the correction token
    req.cached_len, req.device_len = 4, 5
    cm.allocate_paged([req])
    first_page_value = page_table[0, 4].item()
    free_after_first = set(cm.free_slots.tolist())

    # replay rewinds to the SAME (cached_len, device_len) pair to rebuild metadata -- calling
    # allocate_paged again must be a no-op now (alloc_page_bound already covers this range):
    cm.allocate_paged([req])

    assert page_table[0, 4].item() == first_page_value, (
        "a repeat allocate_paged call over an unchanged range must not reassign the page"
    )
    assert set(cm.free_slots.tolist()) == free_after_first, "no new page should be allocated"


def test_rollback_to_an_exact_page_boundary_after_a_speculative_page_does_not_orphan_it():
    """Live crash trigger, stated precisely: a speculative-decode rollback lands keep_cached on
    an EXACT multiple of page_size after a page was already allocated one token into it (for a
    since-rejected draft). div_ceil(keep_cached, page_size) alone is then one page BELOW what
    the request actually owns, so the next round's allocate_paged call re-derives first_page as
    that same already-owned page, grabs a fresh physical page for it, and overwrites the
    page_table row -- orphaning the original (never freed, never reachable again). This is
    equivalent to keep_device == 1 (mod page_size). The fix: req.alloc_page_bound remembers the
    real ownership ceiling across the rollback."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.cached_len, req.device_len = 4, 4
    cm.allocate_paged([req])  # page 0 full and page-aligned: tokens [0,4)

    # round R: draft at position 4 needs a new page; draft rejected, only the correction at
    # position 4 is committed -> keep_cached lands exactly on the page-4 boundary (4 == 1*4).
    req.cached_len, req.device_len = 4, 5
    cm.allocate_paged([req])  # allocates page 1 (tokens [4, 8)) for the draft
    speculative_page_value = page_table[0, 4].item()
    cm.free_spec_reject(req, keep_len=5, alloc_len=5)  # keep_device == alloc_len: no-op free
    req.cached_len, req.device_len = 4, 5  # keep_cached=4 (page-aligned), keep_device=5

    # round R+1: draft chain resets cached_len to device_len - 1 == 4 again, then extends
    # device_len by k=1 to re-verify -- an allocate_paged call over the SAME range as round R's.
    req.cached_len, req.device_len = 4, 5
    cm.allocate_paged([req])

    assert page_table[0, 4].item() == speculative_page_value, (
        "round R+1 must not reassign the page already owned since round R"
    )
    assert speculative_page_value not in set(cm.free_slots.tolist()), (
        "the page is still legitimately owned by the request, not back on the free list"
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


def test_free_req_resources_clears_qsa_pool_and_snapshots():
    from types import SimpleNamespace

    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(8, 4, page_table, "radix")
    tm = TableManager(max_running_reqs=2, page_table=page_table)
    freed_qsa_tables = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        engine=SimpleNamespace(kv_cache=SimpleNamespace(free_req=freed_qsa_tables.append)),
        _spec_snapshot_slots={},
        _spec_qsa_snapshots={0: (torch.zeros(1), torch.zeros(1))},
    )
    stub.free_spec_snapshot_slot = lambda req: SchedulerSpecMixin.free_spec_snapshot_slot(stub, req)

    req = _req(0, prompt_len=4)
    req.table_idx = 1
    req.device_len = 4
    cm.allocate_paged([req])
    req.cached_len = req.device_len
    req.cache_handle = cm.prefix_cache.match_prefix(req.input_ids[:0]).cuda_handle

    Scheduler._free_req_resources(stub, req)

    assert freed_qsa_tables == [1], "QSA pool's free_req must be called with table_idx"
    assert 0 not in stub._spec_qsa_snapshots, "QSA snapshots must be cleared"


def test_idle_asserts_spec_snapshot_slots_empty():
    from types import SimpleNamespace

    import pytest

    page_table = torch.zeros(2, 32, dtype=torch.int32)
    cm = CacheManager(8, 4, page_table, "radix")
    stub = SimpleNamespace(
        cache_manager=cm,
        _spec_snapshot_slots={},
    )
    # Empty -> passes
    Scheduler.run_when_idle(stub)

    # Leaked slot -> raises AssertionError
    stub._spec_snapshot_slots = {101: 3}
    with pytest.raises(AssertionError, match="leaked spec snapshot slots in idle"):
        Scheduler.run_when_idle(stub)


def test_finish_on_verified_draft_token_at_page_boundary_does_not_leak():
    """Live crash (RTX 5080, spec-mtp=1, Turbo4, 16K real serve): EOS lands on a token that was
    itself one of the verify batch's OWN INPUT drafts (committed <= k, not the k+1'th bonus/
    correction token), so its KV was already written by the single verify forward -- unlike
    normal decode's complete_one lag, where the newest token's KV is genuinely still pending.
    _commit_spec_tokens used to leave req.cached_len at the lag value (keep_cached) regardless,
    which is only correct for the pending bonus token. When keep_cached lands exactly on a page
    boundary, that already-written page is excluded from BOTH cache_req's insert/tail-free range
    (which stops at page_ceil(keep_cached)) AND free_spec_reject's range (which starts at
    page_ceil(keep_device) = the page AFTER it) -- orphaned forever. The fix: cached_len at
    finish must be keep_device (not keep_cached) whenever committed <= k."""
    cm, page_table = _make(page_size=4)
    req = _req(0, prompt_len=4)
    req.device_len = 4
    cm.allocate_paged([req])  # fills page 0 exactly: tokens [0,4)
    req.cached_len = req.device_len
    req.cache_handle = cm.prefix_cache.match_prefix(req.input_ids[:0]).cuda_handle

    # k=1 verify window: draft at position 4 (a real verify-batch input), spec_alloc_len=5
    req.device_len = 5
    cm.allocate_paged([req])  # allocates page 1 for the draft position
    committed = 1  # EOS hit on the draft itself (committed <= k=1): its KV IS already written
    start_pos, spec_alloc_len = 4, 5
    k = spec_alloc_len - start_pos
    keep_cached = start_pos + committed - 1  # the old (buggy) value: 4

    # surplus beyond what was committed (none here: keep_device == spec_alloc_len == 5)
    keep_device = start_pos + committed
    if keep_device < spec_alloc_len:
        cm.free_spec_reject(req, keep_len=keep_device, alloc_len=spec_alloc_len)

    # the fix, exactly as applied in _commit_spec_tokens:
    req.cached_len = start_pos + min(committed, k)  # == keep_device == 5, NOT keep_cached (4)
    assert req.cached_len != keep_cached
    cm.cache_req(req, finished=True)
    cm.check_integrity()  # would raise RuntimeError with the old keep_cached-only assignment


def test_spec_snapshot_slot_evicts_radix_snapshots_when_free_list_is_drained():
    from types import SimpleNamespace

    free = []
    pool = SimpleNamespace(alloc=lambda n: [free.pop() for _ in range(n)])
    cm = SimpleNamespace(is_hybrid=True, ensure_mamba_slots=lambda n: free.extend(range(7, 7 + n)))
    stub = SimpleNamespace(
        cache_manager=cm, engine=SimpleNamespace(linear_state_pool=pool), _spec_snapshot_slots={}
    )
    req = SimpleNamespace(uid=3)
    assert SchedulerSpecMixin._spec_snapshot_slot(stub, req) == 7
    assert stub._spec_snapshot_slots == {3: 7}


def test_flush_deferred_replays_feeds_every_pending_token_before_a_plain_decode():
    """A deferred k=1 rejection leaves two unprocessed tokens (device_len - cached_len == 2).
    Before any non-spec forward, which assumes one, the flush must replay all but the last."""
    from types import SimpleNamespace

    replays = []
    pending, current = _req(0, prompt_len=8), _req(1, prompt_len=8)
    pending.cached_len, pending.device_len = 5, 7
    current.cached_len, current.device_len = 6, 7
    stub = SimpleNamespace(decode_manager=SimpleNamespace(running_reqs={pending, current}))
    stub._replay = lambda req, start, n: replays.append((req.uid, start, n))

    SchedulerSpecMixin._flush_deferred_replays(stub)

    assert replays == [(0, 5, 1)]
    assert (pending.cached_len, pending.device_len) == (6, 7)
    assert (current.cached_len, current.device_len) == (6, 7)


def test_replay_window_preserves_inputs_and_bookkeeping():
    """Multi-token replay keeps every logits row but does not commit its sampled successors."""
    from types import SimpleNamespace

    class Event:
        def __init__(self):
            self.synced = False

        def synchronize(self):
            self.synced = True

    for n in (1, 2, 3, 4):
        req = _req(0, prompt_len=8)
        req.cached_len, req.device_len = 3, 3 + n
        token_pool = torch.arange(64, dtype=torch.int64).reshape(2, 32)
        token_pool[0, : req.input_ids.numel()] = req.input_ids
        before_pool = token_pool.clone()
        before_ids = req.input_ids.clone()
        prepared = []
        event = Event()
        stub = SimpleNamespace(
            device=torch.device("cpu"),
            token_pool=token_pool,
            engine=SimpleNamespace(),
        )

        def prepare(batch, *, skip_alloc=False):
            prepared.append((batch, skip_alloc, batch.spec_logits_indices, batch.spec_host_ids))
            batch.padded_reqs = [req]
            positions = torch.arange(3, 3 + n, dtype=torch.int64)
            batch.positions = positions.to(torch.int32)
            return SimpleNamespace(
                input_tuple=(torch.zeros(n, dtype=torch.int64), positions), sample_args=None
            )

        def forward(batch, _sample_args):
            assert torch.equal(batch.input_ids, before_pool[0, 3 : 3 + n])
            # The ordinary engine path completes one pending token. Spec-indexed batches
            # deliberately skip that implicit transition; _replay must restore it itself.
            if batch.spec_logits_indices is None:
                req.complete_one()
            return SimpleNamespace(
                next_tokens_cpu=torch.full((n,), 999, dtype=torch.int64),
                next_tokens_gpu=torch.full((n,), 999, dtype=torch.int64),
                copy_done_event=event,
            )

        stub._prepare_batch = prepare
        stub.engine.forward_batch = forward
        # Exercise the production method with only preparation/forward supplied by CPU fakes.
        SchedulerSpecMixin._replay(stub, req, 3, n)

        batch, skip_alloc, indices, host_ids = prepared[0]
        assert skip_alloc is True
        if n == 1:
            assert batch.is_decode
            assert indices is None and host_ids is None
        else:
            assert batch.is_prefill
            assert torch.equal(indices, torch.arange(n))
            assert host_ids == before_ids[3 : 3 + n].tolist()
            assert host_ids == batch.input_ids.tolist()
        assert torch.equal(batch.input_ids, before_pool[0, 3 : 3 + n])
        assert event.synced
        assert torch.equal(token_pool, before_pool)
        assert torch.equal(req.input_ids, before_ids)
        assert (req.cached_len, req.device_len) == (3 + n, 4 + n)


def test_online_cost_hook_counts_committed_tokens_and_remaining_budget(monkeypatch):
    from freetoken.scheduler.adaptive_mtp import AdaptiveMtpController
    from freetoken.scheduler import spec

    req = _req(0, prompt_len=4)
    req.output_len = 0
    req.device_len = 5
    req.sampling_params = SamplingParams(max_tokens=8, temperature=0)
    controller = AdaptiveMtpController(2)
    scheduler = SimpleNamespace(
        _mtp_controller=controller,
        decode_manager=SimpleNamespace(running_reqs={req}),
        engine=SimpleNamespace(moe_offload_cache=None, num_pages=32),
        finished_reqs=set(),
    )
    clock = iter([1.0, 1.03, 2.0, 2.02])
    monkeypatch.setattr(spec.time, "perf_counter", lambda: next(clock))
    sample = SchedulerSpecMixin._begin_mtp_cycle(scheduler)
    req.append_host(torch.tensor([20, 21, 22], dtype=torch.int32))
    SchedulerSpecMixin._finish_mtp_cycle(scheduler, sample)
    assert controller.cost_summaries[0]["samples"] == 1
    assert abs(controller.cost_summaries[0]["seconds_per_token"] - 0.01) < 1e-12
    assert scheduler._mtp_distribution[0][:2] == [1, 3]

    # One remaining token must use ordinary decode, even during an incomplete probe.
    req.device_len = req.max_device_len - 1
    sample = SchedulerSpecMixin._begin_mtp_cycle(scheduler)
    assert controller.next_depth() == 0
    assert not controller.probing
    req.append_host(torch.tensor([23], dtype=torch.int32))
    scheduler.finished_reqs.add(req)
    SchedulerSpecMixin._finish_mtp_cycle(scheduler, sample)
    assert scheduler._mtp_distribution[0][:2] == [2, 4]


def test_free_request_discards_only_its_pending_draft_kv():
    req = _req(0, prompt_len=4)
    scheduler = SimpleNamespace(_spec_snapshot_slots={}, _mtp_kv_rows=(req.uid, 1, None, None))
    SchedulerSpecMixin.free_spec_snapshot_slot(scheduler, req)
    assert scheduler._mtp_kv_rows is None
    scheduler._mtp_kv_rows = (req.uid + 1, 1, None, None)
    SchedulerSpecMixin.free_spec_snapshot_slot(scheduler, req)
    assert scheduler._mtp_kv_rows[0] == req.uid + 1


def test_run_spec_step_k2_rejection_and_acceptance_positions():
    """Run the production k=2 scheduler path over deterministic CPU fakes."""
    cases = (
        ("reject0", [71, 80, 90], 1, 4, 5, 1, 2, 1, True, 71),
        ("reject1", [50, 81, 90], 2, 5, 6, 2, 2, 1, False, 81),
        ("all_accept", [50, 60, 90], 3, 6, 7, 0, 1, 1, False, 90),
    )
    for (
        name,
        sampled,
        committed,
        cached,
        device,
        replay_count,
        qsa_count,
        ple_count,
        free_last_page,
        pending,
    ) in cases:
        cm, page_table = _make(page_size=1, num_pages=16)
        req = _req(0, prompt_len=4)
        req.cached_len, req.device_len, req.output_len = 3, 4, 0
        req.sampling_params = SamplingParams(max_tokens=16)
        req.cache_handle = cm.prefix_cache.match_prefix(req.input_ids[:0]).cuda_handle
        cm.allocate_paged([req])
        draft_tokens = iter((50, 60))
        draft_steps = []
        replayed = []
        slot_at_replay = []
        qsa_restores = []
        ple_restores = []
        state_copies = []
        token_pool = torch.zeros((2, 32), dtype=torch.int64)
        token_pool[0, 3] = 7
        model_state = SimpleNamespace(_last_residual=torch.zeros((1, 4)))
        model = SimpleNamespace(
            mtp=object(),
            model=model_state,
            config=SimpleNamespace(num_experts_per_tok=2),
        )

        class StatePool:
            def __init__(self):
                self.slot_value = torch.zeros(16, dtype=torch.int64)
                self.slot_value[0] = 123

            def has_slot_state(self, key):
                return key == "ple_ngram_ctx"

            def slot_state(self, _key):
                return self.slot_value

            def copy_from(self, src, dst):
                state_copies.append((src, dst))
                self.slot_value[dst] = self.slot_value[src]

        pool = StatePool()
        scheduler = SimpleNamespace(
            spec_mtp=2,
            device=torch.device("cpu"),
            engine=SimpleNamespace(
                model=model,
                linear_state_pool=pool,
                moe_offload_cache=None,
                forward_batch=None,
            ),
            token_pool=token_pool,
            cache_manager=cm,
            decode_manager=SimpleNamespace(filter_reqs=lambda _reqs: None),
            finished_reqs=set(),
            eos_token_ids=set(),
            toolcall_anchor_id=None,
            _spec_eligible_req=lambda: req,
            _flush_deferred_replays=lambda: None,
            _snapshot_qsa_state=lambda _req: None,
            _spec_snapshot_slot=lambda _req: 9,
            _linear_slot=lambda _req: 0,
            _restore_qsa_state=lambda _req: qsa_restores.append(name),
            _restore_ple_state=lambda _req: ple_restores.append(name),
            _draft_step=None,
            _take_mtp_fill=lambda _req: None,
            _prepare_batch=None,
            _checked_verify_forward=None,
            _commit_spec_tokens=None,
            _replay=None,
            free_spec_snapshot_slot=lambda _req: None,
            _match_stop_str=lambda _req: None,
            send_result=lambda _messages: None,
        )
        # Keep preparation real enough to allocate the speculative pages and
        # provide the same (row, position) mapping as the scheduler.
        verify_pages = []

        def prepare(batch):
            cm.allocate_paged([req])
            verify_pages[:] = page_table[0, 4:6].tolist()
            batch.positions = torch.arange(3, 6, dtype=torch.int32)
            return SimpleNamespace(
                input_tuple=(torch.zeros(3, dtype=torch.long), torch.arange(3, 6)),
                sample_args=None,
            )

        def forward(_batch, _sample_args):
            pool.slot_value[0] = 888
            model_state._last_residual = torch.arange(3, dtype=torch.float32).view(3, 1)
            return SimpleNamespace(
                next_tokens_cpu=torch.tensor(sampled),
                next_tokens_gpu=torch.tensor(sampled),
                copy_done_event=SimpleNamespace(synchronize=lambda: None),
            )

        def draft(_req, pos, residual, _token):
            draft_steps.append((pos, _req.cached_len, _req.device_len))
            pool.slot_value[0] = 900 + pos
            return residual, torch.zeros((1, 2)), torch.tensor([next(draft_tokens)])

        def replay(_req, start, count):
            slot_at_replay.append(pool.slot_value[0].item())
            replayed.append((start, count))
            pool.slot_value[0] = 1000 + count

        scheduler._draft_step = draft
        scheduler._replay = replay
        scheduler._snapshot_ple_state = lambda r: SchedulerSpecMixin._snapshot_ple_state(
            scheduler, r
        )
        scheduler._restore_ple_state = lambda r: (
            ple_restores.append(name),
            SchedulerSpecMixin._restore_ple_state(scheduler, r),
        )
        scheduler._prepare_batch = prepare
        scheduler.engine.forward_batch = forward
        scheduler._checked_verify_forward = lambda *_args: None
        scheduler._commit_spec_tokens = lambda r, tokens, **kwargs: (
            SchedulerSpecMixin._commit_spec_tokens(scheduler, r, tokens, **kwargs)
        )
        scheduler.free_spec_snapshot_slot = lambda _req: None
        scheduler.table_manager = TableManager(max_running_reqs=2, page_table=page_table)
        ran = SchedulerSpecMixin.run_spec_step(scheduler)

        assert ran, name
        assert draft_steps == [(3, 3, 4), (4, 4, 5)], name
        assert (req.cached_len, req.device_len) == (cached, device), name
        assert req.device_len - req.cached_len == 1, name
        assert token_pool[0, req.cached_len].item() == pending, name
        assert replayed == ([(3, replay_count)] if replay_count else []), name
        assert len(qsa_restores) == qsa_count, name
        assert len(ple_restores) == ple_count, name
        assert model_state._last_residual.item() == committed - 1, name
        if replay_count:
            assert slot_at_replay == [123], name
        assert pool.slot_value[0].item() == (1000 + replay_count if replay_count else 888), name
        if free_last_page:
            assert verify_pages[1] in set(cm.free_slots.tolist()), name
        else:
            assert verify_pages[1] not in set(cm.free_slots.tolist()), name
