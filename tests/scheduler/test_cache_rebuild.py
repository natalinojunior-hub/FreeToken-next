"""The cache-rebuild path where it has an in-process seam: the scheduler's idle gate and the
pool/table re-point it performs. The destructive orchestration underneath (graph teardown, pool
resize, page-table refresh, graph re-capture) has no seam worth stubbing and is covered against a
real server by tests/e2e/test_cache_rebuild.py; the maintenance state machine the HTTP layer runs
on top lives in tests/server/test_rebuild_maintenance.py."""

from __future__ import annotations

import torch


def _page_table(max_running_reqs: int, width: int) -> torch.Tensor:
    return torch.zeros((max_running_reqs + 1, width), dtype=torch.int32, device=torch.device("cpu"))


def _setup_context(page_size: int) -> None:
    """Initialize global context if not already done."""
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        # Create minimal context
        ctx = Context(page_size=page_size)
        set_global_ctx(ctx)


def test_cache_manager_rebuild_resets_pages_and_prefix():
    from freetoken.scheduler.cache import CacheManager

    _setup_context(page_size=2)

    pt = _page_table(4, 64)
    cm = CacheManager(num_pages=8, page_size=2, page_table=pt, type="radix")
    # mutate state so we can prove rebuild resets it
    cm.free_slots = cm.free_slots[:3]

    new_pt = _page_table(4, 128)
    cm.rebuild(num_pages=20, page_table=new_pt)

    assert cm.num_pages == 20
    assert cm.page_table is new_pt
    assert cm.free_slots.tolist() == [i * 2 for i in range(20)]
    assert cm.prefix_cache.size_info.total_size == 0
    cm.check_integrity()  # must pass: free_pages(20) + cache_pages(0) == num_pages(20)


def test_table_manager_rebuild_reallocs_token_pool_and_frees_slots():
    from freetoken.scheduler.table import TableManager

    pt = _page_table(4, 64)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    tm.allocate()
    tm.allocate()  # consume 2 slots

    new_pt = _page_table(4, 128)
    tm.rebuild(new_pt)

    assert tm.page_table is new_pt
    assert tm.token_pool.shape == new_pt.shape
    assert tm.available_size == 4  # all slots free again


def _stub_scheduler(*, prefill_runnable: bool, decode_runnable: bool, pending: object | None):
    """A Scheduler shell (no __init__/GPU) wired just enough to drive normal_loop's
    rebuild-drain branch. _execute_pending_rebuild is replaced with a recorder."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=prefill_runnable)
    sched.decode_manager = SimpleNamespace(runnable=decode_runnable)
    sched._pending_rebuild = pending
    sched.receive_msg = lambda blocking: []
    sched._schedule_next_batch = lambda: None
    sched._process_last_data = lambda data: None
    calls = []

    def _exec():
        calls.append(True)
        sched._pending_rebuild = None

    sched._execute_pending_rebuild = _exec
    return sched, calls


def test_normal_loop_executes_pending_rebuild_when_idle():
    # Non-overlap mode (DISABLE_OVERLAP_SCHEDULING) must drain a queued rebuild at the idle
    # safe point, else it hangs until the HTTP request times out.
    from freetoken.scheduler.scheduler import Scheduler

    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=False, pending=object())
    Scheduler.normal_loop(sched)
    assert calls == [True]
    assert sched._pending_rebuild is None


def test_mtp_selects_normal_loop_without_an_environment_flag(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    class Finished(Exception):
        pass

    def normal_loop():
        raise Finished

    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    sched = SimpleNamespace(
        spec_mtp=4,
        engine_stream_ctx=nullcontext(),
        engine=SimpleNamespace(stream=SimpleNamespace(wait_stream=lambda _: None)),
        stream=object(),
        decode_manager=SimpleNamespace(
            running_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(is_greedy=True))]
        ),
        prefill_manager=SimpleNamespace(pending_list=[]),
        normal_loop=normal_loop,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: sched.stream)
    sched._has_greedy_mtp_req = lambda data: Scheduler._has_greedy_mtp_req(sched, data)
    with pytest.raises(Finished):
        Scheduler.run_forever(sched)


def test_sampling_mtp_uses_overlap_and_greedy_pending_uses_normal():
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = SimpleNamespace(
        spec_mtp=2,
        decode_manager=SimpleNamespace(running_reqs=[]),
        prefill_manager=SimpleNamespace(pending_list=[]),
    )
    sampling_req = SimpleNamespace(sampling_params=SimpleNamespace(is_greedy=False))
    greedy_req = SimpleNamespace(sampling_params=SimpleNamespace(is_greedy=True))
    sched.prefill_manager.pending_list = [
        SimpleNamespace(sampling_params=sampling_req.sampling_params, chunked_req=greedy_req)
    ]
    assert Scheduler._has_greedy_mtp_req(sched, None) is False
    sched.prefill_manager.pending_list[0].sampling_params.is_greedy = True
    assert Scheduler._has_greedy_mtp_req(sched, None) is True  # includes chunked prefill
    sched.prefill_manager.pending_list.clear()
    pending_data = (SimpleNamespace(batch=SimpleNamespace(reqs=[greedy_req])), None)
    assert Scheduler._has_greedy_mtp_req(sched, pending_data) is True


def test_sampling_mtp_enters_overlap(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    class Finished(Exception):
        pass

    stream = object()
    sched = SimpleNamespace(
        spec_mtp=2,
        stream=stream,
        engine=SimpleNamespace(stream=object()),
        decode_manager=SimpleNamespace(running_reqs=[]),
        prefill_manager=SimpleNamespace(pending_list=[]),
        overlap_loop=lambda data: (_ for _ in ()).throw(Finished()),
    )
    sched._has_greedy_mtp_req = lambda data: Scheduler._has_greedy_mtp_req(sched, data)
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    with pytest.raises(Finished):
        Scheduler.run_forever(sched)


def test_disable_overlap_flag_keeps_sampling_on_normal_loop(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    class Finished(Exception):
        pass

    stream = object()
    sched = SimpleNamespace(
        spec_mtp=2,
        stream=stream,
        engine_stream_ctx=nullcontext(),
        engine=SimpleNamespace(stream=SimpleNamespace(wait_stream=lambda _stream: None)),
        decode_manager=SimpleNamespace(running_reqs=[]),
        prefill_manager=SimpleNamespace(pending_list=[]),
        normal_loop=lambda: (_ for _ in ()).throw(Finished()),
    )
    sched._has_greedy_mtp_req = lambda data: False
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    with pytest.raises(Finished):
        Scheduler.run_forever(sched)


def test_first_greedy_admission_drains_overlap_batch_once(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    class Finished(Exception):
        pass

    events = []
    stream = SimpleNamespace(wait_stream=lambda other: events.append(("wait", other)))
    engine_stream = SimpleNamespace(wait_stream=lambda other: events.append(("engine_wait", other)))
    greedy = SimpleNamespace(sampling_params=SimpleNamespace(is_greedy=True))
    data = (SimpleNamespace(batch=SimpleNamespace(reqs=[])), "output")
    sched = SimpleNamespace(
        spec_mtp=2,
        engine_stream_ctx=nullcontext(),
        engine=SimpleNamespace(stream=engine_stream),
        stream=stream,
        decode_manager=SimpleNamespace(running_reqs=[]),
        prefill_manager=SimpleNamespace(pending_list=[]),
        _process_last_data=lambda item: events.append(("drain", item)),
        _flush_oom=lambda: events.append(("oom",)),
        _flush_abort_acks=lambda: events.append(("abort",)),
        normal_loop=lambda: (_ for _ in ()).throw(Finished()),
    )
    sched._has_greedy_mtp_req = lambda pending: Scheduler._has_greedy_mtp_req(sched, pending)

    def overlap_loop(last):
        assert last is None
        events.append(("overlap",))
        sched.prefill_manager.pending_list.append(
            SimpleNamespace(sampling_params=greedy.sampling_params, chunked_req=greedy)
        )
        return data

    sched.overlap_loop = overlap_loop
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    with pytest.raises(Finished):
        Scheduler.run_forever(sched)
    assert [event[0] for event in events] == [
        "overlap",
        "wait",
        "drain",
        "oom",
        "abort",
        "engine_wait",
    ]
    assert events[2][1] is data


def test_drain_rechecks_greedy_before_entering_normal_mode(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest

    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    class Finished(Exception):
        pass

    events = []
    stream = SimpleNamespace(wait_stream=lambda _stream: events.append("wait"))
    req = SimpleNamespace(sampling_params=SimpleNamespace(is_greedy=True))
    pending = SimpleNamespace(sampling_params=req.sampling_params, chunked_req=req)
    data = (SimpleNamespace(batch=SimpleNamespace(reqs=[])), "output")
    sched = SimpleNamespace(
        spec_mtp=2,
        stream=stream,
        engine_stream_ctx=nullcontext(),
        engine=SimpleNamespace(
            stream=SimpleNamespace(wait_stream=lambda _stream: events.append("engine_wait"))
        ),
        decode_manager=SimpleNamespace(running_reqs=[]),
        prefill_manager=SimpleNamespace(pending_list=[]),
        _flush_oom=lambda: events.append("oom"),
        _flush_abort_acks=lambda: events.append("abort"),
        normal_loop=lambda: (_ for _ in ()).throw(AssertionError("stale greedy selection")),
    )
    sched._has_greedy_mtp_req = lambda last: Scheduler._has_greedy_mtp_req(sched, last)

    launched = False

    def overlap_loop(last):
        nonlocal launched
        if launched:
            events.append("overlap_next")
            raise Finished
        launched = True
        events.append("overlap_first")
        sched.prefill_manager.pending_list.append(pending)
        return data

    sched.overlap_loop = overlap_loop
    sched._process_last_data = lambda _data: (
        events.append("drain"),
        sched.prefill_manager.pending_list.clear(),
    )
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    with pytest.raises(Finished):
        Scheduler.run_forever(sched)
    assert events == ["overlap_first", "wait", "drain", "oom", "abort", "overlap_next"]


def test_normal_loop_defers_pending_rebuild_while_busy():
    # A queued rebuild must NOT run while prefill/decode is still in flight.
    from freetoken.scheduler.scheduler import Scheduler

    pending = object()
    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=True, pending=pending)
    Scheduler.normal_loop(sched)
    assert calls == []
    assert sched._pending_rebuild is pending  # still queued


def test_rebuild_cache_refreshes_prefill_budget(monkeypatch):
    # A rebuild that shrank the DSV4 window pool must shrink Scheduler.prefill_budget to the new
    # prefill_chunk_budget, or the next long prompt is chunked against the stale (larger) cap.
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=False)
    sched.decode_manager = SimpleNamespace(runnable=False)
    sched.device = torch.device("cpu")
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1), max_extend_tokens=100_000)
    sched.engine = SimpleNamespace(
        rebuild_runtime_cache=lambda **kw: None, num_pages=32, page_table=None
    )
    # engine.page_table unchanged across the (stubbed) rebuild -> no token_pool re-point.
    sched.table_manager = SimpleNamespace(page_table=None)
    # DSV4-like manager: prefill_chunk_budget tracks the (about-to-shrink) window pool; no shared
    # page table, so rebuild_cache's prefix-cache rebuild branch is skipped.
    cache_manager = SimpleNamespace(
        prefill_chunk_budget=5000, rebuild=lambda *a: None, check_integrity=lambda: None
    )
    sched.cache_manager = cache_manager
    sched.table_manager.rebuild = lambda pt: None
    sched.table_manager.token_pool = None
    sched.prefill_budget = min(sched.config.max_extend_tokens, cache_manager.prefill_chunk_budget)
    assert sched.prefill_budget == 5000

    cache_manager.prefill_chunk_budget = 1000  # the (stubbed) engine rebuild shrank the pool
    Scheduler.rebuild_cache(sched, num_pages=16)
    assert sched.prefill_budget == 1000  # tracks the shrunk cap, not the stale 5000
