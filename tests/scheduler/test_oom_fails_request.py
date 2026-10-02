from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.scheduler import Scheduler


def _stub():
    sent, freed, shrinks, aborted = [], [], [], []
    rejected = []
    stub = SimpleNamespace(
        cache_manager=SimpleNamespace(
            page_size=1,
            free_spec_reject=lambda req, keep_len, alloc_len: rejected.append(
                (keep_len, alloc_len)
            ),
        ),
        prefill_manager=SimpleNamespace(abort_req=lambda uid: aborted.append(("p", uid))),
        decode_manager=SimpleNamespace(
            abort_req=lambda uid: aborted.append(("d", uid)), running_reqs=set()
        ),
        _free_req_resources=freed.append,
        send_result=sent.extend,
        device=None,
        engine=SimpleNamespace(shrink_after_oom=lambda: shrinks.append(1)),
    )
    stub._fail_oom_reqs = lambda reqs, e: Scheduler._fail_oom_reqs(stub, reqs, e)
    stub._rejected = rejected
    return stub, sent, freed, shrinks


def test_oom_in_forward_fails_the_batch_not_the_process(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    stub, sent, freed, shrinks = _stub()
    live = SimpleNamespace(
        uid=3,
        table_idx=1,
        device_len=12288,
        cached_len=8192,
        alloc_page_bound=12288,
        cache_handle=SimpleNamespace(cached_len=4096),
    )
    reqs = [live, SimpleNamespace(uid=-1, table_idx=0)]  # -1: the padding dummy

    def boom(_):
        raise torch.OutOfMemoryError("CUDA out of memory")

    stub._forward = boom
    fi = SimpleNamespace(batch=SimpleNamespace(reqs=reqs))
    assert Scheduler._forward_or_fail(stub, fi) is None
    assert sent == []  # handled only after the overlapped batch is drained
    stub.finished_reqs = []
    Scheduler._flush_oom(stub)
    assert [m.uid for m in sent] == [3] and "out of GPU memory" in sent[0].error
    assert freed == [live] and shrinks == [1]
    # every page past the matched prefix goes back; nothing new is committed to the cache
    assert stub._rejected == [(4096, 12288)] and live.cached_len == live.device_len == 4096


def test_oom_skips_request_finished_in_the_drained_batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    stub, sent, freed, shrinks = _stub()
    done = SimpleNamespace(uid=5, table_idx=-1)
    stub.finished_reqs = [done]
    stub._oom_failed = ([done], torch.OutOfMemoryError("CUDA out of memory"))
    Scheduler._flush_oom(stub)
    assert sent == [] and freed == [] and shrinks == [1]


def test_oom_retry_shrinks_before_replaying_exact_forward(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    shrinks = []
    attempts = []
    stub = SimpleNamespace(
        device=None,
        engine=SimpleNamespace(shrink_after_oom=lambda: shrinks.append(True)),
        _forward=lambda forward_input: attempts.append(forward_input) or "replayed",
    )
    item = object()
    result = Scheduler._oom_retry(
        stub, item, torch.OutOfMemoryError("Tried to allocate 2.00 MiB"), 2 << 20
    )
    assert result == "replayed"
    assert shrinks == [True] and attempts == [item]


def test_spec_oom_learns_driver_request_before_dropping_nontransactional_batch():
    errors = []
    stub = SimpleNamespace(
        engine=SimpleNamespace(note_decode_oom=errors.append),
        decode_manager=SimpleNamespace(running_reqs=["active"]),
        run_spec_step=lambda: (_ for _ in ()).throw(
            torch.OutOfMemoryError("Tried to allocate 2.00 MiB")
        ),
        _fail_oom_reqs=lambda reqs, error: errors.append((reqs, error)),
    )

    assert Scheduler._spec_step_or_fail(stub) is True
    assert str(errors[0]) == "Tried to allocate 2.00 MiB"
    assert errors[1][0] == ["active"]


def test_spec_oom_before_commit_rolls_back_and_keeps_the_request(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    stub = SimpleNamespace(device=None, decode_manager=SimpleNamespace(running_reqs=["active"]))
    stub.engine = SimpleNamespace(
        note_decode_oom=lambda e: events.append("note"),
        shrink_after_oom=lambda: events.append("shrink"),
    )
    stub._mtp_controller = SimpleNamespace(fallback_to_k0=lambda: events.append("k0"))

    def step():
        stub._spec_rollback = lambda: events.append("rollback")
        raise torch.OutOfMemoryError("Tried to allocate 2.00 MiB")

    stub.run_spec_step = step
    stub._fail_oom_reqs = lambda reqs, error: events.append("failed")

    assert Scheduler._spec_step_or_fail(stub) is False  # RAW decode runs this iteration
    assert events == ["note", "rollback", "shrink", "k0"]
    assert stub._spec_rollback is None and stub._mtp_cycle_observe is False


def test_first_k1_oom_keeps_k1_second_turns_speculation_off():
    limits = []
    controller = SimpleNamespace(limit_depth=limits.append, fallback_to_k0=lambda: None)
    stub = SimpleNamespace(spec_mtp=5, _mtp_controller=controller, _mtp_controllers={})

    Scheduler._mark_mtp_oom(stub, 3)
    Scheduler._mark_mtp_oom(stub, 1)
    Scheduler._mark_mtp_oom(stub, 1)
    assert limits == [2, 1, 0]


def test_other_errors_still_raise():
    stub, *_ = _stub()

    def boom(_):
        raise ValueError("bug")

    stub._forward = boom
    with pytest.raises(ValueError):
        Scheduler._forward_or_fail(stub, SimpleNamespace(batch=SimpleNamespace(reqs=[])))
