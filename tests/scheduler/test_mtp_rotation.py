# CPU checks of the experimental spec-cycle rotation gate (FREETOKEN_MTP_ROTATE).
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.spec import SchedulerSpecMixin


class _Sched(SchedulerSpecMixin):
    def __init__(self, running, spec_mtp=4, controller=None):
        self.spec_mtp = spec_mtp
        self.decode_manager = SimpleNamespace(running_reqs=running)
        self._mtp_controller = controller
        self.finished_reqs = []  # list: SimpleNamespace fakes are unhashable
        self._save_mtp_depth_profile = lambda _depth: None


_seq = iter(range(1_000_000, 1_000_000_000))


def _req(greedy=True, remain=10):
    # distinct uid keeps SimpleNamespace structural equality from aliasing two stubs
    return SimpleNamespace(
        sampling_params=SimpleNamespace(is_greedy=greedy),
        remain_len=remain,
        uid=next(_seq),
        device_len=5,
        input_ids=torch.zeros(5, dtype=torch.int32),
        append_host=None,
    )


def test_single_mode_strict_without_rotation(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_ROTATE", raising=False)
    a, b = _req(), _req()
    s = _Sched([a])
    assert s._pick_spec_req() is a
    assert s._spec_eligible_req() is a
    s2 = _Sched([a, b])
    assert s2._pick_spec_req() is None
    assert s2._spec_eligible_req() is None


def test_rotation_round_robins_eligible_streams(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_ROTATE", "1")
    a, b, c = _req(), _req(remain=1), _req()
    s = _Sched([a, b, c])  # b never owns a cycle (last token -> raw step)
    picks = [s._pick_spec_req() for _ in range(4)]
    assert picks == [a, c, a, c]
    # the stashed owner is what run_spec_step consumes within the iteration
    assert s._spec_eligible_req() is c  # stash follows the last pick
    s._finish_mtp_cycle(None)
    assert s._spec_eligible_req() is None


def test_rotation_skips_req_that_left_running_set(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_ROTATE", "1")
    a, b = _req(), _req()
    s = _Sched([a, b])
    s._pick_spec_req()  # stashes a
    s.decode_manager.running_reqs = [b]  # aborted between begin and run
    assert s._spec_eligible_req() is None


def test_spec_disabled_picks_nothing(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_ROTATE", "1")
    s = _Sched([_req()], spec_mtp=0)
    assert s._spec_eligible_req() is None


def test_begin_requires_controller(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_ROTATE", "1")
    a = _req()
    s = _Sched([a])
    assert s._begin_mtp_cycle() is None and s._spec_cycle_req is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _controller():
    return SimpleNamespace(
        safe_max_k=4,
        begin_request=lambda uid, epoch: None,
        fallback_to_k0=lambda: None,
        next_depth=lambda: 4,
        observe=lambda *a: None,
        consume_learned_depth=lambda: None,
        cost_summaries={},
        selected_depth=4,
    )


def test_batched_begin_stashes_all_greedy_including_tails(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_BATCHED", "1")
    monkeypatch.delenv("FREETOKEN_MTP_ROTATE", raising=False)
    a, tail = _req(), _req(remain=1)
    sched = _Sched([a, tail])
    sched.engine = SimpleNamespace(moe_offload_cache=None, num_pages=8)
    sched._mtp_controller = _controller()
    sched._mtp_controllers = {a.uid: _controller(), tail.uid: _controller()}
    sample = SchedulerSpecMixin._begin_mtp_cycle(sched)
    assert [p[0] for p in sample] == [a, tail]
    assert sched._spec_cycle_reqs == [a, tail]
    # per-uid controllers exist and tail's depth is decided by remain, not the controller
    assert set(sched._mtp_controllers) == {a.uid, tail.uid}
    sched._mtp_cycle_depths = {a.uid: 4, tail.uid: 0}
    for part in sample:
        part[0].input_ids = torch.zeros(part[1] + 1, dtype=torch.int32)
    SchedulerSpecMixin._finish_mtp_cycle(sched, sample)
    assert sched._spec_cycle_req is None


def test_batched_pick_empty_returns_none(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_BATCHED", "1")
    sched = _Sched([])
    assert SchedulerSpecMixin._begin_mtp_cycle(sched) is None


def test_single_mode_begin_unchanged_under_batched_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_BATCHED", raising=False)
    monkeypatch.delenv("FREETOKEN_MTP_ROTATE", raising=False)
    a = _req()
    sched = _Sched([a])
    sched.engine = SimpleNamespace(moe_offload_cache=None, num_pages=8)
    sched._mtp_controller = _controller()
    sample = SchedulerSpecMixin._begin_mtp_cycle(sched)
    assert isinstance(sample, tuple) and sample[0] is a
