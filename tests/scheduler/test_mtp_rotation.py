# CPU checks of the experimental spec-cycle rotation gate (FREETOKEN_MTP_ROTATE).
from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.scheduler.spec import SchedulerSpecMixin


class _Sched(SchedulerSpecMixin):
    def __init__(self, running, spec_mtp=4, controller=None):
        self.spec_mtp = spec_mtp
        self.decode_manager = SimpleNamespace(running_reqs=running)
        self._mtp_controller = controller


_seq = iter(range(1_000_000, 1_000_000_000))


def _req(greedy=True, remain=10):
    # distinct uid keeps SimpleNamespace structural equality from aliasing two stubs
    return SimpleNamespace(
        sampling_params=SimpleNamespace(is_greedy=greedy), remain_len=remain, uid=next(_seq)
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
