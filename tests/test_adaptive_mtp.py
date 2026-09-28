import pytest

from freetoken.scheduler.adaptive_mtp import AdaptiveMtpController


def _calibrate(controller, uid="request-1", epoch="epoch-a", costs=None):
    controller.begin_request(uid, epoch)
    costs = costs or {0: [0.010] * 8, 1: [0.008] * 4}
    while controller.probing:
        depth = controller.next_depth()
        samples = costs[depth]
        index = controller.cost_summaries[depth]["samples"]
        elapsed = samples[index] if index < len(samples) else samples[-1]
        controller.observe(depth, elapsed, 1)


def test_weighted_seconds_per_committed_token_not_mean_of_ratios():
    controller = AdaptiveMtpController(1)
    costs = {
        0: [10.0, 100.0] + [100.0] * 6,
        1: [5.0, 200.0] + [200.0] * 2,
    }
    tokens = {0: [1, 100] + [100] * 6, 1: [1, 100] + [100] * 2}
    controller.begin_request("weighted", 0)
    seen = {0: 0, 1: 0}
    while controller.probing:
        depth = controller.next_depth()
        index = seen[depth]
        controller.observe(depth, costs[depth][index], tokens[depth][index])
        seen[depth] += 1

    assert (
        sum(t / n for t, n in zip(costs[1][:2], tokens[1][:2])) / 2
        < sum(t / n for t, n in zip(costs[0][:2], tokens[0][:2])) / 2
    )
    assert controller.cost_summaries[0]["seconds_per_token"] == pytest.approx(710 / 701)
    assert controller.cost_summaries[1]["seconds_per_token"] == pytest.approx(605 / 301)
    assert controller.selected_depth == 0
    assert controller.next_depth() == 0


def test_small_measured_gain_is_accepted_without_fixed_percent_threshold():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.0099] * 4})
    assert controller.selected_depth == 1


def test_bounded_observation_windows():
    controller = AdaptiveMtpController(0)
    controller.begin_request("bounded", 1)
    for _ in range(8):
        controller.observe(0, 0.01, 1)
    for _ in range(40):
        controller.observe(0, 0.01, 1)
    assert controller.cost_summaries[0]["samples"] == 32


def test_epoch_change_clears_learning_and_same_request_begin_is_noop():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    plan_depth = controller.next_depth()
    controller.begin_request("request-1", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == plan_depth == 1

    controller.begin_request("request-2", "epoch-b")
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0


def test_abandoned_calibration_discards_partial_samples_and_restarts():
    controller = AdaptiveMtpController(1)
    controller.begin_request("partial", "epoch-a")
    for _ in range(8):
        controller.observe(0, 0.010, 1)
    assert controller.probing
    assert controller.next_depth() == 1

    controller.begin_request("next", "epoch-a")
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0


def test_forced_fallback_mid_calibration_restarts_full_probe_next_request():
    controller = AdaptiveMtpController(2)
    controller.begin_request("partial", "epoch-a")
    for _ in range(8):
        controller.observe(0, 0.010, 1)
    assert controller.next_depth() == 1

    controller.fallback_to_k0()
    controller.begin_request("next", "epoch-a")
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0
    assert controller.cost_summaries[1]["samples"] == 0


def test_periodic_reselection_can_enter_terminal_k0():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    controller._best_depth = lambda: 0

    for _ in range(8):
        controller.observe(1, 0.008, 1)
    assert controller.selected_depth == 0
    assert controller.next_depth() == 0


def test_fallback_is_terminal_until_next_request():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    controller.fallback_to_k0()
    for _ in range(3):
        controller.observe(0, 0.010, 1)
        assert controller.next_depth() == 0
    controller.begin_request("request-2", "epoch-a")
    assert controller.next_depth() == 1


def test_learned_depth_retained_and_measured_drift_schedules_probe():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    controller.begin_request("request-2", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 1

    for _ in range(8):
        if controller.next_depth() == 0:
            break
        controller.observe(1, 0.020, 1)
    assert controller.next_depth() == 0
    controller.begin_request("request-3", "epoch-a")
    assert controller.probing
    assert controller.next_depth() == 0


@pytest.mark.parametrize("safe_max_k", [-1, 5, 1.0, True])
def test_rejects_invalid_safe_depth(safe_max_k):
    with pytest.raises(ValueError):
        AdaptiveMtpController(safe_max_k)


@pytest.mark.parametrize("elapsed", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_elapsed(elapsed):
    controller = AdaptiveMtpController(0)
    controller.begin_request("invalid", 1)
    with pytest.raises(ValueError):
        controller.observe(0, elapsed, 1)


@pytest.mark.parametrize("committed", [0, -1, 1.5, True])
def test_rejects_invalid_committed_tokens(committed):
    controller = AdaptiveMtpController(0)
    controller.begin_request("invalid", 1)
    with pytest.raises(ValueError):
        controller.observe(0, 0.01, committed)
