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


def test_beneficial_drift_keeps_safe_positive_depth_until_reprobe():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    controller.begin_request("request-2", "epoch-a")

    # The selected depth gets faster enough to trigger drift detection, while remaining
    # statistically cheaper than k0. Drift requests fresh calibration; it is not a safety
    # failure and should not force this request onto k0.
    for _ in range(8):
        controller.observe(1, 0.004, 1)

    assert controller.selected_depth == 1
    assert controller.next_depth() == 1
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


def test_calibration_interleaves_positive_depth_probes_after_k0_baseline():
    controller = AdaptiveMtpController(4)
    controller.begin_request("interleaved", "epoch-a")
    plan = []
    while controller.probing:
        depth = controller.next_depth()
        plan.append(depth)
        elapsed = 0.01 if depth == 0 else 0.005 / (depth + 1)
        controller.observe(depth, elapsed, depth + 1)

    assert plan == [0] * 8 + [1, 2, 3, 4] * 4
    assert {depth: controller.cost_summaries[depth]["samples"] for depth in range(5)} == {
        0: 8, 1: 4, 2: 4, 3: 4, 4: 4
    }


def test_monotonic_cheaper_deeper_selects_max_depth():
    """The cert-model regime: with compact verify state + bf16 SSM keeping the expert pool
    VMM-lazy, every deeper draft is cheaper per committed token (k4 < k3 < k2 < k1 < k0). The
    controller must land on safe_max_k, not lock k0 -- this is the auto operating point the
    +20.5% (86.4 -> 104.1 TG) win depends on reaching reliably."""
    controller = AdaptiveMtpController(4)
    costs = {
        0: [0.0154] * 8,
        1: [0.0140] * 4,
        2: [0.0125] * 4,
        3: [0.0108] * 4,
        4: [0.0096] * 4,
    }
    controller.begin_request("monotonic", "epoch-a")
    while controller.probing:
        depth = controller.next_depth()
        index = controller.cost_summaries[depth]["samples"]
        controller.observe(depth, costs[depth][index], 1)
    assert controller.selected_depth == 4
    assert controller.next_depth() == 4


def test_noisy_but_cheaper_depth_is_not_locked_out_on_noise():
    """A depth truly cheaper than k0 but with high per-sample variance (a short, noisy probe).
    The old 'must be significantly BETTER than k0' gate failed on the noise and locked k0 for
    the whole request (~74 TG instead of ~104); the 'eligible unless significantly WORSE' rule
    keeps the cheaper depth, which is the run1-2 k0-lock fix."""
    controller = AdaptiveMtpController(1)
    # k0 tight at 0.010; k1 noisy but centered clearly below it (mean 0.0075).
    costs = {0: [0.010] * 8, 1: [0.012, 0.004, 0.011, 0.003]}
    controller.begin_request("noisy", "epoch-a")
    while controller.probing:
        depth = controller.next_depth()
        index = controller.cost_summaries[depth]["samples"]
        controller.observe(depth, costs[depth][index], 1)
    assert controller.selected_depth == 1


def test_deepest_within_noise_of_cheapest_wins():
    """campaign37 reality: at calibration k4 (the --spec-mtp cap) measured *more* expensive
    than k3 over 4 noisy samples (a cold outlier inflates its cost and se), but the two are
    within noise of each other (k4's cost lower bound sits below k3's upper bound). The cap
    the user asked for must win -- plain min-cost picked k3 (~102 TG) and left k4 (~104 TG,
    'always max TG') on the table. A depth that is PROVABLY slower (low se, above the
    ceiling) is still correctly skipped, so a real residency cliff is not mistaken for noise."""
    controller = AdaptiveMtpController(4)
    costs = {
        0: [0.0154] * 8,
        1: [0.0140] * 4,
        2: [0.0125] * 4,
        3: [0.0096, 0.0095, 0.0097, 0.0096],  # tight ~0.0096, se ~0 -> the cheapest
        4: [0.030, 0.0090, 0.0095, 0.0092],  # one cold outlier -> cost ~0.0144 but large se
    }
    controller.begin_request("deep", "epoch-a")
    while controller.probing:
        depth = controller.next_depth()
        index = controller.cost_summaries[depth]["samples"]
        controller.observe(depth, costs[depth][index], 1)
    assert controller.selected_depth == 4


def test_provably_slower_deep_depth_is_not_picked():
    """The other side: when the deepest draft is genuinely slower (a residency cliff makes k4
    consistently expensive, low se), it is above the cheapest depth's noise ceiling and must
    NOT be selected -- the controller backs off to the cheapest provably-good depth."""
    controller = AdaptiveMtpController(4)
    costs = {
        0: [0.0154] * 8,
        1: [0.0140] * 4,
        2: [0.0125] * 4,
        3: [0.0096] * 4,  # cheapest, tight
        4: [0.0140] * 4,  # consistently slower than k3, tight (low se) -> provably worse
    }
    controller.begin_request("cliff", "epoch-a")
    while controller.probing:
        depth = controller.next_depth()
        index = controller.cost_summaries[depth]["samples"]
        controller.observe(depth, costs[depth][index], 1)
    assert controller.selected_depth == 3
