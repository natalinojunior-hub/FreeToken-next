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
    assert not controller.probing  # a safety/tail clamp does not invent a fresh economics probe
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


def test_beneficial_drift_keeps_safe_positive_depth_without_reprobe():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    controller.begin_request("request-2", "epoch-a")

    # The selected depth gets faster while remaining cheaper than k0. Beneficial drift is
    # neither a safety failure nor a reason to pay for another probe.
    for _ in range(8):
        controller.observe(1, 0.004, 1)

    assert controller.selected_depth == 1
    assert controller.next_depth() == 1
    controller.begin_request("request-3", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 1


def test_cost_rise_keeps_measured_winner_and_counterfactual_audits():
    controller = AdaptiveMtpController(2)
    _calibrate(controller, costs={0: [0.020] * 8, 1: [0.014] * 4, 2: [0.008] * 4})
    for _ in range(8):
        controller.observe(2, 0.010, 1)
    assert controller._stats[2].drifted(harmful_only=True)
    assert controller._best_depth() == 2

    controller.begin_request("still-cheapest", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 2
    # The measured winner still receives fresh k0 comparisons on the audit schedule:
    # a fresh campaign just proved every depth, so the counterfactual is amortized.
    for _ in range(controller._baseline_interval - 8):
        controller.observe(2, 0.010, 1)
    assert controller.probing
    assert controller.next_depth() == 0


@pytest.mark.parametrize("safe_max_k", [-1, 1.0, True])
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
        0: 8,
        1: 4,
        2: 4,
        3: 4,
        4: 4,
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


def test_cold_calibration_keeps_requested_cap_when_candidate_win_is_uncertain():
    """A short cold probe retains the requested cap unless a lower depth clearly wins."""
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


def test_warmed_depth_audit_recovers_k5_without_promoting_k6():
    controller = AdaptiveMtpController(6)
    _calibrate(
        controller,
        costs={
            0: [0.020] * 8,
            1: [0.015] * 4,
            2: [0.013] * 4,
            3: [0.010] * 4,
            4: [0.012] * 4,
            5: [0.040, 0.005, 0.020, 0.020],  # noisy cold measurements
            6: [0.030] * 4,  # proven slower; never promote from one cheap outlier
        },
    )
    assert controller.selected_depth == 3
    assert controller._depth_audit_depth == 5
    assert controller._depth_audit_interval == controller._baseline_interval == 512

    for _ in range(4):
        for _ in range(2 * controller._baseline_interval + 16):
            depth = controller.next_depth()
            if depth == 5:
                break
            controller.observe(depth, 0.020 if depth == 0 else 0.010, 1)
        assert controller.next_depth() == 5
        controller.observe(5, 0.007, 1)
        assert controller.selected_depth != 6

    assert controller.selected_depth == 5
    assert controller.consume_learned_depth() == 5


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


# --- warm start from a persisted depth profile (tuning.mtp_profile) -----------------------


def test_warm_profile_seeds_first_epoch_without_startup_probe():
    """The cached depth avoids startup calibration but stays provisional."""
    controller = AdaptiveMtpController(4, profiled_depth=4)
    controller.begin_request("warm", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 4
    for _ in range(8):
        controller.observe(4, 0.0096, 4)
    controller.begin_request("warm2", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 4
    assert controller.consume_learned_depth() is None  # a warm start learned nothing new


def test_warm_profile_is_invalidated_by_a_later_economic_epoch():
    """A cached optimum must be remeasured after context/cache economics change."""
    controller = AdaptiveMtpController(4, profiled_depth=4)
    controller.begin_request("warm", "epoch-a")
    assert controller.next_depth() == 4
    controller.begin_request("warm", "epoch-b")
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0


def test_costly_initial_profile_audits_k0_then_falls_back():
    """A profile that is wrong in the first prompt regime gets a fresh k0 comparison."""
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("warm", "epoch-a")
    assert controller.next_depth() == 1
    for i in range(512):
        controller.observe(1, 0.030, 1)
        if i < 511:
            assert controller.next_depth() == 1
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[1]["samples"] == 32
    for _ in range(8):
        controller.observe(0, 0.010, 1)
    assert controller.selected_depth == 0
    assert controller.next_depth() == 0
    controller.begin_request("after-audit", "epoch-a")
    assert controller.probing  # an economic k0 fallback is rechecked next request
    assert controller.next_depth() == 0


def test_warm_k0_audit_retains_beneficial_seed_and_never_selects_unmeasured_depth():
    controller = AdaptiveMtpController(3, profiled_depth=2)
    controller.begin_request("warm", "epoch-a")
    for _ in range(512):
        controller.observe(2, 0.008, 1)
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[2]["samples"] == 32
    for _ in range(8):
        controller.observe(0, 0.020, 1)
    assert controller.selected_depth == 2
    assert controller.next_depth() == 2
    assert {k: controller.cost_summaries[k]["samples"] for k in (1, 3)} == {1: 0, 3: 0}


def test_warm_audit_accumulates_across_requests_and_keeps_partial_baseline():
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("one", "epoch-a")
    for _ in range(256):
        controller.observe(1, 0.008, 1)
    controller.begin_request("two", "epoch-a")
    for _ in range(256):
        controller.observe(1, 0.008, 1)
    assert controller.probing
    assert controller.next_depth() == 0
    for _ in range(3):
        controller.observe(0, 0.020, 1)
    controller.begin_request("three", "epoch-a")
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 3
    for _ in range(5):
        controller.observe(0, 0.020, 1)
    assert controller.selected_depth == 1
    assert controller.next_depth() == 1


def test_calibrated_controller_periodically_audits_fresh_k0():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.010] * 8, 1: [0.008] * 4})
    assert controller.selected_depth == 1
    # A completed campaign is the freshest possible counterfactual proof: the first audit
    # waits a full amortized interval, then quadruples on each confirmation.
    assert controller._baseline_interval == 512
    for _ in range(controller._baseline_interval):
        controller.observe(1, 0.008, 1)
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0  # discard stale baseline
    for _ in range(8):
        controller.observe(0, 0.020, 1)
    assert controller.selected_depth == 1
    assert controller.next_depth() == 1
    assert controller._baseline_interval == 2048
    for _ in range(controller._baseline_interval - 1):
        controller.observe(1, 0.008, 1)
        assert not controller.probing
    controller.observe(1, 0.008, 1)
    assert controller.probing  # later audits amortize a validated counterfactual
    assert controller.next_depth() == 0


def test_audit_cost_variation_does_not_invalidate_its_measured_winner():
    controller = AdaptiveMtpController(1)
    _calibrate(controller, costs={0: [0.020] * 8, 1: [0.008] * 4})
    for _ in range(controller._baseline_interval):
        controller.observe(1, 0.008, 1)
    for elapsed in [0.020] * 4 + [0.030] * 4:
        controller.observe(0, elapsed, 1)
    assert controller._stats[0].drifted(harmful_only=True)
    assert controller.selected_depth == 1
    controller.begin_request("after-fresh-audit", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 1


def test_warm_drift_without_raw_baseline_does_not_discard_profile():
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("profiled", "epoch-a")
    for _ in range(8):
        controller.observe(1, 0.008, 1)
    for _ in range(8):
        controller.observe(1, 0.012, 1)
    assert controller.next_depth() == 1  # never changes depth mid-request
    controller.begin_request("after-profiled", "epoch-a")
    assert not controller.probing  # drift has no meaning without a current k=0 comparison
    assert controller.next_depth() == 1

    beneficial = AdaptiveMtpController(1, profiled_depth=1)
    beneficial.begin_request("beneficial", "epoch-a")
    for _ in range(8):
        beneficial.observe(1, 0.012, 1)
    for _ in range(8):
        beneficial.observe(1, 0.008, 1)
    beneficial.begin_request("after-beneficial", "epoch-a")
    assert not beneficial.probing
    assert beneficial.next_depth() == 1


def test_small_cost_rise_does_not_reprobe():
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("noise", "epoch-a")
    for _ in range(8):
        controller.observe(1, 0.0100, 1)
    for _ in range(8):
        controller.observe(1, 0.0104, 1)
    controller.begin_request("after-noise", "epoch-a")
    assert not controller.probing
    assert controller.next_depth() == 1


def test_cold_calibration_exposes_learned_depth_exactly_once():
    """The engine persists whatever a fresh calibration converges to (and only that)."""
    controller = AdaptiveMtpController(4)
    costs = {0: [0.0154] * 8, 1: [0.0140] * 4, 2: [0.0125] * 4, 3: [0.0108] * 4, 4: [0.0096] * 4}
    controller.begin_request("cold", "epoch-a")
    assert controller.consume_learned_depth() is None  # nothing learned before calibration
    while controller.probing:
        depth = controller.next_depth()
        index = controller.cost_summaries[depth]["samples"]
        controller.observe(depth, costs[depth][index], 1)
    assert controller.selected_depth == 4
    assert controller.consume_learned_depth() == 4  # surfaced for saving
    assert controller.consume_learned_depth() is None  # consumed once


def test_cold_calibration_keeps_requested_cap_when_lower_depth_is_not_a_clear_win():
    controller = AdaptiveMtpController(5)
    costs = {
        0: [0.023] * 8,
        1: [0.017] * 4,
        2: [0.014] * 4,
        3: [0.00905, 0.0090, 0.0091, 0.00907],
        4: [0.012] * 4,
        5: [0.008, 0.009, 0.014, 0.018],
    }
    _calibrate(controller, costs=costs)
    assert controller.selected_depth == 5
    assert controller.consume_learned_depth() == 5


def test_force_depth_overrides_profile_and_never_saves(monkeypatch):
    """A measurement force-depth pin owns the depth: it wins over a profile and produces no
    learned depth to persist (a forced run is an instrument, not a calibration)."""
    monkeypatch.setenv("FREETOKEN_MTP_FORCE_DEPTH", "2")
    controller = AdaptiveMtpController(4, profiled_depth=4)
    controller.begin_request("forced", "epoch-a")
    assert controller.next_depth() == 2  # force wins over the profiled 4
    for _ in range(40):
        controller.observe(2, 0.010, 3)
    assert not controller.probing
    assert controller.next_depth() == 2
    assert controller.consume_learned_depth() is None


@pytest.mark.parametrize("bad", [0, 5, -1])
def test_invalid_profiled_depth_is_ignored(bad):
    """Only 1..safe_max_k is honored; an out-of-range or zero profile falls back to a cold
    calibration (a learned 0 is never cached, so it can never be warm-started)."""
    controller = AdaptiveMtpController(4, profiled_depth=bad)
    controller.begin_request("bad", "epoch-a")
    assert controller.probing  # cold calibration, not a warm start


def test_warm_start_beneficial_drift_keeps_profile():
    """A beneficial warmup shift keeps the seed and does not trigger a full probe."""
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("warm", "epoch-a")
    assert controller.next_depth() == 1
    for _ in range(8):
        controller.observe(1, 0.030, 1)  # early: pool cold, slow
    for _ in range(8):
        controller.observe(1, 0.008, 1)  # steady: fast -> beneficial drift
    controller.begin_request("warm2", "epoch-a")
    assert not controller.probing  # profile survived the warmup drift
    assert controller.next_depth() == 1


def test_warm_start_small_cost_rise_does_not_invalidate():
    """A rise below the material-drift floor is noise, not a reason to re-probe."""
    controller = AdaptiveMtpController(1, profiled_depth=1)
    controller.begin_request("warm", "epoch-a")
    assert controller.next_depth() == 1
    for _ in range(8):
        controller.observe(1, 0.0100, 1)  # steady
    for _ in range(8):
        controller.observe(1, 0.0104, 1)  # a ~4% rise: noise, not a regime change
    controller.begin_request("warm2", "epoch-a")
    assert not controller.probing  # small rise ignored -> still warm
    assert controller.next_depth() == 1
