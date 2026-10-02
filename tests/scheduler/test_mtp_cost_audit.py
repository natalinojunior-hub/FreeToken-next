from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.adaptive_mtp import AdaptiveMtpController
from freetoken.scheduler import spec
from freetoken.scheduler.spec import SchedulerSpecMixin


def _req(uid, output_len):
    return Req(
        input_ids=torch.arange(64, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=0),
        cache_handle=SimpleNamespace(cached_len=0),
    )


def test_live_residency_boundary_keeps_learned_mtp_economics():
    controller = AdaptiveMtpController(4, profiled_depth=4)
    cache = SimpleNamespace(cache_size=8192, live_caps=[4096, 1024], _vmm_arenas=[object()])
    scheduler = SimpleNamespace(
        _mtp_controller=controller,
        decode_manager=SimpleNamespace(running_reqs={_req(1, 40)}),
        engine=SimpleNamespace(moe_offload_cache=cache, num_pages=128),
    )
    SchedulerSpecMixin._begin_mtp_cycle(scheduler)
    controller.observe(4, 0.04, 5)

    # A one-row physical shrink crosses both old bit buckets without changing the runtime.
    cache.live_caps = [4095, 1023]
    scheduler.decode_manager.running_reqs = {_req(2, 40)}
    SchedulerSpecMixin._begin_mtp_cycle(scheduler)

    assert controller.next_depth() == 4
    assert not controller.probing
    assert controller.cost_summaries[4]["samples"] == 1


def test_cost_audit_carries_tail_raw_sample_across_requests(monkeypatch):
    from freetoken.scheduler import spec as _spec

    monkeypatch.setattr(_spec, "_SPEC_WARMUP_CYCLES", 0)
    controller = AdaptiveMtpController(1, profiled_depth=1)
    saved = []
    scheduler = SimpleNamespace(
        _mtp_controller=controller,
        decode_manager=SimpleNamespace(running_reqs=set()),
        engine=SimpleNamespace(moe_offload_cache=None, num_pages=32),
        finished_reqs=set(),
        _save_mtp_depth_profile=saved.append,
    )
    step_s = [0.03]
    now = [0.0]

    def perf_counter():
        current = now[0]
        now[0] += step_s[0]
        return current

    monkeypatch.setattr(spec.time, "perf_counter", perf_counter)

    def decode_cycle(req, elapsed_s, *, finish=False):
        scheduler.decode_manager.running_reqs = {req}
        step_s[0] = elapsed_s
        sample = SchedulerSpecMixin._begin_mtp_cycle(scheduler)
        scheduler._mtp_cycle_depth = controller.next_depth()
        req.complete_one()
        req.append_host(torch.tensor([int(req.input_ids[-1]) + 1], dtype=torch.int32))
        if finish:
            scheduler.finished_reqs.add(req)
        SchedulerSpecMixin._finish_mtp_cycle(scheduler, sample)

    # Accumulate the 512 positive observations across short requests in one context epoch.
    for uid in range(32):
        req = _req(uid, 40)
        for _ in range(16):
            decode_cycle(req, 0.03)
    assert controller.probing
    assert controller.next_depth() == 0
    assert controller.cost_summaries[0]["samples"] == 0

    # The final-token guard switches to raw decode while preserving the audit sample.
    tail = _req(3, 2)
    tail.complete_one()
    tail.append_host(torch.tensor([999], dtype=torch.int32))
    decode_cycle(tail, 0.01, finish=True)
    assert controller.cost_summaries[0]["samples"] == 1
    assert controller._auditing
    assert controller.next_depth() == 0

    # A new request resumes the same audit. Seven more raw samples prove the cached k=1 seed
    # slower than k0; the audit must finish cleanly and must not persist a failed profile.
    final = _req(4, 12)
    for _ in range(7):
        decode_cycle(final, 0.01, finish=True)
    assert controller.cost_summaries[0]["samples"] == 8
    assert controller.selected_depth == 0
    assert controller.next_depth() == 0
    assert saved == []


def test_harmful_drift_reprobe_uses_fresh_cost_windows():
    controller = AdaptiveMtpController(2)
    controller.begin_request("initial", "ctx")
    initial_costs = {0: 0.010, 1: 0.009, 2: 0.008}
    while controller.probing:
        depth = controller.next_depth()
        controller.observe(depth, initial_costs[depth], 1)
    assert controller.selected_depth == 2

    # A real regime shift makes the selected depth harmful and another measured depth wins.
    for _ in range(8):
        controller.observe(2, 0.016, 1)
    assert controller._needs_reprobe
    assert controller._best_depth() == 1

    controller.begin_request("after-drift", "ctx")
    assert controller.probing
    assert all(summary["samples"] == 0 for summary in controller.cost_summaries.values())

    # A fresh campaign sees k=2 become the best depth and selects it.
    fresh_costs = {0: 0.010, 1: 0.011, 2: 0.007}
    while controller.probing:
        depth = controller.next_depth()
        controller.observe(depth, fresh_costs[depth], 1)
    assert controller.selected_depth == 2
    assert {k: v["samples"] for k, v in controller.cost_summaries.items()} == {
        0: 8,
        1: 4,
        2: 4,
    }


def test_output_budget_truncation_keeps_controller_depth_and_reports_actual_depth(
    monkeypatch, caplog
):
    controller = AdaptiveMtpController(4, profiled_depth=4)
    controller.begin_request("tail", (16384,))
    req = SimpleNamespace(
        uid="tail",
        sampling_params=SimpleNamespace(is_greedy=True),
        remain_len=2,
        device_len=16384,
        cached_len=0,
    )
    scheduler = SimpleNamespace(
        spec_mtp=4,
        decode_manager=SimpleNamespace(running_reqs=[req]),
        _mtp_controller=controller,
        _spec_eligible_req=lambda: req,
        engine=SimpleNamespace(model=SimpleNamespace(mtp=object())),
        _flush_deferred_replays=lambda: None,
        _snapshot_qsa_state=lambda _req: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    monkeypatch.delenv("FREETOKEN_SPEC_LOOKUP", raising=False)

    try:
        SchedulerSpecMixin.run_spec_step(scheduler)
    except RuntimeError as exc:
        assert str(exc) == "stop"
    else:
        raise AssertionError("expected to stop after the truncated depth is resolved")

    # k=1 is the correct budget-limited execution depth; the controller still expects k=4.
    assert scheduler._mtp_cycle_depth == 1
    assert scheduler._mtp_cycle_observe is False
    assert controller.next_depth() == 4
    assert controller.selected_depth == 4
    assert controller.cost_summaries[1]["samples"] == 0
    assert controller.cost_summaries[4]["samples"] == 0

    req.input_ids = torch.tensor([1, 2])
    saved = []
    scheduler._save_mtp_depth_profile = saved.append
    scheduler._mtp_distribution = {}
    scheduler.finished_reqs = [req]
    monkeypatch.setattr(spec.time, "perf_counter", lambda: 1.0)
    SchedulerSpecMixin._finish_mtp_cycle(scheduler, (req, 1, 0.0))
    assert controller.next_depth() == 4
    assert controller.selected_depth == 4
    assert saved == []
    assert scheduler._mtp_distribution[1][:2] == [1, 1]
    assert "[mtp-economics] uid=tail" in caplog.text


def test_k0_audit_cadence_doubles_while_cached_depth_keeps_winning():
    controller = AdaptiveMtpController(4, profiled_depth=4)
    controller.begin_request("amortize", 0)

    def run(n, elapsed_per_step, committed):
        for _ in range(n):
            depth = controller.next_depth()
            controller.observe(depth, elapsed_per_step, committed)

    def run_to_audit(elapsed_per_step, committed):
        while controller._cycles_since_baseline < controller._baseline_interval:
            run(1, elapsed_per_step, committed)

    run(512, 0.010, 5)  # cached profile: delay the first raw audit until the calibrated window
    run(8, 0.040, 1)
    assert controller._baseline_interval == 2048
    run_to_audit(0.010, 5)
    run(8, 0.040, 1)
    assert controller._baseline_interval == 4096
    run_to_audit(0.010, 5)
    run(8, 0.040, 1)
    assert controller._baseline_interval == 4096
    assert controller.selected_depth == 4
    assert controller.consume_learned_depth() == 4
