"""``freetoken.tuning.candidates``: the v1 candidate matrix and the selection rule.
Pure functions, no server -- results are hand-built ``CandidateResult`` fakes."""

from __future__ import annotations

from freetoken.tuning.candidates import CandidateResult, build_candidates, select_best


def test_matrix_size_without_draft_graph_or_hybrid():
    c = build_candidates(draft_graph_available=False, hybrid_capable=False)
    # {spec_mtp 0,1} x {draft_graph False} x {moe_strategy offload} = 2
    assert len(c) == 2
    assert all(x["draft_graph"] is False for x in c)
    assert all(x["moe_strategy"] == "offload" for x in c)


def test_matrix_size_with_draft_graph_and_hybrid():
    c = build_candidates(draft_graph_available=True, hybrid_capable=True)
    # {spec_mtp 0,1} x {draft_graph False,True} x {moe_strategy offload,hybrid} = 8
    assert len(c) == 8
    assert {x["moe_strategy"] for x in c} == {"offload", "hybrid"}
    assert {x["draft_graph"] for x in c} == {False, True}
    assert c[0]["draft_graph"] is True  # engine default first: wins ties in select_best


def test_matrix_is_data_driven_dicts():
    c = build_candidates(draft_graph_available=False, hybrid_capable=False)
    assert all(isinstance(x, dict) for x in c)
    assert all(set(x) == {"spec_mtp", "defer_replay", "draft_graph", "moe_strategy"} for x in c)


def _r(tg, pp, traceback=False, settings=None):
    return CandidateResult(
        settings or {},
        cold_pp=pp,
        committed_tg=tg,
        ttft_s=0.1,
        peak_vram_mib=100.0,
        had_traceback=traceback,
    )


def test_select_best_picks_highest_tg_within_pp_tolerance():
    results = [
        _r(tg=40.0, pp=1000.0, settings={"id": "a"}),
        _r(tg=50.0, pp=970.0, settings={"id": "b"}),  # 0.97x -> still eligible, wins on tg
        _r(tg=60.0, pp=900.0, settings={"id": "c"}),  # below 0.97 floor -> excluded
    ]
    best = select_best(results)
    assert best.settings["id"] == "b"


def test_select_best_excludes_tracebacks():
    results = [
        _r(tg=100.0, pp=1000.0, traceback=True, settings={"id": "crashed"}),
        _r(tg=10.0, pp=900.0, settings={"id": "ok"}),
    ]
    best = select_best(results)
    assert best.settings["id"] == "ok"


def test_select_best_excludes_missing_measurements():
    results = [
        CandidateResult(
            {"id": "no-pp"},
            cold_pp=None,
            committed_tg=10.0,
            ttft_s=None,
            peak_vram_mib=None,
            had_traceback=False,
        ),
        _r(tg=5.0, pp=500.0, settings={"id": "ok"}),
    ]
    best = select_best(results)
    assert best.settings["id"] == "ok"


def test_select_best_returns_none_when_everything_failed():
    results = [_r(tg=1.0, pp=1.0, traceback=True)]
    assert select_best(results) is None


def test_select_best_returns_none_on_empty_list():
    assert select_best([]) is None


def test_later_candidate_needs_a_margin_over_noise():
    from freetoken.tuning.candidates import CandidateResult, select_best

    def r(tg):
        return CandidateResult({"tg": tg}, 2000.0, tg, 1.0, 1.0, False)

    assert select_best([r(38.0), r(38.9)]).committed_tg == 38.0  # +2.4%: noise, keep default
    assert select_best([r(38.0), r(40.0)]).committed_tg == 40.0  # +5.3%: a real win


def test_tune_prompts_paired_across_candidates():
    from freetoken.tuning.tune_cli import _unique_prompt

    # rep N decodes the same text on every candidate boot; reps within a boot differ
    assert _unique_prompt("x", 1) == _unique_prompt("x", 1)
    assert _unique_prompt("x", 1) != _unique_prompt("x", 2)
