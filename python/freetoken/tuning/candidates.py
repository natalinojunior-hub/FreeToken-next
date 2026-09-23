"""v1 candidate matrix + selection rule for ``ft tune``.

Pure, server-free: ``build_candidates`` is a plain list of dicts (data-driven, easy to
extend -- add a dict, not a code path), and ``select_best`` is a pure function over
already-measured results, so both are unit-testable without ever booting a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def build_candidates(*, draft_graph_available: bool, hybrid_capable: bool) -> list[dict[str, Any]]:
    """{spec_mtp: 0, 1} x {defer_replay: True} x {draft_graph: True/False, only if the env
    exists} x {moe_strategy: offload, hybrid if the checkpoint is hybrid-capable}.

    defer_replay is pinned to True (the current, already-default, known-good value --
    nothing in v1 measures turning it off); it's a field on every candidate for symmetry
    with the profile schema, not a swept axis.
    """
    draft_graph_values = (False, True) if draft_graph_available else (False,)
    moe_strategies = ("offload", "hybrid") if hybrid_capable else ("offload",)
    candidates = []
    for spec_mtp in (0, 1):
        for draft_graph in draft_graph_values:
            for moe_strategy in moe_strategies:
                candidates.append(
                    {
                        "spec_mtp": spec_mtp,
                        "defer_replay": True,
                        "draft_graph": draft_graph,
                        "moe_strategy": moe_strategy,
                    }
                )
    return candidates


@dataclass
class CandidateResult:
    settings: dict[str, Any]
    cold_pp: float | None  # None / no value -> treated as a failed run
    committed_tg: float | None
    ttft_s: float | None
    peak_vram_mib: float | None
    had_traceback: bool


def select_best(
    results: list[CandidateResult], *, pp_tolerance: float = 0.97, tg_margin: float = 0.03
) -> CandidateResult | None:
    """Best median committed TG among the candidates that: had no traceback, produced a
    committed_tg, and whose cold_pp is >= ``pp_tolerance`` x the best cold_pp among the
    surviving (no-traceback) candidates. Candidates are in preference order (defaults
    first): a later one displaces the current pick only if its TG beats it by more than
    ``tg_margin`` -- repeated tunes of MTP on/off at 16K swapped places within 2.5%, which is
    noise, not a win. ``None`` when nothing survives -- the caller must not write a profile
    in that case."""
    survivors = [
        r
        for r in results
        if not r.had_traceback and r.cold_pp is not None and r.committed_tg is not None
    ]
    if not survivors:
        return None
    best_pp = max(r.cold_pp for r in survivors)
    floor = pp_tolerance * best_pp
    eligible = [r for r in survivors if r.cold_pp >= floor]
    if not eligible:
        return None
    best = eligible[0]
    for r in eligible[1:]:
        if r.committed_tg > best.committed_tg * (1 + tg_margin):
            best = r
    return best
