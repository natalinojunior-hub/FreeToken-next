"""``benchmarks/bench_pp_tg.py``'s server-log parsing: pulls expert_slots/kv_pages from the
planner's final ``Plan(...)`` line and kv_device_pages/kv_ram_pages from the KV-RAM-tiering
line (engine/memory_planner.py's ``Plan.__str__``, engine/engine.py's tiering log). Loaded
by path since ``benchmarks/`` isn't a package."""

from __future__ import annotations

import pytest
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "bench_pp_tg", Path(__file__).resolve().parents[2] / "benchmarks" / "bench_pp_tg.py"
)
bench_pp_tg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench_pp_tg)


def test_parses_plan_and_tiering_lines():
    log = (
        "[bench] serve: ...\n"
        "Phase F/G: Solving canonical ledger...\n"
        "PLANNING COMPLETE: Plan(chunk=1024, experts=32, kv_pages=512 (65536 tokens), "
        "expert=4.00GB, kv=2.00GB, fixed=0.50GB, transient=0.10GB, total=6.60GB, "
        "free_after=1.00GB, valid=True)\n"
        "KV RAM tiering: 400 device pages, 112 RAM pages (1.75GB pinned host RAM), "
        "8.00 MiB extra device overhead\n"
    )
    out = bench_pp_tg.parse_engine_log(log)
    assert out == {
        "expert_slots": 32,
        "kv_pages": 512,
        "kv_device_pages": 400,
        "kv_ram_pages": 112,
    }


def test_no_tiering_line_leaves_kv_ram_pages_none():
    log = "PLANNING COMPLETE: Plan(chunk=1024, experts=16, kv_pages=256 (32768 tokens))\n"
    out = bench_pp_tg.parse_engine_log(log)
    assert out["expert_slots"] == 16
    assert out["kv_pages"] == 256
    assert out["kv_device_pages"] is None
    assert out["kv_ram_pages"] is None


def test_empty_log_returns_all_none():
    out = bench_pp_tg.parse_engine_log("")
    assert out == {
        "expert_slots": None,
        "kv_pages": None,
        "kv_device_pages": None,
        "kv_ram_pages": None,
    }


def test_takes_last_match_when_planner_logs_twice():
    log = (
        "PLANNING COMPLETE: Plan(chunk=1024, experts=8, kv_pages=100 (1 tokens))\n"
        "PLANNING COMPLETE: Plan(chunk=1024, experts=16, kv_pages=256 (1 tokens))\n"
    )
    out = bench_pp_tg.parse_engine_log(log)
    assert out["expert_slots"] == 16
    assert out["kv_pages"] == 256


def test_serve_arg_value_parses_space_joined_entry():
    assert bench_pp_tg._serve_arg_value(["--spec-mtp 3"], "--spec-mtp") == "3"
    assert bench_pp_tg._serve_arg_value(["--spec-mtp=3"], "--spec-mtp") == "3"
    assert bench_pp_tg._serve_arg_value(["--other 1"], "--spec-mtp") is None


def test_tg_curve_windows():
    stamps = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]  # 3 steps in 0.3 s, then 3 steps in 0.5 s
    curve = bench_pp_tg.tg_curve(stamps, 3)
    assert curve == pytest.approx([10.0, 6.0])
    assert bench_pp_tg.tg_curve([0.0, 0.1, 0.2, 0.3, 0.4], 3) == pytest.approx([10.0, 10.0])
