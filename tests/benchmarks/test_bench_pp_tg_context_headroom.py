"""Tests for context headroom reporting in benchmarks/bench_pp_tg.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "bench_pp_tg", Path(__file__).resolve().parents[2] / "benchmarks" / "bench_pp_tg.py"
)
bench_pp_tg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench_pp_tg)


def test_parse_context_feasibility_standard():
    log = (
        "2026-04-18 10:00:00 [INFO] KV context feasibility (0.12 MiB per 16-token page, "
        "12.34 GB left for KV after the expert cache): 128K fits, 256K fits, "
        "512K 3.12 GB short, 1024K 15.40 GB short\n"
    )
    assert bench_pp_tg.parse_context_feasibility(log) == 256 * 1024


def test_parse_context_feasibility_all_fit():
    log = (
        "KV context feasibility (0.08 MiB per 16-token page, 24.00 GB left for KV): "
        "128K fits, 256K fits, 512K fits, 1024K fits\n"
    )
    assert bench_pp_tg.parse_context_feasibility(log) == 1024 * 1024


def test_parse_context_feasibility_none_fit():
    log = (
        "KV context feasibility (0.12 MiB per 16-token page, 0.50 GB left for KV): "
        "128K 1.50 GB short, 256K 5.20 GB short\n"
    )
    assert bench_pp_tg.parse_context_feasibility(log) is None


def test_parse_context_feasibility_empty_or_missing():
    assert bench_pp_tg.parse_context_feasibility("") is None
    assert bench_pp_tg.parse_context_feasibility("other server log lines\n") is None


def test_parse_context_feasibility_takes_last():
    log = (
        "KV context feasibility (...): 128K fits, 256K 1.00 GB short\n"
        "KV context feasibility (...): 128K fits, 256K fits, 512K 2.00 GB short\n"
    )
    assert bench_pp_tg.parse_context_feasibility(log) == 256 * 1024


def test_extract_context_headroom(tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "KV context feasibility (...): 128K fits, 256K fits, 512K 1.0 GB short\n"
    )
    res = bench_pp_tg.extract_context_headroom(str(log_file), 45.2)
    assert res["max_runnable_context"] == 262144
    assert res["method"] == "projected"
    sim = res["simulated_tg"]
    assert len(sim) == 4
    assert sim[0] == {"fraction": "25%", "tokens": 65536, "tg_tok_s": 45.2, "method": "projected"}
    assert sim[1] == {"fraction": "50%", "tokens": 131072, "tg_tok_s": 45.2, "method": "projected"}
    assert sim[2] == {"fraction": "75%", "tokens": 196608, "tg_tok_s": 45.2, "method": "projected"}
    assert sim[3] == {"fraction": "100%", "tokens": 262144, "tg_tok_s": 45.2, "method": "projected"}


def test_extract_context_headroom_missing_log():
    res = bench_pp_tg.extract_context_headroom(None, 45.2)
    assert res["max_runnable_context"] is None
    assert res["simulated_tg"] == []


def test_cli_context_headroom_flag():
    args_default = bench_pp_tg.parse_args(["--model", "test"])
    assert args_default.context_headroom is True

    args_off = bench_pp_tg.parse_args(["--model", "test", "--no-context-headroom"])
    assert args_off.context_headroom is False

    args_on = bench_pp_tg.parse_args(["--model", "test", "--context-headroom"])
    assert args_on.context_headroom is True


def test_print_context_headroom(capsys):
    headroom = {
        "max_runnable_context": 262144,
        "simulated_tg": [
            {"fraction": "25%", "tokens": 65536, "tg_tok_s": 42.5, "method": "projected"},
            {"fraction": "50%", "tokens": 131072, "tg_tok_s": 42.5, "method": "projected"},
            {"fraction": "75%", "tokens": 196608, "tg_tok_s": 42.5, "method": "projected"},
            {"fraction": "100%", "tokens": 262144, "tg_tok_s": 42.5, "method": "projected"},
        ],
    }
    bench_pp_tg.print_context_headroom("test-run", headroom)
    captured = capsys.readouterr().out
    assert "==== [test-run] context headroom ====" in captured
    assert "max runnable context: 262144 tokens (from engine KV feasibility)" in captured
    assert "simulated TG @  25% (  65536 tok):  42.50 tok/s [projected]" in captured
    assert "simulated TG @ 100% ( 262144 tok):  42.50 tok/s [projected]" in captured
    assert "projected (decode TG is context-flat to the device hot-window; see --kv-reserve-tokens)" in captured
