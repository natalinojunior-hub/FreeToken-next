"""Exercise the real runner order/warmup policy without a server or GPU."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "bench_arms", Path(__file__).parents[2] / "scripts/bench-arms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interleaves_restores_boot_env_and_warms_only_recapture(monkeypatch, tmp_path):
    runner = load_runner()
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            [
                {"name": "a", "env": {}, "recapture": True},
                {"name": "b", "env": {"FREETOKEN_VERIFY_GRAPH": "0"}, "recapture": True},
                {"name": "raw", "env": {"FREETOKEN_MTP_FORCE_DEPTH": 0}, "recapture": True},
            ]
        )
    )
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench-arms.py",
            "--arms",
            str(arms),
            "--model",
            "fake",
            "--tokens",
            "16",
            "--decode",
            "4",
            "--warmups",
            "2",
            "--repeats",
            "2",
            "--json",
            str(out),
        ],
    )
    monkeypatch.delenv("FREETOKEN_VERIFY_GRAPH", raising=False)
    monkeypatch.delenv("FREETOKEN_MTP_FORCE_DEPTH", raising=False)
    calls = []
    boot = []
    active = {}
    monkeypatch.setattr(runner.bench, "build_prompt_text", lambda *args: "prompt")
    monkeypatch.setattr(runner.bench, "get_json", lambda *args: {"data": [{"id": "fake"}]})
    monkeypatch.setattr(runner.bench, "stop_server", lambda proc: None)

    # Keep stdout open until teardown; readiness pipe must not look like a dead server.
    import threading

    stopped = threading.Event()

    class Pipe:
        def __iter__(self):
            yield b"API server is ready to serve\n"
            stopped.wait(5)

    def popen(*args, **kwargs):
        boot.append(kwargs["env"])
        return SimpleNamespace(stdout=Pipe(), pid=123)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.bench, "stop_server", lambda proc: stopped.set())

    def update(origin, payload):
        active.clear()
        active.update(payload["env"])
        return {
            "status": "ok",
            "applied": payload["env"],
            "recaptured": payload["recapture_graphs"],
        }

    def warm(*args, **kwargs):
        calls.append(("warm", dict(active)))

    def run(*args, **kwargs):
        assert kwargs == {"monitor": False}
        calls.append(("run", dict(active)))
        return {
            "decode_tok_s": 10.0,
            "prefill_tok_s": 20.0,
            "ttft_ms": 800.0,
            "e2e_ms": 1000.0,
            "output_sha1": "raw-diff" if active.get("FREETOKEN_MTP_FORCE_DEPTH") == "0" else "same",
            "prompt_tokens": 16,
            "completion_tokens": 4,
        }

    monkeypatch.setattr(runner, "apply_arm", update)
    monkeypatch.setattr(runner.bench, "stream_completion", warm)
    monkeypatch.setattr(runner.bench, "one_run", run)
    assert runner.main() == 0
    assert len(boot) == 1 and boot[0]["FREETOKEN_DEBUG_RUNTIME"] == "1"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["name"] for row in rows] == ["a", "b", "raw"]
    assert all(len(row["runs"]) == 2 for row in rows)
    assert rows[0]["sha_matches_first"] and not rows[2]["sha_matches_first"]
    measured = [env for kind, env in calls if kind == "run"]
    assert [env["FREETOKEN_MTP_FORCE_DEPTH"] for env in measured] == [
        None,
        None,
        "0",
        None,
        None,
        "0",
    ]
    assert [env["FREETOKEN_VERIFY_GRAPH"] for env in measured] == [None, "0", None, None, "0", None]
    # Initial arm a, then B change to b and B restoration to raw: each warms once.
    assert sum(kind == "warm" for kind, _ in calls) == 6


def test_rejects_layout_arms_before_boot(tmp_path):
    runner = load_runner()
    path = tmp_path / "arms.json"
    path.write_text('[{"name":"bank","env":{"FREETOKEN_MTP_BANK":"new"}}]')
    with pytest.raises(ValueError, match="reboot required"):
        runner.load_arms(str(path))
