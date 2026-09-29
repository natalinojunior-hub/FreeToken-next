"""Tests for ``freetoken.tuning.diagnostics``: configuration, structured JSONL event logging,
unconfigured/disabled no-op guarantees, and best-effort exception resilience. CPU-only."""

from __future__ import annotations

import json
import os

import pytest

from freetoken.tuning import diagnostics


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Pin the cache dir, GPU uuid and reset attribution globals between tests."""
    monkeypatch.setattr(diagnostics, "_cache_dir", lambda: str(tmp_path / "cache"))
    monkeypatch.setattr(diagnostics, "_gpu_uuid", lambda: "GPU-test")
    monkeypatch.delenv("FREETOKEN_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(diagnostics, "_key", None)
    monkeypatch.setattr(diagnostics, "_model_path", None)
    monkeypatch.setattr(diagnostics, "_config_summary", {})
    model = tmp_path / "model"
    model.mkdir()
    (model / "w.gguf").write_bytes(b"\0" * 2048)
    return str(model)


def test_configure_returns_path_and_sets_key(isolated):
    path = diagnostics.configure(isolated, {"spec_mtp": 4, "kv_format": "auto"})
    assert path is not None
    assert diagnostics._key is not None
    assert path.endswith(f"{diagnostics._key}.jsonl")
    assert os.path.exists(os.path.dirname(path))
    assert diagnostics.diagnostics_path(isolated) == path


def test_log_event_appends_valid_json_line(isolated):
    path = diagnostics.configure(isolated, {"spec_mtp": 2})
    diagnostics.log_event(
        "degrade",
        "shed_mtp",
        "--spec-mtp 2 -> 0: tokens need VRAM",
        severity="warn",
        spec_mtp_before=2,
        max_seq_len=8192,
    )
    assert path is not None
    assert os.path.exists(path)
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "degrade"
    assert record["phase"] == "shed_mtp"
    assert record["severity"] == "warn"
    assert record["message"] == "--spec-mtp 2 -> 0: tokens need VRAM"
    assert record["config"]["spec_mtp"] == 2
    assert record["context"]["spec_mtp_before"] == 2
    assert record["context"]["max_seq_len"] == 8192
    assert "ts" in record


def test_log_event_before_configure_is_safe_noop(isolated):
    diagnostics.log_event("warn", "phase1", "unconfigured event", severity="warn", detail=123)
    p = diagnostics.diagnostics_path(isolated)
    if p is not None:
        assert not os.path.exists(p)


def test_disabled_env_prevents_persistence(isolated, monkeypatch):
    monkeypatch.setenv("FREETOKEN_DIAGNOSTICS", "0")
    path = diagnostics.configure(isolated, {"spec_mtp": 1})
    assert path is None
    assert diagnostics._key is None
    diagnostics.log_event("degrade", "kv_format", "KV format -> fp8", severity="warn")
    cache_dir = diagnostics.diagnostics_dir()
    assert not os.path.exists(cache_dir) or not os.listdir(cache_dir)


def test_log_event_never_raises_on_unwritable_dir(isolated, monkeypatch):
    diagnostics.configure(isolated, {})

    def fail_makedirs(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "makedirs", fail_makedirs)
    # Must swallow exception silently and never raise
    diagnostics.log_event(
        "warn", "vram_ledger", "ledger under-modelled", severity="warn", unexplained_bytes=1024
    )
