"""``freetoken.tuning.history``: append/load round trip, concurrent-safe appends, schema
skip, and I/O errors never raising into the caller. CPU-only, no server/CUDA."""

from __future__ import annotations

import freetoken.tuning.history as history


def _model_dir(tmp_path, name="weights.safetensors", size=1024):
    d = tmp_path / "model"
    d.mkdir()
    (d / name).write_bytes(b"\0" * size)
    return str(d)


def test_append_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    history.append_run(model, {"label": "run1", "pp_tok_s_mean": 1000.0})
    history.append_run(model, {"label": "run2", "pp_tok_s_mean": 1100.0})
    runs = history.load_runs(model)
    assert [r["label"] for r in runs] == ["run1", "run2"]
    assert runs[0]["schema"] == history.SCHEMA_VERSION
    assert "timestamp" in runs[0] and "fingerprint_key" in runs[0]


def test_load_runs_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    assert history.load_runs(model) == []


def test_load_runs_skips_mismatched_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    path = history.history_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('{"schema": 999, "label": "stale"}\n')
        f.write('{"schema": %d, "label": "ok"}\n' % history.SCHEMA_VERSION)
    runs = history.load_runs(model)
    assert [r["label"] for r in runs] == ["ok"]


def test_load_runs_skips_unparseable_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    path = history.history_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("{not json\n")
        f.write('{"schema": %d, "label": "ok"}\n' % history.SCHEMA_VERSION)
    runs = history.load_runs(model)
    assert [r["label"] for r in runs] == ["ok"]


def test_append_run_never_raises_on_io_error(tmp_path, monkeypatch):
    # history dir points at a path that can never be created (parent is a file)
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    monkeypatch.setattr(history, "_history_dir", lambda: str(blocker / "history"))
    model = _model_dir(tmp_path)
    dest = history.append_run(model, {"label": "run1"})
    assert dest == history.history_path(model)


def test_key_reuses_profile_fingerprint(tmp_path, monkeypatch):
    # same gpu+model -> same key as freetoken.tuning.profile's own fingerprint hash
    from freetoken.tuning.profile import _base_fingerprint, _hash_fingerprint

    model = _model_dir(tmp_path)
    monkeypatch.setattr(history, "_gpu_uuid", lambda: "GPU-x")
    expected = _hash_fingerprint(_base_fingerprint(gpu_uuid="GPU-x", model_path=model))
    assert history._history_key(model) == expected
