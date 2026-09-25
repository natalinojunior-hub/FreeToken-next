"""``ft history <model>``: prints the persisted run-history log, newest first."""

from __future__ import annotations

import freetoken.tuning.history as history
from freetoken.tuning.history_cli import main as history_main


def _model_dir(tmp_path, name="weights.safetensors", size=1024):
    d = tmp_path / "model"
    d.mkdir()
    (d / name).write_bytes(b"\0" * size)
    return str(d)


def test_prints_no_history_message_when_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    assert history_main([model]) == 0
    assert "sem histórico" in capsys.readouterr().out


def test_prints_newest_first_with_portuguese_headers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(history, "_history_dir", lambda: str(tmp_path))
    model = _model_dir(tmp_path)
    history.append_run(
        model,
        {
            "label": "run1",
            "tokens": 4096,
            "pp_tok_s_mean": 1000.0,
            "tg_tok_s_mean": 40.0,
            "expert_slots": 32,
            "kv_device_pages": 400,
            "kv_ram_pages": 112,
            "output_sha1": "abc123",
        },
    )
    history.append_run(
        model,
        {
            "label": "run2",
            "tokens": 8192,
            "pp_tok_s_mean": 900.0,
            "tg_tok_s_mean": 38.0,
            "expert_slots": 32,
            "kv_device_pages": 800,
            "kv_ram_pages": None,
            "output_sha1": "def456",
        },
    )
    assert history_main([model]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "rótulo" in lines[0] and "contexto" in lines[0] and "hash" in lines[0]
    # newest (run2) first
    assert "run2" in lines[1]
    assert "run1" in lines[2]
    assert "800/-" in lines[1]
    assert "400/112" in lines[2]
