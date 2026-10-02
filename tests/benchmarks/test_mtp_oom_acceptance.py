"""Fault injection must follow real writes and fire once."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("stage", ["verify", "row_commit"])
def test_fault_follows_write_fires_once_and_restores_method(stage, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/mtp_oom_acceptance.py"
    spec = importlib.util.spec_from_file_location("mtp_oom_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("sync"))

    def write(*_args):
        events.append("write")
        return "result"

    engine = SimpleNamespace(
        device="cpu",
        forward_batch=write,
        linear_state_pool=SimpleNamespace(commit_spec_row=write),
    )
    event, restore = module.install_fault(engine, stage)
    owner = engine if stage == "verify" else engine.linear_state_pool
    name = "forward_batch" if stage == "verify" else "commit_spec_row"
    args = (
        (SimpleNamespace(spec_logits_indices=torch.tensor([0])),) if stage == "verify" else (1, 0)
    )
    if stage == "verify":
        assert owner.forward_batch(SimpleNamespace(spec_logits_indices=None)) == "result"
        assert event["injections"] == 0
        events.clear()
    with pytest.raises(torch.OutOfMemoryError):
        getattr(owner, name)(*args)
    assert events == ["write", "sync"]
    assert event["injections"] == 1
    assert getattr(owner, name)(*args) == "result"
    assert event["injections"] == 1
    restore()
    assert getattr(owner, name) is write
