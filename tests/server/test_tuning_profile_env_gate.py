"""``apply_tuning_profile_env_gate``: setdefaults the env-backed v1 tunables from a
persisted ``ft tune`` profile in the parent, before the scheduler spawns. Fakes NVML and
the profile loader -- no CUDA, no server, no real GPU."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from freetoken.server.launch import apply_tuning_profile_env_gate
from freetoken.tuning.profile import CandidateEvidence, Profile, TunedSettings

_KEYS = ("FREETOKEN_SPEC_DEFER_REPLAY", "FREETOKEN_DRAFT_GRAPH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def _args(**over):
    base = dict(model_path="/tmp/model", kv_format="turbo3", max_seq_len_override=16384)
    base.update(over)
    return SimpleNamespace(**base)


def _profile(defer_replay=True, draft_graph=True):
    return Profile(
        schema=1,
        key="anykey",
        chosen=TunedSettings(
            spec_mtp=1, defer_replay=defer_replay, draft_graph=draft_graph, moe_strategy="hybrid"
        ),
        evidence=CandidateEvidence(1000.0, 45.0, 0.2, 12000.0, "2026-09-23"),
    )


def test_no_max_seq_len_override_skips_profile_lookup(monkeypatch):
    called = []
    monkeypatch.setattr("freetoken.gpu_select._nvml_uuids", lambda: called.append(1) or ["u"])
    apply_tuning_profile_env_gate(_args(max_seq_len_override=None), logging.getLogger("t"))
    assert not called
    import os

    assert not any(k in os.environ for k in _KEYS)


def test_no_nvml_skips_profile_lookup(monkeypatch):
    monkeypatch.setattr("freetoken.gpu_select._nvml_uuids", lambda: None)
    apply_tuning_profile_env_gate(_args(), logging.getLogger("t"))
    import os

    assert not any(k in os.environ for k in _KEYS)


def test_no_profile_file_leaves_env_untouched(monkeypatch):
    monkeypatch.setattr("freetoken.gpu_select._nvml_uuids", lambda: ["GPU-fake"])
    monkeypatch.setattr("freetoken.tuning.profile.load", lambda key, path=None: None)
    apply_tuning_profile_env_gate(_args(), logging.getLogger("t"))
    import os

    assert not any(k in os.environ for k in _KEYS)


def test_profile_fills_unset_env_vars(monkeypatch, caplog):
    monkeypatch.setattr("freetoken.gpu_select._nvml_uuids", lambda: ["GPU-fake"])
    monkeypatch.setattr(
        "freetoken.tuning.profile.load", lambda key, path=None: _profile(True, False)
    )
    with caplog.at_level(logging.INFO):
        apply_tuning_profile_env_gate(_args(), logging.getLogger("t"))
    import os

    assert os.environ["FREETOKEN_SPEC_DEFER_REPLAY"] == "1"
    assert os.environ["FREETOKEN_DRAFT_GRAPH"] == "0"
    assert "ft tune profile" in caplog.text


def test_profile_never_overrides_an_already_set_env_var(monkeypatch):
    import os

    os.environ["FREETOKEN_DRAFT_GRAPH"] = "1"  # user explicitly set it
    monkeypatch.setattr("freetoken.gpu_select._nvml_uuids", lambda: ["GPU-fake"])
    monkeypatch.setattr(
        "freetoken.tuning.profile.load", lambda key, path=None: _profile(True, False)
    )
    apply_tuning_profile_env_gate(_args(), logging.getLogger("t"))
    assert os.environ["FREETOKEN_DRAFT_GRAPH"] == "1"  # untouched
    assert os.environ["FREETOKEN_SPEC_DEFER_REPLAY"] == "1"  # still filled
