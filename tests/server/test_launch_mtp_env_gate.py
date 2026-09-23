"""``apply_mtp_env_gate``: auto-sets FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 in the
parent before the scheduler subprocess spawns, when --spec-mtp > 0 and the user hasn't
already set the env var (leaves an explicit "0" alone -- the scheduler still raises)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from freetoken.server.launch import apply_mtp_env_gate

_ENV_KEY = "FREETOKEN_DISABLE_OVERLAP_SCHEDULING"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)


def test_unset_env_gets_set_when_mtp_enabled(monkeypatch, caplog):
    args = SimpleNamespace(spec_mtp=1)
    with caplog.at_level(logging.INFO):
        apply_mtp_env_gate(args, logging.getLogger("test"))
    import os

    assert os.environ[_ENV_KEY] == "1"
    assert "spec-mtp" in caplog.text


def test_spec_mtp_zero_leaves_env_untouched(monkeypatch):
    args = SimpleNamespace(spec_mtp=0)
    apply_mtp_env_gate(args, logging.getLogger("test"))
    import os

    assert _ENV_KEY not in os.environ


def test_user_explicit_zero_is_not_overridden(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "0")
    args = SimpleNamespace(spec_mtp=1)
    apply_mtp_env_gate(args, logging.getLogger("test"))
    import os

    assert os.environ[_ENV_KEY] == "0"


def test_user_explicit_one_is_left_alone(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "1")
    args = SimpleNamespace(spec_mtp=2)
    apply_mtp_env_gate(args, logging.getLogger("test"))
    import os

    assert os.environ[_ENV_KEY] == "1"
