"""--spec-mtp left unset takes the stored `ft tune` choice; explicit flags and multi-request
servers keep their own value."""

from types import SimpleNamespace

import freetoken.tuning.profile as profile_mod
from freetoken.server.args import ServerArgs, _tuned_spec_mtp

KW = {"model_path": "/m", "kv_format": "turbo3", "max_seq_len_override": 16384}


def _stub(monkeypatch, spec_mtp):
    found = None if spec_mtp is None else SimpleNamespace(chosen=SimpleNamespace(spec_mtp=spec_mtp))
    monkeypatch.setattr(profile_mod, "load_for", lambda *a: found)


def test_profile_choice_applies_to_single_request_server(monkeypatch):
    _stub(monkeypatch, 1)
    assert _tuned_spec_mtp({**KW, "max_running_req": 1}) == 1


def test_no_profile_keeps_default(monkeypatch):
    _stub(monkeypatch, None)
    assert _tuned_spec_mtp({**KW, "max_running_req": 1}) == ServerArgs.spec_mtp


def test_multi_request_server_never_gets_mtp(monkeypatch):
    _stub(monkeypatch, 1)
    assert _tuned_spec_mtp({**KW, "max_running_req": 8}) == ServerArgs.spec_mtp
