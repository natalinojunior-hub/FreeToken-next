"""An omitted --spec-mtp enables online depth selection only for native heads."""

from types import SimpleNamespace

import freetoken.tuning.profile as profile_mod
from freetoken.server.args import ServerArgs, _native_mtp_layers, _tuned_spec_mtp, parse_args

KW = {"model_path": "/m", "kv_format": "turbo3", "max_seq_len_override": 16384}


def _stub_hf_config(monkeypatch, layers):
    config = SimpleNamespace(
        to_dict=lambda: {"text_config": {"mtp": {"num_hidden_layers": layers}}}
    )
    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _path: config)


def _parse(monkeypatch, layers, *extra):
    _stub_hf_config(monkeypatch, layers)
    args, _ = parse_args(
        [
            "--model",
            "/models/anon",
            "--dtype",
            "bfloat16",
            *extra,
            "--tool-call-parser",
            "llama3",
            "--reasoning-parser",
            "off",
        ]
    )
    return args


def test_native_head_enables_k4_online_depth_cap_without_profile(monkeypatch):
    _stub_hf_config(monkeypatch, 1)
    assert _tuned_spec_mtp({**KW, "max_running_req": 1}) == 4


def test_old_k0_profile_does_not_disable_online_exploration(monkeypatch):
    _stub_hf_config(monkeypatch, 1)
    monkeypatch.setattr(
        profile_mod,
        "load_for",
        lambda *_: SimpleNamespace(chosen=SimpleNamespace(spec_mtp=0)),
    )
    assert _tuned_spec_mtp({**KW, "max_running_req": 1}) == 4


def test_checkpoint_without_native_head_keeps_default(monkeypatch):
    _stub_hf_config(monkeypatch, 0)
    assert _tuned_spec_mtp({**KW, "max_running_req": 1}) == ServerArgs.spec_mtp


def test_gguf_native_head_uses_existing_metadata_discovery(monkeypatch):
    config = SimpleNamespace(metadata={})
    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _path: config)
    monkeypatch.setattr("freetoken.server.args._native_nextn_layers", lambda _path: 1)
    assert _native_mtp_layers("/m") == 1


def test_gguf_discovered_companion_head_enables_mtp(monkeypatch):
    config = SimpleNamespace(metadata={}, model_type="qwen4exp")
    mtp_config = SimpleNamespace(num_hidden_layers=1)
    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _path: config)
    monkeypatch.setattr("freetoken.server.args._native_nextn_layers", lambda _path: 0)
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.gguf._parse_mtp_config_from_gguf",
        lambda _path: mtp_config,
    )
    assert _native_mtp_layers("/m") == 1


def test_multi_request_server_never_gets_mtp(monkeypatch):
    _stub_hf_config(monkeypatch, 1)
    assert _tuned_spec_mtp({**KW, "max_running_req": 8}) == ServerArgs.spec_mtp


def test_parser_omitted_flag_enables_native_head_for_single_request(monkeypatch):
    args = _parse(monkeypatch, 1)
    assert args.max_running_req == 1
    assert args.spec_mtp == 4


def test_parser_defaults_to_four_requests_without_native_head(monkeypatch):
    args = _parse(monkeypatch, 0)
    assert args.max_running_req == ServerArgs.max_running_req == 4
    assert args.spec_mtp == ServerArgs.spec_mtp == 0


def test_parser_preserves_explicit_multi_request_with_native_head(monkeypatch):
    args = _parse(monkeypatch, 1, "--max-running-requests", "8")
    assert args.max_running_req == 8
    assert args.spec_mtp == 0


def test_parser_explicit_zero_remains_disabled(monkeypatch):
    args = _parse(monkeypatch, 1, "--spec-mtp", "0")
    assert args.max_running_req == ServerArgs.max_running_req
    assert args.spec_mtp == 0
