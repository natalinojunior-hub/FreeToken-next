import os
from types import SimpleNamespace

import pytest

from freetoken.attention import AttnType
from freetoken.engine import engine as eng
from freetoken.env import ENV


def _cfg(fmt, auto, types):
    specs = [SimpleNamespace(attn_type=t) for t in types]
    mc = SimpleNamespace(kv_cache_group_specs=lambda: specs)
    return SimpleNamespace(kv_format=fmt, kv_format_auto=auto, model_config=mc)


def test_full_attention_steps_fp8_turbo4_turbo3(monkeypatch):
    monkeypatch.setattr(
        eng,
        "_required_attn_types",
        lambda mc: frozenset(s.attn_type for s in mc.kv_cache_group_specs()),
    )
    assert eng._kv_fit_ladder(_cfg("fp8", True, [AttnType.FULL])) == (
        "turbo4",
        "turbo3",
    )
    assert eng._kv_fit_ladder(_cfg("turbo3", True, [AttnType.FULL])) == ()
    assert eng._kv_fit_ladder(_cfg("fp8", False, [AttnType.FULL])) == ()  # explicit format


def test_qsa_steps_fp8_turbo4_turbo3(monkeypatch):
    monkeypatch.setattr(
        eng,
        "_required_attn_types",
        lambda mc: frozenset(s.attn_type for s in mc.kv_cache_group_specs()),
    )
    assert eng._kv_fit_ladder(_cfg("fp8", True, [AttnType.QSA])) == ("turbo4", "turbo3")


def test_certified_qsa_keeps_fp8_as_the_ram_tier_floor(monkeypatch):
    monkeypatch.setattr(
        eng,
        "_required_attn_types",
        lambda mc: frozenset(s.attn_type for s in mc.kv_cache_group_specs()),
    )
    config = _cfg("fp8", True, [AttnType.QSA])
    config.model_config.kv_ram_tier_certified = True
    assert eng._kv_fit_ladder(config) == ()


def test_qsa_ram_tiering_requires_certified_family_and_fp8():
    qsa = type("QSAKVCache", (), {})
    device = SimpleNamespace(type="cuda")
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_ram_tier_certified=True),
        kv_format="fp8",
        tp_info=SimpleNamespace(size=1),
    )
    assert eng._kv_ram_tier_unsupported(config, qsa, device) is None
    config.kv_format = "turbo4"
    assert eng._kv_ram_tier_unsupported(config, qsa, device) == "kv_format=turbo4"
    config.kv_format = "fp8"
    config.model_config.kv_ram_tier_certified = False
    assert (
        eng._kv_ram_tier_unsupported(config, qsa, device)
        == "model family not certified for KV in RAM"
    )


def test_certified_qsa_ram_context_needs_only_device_dummy_page():
    qsa = type("QSAKVCache", (), {})
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_ram_tier_certified=True),
        kv_format="fp8",
        page_size=64,
    )
    assert eng._qsa_ram_context_fully_hosted(config, qsa, 256, 16384, True)
    assert not eng._qsa_ram_context_fully_hosted(config, qsa, 255, 16384, True)
    assert not eng._qsa_ram_context_fully_hosted(config, qsa, 256, 16384, False)
    config.kv_format = "turbo4"
    assert not eng._qsa_ram_context_fully_hosted(config, qsa, 256, 16384, True)


def test_forced_mha_fp8_ram_context_can_be_fully_hosted():
    mha = type("MHAKVCache", (), {})
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_ram_tier_certified=False),
        kv_format="fp8",
        kv_tiering="force",
        page_size=64,
    )
    assert eng._qsa_ram_context_fully_hosted(config, mha, 256, 16384, False)


def test_mtp_is_shed_before_the_context_is_refused():
    cfg = SimpleNamespace(spec_mtp=2, max_seq_len=65856)
    assert eng._shed_mtp(cfg) is True and cfg.spec_mtp == 0
    assert eng._shed_mtp(cfg) is False  # nothing left to give: the caller refuses


def test_precision_retry_is_once_per_kv_mtp_candidate(_clean_state_env):
    cfg = _gdn_cfg(4)
    cfg.kv_format = "fp8"
    assert eng._mtp_state_precision_retry_candidate(cfg, True, None) == ("fp8", 4)
    assert eng._mtp_state_precision_retry_candidate(cfg, True, ("fp8", 4)) is None
    cfg.spec_mtp = 0
    assert eng._mtp_state_precision_retry_candidate(cfg, True, ("fp8", 4)) is None


def _gdn_cfg(spec_mtp, has_group=True):
    mc = SimpleNamespace(linear_attention_group=lambda: object() if has_group else None)
    return SimpleNamespace(spec_mtp=spec_mtp, model_config=mc)


@pytest.fixture
def _clean_state_env(monkeypatch):
    """Isolate the process-global env + ENV singleton that _resolve_mtp_state_precision mutates."""
    monkeypatch.delenv("FREETOKEN_MTP_COMPACT_STATE", raising=False)
    monkeypatch.delenv("FREETOKEN_MAMBA_SSM_DTYPE", raising=False)
    saved = ENV.MAMBA_SSM_DTYPE.value
    ENV.MAMBA_SSM_DTYPE.value = "float32"  # the import-default snapshot
    yield
    ENV.MAMBA_SSM_DTYPE.value = saved


def test_resolver_autos_compact_but_keeps_fp32_for_gdn_mtp(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert os.environ["FREETOKEN_MTP_COMPACT_STATE"] == "1"
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"


def test_resolver_noop_without_spec_mtp(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(0))
    assert "FREETOKEN_MTP_COMPACT_STATE" not in os.environ
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"


def test_raw_engine_does_not_inherit_an_automatic_mtp_precision_cut(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"
    eng._resolve_mtp_state_precision(_gdn_cfg(0))
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"


def test_resolver_noop_without_gdn_group(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(4, has_group=False))
    assert "FREETOKEN_MTP_COMPACT_STATE" not in os.environ
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"


def test_resolver_honors_explicit_fp32_override(_clean_state_env, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MAMBA_SSM_DTYPE", "float32")
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert os.environ["FREETOKEN_MTP_COMPACT_STATE"] == "1"  # compact is a free win, still auto
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"  # explicit dtype wins, no bf16 cut


def test_resolver_honors_explicit_bf16_override(_clean_state_env, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MAMBA_SSM_DTYPE", "bfloat16")
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert ENV.MAMBA_SSM_DTYPE.value == "bfloat16"


def test_resolver_honors_explicit_compact_off(_clean_state_env, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_COMPACT_STATE", "0")
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert os.environ["FREETOKEN_MTP_COMPACT_STATE"] == "0"  # setdefault honors the caller's "0"
    assert ENV.MAMBA_SSM_DTYPE.value == "float32"  # pressure retry owns automatic dtype choice
