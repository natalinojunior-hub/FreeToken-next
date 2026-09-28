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


def test_full_attention_steps_fp8_nvfp4_turbo3(monkeypatch):
    monkeypatch.setattr(
        eng,
        "_required_attn_types",
        lambda mc: frozenset(s.attn_type for s in mc.kv_cache_group_specs()),
    )
    assert eng._kv_fit_ladder(_cfg("fp8", True, [AttnType.FULL])) == ("nvfp4", "turbo3")
    assert eng._kv_fit_ladder(_cfg("turbo3", True, [AttnType.FULL])) == ()
    assert eng._kv_fit_ladder(_cfg("fp8", False, [AttnType.FULL])) == ()  # explicit format


def test_qsa_steps_bf16_turbo4_turbo3(monkeypatch):
    monkeypatch.setattr(
        eng,
        "_required_attn_types",
        lambda mc: frozenset(s.attn_type for s in mc.kv_cache_group_specs()),
    )
    assert eng._kv_fit_ladder(_cfg("auto", True, [AttnType.QSA])) == ("turbo4", "turbo3")


def test_mtp_is_shed_before_the_context_is_refused():
    cfg = SimpleNamespace(spec_mtp=2, max_seq_len=65856)
    assert eng._shed_mtp(cfg) is True and cfg.spec_mtp == 0
    assert eng._shed_mtp(cfg) is False  # nothing left to give: the caller refuses


def _gdn_cfg(spec_mtp, has_group=True):
    mc = SimpleNamespace(linear_attention_group=lambda: (object() if has_group else None))
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


def test_resolver_autos_compact_and_bf16_for_gdn_mtp(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert os.environ["FREETOKEN_MTP_COMPACT_STATE"] == "1"
    assert ENV.MAMBA_SSM_DTYPE.value == "bfloat16"


def test_resolver_noop_without_spec_mtp(_clean_state_env):
    eng._resolve_mtp_state_precision(_gdn_cfg(0))
    assert "FREETOKEN_MTP_COMPACT_STATE" not in os.environ
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


def test_resolver_honors_explicit_compact_off(_clean_state_env, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_COMPACT_STATE", "0")
    eng._resolve_mtp_state_precision(_gdn_cfg(4))
    assert os.environ["FREETOKEN_MTP_COMPACT_STATE"] == "0"  # setdefault honors the caller's "0"
    assert ENV.MAMBA_SSM_DTYPE.value == "bfloat16"  # dtype still auto (unset)
