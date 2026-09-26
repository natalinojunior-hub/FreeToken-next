from types import SimpleNamespace

from freetoken.attention import AttnType
from freetoken.engine import engine as eng


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
