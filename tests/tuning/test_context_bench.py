from __future__ import annotations

from freetoken.tuning import context_bench as cb
from freetoken.tuning.context_bench import (
    ContextPoint,
    format_recommendation,
    parse_refusal,
    recommend,
)


def _pt(
    context: int, pp: float, tg: float, kv_ram: bool = False, moe: str = "offload"
) -> ContextPoint:
    return ContextPoint(context=context, pp=pp, tg=tg, kv_ram=kv_ram, moe_strategy=moe)


def test_recommend_normal_curve():
    # TG falls off with context; 128K is within 10% of the 64K peak, 256K is not.
    points = [
        _pt(16384, 4000, 120),
        _pt(65536, 3200, 130),
        _pt(131072, 2500, 125),
        _pt(262144, 1800, 90),
    ]
    rec = recommend(points)
    assert rec.best_tg.context == 65536
    assert rec.best_value.context == 131072  # 125 >= 0.9*130, largest such context
    assert rec.maximum.context == 262144

    lines = format_recommendation(rec)
    assert lines[0] == "64K: melhor TG"
    assert lines[1] == "128K: melhor custo-benefício"
    assert lines[2].startswith("256K: máximo (TG ")
    assert "%" in lines[2] and "PP" in lines[2]


def test_recommend_flat_tg():
    points = [_pt(16384, 4000, 100), _pt(65536, 3500, 100), _pt(131072, 3000, 100)]
    rec = recommend(points)
    # ties -> larger context wins for best_tg
    assert rec.best_tg.context == 131072
    assert rec.best_value.context == 131072
    assert rec.maximum.context == 131072
    lines = format_recommendation(rec)
    assert "igual ao máximo" in lines[1]


def test_recommend_single_point():
    points = [_pt(16384, 4000, 100)]
    rec = recommend(points)
    assert rec.best_tg.context == rec.best_value.context == rec.maximum.context == 16384
    lines = format_recommendation(rec)
    assert lines[2].startswith("16K: máximo (TG +0% e PP +0% vs o melhor)")


def test_parse_refusal_extracts_y():
    log = (
        "some boot noise\n"
        "RuntimeError: o contexto pedido de 300000 tokens não é possível nesse hardware, "
        "o máximo possível é 271360 tokens\n"
    )
    assert parse_refusal(log) == 271360


def test_parse_refusal_none_when_absent():
    assert parse_refusal("server started fine\n") is None


def test_run_context_bench_reuses_history(tmp_path, monkeypatch):
    monkeypatch.setattr("freetoken.tuning.history._cache_dir", lambda: str(tmp_path))

    calls = []

    def fake_measure(model, context, **kwargs):
        calls.append(context)
        return {"ok": True, "pp": 3000.0, "tg": 100.0, "kv_ram": False}

    monkeypatch.setattr(cb, "measure_context_point", fake_measure)

    points1 = cb.run_context_bench("model.gguf", contexts=[16384, 32768])
    assert len(calls) == 2
    assert [p.context for p in points1] == [16384, 32768]

    # Second run against the same model+contexts must hit history, not measure again.
    points2 = cb.run_context_bench("model.gguf", contexts=[16384, 32768])
    assert len(calls) == 2  # unchanged: no new measurements
    assert [p.context for p in points2] == [16384, 32768]

    # --refresh forces re-measurement.
    cb.run_context_bench("model.gguf", contexts=[16384], refresh=True)
    assert len(calls) == 3


def test_run_context_bench_stops_on_refusal_and_measures_y(tmp_path, monkeypatch):
    monkeypatch.setattr("freetoken.tuning.history._cache_dir", lambda: str(tmp_path))

    def fake_measure(model, context, **kwargs):
        if context == 16384:
            return {"ok": True, "pp": 3000.0, "tg": 100.0, "kv_ram": False}
        if context == 32768:
            return {"ok": False, "refusal_max_tokens": 20480}
        if context == 20480:
            return {"ok": True, "pp": 2000.0, "tg": 80.0, "kv_ram": True}
        raise AssertionError(f"unexpected context {context}")

    monkeypatch.setattr(cb, "measure_context_point", fake_measure)

    points = cb.run_context_bench("model.gguf", contexts=[16384, 32768, 65536])
    assert [p.context for p in points] == [16384, 20480]
    assert points[-1].kv_ram is True


def test_cli_parses_positional_model_and_contexts(tmp_path, monkeypatch):
    monkeypatch.setattr("freetoken.tuning.history._cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        cb,
        "measure_context_point",
        lambda model, context, **kwargs: {"ok": True, "pp": 1000.0, "tg": 50.0, "kv_ram": False},
    )
    rc = cb.main(["model.gguf", "--contexts", "16384,32768"])
    assert rc == 0
