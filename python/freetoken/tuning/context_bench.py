"""``ft bench context``: learn how much context a model fits on this machine, and
recommend a trade-off (best TG, best cost-benefit, largest feasible).

Reuses ``freetoken.tuning.tune_cli``'s boot/measure primitives (``_free_port``,
``_wait_for_health``, ``_model_id``, ``_measure_one_rep``, ``_unique_prompt``) for each
context point instead of writing a second server harness, and
``freetoken.tuning.history`` (``append_run``/``load_runs``, label ``"context-bench"``) so
repeated runs against the same model reuse prior measurements.

The recommendation logic (``recommend``) is pure and separate from the I/O
(``run_context_bench``) so it is unit-testable on synthetic points with no server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from freetoken.tuning.history import append_run, load_runs
from freetoken.tuning.tune_cli import (
    _free_port,
    _measure_one_rep,
    _model_id,
    _unique_prompt,
    _wait_for_health,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)

LABEL = "context-bench"
DEFAULT_CONTEXTS = (16384, 32768, 65536, 131072, 262144)
# The engine refuses an over-large context with this exact Portuguese message
# (see engine.py's PageTable init); Y is already rounded down to a multiple of 1024.
_REFUSAL_RE = re.compile(r"m\xe1ximo poss\xedvel \xe9 (\d+) tokens")


@dataclass
class ContextPoint:
    context: int
    pp: float
    tg: float
    kv_ram: bool
    moe_strategy: str


@dataclass
class Recommendation:
    best_tg: ContextPoint
    best_value: ContextPoint
    maximum: ContextPoint


def parse_refusal(log_text: str) -> int | None:
    """The maximum feasible context (Y, tokens) from the engine's refusal message, or
    ``None`` if the log doesn't contain one."""
    m = _REFUSAL_RE.search(log_text)
    return int(m.group(1)) if m else None


def _label(context: int) -> str:
    return f"{context // 1024}K"


def recommend(points: list[ContextPoint]) -> Recommendation:
    if not points:
        raise ValueError("recommend: no measured points")
    best_tg = max(points, key=lambda p: (p.tg, p.context))
    floor = best_tg.tg * 0.9
    value_candidates = [p for p in points if p.tg >= floor]
    best_value = max(value_candidates, key=lambda p: p.context)
    maximum = max(points, key=lambda p: p.context)
    return Recommendation(best_tg=best_tg, best_value=best_value, maximum=maximum)


def format_table(points: list[ContextPoint], best_tg: ContextPoint) -> str:
    header = "contexto\tPP\tTG\tΔTG%\tKV na RAM\texperts"
    rows = [header]
    for p in sorted(points, key=lambda p: p.context):
        dtg = (p.tg - best_tg.tg) / best_tg.tg * 100 if best_tg.tg else 0.0
        rows.append(
            f"{_label(p.context)}\t{p.pp:.1f}\t{p.tg:.1f}\t{dtg:+.1f}%\t"
            f"{'sim' if p.kv_ram else 'não'}\t{p.moe_strategy}"
        )
    return "\n".join(rows)


def format_recommendation(rec: Recommendation) -> list[str]:
    lines = [f"{_label(rec.best_tg.context)}: melhor TG"]
    if rec.best_value.context == rec.maximum.context:
        lines.append(f"{_label(rec.best_value.context)}: melhor custo-benefício (igual ao máximo)")
    else:
        lines.append(f"{_label(rec.best_value.context)}: melhor custo-benefício")
    dtg = (rec.maximum.tg - rec.best_tg.tg) / rec.best_tg.tg * 100 if rec.best_tg.tg else 0.0
    dpp = (rec.maximum.pp - rec.best_tg.pp) / rec.best_tg.pp * 100 if rec.best_tg.pp else 0.0
    lines.append(
        f"{_label(rec.maximum.context)}: máximo (TG {dtg:+.0f}% e PP {dpp:+.0f}% vs o melhor)"
    )
    return lines


# ============================== I/O: boot + measure ==============================


def _prompt_for(context: int) -> str:
    # ~10 tokens/sentence, ~85% of the context -- same shape as ft tune's default prompt.
    return "The quick brown fox jumps over the lazy dog. " * max(1, int(context * 0.85) // 10)


def measure_context_point(
    model: str,
    context: int,
    *,
    kv_format: str = "turbo3",
    moe_strategy: str = "offload",
    reps: int = 3,
    decode_tokens: int = 256,
    prompt_text: str | None = None,
    log_dir: str | None = None,
) -> dict:
    """Boot one real ``ft serve`` at ``--max-seq-len-override context --kv-tiering auto``,
    measure cold PP / committed TG (reusing ft tune's measurement helpers), then kill it.
    Returns ``{"ok": True, "pp", "tg", "kv_ram", ...}`` or, on refusal/boot failure,
    ``{"ok": False, "refusal_max_tokens": Y | None}``."""
    port = _free_port()
    argv = [
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len-override",
        str(context),
        "--kv-format",
        kv_format,
        "--kv-tiering",
        "auto",
        "--moe-strategy",
        moe_strategy,
        "--spec-mtp",
        "0",
        "--max-running-requests",
        "1",
    ]
    log_dir = log_dir or os.environ.get("TMPDIR") or (
        tempfile.gettempdir() if os.name == "nt" else "/tmp"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"ft-bench-context-{context}-{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "freetoken.cli", "serve", *argv],
        start_new_session=True,  # own process group -> kill only this boot's tree
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    text = prompt_text if prompt_text is not None else _prompt_for(context)
    pp = tg = None
    try:
        url = f"http://127.0.0.1:{port}"
        if not _wait_for_health(proc, url):
            log_file.flush()
            with open(log_path) as f:
                log_text = f.read()
            return {"ok": False, "refusal_max_tokens": parse_refusal(log_text)}
        model_id = _model_id(url) or "default"
        samples = []
        for i in range(1 + reps):  # 1 warmup + reps
            r = _measure_one_rep(url, model_id, _unique_prompt(text, i), decode_tokens)
            if r is None:
                break
            if i > 0:
                samples.append(r)
        if not samples:
            return {"ok": False, "refusal_max_tokens": None}
        samples.sort(key=lambda r: r["committed_tg"])
        tg = samples[len(samples) // 2]["committed_tg"]
        pp = sorted(s["cold_pp"] for s in samples)[len(samples) // 2]
    finally:
        if proc.poll() is None:
            if hasattr(os, "killpg"):  # POSIX: signal the whole session created at spawn
                os.killpg(proc.pid, signal.SIGINT)
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                else:
                    proc.kill()  # Windows: no killpg/SIGKILL; TerminateProcess the direct child
        proc.wait(timeout=10)
        log_file.close()

    with open(log_path) as f:
        kv_ram = "KV RAM tiering:" in f.read()
    return {"ok": True, "pp": pp, "tg": tg, "kv_ram": kv_ram}


def _cached_point(model: str, context: int, moe_strategy: str) -> ContextPoint | None:
    match = None
    for rec in load_runs(model):
        if (
            rec.get("label") == LABEL
            and rec.get("context") == context
            and rec.get("moe_strategy") == moe_strategy
        ):
            match = rec  # keep the last (most recent) match
    if match is None:
        return None
    return ContextPoint(
        context=context,
        pp=match["pp"],
        tg=match["tg"],
        kv_ram=match["kv_ram"],
        moe_strategy=moe_strategy,
    )


def _save_point(model: str, point: ContextPoint) -> None:
    append_run(
        model,
        {
            "label": LABEL,
            "context": point.context,
            "pp": point.pp,
            "tg": point.tg,
            "kv_ram": point.kv_ram,
            "moe_strategy": point.moe_strategy,
        },
    )


def run_context_bench(
    model: str,
    *,
    contexts: list[int] | None = None,
    kv_format: str = "turbo3",
    moe_strategy: str = "offload",
    reps: int = 3,
    decode_tokens: int = 256,
    log_dir: str | None = None,
    refresh: bool = False,
) -> list[ContextPoint]:
    """Measure ascending --contexts, reusing history unless ``refresh``. On a refusal,
    parse the max feasible context Y and, if Y >= 16K and not already a measured point,
    measure it (rounded down to 1024) as the final point, then stop."""
    ordered = sorted(set(contexts) if contexts else DEFAULT_CONTEXTS)
    points: list[ContextPoint] = []
    measured: set[int] = set()
    for ctx in ordered:
        cached = None if refresh else _cached_point(model, ctx, moe_strategy)
        if cached is not None:
            points.append(cached)
            measured.add(ctx)
            continue
        result = measure_context_point(
            model,
            ctx,
            kv_format=kv_format,
            moe_strategy=moe_strategy,
            reps=reps,
            decode_tokens=decode_tokens,
            log_dir=log_dir,
        )
        if not result["ok"]:
            y = result.get("refusal_max_tokens")
            if y is not None and y >= 16384:
                y_final = (y // 1024) * 1024
                if y_final >= 16384 and y_final not in measured:
                    final = measure_context_point(
                        model,
                        y_final,
                        kv_format=kv_format,
                        moe_strategy=moe_strategy,
                        reps=reps,
                        decode_tokens=decode_tokens,
                        log_dir=log_dir,
                    )
                    if final["ok"]:
                        cp = ContextPoint(
                            context=y_final,
                            pp=final["pp"],
                            tg=final["tg"],
                            kv_ram=final["kv_ram"],
                            moe_strategy=moe_strategy,
                        )
                        points.append(cp)
                        _save_point(model, cp)
            break
        cp = ContextPoint(
            context=ctx,
            pp=result["pp"],
            tg=result["tg"],
            kv_ram=result["kv_ram"],
            moe_strategy=moe_strategy,
        )
        points.append(cp)
        measured.add(ctx)
        _save_point(model, cp)
    return points


def main(argv: list[str] | None = None, prog: str = "ft bench context") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("model", help="caminho do checkpoint (.gguf ou pasta NVFP4/safetensors)")
    p.add_argument(
        "--contexts",
        default=None,
        help="lista ascendente de contextos a medir, separada por vírgula (padrão: "
        + ",".join(str(c) for c in DEFAULT_CONTEXTS),
    )
    p.add_argument("--kv-format", default="turbo3")
    p.add_argument("--moe-strategy", default="offload")
    p.add_argument("--reps", type=int, default=3, help="repetições medidas por ponto (+1 warmup)")
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--log-dir", default=None)
    p.add_argument(
        "--refresh", action="store_true", help="remedir mesmo se já houver histórico salvo"
    )
    p.add_argument("--json", dest="json_path", default=None, help="também gravar resultado em JSON")
    ns = p.parse_args(argv)

    contexts = [int(c) for c in ns.contexts.split(",")] if ns.contexts else list(DEFAULT_CONTEXTS)
    points = run_context_bench(
        ns.model,
        contexts=contexts,
        kv_format=ns.kv_format,
        moe_strategy=ns.moe_strategy,
        reps=ns.reps,
        decode_tokens=ns.decode_tokens,
        log_dir=ns.log_dir,
        refresh=ns.refresh,
    )
    if not points:
        print("ft bench context: nenhum contexto coube nesse hardware")
        return 1

    rec = recommend(points)
    print(format_table(points, rec.best_tg))
    for line in format_recommendation(rec):
        print(line)

    if ns.json_path:
        payload = {
            "model": ns.model,
            "points": [p.__dict__ for p in points],
            "recommendation": {
                "best_tg": rec.best_tg.context,
                "best_value": rec.best_value.context,
                "maximum": rec.maximum.context,
            },
        }
        with open(ns.json_path, "w") as f:
            json.dump(payload, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
