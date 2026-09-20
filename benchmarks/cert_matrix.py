"""Certification matrix: no-regression gate plus native-vs-GGUF parity, as one command.

Phase 1 of the mission fixes the guards (the NVFP4 checkpoints must never get slower);
phase 2 onward adds GGUF, KV formats, the ledger and MTP. This is the gate that says a
build is still shippable: every native row must clear its guard, and every GGUF row must
be compared against the same-architecture native row so a GGUF regression cannot hide as
"GGUF has no baseline".

Rows are declared here and nowhere else, so a review sees the whole contract at once.

    python benchmarks/cert_matrix.py --dry-run          # print the plan, touch no GPU
    python benchmarks/cert_matrix.py --contexts 16384    # the 16K guard set
    python benchmarks/cert_matrix.py --only gguf        # one family
    python benchmarks/cert_matrix.py --contexts 4096,16384,32768 --json /tmp/cert.jsonl

Each row runs `bench_pp_tg.py`, which spawns its own server, so rows are independent and
one row's failure does not stop the others (it is reported, never skipped). A row whose
checkpoint cannot be served by the current code reports BLOCKED with the reason instead of
disappearing from the table -- a silently absent row is how a regression escapes a gate.

Two honest limits of the parity columns:
* the GGUF files here are not the same weights as their native counterparts (Ornith and
  Tiel-Coder are fine-tunes; the Unsloth/AD builds are different quant recipes), so the
  comparison is throughput, capacity and per-token cost under equal context -- not
  token-for-token output identity;
* expert-bank GGUF rows depend on a single ggml type per bank, which mixed-quant builds
  violate, until the exact-geometry expert cache lands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "bench_pp_tg.py"

# Guards from docs/freetoken-next/PERFORMANCE.md section 3, measured on this host on
# freetoken-next @ cac247a. Keep the two files in step.
GUARDS = {
    "native-35b-a3b": {"pp": 4600.0, "tg": 158.0},
    "native-flash-next": {"pp": 1850.0, "tg": 28.5},
}

# serve args shared by every row so an A/B changes exactly one variable
# (docs/freetoken-next/EXPERIMENTS.md EXP-001: PP needs --cache-type naive, and KV must be
# sized explicitly because --moe-cache-auto leaves only 8268 tokens on a 16 GiB card).
COMMON = ["--num-tokens", "{ctx_plus}", "--cache-type", "naive"]

ROWS = [
    {
        "id": "native-35b-a3b",
        "family": "native",
        "pair": "qwen35moe",
        "model": "/models/Qwen3.6-35B-A3B-NVFP4-FT",
        "mem_ratio": 0.9,
        "note": "NVFP4 + FP8 KV from the checkpoint; the mission's fast proving ground",
    },
    {
        "id": "gguf-ornith-apex",
        "family": "gguf",
        "pair": "qwen35moe",
        "model": "/models/Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf",
        "mem_ratio": 0.9,
        "blocked_by": "mixed expert ggml types across layers (Q3_K x30 + Q4_K x10) against "
        "the single-stride expert slot pool; needs the exact-geometry pool",
    },
    {
        "id": "gguf-tiel-coder",
        "family": "gguf",
        "pair": "qwen35moe",
        "model": "/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf",
        "mem_ratio": 0.9,
        "blocked_by": "unsloth-dynamic build mixes expert types per layer (same gate)",
    },
    {
        "id": "native-flash-next",
        "family": "native",
        "pair": "qwen4exp",
        "model": "/models/Qwen3.8-Flash-Next-NVFP4-Radix",
        "mem_ratio": 0.86,
        "note": "0.9 CUDA-OOMs in unbudgeted transients (Triton autotune, graph capture)",
    },
    {
        "id": "gguf-flash-unsloth-ud",
        "family": "gguf",
        "pair": "qwen4exp",
        "model": "/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS",
        "mem_ratio": 0.86,
        "note": "GGUF UD-IQ4_XS sharded with PLE UVA/mmap and native GGUF operators",
    },
    {
        "id": "gguf-flash-ad-q4km",
        "family": "gguf",
        "pair": "qwen4exp",
        "model": "/models/Qwen3.8-Flash-Next-AD-4.27/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64",
        "mem_ratio": 0.86,
        "blocked_by": "33-shard split; same qwen4exp adapter and PLE-table gaps",
    },
    {
        "id": "gguf-qwen38-27b-iq3s",
        "family": "gguf",
        "pair": "qwen35-dense",
        "model": "/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf",
        "mem_ratio": 0.86,
        "note": "dense qwen35: no expert banks, so the single-stride rule does not apply; "
        "I-quants have no MMQ case, so prefill dequantizes -- measure it, do not hide it",
        "parity_against": None,
    },
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--contexts", default="16384", help="comma list of prompt token counts")
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--only", choices=["native", "gguf"], help="run one family only")
    p.add_argument("--json", dest="json_out", default=None, help="append every row here")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def command(row, ctx, args):
    serve = list(COMMON)
    serve = [str(ctx + args.decode + 64) if a == "{ctx_plus}" else a for a in serve]
    return [
        sys.executable,
        str(BENCH),
        "--model",
        row["model"],
        "--tokens",
        str(ctx),
        "--decode",
        str(args.decode),
        "--repeats",
        str(args.repeats),
        "--mem-ratio",
        str(row["mem_ratio"]),
        "--label",
        f"{row['id']}@{ctx}",
        "--serve-arg",
        " ".join(serve),
    ]


def read_last_row(jsonl: Path) -> dict:
    lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def main(argv=None) -> int:
    args = parse_args(argv)
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    rows = [r for r in ROWS if not args.only or r["family"] == args.only]

    print(
        f"certification matrix: {len(rows)} rows x {len(contexts)} contexts "
        f"({', '.join(str(c) for c in contexts)} tokens)"
    )

    if args.dry_run:
        for row in rows:
            for ctx in contexts:
                why = row.get("blocked_by")
                print(f"\n[{row['id']} @{ctx}] {'BLOCKED: ' + why if why else 'run'}")
                print("  " + " ".join(command(row, ctx, args)))
        print(
            f"\ndry run only; {sum(1 for r in rows if r.get('blocked_by'))} rows carry a "
            f"known blocker and would report BLOCKED rather than be skipped"
        )
        return 0

    probe = Path("/tmp/cert-matrix.jsonl")
    results, failures = [], 0
    for row in rows:
        for ctx in contexts:
            if row.get("blocked_by"):
                print(f"[{row['id']} @{ctx}] BLOCKED {row['blocked_by']}")
                results.append(
                    {"id": row["id"], "ctx": ctx, "status": "BLOCKED", "reason": row["blocked_by"]}
                )
                continue
            cmd = command(row, ctx, args)
            proc = subprocess.run(cmd + ["--json", str(probe)])
            if proc.returncode != 0:
                print(f"[{row['id']} @{ctx}] FAIL bench exited {proc.returncode}")
                results.append({"id": row["id"], "ctx": ctx, "status": "FAIL"})
                failures += 1
                continue
            m = read_last_row(probe)
            guard = GUARDS.get(row["id"])
            status = "OK"
            if guard and ctx == 16384:
                if m.get("PP_mean", 0) < guard["pp"] or m.get("TG_mean", 0) < guard["tg"]:
                    status = f"REGRESSION (guard PP>={guard['pp']}, TG>={guard['tg']})"
                    failures += 1
            print(
                f"[{row['id']} @{ctx}] PP {m.get('PP_mean', 0):.1f} TG {m.get('TG_mean', 0):.2f} "
                f"TTFT {m.get('TTFT_mean', 0):.0f}ms VRAM {m.get('vram_gib_mean', 0):.2f}GiB "
                f"RSS {m.get('server_rss_gib_mean', 0):.1f}GiB  {status}"
            )
            results.append(
                {
                    "id": row["id"],
                    "pair": row["pair"],
                    "family": row["family"],
                    "ctx": ctx,
                    "status": status,
                    **{
                        k: m.get(k)
                        for k in (
                            "PP_mean",
                            "TG_mean",
                            "TTFT_mean",
                            "vram_gib_mean",
                            "server_rss_gib_mean",
                            "itl_p50_mean",
                            "itl_p95_mean",
                        )
                    },
                }
            )

    gguf = {}
    for r in results:
        if r.get("family") == "gguf" and r.get("status") == "OK":
            gguf.setdefault(r["pair"], []).append(r)
    for pair, items in gguf.items():
        base = next(
            (r for r in results if r.get("pair") == pair and r.get("family") == "native"), None
        )
        if not base:
            print(
                f"\n[parity {pair}] no native row on this host -- GGUF numbers have no "
                f"reference; add a native {pair} checkpoint to close the gate"
            )
            continue
        for r in items:
            print(
                f"\n[parity {r['id']} vs {base['id']} @{r['ctx']}] "
                f"PP {100.0 * r['PP_mean'] / base['PP_mean']:.1f}% of native, "
                f"TG {100.0 * r['TG_mean'] / base['TG_mean']:.1f}%, "
                f"VRAM {r['vram_gib_mean'] - base['vram_gib_mean']:+.2f} GiB "
                f"(weights differ between these files: throughput/capacity only)"
            )

    out = Path(args.json_out) if args.json_out else None
    if out:
        with out.open("a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nrows appended to {out}")
    print(
        f"\n{'PASS' if not failures else f'FAIL ({failures} rows)'} -- "
        f"{sum(1 for r in results if r.get('status') == 'BLOCKED')} blocked, "
        f"clear blockers or remove the row; do not leave it silent"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
