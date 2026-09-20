#!/usr/bin/env python3
"""bench-profile.py — Benchmark com validação automática contra anchors PERFORMANCE.md"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ANCHORS = {
    "35B-A3B-16K": {"pp_min": 4600, "tg_min": 158, "vram_max": 15.0, "sha1": "614aa7bcdf59"},
    "Flash-Next-16K": {"pp_min": 1850, "tg_min": 28.5, "vram_max": 15.0, "sha1": "614aa7bcdf59"},
    "35B-A3B-128K": {"pp_min": 3189, "tg_min": 89.3, "vram_max": 14.4, "sha1": None},
    "35B-A3B-256K": {"pp_min": 2354, "tg_min": 63.8, "vram_max": 14.5, "sha1": None},
    "Flash-Next-128K": {"pp_min": 1376, "tg_min": 4.86, "vram_max": 14.84, "sha1": None},
}

MODEL_TO_ANCHOR = {
    "Qwen3.6-35B-A3B-NVFP4-FT": "35B-A3B-16K",
    "Qwen3.6-35B-A3B-NVFP4-FT-128K": "35B-A3B-128K",
    "Qwen3.6-35B-A3B-NVFP4-FT-256K": "35B-A3B-256K",
    "Qwen3.8-Flash-Next-NVFP4-Radix": "Flash-Next-16K",
    "Qwen3.8-Flash-Next-NVFP4-Radix-128K": "Flash-Next-128K",
}


def run_benchmark(model, tokens, decode, repeats, warmups, serve_args, label, json_out):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/models/outros/cuda-13.3",
            "PATH": "/models/outros/cuda-13.3/bin:" + env.get("PATH", ""),
            "LD_LIBRARY_PATH": "/models/outros/cuda-13.3/lib64:" + env.get("LD_LIBRARY_PATH", ""),
            "TORCH_CUDA_ARCH_LIST": "12.0;12.0a",
            "TMPDIR": "/models/desenvolvimento/tmp",
            "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1",
        }
    )

    cmd = [
        "/home/natal/.local/bin/uv",
        "run",
        "--no-sync",
        "benchmarks/bench_pp_tg.py",
        "--model",
        model,
        "--tokens",
        str(tokens),
        "--decode",
        str(decode),
        "--repeats",
        str(repeats),
        "--warmups",
        str(warmups),
        "--label",
        label,
        "--json",
        json_out,
    ]
    if serve_args:
        for arg in serve_args.split():
            cmd.extend(["--serve-arg", arg])

    print(f"[bench-profile] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"[bench-profile] FAILED (exit {result.returncode})")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        return None

    return json_out


def validate_results(json_path, anchor_key):
    with open(json_path) as f:
        data = json.load(f)

    anchor = ANCHORS[anchor_key]
    runs = data.get("runs", [])
    if not runs:
        return False, "Nenhum run no JSON"

    # Pega o último run (steady state)
    last = runs[-1]
    pp = last.get("prefill_tok_s", 0)
    tg = last.get("decode_tok_s", 0)
    vram = last.get("vram_gib", 0)
    sha1 = last.get("output_sha1", "")

    errors = []
    if pp < anchor["pp_min"]:
        errors.append(f"PP {pp:.1f} < {anchor['pp_min']} (anchor)")
    if tg < anchor["tg_min"]:
        errors.append(f"TG {tg:.2f} < {anchor['tg_min']} (anchor)")
    if vram > anchor["vram_max"]:
        errors.append(f"VRAM {vram:.2f}GiB > {anchor['vram_max']}GiB (anchor)")
    if anchor["sha1"] and sha1 != anchor["sha1"]:
        errors.append(f"SHA1 {sha1} != {anchor['sha1']} (anchor)")

    if errors:
        return False, "; ".join(errors)
    return True, f"PP={pp:.1f} TG={tg:.2f} VRAM={vram:.2f}GiB SHA1={sha1[:12]} ✓"


def main():
    ap = argparse.ArgumentParser(description="Benchmark com validação de anchors")
    ap.add_argument("--model", required=True, help="Path do modelo")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--serve-args", default="--num-tokens 16576 --cache-type naive")
    ap.add_argument("--label", default="profile")
    ap.add_argument("--json-out", default="/tmp/bench_profile.jsonl")
    args = ap.parse_args()

    model_name = Path(args.model).name
    anchor_key = MODEL_TO_ANCHOR.get(model_name)
    if not anchor_key:
        print(f"[bench-profile] WARN: Modelo '{model_name}' não tem anchor definido")
        anchor_key = list(ANCHORS.keys())[0]

    print(f"[bench-profile] Modelo: {model_name} -> Anchor: {anchor_key}")
    print(f"[bench-profile] Anchors: {ANCHORS[anchor_key]}")

    json_path = run_benchmark(
        args.model,
        args.tokens,
        args.decode,
        args.repeats,
        args.warmups,
        args.serve_args,
        args.label,
        args.json_out,
    )

    if json_path is None:
        sys.exit(1)

    ok, msg = validate_results(json_path, anchor_key)
    if ok:
        print(f"\n[bench-profile] ✅ PASS: {msg}")
        sys.exit(0)
    else:
        print(f"\n[bench-profile] ❌ FAIL: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
