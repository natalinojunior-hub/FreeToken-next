"""One boot, interleaved runtime arms using bench_pp_tg's exact request/measurement math.

Run under: flock /tmp/gpu.lock .venv/bin/python scripts/bench-arms.py --arms arms.json
--model MODEL --tokens 16384 --decode 256 --repeats 2 --json results.jsonl --log server.log
Arms: [{"name": "default", "env": {}}, {"name": "raw", "env":
{"FREETOKEN_MTP_FORCE_DEPTH": 0}, "reset_mtp_controller": true}].
Optional recapture authorizes B changes; spec overrides measurement args (tokens, decode,
sample, prompt_file, prompt_file_exact, prompt_offset). Geometry/weight flags require reboot.
Missing env keys restore the boot environment, including unset values. Initial boot warms
once; subsequent warmups run once per arm only after an actual graph recapture. Repeats
visit every arm in order (ABAB). Readiness blocks on the server stdout pipe; requests block
on HTTP streams. GPU sampling/polling watchdogs in the canonical bench are disabled here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks import bench_pp_tg as bench  # noqa: E402
from freetoken.server.runtime import RuntimeRequest, classify_knob  # noqa: E402


def apply_arm(origin: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{origin}/v1/admin/runtime",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    if result.get("status") != "ok":
        raise RuntimeError(f"runtime update failed: {result}")
    return result


def arm_args(args: argparse.Namespace, arm: dict) -> argparse.Namespace:
    spec = arm.get("spec", {})
    allowed = {"tokens", "decode", "sample", "prompt_file", "prompt_file_exact", "prompt_offset"}
    if set(spec) - allowed:
        raise ValueError(
            f"unsupported spec (reboot required for engine settings): {set(spec) - allowed}"
        )
    return argparse.Namespace(**(vars(args) | spec))


def load_arms(path: str) -> list[dict]:
    arms = json.loads(Path(path).read_text())
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty JSON array")
    names = set()
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("name"), str) or not arm["name"]:
            raise ValueError("each arm needs a nonempty name")
        if arm["name"] in names:
            raise ValueError("arm names must be unique")
        names.add(arm["name"])
        env = arm.get("env", {})
        if not isinstance(env, dict):
            raise ValueError("arm env must be an object")
        RuntimeRequest.model_validate(
            {
                "env": env,
                "recapture_graphs": arm.get("recapture", False),
                "reset_mtp_controller": arm.get("reset_mtp_controller", False),
                "reset_moe_stats": arm.get("reset_moe_stats", False),
            }
        )
        reboot = [k for k in env if classify_knob(k) == "C"]
        if reboot:
            raise ValueError(f"{arm['name']}: reboot required for {reboot}")
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--log")
    own, rest = parser.parse_known_args()
    args = bench.parse_args(rest)
    arms = load_arms(own.arms)
    if args.repeats < 1 or args.warmups < 0 or args.fresh_server_each_repeat or args.keep_alive:
        raise ValueError("need repeats >= 1, warmups >= 0, one boot and orderly teardown")
    specs = [arm_args(args, arm) for arm in arms]
    if any(spec.tokens < 1 or spec.decode < 2 for spec in specs):
        raise ValueError("need tokens >= 1 and decode >= 2")
    prompts = [
        Path(spec.prompt_file).read_text(errors="replace")
        if spec.prompt_file_exact
        else bench.build_prompt_text(spec.model, spec.prompt_file, spec.tokens, spec.prompt_offset)
        for spec in specs
    ]
    boot_args = argparse.Namespace(**vars(args))
    boot_args.tokens = max(spec.tokens for spec in specs)
    boot_args.decode = max(spec.decode for spec in specs)
    port = bench.free_port()
    origin = f"http://127.0.0.1:{port}"
    cmd = bench.serve_cmd(boot_args, port)
    env = dict(os.environ, FREETOKEN_DEBUG_RUNTIME="1", PYTHONFAULTHANDLER="1")
    env.pop("FREETOKEN_RUNTIME_ENV_FILE", None)
    keys = set().union(*(set(arm.get("env", {})) for arm in arms))
    baseline = {key: env.get(key) for key in keys}
    log_path = own.log or str(Path(args.json_out or "/tmp/bench-arms.jsonl").with_suffix(".log"))
    ready = threading.Event()
    ended = threading.Event()
    started = time.monotonic()
    runs: list[list[dict]] = [[] for _ in arms]
    updates: list[list[dict]] = [[] for _ in arms]
    warmed: set[int] = set()
    boot_s = warmup_s = recapture_s = 0.0
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if b"API server is ready to serve" in line:
                    ready.set()
            ended.set()
            ready.set()  # wake startup on failure too

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            if not ready.wait(args.server_timeout) or ended.is_set():
                raise RuntimeError(f"server failed readiness; see {log_path}")
            boot_s = time.monotonic() - started
            model_id = bench.get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[arms] ready in {boot_s:.2f}s; server log {log_path}", flush=True)
            current = dict(baseline)
            for repeat in range(args.repeats):
                for index, (arm, spec, prompt) in enumerate(zip(arms, specs, prompts)):
                    target = baseline | {
                        k: None if v is None else str(v) for k, v in arm.get("env", {}).items()
                    }
                    baked = [
                        k for k in keys if current.get(k) != target[k] and classify_knob(k) == "B"
                    ]
                    if baked and not arm.get("recapture", False):
                        raise ValueError(f"{arm['name']}: set recapture=true for {baked}")
                    t0 = time.monotonic()
                    applied = apply_arm(
                        origin,
                        {
                            "env": target,
                            "recapture_graphs": bool(baked),
                            "reset_mtp_controller": arm.get("reset_mtp_controller", False),
                            "reset_moe_stats": arm.get("reset_moe_stats", False),
                        },
                    )
                    if applied["recaptured"]:
                        recapture_s += time.monotonic() - t0
                    current = target
                    updates[index].append(applied)
                    if (repeat == 0 and index == 0) or (
                        applied["recaptured"] and index not in warmed
                    ):
                        t0 = time.monotonic()
                        for _ in range(args.warmups):
                            bench.stream_completion(origin, model_id, prompt, spec)
                        warmup_s += time.monotonic() - t0
                        warmed.add(index)
                    row = bench.one_run(origin, model_id, prompt, spec, proc, monitor=False)
                    if (
                        row["prompt_tokens"] != spec.tokens
                        or row["completion_tokens"] != spec.decode
                    ):
                        raise RuntimeError(f"{arm['name']}: invalid prompt/decode token count")
                    row["repeat"] = repeat
                    row["arm"] = arm["name"]
                    row["runtime"] = applied
                    runs[index].append(row)
                    print(
                        f"[arms] r{repeat + 1} {arm['name']}: TG {row['decode_tok_s']:.2f}, SHA {row['output_sha1']}",
                        flush=True,
                    )
        finally:
            bench.stop_server(proc)
            thread.join(timeout=10)
    wall_s = time.monotonic() - started
    reference = runs[0][0]["output_sha1"]
    # Equal request work; separate boots additionally repeat boot + initial warmups.
    separate_s = (
        wall_s
        + (len(arms) - 1) * boot_s
        + sum(
            args.warmups * bench.mean(rows, "e2e_ms") / 1000
            for i, rows in enumerate(runs)
            if i not in warmed
        )
    )
    summaries: list[dict[str, Any]] = []
    for arm, rows, applied in zip(arms, runs, updates):
        summaries.append(
            {
                "name": arm["name"],
                "env": arm.get("env", {}),
                "spec": arm.get("spec", {}),
                "model": args.model,
                "serve_cmd": cmd,
                "n": len(rows),
                "runs": rows,
                "runtime_updates": applied,
                "output_sha1": rows[0]["output_sha1"],
                "sha_matches_first": all(row["output_sha1"] == reference for row in rows),
                "PP_mean": bench.mean(rows, "prefill_tok_s"),
                "PP_min": min(r["prefill_tok_s"] for r in rows),
                "TG_mean": bench.mean(rows, "decode_tok_s"),
                "TG_min": min(r["decode_tok_s"] for r in rows),
                "TTFT_mean": bench.mean(rows, "ttft_ms"),
                "TTFT_min": min(r["ttft_ms"] for r in rows),
                "wall_s": wall_s,
                "boot_s": boot_s,
                "warmup_s": warmup_s,
                "recapture_s": recapture_s,
                "estimated_separate_boot_s": separate_s,
                "server_log": log_path,
            }
        )
    if args.json_out:
        with open(args.json_out, "a") as out:
            for summary in summaries:
                out.write(json.dumps(summary) + "\n")
    print("arm                     TG mean/min     PP mean     TTFT ms   SHA")
    for row in summaries:
        print(
            f"{row['name']:<23} {row['TG_mean']:7.2f}/{row['TG_min']:.2f} {row['PP_mean']:10.1f} {row['TTFT_mean']:10.1f} {'MATCH' if row['sha_matches_first'] else 'DIFF'}"
        )
    print(f"wall {wall_s:.2f}s; equivalent separate boots estimated {separate_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
