"""VRAM certification matrix: one server boot per configuration, every workload on it.

Each configuration boots `ft serve` once and runs, in order, a short prompt, a
~4K prompt and a context-filling prompt (max_seq_len minus decode), streaming
tokens with bench_pp_tg's stall watchdog. A hang (no token for --stall-timeout
seconds, or no first token for --ttft-timeout), a dead server, or an error
marks the configuration FAIL, dumps every Python thread's stack from the server
(SIGUSR1 + faulthandler) into its log, kills it and moves on -- nothing waits
idle. Results stream as JSON lines to --out.

    python benchmarks/cert_matrix.py --out results.jsonl \
        --config "gguf-t3|/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS|--kv-format=turbo3"
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
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_pp_tg as bench  # noqa: E402

_SITE = Path(tempfile.mkdtemp(prefix="cert-site-"))
(_SITE / "sitecustomize.py").write_text(
    "import ctypes, faulthandler, signal\n"
    "faulthandler.register(signal.SIGUSR1, all_threads=True)\n"
    # PR_SET_PTRACER_ANY: let the watchdog's py-spy attach despite yama ptrace_scope=1.
    "ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)\n"
)
_RECON = re.compile(r"\[Phase [CHI]\].*|PLANNING COMPLETE.*|VRAM plan infeasible.*")


def run_config(label: str, model: str, extra: list[str], a: argparse.Namespace) -> dict:
    args = bench.parse_args(
        ["--model", model, "--tokens", str(a.ctx - a.decode - 64), "--decode", str(a.decode)]
        + [f"--serve-arg={x}" for x in [f"--max-seq-len-override={a.ctx}", *extra]]
        + (["--no-graph"] if "--no-graph" in extra else [])
    )
    args.serve_args = [x for x in args.serve_args if x != "--no-graph"]
    args.ttft_timeout, args.stall_timeout = a.ttft_timeout, a.stall_timeout
    port = bench.free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"cert-{label}-", suffix=".log", dir=a.tmp)
    env = dict(os.environ, PYTHONPATH=f"{_SITE}:{os.environ.get('PYTHONPATH', '')}")
    if any("--spec-mtp" in x for x in extra):
        env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    result: dict = {"label": label, "model": model, "args": extra, "log": log_path, "runs": []}
    t0 = time.monotonic()
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            bench.serve_cmd(args, port),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        pump = threading.Thread(target=bench.pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            bench.wait_ready(origin, proc, log_path, a.server_timeout)
            result["boot_s"] = round(time.monotonic() - t0, 1)
            model_id = bench.get_json(f"{origin}/v1/models")["data"][0]["id"]
            for name, tokens in (("short", 256), ("4k", 4096), ("fill", args.tokens)):
                args.tokens, args.label = tokens, f"{label}:{name}"
                prompt = bench.build_prompt_text(model, args.prompt_file, tokens, 0)
                row = bench.one_run(origin, model_id, prompt, args, proc)
                row.pop("output_text", None)
                result["runs"].append(row)
                print(
                    f"[cert] {label}:{name} PP {row['prefill_tok_s']:.0f} TG "
                    f"{row['decode_tok_s']:.2f} VRAM {row['vram_gib']:.2f} GiB",
                    flush=True,
                )
            result["status"] = "PASS"
        except BaseException as exc:  # SystemExit from the bench watchdog included
            result["status"] = "FAIL"
            result["error"] = f"{type(exc).__name__}: {exc}"
            for pid in bench._tree_pids(proc.pid):
                try:
                    os.kill(pid, signal.SIGUSR1)
                except OSError:
                    pass
        finally:
            bench.stop_server(proc)
            pump.join(timeout=10)
    text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    result["reconciliation"] = _RECON.findall(re.sub(r"\x1b\[[0-9;]*m", "", text))
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", required=True, help="label|model|arg,arg,...")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--ttft-timeout", type=float, default=300.0)
    p.add_argument("--stall-timeout", type=float, default=30.0)
    p.add_argument("--server-timeout", type=float, default=1200.0)
    p.add_argument("--tmp", default=os.environ.get("TMPDIR", "/models/desenvolvimento/tmp"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    failed = 0
    for spec in a.config:
        label, model, extra = (spec.split("|") + [""])[:3]
        r = run_config(label, model, [x for x in extra.split(",") if x], a)
        failed += r["status"] != "PASS"
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[cert] {label}: {r['status']} {r.get('error', '')}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
