"""One boot, many configs: re-pin env flags between requests via FREETOKEN_RUNTIME_ENV_FILE.

Usage: python scripts/bench-sweep.py --sweep sweep.json [bench_pp_tg args, e.g. --model M
--tokens 16384 --decode 256 --serve-arg="--spec-mtp 6"]
sweep.json: [{"name": "k0", "env": {"FREETOKEN_MTP_FORCE_DEPTH": 0}}, ...]. The first entry is the
SHA reference. A flag missing from a later entry is reset (removed) so entries stay independent.
Only flags read at call time apply; graph-baked flags need --no-graph.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks import bench_pp_tg as bench  # noqa: E402


def main() -> int:
    pre = argparse.ArgumentParser()
    pre.add_argument("--sweep", required=True)
    pre.add_argument("--out", default=None)
    own, rest = pre.parse_known_args()
    configs = json.loads(Path(own.sweep).read_text())
    args = bench.parse_args(rest)
    prompt = (
        Path(args.prompt_file).read_text(errors="replace")
        if args.prompt_file_exact
        else bench.build_prompt_text(args.model, args.prompt_file, args.tokens, args.prompt_offset)
    )
    tmp_dir = os.environ.get("TMPDIR", "/models/desenvolvimento/tmp")
    fd, log_path = tempfile.mkstemp(prefix="bench-sweep-", suffix=".log", dir=tmp_dir)
    env_file = Path(tmp_dir) / f"sweep-env-{os.getpid()}.json"
    env_file.write_text("{}")
    port = bench.free_port()
    origin = f"http://127.0.0.1:{port}"
    env = dict(os.environ, FREETOKEN_RUNTIME_ENV_FILE=str(env_file))
    if any("--spec-mtp" in a for a in args.serve_args):
        env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    cmd = bench.serve_cmd(args, port)
    results = []
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        pump = threading.Thread(target=bench.pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            bench.wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = bench.get_json(f"{origin}/v1/models")["data"][0]["id"]
            active: set[str] = set()
            for index, cfg in enumerate(configs):
                values = {key: None for key in active} | cfg.get("env", {})
                active = set(cfg.get("env", {}))
                env_file.write_text(json.dumps(values))
                os.utime(env_file, (time.time() + index + 1, time.time() + index + 1))
                for _ in range(cfg.get("warmups", 0)):
                    bench.stream_completion(origin, model_id, prompt, args, proc=proc)
                for _ in range(cfg.get("repeats", 1)):
                    row = bench.one_run(origin, model_id, prompt, args, proc)
                    results.append(
                        {
                            "name": cfg["name"],
                            "tg": row["decode_tok_s"],
                            "pp": row["prefill_tok_s"],
                            "ttft_ms": row["ttft_ms"],
                            "sha": row["output_sha1"],
                            "vram_gib": row["vram_gib"],
                            "rss_gib": row["server_rss_gib"],
                            "text": row["output_text"],
                        }
                    )
                    ref = results[0]["sha"]
                    ref_text, text = results[0]["text"], results[-1]["text"]
                    first_diff = next(
                        (i for i, (x, y) in enumerate(zip(ref_text, text)) if x != y),
                        None if len(ref_text) == len(text) else min(len(ref_text), len(text)),
                    )
                    print(
                        f"[sweep] {cfg['name']:<24} TG {row['decode_tok_s']:7.2f}  "
                        f"PP {row['prefill_tok_s']:7.1f}  sha {row['output_sha1']} "
                        f"{'MATCH' if row['output_sha1'] == ref else f'DIVERGE@char{first_diff}'}",
                        flush=True,
                    )
        finally:
            bench.stop_server(proc)
            pump.join(timeout=10)
            env_file.unlink(missing_ok=True)
    if own.out:
        Path(own.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
