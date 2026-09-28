"""Concurrent-decode probe: aggregate TG at batch size N.

bench_pp_tg.py issues requests strictly sequentially, so --max-running-requests >1 never
actually produces concurrent decode. This reuses that harness's server lifecycle and
streaming client but fans N requests out in parallel, and reports aggregate committed
throughput over the shared decode window alongside per-stream TG.

Concurrency comes from BENCH_CONCURRENCY (default 1); every other flag is bench_pp_tg's.
Each stream gets a distinct prompt offset so the radix prefix cache cannot collapse the N
prefills into one, and so each stream carries its own 16K KV footprint.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import bench_pp_tg as B  # noqa: E402


def main() -> int:
    args = B.parse_args(sys.argv[1:])
    n = int(os.environ.get("BENCH_CONCURRENCY", "1"))
    prompts = [
        B.build_prompt_text(args.model, args.prompt_file, args.tokens, args.prompt_offset + i * 64)
        for i in range(n)
    ]

    port = B.free_port()
    origin = f"http://127.0.0.1:{port}"
    tmp_dir = os.environ.get("TMPDIR", "/models/desenvolvimento/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, log_path = tempfile.mkstemp(prefix="bench-conc-", suffix=".log", dir=tmp_dir)
    cmd = B.serve_cmd(args, port)
    env = dict(os.environ)
    if any("--spec-mtp" in v for v in args.serve_args):
        env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"

    print(f"[conc] N={n} serve: {' '.join(cmd)}", flush=True)
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        pump = threading.Thread(target=B.pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            B.wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = B.get_json(f"{origin}/v1/models")["data"][0]["id"]

            results: list[dict] = [None] * n  # type: ignore[assignment]
            errors: list[str] = []

            def worker(i: int) -> None:
                try:
                    results[i] = B.stream_completion(origin, model_id, prompts[i], args, proc=proc)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"stream {i}: {type(exc).__name__}: {exc}")

            # Warmup: one full request so graphs/pools are hot before the measured window.
            B.stream_completion(origin, model_id, prompts[0], args, proc=proc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0

            if errors:
                print("[conc] ERRORS:", *errors, sep="\n  ")
                return 1

            total_tok = sum(r["usage"]["completion_tokens"] for r in results)
            first = min(r["stamps"][0] for r in results)
            last = max(r["stamps"][-1] for r in results)
            window = last - first
            per_stream = []
            for i, r in enumerate(results):
                st = r["stamps"]
                steps = r["usage"]["completion_tokens"] - 1
                dt = st[-1] - st[0]
                per_stream.append(steps / dt if dt > 0 else 0.0)
                gaps = sorted((b - a) * 1e3 for a, b in zip(st, st[1:]))
                print(
                    f"  stream {i}: tokens={r['usage']['completion_tokens']} "
                    f"TG={steps / dt if dt > 0 else 0:.2f} "
                    f"TTFT={(st[0] - r['t0']) * 1e3:.1f}ms "
                    f"ITL p50={gaps[len(gaps) // 2]:.2f} p95={gaps[int(len(gaps) * 0.95)]:.2f}ms"
                )
            agg = (total_tok - n) / window if window > 0 else 0.0
            stats = B.get_json(f"{origin}/v1/stats")
            print(
                f"\n[conc] N={n} label={args.label}\n"
                f"  AGGREGATE TG = {agg:.2f} tok/s  ({total_tok} tok / {window:.3f}s window)\n"
                f"  wall (incl. skew) = {wall:.3f}s\n"
                f"  per-stream TG mean={statistics.mean(per_stream):.2f} "
                f"min={min(per_stream):.2f} max={max(per_stream):.2f}\n"
                f"  engine decode_tps={stats.get('decode_tps')} prefill_tps={stats.get('prefill_tps')}\n"
                f"  log: {log_path}"
            )
            return 0
        finally:
            B.stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
