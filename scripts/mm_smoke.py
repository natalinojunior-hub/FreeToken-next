"""Vision (mmproj or inline-tower) serve smoke: boot a model, answer one image prompt.

Streams a deterministic solid-color PNG as a base64 ``image_url`` part to
``/v1/chat/completions`` and FAILs unless the server returns non-empty text.
One boot per model; prints a JSON line for the certification record.

    python scripts/mm_smoke.py --model /models/... --json out.jsonl --label ista-mm
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import hashlib
import urllib.request
import zlib
from pathlib import Path


def _solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    row = b"\x00" + bytes(rgb) * size

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return len(body).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    ihdr = size.to_bytes(4, "big") * 2 + b"\x08\x02\x00\x00\x00"
    raw = row * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _chat_image(origin: str, model_id: str, png: bytes, max_tokens: int, timeout: float) -> dict:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    body = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "What color is this image?"},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--model", required=True)
    a.add_argument("--worktree", default=str(Path(__file__).resolve().parent.parent))
    a.add_argument("--pythonpath", default=None)
    a.add_argument("--label", required=True)
    a.add_argument("--json", required=True, type=Path)
    a.add_argument("--context", type=int, default=8192)
    a.add_argument("--max-tokens", type=int, default=32)
    a.add_argument("--server-timeout", type=float, default=1800)
    a.add_argument("--request-timeout", type=float, default=600)
    a.add_argument("--serve-arg", action="append", default=[])
    args = a.parse_args()

    sys.path.insert(0, str(Path(args.worktree) / "benchmarks"))
    import bench_pp_tg as bench

    port = bench.free_port()
    origin = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len-override",
        str(args.context),
    ] + [x for arg in args.serve_arg for x in arg.split("=", 1)]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.json.with_suffix(f".{args.label}.log")
    record = {
        "kind": "mm-smoke",
        "label": args.label,
        "model": args.model,
        "context": args.context,
        "started_at": time.time(),
    }
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=args.worktree,
            env={**os.environ, "PYTHONPATH": args.pythonpath or f"{args.worktree}/python"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pump = threading.Thread(target=bench.pump_output, args=(proc.stdout, log_file), daemon=True)
        pump.start()
        status = "FAIL"
        try:
            bench.wait_ready(origin, proc, str(log_path), args.server_timeout)
            model_id = bench.get_json(f"{origin}/v1/models")["data"][0]["id"]
            # 64x64 red square: the answer must be non-empty; exact wording is not asserted.
            response = _chat_image(
                origin, model_id, _solid_png((220, 30, 30)), args.max_tokens, args.request_timeout
            )
            text = (response["choices"][0]["message"].get("content") or "").strip()
            usage = response.get("usage", {})
            record.update(
                answer_sha256=hashlib.sha256(text.encode(errors="replace")).hexdigest()[:16],
                completion_tokens=usage.get("completion_tokens"),
            )
            if text and usage.get("completion_tokens"):
                status = "PASS"
        except Exception as exc:  # noqa: BLE001 - certification harness reports, never raises
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    record.update(status=status, finished_at=time.time(), log_path=str(log_path))
    with args.json.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
