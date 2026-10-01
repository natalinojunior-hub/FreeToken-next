"""Prefill + decode regression harness for a served model, at a chosen context size.

`bench_decode_moe.py` answers "which MoE backend decodes fastest on an AIME prompt".
This answers the phase-1 question instead: *what does the whole serving path cost at a
given context length, and did a patch move it?* One server, one config, N repeats:

    PP   = prompt_tokens / time-to-first-token        (prefill throughput, what FreeToken
                                                       is unusually good at — the guard)
    TG   = (completion_tokens - 1) / (t_last - t_first)  (bs=1 decode throughput)
    ITL  = inter-token latency p50/p95, TTFT, engine-side throughput from /v1/stats,
           VRAM (server + nvidia-smi), GPU utilisation, host RSS and MemAvailable.

The prompt is a slice of a local corpus (--prompt-file) that has been fixed-pointed with
the checkpoint's own tokenizer to exactly --tokens ids, so `prompt_tokens` is exact and
repeats are byte-identical; nothing is downloaded per run. Sampling is
greedy unless --sample: the workload is throughput, not quality, and temperature adds
run-to-run routing noise.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_pp_tg.py \
        --model /models/Qwen3.6-35B-A3B-NVFP4-FT --tokens 16384 --decode 128 --repeats 3 \
        --label v0.1.3-baseline --json /tmp/pp_tg.jsonl

Extra engine flags pass through verbatim (--serve-arg '--kv-reserve-tokens 4096'), which
is how an A/B changes exactly one knob. JSONL rows keep every measured field plus the
resolved serve command, so a row is enough to reproduce the number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CORPUS = "/models/servers/prompt-235k.txt"
CHARS_PER_TOKEN = 4  # only used to size the text slice that gets tokenized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="checkpoint dir / .ftw dir / .gguf path")
    p.add_argument(
        "--tokens", type=int, default=16384, help="prompt tokens (the context under test)"
    )
    p.add_argument("--decode", type=int, default=128, help="generated tokens per request")
    p.add_argument(
        "--tg-curve", type=int, default=0, metavar="N",
        help="also report decode tok/s per N-token window (e.g. 1024) of the last run",
    )  # fmt: skip
    p.add_argument(
        "--context-headroom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="report max runnable context from KV feasibility and simulated TG projection (default: on)",
    )
    p.add_argument("--repeats", type=int, default=3, help="measured requests after warmup")
    p.add_argument("--warmups", type=int, default=2, help="untimed requests at full context")
    p.add_argument(
        "--fresh-server-each-repeat",
        action="store_true",
        help="restart server before every measured repetition (cold cache)",
    )
    p.add_argument(
        "--token-trace",
        default=None,
        help="absolute JSONL path for the opt-in server token trace",
    )
    p.add_argument("--prompt-file", default=os.environ.get("FREETOKEN_NEXT_PROMPT", DEFAULT_CORPUS))
    p.add_argument(
        "--prompt-file-exact",
        action="store_true",
        help="send --prompt-file verbatim instead of deriving a fixed token slice",
    )
    p.add_argument(
        "--prompt-offset", type=int, default=0, help="token offset into the corpus slice"
    )
    p.add_argument("--gpu", default=None, help="UUID or nvidia-smi index, as ft serve --gpu")
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--sample", action="store_true", help="use the checkpoint's sampling instead of greedy"
    )
    p.add_argument(
        "--serve-arg",
        dest="serve_args",
        action="append",
        default=[],
        help="extra flag for ft serve, verbatim (repeatable)",
    )
    p.add_argument("--label", default="run", help="tag written into the output rows")
    p.add_argument(
        "--server-timeout",
        type=float,
        default=600,
        help="max seconds to wait for server startup (default: 600s)",
    )
    p.add_argument(
        "--stall-timeout",
        type=float,
        default=45.0,
        help="max seconds between token arrivals during decode (default: 45s)",
    )
    p.add_argument(
        "--ttft-timeout",
        type=float,
        default=None,
        help="max seconds before first token; default scales with prompt length",
    )
    p.add_argument("--json", dest="json_out", default=None, help="append result rows here")
    p.add_argument(
        "--no-history",
        action="store_true",
        help="skip appending this run's summary to the per-model run-history log",
    )
    p.add_argument(
        "--keep-alive", action="store_true", help="leave the server running (manual probing)"
    )
    return p.parse_args(argv)


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    for _ in range(100):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        try:
            with socket.socket() as s2:
                s2.bind(("127.0.0.1", p + 1))
            return p
        except OSError:
            continue
    return p


def build_prompt_text(model: str, path: str, tokens: int, offset: int) -> str:
    """A corpus slice that re-tokenizes to exactly `tokens` ids with the checkpoint's tokenizer.

    Sent as text because the server rejects token-id prompt inputs; the fixed point below is
    what makes `prompt_tokens` exact and repeats identical, without a probe request per run.
    """
    text = Path(path).read_text(errors="replace")
    start = offset * tokens * CHARS_PER_TOKEN
    chunk = text[start : start + tokens * CHARS_PER_TOKEN * 2] or text
    if not chunk.strip():
        sys.exit(f"[bench] corpus {path} exhausted at --prompt-offset {offset}")
    from freetoken.utils.hf import load_tokenizer

    # The engine's own loader, not AutoTokenizer: a GGUF checkpoint carries its vocab inside
    # the file, and transformers' loader only knows how to find a directory of HF artifacts.
    # Without this the harness cannot build an exact-token prompt for a .gguf model at all,
    # which is the whole point of the GGUF rows of the certification matrix.
    tok = load_tokenizer(model)
    ids = tok(chunk, add_special_tokens=False)["input_ids"]
    if len(ids) < tokens:
        sys.exit(f"[bench] corpus slice gave {len(ids)} < {tokens} tokens; use a bigger corpus")
    k = tokens
    for _ in range(12):
        s = tok.decode(ids[:k], skip_special_tokens=True)
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n == tokens:
            return s
        k = max(1, min(len(ids), k + (tokens - n)))
    sys.exit(f"[bench] could not fixpoint a {tokens}-token prompt (last length {n})")


def serve_cmd(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
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
        "--max-running-requests",
        "1",
        "--max-seq-len-override",
        str(args.tokens + args.decode + 64),
        "--cuda-graph-max-bs",
        "0" if args.no_graph else "1",
    ]
    for extra in args.serve_args:
        cmd += extra.split()
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    return cmd


class GpuSampler(threading.Thread):
    """Polls nvidia-smi for util + used memory while measuring; keeps the samples."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.util: list[int] = []
        self.mem_mib: list[int] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                out = (
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    .stdout.strip()
                    .splitlines()[0]
                )
                u, m = (int(float(x)) for x in out.split(","))
                self.util.append(u)
                self.mem_mib.append(m)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=5)
        util, mem = sorted(self.util), sorted(self.mem_mib)
        if not util:
            return {}
        return {
            "gpu_util_mean": sum(util) / len(util),
            "gpu_util_p95": util[min(len(util) - 1, int(len(util) * 0.95))],
            "gpu_mem_used_mib_max": mem[-1],
            "gpu_samples": len(util),
        }


def _tree_pids(pid: int) -> list[int]:
    pids, stack = [pid], [pid]
    while stack:
        cur = stack.pop()
        try:
            out = subprocess.run(
                ["ps", "-o", "pid=", "--ppid", str(cur)], capture_output=True, text=True, timeout=5
            ).stdout.split()
        except Exception:
            out = []
        for q in out:
            if int(q) not in pids:
                pids.append(int(q))
                stack.append(int(q))
    return pids


def proc_rss_kib(pid: int) -> int:
    """Sum VmRSS over the process tree (frontend + scheduler/tokenizer workers)."""
    total = 0
    for p in _tree_pids(pid):
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except OSError:
            continue
    return total


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def stream_completion(
    origin: str, model_id: str, prompt: str, args: argparse.Namespace, proc=None
) -> dict:
    body = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.sample:
        body.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
    req = urllib.request.Request(
        f"{origin}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()

    watch_stop = threading.Event()
    last_event_time = [time.monotonic()]
    token_count = [0]
    in_prefill = [True]
    failure: list[str] = []

    ttft_timeout = args.ttft_timeout or max(90.0, (args.tokens / 1000.0) * 2.5 + 30.0)
    stall_timeout = getattr(args, "stall_timeout", 45.0)
    gpu_idle_timeout = getattr(args, "gpu_idle_timeout", 2.0)

    if proc is not None:

        def _fail(msg: str) -> None:
            # Dump every server thread's stack (faulthandler on SIGUSR1, when the
            # server registered it) before killing, then unblock the reader.
            print(f"\n[bench] {msg}", flush=True)
            failure.append(msg)
            spy = Path(sys.executable).with_name("py-spy")
            for pid in _tree_pids(proc.pid) if spy.exists() else []:
                dump = subprocess.run(
                    [str(spy), "dump", "--native", "--pid", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                print(f"[bench] py-spy {pid}:\n{dump.stdout}{dump.stderr}", flush=True)
            for pid in _tree_pids(proc.pid):
                try:
                    os.kill(pid, signal.SIGUSR1)
                except OSError:
                    pass
            watch_stop.wait(2.0)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

        def _gpu_busy() -> bool:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.split()
                return any(int(u) > 0 for u in out)
            except (OSError, ValueError, subprocess.SubprocessError):
                return True  # unknown: never fail on a missing probe

        def _watch():
            last_heartbeat = time.monotonic()
            last_gpu_busy = time.monotonic()
            while not watch_stop.wait(0.5):
                now = time.monotonic()
                # No token AND an idle GPU is a hang, whatever the phase: fail fast
                # instead of waiting out the TTFT/stall ceilings.
                if _gpu_busy():
                    last_gpu_busy = now
                elif (
                    now - last_gpu_busy > gpu_idle_timeout
                    and now - last_event_time[0] > gpu_idle_timeout
                ):
                    return _fail(f"GPU idle for {now - last_gpu_busy:.0f}s with no token: hang")
                if proc.poll() is not None:
                    return _fail(f"server process died with exitcode {proc.returncode}")
                try:
                    health = get_json(f"{origin}/health", timeout=1)
                except (OSError, ValueError):
                    health = None
                if isinstance(health, dict) and health.get("status") == "error":
                    return _fail(f"server reported failure: {health}")
                if in_prefill[0]:
                    elapsed_ttft = now - t0
                    if now - last_heartbeat >= 5.0:
                        last_heartbeat = now
                        print(
                            f"[bench-watchdog] prefill in progress: elapsed {elapsed_ttft:.1f}s / max {ttft_timeout:.1f}s",
                            flush=True,
                        )
                    if elapsed_ttft > ttft_timeout:
                        return _fail(
                            f"TTFT prefill timed out after {elapsed_ttft:.1f}s (max {ttft_timeout:.1f}s)! Server stalled or GPU deadlocked."
                        )
                else:
                    gap = now - last_event_time[0]
                    if now - last_heartbeat >= 5.0:
                        last_heartbeat = now
                        print(
                            f"[bench-watchdog] decoding: {token_count[0]}/{args.decode} tokens | elapsed {now - t0:.1f}s | last gap {gap:.2f}s",
                            flush=True,
                        )
                    if gap > stall_timeout:
                        return _fail(
                            f"Token generation stalled! No token for {gap:.1f}s (stall-timeout: {stall_timeout:.1f}s) after token {token_count[0]}/{args.decode}."
                        )

        threading.Thread(target=_watch, daemon=True).start()

    try:
        try:
            resp = urllib.request.urlopen(req, timeout=int(ttft_timeout + 60))
        except urllib.error.HTTPError as e:
            sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
        with resp:
            for raw in resp:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:") :].strip()
                if payload == b"[DONE]":
                    break
                now = time.perf_counter()
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = choice.get("text")
                    if text is None:
                        text = delta.get("content")
                    if text is None:
                        text = delta.get("reasoning_content")
                    if text is not None:
                        stamps.append(now)
                        pieces.append(text)
                        in_prefill[0] = False
                        token_count[0] += 1
                        last_event_time[0] = time.monotonic()
    except OSError:
        if not failure:
            raise
    finally:
        watch_stop.set()
    if failure:
        sys.exit(f"[bench] {failure[0]}")
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def tg_curve(stamps: list[float], window: int) -> list[float]:
    """Decode tok/s per ``window`` streamed tokens (a trailing partial window included)."""
    return [
        (len(w) - 1) / (w[-1] - w[0])
        for i in range(0, len(stamps) - 1, window)
        if len(w := stamps[i : i + window + 1]) > 1 and w[-1] > w[0]
    ]


def one_run(origin: str, model_id: str, prompt: str, args: argparse.Namespace, proc) -> dict:
    sampler = GpuSampler()
    sampler.start()
    t_send = time.perf_counter()
    r = stream_completion(origin, model_id, prompt, args, proc=proc)
    stats = get_json(f"{origin}/v1/stats")
    gpu = sampler.stop()
    stamps, usage = r["stamps"], r["usage"]
    if len(stamps) < 2:
        sys.exit(f"[bench] need >=2 token events, got {len(stamps)}")
    prompt_tokens = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    if completion != args.decode:
        print(
            f"[bench] WARNING: completion_tokens={completion} != --decode {args.decode}", flush=True
        )
    steps = completion - 1
    decode_time = stamps[-1] - stamps[0]
    ttft = stamps[0] - r["t0"]
    gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
    tp = stats.get("throughput", {}) or {}
    kv = stats.get("kv", {}) or {}
    return {
        "label": args.label,
        "model": args.model,
        "serve_args": list(args.serve_args),
        "prompt_tokens": prompt_tokens,
        "prefill_tok_s": prompt_tokens / ttft if ttft > 0 else 0.0,
        "decode_tok_s": steps / decode_time if decode_time > 0 else 0.0,
        "ms_per_token": decode_time / steps * 1e3 if steps > 0 else 0.0,
        "ttft_ms": ttft * 1e3,
        "e2e_ms": (stamps[-1] - t_send) * 1e3,
        "itl_ms_p50": gaps[len(gaps) // 2],
        "itl_ms_p95": gaps[min(len(gaps) - 1, int(len(gaps) * 0.95))],
        "itl_ms": [round((b - a) * 1e3, 3) for a, b in zip(stamps, stamps[1:])],
        "completion_tokens": completion,
        "engine_prefill_tps": tp.get("prefill_tps"),
        "engine_decode_tps": tp.get("decode_tps"),
        "kv_used_pages": kv.get("used_pages"),
        "kv_total_pages": kv.get("total_pages"),
        "kv_page_size": kv.get("page_size"),
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "server_rss_gib": proc_rss_kib(proc.pid) / 2**20,
        "mem_available_gib": mem_available_gib(),
        "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
        "output_text": r["text"],
        "tg_curve": tg_curve(stamps, args.tg_curve) if getattr(args, "tg_curve", 0) else None,
        **gpu,
    }


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _serve_arg_value(serve_args: list[str], flag: str) -> str | None:
    """The value following ``flag`` among ``--serve-arg`` entries (each entry may itself
    be a space-joined ``"--spec-mtp 3"``, matching how ``serve_cmd`` expands them)."""
    tokens = [tok for entry in serve_args for tok in entry.split()]
    for i, a in enumerate(tokens):
        if a == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


# Planner's `str(Plan)` ("PLANNING COMPLETE: Plan(chunk=1024, experts=32, kv_pages=512 (...)
# ..."), engine/memory_planner.py Plan.__str__; and the KV-RAM-tiering line, engine/engine.py
# ("KV RAM tiering: 400 device pages, 112 RAM pages (...)"), only printed when tiering is on.
_PLAN_RE = re.compile(r"experts=(\d+).*?kv_pages=(\d+)")
_KV_TIERING_RE = re.compile(r"KV RAM tiering: (\d+) device pages, (\d+) RAM pages")


def parse_engine_log(text: str) -> dict[str, int | None]:
    """``expert_slots``/``kv_pages`` from the planner's final ``Plan(...)`` line, and
    ``kv_device_pages``/``kv_ram_pages`` from the KV-RAM-tiering line (``None`` for both
    when tiering never logged, i.e. KV stayed device-only). Takes the last match of each --
    only the final planning decision and boot-time tiering line matter."""
    out: dict[str, int | None] = {
        "expert_slots": None,
        "kv_pages": None,
        "kv_device_pages": None,
        "kv_ram_pages": None,
    }
    plan_matches = _PLAN_RE.findall(text)
    if plan_matches:
        experts, kv_pages = plan_matches[-1]
        out["expert_slots"] = int(experts)
        out["kv_pages"] = int(kv_pages)
    tiering_matches = _KV_TIERING_RE.findall(text)
    if tiering_matches:
        device_pages, ram_pages = tiering_matches[-1]
        out["kv_device_pages"] = int(device_pages)
        out["kv_ram_pages"] = int(ram_pages)
    return out


# KV context feasibility line, engine/engine.py ("KV context feasibility (...): 128K fits, 256K fits, 512K 3.12 GB short, ...")
_FEASIBILITY_RE = re.compile(r"KV context feasibility.*?:\s*(.*)")
_FITS_RE = re.compile(r"(\d+)([KM])\s+fits")


def parse_context_feasibility(text: str) -> int | None:
    """Extract largest context (tokens) that 'fits' from 'KV context feasibility' log."""
    lines = _FEASIBILITY_RE.findall(text)
    if not lines:
        return None
    matches = _FITS_RE.findall(lines[-1])
    if not matches:
        return None
    tokens_list = [int(val) * (1024 if unit == "K" else 1024 * 1024) for val, unit in matches]
    return max(tokens_list)


def extract_context_headroom(log_path: str | None, tg_tok_s: float | None) -> dict:
    """Derive max runnable context from server KV feasibility log and project TG."""
    max_context = None
    if log_path:
        try:
            text = Path(log_path).read_text(errors="replace")
            max_context = parse_context_feasibility(text)
        except OSError:
            pass
    tg = tg_tok_s if tg_tok_s is not None else 0.0
    simulated = []
    if max_context is not None:
        for pct in (25, 50, 75, 100):
            toks = int(round(max_context * pct / 100.0))
            simulated.append(
                {
                    "fraction": f"{pct}%",
                    "tokens": toks,
                    "tg_tok_s": round(tg, 2),
                    "method": "projected",
                }
            )
    return {
        "max_runnable_context": max_context,
        "source": "engine_kv_feasibility",
        "method": "projected",
        "simulated_tg": simulated,
    }


def print_context_headroom(label: str, headroom: dict) -> None:
    print(f"\n==== [{label}] context headroom ====")
    max_ctx = headroom.get("max_runnable_context")
    if max_ctx is not None:
        print(f"  max runnable context: {max_ctx} tokens (from engine KV feasibility)")
        for item in headroom.get("simulated_tg", []):
            pct = item["fraction"]
            toks = item["tokens"]
            tg = item["tg_tok_s"]
            method = item["method"]
            print(f"  simulated TG @ {pct:>4s} ({toks:7d} tok): {tg:6.2f} tok/s [{method}]")
        print(
            "  note: projected (decode TG is context-flat to the device hot-window; see --kv-reserve-tokens)"
        )
    else:
        print("  max runnable context: unknown (no fitting context in engine KV feasibility)")


def append_history(args: argparse.Namespace, summary: dict, log_path: str | None = None) -> None:
    """Record this run's summary into the per-model history log (skipped on
    ``--no-history``). Never lets a history-write failure, or a missing/unreadable server
    log, affect the bench run's own exit code -- ``history.append_run`` already swallows
    I/O errors, and a log read failure here just leaves the KV/expert fields unset."""
    if args.no_history:
        return
    from freetoken.tuning import history

    record = {
        "label": summary.get("label"),
        "tokens": summary.get("prompt_tokens"),
        "decode_tokens": args.decode,
        "spec_mtp": _serve_arg_value(args.serve_args, "--spec-mtp"),
        "serve_args": list(args.serve_args),
        "pp_tok_s_mean": summary.get("PP_mean"),
        "pp_tok_s_min": summary.get("PP_min"),
        "tg_tok_s_mean": summary.get("TG_mean"),
        "tg_tok_s_min": summary.get("TG_min"),
        "output_sha1": summary.get("output_sha1"),
        "kv_total_pages": summary.get("kv_total_pages"),
        "vram_gib_mean": summary.get("vram_gib_mean"),
        "server_rss_gib_mean": summary.get("server_rss_gib_mean"),
        "expert_slots": None,
        "kv_pages": None,
        "kv_device_pages": None,
        "kv_ram_pages": None,
    }
    if log_path:
        try:
            record.update(parse_engine_log(Path(log_path).read_text(errors="replace")))
        except OSError as e:
            print(f"[bench] history: could not read server log {log_path}: {e}", flush=True)
    history.append_run(args.model, record)


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-40:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    t0 = time.monotonic()
    last_print = t0
    last_health_str = "connecting..."
    stall_s = float(os.environ.get("FREETOKEN_BENCH_BOOT_STALL_S", "180"))
    last_size, last_growth = -1, t0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        now = time.monotonic()
        size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        if size != last_size:
            last_size, last_growth = size, now
        elif stall_s > 0 and now - last_growth >= stall_s:
            # A booting server that logs nothing for minutes is hung (not slow): dump every
            # thread's stack into the log (faulthandler on SIGABRT) and fail loudly.
            os.killpg(proc.pid, signal.SIGABRT)
            time.sleep(2.0)  # let the dumps reach the pump before the log is read back
            die_with_log(
                f"BOOT STALL: server log silent for {stall_s:.0f}s; thread stacks dumped above",
                log_path,
            )
        if now - last_print >= 5.0:
            last_print = now
            print(
                f"[bench-watchdog] waiting for server readiness: {now - t0:.1f}s / {timeout:.0f}s (status: {last_health_str})",
                flush=True,
            )
        try:
            health = get_json(f"{origin}/health", timeout=3)
            last_health_str = str(health.get("maintenance") or health.get("status") or health)
        except (OSError, ValueError):
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            print(f"[bench-watchdog] server ready in {time.monotonic() - t0:.1f}s", flush=True)
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def wait_process_exit(proc) -> None:
    try:
        fd = os.pidfd_open(proc.pid)
    except ProcessLookupError:
        return
    try:
        import select

        select.select([fd], [], [])
    finally:
        os.close(fd)
    proc.wait()


def stop_server(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    wait_process_exit(proc)


def run_cold_repeats(args: argparse.Namespace, prompt: str) -> list[dict]:
    """Run each measured request in a new server process and cache."""
    rows: list[dict] = []
    for index in range(args.repeats):
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        tmp_dir = os.environ.get("TMPDIR", "/models/desenvolvimento/tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fd, log_path = tempfile.mkstemp(prefix="bench-pp-tg-", suffix=".log", dir=tmp_dir)
        cmd = serve_cmd(args, port)
        print(f"[bench] cold repeat {index + 1}/{args.repeats}: serve: {' '.join(cmd)}", flush=True)
        env = dict(os.environ)
        trace_path = None
        if args.token_trace:
            trace_path = str(Path(args.token_trace).resolve())
            if args.repeats > 1:
                trace_path = f"{trace_path}.{index + 1}"
            env["FREETOKEN_TOKEN_TRACE"] = trace_path
            print(f"[bench] token trace: {trace_path}", flush=True)
        if any("--spec-mtp" in value for value in args.serve_args):
            env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
        with os.fdopen(fd, "wb") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
            pump.start()
            try:
                wait_ready(origin, proc, log_path, args.server_timeout)
                model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
                row = one_run(origin, model_id, prompt, args, proc)
                row["log_path"] = log_path
                rows.append(row)
                print(
                    f"[bench] cold repeat {index + 1}/{args.repeats}: "
                    f"PP {row['prefill_tok_s']:.1f} TG {row['decode_tok_s']:.2f}",
                    flush=True,
                )
            finally:
                stop_server(proc)
                pump.join(timeout=10)
        if trace_path:
            trace_file = Path(trace_path)
            if not trace_file.is_file() or trace_file.stat().st_size == 0:
                die_with_log(f"token trace missing or empty: {trace_path}", log_path)
            try:
                records = [
                    json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()
                ]
            except (OSError, json.JSONDecodeError) as exc:
                die_with_log(f"invalid token trace {trace_path}: {exc}", log_path)
            if not records or not all(isinstance(item.get("kind"), str) for item in records):
                die_with_log(f"token trace has invalid schema: {trace_path}", log_path)
        print(f"[bench] cold repeat log: {log_path}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = (
        Path(args.prompt_file).read_text(errors="replace")
        if args.prompt_file_exact
        else build_prompt_text(args.model, args.prompt_file, args.tokens, args.prompt_offset)
    )
    if args.fresh_server_each_repeat:
        rows = run_cold_repeats(args, prompt)
        if not rows:
            return 1
        summary = {
            "label": args.label,
            "model": args.model,
            "n": len(rows),
            "prompt_tokens": rows[0]["prompt_tokens"],
            "PP_mean": mean(rows, "prefill_tok_s"),
            "PP_min": min(r["prefill_tok_s"] for r in rows),
            "TG_mean": mean(rows, "decode_tok_s"),
            "TG_min": min(r["decode_tok_s"] for r in rows),
            "TTFT_mean": mean(rows, "ttft_ms"),
            "itl_p50_mean": mean(rows, "itl_ms_p50"),
            "itl_p95_mean": mean(rows, "itl_ms_p95"),
            "vram_gib_mean": mean(rows, "vram_gib"),
            "gpu_util_mean": mean(rows, "gpu_util_mean"),
            "server_rss_gib_mean": mean(rows, "server_rss_gib"),
            "kv_total_pages": rows[-1]["kv_total_pages"],
            "output_sha1": rows[-1]["output_sha1"],
            "runs": rows,
        }
        print(f"\n==== [{args.label}] {summary['prompt_tokens']} tok / {args.decode} gen ====")
        print(f"  PP mean {summary['PP_mean']:9.1f} tok/s (min {summary['PP_min']:.1f})")
        print(f"  TG mean {summary['TG_mean']:9.2f} tok/s (min {summary['TG_min']:.2f})")
        if args.tg_curve:
            for i, tps in enumerate(rows[-1].get("tg_curve") or []):
                lo = i * args.tg_curve
                print(f"  TG curve [{lo}-{lo + args.tg_curve}): {tps:.2f}")
        print(f"  output hashes: {sorted({r['output_sha1'] for r in rows})}")
        if args.context_headroom:
            headroom = extract_context_headroom(rows[-1].get("log_path"), summary.get("TG_mean"))
            summary["context_headroom"] = headroom
            print_context_headroom(args.label, headroom)
        if args.json_out:
            with open(args.json_out, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary) + "\n")
        append_history(args, summary, log_path=rows[-1].get("log_path"))
        return 0
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    tmp_dir = os.environ.get("TMPDIR", "/models/desenvolvimento/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, log_path = tempfile.mkstemp(prefix="bench-pp-tg-", suffix=".log", dir=tmp_dir)
    cmd = serve_cmd(args, port)
    print(
        f"[bench] serve: {' '.join(cmd)}\n[bench] prompt: {args.tokens} tokens, log: {log_path}",
        flush=True,
    )

    rows: list[dict] = []
    env = dict(os.environ)
    if any("--spec-mtp" in a for a in args.serve_args):
        env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            card = get_json(f"{origin}/v1/stats").get("model", {})
            print(
                f"[bench] model_id={model_id} ctx={card.get('ctx')} attn={card.get('attn')} "
                f"moe={card.get('moe')}",
                flush=True,
            )
            for _ in range(args.warmups):
                stream_completion(origin, model_id, prompt, args, proc=proc)
            for i in range(args.repeats):
                row = one_run(origin, model_id, prompt, args, proc)
                rows.append(row)
                print(
                    f"[bench] run {i + 1}/{args.repeats}: PP {row['prefill_tok_s']:.1f}  "
                    f"TG {row['decode_tok_s']:.2f}  TTFT {row['ttft_ms']:.0f} ms  "
                    f"VRAM {row['vram_gib']:.2f} GiB",
                    flush=True,
                )
            if args.keep_alive:
                input("[bench] server up, press enter to stop: ")
        finally:
            if not args.keep_alive:
                stop_server(proc)
            pump.join(timeout=10)

    if not rows:
        return 1
    summary = {
        "label": args.label,
        "model": args.model,
        "n": len(rows),
        "prompt_tokens": rows[0]["prompt_tokens"],
        "PP_mean": mean(rows, "prefill_tok_s"),
        "PP_min": min(r["prefill_tok_s"] for r in rows),
        "TG_mean": mean(rows, "decode_tok_s"),
        "TG_min": min(r["decode_tok_s"] for r in rows),
        "TTFT_mean": mean(rows, "ttft_ms"),
        "itl_p50_mean": mean(rows, "itl_ms_p50"),
        "itl_p95_mean": mean(rows, "itl_ms_p95"),
        "vram_gib_mean": mean(rows, "vram_gib"),
        "gpu_util_mean": mean(rows, "gpu_util_mean"),
        "server_rss_gib_mean": mean(rows, "server_rss_gib"),
        "kv_total_pages": rows[-1]["kv_total_pages"],
        "output_sha1": rows[-1]["output_sha1"],
        "runs": rows,
    }
    print(f"\n==== [{args.label}] {summary['prompt_tokens']} tok / {args.decode} gen ====")
    print(f"  PP     mean {summary['PP_mean']:9.1f} tok/s   (min {summary['PP_min']:.1f})")
    print(f"  TG     mean {summary['TG_mean']:9.2f} tok/s   (min {summary['TG_min']:.2f})")
    if args.tg_curve:
        for i, tps in enumerate(rows[-1].get("tg_curve") or []):
            lo = i * args.tg_curve
            print(f"  TG curve [{lo}-{lo + args.tg_curve}): {tps:.2f}")
    print(
        f"  TTFT   mean {summary['TTFT_mean']:9.1f} ms    ITL p50 {summary['itl_p50_mean']:.2f} "
        f"/ p95 {summary['itl_p95_mean']:.2f} ms"
    )
    print(
        f"  VRAM   mean {summary['vram_gib_mean']:9.2f} GiB   GPU util {summary['gpu_util_mean']:.0f}%"
        f"   RSS {summary['server_rss_gib_mean']:.1f} GiB"
    )
    print(f"  KV pages {summary['kv_total_pages']}  output sha1 {summary['output_sha1']}")
    if args.context_headroom:
        headroom = extract_context_headroom(log_path, summary.get("TG_mean"))
        summary["context_headroom"] = headroom
        print_context_headroom(args.label, headroom)
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(summary) + "\n")
    append_history(args, summary, log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
