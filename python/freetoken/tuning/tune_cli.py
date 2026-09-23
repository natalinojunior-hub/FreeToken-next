"""``ft tune``: boot the v1 candidate matrix against a real server, measure cold PP /
committed TG / peak VRAM, and persist the winner (see ``profile.py`` for the schema/key,
``candidates.py`` for the matrix + selection rule).

    ft tune --model <path> --ctx 16384              # measure + write the profile
    ft tune --model <path> --ctx 16384 --dry-run     # print the candidate matrix + key only

Each candidate is a short, owned-PID ``ft serve`` boot (event-driven readiness via
``/health``, killed by PID -- never by name) on a free port, 1 warmup + N reps (default 3)
of: a unique cold prompt (``cache_prompt: false``, PP measured over processed tokens only,
matching ft-campaign2/coldclient.py) then a fixed-length greedy decode (committed TG).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict

from freetoken.tuning.candidates import CandidateResult, build_candidates, select_best
from freetoken.tuning.profile import (
    CandidateEvidence,
    Profile,
    TunedSettings,
    compute_key,
    save,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _draft_graph_available() -> bool:
    try:
        from freetoken.engine import graph

        return hasattr(graph, "DRAFT_GRAPH_ENV")
    except ImportError:
        return False


def _hybrid_capable(model_path: str) -> bool:
    """Whether this checkpoint's experts have a CPU MoE weight path at all -- reuses the
    engine's own viability predicate rather than re-deriving it."""
    try:
        import torch

        from freetoken.distributed import DistributedInfo
        from freetoken.engine.config import EngineConfig
        from freetoken.engine.engine import _cpu_moe_executor_viable

        config = EngineConfig(
            model_path=model_path,
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.bfloat16,
            attention_backend="fi",
        )
        model_config = config.model_config  # cached_property: parses the checkpoint's config
        return bool(getattr(model_config, "is_moe", False)) and _cpu_moe_executor_viable(
            model_config
        )
    except Exception as e:  # noqa: BLE001 -- unknown model config -> assume not hybrid-capable
        logger.info(f"ft tune: could not probe hybrid capability for {model_path!r}: {e}")
        return False


def _settings_to_argv(
    settings: dict, *, port: int, model: str, ctx: int, kv_format: str
) -> tuple[list[str], dict]:
    """(``ft serve`` argv, extra env) for one candidate. --max-running-requests 1 on every
    candidate (not just spec_mtp>0) so k0/k1 differ only in the tunable being measured --
    matches ft-campaign2/bench.sh's single-request shape. --cuda-graph-max-bs 1 additionally
    whenever spec_mtp > 0 (HARDWARE_TUNING.md's requirement, not auto-applied anywhere in the
    engine -- see D0-autoconfig-audit.md)."""
    argv = [
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len-override",
        str(ctx),
        "--kv-format",
        kv_format,
        "--moe-strategy",
        settings["moe_strategy"],
        "--spec-mtp",
        str(settings["spec_mtp"]),
        "--max-running-requests",
        "1",
        # every candidate, so MTP on/off differ only in the tunable being measured
        "--cuda-graph-max-bs",
        "1",
    ]
    env = dict(os.environ)
    env["FREETOKEN_SPEC_DEFER_REPLAY"] = "1" if settings["defer_replay"] else "0"
    env["FREETOKEN_DRAFT_GRAPH"] = "1" if settings["draft_graph"] else "0"
    return argv, env


def _wait_for_health(proc: subprocess.Popen, url: str, timeout_s: float = 1800) -> bool:
    """Event-driven readiness: poll /health until "serving", the process dies, or timeout.
    No fixed sleep-and-hope -- checks the PID every pass so a crash is reported immediately
    instead of spinning to the timeout (see AGENTS.md's wait-for-server.sh contract)."""
    t0 = time.monotonic()
    while True:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                body = json.loads(resp.read())
            if body.get("maintenance") == "serving":
                return True
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError):
            pass
        if time.monotonic() - t0 > timeout_s:
            return False
        time.sleep(2)


def _peak_vram_mib(pid: int) -> float:
    """Peak VRAM (MiB) across ``pid``'s own process tree, via per-PID nvidia-smi compute-apps
    accounting -- not whole-GPU memory.used, which is wrong on a GPU shared with other jobs."""
    try:
        children = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.split()
        pids = {str(pid), *children}
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        total = 0.0
        for line in out.strip().splitlines():
            p, used = (x.strip() for x in line.split(","))
            if p in pids:
                total += float(used)
        return total
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def _model_id(url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as resp:
            return json.loads(resp.read())["data"][0]["id"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError):
        return None


def _unique_prompt(base_text: str, rep: int) -> str:
    # A nonce prefix, not a repetition of base_text -- MTP's draft accept rate rides on how
    # predictable the continuation is, and a self-repeating prompt biases every candidate's
    # committed TG toward whichever one accepts drafts best on repetition, not on real text.
    # Deterministic per rep: every candidate is a fresh boot (cold anyway), and rep N must
    # decode the same text on every candidate or MTP acceptance/expert misses swing with it.
    nonce = hashlib.sha256(f"ft-tune:{rep}".encode()).hexdigest()[:16]
    return f"[{nonce}] {base_text}"


def _measure_one_rep(url: str, model_id: str, prompt: str, decode_tokens: int) -> dict | None:
    """Streaming completion: TTFT = time to the first chunk, PP = processed prompt tokens /
    TTFT, committed TG = (completion_tokens - 1) / (t_last - t_first) -- mirrors
    ft-campaign2/coldclient.py's math, not a whole-request-wall-clock average of the two."""
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_start = time.monotonic()
    t_first: float | None = None
    t_last: float | None = None
    completion_tokens = 0
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                now = time.monotonic()
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("text") or choices[0].get("finish_reason")):
                    if t_first is None:
                        t_first = now
                    completion_tokens += 1
                    t_last = now
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    if t_first is None:
        return None
    ttft = max(1e-6, t_first - t_start)
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    processed = max(1, prompt_tokens - cached_tokens)
    decode_span = max(1e-6, (t_last or t_first) - t_first)
    return {
        "cold_pp": processed / ttft,
        "committed_tg": max(0.0, (completion_tokens - 1) / decode_span),
        "ttft_s": ttft,
    }


def run_candidate(
    settings: dict,
    *,
    model: str,
    ctx: int,
    kv_format: str,
    reps: int,
    decode_tokens: int,
    prompt_text: str,
    log_dir: str,
) -> CandidateResult:
    port = _free_port()
    argv, env = _settings_to_argv(settings, port=port, model=model, ctx=ctx, kv_format=kv_format)
    label = "-".join(f"{k}{v}" for k, v in sorted(settings.items()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"ft-tune-{label}-{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "freetoken.cli", "serve", *argv],
        env=env,
        start_new_session=True,  # own process group -> kill only this candidate's tree
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    peak_vram = 0.0
    stop_poll = False

    def _poll_vram():
        nonlocal peak_vram
        while not stop_poll and proc.poll() is None:
            peak_vram = max(peak_vram, _peak_vram_mib(proc.pid))
            time.sleep(0.25)

    import threading

    poller = threading.Thread(target=_poll_vram, daemon=True)
    poller.start()
    had_traceback = False
    result_kwargs: dict = {}
    try:
        url = f"http://127.0.0.1:{port}"
        if not _wait_for_health(proc, url):
            had_traceback = True
        else:
            model_id = _model_id(url) or "default"
            samples = []
            for i in range(1 + reps):  # 1 warmup + reps
                prompt = _unique_prompt(prompt_text, i)
                r = _measure_one_rep(url, model_id, prompt, decode_tokens)
                if r is None:
                    had_traceback = True
                    break
                if i > 0:  # discard warmup
                    samples.append(r)
            if samples:
                samples.sort(key=lambda r: r["committed_tg"])
                median = samples[len(samples) // 2]
                result_kwargs = {
                    "cold_pp": sorted(s["cold_pp"] for s in samples)[len(samples) // 2],
                    "committed_tg": median["committed_tg"],
                    "ttft_s": median["ttft_s"],
                }
    finally:
        stop_poll = True
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        poller.join(timeout=5)
        log_file.close()
        with open(log_path) as f:
            if "Traceback" in f.read():
                had_traceback = True

    return CandidateResult(
        settings,
        cold_pp=result_kwargs.get("cold_pp"),
        committed_tg=result_kwargs.get("committed_tg"),
        ttft_s=result_kwargs.get("ttft_s"),
        peak_vram_mib=peak_vram or None,
        had_traceback=had_traceback,
    )


def main(argv: list[str] | None = None, prog: str = "ft tune") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint path to tune")
    p.add_argument("--ctx", type=int, default=16384, help="max-seq-len-override to tune at")
    p.add_argument("--kv-format", default="turbo3")
    p.add_argument("--reps", type=int, default=3, help="measured reps per candidate (+1 warmup)")
    p.add_argument("--decode-tokens", type=int, default=256, help="TG decode length")
    p.add_argument("--gpu-uuid", default=None, help="override the GPU uuid in the profile key")
    p.add_argument(
        "--prompt-file",
        default=None,
        help="representative prompt filling most of --ctx: the profile is keyed by the context "
        "bucket, and MTP's win depends on occupancy and content (default: a synthetic "
        "prompt of about 85%% of --ctx, repetitive, so it overstates MTP acceptance)",
    )
    p.add_argument(
        "--log-dir",
        default=os.environ.get("TMPDIR", "/tmp"),
        help="dir for each candidate's ft serve log (default: $TMPDIR)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the candidate matrix and key, don't launch"
    )
    p.add_argument("-o", "--out", default=None, help="profile path override")
    ns = p.parse_args(argv)

    draft_graph_available = _draft_graph_available()
    hybrid_capable = _hybrid_capable(ns.model)
    candidates = build_candidates(
        draft_graph_available=draft_graph_available, hybrid_capable=hybrid_capable
    )

    gpu_uuid = ns.gpu_uuid
    if gpu_uuid is None:
        try:
            # NVML, not torch.cuda: this process also drives ft-serve subprocesses on the
            # same GPU, so it must never hold its own CUDA context (less free VRAM ->
            # smaller auto-sized expert cache -> a biased TG measurement).
            from freetoken.gpu_select import _nvml_uuids

            uuids = _nvml_uuids()
            gpu_uuid = uuids[0] if uuids else "unknown"
        except Exception:  # noqa: BLE001 -- --dry-run must work with no NVML either
            gpu_uuid = "unknown"
    key = compute_key(
        gpu_uuid=gpu_uuid, model_path=ns.model, kv_format=ns.kv_format, max_seq_len=ns.ctx
    )

    if ns.dry_run:
        print(f"key: {key}")
        print(f"draft_graph_available: {draft_graph_available}  hybrid_capable: {hybrid_capable}")
        for c in candidates:
            print(json.dumps(c))
        return 0

    if ns.prompt_file:
        with open(ns.prompt_file) as f:
            prompt_text = f.read()
    else:
        # ~10 tokens per sentence
        prompt_text = "The quick brown fox jumps over the lazy dog. " * max(
            1, int(ns.ctx * 0.85) // 10
        )

    results = [
        run_candidate(
            c,
            model=ns.model,
            ctx=ns.ctx,
            kv_format=ns.kv_format,
            reps=ns.reps,
            decode_tokens=ns.decode_tokens,
            prompt_text=prompt_text,
            log_dir=ns.log_dir,
        )
        for c in candidates
    ]
    best = select_best(results)
    if best is None:
        print("ft tune: no candidate produced a usable measurement; not writing a profile")
        return 1

    profile = Profile(
        key=key,
        chosen=TunedSettings(
            spec_mtp=best.settings["spec_mtp"],
            defer_replay=best.settings["defer_replay"],
            draft_graph=best.settings["draft_graph"],
            moe_strategy=best.settings["moe_strategy"],
        ),
        evidence=CandidateEvidence(
            cold_pp=best.cold_pp,
            committed_tg=best.committed_tg,
            ttft_s=best.ttft_s,
            peak_vram_mib=best.peak_vram_mib,
            date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
        candidates=[
            {
                "settings": r.settings,
                "result": {k: v for k, v in asdict(r).items() if k != "settings"},
            }
            for r in results
        ],
    )
    dest = save(profile, ns.out)
    print(f"ft tune: wrote {dest}")
    print(json.dumps(profile.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
