#!/usr/bin/env python3
"""Exercise an exact long-context request before, during, and after GPU pressure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


MODEL = "/models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS"
ARCHIVE = Path("/models/desenvolvimento/old/freetoken-next/external/ft-campaign2")
PROMPT_BUILDER = Path.cwd() / "benchmarks/bench_pp_tg.py"
PROMPT_FILE = ARCHIVE / "campaign26/prompt-470k.txt"


def build_prompt(model: str, prompt_file: str, tokens: int) -> str:
    spec = importlib.util.spec_from_file_location("campaign_bench_pp_tg", PROMPT_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import archived prompt builder: {PROMPT_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_prompt_text(model, prompt_file, tokens, 0)


def git_sha(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rss_bytes() -> int | None:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # ru_maxrss is a high-water mark (KiB on Linux, bytes on macOS).
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform.startswith("linux") else 1))


def nvml_memory() -> dict[str, int] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return {
                "total_bytes": int(mem.total),
                "free_bytes": int(mem.free),
                "used_bytes": int(mem.used),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # optional: torch still reports driver-visible physical free bytes
        return None


def resources() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_driver_free_bytes": int(free),
        "torch_driver_total_bytes": int(total),
        "nvml": nvml_memory(),
        "process_rss_bytes": rss_bytes(),
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1024 if sys.platform.startswith("linux") else 1)
        ),
    }


class PeakSampler(threading.Thread):
    """Sample this process's RSS and the visible GPU's NVML usage during generation."""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak: dict[str, int | None] = {
            "process_rss_bytes": 0,
            "nvml_used_bytes": None,
            "nvml_free_bytes": None,
        }
        self.samples = 0

    def run(self) -> None:
        pynvml = handle = None
        nvml_initialized = False
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            nvml_initialized = True
            pynvml = nvml
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pynvml = None
        try:
            while not self.stop_event.is_set():
                self.samples += 1
                rss = rss_bytes()
                if rss is not None:
                    self.peak["process_rss_bytes"] = max(self.peak["process_rss_bytes"] or 0, rss)
                if pynvml is not None and handle is not None:
                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.peak["nvml_used_bytes"] = max(
                            self.peak["nvml_used_bytes"] or 0, int(mem.used)
                        )
                        free = int(mem.free)
                        prior = self.peak["nvml_free_bytes"]
                        self.peak["nvml_free_bytes"] = free if prior is None else min(prior, free)
                    except Exception:
                        pynvml = None
                self.stop_event.wait(self.interval)
        finally:
            if nvml_initialized:
                try:
                    import pynvml as nvml

                    nvml.nvmlShutdown()
                except Exception:
                    pass

    def stop(self) -> dict:
        self.stop_event.set()
        self.join(timeout=5)
        return {**self.peak, "samples": self.samples, "interval_seconds": self.interval}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def graph_counts(llm) -> dict | None:
    runner = getattr(llm.engine, "graph_runner", None)
    if runner is None:
        return None
    return {
        "decode": len(getattr(runner, "graph_map", {})),
        "verify": len(getattr(runner, "verify_graphs", {})),
        "draft": len(getattr(runner, "drafts", {})),
    }


def pool_state(scheduler) -> dict:
    engine = scheduler.engine
    cache = engine.moe_offload_cache
    linear = scheduler.cache_manager.linear_state_pool
    return {
        "kv_pages": int(engine.num_pages),
        "kv_page_size": int(scheduler.cache_manager.page_size),
        "gdn_slots": int(linear.num_slots) if linear is not None else 0,
        "expert_resident_rows": int(cache.resident_rows) if cache is not None else 0,
        "expert_pool_caps": list(cache.live_caps) if cache is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokens", type=int, default=261824)
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--spec-mtp", type=int, choices=(0, 4), default=None)
    parser.add_argument("--prompt-file", default=str(PROMPT_FILE))
    parser.add_argument("--runtime-repo", default=str(Path.cwd()))
    args = parser.parse_args()
    if args.tokens + args.decode > 262144:
        parser.error("prompt plus decode must fit the 262144-token context")
    if not torch.cuda.is_available():
        parser.error("this acceptance run requires CUDA")

    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.message import DetokenizeMsg

    prompt = build_prompt(args.model, args.prompt_file, args.tokens)
    max_context = 16704 if args.tokens == 16384 else 262144
    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        max_seq_len_override=max_context,
        spec_mtp=args.spec_mtp,
    )
    # Offline LLM accepts token-id lists; tokenize without added special tokens so the
    # archived fixed-point prompt stays exactly the requested length.
    prompt_ids = llm.tokenizer.encode(prompt, add_special_tokens=False)
    assert len(prompt_ids) == args.tokens, (len(prompt_ids), args.tokens)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()

    def run(label: str) -> dict:
        token_stamps = []
        send_result = llm.send_result
        sampler = PeakSampler()
        sampler.start()
        started = time.perf_counter()
        try:

            def record_result(messages):
                for msg in messages:
                    if isinstance(msg, DetokenizeMsg) and not (
                        msg.finished and msg.next_token in llm.eos_token_ids
                    ):
                        token_stamps.append(time.perf_counter())
                send_result(messages)

            llm.send_result = record_result
            result = llm.generate(
                [prompt_ids], SamplingParams(temperature=0.0, max_tokens=args.decode)
            )[0]
        finally:
            llm.send_result = send_result
            sampled_peaks = sampler.stop()
        finished = time.perf_counter()
        tokens = list(result["token_ids"])
        assert len(token_stamps) == len(tokens), (len(token_stamps), len(tokens))
        ttft = token_stamps[0] - started if token_stamps else None
        tg_elapsed = token_stamps[-1] - token_stamps[0] if len(token_stamps) > 1 else 0.0
        itls = [b - a for a, b in zip(token_stamps, token_stamps[1:])]
        row = {
            "label": label,
            "output_tokens": len(tokens),
            "text": str(result["text"]),
            "output_sha1": hashlib.sha1(str(result["text"]).encode()).hexdigest(),
            "token_ids": tokens,
            "timing": {
                "wall_seconds": finished - started,
                "ttft_seconds": ttft,
                "prompt_tokens_per_ttft_including_prefix_reuse": (
                    args.tokens / ttft if ttft is not None and ttft > 0 else None
                ),
                "committed_tg_tokens_per_second": (
                    (len(tokens) - 1) / tg_elapsed if tg_elapsed > 0 else None
                ),
                "itl_p50_seconds": percentile(itls, 0.50),
                "itl_p95_seconds": percentile(itls, 0.95),
            },
            "pools": pool_state(llm),
            "resources": resources(),
            "peak_resources_during_generate": sampled_peaks,
            "graphs": graph_counts(llm),
            "expert_stats": None,
        }
        llm.cache_manager.check_integrity()  # idle-only: assert cache and GDN conservation
        return row

    before = run("baseline")
    baseline_geometry = (before["pools"]["kv_pages"], before["pools"]["gdn_slots"])

    hog = None
    pressure_error = None
    try:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        hog_bytes = int(free) - (128 << 20)
        if hog_bytes <= 0:
            raise RuntimeError(f"insufficient free VRAM to establish pressure: {free} bytes")
        hog = torch.empty(hog_bytes, dtype=torch.uint8, device="cuda")
        hog.zero_()  # touch pages so the driver commits the physical allocation
        torch.cuda.synchronize()
        actual_free, _ = torch.cuda.mem_get_info()
        assert actual_free < (256 << 20), ("pressure not established", actual_free)
        under_pressure = run("under_pressure")
    except Exception as exc:
        pressure_error = repr(exc)
        under_pressure = None
    finally:
        if hog is not None:
            del hog
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    recovered = run("recovered")
    rows = [before, under_pressure, recovered]
    report = {
        "runtime_sha": git_sha(args.runtime_repo),
        "model": args.model,
        "prompt_file": args.prompt_file,
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": prompt_sha,
        "config": {
            "dtype": "bfloat16",
            "max_seq_len_override": max_context,
            "max_running_req": llm.config.max_running_req,
            "cuda_graph_max_bs": llm.config.cuda_graph_max_bs,
            "spec_mtp": llm.config.spec_mtp,
            "moe_strategy": llm.config.moe_strategy,
            "moe_cache_auto": llm.config.moe_cache_auto,
            "memory_ratio": "default",
            "explicit_pool_caps": None,
        },
        "pressure_error": pressure_error,
        "runs": rows,
    }
    print(json.dumps(report, ensure_ascii=False))

    assert pressure_error is None, pressure_error
    assert under_pressure is not None
    assert all(row["output_tokens"] == args.decode for row in rows), [
        row["output_tokens"] for row in rows
    ]
    assert before["token_ids"] == under_pressure["token_ids"] == recovered["token_ids"], (
        "greedy output changed under physical-memory pressure"
    )
    assert (
        recovered["pools"]["expert_resident_rows"] > under_pressure["pools"]["expert_resident_rows"]
    ), "physical pressure did not reduce resident expert rows"
    assert all(
        (row["pools"]["kv_pages"], row["pools"]["gdn_slots"]) == baseline_geometry for row in rows
    ), "KV or GDN pool geometry changed during pressure/recovery"


if __name__ == "__main__":
    main()
