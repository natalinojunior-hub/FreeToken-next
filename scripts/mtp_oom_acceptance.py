#!/usr/bin/env python3
"""Compare published RAW IDs with one-shot verify/row-commit OOM and a following request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.bench_pp_tg import DEFAULT_CORPUS, build_prompt_text  # noqa: E402


def install_fault(engine, stage):
    """Raise once after real GPU writes, without changing scheduler recovery logic."""
    event = {"stage": stage, "injections": 0}
    owner = engine if stage == "verify" else engine.linear_state_pool
    attribute = "forward_batch" if stage == "verify" else "commit_spec_row"
    original = getattr(owner, attribute)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        eligible = stage != "verify" or args[0].spec_logits_indices is not None
        if eligible and event["injections"] == 0:
            torch.cuda.synchronize(engine.device)
            event["injections"] += 1
            raise torch.OutOfMemoryError("Injected acceptance OOM: Tried to allocate 2.00 MiB")
        return result

    setattr(owner, attribute, wrapped)
    return event, lambda: setattr(owner, attribute, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", choices=("raw", "verify", "row_commit"), required=True)
    parser.add_argument("--reference", help="RAW JSON artifact from this harness")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--prompt-file", default=DEFAULT_CORPUS)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--attention-backend")
    parser.add_argument("--kv-format", choices=("auto", "bf16", "fp8", "turbo4", "turbo3"))
    parser.add_argument("--kv-ram-tier", dest="kv_tiering", choices=("off", "auto", "force"))
    parser.add_argument("--cache-type", choices=("naive", "radix", "hybrid_radix"))
    args = parser.parse_args()
    if args.stage != "raw" and not args.reference:
        parser.error("fault runs require --reference from a RAW run")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.message import DetokenizeMsg

    os.environ["FREETOKEN_MTP_FORCE_DEPTH"] = str(0 if args.stage == "raw" else args.depth)
    os.environ["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    report = {
        "model": str(Path(args.model).resolve()),
        "stage": args.stage,
        "tokens": args.tokens,
        "decode": args.decode,
        "prompt_offset": args.prompt_offset,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "diff", "--name-only"], text=True)),
        "rows": [],
        "passed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    restore = lambda: None
    try:
        prompt = build_prompt_text(args.model, args.prompt_file, args.tokens, args.prompt_offset)
        report["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        reference = json.loads(Path(args.reference).read_text()) if args.reference else None
        if reference is not None:
            assert reference["passed"] and reference["stage"] == "raw"
            for key in ("model", "tokens", "decode", "prompt_offset", "prompt_sha256"):
                assert report[key] == reference[key], (key, report[key], reference[key])
        llm = LLM(
            model_path=args.model,
            dtype=torch.bfloat16,
            max_seq_len_override=max(16704, args.tokens + args.decode + 64),
            spec_mtp=0 if args.stage == "raw" else args.depth,
            **{
                name: getattr(args, name)
                for name in ("attention_backend", "kv_format", "kv_tiering", "cache_type")
                if getattr(args, name) is not None
            },
        )
        report["resolved"] = {
            name: getattr(llm.engine.config, name, None)
            for name in (
                "attention_backend",
                "kv_format",
                "kv_tiering",
                "kv_ram_dtype",
                "kv_ram_resolved_dtype",
                "max_extend_tokens",
            )
        }
        report["resolved"].update(
            attention_class=type(llm.engine.attn_backend).__name__,
            kv_pool_class=type(llm.engine.kv_cache).__name__,
            cache_type=llm.config.cache_type,
            prefix_cache_class=type(llm.cache_manager.prefix_cache).__name__,
            host_pages=int(getattr(llm.engine, "host_pages", 0)),
        )
        report["resolved"] = {
            key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in report["resolved"].items()
        }
        if reference is not None:
            for key, value in report["resolved"].items():
                if key in ("max_extend_tokens", "host_pages"):
                    continue  # pool capacities can differ when RAW does not load a draft
                assert value == reference.get("resolved", {}).get(key), ("runtime differs", key)
            assert bool(report["resolved"]["host_pages"]) == bool(
                reference.get("resolved", {}).get("host_pages")
            ), "reference KV tier differs"
        ids = llm.tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) == args.tokens, (len(ids), args.tokens)
        if args.stage != "raw":
            event, restore = install_fault(llm.engine, args.stage)
            report["fault"] = event
        for index in range(2):
            published = []
            send = llm.send_result

            def record(messages):
                for msg in messages:
                    if isinstance(msg, DetokenizeMsg) and not (
                        msg.finished and msg.next_token in llm.eos_token_ids
                    ):
                        published.append(int(msg.next_token))
                send(messages)

            llm.send_result = record
            try:
                result = llm.generate(
                    [ids], SamplingParams(temperature=0.0, max_tokens=args.decode, ignore_eos=True)
                )[0]
            finally:
                llm.send_result = send
            report["rows"].append(
                {"label": "fault" if index == 0 else "following", "token_ids": published}
            )
            assert published == result["token_ids"]
            assert len(published) == args.decode, len(published)
            if args.stage != "raw" and index == 0:
                assert report["fault"]["injections"] == 1, "first request never reached fault site"
            if reference is not None:
                assert published == reference["rows"][index]["token_ids"], (
                    f"ID divergence request {index}"
                )
            llm.cache_manager.check_integrity()
        if args.stage != "raw":
            assert report["fault"]["injections"] == 1, "requested fault site was not reached"
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore()
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"acceptance artifact: {output}", flush=True)


if __name__ == "__main__":
    main()
