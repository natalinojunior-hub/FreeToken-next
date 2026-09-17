# EXPERIMENTS — freetoken-next

Append-only log. Every entry: question → setup (exactly reproducible) → result →
verdict (KEEP / LOW_GAIN_SURVIVOR / REJECT / INFORMATIONAL).

## EXP-000 — Establish lineage and host provenance
**Date:** 2026-09-16 · **Verdict:** INFORMATIONAL (complete)
Question: is `next` current upstream FreeToken, and with what toolchain will we measure?
Method: `git fetch --tags upstream`, `git rev-parse HEAD upstream/main`,
`nvcc --version`, `nvidia-smi --query-gpu=...`, `uv`/`pyvenv.cfg` inspection,
`git log af71ba432..HEAD`.
Result: `HEAD == upstream/main == cac247a (v0.1.3)`, 0 behind; anchor build is
`0.1.2+gaf71ba432` = 23 commits behind; SM120 / driver 610.57.04 / torch 2.11.0+cu130 /
triton 3.6.0 / py3.12.14. Working tree had to be restored with `git checkout -f next`
(clone left it empty). Written to PROVENANCE.md.

## EXP-001 — Phase 1 baseline on v0.1.3
**Date:** 2026-09-16 · **Verdict:** **KEEP / PASS** (baseline guards established)
Question: does `v0.1.3` reproduce the 0.1.2 anchors (Flash PP ~1532 / TG ~28.96;
35B-A3B PP ~4104 / TG ~147.1) at 16K within noise?
Setup: `.venv/bin/python benchmarks/bench_pp_tg.py --model <ckpt> --tokens 16384 --decode 128
--repeats 3 --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive"` (greedy,
bs=1, one spawned server per row).
Result 35B-A3B: **PP 4611.1 (min 4605.7), TG 158.83 (min 158.78)**, TTFT 3553 ms,
ITL p50/p95 6.16/6.40 ms, VRAM 14.98 GiB, RSS 22.40 GiB, GPU util 99.8 %,
output sha1 `2a6dca88ffdc`. Repeat spread 0.12 % PP / 0.03 % TG.
Two findings surfaced *by* the run:
1. With default auto-sizing the 16K prompt was **rejected** — `--moe-cache-auto` chose
   6102 expert slots and only 8268 KV tokens, so the anchor configuration must have set KV
   explicitly. Any long-context work has to re-derive that split, not inherit it.
2. The radix prefix cache silently collapses a repeat measurement to `#new-token: 64`
   (16320 cached) — PP would read ~10⁵ tok/s. `--cache-type naive` is mandatory for PP rows.
Raw rows: `docs/freetoken-next/pp_tg.jsonl`.

## EXP-001b — Flash-Next 16K baseline
**Date:** 2026-09-16 · **Verdict:** RUNNING (66 GiB RAM load).

## EXP-002 — GGUF / MTP / TurboQuant corpus and source audits
**Date:** 2026-09-16 · **Verdict:** PENDING (four parallel audits running)
Question: what does upstream already provide for (a) GGUF, (b) MTP, (c) KV quant
backends, (d) VRAM accounting; and what exactly are Turbo3/Turbo4/TCQ/VBR and qwen4exp MTP
in the reference implementation?
Method: read-only source audits of `python/freetoken/`,
`/models/servers/llama-turbo-optimal` (code, not its markdown), and GGUF metadata dumps
of the local corpus. Findings land in ARCHITECTURE.md; only the durable conclusions are
kept here.
