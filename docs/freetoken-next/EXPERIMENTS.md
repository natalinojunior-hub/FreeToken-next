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

## EXP-001b — Flash-Next 16K baseline, and the VRAM headroom wall
**Date:** 2026-09-16 · **Verdict:** **KEEP / PASS** at `--memory-ratio 0.86`; **INFORMATIONAL
failure** at the default 0.9
Setup: same harness, `--model /models/Qwen3.8-Flash-Next-NVFP4-Radix --tokens 16384
--decode 128 --repeats 3 --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive"`.
Attempt 1 and 2 (`--memory-ratio 0.9`, the default): the scheduler worker was killed while
building the ~66 GiB host expert bank (71 %, then 97 % of 192 shards) — host RAM starvation,
because `/tmp` is a 46 GiB tmpfs and 12 GiB of prior-project artifacts plus 17 GiB of
`pytest` temp had `Shmem` at 34 GiB (`MemAvailable` 53 GiB). After moving those off tmpfs
(`MemAvailable` 81.5 GiB) the load completed, then **CUDA OOM at 0.9** twice, at two
different sites: `torch.OutOfMemoryError … Tried to allocate 256.00 MiB` inside
`triton/testing.py:152 get_empty_cache_for_benchmark` reached from
`kernel/fla/chunk_fwd.py:391 chunk_gated_delta_rule_fwd_intra` autotuning, and
`… 192.00 MiB … 9.38 MiB is free` during warmup. So an unbudgeted *transient* (Triton autotune
scratch, graph capture) is what 0.9 leaves no room for — direct evidence for ARCHITECTURE.md §5.
Result at 0.86: **PP 1857.7 (min 1856.4), TG 28.685 (min 28.68)**, TTFT 8819 ms,
ITL p50/p95 34.73/37.71 ms, VRAM 14.86 GiB, GPU util 99.99 %, **server RSS 67.82 GiB**
(anchor ~66.5), KV 259 pages × page_size 64, output sha1 `f8bbaeb7e214`. Resolved config:
`attention_backend='qsa_sparse'` (page size forced to 64), `ple_backend='disk'`
(io_uring + O_DIRECT), `experts: nvfp4 via triton`, `moe_cache_size=1399`.
vs anchors (PP ~1532 / TG ~28.96 / VRAM 15.16 / RAM 66.5): **+21 % PP, −1.0 % TG (noise)**.

| Workload | PP | TG | TTFT | VRAM | RSS | guard |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4 16K | 4611.1 | 158.83 | 3553 ms | 14.98 GiB | 22.40 GiB | ≥4600 / ≥158 |
| Qwen3.8-Flash-Next NVFP4 16K (mr 0.86) | 1857.7 | 28.685 | 8819 ms | 14.86 GiB | 67.82 GiB | ≥1850 / ≥28.5 |


## EXP-002 — GGUF / MTP / TurboQuant corpus and source audits
**Date:** 2026-09-16 · **Verdict:** INFORMATIONAL (complete; reports archived in
`audits/A1…A6`, conclusions in ARCHITECTURE.md §2–§6)
Question: what does upstream already provide for (a) GGUF, (b) MTP, (c) KV quant backends,
(d) VRAM accounting — and what exactly are Turbo3/Turbo4/TCQ/VBR and qwen4exp MTP in the
reference implementation?
Method: six parallel read-only audits — this base (`A1`), llama-turbo-optimal code not
markdown (`A2`), MTP in both engines + GGUF/HF corpus inventory (`A3`), GGUF loader
feasibility (`A4`), FreeToken-Kai's 191-commit delta (`A5`), and the eight upstream PR refs
fetched as `refs/pr/*` (`A6`).
Headline answers: GGUF already loads natively (mmap, packed rows, vendored ggml MMQ/MMVQ/MoE
kernels, 19 decode type cases) but with **one** arch adapter (gemma4), 3 types wired in
Python, and **no shard joining**; **KV quantization does not exist** on the base (no
`--kv-cache-dtype`, `KV_CACHE_DTYPE_BYTES = 2`) but upstream PR #408 (a superset of #354)
defines the 13-seam recipe and an extensible `getattr(BackendInfo, "supports_{x}_kv")` gate;
**MTP is entirely absent** (`mtp.*` is dropped by the loaders, and tests assert that) while
the checkpoints carry 31 (qwen4_exp) / 19 (qwen3_5_moe) MTP tensors and the GDN verify kernel
(`disable_state_update`, `intermediate_states_buffer`, `retrieve_parent_token`) is already
present; **there is no VRAM ledger**, only an implicit `(1-memory_ratio)` remainder and two
independent consumers of one budget formula. Turbo3/4/TCQ/VBR were extracted to byte level
(block structs, `norm`-only scalar, normalized FWHT(128) + sign arrays, trellis 512/256 states
with a 9-bit decode window, per-(layer,side) VBR tiers over a VMM reservation that never
relocates), with `head_dim % 128 == 0` satisfied by our 256 and three ported determinism
oracles.

