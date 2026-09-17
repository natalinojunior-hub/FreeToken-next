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


## EXP-003 — First native GGUF load attempt (port in flight, uncommitted tree)
**Date:** 2026-09-17 · **Verdict:** INFORMATIONAL, load path works; capacity config does not
`ft serve --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf --memory-ratio 0.86
--num-tokens 8192 --max-seq-len-override 8300 --cache-type naive` reached serving with
14.96 GiB PyTorch-allocated and accepted requests — the IQ3_S/IQ4_XS/Q4_K/Q2_S packed tables
map and load. Then `torch.OutOfMemoryError: Tried to allocate 142.00 MiB … 131.38 MiB is
free` inside `kernel/fla/chunk_fwd.py` → `chunk_gated_delta_rule_fwd_h` →
`k.new_empty(B, NT, H, V, K)`: the **GDN prefill transient** for an 8192-token chunk, on a
dense model whose packed weights are fully resident. Same class as EXP-001b: the planner
reserves nothing for the transient (`ARCHITECTURE.md` §5). Corrected probe:
`--memory-ratio 0.8 --num-tokens 4096 --max-seq-len-override 4300 --max-prefill-length 1024`.
Suite against this tree: 3 failed / 1809 passed / 206 skipped — missing
`kernel/aot_models.py` arch entries for the new GGUF arches, an incomplete `__init__.py`
export union, and `test_nvfp4_backends::test_b12x_decode_matches_dequant_reference` which
must first be shown to be a regression rather than flashinfer JIT flake.

## EXP-004 — First native GGUF served correctly (phase 2, gate met)
**Date:** 2026-09-17 · **Verdict:** **KEEP / PASS** (required capability)
`.venv/bin/ft serve --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf
--max-running-requests 1 --memory-ratio 0.8 --num-tokens 4096 --max-seq-len-override 4300
--max-prefill-length 1024 --cuda-graph-max-bs 0 --cache-type naive`
`ft ctl generate "Q: Why is a KV cache needed during autoregressive decoding? A:"
--max-tokens 48` → *"Because each new token must attend to all previous tokens, and
recomputing their representations from scratch at every step would be prohibitively
expensive. The KV cache stores the key and value projections of all previously generated
tokens, so that at step…"* — coherent **and** factually right, so the IQ3_S / IQ4_XS / Q4_K /
Q2_S packed tables are being read by the real kernels, not merely mapped.
Capacity shape that made it work (EXP-003's failure mode, fixed by configuration not code):
eager decode (`--cuda-graph-max-bs 0`) removes graph capture, and the 1024-token chunk cuts
the GDN transient ~8x. Two side observations: the same tree also answered a 1001-token
`/v1/completions` with HTTP 200, and `models/gguf/reader.py:264` raises a NumPy
"not writable" UserWarning on the mmap path (harmless today, worth a `np.ascontiguousarray`
or `flags.writeable` decision rather than a suppressed warning).
Still owed before this is committable: `kernel/aot_models.py` entries for the new GGUF
architectures, the `models/*/__init__.py` export union, `kernel/gguf.py::_module` load order
(the bundled `libcudart.so.13` wins the soname race and shadows torch's runtime — proven
pre-existing at HEAD, not a port regression), and the K-quant CPU path must refuse at
*registration* instead of registering `weight_format=q8_0` and raising `unknown
weight_format` at call time.

## EXP-005 — Close the dirty dummy-page brick (Phase 0)
**Date:** 2026-09-17 · **Verdict:** **KEEP** · commit `1c81064`
Question: the working tree carried a half-finished fix for the fact that every KV pool
allocates `num_pages + 1` (the dummy page padded writes go into) while both budget formulas
priced only usable pages. Was the fix complete, and did it cost anything?
Method: finish the accounting at the one shared seam (`pool_pages()` / `required_bytes()`) and
extend it to the second copy of the bug in `kvcache/base.py::solve_num_pages` (DSV4's own
solver already subtracted the dummy page — the generic template did not); pin the invariant in
tests both ways (`required_bytes(plan) <= budget` and `required_bytes(plan + 1 page) > budget`)
for the planner, the generic solver and `validate_rebuild`; then re-run the 35B-A3B 16K guard.
Result: focused suites green, full `not slow` suite **1813 passed / 206 skipped / 1 failed**
with the failure being the known flashinfer b12x JIT race (`moe/test_nvfp4_backends.py`,
2 passed when run alone — EXP-001 saw the same class on `test_mrope.py`). Guard:
**PP 4610.1 (min 4606.8) / TG 158.75 (min 158.72) / VRAM 14.98 GiB / output sha1
`2a6dca88ffdc`**, i.e. byte-identical generation and both numbers inside the guard
(≥ 4600 / ≥ 158) against the 4611.1 / 158.83 baseline.
Note for every future A/B: the size of the miss was one page, so this brick changes nothing you
can see in throughput -- what it buys is that a plan can no longer promise memory it did not
fund, which is the precondition for the ledger's `pool_budget_bytes` being worth trusting.

## EXP-006 — Authoritative VRAM ledger, brick 1 (Phase 1)
**Date:** 2026-09-17 · **Verdict:** **KEEP** (guards hold; the hidden margin is now an account)
Question: can the engine stop guessing `--memory-ratio`, and does the account it prints
actually add up on real hardware?
Method: `engine/vram_ledger.py` (D-014) — named `Charge(name, bytes, Kind)` lines,
`reserve_bytes` = what must stay empty (Triton autotune arena 256 MiB, CUDA-graph capture peak
256+160/shape MiB, one GDN layer's prefill workspace over `max_extend_tokens`, the live
activation stream, a per-image vision transient, 128 MiB named fragmentation reserve),
`engine_overhead_bytes` = held bytes no pool/expert cost model prices (page table, graph pool,
backend workspaces, PLE), and `ceiling_bytes = ratio x baseline - max(0, reserve - the hole the
ratio already leaves)`. The KV solve, `--moe-cache-auto` and `validate_rebuild` all take those
two numbers; `_calibrate_vram_ledger()` closes the account against
`torch.cuda.memory_allocated` and warns instead of asserting. Inputs from audit A9 (per-consumer
file:line inventory) and the two anchor models.
Result — guards at the default `--memory-ratio 0.9`, three repeats each, same harness:

| Model | PP | TG | TTFT | VRAM | sha1 | guard |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4 16K | 4607.6 (min 4606.8) | 158.49 | 3555.9 ms | 14.59 GiB | `2a6dca88ffdc` | ≥4600 / ≥158 **PASS** |
| Qwen3.8-Flash-Next NVFP4 16K | 1861.9 (min 1860.5) | 28.68 | 8799.9 ms | 14.80 GiB | `f8bbaeb7e214` | ≥1850 / ≥28.5 **PASS** |

Both hashes are the pre-ledger anchors' hashes; the deltas (-0.08 % PP on the 35B, +0.2 % on
Flash) are inside the harness's own repeat spread. **The capability result is that Flash-Next
now serves at 0.9, the default** — EXP-001b's exact failure ("Tried to allocate 256.00 MiB …
209.44 MiB free" inside `get_empty_cache_for_benchmark`) is gone, because the autotune arena,
the graph pool and the GDN prefill workspace are funded lines instead of a hope. At
`--memory-ratio 1.0` the ceiling lands on 12.923 GiB and the plan is the *same geometry* the
hand-found 0.86 gave (1113 expert slots, 259 KV pages, PP 1861.5 / TG 28.66): the account
derives the margin the user used to bisect for.

What the account caught, none of it visible before:
1. `tensor_bytes` billed the offload cache's `prefill_bank_buffers` — views into the first
   `2 x num_experts` slots of its own bank caches — a second time, inflating the expert line
   ~1.9x and making the plan look like it under-priced slots by 2.4x. Fixed by merging
   overlapping byte ranges; the expert line went 5.518 → 3.287 GiB on Flash-Next at 0.9.
2. The GDN prefill workspace is ~0.86 GiB per layer over the default 8192-token chunk on
   Flash-Next (39 % of the whole card), and `chunk_o.py:146`'s `o = torch.zeros_like(v)` is a
   96 MiB slice of it — the allocation that OOM'd. Priced from `LinearGatedDeltaGroupConfig`.
3. The CUDA-graph pool (A9: ~0.2 GB for one captured shape, +0.15 GB per extra shape) and the
   page table were charged *after* the pools were sized, so they came out of the ratio's hole.
   Both are now estimated from config at ledger-open and re-priced from the measurement, which
   is what `engine_overhead_bytes()` exists to carry.
4. `plan_cache_budget` priced the KV reserve in usable pages while the pool allocates one more,
   so a plan could miss its own fit assert by exactly one page (a 1.5 MiB rejection of
   Flash-Next at 0.9); the reserve is now priced as `pool_pages(kv_reserve_pages)`, and the
   greedy expert side hands bytes back when the page floor binds instead of failing startup.
5. After all of that the calibration reads "over-modelled by 0.44 GiB" on Flash-Next at 0.9 —
   the safe direction, and the number Phase 2 starts from: the reserve is currently a
   conservative sum, not a negotiated figure, and `_weights_bytes` is a `mem_get_info` delta
   that includes non-torch overhead (A9 §5.1: measured free is not held bytes).
Residual open items, with owners: price the expert side tables *before* the first plan (they
are only in the account after allocation today); give the encoder cache a byte cap so the mm
line can be a promise instead of a guess; move the ceiling to `engine_allocated_post_weights`
semantics so the account is held-bytes-based end to end.

## EXP-007 — What the account said after each correction (ratio sweep, both anchors)
**Date:** 2026-09-17 · **Verdict:** INFORMATIONAL — the constants are measured now, and two
apparent "plan bugs" turned out to be the account being right and the measurement being wrong
Same 16K harness, same `--num-tokens 16576 --cache-type naive` guard config on both
checkpoints, `--memory-ratio` swept 0.86 / 0.9 / 0.95 / 1.0, reading the printed account each
time. The corrections and what each one moved:

| correction | line it moved | consequence |
|---|---|---|
| merge overlapping byte ranges in `tensor_bytes` | `cache:expert` 10.840 → 10.109 GiB (35B), 5.518 → 3.290 (Flash) | `prefill_bank_buffers` are the first `2 x num_experts` rows of `bank_caches`; billing them twice made a correct plan look like it under-priced a slot by ~2x |
| price `kv_reserve` as `pool_pages(kv_reserve_pages)` | Flash-Next stopped rejecting its own fit assert at 0.9 | the reserve floor is in pages, the greedy fill is in bytes; the miss was exactly the dummy page (1.5 MiB) |
| hand the excess back to the greedy expert side when the floor binds | same | a startup rejection became a slightly smaller expert cache |
| fund `graph:pool` and the page table from formulas before the solve | `pool budget` -0.32 GiB on both anchors | those were the bytes `--memory-ratio 0.86` had been silently reserving |
| price the vision transient per image (192 MiB), not per plausible burst | `reserve` 2.256 → 1.600 GiB | gave back the ~0.3 % of 35B-A3B prefill throughput an over-wide reserve had taken |

Final rows at the default `--memory-ratio 0.9`, three repeats each: 35B-A3B PP 4607.6 /
TG 158.49, account holds 13.739 GiB against the allocator's 13.520 (over-modelled 0.22 GiB),
`uncommitted -0.159 GiB`; Flash-Next PP 1859.3 / TG 28.68, over-modelled 0.44 GiB. Both output
hashes unchanged. Read two caveats before reusing these numbers: `uncommitted` is negative
because `_weights_bytes` is a `mem_get_info` delta that carries non-torch overhead with it, and
Flash-Next's reserve is dominated by one line (`transient:gdn-prefill`, 0.60-0.86 GiB) whose
64-token chunk width is a kernel constant rather than a config knob -- at `max_extend_tokens`
above 8192 that line grows linearly and will start to fight the KV pool, which is Phase 12's
problem with a number already attached to it.

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

