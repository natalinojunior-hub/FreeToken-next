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

## EXP-008 — The ledger decides the split and prices the context targets (Phase 2)
**Date:** 2026-09-17 · **Verdict:** **KEEP** (guards hold; the split now has one owner and an
answer to "how much context does this buy")
Question: with the account in place, does routing the expert/KV decision *through* it change
any measured behaviour, and what does the same account say about 128K/256K/512K/1M before any
of that work exists?
Method: `VramLedger.decide()` computes `pool_budget_bytes(fixed_cache_bytes)` and hands the
existing split rule (`resolve_moe_cache_auto` → `plan_cache_budget`) exactly that money, so no
term is subtracted twice and no second budget formula survives; the returned `MemoryPlan`
carries the decision plus a `ContextFeasibility` row per target context, and the engine logs it
at startup. A unit test pins the parity that matters: with no reserve and no overhead lines,
`decide()` must return what the pre-ledger formula returns.
Guards at the default `--memory-ratio 0.9`, same harness, 3 repeats:

| model | PP | TG | TTFT | VRAM | sha1 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4 16K | 4613.1 (min 4612.1) | 158.53 | 3551.6 ms | 14.59 GiB | `2a6dca88ffdc` (unchanged) |
| Qwen3.8-Flash-Next NVFP4 16K | 1859.8 (min 1857.9) | 28.70 | 8809.4 ms | 14.80 GiB | `f8bbaeb7e214` (unchanged) |

And the plan it printed -- BF16 KV, no compression, no tiering, priced against **what the split
left for KV** (an earlier revision of these rows priced them against the whole pool budget and
so reported "128K fits" on a model whose expert cache had already eaten the budget; that
correction is f6dddb5 and it is the number that matters):

```
35B-A3B    pool budget 10.266 GiB -> 6113 expert slots (10.109 GiB), 0.157 GiB left for KV
           128K needs 2.500 GiB (+2.343)   256K needs 5.000 (+4.843)
           512K needs 10.000 (+9.843)      1M   needs 20.000 (+19.843)
Flash-Next pool budget  3.123 GiB -> 1133 expert slots (2.925 GiB), 0.197 GiB left for KV
           128K needs 3.097 GiB (+2.900)   256K needs 6.192 (+5.995)
           512K needs 12.382 (+12.185)    1M   needs 24.763 (+24.566)
```

Three consequences, and they are the reason this brick exists before TurboKV rather than after
it:
1. The MoE-priority fill (`plan_cache_budget`, which hands the expert cache everything above
   `--kv-reserve-tokens`) is why the anchors never reached 16K under `--moe-cache-auto` without
   `--num-tokens`: it leaves them ~8K tokens of KV. The rows turn that from an anecdote into an
   account line, and EXP-009 turns it into 128K.
2. Flash-Next's 128K cannot be bought that way at all (its whole pool budget is one 128K KV
   pool), so on the hybrid model long context is gated on compressed KV, not on RAM bandwidth --
   the "compress before paging" rule earns its place with a number.
3. 1M BF16 KV at 24.8 GiB matches PERFORMANCE.md §4's independent arithmetic (20-24 GiB), so
   the account and the closed-form model agree on the same model from different directions.
Cost: none measurable. Both hashes are the pre-plan hashes and both PP numbers sit inside the
run-to-run spread of the last four revisions of the same config.

## EXP-009 — 128K is reachable today by buying it with the expert cache (Phase 12 evidence)
**Date:** 2026-09-17 · **Verdict:** **PASS (capability)** — 128K works on Qwen3.6-35B-A3B NVFP4
Question: the plan said a 128K context needs 2.500 GiB of KV while the MoE-first split leaves
0.157 GiB. If the trade is real, asking for the context explicitly must produce a working
server at a measurable TG, predicted in advance.
Method: the same harness, `--kv-reserve-tokens 131136` (the reserve is the only knob; no code
path changed), BF16 KV, `--cache-type naive`, bs=1, 131 072-token prompt, 32 generated, one
repeat, `--memory-ratio 0.9`.
Result: **PP 3188.5 tok/s, TG 89.30 tok/s, TTFT 41 108 ms, ITL p50 10.99 / p95 12.81 ms,
VRAM 14.45 GiB, RSS 22.0 GiB, 131 221 KV pages, output sha1 `d4b5b385a3cf`.** The printed plan
agreed with the run line by line: `pool budget 10.265 GiB -> 4694 expert slots (7.762 GiB) +
131 221 usable KV pages … of the 2.503 GiB left for KV`, and the demand row's prediction of
~4695 surviving expert slots is what the greedy fill produced.
What it means:
1. 128K on a 16 GiB card is not a future feature on the 35B-A3B -- it is one flag, and the flag
   costs 1419 expert slots (6113 -> 4694) while decode stays at 89 tok/s.
2. The reserve-to-context identity (`--kv-reserve-tokens = the context you want`) is the whole
   mechanism `--context auto` needs; what is missing is only the policy for choosing it per
   request mix, and the ledger already answers "what does that context leave for experts".
3. The same arithmetic says Flash-Next cannot buy 128K this way: its pool budget is 3.123 GiB
   and 128K of KV is 3.097 GiB, so funding the context leaves ~0.03 GiB of experts -- which is
   the quantified reason Phase 3 (turbo4 at 4.125 bpv) is on the critical path rather than
   optional: at 4x compression the same 128K costs 0.77 GiB and the expert cache survives.

## EXP-011 — The MoE GGUFs are refused today, with the blocker named (Phase 6/7 target list)
**Date:** 2026-09-17 · **Verdict:** **BLOCKED, blocker identified** (not a regression; a target)
Question: which of the local GGUF MoE checkpoints can `freetoken-next` actually load today? The
subagent audit (A7 §7) claimed all four refuse on mixed expert geometry; claims that a mixed
bank "decodes silently wrong" or that `expert_bytes_per_slot` misses a dimension did not survive
the source (`gguf_experts.py:118-122` raises, and a slot is exactly one layer-expert row), so
this was settled on the host instead of on paper.
Method: `ft serve --model <gguf> --max-running-requests 1 --memory-ratio 0.9 --num-tokens 4096
--max-seq-len-override 4300 --cuda-graph-max-bs 0 --max-prefill-length 1024`, read the refusal.
Result, verbatim:

- `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` → `NotImplementedError: GGUF expert bank
  'gate_up' mixes ggml types across layers ({'Q3_K': [5..34], 'Q4_K': [0..4, 35..39]})`, after
  the message's own reason: one slot pool, one stride, `moe_vec.cuh` addressing rows as
  `expert * nrows * (ncols / qk)` with no padding allowance.
- `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` → same guard on `down`:
  `{'Q6_K': [34, 38, 39], 'IQ4_XS': [0..33, 35..37]}`.
Both fail closed, in the loader, before any GPU allocation -- so there is no silent corruption
to fix, only a capability to add. The dense `Qwen3.8-27B` IQ3_S file serves and benchmarks
(EXP-010/PERFORMANCE.md §9), and `qwen36-35b-a3b-dflash-Q4_K_M.gguf` is the one MoE bank layout
that passes the guard because it is uniform.
What Phase 6 therefore actually needs, in order: (1) expert slot pools keyed by
(bank, ggml type, row geometry) rather than by bank, because a mixed bank is two allocations by
construction; (2) the per-role split, since `gguf_expert_types` records one type per layer for the
fused `gate_up` bank and `gguf_expert_specs:123` then prices both halves with the gate's row bytes.
An earlier draft of this list put "row readers for Q4_K / IQ4_XS / Q3_K" between them; that is
**retracted** -- the GPU path dequantizes every type `BLOCK_SHAPE` names (`--moe-strategy offload`),
so no reader is missing there. The only format gap is CPU-side: `_cpu_moe` dispatches weight formats
for ggml ids 2/12/14 (`cpu_moe_ext.cpp:1367`, `WF_Q4_0/WF_Q4_K/WF_Q6_K`) and
`_resolve_gguf_format` (`cpu_executor.py:86-119`) takes **one** format for both banks, so these
checkpoints run on offload until Q3_K/IQ3_S/IQ4_XS get CPU dot kernels.
**Settled the same day (A7 §8, and it was the audit's own error):** the two byte measurements in this
section disagreed. A7 §8 measured the physical stride of all 1194 expert/weight tensors from their
data offsets (`.qwen/tmp/a7_truth.py`, alignment 32, measured pad 0 on every row) against this tree's
`dequant.py BLOCK_SHAPE`, gguf-py's `GGML_QUANT_SIZES` and llama.cpp's struct arithmetic: **1194/1194
match**, so the ids are *not* renumbered (`Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, IQ4_NL=20,
IQ3_S=21, IQ2_S=22, IQ4_XS=23` -- id for id what `ggml.h:400-413` declares) and the first pass's
276 B / 308 B rows were the gate+up concatenation folded into a single row (a `gate_up` slot is two
rows; Q3_K is 110 B per 256 elements, so its 512-element `ffn_down` row is 220 B, and `276 B` appears
in no file in the corpus). Nothing has to be settled against a reference dequant before Phase 7 keys
the pools; the metadata dead-end stands (`load_gguf_metadata` exposes only `GGUF.version`,
`general.quantization_version` and `general.architecture`, and the `tokenizer.ggml.quant_layout`
key A7 §8 first reported is not present).
**Consequence for the matrix:** these are recorded as BLOCKED rows that name their blocker, which
D-012 requires, and they are not counted as regressions.

## EXP-012 — Turbo KV: the codec, the pool, and a decode that never de-rotates a tile
**Date:** 2026-09-17 · **Verdict:** IN FLIGHT (landed: codec + pool + fused decode; open: prefill, engine wiring)
Question: can FreeToken *consume* compressed KV without paying the materialize tax the reference
measured (-10.8 % TG @16K, -28.7 % @64K)?

The architecture question came first, and it is a measurement, not an opinion: only our Triton
backend can consume a custom KV layout (flashinfer takes a fixed dtype, and
`_resolve_auto_attention_backend` picks `fi` on this host), so what does Triton cost by itself?
**PP 4344.8 / TG 144.11 at 16K on the 35B-A3B vs the fi anchor 4610.8 / 158.53 -- a 9.1 % backend
gap before any codec exists** (`w1_triton.log`; its sha1 `b7c70b36d276` differs from
`2a6dca88ffdc`, so the output-hash guard is per-backend, not global). Stated plainly because it
bounds the work: at 16K the KV is a few percent of the decode step's bytes, so *no* quantizer pays
a 9.1 % backend penalty, and "TG within -3 % of 158.53 at 16K" is not reachable on this path. Where
the codec pays is long context, where KV is precisely the thing being bought out of the expert
cache (256K today costs 5.00 GiB of KV; at turbo4 that is 1.29 GiB).

The trick that makes the read cheap: store in the **rotated** domain, pre-rotate Q (rotate is
orthogonal, so `(Wq)·(Wk)ᵀ == q·kᵀ`), accumulate V rotated, and rotate the output row back once.
Per KV tile that leaves a byte gather, an 8/16-entry lookup and one multiply -- no transform per
tile, which is what LTO's fused prefill lost 6-11 % on.

Landed, four commits (`7fc7d7f..77229c2`), 59 green pins: `kernel/triton/turbo_kv.py` (codec:
layout, corrected-norm semantics, tie rule, 128-element rotation group), `kvcache/turbo_pool.py`
(a paged codes+norm pool that refuses to hand a bf16 view to a bf16 reader),
`kernel/triton/turbo_attn.py` (the tile readers), and one `COMPRESSED` constexpr branch in the
split-k grouped decode kernel -- a branch, not a second kernel, so the compressed path inherits
split-K, head tiling, sinks and graph capture instead of reimplementing them badly, and the 38
existing bf16 pins still pass unchanged.

Three real bugs, each invisible to the obvious test: (1) `SIGNS2` transcribed 127 elements from a
truncated doc dump -- rotations still "worked", because a dropped lane only shifts indices, so the
array is now pinned by length *and* sum; (2) `pack` masked nothing before `<< 6`, so turbo3's third
bit leaked into the neighbouring lane and 7<<6 wrapped uint8; (3) `unpack` sliced turbo3 "all
words, then all bits" where `pack` writes group-major -- exact at head_dim 128, wrong at 256, and
caught only by the GPU tile probe with two groups per row. The wiring test compares against
attention over the *decoded* KV, not the original, so it can only fail on the bookkeeping and never
on quantization error.

Accuracy is at theory, which is the useful surprise: per-vector NMSE 0.0339 (turbo3) and 0.0092
(turbo4) against the Lloyd-Max table for 8/16-level Gaussian quantization (0.0345 / 0.0095) --
slightly better, because the corrected norm projects the reconstruction back onto the input's
sphere. The V-side error of attention equals that number and does **not** move with softmax
sharpness (pinned at three scales), so it is a floor a `(layer, side)` tier schedule has to design
around; the K-side error is the part that grows with logit scale. Bytes: 50 B (turbo3) / 66 B
(turbo4) per token per head per slab per layer vs 256 B bf16 -- 5.12x / 3.88x on a head_dim-128
model -- with `kv_cost`/`unit_bytes` parity pinned so the plan and the allocator cannot disagree.

Next: the extend (prefill) branch, then engine wiring (config key, pool factory, Q rotation and
output rotation in `attention/triton.py`), then the 16K and 256K A/B. No KEEP is claimed here:
there is no serving number yet.

## EXP-013 — The coded readers were instruction-bound; contiguous rewrite REVERTED
**Date:** 2026-09-17 · **Verdict:** **REVERT** (correctness green, performance regressed)

Question EXP-012 left open: what does the coded read path actually cost when it is serving?
Setup: 35B-A3B NVFP4, 16K prompt / 128 generated, `--repeats 2`, `--mem-ratio 0.9`,
`--cache-type naive`, `--kv-format turbo4` (which forces `--attention-backend triton`, see D-017).
Logs `/models/desenvolvimento/tmp/ftnext/{w1_triton,t4_16k,w1_smoke,t4_256k}.log`.

Result, and it was not the result expected: **PP 3560.5 / TG 61.79 / ITL p50 15.98 ms** against
`triton + bf16` at PP 4344.8 / TG 144.11 / ITL 6.79 ms. **4x fewer bytes arriving while ITL doubled**
is the signature of an instruction-bound loop, not a memory-bound one: the first readers indexed a
byte *per element* (`codes + j//2`) and did one L1 gather into the centroid book per element, so a
tile's 256 lanes computed 256 addresses where 128 contiguous loads would do, and the packed layout's
whole advantage was spent on address arithmetic.

Also measured, because it was the obvious suspect and it was not: **the encode is not the
bottleneck.** After replacing the midpoint search's broadcast (`(y >= mid).sum(-1)`, which materializes
`[rows, dim, 16]` per layer -- 67M booleans per 8192-token chunk) with `torch.bucketize(y, mid,
right=True)` -- which *is* the reference's tie rule, "count of boundaries <= value", so the pins did not
move -- a layer-chunk of 32768 groups costs quantize 0.48 ms, rotate 0.30, indices 0.03 (the former
dominant term), pack 0.01, decode 0.85. The prefill crawl at 256K was the readers; the `bucketize`
change is a large constant-factor win on the write path regardless.

Two side findings worth keeping: at 256K the *plan* with turbo4 leaves **5427 expert slots vs 3183**
in bf16 (PERFORMANCE §10) -- the capability half of the phase is proven; and the 256K benchmark was
aborted before it produced throughput, during which a `pkill -f "<bench cmdline>"` matched **its own
command line** and killed the driver while the server kept 14 986 MiB -- the trap JOB_REGISTRY already
documents, re-confirmed the hard way, cleaned by PID with the GPU verified at 2 MiB afterwards.

The attempted rewrite loaded each token's packed row contiguously and expanded it in registers. Its
tests were correct, but the larger live tile caused register pressure and lost throughput.

Where it stopped, and what is verified versus not:

* **Correctness:** the rewrite passed `6 + 94` focused pins, with bf16 pins unchanged.
* **Matched A/B:** `t4b-16k` measured **PP 3524.7 / TG 45.74 / ITL p50 21.69 ms**, versus the
  prior **3560.5 / 61.79 / 15.98 ms**. VRAM stayed 14.20 GiB and the greedy hash stayed
  `49e9819649ba`; the 26% TG loss is a performance regression, not a correctness change.
* **Action:** the seven uncommitted rewrite files were restored to their pre-rewrite state. The next
  reader experiment needs a bounded-tile design that reduces byte address work without materializing
  a full `[N, D]` register tile.
* **Not done at all:** a full-suite run after the wiring; coherence text for a turbo4 generation (only
  throughput hashes so far -- `49e9819649ba` is a deterministic continuation, not a validated one);
  turbo3 end-to-end at any context; CUDA-graph capture evidence for the rotation allocations; a
  vectorized *store* kernel (torch encode is 0.48 ms/layer-chunk, so not urgent); and the VBR tier
  policy that the V-side floor (EXP-012) is meant to drive.

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

## EXP-014 — Remove the unowned dequant scratch reserve
**Date:** 2026-09-17 · **Verdict:** **KEEP**

`modelled_reserves()` had a `dequant_scratch` parameter and reserve line, but repository search
found no caller or matching allocation. The parameter and dead charge were removed rather than
claiming bytes for an unmeasured consumer. The focused ledger suite passes **22/22**. This closes
the accounting gap only; actual compressed-KV reader scratch remains pending until its allocator
reports measured peak bytes.

## EXP-015 — Flash-Next routing counters are exposed, but the 16K run is BLOCKED
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / BLOCKED**

Decision B needed per-layer MoE active/miss counters without changing the decode graph. The
existing graph-safe counters are now exposed through `--moe-collect-stats`; the worker prints
the aggregate and per-layer decode window at shutdown. The parser test passes and `ft serve
--help` shows the option (`56774dc`).

The intended 16K functional run was retried twice with explicit serial expert loading. Both
workers exited while building expert bank `174/192`, before serving a request. The supervisor
initially hid the process status; `b4b200f` now reports it. A detached-server reproduction
ended at `176/192` with `exitcode=-9` (SIGKILL), while foreground benchmark cleanup produced
`exitcode=-15` (SIGTERM). The Flash-Next geometry prices the 48 x 512 NVFP4 bank at **63.46
GiB** before model/runtime overhead; the host had **62 GiB available and no swap**. This is
consistent with host-memory pressure, not a Python exception in the new instrumentation.
The unpinned diagnostic did not reach readiness either. No throughput or routing statistics are
claimed; logs are `/models/desenvolvimento/tmp/ftnext/flash_load_detached_3.log` and
`flash_load_unpinned.log`.
