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

## EXP-016 — Preserve native MTP metadata at the config seam
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / KEEP**

`Qwen4ExpArgs` now carries the checkpoint's `text_config.mtp` metadata, accepts the released
one-layer hybrid shape, and rejects `num_hidden_layers > 1` until the runtime supports it.
The main target loader remains runtime-neutral: the new reader can enumerate `mtp.*`, but no
draft module consumes those tensors, no draft model is built, and no speculative token is
emitted. Focused config and model-skeleton tests pass **29/29**.

The loader now also exposes `iter_mtp_weights()`, which yields the native `mtp.*` namespace,
including the packed MTP expert tensors, without adding them to the target state dict. This is a
reader seam only: no draft module consumes it yet. The qwen4_exp weight suite passes **31/31**.

## EXP-020 — Flash-Next FTW conversion survives host-memory pressure
**Date:** 2026-09-17 · **Verdict:** **PASS / IMPLEMENTED**

The failed conversion was not a Python exception: `earlyoom` terminated the process while the
streamed expert banks accumulated in shared anonymous `mmap` storage. `MADV_DONTNEED` did not
reclaim those pages. Host banks now use private anonymous mappings, which makes the discard
operation effective while preserving the serving allocation path.

The fix was resumed against the existing partial output. The native command completed with
`EXIT:0` in 54.4 s after reaching all 192 expert source shards. The resulting FTW index has
1,415 tensors (1,127 dense and 288 bank entries), all 48 MoE layers, and 10 shards bounded by
8 GiB, and was opened successfully by `FTWReader`. Focused coverage is 45 passed, 6 skipped
for the host-bank and checkpoint suites; the broader Qwen4/FTW/MoE focused set is 75 passed,
6 skipped.

## EXP-017 — FTW streaming conversion reduces RSS but did not produce a valid artifact
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / INCONCLUSIVE**

The existing FTW converter was tested as the bounded-NVMe hypothesis. Command:

```text
PYTHONPATH=python .venv/bin/python -u -m freetoken.cli checkpoint \
  --model /models/Qwen3.8-Flash-Next-NVFP4-Radix \
  --out /models/desenvolvimento/tmp/ftnext/Qwen3.8-Flash-Next-NVFP4-Radix-FTW \
  --moe-backend offload --quant-backend moe.nvfp4=triton --shard-gib 8
```

The layer sink streamed and released banks while writing; observed converter RSS was about
**2.7 GiB**, and the output reached **191/192** expert input shards (about 121 GiB including
dense weights and PLE side files). The process terminated before writing
`freetoken_weight.json`, so the directory is not a usable checkpoint and no serving or
NVMe-speed claim is made. The partial artifact and log remain under
`/models/desenvolvimento/tmp/ftnext/` for forensic inspection only.

## EXP-018 — FTW conversion restart point
**Date:** 2026-09-17 · **Verdict:** **IMPLEMENTED / UNIT-VALIDATED**

`FTWWriter` now commits an atomic `.freetoken_weight.progress.json` after every tensor.
On restart it validates every completed shard, checks committed tensor metadata,
truncates only the active shard back to its durable boundary, and reuses existing
entries. The converter skips already committed dense tensors and per-layer expert-bank
entries; a completed `freetoken_weight.json` is treated as idempotent.

Coverage: `tests/checkpoint/test_ftw_weights.py` resumes after injected bytes in the
active shard and replays both tensors; the checkpoint suite passes **19 passed, 6
skipped**. In the native Flash-Next run, the first interruption left **1,409** committed
entries through layer 46; the next invocation validated and skipped those entries before
reaching the final source shards. The host supervisor then sent **SIGTERM (143)** before
the last layer, but the manifest remained valid and reusable; a full serving artifact is
still pending.

## EXP-019 — Native MTP host contract
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / CORRECTNESS FOUNDATION**

Added `python/freetoken/engine/spec.py` with model-independent contracts for longest-prefix
draft acceptance, fixed-size pipeline messages, paged-cache rollback, GDN convolution-state
rebuild, and PLE n-gram context rebuild. These are pure helpers only; the Qwen4Exp draft
module, separate KV namespace, expert-bank append, scheduler loop, and target-equivalence
runtime gate are still absent. Focused tests pass together with the checkpoint suite.

## EXP-021 — Native MTP draft-layer construction/forward contract, GPU-verified
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / KEEP**

`Qwen4ExpMTP` now builds the released one-layer hybrid draft head: `pre_fc_norm_hidden` /
`pre_fc_norm_embedding` (`GroupedPlusOneRMSNorm`) feed `fc_hidden` / `fc_embedding`
(`LinearReplicated`) into one `Qwen4ExpDecoderLayer` registered at `layer_id = num_layers`,
whose expert bank is redirected to the target's routed experts via `_MTPQuantConfig`.
`with_mtp_layer` extends the model's single full-attention group with that layer id (KV/index
slot only) without changing target depth or `num_moe_layers`. `--spec-mtp` /
`EngineConfig.spec_mtp` are wired end to end but default to 0 (inert).

The first version of `test_mtp_construction_and_forward_contract` crashed the Python
interpreter (`malloc(): unsorted double linked list corrupted`, then a raw `Fatal Python
error: Aborted` under pytest) because it built and ran the model on CPU tensors while
`VocabParallelEmbedding.forward` unconditionally launches a CUDA-only JIT kernel
(`kernel/index.py` -> `index.cu`) with no device guard -- the same crash the sibling
`test_decoder_stack_prefill_and_decode` already avoids with `@requires_cuda` and a
`torch.device("cuda")` construction context. Fixed by building/filling the model and its
generators on `cuda`, matching that pattern; this is a test bug, not a model bug -- the
forward contract itself was never exercised until this fix. `_MTPQuantConfig`'s registration
rejection paths (missing hybrid flag, wrong layer types, foreign RoPE base, target-slot reuse,
ambiguous full-attention groups) are covered without a GPU.

Result: 8/8 MTP-focused tests pass on the host GPU (construction, forward-contract numerics
against a torch reference, quantization routing, geometry rejection) plus 21/21 config tests
and the full checkpoint/moe/qwen4_exp focused set (281 passed, 60 skipped -- the only 2
failures are the pre-existing flashinfer/nvcc-13.3 fp4 build issue, unrelated to this change).

Still missing before any MTP throughput claim: a separate draft KV namespace/expert-bank
append, the draft/verify/rollback scheduler loop, and the target-equivalence gate (greedy
token-stream match against the same target configuration without MTP), per goal item E.

## EXP-022 — Native MTP: KV namespace for the draft layer's QSA slot
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / KEEP**

Two real bugs found and fixed while scoping item E's "KV and verification scratch before
allocation":

1. `create_kvcache_pool`'s QSA branch passed `model_config.num_layers` to `QSAKVCache`'s
   layer-id remap, which only covers `[0, num_layers)`. `with_mtp_layer` registers the draft
   head at `layer_id == num_layers` in the same full-attention group, so pool construction
   raised `ValueError: KV layer id N outside [0, N)` as soon as an MTP layer was registered
   -- independent of `--spec-mtp`, before serving could start. Fixed by widening the remap to
   `max(num_layers, mtp_layer_id + 1)`; the draft layer now gets its own K/V storage slot,
   distinct from every target layer (`tests/models/qwen4_exp/test_config.py::
   test_mtp_registration_gives_the_draft_layer_its_own_kv_storage`).
2. `QSAKVCache.ring_capacity_for(index_ratio, num_speculative_tokens)` already existed
   ("spec decode widens by the draft depth") but nothing ever called it with a real value --
   both the constructor default and `kv_cost`'s fixed-size term always priced 0 speculative
   tokens. `create_kv_pool` -> `create_kvcache_pool` now threads `config.spec_mtp` through to
   both the live allocation and its pre-allocation budget, so the two cannot disagree once a
   draft depth is set (`tests/kvcache/test_qsa_pool.py::test_kv_cost_widens_the_ring_for_spec_mtp`).

`--spec-mtp` still defaults to 0, so neither fix changes any existing serving behavior; they
close correctness gaps that would otherwise surface as a crash or a silent under-allocation
the moment item E's scheduler work turns MTP on. Full checkpoint/kvcache/moe/qwen4_exp focused
set: 601 passed, 61 skipped (same 2 pre-existing flashinfer/nvcc-13.3 failures, unrelated).

Still open for item E: the draft KV/expert-bank append itself (writing draft-step K/V into
this now-correctly-sized slot), the scheduler draft/verify/rollback loop, and the
target-equivalence gate (greedy token-stream match against the same target configuration
without MTP).

## EXP-023 — Native MTP scheduler loop: where the draft/verify step actually has to live
**Date:** 2026-09-17 · **Verdict:** **INFORMATIONAL / BLOCKED (needs architecture decision)**

Scoped item E's remaining piece (draft/verify/rollback scheduler loop) by reading the live
decode/prefill/CUDA-graph paths before touching any of them. Findings:

1. **Decode is hard-wired to exactly one new token per request per step.** `_make_write_tuple`
   (scheduler.py:963) writes one KV slot per request (`req.device_len if req.can_decode else
   -1`); `_make_positions`/`_make_mrope_positions` size their output as
   `sum(r.extend_len for r in batch.padded_reqs)` but every decode `Req` in practice carries
   `extend_len == 1` (see the `test_decoder_stack_prefill_and_decode` decode batch). A verify
   step needs 2+ new positions (the accepted correction token plus each draft) written and
   read back in one forward -- decode's write path does not support that today.
2. **Prefill's admission path does not fit either.** `PrefillAdder`/`ChunkedReq`
   (scheduler/prefill.py) extend a request from its known, fixed `input_ids` established at
   admission (prefix-cache matching, per-request budget/hit-rate accounting, mm-item
   chunking). A draft token is generated on the fly mid-decode; there is no clean seam to
   inject one unplanned token into that machinery without either faking a re-admission or
   duplicating a chunk of PrefillAdder's bookkeeping.
3. **CUDA graph decode replay** (`engine.py:1246 can_use_cuda_graph`/`graph_runner.replay`) is
   captured for fixed decode shapes; it already has an eager fallback path
   (`self.model.forward()` when `use_graph` is False), so an MTP verify step can plausibly
   avoid graph capture entirely by routing through the eager path -- but this still needs the
   decode-shape write/position machinery in (1) to support >1 new token per request first.

Net: implementing item E's scheduler loop is not a bounded bug-fix-sized change like EXP-022;
it requires a genuine multi-token-per-step decode write/position primitive (or an equivalent
new code path) that today's engine does not have anywhere, plus the accept/rollback/CUDA-graph
interaction the goal's preflight explicitly calls out. Escalated to Opus 5 for an architecture
read before further implementation, per CLAUDE.md's ESCALATE rule (subagent given exact files
above, self-contained, no live-path edits made yet).

## EXP-024 — Native MTP: verify-batch seam + rollback primitive (Opus 5 review acted on)
**Date:** 2026-09-17 · **Verdict:** **PARTIAL / KEEP**

Consulted Opus 5 on where the item E draft/verify/rollback step should live (EXP-023's
blocker). Two premises in that scoping were wrong and shrank the real problem: `_make_write_tuple`
only scatters the *sampled* token into the host token pool, not KV -- KV write locations already
come from `batch.out_loc`, which is `extend_len`-generic. The actual one-token-per-step lock-in
is `batch.is_decode` gating hardcoded branches in `build_fla_metadata`, `build_ple_metadata`, and
`ParallelLMHead.forward`. Recommended seam: run a verify step as a `phase="prefill"` `Batch` with
`extend_len==2` for the single running request, entered directly (not through `PrefillAdder`,
which is tied to fixed `input_ids` known at admission) -- this reuses the already-correct
extend-aware FLA/PLE prefill paths and never touches CUDA graph capture
(`can_use_cuda_graph` is decode-only, so a prefill-phase batch is eager for free).

Landed the two pieces this enables safely on their own, both strictly opt-in and no-op today:

- `Batch.spec_logits_indices` (core.py): when set, `ParallelLMHead.forward` selects it instead
  of `get_last_indices(bs)` (one logit row per drafted position, not per request), and
  `Engine.forward_batch` skips the automatic `req.complete_one()` and stops truncating logits to
  `batch.size` -- a real bug the review found: that truncation would have silently dropped a
  verify batch's second logit row (silent wrong-accept, not a crash).
- `CacheManager.free_spec_reject` (scheduler/cache.py): consumes `pages_to_free` (its arithmetic
  was already correct, just never called) to return a rejected verify window's whole unused
  pages, without touching the prefix cache (a rejected window was never `cache_req`'d). No-op
  when `keep_len`/`alloc_len` round up to the same page -- the common case once page_size exceeds
  the draft depth. 3 focused tests (`tests/scheduler/test_spec_reject_frees_pages.py`).

Review also found two of the five `engine/spec.py` helpers are superseded rather than needed:
`rebuild_conv_state` only rebuilds the GDN conv (width-window) state, but the GDN delta-rule
*recurrent* state also advanced and can't be reconstructed from a width-window, and the GDN op
overwrites its live state in place during forward (no `prev_state` available after the fact
anyway). `ngram_context_after`'s convention is unpinned and redundant. Both are superseded by
`LinearStatePool.copy_from(src, dst)`, which already snapshots/restores GDN conv + GDN recurrent
+ every `slot_states` entry together -- and Qwen4Exp's PLE n-gram context (`PLE_NGRAM_STATE`) is
one of those slot states, so one `copy_from` covers all three. `accept_drafts` was traced and
confirmed correct as-is.

Full checkpoint/kvcache/moe/qwen4_exp/scheduler/layers/dsv4 focused set: 720 passed, 61 skipped
(same 2 pre-existing flashinfer/nvcc-13.3 failures).

Still open for item E: the spec loop itself (own module, driving draft-token injection into the
token pool, the verify Batch construction, `accept_drafts` call, and `LinearStatePool`
snapshot/restore around it), the greedy-only/single-request/non-overlap gate, and the
target-equivalence test (`spec_mtp=1` byte-identical token stream vs `spec_mtp=0`, swept across
prompt lengths crossing an `index_ratio` boundary per the review's QSA pending-ring warning).

## EXP-025 — Native MTP: complete scheduler spec-loop algorithm (Opus 5 review, round 2)
**Date:** 2026-09-17 · **Verdict:** **INFORMATIONAL / DESIGN COMPLETE, NOT YET IMPLEMENTED**

Second Opus 5 consult, generalized over a configurable draft depth `k = config.spec_mtp`
(operator requirement: n_max must be tunable, default 1, benchmarkable at 2/3 -- not hardcoded).
Confirmed and extended EXP-024's seam with a concrete, file:line-verified algorithm:

**Key findings beyond EXP-024:**
- `Qwen4ExpMTP.forward` is NOT stateless: at `layer_id == num_layers` it runs a real
  `Qwen4ExpAttention` forward against the paged QSA pool's own reserved slot (`attention.py:163-165`).
  Drafting k>1 tokens is genuinely autoregressive over the draft head's own KV, not k
  independent re-reads of `_last_residual`. The KV slot EXP-022 sized is load-bearing.
- `scheduler._forward`'s `output_mapping` (`_make_write_tuple`) writes exactly one token per
  request unconditionally; a `(k+1)`-row verify result would either crash (shape mismatch) or
  (via `-1` for a finished req, a valid negative index) silently misdirect a write. The spec
  loop must bypass `scheduler._forward` and `_process_last_data` entirely, replicating their
  EOS/stop-string/`append_host`/`cache_req`/`DetokenizeMsg` bookkeeping itself, m times per step.
- `pad_batch` is decode-only (`graph.py:204-221`), so a `phase="prefill"` verify batch is never
  padded -- `spec_logits_indices = arange(k+1)` is exact, no graph-buffer offset to account for.
- New gap found: `Sampler.sample`'s non-greedy path sizes `temperatures`/`top_k`/`top_p` by
  request count, which breaks on a `(k+1)`-row verify batch of 1 request; needs a
  `repeat_interleave`-based `_expand_sample_args` helper (not yet added). Greedy-only for now
  sidesteps it (`temperatures is None` short-circuits).
- **The throughput-critical gap**: the draft layer's KV over the ORIGINAL PROMPT is never
  populated (prefill only runs the target stack), so the first verify window after a prefill
  drafts against garbage context -- correct (target verify still exact) but acceptance is
  ~random until one deliberate MTP pass over the prefill's `_last_residual` window backfills it.
  Without this, measured acceptance rate would be misleadingly bad and TG would look like MTP
  overhead with no offsetting benefit.
- `--spec-mtp` has no concurrency gate today (nothing ties it to `max_running_req`); isolation
  is purely "the spec loop builds its own single-request Batch," so other live requests are
  physically unaffected either way, but a single-request enforcement (loud refusal otherwise,
  per the codebase's existing "refuse clearly" pattern) is still recommended before enabling it.
- Overlap scheduling (`ENV.DISABLE_OVERLAP_SCHEDULING=False`, the default) does not fit: the
  accept decision is host-side and must land before the next forward launches. `--spec-mtp > 0`
  needs a startup-time refusal unless overlap scheduling is disabled -- not yet implemented.

Full per-step algorithm (draft chain -> verify Batch(phase="prefill", extend_len=k+1) ->
`accept_drafts` -> commit-or-reject with `CacheManager.free_spec_reject` +
`LinearStatePool.copy_from` snapshot/restore + a plain re-forward of the accepted tokens on
reject to re-derive the next chain's seed residual) is recorded in full, file:line-referenced
pseudocode in this session's transcript; not reproduced here to keep this entry scannable.

**Not implemented this session.** This is the single highest-blast-radius piece in the
codebase (the live decode scheduling loop shared by every request, MTP or not) and cannot be
correctness-tested without a real serve run against the actual checkpoint -- no unit or toy-GPU
test reaches `Scheduler`/`CacheManager`/`TableManager` wired together today. Implementing it
blind, without that integration-test capability, was judged too high-risk for this pass.
Everything up to and including this design is real and durable; the scheduler loop itself,
the prefill-window MTP warm-up pass, `_expand_sample_args`, and the single-request startup
refusal are the exact, ready-to-implement next unit for item E.

## EXP-026 — Native MTP scheduler loop implemented (untested live)
**Date:** 2026-09-17 · **Verdict:** **IMPLEMENTED / UNVALIDATED-LIVE**

`scheduler/spec.py` (`SchedulerSpecMixin.run_spec_step`) implements EXP-025's design in full,
generalized over `k = config.spec_mtp`. Wired into `normal_loop`; refuses loudly at `Scheduler.
__init__` if `--spec-mtp>0` with overlap scheduling enabled or `max_running_req != 1`. Inert
(0 regressions) at the default `--spec-mtp 0`: 720 passed, 61 skipped, same 2 pre-existing
flashinfer/nvcc-13.3 failures. No test in this repo wires Scheduler+CacheManager+TableManager+
a real model together, so the actual draft/verify/accept/reject/EOS-truncation logic is
unverified by execution -- correctness rests on the file:line-verified design only. Operator
decision: implement now rather than wait for a live checkpoint (recorded per their explicit
choice). First real validation must happen at a live 16K serve per the goal's preflight gates
(MTP acceptance/target-equivalence), before any 256K attempt.

## EXP-027 — Native MTP first live serve: real bugs found, then a CONFIRMED content divergence
**Date:** 2026-09-17 · **Verdict:** **BLOCKED / CRITICAL CORRECTNESS BUG**

First actual `--spec-mtp 1` serve against the real Flash-Next checkpoint (16K, naive cache,
`--max-running-requests 1`, `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`), after freeing host RAM
(21 GiB of stale `/tmp` tmpfs files from unrelated past sessions were competing with the ~63 GiB
NVFP4 expert-bank footprint and causing an earlyoom kill during weight loading -- same
pre-existing constraint as EXP-015, not caused by MTP; see AGENTS.md's new "never use RAM as
storage" rule). Three real bugs found and fixed in sequence, each confirmed by the next serve
attempt getting further:

1. `iter_mtp_weights` was a reader seam only (EXP-016) -- nothing consumed it, so
   `Qwen4ExpMTP`'s own dense weights were never loaded: `KeyError: 'mtp.pre_fc_norm_hidden.weight'`.
   Fixed: `iter_mtp_weights` now fuses q/k/v -> qkv_proj and HC block-inject parts exactly like
   `iter_weights` (reusing `_DenseFuser`), skips the draft's packed expert-bank tensors
   (`_MTPQuantConfig` reuses the target's), and `Engine._load_weight_state_dict` chains it in.
2. `iter_mtp_weights` wasn't exported from `qwen4_exp/__init__.py`, so `_load_attr` raised
   `AttributeError`. Fixed: exported.
3. `_init_offload_moe_cache`'s `assert len(layers) == num_moe_layers` failed (49 vs 48): the
   generic `iter_offload_moe_layers` walk also finds the draft's own `Qwen4ExpMoE` instance.
   Deeper bug underneath: that instance was built with `layer_id == config.num_layers` (its own
   KV/attention slot), but `_MTPQuantConfig` routes its weights onto the TARGET's bank at
   `first_k_dense_replace` -- so at runtime it would have indexed the offload cache's per-layer
   arrays out of bounds. Added `moe_layer_id` to `Qwen4ExpDecoderLayer` to alias the two, and
   adjusted the engine's count assert to expect one extra (aliased) layer when MTP is registered.
4. A real per-token accounting bug in the new spec loop itself: `run_spec_step` pre-set
   `req.cached_len`/`device_len` to the WHOLE verify window's end before `_commit_spec_tokens`'
   per-token EOS/stop/length loop ran, so `hit_length` (which reads live `device_len` via
   `req.can_decode`) saw the window's FINAL length for every token in it instead of each token's
   own position. Fixed: `_commit_spec_tokens` now takes `start_pos` and advances
   `cached_len`/`device_len` one token at a time, exactly like `complete_one()`.

After all four fixes, the server serves real requests without crashing. A short greedy prompt
("The capital of France is", max_tokens swept 1/2/3/5/32) matched the `--spec-mtp 0` baseline's
token content exactly at every length tested -- but a longer, more complex prompt ("Write a
short poem about the ocean...", max_tokens=80) **diverges from the baseline in actual content**,
not just length: both start identically, then differ starting around token 13 (`"sea".` vs
`"sea."` and completely different continuations after). Content divergence under greedy
decoding with an `accept_drafts` contract that only ever commits target-verified tokens means
the verify forward itself (a `phase="prefill"` `Batch` over a 2-token window) is computing
something numerically different from what a real decode step at those same positions would
compute -- not a bookkeeping bug. Prime suspect per the Opus review's own flagged risk (EXP-025
point 8): the QSA `pending_ring`/compressed-group logic keyed by `position % index_ratio` may
behave differently for a prefill-phase multi-position extend than for true decode, corrupting
attention for prompts whose length interacts with `index_ratio` boundaries -- untested at the
short prompt's length, triggered at the longer one's.

**Item E status change: PARTIAL -> BLOCKED on a confirmed correctness bug.** Do not claim
target-equivalence past this point. Next step: escalate this exact repro (both prompts, both
outputs, this file) to Opus 5 for a third architecture pass focused specifically on the QSA
attention path's behavior under a `phase="prefill"` multi-token verify window, before any further
implementation or any throughput measurement.

## EXP-028 — Native MTP: content-correctness CONFIRMED live (fix validated)
**Date:** 2026-09-17 · **Verdict:** **MEASURED / KEEP**

Root cause of EXP-027's divergence found by Opus 5's third pass and fixed (see the fix commit
for full detail): `run_spec_step`'s reject-path replay forward temporarily rewrote
`req.cached_len`/`device_len` to re-derive the next chain's seed residual, and never restored
them -- the step ended one token behind reality, so the next spec step re-drafted an
already-committed token and re-applied its GDN update onto a recurrent+conv state that already
had it, permanently corrupting the state from the first rejection onward. Also fixed:
`LinearStatePool.copy_from` used `req.linear_slot_idx`, which is `None` under `--cache-type
naive` (only hybrid-radix allocates it) -- added `_linear_slot()` to key by `table_idx` under
naive, matching `build_fla_metadata`'s existing convention. Also fixed a latent k>=2 bug: the
draft chain never advanced `cached_len`/`device_len` per step, so draft position i>=1 couldn't
see its own prior draft-step KV (`prepare_metadata` always saw `seqlens_k=d`). Confirmed NOT the
QSA `pending_ring`/`index_ratio`-boundary risk EXP-025 had flagged as top suspect -- that path
already handles a non-aligned `cached_len` correctly (traced with file:line evidence).

Retest against the real Flash-Next checkpoint (`--spec-mtp 1`, k=1, greedy, 16K, naive cache,
single request): both the short prompt ("The capital of France is", `max_tokens` in
{1,2,3,5,32}) and the longer prompt that previously diverged ("Write a short poem about the
ocean...", `max_tokens=80`) now produce **content byte-identical** to the `--spec-mtp 0`
baseline at every length tested. The only remaining difference from baseline is the
pre-existing, MTP-unrelated `max_tokens` vs `completion_tokens` off-by-one (EXP-027 first
observation, reproduced independently without `--spec-mtp`) -- MTP returns exactly
`max_tokens` tokens where the baseline convention returns `max_tokens - 1`; content matches
regardless, so this is a display/counting nuance, not a correctness issue, and is out of this
campaign's scope to chase further.

**Item E status: draft/verify/rollback loop now content-correct for k=1 on real hardware.**
Still open: per-step acceptance-rate logging didn't surface in the server log (added
`logger.info` in `run_spec_step`, but it did not appear in output despite the spec path
demonstrably running -- minor observability gap, not a correctness blocker, needs a follow-up
look at scheduler-subprocess log routing). k=2/3 not yet live-tested (the latent bug above was
fixed by inspection, not by an executed k>=2 run). No throughput (PP/TG/acceptance-rate)
measurement has been taken -- that is the next step, and only after it can the goal's
target-equivalence and MTP preflight gates be marked satisfied.

## EXP-029 — MTP host-RAM OOM during benchmarking: confirmed pre-existing margin, not a leak
**Date:** 2026-09-17 · **Verdict:** **INFORMATIONAL / NOT A BUG**

Repeated `earlyoom` SIGTERMs while benchmarking `--spec-mtp 1` (exitcode=-15, during expert-bank
loading, ~190/192) looked at first like an MTP-specific regression. Measured directly instead of
retrying blind: sampled the scheduler subprocess's RSS every 1s through a full `--spec-mtp 1`
load. Peak RSS was **68.37 GiB**, transiently dropping system-available memory to 9.79% for
about one second (earlyoom's SIGTERM threshold is 10%) before settling to a 66.6 GiB steady
state with ~11% available. This peak is statistically identical to the plain baseline's already
-measured 68.4 GiB RSS (`bench_pp_tg.py`'s own baseline run) -- MTP does not use meaningfully
more host RAM. The failures are a timing race between earlyoom's poll interval and this
pre-existing, already-documented (EXP-015) razor-thin margin, not a leak or regression
introduced by this session's MTP work. Equally possible for a baseline run that happens to get
unlucky. Operator rule going forward: at most 2 retries on any repeating failure, then measure
instead of retrying again -- this entry is the result of following that rule.

## EXP-030 — moe_cache_auto silently downgraded an explicit --num-tokens instead of refusing
**Date:** 2026-09-17 · **Verdict:** **FIXED / KEEP**

Found while trying to benchmark MTP's PP/TG at 16K: `--spec-mtp 1 --num-tokens 16576` started
cleanly, logged "Allocating 8256 tokens for KV cache" (HALF of the requested 16576, and half of
what `--spec-mtp 0` gets with the identical flag), and only surfaced the shortfall as a confusing
`torch.OutOfMemoryError` deep inside a GDN kernel once a real 16384-token prompt was actually
prefilled. Root cause: `Engine._resolve_auto_moe_cache_size` fed `vram_ledger.decide()` a
`kv_reserve_tokens` floor of `max(config.kv_reserve_tokens, min_reserve)` -- using only the
`--kv-reserve-tokens` flag (default 8192) as the KV floor for the MoE/KV split, never looking at
an explicit `--num-tokens`/`--num-pages` override at all. The override was correctly *preserved*
after the split (`if config.num_page_override is None: ...`), but the split itself, run first,
had already decided how much VRAM to hand to experts based on the wrong (smaller) floor, and once
an MTP-registered layer's extra weights/index slab made the budget tighter than the `--spec-mtp 0`
case, "smaller floor, more experts" resolved to a KV allocation below what the user actually asked
for and the request actually needed.

Fix: fold `num_page_override * page_tokens` into the `kv_reserve_tokens` floor passed to
`decide()`. `decide()` already asserts ("cache budget too small: minimum plan ... needs X > budget
Y") when the resulting plan can't fit the budget -- so an unfundable `--num-tokens` now fails
loudly at startup through that existing guard instead of silently under-provisioning and OOMing
mid-request. No-op when no override is given. New regression test:
`tests/engine/test_cache_budget.py::test_engine_resolve_auto_moe_cache_size_refuses_undersized_num_tokens_override`.
845 passed, 63 skipped (2 pre-existing unrelated failures) across the focused suite.

## EXP-031 — Host-RAM OOM during MTP benchmarking: root-caused with Opus 5, not a leak
**Date:** 2026-09-17 · **Verdict:** **FIXED / KEEP (environment, not code)**

EXP-029 (same session, same symptom) concluded "pre-existing razor-thin margin, retry at most
twice" -- true but incomplete. The operator pushed back ("only 8-9GB RAM in use, not a RAM
problem") after seeing `free -h` post-crash, and asked for a full investigation via Opus 5 rather
than accepting the first explanation. That investigation (read-only shell diagnostics + engine
source review) found:

1. **The operator's post-crash `free -h` and the crash itself are both true, no contradiction**:
   `journalctl -u earlyoom` at the exact kill timestamp shows `VmRSS 69428 MiB` for the dying
   process and `mem avail: 7580 of 77673 MiB (9.76%)` -- the peak is real and transient; `free -h`
   run afterward measures a different moment, after those ~69 GiB were already freed by the
   SIGTERM. Six independent kills all showed VmRSS 69.0-69.6 GiB.
2. **The "~15.7 GiB gap" between earlyoom's reported total (~77.6 GiB) and true `MemTotal`
   (~93.4 GiB) is not a bug or a hidden reservation** -- `earlyoom --dryrun -r 1` prints `mem
   total` and `user mem total` as two DIFFERENT, correctly-labeled numbers; earlyoom's threshold
   check uses the second (live-recomputed, nets out shmem/tmpfs), confirmed by the journal
   showing that denominator moving across a huge range (2279 -> 81632 MiB) that `MemTotal`
   physically cannot. Practical consequence: **every 1 GiB parked in `/tmp` (a 46 GiB tmpfs)
   costs ~0.9 GiB of the 10% SIGTERM margin**, not a neutral 1 GiB of a fixed pool.
3. **Swap is 0** (`SwapTotal: 0`), and earlyoom only fires when memory AND swap are both below
   threshold -- with no swap, the swap half of that condition is permanently satisfied, so the
   memory trigger is effectively unguarded compared to a swap-enabled host.
4. **The expert-bank loader itself is already tight, nothing to patch for RAM**: private
   anonymous mmap (EXP-020), `drop_page_cache` before AND after every shard
   (`nvfp4_banks.py:114-123`), pin-after-fill instead of a redundant zero-fill pass
   (`host_banks.py:1-17`), chunked O_DIRECT reads with `posix_fadvise(DONTNEED)` bypassing page
   cache entirely for bulk reads. The 68-70 GiB peak is the 48x512 NVFP4 bank set itself
   (EXP-015's own 63.46 GiB price) plus runtime overhead -- real and irreducible without a
   different residency strategy (streamed/bounded loading, future work, not this session).
5. **The kills missed the threshold by only 78-187 MiB out of 93 GiB (~0.1-0.2%)**, while ~3.3
   GiB of stale, unrelated tmpfs artifacts (old llama.cpp build trees, an npm cache, a benchmark
   log from 2026-09-16 tool sessions unrelated to this campaign) were sitting in `/tmp` the whole
   time -- 16-38x the deficit that actually mattered.

**Fix applied**: moved the ~3.3 GiB of stale `/tmp` artifacts to `/models/backup/pytest-of-natal/
tmp-leftovers-20260917/` (disk-backed, `natal`-writable; `/models/backup` itself is root:root and
needs `sudo mkdir`+`chown` per subfolder -- already done for this path). `systemctl mask
tmp.mount` (done earlier this session, confirmed `systemctl is-enabled tmp.mount` -> `masked`,
symlinked to `/dev/null`) makes this permanent from the next reboot onward: `/tmp` will no longer
mount as tmpfs at all, closing this entire failure class structurally rather than requiring
per-session cleanup.

**Secondary finding, flagged not fixed** (out of scope, a speed issue not a RAM one):
`python/freetoken/moe/expert_banks.py:352`'s `_host_ram_fits_parallel` guard compares available
RAM against the checkpoint's TOTAL file size (125.9 GiB) instead of the actual bank size (63.46
GiB) -- on this host that guard can never pass, so the parallel expert-loading path is
permanently unreachable (always silently falls back to serial, a load-time regression, not a
correctness or RAM one). The same guard separately budgets the transient peak as one shard when
`models/weight.py:86` documents `(prefetch+1)` = 3 shards (largest 9.99 GiB) -- fixing only the
bank-size half without the transient half could newly allocate ~30 GiB of anonymous whole-shard
buffers on top of the banks. Fix both together if parallel loading is ever revisited; not done
this session.

## EXP-032 — Split Turbo4 path runs on the real Flash-Next checkpoint

**Date:** 2026-09-18 · **Verdict:** **DISCOVERY PASS / NOT A CERTIFICATION ROW**

Setup: RTX 5080, real host, `--kv-format turbo4`, `--memory-ratio 0.86`, `--cuda-graph-max-bs 0`,
`FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`, naive cache, 16K prompt, 16 generated tokens, one
warmup and one measured repeat. Result: **PP 1713.7 / TG 24.72**, TTFT 9560.6 ms, VRAM 14.86
GiB, GPU 99%, RSS 69.8 GiB, output sha1 `17f277f43565`.

This confirms that the separate Triton decompression kernel, bounded workspace and existing QSA
attention path execute end to end on the target checkpoint. It is not comparable to the certified
CUDA-graph/overlap baseline and is not a repeated A/B gate. The graph/overlap path remains under
investigation.

## EXP-033 — MTP=1 + Turbo4 reaches speculation; finished-page cleanup exposed

**Date:** 2026-09-18 · **Verdict:** **BLOCKED / FIX PATCHED, NOT RE-TESTED**

After fixing MRoPE positions for the manually assembled draft batch and aligning the
`ForwardOutput.copy_done_event` field, the real 16K MTP=1 + Turbo4 probe reached speculative
decode and logged `k=1 m=2 accepted=1/1`. It then failed the idle cache check with one missing
page: `free_pages(258) + cache_pages(0) != num_pages(259)`.

Root cause: a verify window can allocate the page containing its correction token, while finished
request cleanup released only through `cached_len`. The cleanup now also releases the allocated
`device_len` page tail. The patch is committed with focused tests already passing before this last
cache-tail edit; a post-edit focused test and live PP/TG run remain pending.

## EXP-034 — EXP-033's `alloc_end` patch was itself wrong; the real leak was three separate bugs

**Date:** 2026-09-18 · **Verdict:** **f9354cd's cache.py fix REVERTED; three different real bugs
found and fixed instead; live MTP=1 + Turbo4 now runs to completion with no crash**

Session start: re-ran the full non-slow suite before touching anything live. `tests/scheduler/`
alone: **11 pre-existing failures** (`test_abort_inflight_prefill.py`, `test_dsv4_generic_manager.py`,
`test_scheduler_chunked_prefill.py`, `test_swa_pagesize.py`), all `free_pages+cache_pages !=
num_pages`, all **over-counted** (too many free, not too few). Bisected: these regressed with
f9354cd's own `cache.py` change (checked out `f9354cd~1`'s `cache.py` alone, same tests pass).
Root cause of THAT regression: `cached_len < device_len` is a **standing invariant** in this
codebase (device_len is always the next, not-yet-written slot -- see `Req.__init__`'s own assert
and `complete_one()`), not a spec-only artifact. f9354cd's `alloc_end =
div_ceil(max(cached_len, device_len), page_size)` collapses to `device_len` always, and at
`page_size=1` (these tests) that unconditionally frees one MORE page than `_padded_tail`'s
original `cached_len`-only bound -- a page never allocated to the request in the first place.
**Fix: reverted all three `alloc_end` hunks in `cache.py` back to `_padded_tail`/no-op** (the
pre-f9354cd code). Full `tests/scheduler/` green again (104 passed) before proceeding.

With `cache.py` reverted, the ORIGINAL EXP-033 symptom (`free_pages(258)+cache_pages(0)!=
num_pages(259)`, live, RTX 5080, `--spec-mtp 1 --kv-format turbo4`, 16384-token prompt = exactly
256 pages at page_size=64) was investigated from first principles instead of patched again at the
same spot, and turned out to be **three independent bugs**, all specific to `spec.py`, all masked
by the fact that no unit test ever exercised a page-ALIGNED prompt length (256*64) with k>=1:

1. **`_commit_spec_tokens`'s mid-window finish never reclaimed the verify's surplus allocation.**
   `_free_req_resources` (called mid-loop, on `finished_now`) recycles `table_idx` using only the
   truncated `cached_len`/`device_len` at the finish point -- it has no way to know the verify
   batch's own `_prepare_batch` allocated further out (`d+k`). Fix: `_commit_spec_tokens` now
   takes `spec_alloc_len` and calls `free_spec_reject` itself, BEFORE `_free_req_resources` frees
   `table_idx` -- after that, `free_spec_reject(req, ...)` would read `page_table[-1]` (Python
   negative indexing, not an error) and silently corrupt a DIFFERENT request's row instead.

2. **`free_spec_reject`'s `keep_len` argument used the wrong convention at run_spec_step's two
   call sites.** `req.cached_len` always LAGS the last real, committed token by one (the standing
   invariant from #above) -- `free_spec_reject`'s own contract (proven by
   `test_reject_crossing_a_page_boundary_frees_the_speculative_page`) needs a boundary where index
   `keep_len` itself is already fully speculative, i.e. `req.device_len` (`keep_len+1`), not
   `req.cached_len`. Passing the lag value only differs from the correct one at an EXACT page
   boundary -- which a 16384-token prompt at page_size=64 hits on its very first spec step
   (`d-1=16384`) -- where it wrongly handed back the page still holding the just-committed,
   real correction token (an OVER-free of a live page, not an under-free of a dead one).

3. **The GDN-state "replay" (undoing draft KV, re-forwarding committed tokens as a plain prefill)
   re-ran `allocate_paged` over a range whose pages were already assigned by the verify step's own
   `_prepare_batch` moments earlier.** `allocate_paged` has no memory of a prior call; calling it
   twice on an unchanged `(cached_len, device_len)` pair that straddles a page boundary grabs a
   SECOND, different physical page and overwrites the `page_table` row that held the first one --
   silently orphaning it (never in `free_slots`, never referenced again). This is what actually
   produced the crash: at page_size=64, a 16384-token prompt makes `d-1` land exactly on the page
   256/257 boundary on the request's very first spec step, and every rejected-draft replay after
   that re-triggers the same double-allocate on the SAME row until the request finishes and the
   orphaned page surfaces as `free_pages(258)+cache_pages(0)!=num_pages(259)`. Fix: `_prepare_batch`
   gained a `skip_alloc: bool` parameter (default `False`, every other call site unaffected); the
   replay call passes `skip_alloc=True`.

4. **A fourth, unrelated leak surfaced once the crash stopped happening**: `LinearStatePool
   exhausted: need 1, have 0` on the benchmark's 2nd repeat. `_spec_snapshot_slot` allocates a GDN
   snapshot slot per `req.uid`, freed only by `run_spec_step`'s own `if finished:` branch -- but
   `_spec_eligible_req` disables spec on a request's OWN LAST token (`remain_len <= 1`), so every
   request's true finish goes through the PLAIN (non-spec) decode path instead, which never calls
   `free_spec_snapshot_slot` at all. 100% leak rate, one request finishing = one dead slot forever.
   Fix: moved the release into `_free_req_resources`, the one cleanup path every finish route
   (abort, plain decode, spec) shares, guarded by `hasattr` for scheduler-unit-test stubs that
   don't mix in `SchedulerSpecMixin`.

Each fix has a dedicated regression test in `tests/scheduler/test_spec_reject_frees_pages.py`,
confirmed to fail on the pre-fix code and pass after (`git stash`/manual revert + rerun, not just
inspection). Full non-slow suite: **1934 passed, 1934 passed both times** (the 1 failure is the
pre-existing `flashinfer` `nvcc`/`ptxas` alignment error on this host's CUDA 13.3, unrelated,
already documented).

**Live result** (RTX 5080, real serve, no sandbox, `TMPDIR=/models/desenvolvimento/tmp`):
`--spec-mtp 1 --kv-format turbo4 --memory-ratio 0.86 --cuda-graph-max-bs 0 --cache-type naive
--num-tokens 16576`, `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`, 16384-token prompt, greedy. Two
repeats after 1 warmup now run to completion with **no crash**: PP 1715.2 (matches the MTP-free
Turbo4 anchor of ~1714-1716, EXP-032), but **TG collapsed to 0.79 tok/s mean (0.52-1.07 tok/s
across the two repeats)** against the MTP-free Turbo4 baseline of TG 24.6 (same run, same host,
same config, `--spec-mtp` simply omitted) -- roughly a **31-50x** decode slowdown, ITL p50
1.2-2.4 **seconds** per token at 99.7% GPU util. This is a real, valid PP/TG measurement (goal
priority 3), not a crash -- but it is not a usable number yet: at 99% GPU util the cost is real
compute, not idle waiting, consistent with `--cuda-graph-max-bs 0` (CUDA graph is disabled for
`--spec-mtp` today) forcing every draft+verify step through eager, likely-unfused Triton
launches instead of a captured replay. Ties directly into goal priority 4 (CUDA graph/overlap for
MTP) -- not yet investigated this session.

**New, separate, NOT YET FIXED finding**: the two repeats' `output_sha1` **differ** from each
other (`2d8355daf32e` vs `6e43ad8484ac`) despite identical prompt/config/greedy sampling on the
same warmed server. Isolated: two INDEPENDENT fresh-server single-request runs (decode=32, no
warmup) produce IDENTICAL sha1 (`f423b6a1bade` both times) with near-identical PP/TG -- so a
single request's own decode is fully deterministic and reproducible. The divergence only appears
across SEQUENTIAL requests in the same server session (warmup -> repeat1 -> repeat2), pointing to
residual state leaking from one finished request into the next one's initial state somewhere in
the MTP/spec path (candidates: `_last_residual`, the reused `table_idx`/GDN slot's leftover
content, or a page_table row not fully re-initialized before reuse). **Do not trust any
multi-request MTP+Turbo4 benchmark's content-correctness until this is root-caused** -- the
PP/TG throughput numbers above are still valid (they only measure timing), but the "byte-identical
to spec-mtp=0" correctness bar from EXP-027/028 has NOT been re-confirmed for Turbo4 at 16K, and
this session found it does NOT hold across repeated requests. Next session: reproduce with
`--repeats 2 --warmups 0` (no warmup) to see if 2 is already enough to diverge, then diff the
two repeats' generated token sequences to find the first divergence point, then trace what request
state that position's draft/verify step reads that isn't reset between requests.

## EXP-035 — Opus 5 review reframes both open bugs; Rung 1 timing localizes the TG cost

**Date:** 2026-09-18 · **Verdict:** **Two structural findings, not yet fixed; session paused by
operator to continue on a different model**

Consulted an Opus 5 subagent (read-only code review, no changes) to plan execution of goal
priorities 4-6. Its key finding, from reading the code (not speculation): `python/freetoken/
models/qwen4_exp/gdn.py:154` branches the Gated DeltaNet layer on `batch.is_decode` -- decode
uses `gdn_decode_fla` (fused recurrent, gating computed in-kernel), prefill uses
`gdn_prefill_chunk_fla` (chunked delta-rule, `CHUNK_SIZE=64` in `kernel/fla/chunk.py:28`, no
small-T short-circuit, gating precomputed in Python). `scheduler/spec.py` builds every MTP
forward (draft steps, verify, GDN replay) as `Batch(reqs=[req], phase="prefill")` -- so MTP runs
a structurally different GDN kernel than the no-MTP baseline, even at T=1 where no batching
occurs. This reframes the single "non-determinism" finding from EXP-034 into two bugs: **Bug A**
(content inequivalence, deterministic, caused by the kernel swap) and **Bug B** (cross-request
divergence, non-deterministic, needs a state carrier not cleared between requests -- leading
hypothesis: the QSA pending ring, `qsa_sparse.py:419/433`, read by a subsequent forward when the
spec path's `cached_len` rewind at `spec.py:220` isn't `index_ratio`-aligned).

Also: the previous session's "Rung 2" TG hypothesis (`cache_req(finished=False)` at `spec.py:256`
costing O(context) per token via radix insertion) is **checked and dead** -- under
`--cache-type naive` (the benchmarked config) `NaivePrefixCache.insert_prefix` is a constant-time
no-op (`kvcache/naive_cache.py:29-30`); under `hybrid_radix` the hybrid path never even sets
`mamba_last_track_seqlen` from a spec-sized forward (`attention/linear.py:119`, `c < 1` for any
`extend_len <= 64`). Do not re-derive this; it does not apply to either code path actually
exercised.

**Rung 1 (per-phase timing) implemented and run live** (RTX 5080, `--spec-mtp 1 --kv-format
turbo4`, k=1, `--decode 16`, `FREETOKEN_DEBUG_SPEC_TIMING=1`, new instrumentation in `spec.py`
following the existing `qsa_sparse.py:292-304` `mark()` pattern). Result, consistent across
every step in the run:

| phase | time |
|---|---|
| `verify_forward` (T=k+1=2) | ~1.191-1.195 s |
| `gdn_replay` (T=committed, usually 1) | ~1.192-1.195 s |
| `draft_chain` (T=1 per draft) | ~0.051 s |
| `snapshot`, `verify_prepare_batch`, `commit`, `free_spec_reject`, `cache_req` | <0.001 s each |

`verify_forward` and `gdn_replay` dominate and cost almost EXACTLY the same despite different T
(2 vs 1) -- the cost does not scale with batch size, which points at something scaling with
`cached_len` (~16K at this point in the benchmark) inside `chunk_gated_delta_rule`/its chunk-
offset preparation, not with the tiny `extend_len` MTP actually needs. Not yet traced further
(session paused here by the operator, to continue this specific thread on a different model to
conserve Claude usage). Next: read `chunk_gated_delta_rule` and `prepare_chunk_offsets` for any
computation over the full `cu_host`/cached-length range instead of just the new tokens, and diff
the SAME timing marks between a `--decode 4` cold step and a warm step (the two known regimes) to
see whether the ~1.19s figure itself is regime-dependent or constant regardless of warm/cold.

## EXP-036 — Offline GDN kernel equivalence: Bug A confirmed and quantified

**Date:** 2026-09-18 · **Verdict:** **Confirmed** (`test_decode_prefill_gdn_kernel_inequivalence` in `tests/models/qwen4_exp/test_gdn.py`)

Quantified the structural discrepancy between `gdn_decode_fla` (fused decode kernel) and `gdn_prefill_chunk_fla` (chunked prefill kernel, `CHUNK_SIZE=64`) on identical inputs and initial state:
- T=1 vs T=1 (step 1 / 8 / 64):
  - op output max abs delta: 1.46e-3 / 1.95e-3 / 1.95e-3
  - core_out max abs delta: 1.91e-6 / 5.72e-6 / 7.63e-6
  - recurrent state max abs delta: 5.43e-4 / 9.48e-4 / 1.43e-3
  - conv state max abs delta: 0.00e+00 (identical at T=1)
- T=1 sequential decode vs Prefill T=2 / T=4 (MTP verify shape):
  - conv state max abs delta: 1.95e-3 (step 8) / 9.77e-4 (step 64)
  - op output max abs delta: up to 3.91e-3 (T=2, step 64) / 2.93e-3 (T=4, step 64)
- Impact on greedy argmax: across 640 steps (10 trials x 64 tokens), the numerical delta between decode and prefill paths flipped the greedy argmax logit 1.88% of the time for T=1, 1.41% for T=2, and 1.25% for T=4.
Conclusion: Bug A (deterministic content inequivalence between baseline decode and MTP verify/draft) is structurally real and verified offline.

