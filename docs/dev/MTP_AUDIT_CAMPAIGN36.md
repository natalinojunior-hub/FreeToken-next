# Campaign 36 — MTP end-to-end audit

Date: 2026-09-28  
HEAD: `ec1519530147bfffdc737bd17640bbb8085f0865` (`next`)

## Result

Campaign 35 records raw-k0 closure. Campaign 36 remains **IN PROGRESS**: the serving depth controller is not adaptive, AD warm-prefix containment is pending, and the full requested correctness, performance and certification gates are not complete. Initial preflights did run after selecting the actual GGUF directory; the parent-directory tokenizer error was a path error, not an environmental blocker.

## Execution map

`SchedulerSpecMixin.run_spec_step` (`python/freetoken/scheduler/spec.py:407`) performs:

| stage | current behavior | evidence / limit |
|---|---|---|
| eligibility | exactly one greedy request; overlap disabled | `spec.py:101`, `scheduler.py:160` |
| draft | `k` sequential MTP forwards; draft graph optional | `spec.py:471-501`, `engine/graph.py:171-461` |
| verify | one target prefill over deferred rows + drafts | `spec.py:511-527` |
| acceptance | host prefix loop, correction token retained | `engine/spec.py:23-30` |
| state | snapshots QSA/PLE/linear state; rollback/replay on rejection | `spec.py:454-467`, `601-624` |
| synchronization | mandatory `copy_done_event.synchronize()` each cycle | `spec.py:526` |
| graphs | separate captures per verify shape; eager fallback on capture failure | `engine/graph.py:64-71`, `347-461` |
| controller | serving path returns configured `base_k` unchanged; legacy tests only cover context heuristic | `scheduler/adaptive_mtp.py:29-40`, `tests/scheduler/test_adaptive_mtp.py` |

The draft-KV warmup helper exists (`spec.py:58-99`) but has no scheduler caller. Enabling it unconditionally would add a full prompt MTP pass; it remains an unproven tradeoff.

## Historical physical measurements

Initial preflight (IQ3_XXS, 4,104 prompt tokens, 16 generated, graphs enabled by default, one repeat, linear-state-cache-ratio 2) completed with the fixed prompt file: k0 **47.93 TG**, ITL p50/p95 **20.39/32.40 ms**, allocator-reserved VRAM **5.14 GiB**, SHA `238f7db579b7`; k1 **29.36 TG**, ITL p50/p95 **38.05/57.92 ms**, allocator-reserved VRAM **13.45 GiB**, same SHA. These VRAM figures exclude VMM expert residency and are not physical GPU peaks. The visible log contains 12 k1 cycles, with 3 accepted drafts (25%); the logs and structured rows were produced in an ephemeral context-mode sandbox and are not retained. Consequently these are historical short-run observations, not reviewable canonical performance evidence. Canonical 16K/256 runs with ratio 0 and persistent evidence are being collected under `/models/desenvolvimento/old/freetoken-next/campaign36-mtp/resume/`.

## Closure and limits

- Raw k0: CLOSED/WIN. Campaign-35 certification covers 16K, 64K, 128K and ~256K, LRU-3, adaptive-VRAM, SHA and pressure gates.
- MTP k selection: OPEN. No online acceptance/cost/VRAM hysteresis controller exists; production serving uses fixed `base_k`.
- MTP physical baseline: measured at 4K only; k1 is a loss (29.36 vs 47.93 TG) with much higher VRAM and ITL. 16K/64K/128K/~256K MTP certification remains OPEN.
- State transaction: correctness helpers pass, but complete accepted-boundary proof for PLE/QSA/GDN/linear state remains unproven; replay is still required on the non-zero-replay path.
- Warm-prefix AD divergence: fixed by `e7d756b`; prefix checkpoints now retain the fp32 accumulator instead of copying the rounded bf16 output workspace. AD cold/warm 2K/32 both match the correct cold/naive SHA `2146e3adeb5c` while retaining reuse.

## Verification

Focused MTP/graph/launch/profile tests: **17 passed**. Full `make ci`: **2311 passed, 209 skipped, 17 deselected**; format, lint and mypy also passed.

## Canonical resumption measurements

Persistent evidence directory: `/models/desenvolvimento/old/freetoken-next/campaign36-mtp/resume/`.
All runs below use the actual IQ3_XXS GGUF directory, 16,384 prompt tokens, 256 greedy output
tokens, graphs enabled, `--linear-state-cache-ratio 0`, stats enabled, and no pool-caps override.

| path | cold TG | warm TG | SHA | status |
|---|---:|---:|---|---|
| `k0.jsonl` | 64.506 | 65.566 | `76a5508fd576` | raw baseline revalidated |
| `k1.log` | — | — | — | failed: rollback snapshot allocation exhausted GDN pool |
| `k1-floor.jsonl` | 56.327 | 55.367 | `76a5508fd576` | passes after working-set sizing fix |
| `c36-k1-fixed` | 61.34 | 59.27 | `76a5508fd576` | post-VMM fix fixed depth k=1 (mean 60.30 TG) |
| `c36-k2-fixed` | 53.41 | 51.38 | `76a5508fd576` | post-VMM fix fixed depth k=2 (mean 52.39 TG) |
| `c36-k1-adaptive` | 60.90 | 62.26 | `76a5508fd576` | adaptive depth controller k=1 (mean 61.58 TG) |
| `c36-k2-adaptive` | 60.00 | 61.21 | `76a5508fd576` | adaptive depth controller k=2 (mean 60.61 TG) |
| `c36-k1-graph-draft` | 61.05 | 62.37 | `76a5508fd576` | draft-graph + adaptive controller (mean 61.71 TG) |
| `c36-k0-raw` | 64.88 | 65.92 | `76a5508fd576` | raw baseline (mean 65.40 TG) |

`886046c` fixes the pool floor for both naive and hybrid caches: MTP paths without native
per-row state buffers need one additional non-evictable rollback slot per request. The same
floor is consumed by initial pool sizing, the VRAM planner and runtime rebuild validation.
Focused allocator/planner/rollback tests: 42 passed; ruff and mypy passed. Real k1 cold and
warm requests now complete, while raw k0 sizing stays unchanged. A final full CI after all
campaign changes remains required.

The configured-depth resolver now also clamps drafts to `remain_len - 1`, preventing a
verify window from extending beyond the remaining output budget. Focused scheduler tests:
20 passed. The original depth sweep completed. The user narrowed active scope to k1–k4,
with k0 reference/fallback; k5/k6 artifacts are historical only. Economics control remains
unimplemented. The ZIP from earlier turns is an interim snapshot, not the final requested
closure artifact.

## Verification & Profiling Reduction

- Full `make ci`: **2354 passed, 209 skipped, 17 deselected, 0 warnings/failures**; format, lint, and mypy passed clean.
- CUDA software profiles reduced from Nsight traces (`k0`, `k1`, `k2`):
  - In `k0`, execution is bounded by target forward CUDA graph execution and event synchronization (`3586.48 ms` across 255 tokens, ~14.06 ms/tok).
  - In `k1`, `MTPDraft` adds 132.1 ms GPU kernel duration, while `MTPReplay` adds 213.6 ms and 177.9 ms sync overhead on rejected paths. Target verify window execution adds 4841.2 ms, leading to higher overall cycle latency than raw single-token generation.
  - In `k2`, `MTPDraft` kernel duration increases to 205.5 ms, while `MTPReplay` scales up to 1165.1 ms with 970.7 ms runtime synchronization, explaining the collapse in fixed k=2 throughput to 52.39 TG.
- Policy Gate: Per mandate, because MTP k>=1 does not exceed raw k0 throughput (65.40 TG vs 61.71 TG max), long-context certification sweeps (64K, 128K, 256K) for MTP are **OMITTED** to avoid unneeded resource expenditure on a sub-optimal decode configuration.

## Production decision

**NOT READY.** The k0 engine is production-ready under the campaign-35 contract (65.40 TG on 16K/256).
On RTX 5080 with ISTA 16K/256, MTP speculation incurs a physical net loss:
- Fixed k=1: 60.30 TG (-7.8% vs k0)
- Fixed k=2: 52.39 TG (-19.9% vs k0)
- Adaptive k=1: 61.58 TG (-5.8% vs k0)
- Adaptive k=2: 60.61 TG (-7.3% vs k0)

The verification forward overhead and draft chain overhead exceed the latency savings of accepted drafts on this memory/compute geometry. The serving default MUST remain k0 (`--spec-mtp 0`). MTP should only be activated with explicit opt-in or when higher batch sizes or faster draft models offset verification cost.
