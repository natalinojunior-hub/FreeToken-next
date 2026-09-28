# Campaign 36 — MTP end-to-end audit

Date: 2026-09-28  
HEAD: `ec1519530147bfffdc737bd17640bbb8085f0865` (`next`)

## Result

Raw k0 remains closed by campaign 35. The current MTP implementation is correctness-tested but is **NOT READY for production hardening**: no fresh post-campaign-35 physical MTP baseline was obtainable from the harness because the local checkpoint has no usable tokenizer backend, and the serving depth controller is not adaptive in production.

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

Fresh HEAD preflight (IQ3_XXS, 4,104 prompt tokens, 16 generated, graph off, one repeat) completed with the fixed prompt file: k0 **47.93 TG**, ITL p50/p95 **20.39/32.40 ms**, VRAM **5.14 GiB**, SHA `238f7db579b7`; k1 **29.36 TG**, ITL p50/p95 **38.05/57.92 ms**, VRAM **13.45 GiB**, same SHA. The server trace showed about 3 accepted draft tokens in 13 k=1 cycles (~23%), so speculation loses materially to k0. This is a short preflight, not long-context certification. Historical runs were worse (k0 35.36 TG; k1 22.43 TG at 55.5% acceptance); own MTP expert banks were fixed later (`a9d30f4`), but no throughput win followed.

## Closure and limits

- Raw k0: CLOSED/WIN. Campaign-35 certification covers 16K, 64K, 128K and ~256K, LRU-3, adaptive-VRAM, SHA and pressure gates.
- MTP k selection: OPEN. No online acceptance/cost/VRAM hysteresis controller exists; production serving uses fixed `base_k`.
- MTP physical baseline: measured at 4K only; k1 is a loss (29.36 vs 47.93 TG) with much higher VRAM and ITL. 16K/64K/128K/~256K MTP certification remains OPEN.
- State transaction: correctness helpers pass, but complete accepted-boundary proof for PLE/QSA/GDN/linear state remains unproven; replay is still required on the non-zero-replay path.
- Warm-prefix AD divergence: documented workaround remains `--cache-type naive`; exact GDN boundary recomposition is not implemented.

## Verification

Focused MTP/graph/launch/profile tests: **17 passed**. Full `make ci`: **2311 passed, 209 skipped, 17 deselected**; format, lint and mypy also passed.

## Production decision

**NOT READY.** The k0 engine is production-ready under the campaign-35 contract. The serving default now stays at k0 when no local measured profile exists; a stored profile can still opt into MTP. MTP needs long-context physical certification, an online economics controller with hysteresis and safe k0 fallback, and an end-to-end accepted-state proof before it can be enabled automatically.
