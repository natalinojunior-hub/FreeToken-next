# relatorio-chatgpt-status-atual.md

## Evidence Package Next Master Prompt

### Current Architecture Actually Running

The engine initialization path currently executes the **legacy VRAM ledger** (`python/freetoken/engine/vram_ledger.py`) with an optional **multi-stage planner** (`python/freetoken/engine/memory_planner.py`) invoked only when `--moe-cache-auto` is enabled.

The memory planner is called **after** expert banks are loaded but **before** the final `OffloadMoeCache` and KV pool are created. The planner:
1. Measures physical baselines (Phase A)
2. Builds static cost model from geometries (Phase B)
3. Creates minimal probe pools (Phase C)
4. Measures post-mandatory snapshot (Phase A continued)
5. Runs runtime calibration with a probe prefill (Phase D)
6. Solves prefill chunk size via binary search (Phase F)
7. Solves expert slots and KV pages from residual (Phase G)
8. Constructs final pools (Phase H)
9. Runs final validation prefill (Phase I)

The legacy ledger path still runs in parallel for diagnostic logging; the planner's plan overrides `config.moe_cache_size`, `config.moe_prefill_overlap`, and `config.num_page_override`.

The planner code exists (1168 lines) and is invoked on the automatic path. The legacy ledger remains the default for manual `--moe-cache-size` / `--num-pages` configurations.

### Work Successfully Completed

| Capability | Status | Evidence |
|---|---|---|
| Multi-stage planner module (`memory_planner.py`) | Implemented | 1168 lines, compiles, CI passes |
| Phase A: Physical memory snapshots | Implemented | `take_physical_snapshot()` uses `mem_get_info`, `memory_allocated`, `memory_reserved`, peaks |
| Phase B: Exact static cost model | Implemented | Computes expert/slot, kv/page, GDN state, page table from geometries |
| Phase C: Minimal probe configuration | Implemented | Creates minimal expert cache, KV pool, linear pool, attention backend |
| Phase D: Runtime calibration probe | Implemented | Runs probe prefill, measures allocator/reserved/driver deltas |
| Phase E: Exact GDN sizing | Implemented | Calculates `h` tensor shape `B×NT×H×V×K`, compares to estimator |
| Phase F: Prefill chunk binary search | Implemented | Solves largest chunk fitting transient budget |
| Phase G: Expert cache + KV page solver | Implemented | Allocates from residual after fixed + required KV + transients |
| Phase H: Final pool construction | Implemented | Destroys probes, verifies reclamation, builds final pools |
| Phase I: Final validation prefill | Implemented | Runs real forward at chosen chunk, checks headroom |
| Engine integration | Implemented | `_init_offload_moe_cache` calls planner when `moe_cache_auto=True` |
| CI (lint, typecheck, tests) | Passing | 1957 passed, 206 skipped, 15 deselected |
| Expert geometry from `method.layout()` | Implemented | Uses kernel layout for exact per-slot bytes |
| GDN `h` allocation site identified | Documented | `chunk_delta_h.py:293`, 192 MiB, shape `B,NT,H,V,K` |

### Work Partially Completed

| Area | Current State | Gap |
|---|---|---|
| Automatic prefill chunk solver | Solves chunk but does not write back to scheduler | `config.max_extend_tokens` remains static default 8192 |
| Context-first allocation | Planner knows `max_seq_len` but expert solver can reduce KV below requested | Phase G clamps KV pages to `required_kv_pages` but residual may force reduction |
| Physical validation before readiness | Phase I runs validation but **scheduler worker crashes on OOM** | Validation catches OOM but worker still dies; no safe recovery |
| Automatic replan on validation failure | Skeleton exists (`if not valid: raise`) | No actual replan logic; raises `RuntimeError` |
| Memory-ratio removal | Planner uses physical budget directly | `memory_ratio` field/CLI still exists; legacy ledger still uses it |
| Heuristic constants | Planner uses some (Triton 128 MiB, workspace 128 MiB, frag 64 MiB) | Not measured; hard-coded in `RuntimeCalibration` defaults |
| CUDA Graph integration | Graph capture measured if enabled | In latest run `cuda_graph_max_bs=0` so not active |
| MTP integration | Planner accepts `prefill_overlap` but MTP k>0 not tested with planner | `spec_mtp` not passed to planner; no MTP-aware chunk sizing |
| Turbo3/Turbo4-specific planning | Pool class `kv_cost()` provides geometry | Planner uses it but no backend-specific reserve tuning |

### Work Not Started

| Area | Status |
|---|---|
| Persistent calibration profile (disk) | Not implemented |
| Complete allocator/driver accounting reconciliation | Not implemented |
| Turbo3 16K automatic path test | Not run |
| MTP k=1..6 with automatic planner | Not tested |
| CUDA graphs with automatic planner | Not tested |
| Scheduler `max_extend_tokens` as solver output | Not connected |
| Real 16K Turbo4 generation completion | Not achieved |
| Non-PyTorch memory accounting | Not implemented |
| Cross-rank TP memory validation | Not implemented |

### Current Most Important Blocker

**CRITICAL: First real prefill OOM at 192 MiB GDN `h` allocation** (file: `python/freetoken/kernel/fla/chunk_delta_h.py:293`)

- **Evidence**: `benchmarks/results/campaign-20260920/auto-context-16k-k0.log`
- **Driver free at failure**: 171.38 MiB
- **Failed allocation**: 192.00 MiB
- **Root cause**: Planner commits persistent pools (expert + KV) against a budget whose transient reserve model does not upper-bound the real first GDN prefill. The planner's Phase F solves chunk assuming transients fit, but Phase G then allocates expert/KV from residual, leaving effectively zero headroom for the actual first prefill.
- **Why it blocks certification**: The engine announces readiness ("API server is ready to serve"), accepts the request, then the scheduler worker crashes on the first real prefill. No complete generation or benchmark result is produced.

### Latest Real Turbo4 16K Result

| Metric | Value | Source |
|---|---|---|
| **Test** | `auto-context-16k-k0.log` | 2026-09-21 16:45:57 |
| **Model** | Qwen3.8-Flash-Next-NVFP4-Radix | |
| **Context** | 16,384 (prompt ~15,800) | `--max-seq-len-override=16384` |
| **KV Format** | Turbo4 (qsa_sparse) | `--kv-format=turbo4` |
| **MoE Strategy** | offload | `--moe-strategy=offload` |
| **Expert Cache** | Automatic (`--moe-cache-auto`) | 1,262 slots planned |
| **MTP** | k=0 (disabled) | `--disable-moe-prefill-overlap` |
| **CUDA Graphs** | Disabled | `--cuda-graph-max-bs=0` |
| **Planner Plan** | 1,262 experts, 259 KV pages (16,576 tokens) | Log |
| **Startup Free VRAM** | 15.18 GiB | `mem_get_info` before model |
| **Post-Init Free VRAM** | 1.24 GiB | Driver free after all pools |
| **Measured Transient Peak** | 0.00 GiB (calibration) vs 1.26 GiB modelled | Ledger `measured:transient-peak=0.000` |
| **First Prefill Chunk** | 8,192 tokens | Scheduler `max_extend_tokens=8192` |
| **Completion Status** | **FAILED** | Scheduler worker OOM at GDN `h` |
| **Output Validity** | No SHA1, no generation | Worker exitcode=1 |
| **PP Speed** | Not measured | Request accepted but prefill failed |
| **TG Speed** | Not measured | No decode phase reached |

### Latest Real Turbo3 16K Result

| Metric | Value | Source |
|---|---|---|
| **Status** | **NOT TESTED** with automatic planner | No log found for Turbo3 + `moe-cache-auto` + 16K |

### Current Selected Prefill Strategy

- **Chunk size**: Fixed at `SchedulerConfig.max_extend_tokens = 8192` (scheduler default)
- **Planner solves**: Largest chunk fitting transient budget via binary search (Phase F)
- **Result from latest run**: 8192 tokens (max allowed) — planner chose max chunk
- **Total context**: Preserved at 16,384 via KV pages (259 × 64 = 16,576 tokens)
- **Chunk selection basis**: Planner's `phase_f_solve_prefill_chunk()` uses `phase_e_gdn_sizing()` estimator + hard-coded transient reserves (Triton 128 MiB, workspace 128 MiB, etc.)
- **Problem**: Chunk solver uses `physical_budget` (post-mandatory free = 2.44 GiB) as available transient, but this budget already includes persistent pools. The planner then allocates expert/KV from the same budget in Phase G, creating a circular dependency where the "available transient" is not actually available.

### Current Expert/KV Allocation Strategy

| Step | Logic | Source |
|---|---|---|
| 1. Expert bytes/slot | From `method.layout()` (NVFP4 kernel geometry) | `memory_planner.py:_expert_bytes_per_slot()` |
| 2. Min expert slots | `num_experts × 2` if prefill_overlap else `num_experts` | 512 → 1024 (overlap=False in latest run) |
| 3. Max expert slots | `num_moe_layers × num_experts` = 48 × 512 = 24,576 | Capped by method slot limit |
| 4. KV bytes/page | From pool class `kv_cost()` | QSA pool: 0.43 MiB/page |
| 5. Required KV pages | `ceil(max_seq_len / page_tokens)` = 256 for 16384/64 | Phase G `required_kv_pages` |
| 6. Physical budget | Post-mandatory `driver_free` = 2.44 GiB | Phase A post-mandatory |
| 7. Transient reserve | `gdn_peak + triton + workspace + graph + activation` = ~1.26 GiB | Phase D calibration + hard-coded |
| 8. Semi-persistent | `graph_pool + workspace` = 0.12 GiB | Phase D calibration |
| 9. Fixed overhead | KV fixed + dummy + GDN state + page table + aux = ~1.96 GiB | `StaticCostModel.fixed_overhead_bytes()` |
| 10. Expert slots | `(budget - fixed - required_kv - transient - semi_persistent) / per_expert` | Phase G |
| 11. KV pages | Remaining after expert allocation | Phase G |

**Result from latest run**: 1,262 expert slots (3.259 GiB), 259 KV pages (0.110 GiB), leaving 0 headroom.

### Whether Final Physical Validation Exists

**PARTIAL — validation runs but worker crashes on OOM**

- `phase_i_final_validation()` runs a real prefill forward at chosen chunk
- It catches `torch.cuda.OutOfMemoryError` and returns `(False, msg)`
- However, the planner's `plan()` method **raises `RuntimeError`** on validation failure instead of replanning
- The scheduler worker is a separate process; when OOM occurs inside `model.forward()`, the worker dies (exitcode=1)
- The API supervisor detects worker death and shuts down the server
- **No safe recovery path exists** — the process cannot continue after worker OOM

### Whether Automatic Replanning Exists

**NO — skeleton only**

```python
if not valid:
    logger.warning_rank0(f"Validation failed: {msg}. Attempting replan...")
    # For now, fail - full replan logic would go here
    raise RuntimeError(f"Final validation failed: {msg}")
```

No logic to reduce chunk, reduce experts, or retry with modified parameters.

### Whether Memory-Ratio Remains Active

**YES — partially**

| Location | Status |
|---|---|
| `EngineConfig.memory_ratio` field | Exists, default 1.0 |
| CLI `--memory-ratio` | Exists in `server/args.py` |
| Legacy `ceiling_bytes()` | Still uses ratio in `cache_budget.py` |
| Legacy `VramLedger.decide()` | Still uses ratio via `ceiling_bytes` |
| New planner path | **Does not use ratio** — uses physical `driver_free` directly |

The automatic planner bypasses `memory_ratio` by using `post_mandatory_snapshot.driver_free` as the physical budget. However, the field, CLI option, and legacy ledger calculations remain in the codebase.

### Remaining Heuristic Constants Affecting Correctness

| Constant | Value | Location | Status |
|---|---|---|---|
| `TRITON_AUTOTUNE_ARENA` | 128 MiB | `vram_ledger.py:45` / planner hard-coded | Not measured |
| `BACKEND_WORKSPACE` | 128 MiB | `vram_ledger.py:49` / planner hard-coded | Not measured |
| `FRAGMENTATION_RESERVE` | 64 MiB | `vram_ledger.py:53` / planner hard-coded | Not measured |
| `GRAPH_POOL_FIRST_SHAPE` | 200 MiB | `vram_ledger.py:46` | Not measured (graphs disabled) |
| `GRAPH_POOL_EXTRA_SHAPE` | 150 MiB | `vram_ledger.py:47` | Not measured |
| `GRAPH_CAPTURE_PEAK` | 128 MiB | `vram_ledger.py:48` | Not measured |
| `MM_ENCODER_PEAK` | 192 MiB | `vram_ledger.py:51` | Not active (text-only) |
| `GRAPH_CAPTURE_EXTRA_SHAPE` | 160 MiB | `vram_ledger.py:52` | Not measured |
| `LIVE_ACTIVATION_TENSORS` | 6 | `vram_ledger.py:54` | Not measured |
| `GDN_CHUNK_SIZE` | 64 | `vram_ledger.py:41` | Kernel constant |
| `SchedulerConfig.max_extend_tokens` | 8192 | `scheduler/config.py:17` | Not solver-driven |

### Current Test/CI Status

| Suite | Result | Relevance |
|---|---|---|
| `make format` | **PASS** | 2 files reformatted |
| `make lint` | **PASS** | Ruff all checks passed |
| `make typecheck` | **PASS** | MyPy 427 files, no issues |
| `make test` (fast) | **PASS** | 1957 passed, 206 skipped, 15 deselected |
| `tests/engine/test_vram_ledger.py` | **PASS** | Pure arithmetic, no GPU |
| `tests/engine/test_cache_budget.py` | **PASS** | Pure arithmetic, no GPU |
| `tests/moe/test_offload.py` | **PASS** | CPU/fake device only |
| Real 16K Turbo4 automatic | **NOT TESTED** | No passing GPU run |
| Real 16K Turbo3 automatic | **NOT TESTED** | No GPU run |
| MTP k>0 with planner | **NOT TESTED** | No GPU run |
| CUDA graphs with planner | **NOT TESTED** | No GPU run |

### Current Performance Numbers

| Configuration | PP (tok/s) | TG (tok/s) | VRAM | Status |
|---|---|---|---|---|
| Flash-Next NVFP4 16K auto | N/A | N/A | OOM | **FAIL** |
| Flash-Next NVFP4 4K (IQ4_XS) | ~194 | ~31 | OK | PASS (GGUF) |
| Flash-Next NVFP4 16K MTP k=4 | ~194 | ~27.8 | OK | PASS (GGUF, sha1 match) |
| Flash-Next NVFP4 16K hybrid | N/A | 30.51 | OK | PASS (manual cache) |
| 35B-A3B (baseline) | N/A | 6.3 ms/tok | OK | Reference |

### Files Still Need Changes

| File | Required Change |
|---|---|
| `python/freetoken/engine/memory_planner.py` | Fix circular budget in Phase F/G; connect chunk to scheduler; implement replan |
| `python/freetoken/engine/engine.py` | Ensure planner runs before any persistent allocation; safe worker recovery |
| `python/freetoken/scheduler/config.py` | Accept `max_extend_tokens` from planner |
| `python/freetoken/scheduler/scheduler.py` | Use planner-chosen chunk; safe OOM handling |
| `python/freetoken/kernel/fla/chunk_delta_h.py` | Log exact `B,NT,H,V,K,dtype` at allocation site |
| `python/freetoken/engine/vram_ledger.py` | Remove heuristic constants or make them measured |
| `python/freetoken/engine/cache_budget.py` | Remove `memory_ratio` from automatic path |
| `python/freetoken/server/args.py` | Deprecate `--memory-ratio` for auto mode |

### Dead/Partial Implementations Should Be Removed

| Item | Location | Reason |
|---|---|---|
| `physical_budget_hint` logic | `memory_planner.py:352-354` | Unused, dead code |
| Hard-coded `RuntimeCalibration` defaults | `memory_planner.py:515-518` | Replace with measurements |
| `_run_probe_prefill` duplicate page_table setup | `memory_planner.py:560-565` | Already done in Phase C |
| Legacy ledger `plan_for_context` priority inversion | `vram_ledger.py` | Unused by planner path |
| `measured:transient-peak` always 0 | `vram_ledger.py` | Not measuring real transient peak |

### Smallest Remaining Engineering Milestones

| # | Milestone | Exit Criterion |
|---|---|---|
| 1 | Fix Phase F/G budget circularity | Planner solves chunk + experts + KV without double-counting budget |
| 2 | Connect planner chunk to scheduler | `config.max_extend_tokens` = planner output |
| 3 | Instrument GDN `h` allocation | Log exact shape/dtype at `chunk_delta_h.py:293` |
| 4 | Run 16K Turbo4 with fixed planner | Server ready + first prefill completes + generation completes |
| 5 | Implement safe replan on validation fail | Reduce chunk → retry → reduce experts → retry |
| 6 | Run 16K Turbo3 automatic | Same as Turbo4 |
| 7 | Test MTP k=1..6 with planner | SHA1 match k=0 baseline |
| 8 | Test CUDA graphs with planner | Graph capture + replay without OOM |
| 9 | Remove `memory_ratio` from auto path | Planner path only; legacy path for manual |
| 10 | Persistent calibration profile | Save/load measured peaks by GPU/model/backend |
| 11 | Full CI + benchmark certification | `make ci` + `make bench` + `cert_matrix.py` 16K rows PASS |

### Evidence Missing Next Model Must Collect Before Editing Code

1. **Exact `h` tensor shape at failure** — Run diagnostic wrapper at `chunk_delta_h.py:293` logging `B,NT,H,V,K,dtype,itemsize,bytes`
2. **Peak allocated/reserved immediately before GDN** — Snapshot `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `max_memory_reserved`, `mem_get_info` before each GDN layer in probe
3. **Driver free after each major allocation** — Trace `mem_get_info` after weights, expert cache, KV pool, GDN state, attention backend
4. **Complete expert auxiliary tensor total** — Tensor-walk `OffloadMoeCache` after allocation including all reachable tensors
5. **Allocator reserved blocks at OOM** — Compare `memory_reserved` vs `mem_get_info` at failure moment
6. **Smaller chunk OOM test** — Run same config with `max_extend_tokens=4096,2048,1024` to find minimum viable chunk
7. **Turbo3 16K automatic baseline** — Run `--kv-format turbo3 --moe-cache-auto --max-seq-len-override 16384`
8. **Probe reclamation verification** — Measure `driver_free` before Phase C, after Phase H cleanup, confirm reclamation

### Final Current Status

**STATUS: IMPLEMENTATION INCOMPLETE**

The multi-stage planner architecture is implemented and invoked, but:
- The budget circularity between Phase F (transient) and Phase G (persistent) leaves zero headroom
- The first real prefill OOMs at 192 MiB GDN `h` allocation with only 171 MiB free
- No safe recovery or replanning exists
- No 16K Turbo4/Turbo3 generation has completed with the automatic planner
- Scheduler `max_extend_tokens` is not driven by the planner
- Heuristic constants remain unmeasured
- `memory_ratio` still exists in legacy path and CLI