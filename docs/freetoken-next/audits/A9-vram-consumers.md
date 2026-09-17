# A9 - VRAM consumers audit (branch `next` @ 86af2d3)

Scope: every GPU byte the engine spends outside {weights, KV pool, MoE expert cache, GDN state
pool}, so `engine/vram_ledger.py` (draft present, 331 lines) can charge it instead of letting
`(1 - memory_ratio)` hide it. All line numbers verified at HEAD.

## 1. Today's measurement flow (allocation order)

| # | site | what it measures / does | blind to |
|---|------|--------------------------|----------|
| 1 | engine.py:313 `_ensure_expandable_segments` (def 1119-1133) | allocator policy, not a measurement; exists because NVFP4 dequant churn fragments reserved memory | everything |
| 2 | engine.py:332-334 `free_min, free_max = _sync_get_memory()` | `_sync_get_memory` def 789-806: `synchronize + empty_cache + reset_peak_memory_stats + mem_get_info`, gloo all_reduce. `init_free_memory = free_max` (KV sizing), `self._baseline_free = free_min` (rebuild baseline). CUDA context + pynccl comm exist here (comm built at 325 -> `_init_communication` 484-495) so they are *inside* "free" | anything allocated later |
| 3 | engine.py:352 `place_encoder_weights` (before the snapshot, so host-streamed towers are not charged as weights) | moves/pins vision towers | 2x-block GPU staging (A9 section 2.6) |
| 4 | engine.py:353-354 `post_weights_free`; `_weights_bytes = baseline - post` | delta-accounting for weights | the delta also silently absorbs `load_state_dict` temp copies; nothing else runs in between |
| 5 | engine.py:360 `self._post_weights_free` | stable budget reference for cache sliders (comment 355-359 admits graphs/workspaces make query-time readings drift) | - |
| 6 | engine.py:368 `_host_tables_bytes = load_host_tables` | PINNED HOST bytes only (ple/model.py:144-210); feeds `_pin_budget_bytes` (engine.py:1299-1310, host RAM, WSL-only) | its GPU staging buffers |
| 7 | engine.py:370 `_init_offload_moe_cache` (def 592-680) -> `_resolve_auto_moe_cache_size` (565-590) | MoE cache sized BEFORE KV: `resolve_moe_cache_auto` (cache_budget.py:120-152) on ONE budget = `ratio*baseline - weights - fixed` (net_cache_budget_bytes 28-37), MoE-first split (plan_cache_budget 72-116); writes `num_page_override` (671) so KV reuses the same plan | the expert-cache LRU metadata slabs, alphas, workspaces, autotune |
| 8 | engine.py:396-401 KV solve | `new_free = _sync_get_memory()[1]` (MAX); `_startup_kv_budget` (70-74): `ratio*init - (init-new_free)`; minus `state_pool_bytes(config)`; `solve_num_pages` (kvcache/base.py:65-83, dummy page priced via DUMMY_PAGES) | everything between #2 and #8 is charged to *weights* via `init - new_free` |
| 9 | engine.py:403-432 | `create_kv_pool(+1 dummy)`, LinearStatePool (412-424), page table (428-432) | - |
| 10 | engine.py:439-444 `create_attention_backend`, `Sampler` | workspaces allocated here, AFTER the KV solve decided num_pages | see 2.2 |
| 11 | engine.py:446 `post_free_memory` | log only | - |
| 12 | engine.py:463-475 `GraphRunner(...)` | `free_memory=init_free_memory` used ONLY to pick the bs list (graph.py:83-98); free before/after capture queried-and-logged (graph.py:157-159, 176-179, 204-205); shared graph mempool created (graph.py:193-200); `_warmup_prefill` (477-479, def 1013) fires Triton autotune | the whole capture + autotune peak |
| 13 | engine.py:852-990 `rebuild_runtime_cache` | fit-check `validate_rebuild` (base.py:90-126) replays the same `ratio*baseline - weights - fixed` formula; teardown 951-975; re-measure `free_min` at 982 only to hand GraphRunner a number | reserves; `_sync_get_memory` at 982 is empty_cache-based so freed graph pools read back as "free" |

Net: three real snapshots (332, 353, 396) + one log (446) + rebuild (982). The `(1-ratio)`
remainder (cache_budget.py:35 "is the CUDA-graph/activation headroom") is policy-by-comment,
never by bytes.

## 2. Consumer line items (NOT in {weights, KV, MoE slots, GDN state})

2.1 CUDA graphs (engine/graph.py) - PERSISTENT / SEMI-PERSISTENT
- `GraphCaptureBuffer.logits [max_graph_bs, vocab] fp32` + ids/pos/out_loc/table_idx int32 - graph.py:40-48, allocated 166-167. vocab 152k x bs 160 = ~93 MiB; never budgeted.
- shared graph mempool: `torch.cuda.graph(graph, pool=pool)` graph.py:193-195, `pool = graph.pool()` 199-200 - holds every captured intermediate (MoE `intermediate_cache*`, attention scratch, fla state copies) for the biggest captured bs; only queried-and-logged (176-179, 204-205), never reserved; loop keeps capturing even as `avail_mem` shrinks.
- capture warmup forward (186-192, outside capture) - TRANSIENT peak on top of everything.

2.2 Attention backend workspaces - SEMI-PERSISTENT
- flashinfer `float_workspace_buffer = max(256 MiB, qo_local*padded_batch*cta_tile_q*head_dim*4 + 32 MiB)` - fi.py:95-116; shared by prefill+decode wrappers (117-128), int workspace re-used (129-131, flashinfer-internal size). NOTE: draft ledger `BACKEND_WORKSPACE = 128 MiB` (vram_ledger.py:57) UNDER-charges fi by >=2x.
- trtllm: flat 128 MiB, trtllm.py:45-46.
- triton: `TritonCaptureData` - page_table [max_bs, max_seq_len] int32, attn_logits [max_bs, q_heads, 8, head_dim] fp32, lse [max_bs, q_heads, 8] fp32 - triton.py:38-58; growable per-bs decode scratch triton.py:116-128; swa capture table triton.py:287-288; init_capture_graph 289-293.
- capture scratch (all backends): `BaseCaptureData.create` attention/utils.py:16-24 (page_table [max_bs, max_seq_len] int32), called fi.py:269-270.
- per-backend persistent bufs: dsa.py:336 `_kvlen_buf`, qsa_sparse.py:503-510 capture dict, m3_sparse.py:380.

2.3 Triton autotune - TRANSIENT (must be FUNDABLE at peak)
- `do_bench` takes a 256 MiB L2-flush arena per call: `.venv .../triton/testing.py:152` -> `backends/nvidia/driver.py:754-761` (`256*1024*1024` bytes). This is the "wanted 256 MiB, 209 MiB free" OOM. `TRITON_AUTOTUNE_ARENA = 256 MiB` confirmed.
- live autotune sites: kernel/fla GDN prefill (kda.py:198,309,495,695,862,1265; chunk_fwd.py:30; chunk_delta_h.py:29; solve_tril.py:34,109,234; cumsum.py:75; kda_chunk_delta_h.py:43) tuned during capture-warmup/`_warmup_prefill`; sampling.py:70,101,173,202 (eager first decode, AFTER graphs exist); minimax_m3_sparse.py:187. Disk cache only avoids re-benching (kernel/triton/autotune_cache.py:25-35; autotuner.py:191-208).
- `reset_to_zero` arg clones during benchmarking (autotuner.py:228-235) - small TRANSIENT.

2.4 MoE-path extras - PERSISTENT metadata + TRANSIENT scratch
- LRU/index arrays on GPU: offload_cache.py:183-262 - slot_for_id 4B*L*E, id_of_slot 4B*C, usage 8B*C, step/active ~0, expert_recency 8B*L*E, evict_slots+src_indices 4B*max(E,C) each (201-202, 517-518), lru_stats 8B*L*N_STATS (240-241), decode_freq 8B*L*E (261-262), copy descriptor ptr tables 435-440. ~O(L*E*24B + C*12B); for L=64,E=256,C=8192: ~5 MiB.
- marlin/nvfp4 per-expert alphas: expert_banks.py:101-103 `[L*E]` fp32 per resident role; deliberately excluded from `expert_bytes_per_slot` (cache_budget.py:19-22) - unmodelled but tiny.
- prefill-overlap double buffer borrows 2x[E] slots from the cache itself (offload_cache.py:171-175) - ACCOUNTED.
- fused MoE intermediates: fused.py:228-241 `cache = M*topk*max(N, w2d)*dtype` + intermediate_cache2; triton decode path fused.py:338-364 (3 caches + out). TRANSIENT eager / graph-pool-resident captured; driver M = chunk tokens x topk.
- dequant scratch: variable-size NVFP4->BF16 expert blocks during offload prefill (rationale in engine.py:1120-1128) - TRANSIENT, sized by active-expert count per layer; needs a host measurement constant, not a formula.
- cpu_executor: `out = empty_like(hidden)` per call (cpu_executor.py:684) + ds_fp4 GPU prequant grid (355-375) - TRANSIENT, O(tokens*hidden); its banks/flags are pinned host.

2.5 Pinned/staging GPU counterparts - PERSISTENT
- weight_stream.py:57-58: pinned host bank (host) + `staging = torch.empty((2, row_bytes), device)` GPU = 2 x one vision block's bytes (def `device_bytes` 71-73; instantiated per-vision e.g. models/qwen3_vl/vision.py:246, muse_glimmer/vision.py:133, gemma4/vision.py:208, minimax_m3/vision.py:112, glm5_next/vision.py:107).
- PLE pinned UVA: eager growable + per-captured-size staging `[rows, head_dim]` bf16 - ple.py:152-172 (`_graph_staging` dict outlives rebuilds).
- PLE disk backend: `_graph_dev` = max(256, cuda_graph_max_bs) x heads x head_dim and `_eager_dev` = max_extend_tokens x heads x head_dim (uint8/bf16) - ple_disk.py:147-156; pinned halves host-only.
- offload prefill hit buffers (host pinned): offload_cache.py:635-645.

2.6 Page tables / radix metadata
- engine page_table: engine.py:428-432 `(max_running_req+1) x _page_table_width(max_seq_len, page_size)` int32 - O(R*C*4B): 128k ctx x 321 rows = ~160 MiB. NOT budgeted.
- scheduler free_slots: scheduler/cache.py:38, 562 int32[num_pages] (4B/page) + offsets arange 616 (transient).
- hybrid full_to_swa_index_mapping int64[full_tokens+page+1] + `_swa_free` int32[swa_tokens]: hybrid_swa_pool.py:150-159.
- DSV4 full_to_window int64[full_token+1]: dsv4_paged_pool.py:204-206 (priced in cost model, see 5).
- radix/hybrid/swa trees (kvcache/radix_cache.py, hybrid_radix_cache.py, swa_radix_cache.py): Python host objects - host RAM, NOT VRAM.

2.7 Distributed - PERSISTENT, OUTSIDE the caching allocator
- pynccl symmetric buffer: engine.py:492-495 `max_bytes = max_forward_len*hidden*itemsize`, capped by env.py:73 (1 GiB), `ncclMemAlloc` in kernel/csrc/src/pynccl.cu:75-83 - invisible to `torch.cuda.memory_allocated`, visible only to `mem_get_info`. Allocated BEFORE baseline (comm at 325) so it is netted out today, but a rebuild-era ledger sized off `allocated` would miss it.
- real NCCL (tp>1 without pynccl): comm buffers, driver-internal - needs measurement.

2.8 Encoder / mm
- `EncoderCache(storage="cuda")` holds `[rows, proj_dim]` embeddings on GPU with NO capacity bound (mm/encoder_cache.py:23-53; default "cpu", mm/config.py:19).
- encoder forward activations + `_warmup_encoders` dummy items (engine.py:525-529): TRANSIENT; mm draft constant `MM_ENCODER_PEAK=768 MiB` matches the 0.4-1.2 GB observed band (host measurement).
- streamed towers: line 2.5 staging.

2.9 MTP / speculative decoding: NOT PRESENT. Only MTP weight *dropping* (models/qwen4_exp/weight.py:89, glm5_next/weight.py:11, glm_moe_dsa/weight.py:7). No draft runner, no extra buffers.

2.10 Misc: extra `torch.cuda.Stream`/events (copy/prefill streams) each accrue a small cuBLAS
workspace per handle; CUDA context itself is pre-baseline (excluded from `free`, so a
total-VRAM-denominated ledger must charge ~0.4-0.6 GiB as `device:context`).

## 3. Formulas for the ledger (best closed form; M = measurement needed)

| line item | formula | driver |
|---|---|---|
| graph logits buffer | `max_graph_bs * vocab * 4` + `8 * max_graph_bs` (int32 aux) | bs-list x vocab; M |
| graph mempool (captured activations) | ~ `max_bs_set` MoE intermediates + attn scratch: `M*topk*max(2I, I)*dtype` (fused.py:228) + `M*hidden` terms; today only M: measure `free_before_capture - free_after` | bs, topk, inter, hidden |
| fi workspace | `max(256MiB, qo_local*ceil(2*SMs/kv_local)*cta_q*head_dim*4 + 32MiB)` (fi.py:104-113) | heads, head_dim, SM count |
| trtllm workspace | 128 MiB (trtllm.py:45) | fixed |
| triton capture | `max_bs*max_seq_len*4 + max_bs*qh*8*hd*4 + max_bs*qh*8*4` | bs, ctx, heads |
| capture scratch (base) | `max_bs*max_seq_len*4 + O(max_bs)` (utils.py:22) | bs, ctx |
| autotune arena | 256 MiB (triton driver), + one kernel's arg-clones; TRANSIENT-but-fundable, max-overlap with live scratch | fixed M |
| MoE cache metadata | `L*E*24 + C*12 + max(E,C)*8` bytes | L, E, cache_size |
| marlin alphas | `roles * L * E * 4` | fixed |
| MoE intermediates (eager prefill) | `chunk_tokens*topk*max(2I, w2d)*dtype` (+ `chunk*topk*I/2*dtype`) | chunk, topk, I |
| dequant scratch | M: largest single-layer NVFP4->BF16 block = `E_active * slot_bytes` | active experts |
| weight_stream staging | `2 * block_bytes` (one streamed block) | block size |
| PLE UVA staging | `(captured_bs_set sizes + eager max run) * head_dim * 2` | head_dim, bs |
| PLE disk GPU | `(max(256, cg_max_bs) + max_extend_tokens) * n_heads * head_dim` | ctx |
| page_table | `(R+1) * align_ceil(align_ceil(min(max_seq_len, kv_tokens), P), 32) * 4` | R, ctx |
| free_slots | `num_pages * 4` | pages |
| hybrid mapping | `full_tokens * 8 + swa_tokens * 4` | tokens |
| pynccl | `min(max_seq_len * hidden * itemsize, 1 GiB)` (+ NCCL internals M) | ctx, hidden |
| mm encoder peak | M (768 MiB band) | images |
| encoder cache (cuda mode) | unbounded -> ledger must cap it: `n_images * rows * proj * 2` | mm traffic |
| GDN prefill transient | `gdn_prefill_bytes` (vram_ledger.py:69-95) - keep | chunk, heads, k, v |

Flag MEASURE-ONLY: graph mempool peak, NCCL internals, dequant scratch, mm peak, cublas
per-stream workspaces.

## 4. Where the ledger hooks

`charge(...)` sites (startup order = engine.py): baseline at 332-334 -> `open_ledger`;
weights 353-354; host staging 368 + 2.5 GPU counterparts; MoE cache 370/660 (slots already the
negotiable line; add metadata 2.4 as a charge); encoders 352/380-385; KV solve 396-405 ->
replace `_startup_kv_budget` + `solve_num_pages` inputs with `ledger.pool_budget_bytes()`;
LinearStatePool 412; page table 428; backend workspaces 439; graph buffers+pools 463-475
(graph.py:157-205: charge capture, assert, don't just log); `_warmup_prefill` 479; rebuild:
`validate_rebuild` (base.py:90-126) and the resize steps 951-975 (`release`/`charge` per
teardown/realloc), `free_min` at 982.

Earliest KV decision: engine.py:401 (`solve_num_pages`), fed by 396-400; with
`--moe-cache-auto` the effective decision is earlier - engine.py:660-671 (`num_page_override`
pinned there from `baseline_free/weights_bytes`, `state_pool_bytes` subtracted at 571).

Ordering problem: today MoE is solved at 370, KV at 401, but workspaces (439), graphs (463) and
autotune (479 + first eager decode) come AFTER and are funded from the `(1-ratio)` gap. Before
KV is allocated the ledger must already reserve: fi/trtllm workspace (256-300 MiB), capture
scratch + logits buffer (~100-150 MiB at bs160/152k), the graph-pool peak estimate, the 256 MiB
autotune arena, one GDN prefill chunk transient, and the fragmentation reserve. Draft fixes:
`BACKEND_WORKSPACE=128 MiB` is wrong for fi (>=256 MiB, fi.py:113); `GRAPH_CAPTURE_PEAK x
graph_shapes` (vram_ledger.py:49-52, 138-143) should key off `max(cuda_graph_bs)` not shape
count since the pool is shared (graph.py:199-200); autotune + capture peaks OVERLAP with live
prefill scratch, so max-per-phase, sum-across-phases (as drafted) is right.

## 5. `unit_bytes()` vs `kv_cost()` honesty

| pool | verdict | evidence |
|---|---|---|
| mha_pool.py | HONEST | alloc 51-55 = single slab; unit 101-104 measures it; kv_cost 86-96 fixed=0; dummy priced (base.py:74-76) |
| hybrid_swa_pool.py | UNDER-reports | unit 345-352 measures only the two tier buffers; `full_to_swa_index_mapping` int64 (150-156, priced in kv_cost:320 at 8B/token) and `_swa_free` int32 (158-159, priced NOWHERE) are absent from unit_bytes. Error ~12 B/token (24 MiB per 2M tokens) |
| dsa_pool.py (MLA/DSA) | HONEST | unit 128-130, 192-199 floor-divide both slabs; kv_cost 114-122 spec-sum includes index slab (base.py:34, bf16 hardcode flagged there) |
| KpoolDSA (dsa_pool.py:208) | kv_cost UNDER-prices | alloc adds scratch rows (`_index_rows` 230-232: `+num_req_slots`) and tails `_tail_k/_tail_gate` (239-245) but kv_cost fixed=0 (inherited 114-122). Unmodelled = `L_idx*R*D_idx*2 + 2*L_idx*R*ratio*D_head*2` bytes (MiB-scale at R=4 default, linear in max_running_req). unit_bytes (192-199) DOES measure them -> rebuild fit-check sees the delta; startup does not |
| qsa_pool.py | small UNDER-price | ring priced (kv_cost 162-180 fixed); `_cmp_k_buffer` scratch rows `num_req_slots` (119-124) NOT priced; unit 181-193 deliberately excludes them (shadow-only) -> consistent with itself, ~`L_idx*R*D_idx*2` B unmodelled either way; ring uses `_INDEX_DTYPE_BYTES=2` (33) asserted == alloc dtype (85) |
| bsa_pool.py | HONEST-by-inheritance | unit 110-116 measures slab+index; kv_cost is MHA's (86-96) and prices the index slab only through the base spec formula's hardcoded 2 B bf16 (base.py:22-34) - correct today (factory passes engine dtype, kvcache/__init__.py:114-120), silently wrong if the slab dtype changes |
| dsv4_paged_pool.py | MODEL, not measurement | unit 439-442 returns `dsv4_kv_unit_bytes/dsv4_window_unit_bytes` (dsv4_cost_model.py:101-133, ceil/token); solve is byte-exact vs `_alloc_buffers` (dsv4_paged_pool.py:196-260) incl. full_to_window (cost model:109) and scratch rows (kv_cost 317-328 passes n_scratch). No missing tier found; residual risk is drift -> ledger should assert summed-model == summed-measured once at startup |
| linear_state_pool.py | HONEST (no unit_bytes) | `state_pool_bytes` 263-277 uses `ssm_state_dtype` fp32 for recurrent (251-252) matching alloc 67-77; slot formula 280-290 (`4R + ceil(ratio*R) + 1`); slot_states 98-106 priced |

Ledger consequence: charge KV from `kv_cost` per tier PLUS the fixed-term gaps above
(hybrid mapping, Kpool/QSA scratch rows), and cross-check against measured `unit_bytes` at
startup - the two Kpool/QSA omissions are exactly the kind of "few MiB nobody owns" that
compounds with the graph/autotune reserve.
