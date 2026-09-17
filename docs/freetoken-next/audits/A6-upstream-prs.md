I have everything. Now write the report — dense markdown, sections 1-6, ≤250 lines, evidence with refs/pr/N:path:LINE.

Let me be careful and complete.

Key numbers to include:
- Change list for a new dtype.
- Conflicts: 354 vs 408 vs 113.
- Rebase checks: 113 rc=0, 460 rc=0, 69 rc=0, 494 rc=0; 354 rc=1 (5 files), 408 rc=1 (same 5), 337 rc=1 (1 file), 447 rc=1 (6 files).

Note the important nuance: #113 merges clean textually but conflicts semantically with #354/#408 (same CLI flag, different dest). And 113's base is 49 commits behind → its clean merge is because it only touches DSV4 files that upstream didn't change.

Let me write the report.
# FreeToken PR audit — KV-format seams

Base = `cac247a` (v0.1.3). All `path:LINE` citations are **in the PR ref tree** (`refs/pr/N:path:LINE`), verified with `git show`.

---

## 1. KV dtype recipe — ordered change-list to add one `--kv-cache-dtype` value

Ancestry first (decisive): **`refs/pr/354` is a strict ancestor of `refs/pr/408`** (`git merge-base --is-ancestor` → YES), and of `refs/pr/460`. `refs/pr/113` is independent (`not in 354`). So #408 *is* #354 + nvfp4; #460 *is* #408 + `fp8-fp4`.

### Canonical file list (13 seams — #354 is the minimal, cleanest instance)

| # | File | What the hunk does |
|---|---|---|
| 1 | `python/freetoken/server/args.py:372` | `parser.add_argument("--kv-cache-dtype", dest="kv_quant", choices=["auto","bf16","fp8"])`, `default=ServerArgs.kv_quant`. #408:403 → `choices=[…,"nvfp4"]`. |
| 2 | `python/freetoken/engine/config.py:67` | `EngineConfig.kv_quant: str = "none"` (the *stored* value, not the CLI spelling). |
| 3 | `python/freetoken/engine/engine.py:118` | `KV_QUANT_ALIASES = {"auto":"none","bf16":"none","none":"none","fp8":"fp8"}` — CLI spelling → config token. #408:121 adds `"nvfp4":"nvfp4"`. |
| 4 | `engine/engine.py:121 _resolve_kv_quant` | normalize+reject unknown. `engine.py:132 _backend_supports_kv_quant(name, kv_quant)` — splits comma backend strings, asks each `BackendInfo`. #408 generalises the lookup to `getattr(info, f"supports_{kv_quant}_kv")` (`refs/pr/408:…/engine.py:143`) — **that f-string is the extensible seam.** |
| 5 | `engine/engine.py:1359` in `_adjust_config` | resolve `kv_quant` **before** the backend tree; `engine.py:1367 quant_unsupported = required_attn_types - {FULL,SWA,QSA,MLA,DSA}` → `ValueError`; passes `kv_quant=` into `_resolve_auto_attention_backend` (`engine.py:1401`) so auto *skips* incapable backends; then `_validate_attention_backend_choice` (`engine.py:190+`) refuses an explicit incapable one, listing valid names. |
| 6 | `python/freetoken/attention/__init__.py:36` | `BackendInfo.supports_fp8_kv: bool = False` (new field). Opt-ins: `fa` `:91`, `dsa` `:113`, `qsa_sparse` `:146`. #408:41 adds `supports_nvfp4_kv` and sets it on the same three (`:92,:116,:153`). `triton` inherits `fa`'s registration block; `dsv4_sparse`/`m3_sparse`/`fi`/`trtllm` stay `False`. |
| 7 | `python/freetoken/kvcache/base.py` | `FP8_KV_SCALE_BYTES = 4` `:13`; `kv_storage_bytes_per_elem(config) -> int` `:22` (bytes/element of the code buffer); `kv_scale_bytes_per_token(spec, config)` `:37` (sidecar priced *here*, not in the pool, "so kv_cost and the pool's own allocation can never disagree"); `spec_kv_bytes_per_token` `:49` multiplies by it; `BaseKVCachePool.kv_quant: str = "none"` `:87`; `k_scale()`/`v_scale()` `:198,:203`; **`dtype` `:213` = compute dtype, `store_dtype` `:222` = buffer element type** — the central new contract. #408:215 adds `k_block_scale()`/`v_block_scale()`. |
| 8 | `python/freetoken/kvcache/__init__.py:79` | `_reject_unsupported_quant(pool, kv_quant)` called for `DSV4 paged` `:104` and `BSA` `:186`; `kv_quant=kv_quant` threaded into every pool ctor at `:133,158,226,247,267,280`. |
| 9 | `kvcache/mha_pool.py:12 _kv_store_dtype(dtype, kv_quant)`; `:73 _alloc(...)`; `:174 store_kv`; `:164 k_scale/v_scale`; `:216 store_dtype`. Same trio duplicated per family: `hybrid_swa_pool.py:30 _alloc_group_storage` (+`:258` scales, `:268` store), `qsa_pool.py:66` (KV tiers only), `dsa_pool.py:64 _alloc`/`:97 latent_scale`/`:120` store. |
| 10 | `kernel/triton/kv_quant.py` (new module) | the format's encode kernel + dtype contract (§b,§c). #408 adds sibling `kernel/triton/kv_nvfp4.py`. |
| 11 | `kernel/triton/attention.py:17 _kv_dequant_scale`; `HAS_KV_SCALE: tl.constexpr` at `:110,:256,:620,:774` (4 kernels); wrapper params `:464-465`/`:953-954`. Also `:47 _select_extend_tile` takes the KV element size. |
| 12 | `kernel/triton/qsa/attend.py:58` + `kernel/triton/glm_dsa_sparse.py:41` (+ `:375 glm_dsa_sparse_attn(pool_scale=…)`), `qsa/score.py` — the *other* readers of the same pool. |
| 13 | `attention/triton.py:158-160` (fetch `k_scale/v_scale` from pool, assert pair, pass to all 3 call sites `:186,:208,:222`), `attention/qsa_sparse.py:292`, `attention/dsa.py:204`. Plus `kernel/aot_models.py:38 KV_CACHE_DTYPE_BYTES/FP8_KV_CACHE_DTYPE_BYTES`, `store_element_sizes(model, dtype_bytes)` `:397`, `aggregate_store_element_sizes()` `:421` (a missing row size = JIT fallback = fails the `FREETOKEN_DISABLE_JIT=1` release gate). Docs: `docs/cli.md:74` table row + `### FP8 KV cache` section; `docs/models.md`. |

### (b) Code-buffer layout chosen by each PR

| PR | code buffer dtype/shape | scales | token→scale index |
|---|---|---|---|
| **#354 fp8** | `torch.uint8`, **same shape as the 16-bit buffer**: `(2, L, pages, page_size, heads, head_dim)` (`mha_pool.py:88`, `alloc_codes` = `kv_quant.py:61` `torch.zeros`) | **separate sidecar tensor** `torch.zeros((2, L, pages*page_size, heads), fp32)` (`mha_pool.py:93`) | `scale[slot, head]`, same slot as the code row; `k_scale(layer)` = `_scale_buffer[0][dense(layer)]` (`mha_pool.py:164`) — `page_size` collapsed out of the code shape deliberately |
| **#408 nvfp4** | `torch.uint8`, **half width**: last dim `head_dim//2`, low nibble = elem 2i (`kv_nvfp4.py:50,78`) | **two sidecars**: fp32 row `[slots, heads]` + `torch.zeros((2,L,slots,heads, head_dim//16), uint8)` E4M3 block scales (`mha_pool.py` @408 `:112`) | `block_ptr[(slot*HEADS + h) * (D//16) + dim//16]`, `row_ptr[slot*stride + h]` (`kv_nvfp4.py:52-56`); pool stores `self._head_dim` so rebuild never confuses packed width with model width |
| **#113 fp8_e4m3 (DSV4)** | `torch.float8_e4m3fn` **native type in the buffer** (`dsv4_kv_quant.py:36 STORAGE_DTYPE`), `[slots, head_dim]` per pool (`dsv4_paged_pool.py:232`) | sidecar `[slots, head_dim//32]` **fp16**, `None`-filled list when off (`dsv4_paged_pool.py:238,270`) | per **32 elements along head_dim**, dequant applied *before* the dot (scale varies along the reduction dim) — `dsv4_kv_quant.py:13` |
| **#460 fp8-fp4** | one packed `uint8` row = **codes followed by scale bytes in the same row** (`docs/deepseek-v41.md:178`), sizes from `kvcache/dsv41_layout.py:6 dsv41_row_bytes()` → `(512+16, 256+32, 64+4)` | inline, no sidecar tensor | row stride only |

Sidecar (354/408/113) keeps `k_cache()`'s geometry untouched so "every index into them are unchanged"; packed-row (460) trades that for page-reuse/rebuild moving scales with codes (`dsv41.md:178`).

### (c) Encode-on-write / decode-restore paths (kernel names + file:line)

Encode (#354): `kernel/triton/kv_quant.py:84 `_kv_quant_scatter_kernel`` — grid `(tokens, heads)`, one program per (token, kv_head) row; `scale = max(amax,1e-10)/448`, `clamp(±448)`, `e4m3_f32_to_u8(round_e4m3(q))`, stores codes **and** `sk`/`sv` in the same launch. Host wrapper `kv_quant.py:133 quantize_kv_to_cache(k,v,out_loc,k_cache,v_cache,k_scale,v_scale)` with 6 dtype/shape asserts. Latent/MLA variant: `:189 _kv_quant_rows_scatter_kernel` / `:207 quantize_rows_to_cache`. NVFP4 (#408): `kv_nvfp4.py:65 _quantize_row` (`row_scale = amax/(6*448)`, E4M3 block scale via `tl.div_rn`, encoded against the *stored* scale), `:87 _scatter_rows` (MLA), `:116 _scatter`, wrappers `:94 quantize_nvfp4_rows_to_cache`, `:124 quantize_nvfp4_to_cache`; both pass `enable_fp_fusion=False`. #113: `kernel/triton/dsv4/kv_quant.py:19 _store_kv_quant_kernel` / `:54 store_kv_quant(pool, scales, slots, kv)`, called from `dsv4_paged_pool.py:161 _store_kv_quant`.

Decode (#354): `e4m3_compat.py:214 kv_load_e4m3_tile_f32(ptrs,mask)` and `:246 kv_load_e4m3_tile_scaled16(...)` (+ `:242 KV_TILE_SCALE = tl.constexpr(256.0)`), scale folded at `attention.py:17 _kv_dequant_scale` → `s * KV_TILE_SCALE`. Readers: `_paged_attention_kernel:81`, `_decode_grouped_stage1_kernel:218`, `_extend_attention_kernel:589`, `_extend_attention_split_kernel:737` (all `HAS_KV_SCALE`); QSA `qsa/attend.py:58`; DSA/MLA `glm_dsa_sparse.py:90-100,277-287` (`row_scale = tl.load(pool_scale_ptr + idxs*stride_ps)`). NVFP4 (#408): single reader `kv_nvfp4.py:37 load_nvfp4(ptr, block_ptr, row_ptr, slots, head, dims, …)`, `KV_NVFP4: tl.constexpr` at `attention.py:114,275` + qsa/attend + glm_dsa_sparse.

Restore ordering detail (#408 docs/cli.md:225): *"Paged MHA prefill uses fresh compute-dtype K/V while cached prefixes are restored, as in the FP8 path. MLA/DSA stores fresh latent rows first and reads the quantized cache in both prefill and decode."*

### (d) CUDA-Graph capture safety

Two distinct traps, both recorded in `refs/pr/354` commit `03fb043` body and `kv_quant.py:16-30` / `e4m3_compat.py:181-250`:

1. **fp8 type in a kernel signature** — `pool.dtype` reported e4m3, so QSA's indexer fed an fp8 tensor to `tl.dot` → `Unsupported rhs dtype fp8e4nv` at graph capture. Fix = split the contract: `dtype` is the **compute** dtype (16-bit), `store_dtype` the buffer type (`base.py:213/222`), pinned by a runtime assert in `attention/qsa_sparse.py:117` (`assert self.dtype.itemsize == 2`).
2. **fp8 type in the *pointer*** — `"cannot cast int32 to fp8e4nv"`, raised at CUDA graph capture on sm_100. Triton does **not** statically prune a branch on a pointer's element type; it type-checks the dead arm, whose `tl.load(..., other=0)` int fill is illegal against an fp8 pointer. The compile-time probe `e4m3_native_cx()` was the other failed route (it answers independently from the host that allocated the buffer and disagreed on sm_100; a `@constexpr_function` cannot reference the host `e4m3_native()` because Triton hashes constexpr fns by AST walk → `Unsupported function referenced`, `e4m3_compat.py:133-146`).

**Resolution:** codes live in a plain `uint8` buffer on *every* arch (`kv_codes_dtype()` is a constant, `kv_quant.py:49`), the fp8 type never appears in any signature, and the decode is **straight-line software** (`kv_load_e4m3_tile_f32` = `e4m3_u8_to_f32(tl.load(ptrs, mask, other=0))`) — no arm to prune. Probes stay separate; `warn_if_probes_disagree()` (`e4m3_compat.py:85`) logs a disagreement once instead of fixing it. Quantize+scatter are **fused into one launch** so `out_loc` stays a device tensor under capture (`kv_quant.py:26-30`). #408 keeps this (`kv_nvfp4.py` is all `uint8`/`fp32`, no fp8/fp4 pointer types) and pins it in `tests/kernels/test_kv_nvfp4.py:252 test_store_cuda_graph_replay_changes_slots` + `:317 test_backend_decode_graph_replays_new_kv_and_page_tables`. #113 did **not** adopt it — it uses a native `float8_e4m3fn` buffer and simply rejects `< sm_89` at config time (`engine.py` @113 `_validate_kv_cache_dtype`).

### (e) Backend gating

`BackendInfo` is the capability record (`attention/__init__.py:20-38`); default `False`, so a new format is denied to every backend until declared. The gate runs twice: in `auto` resolution (`_resolve_auto_attention_backend(..., kv_quant=)`, `engine.py:1401` — candidate *skipped*) and in explicit validation (`engine.py:231-246` — `ValueError` naming valid backends), **before any weight is resident**. Rationale verbatim (`attention/__init__.py:36-40`): backends that hand the cache to an external kernel "must opt out until that kernel is proven to apply our scale layout". #408 makes it generic via `supports_{kv_quant}_kv`. A new format therefore only adds one bool field + 3 registrations.

### (f) Pool budgeting + live rebuild

Budget arithmetic is quantization-aware at exactly one place, `base.py:22/37/49`, which feeds `MHAKVCache.kv_cost` (`refs/pr/354:mha_pool.py:126`, returns `per_token*page_size, 0, page_size, 0`) and thus `engine/cache_budget.py:45,84,123` (`num_pages = max(remaining // cache_per_page, kv_reserve_pages)`; `kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)`). **Reservation granularity stays `page_size`** — nothing new; the *unit* just gets cheaper, so `--num-pages`/`--num-tokens`/`--moe-cache-auto` all resolve automatically. `cache_per_page` includes codes+scale because both are `// tokens`-divided in `unit_bytes()` (`mha_pool.py:141`, `hybrid_swa_pool.py:436`, `dsa_pool.py:152`; #408 adds `// tokens` for the block buffer at `mha_pool.py:164`) — `ft ctl stats` / `/v1/cache/status` report the smaller number (`docs/cli.md:96`).

Live rebuild (`POST /v1/cache/rebuild`, `server/api_server.py:572` → `engine.py:841`): each pool's `rebuild()` now drops `_scale_buffer` **before** `torch.cuda.empty_cache()` and re-enters the *shared* allocator, so codes+cales can never drift (`mha_pool.py:112-118` → `_alloc`; `hybrid_swa_pool.py` via `_group_geometry`/`_alloc_group`, which added a `quantized` flag to the geometry tuple; `dsa_pool.py:129`). `QSAKVCache` also nulls `_scale_buffer` on the OOM-rollback path (`qsa_pool.py:157`). The pinned invariant: `unit_bytes()` must equal what `kv_cost()` priced — `test_mha_pool_fp8.py:164`, `test_qsa_pool_fp8.py:141`, `test_kv_nvfp4.py:234`, `test_dsa_pool.py:181`.

### (g) Bytes/token claims — verified against the code

| Format | per (token, kv head) row, `head_dim=128` | source |
|---|---|---|
| bf16 | 128×2 = **256 B** | `spec_kv_bytes_per_token` |
| #354 fp8 | 128×1 + 4 = **132 B** (1.94×) | `base.py:22,46`; `test_mha_pool_fp8.py:178 test_fp8_lands_just_above_half_the_bytes`; commit body "~3% at head_dim 128" ✓ (4/128) |
| #408 nvfp4 | 128/2=64 codes + 128/16=8 block + 4 row = **76 B** (3.37×) ✓ *claim correct* | `kv_scale_bytes_per_token` @408:47 `scale_bytes += head_dim//16`; `row_bytes = head_dim//2` @:62 |
| #408 MLA latent, 512 | 256 + 32 + 4 = **292 B** vs 1024 bf16 ✓ | `docs/cli.md:216` |
| #113 fp8_e4m3 | `1 + 2/32` = **1.0625 B/elem** → 512-dim DSV4 row = 544 B vs 1024 | `dsv4_kv_quant.py:38 BYTES_PER_ELEMENT`; `dsv4_cost_model.py:53 _kv_bytes` = `ceil(head_dim*1.0625)` |
| #460 fp8-fp4 | window 528 / compressed 288 / index 68 B ✓ (arithmetic checks) | `dsv41_layout.py:16-18` |

Caveat on #408's marketing: 76 B is **per K row and per V row separately** — a 2-slab MHA token costs `2 × heads × 76` (`(1 if spec.mla else 2)`), so "76 vs 256" is a row ratio, not a token ratio.

### (h) Tests to mirror

`tests/engine/test_kv_quant_config.py` (CUDA-free, pure config): `test_kv_quant_spellings` (alias table), `test_only_the_backends_that_read_scales_declare_fp8_support` (the gate is a whitelist, not a blacklist), `test_auto_avoids_the_fast_backends_for_fp8`, `test_explicit_unsupported_backend_is_rejected`, `test_pool_families_without_a_scale_read_path_are_rejected`, `test_{qsa,swa}_keeps_fp8_available`; #408 adds `test_nvfp4_{auto_selects_triton,rejects_unsupported_pools_before_allocation,rejects_backends_without_its_layout,rejects_partial_blocks,accepts_hybrid_swa,accepts_qsa}`.
`tests/kernels/test_kv_fp8.py`: `test_scale_is_amax_over_e4m3_max`, `test_codes_match_the_reference_quantizer_and_reconstruction_is_close`, `test_encoder_inverts_the_grid_through_the_scale_one_path`, `test_zero_row_stays_finite_and_exact`, `test_codes_are_plain_bytes_and_the_kernel_decode_matches_torch`, **`test_kv_codec_has_no_arch_or_dtype_branch`** (the trap can't come back).
`tests/kernels/test_e4m3_compat.py`: `test_constexpr_probe_never_references_a_host_function`, `test_decode_f32_bitexact`, `test_decode_f16_x128_bitexact`, `test_round_to_grid_bitexact`, `test_kv_tile_scaled16_agrees_with_the_f32_loader`.
`tests/kernels/test_triton_attention.py`: `test_decode_paged_attention_decodes_fp8_scales`, `test_paged_attention_decodes_fp8_scales`, `test_extend_paged_attention_decodes_fp8_scales`, `test_select_extend_tile_uses_kv_cache_element_size`.
`tests/kvcache/test_mha_pool_fp8.py`: `test_kv_store_dtype_selection`, `test_fp8_pool_keeps_geometry_and_adds_scale_views`, `test_layer_ids_remap_applies_to_scales_too`, `test_store_kv_scatters_codes_and_scales`, `test_rebuild_resizes_codes_and_scales_together`, `test_unit_bytes_matches_the_cost_model_that_sized_the_pool`, `test_fp8_lands_just_above_half_the_bytes`, `test_hybrid_swa_pool_also_separates_compute_and_store_dtype`.
`tests/kvcache/test_qsa_pool_fp8.py`: `test_index_tiers_stay_16_bit_whatever_the_kv_store_does`, `test_store_kv_writes_the_slot_the_attend_kernel_will_read`, `test_factory_threads_kv_quant_into_the_qsa_pool`.
`tests/kernels/test_qsa_fp8.py`: `test_fp8_codes_match_the_bf16_cache_bit_for_bit`, `test_scale_arguments_are_validated`, **`test_qsa_scoring_refuses_fp8_operands`**.
`tests/kernels/test_kv_nvfp4.py` (408): `test_latent_scatter_matches_reference_and_preserves_prefix`, `test_e2m1_grid_and_round_to_even_boundaries`, `test_attention_reads_packed_cache`, `test_pool_budget_rebuild_and_layer_mapping`, `test_store_cuda_graph_replay_changes_slots`, `test_backend_decode_graph_replays_new_kv_and_page_tables`.
`tests/kernels/test_qsa_nvfp4.py`: `test_qsa_nvfp4_matches_its_bf16_decode`, `test_qsa_nvfp4_requires_both_block_scale_tensors`, `test_qsa_nvfp4_splitk_reads_value_rows`.
`tests/kvcache/test_dsv4_kv_quant.py` (113): `test_bytes_per_element_is_what_the_cost_model_assumes`, `test_scale_width_rejects_a_ragged_head_dim`, `test_round_trip_error_is_one_e4m3_step`, `test_store_kernel_matches_the_reference`, `test_store_kernel_leaves_other_slots_untouched`, `test_quantized_attention_{tracks_the_bf16_reference,honours_masked_columns,on_the_splitk_decode_path,respects_cmp_counts}`.
Plus `tests/kvcache/test_kv_cache_rebuild.py:64+` (fp8 `unit_bytes == (layers*latent + layers*4, 0)`), `tests/models/qwen4_exp/test_qsa_backend.py:277`, `tests/attention/test_dsa_kpool.py:79`.

### (i) Stated performance caveats

- #354 commit `05861fb`: scale moved **after** the dot (`scores*s_k`, `(p*s_v)@v`) — "head_dim/BLOCK_M fewer multiplies", removed the fp32 widen need, and *improved* accuracy (worst abs err 0.281→0.0996 on sm_86). `73ca76c`: extend tile was charging K/V 2 B/elem regardless of cache → fp8 got a smaller tile than it had room for.
- #354 `e4m3_compat.py:92,113`: probe-disagreement box runs the software decode, "bit-exact … but slower".
- #408 `docs/cli.md:230`: **"Capacity savings do not guarantee faster decode; packing, reconstruction, and the selected attention backend affect throughput."** Its instrument is `benchmarks/bench_kv_quant.py` (scatter + paged-decode latency per format, `--lengths 1024,8192,32768`).
- #113 `dsv4_kv_quant.py:13`: "Storage is what this buys, not tensor-core throughput."
- #460 `docs/deepseek-v41-kv-quantization-plan.md:28`: "decoding packed values also costs work … must be measured"; `deepseek-v41.md:39,112`: throughput **not** measured. → **NOT FOUND** anywhere: a hard "Triton restore path is slower than BF16/FP8" measurement. #408's own commits (`2f554c9`, `04d4621`) have **empty commit bodies** — no perf numbers shipped.

---

## 2. Which base to build on, and the conflict graph

`git merge-tree --write-tree cac247a refs/pr/N` (exit≠0 = conflict):

| PR | merge-tree | conflicting files (all vs `cac247a`) |
|---|---|---|
| **#408** | rc=1 | `docs/models.md`, `attention/triton.py`, `kernel/triton/attention.py`, `kvcache/__init__.py`, `kvcache/qsa_pool.py` |
| #354 | rc=1 | **same 5 files** |
| #113 | **rc=0** | none (DSV4-only files untouched upstream since `0ab982f`) |
| #460 | **rc=0** | none (already contains cac247a) |
| #69 | **rc=0** | none |
| #494 | **rc=0** | none |
| #337 | rc=1 | `models/nvfp4_banks.py` only |
| #447 | rc=1 | `engine/engine.py`, `models/qwen4_exp/{config,weight}.py`, `models/weight.py`, `scheduler/scheduler.py`, `tests/models/qwen4_exp/test_config.py` |

Who broke what (354→408 conflict, `git log af71ba4..cac247a -- <file>`): `kernel/triton/attention.py` + `attention/triton.py` → `84d236c feat(gemma4): serve image input (#467)`; `kvcache/__init__.py` + `kvcache/qsa_pool.py` → `08d728d feat(mm): serve image input on the Qwen families (#454)`; `docs/models.md` → 6 commits (`#486,#480,#481,#479,#467,#454`). `kvcache/base.py` had **zero** upstream commits since the branch point — the sizing seam is untouched, i.e. cheap to re-apply.

**Overlap / supersession:** #354 ⊂ #408 ⊂ #460 by commit ancestry — #408 *literally contains* #354's tip (`9b103b0` is an ancestor), including its review fixes. Building on #354 alone means re-doing #408's rebase. **#113 is not superseded but is mutually exclusive at the CLI**: it declares `--kv-cache-dtype` with `dest="kv_quant_cache_dtype"`… precisely `dest="kv_cache_dtype"`, choices `("auto","fp8_e4m3")` (`refs/pr/113:args.py:508`), vs #354's `dest="kv_quant"`, choices `auto|bf16|fp8` — argparse rejects a duplicate `--kv-cache-dtype` option string, so merging #113 with #408 is a **hard conflict on the flag itself even though `merge-tree` is clean** (verified: the 113↛cac247a merged tree keeps exactly **1** occurrence, `args.py:681`). #113's merge-base is 49 commits behind the base; its `_validate_kv_cache_dtype` hook and its `dsv4_args.kv_quant` stamping are the parts worth stealing; its native-fp8-pointer approach contradicts #354's hard-won `uint8` rule.

**→ Base on #408** (it *is* the fp8+nvfp4 stack, generalised gate, 5 conflicting files all shallow and named). Merge order: #408 first, then rename #113's flag to something DSV4-scoped or fold `fp8_e4m3` into `KV_QUANT_ALIASES` and its `bool kv_quant` into the string token.

---

## 3. DSpark speculative decoding shape (#69)

Module docstring is the spec: `refs/pr/69:python/freetoken/models/deepseek_v4/dspark.py:1-28` — semi-autoregressive **block** drafter under `mtp.{0..n_mtp_layers-1}`, proposes `dspark_block_size` tokens/pass, Markov transition bias + confidence gate; draft layer ids continue the target's (`n_layers + k`) so expert banks/slot caches/KV pools address both with one index space.

| Stage | Where | Notes |
|---|---|---|
| flags | `server/args.py` `--speculative-dspark` (`+` block); `engine/config.py` +20 | `engine.py:1537` rejects a checkpoint with no drafter |
| batch → block | `scheduler/scheduler.py:910 _maybe_make_speculative` | verify is **not a new forward kind — it is a PREFILL of `1+k` tokens/req**: `req.append_host(noise)`, `device_len += k`, `batch.phase="prefill"`, `batch.speculative=True`, `batch.spec_block=k`, `batch.release_tail = cache_manager.release_speculative_tail`. `_SpeculativeConfig(block_size, noise_token_id)` `:1012`, `_speculative_config(config)` `:1018` |
| draft tokens | `engine/engine.py:1297 draft_into_batch` | snapshots carry first, `model.catch_up_draft_context(batch)`, `model.draft()`, `sampling_probs` → **q**, proposes by *sampling* q (argmax would void the p/q guarantee), writes `batch.input_ids[1:] = proposed[:-1]`; `dspark.py:243 propose` returns logits not tokens; `:280 block_input_ids` = last token + `noise_token_id`×(k−1) |
| variable-length handling | `core.py` `Batch.speculative/spec_block/draft_confidence/draft_probs/carry_snapshot/release_tail/spec_emitted` (7 new fields) | **no variable-shape graph**: the width reduction happens in the *host* loop, and `draft_width(confidence, threshold, block)` (`dspark.py:440`) truncates `proposed[:width]`. `#69`'s hybrid path (`f0f3f07`) does not enable CUDA graphs for spec decode — verify rides the prefill graph path with `1+k` per req; `_finish_speculative:1184` *asserts* `logits.shape[0] == (1+k)*n_reqs`, i.e. the graph keeps the full width and acceptance collapses after. NOT FOUND: a padded/adaptive-width graph capture for spec decode. |
| acceptance | `dspark.py:343 accepted_prefix` (greedy, prefix-only, bonus token at first disagreement) / `:399 rejection_accept` (`px > u*qx` product form, residual `max(0,p−q)` resample) / `:366 sampling_probs` (temperature+top-k+top-p applied to **both** p and q) | |
| commit | `engine.py:1154 _finish_speculative` | per-req: `width = min(draft_width(...), max_device_len-start-1)`, `req.input_ids = req._ids_buf[:keep]`, `append_host(bonus)`, `cached_len,device_len = keep, keep+1`; `core.py Req.complete_n(n)` (new, advances per request not per batch) |
| rollback (KV) | `scheduler/cache.py:260 release_speculative_tail` | frees the abandoned tail's pages **and** SWA slots *before* `device_len` drops — otherwise a silent leak surfacing only as `SWA-slot leak: free(11520)+tree(256) != capacity(12160)` at the next idle integrity check |
| rollback (state) | `models/deepseek_v4/rollback.py:33 CarrySnapshot` / `:88 needs_rollback` | `read_carry_blocks/write_carry_blocks` per (layer, tier ∈ {`attn`,`idx`}), `.clone()`d; restore is idempotent. `needs_rollback` = only when the last kept and first dropped position share a window page (`//page_size` equal) |
| driver | `dspark.py:465 SpeculativeLoop.step:491` → `StepResult(tokens, drafted, verified, accepted, rolled_back)` | the 3 orderings that "go wrong in ways that do not crash": snapshot **before** draft, width **before** verify, prefix+bonus always ≥1 token. Ops injected → GPU-free testable |
| model side | `models/deepseek_v4/model.py` `draft()`, `catch_up_draft_context()`, `last_aux_hidden()`, `last_aux_addressing()`, `embed_tokens()`, `logits()`; `dspark.py:178 store_context_kv`, `:225 catch_up_context`, `:54 MarkovHead`, `:77 ConfidenceHead`, `:102 DSparkDrafter`, `:301 hc_head`, `:313 head_hidden`, `:529 window_cols_for_block` | |

**DSV4-specific:** `aux_hidden` taps at `dspark_target_layer_ids` (40,41,42); the compressor/indexer **carry** (§`rollback.py:1-28`) is DSV4's rolling per-request state — a plain paged pool needs no snapshot at all; the FP4 offload expert banks + hyper-connections inside draft layers; `n_mtp_layers` full DSV4 blocks; `models/deepseek_v4/parallel.py`.
**Generalisable to `qwen4_exp` MTP:** everything from `scheduler.py:910` down — `_SpeculativeConfig`/`_maybe_make_speculative` (1+k-prefill trick), the 7 `Batch` fields + `Req.complete_n`, `accepted_prefix`/`rejection_accept`/`sampling_probs`/`draft_width`/`SpeculativeLoop`/`StepResult` (pure, model-free), `_finish_speculative` + `draft_into_batch` as templates, `release_speculative_tail` (any paged+SWA pool), `cache.py:351,366` reply accounting. **Missing for `qwen4_exp` today:** the base *drops* `mtp.*` (`models/qwen4_exp/weight.py:89` `if raw_name.startswith("mtp."): skip`; same in `qwen3_5_moe/weight.py:58`, `glm5_next/weight.py:48`) — a loader is new work. Two ready seams: `kvcache/qsa_pool.py:49 ring_capacity_for(index_ratio, num_speculative_tokens)` already sizes the QSA ring for spec depth, and `kernel/fla/fused_sigmoid_gating_recurrent.py:59 HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` + `:319` "they can differ under `--speculative-adaptive`" already anticipate GDN-state tree masking (qwen4_exp is the GDN/hybrid-linear family). #69's `#70` sibling is not fetched → the [2/3] series is incomplete.

---

## 4. NVMe expert tier (#337)

`refs/pr/337:python/freetoken/moe/disk_tier.py` (685 new lines). Model: **VRAM ← RAM ← NVMe**, RAM keeps experts `[0, ram_experts)`, tail stays on disk in the **original safetensors shards** (no FTW conversion; each expert tensor is contiguous → a bank row is 1–2 aligned `preadv`s, `disk_tier.py:1-30`).

- **Data structure:** `DiskTierSpec(ram_experts)` `:42` (engine→loader, per `expert_ram_experts`); `Nvfp4DiskIndex` `:182` built from `_read_safetensors_offsets(path)` `:168` + shard scan, answering `row_segments(bank_idx, layer, expert) -> [(fd_idx, off, nbytes)]` `:241`; `DiskTier` `:251` holds `cache.banks` as `[(per_layer_host, gpu_cache)]`, per-bank `_row_bytes`, `_dst_slices` (fused gate|up split at the row midpoint), a thread-local pinned **staging ring** `_staging_ring()` `:323` sized `max_row` rounded to 4096 (+2 pages), fd cache with O_DIRECT and a **plain-preadv fallback** for tmpfs/overlayfs `_fd()` `:296`, `ThreadPoolExecutor(workers=8)`.
- **Admission:** none of its own — the GPU slot cache's existing LRU assigns the slot; `fetch_pending` reads `cache.src_indices/evict_slots/num_indices` and treats `expert >= self._ram` as disk-resident (`:646`), i.e. **admission by expert id, decided at load, immutable at runtime**. `materialize_layer(cache, layer_id, expert_ids)` `:594` for whole-layer/prefill.
- **Eviction:** `cache.usage[disk] = cache.step` (`:640`, just before `fetch_pending`) — disk experts consume an LRU slot but never a PCIe budget. Nothing evicts *from* RAM to disk. Post-load `release_bank_tails()` `:52` `MADV_DONTNEED`s the tail rows (`:277 test_release_range_frees_pages`), `tail_resident_bytes` `:74` / `check_tail_unbacked` `:96` verify it; **best-effort** — skips + warns when `N*row_bytes` isn't 4096-aligned.
- **Readahead: none.** `_fetch_expert` is submitted per miss and `f.result()`-joined immediately, then `_sync_fetches()` `:396` — **synchronous fetch, the layer waits for its disk misses** (module header `:19`). No overlap, no prediction, no O_DIRECT prefetch window.
- **Budget claims:** it does **not** go through the memory planner. `--expert-ram-experts N` *is* the RAM budget (N pinned experts/layer, `args.py` `+`); VRAM is untouched (slot cache size unchanged). Host release is via `MADV_DONTNEED`, not a reservation. Integration = `OffloadMoeCache.attach_disk_tier(index, ram_experts, workers)` `offload_cache.py:1028` + `_disk_tier` `:161` + `copy_missing` shrink-to-RAM-remainder `:1040` + `rebuild` → `_disk_tier.refresh(self)` `:537`. Gating collects **all** unmet preconditions and raises once: `0 < ram_experts < E`, `decode_target == "gpu"`, `moe_prefill_overlap` off, `cuda_graph_max_bs == 0`, native `nvfp4` banks (`engine.py:547-569`, `offload_cache.py:1032-1035`). Fails loud if the provider returns no disk index after releasing rows (`engine.py:656` `NotImplementedError`). Debug/verify env: `FT_DISK_TIER_VERIFY` → `_verify_slot:410`, `_ref_row:442`, `_identify_overwriter:465`, `verify_ram:496`, `verify_decode_mapping:559`. Stats: `{"experts_fetched","bytes_fetched"}` `:684`.

---

## 5. GGUF today

**Base (`cac247a`) already owns a full native-GGUF path.** Entry points:

| Seam | file:line |
|---|---|
| GGUF file/config reader | `models/gguf/reader.py:23 is_gguf_path`, `:42 gguf_config_source`, `:60 write_metadata_gguf`, `:102 GgufTensor(.packed())`, `:130 load_gguf_metadata`, `:136 gguf_architecture`, `:143 iter_gguf_tensors`, `:174 gguf_tensor_names` |
| HF-config shim | `models/gguf/config.py:25 GgufConfigShim`, `:61 build_gguf_shim` |
| tokenizer | `models/gguf/tokenizer.py:19 load_gguf_tokenizer`, `:55 gguf_eos_token_ids` |
| type metadata + torch reference dequant | `models/gguf/dequant.py:31 BLOCK_SHAPE`, `:49 row_bytes` (**single source of truth for packed row bytes**, shared by packed weights and expert banks), `:67 dequant_q4_0`, `:82 dequant_q6_k`, `:124 dequantize` |
| packed linear/embedding layers | `layers/gguf.py:43 fused_mul_mat_gguf`, `:68 GGUFLinear`, `:91 GGUFEmbedding` |
| vendored ggml CUDA | `kernel/gguf.py:51 _module`, `:79 ggml_dequantize`, `:86 ggml_mul_mat_vec_a8`, `:93 ggml_mul_mat_a8`, `:100 ggml_moe_a8`, `:118 ggml_moe_a8_vec`, `:131 ggml_moe_get_block_size` + `kernel/csrc/gguf/{mmq,mmvq,vecdotq,dequantize,moe,moe_vec,ggml-common}.cuh` |
| model adapter (only one arch) | `models/gemma4/gguf.py:51 parse_gguf_config`, `:31 _full_rotary_dim`, `:187 iter_gguf_weights`, `:306 is_gguf_model`, `:311 GGUFTiedLMHead`, `:340 convert_gemma4_to_gguf`, `:391 load_q4_0_expert_sources`, `:382 _q4_0_expert_specs` |

Base coverage is **narrow**: dequant/reference = `F32/F16/BF16/Q4_0/Q8_0/Q6_K` only (`dequant.py:24-47`), kernel dispatch sets `_MMVQ = _MMQ = _DEQUANT = {Q4_0,Q8_0,Q6_K}` (`layers/gguf.py:51-53`), and **one architecture** (gemma4; `models/gguf/config.py` + `tests/models/test_gemma4_gguf_rope.py` are the only non-gemma4 GGUF code/tests). No generic `GgufForCausalLM`.

**#494 = "mixed-quant" precisely:** the base **hard-codes the type per role** — qkv/o/shared-MLP → `GGML_Q4_0`, embedding+LM head → `GGML_Q6_K`, and `expert_quant="q4_0"`/`moe_weight_format="q4_0"` (docstring `gemma4/gguf.py:8-10`) — so a checkpoint whose tensor table mixes types per layer or per role cannot be built. Three fixes:
1. **discover the layout** from the tensor table without touching payloads: `gemma4/gguf.py:51 _gguf_quant_layout(model_path, num_layers, num_experts)` → `{embedding, qkv[layer], attn_output[layer], shared_gate_up[layer], shared_down[layer], expert_gate_up[layer], expert_down[layer], expert_gate_up_bytes[layer], expert_down_bytes[layer]}`, validated (a role disagreeing across layers → `ValueError`; missing expert tensor → `ValueError`), `None` for metadata-only FTW sources (keeps legacy defaults). Stored on the new `ModelConfig.gguf_quant_types` (`models/config.py:317`).
2. **thread per-tensor types** into the ops: `convert_gemma4_to_gguf` swaps with `layout[role][layer_id]` instead of constants; `layers/moe.py`/`moe/expert_banks.py`/`moe/fused_q4_0.py`/`offload_cache.py` take **independent gate_up/down quant types** (`gguf_quant_types=banks.gguf_quant_types`, `engine.py:693`); packed expert bytes are preserved while slots are padded (loader passes the slot stride to the CUDA kernel).
3. **widen the type table**: `dequant.py:22-56` adds `Q4_1,Q5_0,Q5_1,Q2_K,Q3_K,Q4_K,Q5_K,IQ1_M,IQ1_S,IQ2_S,IQ2_XS,IQ2_XXS,IQ3_S,IQ3_XXS,IQ4_NL,IQ4_XS` to `BLOCK_SHAPE`/`GGML_NAME`, with reference dequant **delegated to `gguf.quants.dequantize`** (`_dequant_gguf_py`), and `layers/gguf.py:48-78` splits `_STANDARD_AND_K` (MMVQ+MMQ) from `_IQ` (**MMVQ + dequant fallback only** — the vendored MMQ switch has no IQ case). New CUDA kernel entry points in `csrc/gguf/gguf_kernel.cu` + `moe_vec.cuh`.
Tests: `tests/models/test_gemma4_gguf_quant.py:9 test_cuda_quant_types_have_packed_row_layouts`, `:67 test_quant_layout_detects_dense_and_per_layer_expert_types`, `:104 test_mixed_expert_loader_pads_slots_but_preserves_packed_bytes`, `:144 test_gguf_expert_gemm_passes_independent_quant_types`, `:41 test_gguf_config_shim_can_be_masked_like_hf_config`.
→ For a Phase-2 GGUF plan: reader/type-metadata/packed-linear/ggml-CUDA-plumbing already exist; what doesn't is a **second architecture adapter** and any type the vendored kernels lack.

---

## 6. Verdict table

| PR | Verdict | Reason (evidence above) |
|---|---|---|
| **#408** | **NEEDS-PORT — and this is the one to port** | Superset of #354 by ancestry; `merge-tree` rc=1 on 5 shallow files whose upstream authors are named (`#467`, `#454`); `kvcache/base.py` untouched upstream. Empty commit bodies + no perf numbers ⇒ unreviewed state (NOT FOUND: any "address review" commit). |
| #354 | REFERENCE-ONLY | Strict ancestor of #408. Best-documented recipe (full commit bodies, trap rationale) but taking it alone loses nvfp4 + the generic gate. rc=1, same 5 files. |
| #113 | REFERENCE-ONLY | Textually clean (`rc=0`) but semantically collides on `--kv-cache-dtype` (`dest=kv_cache_dtype`/`{auto,fp8_e4m3}`) and uses **native fp8 pointers**, the exact thing #354 forbids; 49 commits stale, DSV4-only, 1 commit, no review trail. Lift `_validate_kv_cache_dtype` + the `dsv4_args.kv_quant` **stamp-before-pool-exists** ordering + `--moe-cache-auto` coupling. |
| #460 | REBASE-CLEAN, REFERENCE-ONLY | `rc=0`, contains cac247a. +14,829 across 132 files, model-new; but `dsv41_layout.dsv41_row_bytes()` + the design doc are the clearest packed-row (codes-then-scales) precedent, and its own doc admits throughput unmeasured. |
| #69 | REBASE-CLEAN | `rc=0`; but it is `[2/3]` of a series (`refs/pr/70` absent) and bundles unrelated TP/NUMA/banner work; acceptance/rollback/loop logic is pure and DSV4-independent. |
| #337 | NEEDS-PORT (small) | `rc=1`, **one** file (`models/nvfp4_banks.py`), 2 commits, last one its own bugfix; explicitly self-labelled "v0 prototype": no readahead, no CUDA graphs, nvfp4-only, id-partitioned not LRU. |
| #494 | REBASE-CLEAN | `rc=0`, 1 commit, GGUF-only, additive to `ModelConfig`; narrowest blast radius in the set. |
| #447 | NEEDS-PORT | `rc=1` on 6 files incl. `engine/engine.py` and `qwen4_exp/{config,weight}.py`; **only PR with an explicit review trail** ("address review — Copilot review on #447", 5 substantive fixes incl. a `copy_missing` stale-`_pending_src_layer` slot-overwrite bug). |

### Highest-value lifts for the roadmap

1. **KV formats — the four seams #408 generalised** (build all of Turbo3/4/TCQ/VBR on these, no new paths): `getattr(BackendInfo, f"supports_{kv_quant}_kv")` (engine.py:143) for capability; `base.kv_storage_bytes_per_elem` + `kv_scale_bytes_per_token` + `spec_kv_bytes_per_token` (base.py:22/37/49) as the *only* cost model, asserted equal to `unit_bytes()`; `dtype` (compute) vs `store_dtype` (buffer) with the `itemsize==2` assert at the backend; and one `_alloc()` shared by first allocation *and* `rebuild()` (`mha_pool.py:73`, `hybrid_swa_pool._alloc_group_storage`). Add to that the two hard rules: **`uint8` codes, no fp8/fp4 type in any kernel signature or pointer**, and **fused quantize+scatter in one launch** with device `out_loc`.
2. **MTP / spec decode — `scheduler.py:910` + `dspark.py:343-528`.** "Verify is a `1+k`-token prefill, phase flipped" plus `Req.complete_n`, the 7 `Batch` fields, `accepted_prefix`/`rejection_accept`/`draft_width`/`SpeculativeLoop` are model-free; only `CarrySnapshot` is DSV4's compressor (qwen4_exp's analogue is the GDN state ring — note `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` and `ring_capacity_for(num_speculative_tokens=…)` already exist in the base). Must-copy bug: `release_speculative_tail` **before** lowering `device_len`.
3. **Tiering — the *precondition* pattern, not the tier.** #337's "collect all unmet preconditions, raise once" (`engine.py:547`) + `attach_disk_tier` asserts + `refresh()` on rebuild + `MADV_DONTNEED`-after-load with a warn-don't-abort alignment rule; and #447's `OwnerCacheGeometry`/`partition_route` for a real multi-level ownership model.
4. **GGUF — `_gguf_quant_layout()`** (#494) as the pattern for "discover formats from the tensor table, never assume per role", plus `ModelConfig.gguf_quant_types` as the carrier into the layer/bank constructors; `row_bytes()` stays the single source for packed sizes.
5. **Memory accounting — `kv_scale_bytes_per_token` priced in `base.py`, not in the pool** (so `kv_cost` and the allocation cannot disagree) + the invariant tests (`test_unit_bytes_matches_the_cost_model_that_sized_the_pool`, `test_latent_budget_matches_allocations_and_rebuild`, `test_bytes_per_element_is_what_the_cost_model_assumes`) + `kernel/aot_models.aggregate_store_element_sizes()` — every new code width needs its row size registered or the `FREETOKEN_DISABLE_JIT=1` release gate fails.