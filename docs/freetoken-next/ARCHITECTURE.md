# ARCHITECTURE — freetoken-next

Current truth about the design. §1–2 describe the inherited base as it exists; §3–7 are
target designs, each with the file:line seams a patch must touch. Evidence lives in
`audits/A1…A6` (source audits of this base, llama-turbo-optimal, the MTP/GGUF corpus, GGUF
feasibility, FreeToken-Kai, and the upstream PR refs) — this file states conclusions, not
the whole investigation.

## 1. Inherited subsystem map (upstream v0.1.3, `cac247a`)

```
python/freetoken/            engine, installed as `freetoken` with the `ft` CLI
  server/                    OpenAI / Anthropic / Responses HTTP APIs, streaming, tool parsers
  scheduler/                 chunked prefill (PrefillAdder, max_extend_tokens=8192), cache
                             manager, commit/window locking, overlap_loop
  kvcache/                   paged pools (mha/dsa/hybrid_swa/dsv4/linear_state) + radix caches
  moe/                       expert offload cache, CPU/GPU/hybrid executors, host banks
  models/                    registry + per-family loaders; models/gguf/ = native GGUF reader
  kernel/                    CUDA/Triton kernels, JIT, csrc/ (_pinned_tensor, _cpu_moe,
                             _ple_store, gguf/, csrc/jit/store.cu)
  layers/, attention/        fused ops; backends triton/fa/fi/trtllm/dsv4_sparse/dsa/
                             m3_sparse/qsa_sparse behind a BackendInfo capability matrix
  engine/                    cache_budget.py, config resolution, CUDA-graph capture, pin budget
  checkpoint/                HF -> FTW fast-load conversion
tests/ mirrors the subsystem tree; benchmarks/ has bench_decode_moe, bench_load_weight_generic,
bench_offload_cache_copy and (new) bench_pp_tg.
```

Auto-resolution is the product promise: `ft serve --model X` resolves dtype, attention
backend, MoE strategy/cache, KV capacity, graph sizes, parsers from checkpoint + GPU.
Measured on this host for Qwen3.6-35B-A3B: `moe_strategy='offload'`,
`attention_backend='fi'`, `cache_type='hybrid_radix'`, `page_size=1`; for Flash-Next:
`attention_backend='qsa_sparse'` with **page size overridden to 64**, `ple_backend='disk'`
(io_uring, O_DIRECT, wait-sync), `experts: nvfp4 via triton`.

## 2. What the base already provides (verified, A1/A4)

- **Quant abstraction is weights-only and real**: `QuantKind`
  (`layers/quantization/scheme.py:20`: none/fp8_tensor/fp8_block/mxfp8/nvfp4/mxfp4),
  `QuantConfig` dialects (`configs/{compressed_tensors,fp8,modelopt,mxfp4}.py`, factory
  `configs/base.py:112 from_hf`), `QuantMethod` with an ordered kernel table
  (`method.py:21 select_kernel`) and `finalize_quant()` (`method.py:64`), global
  `QuantBackend` selection (`quant_backend.py`, CLI `--quant-backend`).
  **KV has none of this**: `KV_CACHE_DTYPE_BYTES = 2` (`kernel/aot_models.py:36`), KV pool
  dtype == compute dtype (`engine/engine.py:312,396`).
- **Paged KV**: one slab per layer, `torch.empty((2, L, num_pages, page_size, kv_heads, head_dim))`
  (`kvcache/mha_pool.py:50`); append via `store_kv` → JIT CUDA scatter
  (`kernel/store.py:30` → `kernel/csrc/jit/store.cu:28`); growth via
  `CacheManager.allocate_paged` (`scheduler/cache.py:260`, called `scheduler.py:810`);
  live resize via `rebuild_from_config`/`validate_rebuild` (`kvcache/base.py:128,83`);
  `spec_kv_bytes_per_token` is the cost model (`kvcache/base.py:19`).
  **Host-RAM KV: not present** — only `# TODO: support HiCache` (`kvcache/base.py:197`).
- **Native GGUF exists but narrow** (§3).
- **Expert offload**: `OffloadMoeCache` (`moe/offload_cache.py:104`), 8 bank schemas
  (`_BANK_SCHEMAS` `:33–96`: bf16, fp8_block, **q4_0**, nvfp4, nvfp4_marlin, nvfp4_b12x,
  mxfp4_triton, ds_fp4), device-side LRU admission (`ensure_experts` `:843`, `flashlib`),
  fused multi-bank copy (`copy_missing` `:1011`), pinned host source banks
  (`moe/host_banks.py:78 HostBank`, pin-after-fill via `PinPipeline` `:286`), prefill
  double-buffer (`:645–841`), D2D hit reuse (`_hit_d2d_usable` `:703`, off by default),
  counters (`decode_miss_stats` `:928`).
- **Pinned-host precedent for a huge table**: Flash-Next PLE `PinnedUVATable`
  (`models/qwen4_exp/ple.py:121`, "47.7 GiB FP8 n-gram store, gathers rows over UVA") and
  the `_ple_store` disk row store. These are the models for RAM-tiered KV.
- **CUDA graphs**: per-bs capture (`engine/graph.py:143`), static `GraphCaptureBuffer`
  (`:29–75`), `prepare_for_capture/prepare_for_replay` (`attention/base.py:66–72`); stable
  shapes required: bs ladder (`graph.py:87`), page-table row width
  (`engine/engine.py:421–430`), pool **object identity across rebuilds**
  (`mha_pool.py:59`, `needs_rebind_on_rebuild` `kvcache/base.py:45`).

## 3. Native GGUF (Phase 2)

**Already there** (`models/gguf/`, `layers/gguf.py`, `kernel/gguf.py`,
`kernel/csrc/gguf/`): `is_gguf_path`/`iter_gguf_tensors` (`models/gguf/reader.py:23,143`)
hand out **zero-copy uint8 `[rows, row_bytes]` views over `np.memmap`**
(`GgufTensor.packed()` `:110`) — no whole-file dequant, GGUF stays immutable, and the
dequant happens *inside* vendored ggml CUDA kernels: `ggml_mul_mat_vec_a8` (MMVQ, ~19 type
cases), `ggml_mul_mat_a8` (MMQ, 14 cases), `ggml_dequantize`, `ggml_moe_a8`,
`ggml_moe_a8_vec` (`kernel/gguf.py:79–131`). `GGUFLinear`/`GGUFEmbedding` keep rows packed
(`layers/gguf.py:68,91`); `GGUFTiedLMHead` shares the embedding's packed `qweight`
(`models/gemma4/gguf.py:312`). Scales are **inline in each ggml block**, so one bank per
projection is enough (unlike the engine's 6-bank NVFP4 layout).

**The four gaps**, in order of cost:

| Gap | Where | Work |
|---|---|---|
| Arch coverage: **one** adapter (gemma4) | `models/gguf/config.py:20 GGUF_ARCH_TO_REGISTRY`, `models/register.py:263` | per-family `parse_gguf_config` + `iter_gguf_weights` (name map + packed `cat(dim=0)` fusion), model-side `GGUF*` swap, registry spec, `kernel/aot_models.py:58 arch_aliases` |
| Type coverage is **Python-side only** | `layers/gguf.py:35–37` (`_MMVQ=_MMQ=_DEQUANT={Q4_0,Q8_0,Q6_K}`), `models/gguf/dequant.py:30 BLOCK_SHAPE` | widen the two tables to the cases the CUDA switch really has (Q4_1, Q5_0/1, Q2/3_K, **Q4_K, Q5_K**, IQ4_NL/IQ4_XS via MMVQ); `dequant_q8_0` reference is missing entirely |
| **No shard joining** | `models/gguf/reader.py:23` accepts one file; gguf-py has no split support | resolve `general.split_count`/`split_no`, mmap each part, per-file local offsets (llama.cpp does this in C++ at `src/llama-model-loader.cpp:1021-1045`) |
| GGUF bypasses `QuantConfig` | `models/register.py:321-326` returns `None`; `engine/engine.py:772` FIXME ("q4_0 banks have no quant method yet") | register GGUF types as a dialect later; **not** in the first increment |

Per-type verdict (A4 §3, block sizes from `csrc/gguf/ggml-common.h`): F32/F16/BF16/Q4_0/
Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/**Q4_K**/**Q5_K**/Q6_K = **DIRECT** (kernel case exists,
tables need widening); IQ4_NL/IQ4_XS = direct for decode (MMVQ), dequant-fallback for
prefill; IQ1/IQ2/IQ3 family = decode-only today, prefill materializes the weight per call
→ **NEW-KERNEL**; TQ1_0/TQ2_0, ggml MXFP4/NVFP4, Q1_0/Q2_0/Q2_0_G128 = **NEW-KERNEL**;
turbo types are **runtime-only KV codecs, never GGUF tensors** → irrelevant to the loader.

**Local corpus** (A3 §C): single-file, servable now → `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf`
(16.24 GiB, arch `qwen35moe`, Q4_K_M, 41 blocks, 256 experts, `nextn` at blk.41) and
`Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` (17.35 GiB, same geometry, IQ4_XS); dense-hybrid
`Qwen3.8-27B-…-IQ3_S-MTP-…gguf` (11.15 GiB, arch `qwen35`); sharded → the two
`qwen4exp` targets (3 and 33 shards, 87–88 GiB, `per_layer_token_embd.weight [160, 3.2e8]`
IQ4_NL/Q5_1 ≈ 45 GiB) which also need shard joining + PLE-table mapping.

**Increments**: I1 = upstream **PR #131** ported onto this base (`refs/pr/131`; 21 ggml types,
GGUF expert banks, K-quant CPU kernels, qwen35moe/qwen3moe/deepseek_v4 adapters, shard
joining, its own tests) — it merges with 7 conflicts because it predates `#418`/`#427`, so the
leaf files are adopted and those seams re-authored (D-009). I2 = per-layer expert geometry in
the slot pool (`_BANK_SCHEMAS`, `expert_banks.py:175 _PROVIDERS`), which is what actually
unblocks the two local MoE GGUF files: `Ornith-…-APEX-MTP-I-Compact.gguf` mixes Q3_K ×30 +
Q4_K ×10 and `Tiel-Coder-…-UD-IQ4_XS.gguf` is unsloth-dynamic, both against one stride — I1
alone correctly refuses them. I3 = `qwen4exp` adapter + `per_layer_token_embd` → PLE table
(needed by both sharded Flash builds). I4 = prefill kernels for the IQ types (MMQ has no IQ
case, so I-quant prefill dequantizes today). `mtp.*`/`nextn.*` stay dropped with a logged
warning in every adapter until §6 lands.

**Discover, never assume, the per-role type** — upstream #494's pattern (`refs/pr/494`,
`merge-tree` clean): `_gguf_quant_layout(model_path, num_layers, num_experts)` reads the
*tensor table* only (no payloads) into a per-role, per-layer map, raises when a role disagrees
across layers, and the result rides `ModelConfig.gguf_quant_types` into the layer and bank
constructors, which take **independent gate_up / down types** and pad the slot stride while
preserving packed bytes; dispatch splits `_STANDARD_AND_K` (MMVQ+MMQ) from `_IQ` (MMVQ +
dequant fallback). `row_bytes()` stays the single source for packed sizes. This is the shape
I2 must keep.

## 4. Turbo3 / Turbo4 / TCQ / VBR KV (Phases 3–4)

Spec extracted from `/models/servers/llama-turbo-optimal` (A2). Runtime-only KV types
(`ggml/include/ggml.h:433-447`), **fixed-size rows**, one `ggml_half norm` per block and
nothing else — no zero/min/scale array:

| type | blck_size | type_size | bpv | B/head@256 (= our `head_dim`) |
|---|---|---|---|---|
| turbo2 | 32 | 10 | 2.5 | 80 |
| **turbo3** | 32 | 14 | 3.5 | 112 |
| **turbo4** | 128 | 66 | 4.125 | 132 |
| turbo8 | 128 | 130 | 8.125 | 260 |
| turbo3_tcq / turbo2_tcq / turbo1_tcq | 128 | 52 / 36 / 20 | 3.25 / 2.25 / 1.25 | 104 / 72 / 40 |

Against the numbers in PERFORMANCE.md §4 this is the whole point: Flash-Next 1M context
24 GiB (BF16) → **13.2 GiB at turbo4**, **11.2 GiB at turbo3**, 8.6 GiB at turbo2;
35B-A3B 20 GiB → 11 GiB / 9.3 GiB / 7.6 GiB. **Both fit inside our KV budget at 1M with
Turbo3 alone.**

Algorithm (encode, CUDA is authoritative — the CPU refs use a *different* rotation and the
turbo2/3/TCQ refs are stubs that zero the payload; never cross-validate them):
mean-sub tap (optional, per-model baked calibration) → per-channel InnerQ scale → L2 norm →
normalize → `x *= s1` → **normalized FWHT over 128** (× 1/√128 = 0.08838834764831845) →
`x *= s2` → nearest centroid (`<` against midpoints; a tie takes the higher index) → store
`norm` (+ `recon_norm`). Turbo3 splits the 3-bit index across two arrays (`qs` low 2 bits
4/byte, `signs` high bit 8/byte); turbo4 packs nibbles low-first; turbo8 is a uniform
256-level grid with per-block absmax. **Inverse = the same FWHT call with s1/s2 swapped**
(it is an involution). TCQ = bitshift trellis (t3: 512 states, L=9; t2: 256; t1: 256) with a
single Viterbi pass over the 128 coordinates of one rotation group, sequential backtrace,
**codebook indexed by state and separate for K and V**, on-disk bits = 6 (or 7) initial-state
bits then the 128 symbols so decode reads a 9-bit window at bit offset `3t` (no state
tracking), plus a V-side norm alpha (1.04 encode / 1.02–1.26 decode). VBR is **not** a type:
it is a per-(layer, K/V side) tier controller over contiguous cell ranges, retiering by
in-place whole-row transcode (reverse tile order), physical pages mapped/unmapped through a
VMM reservation so **addresses never move** — graph `epoch` invalidation on move
(`fattn.cu:1374-1470`).

Constraints that our design must honour:
1. `head_dim % 128 == 0` (ours is 256 ✓), pad per-head to a multiple of 128, `head_dim > 512`
   falls back to f16.
2. K and V are separate tensors with **independent types** (mixed tiers are the point of VBR).
3. RoPE is applied **before** quantization; therefore Q must be FWHT-rotated before QK and
   the attention **output un-rotated** afterwards (`llama-graph.cpp:3045-3070`) — that
   rotation pair is part of the attention backend, not the pool.
4. Written quantized at append time (`ggml_set_rows` in `cpy_k`/`cpy_v`).
5. Fused dequant-in-attention is **decode-only** (`Q->ne[1] <= 4`); prefill materializes, and
   forcing fused prefill measured −6…−11 %. Asymmetric (non-adjacent) tier pairs fall back to
   materialize at −13…−15 % TG.
6. Scratch VRAM must be *budgeted*, not discovered: LTO keeps per-context
   `ggml_cuda_fattn_scratch` + a projection `(row_k+row_v) * wm_cells` folded into the fit
   test (`llama-kv-cache.cpp:4565-4571`), and a 128K OOM was traced to two 120 MiB scratch
   reservations. This is exactly the §5 ledger's job.

**Port plan onto the base** — build on upstream **PR #408's seams** (A6 §1/§2: #408 is a strict
superset of #354, and #354 of #113's flag spelling; port #408, reference the rest), whose
extensible core is: `--kv-cache-dtype` → `EngineConfig.kv_quant` → alias table →
`getattr(BackendInfo, f"supports_{kv_quant}_kv")` capability gate that makes *auto* skip
incapable backends and startup *refuse* (not ignore) an incapable explicit choice;
`kv_storage_bytes_per_elem` + `kv_scale_bytes_per_token` feeding the single cost model
`spec_kv_bytes_per_token`, asserted equal to `unit_bytes()`; **`dtype` (compute) vs
`store_dtype` (buffer)** with the `itemsize==2` assert at the backend boundary; one `_alloc()`
shared by first allocation and `rebuild()`. Then, per Turbo tier:
(a) storage: pool allocates packed rows (`[num_pages, page_size, kv_heads, row_bytes]` uint8)
+ a `norm` sidecar (or inline, matching the block struct);
(b) write: a quantizing variant of `kernel/csrc/jit/store.cu` / Triton `_store` that runs the
norm→FWHT→centroid chain inline and never materializes the row;
(c) read: **first increment** a fused Triton decode attention that consumes packed rows
directly into shmem (the LTO `fattn-mma-turbo` shape, and our `attention/triton.py` +
`qsa_sparse.py` are the hosts); dequant-to-persistent-scratch is only an interim and must be
measured, because the LTO data says materialize costs 13–15 % TG;
(d) Q-side rotation + output un-rotation inside the backend, so the pool stays geometry-simple;
(e) TCQ only after turbo3/turbo4 are end-to-end green (LTO's own conclusion: "TCQ's extra
compression is not a net win here" at 3.25 vs 3.5 bpv — 33.64 vs 39.29 TG).
Preserve BF16/FP8(+#408 NVFP4) and A/B them all with `--kv-cache-dtype`.

**Four rules #354/#408 already paid for — the port inherits them, it does not relitigate them**
(A6 §1d, §1f, §1h):
1. **Codes live in a plain `uint8` buffer on every architecture, and the fp8/fp4 type never
   appears in a kernel signature *or* a pointer.** Two independent routes failed on real
   hardware: the compile-time probe (`e4m3_native_cx()`) answers independently from the host
   that allocated the buffer and disagreed on sm_100, and branching on a pointer's element
   type is *not* statically pruned — Triton type-checks the dead arm and dies at CUDA-graph
   capture (`cannot cast int32 to fp8e4nv`). What remains is a straight-line software decode.
   Turbo3/Turbo4 are naturally on the same side of this (their payload is `qs` bytes + one
   `ggml_half` norm), and TCQ's 9-bit window decode is straight-line too.
2. **Quantize and scatter fuse into one launch** with `out_loc` kept a device tensor, so
   capture never sees a host sync.
3. `unit_bytes()` **must equal** what `kv_cost()` priced, in every pool family, after
   `rebuild()` too — that invariant is what makes the §5 ledger honest, and #408 pins it with
   tests (`test_unit_bytes_matches_the_cost_model_that_sized_the_pool`,
   `test_rebuild_resizes_codes_and_scales_together`, `test_kv_codec_has_no_arch_or_dtype_branch`).
4. Every new code width needs its row size registered in `kernel/aot_models.py`
   (`aggregate_store_element_sizes()`), or the `FREETOKEN_DISABLE_JIT=1` release gate fails —
   which is the cheapest possible enforcement that a new KV format cannot silently fall back.

Two further caveats recorded there: #408's own docs state *"capacity savings do not guarantee
faster decode"*, and its 76 B-vs-256 B figure is **per K row and per V row separately**, not a
token figure; and #113 (closed, DSV4-only) collides on the same `--kv-cache-dtype` option
string while doing the exact thing rule 1 forbids — read it for `_validate_kv_cache_dtype`
and the stamp-before-pool-exists ordering, don't merge it.

**Determinism oracles to port** (A2 §7): `vbr_transcode_anchor_test` LCG pattern
(`r = i*1103515245+12345; (r&0xFFFF)/32768-1`) with byte-identical A→A and in-place-vs-copies
A→B; the coverage mandate line (`docs/autopilot.md:657`: widths 128/256/384, I32/I64 indices,
1 and 7 rows, K/V, coop-vs-generic **bit-exact**, mean-sub on/off, GQA 1…16, batches 1/4/32,
`NMSE <= 5e-4`); `k_set_rows_turbo3_coop` Blackwell path (SM120 is our card; that kernel was
worth +10.5 % TG bit-exact).

## 5. Authoritative VRAM accounting (Phase 6)

**There is no ledger today** (A1 §6). Consumers re-derive from a handful of `mem_get_info`
snapshots taken in a fixed order — `engine/engine.py:778 _sync_get_memory` (sync +
`empty_cache` + `reset_peak_memory_stats`) → `_baseline_free` (:318) → post-weights
`_weights_bytes` (:345) / `_post_weights_free` (:352) — and then `net_cache_budget_bytes`
(`engine/cache_budget.py:29`) is shared by *two independent* decisions: `_startup_kv_budget`
(`engine.py:70`, KV pages) and `resolve_moe_cache_auto` (`cache_budget.py:88`, expert slots),
**with MoE sized first and KV consuming the residue** (`engine.py:557→584` before the page
solve at `:389–393`, GDN `state_pool_bytes` subtracted at `:392`). `(1 - memory_ratio)` is
the *implicit* headroom for everything else: **CUDA graphs are never budgeted**
(`engine/graph.py:101,156,171` only query and log), backend workspaces are allocated blind
(`attention/fi.py:114`, `trtllm.py:45`, `triton.py:43`), and the host side has a separate pin
quota (`engine.py:1281 _pin_budget_bytes`, `FREETOKEN_PIN_BUDGET_GB`, `_check_pin_budget`
:1309) that VRAM decisions never consult.

Three measured consequences on this host, all from EXP-001/001b:
1. Auto-sizing chose **6102 expert slots + 8268 KV tokens** on a 16 GiB card → the 16K anchor
   config was unreachable without `--num-tokens`; 1.08–1.19 GiB was still reported free.
2. Flash-Next at `--memory-ratio 0.9` **CUDA-OOM'd inside Triton autotuning**
   (`get_empty_cache_for_benchmark` wanting 256 MiB, 209 MiB free) at 14.64 GiB PyTorch-
   allocated — an unbudgeted, transient, JIT-only consumer. It succeeded at 0.86.
3. Per-sequence GDN state (966 MiB / 1737 MiB, PERFORMANCE.md §4) is pooled at
   `linear_state_cache_ratio=2.0` and is priced against nothing.

**Design**: one `VramLedger` object owned by the engine, created from the `_baseline_free`
snapshot, with named consumers that `reserve(name, bytes, kind=permanent|transient|resizable)`
before allocating and `commit`/`release` after, plus `reserve_headroom()` for
(graph capture, per-backend workspaces, JIT/autotune scratch, dequant scratch, staging).
`net_cache_budget_bytes` becomes a *query* against remaining headroom instead of a formula;
`validate_rebuild` (`kvcache/base.py:83`) and `rebuild_runtime_cache` (`engine.py:841`) become
the resize API (`ledger.reprice(...)`), keeping the identity-preserving rebuild contract
(`mha_pool.py:59`). The dequant/restore scratch projection from §4(c) is a *first-class
line item*, following LTO's `(row_k+row_v) × cells` form. Then the governor chooses among
{KV tier/format, KV→pinned-RAM tier, expert-cache size, GDN state slots, MTP budget, PLE
placement, workspace size} by measured marginal effective-TG per MiB, with the runtime
flight-recorder (§7) as its input. `--memory-ratio` stays as a cap, not as the policy.

## 6. MTP / speculative decoding map (Phases 9–10)

**Nothing exists on the base** (A3 §A2 "NOT PRESENT": no draft/verify/accept/rollback path;
`--kv-cache-dtype`-era flags absent; the only speculative machinery upstream is DSV4 DSpark in
`refs/pr/69`, `merge-tree` clean). The loaders **throw the head away**:
`models/qwen4_exp/weight.py:9,89`, `models/qwen3_5_moe/weight.py:58` (and upstream issue #421
says the same). Tests currently *assert* that drop (`tests/models/qwen4_exp/test_weight.py:234`,
`test_weight_ckpt.py:213`) — they must be inverted, not duplicated.

**The data is there.** Flash-Next HF checkpoint (`/models/Qwen3.8-Flash-Next-NVFP4-Radix`)
carries `text_config.mtp = {hybrid: true, layer_types: ["full_attention"],
mtp_use_hidden_state_from_layer: null, num_hidden_layers: 1, rope_theta: 1e7}` and **31 BF16
MTP tensors**: `mtp.{pre_fc_norm_embedding,pre_fc_norm_hidden,fc_embedding,fc_hidden}`,
`mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}`,
`mtp.layers.0.{attn,mlp}_hyper_connection.*`, `mtp.layers.0.self_attn.{q,k,v,o}_proj` +
`q_norm/k_norm` + `self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}`,
`mtp.layers.0.mlp.{gate,shared_expert.*,shared_expert_gate,experts.gate_up_proj,experts.down_proj}`.
**No `mtp.embed_tokens`, no `mtp.lm_head` → the head and embedding are always shared.**
GGUF sidecars exist for the same block (`/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/MTP/`,
3 files at `blk.48` of a 49-block geometry, `qwen4exp.nextn_shared_target_tensors` marking
tensor sharing with the target, `nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}`
at Q4_K/Q5_0/Q8_0) and the 35B-A3B NVFP4 checkpoint has 19 `mtp.*` tensors with
`mtp_num_hidden_layers=1`.

Integration map (each step is bounded by an existing seam):
1. **Load**: stop dropping `mtp.*`; route `mtp.layers.0.mlp.experts.*` into the expert banks;
   fuse `fc_embedding`+`fc_hidden` into one `eh_proj [2·n_embd, n_embd]`; map
   `pre_fc_norm_*`→`enorm`/`hnorm`, `hyper_connection_mixer.*`→`nextn.hc_head_*`.
2. **Config**: surface `text_config.mtp` in `Qwen4ExpArgs`; assert `num_hidden_layers <= 1`.
3. **Model**: `Qwen4ExpMTPDecoderLayer` composed from the existing `GatedResidual`
   (`models/qwen4_exp/hc.py`), `Qwen4ExpAttention` **minus the indexer** (draft is dense) and
   `Qwen4ExpMoE`, reusing `model.embed_tokens`/`lm_head`.
4. **Handoff**: expose the pre-collapse wide residual `R [T, hc_count·hidden] = [T, 10240]`
   from `Qwen4ExpModel.forward` (`model.py:150–168`) as a graph output.
5. **Engine**: `engine/engine.py:987–1005` emits 1 logit row and `req.complete_one()`; needs
   N+1 rows, a prefix-match accept, and accepted-length-driven `complete_n`; graphs
   (`engine/graph.py:79–99,140–204`) must gain a verify-length axis (or pad to `n_max+1` and
   mask) and be keyed on the nextn flag.
6. **KV**: rollback for the trunk only (paged pool + `linear_state_pool`), and a **separate
   single-layer dense KV pool for the draft** with no GDN/PLE state;
   `kvcache/qsa_pool.py:49–51` already reserves ring headroom for
   `num_speculative_tokens` but nobody passes it one.
7. **GDN verify is already implemented**: `kernel/fla/fused_sigmoid_gating_recurrent.py`
   accepts `disable_state_update`, `intermediate_states_buffer`, `intermediate_state_indices`
   and `retrieve_parent_token` (`:263–280`) — the multi-token path only has to pass them.
   This is the single largest reusable asset in the whole map.
8. **Carry/rollback state machine**: port the semantics (not the code) of LTO's
   `pending_h`/`verify_h` lifecycle, including "stale carry ⇒ drop the draft sequence, keep
   the newest target row" and "never `accept()` twice".
9. **PLE**: the draft carries no PLE tensors, but `model.py:152` asserts exactly one PLE layer
   and the disk row table sizes graphs by `max_graph_rows` — both must widen to verify shapes.
10. **Flags/UX**: `--spec-type/--num-speculative-tokens/--spec-draft-{p,n}-min` (nothing
    exists); oracle = llama.cpp/Unsloth qwen4exp MTP + `refs/pr/69`.

Metrics discipline (§MTP PERFORMANCE GOAL): report draft latency, verify latency, accepted
tokens/step, rejection/rollback cost, and **effective TG** separately. LTO's numbers say
verify dominates: turbo4 target-only 34.29 TG vs n_max=1/2/3 = 39.36/41.25/41.75 TG, i.e.
MTP is +22 % effective but only if verify is not a materializing path — which is why Phase 10
(MTP + TurboKV) must keep the fused decode kernel (§4c) tier-aware.

## 7. Observability (flight recorder)

Existing surfaces: `/v1/stats` (`throughput.{prefill_tps,decode_tps}`, `vram_bytes`,
`kv/mamba/swa` pools), `/v1/cache/status` (`kvcache/cache_status.py:14 compute_cache_unit_bytes`),
`POST /v1/cache/rebuild`, `--moe-collect-stats` + `decode_miss_stats()`,
`--prefill-hit-d2d` counters, and the scheduler decode log line
(`#token`, `token usage`, `#mamba-slot`, `gen throughput`). Gaps for the mission: H2D/D2D byte
totals, KV page migration/eviction, graph capture vs replay time, per-stage exposed idle. The
recorder will be a ring of these counters read at `/v1/stats`, priced so that reading it costs
less than 1 % TG; §5's ledger is its first instrument.

## 8. FreeToken-Kai: what the fork already solved

`yuuki-net/FreeToken-Kai` branch `kai` (`9dc5412`) has merged **our exact base**, so its delta
is `git diff cac247a..HEAD` = 191 commits / 203 files / **+31 851 −583**. Full audit in
`audits/A5-freetoken-kai.md`. Its author commits carry Claude co-author trailers and every
measurement in its docs was taken on **WSL2** (2060 / 3060 / 4060 Ti), never native Linux —
so its numbers are transferable as *directions*, not as this host's guards.

| Kai subsystem | State there | Value / risk for us |
|---|---|---|
| **GGUF** | **abandoned** — `docs/gguf.md` (`9784d13`) is a negative result: UD checkpoints vary the expert ggml type per layer and the offload slot pool is a single allocation with a single stride | the blocker is *structural* and identical to the constraint #131 documents; the useful output is its size table (NVFP4 experts 68.1 GB @4.51 bpw vs UD-Q2_K_XL 46.1 GB @3.05; **most of the saving is the PLE table 51.2 → 28.8 GB**) and its rate finding: hybrid decode needs ~44 GB/s from the CPU MoE path, so a scalar dequant kernel will not be the same order |
| **KV quant** `--kv-cache-dtype {auto,q8_0,q4_0}` | complete + measured (`kvcache/kv_quant.py`, `kernel/triton/kv_quant.py:27 _quantize_store_kernel`, gate `kvcache/__init__.py:137-170`, reads `attention/triton.py:157-220`, `qsa/attend.py:356-500`) | port the spec object, page geometry, refusal gate and the `--moe-cache-auto` coupling (freed KV bytes → expert slots, 358→902 on a 2060); **its measured −33 % dense TG at 30K (flat −1.4 % on Flash-Next) is the cautionary datum for §4(c)** — do not port its Triton forks |
| **Long context** | bought with VRAM, not tiered: host embedding, kv-quant, `--kv-reserve-tokens`/`--max-seq-len-override`. **No live-KV RAM/disk tier exists** | our Phase 5 stays greenfield; Kai confirms nobody has solved it upstream |
| **Host MoE banks** `--moe-bank-ram` | ~4 000 lines, well-tested: `moe/bank_file.py` (FTMB v2, A/B manifest, per-block SHA-256, journal), `mapped_bank.py`, `bank_tier.py`, `bank_rewarm.py`, `swap_back.py` | highest-value Kai subsystem for us, but it makes the bank **the only copy** of the experts and collides head-on with Phase 7 (exact-geometry expert cache) and with `#418/#427`'s expert packing ownership |
| **VRAM / pin accounting** | **two ledgers + three ad-hoc consumers**, not one: pin side is a single accumulator `_host_tables_bytes` (`engine.py:675-680`, adds `_mtp_bank_bytes`, `--host-embedding`, `_encoder_pinned_bytes` via `_encoder_bank_bytes` `:460`) on top of base's `_pin_budget_bytes` `:1281`; VRAM side is `plan_cache_budget` (boot) + a **per-prefill transient sizer** (`_measure_prefill_transient` `:2436` — one 1024-token dummy forward, `max_memory_allocated − held`; `_fit_prefill_chunk` `:2265`; `prefill_chunk_now` `:2479` re-solved before every prefill) + `engine/spec_graph.py` `mem_get_info` guards + a duplicated `PIN_BUDGET_FRACTION` in `moe/disk_probe.py` | the pin accumulator is *the* seam Phase 6 must extend, not fork (≈30 lines); **`cache_budget.py:44-52 pool_pages` fixes a real base bug — every KV pool allocates one page past the usable ones and the generic arithmetic used to price only the usable pages** (`71abe2c`), plus `CacheBudgetTooSmall`/`shortfall_fixes`/`explain_shortfall` diagnostics that our governor will need to explain itself; land those *before* the governor rewrites the file. The transient sizer is the actuator our EXP-001b OOM proves is missing — fold it into the ledger rather than adding a fourth `mem_get_info` caller |
| **`--dense-quant fp8`** | `layers/quantization/at_load.py` (`AT_LOAD_FP8` per-row e4m3, `KEEP_BF16` = `*.gate`, `*.shared_expert_gate`, `*hyper_connection*`, `*.indexer`, `*.ple`, `*.in_proj_b*`, `visual`) + `kernel/triton/fp8_pertensor_linear.py` | halves resident dense VRAM (Flash-Next 4.9 → 2.9 GB/card, "what leaves room for 128k"); the keep-bf16 list is exactly the set our own precision work must not touch |
| **MTP `--spec-mtp K`** | present for `qwen3_5_moe` **and** `qwen4_exp` (§6 of the audit; `engine/spec.py`, `engine/spec_graph.py`, `models/mtp_quant.py`) | oracle for Phase 9's acceptance/rollback arithmetic; see D-011 for why we implement rather than copy |
| **Prefill knobs** | `--prefill-chunk-budget` (0.75 measured), `--prefill-profile`, `--mm-encoder-dtype`, `--pp-prefill-group`/`--pp-send-ahead`; the `daf0a73` gloo backlog fix | the backlog fix is a base-level multi-rank leak worth taking standalone |
| **`ft doctor disk`** | `moe/disk_probe.py` (939 lines, stdlib-only: PCI topology chain, chipset-vs-GPU shared uplink, `readahead`, `memlock_limit`, `mapped_by`, random-row-read) + `gpu_probe.h2d_rate` + `disk_doctor.py` | our host is native Linux with tmpfs `/tmp` (see STATE.md gotchas) — exactly the diagnostic that finds that class of problem; blocked on Phase 6 giving it one source of truth |
| Negative results | more expert slots 358→902 = **+2 %**; fewer CPU-locked layers = +2-3 %; thread count inside noise; run-to-run spread up to **20 %**; `FREETOKEN_PREMAP_VRAM=1` did not help; **`read_ahead_kb` alone = 2.5×** | the 20 % spread number is the floor for what we may call a gain on small cards; the readahead result is a free win to re-test on this box |

