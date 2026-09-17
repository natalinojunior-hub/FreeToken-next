I have all evidence. Write the report now.

Keep ~180 lines. Dense.

# GGUF native-loader feasibility & design audit — FreeToken 0.1.3

**Headline:** a native GGUF loader **already exists and is committed** (`git log -- python/freetoken/models/gguf/` → `3af9d90 feat: initial open-source release`). It is *not* whole-file-dequantizing: weights stay packed and are dequantized inside vendored ggml CUDA kernels. It is narrow in exactly three axes: **1 architecture (gemma4)**, **3 GGML types wired in Python (Q4_0/Q6_K/F32-F16-BF16)**, and **single-file only (no shard joining)**. Most of your work is widening existing seams, not building new ones.

---

## 1. Engine's tensor contract

There is **no custom quant container and no opaque weight object**. A weight is a plain `torch.Tensor` held as a normal instance attribute of a `BaseOP`.

- `python/freetoken/layers/base.py:15` `class BaseOP` — the contract root.
- `layers/base.py:19-30` `state_dict()` walks `self.__dict__`, exporting every non-`_`-prefixed `torch.Tensor` (recursing into child `BaseOP`s). Attribute name **is** the checkpoint key.
- `layers/base.py:32-53` `load_state_dict()` is the gate a quantized weight must pass: `layers/base.py:45` `assert param.shape == item.shape and param.dtype == item.dtype`. **Shape and dtype of the placeholder built at construction time are the entire contract.** `layers/base.py:53` raises on leftover keys.
- Device is *not* asserted here; it is imposed upstream by `engine/engine.py:275-292` (`_materialize_loaded_weight_state_dict`), which does `weight.to(device=device, dtype=expected.dtype)` per key. Called from `engine/engine.py:513-519`, consumed at `engine/engine.py:333` `self.model.load_state_dict(...)`.

Per layer kind:

| need | declaring code | what a quant weight must be |
|---|---|---|
| (a) dense matmul | `layers/linear.py:14-46` `_LinearTPImpl` → `linear.py:41` `quant_method.create_weights(self)` | attributes set by a `LinearMethod`, e.g. `layers/quantization/linear/unquantized.py:31` `layer.weight = torch.empty(out, in)`; native-GGUF alternative `layers/gguf.py:68-89` `GGUFLinear` with `self.qweight` `[out, row_bytes] uint8` + `self._quant_type` |
| (b) MoE expert lookup | `layers/moe.py:84-87` (`quant_method = None` when no config; else `create_weights`), dispatch at `layers/moe.py:403-435`. Resident path needs `ExpertView` (`layers/quantization/moe/base.py:90-97`); **offload path needs only host bank tensors whose names come from `moe/offload_cache.py:36-77` `_BANK_SCHEMAS[fmt]`**, unpacked positionally at `layers/moe.py:431` (`gate_up, down = views`). GGUF's q4_0 branch is `layers/moe.py:425-434`. |
| (c) embedding table | `layers/embedding.py:15-59` `VocabParallelEmbedding` (`:31` `self.weight = torch.empty(n_tp, dim)`) via `kernel/index.py:43-56` `indexing()` (element_size-keyed JIT CUDA gather, `kernel/index.py:12-30`). Native-GGUF alternative `layers/gguf.py:91-128` `GGUFEmbedding`, which deliberately **bypasses** `indexing()` and gathers packed rows with `index_select` (`layers/gguf.py:120`) — documented at `kernel/aot_models.py:19`. |

**FTW (the existing quantized checkpoint) satisfies it as follows.** On-disk layout `checkpoint/ftw.py:26-31`: `<dir>/freetoken_weight.json` (index) + `freetoken-NNNNN.ftw` shards; constants `ftw.py:50-55` (`INDEX_NAME`, `FORMAT_TAG="freetoken_weight"`, `ALIGN=4096`, `_SHARD_FMT`). Index entry schema `ftw.py:184` `{"name","kind","dtype",...}`; two kinds (`ftw.py:18-24`): `kind="weight"` = *exactly what `iter_weights` yields, post-fusion/post-TP, fed straight to `load_state_dict`*, and `kind="experts_bank"` = post-repack banks, per-layer entries named `base#LNNNNN` (`ftw.py:60-67`). Reader: `ftw.py:105 is_ftw_checkpoint`, `ftw.py:365 iter_ftw_weights`, `ftw.py:211 FTWReader`. Replay point: `models/weight.py:226-228`.

**Quant configs.** `hf_quant_config.json` is *not* a first-class format: `utils/hf.py:194-204 sidecar_quantization_config` reads it only as a legacy ModelOpt sidecar and injects `{"quant_method":"modelopt", **quant}` into `config.quantization_config` (`utils/hf.py:206-220`), which `QuantConfig.from_hf` (`layers/quantization/configs/base.py:93-108`) then routes to a registered dialect. Kinds are `layers/quantization/scheme.py:14-22` (`NONE/FP8_TENSOR/FP8_BLOCK/MXFP8/NVFP4/MXFP4`). **GGUF explicitly opts out**: `models/register.py:321-326` returns `None` for `parse_config == "parse_gguf_config"`, i.e. the GGUF path never enters the QuantKind registry (`models/config.py:271 quant`), and layers are built dense then swapped (`models/gemma4/model.py:96-99`).

---

## 2. Kernel-side truth

For a GGUF quant weight the dequant happens **inside the matmul**; nothing is pre-dequantized. Available GPU kernels:

| op (Python wrapper) | kernel | accepted `quant_type` (exact `case` labels) | expected layout |
|---|---|---|---|
| `kernel/gguf.py:86-91 ggml_mul_mat_vec_a8` (MMVQ, GEMV) | `csrc/gguf/gguf_kernel.cu:94`, switch `:110-189` | `2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29` | `W` = `[out, in/block*type_size]` **uint8 rows of whole ggml blocks, ggml storage order**, contiguous. Activations quantized to `block_q8_1` in-kernel (`gguf_kernel.cu:23-59`, `kx` padded to 512). |
| `kernel/gguf.py:94-99 ggml_mul_mat_a8` (MMQ, prefill) | `gguf_kernel.cu:192`, switch `:209-318` | `2,3,6,7,8,10,11,12,13,14` **only** | identical `W` layout; tile-interleaved reads inside kernel |
| `kernel/gguf.py:79-83 ggml_dequantize` | `gguf_kernel.cu:74-91` → `dequantize.cuh:540-583 ggml_get_to_cuda` | `2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29` | same packed rows → dense `[m,n]` |
| `kernel/gguf.py:102-119 ggml_moe_a8` | `gguf_kernel.cu:335`, switch `:355-518` | `2,3,6,7,8,10,11,12,13,14` | `W` = `[E, row, row_bytes]` stacked experts + `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` |
| `kernel/gguf.py:122-131 ggml_moe_a8_vec` | `gguf_kernel.cu:541`, switch `:559-811` | `2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29` | stacked packed banks; takes `topk_ids` directly. Used by `moe/fused_q4_0.py:22-57`. |
| CPU W4A16 MoE GEMV | `kernel/csrc/cpu_moe/cpu_moe_ext.cpp:1111-1130` (`q4_0_dot`) | Q4_0 | **the same packed banks**, no scale side-array (fp16 scale inline per 32) |

Other low-bit GPU paths in the engine (all **incompatible with ggml block layout**, for contrast): `layers/quantization/linear/nvfp4.py:116-124` (`weight [O,K/2] uint8` + `weight_scale [O,K/16] e4m3` + `weight_global [O] fp16`), `linear/fp8_block.py:53-60`, `linear/mxfp8.py:43-48`, `linear/fp8_tensor.py:80-84`, and MoE `layers/quantization/moe/nvfp4.py:37/224/488` (triton / marlin / b12x repacks), `moe/mxfp4.py:14/56`, `moe/fp8_block.py:20`. **NOT FOUND:** any generic `w4a16`/`awq`/`exl2` kernel, any `torch._scaled_mm` call site, any `fp4`-native dense linear outside NVFP4.

**Scale layout is inline, never side-array.** All ggml quant types pack the scale(s) *inside* the block (e.g. `csrc/gguf/ggml-common.h:24 block_q4_0` `{ggml_half d; uint8_t qs[16]}`, `:91 block_q4_K`, `:109 block_q6_K`). This is exactly why the engine's NVFP4/FP8 bank schemas (`moe/offload_cache.py:52-76`, 6-4 named banks) cannot accept a ggml tensor without repacking, and why ggml types instead need only **one** bank per projection.

**The narrowing is Python-side, not kernel-side:** `layers/gguf.py:35-37` sets `_MMVQ = _MMQ = _DEQUANT = {Q4_0, Q8_0, Q6_K}` and `models/gguf/dequant.py:30-37 BLOCK_SHAPE` lists only `F32,F16,BF16,Q4_0,Q8_0,Q6_K`. Enabling Q4_K/Q5_K/IQ* is mostly editing those two tables.

---

## 3. GGUF type → engine mapping

Block sizes from `llama-turbo-optimal/gguf-py/gguf/constants.py:5748-5784 (GGML_QUANT_SIZES)`; type ids from `ggml/include/ggml.h:390-446`; structs from `freetoken/kernel/csrc/gguf/ggml-common.h`. `K=QK_K` where applicable.

| GGUF type (id) | blk | B/blk | bpw | scale layout | verdict | closest existing kernel |
|---|---|---|---|---|---|---|
| F32 (0) | 1 | 4 | 32 | none | **DIRECT** | `layers/gguf.py:53-54` (`x @ qweight.T`) |
| F16 (1) / BF16 (30) | 1 | 2 | 16 | none | **DIRECT** | same |
| Q4_0 (2) | 32 | 18 | 4.50 | fp16 `d` per 32, inline | **DIRECT** (live today) | MMVQ `gguf_kernel.cu:110`, MMQ `:210` |
| Q4_1 (3) | 32 | 20 | 5.00 | `d`+`min` fp16 inline | **DIRECT** (kernel has it; `layers/gguf.py:35` set + `BLOCK_SHAPE` missing) | MMVQ/MMQ `:115/:222` |
| Q5_0 (6) / Q5_1 (7) | 32 | 22/24 | 5.5/6.0 | fp16 `d`(+`min`) + 4B `qh` bitmap | **DIRECT** | `:119/:123`, `:234/:246` |
| Q8_0 (8) | 32 | 34 | 8.50 | fp16 `d` per 32, int8 qs | **DIRECT** — *but* `models/gguf/dequant.py:118 _DEQUANT` lacks q8_0, so `dequantize()` (reference path) raises. Fix there. | `:127/:258` |
| Q2_K (10) | 256 | 84 | 2.625 | `d`,`dmin` fp16 + 16B `scales` (8×5-bit) | **DIRECT** (Q4_0-style; add to sets) | `:131/:270` |
| Q3_K (11) | 256 | 110 | 3.44 | fp16 `d` + 12B packed sub-scales + 32B `hmask` | **DIRECT** | `:135/:282` |
| **Q4_K** (12) | 256 | 144 | 4.50 | fp16 `d`,`dmin` + 12B 6-bit sub-scales for 8×32 groups | **DIRECT** (kernel ready; needs `BLOCK_SHAPE` + `_MMVQ/_MMQ`/`GGML_Q4_K` const + dequant ref) | MMQ `gguf_kernel.cu:294` → `mmq.cuh:703 ggml_mul_mat_q4_K_q8_1_cuda`; MMVQ `:139` |
| **Q5_K** (13) | 256 | 176 | 5.50 | as Q4_K + 32B `qh` | **DIRECT** (same wiring gap) | `:306` → `mmq.cuh:778` |
| Q6_K (14) | 256 | 210 | 6.56 | fp16 `d` + 16 int8 sub-scales per 16 | **DIRECT** (live today) | `dequant.py:95 dequant_q6_k`, MMQ `:318` |
| Q8_K (15) | 256 | 292 | 9.13 | fp32 `d` + 32 int8 | n/a — activation-only type | **NOT FOUND** anywhere |
| IQ4_NL (20) | 32 | 18 | 4.50 | fp16 `d`, symmetric 16-level LUT | **DIRECT decode / REPACK-ONCE prefill** (no MMQ case) | MMVQ `:167`, `dequantize.cuh:569` |
| IQ4_XS (23) | 256 | 136 | 4.25 | fp16 `d` + 4B `scales` (8×4-bit) | **DIRECT decode / dequant-fallback prefill** | MMVQ `:179`, `moe_vec` `:786` |
| IQ2_XXS (16) 2.06 / IQ2_XS (17) 2.31 / IQ2_S (22) 2.56 / IQ3_XXS (18) 3.06 / IQ3_S (21) 3.44 / IQ1_S (19) 1.56 / IQ1_M (29) 1.75 | 256 | 66/74/82/98/110/50/56 | — | fp16 `d` + **embedded codebook index bytes** | **DIRECT decode only**; prefill falls to `layers/gguf.py:60-64` dequant→bf16 matmul (materializes the weight per call — unacceptable). Mark **NEW-KERNEL** for prefill. | MMVQ `:151-189`; `dequantize.cuh:545-581` |
| TQ1_0 (34) 1.69 / TQ2_0 (36) 2.06 | 256 | 54/66 | — | ternary/quaternary, no per-block scale | **NEW-KERNEL** — no `case 34/35` in `gguf_kernel.cu`, absent from vendored `ggml-common.h` | — |
| MXFP4 (39) 4.25 / NVFP4 (40) 4.50 | 32/64 | 17/36 | — | e8m0 / e4m3 block scale | **NEW-KERNEL** (ggml's layout ≠ engine's NVFP4 6-bank layout, `offload_cache.py:52-63`) | `linear/nvfp4.py:116` |
| Q1_0 (41) 1.13 / Q2_0 (42) 2.25 | 128/64 | 18/18 | — | fp16 `d` inline | **NEW-KERNEL**; **absent from vendored `ggml-common.h`** | — |
| Q2_0_G128 (53, fork-only, `ggml.h:445`) | 128 | — | 2.0 | — | **CPU-ONLY / NOT LOADABLE**: id 53 has **no entry in `GGML_QUANT_SIZES`** (`constants.py:5748-5784`), so `models/gguf/reader.py:150 GGML_QUANT_SIZES[t.tensor_type]` raises `KeyError` before any kernel runs | — |
| Turbo3/4/2/8/1_TCQ (43-52) & VBR | — | — | — | KV-cache codecs (`ggml.h:435-444`; `ggml/include/ggml-vbr.h:3-4` "TurboQuant KV-cache support … turbo-typed KV tensors"; `ggml-turbo-meansub.h:3-6` bakes means into the *runtime*, not the GGUF) | **out of scope for a weight loader** | — |

`DIRECT` = zero repack: hand `GgufTensor.packed()` (`models/gguf/reader.py:110-112`, a `torch.from_numpy` uint8 view over `np.memmap`) straight to `GGUFLinear.qweight`. `REPACK-ONCE` cache would live next to the FTW, as an extra `kind="experts_bank"` / `kind="weight"` entry in `freetoken_weight.json` (`checkpoint/ftw.py:184`), exactly how NVFP4 marlin/b12x repacks are already persisted (`kernel/aot_models.py:97-99`, `moe/offload_cache.py:59-63`).

---

## 4. Sharding + tensor-name mapping

**Sharding is NOT supported — NOT FOUND.** `models/gguf/reader.py:23-27 is_gguf_path` accepts only `os.path.isfile(path) and path.endswith(".gguf")`, and its own docstring says "the only GGUF layout FreeToken loads directly". `reader.py:120-124 _reader()` = `functools.cache`d `gguf.GGUFReader(path)`, which does `np.memmap(path)` on that one file (`llama-turbo-optimal/gguf-py/gguf/gguf_reader.py:137-138`). gguf-py has **no** split joining (`grep n_parts|of-|all_tensors gguf_reader.py` → zero hits); llama.cpp does it in C++ only: `src/llama-model-loader.cpp:507-527 llama_get_list_splits`, `:1021-1045` (`LLM_KV_SPLIT_COUNT`/`LLM_KV_SPLIT_NO`, "model must be loaded with the first split"), keys at `gguf-py/gguf/constants.py:273-275`. A `-00001-of-00005` set therefore parses the full tensor table from part 1 but reads tensor data past the mmap end for parts 2..N. **This is the single biggest missing piece and it needs no kernel work** — resolve `split.count`, open all parts, and build views over a per-file mmap (data offsets in GGUF v3 shards are file-local from part 2 on; llama.cpp sums sizes to a contiguous logical range at `llama-model-loader.cpp:964`).

**Name mapping.** There is no central name map for checkpoint→engine names; each family's `models/<fam>/weight.py` hard-codes it. The reference dense one is `models/llama/weight.py:13-19` (`_MERGE_RULES`: `.q_proj/.k_proj/.v_proj → .qkv_proj`, `.gate_proj/.up_proj → .gate_up_proj`) applied by `models/loader.py:130-160 iter_merged_tensors`, with TP sharding in `models/loader.py:22-48 shard_tensor` and generic root/segment renames declared as data in `models/register.py:35-88` (`_DENSE_PACKED`, `_GEMMA4_SEGMENTS`) and consumed by `layers/quantization/names.py:63-70 NameMap` — but `NameMap` **only rewrites names for quant-config lookups**, it never rewrites weight keys.

The GGUF translator is therefore per-family and lives in the family's `gguf.py`: `models/gemma4/gguf.py:151-169 _LAYER_SCALAR_MAP` (gguf suffix → freetoken dotted attr) plus the hand-rolled if/elif chain at `models/gemma4/gguf.py:236-266` (`attn_q/k/v`, `attn_output`, `ffn_gate/up/down`, `ffn_gate_inp(.scale)`, `ffn_down_exps.scale`, `token_embd`, `output_norm`, `rope_freqs`), with fusion by `torch.cat(..., dim=0)` over packed rows at `models/gemma4/gguf.py:276-287` — legal because rows share the input dim, hence `row_bytes` (`models/gguf/reader.py:157-158`). **Plug-in point for llama/qwen/ds GGUF: a new `models/<fam>/gguf.py` exposing `parse_gguf_config` + `iter_gguf_weights`, registered as a `ModelSpec` override** — the mechanism already exists at `models/register.py:263-268` (`"Gemma4GGUFForCausalLM": ModelSpec(..., parse_config="parse_gguf_config", iter_weights="iter_gguf_weights")`), reached through `models/weight.py:209-244 load_weight` and guarded by `tests/models/test_models_registry.py:6-21`. Arch→registry-key table: `models/gguf/config.py:20-22 GGUF_ARCH_TO_REGISTRY` (currently one entry).

**Tied embeddings** are handled twice, and both work off the presence of `output.weight`: config derivation at `models/gguf/config.py:71-86` (`tie_word_embeddings = "output.weight" not in names`, with a KV fallback `freetoken.output_weight_present` for metadata-only GGUF, `reader.py:36-39` + `reader.py:60-99 write_metadata_gguf`); runtime at `models/gemma4/gguf.py:312-338 class GGUFTiedLMHead`, which holds only a *reference* to the embedding's `qweight` and returns an empty `state_dict` (`:325-329`) so `load_state_dict` pops `lm_head.weight`/`bias` instead of failing (`:331-334`) — installed at `models/gemma4/gguf.py:375-376`. HF-side analogue: `layers/embedding.py:62-88 ParallelLMHead(tied_embedding=...)`, used at `models/gemma4/model.py:85-87`. Note `iter_gguf_weights` in `gemma4/gguf.py` yields nothing for `output.weight` (it is not in the map) — for a **untied** llama GGUF you must add an `output.weight → lm_head.qweight` case.

**`mtp.*`: NOT FOUND — dropped by every family.** `models/qwen4_exp/weight.py:9,89`, `models/qwen3_5_moe/weight.py:58`, `models/glm4_moe/config.py:28`, `models/deepseek_v4/weight.py:178`. A GGUF loader should skip `mtp.*` the same way and say so loudly.

---

## 5. mmap / lazy load — already present, reuse it

- **GGUF is already mmap'd and lazy.** `gguf.GGUFReader.__init__` → `np.memmap(path)` (`gguf_reader.py:138`); `models/gguf/reader.py:160-162` re-views those bytes as `uint8 [rows, row_bytes]`; `reader.py:110-112 GgufTensor.packed()` returns `torch.from_numpy(...)` — zero-copy, no read until the kernel touches pages. Only the tiny F32 norm/router/scalar tensors are eagerly dequantized (`models/gemma4/gguf.py:172-175 _to_bf16`). No whole-file dequant exists anywhere on this path.
- **FTW reader:** `checkpoint/ftw.py:225` `_maps` cache, `ftw.py:270-292 FTWReader._map()` = `mmap.mmap(fd, 0, PROT_READ)` + `MADV_SEQUENTIAL`, used as the *fallback* when `O_DIRECT` is rejected (`ftw.py:229-231, 241-262`). Fast path is chunked multi-threaded `os.preadv` with `O_DIRECT` (`ftw.py:38-68 _read_shard_odirect_parallel` in `models/weight.py`, `ftw.py:70-93 _pread_into`). Page-cache eviction: `models/loader.py:71-79 drop_page_cache`.
- **Safetensors lazy/random access:** `models/loader.py:163-197 class ShardReader` — `safetensors.safe_open(file, framework="pt", device=...)` handles opened lazily, per-name lookup across shards via `safetensors_weight_map` (`loader.py:59-73`). Not `np.memmap`, and no `torch.UntypedStorage.from_file` and **no `fsspec` anywhere — NOT FOUND** (`grep fsspec` → 0 hits).
- **Host-resident expert tensors** (`python/freetoken/moe/`): `moe/host_banks.py:78-160 class HostBank` — backing is either `"mmap"` (a **lazy anonymous `mmap.mmap(-1, size)`**, `host_banks.py:112`, so pages commit only on fill) or `"cuda"` (`cudaHostAlloc` born-pinned, `host_banks.py:69-75`). Residency classes `host_banks.py:44-54`: `PINNED` / `LOCKED` (`mlock`) / `PAGEABLE`. Pinning is **pin-after-fill** via `cudaHostRegister` (`host_banks.py:131-143`), driven by a single background thread `host_banks.py:286-350 class PinPipeline` and `host_banks.py:351 class LayerCompletionTracker`; allocation entry `host_banks.py:207 alloc_layer_banks(specs, num_layers)`; the mmaps are leaked on purpose (`host_banks.py:57-58 _LIVE_BUFFERS`). GGUF banks are allocated exactly this way: `models/gemma4/gguf.py:416-417` and pinned via `gguf.py:437-443`. The returned `dict[str, list[Tensor]]` is handed to `moe/expert_banks.py:155-172 _q4_0_banks` → `ExpertBanks("q4_0", ...)`, the only format with a bespoke provider (`expert_banks.py:175-178 _PROVIDERS`).

---

## 6. Verdict

Files/functions a minimal GGUF loader must touch, in dependency order (★ = must edit, ☆ = usually untouched):

1. ☆ `models/gguf/reader.py:143 iter_gguf_tensors` / `:110 packed` — the mmap view; already generic.
2. ★ `models/gguf/reader.py:23 is_gguf_path` + new split-file resolution — **only if** sharded GGUF is in scope.
3. ★ `models/gguf/dequant.py:30-47 BLOCK_SHAPE / GGML_NAME` and `:118 _DEQUANT` — add the types you serve (Q8_0 reference dequant is currently missing entirely).
4. ★ `layers/gguf.py:33-37 _UNQUANTIZED/_MMVQ/_MMQ/_DEQUANT` — widen the dispatch sets to match `gguf_kernel.cu`'s real coverage.
5. ★ `models/gguf/config.py:20-22 GGUF_ARCH_TO_REGISTRY` — add the architecture key.
6. ★ `models/<fam>/config.py` — new `parse_gguf_config(shim) -> ModelConfig` (copy `models/gemma4/gguf.py:47-142`).
7. ★ `models/<fam>/gguf.py` — new `iter_gguf_weights` (name map + packed fusion) + `convert_<fam>_to_gguf` (copy `models/gemma4/gguf.py:186-377`).
8. ★ `models/<fam>/model.py` — call the swap (copy `models/gemma4/model.py:96-99`).
9. ★ `models/register.py` (near `:263`) — register `"<Fam>GGUFForCausalLM"` with the two overrides; ★ `models/<fam>/__init__.py` re-exports (cf. `models/gemma4/__init__.py:3-8`); ★ `kernel/aot_models.py:58-66 arch_aliases` (enforced by `tests/models/test_models_registry.py`).
10. MoE only: ★ `moe/offload_cache.py:36-92` (`_BANK_SCHEMAS`/`_BANK_BYTES_PER_EXPERT`), ★ `moe/expert_banks.py:175-178 _PROVIDERS`, ★ `layers/moe.py:425-435`, ★ `moe/cpu_executor.py:72,407-430`. Remove the `FIXME` branches at `engine/engine.py:772` and `register.py:321-326` only if GGUF ever joins the QuantKind registry — **recommended against for a first increment.**

**Smallest first increment (dense, unsharded, TP=1, Llama-3-8B-class `Q4_K_M`):** no new file under `kernel/`, no CUDA change, no FTW change. `Q4_K_M` puts `Q4_K` on `blk.N.attn_q/k/v/o` + `ffn_up/gate` (weight_a), `Q6_K` on `ffn_down` (weight_b), `Q8_0` on `token_embd`/`output`, and F32 on all norms — every one of which already has a case in `ggml_mul_mat_vec_a8`/`ggml_mul_mat_a8`/`ggml_dequantize`. Steps: extend the two type tables (3, 4) → parse llama KV metadata into `ModelConfig` (6) → translate names, keeping packed rows and fusing qkv/gate_up by `cat(dim=0)` (7) → swap `Linear`/`VocabParallelEmbedding` for `GGUFLinear`/`GGUFEmbedding` and `lm_head` for `GGUFTiedLMHead` (7, 8) → register (9). Config/tokenizer/sampling already come for free (`models/gguf/tokenizer.py`, `utils/hf.py:33-38, 108-116, 239-247`), and so does FTW conversion of the result (`checkpoint/convert.py:73-86`).

```python
# models/llama/gguf.py  — the load-bearing part of the increment
def parse_gguf_config(shim):                       # mirror gemma4/gguf.py:47
    m = {k.split(".", 1)[1]: v for k, v in shim.metadata.items() if k.startswith("llama.")}
    return ModelConfig(num_layers=int(m["block_count"]), hidden_size=int(m["embedding_length"]),
                       num_qo_heads=int(m["attention.head_count"]),
                       num_kv_heads=int(m["attention.head_count_kv"]),
                       head_dim=int(m.get("attention.key_length", hidden // nq)),
                       intermediate_size=int(m["feed_forward_length"]), vocab_size=shim.vocab_size,
                       tie_word_embeddings=shim.tie_word_embeddings, rms_norm_eps=...,
                       rotary_config=RotaryConfig(...), model_type="llama",
                       architectures=list(shim.architectures))          # expert_quant stays "none"

_MAP = {"attn_q": ("self_attn.qkv_proj", "q"), "attn_k": ("self_attn.qkv_proj", "k"),
        "attn_v": ("self_attn.qkv_proj", "v"), "attn_output": ("self_attn.o_proj", None),
        "ffn_gate": ("feed_forward.gate_up_proj", "gate"), "ffn_up": ("feed_forward.gate_up_proj", "up"),
        "ffn_down": ("feed_forward.down_proj", None)}

def iter_gguf_weights(model_path, device, *, include_moe_experts, include_non_moe):
    buf = {}
    for t in iter_gguf_tensors(model_path):                 # zero-copy over np.memmap
        if t.name == "token_embd.weight":   yield "model.embed_tokens.qweight", t.packed()
        elif t.name == "output.weight":     yield "lm_head.qweight", t.packed()
        elif t.name.endswith("_norm.weight"): yield _scalar(t)          # F32 -> bf16
        elif t.name.startswith("blk."):
            layer, suffix = int(t.name.split(".")[1]), t.name.split(".", 2)[2]
            if suffix.startswith("mtp.") or layer >= cfg.num_layers: continue
            mod, slot = _MAP[suffix.removesuffix(".weight")]
            key = f"model.layers.{layer}.{mod}.qweight"
            (buf.setdefault(key, {})[slot] if slot else buf.setdefault(key, [None]))  # ...
            yield from _flush_when_complete(buf)            # torch.cat(parts, dim=0)
```