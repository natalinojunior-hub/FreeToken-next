I have everything. Let me write the final report now, dense, ~200 lines.

Let me compute bytes/token for head_dim=128 and 256, n_kv heads:
- turbo3: 14 B/32 elem = 0.4375 B/elem → head 128 = 56 B; head 256 = 112 B
- turbo2: 10/32 = 0.3125 B/elem → 40 B / 80 B
- turbo4: 66/128 = 0.515625 → 66 B / 132 B
- turbo8: 130/128 = 1.015625 → 130 B / 260 B
- turbo3_tcq: 52/128 = 0.40625 → 52 / 104
- turbo2_tcq: 36/128 = 0.28125 → 36 / 72
- turbo1_tcq: 20/128 = 0.15625 → 20 / 40
- f16: 2 B/elem → 256 / 512

I'll present a table with bpv, B/elem, and per-head bytes.

Write it up.
## TurboQuant / TCQ / VBR — spec extracted from code (`/models/servers/llama-turbo-optimal`)

All paths absolute. There is **no `GGML_TYPE_META`** symbol; per-type metadata is `ggml_type_traits` (`/models/servers/llama-turbo-optimal/ggml/src/ggml.c:776-860`) + `ggml_type_traits_cpu` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cpu/ggml-cpu.c:421-446`). **`QK_K` (256) is not used by any turbo type** — each type defines its own `QK_TURBO*`.

### 1. Type identity

`/models/servers/llama-turbo-optimal/ggml/include/ggml.h:433-447` (ids were bumped +1 to free slot 42 for upstream `Q2_0`; these are **runtime-only KV codecs, never written to GGUF**):

| enum | id | `blck_size` | `type_size` | bpv | B/elem | B/head@128 | B/head@256 |
|---|---|---|---|---|---|---|---|
| `TURBO3_0` ("turbo3") | 43 | `QK_TURBO3`=32 | 14 | 3.5 | 0.4375 | 56 | 112 |
| `TURBO4_0` ("turbo4") | 44 | `QK_TURBO4`=128 | 66 | 4.125 | 0.515625 | 66 | 132 |
| `TURBO2_0` ("turbo2") | 45 | `QK_TURBO2`=32 | 10 | 2.5 | 0.3125 | 40 | 80 |
| `TURBO3_TCQ` | 46 | 128 | 52 | 3.25 | 0.40625 | 52 | 104 |
| `TURBO2_TCQ` | 47 | 128 | 36 | 2.25 | 0.28125 | 36 | 72 |
| `TURBO8_0` ("turbo8") | 48 | `QK_TURBO8`=128 | 130 | 8.125 | 1.015625 | 130 | 260 |
| `TURBO1`/`TURBO1_NSN`/`TURBO1_CQ` | 49/50/51 | 128 | 18/20/18 | — | — | — | **RESERVED, codec removed 2026-07-05** (`ggml.h:441-443`) |
| `TURBO1_TCQ` | 52 | 128 | 20 | 1.25 | 0.15625 | 20 | 40 |

`GGML_TYPE_COUNT=54`. Rotation group is always 128 (`QK_TURBO3_GROUP`/`QK_TURBO2_GROUP`=128, `ggml-common.h:312,323`; `TURBO_D 128`, `ggml-turbo-quant.c:26`) — i.e. turbo2/turbo3 pack **4 sub-blocks of 32** but write the **same group-level corrected norm into all four** (`turbo-quant-cuda.cuh:912-916`).

Verbatim layouts, `/models/servers/llama-turbo-optimal/ggml/src/ggml-common.h:311-417`:

```c
typedef struct { ggml_half norm;         //  2B: (corrected) group L2 norm
                 uint8_t  qs[32/4];      //  8B: lower 2 bits of the 3-bit index, 4/byte
                 uint8_t  signs[32/8]; } block_turbo3_0;   //  4B: upper 1 bit, 8/byte  // 14B
typedef struct { ggml_half norm; uint8_t qs[32/4]; } block_turbo2_0;   // 10B  (2-bit idx, 4/byte)
typedef struct { ggml_half norm; uint8_t qs[128/2]; } block_turbo4_0;  // 66B  (4-bit idx, low nibble first)
typedef struct { ggml_half norm; uint8_t qs[128];    } block_turbo8_0; // 130B (8-bit idx; norm = L2norm*absmax)
typedef struct { ggml_half norm; uint8_t qs[49]; uint8_t pad; } block_turbo3_tcq; // 52B: 390-bit trellis stream +2 pad
typedef struct { ggml_half norm; uint8_t qs[33]; uint8_t pad; } block_turbo2_tcq; // 36B: 262 bits
typedef struct { ggml_half norm; uint8_t qs[17]; uint8_t pad; } block_turbo1_tcq; // 20B: 135 bits
```
No `d`/scale/min/zero fields exist — the only scalar is `norm` (`GGML_COMMON_AGGR_U`-style unions are unused). `static_assert`s at `ggml-common.h:316,327,339,351,361,371,417` pin the byte counts.

**VBR is not a type.** It is a runtime tier controller: `enum vbr_tier {T8, T4, T3_TCQ, T2_TCQ, T1_TCQ}` (`/models/servers/llama-turbo-optimal/src/llama-kv-cache.cpp:80-87`), `vbr_tier_type()` → the five types above (`:228-237`), movability gate `vbr_type_is_movable()` = F16 + the 5 turbo types (`:222-226`). Membership predicate `ggml_is_turbo_kv_type()` = the 7 live turbo types (`/models/servers/llama-turbo-optimal/ggml/src/ggml.c:1455-1468`) — note `TURBO1_*` reserved slots deliberately excluded.

### 2. Bit-packing algorithm

Shared encode pipeline (CUDA, authoritative): mean-sub (optional tap) → per-channel InnerQ scale → **L2 norm → normalize → `x*=s1` → normalized FWHT(128) → `x*=s2`** → nearest-centroid → store **norm/recon_norm**.
- FWHT: `turbo-quant-cuda.cuh:764-776` (butterfly stages h=1..64, then `*= 0.08838834764831845f` = 1/√128); `turbo_rotate_forward_cuda` `:779-783`; sign arrays `d_turbo_wht_signs1/2[128]` `:714-717` (±1 only, "from turbo-wht.h, seed=42 rotation, seed=1042 QJL").
- Inverse = **same call with s1/s2 swapped** (`turbo-quant-cuda.cuh:1110-1112, 1211`) because the normalized FWHT is an involution.
- Turbo3 = 3-bit: `idx = (low2 from qs) | (hi1 from signs)<<2`, `:903-906`. Turbo4 = 4-bit: `qs[j/2] = (idx(j+1)<<4) | idx(j)` `:1021-1024`. Turbo2 = turbo3 minus `signs`. Turbo8 = **uniform 256-level grid + per-block absmax**, `idx = lrintf(x*inv_absmax*127.5f + 127.5f)` clamped [0,255], `norm = half(norm*absmax)` (`:1078-1092`), centroids `=(i-127.5)/127.5` (`:679+`).
- No residual/delta stage. No QJL anywhere (removed: `:718` "QJL sign arrays removed — turbo4 now uses pure 4-bit PolarQuant"); the `ggml.h:435-436` "2-bit PolarQuant + 1-bit QJL" comment is **stale** vs. the 3-bit-centroid code actually used.
- Clamp rules: `norm > 1e-10f` guard everywhere; nearest-centroid is strict `<` against midpoints, so a value exactly on a midpoint takes the **higher** index (`:785-830`; CPU mirror `ggml-turbo-quant.c:236-287`, 4-bit via an explicit binary-search tree, 8-bit via lower_bound).

**TCQ** (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/turbo-quant-cuda.cuh`, encoder `k_set_rows_turbo3_tcq:1498`, `k_set_rows_turbo2_tcq:1960`, turbo1 `:2384`):
- Right-shift **bitshift trellis**; t3: k=3, L=9, **512 states**, `ns=(prev>>3)|(out<<6)` (`:1665`); t2: k=2, L=8, **256 states**, `ns=(prev>>2)|(out<<6)` (`:2097-2098`); t1: k=1, L=8, **256 states**, `ns=(prev>>1)|(out<<7)` (`:2557-2560`).
- One Viterbi **forward pass over exactly 128 steps** (128 coords = one FWHT group), per-step cost `min over 2^k predecessors of cost_rd[base|p] + (x[t]-codebook[state])^2`, tie → **lowest** predecessor index (`:1655-1668`, probe `TURBO_TCQ_TIEHI`), then a **sequential single-thread backtrace** from the min-cost final state (`:1731-1741`).
- Codebook is indexed **by the 9/8-bit state**, not by the symbol: `recon_t = codebook[state_t]`. Codebooks are **separate for K and V** and split encode-side/decode-side: `d_turbo3_tcq_codebook[512]` `:1369`, `_v[512]` `:1301`, `d_turbo2_tcq_codebook[256]` `:1887`, `_v` `:1851`, `d_turbo1_tcq_codebook[256]` `:2285`, `_v` `:2319`; decode copies in `/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn-common.cuh:102,170,240,276` and `/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn.cu:881,915`. Env: `TURBO_TCQ_CB[_K|_V]`, `TURBO1_TCQ_CB_*`, hot-swap by mtime (`turbo-quant-cuda.cuh:1446-1474`).
- On-disk bit layout (LSB-first): `qs` = **6 bits of initial state** (`(state>>3)&0x3F` for t3/t2, `(state>>1)&0x7F` = 7 bits for t1) **then the 128 symbols packed back-to-back** (`:1778-1800`, `:2216-2238`, `:2544-2561`). Decode is a **9-bit window read at bit offset `3t`** (t2: 8 bits at `2t`; t1: 8 bits at `t`) — the 6/7-bit prefix makes the window land exactly on `out[t-2]|out[t-1]<<3|out[t]<<6`, so decode needs no state tracking (`:1816-1846`, `:2244-2270`, `:2625-2633`).
- **Decode-side codebook mismatch to reproduce:** `dequantize_turbo3_tcq`/`_2_tcq` (get-rows path) always use the **K** table, while the fused loader takes `cb` as a parameter and applies `alpha_v`.
- Stored norm is multiplied by a tier alpha at encode (`d_tcq_norm_alpha` K=1.0 / `_v` V=1.04, `:1439-1440`) and again at decode by `TURBO_TCQ_ALPHA_V_T3=1.02 / T2=1.06 / T1=1.26` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/turbo-tcq-alpha.cuh:11-13`, dispatch `fattn.cu:461-482`).

**VBR** — what varies is **per-(layer, K/V side) tier over contiguous cell ranges**, never per token; all tiers are fixed-size rows, so paging/seek is a plain row index with no bit offsets. Physical layout: one VMM virtual-address reservation per KV tensor with fixed page-aligned per-tensor offsets, physical pages mapped on demand and unmapped after a degrade, **never relocated** (`/models/servers/llama-turbo-optimal/ggml/include/ggml-vbr.h:27-30`). Watermark is measured in **cells padded to 256** (`/models/servers/llama-turbo-optimal/src/llama-kv-cache.cpp:4019`), span math `vbr_span_of()` compares row_bytes(A) vs row_bytes(B) × cells (`:877-918`), commit granularity forced ≥64 KiB on ROCm (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/vbr-vmm-policy.h:5-17`). Retiering = **in-place transcode of whole rows**, tiled 256 cells at a time, **reverse tile order** for in-place safety (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/vbr-transcode.cu:115,477,520-521`).

### 3. Reference encode/decode paths

`/models/servers/llama-turbo-optimal/ggml/src/ggml-turbo-quant.c`: `quantize_row_turbo2_0_ref:289` · `dequantize_row_turbo2_0:300` · `turbo3_0_ref:330` / `dequant:343` · `turbo4_0_ref:458` / `dequant:508` · `turbo8_0_ref:554` / `dequant:601` · `turbo3_tcq_ref:376` / `dequant:388` · `turbo2_tcq_ref:417` / `dequant:429` · `turbo1_tcq_ref:645` / `dequant:655`; bulk `quantize_turbo*:312,358,536,628,399,440,660`.

**Critical for a port — these are only partially real:**
- turbo2/3 refs compute the norm and then `memset(qs,0)` / `memset(signs,0)` — "Stub — Metal shader handles quantize on GPU" (`:331`), so **a CPU round-trip through turbo3 returns centroids[0]**. Same for both TCQ refs (`:377,418`) whose dequant returns all zeros (`:388-397,429-438`); turbo1_tcq stub dequant `memset`s to 0 and its stub norm multiplies by `0.08838834764831845f` (`:645-653`) — different from every other tier.
- **turbo4/turbo8 CPU refs use a different rotation than CUDA**: a dense 128×128 random-orthogonal matrix from `turbo_prng_seed(42)` + LCG(`*6364136223846793005+1442695040888963407`) + Box-Muller + modified Gram-Schmidt QR (`:130-190`, applied via `matvec:223`). CUDA uses FWHT+signs. **Do not cross-validate CPU vs CUDA.**
- CPU turbo8 uses 256 **Lloyd-Max** `CENTROIDS_8BIT` (`:50-90`) while CUDA uses the **uniform** grid + absmax (`turbo-quant-cuda.cuh:1078-1092`) — a real semantic divergence; the CUDA path is the shipped one.
- Real encoder/decoder = CUDA only. Encode: `k_set_rows_turbo3:832`, `k_set_rows_turbo2:1228`, `quantize_f32_turbo4_0_block:1006`, `quantize_f32_turbo8_0_block:1052`, TCQ `:1498/:1960/:2384`; decode: `dequantize_turbo{3_0:941,4_0:1039,8_0:1097,2_0:1287,3_tcq:1818,2_tcq:2246,1_tcq:2627}`.
- **No CPU SIMD path exists**: `vec_dot = NULL`, `vec_dot_type = F32` for turbo2/3/4/8 (`ggml-cpu.c:421-446`), no CPU entries at all for TCQ types, and CPU `FLASH_ATTN_EXT` has explicit `// no-op` cases for `TURBO2_0/3_0/4_0/8_0` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cpu/ops.cpp:5957-5965`).
- Bit-exactness caveat: the coop kernels use a **serial shared-memory reduction, not a tree, deliberately** "to preserve FMA rounding order" (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/set-rows.cu:1188-1190`); recon-norm sums accumulate in coordinate order (`turbo-quant-cuda.cuh:907-909`).

### 4. GPU kernels

Two modes coexist, selected in `/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn.cu:2325-2392`:

**(a) Fused dequant-in-attention (flash-attn style)** — `ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ,DV,ncols1,ncols2,type_K,type_V>` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn-mma-turbo.cuh:16-190`) launching the templated `flash_attn_ext_f16<...,type_K,type_V>` with `need_f16_K=false, need_f16_V=false` (`:186-189`, "raw turbo data passes through"). Tile loaders in `/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn-mma-f16.cuh`: `flash_attn_ext_turbo4_load_tile:558`, `_turbo8_:609`, `_q8_0_:~655`, generic `_turbo_load_tile<turbo_type,...>:707` (turbo3/turbo2/tcq3/tcq1/tcq2 branches `:725,:755,:775,:797,:821`) — dequant directly into `half2` shmem tiles, no f16 buffer.
Geometry: `nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ,DV,ncols,cc)`, `nwarps = nthreads/warp_size`, grid via `launch_fattn<DV,ncols1,ncols2>` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/fattn-common.cuh:1915`); **`nstages` forced to 0** because cp.async cannot do ALU dequant, so tile_K/tile_V share one shmem region (`fattn-mma-turbo.cuh:27-28,36`); smem = `max(combine, Q + KV + mask)` computed at `:38-46` and `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` once per device (`:59-99`, with the RDNA >32 KB-LDS/graph-capture note `:63-67`). Instances are autogenerated: `/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/template-instances/fattn-mma-turbo-*.cu` (D∈{128,256}, ncols∈{8,16,32,64}); declared at `fattn-mma-turbo.cuh:193-246`.
Gate: decode only (`Q->ne[1] <= 4`), D∈{128,256}, `turing_mma_available || amd_wmma_available` (`fattn.cu:2387-2392`); kill switch `GGML_TURBO_MMA_FUSED=0` (`:2330`); matched pairs plus an **adjacent-tier asymmetric allowlist** (`:2352-2372`), incl. `f16↔t8` and `q8_0:turbo4`.

**(b) Expand-then-attention (materialize)** — bulk dequant to f16 then generic MMA. Kernels `k_turbo2/3/4/8_dequant_f16` (`fattn.cu:521,543,630,819`), `*_inv_fwht` variants (`:696,747,784,842,1048,1090,1132`), `k_turbo1_tcq_dequant_f16[:979]/_rot[:1019]`, TCQ `k_turbo3_tcq_dequant_f16[:572]`/`k_turbo2_tcq_dequant_f16[:601]`. Geometry: `dim3 grid(K->ne[1],K->ne[2],K->ne[3])`, block `K->ne[0]` (non-FWHT) or **128** (FWHT variants) — `:1637-1687` (K), `:1698-1761` (V), `:2647-2674`; VBR path `dim3 grid(n_cells,1,1)` block 128 (`:1308-1349`) with `k_vbr_f16_gather`/`k_vbr_f16_to_f32_scaled` (`:1273-1361`) and assert `ne0 % 128 == 0` (`:1304`). Q pre-rotation `k_turbo_fwht_forward<<<n_q_groups,128>>>` (`:1235`, launched `:1803,:2427`).

**Scratch VRAM accounting** (the known concern): `struct ggml_cuda_fattn_scratch { q_rot_buf; side k; side v; epoch }` and `ggml_cuda_vbr_transcode_workspace` are **per backend context**, not per device, so independent llama contexts do not alias them (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/common.cuh:1449-1499`; rationale `:1489-1499`). Per side: VMM pool (contiguous VA once, 2 MiB physical pages on demand, mapped to **exact** attended width) with `cudaMalloc`+`next_pow2` fallback; grow-only, `epoch++` on every address move to invalidate captured graphs (`fattn.cu:1374-1470`). Boundary reserve `ggml_backend_cuda_kv_dequant_scratch_reserve` (`fattn.cu:1478-1490`); **projection = `(scratch_k_row + scratch_v_row) * wm_cells`** (`/models/servers/llama-turbo-optimal/src/llama-kv-cache.cpp:4565-4571`) and it is folded into the budget test (`:4505-4520`); single-side growth example "217 MiB with 192 MiB headroom" (`:4524-4530`); abort message on exhaustion (`fattn.cu:1467`). Authoritative materialize condition `ggml_vbr_kv_dequant_sides()` (`ggml/include/ggml-vbr.h:31-44`) — "edit HERE only". Transcode workspace planes f16+f32+idx+mean, 128-byte-aligned, `TILE=256` → ~6 MiB f32 at ne0=6144 (`vbr-transcode.cu:42-100,474-477`). TCQ encoder backtrace: per-device `cudaMalloc` of `ne_total_groups*128*64` (t3/t2) or `*128*128` (t1) bytes (`set-rows.cu:412-420,1814,1904,1961`), replaced by 8 KiB dynamic shmem when `TURBO_TCQ_SHARED_BT` (default on) (`:1793-1800`). Debug leaks: `TURBO_EXTRACT` allocates `max_samples*4B` + tags and writes `/tmp/turbo_postrot.bin` (`turbo-quant-cuda.cuh:88-160`); sink stash `cudaMalloc(ne0*n_sink*2)` per tensor (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/turbo-sink.cu:22-33`).

**SM120/Blackwell**: one specialization only — `use_t3_coop = (cc == GGML_CUDA_CC_BLACKWELL)` selects `k_set_rows_turbo3_coop<<<groups,128>>>` instead of the 256-thread-per-block generic (`set-rows.cu:1699-1716`, kernel `:1192`); `GGML_CUDA_CC_BLACKWELL 1200` (`common.cuh:71`). No Blackwell-specific attention kernel: the fused path is `turing_mma_available`-gated and shared. HIP branch differs: `TCQ3_ENC_NT` 128 (RDNA wave32) vs 512 (NVIDIA) (`turbo-quant-cuda.cuh:1491-1497`) and RDNA4 picks 512/256/128 by batch (`set-rows.cu:1836-1850`).

### 5. Attention semantics

- K and V are **separate tensors with independent types**, incl. mixed pairs (`fattn-mma-turbo.cuh:34` `V_is_K_view=false`; allowlist `fattn.cu:2352-2372`).
- **RoPE before quantization**: `set-rows.cu:539` dumps "raw pre-FWHT (**post-RoPE**) K/V rows"; the cache is written with `ggml_set_rows` in `cpy_k`/`cpy_v` (`llama-kv-cache.cpp:12190, 12233, 12254`) — **quantized at append time**, one row per token per (head-dim block).
- Both K and V are stored **rotated (rotated space)**; therefore Q must be FWHT'd before `Q·K` (handled inline in the vec kernel's shmem FWHT and by `k_turbo_fwht_forward` for prefill MMA — `llama-graph.cpp:3244-3246`, `fattn.cu:1235`), and the attention **output is un-rotated in-graph** with `ggml_turbo_wht(ctx, cur, /*inverse=*/1)` + `+mu_V` (except `TURBO8_0`) in `build_attn_v_unrotate` (`/models/servers/llama-turbo-optimal/src/llama-graph.cpp:3045-3070`).
- head_dim must be a multiple of 128: cache allocation pads per-head to `ceil(head/128)*128` — "Turbo head padding: FWHT requires head_dim % 128 == 0 … zeros contribute nothing via Parseval's theorem" (`llama-kv-cache.cpp:1207-1226`), input padded in `cpy_k:12162-12172` and Q padded/cropped in the graph (`llama-graph.cpp:3249-3270`); `head_dim > 512` ⇒ **fall back to f16** ("turbo FA limit", `llama-kv-cache.cpp:1200-1202`); assert `ne0 % 128 == 0` (`fattn.cu:1304`).
- Affine mean-sub tap: per-(model id, layer, channel) baked calibration, `ggml_turbo_meansub_model_id/table` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-turbo-meansub.cpp:43,77`, data `/models/servers/llama-turbo-optimal/ggml/src/ggml-turbo-meansub-data.inc`), K tap decode-free, V tap restored in graph; `TURBO8_0` deliberately excluded (`set-rows.cu:286-300`, `llama-graph.cpp:3062-3067`).
- Per-layer dtype selection exists as **`VBR_LAYER_SCHEDULE`** (`<il0>-<il1>:<k|v>:<tier>;…`, `llama-kv-cache.cpp:545-623,993`) and `--vbr-policy` ladders; the runtime degrade cursor moves one `(layer,side)` step at a time (`llama-kv-cache.cpp:3120-3140`, orders in `/models/servers/llama-turbo-optimal/src/llama-vbr-degrade-orders.inc`, `VBR_DEGRADE_ORDER` override).
- **Long-context constraints found in code (not markdown):** fused path is decode-only (`Q->ne[1]<=4`, `fattn.cu:2387`) — prefill always materializes (`:2374-2380`, `TURBO_FUSED_PREFILL` measured −6%…−11% at d8192, default off); non-adjacent mixed tiers "DO occur live and fall to the materialize path (measured −13-15% tg32 @ d8192)" (`fattn.cu:2349-2352`); the `f16→t8` entry band previously cost "−50% @ d64k" (`fattn-mma-turbo.cuh:229-235`); `q8_0:turbo4` fallback penalty "0% @d0, −10.8% @d16k, −28.7% @d64k" (`fattn.cu:2369-2371`); **`turbo1_tcq` fused decode is not InnerQ-scale-aware → forced onto materialize when calibrated** (`fattn.cu:2336-2341`, `turbo-quant-cuda.cuh:539-549`); a turbo side's dequant-scratch floor "cannot itself be degraded away", so the ladder can be permanently over budget → stability latch (`llama-kv-cache.cpp:3120-3140`); `GGML_ABORT` on scratch-grow exhaustion (`fattn.cu:1467`); 128K OOM reproduced by two ~120 MiB scratch reservations (`122880*1024`/side, `/models/servers/llama-turbo-optimal/docs/autopilot.md:541`); context-shift/self-extend disabled under dynamic VBR because the shift graph would touch unmapped pool pages (`/models/servers/llama-turbo-optimal/docs/vbr.md:88-89`).

### 6. Measured numbers

Bytes/token, exact from block sizes (K+V, full-attention layers only, `head_count_kv=4`, `key_len=value_len=256`, Qwen3.6-27B 16/64 KV layers): q8_0 K / turbo3 V = **24,576 B/token**; turbo1_tcq both sides = **5,120 B/token**; whole-model floor-tier ≈ **20,800 B/token** — `/models/servers/llama-turbo-optimal/docs/research/master-performance-opportunity-audit.md:528-545` (explicitly a desk calculation from `ggml-common.h`, not a run). Depth-scaled transfer: 403 MB @16k / 1.61 GB @64k / 6.44 GB @256k, 6.96/27.8/111.2 ms at 57.9 GB/s PCIe; ÷4.8 at floor tier.

Quality (KLD vs f16 KV, teacher-forced, deterministic): turbo1_tcq joint median **0.050923** @ ctx8192/24 chunks on Qwen3.6-27B-Q6_K, ladder anchor 0.0691 → 0.0569 → 0.0509 — `/models/servers/llama-turbo-optimal/turbo1_tcq_codebooks/PROVENANCE.md:3-8` (trained on a 14×4090 box, sm_89). turbo3_tcq ≈ **40% lower KLD than scalar turbo3 at the same 3.25 vs 3.5 bpv**; `turbo3_tcq K + turbo2_tcq V` = 2.75 bpv is **15-17% lower KLD than the reverse** — `/models/servers/llama-turbo-optimal/README.md:427-457`. Per-model price panels: 3169 KLD cells, matrix v3 (`llama-vbr-degrade-orders.inc:1-8`).

Speed (mixed machines — RTX 3090, RTX 5080/sm_120, gfx1151/1201; **no FP8/NVFP4-KV comparison exists in-repo**):
- Fixed tiers, ctx4096 code workload, MTP: turbo3 (3.5 bpv) **39.29 TG**; turbo3_tcq (3.25) **33.64 TG**; turbo4 (4.125) target-only **34.29**, n_max 1/2/3 **39.36/41.25/41.75 TG** — `/models/servers/llama-turbo-optimal/memory/STATE.md:1152-1166`; conclusion "TCQ's extra compression is not a net win here".
- ctx8192 turbo4: 35.20 TG / 599.52 PP (target-only), 34.22 TG / 586.57 PP (MTP n_max=2); ctx16384 turbo4 n_max=3: **38.07 TG / 571.39 PP**, `STATE.md:1169-1183`.
- Static turbo3:turbo3 @ ctx65536: **341.3 pp / 59.4 tg** — `/models/servers/llama-turbo-optimal/docs/research/vbr-long-context-oom-capacity-audit.md:456-458`.
- Dense 65536 certified profile, target `q8_0` / draft-V `turbo3`: 1714.04/79.32 (atomic) vs 1707.36/76.97 (fork) pp/tg — `docs/autopilot.md:607`; MoE 262144 with `q8_0/turbo4`: 1211.62/45.45 — `:599`.
- VBR controller steady-state: 18.7 → **48.2 tg** after the budget-guard fix (`vbr-long-context-oom-capacity-audit.md:446-452`).
- `k_set_rows_turbo3_coop` (Blackwell) fix: **+0.96% pp / +10.5% tg**, bit-exact — `/models/servers/llama-turbo-optimal/LESSONS.md:22`.
- Do **not** copy one number set blindly: `LESSONS.md:80` documents a `-ctv turbo3` CLI string resolving to `TURBO3_0` while a *different* parser in `llama-kv-cache.cpp` aliases bare "turbo3" to the **TCQ** codec.

### 7. Test vectors / correctness oracles

- **Deterministic byte-level round-trip oracle (best candidate):** `llama_kv_cache::vbr_transcode_anchor_test()`, env `VBR_TRANSCODE_TEST`, `/models/servers/llama-turbo-optimal/src/llama-kv-cache.cpp:9138-9240` — synthesizes valid turbo8 from a fixed LCG pattern `r = i*1103515245u + 12345u; (r & 0xFFFF)/32768.0f - 1.0f` (`:9200-9202`), then (a) A→A transcode must be **byte-identical**, (b) A→B separate-dst vs in-place must be byte-identical (the in-place trailing invariant). Declared `/models/servers/llama-turbo-optimal/src/llama-kv-cache.h:1645`, row-count knob `VBR_TRANSCODE_TEST_N`, run with `TURBO_MEANSUB_OFF=1`; armed from `apply_ubatch` (`llama-kv-cache.cpp:3933`). Per-transcode error report: `VBR_TRANSCODE_FIDELITY` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/vbr-transcode.cu:127,364-380,469`).
- **CPU round-trip smoke:** `/models/servers/llama-turbo-optimal/tests/test-turbo-quant.c` (tests 1-3, MSE/cosine/norm printout; **caveat:** tests 1-2 exercise the turbo3 stub, so only test 3 (turbo4) is meaningful).
- **GPU runtime suite:** `/models/servers/llama-turbo-optimal/tests/test-cuda-turbo3-runtime.cpp` — `--case backend|smoke|set_rows|mul_mat_id|fattn_turbo3` (`:17-23`); `set_rows` uses **exact `!=` float comparison** against a hand-computed expected value (`:220-231`), `fattn_turbo3` builds a public `FLASH_ATTN_EXT` graph with turbo K/V and cross-checks CPU-vs-CUDA (`:346-443`, tolerance `1e-5` at `:312`).
- **Coverage mandate (use as your oracle spec):** `/models/servers/llama-turbo-optimal/docs/autopilot.md:657` — Turbo3 widths 128/256/384, I32/I64 indices, 1 and 7 rows, K/V, **cooperative vs generic bit-exact equivalence**, K/V PFH1 mean-sub on/off, InnerQ calibration/transition/dispatch, D=128/256, GQA 1/2/3/4/5/8/12/16, batches 1/4/32, KV 128/129/1024, T3/T3 · Q8/T3 · T3/Q8, `NMSE <= 5e-4`, non-zero dispatch counters.
- **Quality-only per-tier harness:** `k_set_rows_ragged_roundtrip` (`/models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/set-rows.cu:731-800`) + `turbo_roundtrip_block_to_orig` / `ragged_lookup_tier` / `ragged_band{pos,layer,cb,kv,tier}` (`turbo-quant-cuda.cuh:1105-1220`, tier codes `16=f16, 8, 4, 3, 2, 11=t1tcq, 22, 33`, schedule via `TURBO_RAGGED_SCHEDULE`, `set-rows.cu:424-522`) — quantize→dequantize→inverse-FWHT and store as f16 so a plain f16 attention reproduces the real rotated-domain score.
- **Trellis bit-exactness instruments:** `TURBO_TCQ_DUMP_ERRORS=N` dumps post-FWHT inputs + output symbols (`set-rows.cu:98-131`, `turbo-quant-cuda.cuh:56-60,1722-1740`) consumed by `/models/servers/llama-turbo-optimal/scripts/analyze_tcq_errors.py` and `/models/servers/llama-turbo-optimal/scripts/tcq_diagnostics.cuh`; `TURBO_EXTRACT` post-rotation corpus (`turbo-quant-cuda.cuh:88-160`) feeds `scripts/tcq_train_{1bit,2bit,v2,product,free_init,rshift,tailbiting_prototype}.py`, `scripts/tcq_free_init_test.py`, `scripts/analyze_norm_correction.py`.
- Controller/state tests (not codec): `/models/servers/llama-turbo-optimal/tests/test-vbr-{policy,downward,physical,transaction,artifact,artifact-capture,artifact-adopt,hard-seal,vmm,representation-epoch}.cpp`, `test-llama-bench-vbr.cpp`; full-ladder gate evidence `F16→T8→T4→T3→T2→T1` with QSA index bit-identical across retiers — `/models/servers/llama-turbo-optimal/docs/qwen4-vbr-plan.md:420`.

**NOT FOUND anywhere in the repo:** a Python/NumPy reference implementation of any turbo codec; a bit-exact spec/test for the TCQ *decoder* outside the CUDA kernels; any per-token variable-length/bit-offset index (all tiers are fixed-row); FP8 or NVFP4 **KV-cache** quality comparisons against Turbo (NVFP4 appears only as weight-quant / a model name); and any `GGML_TYPE_META`-style reflection API.