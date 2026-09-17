# A8 — turbo3 / turbo4 KV codec byte-level spec (from llama-turbo-optimal)

Source: `/models/servers/llama-turbo-optimal` @ d0a274167. All file:line below are relative to that root.
The **CUDA path is the authoritative codec**; the CPU ref files diverge (see §2.0). Do not port `ggml/src/ggml-turbo-quant.c` as-is.

## 1. Row layout

Type ids: `GGML_TYPE_TURBO3_0 = 43`, `GGML_TYPE_TURBO4_0 = 44` (`ggml/include/ggml.h:435-436`). The ggml.h
comments ("2-bit PolarQuant + 1-bit QJL" / "3-bit + QJL") are **stale**: both are pure PolarQuant + RWHT,
no QJL (`ggml/src/ggml-turbo-quant.c:2-8`, `turbo-quant-cuda.cuh:676` "QJL sign arrays removed").

### turbo3 — `block_turbo3_0` (`ggml/src/ggml-common.h:309-316`)

```c
#define QK_TURBO3 32        // storage block = 32 elems
#define QK_TURBO3_GROUP 128 // rotation group = head_dim chunk
typedef struct {
    ggml_half  norm;                  //  2 B
    uint8_t    qs[QK_TURBO3 / 4];     //  8 B: low 2 bits of idx, 4 per byte
    uint8_t    signs[QK_TURBO3 / 8];  //  4 B: bit 2 (high) of idx, 8 per byte
} block_turbo3_0;                     // 14 B / 32 elems = 3.5 bpv
```
Verified 14 B (static_assert `ggml-common.h:316`). One rotation group (128 elems = head_dim chunk) =
**4 sub-blocks that all store the same `norm`** (`turbo-quant-cuda.cuh:932-936`).

Bit packing / unpacking (element `j` in 0..31, exact code `turbo-quant-cuda.cuh:924-925`,
decode twins `fattn.cu:727-728`, `set-rows.cu:1293-1305`):
```
idx(j) = ((qs[j/4] >> ((j%4)*2)) & 0x3) | (((signs[j/8] >> (j%8)) & 0x1) << 2)
recon  = centroid3[idx] * norm              // rotated domain, no inverse transform here
```

`norm` semantics: **fp16 corrected scale, not raw L2**: `norm = ||x||_group / ||c||_group` where
`||c|| = sqrt(sum_j centroid3[idx_j]^2)` over the 128-elem group (comment `turbo-quant-cuda.cuh:931-935`:
"store corrected norm so dequant(x) has exact original L2 norm"). Fallback `norm = ||x||` if recon_norm <= 1e-10.
Decoded as a plain per-element multiply (`fattn-common.cuh:802`).

### turbo4 — `block_turbo4_0` (`ggml/src/ggml-common.h:356-361`)

```c
#define QK_TURBO4 128
typedef struct {
    ggml_half  norm;                  //  2 B
    uint8_t    qs[QK_TURBO4 / 2];     // 64 B: 4-bit idx, 2 per byte, low nibble = even j
} block_turbo4_0;                     // 66 B / 128 elems = 4.125 bpv
```
Verified 66 B (static_assert `:361`). Block == rotation group (128). Nibble order **low-first**,
encode `dst->qs[j/2] = (idx[j+1] << 4) | idx[j]` (`turbo-quant-cuda.cuh:1020-1023`),
decode `idx = (j&1) ? qs[j/2]>>4 : qs[j/2] & 0xF` (`turbo-quant-cuda.cuh:1043`, `fattn.cu:769`).
`norm` = corrected norm above **× `d_turbo4_alpha`** (`turbo-quant-cuda.cuh:1033`); alpha default 1.0
("KLD-optimal: any scaling hurts", `:1001-1003`), env `TURBO4_ALPHA` (`set-rows.cu:20-32`).

## 2. Encode pipeline (CUDA, authoritative)

Kernels: generic `k_set_rows_turbo3` (`turbo-quant-cuda.cuh:833-937`), coop twins
`k_set_rows_turbo3_coop` (`set-rows.cu:1192-1302`), `k_set_rows_turbo4_coop` (`set-rows.cu:1114-1179`),
generic turbo4 block `quantize_f32_turbo4_0_block` (`turbo-quant-cuda.cuh:1007-1036`). Ordered steps
(turbo3 line refs `turbo-quant-cuda.cuh:845-937` / `set-rows.cu:1216-1301`; turbo4 `set-rows.cu:1140-1178`):

1. **Load 128-elem group** from f32 src. Assert `ne00 % 128 == 0` (`set-rows.cu:1678`, `:246`).
2. **InnerQ calibration** (optional accumulation, both K and V, on RAW pre-tap values): per-channel
   sum-of-squares + running `atomicMax|K_i|` CAS + count (`turbo-quant-cuda.cuh:860-885`).
3. **Mean-sub (affine tap)**: `x[j] -= kmean_mu[i00+j]` in the **raw domain, pre-norm pre-rotation**
   (`turbo-quant-cuda.cuh:886-889`). Gating: dst tensor named `cache_k_l<N>` / `cache_v_l<N>`,
   N < 128, ne00 <= 2048, and a table is loaded (`set-rows.cu:286-299`, `:1694-1699`); **off for turbo8**
   (`set-rows.cu:290`, comment `:287-289`: t8 "zero median gain and +9.5% mean KLD cost", t4/t3 kept,
   "t4 median -33%, t3 -39%"). Model id parsed from `_ms<id>` suffix (`set-rows.cu:281-284`).
   Calibration constants: baked per-(arch, n_layer, n_embd) dense [128 layers][2048 channels] fp32 K and V
   mean slabs (`ggml/src/ggml-turbo-meansub.cpp:11-23,43-54`, data `ggml-turbo-meansub-data.inc`;
   limits MAX_L=128, MAX_C=2048, MAX_MODELS=16 `ggml/include/ggml-turbo-meansub.h:12-14`); layers with
   probe cnt < 100 get mu=0 (`turbo-quant-cuda.cuh:318-320`); env `TURBO_KMEAN_SUB`/`TURBO_VMEAN_SUB`
   override with PFH1 dumps, `TURBO_MEANSUB_OFF` kills the tap (`turbo-quant-cuda.cuh:277`).
   K tap needs no restore (softmax shift-invariant, per-head constant, comment `:244-250`); **V tap is
   undone at graph level** by adding mu_V back after attention (weights sum to 1) — `src/llama-graph.cpp:3060-3069`.
4. **InnerQ per-channel scale**: `x[j] *= d_innerq_channel_scale[j]` — **turbo3 only**
   (`turbo-quant-cuda.cuh:890`, `set-rows.cu:1245`); turbo4/turbo8 encoders never apply it (asymmetry
   with the decode side, which multiplies `scale_inv` — benign only because default is identity).
   Derivation (`turbo-quant-cuda.cuh:415-596`): mode 0 default RMS rule `scale_i = (mean_rms/channel_rms_i)^strength`,
   `strength` default 0.5 (`TURBO_INNERQ_STRENGTH`), clamped to [0.5, 2.0]; mode 1 = 1/sqrt(max|K_i|)
   normalized to geometric mean 1 (`:481-505`). **Auto-disable to identity if max scale ratio < 1.2**
   (`:546-558`) => in default builds InnerQ is off; calibrated scales then require decode to use the
   pushed inv-scale symbols (`:533-544` fail-loud note).
5. **Group L2 norm**: serial `norm_sq += x[j]*x[j]`, `grp_norm=sqrt`, `inv_norm = grp_norm>1e-10 ? 1/grp_norm : 0`,
   normalize (`turbo-quant-cuda.cuh:891-895`).
6. **Forward rotation** `turbo_rotate_forward_cuda` (`turbo-quant-cuda.cuh:779-784`): `x *= s1; fwht128(x); x *= s2`,
   where `fwht128` (`:764-774`) is the in-place butterfly (h=1..64, `a+b / a-b`) followed by
   `x[i] *= 0.08838834764831845f` (= 1/sqrt(128), folded inside the FWHT call). s1/s2 are **128-float ±1
   arrays, not scalars** (`:714-717`; generation: RWHT compression of the seed=42 rotation, `ggml/src/ggml-metal/turbo-wht.h:1-3`,
   `turbo-wht.cu:4-17` — exact values, identical in both TUs and in `fattn-common.cuh:312-315`):

   `signs1`: `-1 1 1 -1 -1 1 -1 1 -1 -1 1 1 1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 1 -1 1 1 -1 -1 -1 -1 1 1 -1 1 1 -1 1 -1 1 1 -1 -1 1 -1 1 1 1 1 -1 -1 -1 -1 -1 1 -1 1 1 1 1 -1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 1 1 -1 -1 1 1 1 -1 -1 1 1 -1 1 1 -1 1 -1 -1 1 1 -1 1 -1 1 -1 1 1 1 1 -1 1 -1 1 1 -1 1 1 -1 -1 -1 -1 -1 1 1 -1 1 1 -1 1`
   `signs2`: `1 1 1 1 -1 1 1 -1 1 -1 -1 -1 1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 -1 -1 1 -1 1 1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1 -1 1 1 1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 1 -1 1 -1 -1 -1 -1 1 -1 1 -1 1 -1 -1 1 1 -1 1 -1 1 1 -1 1 -1 -1 -1 -1 1 -1 -1 1 -1 1 -1 1 1 1 -1 -1 1 -1 1 -1 1 1 -1 -1 1 -1 1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 -1`

7. **Nearest centroid** = threshold search on midpoint tables; **tie rule: strict `<`, equality takes the
   higher index** (so val==0.0 falls to index 4 for turbo3). 3-bit: linear 7-way (`turbo-quant-cuda.cuh:786-796`);
   4-bit: 4-level binary search on 15 mids (`:798-828`). Stock tables (compile default `TURBO3_SKEW_EXP=0`):
   ```
   CENTROIDS_3BIT[8]  = -0.190685 -0.117832 -0.065717 -0.021460 0.021460 0.065717 0.117832 0.190685   (turbo-quant-cuda.cuh:617-620; ggml-turbo-quant.c:33-36)
   MID_3BIT[7]        = -0.154259 -0.091775 -0.043589  0.0      0.043589  0.091775  0.154259          (turbo-quant-cuda.cuh:621-624)
   CENTROIDS_4BIT[16] = -0.241556 -0.182907 -0.143047 -0.111065 -0.083317 -0.058069 -0.034311 -0.011353
                         0.011353  0.034311  0.058069  0.083317  0.111065  0.143047  0.182907  0.241556 (turbo-quant-cuda.cuh:659-663; ggml-turbo-quant.c:39-44)
   MID_4BIT[15]       = -0.212232 -0.162977 -0.127056 -0.097191 -0.070693 -0.046190 -0.022832 0.0
                         0.022832  0.046190  0.070693  0.097191  0.127056  0.162977  0.212232         (turbo-quant-cuda.cuh:665-669; ggml-turbo-quant.c:45-49)
   ```
   Convention: Lloyd-Max for N(0, 1/128) post-normalize+rotate coords (`ggml-turbo-quant.c:31-38`,
   `turbo-quant-cuda.cuh:657`). Alternate books exist: `TURBO3_SKEW_EXP=1/2` symmetric-retrained /
   asymmetric books + per-coord skew sign arrays `d_turbo3_ss_k/v[128]` folded into idx search or decode
   sign flip (`turbo-quant-cuda.cuh:600-615, 627-636`; decode side `fattn.cu:560-567`, `fattn-mma-f16.cuh:745-750`) —
   **stock default = plain book, SKEW=0**. Runtime file swap `TURBO_CB_T2/T3/T4/T8` (`turbo-quant-cuda.cuh:722-760`).
8. **Pack** (§1 formulas) and **store corrected norm** (fp16(grp_norm/recon_norm); turbo4 × alpha).
9. **Geometry rules**: KV cache tensor head stride is zero-padded up to `ceil(head_dim/128)*128` at
   allocation when the cache type is turbo — "contribute nothing via Parseval's theorem"
   (`src/llama-kv-cache.cpp:1207-1235`); `head_dim > 512` with a turbo cache type **falls back to f16**
   per K/V side ("Turbo FA vec kernel supports head_dim <= 512", `src/llama-kv-cache.cpp:1184-1205`).

### 2.0 CPU ref status (do not trust)
`quantize_row_turbo3_0_ref` is a **stub**: writes norm + zero indices (`ggml-turbo-quant.c:330-341`).
`dequantize_row_turbo3_0` (`:343-357`) returns rotated-domain values with no inverse rotation.
`quantize_row_turbo4_0_ref` (`:458-506`) is functional but rotates with a **dense 128x128 QR-random
matrix** (Box-Muller LCG `state = state*6364136223846793005 + 1442695040888963407`, seed 42,
`:134-186,26-27`) — *not* the CUDA RWHT — so CPU-encoded bytes differ from CUDA. CPU traits advertise
`vec_dot = NULL` for both types (`ggml/src/ggml-cpu/ggml-cpu.c:427-438`).

## 3. Decode pipeline

Per-element inverse of §2 (rotated domain unless noted):
- `val_rot = centroid[idx(j)] * norm` (turbo3 `turbo-quant-cuda.cuh:941-954`; turbo4 `:1039-1050`).
- Original-domain materialization (`k_turbo3_dequant_f16_inv_fwht` `fattn.cu:696-743`,
  `k_turbo4_dequant_f16_inv_fwht` `fattn.cu:747-778`, warp-coop twins in `set-rows.cu:1088-1103`):
  ```
  t = centroid[idx]                                  // f32
  t = fwht128_butterfly_inplace(t * s2[tid])         // butterfly = inverse of forward butterfly
  out = t * (1/sqrt(128)) * s1[tid] * innerq_scale_inv[tid] * norm   // cast half2 pair-store
  ```
  (`fattn.cu:736-741`, `:773-776`). **Claim verified**: the inverse uses the same butterfly with s1/s2
  swapped (forward = s1, H, s2; inverse = s2, H, s1 — H self-inverse up to 128, normalized by 1/sqrt(128)
  inside; offline twin comment `scripts/e9_rotation_offline.py:12-14`). The decode additionally folds the
  InnerQ inv-scale (`:739`) — omit when InnerQ is identity (default).
- Attention-domain contract: stored K/V stay rotated; the fused/vec paths **rotate Q forward** instead of
  inverse-rotating K (`k_turbo_fwht_forward` `fattn.cu:1235-1259`, applied `:2425-2433`), and the graph
  **inverse-rotates the attention output** (V un-rotation, `src/llama-graph.cpp:3043-3058`) then restores
  mu_V (`:3060-3069`). The materialize path instead inverse-FWHTs K to the original domain and keeps Q
  unrotated (`fattn.cu:2636-2642` comment; exception `k_t3_use_rotated` `:2628-2630`).

## 4. Attention consumption (CUDA)

- **Fused MMA decode** (`ggml/src/ggml-cuda/fattn-mma-turbo.cuh`): reuses `flash_attn_ext_f16` templated
  on `type_K/type_V`, dequant into half2 shmem tiles — no fp16 scratch. Gate (`fattn.cu:2385-2389`):
  `turbo_mma_fused` (env `GGML_TURBO_MMA_FUSED`, `:2329-2336`) && (matched K==V `:2341` or asym pair
  `:2353-2374`) && **`(Q->ne[1] <= 4 || turbo_fused_prefill)`** && D ∈ {128,256} && (Turing-MMA or ROCm-WMMA).
  Instance table incl. (TURBO3_0,TURBO3_0) and (TURBO4_0,TURBO4_0) at both D `:2456-2477`; asym pairs
  D=256-only. Tile machinery: `flash_attn_ext_turbo4_load_tile` (`fattn-mma-f16.cuh:558-604`, per-block
  16×`cent*norm` half LUT, 64 byte-stores/row), general loader for turbo3/2/tcq (`:706-760`,
  row-per-thread); K call sites `:980/:998`, V `:1362/:1374`. Turbo forces `nstages=0` (cp.async can't do
  ALU dequant) and tile_K/tile_V share one shmem stage (`fattn-mma-turbo.cuh:27-30`); shmem sizes `:36-47`.
  GQA tiling: `ncols2` from gqa_ratio %8/%4/%2 (`fattn.cu:263-300`), `ncols1` picked by `Q->ne[1] <=
  8/ncols2 / 16/ncols2 / 32/ncols2` (`:236-261`).
- **Decode default is NOT the native VEC kernel**: matched-but-unfused or override-routed pairs
  dequant to an f16 scratch and run generic MMA ("MMA tensor cores on fp16 beat VEC scalar on turbo
  bits ... ~1% slower native", `fattn.cu:2528-2530`; predicate `do_decode_dequant` `:2570-2574`,
  native opt-in `GGML_TURBO_DECODE_NATIVE` `:2532`). Native VEC KQ dot exists for both types
  (`vec_dot_fattn_vec_KQ_turbo3_0` `fattn-common.cuh:780-841`, `_turbo4_0` `:843-882`; V
  `dequantize_V_turbo3_0/:1323`, `_turbo4_0/:1363`; wiring `:1560,1562,1592,1594`; turbo lanes
  `nthreads_KQ=128/cpy_nb` `fattn-vec.cuh:91-103`).
- **Prefill** (Q->ne[1] > 1, D<=256): `ggml_cuda_turbo_prefill_attend` — dequant-to-f16 + MMA
  (`fattn.cu:2483-2489`, body `:1605+`). Fused prefill exists but was measured a **loss** ("neutral at d0
  but -6% (t8/t4) to -11% (t3/t1_tcq) at d8192 — re-decoding the K/V tile once per 64-column block",
  `:2375-2383`).
- **Measured consequences in comments**: materialize fallback costs "0% @d0, -10.8% @d16k, -28.7%
  @d64k" tg vs fused (q8_0-K pair, `:2366-2369`); non-adjacent mixed VBR tiers "measured -13-15% tg32
  @ d8192" (`:2351-2352`); scratch buffer is VMM-backed after a 500 MiB contiguous-growth OOM at
  131k->262k (`:2630-2648`).
- **`k_set_rows_turbo3_coop` (SM120/Blackwell path)**: launched *only* when
  `devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL` (1200, `common.cuh:71`), non-HIP/MUSA, and
  `TURBO_EXTRACT` unarmed (`set-rows.cu:1701-1722`; `#else use_t3_coop=false :1705`). Differences vs
  generic: 128 threads/group instead of 1 thread/group (generic spills `float x[128]`, 512 B/thread,
  serial FWHT, ~13x slower, "~215 ms/prefill" on 26k tokens, `:1181-1187`); butterfly via
  `__shfl_xor_sync` passes h<=16 + 2 smem passes h=32/64 (`ragged_fwht128_butterfly_inplace`
  `:1026-1046`); **norm reductions deliberately serial over smem (`:1250-1257`) so the kernel is
  bit-exact with the generic one**, unlike turbo4_coop's tree reductions which are only ULP-equal
  (`:1106-1112`, `:1145-1154`). LESSONS.md:22 records +0.96% prompt / +10.5% gen t/s from this fix.
  turbo4's coop path is **not** arch-gated (`set-rows.cu:1724-1741`).

## 5. Determinism oracles

- **`vbr_transcode_anchor_test`** (`src/llama-kv-cache.cpp:9145`; purpose comment `:9138-9144`). Armed
  by env `VBR_TRANSCODE_TEST`, fires once from `apply_ubatch` on the 2nd call after >= 512 used cells so
  InnerQ decode scales are identity-initialized first (`:3930-3944`). LCG input pattern (`:9199-9201`):
  ```c
  uint32_t r = (uint32_t) i * 1103515245u + 12345u;
  hp[i] = (float)(r & 0xFFFF) / 32768.0f - 1.0f;   // deterministic non-zero, in [-1, 1)
  ```
  `N` default 256 rows (`VBR_TRANSCODE_TEST_N`, `:9186-9187`); width = live layer's ne0 (fallback 1024,
  `:9178-9184`); tensor named `cache_k_l3` to force the K codebook (`:9194`). Checks: turbo8 A->A
  transcode byte agreement (`:9227-9236`); every direct ladder promote edge t8->t4/t3/t2 and
  t4->t3/t2, t3->t2 **in-place == separate-dst byte-identical**, dequant-adjudicated when TCQ
  don't-care bits differ (`:9333-9441`); cross-domain reconstruct to t8/f16 in-place==separate under
  `GGML_ASSERT` (`:9443-9499`). K and V each exercised (`is_v` loop). Tap-off runs use
  `TURBO_MEANSUB_OFF=1` (`:9142-9143`).
- **Coverage mandate** is prose, not code: `docs/autopilot.md:657` — "Required minimum coverage: Turbo3
  widths 128/256/384, I32/I64 indices, one and seven rows, K/V, and cooperative versus generic
  equivalence; negative indices -1, first, and last with sentinels; K/V PFH1 mean-sub enabled and
  disabled; InnerQ calibration, transition, and dispatch; attention D=128/256, GQA 1/2/3/4/5/8/12/16,
  batches 1/4/32, KV 128/129/1024, and T3/T3, Q8/T3, and T3/Q8; ... Require NMSE <= 5e-4 ... and
  bit-exact equivalence where specified. DKQ512 with odd GQA greater than one remains unsupported."
  The suite implementing it **does not exist yet** (`memory/LESSONS.md:401-403`: "the Sol-defined suites
  must be implemented and run independently").
- **NMSE computation** (the only implemented numerical gate): `tests/test-cuda-turbo3-runtime.cpp` —
  fixed case D=128, n_queries=4, n_kv=64, heads=2 (`:435-438`); CPU-ref vs CUDA0 FLASH_ATTN_EXT;
  `nmse = sum((cuda-cpu)^2) / sum(cpu^2)` over all outputs (`:486-496`), limit `constexpr double
  max_nmse = 5.0e-4` (`:498-500`), non-finite = fail (`:489-492`). Inputs are deterministic sin/cos
  patterns (`:449-453`). **Status**: the case SKIPs (exit 77) whenever the CPU turbo3 traits lack
  `vec_dot` (`:426-435`), which they do (`ggml-cpu.c:427-432`); and CPU-side quantization of the K/V
  inputs goes through the `quantize_turbo3_0` stub (`:457-460` -> `ggml.c:8392` ->
  `ggml-turbo-quant.c:330`). set_rows oracle case: 4-row dst / 2 writes, I64 ids {3,1}, exact-value
  compare (`:173-237`).

## 6. Triton translation notes

Per 128-elem group, row base `p` (turbo3: `p + blk*14`, blk = g*4..g*4+3 of the group; turbo4: `p + g*66`):
- turbo3 needs 3 loads: `norm` fp16 (1 per group, any of the 4 copies), `qs` 8B (`tl.load(p+2+tl.arange(0,8))`),
  `signs` 4B (`p+10..13`); turbo4: `norm` fp16 + `qs` 64B. Index math per element j: exactly §1 formulas —
  vectorize as `low2 = (qs[j//4] >> ((j%4)*2)) & 3`, `hi1 = (signs[j//8] >> (j%8)) & 1`,
  `w = centroid3[low2 | hi1<<2]`. turbo4: `w = centroid4[(qs[j//2] >> (4*(j&1))) & 0xF]`.
- Constexpr: QK (32/128), group=128, both centroid + midpoint tables (they become `tl.constexpr` arrays or
  a gather over a constant LUT of 8/16 floats; precompute `centroid*norm` halves LUTs exactly like
  `fattn-mma-f16.cuh:583-587, 731-734`). Row pointer strides are byte offsets: `nb1 = (ne00/32)*14` (turbo3)
  / `(ne00/128)*66` (turbo4) from the ggml type table (`ggml.c:775-790`).
- Decode-with-inverse-rotation (materialize) needs the whole 128-group in one tile: FWHT as log2(128)=7
  butterfly stages. CUDA splits it warp-wise (`__shfl_xor_sync` h<=16, smem h=32/64, `fattn.cu:659-681`);
  Triton must instead keep the 128 values in registers and emulate the same pair ops via reshape/xor —
  any order is numerically equal only up to fp reassociation, so pick one order and use it in encode
  *and* decode or accept index flips at midpoint ties.
- Determinism trap: turbo3's CUDA encoder is bit-exact-serial-reduction on purpose (`set-rows.cu:1186-1192`).
  `tl.sum` is tree-order; if FreeToken wants to reproduce LTO bytes, force the sum order (explicit loop)
  or do not claim bit-exactness. Norm quantization to fp16 (`__float2half` == RTNE) and the strict-`<`
  midpoint tie rule (§2.7) must match.
- Geometry: pad head_dim to 128 multiples (zero pad, §2.9); head_dim>512 -> plain f16 (or f16-fallback
  policy); 14/66 B rows break 16B alignment only for turbo3 `signs`-offset reads — plain `tl.load` of
  int8 lanes is fine.
- If porting the attention-fused variant: replicate Q-forward-rotation (one `rotate_fwd` per Q row,
  §3), the `Q<=4`-tokens gate is a *dispatch* policy, not a codec constraint; V restore = inverse
  rotation of the attention output + mu_V add (§3), valid because softmax weights sum to 1.

## Verification notes
- Row sizes, packing, tables, formulas: read from source (line refs above); nothing was executed.
- Repo-wide grep confirms no other definition of the 3/4-bit books besides the three TURBO3_SKEW variants
  (`turbo-quant-cuda.cuh:600-624`, `fattn-common.cuh:9-30`) and the CPU copies (`ggml-turbo-quant.c:29-49`).
