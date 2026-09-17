# A7 — GGUF MoE expert-bank geometry audit

Read-only audit, branch `next` @ 86af2d3 (+ uncommitted `cache_budget.py`/`kvcache/base.py` edits;
`expert_bytes_per_slot` is unchanged by them); line numbers are the working tree. Corpus generator:
`.qwen/tmp/a7_corpus.py` (metadata-only via `freetoken.models.gguf.reader._reader`, never reads tensor bytes).

## 1. CORPUS — every top-level GGUF under /models (7 files)

| file | bytes | arch | expert cfg | tensors | expert stacks |
|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf | 17,437,861,408 | qwen35moe | block_count 41 (40+1 nextn), expert_count 256, used 8, I=512, H=2048 | 753 | yes |
| Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf | 18,629,540,384 | qwen35moe | same dims | 753 | yes |
| Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf | 11,975,960,640 | qwen35 | 65 blk, H=5120, ffn=17408, nextn 1 | 866 | 0 |
| Qwen3.8-27B-GSQ-RCO-IQ3_S-mmproj.gguf | 931,146,528 | clip | — | 334 | 0 |
| Qwen3.6-35B-A3B-mmproj.gguf | 927,606,944 | clip | — | 334 | 0 |
| dflash-draft-Ornith15.gguf | 782,818,688 | dflash | 6 blk | 69 | 0 |
| qwen36-35b-a3b-dflash-Q4_K_M.gguf | 235,691,744 | dflash | 6 blk | 69 | 0 |

Depth-2 adds only mmproj (`clip`) files; no sharded MoE GGUF sets outside the top level. Both MoE
files mix types **across layers within one bank role** (counts = layers using that type):

Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf:

```
blk.%d.ffn_gate_exps.weight -> {Q4_K:10 [0-4,35-39], Q3_K:30 [5-34], Q8_0:1 [40=nextn]}  ne=[2048,512,256]
blk.%d.ffn_up_exps.weight   -> identical to gate_exps
blk.%d.ffn_down_exps.weight -> {Q4_K:10 [0-4,35-39], Q3_K:30 [5-34], Q8_0:1 [40]}         ne=[512,2048,256]
blk.%d.ffn_gate_shexp.weight/ffn_up_shexp/ffn_down_shexp -> {Q6_K:40, Q8_0:1}  (resident, not pooled)
blk.%d.ffn_gate_inp_shexp.weight -> {F32:41}
```

Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf:

```
blk.%d.ffn_gate_exps.weight -> {IQ3_S:40 [0-39], Q8_0:1 [40=nextn]}   ne=[2048,512,256]
blk.%d.ffn_up_exps.weight   -> identical
blk.%d.ffn_down_exps.weight -> {IQ4_XS:37 [0-33,35-37], Q6_K:3 [34,38,39], Q8_0:1 [40]}  ne=[512,2048,256]
blk.%d.ffn_*_shexp.weight   -> {Q8_0:41} ; ffn_gate_inp_shexp F32 x41
```

The Q8_0 `x1` is always layer 40 (the NextN/MTP block): the served model is
`block_count - nextn` = 40 layers deep
(`/models/desenvolvimento/freetoken-next/python/freetoken/models/qwen3_5_moe/gguf.py:106-107`) and loaders
skip `layer >= num_layers` (`.../models/qwen3_5_moe/gguf_experts.py:58-59`), so MTP never reaches the
pool. Mixed set per served bank: gate_up **10 Q4_K / 30 Q3_K**, down **10 Q4_K / 30 Q3_K** (Ornith);
down **3 Q6_K / 37 IQ4_XS** over a uniform IQ3_S gate_up (Tiel) — exactly what `layers/moe.py:436-440`
documents as un-servable.

## 2. CURRENT POOL CODE — where the uniform stride is chosen

The uniform-stride assumption sits at five independent points; mixed banks are **rejected, never
coerced or padded** (padding is documented as silently wrong, see §5).

1. Config-time sizing silently returns `None` on a mixed bank (no raise, so the byte estimate is
   just skipped) — `gate_up, down = set(types["gate_up"]), set(types["down"]); if len(gate_up) != 1 or len(down) != 1: return None`,
   `/models/desenvolvimento/freetoken-next/python/freetoken/models/qwen3_5_moe/gguf.py:96-98`.
2. Provider-level user-facing reject — `/models/desenvolvimento/freetoken-next/python/freetoken/moe/expert_banks.py:204-232`
   (`_gguf_banks`, defined at :180):
```python
    # One pool per bank is shared by every layer, and moe_vec.cuh addresses it without a
    # padding allowance, so a bank whose type varies by layer cannot be served. [...]
    resolved = {}
    for name in ("gate_up", "down"):
        distinct = sorted(set(types[name]))
        if len(distinct) != 1:
            spread = {GGML_NAME.get(t, t): [i for i, x in enumerate(types[name]) if x == t]
                      for t in distinct}
            raise NotImplementedError(f"GGUF expert bank {name!r} mixes ggml types across layers ({spread}). ...")
        resolved[name] = distinct[0]
```
   and the surviving single type per bank leaves the loader at `expert_banks.py:234`
   (`gguf_expert_types=(resolved["gate_up"], resolved["down"])`). Same reject is duplicated in
   `qwen3_moe/gguf_experts.py:113-119` and `deepseek_v4/gguf_experts.py:112-119`.
3. Per-slot byte size — `/models/desenvolvimento/freetoken-next/python/freetoken/engine/cache_budget.py:17-28`:
```python
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())
```
   `t[0]` is **layer 0's** bank, i.e. the per-slot price is read off one arbitrary layer's stride.
   `expert_banks.py:341-345` (`bank_bytes_estimate`) does the same with one type per bank.
4. GPU pool allocation — `/models/desenvolvimento/freetoken-next/python/freetoken/moe/offload_cache.py:351-373`
   (`set_bank_sources`): `head = per_layer[0]` (:354) is the geometry source of truth, layer
   uniformity is asserted (:362-367), then one pool per bank is allocated from that head (:369-373):
```python
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device)
```
   Repeated verbatim on resize at `offload_cache.py:505-509` (`rebuild`). The bank list per format
   is a name tuple, not a geometry tuple: `_BANK_SCHEMAS["gguf"] = ("gate_up", "down")`
   (`offload_cache.py:36-53`), field `gguf_expert_types: tuple[int, int] | None` (`:135`),
   documented "Per-bank ... but NOT per-layer" (`:130-134`).
5. Kernel dispatch — `/models/desenvolvimento/freetoken-next/python/freetoken/layers/moe.py:444-453`
   pulls one `(t_gate_up, t_down)` pair off the cache for every layer.
   The row bytes themselves come from
   `/models/desenvolvimento/freetoken-next/python/freetoken/models/gguf/dequant.py:146-157` (`row_bytes`)
   and `.../models/qwen3_5_moe/gguf_experts.py:123-124` (`rb = row_bytes(elems, distinct[0])`,
   `shape = (E, 2*I, rb) | (E, H, rb)`); the gate+up row-concat requires gate/up to share a type
   (`gguf_experts.py:78-83`).

## 3. GEOMETRY TABLE — types seen in expert tensors (H=2048, I=512, E=256)

`BLOCK_SHAPE` from `/models/desenvolvimento/freetoken-next/python/freetoken/models/gguf/dequant.py:53-77`.
gate_up slot = `2*I*row_bytes(H,t)` bytes; down slot = `H*row_bytes(I,t)` bytes (one expert, one bank).

| ggml type | blk elems | B/blk | B/elem | row_bytes(H)=rb | row_bytes(I) | gate_up slot | down slot | MMVQ/MoE-vec | grouped MMQ | CPU GEMV | torch dequant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 32 | 34 | 1.0625 | 2176 | 544 | 2,228,224 | 1,114,112 | yes (`dequant.py:128-135`) | yes (`:137-141`) | NO | NO |
| Q6_K | 256 | 210 | 0.8203 | 1680 | 420 | 1,720,320 | 860,160 | yes | yes | yes (`cpu_executor.py:83`) | yes (`:215-218`) |
| Q4_K | 256 | 144 | 0.5625 | 1152 | 288 | 1,179,648 | 589,824 | yes | yes | yes | NO |
| IQ4_XS | 256 | 136 | 0.5312 | 1088 | 272 | 1,114,112 | 557,056 | yes | **no** | NO | NO |
| Q3_K / IQ3_S | 256 | 110 | 0.4297 | 880 | 220 | 901,120 | 450,560 | yes | **no** | NO | NO |

Full combined slot (gate_up+down): Q3_K/IQ3_S 1,351,680 B (1.289 MiB), IQ4_XS-down 1,458,176 B,
Q4_K 1,769,472 B, Q6_K-down 1,761,280 B, Q8_0 3,342,336 B.

SM120 status: the ggml extension is JIT-built with **no arch flags at all**
(`/models/desenvolvimento/freetoken-next/python/freetoken/kernel/gguf.py:63-74` — `load()` gets only
`-O3 --expt-relaxed-constexpr -ccbin ...`), so it compiles for the device torch reports; there is
**NOT PRESENT** any sm_120 allowlist, SASS prebuild entry, or `__CUDA_ARCH__` gate in the MoE path
(`csrc/gguf/moe.cuh`/`moe_vec.cuh`/`mmq.cuh` have none; the only `__CUDA_ARCH__` uses are in
`csrc/gguf/vecdotq.cuh`). All six corpus types are therefore MMVQ-capable on SM120. Two coverage
notes for per-geometry serving: I-quants have no grouped MMQ (`dequant.py:137-141`) and
`ggml_moe_a8` has **no Python caller** (`kernel/gguf.py:100`, `__all__:139`) — decode *and* prefill
both go through `ggml_moe_a8_vec` twice (`/models/desenvolvimento/freetoken-next/python/freetoken/moe/fused_q4_0.py:74,77`),
so MMVQ coverage is the only thing a new pool must satisfy.

## 4. KEYING PROPOSAL

Minimum key for one expert pool:
`(role, ggml_type, row_elems, row_bytes, n_rows_per_slot, dtype=uint8, feat_bytes % 16, device)`
where `row_elems` ∈ {H (gate_up), I (down)}, `row_bytes = row_bytes(row_elems, ggml_type)`
(`dequant.py:146`), `n_rows_per_slot` = 2I or H. Alignment: `feat_bytes = n_rows*row_bytes` must be
%16 for the fused copy (`offload_cache.py:417-418`, `kernel/fast_index_copy.py:175-179`) and %128
for the legacy per-bank path (`offload_cache.py:385-390`); all table entries satisfy both
(Q6_K's `rb(I)=420` is not %16 but the 2048-row slot `860,160` is).

Attach points (in load order):
- `models/config.py:300` — `gguf_expert_types: tuple[int,int]` becomes per-layer
  `dict[role, list[int]]` (already computed by `gguf_expert_types()`, `qwen3_5_moe/gguf_experts.py:36-94`).
- `moe/expert_banks.py:234` — stop resolving to one type; keep the per-layer list.
- `moe/qwen3_5_moe/gguf_experts.py:117-125` — one spec per `(role, type)` instead of per role.
- `moe/offload_cache.py:36-53,135,351-373,505-511` — pool dict keyed by the tuple; `layers/moe.py:444-453`
  and `bank_views()` (`offload_cache.py:610-616`) resolve a layer to its pool.
- `engine/cache_budget.py:17-28` + `moe/expert_banks.py:341-345` — per-slot bytes become the
  byte-weighted mean over geometries (planner already consumes a single int, `plan_cache_budget:98-108`).
- `checkpoint/convert.py:313` / `checkpoint/ftw.py:618` — FTW persists only the `quant_format`
  string; a per-pool ggml type field is **NOT PRESENT** and must be added or converted checkpoints lose it.

Precedent for exactly this split already exists on the resident (non-pooled) side: when a shared
expert's gate and up types differ, `models/qwen3_5_moe/gguf.py:673-682` stops fusing them and emits
`gate_up_proj.qweight_0` / `.qweight_1` instead of one `qweight` — per-geometry tensors, one stride each.
Open design question the key does not answer: whether each geometry pool gets the full
`cache_size` slots (multiplies the pool's VRAM by the number of geometries) or the budget is split
per pool; `validate_rebuild` (`offload_cache.py:463-469`) only enforces `cache_size >= num_experts`
per cache, not per pool.

Quantified benefit of per-geometry pools (bytes a single worst-stride pool would burn; today the
alternative is not "waste" but outright refusal to load):
- Ornith: Δgate_up 278,528 + Δdown 139,264 = **417,792 B/slot (408 KiB, +30.9 % over the Q3_K
  slot)**; 30/40 layers sit at the cheaper stride; 1.59 GiB at 4,096 slots, **3.98 GiB at the
  10,240-slot ceiling** (40 layers × 256 experts; `hi = min(total_experts, max_slots)` at
  `cache_budget.py:98`).
- Tiel: Δdown **303,104 B/slot (296 KiB, +54.4 % on the down bank, +20.8 % of the whole slot)**;
  37/40 layers cheaper; 2.89 GiB at 10,240 slots.

## 5. RISKS a per-geometry pool collides with (all file:line verified)

1. **Fully-packed addressing, no padding slack** —
   `/models/desenvolvimento/freetoken-next/python/freetoken/kernel/csrc/gguf/moe_vec.cuh:24,30`
   (`blocks_per_row = ncols / qk;` / `x = ((const block_q_t*)vx) + expert * nrows * blocks_per_row;`).
   Padding a cheap layer to a wider stride is silently wrong, not merely slow —
   `.../models/qwen3_5_moe/gguf_experts.py:7-13`: "never pad a smaller-type layer up to a larger
   type's stride, because the kernel would then read every block at the wrong offset and return
   plausible-looking garbage" (same warning, `models/config.py:293-297`).
2. **Layer-uniform bank identity** — `set_bank_sources` requires every layer of a bank to equal
   `head` in shape and dtype (`offload_cache.py:354,362-367`) and validates bank names as a *set*
   against the schema (`:327-331`), so two pools with the same role (`gate_up/Q3_K`,
   `gate_up/Q4_K`) collide at that assert and at `canonical_role` (`:322-331`).
3. **CUDA-graph-stable copy descriptors** — the descriptor is built once because "the addresses are
   fixed for the cache's lifetime so the descriptor tensors are CUDA-graph safe"
   (`offload_cache.py:392-399`); `feat_bytes`/`dst_ptrs` are one entry **per bank** (`:435-443`) and
   `copy_missing` indexes `self._copy_src_ptrs[layer_id]` with `layer_id` static per captured node
   (`:1039-1053`). Kernel side, the row move is `src + ps*feat` / `dst + pd*feat` with a single `feat`
   per bank (`/models/desenvolvimento/freetoken-next/python/freetoken/kernel/csrc/jit/fast_index_copy.cuh:476-478,497-510`).
   More pools ⇒ more entries and a per-layer pool lookup inside a captured launch.
4. **`rebuild` reallocates and re-plans** — `offload_cache.py:477-511` frees and re-allocates every
   slot cache then rebuilds the copy plan; with per-geometry pools the slot↔pool binding must be
   re-derived and graphs re-captured (the rebuild path destroys the CUDA graphs *before* calling
   `moe_offload_cache.rebuild` — `engine/engine.py:1081-1082` then `:1092`, audit-time snapshot).
5. **Prefill overlap borrows slots by position, not by pool** — the double buffers view the first
   `2*num_experts` slots of *each* bank as `(2, E, *shape[1:])` (`offload_cache.py:618-628`),
   `_invalidate_prefill_buffer` assumes `slot_start = buffer_id*num_experts` maps 1:1 onto expert
   ids (`:647-655`), and `alphas_for_layer`/`materialize_layer` rely on "position == expert id"
   (`:600-608`, `layers/moe.py:347-380`). A per-geometry split breaks that bijection for any layer
   whose pool is not the buffer's pool.
6. **Slot bookkeeping is bank-agnostic** — `slot_for_id[num_layers, num_experts]` / `id_of_slot[cache_size]`
   in a flat `layer*E+expert` id space, and `evict_slots`/`src_indices` are shared across banks
   (`offload_cache.py:181-202,214-217,1043-1044`). Slots stop being pool-independent.
7. **Budget planner takes one scalar** — `per_expert_bytes` from layer 0 (`cache_budget.py:28`,
   used at `engine/engine.py:703,948`, audit-time); the MoE-first greedy split (`cache_budget.py:98-108`)
   and the runtime fit check both assume a single stride.
8. **CPU/hybrid decode is single-format** — `_resolve_gguf_format` needs one `weight_format` for
   both banks and rejects mixed pairs (`/models/desenvolvimento/freetoken-next/python/freetoken/moe/cpu_executor.py:86-100`,
   `_GGML_TO_CPU_FMT = {2:"q4_0",12:"q4_k",14:"q6_k"}` at `:83`); covered by
   `/models/desenvolvimento/freetoken-next/tests/moe/test_cpu_moe_kquant.py:175-184`. Per-geometry
   GPU pools do not help `--moe-strategy cpu|hybrid`, and Q3_K/IQ3_S/IQ4_XS/Q8_0 have no CPU kernel at all.
9. **No regression test protects the provider reject** — `grep "mixes ggml types" tests/` →
   NOT PRESENT; only the CPU-format mixed-type path is tested. Any per-geometry change removes an
   untested guard rather than a tested one.
10. **Host banks are per-layer page-aligned allocations** — `alloc_layer_banks`
    (`/models/desenvolvimento/freetoken-next/python/freetoken/moe/host_banks.py:207-216`, `HostBank`
    page-alignment `:79-103`) already tolerates per-layer sizes, so the host side is NOT a blocker;
    the GPU pool, the copy descriptor, and the prefill position invariant are.

## 7. REFUSED-TODAY MATRIX (Phase 6 target list)

Predicate run per model: `gguf_architecture` -> `build_gguf_shim` (`models/gguf/config.py:69-104`)
-> the family's `parse_gguf_config(shim)` -> `gguf_expert_types(path, cfg.num_layers)` ->
`gguf_expert_specs(cfg, types)` (`qwen3_5_moe/gguf_experts.py:118-122`), CPU-only, harness
`.qwen/tmp/a7_matrix.py`, output `.qwen/tmp/a7_matrix3.jsonl` (42 shard-set records over
`/models/**/*.gguf`, dev tree + `/models/backup` excluded; 33 of the 42 are llama.cpp source trees
under `/models/servers/**`, i.e. not serving targets).

| model (GGUF set) | arch | cfg parses | gate_up types | down types | terminal verdict |
|---|---|---|---|---|---|
| `/models/Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` | qwen35moe | yes (L=40, E=256, H=2048, I=512, expert_quant=gguf) | Q3_K x30 [5-34], Q4_K x10 [0-4,35-39] | Q3_K x30, Q4_K x10 | **REFUSED** (geometry) |
| `/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` | qwen35moe | yes (same dims) | IQ3_S x40 (uniform) | IQ4_XS x37 [0-33,35-37], Q6_K x3 [34,38,39] | **REFUSED** (geometry) |
| `/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf` | qwen35 | yes (L=64, E=0) | none | none | NO-ROUTED-EXPERTS (dense; no pool exists) |
| `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS/*-0000{1,2,3}-of-00003.gguf` (88 GB) | **qwen4exp** | **no** | n/a | n/a | REFUSED-EARLIER (arch mapping) |
| `/models/Qwen3.8-Flash-Next-AD-4.27/…Q4_K_M-M64/*-000NN-of-00033.gguf` (89 GB, 33/33 present) | **qwen4exp** | **no** | n/a | n/a | REFUSED-EARLIER (arch mapping) |
| `…/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/MTP/mtp-Qwen3.8-Flash-Next-{Q4_K_M,shared-Q4_K_M,shared-Q8_0}.gguf` | **qwen4exp** | **no** | n/a | n/a | REFUSED-EARLIER (arch mapping) |
| `/models/Qwen3.6-35B-A3B-mmproj.gguf`, `/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mmproj.gguf` | clip | n/a | n/a | n/a | not an engine checkpoint (vision projector) |
| `/models/dflash-draft-Ornith15.gguf`, `/models/qwen36-35b-a3b-dflash-Q4_K_M.gguf` | dflash | n/a | n/a | n/a | REFUSED-EARLIER (arch mapping; draft model) |
| `/models/servers/llama-turbo-optimal/build/tests/test-models/{qwen3moe-moe,deepseek4-moe}.gguf` | qwen3moe / deepseek4 | yes | F32 uniform | F32 uniform | PAST-THE-GUARD (fixtures; die next at the kernel, note 4) |
| `/models/servers/llama-turbo-optimal/build/tests/test-models/qwen35moe-moe.gguf` | qwen35moe | **no** | — | — | REFUSED-EARLIER: `missing required key qwen35moe.full_attention_interval` |

Verbatim geometry refusals (what a user sees today; the same message is re-raised with layer names
by `moe/expert_banks.py:218-226` before this point in a real load):

```
Ornith: ValueError: expert bank 'gate_up' mixes ggml types across layers ([11, 12]); a bank must be uniform because its slot pool is one allocation with one stride
Tiel:   ValueError: expert bank 'down'   mixes ggml types across layers ([14, 23]); a bank must be uniform because its slot pool is one allocation with one stride
```

Notes (Phase 6 scope):
1. **Only two checkpoints in the whole corpus are blocked by geometry alone** — Ornith and Tiel, both
   `qwen35moe`, both otherwise fully parseable. That is the unblock list for a geometry-keyed pool.
2. The Qwen3.8-Flash-Next sets (177 GB, 2 real MoE checkpoints + 3 MTP adapters) die earlier at
   `GGUF_ARCH_TO_REGISTRY` (`models/gguf/config.py:20-27`): `ValueError: GGUF architecture 'qwen4exp'
   is not supported (known: ['deepseek4', 'gemma4', 'qwen35', 'qwen35moe', 'qwen3moe'])`. They need a
   model package + `parse_gguf_config` + `iter_gguf_weights` + `gguf_experts` (registry entries at
   `models/register.py:270-284`), i.e. an arch-mapping project, not a pool change.
3. `load_gguf_expert_sources` exists for all three registered MoE families (`qwen3_moe`,
   `qwen3_5_moe`, `deepseek_v4`, each `gguf_experts.py:125/129/126`) — no registered arch is missing
   a row reader. Unmapped qwen arch strings seen in the corpus: qwen4exp, qwen, qwen2, qwen2moe,
   qwen2vl, qwen3, qwen3next, qwen3tts, qwen3vl, qwen3vlmoe.
4. PAST-THE-GUARD is not servable: those two fixtures pass with `row_bytes(F32)` banks, and F32 is not
   in `MOE_VEC_TYPES` (`models/gguf/dequant.py:128-135`), so `fused_experts_gguf` rejects them at
   `moe/fused_q4_0.py:54-60`.
5. MTP/NextN is a separate blocker: the served model is `block_count - nextn` deep and NextN weights
   are dropped with "GGUF paths do not do speculative decoding"
   (`models/qwen3_5_moe/gguf.py:548-556`). `per_layer_token_embd` is **NOT PRESENT** anywhere in this
   tree (`grep -rn per_layer_token_embd python/freetoken` -> empty).
6. Shard sets were validated by the reader itself (`reader.py:53-113` completeness check, `:296-317`
   declared-vs-found tensor counts); both Qwen3.8 sets are complete (3/3 and 33/33), so their refusal
   is not a truncated-download artifact. And per correction 1: a mixed bank **fails closed** — there is
   no silent mis-decode and no padding tax in the current code, because the geometry is never accepted.

## ERRATA vs. the reviewer's three corrections (verified against HEAD 86af2d3)

1. Agreed and already stated: mixed banks **fail closed** (`gguf_experts.py:118-122` ValueError ->
   `expert_banks.py:218-226` NotImplementedError). This dossier never claimed silent corruption or a
   padding tax; §2 says "mixed banks are **rejected, never coerced or padded**" and §4 says "today the
   alternative is not waste but outright refusal to load". The §4 byte figures are the *counterfactual*
   cost of the naive pad-to-worst fix, which §5.1 records as silently wrong. Phase 6's benefit is
   therefore "loads at all", with the byte figures as the ceiling on what any single-pool design would
   have to pay.
2. Agreed and already stated: `expert_bytes_per_slot` (`engine/cache_budget.py:17-28`) is correct —
   a source is `[E, 2I, rb]` uint8 and a slot is one (layer, expert) row, with
   `total_experts = num_moe_layers * num_experts` (`engine/engine.py:696`; the reviewer's "a slot is
   one (layer, expert) pair" is the right reading of `cache_budget.py:28`, whose `t[0][0]` indexes
   layer 0 / expert 0 of a `[E, 2I, rb]` source). §2 point 3 only
   observes that the stride is read off layer 0, which is exactly why a mixed bank is a problem; no fix proposed.
3. Not reproducible in this tree. `/models/desenvolvimento/freetoken-next/python/freetoken/models/gguf/dequant.py`
   is 279 lines (no :354-364); `grep -rn "turbo|qjl|Q8_3|276"` over `models/gguf/` and
   `kernel/csrc/gguf/` returns nothing; `BLOCK_SHAPE` (`:53-77`) is documented as derived from
   `kernel/csrc/gguf/ggml-common.h:18-192` and gives Q3_K=(256,110), IQ4_XS=(256,136).
   Physical confirmation from the Ornith tensor table: `blk.5.ffn_down_exps.weight` (Q3_K,
   ne=[512,2048,256]) is 115,343,360 B = rows x `row_bytes(512,Q3_K)`=220 exactly, and
   `blk.0` (Q4_K) 150,994,944 B = rows x 288 exactly; the whole-file tensor sum is
   17,426,870,784 B against a 17,437,861,408 B file (10.5 MiB header+alignment slack). So §3's numbers
   are the file's real strides on this branch; a turbo/block_q8_3_turbo redefinition would be a
   different tree and would invalidate §3/§4 wholesale.

## 8. STRIDE-vs-FILE TRUTH TABLE

`actual B/row` = `(next tensor's data_offset - this tensor's data_offset) // rows`, rows = product of
every ggml dim above the fastest; taken from the tensor-info offsets only, never from a type-size
table (`/models/desenvolvimento/freetoken-next/.qwen/tmp/a7_truth.py` →
`.qwen/tmp/a7_truth.txt`, 1194 tensors over 11 files/shard-sets; alignment 32, measured pad = 0 on all).
`llama.cpp-derived` comes from the fork's own block structs (`/models/servers/llama-turbo-optimal/ggml/src/ggml-common.h`:
`QK_K=256` :89, `K_SCALE_SIZE=12` :90, `IQ3S_N_SCALE=QK_K/64` :543, asserts at :438-589), independent
of FreeToken's `BLOCK_SHAPE`. `n` = tensors with that (model, role, type, elems).

| model | role | id | repo `GGML_NAME` | llama.cpp name for that id | elems/row | actual B/row | repo `row_bytes` | llama.cpp-derived | n | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Ornith (qwen35moe) | blk.%d.ffn_gate_exps / up_exps | 12 | Q4_K | `GGML_TYPE_Q4_K = 12` | 2048 | 1152 | 1152 | 8×144 = 1152 | 10+10 | MATCH |
| Ornith | blk.%d.ffn_gate_exps / up_exps | 11 | Q3_K | `Q3_K = 11` | 2048 | 880 | 880 | 8×110 = 880 | 30+30 | MATCH |
| Ornith | blk.%d.ffn_gate_exps / up_exps | 8 | Q8_0 | `Q8_0 = 8` | 2048 | 2176 | 2176 | 64×34 = 2176 | 1+1 (blk.40) | MATCH |
| Ornith | blk.%d.ffn_down_exps | 12 / 11 / 8 | Q4_K / Q3_K / Q8_0 | same ids | 512 | 288 / 220 / 544 | 288 / 220 / 544 | 2×144 / 2×110 / 16×34 | 10/30/1 | MATCH |
| Ornith | blk.%d.ffn_down_shexp | 14 / 8 | Q6_K / Q8_0 | `Q6_K = 14` | 512 | 420 / 544 | 420 / 544 | 2×210 / 16×34 | 40 / 1 | MATCH |
| Tiel (qwen35moe) | blk.%d.ffn_gate_exps = up_exps | 21 / 8 | IQ3_S / Q8_0 | `IQ3_S = 21` | 2048 | 880 / 2176 | 880 / 2176 | 8×110 / 64×34 | 40+40 / 1+1 | MATCH |
| Tiel | blk.%d.ffn_down_exps | 23 / 14 / 8 | IQ4_XS / Q6_K / Q8_0 | `IQ4_XS = 23`, `Q6_K = 14` | 512 | 272 / 420 / 544 | 272 / 420 / 544 | 2×136 / 2×210 / 16×34 | 37/3/1 | MATCH |
| Tiel | blk.%d.ffn_{gate,up,down}_shexp | 8 | Q8_0 | `Q8_0 = 8` | 2048/512 | 2176 / 544 | 2176 / 544 | 64×34 / 16×34 | 41×3 | MATCH |
| Qwen3.8-Flash-Next UD-IQ4_XS (qwen4exp, 48 L, E=512, H=2560, I=640) | blk.%d.ffn_gate_exps / up_exps | 21 / 23 | IQ3_S / IQ4_XS | same ids | 2560 | 1100 / 1360 | 1100 / 1360 | 10×110 / 10×136 | 47+47 / 1+1 | MATCH |
| …UD-IQ4_XS | blk.%d.ffn_down_exps | 20 / 8 | IQ4_NL / Q8_0 | `IQ4_NL = 20` | 640 | 360 / 680 | 360 / 680 | 20×18 / 20×34 | 43/5 | MATCH |
| …UD-IQ4_XS | blk.%d.ffn_down_shexp | 8 | Q8_0 | — | 640 | 680 | 680 | 20×34 | 48 | MATCH |
| Qwen3.8-Flash-Next AD-4.27 Q4_K_M (qwen4exp) | blk.%d.ffn_gate_exps / up_exps | 22 / 21 | IQ2_S / IQ3_S | `IQ2_S = 22` | 2560 | 820 / 1100 | 820 / 1100 | 10×82 / 10×110 | 36+36 / 12+12 | MATCH |
| …AD-4.27 | blk.%d.ffn_down_exps / _shexp | 20 / 8 | IQ4_NL / Q8_0 | — | 640 | 360 / 680 | 360 / 680 | 20×18 / 20×34 | 48/48 | MATCH |
| mtp-Qwen3.8-Flash-Next-shared-Q8_0 | ffn_{gate,up,down}_exps, down_shexp, hc_ffn_up | 8 | Q8_0 | `Q8_0 = 8` | 2560/640/320 | 2720/680/340 | identical | 80×34, 20×34, 10×34 | 5 | MATCH |
| mtp-Qwen3.8-Flash-Next-{Q4_K_M, shared-Q4_K_M} | ffn_gate_exps / up_exps ; ffn_down_exps / down_shexp ; hc_ffn_up | 12 ; 8 ; 6 | Q4_K ; Q8_0 ; Q5_0 | `Q4_K = 12`, `Q5_0 = 6` | 2560/640/320 | 1440 ; 680 ; 220 | identical | 10×144 ; 20×34 ; 10×22 | 4 ; 4 ; 2 | MATCH |
| Qwen3.8-27B-GSQ (qwen35, dense `blk.%d.ffn_{gate,up,down}`) | 23 (role,type) combos; ids 10 Q2_K, 12 Q4_K, 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 29 IQ1_M | — | every name = its own id | — | 5120 (gate/up), 17408 (down) | 1120-2880 (gate/up), 5032-9792 (down) | identical to actual | gate/up: 20×{56,66,74,84,98,110,136,144}; down: 68×{74,84,98,110,136,144} | 195 | MATCH |
| qwen36-35b-a3b-dflash-Q4_K_M (dflash) | blk.%d.ffn_{gate,up} / ffn_down | 12 / 14 | Q4_K / Q6_K | — | 2048 / 6144 | 1152 / 3456 / 5040 | identical | 8×144 / 24×144 / 24×210 | 18 | MATCH |
| dflash-draft-Ornith15 (dflash) | blk.%d.ffn_* , attn_output | 30 | BF16 | `BF16 = 30` | 2048/6144/4096 | 4096/12288/8192 | identical | 2 B/elem | 24 | MATCH |
| Qwen3.6-35B-A3B-mmproj (clip) | v.blk.%d.ffn_{up,down} | 1 | F16 | `F16 = 1` | 1152/4304 | 2304/8608 | identical | 2 B/elem | 54 | MATCH |

Verdict (no exceptions found):
1. **The enum is NOT renumbered.** `/models/desenvolvimento/freetoken-next/python/freetoken/models/gguf/dequant.py:28-49`
   (`Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, IQ3_S=21, IQ4_XS=23, IQ1_M=29, BF16=30`) is
   value-identical (id by id) to `/models/servers/llama-turbo-optimal/ggml/include/ggml.h:400-404,411-413`
   (`Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14`; `Q8_K = 15`, `IQ3_S = 21`, `IQ4_XS = 23`).
   The premise "llama.cpp has Q4_K=10, Q5_K=11, Q6_K=12" is false in both this fork and upstream — ids
   10/11 are Q2_K/Q3_K.
2. **Zero mis-decode candidates: 1194/1194 tensors MATCH**, pad=0, and the repo's `row_bytes`, the
   installed gguf-py `GGML_QUANT_SIZES`, and the fork's `static_assert` struct arithmetic all agree
   with the bytes the writer actually laid down. There is no (model, role) where the file's stride
   matches a *different* name's row size, so no bank is being fed to the wrong kernel and back.
3. Q3_K is 110 B per 256 elements, confirmed physically (512-elem row = 220 B, not 276); `276` appears
   nowhere in the corpus, and `276 = 16 × 17.25` is not realisable with integer GGUF block sizes.
   That fork's turbo codecs are ids 43-51 (`ggml.h:433-443`, `block_turbo3_0` = 14 B / `QK_TURBO3`=32,
   `ggml-common.h:315`) — a KV-cache codec family, not a redefined Q3_K — and no weight tensor in any
   of these files uses an id ≥ 43 (max id seen is 30). `block_q8_3_turbo` is NOT PRESENT in that tree.
4. **Phase 6 therefore starts by keying pools, not by correcting the type table.** §3's geometry table
   and §4's stride/waste arithmetic stand as written. The same pass shows the two `qwen4exp` sets are
   themselves geometry-mixed (UD-IQ4_XS: gate_up 47 IQ3_S + 1 IQ4_XS, down 43 IQ4_NL + 5 Q8_0;
   AD-4.27: gate_up 36 IQ2_S + 12 IQ3_S, down uniform IQ4_NL), so once their arch lands they hit this
   same guard — 192 expert tensors each at E=512, H=2560, I=640, the largest future win.
5. The `per_layer_token_embd` the reviewer expected is a *checkpoint* role, not a FreeToken symbol:
   `per_layer_token_embd.weight` (IQ4_NL, `ne=[160, 320001536]`) plus ~190 `blk.N.hc_ffn_*`
   hyper-connection tensors exist in the `qwen4exp` shards and have no reader here — consistent with
   §7 notes 2-3 (arch mapping first, pool keying second for those two sets).


