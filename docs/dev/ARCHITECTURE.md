# ARCHITECTURE — freetoken-next

**Design atual, subsistemas, seams, file:line references.** Source: `old/docs/freetoken-next/ARCHITECTURE.md` (416 linhas) + Audits A1-A9.

---

## 1. Mapa de Subsistemas Herdados (upstream v0.1.3 `cac247a`)

```
python/freetoken/
  server/          OpenAI/Anthropic/Responses HTTP, streaming, tool parsers
  scheduler/       Chunked prefill (PrefillAdder, max_extend_tokens=8192), cache manager, commit/window locking, overlap_loop
  kvcache/         Paged KV pools, radix prefix caches (radix/), dummy-page brick
  moe/             OffloadMoELayer, expert banks (CPU/GPU/hybrid), quantized experts (fp8/nvfp4/mxfp4/q4_0)
  models/          Registry + per-arch loaders (qwen4_exp, glm*, deepseek_v4, gguf)
  kernel/          CUDA/Triton kernels, JIT cache, C++ extensions (csrc/)
  layers/attention/ Fused ops, attention backends (QSA, TurboKV, dense)
  engine/          Cache budget planning, config resolution, VRAM ledger
  checkpoint/      HF → FTW fast-load conversion
```

---

## 2. Seams Críticos (File:Line)

| Seam | Arquivo:Linha | Descrição |
|------|---------------|-----------|
| VRAM Ledger | `engine/vram_ledger.py` | Single source: ceiling, reserve, overhead, expert/KV split, context rows |
| Cache Budget | `engine/cache_budget.py:557` | `plan_cache_budget` + `_resolve_auto_moe_cache_size` |
| TurboKV/QSA Split | `kvcache/qsa_pool.py` | Delegates full-attention to `TurboMHAKVCache` |
| TurboKV Decompress | `kernel/triton/qsa/decompress.py` | Separate kernel before dense QSA attention |
| MoE Offload Cache | `moe/offload_cache.py:103-147` | `OffloadMoeCache` dataclass, LRU, H2D async, double-buffer |
| Expert Banks | `moe/expert_banks.py:275` | `load_expert_banks` → HostBank pinned + GPU slot cache |
| MTP Spec Loop | `scheduler/spec.py` | Draft+verify as `Batch(phase="prefill")` per token |
| MTP Model | `models/qwen4_exp/model.py` | `Qwen4ExpForCausalLM.forward` carry shift |
| GGUF Loader | `models/gguf/reader.py:123` | `GGUFReader` + shard list; `config.py:20` arch registry |
| Chunked Prefill | `scheduler/prefill_adder.py`, `scheduler/cache.py:260` | `allocate_paged` |
| Radix Cache | `kvcache/radix/radix_cache.py` | `RadixPrefixCache`, `SWARadixCache`, `HybridRadixCache` |

---

## 3. TurboKV/QSA Split Architecture (Pillar 1 + 2)

```
┌─────────────────────────────────────────────────────────────┐
│                    QSA Sparse Attention                     │
│  (existing dense attention kernel on FP16 tiles)            │
└─────────────────────────┬───────────────────────────────────┘
                          │ FP16 tiles (bounded workspace)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Turbo4 Decompression Kernel                    │
│  (in-SRAM FP4 dequant: block_n=32, num_stages=1)            │
│  Input: compressed slabs (kn_ptr, vn_ptr, cent_ptr, etc.)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ Compressed KV (4-bit)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              TurboMHAKVCache (via qsa_pool.py)              │
│  Page allocation + index reconstruction                     │
└─────────────────────────────────────────────────────────────┘
```

**Por que split?** Fused dequant+attention em 1 kernel → ptxas host-RAM 70 GiB → OOM. Split evita isso.

---

## 4. VRAM Ledger Flow

```
Engine Startup
    │
    ▼
_sync_get_memory() → baseline_free (post-weights)
    │
    ▼
VramLedger.decide() → MemoryPlan
    ├── expert_cache_bytes (from MoE budget)
    ├── kv_cache_bytes (remaining after reserve)
    └── context_rows (priced via ContextDemand)
    │
    ▼
Measured peak → ratchets reserve floor for next plan
```

---

## 5. MTP Speculative Pipeline

```
Prefill (prompt)
    │
    ▼
MTP Warm-up (carry shift: h_{t-1} + x_t → draft x_{t+1})  ← EXP-043 fix
    │
    ▼
Spec Loop (per decode step):
    ├── Draft forward (k tokens) → draft logits
    ├── Target verify (parallel) → accept/reject
    ├── Accepted tokens → append KV, update carry
    └── Rejected → QSA snapshot/restore (EXP-039)
```

---

## 6. GGUF Native Loader

- **Zero load-time dequant:** weights stay packed uint8 rows (`GGUFLinear.row_bytes()`)
- **Dispatch:** M≤6 → mmvq, else mmq, else dequantize (vendored llama.cpp kernels)
- **Mixed quant support:** Q4_0, Q8_0, Q6_K, IQ4_XS via `BLOCK_SHAPE` table
- **MoE Geometry:** Expert banks keyed by (bank, role, type) — Phase 7 unblock

---

## 7. Referências Cruzadas

| Documento | Conteúdo |
|-----------|----------|
| `CONTEXT.md` | Visão geral + próximos passos |
| `DECISIONS.md` | D-001 a D-023 rationale |
| `PERFORMANCE.md` | Métricas medidas |
| `EXPERIMENTS.md` | EXP-000 a EXP-045 setup/result/verdict |
| `old/docs/freetoken-next/audits/A1-A9.md` | Auditorias fonte detalhadas |
| `LESSONS.md` | Sintoma→causa→fix patterns |

---

## 8. Próximas Mudanças Arquiteturais (Roadmap)

1. **Phase 7:** Expert pool keyed by (bank, role, type) — `moe/offload_cache.py`
2. **Phase 10:** Pillar 2 Zero-replay GDN — `scheduler/spec.py` + `models/qwen4_exp/`
3. **Phase 12:** TCQ/VBR policy — per-(layer,side) tier schedule
4. **Phase 13:** PLE tiered/paged RAM KV — bounded-read sparse/windowed