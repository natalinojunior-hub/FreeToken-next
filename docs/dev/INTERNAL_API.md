# INTERNAL_API — APIs Internas Críticas freetoken-next

**Última atualização:** 2026-09-19 | **Versão:** v0.1.3 (`cac247a`)
**Uso:** Referência para debug, instrumentação e mudanças arquiteturais

---

## SchedulerSpecMixin (scheduler/spec.py)

### `run_spec_step() -> bool`
**Linha:** ~285  
**Descrição:** Executa um passo de decodificação especulativa MTP para request elegível.  
**Retorna:** `True` se rodou spec (caller deve pular `_schedule_next_batch`/`_forward`), `False` caso contrário.

**Fluxo:**
1. `_spec_eligible_req()` → request único greedy com `remain_len > 1`
2. `_snapshot_qsa_state(req, d, k)` — snapshot ring/scratch/cmp antes do draft
3. **Draft chain:** `k` passos autoregressivos via draft head MTP (linha 317-353)
4. **Verify batch:** Forward único com `spec_logits_indices` para `k+1` posições (linha 355-380)
5. `accept_drafts(sampled, drafts)` → tokens aceitos (linha 374)
6. `_commit_spec_tokens(req, accepted, start_pos=d, spec_alloc_len=d+k)` — commit com rollback/replay se rejeição parcial

**Instrumentação (EXP-048):**
```bash
FREETOKEN_DEBUG_EXP048=1 python benchmarks/bench_pp_tg.py ...
# Logs: [exp048-step], [exp048-verify], [exp048-rollback], [exp048-replay]
FREETOKEN_DEBUG_SPEC_TOP2=1 python ...
# Logs: [spec-draft-top2], [spec-verify-top2] com top1/top2 logits
```

### `_commit_spec_tokens(req, tokens, start_pos, spec_alloc_len) -> int`
**Linha:** 137  
**Descrição:** Commita tokens aceitos aplicando EOS/stop/length finish logic por token.  
**Retorna:** Quantos tokens de fato contaram (para em finish reason).

**Detalhe crítico:** Avança `cached_len`/`device_len` UM token por iteração (igual a `complete_one()`), pré-seta fim da janela inteira para `can_decode` refletir corretamente.

### `_snapshot_qsa_state(req, d, k) -> None`
**Linha:** 77  
**Descrição:** Salva estado QSA (pending_ring, cmp_scratch, cmp_k_buffer rows) antes do draft chain mutar.  
**Parâmetros:** `d` (posição atual), `k` (profundidade MTP) — usado para identificar `closing_rows` via `page_table`.

### `_restore_qsa_state(req) -> None`
**Linha:** 114  
**Descrição:** Restaura snapshot QSA em caso de rejeição parcial (rollback).  
**Inclui:** `cmp_k_buffer[cmp_rows]` se salvo.

### `_spec_eligible_reqs() -> list[Req]`
**Linha:** 48  
**Descrição:** Retorna TODOS requests running elegíveis (greedy, `remain_len > 1`). Remove trava `len(running) != 1`.

### `_spec_eligible_req() -> Req | None`
**Linha:** 62  
**Wrapper:** Retorna primeiro de `_spec_eligible_reqs()` ou `None`.

### `clear_mtp_slot() -> None` (QSAKVCache method)
**Arquivo:** `kvcache/qsa_pool.py`  
**Descrição:** Zera **todos** tiers do slab MTP (camada 48) ao reciclar request: `cmp_k`, `pending_ring`, `_k_codes`, `_k_norm`, `_v_codes`, `_v_norm`, BF16 `_kv_buffer`.  
**Chamado em:** `QSAKVCache.free_req()` → garante determinismo cross-request (EXP-046).

---

## DetokenizeManager (tokenizer/detokenize.py)

### `detokenize(msgs: List[DetokenizeMsg]) -> List[str]`
**Linha:** 91  
**Descrição:** Streaming detokenização incremental com offsets progressivos por UID.  
**Fix EXP-048 (linha 91-97):** Se lote contém múltiplas msgs mesmo `uid`, processa **sequencialmente** para manter invariantes de `read_offset`/`surr_offset`.

**DetokenizeMsg campos:**
- `uid`: Request identifier
- `next_token`: Token int
- `finished`: bool
- `finish_reason`: "stop" | "length" | None
- `matched_stop`: matched stop string ou None
- `stop_strs`: Lista de stop strings ou None

**DecodeStatus por UID:**
- `decoded_ids`: Lista completa de tokens emitidos
- `decoded_str`: Texto já emitido (flushado)
- `read_offset`: Índice até onde `batch_decode` leu
- `surr_offset`: Índice do prefixo "surrogate" para recálculo
- `sent_offset`: Quantos chars já enviados ao cliente

---

## VRAM Ledger (engine/vram_ledger.py)

### `VRAMAccount`
**Single source of truth** para: ceiling, reserve, overhead, expert/KV split, context rows.

### `allocate_kv_pages(num_pages) -> int`
Aloca páginas KV, atualiza account. Retorna páginas alocadas.

### `allocate_expert_pages(bank, role, type) -> int`
Aloca páginas para expert bank com geometria `(bank, role, type)`.

### Reservas One-Time (Otimizadas LTO)
| Reserva | Original | Atual | Ganho |
|---------|----------|-------|-------|
| `TRITON_AUTOTUNE_ARENA` | 256 MiB | 128 MiB | 128 MiB |
| `GRAPH_CAPTURE_PEAK` | 256 MiB | 128 MiB | 128 MiB |
| `FRAGMENTATION_RESERVE` | 128 MiB | 64 MiB | 64 MiB |

---

## MoE Offload Cache (moe/offload_cache.py)

### `OffloadCache`
- `get_expert(bank, role, type) -> ExpertBank` — busca/fetch assíncrono
- `put_expert(bank, role, type, bank_data)` — eviction LRU
- `prefetch_next(req)` — double-buffer prefill

### `ExpertBank`
- `weight: Tensor` — pesos quantizados (host ou device)
- `quant_type: str` — Q3_K, Q4_K, IQ3_S, etc.
- `shape: Tuple` — geometria exata

### Parâmetros de Tuning (env vars)
```bash
FREETOKEN_MOE_SMALL_BANK_FEAT_BYTES=65536    # 64 KiB (era 256 KiB)
FREETOKEN_HYBRID_MAX_FETCH=4                  # paralelos H2D
FREETOKEN_HYBRID_FETCH_FRACTION=0.5           # overlap fetch/compute
```

---

## QSAKVCache (kvcache/qsa_pool.py)

### `QSAKVCache`
- `pending_ring: Tensor[layers, pages]` — códigos comprimidos pendentes
- `cmp_k_buffer: Tensor[layers, cmp_rows]` — buffer scratch compressão
- `_k_codes, _k_norm, _v_codes, _v_norm, _kv_buffer` — tiers por layer

### `free_req(req) -> None`
**Chamado em:** `_free_req_resources` (caminho compartilhado abort/decode/spec)  
**Ações:** 
1. Libera páginas KV via `page_table`
2. **`clear_mtp_slot()`** — zera slab MTP (camada 48) todos tiers
3. Libera slot `linear_state_pool` (GDN snapshot)

### `cmp_k_buffer` layout
- `[0:cmp_scratch_base]` — base comprimida (persistente por layer)
- `[cmp_scratch_base + table_idx]` — scratch por request (temporário)

---

## LinearStatePool (engine/engine.py)

### `LinearStatePool`
- `conv_states: Tensor[layers, max_slots, ...]` — estados convolucionais GDN
- `recurrent_states: Tensor[layers, max_slots, ...]` — estados recorrentes GDN

### `_spec_snapshot_slots: Dict[uid, int]`
Slot reservado por request para snapshot GDN durante spec.

### `free_spec_snapshot_slot(req) -> None`
Libera slot do pool linear ao finalizar request (spec ou não).

---

## Engine Forward (engine/engine.py)

### `_forward_decode(batch) -> ForwardOutput`
**Linha:** ~1320  
**Fluxo MTP:**
1. `batch.spec_logits_indices` não-None → spec step
2. Loop `req.complete_one()` para cada request no batch (linha 1347)
3. `sampler.sample(batch_logits)` → `next_tokens_gpu`
4. Retorna `ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)`

### `graph_runner.replay(batch)`
Replay CUDA graph para decode (quando `use_graph=True`).

---

## Sampler (engine/sampler.py)

### `sample(logits, args) -> Tensor[int32]`
Greedy (temp=0) ou sampling (top-k, top-p, temperature).  
Retorna `next_tokens` shape `[batch_size]` int32.

---

## CacheManager (scheduler/cache.py)

### `allocate_paged(req, num_tokens, skip_alloc=False) -> int`
Aloca páginas para request.  
**`skip_alloc=True`** em replay paths evita page leak (não-idempotente).

### `check_integrity() -> bool`
Valida consistência page_table, ref_counts, free_list.  
**Crash se False:** `CacheManager integrity check failed`

---

## Model Registry (models/registry.py)

### `load_model(path) -> Model`
Carrega checkpoint (FTW/GGUF/HF), resolve arquitetura, retorna modelo com config.

### `ModelConfig`
- `num_layers`, `num_experts`, `num_active_experts`
- `hidden_size`, `num_heads`, `head_dim`
- `moe_geometry`: Dict[layer, (bank, role, type)]
- `quantization`: por tensor

---

## Instrumentação / Debug Flags

| Env Var | Efeito | Arquivo |
|---------|--------|---------|
| `FREETOKEN_DEBUG_EXP048=1` | Logs passo-a-passo EXP-048 | `spec.py` |
| `FREETOKEN_DEBUG_SPEC_TOP2=1` | Logits top-2 draft/verify | `spec.py` |
| `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1` | Desabilita overlap scheduling | `engine.py` |
| `TORCH_COMPILE_DEBUG=1` | Debug torch.compile | - |

---

## Padrões de Chamada Críticos

```python
# Spec step completo (scheduler/spec.py)
if self.run_spec_step():
    return  # Skip normal schedule/forward

# Commit tokens com finish logic (scheduler/spec.py)
committed = self._commit_spec_tokens(req, accepted, start_pos=d, spec_alloc_len=d+k)

# Detokenização streaming (tokenizer/detokenize.py)
outputs = detokenize_manager.detokenize([
    DetokenizeMsg(uid=req.uid, next_token=t, finished=fin, ...)
    for t in committed_tokens
])

# VRAM allocation (engine/vram_ledger.py)
kv_pages = vram_ledger.allocate_kv_pages(num_pages)
expert_pages = vram_ledger.allocate_expert_pages(bank, role, type)

# MoE fetch (moe/offload_cache.py)
expert_bank = offload_cache.get_expert(bank, role, type)
```
