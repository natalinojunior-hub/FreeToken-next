# ERRORS — Catálogo Estruturado de Erros freetoken-next

**Formato:** `CÓDIGO | SINTOMA | CAUSA RAIZ | FIX | PREVENÇÃO | ARQUIVO`
**Ordenado por:** Componente → Frequência → Severidade
**Última atualização:** 2026-09-19

---

## MoE / Expert Offload

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **MOE-001** | `earlyoom killed python` ~69GB RAM | tmpfs `/tmp` = RAM-backed; Flash-Next precisa ~63GiB host | `export TMPDIR=/models/desenvolvimento/tmp`; `sudo systemctl mask tmp.mount` | SEMPRE exportar TMPDIR antes de serve/checkpoint | `moe/offload_cache.py` |
| **MOE-002** | `--moe-cache-auto` ignora `--num-tokens` | Auto-split KV/experts roda antes de parse de `--num-tokens` | `--num-tokens` vira piso obrigatório | Documentar em RUNBOOKS.md #1 | `moe/offload_cache.py` |
| **MOE-003** | `KeyError: expert_geometry` / geometria inválida | Pool keyed apenas por `(bank)` não `(bank, role, type)` | Phase 7: keying por geometria exata | Validador em `expert_banks.py` | `moe/expert_banks.py` |
| **MOE-004** | TG 0.47 tok/s (MoE quebrado) | Expert banks não carregados / quantização errada | Verificar `--moe-strategy offload` e `--moe-cache-auto` | Benchmark TG ≥ 25 tok/s (EXP-045 anchor) | `moe/offload_cache.py` |

---

## KV Cache / Paging / Radix

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **KVC-001** | `CacheManager integrity check failed` | `allocate_paged` não-idempotente; replay rebobina e realoca | `_prepare_batch(..., skip_alloc=True)` em replay paths | Testar `check_integrity()` original falhando antes | `scheduler/cache.py` |
| **KVC-002** | Page leak: VRAM cresce sem bound | `_spec_eligible_req` desliga spec no último token, cleanup só no branch spec | Mover `free_spec_snapshot_slot` para `_free_req_resources` | Suite completa scheduler após fix cache/paginação | `scheduler/spec.py` |
| **KVC-003** | Non-determinismo requests sequenciais | Slab KV do draft slot MTP (camada 48) sujo por páginas pré-alocadas | `clear_mtp_slot()` em `QSAKVCache.free_req` (todos tiers) | Validar 4 reqs sequenciais bit-identical sha1 | `kvcache/qsa_pool.py` |
| **KVC-004** | VRAM ledger over-modelled | Reservas estáticas one-time aprisionam ~320 MiB headroom | Reduzir `TRITON_AUTOTUNE_ARENA`, `GRAPH_CAPTURE_PEAK`, `FRAGMENTATION_RESERVE` | Auditoria LTO cruzada | `engine/vram_ledger.py` |

---

## MTP / Speculative Decode

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **MTP-001** | Accept-rate ~0% primeiros tokens | Draft head KV nunca populado sobre prompt original (prefill-window warm-up ausente) | Implementar prefill-window MTP warm-up (EXP-025 gap) | Gate EXP-025 em ROADMAP | `scheduler/spec.py` |
| **MTP-002** | sha1 diverge k≥1 vs k=0 (EXP-048) | `DetokenizeManager.detokenize()` acumula tokens antes de atualizar offsets para múltiplos msgs mesmo UID | Processar sequencialmente quando `len(msgs) > len({m.uid})` | Teste tokenizer/detokenize batch | `tokenizer/detokenize.py:91` |
| **MTP-003** | k=2/k=3 divergem do baseline k=0 | Aceitação baixa (3.3% full 2/2) + carry shift não validado live | Instrumentar accept-rate por step; prefill-window warm-up | Gate content equivalence k=2/k=3 em QA.md | `scheduler/spec.py` |
| **MTP-004** | `AttributeError: _mtp_slot` | Tentativa de snapshot/restore QSA restrito à camada MTP via slot inexistente | `_snapshot_qsa_state`/`_restore_qsa_state` todas camadas ou via `_mtp_slot` válido | Bug real de escopo, mas não causa EXP-048 | `scheduler/spec.py:74` |

---

## Tokenizer / Detokenização

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **TOK-001** | Texto duplicado `service--0` vs `service-0` | Batch detokenização múltiplos tokens mesmo UID: offsets calculados após append de todos | Detectar repetição UID no lote; processar sequencialmente | `pytest tests/tokenizer/` + teste batch multi-UID | `tokenizer/detokenize.py:91` |
| **TOK-002** | Streaming repete/gagueja | `DetokenizeManager` assumia 1 msg/uid/lote | Mesmo fix TOK-001 | - | `tokenizer/detokenize.py` |

---

## CUDA / Kernel / JIT / Compilação

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **CUDA-001** | `ptxas fatal: Out of memory in register allocation` | Kernel fundido Turbo4 dequant + QSA attention explode registradores | Split kernel: decompressão separada (`decompress.py`) + attention denso | `block_n=32`, `num_stages=1` kernels comprimidos | `kernel/triton/qsa/decompress.py` |
| **CUDA-002** | `Triton fp4_quantization_120f` não compila | `quantization.cu:488` alignment error nvcc 13.3 | Test skip isolado; não bloqueia suite principal | `pytest.mark.skipif` env flag | `tests/kernel/test_fp4.py` |
| **CUDA-003** | SegFault import `freetoken` | Extensões C++ desatualizadas / JIT cache corrompido | `make rebuild` | `make preflight` antes de bench | `kernel/csrc/` |

---

## Scheduler / Batching / Pipeline

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **SCH-001** | Travamento servidor (sem logs >60s) | Overlap scheduling + MTP + cuda-graph deadlock | `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1` + `--cuda-graph-max-bs 0` | Exportar ANTES do python; documentado em RUNBOOKS | `engine/engine.py` |
| **SCH-002** | `len(running) != 1` trava MTP multi-request | Trava hardcoded para single request | `_spec_eligible_reqs()` retorna lista; `_spec_eligible_req()` wrapper | Testar multi-request MTP | `scheduler/spec.py:48` |

---

## GGUF / Checkpoint Loading

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **GGUF-001** | Load falha: `block_shape` / `stride` mismatch | Geometria MoE derivada de offset não de arquivo | Ler `expert_geometry` do GGUF metadata | Validador em `gguf/reader.py` | `models/gguf/reader.py` |
| **GGUF-002** | IQ3_S 27B: RSS 2.17GiB mas VRAM alta | Quantização mista não propagada para VRAM ledger | Ledger deve contabilizar quantização por tensor | Auditoria `vram_ledger.py` | `engine/vram_ledger.py` |

---

## Ambiente / Sistema

| Código | Sintoma | Causa Raiz | Fix | Prevenção | Arquivo |
|--------|---------|------------|-----|-----------|---------|
| **ENV-001** | `uv: command not found` | uv não no PATH ou não instalado | `/home/natal/.local/bin/uv` ou `pip install uv` | `make preflight` verifica | - |
| **ENV-002** | `CUDA_HOME` aponta para versão errada | Default aponta para 12.x, precisa 13.3 | `export CUDA_HOME=/models/outros/cuda-13.3` | Export no `.bashrc` ou `Makefile` | `Makefile:3` |
| **ENV-003** | `TORCH_CUDA_ARCH_LIST` errado | Default não inclui SM120 (RTX 5080) | `export TORCH_CUDA_ARCH_LIST="12.0;12.0a"` | Export no `.bashrc` ou `Makefile` | `Makefile:6` |

---

## Como Adicionar Novo Erro

```markdown
| **XXX-###** | Sintoma exato (copiar do log) | Causa raiz técnica (1 linha) | Fix aplicado (comando ou arquivo:linha) | Prevenção (checklist/doc) | Arquivo principal |
```

**Regras:**
- Append-only (nunca apagar ou reordenar)
- Código: `COMPONENTE-###` (MOE, KVC, MTP, TOK, CUDA, SCH, GGUF, ENV)
- Um erro por linha
- Referenciar EXP/LESSONS/QA quando aplicável
| KVC-005 | VRAM ledger mismatch após replay spec | _commit_spec_tokens não restaura cmp_k_buffer rows fechadas | adicionar cmp_rows restore em _restore_qsa_state | validar EXP-048 replay | scheduler/spec.py:114 |
| TEST-001 | Teste doc-append | script testado | validado | make doc-test | scripts/doc-append.py |
| DOC-001 | EXP-050 claim prefill-window MTP warmup 45% accept | Documentação afirma implementação mas scheduler/spec.py não tem código para popular draft KV no prefill | Corrigir docs CONTEXT.md/STATE.md removendo claim falso | Verificar implementação antes de documentar | docs/dev/CONTEXT.md, docs/dev/STATE.md |
| DOC-002 | STATE.md afirma multi-request MTP suportado | Código mantém single-request only (_spec_eligible_req com len(running)!=1) | Corrigir STATE.md para refletir realidade do código | Verificar implementação antes de documentar | docs/dev/STATE.md |
