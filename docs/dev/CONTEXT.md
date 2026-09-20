# CONTEXT — freetoken-next

**Projeto:** FreeToken Next — Motor de inferência MoE edge-native para modelos de fronteira (290B+) em hardware consumer (RTX 5080 16GB).

**Base upstream:** FreeToken v0.1.3 (`cac247a`), branch `next` (0 commits behind upstream/main).

**Hardware alvo:** RTX 5080 (15.51 GiB VRAM, SM120 Blackwell, PCIe 4.0), 96 GB DDR5 RAM, NVMe.

**Checkpoints principais:**
- **Família Qwen 3.8 Flash Next** — **ALVO ÚNICO EXCLUSIVO ATUAL**:
  - `Qwen3.8-Flash-Next-NVFP4-Radix` (`/models/Qwen3.8-Flash-Next-NVFP4-Radix`): 48 layers (36 GDN, 12 QSA), 256 experts, MTP nativo em Blackwell SM120.
  - `Qwen3.8-Flash-Next-Unsloth-IQ4_XS` (`/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS`): Checkpoint GGUF compacto (UD-IQ4_XS sharded + MTP heads em `MTP/`), alvo de alta performance TG por menor tamanho e footprint de memória.
- *Demais modelos (Qwen 35B-A3B, Ornith, Tiel, etc.) temporariamente BLOQUEADOS para novas features/testes; foco 100% na família Qwen 3.8 Flash Next.*

---

## Estado Atual (2026-09-19)

### ✅ CONCLUÍDO — Roadmap Lines 1-8, 11
- **Line 1-3:** Lineage, build, baselines reproduzidos (35B: PP 4611/TG 158.8; Flash: PP 1858/TG 28.7)
- **Line 4:** 9 auditorias fonte (A1-A9) completas e integradas
- **Line 5-6:** VRAM Ledger + Governor operacional; 128K/256K funcionando no 35B-A3B
- **Line 7:** Loader GGUF nativo committed; primeira linha densa medida (IQ3_S 27B: PP 2417/TG 25.3, RSS 2.17 GiB)
- **Line 8:** Turbo4 + MTP **certificados** — bit-identical ao baseline Triton+BF16 (sha1 `614aa7bcdf59`)
  - Bug A (GDN decode vs prefill): caracterizado, delta ~1.95e-3
  - Bug B (QSA carrier leak): corrigido via `free_req` + snapshot/restore
  - TG MoE: 0.79 → 25 tok/s (~25x) via micro-batch decode path
  - Multi-token k=2/3 validado live
  - Long-context 128K certificado: PP 1376, TG 4.86, 14.84 GiB VRAM
  - Pillar 1: Kernel fused In-SRAM FP4 dequant (elimina DRAM workspace)
  - MTP carry shift: aceitação draft 90.9% (86.4% full 2/2)
  - MTP2 64-tok: 2.82 tok/step, TG 25.6-26.2, sha1 `ed45eb6cc897`
- **Line 11:** MTP nativo qwen4exp validado (EXP-043/044/045)

### 🔄 EM ANDAMENTO — Roadmap Lines 9-10
- **Line 9 (Phase 7):** Pool de experts keyed por geometria exata (bank, role, type) — desbloqueia Ornith/Tiel MoE GGUF
- **Line 10 (Phase 10):** Fusão MTP + TurboKV (D-022/023)
  - Pillar 2: Zero-replay GDN
  - Split-kernel Turbo4/QSA: decompressão separada + attention denso (evita ptxas host-OOM)

### ✅ CONCLUÍDO NESTA SESSÃO (2026-09-19)
- **Benchmark MTP Varredura Completa NVFP4 (k=0..6):**
  - $k=0$ (Greedy baseline): **TG 27.73 tok/s**, PP 1681 tok/s, VRAM 13.75 GiB, sha1 `c0e2b6c30ac9`
  - $k=1$ (MTP Speculative): **TG 24.39 tok/s**, PP 1670 tok/s, VRAM 14.49 GiB, sha1 `17f277f43565`, 100% accept (1/1)
  - $k=2$ (MTP Multi-Token): **TG 25.09 tok/s** (best MTP TG), PP 1687 tok/s, VRAM 14.49 GiB, sha1 `17f277f43565`, 100% accept (2/2)
  - $k=3$: **CRASH corrigido (2026-09-20)** — shape mismatch GDN layer [48,128,128] vs [1,48,128,128] durante verify forward; fix `.squeeze(0)` na captura em `gdn.py`. Não retestado o TG de k=3 após o fix.
  - $k=4,5,6$: Não testados (k=3 falhou)
  - **Conclusão:** Para NVFP4 + naive cache + Radix backend em 16K context, MTP overhead excede benefício — melhor TG em k=0 (27.73 tok/s). MTP k=2 é melhor entre configs MTP (25.09 tok/s) mas ainda 9.5% abaixo do baseline.
- **GGUF Adapter qwen4_exp — RESOLVIDO (2026-09-20):**
  - Checkpoint `Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS` carrega e serve end-to-end.
  - Bugs corrigidos em `gguf.py`: indexer `index_kv_heads` (usava GQA kv_heads em vez do 1 fixo do indexer), PLE `ple_layer_index` (índice absoluto vs local divergente entre init e geração de pesos), `ModelConfig` sem `slot_states=ple_slot_states(qwen4_args)` (causava `PLE needs ple_ngram_ctx slot state`).
  - Verificado com `benchmarks/bench_pp_tg.py` (4096 tok / 32 decode, TG 31.36 tok/s) e `cert_matrix.py` a 16384 tokens (falha residual é stall-timeout de harness por lentidão real de prefill IQ4_XS sem MMQ, não bug de carregamento). Ver `PERFORMANCE.md`.
- **Documentação Atualizada:** `docs/dev/PERFORMANCE.md` (anchors NVFP4 k=0,1,2,3), `docs/dev/STATE.md` (status atual)

### ⏳ PENDENTE — Roadmap Lines 12-18
- Line 12: TCQ/VBR policy
- Line 13: PLE tiered/paged RAM KV
- Line 14: Adaptive MTP + `--context auto`
- Line 15: Certificação 512K/1M (bloqueado por RoPE do checkpoint)
- Line 16: Matriz de certificação verde (`benchmarks/cert_matrix.py`)
- Line 17: Gaps de matriz (shard joining, qwen4exp GGUF adapter, PLE mapping)

---

## Métricas de Referência (Anchors)

| Modelo | Contexto | PP (tok/s) | TG (tok/s) | VRAM | RSS | GPU Util |
|--------|----------|------------|------------|------|-----|----------|
| 35B-A3B | 16K | 4611 | 158.8 | 14.98 GiB | ~20 GiB | 99.8% |
| Flash-Next | 16K | 1858 | 28.7 | 14.86 GiB | 67.8 GiB | 99.99% |
| 35B-A3B | 128K | 3189 | 89.3 | 14.4 GiB | 22.0 GiB | — |
| 35B-A3B | 256K | 2354 | 63.8 | 14.5 GiB | 22.0 GiB | — |
| Flash-Next | 128K | 1376 | 4.86 | 14.84 GiB | — | — |
| **Flash-Next NVFP4 (naive cache) k=0** | 16K | **1681** | **27.73** | 13.75 GiB | 69.7 GiB | 94.4% |
| **Flash-Next NVFP4 (naive cache) k=1** | 16K | 1670 | 24.39 | 14.49 GiB | 69.7 GiB | 87.3% |
| **Flash-Next NVFP4 (naive cache) k=2** | 16K | 1687 | **25.09** | 14.49 GiB | 70.4 GiB | 92.4% |
| **Flash-Next NVFP4 (naive cache) k=3** | 16K | — | **CRASH** | — | — | — |

---

## Arquitetura Crítica

1. **VRAM Ledger** (`engine/vram_ledger.py`): Single source of truth para ceiling, reserve, expert/KV split, context rows
2. **TurboKV/QSA Split** (`qsa_pool.py` + `kernel/triton/qsa/decompress.py`): Decompressão separada evita ptxas host-RAM explosion
3. **MoE Offload Cache** (`moe/offload_cache.py`): LRU expert banks, H2D assíncrono, double-buffer prefill
4. **MTP Speculative** (`scheduler/spec.py` + `models/qwen4_exp/`): Draft head nativo, carry shift, multi-token
5. **GGUF Loader** (`models/gguf/`): Native, sem dequant load-time, suporta mixed quant (Q3_K/Q4_K/Q6_K/IQ4_XS)

---

## Comandos Operacionais

```bash
cd /models/desenvolvimento/freetoken-next
source .venv/bin/activate
export TMPDIR=/models/desenvolvimento/tmp

# Servidor base
.venv/bin/ft serve --model /models/Qwen3.6-35B-A3B-NVFP4-FT --num-tokens 16576 --cache-type naive

# Benchmark PP/TG
.venv/bin/python benchmarks/bench_pp_tg.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
    --tokens 16384 --decode 128 --repeats 3 --label <tag> \
    --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive" --json /tmp/pp_tg.jsonl

# Testes
.venv/bin/python -m pytest tests -m "not slow" -q --basetemp=/models/desenvolvimento/tmp
```

---

## Pre-Flight Check (Obrigatório antes de qualquer benchmark)

1. `free -h` — RAM livre, cache limpo, nada em swap/tmpfs
2. `nvidia-smi` — VRAM livre, sem processos remanescentes
3. `ps aux | grep -E '(python|tail)'` — matar órfãos/zumbis
4. `du -sh /tmp/*` — verificar lixo acumulado (tmpfs = RAM real)

---

## Referências Cruzadas

| Arquivo | Papel | Conteúdo Principal |
|---------|-------|-------------------|
| `STATE.md` | Estado executável atual | Snapshot 2026-09-19, handoffs, gates, comandos |
| `ROADMAP.md` | Plano fases 1-18 | Status, gates, dependências |
| `LESSONS.md` | Padrões sintoma→causa→fix | 164 entradas validadas em hardware real |
| `DECISIONS.md` | Registro imutável D-001 a D-023 | Por que cada escolha arquitetural |
| `PERFORMANCE.md` | Números medidos | Tabelas PP/TG/VRAM/RSS por config |
| `ARCHITECTURE.md` | Design atual | Subsistemas, seams, file:line references |
| `EXPERIMENTS.md` | Diário empírico | EXP-000 a EXP-045, setup/result/verdict |
| `AGENTS.md` | Instruções para agentes IA | Regras, workflows, comandos de verificação |
| `CLAUDE.md` | Ponte para AGENTS.md | Regra única: leia AGENTS.md primeiro |
| `QA.md` | Quality Assurance | Gates, validações, checklists de release |

---

## Próximos Passos Imediatos (Prioridade)

1. ~~Investigar crash k=3 (GDN shape mismatch)~~ — corrigido em `gdn.py` (`.squeeze(0)` na captura), suite mínima 21/21 passando (2026-09-20).
2. ~~Resolver mismatches arquiteturais GGUF Unsloth-IQ4_XS (PLE dims, indexer heads)~~ — 3 bugs corrigidos em `gguf.py` (indexer `index_kv_heads`, PLE `ple_layer_index`, `ModelConfig` sem `slot_states`); verificado end-to-end (2026-09-20). Ver `PERFORMANCE.md`.
3. ~~Executar `benchmarks/cert_matrix.py` para matriz completa de certificação~~ — rodado a 16384 ctx (2026-09-20): 1 regressão de guard (PP em `native-35b-a3b`) + 2 falhas de VRAM/OOM pré-existentes não investigadas. Ver `PERFORMANCE.md` e `STATE.md`.
4. Investigar regressão de guard PP em `native-35b-a3b` (-0.5%, `cert_matrix.py`) e falhas de VRAM em `native-flash-next`/`gguf-qwen38-27b-iq3s` a 16384 ctx.
5. Testes 512K/1M diferidos (requerem flag `--allow-rope-extend`)
6. **MTP Speculative Decode Optimization (NVFP4-Radix)**: K=4 achieves 27.8 tok/s TG (-7% vs k=0 30.0), PP 1732, SHA1 `573a19610680` matches k=0 greedy baseline. Fixes: warmup full prefill context (no `spec_logits_indices`), decode replay for SHA1 equivalence, `_last_residual` seeding, adaptive gating via `adaptive_mtp.py`. Target: batched multi-token decode replay to reach +35% TG (39.7 tok/s).