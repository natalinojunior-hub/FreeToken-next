# CONTEXT — freetoken-next

**Projeto:** FreeToken Next — Motor de inferência MoE edge-native para modelos de fronteira (290B+) em hardware consumer (RTX 5080 16GB).

**Base upstream:** FreeToken v0.1.3 (`cac247a`), branch `next` (0 commits behind upstream/main).

**Hardware alvo:** RTX 5080 (15.51 GiB VRAM, SM120 Blackwell, PCIe 4.0), 96 GB DDR5 RAM, NVMe.

**Checkpoints principais:**
- Qwen3.8-Flash-Next-NVFP4 (48 layers, 36 GDN, 12 QSA, 256 experts, MTP nativo)
- Qwen3.6-35B-A3B-NVFP4 (48 layers, MoE 8 experts ativos)

---

## Estado Atual (2026-09-18)

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
| `STATE.md` | Estado executável atual | Snapshot 2026-09-18, handoffs, gates, comandos |
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

1. **Validar split-kernel Turbo4/QSA** em 16K contra baseline Triton+BF16 (sha1 match)
2. **Medir accept-rate + TG** MTP=1 com Turbo4 ativo (goal priority 4)
3. **Resolver non-determinismo** requests sequenciais same-server (logging logits top-2)
4. **Live test k=2/k=3** content equivalence (short + ocean poem prompt)
5. **Implementar prefill-window MTP warm-up** (EXP-025 gap)
6. **Phase 7:** Expert pool keyed by (bank, role, type) — unblock Ornith/Tiel MoE GGUF