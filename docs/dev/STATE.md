# STATE — freetoken-next (Root)

**Snapshot:** 2026-09-18 | **Branch:** `next` @ `cac247a` (v0.1.3) | **Hardware:** RTX 5080 16GB / 96GB RAM

---

## Status Resumido

| Fase | Item | Status | Evidência |
|------|------|--------|-----------|
| 1-3 | Lineage, Build, Baselines | ✅ Done | `git fetch` 0 behind; PP/TG anchors reproduzidos |
| 4 | Source Audits A1-A9 | ✅ Done | `old/docs/freetoken-next/audits/A1-A9.md` |
| 5-6 | VRAM Ledger + Governor | ✅ Done | `engine/vram_ledger.py`; 128K/256K working |
| 7 | GGUF Loader Native | ✅ Committed | Dense IQ3_S measured: PP 2417/TG 25.3, RSS 2.17 GiB |
| 8 | **Turbo4 + MTP Certified** | ✅ **Closed** | Bit-identical sha1 `614aa7bcdf59`; TG 0.79→25 tok/s |
| 9 | Phase 7: Expert Pool Geometry | 🔄 In Progress | Keyed by (bank, role, type) |
| 10 | Phase 10: MTP + TurboKV Fusion | 🔄 Specified (D-022) | Split-kernel path running |
| 11 | Native MTP qwen4exp | ✅ Validated | EXP-043/044/045: 90.9% accept, MTP2 certificado |
| 12-18 | TCQ/VBR, PLE, Adaptive, 512K+, Cert Matrix | ⏳ Pending | Gates definidos em ROADMAP.md |

---

## Métricas Atuais (Anchors de Regressão)

```bash
# 35B-A3B @ 16K (guard: PP≥4600, TG≥158)
PP: 4611  TG: 158.8  VRAM: 14.98 GiB  RSS: ~20 GiB  GPU: 99.8%

# Flash-Next @ 16K (guard: PP≥1850, TG≥28.5)
PP: 1858  TG: 28.7  VRAM: 14.86 GiB  RSS: 67.8 GiB  GPU: 99.99%

# 35B-A3B @ 128K
PP: 3189  TG: 89.3  VRAM: 14.4 GiB  RSS: 22.0 GiB

# 35B-A3B @ 256K
PP: 2354  TG: 63.8  VRAM: 14.5 GiB  RSS: 22.0 GiB
```

---

## Handoff Crítico (Próxima Sessão)

**Prioridade 1:** Validar split-kernel Turbo4/QSA em 16K (sha1 match baseline)
**Prioridade 2:** Medir TG + accept-rate MTP=1 com Turbo4 ativo
**Prioridade 3:** Debug non-determinismo sequential requests (log logits top-2)
**Prioridade 4:** Live test k=2/k=3 content equivalence
**Prioridade 5:** Phase 7 expert pool geometry keying

---

## Arquivos de Referência Vivos

- **Contexto completo:** `CONTEXT.md`
- **Roadmap detalhado:** `ROADMAP.md` / `docs/freetoken-next/ROADMAP.md`
- **Estado detalhado:** `docs/freetoken-next/STATE.md`
- **Lições aprendidas:** `LESSONS.md`
- **Decisões imutáveis:** `DECISIONS.md` / `docs/freetoken-next/DECISIONS.md`
- **Performance medida:** `PERFORMANCE.md` / `docs/freetoken-next/PERFORMANCE.md`
- **Arquitetura:** `ARCHITECTURE.md` / `docs/freetoken-next/ARCHITECTURE.md`
- **Experimentos:** `EXPERIMENTS.md` / `docs/freetoken-next/EXPERIMENTS.md`
- **QA Gates:** `QA.md`
- **Instruções agentes:** `AGENTS.md`

---

## Ambiente

```bash
cd /models/desenvolvimento/freetoken-next
source .venv/bin/activate
export TMPDIR=/models/desenvolvimento/tmp
# Pre-flight: free -h && nvidia-smi && ps aux | grep -E '(python|tail)'
```