# EXPERIMENTS — freetoken-next

**Diário empírico append-only.** EXP-000 a EXP-045. Source: `old/docs/freetoken-next/EXPERIMENTS.md` (1291 linhas).

---

## Formato

`EXP-NNN` — **Question** → **Setup** (reproduzível) → **Result** → **Verdict** (KEEP / LOW_GAIN_SURVIVOR / REJECT / INFORMATIONAL)

---

## Resumo por Categoria

### Lineage & Baselines (EXP-000 a EXP-002)
| Exp | Foco | Resultado |
|-----|------|-----------|
| EXP-000 | Lineage host provenance | `cac247a` = v0.1.3, 0 behind, SM120, py3.12.14 |
| EXP-001 | 35B-A3B baseline 16K | PP 4611, TG 158.8, VRAM 14.98 GiB |
| EXP-001b | Flash-Next baseline 16K | PP 1858, TG 28.7, VRAM 14.86 GiB, RSS 67.8 GiB |

### VRAM Ledger & Governor (EXP-003 a EXP-009)
| Exp | Foco | Resultado |
|-----|------|-----------|
| EXP-003 | Dummy-page brick | `num_pages + 1` priced in both budgets |
| EXP-004 | GGUF dense IQ3_S | PP 2417, TG 25.3, RSS 2.17 GiB |
| EXP-005 | Brick re-measure | PP 4610.1, TG 158.75, sha1 identical |
| EXP-006 | VRAM ledger brick 1 | Flash-Next serves at `--memory-ratio 0.9` |
| EXP-007 | 128K/256K arithmetic | 35B: 128K TG 89.3, 256K TG 63.8 |
| EXP-008 | VRAM governor bricks 2-4 | Guards hold, hashes unchanged |
| EXP-009 | Final gate | 35B PP 4610.8, Flash PP 1857.3, hashes unchanged |

### MTP Native & Speculative (EXP-020 a EXP-045)
| Exp | Foco | Resultado |
|-----|------|-----------|
| EXP-021/022/024 | MTP config/reader/pipeline | Weight loading, MoE aliasing, scheduler spec-loop |
| EXP-025 | Prefill-window MTP warm-up | **GAP** — draft KV não populado sobre prompt original |
| EXP-026 | MTP k=1 live serve | Content byte-identical to `--spec-mtp 0` |
| EXP-027/028 | 6 real bugs fixed | KeyError peso MTP, export pkg, MoE layer-count assert, cached_len/device_len accounting, state-corrupting reject rewind |
| EXP-029/031 | earlyoom host-RAM | ~68-70 GiB peak RSS irreducible; `--spec-mtp` não adiciona RAM |
| EXP-030 | `--moe-cache-auto` bug | `--num-tokens` explícito vira piso obrigatório |
| EXP-033 | Page leak `allocate_paged` | Fix: `_prepare_batch(skip_alloc=True)` em replay |
| EXP-034 | Real page leak (3 bugs spec.py) | `cache.py` patch revertido; fix em `spec.py` |
| EXP-035 | Determinism probe `--decode 4` | 2 fases estáveis: req 1-2 (sha1 A), req 3-6 (sha1 B) |
| EXP-036 | Bug A: GDN decode vs prefill | Delta ~1.95e-3, flips greedy ~1.5% steps |
| EXP-037 | TG MoE bottleneck | 96.3% prefill forward = materializing 512 experts × 48 layers |
| EXP-038 | Bug B: QSA carrier leak | `pending_ring` + `_cmp_k_buffer` surviving recycled `table_idx` |
| EXP-039 | QSA snapshot/restore | `free_req` zeros state; greedy sha1 bit-identical cross-runs |
| EXP-040 | MoE micro-batch decode | `verify_forward` 1.203s→0.0735s (16.3x); TG 0.79→25 tok/s |
| EXP-041 | Long-context 128K Turbo4 | PP 1376, TG 4.86, 14.84 GiB VRAM, 2053 pages (0.871 GiB KV) |
| EXP-042 | Pillar 1: In-SRAM FP4 dequant | Kernel fused `_qsa_sparse_paged_gqa_splitk_kernel`, zero DRAM workspace |
| EXP-043 | MTP carry shift | Draft accept 90.9% (86.4% full 2/2) |
| EXP-044 | Multi-token detokenizer | Sequential per-uid processing, 100% dedup |
| EXP-045 | MTP2 64-token cert | 2.82 tok/step, TG 25.6-26.2, sha1 `ed45eb6cc897` |

### TurboKV/QSA Integration (EXP-010 a EXP-019)
| Exp | Foco | Resultado |
|-----|------|-----------|
| EXP-010 | GGUF matrix row | IQ3_S 27B measured |
| EXP-011 | Re-measure quantities | Every gating number re-measured before plan |
| EXP-012 | Triton + BF16 baseline | PP 1814.7, TG 21.17, sha1 `614aa7bcdf59` |
| EXP-013 | Packed layout reader | Byte-per-element doubled ITL; fix: load once, split registers |
| EXP-014 | Turbo3/Turbo4 codec | A8 audit: turbo3 112B, turbo4 132B per 256-row |
| EXP-015 | GGUF MoE host-RAM | 63.46 GiB bank > 62 GiB avail → blocked |
| EXP-016 | 512K/1M blocked | RoPE 262K pos + BF16 KV |
| EXP-017/018/020 | FTW cold-bank | Resumable converter, 73.53 GiB Flash-Next FTW valid |

---

## Verdicts Agregados

| Verdict | Count | Exemplos |
|---------|-------|----------|
| **KEEP** | ~25 | EXP-001, 004, 006, 007, 008, 009, 026, 034, 039, 040, 041, 042, 043, 044, 045 |
| **INFORMATIONAL** | ~12 | EXP-000, 002, 003, 005, 011, 012, 014, 016, 017, 018, 020 |
| **REJECT** | ~3 | EXP-025 (gap), 029 (earlyoom race), 033 (wrong fix) |
| **LOW_GAIN_SURVIVOR** | ~2 | EXP-010, 013 |

---

## Experimentos Críticos Pendentes

1. **Split-kernel Turbo4/QSA 16K validation** (sha1 match baseline)
2. **MTP=1 + Turbo4 TG + accept-rate measurement** (goal priority 4)
3. **Non-determinismo sequential requests** (log logits top-2)
4. **k=2/k=3 live content equivalence**
5. **Prefill-window MTP warm-up implementation** (EXP-025)
6. **Phase 7 expert pool geometry keying**

---

## Referência Completa

`old/docs/freetoken-next/EXPERIMENTS.md` — Setup exato, comandos, logs, métricas brutas, verdicts detalhados para todos 45 experimentos.