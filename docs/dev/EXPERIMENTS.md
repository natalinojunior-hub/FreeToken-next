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
| EXP-046 | Determinismo sequential pós-fix MTP slot | 4 reqs sequenciais (`--decode 4 --repeats 4 --warmups 0`, same server) todos sha1 `614aa7bcdf59` = baseline; carrier ring/cmp_base/scratch pré-draft idênticos cross-req; PP 1776, TG 10.34 (req1 7.76 warm-up residual experts, req2-4 11.2), VRAM 14.76 GiB |
| EXP-047 | k=2/k=3 live content equivalence vs k=0 (SUPERSEDED by EXP-048) | 16384 tok / decode 64 / 4 repeats, RTX 5080, `--kv-format=turbo4 --cache-type=naive`. Causa identificada e resolvida em EXP-048 (bug no `detokenize.py`, múltiplos tokens por UID). Validado live pós-fix: k=2 decode 64 gera sha1 `ed45eb6cc897` 100% bit-identical ao baseline k=0. |
| EXP-048 | Divergência determinística k>=1 vs k=0 (RESOLVIDO & CERTIFICADO LIVE) | Causa raiz: bug no `DetokenizeManager.detokenize()` (`python/freetoken/tokenizer/detokenize.py`). O modelo e o verify batch de MTP foram 100% determinísticos e idênticos ao baseline k=0. Porém, ao aceitar m=2 tokens em um único passo, o scheduler enviou múltiplos `DetokenizeMsg` do mesmo `uid` no mesmo lote. `detokenize()` acumulava todos os tokens no `DecodeStatus` antes de atualizar offsets, fazendo o segundo token reutilizar texto e decodificar fatia duplicada (`--0` em vez de `-0`), corrompendo o sha1 do texto retornado. Fix: processamento sequencial quando há múltiplos msgs por UID no batch. Validação live em hardware real (RTX 5080, Flash-Next 16K, turbo4+naive, artefatos em `benchmarks/results/`):<br>- `--decode 16`: k=0 sha1 `17f277f43565` (TG 20.72), k=1 sha1 `17f277f43565` (TG 9.32), k=2 sha1 `17f277f43565` (TG 8.07) — 100% bit-identical.<br>- `--decode 64`: k=0 sha1 `ed45eb6cc897` (TG 22.94), k=1 sha1 `ed45eb6cc897` (TG 11.06), k=2 sha1 `ed45eb6cc897` (TG 8.43) — 100% bit-identical. |

| EXP-049 | Accept-rate tuning k=2 com carry shift | 16384 tok / decode 64 / 4 repeats / RTX 5080 / turbo4+naive / spec-mtp 2 | Accept full 2/2: 12% (era 3.3%); TG 12.4 tok/s (era 8.81); sha1 match baseline 3/4 runs | INFORMATIONAL — carry shift ajuda mas warm-up gap persiste |
| EXP-050 | Prefill-window MTP warm-up (EXP-025 gap) | 16384 tok / decode 64 / RTX 5080 / turbo4+naive / spec-mtp 2 / populate draft KV over prompt | Draft KV populated on prefill; accept-rate full 2/2: 45% (era 3.3%); TG 18.2 tok/s; sha1 match baseline | KEEP — resolve EXP-025 gap, unblock k>=2 production |
| EXP-051 | Estudo repos externos (atomicmilkshake/FreeToken fork + UnsignedChad/windows-freetoken-mtp) para portabilidade GGUF/TurboQuant/MTP | Analise via GitHub API compare-diff (sem clone): atomicmilkshake/FreeToken (fork de FlashML-org/FreeToken, diverged +37/-46 commits) e windows-freetoken-mtp (repo standalone, mtp.md+mtp.py+bench_mtp_accept.py) | (1) GGUF loader do fork alvo qwen3_moe/qwen3_5_moe e DESCARTA bloco NextN/MTP no GGUF (nao suporta MTP em GGUF); nao resolve mismatch PLE/indexer do Unsloth-IQ4_XS (arquitetura diferente, qwen4_exp). (2) Fork adiciona kvcache/quant_codec.py: codec KV tq2/tq3/tq4 (2/3/4-bit, Lloyd-Max + norm L2 por bloco + nibble-pack), com kernels CUDA JIT espelhados em csrc/jit/quant_codec.cuh e quant_store.cu/quant_dequant.cu -- arquitetura-agnostico, candidato real a portar como TurboKV 4-bit no QSA pool. (3) windows-freetoken-mtp NAO tem loop de decode MTP produtivo (README/mtp.md admitem explicitamente 'production --mtp loop remains the way to realize speedup, not yet wired'); alpha=0.894 e 144 tok/s sao projetados via harness pure-torch offline, nao medidos em serve real; arquitetura alvo e Qwen3.6-35B-A3B (qwen3_5_moe), nao Qwen3.8-Flash-Next. Design de rollback GDN sugerido (snapshot/restore via HybridRadixCache) e verify via prefill/extend chunked -- mas isso e o MESMO padrao que LESSONS.md ja aponta como causa da lentidao MTP k=1/k=2 em nossa engine (Batch phase=prefill por token + host syncs). | NENHUM fix direto portavel para os 2 blockers abertos (GGUF PLE/indexer mismatch, crash k=3 GDN shape). Unico item de valor real: extrair o codec tq2/tq3/tq4 (kvcache/quant_codec.py do fork) como base para 4-bit KV no QSA pool, mencionado em LESSONS.md como cura real do MTP k=1 OOM. Nao clonar/mergear nada; portar sob avaliacao humana (licenca Apache-2.0 compativel). |
| EXP-052 | IQ4_XS cold production k0/k1 and opt-in trace | RTX 5080, HEAD 437b5a6, same 4101-token prompt/SHA, turbo3, ctx16K, one request/no warmup; separate 96-token k1 diagnostic with 8 profiled SPEC cycles and Y=1/2/4 geometry microprobe | k0 36.60 TG/1456 PP; auto k1 42.53/1466; trace 27.39 instrumented, 57 SPEC cycles, 8 profiled; Y=2/4 only tiny dense-kernel differences, no end-to-end evidence; output hashes differ | INFORMATIONAL — single pair and diagnostic; no quality/equivalence claim |
| EXP-053 | Auditoria de oportunidades adicionais após Campaign 7 | HEAD 437b5a6; auditorias Luna em transferência de experts, compute GGUF e scheduler/MTP; leitura dos caminhos ativos e métricas Campaign 7 | Nenhum candidato seguro encontrado. Warm-seed de experts limita-se ao primeiro ciclo (<1% TG); _adaptive_mtp_controller não é instanciado; fusão de redução/escala no MoE tem teto inferior ao critério de probe e risco de captura/ordem numérica. | Encerrar a busca nestes mecanismos; manter como próximas hipóteses somente residência compatível com rota comprovada e microprobe CUDA com >=1 ms/ciclo end-to-end. |
| EXP-054 | Auditoria integral PP/TG após Campaign 7 | HEAD 7aea2ec; auditoria paralela de engine/scheduler, MoE/offload, PP/MTP/GDN/QSA, kernels, memória e documentação; revisão dos commits e artefatos de campanha | Nenhum mecanismo adicional com ganho reproduzível foi encontrado. O banco próprio de experts MTP descrito como pendente em relatorio-estudo-mtp-lto.md já está implementado em a9d30f4 e elevou aceitação documentada de 55.5% para 75.2%. Caminhos ativos já cobrem graph decode/draft/verify, cópia fused, pools por geometria, D2D hits, overlap, hybrid, TurboKV/QSA e chunking conservador. | Não alterar produção. Hipóteses restantes exigem projeto e A/B: residência guiada por rota/decode_freq, zero-replay completo com snapshots QSA/PLE e aceitação device-side; não há prova de ganho nem orçamento seguro. PP frio atual ~1456-1466 e TG ~36.6 k0/~42.5 k1 permanecem referência matched Campaign 7. |
| EXP-055 | Campaign 8: hipóteses de alto risco em worktrees reversíveis | Três worktrees isolados para residência, zero-replay e aceitação device-side; replay CPU, testes de rollback/GDN/spec e revisão de invariantes | Residência: LRU reproduz misses em +0.018%/-0.034%; Belady economiza 38-42% mas exige rotas futuras. Zero-replay: custo k1 sub-1%, k2/k3 abaixo do break-even e faltam snapshots QSA/PLE/KV completos. Aceitação device-side: intervalo medido ~0.12 ms/ciclo (~0.3% TG), com rollback/páginas/EOS ainda host-bound. Testes direcionados passaram. | Nenhuma hipótese atingiu o limiar solicitado de >5% TG com PP <=3% de regressão e preservação de qualidade/VRAM/fallback; não implementar. |
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