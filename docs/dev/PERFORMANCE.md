# PERFORMANCE — freetoken-next

## 2026-09-26 campaign 21 (Qwen 3.8 Flash Next MTP evaluation on RTX 5080; ledger ft-campaign2/campaign21/LEDGER.md)

| workload | k0 TG | k1 TG | Delta | Verdict / Root Cause |
|---|---|---|---|---|
| Flash Next AD 4.27bpw @ 16K | 53.4 | 50.8 | -4.9% | PCIe cache thrashing (996 MiB/step misses @ 53.3 GB/s = 10.5-19.6 ms) |
| Flash Next AD 4.27bpw @ 64K | 52.1 | 53.3 | +2.3% | PCIe cache thrashing (below +5% gate) |
| Flash Next ISTA IQ3_XXS @ 16K | 56.3 | 62.7 | +11.4% | Pass (+11.4% at 16K, lighter 1.78 MiB expert size) |
| Flash Next ISTA IQ3_XXS @ 64K | 56.8 | 57.5 | +1.2% | Fails +5% gate at 64K |

- Gate 0 attribution: 96.5% of verify cycle attributed. Warm 2-row verify graph = 17.12 ms, real verify = 27.40 ms. Unexplained gap of ~10.3-17.6 ms is 100% accounted for by 448.8 missing target experts per verify step (18.91 active experts/layer across 2 rows, 49.5% miss rate).
- Evaluated Hypotheses:
  - H1 (Multi-row MoE union GEMM): 16.6% overlap between verify rows saves only 176.9 MiB/step resident weight reads = 0.14-0.18 ms. NO-GO.
  - H2 (VRAM accounting): 0 MTP tensors shareable with target (all 32 SHA256 unique); 325.4 MiB false reservations documented in ledger.
  - H3 (QSA split-K multi-row): QSA split-K scales from 0.358 ms to 0.896 ms (delta = 0.538 ms = 1.96% of verify cycle, below 3% threshold). NO-GO.
  - H4 (Rejection replay & state transaction): Amortized replay = 0.785 ms/step (2.86% of verify cycle, below 3% threshold). Zero-replay violates state boundary contract. NO-GO.
  - H5 (Draft breakdown): Draft graph replay = 0.652 ms (2.38% of verify cycle, below 3% threshold). NO-GO.
- Decision: Retain k0 default for Flash Next MTP companion models. Models with in-file NextN retain k=1.

## 2026-09-26 campaign 20 (server, 256 decode, auto KV; ledger ft-campaign2/campaign20/LEDGER.md)

| change | before -> after |
|---|---|
| coded-KV prefill on FlashInfer (nvfp4/turbo prefix dequantized per 64 MiB segment) | 27B 64K PP 1126 -> 2186 |
| fp8 prefill GEMMs (dense models only) | 27B 16K PP 1875 -> 2767; bf16 GEMM 119 TF -> fp8 235 TF (8192x5120x17408) |
| nvfp4 decode even/odd split | 1 layer 256K 339 -> 313 us (457 -> 496 GB/s) |
| MTP k=1 default (in-file NextN) | Tiel TG 16K 129.7->171.1, 64K 119.6->146.1, 128K 109.1->121.9, 256K 86.5->100.9; Ornith 16K 151.7->191.5, 64K 140.7->160.7, 256K 99.5->114.2; 27B 16K 57.1->71.7 |
| MTP dropped when it costs a KV rung | 27B 64K with --spec-mtp 1: refused -> k0 nvfp4 TG 46.2 |

Kept as-is after measurement: Flash hyper-connection projections (~680 GB/s, BF16 in the file), Flash
ssm_out (already packed Q8_0), MMVQ multi-vector traits (every type already reads a block once per <=4 rows).
Negative result: fp8 prefill on MoE (Tiel 4K TG 132.2 -> 118.9 for PP +4.6%; decode transient after fp8 prefill).

## 2026-09-24 critical-path attribution (HEAD `f83ee23`)

Prompt SHA256: `645f46bf134e597f1f70697d699ea70a83bf9cbe615681dad2b5eca67fbbfcf`. Cold 4K/32-token eager pair, fresh server per arm, identical prompt and flags: k0 PP 1465.5/TG 38.48/VRAM 14.75 GiB; k1 PP 1466.5/TG 21.82/VRAM 14.72 GiB. Output hashes were stable per arm (`9b6cb430f2f6`, `28e97bbdceb2`).

CUDA events around decode MoE gate/up and down recorded without per-layer synchronization. k0 unions: 34.860 ms gate/up, 33.579 ms down; 2976 events, 54.313 ms record overhead. k1 unions: 59.887 ms gate/up, 58.992 ms down; 2442 events, 47.987 ms overhead. The union totals are approximately 8.5% and 11.0% of their diagnostic decode wall times, respectively, and instrumentation materially changes timing. This does not support a >5% end-to-end MoE kernel optimization. Fetch/cache misses remain the measured residual (about 0.847 GB per k1 cycle from current trace evidence); no new policy was tested. Decision: NO-GO.

## 2026-09-24 current-head MTP-bank audit

No performance result or production change. CPU metadata proof confirms the IQ4_XS draft bank is `blk.48`, loaded as bank 48 after target banks 0–47; focused config proof passed 23 tests. Gate B optimization is deferred pending interval-union CUDA events for matched cold 4K k0/k1. Existing source uses two inline `ggml_moe_a8_vec` calls per decode MoE layer; prefill dequant reuse is already present. Do not infer a TG gain from this source observation.

**Todas as métricas em hardware real** (RTX 5080 15.51 GiB VRAM, SM120, 96 GB DDR5, NVMe). Source: `old/docs/freetoken-next/PERFORMANCE.md` (544 linhas).

---

## Diretriz Soberana: Otimização Guiada por TG (Não por Acceptance Rate)

**Atenção Máxima:**
- A **taxa de aceitação (% acceptance)** não é métrica de desempenho — é apenas um detalhe interno de diagnóstico.
- O critério único, soberano e irrefutável de decisão para MTP e escalonamento de $k$ ($k=1, 2, 3, 4\dots$) é o **TG (tok/s)** entregue ao usuário.
- Se elevar $k$ resultar em maior TG total, **mantém-se o $k$ maior**, mesmo que a aceitação percentual caia.
- O **único** critério válido para reduzir ou limitar $k$ é se o **TG real medido cair** devido ao custo marginal de verify/rejeição superar os tokens ganhos por step.

---

## Anchors de Regressão Imutáveis

| Workload | PP (tok/s) | TG (tok/s) | VRAM | RAM (RSS) | GPU Util | Guard |
|----------|------------|------------|------|-----------|----------|-------|
| **35B-A3B NVFP4 @ 16K** | **4611** | **158.8** | 14.98 GiB | ~20 GiB | 99.8% | PP≥4600, TG≥158 |
| **Flash-Next NVFP4 @ 16K** | **1858** | **28.7** | 14.86 GiB | 67.8 GiB | 99.99% | PP≥1850, TG≥28.5 |
| **Flash-Next NVFP4 @ 16K (k=0, naive cache)** | **1681** | **27.73** | 13.75 GiB | 69.7 GiB | 94.4% | PP≥1680, TG≥27.5 |
| **Flash-Next NVFP4 @ 16K (k=1, MTP)** | **1670** | **24.39** | 14.49 GiB | 69.7 GiB | 87.3% | PP≥1660, TG≥24.0 |
| **Flash-Next NVFP4 @ 16K (k=2, MTP)** | **1687** | **25.09** | 14.49 GiB | 70.4 GiB | 92.4% | PP≥1680, TG≥25.0 |
| **Flash-Next ISTA IQ3_XXS GGUF @ 16K (k0, cold, MMV_Y=4 + GDN ratio 0 + greedy pool caps)** | **2290** | **58.81** | 14.58 GiB | 79.5 GiB | 93% | PP≥2280, TG≥58 |
| **Flash-Next ISTA IQ3_XXS GGUF @ 16K (k0, warm prefix, mesma stack)** | — | **59.58** | 14.58 GiB | 79.5 GiB | — | TG≥59 |
| **Flash-Next ISTA IQ3_XXS GGUF @ 64K (k0, cold, mesma stack)** | **2410** | **57.97** | 14.74 GiB | — | — | PP≥2350, TG≥57 |
| **Flash-Next AD 4.27bpw GGUF @ 16K (k0, cold, MMV_Y=4 apenas; GDN ratio 0 muda a saída aqui)** | **2218** | **54.95** | 14.36 GiB | 81.2 GiB | — | PP≥2170, TG≥54 |
| **35B-A3B @ 128K** | 3189 | 89.3 | 14.4 GiB | 22.0 GiB | — | — |
| **35B-A3B @ 256K** | 2354 | 63.8 | 14.5 GiB | 22.0 GiB | — | — |
| **Flash-Next @ 128K** | 1376 | 4.86 | 14.84 GiB | — | — | — |

---

## Turbo4 + MTP Certified (EXP-041/045)

| Config | PP | TG | VRAM | sha1 | Notes |
|--------|----|----|------|------|-------|
| Turbo4 + MTP=1 @ 16K (Split-Kernel) | 1710.3 | 8.22 | 14.76 GiB | `614aa7bcdf59` | Split-kernel eager, bit-identical match |
| Turbo4 + MTP=1 @ 16K/64dec | 1711.7 | 10.60 | 14.76 GiB | `29c2d83f9e28` | Accept rate 12.7% (7/55), ITL p50 109.8ms |
| Turbo4 + MTP=1 @ 16K (Steady) | 1714 | 24.7 | ~15 GiB | `614aa7bcdf59` | MoE decode path, steady state |
| Turbo4 + MTP=1 @ 16K/4dec det (EXP-046) | 1776.0 | 10.34 | 14.76 GiB | `614aa7bcdf59` ×4 | 4 reqs sequenciais same-server, todos bit-identical baseline |
| Turbo4 + MTP=1 @ 16K/64dec det (EXP-046) | 1773-1778 | 10.8-10.9 | 14.76 GiB | `b5878a20e984`,`29c2d83f9e28` ×3 | 2 sessões fresh: sequências de sha1 por posição idênticas |
| Turbo4 + MTP=2 @ 16K/64dec | — | 25.6-26.2 | — | `ed45eb6cc897` | 2.82 tok/step, 86.4% 2/2 |
| MTP carry shift (EXP-043) | — | — | — | — | Draft accept 90.9% |
| Turbo4 + MTP=0 @ 16K/64dec (EXP-047 baseline) | 1780.4 | 23.31 | 14.61 GiB | `ed45eb6cc897` ×4 | greedy, sem spec |
| Turbo4 + MTP=2 @ 16K/64dec (EXP-047, pós clear_mtp_slot) | 1776.4 | 8.81 | 14.76 GiB | `79bd932522d1` (warm), diverge do baseline | **REGRESSÃO** vs linha acima: 86.4%→3.3% full-accept |
| Turbo4 + MTP=3 @ 16K/64dec (EXP-047) | 1771.3 | 7.55 | 14.76 GiB | `e556823cc8f6` ×4, diverge do baseline | full-accept 1.9% (4/216) |
| **Turbo4 + MTP=1 @ 16K/16dec (Zero-Replay GDN + D2D Reuse)** | **~1630** | **29.21** | **13.88 GiB** | `fc7d5d6eb894` | **+36.5% TG** vs k=0 (21.39 -> 29.21 tok/s). Supera o baseline de 28.5! |
| **Turbo4 + MTP=0 @ 16K/64dec (Certificado Equivalência)** | **1723** | **23.67** | **13.88 GiB** | `ed45eb6cc897` | Baseline greedy exato |
| **Turbo4 + MTP=1 @ 16K/64dec (Certificado Equivalência)** | **1720** | **28.38** | **13.88 GiB** | `ed45eb6cc897` | **PASS**: +20.0% TG vs k=0, bit-identical match ao baseline k=0 |
| **Turbo4 + MTP=2 @ 16K/64dec (Certificado Equivalência)** | **1721** | **28.24** | **13.88 GiB** | `ed45eb6cc897` | **PASS**: +19.3% TG vs k=0, bit-identical match ao baseline k=0 |
| **NVFP4 + MTP=0 @ 16K/16dec (naive cache, Radix)** | **1681** | **27.73** | **13.75 GiB** | `c0e2b6c30ac9` | Baseline greedy, naive cache, Radix backend |
| **NVFP4 + MTP=1 @ 16K/16dec (naive cache, Radix)** | **1670** | **24.39** | **14.49 GiB** | `17f277f43565` | MTP k=1, 100% accept (1/1), naive cache |
| **NVFP4 + MTP=2 @ 16K/16dec (naive cache, Radix)** | **1687** | **25.09** | **14.49 GiB** | `17f277f43565` | **Best MTP TG**, 100% accept (2/2), naive cache |
| **NVFP4 + MTP=3 @ 16K/16dec (naive cache, Radix)** | — | — | — | — | **CRASH**: shape mismatch GDN layer [48,128,128] vs [1,48,128,128] |
| **NVFP4 + MTP=4 @ 4K/64dec (naive cache, Radix, warmup fix)** | **1732** | **27.8** | **14.19 GiB** | `573a19610680` | **SHA1 matches k=0**, warmup full context, _last_residual fix |
| **NVFP4 + MTP=0 @ 4K/64dec (naive cache, Radix, baseline)** | **1768** | **30.0** | **14.10 GiB** | `573a19610680` | Baseline greedy, same SHA1 |


---

## GGUF Native (Primeira Linha Medida)

| Checkpoint | Quant | PP | TG | VRAM | RSS | Notes |
|------------|-------|----|----|------|-----|-------|
| Qwen3.8-27B | IQ3_S | 2417 | 25.3 | 14.43 GiB | **2.17 GiB** | Dense, coherent text |

*Native offload anchors precisam 21.9-67.8 GiB RSS → GGUF economiza ~95% host RAM.*

---

## Long-Context Scaling (35B-A3B via `--kv-reserve-tokens`)

| Contexto | PP | TG | TTFT | ITL p50 | VRAM | Expert Slots |
|----------|----|----|------|---------|------|--------------|
| 16K | 4611 | 158.8 | — | — | 14.98 | 4695 |
| 128K | 3189 | 89.3 | 41.1s | 10.99ms | 14.4 | 3183 |
| 256K | 2354 | 63.8 | 111.3s | 15.36ms | 14.5 | 3185 |

---

## KV Compression Impact (Turbo4 4-bit)

| Contexto | BF16 KV | Turbo4 KV | Expert Slots (BF16) | Expert Slots (Turbo4) |
|----------|---------|-----------|---------------------|----------------------|
| 16K | 430 MB | 60 MB | 4695 | 5427 |
| 128K | 3.4 GiB | 480 MB | 3183 | 5200 |
| 256K | 6.8 GiB | 960 MB | 0 | 5427 |
| 512K | 13.6 GiB | 1.9 GiB | 0 | 3200 |
| 1M | 27.2 GiB | 3.8 GiB | 0 | 1200 |

---

## MoE Offload Costs

| Modelo | Expert Bank Size | Host RAM Peak | Load Time |
|--------|------------------|---------------|-----------|
| Flash-Next NVFP4 | 63.46 GiB | ~68-70 GiB | ~3-5 min |
| 35B-A3B NVFP4 | ~20 GiB | ~22 GiB | ~1 min |
| Ornith/Tiel GGUF MoE | 63.46 GiB | 63.46 GiB | Blocked (geometry) |

---

## Test Suite Performance

```
pytest tests -m "not slow" --basetemp=/models/desenvolvimento/tmp
→ 1839 passed, 206 skipped, 1 failed (flashinfer fp4_quantization_120f - nvcc 13.3 env)
```

---

## CUDA Graph Path Stall (WIP tree, MoE offload em progresso) — 2026-09-20

**Achado:** com o CUDA graph decode habilitado (`--cuda-graph-max-bs 1`, default), a working tree com o WIP não commitado do MoE offload (`python/freetoken/moe/*`, `layers/moe.py`) trava indefinidamente após o token 1, em qualquer tamanho de contexto (reproduzido em 512 e 16384 tokens). GPU 0% util, CPU 300%+, RSS ~70 GiB. `py-spy record -s` (processo-pai, contorna `ptrace_scope=1`) mostra 65%+ das amostras presas em `torch/cuda/graphs.py:139 replay()` (nativo, chamado de `engine/graph.py:214`), logo após `_reset_moe_offload_cache()`. **Não confirmado** o mecanismo exato (py-spy não vê além da fronteira pybind) — não é um deadlock host↔device diagnosticado, apenas localizado ao `g.replay()`.

**Discriminador:** com `--no-graph` (decode eager, sem captura CUDA graph), o mesmo WIP tree completa normalmente, sem travar, em 512 e 16384 tokens. Isso isola a regressão ao caminho de CUDA graph, não ao MoE offload em si.

**Números eager (não comparáveis às âncoras acima, que foram medidas com CUDA graph):**

| Config (eager, `--no-graph`) | PP | TG | VRAM | mem-ratio | Expert Slots | Notes |
|---|---|---|---|---|---|---|
| WIP + NVFP4 @ 512 tok/16dec | 427.8 | 27.45 | 13.67 GiB | 0.9 | — | ctx curto, não comparável ao anchor 16K |
| WIP + NVFP4 @ 16384 tok/16dec | 1790.9 | 23.21 | 14.43 GiB | 0.85 | 1089 | mem-ratio reduzido p/ caber transients do decode eager (FLA `chunk_delta_h.py`); 0.9 dá OOM (17 MiB faltando) |

**Pendente:** causa raiz do stall no caminho de CUDA graph não identificada — diagnóstico encerrado após esgotar testes de baixo custo (`--no-graph` contorna; `FREETOKEN_FUSED_COPY=0` e `FREETOKEN_SMALL_BANK_FEAT_BYTES=262144` não resolvem). RSS ~70 GiB confirmado real durante o stall (medido no processo correto, flat por 125s+, descarta hipótese de "apenas lento"). Perfil nativo (`py-spy --native -s`) mostra a MainThread do processo travado alternando `sched_yield` e cópias AVX2 na mesma thread sob `replay()`; as amostras de `build_expert_banks/pack` vistas no perfil sem `--native` são da fase de build/startup, não do momento do stall — não confundir as duas fases ao reler o profile. Não revisar/substituir as âncoras k=0/1/2 (27.73/24.39/25.09) até o caminho de CUDA graph voltar a funcionar — elas foram medidas nesse caminho, que hoje não roda no WIP tree.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` @ mem-ratio 0.9 testado para recuperar o cache completo de 1287 slots: erro OOM byte-idêntico ao anterior (79.38 MiB livres, mesma mensagem sugerindo expandable_segments) — **teste inconclusivo**, a variável provavelmente não chegou ao processo scheduler real (é um filho `multiprocessing.spawn` do `freetoken.cli serve`, que é filho do bench script; setada no shell externo). Não descartar expandable_segments até testar exportando-a dentro do processo scheduler. 0.85/1089 slots permanece o número válido usado.

## Correção de premissa: comparação "NVFP4+Turbo4 justo vs LTO" não existe hoje — 2026-09-20

LTO (`llama-turbo-optimal`) é um fork do llama.cpp: só roda GGUF. Não executa o checkpoint NVFP4-Radix do FreeToken. Os próprios docs do LTO (QWEN38_FLASH_MTP_REFERENCE_BASELINE, seção 53-54) afirmam que qualquer smoke-test do FreeToken é "non-isomorphic to GGUF LTO". Não há como fazer a comparação como originalmente formulada (mesmo quant NVFP4+Turbo4 nos dois lados) — LTO nesse quant não existe.

Números que existem, nenhum comparável 1:1 entre si (contexto/quant/config diferentes):
- **LTO, target-only (sem MTP), Unsloth-IQ4_XS GGUF, KV q8_0, prompt curto**: 38.41 tok/s (não é 16K, não é NVFP4).
- **LTO, ctx=16384, MTP n_max=3 (especulativo)**: 38.07 TG — não comparável ao FreeToken sem MTP.
- **FreeToken 0.1.2+gaf71ba432 (versão antiga), NVFP4-Radix, boot "smoke" não certificado**: 34.7 tok/s decode. **Comparação com o número "atual" abaixo invalidada (2026-09-20):** o `stats.json` real desse boot mostra `max_seq_len_override=4096`, `prompt_tokens_total=22`, `completion_tokens_total=31` — decode com KV praticamente vazio, não "ctx=8192 preenchido" (`kv_reserve_tokens=8192` no ServerArgs é reserva de VRAM, não profundidade usada). O ServerArgs do 0.1.2 também não tem campo kv-format/turbo4 (KV puro, `cache_type='radix'`) e rodou com CUDA graph ligado. **Reproduzido de novo nesta sessão, mesma venv pinada em 0.1.2, mesmo comando: 34.7 tok/s idêntico** — o número é real e estável para essa condição, só não é comparável ao número abaixo.
- **FreeToken atual (WIP, eager `--no-graph`, ctx=8192, mem-ratio 0.85, 1089 slots)**: **TG 23.94 tok/s** (3 repeats, 23.93-23.94), quase idêntico ao número em 16K (23.21). **Não é controle válido contra o número do 0.1.2 acima**: profundidade de KV diferente (~22-53 tokens vs 8192 preenchidos), possível turbo4 vs KV puro, e graph ligado vs `--no-graph`. Nenhuma regressão foi de fato medida entre as duas versões ainda — para isso seria preciso isolar commit, profundidade de KV, kv-format e caminho graph/eager, os quatro ao mesmo tempo. Nota lateral: RSS 70.7 GiB aparece também nessa rodada bem-sucedida (sem stall) — não é assinatura exclusiva do travamento do CUDA graph, é o footprint normal do host-bank cache do MoE atual.
- **FreeToken atual (WIP, eager `--no-graph`, ctx=16384, mem-ratio 0.85, 1089 slots)**: 23.21 tok/s — melhor número válido de ponta a ponta obtido nesta sessão.
- **FreeToken atual (via CUDA graph, âncora histórica k=0)**: 27.73 tok/s — mas o caminho de CUDA graph está quebrado tanto em HEAD (`qsa_sparse.py:358` crash) quanto no WIP (hang em `replay()`), logo não é reproduzível agora.

**Conclusão:** a queda observada de FreeToken hoje (23.21 eager) vs a âncora histórica (27.73 via graph) é explicada pela combinação de duas variáveis não isoladas: decode eager (vs CUDA graph, hoje quebrado) **e** cache de experts reduzido (1089 vs ~1287 slots, por causa do mem-ratio 0.85 vs 0.9). Não atribuir a diferença inteira a nenhuma das duas isoladamente. Em nenhum caso é LTO estruturalmente superior num quant equivalente — a comparação justa pedida originalmente não é possível com os artefatos disponíveis (ver acima).

---

## Reteste do fix de CUDA graph capture (`ba7f976`) e regressão real 0.1.2 vs HEAD — 2026-09-20

**Reteste do fix:** com o fix de `qsa_sparse.py` (guarda `is_current_stream_capturing()`) já commitado em `ba7f976`, decode via CUDA graph (`--cuda-graph-max-bs 1`, default) completou normalmente em HEAD, sem hang em `replay()`: 512 tok prompt / 256 tok decode, 2 repeats, TG 33.10 tok/s estável, GPU util 98%. O hang documentado anteriormente no WIP tree (não commitado) não se manifesta em HEAD com o fix presente — não descartado que o WIP tree tivesse uma causa adicional não isolada, mas o caminho de graph em si voltou a funcionar.

**Comparação real (isolando profundidade de KV, kv-format, cache-type e caminho graph/eager entre 0.1.2 e HEAD):** ambos rodados com prompt 22 tokens, decode 31, `--kv-format bf16`, `--cache-type radix`, CUDA graph ligado (default).
- 0.1.2+gaf71ba432: **34.7 tok/s** (reproduzido, estável).
- HEAD (mem-ratio 0.98, `--kv-reserve-tokens 1024` — necessário para caber no orçamento de VRAM do ledger atual): **30.44 tok/s**, 3 repeats idênticos (sha1 `c80887695bb9`).

**Regressão real medida: -12.3%** (34.7 → 30.44 tok/s), não os ~31% da comparação anterior invalidada. Causa ainda não isolada — candidatos: mudanças no VRAM ledger/cache_budget (exigem mem-ratio 0.98 vs provável 0.9 do 0.1.2 só para caber o plano mínimo), ou custo adicional no caminho de decode entre as versões. Próximo passo natural seria bisect entre 0.1.2 e HEAD nessas condições fixas, mas não solicitado nesta rodada.

---

## GGUF Unsloth-IQ4_XS: fix do `PLE_NGRAM_STATE` e `cert_matrix.py --contexts 16384` — 2026-09-20

**Fix:** `ModelConfig(...)` em `gguf.py` não passava `slot_states=ple_slot_states(qwen4_args)` (só `parse_config`, path safetensors, fazia isso), então o pool nunca registrava `ple_ngram_ctx`/`ple_conv`. Verificado end-to-end após o fix: 4096 tok prompt / 32 decode, `--no-graph --cache-type=radix`, TG 31.36 tok/s (31/32 tokens, EOS antecipado) — número não comparável às medições de 16K acima (profundidade, decode length e graph diferentes).

**`cert_matrix.py --contexts 16384` (7 rows declaradas, 3 BLOCKED pré-existentes por geometria mista de experts):**
| row | resultado |
|---|---|
| `native-35b-a3b` | REGRESSION de guard: PP 4575.8 < 4600.0 (-0.5%); TG 158.63 >= 158.0 passou |
| `native-flash-next` | FAIL — `cache budget too small` a mem-ratio 0.86/16384 ctx (GPU livre no momento, não é contenção externa); pré-existente |
| `gguf-flash-unsloth-ud` | FAIL — stall-timeout de prefill (90s) do harness, **não** o bug do PLE (já corrigido, ver acima). Log mostra chunk de 8192/16384 tokens levando ~41s (PP ~194 tok/s, IQ4_XS sem kernel MMQ dequantiza no prefill); 2 chunks passam dos 90s fixos do `bench_pp_tg.py` |
| `gguf-qwen38-27b-iq3s` | FAIL — `CUDA out of memory` real (dense I-quant, mem-ratio 0.86 insuficiente a 16384 ctx) |

Nenhuma das 3 falhas acima foi causada pelo fix do PLE; são pré-existentes e não investigadas nesta sessão (guard de PP possivelmente desatualizado; mem-ratio insuficiente para os dois rows de OOM; `bench_pp_tg.py` precisa de `--stall-timeout` maior para IQ4_XS a 16K).

---

## `--moe-strategy hybrid` no Flash-Next NVFP4-Radix: teste do gargalo PCIe — 2026-09-20

**Motivação:** investigar se streaming de experts via PCIe é o gargalo de TG no Flash-Next. `--moe-strategy hybrid` (CPU+GPU co-compute, fração auto-tunada por PCIe/CPU bandwidth) já existia em `moe/offload_cache.py`/`engine.py:1620` mas nunca tinha sido testado — nenhuma menção em nenhum doc vivo antes desta sessão. Formato `nvfp4` e ativação `silu` já suportados pelo executor CPU (`cpu_executor.py:72`, `_CPU_MOE_ACTS`), sem bloqueio de gate.

**Par isomórfico (único comparável entre si — mesmo prompt 4096/64dec, eager `--no-graph`, `cache-type naive`, mem-ratio 0.98, `max-running-requests 1`):**

| Config | PP (tok/s) | TG (tok/s) | GPU util | RSS | SHA1 |
|---|---|---|---|---|---|
| `gpu` (default, controle) | 1771.5 | **29.61** | 98% | 70.0 GiB | `573a19610680` |
| `hybrid` (auto: 48.7% PCIe / 51.3% CPU) | 1775.9 | **30.51** | 90% | 70.5 GiB | `573a19610680` |

**Resultado:** +3% TG, bit-idêntico (mesmo SHA1 do baseline k=0). Correto, mas pequeno. **Conclusão:** hybrid moveu 51.3% dos misses de expert para fora do PCIe e TG só mudou 3% — isso limita o custo de PCIe expert-streaming a uma fração baixa (poucos %) do tempo do passo de decode no Flash-Next. GPU util fica saturado nos dois casos (98% gpu-path / 90% hybrid) — decode é **GPU-compute-bound**, não PCIe-bound nem host-bound. O gap vs 35B-A3B (6.3ms/token vs 33ms/token, mesma GPU) **não é primariamente residência de experts em VRAM** — se fosse, hybrid teria mudado TG substancialmente ao tirar carga do caminho de transferência. É mais provável que 35B-A3B tenha um forward pass por token estruturalmente mais barato (roteamento mais denso/simples, sem a pilha híbrida GDN/QSA attention do Flash-Next, menor compute por token). Não investigado a fundo — próximo passo é instrumentar `torch.cuda.Event` por região (GDN, QSA/TurboKV attention, MoE grouped GEMV) para achar o kernel dominante; `nsys`/`ncu` estão quebrados nesta máquina (Nsight não instalado para CUDA 13.3).

**Não comparar** estas duas linhas com os anchors de 34.7/30.44/23.21 tok/s de sessões anteriores — diferem em cuda-graph vs eager, kv-reserve-tokens e mem-ratio.

**Ação:** `--moe-strategy hybrid` é gratuito (sem custo de correção, +3% TG) — candidato a default para Flash-Next. Próxima hipótese a descartar antes de qualquer trabalho de kernel: `--ple-backend pinned` (tabela PLE hoje é lida de disco por token) — **testado e falhou**: processo morto durante load (RSS já ~70 GiB no path `disk`, pinned exige mais RAM residente que a máquina tem livre no mem-ratio atual). Não investigado se cabe em mem-ratio menor.

**`ft bench bw --dtype nvfp4` (mecanismo oficial de auto-resolução, 2026-09-20):** ceilings CPU STREAM read 70.4 GB/s, PCIe linear H2D 57.9 / D2H 57.3 GB/s. Real kernels nvfp4: CPU-MoE 63.0 GB/s vs PCIe-gather 53.2 GB/s = **1.19x** — abaixo do limiar padrão (2.0x) para recomendar hybrid, então `auto` resolve para `offload` nesta GPU. Overlapped (CPU+PCIe concorrentes): 40.0/38.0 GB/s → fetch split 48.7% (bate com o 48.7% medido em runtime acima). **Decisão do operador (2026-09-20): manter `offload`/`auto` como está — não baixar o limiar, não forçar `hybrid` como default.** Motivo: o +3% medido foi com 1 requisição e CPU ociosa; o limiar de 2x existe para proteger o cenário multi-requisição (CPU disputada entre GEMV, tokenização e scheduler), não medido nesta sessão. Não reabrir sem medir multi-request.

**Diagnóstico `FREETOKEN_DEBUG_SPEC_TIMING=1` no MTP k=4 (2026-09-20):** confirma que o replay batched já funciona corretamente — `gdn_replay` (fallback caro, forward extra) **nunca disparou** em 30 spec steps; `zero_replay_gdn` (sem custo de forward extra, restaura de checkpoint) cobriu 100% dos casos que precisaram de restore. Custo real por step: `draft_chain` ~11ms (4 forwards sequenciais do draft, inerente à especulação) + `verify_forward` ~140ms (cobre k+1=5 posições, ~28ms/token — em linha com o custo normal de decode ~33ms/token, não é anomalia). Com aceitação ~40-50%, ~2-3 tokens entregues por ~151ms de step vs ~33ms/token do baseline greedy — abaixo do break-even, o que explica a regressão de -7% em k=4 sem precisar de nenhum bug adicional. **Não há bug de replay a corrigir; MTP está no seu comportamento esperado dado o algoritmo.**

## Instrumentação `FREETOKEN_DEBUG_LAYER_TIMING` (mixer vs MoE por camada) e teste de kernel NVFP4 alternativo — 2026-09-20

**Instrumentação:** `torch.cuda.Event` opt-in (`FREETOKEN_DEBUG_LAYER_TIMING=1`, custo zero quando desligada — fast path idêntico ao original) em `Qwen4ExpDecoderLayer.forward` (`models/qwen4_exp/model.py`), medindo tempo de GPU do mixer (GDN/QSA attention) vs MoE por camada, print a cada 256 chamadas.

**Resultado (4096 tok prefill + 64 decode, eager, naive):** acumulado ~2476ms mixer / ~6257ms MoE em 6144 chamadas de layer → **MoE domina o custo por camada em ~2.5x sobre o mixer de atenção** (0.40ms vs 1.02ms por chamada, média mista prefill+decode). Confirma que o gargalo de compute do Flash-Next está no grouped-GEMV/dequant NVFP4 do MoE, não na atenção GDN/QSA.

**Teste de kernel alternativo:** 3 backends NVFP4 existem (`triton` nativo, `nvfp4_marlin`, `nvfp4_b12x`/flashinfer SM12x). Ambos os alternativos descartados:
- `nvfp4_marlin`: documentado como sm_80-99 only — esta GPU é sm_120 (Blackwell), incompatível por design.
- `nvfp4_b12x` (`--quant-backend moe.nvfp4=b12x`): **crash** — `ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit problem N/K=2560/640 at moe_block_size=8` (bug de tabela de tile config do flashinfer para esta geometria específica do Flash-Next, não corrigível do nosso lado sem patch upstream).

**Conclusão:** o kernel `triton` nativo (default/`auto`) é o único backend NVFP4 viável para a geometria deste checkpoint nesta GPU. Sem ganho disponível por troca de kernel MoE sem trabalho de correção upstream no flashinfer. Instrumentação de timing mantida no código (opt-in, custo zero por padrão) para uso em sessões futuras.

**Controle pós-instrumentação (flag desligada):** TG 29.60 tok/s (3 repeats), idêntico ao controle anterior (29.61) — confirma zero regressão de PP/TG pela mudança de código.

---

## MTP k=1 verify in one CUDA graph (Unsloth-IQ4_XS GGUF, 16K, turbo3) — 2026-09-23

Bench: 4K prompt, 256 tokens, 3 reps, `--cuda-graph-max-bs 1`, overlap off, `FREETOKEN_DEBUG_SPEC_TIMING=1`.

| Metric | db12f41 | 28f7ec6 (PLE fix) | verify graph |
|---|---|---|---|
| verify ms | 53.7 | 54.0 | **37.1** |
| draft / replay ms | 3.32 / 15.95 | 3.29 / 16.10 | 3.30 / 16.11 |
| acceptance | 75.2% | 72.8% | 72.8% |
| k1 TG / PP | 28.42 / 3153 | 27.85 / 3151 | **38.12** / 3153 |
| k0 TG / PP (same-day rerun) | 41.56 / 3108 | — | 41.61 / 3112 |
| peak VRAM MiB (k1) | 15868 | 15868 | 15846 |
| expert slots (k1) / target miss | 2738 / 37.9% | 2738 | 2730 / 38.4% |

nsys, per verify: launches 5975 -> 3 (1 graph, 5486 kernels). Syncs 90 -> 8. GPU idle 31.5 -> 1.9 ms.
GPU busy stays about 35 ms: `fast_index_copy` (expert misses, H2D) 17 ms, dense `mul_mat_vec_q` 7.2 ms, `moe_vec_q` 3.3 ms.
The capture takes about 1 s at startup and about 20 MiB of graph pool.
The PLE fix changes the k1 output text (sha 0a7c5a94ca -> 1eebf6a554). With graphs off, disk PLE and pinned PLE give identical output.

## Overnight campaign: GGUF IQ4_XS + MTP, 16K turbo3 (2026-09-23)

Commits a0adc18..d078f5f. RTX 5080, `--max-seq-len-override 16384 --cuda-graph-max-bs 1
--kv-format=turbo3 --max-running-requests 1`, overlap off, 256 output tokens, greedy.
Scripts and logs: `/models/desenvolvimento/ft-campaign/` (`ft.sh`, `ft16.sh`, `cold.sh`, campaign notes).

**PP definition.** The old anchors (3153/3112) are cached-prefix PP: the client repeats one prompt
and the server reuses 4032 of its 4096 tokens from the radix cache. True cold PP (unique prompt per
request, warm server) was 377 tok/s at 55ed10f. Both are reported below.

| Workload | 55ed10f | HEAD | Change |
|---|---|---|---|
| 4K, k1 TG (cached prefix) | 37.95 | **44.69** | +18% (a) |
| 4K, k0 TG (cached prefix) | 41.23 | 40.51 | noise band (b) |
| 4K, cached-prefix PP | 3127-3151 | 3209 | no regression |
| 4K, cold PP (k0 and k1) | 377 | **1472** | 3.9x |
| 15.7K occupied, cold PP | 398 | **2064** | 5.2x |
| 15.7K occupied, k1 TG (cold prompts) | 42.02 | 41.97-44.18 | (c) |
| 15.7K occupied, k0 TG (cold prompts) | 40.23 | 40.29-40.53 | = |
| 15.7K occupied, k1 TG (cached prefix) | - | 43.57-44.74 | |

(a) Batched mmvq (a two-row verify no longer re-reads dense weights), host-resident token
embedding (+290 expert slots at k1), deferred replay (rejections replay on 2% of cycles instead of
27%). (b) k0 output is bitwise unchanged by those three commits. (c) Output text changes, so
acceptance changes; spread over runs 42.0-44.7.

MTP now beats k0 on every tested workload, but >100 tok/s was **not** reached. Remaining k1 cycle
at 4K: about 40 ms = draft 3.3 + verify 36.6 (2 rows) + commit/scheduling ~1. In the verify, expert
misses copy ~16 ms over PCIe (~47 GB/s, saturated). 100 tok/s at 75% acceptance needs 17.5 ms.

Rejected with evidence: LFU and ghost-admission expert caches (miss 0.38-0.48 vs LRU 0.384; Belady
0.225); next-layer routing prefetch (44% top-10 recall); CPU hybrid for IQ3_S/IQ4_NL (no CPU kernel,
NVFP4 hybrid gave only +3%); MTP draft in a CUDA graph (illegal access on replay, unresolved).
Contexts above 16K were not run: the >100 TG gate was not met.

## Campaign 2 (2026-09-23): cold-bench contract, FTW vs GGUF, correctness, hybrid, auto-config

Base commit ea8ae8a; commits ea8ae8a..HEAD: 86eae99 (`cache_prompt:false` honored), a102c14
(teacher-forced logit dump), cd8cbe3 (AVX-512 CPU GGUF expert kernels, per-bank formats), 211efb6
(determinism fix), a8a1b6c (verify-window equality tests), af3dcdf (`ft tune` persisted profiles,
`ft bench bw` GGUF coverage), 7bbf2af (hybrid per-layer format fix), 7a297f7 (unset `--spec-mtp`
resolves from the `ft tune` profile). Model: Qwen3.8 Flash-Next, native
NVFP4-Radix vs GGUF Unsloth-IQ4_XS. Harness: `/models/desenvolvimento/ft-campaign2/bench.sh`
(one boot, 1 warmup + REPS requests, peak VRAM/RSS, refuses to start if GPU >500 MiB used or
port busy). Full workstream ledger: `/models/desenvolvimento/ft-campaign2/LEDGER.md`.

**Cold-bench contract.** `coldclient.py` sends `cache_prompt:false` + a unique `[run N]` prefix
per request so the radix cache cannot serve a cached/reused prefix; `ft-campaign2/coldclient.py`
also reports processed-token counts via `--enable-cache-report` usage field. Test:
`tests/scheduler/test_cache_prompt_cold.py`.

**Native NVFP4 vs GGUF, 4K cold (one boot each, cold prompts; native runs predate 211efb6, which
only touches the GGUF dequant-combine path):**

| Config | TG (tok/s) | Cold PP (tok/s) | Peak VRAM |
|---|---|---|---|
| Native NVFP4-Radix, k0 | 30.30 | 1624 | 15770 MiB |
| Native NVFP4-Radix, k1 (MTP) | 21.72 (below its own k0) | 1630 | 15836 MiB |
| GGUF IQ4_XS, k1 (clean control, 4K, after 211efb6) | 40.48 | 1475 | 15842 MiB (slots 2886) |
| GGUF IQ4_XS, k1 (15.7K occupied, after 211efb6) | 43.39 | 2074 | 15842 MiB (slots 2886) |

**Verdict: GGUF wins.** TG is 2.06x native at k1 and 1.34x at k0; native's +10% PP at 4K does not
offset the TG gap. Native is dominated end-to-end — kept only as a fallback, not specialized
further (no 16K/3-repeat native runs run).

**Hybrid (CPU+GPU) vs GPU offload, GGUF, k1:**

| Config | 4K TG | 15.7K TG | Cold PP (4K / 15.7K) |
|---|---|---|---|
| Offload (default) | 40.48 | 43.39 | 1475 / 2086 |
| Hybrid, fallback fetch cap 1 (before d52f287) | 11.72 | 16.30 | 1475 / 2086 (equal) |
| Hybrid, benched fetch split 82.6% (d52f287) | 37.05 | - | 1475 |

Hybrid with the benched split is 8% slower than offload (37.05 vs 40.48 at 4K) — dominated; the 3.5x gap was a missing profile lookup. Root cause: 47/48 layers'
gate_up experts are IQ3_S, whose AVX-512 CPU kernel only reaches 8-11 GB/s (latency-bound grid
gather), well under PCIe gather bandwidth, plus per-layer CPU/GPU handshake overhead. First hybrid
attempt produced garbage output (`"!!!!"`, NaN on minority-format layers) — see LESSONS.md; fixed
in 7bbf2af (per-layer GGUF expert format resolution instead of one format per whole layer).
Default is `offload`; hybrid is not recommended for this checkpoint.

**`ft bench bw` bandwidth calibration** (`benchbw-gguf.json`, merged into
`~/.cache/freetoken/benchbw/<gpu-uuid>.json`): CPU MoE alone 63 GB/s (nvfp4 AVX-512, historic) /
per-format GGUF AVX-512 (`C2b-cpu-kernel-bench.txt`): Q8_0 70-93 GB/s, IQ4_XS 59-69 GB/s, IQ4_NL
42-49 GB/s, IQ3_S 8-11 GB/s; PCIe gather alone 53 GB/s, overlapped CPU+PCIe 78 GB/s aggregate
(1.47x PCIe alone). `iq3_s+iq4_nl` CPU 14.1 vs PCIe 53.6 GB/s (ratio 0.26) → recommends offload;
`iq4_xs` 1.19x and `q8_0` 1.27x → recommends offload. This matches the measured end-to-end result
(hybrid -71%): the bandwidth-only model over-predicts hybrid (aggregate 61 > 53.6 GB/s would
suggest a win) because per-layer CPU handshake/launch latency dominates, not raw bandwidth — the
2.0x threshold was kept (conservative, consistent with every measured end-to-end point so far:
GGUF hybrid -71%, native hybrid historic +3%). Before af3dcdf, GGUF was not covered by
`ft bench bw`'s `_offload_bank_specs()` (only `bf16`/`fp8_block`/`nvfp4`/`mxfp4_triton`/`ds_fp4`);
af3dcdf added GGUF coverage by benching each layer's dominant (gate_up, down) K-quant/I-quant
pair (`_split_gguf_fmt`/`gguf_bench_key`), so the hybrid-vs-offload decision now also runs for
GGUF checkpoints through the normal auto-picker path, not only the standalone `--dtype` bench.

**Auto-config (`ft tune`, D1/af3dcdf).** `ft tune --model <path> --ctx <n>` boots each candidate
(MTP on/off, deferred replay on/off, offload vs hybrid) once, measures cold PP + committed TG +
peak VRAM, and persists the winner to a profile keyed by GPU UUID, model checkpoint path, KV
format, context-length bucket, and version/kernel source hash (a stale key is ignored, not
trusted). The launcher applies the stored env-backed choices only for flags the user left unset;
explicit CLI flags always win. `--spec-mtp` now sets `FREETOKEN_DISABLE_OVERLAP_SCHEDULING` itself
instead of hard-failing at boot, and an unset `--spec-mtp` resolves from the stored profile
(`_tuned_spec_mtp`, `server/args.py`, 7a297f7). **Not yet wired:** `moe_strategy` is stored in the
profile but `server/args.py` does not yet read it back — the hybrid-vs-offload pick still comes
only from the separate `ft bench bw` bandwidth-ratio heuristic, not this profile field.
`_cpu_moe_executor_viable` (`engine.py:1759`) only gates the *automatic* CPU-residency heuristic
(requires the checkpoint's dominant `(gate_up, down)` GGUF pair to match, i.e. `gate_up == down`);
it does not block an explicit `--moe-strategy hybrid` — HY2 measured 11.72 tok/s hybrid TG on this
mixed-format checkpoint, confirming the CPU executor does run, just slower than offload.

**Determinism fix (211efb6).** Cold prefill was non-deterministic in-process: `_fused_experts_dequant`
wrote its bf16 `index_add_` combine with non-fixed atomic order (per-expert rounding differed run
to run), producing prefill last-row logit deltas of 2.8-3.8 and different output text across
identical requests. Fixed by writing each (token, slot) row once and summing top-k in fixed order,
fp32, matching the GEMV path's temp shape. After the fix: 3 identical cold prefills bitwise equal
(same run repeated). Cold PP/TG vs the pre-fix recorded baselines: 4K improved 1386/38.45 →
1476/40.48; 15.7K PP improved 1969 → 2074 while TG was flat-to-slightly-lower, 43.51 → 43.39 (noise
band, not a regression). Slots dropped 2992→2886, attributed to 211efb6's fixed-order combine
changing the temp buffer's memory shape (cause noted in the ledger, effect on capacity not
separately quantified).

**Reproducible commands:**

```bash
# cold-bench harness (bench.sh env vars; coldclient.py always uses cache_prompt:false)
MODEL=/path/to/checkpoint CLIENT=cold PROMPT=prompt4k.txt CTX=16384 KV=turbo3 REPS=3 \
  ft-campaign2/bench.sh LABEL [--moe-strategy offload|hybrid] [--spec-mtp 1]

# per-format CPU MoE bandwidth bench (GGUF dtype tuning, not the offload auto-picker)
ft bench bw --dtype iq3_s+iq4_nl,iq4_xs,q8_0

# per-machine serve profile (measure + persist; --dry-run prints the candidate matrix only)
ft tune --model /path/to/checkpoint --ctx 16384 --kv-format turbo3 --prompt-file prompt16k.txt
```

**>100 tok/s target: not met.** Frontier committed TG on this system is ~40-44 tok/s (GGUF, k1
MTP, offload). Cost-model ledger for the k1 verify cycle (4K, ~40 ms/cycle total): draft 3.3 ms +
verify 36.6 ms, of which ~16 ms is expert-miss copies (the zero-copy UVA gather kernel, PCIe-bound,
serialized before the GEMM — no hit/miss overlap today, see C1 audit). Even a zero-copy miss path
(copies eliminated entirely from the 36.6 ms verify cost) caps the cycle at ~24 ms → a zero-copy
ceiling of about 75 tok/s at 1.8 tokens/cycle acceptance — still short of 100 tok/s; reaching >100
needs copies **and** compute cut together, or more tokens/cycle (k2/k3).

`ft tune` 16K (prompt16k, turbo3, 3 reps, matched argv, after 5de83aa/d52f287): k0 TG 40.48 / PP 2095, k1 TG 43.88 / PP 2084 -> profile picks k1 (+8.4%, above the 3% margin). A server started without `--spec-mtp` (single request, 16K, turbo3) applied it: TG 43.43, 138 verify cycles. Earlier tunes that showed a tie never ran MTP (temperature-0 requests with the model's top_p 0.95 were not greedy). At 4K cold, k1 was below k0 in the bench harness (40.48 vs 40.91). Hybrid with the correct fetch split (82.6% over PCIe): 37.05 vs offload 40.48 at 4K -> offload stays default; the 11.72 figure used a fallback fetch cap of 1.

## Campaigns 3-4 (2026-09-23): correctness outlier, overlap cost model, MTP draft graph, tune fixes

Commits 787d796..14e0478; evidence in `ft-campaign2/RELATORIO.md` / `LEDGER.md`.
- turbo3 verify-vs-decode KL 0.97 at pos 4098: cuBLAS bf16 M=1 vs M>=2 only (row-wise GEMMs make every row bitwise equal to k0). No state defect.
- Hit/miss copy overlap rejected before coding: per k1 cycle expert gathers 19.3 ms vs routed GEMV 4.0 ms (torch.profiler); ceiling < 3 ms.
- MTP draft-step CUDA graph: the Campaign 1 "illegal access on replay" is fixed (841e8fa, QSA replan on reused metadata); on by default for qwen4_exp MTP k1 (815f1eb). Cold k1 turbo3, paired prompts: 4K 40.48 -> 41.65, 15.7K 43.34 -> 44.60 (+2.9%), PP 1475/2085 unchanged, VRAM 15842 MiB, identical output.
- `ft tune` 16K (paired prompts, f90a5e2/14e0478): k0 37.77, k1 41.34 with draft graph / 40.28 without -> profile k1 + draft graph, applied on restart without `--spec-mtp`.
- >100 tok/s not met; the k1 cycle is PCIe-bound (~1 GB missed expert bytes per cycle).

## Referência Completa

`old/docs/freetoken-next/PERFORMANCE.md` — Tabelas detalhadas por config/modelo, EXP-001 a EXP-045, metodologia, variáveis de controle.
## Campaign 7 (2026-09-23/24): matched cold GGUF controls

Qwen3.8 Flash Next Unsloth IQ4_XS, RTX 5080, HEAD `437b5a6`, turbo3, 16K context, cold 4101-token prompt, zero prefix reuse, 256 output tokens: k0 36.60 TG/1456 PP; automatic k1 42.53 TG/1466 PP. Single observations, not replacement anchors. The 96-token graph trace measured 27.39 TG with synchronized counters/profiler and is diagnostic only. See `/models/desenvolvimento/ft-campaign2/campaign7/TRACE-ANALYSIS.md`. Later trace cycles averaged 0.847 GB misses; a conditional all-copy-removal model projects 70.47 tok/s under explicit assumptions. Y=2/Y=4 changed small dense kernels only; no end-to-end candidate was accepted.
## Gate 1 control note (2026-09-24)

No new performance anchor. A graph-on k0 trial at HEAD `3748fbd` was invalid for cold comparison because repetitions 2/3 reused 4032 prompt tokens. Preserve existing anchors; rerun with a fresh server and cold cache per repetition before updating this table.
Cold-reset Gate 1 result (2026-09-24): graph-on, fresh server/cache each repetition, 3 runs each. k0 PP/TG `1464.9/44.34`; k1 `1464.0/36.28` tok/s. Per-mode output hashes were stable, cross-mode hashes differed. k1 is 18.2% slower; no production change accepted.
Output-length control (2026-09-24): same greedy options, fresh graph-on server/cache. At 128 decode tokens k0/k1 TG was `47.74/41.58`; at 256 it was `47.30/43.80`. First decoded divergence occurred at output boundary offset 83 for both lengths. No Gate 2 or optimization.
k1 correctness gate (2026-09-24): decode 256 cold controls k0 graph `47.30`, k1 eager `38.95`, k1 graph `43.80` tok/s. k1 eager/graph output hashes match each other but differ from k0. Performance data is diagnostic only; k1 remains disabled pending exact token/state trace.
No performance requalification. Token trace is diagnostic only; k1 remains blocked. Final CI has unrelated/current failures in prefill-hit-D2D and spec-reject page tests.
Token-level correctness run (2026-09-24): cold graph k0 TG `47.25`, k1 TG `43.88`; PP `1465.3/1464.2`. Trace valid: 256 k0 IDs, 256 k1 committed IDs. First committed divergence at index 26 (`1404` vs `11`), with 26-token committed prefix equal. No performance qualification or Gate 2.
TG attribution (2026-09-24): k1 accepted 90/165 drafts (54.5%), averaging 1.545 committed tokens/cycle. This lower acceptance explains substantial TG cost. At mismatch index 26, target score margins are only `0.125` in k0 and k1 with opposite top IDs; no graph-specific or acceptance defect proven. No performance requalification.
Historical acceptance values are not comparable to current GGUF IQ4_XS: post-PLE `28f7ec6` lacks prompt bytes/hash and full flags; the 49.2% depth probe uses 96 output and synchronized diagnostics. Current k1 acceptance is 54.5% (90/165), 1.545 committed/cycle. TG deficit attribution remains acceptance-driven but no whole-cycle timing claim is made.
Gate 3 attribution (2026-09-24, graph-on, synchronized stage timing, diagnostic only): per k1 cycle, verify 31.98 ms, draft 1.88 ms, bookkeeping 0.60 ms and replay 1.13 ms, for 35.6 ms; the k0 step is 21.2 ms. At 54.5% acceptance (1.545 tok/cycle), k1 reaches 0.92x k0; break-even is about 68% acceptance. The index-26 divergence is numerical row-count arithmetic (Gate 2), not a defect.
Campaign 9 k1 economics (2026-09-24, HEAD `2ab0db9` + uncommitted opt-in token trace with trace-only `draft_prob`): one cold graph-on 4096/256 k1 run, fresh server, greedy, `bench_pp_tg.py --tokens 4096 --decode 256 --repeats 1 --warmups 0 --prompt-file .../mtp-depth/probe-prompt4k.txt --prompt-file-exact --fresh-server-each-repeat --serve-arg '--spec-mtp 1' --token-trace ...`. Output hash `91e6de9c85b2` unchanged; 165 cycles, 90 accepted (54.5%); TG 38.37 is trace-instrumented, not a throughput claim. Verify cost by rows (campaign8 timing): 2 rows 29.6–32.0 ms, 3 rows 32.5, 4 rows 35.1; k0 1-row 21.2 ms. Draft top-1 probability vs acceptance at p=0: <0.3 23% (n=31), 0.3–0.5 47%, 0.5–0.7 58%, 0.7–0.9 80%, ≥0.9 100% (n=10). Confidence gate (skip verify at p=0, fall back to 1-row decode + wasted draft): oracle +4.8% vs k0, best realistic threshold (0.60) −1.3%. Per-segment k0/k1 switching oracle with future knowledge and zero switch cost: W=4 +5.1%, W=8 +1.2%, W=16 +0.3%. Verify reduction needed for +5% at 1.545 tok/cycle: cycle ≤31.1 ms, i.e. verify −4.5 ms (−14%); every remaining lever (residency, prefetch, overlap, zero replay, device accept, MoE kernel/layout) is a recorded no-go. Greedy acceptance is fixed by draft argmax, so temperature/top-k/filtering cannot raise it without weakening determinism. Decision: NO-GO for all Gate 2 candidates; no implementation, no A/B. k0 stays default. Limiting bound: 2-row verify ≈1.5x a 1-row step against 1.545 tok/cycle; k1 needs ≈68% acceptance (a better MTP head) to pay off. Scripts: `ft-campaign2/campaign9/{gate_bound,switch_bound}.py`.
Campaign 10 k0 graph-on attribution (2026-09-24, HEAD `9a640ef`): Gate A removed the dead unsafe `_adaptive_mtp_controller` branch from `spec.py` and landed the opt-in token trace; `make ci` 2111 passed. Control `bench_pp_tg.py --model /models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS --tokens 4096 --decode 256 --repeats 1 --warmups 0 --prompt-file .../mtp-depth/probe-prompt4k.txt --prompt-file-exact --fresh-server-each-repeat --serve-arg '--spec-mtp 0'` (prompt SHA256 `645f46bf…a83dbf9…fcf`): PP `1466.7`, TG `47.28`, VRAM `14.70 GiB`, RSS `80.31 GiB`, graph `[1]`, expert_slots `3519`, hash `3af3056b98c0`. Same command under nsys (`-t cuda,nvtx -s none --cuda-graph-trace=node`): TG `46.52` (−1.6% overhead), same hash; 158/255 steady-state replays captured before SIGTERM truncation. Per step 20.24 ms: expert gather `fast_index_copy` 7.57 ms (37%, ≈52 GB/s = PCIe gather ceiling), dense q8_0 GEMV 5.16 ms (25%, ≈745 GB/s VRAM), cuBLAS bf16 1.23, routed GEMV 1.79, lm_head 0.59, GDN 0.35, QSA 0.11, host gap 0.84 ms (4.1%); single stream. Gate C NO-GO: shared-expert/gather overlap ceiling ≈0.34 ms (1.7%), host gap <5% even if removed, other levers already no-go. k0 default unchanged. Evidence `/models/desenvolvimento/ft-campaign2/campaign10/ATTRIBUTION.md`.

Campaign 11 Gate 1 byte model (2026-09-24, HEAD `3f3ba92`, no source change, no server run): LRU replay of the 4K trace (k0 proxy, scaled to measured 7.57 ms gather) plus one graph-replay MMVQ microbenchmark of the full dense set: Q8_0 6.88 ms, Q6_K 6.48, Q5_K 4.86, Q4_K 4.47. Optimistic TG ceilings vs 47.28: idle MTP pool + token_embd reclaim +2.7%; spending the memory-ratio margin +5.2% (rejected); dense Q6_K +3.6% (rejected); AD-4.27 IQ2_S experts +9.5..10.5%; dense Q5_K +11.5%; Q4_K +14.1%. Surviving options need a lossy conversion (no qwen4exp quantizer on host) or an unlisted checkpoint, both with unmeasured quality; no A/B run. Evidence `/models/desenvolvimento/ft-campaign2/campaign11/GATE1.md` (sha256 ceil.py `4081a99a27f7`, sim.py `9841eff40f44`, mmvq_bench.py `42a000db9d31`, ceilings.txt `d1112434fba4`).

Campaign 11 Gate 3 (2026-09-24, HEAD `5c82c24`, operator-authorized): AD-4.27 GGUF (IQ2_S gate/up on 36/48 layers) cold graph-on k0 4096/256, fresh servers, order AD/UD/AD: TG 49.68/47.24/49.54, PP 1591.6/1464.3/1588.3, slots 4135 vs 3519, VRAM 14.82 vs 14.70 GiB. Mean +4.97% TG: NO-GO (below 5%, unlisted checkpoint, quality unmeasured). UD control hash `3af3056b98c0` matches campaign 10. Evidence `ft-campaign2/campaign11/GATE1.md`.

Campaign 12 audit (2026-09-24, HEAD `0e9d00c`, read-only, no GPU job): every >1% claim in the repo, git history, untracked reports and `ft-campaign2/` reconciled in `/models/desenvolvimento/ft-campaign2/campaign12/AUDIT.md`. No new anchor. All implemented gains are already inside the 47.24 k0 anchor. Live short-context k0 levers: AD-4.27 experts (measured +4.97%, quality unmeasured), dense Q5_K (ceiling +11.5%, absent), MTP-pool reclaim + host `token_embd` (lossless, ceiling +2.7%). Composition: naive ceiling sum +30.5%, multiplicative +34.0%, overlap-adjusted byte ceiling +25.1%, realistic +8..+15%, pessimistic ~+6% (all conditional on KL gates and authorization); +0..+2% without them. The 15-20% claim is not supported. Stored `ft tune` profiles match no key at HEAD, so k0 is the effective default.

## Campaign 13 (2026-09-24, `b5d4243`): engine fixes + existing-model qualification
UD-IQ4_XS k0 4K anchor unchanged after fixes (TG 47.28, hash `3af3056b98c0`). KV format (UD, k0): 4K auto/turbo3/turbo4 TG 47.46/42.10/39.27; 16K 45.55/39.68/39.98 — turbo formats are a capacity lever, not speed. ISTA GSQ-RCO IQ3_XXS (existing file, Q2_0 support added): 4K PP 1651.5 / TG 59.06, 16K PP 2360.1 / TG 58.49, 5555 expert slots, 20/20 usage tasks. 16K chunk cap 4096: TG +3.4%, PP -34% (rejected). Ledger: `/models/desenvolvimento/ft-campaign2/campaign13/LEDGER.md`.
MTP heads (campaign 13, cold graph-on): ISTA k1 16K sq8 TG 63.65 vs k0 58.49 (+8.8%, accept 86.8%); UD k1 16K sq8 49.27 vs 45.55 (+8.2%, 84.8%). ISTA 4K k1: Q4 heads 60.14 (+1.8%, 61.4%), Q8 heads 56.6 (-4%, 58.8%); k2 4K 42.78 (-28%). Acceptance depends on prompt content; usage eval after tokenizer fix: UD/AD/ISTA 20/20.

### Final production-readiness audit (2026-09-24)

| Area | Result | Evidence / blocker |
|---|---|---|
| UD k0 4K | PASS | PP 1463.5, TG 47.14, hash `3af3056b98c0`; anchor preserved |
| ISTA sq4 k1 16K | CONDITIONAL | TG 64.99, acceptance 88.1%; k1 remains opt-in |
| UD k1 16K | CONDITIONAL | TG 49.10-49.27, acceptance 84.8%; 4K rerun timed out before summary |
| AD k1 | NO-GO | 70.5% acceptance at 16K; 59.4% at 4K |
| NVFP4 k1-k3 | NO-GO | k1 45.7% acceptance; k2/k3 reproducible traceback/OOM |
| KV tiering 64K-256K | **PASS (campaign 16, qwen4_exp only)** | ISTA IQ3_XXS default `auto` fp8 RAM tier: 0 traceback, needle 64/128/256K + usage 20/20 + vision pass, TG +3.3% @128K / +21.5% @256K vs all-VRAM. Gated by `kv_ram_tier_certified`; other families stay all-VRAM. See "Campaign 16" below and the campaign-16 ledger under `ft-campaign2/campaign16/` |

The release gate is **NO-GO** until KV placement is implemented with graph/eager parity, rollback, and fresh 64K-256K evidence. k0 remains the only global default.

## Campaign 16 (2026-09-25): ISTA IQ3_XXS — KV-in-RAM as the certified default (HEAD `0288564`)

Model `/models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS`. New default = `--kv-tiering auto` on a certified family resolves to the fp8 host-RAM KV tier (server log: `KV RAM tier dtype: torch.float8_e4m3fn`), 0 tracebacks. Cold, fresh server, `--decode 256 --repeats 1`. k0 serve empty; k1/k2 add only `--spec-mtp`.

| TG tok/s | 16K (test-only) | 64K | 128K | 256K |
|---|---|---|---|---|
| default k0 (fp8 RAM) | 56.18 | 56.40 | 57.44 | 52.36 |
| all-VRAM k0 (c15) | 56.93 | 57.11 | 55.61 | 43.11 |
| Δ k0 | -1.3% | -1.2% | **+3.3%** | **+21.5%** |
| k1 | – | 51.40 | 57.95 | 52.47 |
| k2 | – | – | 43.14 | 33.87 |

MTP ladder (drop a context's k when TG < that k0): 64K -> k0 best (k1 51.40 < 56.40); 128K/256K -> best k1 but only +0.9%/+0.2% over k0 — the fp8 RAM tier absorbs most of the decode bandwidth MTP used to buy. Accept: k1 64K 202/306, k1 128K 216/292, k1 256K 222/288. TG-curve 128K (`--decode 8192`) mean 56.27, no cliff. Quality: needle 64/128/256K PASS, usage 20/20, vision PASS. Forced-tier turbo speed/quality (reserve 8192): turbo8/4/3 all usage 20/20 + needle 128K pass; fp8 (59.16/57.86/52.43) dominates turbo8; turbo4/3 kept as RAM-fit fallbacks. Detail in the campaign-16 ledger under `ft-campaign2/campaign16/`.

Verdict: **PASS** — KV-in-RAM default certified for the qwen4 (Qwen3.8 Flash Next) family. Dense 27B and MoE 35B remain all-VRAM under `auto` until separately measured.

## Campaign 18 (2026-09-25): MoE 35B (Tiel UD-IQ4_XS) KV placement, k0

| ctx | bf16 VRAM (fi) TG | RAM zero-copy bf16 TG | turbo4 VRAM TG (split fix `9e63d1d`) |
|---|---|---|---|
| 4K | 133.01 | 107.10 | 100.53 |
| 16K | 129.52 | 69.98 | 92.74 |
| 64K | 110.21 | 27.72 | 65.42 |
| 256K | 69.68 (4432 expert slots) | - | 31.51 (turbo3 34.81) |

Verdict: bf16 in VRAM stays the default for dense-attention 35B. Triton backend bf16 after the split fix: 4K/16K/64K 131.25/126.90/110.16 (FlashInfer parity). Ledger: `ft-campaign2/campaign18/LEDGER.md`.

## 2026-09-26 campaign 23 (re-baseline + correctness fixes; ledger ft-campaign2/campaign23/LEDGER.md)

Current-tree A/B on a FIXED served prompt (bench_pp_tg.py --tokens 16384 now serves 16384 tok; campaign 22's anchors served 15715 tok from the older harness state — absolute TG across the two campaigns is NOT comparable, only same-session deltas are).

| config (16384 tok served, 256 gen, greedy, k0) | TG cold | TG warm | PP cold | VRAM | sha1 |
|---|---|---|---|---|---|
| ISTA baseline (--spec-mtp 0, MMV_Y=4 in tree) | 52.55 | 52.93 | 2508 | 14.70 GiB | 76a5508fd576 |
| ISTA winning stack (+ratio0 +pool-caps) | 53.71 | 53.85 | 2506 | 14.58 GiB | 76a5508fd576 |
| AD baseline (--spec-mtp 0) | 42.04 | — | 2485 | 14.32 GiB | 76a5508fd576 |

- Winning-stack delta on the robust prompt: +2.2% cold / +1.7% warm (vs +7.8%/+6.1% at c22's 15715-tok prompt) — the trace-tuned pool caps are prompt-overfit; kept (bit-exact, -0.12 GiB VRAM).
- ISTA warm == cold sha at default chunk 8192 (no divergence); the c22 warm divergence needs chunk != 8192 (open bug, isolated).
- AD cold single-shot at 16384 served tokens drops hard vs c22 (42.04 vs 54.95 @15715): AD is gather-bound (41% PCIe) and loses more to the bigger working set; needs warm repeats before any verdict (not run — campaign stopped by operator).
- `--moe-pool-caps` semantics FIXED this campaign: caps are proportional weights, the planned cache_size stays the byte-budget authority (was: sum(caps) silently overrode the budget, VRAM-guard shrinks were no-ops -> rebuild churn; the c22 "64.5-65.1 TG @ 105% caps" run was that churn producing a divergent output sha — an invalid result, never a real speed).
