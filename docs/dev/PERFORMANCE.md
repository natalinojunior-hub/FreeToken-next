# PERFORMANCE — freetoken-next

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
