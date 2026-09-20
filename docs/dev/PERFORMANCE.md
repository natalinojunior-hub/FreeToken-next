# PERFORMANCE — freetoken-next

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

LTO (`llama-turbo-optimal`) é um fork do llama.cpp: só roda GGUF. Não executa o checkpoint NVFP4-Radix do FreeToken. Os próprios docs do LTO (`QWEN38_FLASH_MTP_REFERENCE_BASELINE.md:53-54`) afirmam que qualquer smoke-test do FreeToken é "non-isomorphic to GGUF LTO". Não há como fazer a comparação como originalmente formulada (mesmo quant NVFP4+Turbo4 nos dois lados) — LTO nesse quant não existe.

Números que existem, nenhum comparável 1:1 entre si (contexto/quant/config diferentes):
- **LTO, target-only (sem MTP), Unsloth-IQ4_XS GGUF, KV q8_0, prompt curto**: 38.41 tok/s (não é 16K, não é NVFP4).
- **LTO, ctx=16384, MTP n_max=3 (especulativo)**: 38.07 TG — não comparável ao FreeToken sem MTP.
- **FreeToken 0.1.2+gaf71ba432 (versão antiga), NVFP4-Radix, boot "smoke" não certificado**: 34.7 tok/s decode. **Comparação com o número "atual" abaixo invalidada (2026-09-20):** o `stats.json` real desse boot mostra `max_seq_len_override=4096`, `prompt_tokens_total=22`, `completion_tokens_total=31` — decode com KV praticamente vazio, não "ctx=8192 preenchido" (`kv_reserve_tokens=8192` no ServerArgs é reserva de VRAM, não profundidade usada). O ServerArgs do 0.1.2 também não tem campo kv-format/turbo4 (KV puro, `cache_type='radix'`) e rodou com CUDA graph ligado. **Reproduzido de novo nesta sessão, mesma venv pinada em 0.1.2, mesmo comando: 34.7 tok/s idêntico** — o número é real e estável para essa condição, só não é comparável ao número abaixo.
- **FreeToken atual (WIP, eager `--no-graph`, ctx=8192, mem-ratio 0.85, 1089 slots)**: **TG 23.94 tok/s** (3 repeats, 23.93-23.94), quase idêntico ao número em 16K (23.21). **Não é controle válido contra o número do 0.1.2 acima**: profundidade de KV diferente (~22-53 tokens vs 8192 preenchidos), possível turbo4 vs KV puro, e graph ligado vs `--no-graph`. Nenhuma regressão foi de fato medida entre as duas versões ainda — para isso seria preciso isolar commit, profundidade de KV, kv-format e caminho graph/eager, os quatro ao mesmo tempo. Nota lateral: RSS 70.7 GiB aparece também nessa rodada bem-sucedida (sem stall) — não é assinatura exclusiva do travamento do CUDA graph, é o footprint normal do host-bank cache do MoE atual.
- **FreeToken atual (WIP, eager `--no-graph`, ctx=16384, mem-ratio 0.85, 1089 slots)**: 23.21 tok/s — melhor número válido de ponta a ponta obtido nesta sessão.
- **FreeToken atual (via CUDA graph, âncora histórica k=0)**: 27.73 tok/s — mas o caminho de CUDA graph está quebrado tanto em HEAD (`qsa_sparse.py:358` crash) quanto no WIP (hang em `replay()`), logo não é reproduzível agora.

**Conclusão:** a queda observada de FreeToken hoje (23.21 eager) vs a âncora histórica (27.73 via graph) é explicada pela combinação de duas variáveis não isoladas: decode eager (vs CUDA graph, hoje quebrado) **e** cache de experts reduzido (1089 vs ~1287 slots, por causa do mem-ratio 0.85 vs 0.9). Não atribuir a diferença inteira a nenhuma das duas isoladamente. Em nenhum caso é LTO estruturalmente superior num quant equivalente — a comparação justa pedida originalmente não é possível com os artefatos disponíveis (ver acima).

---

## Referência Completa

`old/docs/freetoken-next/PERFORMANCE.md` — Tabelas detalhadas por config/modelo, EXP-001 a EXP-045, metodologia, variáveis de controle.