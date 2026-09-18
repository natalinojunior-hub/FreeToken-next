# PERFORMANCE — freetoken-next

**Todas as métricas em hardware real** (RTX 5080 15.51 GiB VRAM, SM120, 96 GB DDR5, NVMe). Source: `old/docs/freetoken-next/PERFORMANCE.md` (544 linhas).

---

## Anchors de Regressão Imutáveis

| Workload | PP (tok/s) | TG (tok/s) | VRAM | RAM (RSS) | GPU Util | Guard |
|----------|------------|------------|------|-----------|----------|-------|
| **35B-A3B NVFP4 @ 16K** | **4611** | **158.8** | 14.98 GiB | ~20 GiB | 99.8% | PP≥4600, TG≥158 |
| **Flash-Next NVFP4 @ 16K** | **1858** | **28.7** | 14.86 GiB | 67.8 GiB | 99.99% | PP≥1850, TG≥28.5 |
| **35B-A3B @ 128K** | 3189 | 89.3 | 14.4 GiB | 22.0 GiB | — | — |
| **35B-A3B @ 256K** | 2354 | 63.8 | 14.5 GiB | 22.0 GiB | — | — |
| **Flash-Next @ 128K** | 1376 | 4.86 | 14.84 GiB | — | — | — |

---

## Turbo4 + MTP Certified (EXP-041/045)

| Config | PP | TG | VRAM | sha1 | Notes |
|--------|----|----|------|------|-------|
| Turbo4 + MTP=1 @ 16K | 1714 | 24.7 | ~15 GiB | `614aa7bcdf59` | Split-kernel, eager |
| Turbo4 + MTP=2 @ 16K/64dec | — | 25.6-26.2 | — | `ed45eb6cc897` | 2.82 tok/step, 86.4% 2/2 |
| MTP carry shift (EXP-043) | — | — | — | — | Draft accept 90.9% |

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

## Referência Completa

`old/docs/freetoken-next/PERFORMANCE.md` — Tabelas detalhadas por config/modelo, EXP-001 a EXP-045, metodologia, variáveis de controle.