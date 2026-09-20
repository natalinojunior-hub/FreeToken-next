# HARDWARE_TUNING — Guia de Tuning por Hardware freetoken-next

**Hardware alvo:** RTX 5080 (15.51 GiB VRAM, SM120 Blackwell, PCIe 4.0), 96 GB DDR5, NVMe
**Base:** FreeToken v0.1.3 (`cac247a`)

---

## Variáveis de Ambiente Obrigatórias

```bash
export CUDA_HOME=/models/outros/cuda-13.3
export PATH="/models/outros/cuda-13.3/bin:$PATH"
export LD_LIBRARY_PATH="/models/outros/cuda-13.3/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0;12.0a"     # SM120 = RTX 5080
export TMPDIR=/models/desenvolvimento/tmp      # Disco real 1.3+ TB, NÃO tmpfs
export FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1  # Evita deadlock MTP + cuda-graph
export MAX_JOBS=24                             # Para compilação paralela
```

---

## Tabelas de Tuning por Modelo/Contexto

### Qwen3.8-Flash-Next-NVFP4-Radix (Flash-Next 290B, 48 layers, 256 experts)

| Contexto | `num_tokens` | `memory_ratio` | `cache_type` | `kv_format` | `spec_mtp` | PP (tok/s) | TG (tok/s) | VRAM (GiB) | Notas |
|----------|--------------|----------------|--------------|-------------|------------|------------|------------|------------|-------|
| 16K | 16512 | 0.86 | naive | turbo4 | 0 | 1858 | 28.7 | 14.86 | Baseline Triton+BF16 |
| 16K | 16512 | 0.86 | naive | turbo4 | 1 | 1714 | 24.7 | ~15.0 | Split-kernel, steady |
| 16K | 16512 | 0.86 | naive | turbo4 | 1 | 1710 | 8.22 | 14.76 | Split-kernel, cold |
| 128K | 131072 | 0.86 | naive | turbo4 | 0 | 1376 | 4.86 | 14.84 | Long-context certificado |

**Flags serve recomendadas 16K MTP=1:**
```bash
ft serve --model /models/Qwen3.8-Flash-Next-NVFP4-Radix \
  --num-tokens 16512 \
  --cache-type naive \
  --kv-format turbo4 \
  --spec-mtp 1 \
  --cuda-graph-max-bs 0 \
  --memory-ratio 0.86
```

### Qwen3.6-35B-A3B-NVFP4-FT (35B MoE, 8 experts ativos)

| Contexto | `num_tokens` | `memory_ratio` | `cache_type` | PP (tok/s) | TG (tok/s) | VRAM (GiB) |
|----------|--------------|----------------|--------------|------------|------------|------------|
| 16K | 16576 | 0.9 | naive | 4611 | 158.8 | 14.98 |
| 128K | 131072 | 0.9 | naive | 3189 | 89.3 | 14.4 |
| 256K | 262144 | 0.9 | naive | 2354 | 63.8 | 14.5 |

**Flags serve recomendadas 16K:**
```bash
ft serve --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
  --num-tokens 16576 \
  --cache-type naive \
  --memory-ratio 0.9
```

---

## Parâmetros Críticos

### `memory_ratio` — Trade-off VRAM vs TG
| Valor | Uso | Efeito |
|-------|-----|--------|
| 1.0 | Ceiling máximo | Mais KV context, risco OOM |
| **0.86** | **Hand-tuned (recomendado)** | ~320MiB headroom resgatado para pool KV/experts |
| 0.9 | Default | Equilibrado |
| <0.8 | Conservativo | TG cai, mais seguro |

### `num_tokens` — Piso Obrigatório
- **NUNCA** omitir — `--moe-cache-auto` decidia split ANTES de parse
- Flash-Next 16K: `16512` (16384 prompt + 128 decode)
- 35B-A3B 16K: `16576` (16384 prompt + 192 decode)
- Long-context: `prompt_tokens + decode_tokens + 128` margem

### `cache_type`
| Tipo | Uso | Nota |
|------|-----|------|
| **naive** | Benchmarks, certificação | Sem prefix cache fake PP |
| radix | Produção, multi-request | Prefix sharing real |

### `kv_format`
| Formato | Compressão | VRAM savings | Compatibilidade |
|---------|------------|--------------|-----------------|
| **turbo4** | 4-bit | ~4x vs BF16 | Flash-Next, 35B-A3B |
| bf16 | Nenhuma | Baseline | Todos |

### `spec_mtp` (MTP depth)
| Valor | Draft tokens/step | TG gain | Accept-rate | Caveat |
|-------|-------------------|---------|-------------|--------|
| 0 | 0 (baseline) | 1x | N/A | Determinístico |
| **1** | **1** | **~2x** | **~90% (com carry shift)** | Requer `--cuda-graph-max-bs 0` |
| 2 | 2 | ~2.8x | 86.4% full 2/2 | Prefill-window warm-up gap |
| 3 | 3 | ~2.5x | 1.9% full 3/3 | Regressão TG pós clear_mtp_slot |

### `cuda_graph_max_bs`
| Valor | Efeito |
|-------|--------|
| **0** | **Desabilita cuda-graph (obrigatório para MTP)** |
| >0 | Habilita, mas trava com MTP + overlap scheduling |

---

## MoE Offload Tuning

| Parâmetro | Default | Tuned | Efeito |
|-----------|---------|-------|--------|
| `_SMALL_BANK_FEAT_BYTES` | 256 KiB | **64 KiB** | Funde mais bancos quantizados, menos cópias síncronas |
| `hybrid_max_fetch` | 1 | **4** | Paralelismo fetch H2D 4x |
| `hybrid_fetch_fraction` | 1.0 | **0.5** | Overlap fetch/compute |
| `moe_cache_size` | auto | 0 (offload) | VRAM para KV, experts em host RAM |

```bash
export FREETOKEN_MOE_SMALL_BANK_FEAT_BYTES=65536
export FREETOKEN_HYBRID_MAX_FETCH=4
export FREETOKEN_HYBRID_FETCH_FRACTION=0.5
```

---

## VRAM Ledger Reservas (engine/vram_ledger.py)

| Reserva | Original | Otimizado | Ganho |
|---------|----------|-----------|-------|
| `TRITON_AUTOTUNE_ARENA` | 256 MiB | **128 MiB** | 128 MiB |
| `GRAPH_CAPTURE_PEAK` | 256 MiB | **128 MiB** | 128 MiB |
| `FRAGMENTATION_RESERVE` | 128 MiB | **64 MiB** | 64 MiB |
| **Total** | 640 MiB | **320 MiB** | **~320 MiB** |

---

## Checklist Pré-Benchmark por Hardware

### RTX 5080 (16GB)
- [ ] `free -h` → RAM livre ≥ 20 GiB
- [ ] `nvidia-smi` → VRAM livre ≥ 14 GiB, 0 processos
- [ ] `du -sh /tmp/*` → < 1 GiB
- [ ] `export TMPDIR=/models/desenvolvimento/tmp`
- [ ] `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`
- [ ] `--cuda-graph-max-bs 0` se MTP>0
- [ ] `--num-tokens` explícito = piso
- [ ] Mínimo 3 repeats, warmups declarados
- [ ] JSONL output salvo

### Outras GPUs (RTX 30/40 series)
- Ajustar `TORCH_CUDA_ARCH_LIST`: 8.0 (Ampere), 8.9 (Ada)
- `memory_ratio` pode precisar ser menor (ex: 0.8 para 12GB)
- `num_tokens` proporcional à VRAM

---

## Referências
- `PERFORMANCE.md` — Tabelas completas PP/TG/VRAM/RSS
- `RUNBOOKS.md` — Procedimentos "modelo não carrega", "bench falha determinismo"
- `QA.md` — Gates de validação e checklists
- `EXPERIMENTS.md` — EXP-043 (MTP carry shift), EXP-045 (MTP2 cert), EXP-048 (detokenize fix)
