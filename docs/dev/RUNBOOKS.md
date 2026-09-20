# RUNBOOKS — Procedimentos Operacionais freetoken-next

**Última atualização:** 2026-09-19 | **Hardware:** RTX 5080 16GB / 96GB RAM

---

## Índice
1. [Modelo não carrega / OOM no serve](#1-modelo-não-carrega--oom-no-serve)
2. [Bench falha determinismo (sha1 diverge)](#2-bench-falha-determinismo-sha1-diverge)
3. [Kernel JIT não compila / SegFault](#3-kernel-jit-não-compila--segfault)
4. [MoE offload earlyoom / host RAM exausto](#4-moe-offload-earlyoom--host-ram-exausto)
5. [Servidor trava / não responde](#5-servidor-trava--não-responde)
6. [VRAM Ledger ceiling excedido](#6-vram-ledger-ceiling-excedido)
7. [Checkpoint GGUF não carrega / geometria inválida](#7-checkpoint-gguf-não-carrega--geometria-inválida)

---

## 1. Modelo não carrega / OOM no serve

### Sintoma
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
CacheManager integrity check failed
```

### Diagnóstico
```bash
free -h          # RAM livre ≥ 20 GiB?
nvidia-smi       # VRAM livre ≥ 14 GiB? Processos zumbis?
du -sh /tmp/*    # tmpfs < 1 GiB?
```

### Causas Comuns
| Causa | Fix |
|-------|-----|
| `--num-tokens` não explícito (padrão 8192) | `--num-tokens 16512` (Flash-Next 16K) ou `33152` (35B 16K) |
| `memory_ratio` alto demais | `--memory-ratio 0.86` (hand-tuned) ou `0.9` (default) |
| Cache tipo errado | `--cache-type naive` (evita prefix cache fake PP) |
| `--moe-cache-auto` decide antes de `--num-tokens` | `--num-tokens` é piso obrigatório, nunca sugestão |

### Validação
```bash
make preflight
ft serve --model /models/Qwen3.8-Flash-Next-NVFP4-Radix --num-tokens 16512 --cache-type naive --memory-ratio 0.86
```

---

## 2. Bench falha determinismo (sha1 diverge)

### Sintoma
```
sha1 k=0: 17f277f43565
sha1 k=1: 8595ff5f6a83  (diverge)
```

### Diagnóstico
```bash
# 1. Rodar baseline k=0
make bench
# 2. Rodar k=1 mesmo prompt
FREETOKEN_DEBUG_EXP048=1 python benchmarks/bench_pp_tg.py --spec-mtp 1 --decode 16 --tokens 16384 --repeats 1 --json /tmp/det.jsonl
# 3. Comparar logs: grep "exp048" /models/desenvolvimento/tmp/bench-pp-tg-*.log
```

### Causas Comuns
| Causa | Fix | Arquivo |
|-------|-----|---------|
| Detokenize batch múltiplos tokens mesmo UID | Processar sequencialmente | `python/freetoken/tokenizer/detokenize.py:91` |
| Estado KV não zerado entre requests | `clear_mtp_slot()` em `free_req` | `python/freetoken/kvcache/qsa_pool.py` |
| Ring/scratch não restaurado | `_restore_qsa_state` todas camadas | `python/freetoken/scheduler/spec.py:101` |

### Validação
```bash
# 2 processos servidor SEPARADOS, 1 request cada
python benchmarks/bench_pp_tg.py --decode 4 --repeats 4 --warmups 0 --json /tmp/det1.jsonl &
python benchmarks/bench_pp_tg.py --decode 4 --repeats 4 --warmups 0 --json /tmp/det2.jsonl &
# sha1 devem ser idênticos cross-process
```

---

## 3. Kernel JIT não compila / SegFault

### Sintoma
```
ptxas fatal: Out of memory in register allocation
Triton kernel compilation failed
Segmentation fault (core dumped)
```

### Diagnóstico
```bash
# Verificar se cache JIT corrompido
rm -rf ~/.cache/triton ~/.cache/nvcc
# Recompilar extensões C++
make rebuild
```

### Causas Comuns
| Causa | Fix |
|-------|-----|
| Kernel fundido muito grande (Turbo4 + QSA) | Split kernel: decompressão separada (`kernel/triton/qsa/decompress.py`) |
| `TORCH_CUDA_ARCH_LIST` errado | `export TORCH_CUDA_ARCH_LIST="12.0;12.0a"` (RTX 5080 = SM120) |
| `CUDA_HOME` não aponta para 13.3 | `export CUDA_HOME=/models/outros/cuda-13.3` |

### Validação
```bash
make rebuild
make test
```

---

## 4. MoE offload earlyoom / host RAM exausto

### Sintoma
```
earlyoom: killed process python (pid 12345)
free -h mostra 8-9GB usado mas pico real era ~69GB
```

### Diagnóstico
```bash
free -h
# earlyoom recalcula "user mem total" descontando tmpfs/shm
df -h /tmp          # tmpfs = RAM real!
ps aux --sort=-%mem | head -20
```

### Fix Imediato
```bash
# Limpar tmpfs
rm -rf /tmp/*
# Desabilitar tmpfs permanentemente (requer sudo)
sudo systemctl mask tmp.mount
sudo reboot
```

### Prevenção
```bash
export TMPDIR=/models/desenvolvimento/tmp  # disco real 1.3+ TB
# SEMPRE antes de ft serve, ft checkpoint, pytest
```

---

## 5. Servidor trava / não responde

### Sintoma
```
curl /v1/completions timeout
nenhum log novo por > 60s
```

### Diagnóstico
```bash
# Verificar se processo vivo
ps aux | grep ft
# Verificar GPU
nvidia-smi
# Verificar logs
tail -f /models/desenvolvimento/tmp/bench-pp-tg-*.log
```

### Fix
```bash
# Matar servidor travado
pkill -f "ft serve"
pkill -f "python.*bench_pp_tg"
# Limpar estado
make preflight
```

### Prevenção
```bash
# SEMPRE usar wait-for-server.sh (event-driven)
ft serve & PID=$!
./scripts/wait-for-server.sh 8000 $PID
# timeout global pytest já configurado: --timeout=60
```

---

## 6. VRAM Ledger ceiling excedido

### Sintoma
```
VRAM ledger over-modelled account by X GiB
KV context feasibility: 128K X GiB short
```

### Diagnóstico
```bash
ft serve --model ... --memory-ratio 0.86  # imprime ledger
grep "VRAM ledger" /models/desenvolvimento/tmp/bench-pp-tg-*.log
```

### Fixes
| Ação | Arquivo |
|------|---------|
| Reduzir `TRITON_AUTOTUNE_ARENA` 256→128 MiB | `engine/vram_ledger.py` |
| Reduzir `GRAPH_CAPTURE_PEAK` 256→128 MiB | `engine/vram_ledger.py` |
| Reduzir `FRAGMENTATION_RESERVE` 128→64 MiB | `engine/vram_ledger.py` |
| Usar `--kv-format turbo4` (compressão 4-bit) | flag serve |

---

## 7. Checkpoint GGUF não carrega / geometria inválida

### Sintoma
```
KeyError: 'expert_geometry' ou 'bank_shape'
MoE pool key mismatch: expected (bank, role, type)
```

### Diagnóstico
```bash
# Verificar geometria do checkpoint
python -c "
from freetoken.models.gguf.reader import GGUFReader
r = GGUFReader('/models/seu-modelo.gguf')
for k in r.keys():
    if 'expert' in k.lower() or 'moe' in k.lower():
        print(k, r[k].shape)
"
```

### Fix
- Pool de experts deve ser keyed por `(bank, role, type)` exato
- Verificar `moe/offload_cache.py` e `moe/expert_banks.py`
- Phase 7: implementar keying por geometria exata (desbloqueia Ornith/Tiel)

---

## Comandos Úteis Rápidos

```bash
# Pre-flight completo
make preflight

# Rebuild tudo
make rebuild

# CI local (lint + typecheck + tests)
make ci

# Benchmark com validação anchors
make bench

# Formatar código
make format

# Logs mais recentes
ls -lt /models/desenvolvimento/tmp/bench-pp-tg-*.log | head -5
```

## 8. Nova issue descoberta

**Adicionado:** 2026-09-19 (auto)

### Sintoma
Sintoma: VRAM ledger mismatch

### Diagnóstico
Diagnóstico: _restore_qsa_state não cobre cmp_rows

### Fix
Fix: adicionar cmp_rows restore em scheduler/spec.py:114

---
