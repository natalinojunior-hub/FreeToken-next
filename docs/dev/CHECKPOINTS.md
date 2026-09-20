# CHECKPOINTS — Modelos Suportados e Formatos freetoken-next

**Última atualização:** 2026-09-19 | **Formatos:** FTW (native), GGUF (native), HF (convert)

---

## Checkpoints Principais (Produção)

| Modelo | Arquitetura | Params | Experts | Ativos | Quant | Formato | Path | Status |
| Test-Model | Dense-7B | 1 | 1 | BF16 | FTW | /models/Test-Model | Testing |
|--------|-------------|--------|---------|--------|-------|---------|------|--------|
| Qwen3.8-Flash-Next | MoE 48L | ~290B | 256 | 8 | NVFP4 | FTW | `/models/Qwen3.8-Flash-Next-NVFP4-Radix` | ✅ Certificado 16K/128K |
| Qwen3.6-35B-A3B | MoE 48L | 35B | 128 | 8 | NVFP4 | FTW | `/models/Qwen3.6-35B-A3B-NVFP4-FT` | ✅ Certificado 16K/128K/256K |
| GLM-5.2 | MoE | ~100B | TBD | TBD | FP8 | GGUF | TBD | 🔄 Phase 7 |
| DeepSeek-V4-Flash | MoE | ~200B | TBD | TBD | MXFP4 | GGUF | TBD | 🔄 Phase 7 |
| Ornith-35B | MoE | 35B | TBD | TBD | Q3_K/IQ3_S | GGUF | `/models/Ornith-35B-GGUF` | ⏳ Phase 7 (geometry keying) |
| Tiel-35B | MoE | 35B | TBD | TBD | Q4_K | GGUF | `/models/Tiel-35B-GGUF` | ⏳ Phase 7 (geometry keying) |

---

## Formato FTW (FreeToken Weight) — Nativo

### Características
- Fast-load: mmap + lazy page-in, sem dequant no load
- Suporta quantização mista por tensor (NVFP4, MXFP4, FP8, BF16)
- Metadata: `ftw_metadata.json` com geometria MoE, VRAM hints

### Conversão HF → FTW
```bash
ft checkpoint convert \
  --model /path/to/hf-checkpoint \
  --output /models/MODEL-FTW \
  --quant nvfp4 \
  --moe-geometry auto
```

### Validação FTW
```bash
ft checkpoint verify /models/MODEL-FTW
# Deve imprimir: Geometria MoE, quantização por layer, VRAM estimada
```

---

## Formato GGUF — Native Loader (Sem dequant load-time)

### Características
- Loader nativo `models/gguf/reader.py` + `models/gguf/config.py`
- Suporta quantização mista: Q3_K, Q4_K, Q6_K, IQ3_S, IQ4_XS, IQ4_NL
- **Sem dequant no load** — pesos ficam quantizados em VRAM/host
- Geometria MoE derivada do arquivo GGUF (blocos `expert_*`)

### Checkpoints GGUF Testados

| Modelo | Quant | Tamanho | VRAM (16K) | TG (tok/s) | Status |
|--------|-------|---------|------------|------------|--------|
| Qwen3.6-35B-A3B | IQ3_S | 27B | 2.17 GiB RSS | 25.3 | ✅ Dense baseline |
| Ornith-35B | Q3_K | ~20B | TBD | TBD | ⏳ Phase 7 |
| Tiel-35B | Q4_K | ~22B | TBD | TBD | ⏳ Phase 7 |

### Carregamento GGUF
```bash
ft serve --model /models/Ornith-35B-GGUF \
  --num-tokens 16576 \
  --cache-type naive \
  --moe-strategy offload \
  --moe-cache-auto
```

### Geometria MoE GGUF (Phase 7)
**Problema:** Pool de experts keyed apenas por `bank` → colisão Ornith/Tiel
**Fix:** Key por `(bank, role, type)` exato
- `role`: `gate_up` | `down` | `shared`
- `type`: `Q3_K` | `Q4_K` | `IQ3_S` | `IQ4_XS` | `BF16`
- `bank`: índice do expert bank (0..N)

```python
# moe/expert_banks.py - Nova chave
pool_key = (bank_idx, role, quant_type)  # ex: (3, "gate_up", "Q3_K")
```

---

## Conversão e Reparo

### FTW Hotfix (checkpoints legados)
```bash
ft checkpoint hotfix /models/OLD-CHECKPOINT \
  --output /models/FIXED-CHECKPOINT \
  --fix-geometry --fix-quant-meta
```

### GGUF → FTW (experimental)
```bash
ft checkpoint convert \
  --model /models/MODEL.gguf \
  --output /models/MODEL-FTW \
  --from-gguf
```

---

## Validação de Checkpoint (Pre-Serve)

```bash
# Verificar se carrega sem OOM
ft serve --model /models/CHECKPOINT --num-tokens 8192 --cache-type naive --dry-run

# Verificar geometria MoE
python -c "
from freetoken.models.registry import load_model
model = load_model('/models/CHECKPOINT')
print('Experts:', model.config.num_experts)
print('Active:', model.config.num_active_experts)
print('Layers:', model.config.num_layers)
"
```

---

## Estrutura de Diretórios Esperada

```
/models/
├── Qwen3.8-Flash-Next-NVFP4-Radix/      # FTW native
│   ├── ftw_metadata.json
│   ├── model.safetensors (ou shards)
│   └── tokenizer/
├── Qwen3.6-35B-A3B-NVFP4-FT/            # FTW native
│   ├── ftw_metadata.json
│   ├── model.safetensors
│   └── tokenizer/
├── Ornith-35B-GGUF/                      # GGUF native
│   ├── ornith-35b-q3_k.gguf
│   └── tokenizer/
├── Tiel-35B-GGUF/                        # GGUF native
│   ├── tiel-35b-q4_k.gguf
│   └── tokenizer/
└── desenvolvimento/
    └── tmp/                              # TMPDIR (disco real)
```

---

## Referências
- `models/gguf/reader.py` — Loader GGUF nativo
- `models/gguf/config.py` — Config parsing + geometria MoE
- `checkpoint/` — Conversão HF→FTW
- `RUNBOOKS.md#7` — Checkpoint GGUF não carrega
- `HARDWARE_TUNING.md` — Parâmetros por modelo
- `EXPERIMENTS.md` — EXP-015 (GGUF MoE host-RAM), EXP-017/018/020 (FTW cold-bank)
