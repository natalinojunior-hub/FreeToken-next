#!/usr/bin/env bash
# Realiza checagem de saúde e limpeza do ambiente antes de benchmarks pesados.

echo "=== FreeToken Next: Pre-Flight Check ==="

# 1. Matar zumbis
echo "[1/4] Caçando processos zumbis (python, tail)..."
pkill -u "$USER" -f "bench_pp_tg.py" && echo "  - Morto: bench_pp_tg.py" || true
pkill -u "$USER" -f "ft serve" && echo "  - Morto: ft serve" || true
pkill -u "$USER" -x "tail" && echo "  - Morto: tail" || true
echo "  OK."

# 2. Limpeza de TMPDIR
echo "[2/4] Limpando TMPDIR (/models/desenvolvimento/tmp)..."
rm -rf /models/desenvolvimento/tmp/*
echo "  OK."

# 3. Checagem de RAM do Host
echo "[3/4] Checando RAM do Host..."
RAM_LIVRE_MB=$(free -m | awk '/^Mem:/ {print $7}')
if [ "$RAM_LIVRE_MB" -lt 20000 ]; then
    echo "  [AVISO] Memória livre baixa: ${RAM_LIVRE_MB}MB. Recomendado >= 20000MB para Flash-Next."
else
    echo "  OK (${RAM_LIVRE_MB}MB livres)."
fi

# 4. Checagem de VRAM (NVIDIA)
echo "[4/4] Checando VRAM da GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | awk -F', ' '{print "  Uso: " $1 " / " $2}'
else
    echo "  [AVISO] nvidia-smi não encontrado."
fi

echo "=== Sistema Pronto ==="
