#!/usr/bin/env bash
# preflight.sh — Validação obrigatória antes de qualquer benchmark/serve
# Uso: ./scripts/preflight.sh [--strict]
# Exit codes: 0=OK, 1=WARN, 2=FAIL, 3=SUDO_NEEDED

set -euo pipefail

STRICT="${1:-}"
WARN=0
FAIL=0

log()   { echo -e "\033[1;34m[PREFLIGHT]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; WARN=1; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m $*"; FAIL=1; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }

# 1. RAM
log "Verificando RAM..."
# Drop clean page cache via posix_fadvise to maximize MemAvailable for earlyoom headroom
python3 -c "
import os, glob
for p in glob.glob('/models/*/*.safetensors') + glob.glob('/models/*/*.ftw'):
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass
" 2>/dev/null || true
RAM_FREE=$(free -g | awk '/^Mem:/ {print $7}')
if (( RAM_FREE < 20 )); then
    warn "RAM livre: ${RAM_FREE}GiB (recomendado ≥20GiB para Flash-Next MoE offload)"
else
    ok "RAM livre: ${RAM_FREE}GiB"
fi

# 2. Swap
SWAP_USED=$(free -g | awk '/^Swap:/ {print $3}')
if (( SWAP_USED > 0 )); then
    warn "Swap em uso: ${SWAP_USED}GiB (deve ser 0)"
else
    ok "Swap: 0GiB"
fi

# 3. tmpfs /tmp
TMP_USAGE=$(du -sh /tmp 2>/dev/null | awk '{print $1}' || echo "0")
TMP_USAGE_GB=$(echo "$TMP_USAGE" | sed 's/[A-Z]//g')
if [[ "$TMP_USAGE" == *G* ]] && (( ${TMP_USAGE_GB%.*} > 1 )); then
    warn "/tmp tmpfs: ${TMP_USAGE} (recomendado <1GiB). Rode: rm -rf /tmp/*"
else
    ok "/tmp tmpfs: ${TMP_USAGE}"
fi

# 4. VRAM
log "Verificando GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi não encontrado. Driver NVIDIA instalado?"
else
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if (( VRAM_FREE < 14000 )); then
        warn "VRAM livre: ${VRAM_FREE}MiB (recomendado ≥14GiB para 16K context)"
    else
        ok "VRAM livre: ${VRAM_FREE}MiB / ${VRAM_TOTAL}MiB"
    fi
    if (( PROCS > 0 )); then
        warn "Processos na GPU: $PROCS (deve ser 0 antes de serve/bench)"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    else
        ok "GPU sem processos"
    fi
fi

# 5. Processos zumbis python/tail
log "Verificando processos órfãos..."
ORPHANS=$(ps aux | grep -E '(python|tail)' | grep -v grep | grep -v "preflight" | wc -l)
if (( ORPHANS > 0 )); then
    warn "Processos python/tail ativos: $ORPHANS"
    ps aux | grep -E '(python|tail)' | grep -v grep | grep -v "preflight"
    if [[ "$STRICT" == "--strict" ]]; then
        log "Matando órfãos (--strict)..."
        pkill -9 -f "python.*bench" 2>/dev/null || true
        pkill -9 -f "ft serve" 2>/dev/null || true
        killall tail 2>/dev/null || true
        ok "Órfãos limpos"
    fi
else
    ok "Sem processos órfãos"
fi

# 6. TMPDIR
log "Verificando TMPDIR..."
if [[ -z "${TMPDIR:-}" ]]; then
    warn "TMPDIR não exportado. Use: export TMPDIR=/models/desenvolvimento/tmp"
else
    ok "TMPDIR=${TMPDIR}"
    if [[ ! -d "$TMPDIR" ]]; then
        warn "TMPDIR não existe, criando..."
        mkdir -p "$TMPDIR"
    fi
fi

# 7. CUDA_HOME e versão do toolkit (CUDA 13.3 obrigatório para Blackwell SM120)
log "Verificando CUDA (mínimo 13.3 para Blackwell SM120)..."
if [[ -z "${CUDA_HOME:-}" ]]; then
    warn "CUDA_HOME não exportado. Use: export CUDA_HOME=/models/outros/cuda-13.3"
else
    ok "CUDA_HOME=${CUDA_HOME}"
    if [[ ! -d "$CUDA_HOME" ]]; then
        fail "CUDA_HOME não existe: $CUDA_HOME"
    fi
fi
NVCC_BIN="${CUDA_HOME:-/models/outros/cuda-13.3}/bin/nvcc"
if [[ ! -x "$NVCC_BIN" ]]; then
    NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
fi
if [[ -n "$NVCC_BIN" && -x "$NVCC_BIN" ]]; then
    NVCC_VER=$("$NVCC_BIN" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -1)
    # Compara semanticamente se NVCC_VER >= 13.3
    NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    NVCC_MINOR=$(echo "$NVCC_VER" | cut -d. -f2)
    if (( NVCC_MAJOR < 13 || (NVCC_MAJOR == 13 && NVCC_MINOR < 3) )); then
        fail "nvcc versão ${NVCC_VER} detectada. RTX 5080 (SM120) exige estritamente CUDA >= 13.3!"
    else
        ok "nvcc ${NVCC_VER} (CUDA >= 13.3 verificado e ativo: $NVCC_BIN)"
    fi
else
    fail "nvcc não encontrado em ${CUDA_HOME}/bin nem no PATH!"
fi

# 8. TORCH_CUDA_ARCH_LIST
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    warn "TORCH_CUDA_ARCH_LIST não exportado. RTX 5080 precisa: 12.0;12.0a"
else
    ok "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

# 8b. CPU / Paralelismo de Compilação (Ryzen 9 9900X = 24 threads)
NPROC_COUNT=$(nproc 2>/dev/null || echo "1")
if (( NPROC_COUNT >= 24 )); then
    ok "CPU Threads detectados: ${NPROC_COUNT} (Ryzen 9 9900X Zen 5)"
else
    warn "CPU Threads detectados: ${NPROC_COUNT} (esperado 24)"
fi
if [[ "${MAX_JOBS:-}" != "24" ]]; then
    warn "MAX_JOBS não está definido como 24 (atual: '${MAX_JOBS:-unset}'). Use: export MAX_JOBS=24"
else
    ok "MAX_JOBS=24"
fi

# 9. uv / Python env
log "Verificando ambiente Python..."
if command -v uv &>/dev/null; then
    ok "uv: $(uv --version)"
elif command -v /home/natal/.local/bin/uv &>/dev/null; then
    ok "uv: $(/home/natal/.local/bin/uv --version) (em ~/.local/bin)"
else
    warn "uv não no PATH. Instale: pip install uv"
fi

if [[ -d ".venv" ]]; then
    ok ".venv existe"
else
    warn ".venv não encontrado. Rode: make install"
fi

# 10. Checkpoints principais
log "Verificando checkpoints..."
for CKPT in "/models/Qwen3.8-Flash-Next-NVFP4-Radix" "/models/Qwen3.6-35B-A3B-NVFP4-FT"; do
    if [[ -d "$CKPT" ]]; then
        ok "Checkpoint: $CKPT"
    else
        warn "Checkpoint não encontrado: $CKPT"
    fi
done

# Resumo
echo
if (( FAIL > 0 )); then
    fail "PREFLIGHT FALHOU — corrija erros acima antes de prosseguir"
    exit 2
elif (( WARN > 0 )); then
    warn "PREFLIGHT COM AVISOS — revise itens acima"
    [[ "$STRICT" == "--strict" ]] && exit 1 || exit 1
else
    ok "PREFLIGHT OK — ambiente pronto"
    exit 0
fi
