#!/usr/bin/env bash
# bench-mtp-equiv.sh — Roda bench_pp_tg.py em dois braços (--spec-mtp 0 vs N) e
# compara sha1 do output. PASS = bit-identical; FAIL = diverge (imprime ambos sha1).
# Uso: ./scripts/bench-mtp-equiv.sh --model PATH [--mtp-k N] [--tokens N] [--decode N]
#      [--mem-ratio R] [--extra-serve-arg "..."] (repetível) [--label-prefix TAG]

set -euo pipefail

export MAX_JOBS="${MAX_JOBS:-24}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"
export TMPDIR="${TMPDIR:-/models/desenvolvimento/tmp}"
export CUDA_HOME="${CUDA_HOME:-/models/outros/cuda-13.3}"
export PATH="/models/desenvolvimento/freetoken-next/.venv/bin:$CUDA_HOME/bin:/home/natal/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

MODEL=""
MTP_K=1
TOKENS=16384
DECODE=16
MEM_RATIO=0.98
LABEL_PREFIX="mtpeq"
EXTRA_SERVE_ARGS=()
NO_GRAPH=""
SCRATCH="${TMPDIR:-/tmp}/bench-mtp-equiv.$$"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --mtp-k) MTP_K="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --decode) DECODE="$2"; shift 2 ;;
        --mem-ratio) MEM_RATIO="$2"; shift 2 ;;
        --extra-serve-arg) EXTRA_SERVE_ARGS+=("$2"); shift 2 ;;
        --label-prefix) LABEL_PREFIX="$2"; shift 2 ;;
        --no-graph) NO_GRAPH="--no-graph"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ERRO: --model obrigatorio" >&2; exit 2; }

mkdir -p "$SCRATCH"
echo "[bench-mtp-equiv] scratch: $SCRATCH"

echo "[bench-mtp-equiv] preflight..."
# No --strict: preflight.sh --strict does `killall tail`, which also kills
# any Monitor/tail -F currently watching this script's own log output.
set +e
"$(dirname "$0")/preflight.sh"
PREFLIGHT_RC=$?
set -e
if [[ $PREFLIGHT_RC -ge 2 ]]; then
    echo "[bench-mtp-equiv] FAIL: preflight failed (exit $PREFLIGHT_RC)" >&2
    exit 1
fi

run_arm() {
    local k="$1" label="$2" json="$3" log="$4"
    local pid
    local -a serve_arg_flags=()
    for a in "${EXTRA_SERVE_ARGS[@]}"; do
        serve_arg_flags+=("--serve-arg=$a")
    done
    TMPDIR=/models/desenvolvimento/tmp FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
        .venv/bin/python benchmarks/bench_pp_tg.py \
        --model "$MODEL" --tokens "$TOKENS" --decode "$DECODE" \
        --repeats 1 --warmups 0 --label "$label" --mem-ratio "$MEM_RATIO" $NO_GRAPH \
        --serve-arg="--spec-mtp $k" "${serve_arg_flags[@]}" \
        --json "$json" > "$log" 2>&1 < /dev/null &
    pid=$!

    wait "$pid" || true
    if [[ ! -s "$json" ]]; then
        return 1
    fi
    python3 -c 'import json,sys; from pathlib import Path; rows=[json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]; raise SystemExit(0 if rows and rows[-1].get("output_sha1") else 1)' "$json" || return 1
    return 0
}

echo "[bench-mtp-equiv] arm k=0 (baseline)..."
if ! run_arm 0 "${LABEL_PREFIX}_k0" "$SCRATCH/k0.jsonl" "$SCRATCH/k0.log"; then
    echo "[bench-mtp-equiv] FAIL: baseline (k=0) did not complete. See $SCRATCH/k0.log" >&2
    exit 1
fi

echo "[bench-mtp-equiv] arm k=$MTP_K (mtp)..."
if ! run_arm "$MTP_K" "${LABEL_PREFIX}_k${MTP_K}" "$SCRATCH/k${MTP_K}.jsonl" "$SCRATCH/k${MTP_K}.log"; then
    echo "[bench-mtp-equiv] FAIL: mtp (k=$MTP_K) did not complete. See $SCRATCH/k${MTP_K}.log" >&2
    exit 1
fi

VALUES=$(python3 -c 'import json,sys; from pathlib import Path; rows=[json.loads(Path(name).read_text().splitlines()[-1]) for name in sys.argv[1:]]; print(rows[0]["output_sha1"], rows[0]["TG_mean"], rows[1]["output_sha1"], rows[1]["TG_mean"])' "$SCRATCH/k0.jsonl" "$SCRATCH/k${MTP_K}.jsonl")
read -r SHA_K0 TG_K0 SHA_KN TG_KN <<< "$VALUES"

echo "[bench-mtp-equiv] k=0 sha1=$SHA_K0 TG=$TG_K0"
echo "[bench-mtp-equiv] k=$MTP_K sha1=$SHA_KN TG=$TG_KN"

if [[ "$SHA_K0" == "$SHA_KN" ]]; then
    echo "[bench-mtp-equiv] PASS: sha1 match ($SHA_K0)"
    exit 0
else
    echo "[bench-mtp-equiv] FAIL: sha1 diverge ($SHA_K0 != $SHA_KN)"
    exit 1
fi
