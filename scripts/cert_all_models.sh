#!/usr/bin/env bash
# Universal certification matrix: every /models checkpoint, serial on one GPU, evented status.
# Per model: automatic MTP bench 16K, RAW (depth-0) bench 16K, vision smoke (mmproj or inline),
# and a 256K qualification attempt (planner verdict recorded when infeasible - measured, not skipped).
set -u
REPO=/models/desenvolvimento/freetoken-next
A=/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929
ST=$A/cert-all.status
PROMPT=/models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign26/prompt-470k.txt
export TMPDIR=/models/desenvolvimento/tmp CUDA_HOME=/models/outros/cuda-13.3
export PATH=/models/outros/cuda-13.3/bin:/home/natal/.local/bin:$PATH
export PYTHONPATH=python
PY=$REPO/.venv/bin/python
cd "$REPO" || exit 9

status() { echo "$(date -Is) $*" >> "$ST"; }
phase() { # phase name cmd... : record rc, keep going on failure (verdicts are data)
  local name=$1; shift
  status "START $name"
  if "$@" >> "$A/cert-all.log" 2>&1; then status "PASS $name"; else status "FAIL $name rc=$?"; fi
}

bench() { # bench name model extra-env... -- extra-serve-args...
  local name=$1 model=$2; shift 2
  local envs=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  extra=("$@")
  env ${envs[@]+"${envs[@]}"} "$PY" benchmarks/bench_pp_tg.py \
    --model "$model" --tokens 16384 --decode 256 --repeats 6 --warmups 1 \
    --prompt-file "$PROMPT" --serve-arg=--max-seq-len --serve-arg=16704 \
    ${extra[@]+"${extra[@]}"} --no-history --json "$A/cert-all.jsonl" --label "$name"
}

AD=/models/Qwen3.8-Flash-Next-AD-4.27/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64
PFE=/models/Qwen3.8-Flash-Next-pfeifferj-3.5bit
RAD=/models/Qwen3.8-Radix-NVPF4
FT=/models/Qwen3.6-35B-A3B-NVFP4-FT

status "cert-all begin pid=$$"

phase ad-16k-mtp      bench ad-16k-mtp "$AD"
phase ad-16k-raw      bench ad-16k-raw "$AD" FREETOKEN_MTP_FORCE_DEPTH=0 --
phase ad-mm           "$PY" scripts/mm_smoke.py --model "$AD" --label ad-mm --json "$A/cert-all.mm.jsonl"

phase pfe-16k-mtp     bench pfe-16k-mtp "$PFE"
phase pfe-16k-raw     bench pfe-16k-raw "$PFE" FREETOKEN_MTP_FORCE_DEPTH=0 --
phase pfe-mm          "$PY" scripts/mm_smoke.py --model "$PFE" --label pfe-mm --json "$A/cert-all.mm.jsonl"

phase rad-16k-mtp     bench rad-16k-mtp "$RAD"
phase rad-16k-raw     bench rad-16k-raw "$RAD" FREETOKEN_MTP_FORCE_DEPTH=0 --
phase rad-mm          "$PY" scripts/mm_smoke.py --model "$RAD" --label rad-mm --json "$A/cert-all.mm.jsonl"

phase ft-16k-mtp      bench ft-16k-mtp "$FT"
phase ft-16k-raw      bench ft-16k-raw "$FT" FREETOKEN_MTP_FORCE_DEPTH=0 --
phase ft-mm           "$PY" scripts/mm_smoke.py --model "$FT" --label ft-mm --json "$A/cert-all.mm.jsonl"

# 256K qualification attempts (ISTA/AD/pfeiffer GGUF; Radix/FT largest feasible context recorded)
qualify() { # qualify name model ctx
  local name=$1 model=$2 ctx=$3
  phase "$name" "$PY" scripts/qualify_long_context.py --model "$model" --worktree "$REPO" \
    --context "$ctx" --mode usage --out "$A/cert-all.qual.jsonl"
}
qualify ista-256k-usage /models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS 262144
qualify ad-256k-usage   "$AD" 262144
qualify pfe-256k-usage  "$PFE" 262144
qualify rad-max-usage   "$RAD" 262144
qualify ft-max-usage    "$FT" 262144

status "cert-all done"
touch "$A/cert-all.done"
