#!/usr/bin/env bash
# Supported-model certification matrix: serial on one GPU, evented status.
# MTP is measured only for families with an implemented head; unsupported is recorded explicitly.
set -u
REPO=/models/desenvolvimento/freetoken-next
A=${A:-/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929}
mkdir -p "$A"
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
  if [ -f "$A/cert-all.jsonl" ] && grep -q "\"label\": \"$name\"" "$A/cert-all.jsonl"; then
    status "CACHED $name (verdict already in cert-all.jsonl)"; return 0
  fi
  local envs=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  extra=("$@")
  local mtp_args=(--serve-arg=--spec-mtp --serve-arg=5)
  if [[ " ${envs[*]} " == *" FREETOKEN_MTP_FORCE_DEPTH=0 "* ]]; then mtp_args=(); fi
  env ${envs[@]+"${envs[@]}"} "$PY" benchmarks/bench_pp_tg.py \
    --model "$model" --tokens 16384 --decode 256 --repeats 6 --warmups 1 \
    --prompt-file "$PROMPT" --serve-arg=--max-seq-len --serve-arg=16704 \
    "${mtp_args[@]}" \
    ${extra[@]+"${extra[@]}"} --no-history --json "$A/cert-all.jsonl" --label "$name"
}

AD=/models/Qwen3.8-Flash-Next-AD-4.27/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64
PFE=/models/Qwen3.8-Flash-Next-pfeifferj-3.5bit
RAD=/models/Qwen3.8-Radix-NVPF4
FT=/models/Qwen3.6-35B-A3B-NVFP4-FT

status "cert-all begin pid=$$"

ISTA=/models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS
phase ista-16k-mtp    bench ista-16k-mtp "$ISTA"
phase ista-16k-raw    bench ista-16k-raw "$ISTA" FREETOKEN_MTP_FORCE_DEPTH=0 --

phase ad-16k-mtp      bench ad-16k-mtp "$AD"
phase ad-16k-raw      bench ad-16k-raw "$AD" FREETOKEN_MTP_FORCE_DEPTH=0 --
mm() { # mm name model : vision smoke only when the checkpoint carries an mmproj sidecar
  local name=$1 model=$2
  if [ -f "$A/cert-all.mm.jsonl" ] && grep -q "\"label\": \"$name\"" "$A/cert-all.mm.jsonl"; then
    status "CACHED $name"; return 0
  fi
  if ! ls "$model"/*mmproj*.gguf >/dev/null 2>&1; then
    status "SKIP $name (no mmproj sidecar)"; return 0
  fi
  "$PY" scripts/mm_smoke.py --model "$model" --label "$name" --json "$A/cert-all.mm.jsonl"
}

phase ista-mm-v3      mm ista-mm-v3 /models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS
phase ad-mm-v3        mm ad-mm-v3 "$AD"
phase pfe-mm-v3       mm pfe-mm-v3 "$PFE"
phase rad-mm          mm rad-mm "$RAD"
phase ft-mm           mm ft-mm "$FT"

phase pfe-16k-mtp     bench pfe-16k-mtp "$PFE"
phase pfe-16k-raw     bench pfe-16k-raw "$PFE" FREETOKEN_MTP_FORCE_DEPTH=0 --

phase rad-16k-mtp     bench rad-16k-mtp "$RAD"
phase rad-16k-raw     bench rad-16k-raw "$RAD" FREETOKEN_MTP_FORCE_DEPTH=0 --

status "UNSUPPORTED ft-16k-mtp (Qwen3.6-35B-A3B family has no wired native MTP head)"
phase ft-16k-raw      bench ft-16k-raw "$FT" FREETOKEN_MTP_FORCE_DEPTH=0 --

# GATE (operator): the 256K phase runs only if every 16K MTP+RAW bench passed.
if grep -E " FAIL .*16k" "$ST" | awk '{print $3}' | sort -u | while read -r nm; do
    grep -q "\"label\": \"$nm\"" "$A/cert-all.jsonl" || echo UNCACHED
  done | grep -q UNCACHED; then
  status "cert-all: 16K gate FAILED -> skipping 256K phase"; exit 1
fi
status "cert-all: 16K gate OK -> entering 256K phase"
if [ "${CERT_16K_ONLY:-0}" = 1 ]; then
  status "cert-all: 16K-only matrix complete"
  exit 0
fi

# 256K qualification attempts (ISTA/AD/pfeiffer GGUF; Radix/FT largest feasible context recorded)
qualify() { # qualify name model ctx
  local name=$1 model=$2 ctx=$3
  if [ -f "$A/cert-all.qual.jsonl" ] && grep -q "\"label\": \"$name\"" "$A/cert-all.qual.jsonl"; then
    status "CACHED $name"; return 0
  fi
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
