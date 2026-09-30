#!/usr/bin/env bash
# MTP closure pipeline: serial GPU gates, event status per phase, fail-fast.
set -u
REPO=/models/desenvolvimento/freetoken-next
A=/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929
ST=$A/closure-orchestrator.status
MODEL=/models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS
RAW_SHA=76a5508fd576399d4569b53207c0b70af427ff82
K4_BASELINE=103.58
export TMPDIR=/models/desenvolvimento/tmp CUDA_HOME=/models/outros/cuda-13.3
export PATH=/models/outros/cuda-13.3/bin:/home/natal/.local/bin:$PATH
export PYTHONPATH=python
PY=$REPO/.venv/bin/python
cd "$REPO" || exit 9

status() { echo "$(date -Is) $*" >> "$ST"; }
finish() { # finish PASS|FAIL phase
  echo "$(date -Is) VERDICT $1 $2" >> "$ST"
  touch "$A/closure-orchestrator.$1"
  exit $([ "$1" = PASS ] && echo 0 || echo 1)
}
run() { # run phase cmd... ; log to $A/phase.log, hard-fail on rc!=0
  local name=$1; shift
  status "START $name"
  if "$@" >> "$A/$name.log" 2>&1; then status "PASS $name"; else status "FAIL $name rc=$?"; finish FAIL "$name"; fi
}

rm -f "$A"/closure-orchestrator.PASS "$A"/closure-orchestrator.FAIL
status "ORCHESTRATOR begin pid=$$"

preflight_phase() { make preflight; local rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; } # rc=2: warnings only (known absent checkpoints)
run preflight-final preflight_phase

# Full CI (CPU lint/type/test; GPU tests execute since GPU is idle)
run full-ci-final bash -c "make ci"
# frozen-source guard: formatting must not have touched runtime source
if ! git diff --quiet HEAD -- python tests; then status "FAIL ci-mutated-source"; finish FAIL "ci-source-drift"; fi

bench() { # bench phase-name tokens serve-ctx label
  local name=$1 tokens=$2 ctx=$3 label=$4
  run "$name" "$PY" benchmarks/bench_pp_tg.py \
    --model "$MODEL" --tokens "$tokens" --decode 256 --repeats 3 --warmups 1 \
    --prompt-file /models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign26/prompt-470k.txt \
    --serve-arg=--max-seq-len --serve-arg="$ctx" --label "$label" --no-history \
    --json "$A/$label.jsonl"
}
GATE16() { $PY - "$A" "$RAW_SHA" "$K4_BASELINE" <<'EOF'
import json, sys
arc, raw_sha, base = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = [json.loads(l) for l in open(f"{arc}/auto-final16.jsonl") if l.strip()]
r = [x for x in rows if x["label"] == "auto-final16"][-1]
measured = r["runs"][-r["n"]:]
shas = {run["output_sha1"] for run in measured}
toks = {run["completion_tokens"] for run in measured}
assert r["TG_mean"] >= base, f"TG_mean {r['TG_mean']:.4f} < {base}"
assert shas == {raw_sha}, f"output parity fail: {shas}"
assert toks == {256}, f"token-count fail: {toks}"
print(f"GATE16 OK TG_mean={r['TG_mean']:.4f} TG_min={r['TG_min']:.4f} sha={shas.pop()[:12]}")
EOF
}
GATE256() { $PY - "$A" <<'EOF'
import json, sys
arc = sys.argv[1]
def last(label):
    rows = [json.loads(l) for l in open(f"{arc}/{label}.jsonl") if l.strip()]
    return [x for x in rows if x["label"] == label][-1]
r16, r256 = last("auto-final16"), last("auto-final256")
m256 = r256["runs"][-r256["n"]:]
shas = {run["output_sha1"] for run in m256}
toks = {run["completion_tokens"] for run in m256}
loss = 100.0 * (1 - r256["TG_mean"] / r16["TG_mean"])
assert shas == {m256[0]["output_sha1"]}, f"256K parity fail: {shas}"
assert toks == {256}, f"256K token-count fail: {toks}"
assert loss <= 10.0, f"256K TG loss {loss:.2f}% > 10% vs paired 16K"
print(f"GATE256 OK TG_mean={r256['TG_mean']:.4f} loss={loss:.2f}% sha={shas.pop()[:12]}")
EOF
}

bench bench16-final 16384 16704 auto-final16
# 16K gate decides ONLY 256K certification; validation batch always completes.
status "START gate16"
if GATE16 >> "$A/gate16.log" 2>&1; then status "PASS gate16"; GATE16_OK=1; else status "FAIL gate16"; GATE16_OK=0; fi
run pressure-final16 "$PY" scripts/physical_pressure_acceptance.py --tokens 16384 --decode 256
run usage-final16 "$PY" scripts/qualify_long_context.py --model "$MODEL" --context 16704 --mode usage --out "$A/usage-final16.json"

if [ "$GATE16_OK" = 0 ]; then status "256K NOT CERTIFIED: gate16 failed (mean TG below $K4_BASELINE)"; finish FAIL gate16-no-256k; fi
bench bench256-final 261824 262144 auto-final256
run gate256 GATE256
run pressure-final256 "$PY" scripts/physical_pressure_acceptance.py --tokens 261824 --decode 256
run needle-final256 "$PY" scripts/qualify_long_context.py --model "$MODEL" --context 262144 --mode needle --out "$A/needle-final256.json"
run usage-final256 "$PY" scripts/qualify_long_context.py --model "$MODEL" --context 262144 --mode usage --out "$A/usage-final256.json"

status "ALL-GATES PASS"
finish PASS all
