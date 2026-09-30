#!/usr/bin/env bash
# Paired 16K RAW vs automatic MTP certification. Each process runs warmup + 3 samples.
set -euo pipefail

usage() {
  echo "Usage: $0 --model PATH --prompt-file PATH --output-dir DIR"
}
MODEL="" PROMPT="" OUT=""
while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" && -n "$PROMPT" && -n "$OUT" ]] || { usage >&2; exit 2; }
[[ -e "$MODEL" && -r "$PROMPT" ]] || { echo "Model/prompt not accessible" >&2; exit 2; }
mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "Python executable not found: $PY" >&2; exit 2; }

git rev-parse HEAD > "$OUT/git-head.txt"
git status --short > "$OUT/git-status.txt"
"$PY" - "$MODEL" "$PROMPT" "$OUT" <<'PY' > "$OUT/environment.json"
import hashlib, json, os, platform, subprocess, sys
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
model,prompt,out=sys.argv[1:]
try: gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
except OSError: gpu='unavailable'
env={k:v for k,v in os.environ.items() if k.startswith('FREETOKEN_') or k in ('CUDA_HOME','PATH')}
diff=subprocess.run(['git','diff','--binary','HEAD','--','python/freetoken'],capture_output=True).stdout
untracked=subprocess.run(['git','ls-files','--others','--exclude-standard','--','python/freetoken'],capture_output=True,text=True).stdout.splitlines()
dirty={path:sha(path) for path in untracked if os.path.isfile(path)}
json.dump({'model':os.path.realpath(model),'prompt':os.path.realpath(prompt),'prompt_sha256':sha(prompt),'output_dir':os.path.realpath(out),'python':sys.version,'platform':platform.platform(),'gpu':gpu,'environment':env,'runtime_source_diff_sha256':hashlib.sha256(diff).hexdigest(),'untracked_runtime_files_sha256':dirty,'benchmark':'benchmarks/bench_pp_tg.py','prompt_tokens':16384,'decode_tokens':256,'max_seq_len':16704,'warmups':1,'measured_per_boot':3,'boots_per_arm':1,'order':'RAW,MTP','raw_spec_mtp':0,'automatic_mtp_cap':5},sys.stdout,indent=2); print()
PY

run() {
  local arm="$1" spec="$2" batch="$3"
  env -u FREETOKEN_MTP_FORCE_DEPTH -u FREETOKEN_MTP_FIXED_DEPTH -u FREETOKEN_TOKEN_TRACE \
    PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - "$ROOT" "$MODEL" "$PROMPT" "$OUT" "$arm" "$spec" "$batch" <<'PY'
import os,runpy,sys
root,model,prompt,out,arm,spec,batch=sys.argv[1:]
for key in list(os.environ):
    if key.startswith('FREETOKEN_DEBUG_'): os.environ.pop(key)
os.environ['PYTHONPATH']=root+'/python'+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
sys.argv=['benchmarks/bench_pp_tg.py','--model',model,'--prompt-file',prompt,'--tokens','16384','--decode','256','--repeats','3','--warmups','1','--label',f'ista16k-{arm}-{batch}','--serve-arg=--max-seq-len','--serve-arg=16704',f'--serve-arg=--spec-mtp {spec}','--no-history','--json',f'{out}/{arm}-{batch}.jsonl']
runpy.run_path('benchmarks/bench_pp_tg.py',run_name='__main__')
PY
}
run raw 0 1 2>&1 | tee "$OUT/raw-1.log"
run mtp 5 1 2>&1 | tee "$OUT/mtp-1.log"

"$PY" - "$OUT" <<'PY' > "$OUT/summary.json"
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); result={}
failed=[]
for arm in ('raw','mtp'):
    rows=[]
    for batch in (1,):
        f=p/f'{arm}-{batch}.jsonl'; rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    measured=[]
    for row in rows:
        measured += row.get('runs',[])[-3:]
    if len(measured)!=3: failed.append(f'{arm}: expected 3 samples, got {len(measured)}')
    if not measured:
        result[arm]={'repeats':0,'tg_mean':None,'tg_min':None,'output_sha1s':[],'completion_tokens':[],'floor_106_pass':None}
        failed.append(f'{arm}: no measured samples')
        continue
    tg=[x['decode_tok_s'] for x in measured]
    result[arm]={'repeats':len(measured),'tg_mean':sum(tg)/len(tg),'tg_min':min(tg),'pp_mean':sum(x['prefill_tok_s'] for x in measured)/len(measured),'ttft_mean_ms':sum(x['ttft_ms'] for x in measured)/len(measured),'itl_p50_mean_ms':sum(x['itl_ms_p50'] for x in measured)/len(measured),'itl_p95_mean_ms':sum(x['itl_ms_p95'] for x in measured)/len(measured),'ram_rss_gib_mean':sum(x['server_rss_gib'] for x in measured)/len(measured),'vram_gib_mean':sum(x['vram_gib'] for x in measured)/len(measured),'output_sha1s':sorted({x['output_sha1'] for x in measured}),'completion_tokens':sorted({x['completion_tokens'] for x in measured}),'floor_106_pass':min(tg)>=106 if arm=='mtp' else None}
    if len(result[arm]['output_sha1s'])!=1: failed.append(f'{arm}: output SHA set is not singleton')
    if result[arm]['completion_tokens']!=[256]: failed.append(f'{arm}: completion token counts are not exactly 256')
    if arm=='mtp' and min(tg)<106: failed.append(f'{arm}: TG floor 106 missed (min={min(tg):.3f})')
    result[arm]['raw_json_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob(f'{arm}-*.jsonl'))}
result['output_parity']=result['raw']['output_sha1s']==result['mtp']['output_sha1s']
if not result['output_parity']: failed.append('RAW/MTP output SHA parity failed')
result['failed_contracts']=failed
print(json.dumps(result,indent=2))
if failed: raise SystemExit(1)
PY
