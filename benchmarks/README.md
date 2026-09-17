# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_pp_tg.py`** — prefill and decode throughput at a chosen context length, one server
per row: PP from TTFT, TG from streamed token arrivals, plus TTFT, ITL p50/p95, VRAM, host RSS
and sampled GPU utilisation. The prompt is a local corpus slice fixed-pointed to exactly
`--tokens` ids, greedy, repeated `--repeats` times, and each JSONL row records the serve
command that produced it. PP rows need `--serve-arg "--cache-type naive"`, or the radix prefix
cache reuses the prompt and reports a prefill that never ran.

```bash
python benchmarks/bench_pp_tg.py --model /path/to/model --tokens 16384 --decode 128 \
    --repeats 3 --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive"
```

**`cert_matrix.py`** — the final gate over both. Runs `bench_pp_tg.py` across the declared
native `-FT`/NVFP4 rows and their same-architecture GGUF counterparts, fails the build on a
native regression against its guard, and prints each GGUF row as a percentage of its native
row. A checkpoint the current code cannot serve reports its blocker instead of dropping out of
the table; `--dry-run` prints the plan without touching the GPU.

```bash
python benchmarks/cert_matrix.py --dry-run
python benchmarks/cert_matrix.py --contexts 4096,16384,32768 --json /tmp/cert.jsonl
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
