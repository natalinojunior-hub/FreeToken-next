# PERFORMANCE — freetoken-next

Every number here is produced on the real host (`rtx5080`, RTX 5080 / SM120, 96 GB DDR5).
No container/sandbox measurement is accepted as evidence.

## 1. Immutable regression anchors

Reference build for both anchors: `freetoken 0.1.2+gaf71ba432` (commit `af71ba432`,
23 commits behind this tree's base), NVFP4 checkpoints, 16K discovery context, no MTP,
D2D expert reuse disabled. Recorded 2026-09 (pre-existing host measurements; re-derived
below on `v0.1.3` as the gate for Phase 1).

| Workload | PP tok/s | TG tok/s | VRAM | RAM | GPU util |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4, 16K | ~1532 | ~28.96 | ~15.16 GiB | ~66.5 GiB | ~96 % |
| Qwen3.6-35B-A3B NVFP4, 16K | ~4104 | ~147.1 | ~15.04 GiB | ~20.5 GiB | — |

Gate for every future patch: Flash **PP ≥ 1500** at 16K-class conditions, 35B-A3B
**PP ≥ 4000 / TG ≥ 140** unless a measured long-context gain justifies the tradeoff.

## 2. Measured host ceilings (from the existing `ft bench bw` profile)

`~/.cache/freetoken/benchbw/GPU-85428346-1503-a189-0b52-7eb8049199f1.json`, v4,
2026-09-07, 12 physical cores:

| Quantity | Value | Meaning for design |
|---|---|---|
| `cpu_stream_read_gbs` | 64.7 | host RAM read bandwidth ceiling |
| `pcie_linear_h2d_gbs` | 57.76 | **the PCIe wall** for any RAM-tiered KV/expert scheme |
| `pcie_linear_d2h_gbs` | 57.19 | — |
| nvfp4 expert bytes | 7 974 912 / expert | — |
| bf16 expert bytes | 9 437 184 / expert | — |
| nvfp4 `cpu_moe_gbs` vs `pcie_gather_gbs` | 56.63 vs 52.60 (ratio 1.08) | CPU MoE is **not** ~2× PCIe on this box → `offload` recommended, `hybrid` off by default |
| bf16 `cpu_moe_gbs` vs `pcie_gather_gbs` | 66.53 vs 52.17 (`avx512bf16`) | same conclusion for bf16 |

Consequence for Phase 5/RAM-offload KV: at 57.76 GB/s linear H2D, moving 1 GiB costs
~18 ms. Any KV page-in strategy that puts more than a few hundred MiB/s on the exposed
decode path is dead on arrival — transfers must overlap compute or be amortised over
many tokens.

## 3. Baseline on this tree (freetoken-next @ `cac247a`, v0.1.3) — MEASURED 2026-09-16

Harness: `benchmarks/bench_pp_tg.py` (added in this commit) — one `ft serve`, greedy
`/v1/completions` with a corpus slice (`/models/servers/prompt-235k.txt`) fixed-pointed with
the checkpoint's own tokenizer to exactly N ids, 2 warmups excluded, 3 measured repeats.
PP = prompt_tokens / TTFT; TG = (completion−1)/(t_last−t_first); ITL from SSE arrival
stamps; VRAM from `/v1/stats`; GPU util from sampled `nvidia-smi` (4 Hz); host RSS summed
over the server process tree. Rows keep the full serve command, so a number is reproducible.

`ft serve --model /models/Qwen3.6-35B-A3B-NVFP4-FT --max-running-requests 1
--max-seq-len-override 16576 --memory-ratio 0.9 --cuda-graph-max-bs 1
--num-tokens 16576 --cache-type naive`

| Metric (16 384-token prompt, 128 generated, bs=1, greedy) | v0.1.3 mean (min) | 0.1.2 anchor | Δ |
|---|---|---|---|
| **PP** tok/s | **4611.1** (4605.7) | ~4104 | **+12.3 %** |
| **TG** tok/s | **158.83** (158.78) | ~147.1 | **+8.0 %** |
| TTFT | 3553 ms | — | — |
| ITL p50 / p95 | 6.16 / 6.40 ms | — | — |
| VRAM | 14.98 GiB | ~15.04 | −0.06 |
| server RSS | 22.40 GiB | ~20.5 | +1.9 |
| GPU util (sampled mean) | 99.8 % | — | — |
| KV pages | 16 576 (page_size 1) | — | — |
| output sha1 | `2a6dca88ffdc` | — | determinism pin |

Spread across repeats: PP 0.12 %, TG 0.03 % — far below the 2 % gate, so this harness can
resolve a LOW_GAIN_SURVIVOR.

**Phase-1 verdict: PASS.** v0.1.3 is *faster* than the 0.1.2 anchor on both axes, consistent
with the 23 skipped commits (notably `#438`/`#427`/`#428` quant read-path and
`#367` paged-KV reservation). The anchors in §1 are therefore **lower bounds** here, and these
v0.1.3 numbers become the new immutable guards: **35B-A3B ≥ 4600 PP / ≥ 158 TG @16K**.
Caveat for comparability: the anchor's `--num-tokens` / `--cache-type` were not recorded, so
the +12 % may partly be configuration, not code; the delta that matters going forward is
measured against *this* row, same command.

**New immutable guards from this run: 35B-A3B ≥ 4600 PP / ≥ 158 TG @16K**
(`--memory-ratio 0.9`), and **Flash-Next ≥ 1850 PP / ≥ 28.5 TG @16K** at
`--memory-ratio 0.86` — at 0.9 Flash-Next **CUDA-OOMs** in unbudgeted transients (Triton
autotune wants 256 MiB when 209 MiB are free; see EXPERIMENTS.md EXP-001b). Raw rows and the
full per-repeat detail: `docs/freetoken-next/pp_tg.jsonl`, EXPERIMENTS.md EXP-001/001b.

## 4. Context memory physics (computed from the local checkpoints, 2026-09-16)

From `config.json` of `/models/Qwen3.6-35B-A3B-NVFP4-FT` and
`/models/Qwen3.8-Flash-Next-NVFP4-Radix` (`text_config`), not from any engine report.

| Quantity | Qwen3.6-35B-A3B (`qwen3_5_moe`) | Qwen3.8-Flash-Next (`qwen4_exp`) |
|---|---|---|
| layers (linear : full attention) | 40 = 30 : 10 (`full_attention_interval=4`) | 48 = 36 : 12 (`full_attention_interval=4`) |
| attention shape | `num_key_value_heads=2`, `num_attention_heads=16`, `head_dim=256`, `hidden_size=2048` | `num_key_value_heads=2`, `num_attention_heads=24`, `head_dim=256`, `hidden_size=2560` |
| MoE | 256 experts, top-8, `moe_intermediate_size=512` | 512 experts, top-10, `moe_intermediate_size=640` |
| trained max context | **262144** | **262144** |
| sparse attention | — | indexer: 1 kv head, 4 n-heads, `head_dim=128`, **`indexer_budget=2048`**, compress ratio 4 |
| GDN (linear) state shape | 16 k-heads × 32 v-heads × 128 × 128, conv kernel 4 | 16 k-heads × 48 v-heads × 128 × 128, conv kernel 4 |

**Per-token full-attention KV (only the `full_attention` layers hold a KV row):**

| KV format | bytes/token 35B-A3B | bytes/token Flash-Next | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|
| BF16 (2·2·256·2·n_full) | 20 480 | 24 576 | 2.50 / 3.00 GiB | 5.00 / 6.00 GiB | 10.0 / 12.0 GiB | **20.0 / 24.0 GiB** |
| FP8 + per-row fp32 scale | 10 400 | 12 480 | 1.27 / 1.52 | 2.54 / 3.04 | 5.08 / 6.08 | 10.2 / 12.2 GiB |
| 4-bit (NVFP4/Turbo4-class) + scales | 5 280 | 6 336 | 0.64 / 0.77 | 1.28 / 1.54 | 2.56 / 3.08 | 5.1 / 6.1 GiB |

*(left number = 35B-A3B, right = Flash-Next; scales = 4 B per (token, kv-head) row, +16-value
E4M3 block scale for the 4-bit row.)

**Per-sequence recurrent state (independent of context length, `mamba_ssm_dtype=float32`):**
35B-A3B 32.0 MiB × 30 layers = **966 MiB**; Flash-Next 48.0 MiB × 36 = **1737 MiB**
(plus conv state 0.19 / 0.25 MiB). The engine holds these in a slot pool sized by
`linear_state_cache_ratio=2.0` — on this 16 GiB card, 4 concurrent sequences of Flash-Next
cost ~7 GiB of state before any KV is allocated. **This is the number that actually
competes with KV and expert cache, and no current planner weighs it against them.**

Consequences that steer the design:

1. 256K is inside both checkpoints' trained length; 512K/1M need RoPE extension and must be
   reported as such.
2. At 1M, BF16 KV (20–24 GiB) exceeds this card 1.5×; 4-bit KV (5–6 GiB) fits in VRAM with
   no RAM tier at all. **Compression, not paging, is the first-order answer for the dense
   attention share of these hybrid models** — the tier work matters for the *expert/PLE* side
   and for 512K+.
3. PCIe is 57.76 GB/s (§2): streaming a whole 1M BF16 KV per decode is impossible, so any
   RAM tier must move *selected* pages. `indexer_budget=2048` on Flash-Next means only ~2048
   tokens/layer are read per step → ~12 MiB/step if every read missed = 0.21 ms at 57.76 GB/s.
   That is the break-even that makes RAM-backed KV viable on qwen4_exp and *not* on a dense
   attention model.
4. Measured expert-cache auto-sizing on this host (`--moe-cache-auto`): with 15.18 GiB free
   after weights it chose **6102 expert slots and left 8268 KV tokens** — i.e. the planner
   traded 4× the KV the 16K anchor needed in favour of expert slots, then reported 1.08 GiB
   still free after CUDA-graph capture. That slack plus the state-pool blindness above is the
   evidence for Phase 6 (one ledger) and for the KV-reserve floor.

## 5. Comparison rules
Constant across an A/B: model, quant, prompt, context, generation length, sampling,
MTP state, KV type, cache config, build, profiler state. Discovery uses 4K/16K/32K only;
128K/256K are promotion-stage, 512K/1M are certification-stage (2–3 runs, report
mean/variation). Gates: KEEP ≥10 % reproducible end-to-end or a required capability;
LOW_GAIN_SURVIVOR 2–10 % (kept, never discarded); REJECT ≤2 %/noise/incorrect.

## 6. Certification matrix — the final no-regression gate (D-012)

`python benchmarks/cert_matrix.py --contexts 4096,16384,32768` (add 128K/256K at
certification). Guards are declared in the script and must stay in step with §3. Native rows
must clear their guard; GGUF rows are reported as a percentage of the **same-architecture
native row**; an unservable row reports BLOCKED with its blocker rather than vanishing.

| pair | native row | GGUF row(s) | guard / current state |
|---|---|---|---|
| `qwen35moe` | `/models/Qwen3.6-35B-A3B-NVFP4-FT` | `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf`, `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` | PP ≥ 4600 / TG ≥ 158. **Both GGUF rows BLOCKED**: expert banks mix Q3_K ×30 + Q4_K ×10 (and IQ4_XS UD) against a single-stride slot pool → needs the exact-geometry pool (Phase 7) |
| `qwen4exp` | `/models/Qwen3.8-Flash-Next-NVFP4-Radix` | `…-Unsloth-IQ4_XS/UD-IQ4_XS` (3 shards) + its `MTP/` sidecars, `…-AD-4.27…-Q4_K_M-M64` (33 shards) | PP ≥ 1850 / TG ≥ 28.5 at `--memory-ratio 0.86`. **GGUF rows BLOCKED** on shard joining, a `qwen4exp` GGUF adapter, and mapping `per_layer_token_embd` (IQ4_NL / Q5_1, ~45 GiB) onto the PLE table |
| `qwen35` dense | **none on this host** | `Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf` | no parity reference — the script prints that explicitly; close it with a native FP8/NVFP4 Qwen3.8-27B or a same-weights GGUF conversion. Dense has no expert banks, so only the I-quant prefill path (dequant + plain matmul) is under test here |
| KV format A/B | the same native row | `--kv-cache-dtype bf16 / fp8 / nvfp4 / turbo3 / turbo4 / tcq / vbr` | every format measured at 16K **and** 32K against its own BF16 row; an unmeasured TG cost blocks the merge (Kai's −33 % dense @30K is the precedent this rule exists for) |
| MTP | native row, speculation off | same row, `n_max` = 1 / 2 / 3 | effective TG is the metric; draft latency, verify latency, accepted tokens, rejection and rollback cost reported separately |

Parity is throughput and capacity, not output identity (D-013): these GGUF files carry
different weights or a different quant recipe than their native counterpart, so
token-for-token equality requires a same-weights pair.

## 7. The VRAM account, and what it costs to stop guessing the ratio

Every row here is `benchmarks/bench_pp_tg.py` on the real host, 16 384-token prompt, 128
generated, bs=1, greedy, `--cache-type naive`, 3 repeats; the only column that varies is
`--memory-ratio`. "serves" is the gate that matters first: before `engine/vram_ledger.py`
(D-014, EXP-006) Flash-Next **CUDA-OOM'd at 0.9** and only ran at the bisected 0.86.

| model | `--memory-ratio` | serves? | PP | TG | VRAM | expert slots | KV pages |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4 | 0.9 (default) | yes | 4607.6 | 158.49 | 14.59 GiB | 6043 | 16576 x 1 |
| Qwen3.8-Flash-Next NVFP4 | 0.86 | yes (pre-ledger anchor) | 1857.7 | 28.685 | 14.86 GiB | 1399 | 259 x 64 |
| Qwen3.8-Flash-Next NVFP4 | 0.9 | **yes (was OOM)** | 1861.9 | 28.68 | 14.80 GiB | 1274 | 259 x 64 |
| Qwen3.8-Flash-Next NVFP4 | 1.0 | **yes (was OOM)** | 1861.5 | 28.66 | 14.79 GiB | 1113 | 259 x 64 |
| Qwen3.8-Flash-Next NVFP4 | 0.95 | same ceiling as 1.0 | 1862.5 | 28.67 | 14.79 GiB | 1113 | 259 x 64 |

Two things that number sequence is telling. First, the guards are not pay-for-anything
decisions: the pre-ledger 0.86 row and the post-ledger 0.9/1.0 rows are the same throughput to
within 0.2 % with the same output hash (`f8bbaeb7e214`), so accounting for the peak costs
nothing that was ever actually available. Second, 0.95 and 1.0 now resolve to the *same* plan
(1113 expert slots, 259 pages) because `ceiling_bytes` floors the ratio at the modelled peak:
past the point where the reserve binds, raising `--memory-ratio` stops being a gamble and
stops being a way to win context, which is exactly what the hand-found 0.86 was pretending to
be. The expert-slot column is where the reserve is paid for, and the ledger names the lines it
paid (256 MiB autotune arena, 256 + 160 MiB/shape graph capture, 200 + 150 MiB/shape graph
pool, ~880 MiB GDN prefill workspace, ~240 MiB activations, 192 MiB per-image vision
transient, 128 MiB fragmentation).

The calibration line printed with each report is the honesty check on those numbers: the
account's held bytes against `torch.cuda.memory_allocated`. It read "over-modelled by
~0.2-0.4 GiB" on both anchors after the two measurement bugs EXP-006 lists were fixed, which
is the safe direction (context left unspent) and the quantity Phase 2 has to negotiate down.

## 8. Long context, measured (Qwen3.6-35B-A3B NVFP4, bs=1, greedy, real host)

The plan said a context length is bought out of the expert cache, so that is what was done.
`--kv-reserve-tokens <context>` is the only knob: BF16 KV, `--cache-type naive`,
`--memory-ratio 0.9`, prompt taken from a fixed 9 MB corpus slice with `--prompt-file` so a
repeat measures a prefill and not a cached prefix.

| context | PP (tok/s) | TG (tok/s) | TTFT | ITL p50 / p95 | VRAM | RSS | expert slots kept |
|---|---|---|---|---|---|---|---|
| 16 384 (guard) | 4611.3 | 158.54 | 3.55 s | 6.17 / 6.45 ms | 14.59 GiB | 21.9 GiB | 6113 |
| 131 072 (128K) | 3188.5 | 89.30 | 41.1 s | 10.99 / 12.81 ms | 14.45 GiB | 22.0 GiB | 4694 (plan said 4695) |
| 261 900 (256K) | 2353.7 | 63.83 | 111.3 s | 15.36 / 18.85 ms | 14.41 GiB | 22.0 GiB | 3185 (plan said 3183) |

The last two rows were first produced by hand (`--kv-reserve-tokens 131136`), and then
re-produced by `--kv-reserve-context`, which now makes the plan buy the context itself: 131 072
tokens served with PP 3189.0 / TG 107.18 / TTFT 41.1 s / VRAM 14.45 GiB at 32 generated tokens
(TG rises over the row above only because that row generated 64, and ITL grows with the
context the decode has to re-read).

Three things to read out of this table:

1. **TG degrades gracefully with context** (158.5 → 89.3 → 63.8) with GPU utilisation pinned at
   100 %: at 16K decode is expert-bandwidth-bound, and by 256K it is also paying for 262 000
   tokens of attention reads -- the two costs add, they do not substitute for each other.
2. **Prefill is where long context actually costs**: TTFT 41 s at 128K and 111 s at 256K, while
   PP itself falls to 2354 tok/s because the expert cache shrank to pay for the KV. Chunked
   prefill and overlap work is measured against those numbers, not against TG.
3. **512K and 1M are blocked before memory is the question.** The checkpoint's RoPE table is
   262 144 positions, so `--max-seq-len-override 524384` is refused outright by the engine. And
   the plan prices 512K of BF16 KV at 10.000 GiB -- the entire pool budget, zero resident
   experts -- with 1M at 20.000 GiB. Past 256K the order is therefore fixed: a rope-scaled
   checkpoint, then a ~4x compressed KV format (turbo4 at 4.125 bpv puts 512K back near
   2.6 GiB and 1M near 5.1 GiB, both affordable alongside a real expert cache), and only then
   RAM tiering, which the measured 57.76 GB/s PCIe ceiling (§2) makes a decode-latency problem
   rather than a capacity one.

These rows are the certification baseline for Phases 3-5 and 11: a compressed or tiered KV
format has to beat them at equal context, and the ledger prints the comparison instead of a
fresh guess.

## 9. First GGUF row measured (Qwen3.8-27B IQ3_S, dense, 4K/128)

`bench_pp_tg.py --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf --tokens 4096
--decode 128 --repeats 2 --memory-ratio 0.9 --cache-type naive --max-prefill-length 1024
--num-tokens 4096` (eager decode, as EXP-004 established):

| model | PP | TG | TTFT | ITL p50 / p95 | VRAM | RSS |
|---|---|---|---|---|---|---|
| Qwen3.8-27B IQ3_S GGUF | 2416.6 | 25.29 | 1693 ms | 39.39 / 43.45 ms | 14.43 GiB | **2.17 GiB** |

Output was coherent and the KV geometry is 4096 pages x 1 token. The account the run printed is
the interesting part: `weights:model 13.211`, `cache:gdn-state 0.287`, `cache:kv 0.250`,
reserve 0.788 against a ceiling of 13.661 (ratio 0.9 x baseline 15.179) -- a 27B dense IQ3_S
checkpoint leaves effectively nothing spare on this card, and the feasibility row says so for
each target: `128K 8.00 GiB short, 256K 16.00 GiB short, 512K 32.00 GiB short,
1M 64.00 GiB short`, at 0.06 MiB of BF16 KV per token. Long context on the dense 27B is
therefore not a scheduling question: it is 13.2 GiB of resident weights against a 15.51 GiB
card, and only a compressed KV format (which turns 8.00 GiB of 128K KV into ~2.1 GiB) plus a
smaller resident weight footprint would move it.

Two things this row establishes for the matrix:

1. **A GGUF checkpoint can be benchmarked at all** -- the harness had to be changed to tokenize
   through the engine's own `load_tokenizer` (§STATE), which is why this is the first one.
2. **2.17 GiB of RSS against the native NVFP4 anchors' 21.9-67.8 GiB.** A quantized single-file
   checkpoint streams its weights from page cache instead of pinning a host expert bank, so on
   this 91 GiB box the GGUF path leaves ~65 GiB of RAM free that the native offload path
   consumes. That is the trade the certification matrix has to weigh against the throughput
   delta, and it is the reason a 93 GB GGUF can be the better deployment even at lower TG
   (FINAL PERFORMANCE POLICY). Native comparison for this row is
   `NO_NATIVE_REFERENCE_AVAILABLE`: there is no native FreeToken 27B dense checkpoint on the host.

## 10. Compressed KV, measured (Qwen3.6-35B-A3B NVFP4, bs=1, greedy, 16K, `--cache-type naive`)

2026-09-17. Same checkpoint, same context, same budget; the only variables are the KV format and
the backend that can read it. The turbo4 row used the **first** tile readers -- one byte-gather and
one L1 lookup *per element* -- which is the number that triggered the rewrite in EXP-013. It is kept
as the baseline to beat, not as the current cost.

| arm | PP tok/s | TG tok/s | ITL p50 | VRAM | output sha1 |
|---|---|---|---|---|---|
| flashinfer + bf16 (the anchor, D-012 guard) | 4610.8 | 158.53 | — | 14.98 | `2a6dca88ffdc` |
| triton + bf16 (backend cost alone) | 4344.8 | 144.11 | 6.79 ms | 14.34 | `b7c70b36d276` |
| triton + turbo4 (per-element readers) | 3560.5 | 61.79 | 15.98 ms | 14.20 | `49e9819649ba` |

Read this as three separate costs, which is the whole point of the table. **The backend costs 9.1 %
TG** (158.53 -> 144.11) before any codec exists. **The readers cost a further 57 %** (144.11 ->
61.79) because they were instruction-bound, not bandwidth-bound: 4x fewer bytes arriving while ITL
doubled says the work was per-element address arithmetic, not memory. The codec's *accuracy* cost is
pinned separately in EXP-012 (V-side output NMSE equals the book's Lloyd-Max distortion, independent
of attention sharpness). D-017 states what the guard can and cannot mean on this path.

The hash column matters: a different backend is a different greedy continuation, so
`2a6dca88ffdc` is the *fi* guard and `b7c70b36d276` the *triton* one. Any compressed-KV guard has to
name its backend, and "output unchanged" is only a claim within one.

### What the account does with the freed bytes

The reason to compress on this host is not 16K throughput. It is that context stops costing the
expert cache its memory. Same checkpoint, `--kv-reserve-context`, 256K requested:

| KV format | KV for 256K | expert slots left | 512K | 1M |
|---|---|---|---|---|
| bf16 (EXP-009) | 5.000 GiB (the whole pool budget) | 3183 | unfundable | unfundable |
| turbo4 | 1.289 GiB | **5427 (+70 %)** | 4647 slots / +1.288 GiB over plan | 3088 slots / +3.867 GiB |

turbo4's own plan rows read `128K: fits`, `256K: fits` where bf16 said `256K: needs 5.000 GiB
(+0.003 over)`. 512K/1M stay blocked *by the checkpoint* (its RoPE table is 262 144 positions), but
the physics changed character: at bf16 a 1M context was "the entire pool budget, zero experts", and
at turbo4 it is "a smaller expert cache". That is the difference D-016 was about -- compressed KV
buys context outright, where paging would have bought it at 2.2 tok/s.
