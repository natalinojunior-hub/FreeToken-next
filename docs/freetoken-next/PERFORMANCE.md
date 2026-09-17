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
