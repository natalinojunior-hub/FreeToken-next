# State — 2026-09-23

## Critical-path MoE profiling — 2026-09-24

HEAD `f83ee23`. Prompt file SHA256 `645f46bf134e597f1f70697d699ea70a83bf9cbe615681dad2b5eca67fbbfcf`. Matched cold 4096-token, 32-token-output, eager (`--no-graph`) runs used separate fresh servers, identical flags and zero prefix reuse. Uninstrumented: k0 PP 1465.5, TG 38.48, VRAM 14.75 GiB, output `9b6cb430f2f6`; k1 PP 1466.5, TG 21.82, VRAM 14.72 GiB, output `28e97bbdceb2`.

Opt-in `FREETOKEN_MOE_EVENTS` recorded CUDA events around the two existing decode `ggml_moe_a8_vec` calls without per-layer synchronization; one final collection occurred on termination. k0: 1488 gate/up + 1488 down samples, interval unions gate/up 34.860 ms and down 33.579 ms, record overhead 54.313 ms. k1: 1221 + 1221 samples, unions 59.887 ms and 58.992 ms, overhead 47.987 ms. Event totals are about 8.5% of k0 and 11.0% of k1 diagnostic decode wall time, with substantial injected overhead; they are attribution evidence only. The measured region is not the dominant end-to-end bottleneck. Existing current-head miss evidence remains about 0.847 GB per k1 cycle; no new cache policy was tested. Decision: NO-GO for kernel changes. Next action: add outer eager verify/draft/replay events or close the path; do not optimize MoE GEMV from this profile.

## Current-head MTP bank audit — 2026-09-24

At HEAD `ab7fe76`, Gate A is closed with no production change. CPU/metadata proof found the IQ4_XS MTP shard contains `blk.48.ffn_{gate,up,down}_exps.weight` (512 experts, GGML type 8). `qwen4_exp/gguf.py` derives block 48, appends it after target banks 0–47, and `ModelConfig.num_moe_layers=49` with `mtp_expert_bank=True`; `Qwen4ExpMTP` routes its MoE layer to bank 48. Focused proof: `tests/models/qwen4_exp/test_config.py` 23 passed. The old claim that MTP aliases target layer 0 is false on current HEAD. `weight.py` still has a stale explanatory comment saying draft expert tensors are skipped; the separate expert-source loader is the actual path.

Gate B remains no-go for implementation. Current IQ4_XS decode calls `ggml_moe_a8_vec` twice per MoE layer (gate/up and down); large-prefill dequant reuse already exists. No current interval-union GPU event profiler covers this path, so there is no evidence for a safe >5% mechanism. Next distinct step is one matched cold 4K k0/k1 eager event profile before any kernel edit.

## MTP depth diagnostic handoff

Production remains on k1; this diagnostic did not remeasure or change the production profile. Acceptance survivors: k1 31/63; k2 q1=27/54, q2=14/27; k3 q1=26/53, q2=13/26, q3=3/13. Expert misses per SPEC committed token (not all request tokens): k1/k2/k3 = 0.549/0.646/0.791 GB. Counters synchronize each cycle. Optimistic k2 cost is 55.27 ms versus a 48.8 ms break-even; k2+ is NO-GO. k4, k5, and 16K probes were not run for economic reasons. CPU coverage uses fake target/draft outputs, replay, and a scalar pool-slot control-flow fake; it does not prove Qwen/GDN/PLE numerical parity. EOS/cancel integration remains untested. Evidence: `/models/desenvolvimento/ft-campaign2/mtp-depth/k2-correctness.md` and `k2-correctness.log`. Current integrated validation record: `/models/desenvolvimento/ft-campaign2/mtp-depth/ci-final.log`; consult its recorded exit status. The CI result cited below is historical. Retain k1 and evaluate routing-compatible ways to lower target-pool expert bytes using the existing 4K cost model before implementation or GPU A/B.

Doing: activation complete; expert-cache policy NO-GO. Evidence: Campaign 5
ledger and expert pool analysis under `/models/desenvolvimento/ft-campaign2/e1/`.

- The final 16K GGUF/turbo3 schema-2 profile `3528c0705928df183b3d878c` selects
  MTP k1 and the draft graph: 41.04 committed tok/s, cold PP 2077 tok/s.
  A normal restart without `--spec-mtp` loaded k1 and captured verify/draft
  graphs. `FREETOKEN_DRAFT_GRAPH=0` disables draft capture. Three schema-1
  profiles with graph disabled remain invalid.
- Commits `a96d22a` and `f75f717` add opt-in eager expert tracing and repair
  the server readiness gate. Tracing is diagnostic, not a TG benchmark.
- Actual 2,886-slot LRU misses: 137.57 GB over 149 cycles at 4K and 129.46 GB
  over 145 cycles near 16K. Replay matches within 0.04%; hindsight Belady
  saves 38-42% bytes but requires future routing. Fixed-budget pool shifts
  increased 4K misses. No cache-policy change or production A/B followed.
- Residual bottleneck: 0.89-0.92 GB expert misses and 17.8-18.4 ms copy per
  k1 cycle. The optimistic Belady cost model reaches only about 50/55 tok/s
  at 4K/16K; the >100 committed tok/s goal remains unmet.
- Final integrated CI command: `TMPDIR=/models/desenvolvimento/tmp make ci`.
  Its result is stored in `/models/desenvolvimento/ft-campaign2/e1/ci-final.log`.

Next: evaluate a routing-compatible way to lower target-pool expert bytes per
committed token. Use a 4K cost model before any implementation or GPU A/B.
Do not retry unchanged LFU/ghost, CPU hybrid, or copy/compute overlap.
- Campaign 7 (2026-09-24): matched cold IQ4_XS controls at HEAD `437b5a6` measured k0 36.60 TG/1456 PP and automatic k1 42.53 TG/1466 PP, with 4101 prompt tokens, zero prefix reuse, and 256 output tokens. k1 resolved profile `3528c0705928df183b3d878c`; 109/145 drafts accepted. One pair only; output hashes differ, no quality equivalence claim.
- Opt-in graph trace used 96 output tokens, 57 SPEC cycles, and 8 profiled cycles. Later cycles averaged 0.847 GB expert misses; target/MTP totals were 55.394/1.467 GB. Synchronized profiler TG is diagnostic. Conditional removal of all later-cohort copies projects 70.47 tok/s under explicit assumptions, not a measurement or physical ceiling. Y=1/2/4 geometry changed only tiny dense kernels; production remains Y=1.
