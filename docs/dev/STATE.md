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
## 2026-09-24 Gate 1 control status

Gate 1 is blocked. The graph-on k0 harness run at HEAD `3748fbd` reused 4032 prompt tokens on repetitions 2 and 3, so its PP mean is invalid. See `/models/desenvolvimento/ft-campaign2/campaign8/gate1-abort-20260924.md`. Do not compare k1 or change production code until the harness starts a fresh server and cold cache for each repetition.
Cold-reset harness fix validated. Three fresh graph-on servers per mode produced k0 PP/TG `1464.9/44.34`, k1 `1464.0/36.28` tok/s. Output hashes were stable within each mode but differed between modes; k1 is 18.2% slower and fails parity, so optimization is blocked.
Output-length check complete: with identical greedy sampling and fresh graph-on servers, 128 tokens gave k0/k1 TG `47.74/41.58`; 256 gave `47.30/43.80`. First decoded divergence is at output boundary offset 83 in both lengths. k1 logs confirm graph capture and `k=1` draft/verify records. Optimization remains suspended; Gate 2 not started.
k1 correctness gate blocked. At decode 256, k0 graph TG `47.30`, k1 eager `38.95`, k1 graph `43.80` tok/s. k1 eager and graph hashes match (`91e6de9c85b2`), but both differ from k0 (`3af3056b98c0`). Graph capture and speculative cycles are confirmed. Exact token IDs/logits are unavailable through current SSE; `logprobs` is rejected. Keep k0 default and add opt-in internal token/state trace before any Gate 2 work.
Added opt-in `FREETOKEN_TOKEN_TRACE` for decode/spec draft/verify/commit token records with bounded flush. No exact GPU token trace was accepted yet; k1 remains disabled. Final `make ci` currently has 3 failures: two `tests/moe/test_prefill_hit_d2d.py` and one `tests/scheduler/test_spec_reject_frees_pages.py`; no fix was inferred from them.
Token trace delivery recovered. Harness now accepts `--token-trace`, injects the absolute path into the server environment, prints it, validates nonempty JSONL after process reap, and distinguishes it from result JSON. Smoke command used `--tokens 4096 --decode 1`; trace contained one `decode` ID record. Paired cold graph traces at 4096/256 produced 256 k0 decode IDs and 256 k1 committed IDs. First committed divergence: index 26, k0 token `1404`, k1 token `11`; prior 26 committed IDs match. k1 remains blocked; no causal state proof or fix.
Cause investigation complete: first committed mismatch remains index 26 (`k0=1404`, `k1=11`). At the aligned prefix, k0 logits are `1404=18.875`, `11=18.75`; k1 eager target scores are `11=19.125`, `1404` second, margin `0.125`. Both choose own top token. This is a near-tie trajectory difference; no acceptance, slot-index or graph defect proven. k1 acceptance is 90/165 (54.5%), 1.545 committed tokens/cycle, explaining TG loss pressure. Keep k1 disabled.
Historical acceptance audit: post-PLE `28f7ec6` reports 75.2% -> 72.8%, but prompt bytes/hash and complete flags are unavailable. Separate 96-token depth probe reports 49.2% under different protocol. Current valid k1 trace is 90/165 (54.5%), 1.545 committed/cycle. At mismatch 26, k0/k1 top scores differ by a 0.125 near-tie; no causal state defect proven. k0 remains default.
Gate 2 closed (2026-09-24, HEAD `3748fbd`): k1 eager `FREETOKEN_DEBUG_LOGIT_DUMP` run was compared against the k0 eager dump by absolute position. At the first verify (identical post-prefill state and token), row-0 logits already differ from k0 by max |d| 1.125; within k1 alone, a 2-row versus a 3-row verify over identical state differs by 0.559. Matched-history rows stay flat at 0.5–1.4 max |d| across the common prefix. GDN recurrent relative drift grows smoothly 0.8% -> 1.8%, and replay cycles show no jump versus accept-all cycles. Source audit (`campaign8/k1-source-audit-20260924.md`): sampler/acceptance, GDN verify (`_spec_verify_step_by_step` uses the decode kernels), QSA and MoE are row-independent; the residual comes from multi-row GEMV/GEMM reduction order. Verdict: the index-26 flip (margin 0.125) is numerical, and no defect was found. No parity forcing.
Gate 3 (graph-on cold 4096/256, `FREETOKEN_DEBUG_SPEC_TIMING`, synchronized attribution only): per k1 cycle, verify_forward is 31.98 ms, draft 1.88 ms, snapshot+prepare+commit 0.60 ms and replay 1.13 ms on average, for 35.6 ms total, consistent with the uninstrumented 1.545/43.88 = 35.2 ms. The k0 step is 21.2 ms (47.25 tok/s). Verify costs 1.51x a decode step; break-even needs about 1.68 committed tokens per cycle (about 68% acceptance) against the measured 54.5%. The loss comes from low acceptance multiplied by the 2-row verify cost. Draft, replay and graph overhead are small (<8%), and there is no implementation regression. Keep k0 as default. Next: raise acceptance or lower the verify expert traffic with a new measured mechanism; the closed cache/hybrid/overlap/MoE-kernel/draft-graph hypotheses are not reopened.
