# MTP regression closure — 2026-09-29

## State at session stop

The automatic MTP regression work is **not certified complete**. Runtime fixes were transferred from the isolated `closure-runtime` worktree onto the main checkout. The final user instruction stopped the last validation batch and all agents. Do not resume without a new user request.

Final CPU CI passed: 1,967 passed, 833 skipped, 17 deselected, 3 warnings. The focused GPU batch passed 50 tests (GDN, QSA backend, VMM residency). It includes strict GDN transaction parity: FP32/BF16, row checkpoints/compact replay, accepted prefixes 0–4, each recurrent output and the next decode matched RAW exactly. VMM resize fault injection, geometry, and VRAM guard passed 47 tests with 7 CUDA skips; scheduler transaction matrix passed 24 CPU tests. The total source-tree SHA-256 recorded after formatting is `0798658043cc97fc066dfa5b31a87b2a6e482c948c1f6fa0787381fb569a5fb0` (716 files across `python`, `scripts`, and `tests`).

The automatic 16K final benchmark did **not** run. Its server was stopped at API-ready before any measured request; there is no final JSON/result. Final pressure and quality runs, full GPU CI, and 256K certification also did not run. The latest valid automatic 16K run is `auto-v7`: 102.2586 tok/s mean, 98.9293 minimum; six 256-token outputs had SHA-1 `76a5508fd576`. It predates the final BF16 recurrent-state, VMM rollback, idle-regrowth, and pressure-aware precision fixes. The measured forced-k4 ratio-0 reference was 103.58 tok/s; same-pool forced-k4 was 102.6289. Do not compare these as a final paired benchmark.

QSA `[16,32,4]` vs `[32,16,4]` A/B/A produced about 100.06/99.90/100.30 tok/s; the stock shape was restored. The reported 9% was a kernel-level result, not an end-to-end gain. The 113 tok/s target is unproven. No final 256K result or real accepted-prefix QSA-vs-RAW transaction proof exists. Earlier usage quality was 20/20, before the final fixes; it is not final certification.

## Root causes fixed in the candidate

- MTP defaults now resolve from model-head metadata; single-head/single-request defaults select k4 without additional operating flags. Explicit choices remain authoritative.
- Prefix pruning no longer becomes terminal k0; the final-token fallback remains available. Accepted outputs and page rollback are covered by the k4 0–4 CPU matrix.
- RAW generation now finishes after the requested device tokens instead of losing the overlap-delayed final token. Offline result IDs remain unique across repeated calls.
- State floors are included in, not added after, the pool budget split. GDN snapshot demand follows the request ratio. At one request and ratio 2, the GDN pool is seven total slots; lower demand is not guessed by deleting rollback state.
- Scheduler re-probes no longer invalidate their own controller evidence. Live caps no longer cause false economic-epoch changes; source fingerprints include all runtime source. Tail-clamped depth samples reflect work actually done.
- VMM backing grows/shrinks by native granules, so retained prefixes stay mapped and warm. Live grow prices the exact post-floor per-pool geometry, including non-monotone splits; a cached curve reduced a 10,000-candidate scan to 0.526 ms. Foreign-memory grow refusals no longer become permanent learned reserve debt. Partial resize failures compensate to the safe resident prefix; cleanup failures are fatal rather than leaving cache accounting inconsistent. Idle regrow uses the actual compensated footprint and does not learn refused bytes.
- Greedy MTP requests automatically transition through the serialized scheduler loop; non-greedy requests retain overlap scheduling. Target-fed KV priming is automatic, capability-checked, and bounded to the existing eight-row chunk. The Qwen4 MTP head preserves all row-wise state work while running MLP/MoE on the last row only.
- BF16 GDN state now rounds each recurrent carry at the same storage boundary as one-token decode. Automatic precision starts at FP32 plus compact state, and retries BF16 only when an automatic GDN+MTP memory candidate fails its actual VMM floor or context plan. Retry is scoped to each KV/MTP candidate; explicit dtype settings are preserved.

## Required next steps

1. Re-run the final, bare automatic ISTA 16K benchmark, six measured repeats after one warmup. Compare outputs to the archived RAW SHA and compare TG with the 103.58 tok/s forced-k4 reference. Record per-run depth, audit and pressure data.
2. Run the 16K physical-pressure/recovery check and final 20-task usage quality suite on the same frozen source. Run the full GPU CI suite; the 50-test GPU batch is focused coverage, not full CI.
3. Only if 16K automatic MTP meets the verified k4 baseline, certify 256K. Require output parity, recovery under pressure, long-needle and usage quality, and mean TG loss no greater than 10% versus the paired 16K result.
4. Test AD and NVFP4 independently at their supported contexts. Pfeiffer has no MTP artifact and should be treated as automatic RAW. Unsloth weights were unavailable. No 512K decode certification was found in the archive.
5. Revisit accepted-prefix QSA commit-vs-RAW proof across the compressed 128-row boundary before claiming complete state-transaction certification.

## Bare automatic benchmark commands

Run one GPU job at a time from the repository root. These benchmark-harness runs use model plus context length; the `serve-arg` only sets that context. Archive each JSON and log under `/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929/`.

```bash
TMPDIR=/models/desenvolvimento/tmp CUDA_HOME=/models/outros/cuda-13.3 \
PATH=/models/outros/cuda-13.3/bin:$PATH PYTHONPATH=python \
.venv/bin/python benchmarks/bench_pp_tg.py \
  --model /models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS \
  --tokens 16384 --decode 256 --repeats 6 --warmups 1 \
  --prompt-file /models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign26/prompt-470k.txt \
  --serve-arg=--max-seq-len --serve-arg=16704 \
  --label auto-final16 --no-history \
  --json /models/desenvolvimento/old/freetoken-next/mtp-regression-20260929/auto-final16.jsonl
```

Only after 16K passes its gates, repeat with `--tokens 261824`, `--serve-arg=262144`, and label `auto-final256`; use a distinct JSON path. Pair the 256K mean with the final 16K mean and enforce the 10% loss limit.

## Evidence and safeguards

Archived session evidence is under `/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929/`. `final-runtime.patch` contains the complete candidate runtime patch relative to checkpoint `c6ef2af`; all 15 changed runtime files match the candidate byte-for-byte. Candidate tests also match except for the unrelated IQ4 tolerance change, which was deliberately left untouched. The checkout had substantial pre-existing work, including archive/doc removals. Those unrelated changes were preserved and must not be swept into the MTP task commit. The local benchmark harness file `scripts/bench_runner.py` is pre-existing work.

Runtime and session evidence commit: `c8c30a9` (`fix(mtp): restore automatic state and residency correctness`). Documentation/ai-memory closure followed the user's instruction to stop validation and all agents.

The original 512K claim is recorded as user history, not verified decode evidence. Capacity planning, prompt ingestion and decode certification are separate gates.

## Final validation — resumed run, 2026-09-29 (HEAD `317588b`, source `c8c30a9`)

The user authorized resuming validation. All checks ran serially on the frozen source (no runtime file changed during the batch; `git diff HEAD -- python tests` stayed empty). Evidence: `/models/desenvolvimento/old/freetoken-next/mtp-regression-20260929/` (`closure-orchestrator.status`, `full-ci-final.log`, `bench16-final.log`, `auto-final16.jsonl`, `diag16.jsonl`, `pressure-final16.result.json`, `usage-final16.json`, `val16.status`).

- Full CI (GPU attached): `make ci` passed — 2693 passed, 209 skipped, 17 deselected, 3 warnings. Formatting left all 685 source files unchanged.
- Bare automatic ISTA 16K (`auto-final16`, one warmup + six measured, context 16704, no flags beyond context): mean TG 102.066 tok/s, min 98.93→99.11 range, all six outputs 256 tokens with SHA-1 `76a5508fd576` — RAW parity holds. **The 103.58 tok/s gate was not met**, so 256K certification was not run and is not claimed. Repeat diagnostics on the same frozen source: bare rerun mean 101.612; explicit `FREETOKEN_MAMBA_SSM_DTYPE=bfloat16` (skips the FP32-first probe) mean 102.215 — the probe is not the tax.
- Physical pressure/recovery at 16K passed: before/under/recovered all exact 256 tokens, identical SHA, `pressure_error=None`, KV 389 pages and GDN 7 slots invariant, expert rows 5853→5773→5950 (recovered above baseline).
- Usage quality at 16K passed: 20/20 tasks (`gates.usage_20_of_20 = true`).

### Measured attribution of the 103.58 gap (~1.5–2%)

1. Automatic GDN snapshot default (`linear_state_cache_ratio=2.0`) prices ~59–90 expert rows below the ratio-0 opt-out that produced the 103.58 reference; the reference itself is correct (SHA `76a5508fd576`) but used an explicit opt-out and pre-final-fix code — the pairs were never re-run on the final HEAD.
2. Final correctness fixes (BF16 carry rounding, FP32-first probe ladder, probe-conservative reserves) cost roughly 0.4–1% end-to-end versus the old-code same-pool 102.63; the FP32→BF16 discard path leaks no cache, but a discarded probe candidate is never explicitly destroyed (`python/freetoken/engine/engine.py:1884-1912`) — unproven residue, worth an isolated experiment.
3. Steady automatic TG already matches the legitimate same-pool forced-k4 reference within noise; the remaining mean loss is the ratio-default VRAM economics plus in-run prefill-pressure shrink churn (guard shrinks ~100 rows at prefill and regrows mid-decode).

### 113.36 tok/s path — verified status

`notes/open-investigations.md` (#1/#2) recorded 113.36 = 104 × a QSA M=5 kernel-level +9%. Root cause #1 (premature k0 via under-priced transients) is fixed by `c8c30a9`; automatic runs keep k4 (365/384 cycles). The M=5 candidate `[32,16,4]` was implemented, tested end-to-end A/B/A (≈100.06/99.90/100.30 — no gain; kernel-level 9% did not translate), reverted, and is absent from the tree. Conclusion: **113.36 is falsified as written**; its re-evaluation precondition ("once k4 headroom is cleared") is still unmet because automatic sits ~1.5% below 104. Remaining levers, in measured order of size: adaptive snapshot-cache economics (price ratio-2.0 default against measured prefix-hit benefit under binding VRAM), prefill-transient overpricing that triggers mid-run shrink/regrow churn, the ~0.4–1% correctness-fix dtype cost, and the probe-discard cleanup experiment. None is certified; none may be claimed without a bare automatic 16K rerun meeting its gate.

### Open items after this run

1. Re-pair the references on the final HEAD (forced-k4 ratio-0 and ratio-2.0, six measured repeats) to convert attribution items 1/2 from cross-code inference into same-code measurements.
2. Root-fix batch for the gap above; each fix requires CI + bare 16K rerun before any gate claim.
3. 256K certification only after a bare automatic 16K meets 103.58 on frozen source (not met 2026-09-29).
4. Accepted-prefix QSA commit-vs-RAW proof across the compressed 128-row boundary remains open.
5. The validation harnesses used here were copied from the closure worktree into `scripts/` (`physical_pressure_acceptance.py`, `qualify_long_context.py`) plus the serial `closure_orchestrator.sh`; pre-existing checkout WIP remains untouched and uncommitted.
