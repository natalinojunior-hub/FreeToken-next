# JOB_REGISTRY — freetoken-next campaign

Every asynchronous process, closed on completion. Logs under
`/models/desenvolvimento/tmp/ftnext/` — never `/tmp` (a 46 GiB tmpfs that competes with host
expert banks; 26 GiB of pytest temp was moved out of it on 2026-09-17 to
`/models/desenvolvimento/tmp/tmpfs-recovered/`, which is what freed the Flash-Next loads).
GPU jobs are strictly serial: nothing else may touch the card while a benchmark or suite run
is live.

| ID | Kind | Purpose | Command / task | Output | Status |
|---|---|---|---|---|---|
| suite-p0 | pytest | full `not slow` suite on the dummy-page brick | `pytest tests -m "not slow" -q --timeout=1200` | `tmp/ftnext/suite_p0.log` | done: 1813 passed, 206 skipped, 1 failed (flashinfer b12x JIT race; 2 passed standalone) |
| guard-p0 | bench | 35B-A3B 16K guard on the dummy-page brick | `bench_pp_tg.py --tokens 16384 --decode 128 --repeats 3` | `guard_p0.log`, `guard_p0.jsonl` | done: PP 4610.1 / TG 158.75, output sha1 identical to the 4611.1 baseline |
| A7 | subagent | GGUF MoE geometry audit → refused-today matrix → stride-vs-file truth table | read-only, 3 rounds | `audits/A7-gguf-moe-geometry.md` §1-§8 | done |
| A8 | subagent | turbo3/turbo4/TCQ byte-level codec spec from llama-turbo-optimal | read-only, 2 rounds | `audits/A8-turbo-codec-spec.md` | done |
| A9 | subagent | unaccounted VRAM consumer inventory + size formulas | read-only | `audits/A9-vram-consumers.md` | done |
| flash-086 | bench | Flash-Next 16K anchor with the ledger at `--memory-ratio 0.86` | same harness, `--mem-ratio 0.86` | `flash_ledger_086.log` | done: PP 1862.5 / TG 28.67, sha1 `f8bbaeb7e214` (anchor 1857.7 / 28.685) |
| flash-100 | bench | the capability test: Flash-Next at `--memory-ratio 1.0` | same, `--mem-ratio 1.0` | `flash_ledger_100.log` | done: PP 1861.5 / TG 28.66 — serves, where 0.9 CUDA-OOMed before the ledger |
| suite-p1 | pytest | full `not slow` suite with the ledger | `pytest tests -m "not slow" -q --basetemp=.../pytest-bt` | `suite_p1.log`, `suite_p1b.log` | done: 1830 passed, 206 skipped, 1 failed (same flashinfer race) |
| p2/p3/p4/p5 | bench | anchor guards at 0.9 across the ledger's calibration revisions | same harness, `--mem-ratio 0.9` | `p2_anchors.log`, `p3_flash.log`, `p4_anchors.log`, `p5.log` | p5: Flash PP 1861.9 / TG 28.68 ✓; 35B pending |
