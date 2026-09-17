# JOB_REGISTRY — freetoken-next campaign

Every asynchronous process, closed on completion. Logs under `/models/desenvolvimento/tmp/ftnext/`
— never `/tmp`: a 26 GiB pile of pytest temp there is what starved the Flash-Next host-bank load
(moved out to `tmp/tmpfs-recovered/`, which is what made the 66 GiB bank loadable again). GPU
jobs are strictly serial: nothing else may touch the card while a benchmark or a suite run is
live, and a `pkill` pattern that also matches the killer's own command line will end the job.

| ID | Kind | Purpose | Command / task | Output | Status |
|---|---|---|---|---|---|
| suite-p0 | pytest | full `not slow` suite on the dummy-page brick | `pytest tests -m "not slow"` | `suite_p0.log` | done: 1813 passed / 206 skipped / 1 failed (flashinfer b12x JIT race; 2 passed alone) |
| suite-p1, p1b, p1c, p9 | pytest | same suite after each ledger revision | idem, `--basetemp=` on disk | `suite_p1*.log`, `p9.log` | done: **1834 passed / 206 skipped / 1 failed**, same single race |
| guard-p0 | bench | 35B-A3B 16K guard on the dummy-page brick | `bench_pp_tg.py --tokens 16384 --decode 128 --repeats 3` | `guard_p0.log` | done: PP 4610.1 / TG 158.75, output sha1 identical to baseline |
| flash-086 / -100 / -090 | bench | Flash-Next 16K at three ratios with the ledger | same, `--mem-ratio` | `flash_ledger_*.log`, `p5.log`, `g.log` | done: 0.86 PP 1862.5 / TG 28.67; 1.0 PP 1861.5; 0.9 PP 1857.3 / TG 28.68 — **0.9 served, where EXP-001b OOM'd** |
| g-35b-090, p5, p7, p9 | bench | 35B-A3B guard after each calibration revision | idem | `g.log`, `p*.log` | done: PP 4601.5 → 4615.3 → 4607.6 → **4611.3**, TG 158.49-158.75, sha1 always `2a6dca88ffdc` |
| g-35b-128k | bench | 128K context bought out of the expert cache | `--tokens 131072 --decode 64 --kv-reserve-tokens 131136` | `g.log` | done: **PP 3188.5 / TG 89.30 / TTFT 41.1 s / VRAM 14.45 GiB**, sha1 `d4b5b385a3cf` |
| g35-262144, -524288 | bench | 256K / 512K feasibility | idem | `long2.log` | blocked by the checkpoint's RoPE table (262144 positions) — the harness's +64 headroom pushed the request 96 positions past it; retried at 261 900 |
| g-flash-128k | bench | can Flash-Next buy 128K the same way | idem | `long.log` | **refused in arithmetic** by the plan (needs 6.17 GiB against a 3.35 GiB budget) — the correct fail-closed answer, and the reason Phase 3 exists |
| gguf-27b | serve | does the dense GGUF still serve after the ledger resized the KV budget | `ft serve --model …IQ3_S…gguf --memory-ratio 0.9 --num-tokens 4096 --cuda-graph-max-bs 0 --max-prefill-length 1024` | `gguf_serve.log` | server ready at 0.9 (EXP-004 needed 0.8); generation not yet re-measured |
| A7 | subagent | GGUF MoE geometry → refused-today matrix → stride-vs-file truth table | read-only, 3 rounds | `audits/A7-gguf-moe-geometry.md` §1-§8 | done |
| A8 | subagent | turbo3/turbo4/TCQ byte-level codec spec + row-size correction | read-only, 2 rounds | `audits/A8-turbo-codec-spec.md` | done |
| A9 | subagent | unaccounted VRAM consumers + size formulas | read-only | `audits/A9-vram-consumers.md` | done |
