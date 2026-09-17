# JOB_REGISTRY — freetoken-next campaign

Every asynchronous process, closed on completion. Logs live under `/models/desenvolvimento/tmp/ftnext/`
— never `/tmp`: a 26 GiB pile of pytest temp there is what starved the Flash-Next host-bank load
twice (moved to `tmp/tmpfs-recovered/`, which took `MemAvailable` from 17 GiB to 77 GiB and made the
66 GiB bank loadable again). GPU jobs are strictly serial: nothing may touch the card while a
benchmark or suite run is live, and a `pkill` pattern that also matches the killer's own command
line ends the job that issued it.

| ID | Kind | Purpose | Command / task | Output | Status |
|---|---|---|---|---|---|
| suite-p0 | pytest | full `not slow` suite on the dummy-page brick | `pytest tests -m "not slow"` | `suite_p0.log` | done: 1813 passed / 206 skipped / 1 failed |
| suite-p1 → suite-end | pytest | same suite after each ledger, planner and ABI revision | idem, `--basetemp=` on disk | `suite_p1*.log`, `suite_final.log`, `suite_end.log` | done: 1813 → 1834 → 1837 → **1839 passed**, 206 skipped, 1 failed — the one failure is the flashinfer `fp4_quantization_120f` nvcc 13.3 compile defect (`quantization.cu:488`), reproduced with this branch's work stashed and green when its file runs alone |
| guard-p0 | bench | 35B-A3B 16K guard on the dummy-page brick | `bench_pp_tg.py --tokens 16384 --decode 128 --repeats 3` | `guard_p0.log` | done: PP 4610.1 / TG 158.75, sha1 identical to the 4611.1 baseline |
| g/p5/p7/p9 revisions | bench | 35B-A3B guard after each ledger calibration revision | idem at `--mem-ratio 0.9` | `g.log`, `p*.log` | done: PP 4601.5 → 4607.6 → 4611.3 → 4615.3, TG 158.49-158.75, sha1 always `2a6dca88ffdc` |
| flash-086 / -090 / -095 / -100 | bench | Flash-Next 16K across the ratio range the ledger replaces | same, `--mem-ratio` | `flash_ledger_*.log`, `p3/p5/g.log` | done: 0.86 PP 1862.5 / TG 28.67 (anchor), 0.9 PP 1857.3-1861.9 / TG 28.68 **where EXP-001b OOM'd**, 1.0 PP 1861.5 / TG 28.66, all sha1 `f8bbaeb7e214`; 0.95/0.75 died on **host** RAM during the 66 GiB bank build, not on the plan |
| g-35b-128k | bench | 128K bought out of the expert cache by hand | `--tokens 131072 --decode 64 --kv-reserve-tokens 131136` | `g.log` | done: PP 3188.5 / TG 89.30 / TTFT 41.1 s / VRAM 14.45 GiB, sha1 `d4b5b385a3cf` |
| g35-256k / 512k | bench | 256K and 512K feasibility | `--kv-reserve-tokens <ctx>`, 9 MB corpus | `long3.log`, `long2.log` | done: 256K **PP 2353.7 / TG 63.83** (sha1 `03ac272da761`); 512K refused — the checkpoint's RoPE table is 262 144 positions |
| ctxauto | bench | does `--kv-reserve-context` buy 128K unaided | `--kv-reserve-context`, no hand-set reserve | `ctxauto.log` | done: plan and result agree (4694 slots, 131 221 pages), PP 3189.0 / TG 107.18 at 32-token decode |
| g-flash-128k | bench | can Flash-Next buy 128K the same way | idem | `long.log` | **refused in arithmetic** (needs 6.17 GiB against a 3.35 GiB budget) — the fail-closed answer, and the quantified reason Phase 3 is on the critical path |
| gguf-27b | serve + bench | the dense GGUF row of the matrix, and a regression check on the resized KV budget | `bench_pp_tg.py --model …IQ3_S…gguf --tokens 4096 --decode 128`, plus a direct `ft serve` at 0.9 | `gguf27b_run.log`, `gguf_serve*.log` | done: serves at the default 0.9 (EXP-004 needed 0.8); PP 2416.6 / TG 25.29 / VRAM 14.43 GiB / **RSS 2.17 GiB**, sha1 `9035a116947b` |
| ornith / tiel | serve | do the MoE GGUFs load today | `ft serve --model <gguf>` at 4K | `ornith_try.log`, `tiel_try.log` | done: both fail closed in the loader on mixed expert geometry, with the layer map named (EXP-011) |
| build-ext | build | rebuild `_cpu_moe` for the weight-format probe | `MAX_JOBS=6 python setup.py build_ext --inplace` | `build_ext.log` | done: `max_weight_format_id() == 6`, all seven formats dispatch |
| A7 | subagent | GGUF MoE geometry → refused-today matrix → stride-vs-file table | read-only, 3 rounds | `audits/A7-gguf-moe-geometry.md` | done; its first-pass headline claims were corrected against the source and the host, and §8's offset-derived table closed the byte-layout question (1194/1194 tensors match; ids are llama.cpp's) — EXP-011 |
| A8 | subagent | turbo3/turbo4/TCQ byte layout + the row-size correction | read-only, 2 rounds | `audits/A8-turbo-codec-spec.md` | done; §1 confirms ARCHITECTURE §4's bytes (turbo3 14 B/32 elems = 112 B per 256-element row, turbo4 66 B/128 = 132 B) and retracts the 138/276 B figures that an intermediate pass had proposed |
| fmt-coverage | read | what actually blocks the refused MoE GGUFs at execution time | source pass over `cpu_moe_ext.cpp` / `cpu_executor.py` / `dequant.py` | this file | done: no GPU reader is missing (offload dequantizes every `BLOCK_SHAPE` type); `_cpu_moe` has dot kernels for ids 2/12/14 only and `_resolve_gguf_format` takes one format for both banks |
| A9 | subagent | unaccounted VRAM consumers, size formulas, measured graph/autotune/JIT numbers | read-only | `audits/A9-vram-consumers.md` | done; the ledger's line items are built from it and calibrated against the anchors |
