# STATE — freetoken-next

Snapshot date: 2026-09-17. This is the current truth; history goes to
EXPERIMENTS.md / DECISIONS.md, not here.

## Where we are

**Phase 1 (clean fast base) DONE. Phase 6 (the VRAM ledger + governor) DONE: one account
owns the ceiling, the reserve, the expert/KV split and the context rows, and a measured peak
now ratchets its own floor. 128K and 256K work on the 35B-A3B. Phase 2 (native GGUF) is
committed and has its first measured row; its MoE rows are gated by Phase 7, which is now
known to be a pool-keying change and not a missing kernel.**

| Step | Status | Evidence |
|---|---|---|
| Base = current upstream FreeToken `cac247a` (v0.1.3), branch `next` | done | `git fetch upstream` → 0 behind; PROVENANCE.md §1 |
| Durable docs created | done | `docs/freetoken-next/` (7 files + audits A1…A9) |
| Builds cleanly on host | done | editable `.[accel]` install in `.venv` (py 3.12.14, torch 2.11.0+cu130, triton 3.6.0); `_pinned_tensor`, `_cpu_moe`, `_ple_store` all compiled; `ft --version` → 0.1.3 |
| Test suite green | done | `pytest tests -m "not slow" --basetemp=<disk>` → **1839 passed, 206 skipped, 1 failed**. The one failure is environmental, not a regression: flashinfer's `fp4_quantization_120f` module does not compile with this host's nvcc 13.3 (`quantization.cu:488: error: alignment cannot be set to less than the default alignment`), and the two b12x tests in `tests/moe/test_nvfp4_backends.py` either skip or fail depending on which test file built first in the process. Reproduced with every change on this branch stashed, and green when the file runs alone (EXP-008) |
| Baseline reproduces anchors | **PASS on both models** | 35B-A3B @16K: PP 4611 / TG 158.8 / VRAM 14.98 GiB / 99.8 % util; Flash-Next @16K: PP 1857.7 / TG 28.685 / VRAM 14.86 GiB / RSS 67.82 GiB / 99.99 % util (PERFORMANCE.md §3, EXPERIMENTS.md EXP-001/001b) |
| Dummy-page brick (`1c81064`) | **done, guards hold** | the pool allocates `num_pages + 1` and both budget formulas now price it, including `kvcache/base.py::solve_num_pages`; 35B @16K re-measured PP 4610.1 / TG 158.75, **output sha1 identical** (EXP-005) |
| VRAM ledger, brick 1 | **done, guards hold at the default ratio** | `engine/vram_ledger.py` + `cache_budget.ceiling_bytes`; one account, printed at startup, `--memory-ratio` a cap and the modelled peak a floor with only the shortfall charged. **Flash-Next now serves at `--memory-ratio 0.9`** (EXP-001b's OOM is gone) and at 1.0 lands on the same geometry as the hand-found 0.86 (EXP-006) |
| VRAM governor, bricks 2-4 | **done, guards hold** | `VramLedger.decide()` owns the expert/KV split and prints a `MemoryPlan` with 128K/256K/512K/1M rows priced against what the split *leaves* for KV; `ContextDemand` + `plan_for_context` price the same targets from the other side and back the `--kv-reserve-context` flag; a measured peak ratchets the reserve floor up for every later plan. Final gate: 35B PP 4610.8 / TG 158.53, Flash PP 1857.3 / TG 28.68, hashes unchanged (EXP-008, EXP-009) |
| **128K / 256K practical (35B-A3B)** | **PASS** | bought out of the expert cache with `--kv-reserve-tokens`: **128K PP 3188.5 / TG 89.30 / TTFT 41.1 s / ITL p50 10.99 ms**, **256K PP 2353.7 / TG 63.83 / TTFT 111.3 s / ITL p50 15.36 ms**; VRAM 14.4-14.5 GiB, RSS 22.0 GiB. The plan predicted the surviving expert cache to one slot (4695 vs 4694, 3183 vs 3185) (EXP-009) |
| 512K / 1M on these checkpoints | **BLOCKED by the checkpoint, not the engine** | `--max-seq-len-override 524384` is refused: the Qwen3.6-35B-A3B RoPE table is 262 144 positions, so anything past 256K needs a rope-scaled config. Memory agrees: the plan prices 512K BF16 KV at 10.000 GiB, which is the entire pool budget (0 experts) and 1M at 20.000 GiB (EXP-008, EXP-009) |
| **First GGUF matrix row measured** | **PASS** | Qwen3.8-27B IQ3_S dense: PP 2416.6 / TG 25.29 / TTFT 1.69 s / VRAM 14.43 GiB and **RSS 2.17 GiB** (the native offload anchors need 21.9-67.8 GiB); its own account says 128K KV is 8.00 GiB short of what 13.2 GiB of resident weights leave behind (PERFORMANCE.md §9) |
| CPU MoE ABI gate | **done** | `_cpu_moe.max_weight_format_id()` + `compiled_extension_supports_format()`: a `.so` predating the GGUF K-quants now refuses `q4_0/q4_k/q6_k` with the rebuild instruction instead of faulting inside a worker thread that already holds the bank table (the activation probe's twin) |
| Benchmark a bare `.gguf` | **done** | `benchmarks/bench_pp_tg.py` now tokenizes through `utils/hf.load_tokenizer` (the engine's own GGUF vocab path) instead of `AutoTokenizer`, which can only read an HF directory |
| Source audits (9, parallel) | **done** | `docs/freetoken-next/audits/A1…A9.md`; A7 = GGUF MoE geometry + refused-today matrix + the stride-vs-file table that closed the byte-layout question, A8 = turbo codec byte layout (turbo3 112 B / turbo4 132 B per 256-element row), A9 = unaccounted VRAM consumers |
| Phase 2 GGUF loader | **committed** (`86af2d3`); dense row measured, MoE rows gated | EXP-004: the IQ3_S 27B GGUF generated coherent, factually correct text and the NextN/MTP drop warned as designed; EXP-010 measured it (PP 2416.6 / TG 25.29 / RSS 2.17 GiB). The two MoE GGUFs that geometry blocks are gated by Phase 7 pool-keying, not by a missing kernel |
| Phase 3 Turbo KV | **codec + pool + both fused paths landed; contiguous reader rewrite reverted** | `turbo_kv` (NMSE at Lloyd-Max theory: 0.0339 turbo3 / 0.0092 turbo4), `turbo_pool` (50 B / 66 B per token-head-slab vs 256 B bf16 = 5.12x / 3.88x, `kv_cost`/`unit_bytes` parity pinned), `turbo_attn` readers, `COMPRESSED` branches in the decode *and* prefill kernels, and `--kv-format` wired through config/factory/ledger. Serves on the host: plan gives **6183 slots + 8440 pages** where bf16 gave 6113 + 8238, and at **256K: 5427 slots vs 3183** (PERFORMANCE §10, EXP-012/013). The contiguous-load rewrite passed correctness pins but regressed the matched 16K A/B to TG 45.74, so the pre-rewrite readers are restored; EXP-013 records the evidence and next bounded-tile hypothesis |
| Decision B MoE routing instrumentation | **partial / blocked on host load** | `--moe-collect-stats` is parser-tested and logs graph-safe aggregate/per-layer counters at worker shutdown (`56774dc`); detached reproduction reached only bank 176/192 and was killed with `exitcode=-9`, consistent with the 63.46 GiB bank footprint exceeding the host's 62 GiB available RAM (EXP-015) |
| Native MTP config/reader seam | **content-correct live for k=1; throughput unmeasured** | Full pipeline implemented and validated against the real checkpoint (`--spec-mtp 1`, 16K, single request, naive cache): weight loading, MoE-bank aliasing, scheduler spec-loop (EXP-021/022/024/025/026). Five real bugs found and fixed via live serving (EXP-027/028), the last a state-corrupting reject-path rewind bug found by a third Opus 5 pass. Retest confirms MTP output is content byte-identical to `--spec-mtp 0` on both a short and a previously-diverging longer prompt. k=2/3 fixed by inspection, not yet live-tested. No PP/TG/acceptance measurement taken yet -- that and a k=2/3 live check are the next steps before any preflight gate can be marked satisfied |
| NVMe/FTW cold-bank hypothesis | **converter resumable; artifact proven** | `FTWWriter` checkpoints every tensor and validates/truncates on restart; dense and streamed MoE entries resume without rewriting. Private host-bank mappings make `MADV_DONTNEED` reclaim conversion pages. Native Flash-Next resumed to a valid 73.53 GiB FTW with 48 layers and 10 shards (EXP-017/018/020); serving and throughput remain unmeasured |
| Final gate declared | done | `benchmarks/cert_matrix.py` (D-012, PERFORMANCE.md §6): native `-FT` rows must clear their guard and every same-arch GGUF row reports parity against them |

## Findings that already changed the plan

1. **KV compression, not paging, is the first lever for these hybrids.** At 1M the
   full-attention KV is 20 GiB (35B-A3B) / 24 GiB (Flash-Next) in BF16 but only ~5 / ~6 GiB
   at 4-bit — it fits on the card without a RAM tier. Paging matters for experts/PLE and
   512K+.
2. **The per-sequence recurrent state is a hidden giant**: 966 MiB (35B-A3B) and 1737 MiB
   (Flash-Next) per sequence in fp32, pooled at `linear_state_cache_ratio=2.0`, and it is
   *not* weighed against KV/expert-cache in the current planner. Phase 6 must own it.
3. **The default auto-sizer cannot serve the mission's own anchors**: it gave 8268 KV tokens
   (< 16K) and left 1.08 GiB VRAM unspent. The `--kv-reserve-tokens` floor (8192) is a
   constant, not a policy.
4. **Prefix cache silently fakes a PP measurement** (`#new-token: 64` of 16384) → harness uses
   `--cache-type naive`. Any future A/B must state this.
5. **Prior art is closer than expected** (PROVENANCE.md §5): upstream already has open PRs that
   add exactly the KV-dtype plumbing seam we need (`refs/pr/354`, `refs/pr/408`, `refs/pr/113`),
   the only speculative-decoding machinery (`refs/pr/69`), an NVMe expert tier
   (`refs/pr/337`), a GGUF mixed-quant fix (`refs/pr/494`); and **FreeToken-Kai**
   (`/models/desenvolvimento/reference/freetoken-kai`) is a 191-commit, +31.8 k-line fork that
   has *already merged our exact base* and ships GGUF / KV-quant / host-bank / long-context /
   VRAM-accounting work with its own docs. Port-with-provenance beats reinvention where the
   mechanism is sound.
6. TurboQuant is explicitly requested upstream (**issue #141**) and absent — our Turbo3/Turbo4
   backend is the differentiator, not a duplicate.
7. `qwen4_exp` MTP tensors exist in the local checkpoints (both configs list `mtp.*` in the
   quantization *ignore* set) but upstream's loader **drops `mtp.*`** (issue #421). Phase 9 is
   greenfield on our base; `refs/pr/69` is the reference loop shape.
8. **The GGUF MoE unblock is smaller than the audits first claimed, and the difference is the
   kind of mistake worth remembering.** A7's first pass reported expert rows of 276/308 B and
   concluded the checkpoints carry a re-quantizer's private layouts; §8's offset-derived table
   (1194/1194 tensors) shows llama.cpp's byte counts under llama.cpp's own ids, and the "missing
   row readers" turned out to be missing *CPU dot kernels* only — `--moe-strategy offload` already
   dequantizes every `BLOCK_SHAPE` type. Two numbers that disagreed and nobody settling them
   against the host would have sent Phase 7 to write kernels for a layout that does not exist.
   Every quantity that gates a phase now gets re-measured before it enters a plan (EXP-011).
9. **Compressed KV is a capacity lever here, not a 16K speed lever -- and the two costs must be kept
   apart.** Only our Triton kernels can read a coded tile, and Triton with *uncompressed* KV already
   costs 9.1 % of TG at 16K (158.53 -> 144.11), where the KV is a few percent of the decode step's
   bytes; no quantizer can recover that there, so the phase's own 16K gate was unreachable by
   construction rather than by poor work (D-017). What compression does buy is measured: 256K leaves
   5427 expert slots against bf16's 3183, and 512K/1M stop being "the whole pool budget, zero
   experts". Keep the backend penalty and the codec penalty in separate columns, or both look like the
   codec's fault (PERFORMANCE §10).
10. **A packed layout only wins if the reader stays out of the address business.** The first coded
    readers fetched a byte per element and gathered the centroid book per element: 4x fewer bytes
    arrived while decode ITL doubled (6.79 -> 15.98 ms). The fix is loading each token's packed row
    once and splitting it in registers, which is correctness-green but not yet re-measured (EXP-013).

## Running

```bash
cd /models/desenvolvimento/freetoken-next
.venv/bin/ft serve --model /models/Qwen3.6-35B-A3B-NVFP4-FT --num-tokens 16576 --cache-type naive
.venv/bin/python benchmarks/bench_pp_tg.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
    --tokens 16384 --decode 128 --repeats 3 --label <tag> \
    --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive" --json /tmp/pp_tg.jsonl
.venv/bin/python -m pytest tests -m "not slow" -q
```

Environment gotchas: `UV_CACHE_DIR` must be writable (`/models/desenvolvimento/.uvcache`; the
configured default is root-owned). Never use RAM as storage, under any circumstance: `/tmp` is
a 46 GiB tmpfs that competes with host banks (offload expert banks, PLE tables) and can starve
or OOM-kill a serving process that has nothing to do with what filled it — this bit an actual
`--spec-mtp` validation run (earlyoom killed the worker at ~150/192 experts while 21 GiB of
stale files from unrelated past sessions sat in `/tmp`, 2026-09-17). `/models` has 1.3+ TB free
on real disk; always set `TMPDIR=/models/desenvolvimento/tmp` (checkpoint, serve, pytest
`--basetemp`, any scratch download) and check `free -h` / `du -sh /tmp/*` before a large run.
There is no swap.
