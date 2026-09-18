# STATE — freetoken-next

Snapshot date: 2026-09-18. This is the current truth; history goes to
EXPERIMENTS.md / DECISIONS.md, not here.

## Current checkpoint (2026-09-18, session 3 — ROADMAP Line 8 Closed)

ROADMAP Line 8 (Turbo4 + MTP: correctness, determinism, throughput and certification) is **fully closed**.
ROADMAP Line 12 (Phase 10: Fused verify MTP+TurboKV) is **formally specified (D-022)**.

Summary of resolutions:
1. **Bug A (GDN Decode vs Prefill Inequivalence):** Characterized offline (EXP-036). Quantified ~1.95e-3 max abs delta flipping greedy argmax ~1.5% of steps between fused decode and chunked prefill kernels.
2. **Bug B (Multi-Request State Carrier Leak):** Root cause isolated (EXP-038) to QSA `pending_ring` and `_cmp_k_buffer[scratch]` surviving on recycled `table_idx`. Fixed via `QSAKVCache.free_req(table_idx)` in `_free_req_resources` and QSA state snapshot/restore on speculative reject (EXP-039). Verified: initial `pending_ring` hash bit-identical (`6.707354e+03`) across all requests; greedy output sha1 bit-identical across sequential runs (`614aa7bcdf59`).
3. **Equivalence vs Baseline:** Turbo4 + MTP output sha1 (`614aa7bcdf59`) matches baseline `triton + bf16` at 16K byte-for-byte.
4. **TG Bottleneck (0.79 tok/s -> 25 tok/s):** Localized 96.3% of prefill forward cost to MoE materializing all 512 experts across 48 layers (EXP-037). Resolved by adding `_use_decode_path` for speculative micro-batches ($T \le 8, \text{len}(\text{reqs}) == 1$) in `OffloadMoELayer` (EXP-040). `verify_forward` dropped from 1.203s to 0.0735s (16.3x faster); `gdn_replay` dropped from 1.204s to 0.0407s (29.6x faster); live TG jumped from 0.79 tok/s to 19.28 tok/s mean (24.8–25.0 tok/s steady state, ~25x speedup).
5. **Multi-Token Speculation:** $k=2$ and $k=3$ validated live on `--decode 32` with successful multi-token acceptances (`accepted=2/2`, `accepted=3/3`) and zero memory leaks.
6. **Long-Context Turbo4 128K Certified:** Executed 131,072-token prefill + decode on RTX 5080 (EXP-041): PP 1376.1 tok/s, TG 4.86 tok/s, 14.84 GiB VRAM (under 15.51 GiB card ceiling), 2053 pages (0.871 GiB KV). 256K ledger physics confirmed (KV 1.738 GiB leaves 517 expert slots).
7. **Watchdog & Memory Hardening:** Added active server process watchdog to `bench_pp_tg.py` to abort immediately on crash/exit; chunked `rotate` and `inv_rotate` in `turbo_kv.py` into 4096-row blocks, dropping transient peak allocation from >500 MiB to ~2 MiB; replaced Triton JIT in `_materialize_layer_gpu` with in-place PyTorch indexing.

The previous host-RAM/`earlyoom` blocker is no longer active: on 2026-09-18 the
host reports 91 GiB total, 6.5 GiB used, 84 GiB available, and only 137 MiB in
`/tmp`; `/tmp` is backed by the NVMe filesystem rather than tmpfs. RAM hygiene
remains a preflight check, but it is not currently preventing the benchmark.

The working tree contains an additional, uncommitted TurboKV/QSA integration:
`qsa_pool.py` delegates the full-attention pool to `TurboMHAKVCache`,
`qsa_sparse.py` detects compressed slabs and materializes a bounded page workspace,
and `kernel/triton/qsa/decompress.py` provides a separate Triton decompression
kernel before the existing dense QSA attention kernel. This is the intended escape
from the fused deserialization/attention JIT that caused the ptxas host-RAM
failure, but it still needs correctness, compilation, and performance validation.

The next phase is therefore not yet QSA pruning. First validate this split-kernel
path at 16K against `triton + bf16`; then measure the long-context memory benefit.
Only after that gate passes should block pruning, score reuse, or MoE/PLE prefetch
be implemented.

Validation on 2026-09-18: the targeted TurboKV/QSA set is green (79 passed, one
warning), the QSA backend suite passes on the real GPU, the split Turbo4 CUDA
test passes, and the isolated decompressor passes with checkpoint geometry. The
MTP path exposed and received fixes for MRoPE positions and the current
`ForwardOutput.copy_done_event` name; focused scheduler/QSA tests pass (14
passed). The latest cache-tail fix is committed below but still needs a fresh
focused test and live benchmark.

## Handoff (2026-09-18, session 2 ending)

**Read this section first.** Item E (native MTP) continues. EXP-034 fixed the real page-leak
(three bugs in `spec.py`, not the `cache.py` patch session 1 committed -- that patch was reverted,
it caused its own regression). MTP=1+Turbo4 now runs to completion live. What is NOT done, in
order of priority (goal priorities 4-5 next):

0. **MTP throughput redesign (goal priority 4, "CUDA Graph/overlap").** Live TG is 0.79 tok/s
   mean vs 24.6 without MTP (31-50x slower, ITL 1.2-2.4s/token) at 99.7% GPU util -- real
   compute, not idle wait, and **not simply `--cuda-graph-max-bs 0`**: the MTP-free arm used the
   SAME flag and hit 24.6 tok/s, so graph-disabled alone isn't the discriminator (ruled out by
   review, see PERFORMANCE.md). Sharper lead: the `--decode 4` determinism probe (below) showed
   the SAME code/flags giving ITL 1294ms (requests 1-2) vs 47ms (requests 3-6) -- a 27x spread
   with no code change, so the constant-per-step `phase="prefill"` overhead and host syncs
   (`tok_prev.item()`, `out.copy_done_event.synchronize()`) are NOT the dominant cost; something
   shape/compile-cache-dependent is. The 47ms regime does not persist at `--decode 64` (reverts
   to ~1239ms) -- **the actual question is why the fast regime doesn't hold at longer decodes**,
   not whether CUDA graph capture would help (never measured this session). This is the same
   pending work as ROADMAP phase 10/12 ("MTP + TurboKV fused verify: no materialize-path
   penalty") -- scope it as a real redesign, multi-session, after the discriminator above is
   understood, not before.
1. **Non-determinism across sequential requests in the same server session (found this session,
   NOT fixed, root cause still unknown).** `req.uid` (unique per real request) and
   `model.model._last_residual` (overwritten by every forward) were checked and ruled out.
   Localized with `--decode 4` (cheap to iterate): 6 sequential requests gave TWO stable phases,
   not continuous drift -- requests 1-2 identical to each other (`sha1 614aa7bcdf59`, ~0.6
   tok/s), requests 3-6 identical to each other but different from 1-2 (`sha1 229272fa84fd`,
   ~2.33 tok/s, a clean one-time transition, not a leak that keeps compounding). Looked like a
   JIT/autotune warmup settling after ~2 real calls -- **but this does NOT generalize**: rerunning
   at `--decode 64 --warmups 2 --repeats 2` (4 total requests, same positions where `--decode 4`
   had already stabilized) still gave two DIFFERENT sha1 for repeat1 vs repeat2. So whatever
   causes this is sensitive to decode length / total step count in a way not yet characterized,
   not just "how many prior requests." **More black-box benchmarking will not resolve this.**
   Concrete next step: `spec.py:195`'s existing `logger.info(f"spec: k=... m=... accepted=...
   drafts=... sampled=...")` fires every step -- add the raw logits' top-2 values (not just the
   argmax winner) to that line, rerun `--decode 4 --repeats 6`, and diff phase-1 (requests 1-2)
   against phase-2 (requests 3-6) logs. A flipped argmax between near-tied logits means
   floating-point/kernel-config (numerics); the DRAFT tokens differing before the logits do
   means a real state bug. ~20 minutes, discriminates the two live hypotheses.
   ```bash
   FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 TMPDIR=/models/desenvolvimento/tmp \
     .venv/bin/python benchmarks/bench_pp_tg.py \
     --model /models/Qwen3.8-Flash-Next-NVFP4-Radix --tokens 16384 --decode 4 \
     --repeats 6 --warmups 0 --label mtp1-turbo4-repro \
     --serve-arg "--cache-type naive" --serve-arg "--max-running-requests 1" \
     --serve-arg "--num-tokens 16576" --serve-arg "--spec-mtp 1" \
     --serve-arg "--kv-format turbo4" --serve-arg "--memory-ratio 0.86" \
     --serve-arg "--cuda-graph-max-bs 0" \
     --json /models/desenvolvimento/tmp/mtp_validate/repro.jsonl
   ```
2. **k=2/k=3 live validation.** The draft-chain per-step position bug (draft step i>=1 couldn't
   see its own prior KV) was fixed by code inspection during the k=1 debugging, never exercised
   live. Run the same greedy content-equivalence check (short + the "ocean poem" prompt that
   caught the k=1 bug) at `--spec-mtp 2` and `--spec-mtp 3` before trusting them.
3. **Acceptance-rate logging is silent.** `scheduler/spec.py`'s `logger.info(f"spec: k=...")`
   line never appeared in the server log despite the spec path demonstrably running (content
   fix confirmed). Minor, not a correctness blocker — worth a quick look at scheduler-subprocess
   log routing before relying on acceptance-rate numbers for the goal's report.
4. **The prefill-window MTP warm-up pass is still missing** (EXP-025's throughput-critical gap):
   the draft head's own KV never gets populated over the ORIGINAL PROMPT, only over positions
   generated after MTP engages. Correctness is unaffected (target verify is still exact) but
   acceptance rate is likely poor for the first several tokens after any prefill. Not
   implemented this session; needed before trusting a measured acceptance rate as representative.
5. Only after 1-4: the goal's preflight gates (target-equivalence ✅ done, acceptance/rollback
   timing measured, a credible 256K projection) can be evaluated, and only then does a 256K
   certification attempt become in-scope — not before, per the goal's own hard test budget.

**Non-MTP loose end**: `python/freetoken/moe/expert_banks.py:352`'s `_host_ram_fits_parallel`
guard was flagged (not fixed) by the Opus RAM investigation (EXP-031) — it compares against
total checkpoint size (125.9 GiB) instead of the actual bank size (63.46 GiB), so the parallel
expert-loading path is permanently unreachable on this host (silent perf regression, always
serial-loads), AND its transient-memory budget separately under-counts the prefetch window
(1 shard budgeted vs the documented 3). Both need fixing together if parallel loading is ever
revisited — fixing only one half first could cause a NEW ~30 GiB transient OOM.

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
| Phase 3 Turbo KV | **split path runs; A/B gate in progress** | Separate decompression + bounded dense workspace + existing QSA attention is live on Flash-Next: Turbo4 16K eager/no-overlap = PP 1713.7 / TG 24.72. MTP=1 reaches speculation but is blocked from a valid row by a one-page cache leak; the tail-free fix is not re-benchmarked yet. |
| Decision B MoE routing instrumentation | **partial / blocked on host load** | `--moe-collect-stats` is parser-tested and logs graph-safe aggregate/per-layer counters at worker shutdown (`56774dc`); detached reproduction reached only bank 176/192 and was killed with `exitcode=-9`, consistent with the 63.46 GiB bank footprint exceeding the host's 62 GiB available RAM (EXP-015) |
| Native MTP config/reader seam | **content-correct live for k=1; throughput STILL unmeasured** | Full pipeline implemented and validated against the real checkpoint (`--spec-mtp 1`, 16K, single request, naive cache): weight loading, MoE-bank aliasing, scheduler spec-loop (EXP-021/022/024/025/026). Six real bugs found and fixed via live serving (EXP-027/028: weight-loading `KeyError`, missing package export, MoE offload-cache layer-count assert + `moe_layer_id` aliasing, a per-token `cached_len`/`device_len` accounting bug, and the state-corrupting reject-path rewind bug found by a third Opus 5 pass). Retest confirms MTP output is content byte-identical to `--spec-mtp 0` on both a short and a previously-diverging longer prompt. k=2/3 fixed by inspection, not yet live-tested. A real `--moe-cache-auto` correctness bug was also found and fixed along the way (EXP-030: an explicit `--num-tokens` was silently downgraded instead of refusing, once the MTP layer's extra VRAM made the auto split tight). **No PP/TG/acceptance measurement has been taken yet** — every attempt so far died to a host-RAM `earlyoom` race during expert-bank loading (EXP-029/031), now understood and fixed (see below); the next session's first job is to actually run the benchmark |
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
or OOM-kill a serving process that has nothing to do with what filled it. Confirmed by a live
Opus 5 investigation (EXP-031): `earlyoom`'s reported "total" is NOT `MemTotal` — it's a
live-recomputed `user mem total` that nets out shmem/tmpfs (`/usr/bin/earlyoom --dryrun -r 1`
proves this: it prints `mem total` and `user mem total` as two different, correct numbers), so
**every 1 GiB parked in `/tmp` costs ~0.9 GiB of the 10 % SIGTERM margin**, not 1 GiB of a fixed
93 GiB pool. There is no swap (`SwapTotal: 0`), so earlyoom's memory trigger is permanently
unguarded (the swap half of its AND-condition is trivially satisfied). The Flash-Next expert-bank
loader itself is already tight (private anonymous mmap + `drop_page_cache` before/after every
shard, EXP-020/EXP-031) — the ~68-70 GiB peak RSS is real and irreducible without a different
residency strategy; it is not a leak. `--spec-mtp` adds no measurable host RAM over the
`--spec-mtp 0` baseline (EXP-031's RSS trace: 68.37 GiB peak either way).
**Fix applied this session**: `/tmp` cleared of ~21 GiB then ~3.3 GiB more of stale artifacts
from unrelated past tool sessions (moved to `/models/backup/pytest-of-natal/` and
`.../tmp-leftovers-20260917/`, which are natal-writable; `/models/backup` itself is root:root and
needs `sudo mkdir + chown` per subfolder). `tmp.mount` is `systemctl mask`ed (confirmed
`is-enabled` → `masked`, symlinked to `/dev/null`), so **after the next reboot `/tmp` stops being
tmpfs entirely** and this whole class of failure goes away permanently; until then, re-check
`du -sh /tmp/*` before any large `ft serve`/checkpoint run — stale build trees from OTHER tools
on this shared machine (llama.cpp builds, npm caches, old benchmark logs) reaccumulate there.
`/models` has 1.3+ TB free on real disk; always set `TMPDIR=/models/desenvolvimento/tmp`
(checkpoint, serve, pytest `--basetemp`, any scratch download).
