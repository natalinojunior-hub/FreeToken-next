# STATE — freetoken-next

Snapshot date: 2026-09-16. This is the current truth; history goes to
EXPERIMENTS.md / DECISIONS.md, not here.

## Where we are

**Phase 1 (clean fast base) DONE. Phase 2 (native GGUF) in flight. Phase 6 (the VRAM
ledger) has its first brick landed: the account exists, prints, and is honest enough to
show its own holes.**

| Step | Status | Evidence |
|---|---|---|
| Base = current upstream FreeToken `cac247a` (v0.1.3), branch `next` | done | `git fetch upstream` → 0 behind; PROVENANCE.md §1 |
| Durable docs created | done | `docs/freetoken-next/` (7 files + audits A1…A9) |
| Builds cleanly on host | done | editable `.[accel]` install in `.venv` (py 3.12.14, torch 2.11.0+cu130, triton 3.6.0); `_pinned_tensor`, `_cpu_moe`, `_ple_store` all compiled; `ft --version` → 0.1.3 |
| Test suite green | done | `pytest tests -m "not slow"` → **1837 passed, 206 skipped, 1 failed**. The one failure is an environment defect, not a regression: flashinfer's `fp4_quantization_120f` JIT module does not compile with this host's nvcc 13.3 (`quantization.cu:488: error: alignment cannot be set to less than the default alignment` at `-gencode=arch=compute_120f,code=sm_120f`), which `tests/moe/test_nvfp4_backends.py`' two b12x tests either surface or skip depending on import order. Reproduced with this branch's changes stashed (EXP-008) |
| Baseline reproduces anchors | **PASS on both models** | 35B-A3B @16K: PP 4611 / TG 158.8 / VRAM 14.98 GiB / 99.8 % util; Flash-Next @16K: PP 1857.7 / TG 28.685 / VRAM 14.86 GiB / RSS 67.82 GiB / 99.99 % util (PERFORMANCE.md §3, EXPERIMENTS.md EXP-001/001b) |
| Dummy-page brick (`1c81064`) | **done, guards hold** | the pool allocates `num_pages + 1` and both budget formulas now price it, including `kvcache/base.py::solve_num_pages`; 35B @16K re-measured PP 4610.1 / TG 158.75, **output sha1 identical** (EXP-005) |
| VRAM ledger, brick 1 | **done, guards hold at the default ratio** | `engine/vram_ledger.py` + `cache_budget.ceiling_bytes`; one account, printed at startup, `--memory-ratio` a cap and the modelled peak a floor with only the shortfall charged. **Flash-Next now serves at `--memory-ratio 0.9`** (EXP-001b's OOM is gone) and at 1.0 lands on the same geometry as the hand-found 0.86 (EXP-006) |
| VRAM governor, brick 2 | **done, guards hold** | `VramLedger.decide()` owns the expert/KV split and prints a `MemoryPlan` with 128K/256K/512K/1M rows priced against what the split *leaves* for KV, plus the reverse `ContextDemand` trade. 35B PP 4611.3 / TG 158.54, Flash PP 1857.3 / TG 28.68, hashes unchanged (EXP-008) |
| **128K / 256K practical (35B-A3B)** | **PASS** | bought out of the expert cache with `--kv-reserve-tokens`: **128K PP 3188.5 / TG 89.30 / TTFT 41.1 s / ITL p50 10.99 ms**, **256K PP 2353.7 / TG 63.83 / TTFT 111.3 s / ITL p50 15.36 ms**; VRAM 14.4-14.5 GiB, RSS 22.0 GiB. The plan predicted the surviving expert cache to one slot (4695 vs 4694, 3183 vs 3185) (EXP-009) |
| 512K / 1M on these checkpoints | **BLOCKED by the checkpoint, not the engine** | `--max-seq-len-override 524384` is refused: the Qwen3.6-35B-A3B RoPE table is 262 144 positions, so anything past 256K needs a rope-scaled config. Memory agrees: the plan prices 512K BF16 KV at 10.000 GiB, which is the entire pool budget (0 experts) and 1M at 20.000 GiB (EXP-008, EXP-009) |
| **First GGUF matrix row measured** | **PASS** | Qwen3.8-27B IQ3_S dense: PP 2416.6 / TG 25.29 / TTFT 1.69 s / VRAM 14.43 GiB and **RSS 2.17 GiB** (the native offload anchors need 21.9-67.8 GiB); its own account says 128K KV is 8.00 GiB short of what 13.2 GiB of resident weights leave behind (PERFORMANCE.md §9) |
| CPU MoE ABI gate | **done** | `_cpu_moe.max_weight_format_id()` + `compiled_extension_supports_format()`: a `.so` predating the GGUF K-quants now refuses `q4_0/q4_k/q6_k` with the rebuild instruction instead of faulting inside a worker thread that already holds the bank table (the activation probe's twin) |
| Benchmark a bare `.gguf` | **done** | `benchmarks/bench_pp_tg.py` now tokenizes through `utils/hf.load_tokenizer` (the engine's own GGUF vocab path) instead of `AutoTokenizer`, which can only read an HF directory |
| Source audits (9, parallel) | **done** | `docs/freetoken-next/audits/A1…A9.md`; A7 = GGUF MoE geometry + refused-today matrix, A8 = turbo codec byte layout, A9 = unaccounted VRAM consumers |
| Phase 2 GGUF loader | **first model serves correctly** (port itself still uncommitted) | EXP-004: the IQ3_S 27B GGUF generated coherent, factually correct text and the NextN/MTP drop warned as designed; owed before commit: `kernel/aot_models.py` arch entries, the `models/*/__init__.py` export union, `kernel/gguf.py` `libcudart` load order, and refusing K-quant CPU formats at registration |
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
configured default is root-owned). `/tmp` is a 46 GiB tmpfs that competes with host banks —
set `TMPDIR=/models/desenvolvimento/tmp` for big-model runs. There is no swap.
