# STATE — freetoken-next

Snapshot date: 2026-09-16. This is the current truth; history goes to
EXPERIMENTS.md / DECISIONS.md, not here.

## Where we are

**Phase 1 (clean fast base) is DONE.** Phase 2 (native GGUF) is next and the source
audits that decide its design are in flight.

| Step | Status | Evidence |
|---|---|---|
| Base = current upstream FreeToken `cac247a` (v0.1.3), branch `next` | done | `git fetch upstream` → 0 behind; PROVENANCE.md §1 |
| Durable docs created | done | `docs/freetoken-next/` (7 files) |
| Builds cleanly on host | done | editable `.[accel]` install in `.venv` (py 3.12.14, torch 2.11.0+cu130, triton 3.6.0); `_pinned_tensor`, `_cpu_moe`, `_ple_store` all compiled; `ft --version` → 0.1.3 |
| Test suite green | done | `pytest tests -m "not slow"` → **1746 passed, 206 skipped, 1 failed**, and that one (`kernels/test_mrope.py`) is a flashinfer JIT first-build race — re-run alone: **6 passed** |
| Baseline reproduces anchors | **PASS on both models** | 35B-A3B @16K: PP 4611 / TG 158.8 / VRAM 14.98 GiB / 99.8 % util; Flash-Next @16K: PP 1857.7 / TG 28.685 / VRAM 14.86 GiB / RSS 67.82 GiB / 99.99 % util (PERFORMANCE.md §3, EXPERIMENTS.md EXP-001/001b) |
| Source audits (6, parallel) | **done** | `docs/freetoken-next/audits/A1…A6.md`; conclusions merged into ARCHITECTURE.md §2–§6 |
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
