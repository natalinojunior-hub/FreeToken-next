# DECISIONS — freetoken-next

Append-only. One decision per entry; supersede rather than rewrite.

## D-001 — Base is the current upstream tip, not an old fork
**Date:** 2026-09-16 · **Status:** accepted
`next` branches from `upstream/main` = `cac247a` = `v0.1.3`, which after `git fetch` is
0 commits behind. The pre-existing `/models/servers/freetoken` install (`0.1.2+gaf71ba432`,
23 commits older) is kept untouched as the *anchor* build, not as the base.
**Why:** the mission requires current-upstream lineage, and the 23 skipped commits contain
exactly the abstractions later phases build on (`#418` QuantConfig/Scheme/Method layers,
`#427/#426/#438` checkpoint→reader handoff, `#428` qwen4_exp block-fp8, `#367` paged-KV
reservation granularity, `#420` Flash-Next PLE table).

## D-002 — Dedicated editable env; reference env stays read-only
**Date:** 2026-09-16 · **Status:** accepted
Build into `/models/desenvolvimento/freetoken-next/.venv` (`uv venv --python 3.12`,
`uv pip install -e ".[accel]"`), with `UV_CACHE_DIR=/models/desenvolvimento/.uvcache`
because the configured default `/models/outros/cache/uv-cache` is root-owned.
**Why:** A/B against the installed 0.1.2 anchor must remain possible at any time; sharing
its site-packages would destroy that. Python 3.12 because the system 3.14 has no torch build.

## D-003 — Local research fork: commit locally, never push
**Date:** 2026-09-16 · **Status:** accepted
Upstream `AGENTS.md` bars autonomous agents from pushing or opening PRs/issues. No remote
push, no `gh` write operations; `next` lives only on this machine.

## D-004 — GGUF is an immutable source of truth
**Date:** 2026-09-16 · **Status:** accepted
No whole-file dequantization at load, no in-place rewrite of a GGUF. If a repacked layout
is faster on the hot path, it becomes a *derived cache/overlay* keyed by file identity,
never a mutation of the user's file.
**Why:** the user's GGUF corpus is large and shared with other engines; a loader that
rewrites or explodes it is unacceptable regardless of speed.

## D-005 — New KV formats enter through the existing quant abstraction
**Date:** 2026-09-16 · **Status:** revised after audit A1/A6
A1 found **no KV quantization seam at all** on `cac247a` (`KV_CACHE_DTYPE_BYTES = 2`,
`kernel/aot_models.py:36`; no `--kv-cache-dtype`). The `QuantConfig → QuantScheme →
QuantMethod` layers are weights-only. So D-005 becomes: Turbo3/Turbo4/TCQ/VBR are to be built
on **upstream PR #408's seams**, ported forward onto `cac247a` — `--kv-cache-dtype` →
`EngineConfig.kv_quant` → alias table → the generic capability gate
`getattr(BackendInfo, "supports_{kv_quant}_kv")` that makes auto *skip* and startup *refuse*
incapable backends; `kv_storage_bytes_per_elem` + `kv_scale_bytes_per_token` as the only cost
model, asserted against `unit_bytes()`; `dtype` (compute) vs `store_dtype` (buffer); one
`_alloc()` shared by first allocation and `rebuild()`. A6's rebase check: #408 is a strict
superset of #354 (`merge-tree` rc=1 on 5 files both) and #113 collides semantically on the
same flag while using native fp8 pointers, which #354 explicitly rejects. No new one-off path.
**Why:** A/B between KV formats is a requirement, and porting the seam upstream already
reviewed is cheaper than inventing one that later has to be reconciled with #408.

## D-006 — Preserve 0.1.3 prefill behaviour while adding decode features
**Date:** 2026-09-16 · **Status:** accepted
Any patch that cannot show PP within noise of the anchors is reverted, not optimised
"later". The scheduler, expert streaming, and quant kernels are only replaced with causal
wall-clock evidence that the existing mechanism is the bottleneck.

## D-007 — Flash-Next regression config is `--memory-ratio 0.86`
**Date:** 2026-09-16 · **Status:** accepted (until Phase 6 replaces it)
At the default 0.9 the engine cannot warm up on this card: it CUDA-OOMs in unbudgeted
transients (Triton autotune's 256 MiB benchmark cache; graph capture). All Flash-Next A/Bs
use 0.86 and state it, so the ratio is not an accidental variable in a comparison. Phase 6's
ledger is what removes the need for this.

## D-008 — GGUF phase order: widen, then adapt, then shard
**Date:** 2026-09-16 · **Status:** accepted
A4 shows the loader, the mmap'd packed-row reader and the vendored ggml MMQ/MMVQ/MoE kernels
already exist and already keep weights compressed until the consuming kernel; the narrow parts
are two Python tables, the per-family name maps, and shard joining. So: I1 = `qwen3_5_moe`
adapter + type tables (target: single-file Q4_K_M `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf`),
I2 = ggml expert-bank schemas for Q4_K/Q6_K/Q8_0, I3 = shard joining, I4 = `qwen4exp`
(incl. `per_layer_token_embd` → PLE), I5 = prefill kernels for the IQ types. GGUF stays out of
the `QuantKind` registry until it is not a FIXME (`engine/engine.py:772`).
**Why:** every increment must be servable and measurable; a sharded or IQ-prefill first step
would spend weeks before any number exists.

