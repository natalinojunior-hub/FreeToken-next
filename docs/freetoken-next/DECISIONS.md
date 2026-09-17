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
**Date:** 2026-09-16 · **Status:** proposed (confirm against audit A1)
Turbo3/Turbo4/TCQ/VBR are to be registered as quant scheme/method backends on the
`QuantConfig → QuantScheme → QuantMethod` layers introduced by `#418`, alongside BF16/FP8/
NVFP4, rather than as ad-hoc branches in the attention or cache code. A/B between KV
formats is a requirement, so selection must be reachable from config at runtime.

## D-006 — Preserve 0.1.3 prefill behaviour while adding decode features
**Date:** 2026-09-16 · **Status:** accepted
Any patch that cannot show PP within noise of the anchors is reverted, not optimised
"later". The scheduler, expert streaming, and quant kernels are only replaced with causal
wall-clock evidence that the existing mechanism is the bottleneck.
