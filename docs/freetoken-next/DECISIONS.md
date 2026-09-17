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

## D-009 — Phase 2 sources: port upstream PR #131, do not merge it
**Date:** 2026-09-16 · **Status:** accepted
A5 (FreeToken-Kai audit) shows Kai **abandoned** GGUF (`docs/gguf.md` is a negative result:
UD checkpoints vary the expert ggml type per layer, and the offload slot pool is one
allocation with one row stride) and points at upstream **PR #131 `feat/generic-gguf`**
(`refs/pr/131`, 37 files, +5920/−220, 36 commits, head `bb432e8` 2026-08-25) as the real
source: 21 ggml types, GGUF expert banks, K-quant CPU kernels, loaders for **qwen35moe**,
qwen3moe and deepseek_v4, **split-shard loading**, and tests (`tests/models/test_gguf_*.py`,
`test_qwen35moe_gguf.py`, `tests/moe/test_cpu_moe_kquant.py`).
Attempted `git merge refs/pr/131` → **7 conflicting files** (`engine/engine.py`,
`layers/moe.py`, `moe/expert_banks.py`, `moe/cpu_executor.py`, two `models/*/__init__.py`,
`docs/models.md`), because #131 predates `#418`/`#427`: HEAD's `expert_banks.py` builds banks
from `QuantMethod.layout()` and `ExpertBanks(kind=…, kernel=…, layout=…)`, and
`_resolve_auto_moe_cache_size` gained a `method` argument. So: **adopt the PR's leaf files and
re-author those seven seams against HEAD** — a port, keeping the PR's comments and its
constraints documented (uniform expert type per bank, TP=1, `nextn`/MTP dropped). Provenance
row goes to PROVENANCE.md §4 on landing.
**Why:** a merge would silently revert the QuantConfig refactor's bank construction; a port
keeps both. Writing it from scratch wastes 5 900 reviewed lines.

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

## D-010 — KV quantization has two templates, and a measured warning
**Date:** 2026-09-16 · **Status:** accepted
A5 found FreeToken-Kai already ships `--kv-cache-dtype {auto,q8_0,q4_0}` on **our exact base**
(`kvcache/kv_quant.py`, `kernel/triton/kv_quant.py:_quantize_store_kernel`, pool wiring
`mha_pool.py:25`/`hybrid_swa_pool.py:39`/`qsa_pool`, gate `kvcache/__init__.py:137-170`, reads
`attention/triton.py:157-220` and `qsa/attend.py:356-500`; SHAs `cd385f8`, `ef9be8c`, `64de6c0`,
`6920bfa`, `7321ad1`), with bytes/token and per-family refusal tables measured. Its measured
cost is the warning this decision is about: **−33 % TG on a dense model at 30K, flat −1.4 % on
Flash-Next** with q4_0 — i.e. naive quantize-then-dequant-in-attention *does* destroy decode,
exactly the failure ARCHITECTURE.md §4 is written to avoid.
Position: take from Kai the **spec object, page geometry, the startup refusal gate and the
`--moe-cache-auto` coupling** (freed KV bytes becoming expert slots), and from #408 the
**capability seam** (`supports_{x}_kv`, `dtype` vs `store_dtype`, shared `_alloc()`); write the
Turbo3/Turbo4 encode/decode and the fused read path ourselves against the LTO semantics, and
never ship a KV format whose 16K/32K A/B has not shown the TG cost.

## D-011 — Phase 9 has a working reference on our base
**Date:** 2026-09-16 · **Status:** accepted
Kai implements `--spec-mtp K` for both `qwen3_5_moe` **and** `qwen4_exp` (`engine/spec.py` with
`accept_drafts`, `pages_to_free`, `rebuild_conv_state`, `ngram_context_after`;
`engine/spec_graph.py` 335 lines with a `FT_SPEC_GRAPH_MIN_FREE_MB=256` graph guard;
`models/mtp_quant.py`; `attention/base.py:init_spec_capture`/`stage_spec`;
`qwen4_exp/ple.py:407`; SHAs `a8c262b`, `c6d130f`, `d93bd3f`, `153b4a5`, `d0c0cc5`, `2a461ae`,
`d7c7fa4`). It is a correctness oracle for the acceptance/rollback arithmetic and a template
for graph handling — but A5 also flags its costs: it adds a bank layer *after* placement is
solved, keeps K+1 GDN states resident (~250 MB at K=3), spends a KV layer, and multiplies
expert traffic. So Phase 9 reads it and #69, implements against the §5 ledger rather than
copying the graph-side bookkeeping.

## D-012 — Final validation is a matrix over native `-FT`/NVFP4 and same-arch GGUF
**Date:** 2026-09-17 · **Status:** accepted (user requirement)
After every phase a build is shippable only if `benchmarks/cert_matrix.py` passes:
1. **no regression** on the native FreeToken-compatible checkpoints — every native row must
   clear its guard at 16K (35B-A3B ≥ 4600 PP / ≥ 158 TG; Flash-Next ≥ 1850 PP / ≥ 28.5 TG at
   `--memory-ratio 0.86`);
2. **native-vs-GGUF parity** — each GGUF row is reported as a percentage of the
   same-architecture native row at equal context, so a GGUF path cannot hide behind "no
   baseline";
3. a row whose checkpoint the current code cannot serve reports **BLOCKED naming the
   blocker**, never silently absent.
**Why:** "loads GGUF" is not the gate; the GGUF path has to be measured against the NVFP4
path on the same architecture, and the only way to keep both true across thirteen phases is
one declared matrix that fails the build.

## D-013 — Parity is throughput/capacity, not output identity
**Date:** 2026-09-17 · **Status:** accepted
The local GGUF files are not the same weights as their native counterparts (Ornith and
Tiel-Coder are fine-tunes on the Qwen3.6-35B-A3B geometry; the Unsloth/AD Flash builds are
different quant recipes; dense `qwen35` has no native counterpart on this host at all). The
matrix therefore compares PP, TG, TTFT, ITL, VRAM and RSS at equal context plus a qualitative
sanity generation — **not** token-for-token equality. Equality needs a same-weights pair
(convert a native checkpoint to GGUF and back, or fetch official Qwen3.8-27B native), which
is recorded as a matrix gap, not silently dropped.


