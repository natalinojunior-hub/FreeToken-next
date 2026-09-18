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

## D-014 — The VRAM ledger owns the reserve; `--memory-ratio` is a cap, not the policy
**Date:** 2026-09-17 · **Status:** accepted (`engine/vram_ledger.py`, `cache_budget.ceiling_bytes`)
One object opens the byte account after the weights load and owns every number the planner
consumes. Each consumer is a named line item with a `Kind`; `reserve_bytes` is what must stay
empty (Triton's 256 MiB autotune arena, the CUDA-graph capture peak, the gated-delta-net
per-layer prefill workspace over one scheduler chunk, the live activation stream, the named
fragmentation reserve) and `engine_overhead_bytes()` is what the engine holds but no pool or
expert cost model prices (page table, graph pool, backend workspaces, PLE). The ceiling is
`ratio x baseline` minus only the **shortfall** between the modelled reserve and the hole the
ratio already leaves, so: (a) every shipped default ratio reproduces the pre-ledger budget
byte-for-byte, which is what keeps the two performance guards intact; (b) raising the ratio
toward 1.0 stops being a gamble, because the plan funds the peak it used to hope for; (c) the
KV solve, `--moe-cache-auto` and `validate_rebuild` all take the same two numbers from the same
place, so they cannot disagree about what "fits" means. `unit_bytes`/measured lines are
compared against the account at the end of init and a gap is warned, never asserted.
**Why:** before this, the only thing between a filled pool and a CUDA OOM was an unmodelled
`(1 - memory_ratio)` hole whose size nobody could state -- EXP-001b had to bisect
`--memory-ratio` by hand (0.9 OOM, 0.86 works) and the same hidden margin is what caps the
context the mission needs. EXP-006 shows the account catching two real holes the formula could
not see (an aliased-view double count, and 2.2 GiB of expert-cache side tables the per-slot
price omits).
**Consequence for D-007:** `--memory-ratio 0.86` is no longer the only working Flash-Next
config; the default 0.9 serves with the reserve modelled (EXP-006). 0.86 stays in the guard
table as the recorded anchor condition so the comparison remains matched.

## D-015 — MoE-priority stays the default; context is bought, not assumed
**Date:** 2026-09-17 · **Status:** accepted (`VramLedger.decide`, `MemoryPlan.kv_budget_bytes`)
The auto plan fills the expert cache first and lets KV keep the floor (`plan_cache_budget`'s
`--kv-reserve-tokens`), because on this machine resident experts are what TG is made of. The
ledger keeps that policy but stops letting it be silent: every engine now prints, per context
target, the bytes KV would need *against what the split actually left for KV* -- and on a 16 GiB
card with an 8 GiB dense slice that number is brutal (the 35B-A3B auto plan leaves 0.157 GiB for
KV, so 128K needs +2.343 GiB of somebody else's memory). Two consequences, both deliberate:
1. Long context is bought with an explicit trade the operator can see and choose: raise
   `--kv-reserve-tokens` (expert cache down, context up), compress the KV format (§4's
   turbo4: 4.125 bpv, ~4x smaller), or tier to RAM (§10's PCIe-bound economics). The plan
   prints the shortfall for each so the choice is arithmetic, not experiment.
2. No "auto context" flag is exposed until the plan is the thing that made the choice; the
   feasibility rows are the prerequisite for §11's `--context auto` and are already in the log.
**Why:** EXP-008 measured that a pool-budget-priced feasibility table reads as "128K fits" on a
model whose expert cache has already eaten the whole budget. Pricing the same rows against what
the split leaves is the difference between a plan and a fantasy, and it is the number the
compressed-KV phases have to beat.

## D-016 — No per-token RAM paging: decode re-reads the whole context, and PCIe says no
**Date:** 2026-09-17 · **Status:** accepted (economic gate computed before any paging work)
The Phase-11 gate asked for arithmetic before implementation, so here it is, from the KV bytes
the ledger reports on this host and the measured 57.76 GB/s host link (PERFORMANCE.md §2).
Attention is a full scan, so a RAM tier that holds part of the live context is re-read **once
per generated token**, not once per request:

| model | BF16 KV/token | 128K one step | 1M one step | 1M ceiling | 1M at turbo4 (4.125 bpv) |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 20.24 KiB | 47 ms | 376 ms | **2.66 tok/s** | 5.22 GiB, 10.3 tok/s |
| Qwen3.8-Flash-Next (paged part) | 24.86 KiB | 58 ms | 462 ms | **2.16 tok/s** | 6.41 GiB, 8.4 tok/s |

At 30 tok/s the whole link buys 1.8 GiB of fetchable KV per step -- under 93 000 tokens of
either model's BF16 context -- so every tiered configuration that keeps 128K or more of live
context in RAM is physically guaranteed to destroy TG, which is the outcome the gate exists to
prevent. Decision: **do not build per-token hot/warm page migration.** RAM keeps the roles it
already earns -- the pinned expert banks, the PLE table, the host radix metadata -- and gains
exactly one new one: streaming a *cold* prefix in once, at prefill speed, where the PCIe cost is
paid per prompt instead of per token. Long context is bought the other two ways the ledger
prices: compressed KV (4x turns the 35B-A3B's 1M from 20.24 GiB into 5.22 GiB, which fits VRAM
outright and removes the fetch entirely) and, inside VRAM, the expert cache (D-015).
**Why:** the arithmetic is not close -- 2.2-2.7 tok/s against a 28.7-158.5 tok/s baseline -- and
a paging subsystem built on that premise would have cost the phases that actually deliver
context (3-5, 12) their remaining budget.

## D-017 — Compressed KV is a long-context lever; the 16K native guard stays on the flashinfer path

**Date:** 2026-09-17 · **Status:** binding · **Evidence:** PERFORMANCE.md §10, EXP-012, EXP-013

**Context.** The campaign's TurboKV gate was written as "TG within a few percent of the 16K anchor,
with >=3.5x fewer KV bytes/token". Landing the codec made the first half measurable, and it is not
reachable: only our Triton kernels can read a coded tile (flashinfer takes a fixed dtype and a fixed
slab layout), and Triton with *uncompressed* KV already costs 9.1 % of TG at 16K (158.53 -> 144.11).
At that context the KV is a few percent of the decode step's bytes, so no quantizer -- not turbo4,
not a hypothetical lossless one -- can recover a 9.1 % backend penalty there. The measurement, not a
preference, is what splits the gate.

**Decision.** `--kv-format turbo3|turbo4` is opt-in and is certified on two axes: against the
**same-backend** bf16 arm (`triton + bf16`), and against the bytes the account gets back at long
context. It is never certified by the flashinfer 16K guard, and the flashinfer guard itself keeps its
exact form: the compressed work must not move `PP >= 4600 / TG >= 158 / sha1 2a6dca88ffdc`, which is
why the kernel change is a compile-time-dead `COMPRESSED` constexpr branch rather than a second
attention implementation, and why "the bf16 call is bit-identical to itself" is a pinned test.

Consequences, binding:
1. A compressed arm reports its own baseline and its own output hash. A different backend is a
   different greedy continuation (`b7c70b36d276` for triton vs `2a6dca88ffdc` for fi), so
   "output unchanged" is only meaningful within one backend, and every row must name it.
2. The phase's success metric is what the freed bytes buy: 256K with **+70 % expert slots**
   (3183 -> 5427) and `128K/256K: fits` where bf16 could not fund 256K, then TG measured *there*.
3. The no-hidden-materialization rule survives unchanged: fused tile reads with bounded scratch, never
   a context-sized dequant buffer, and whatever scratch exists is a ledger line.
4. If the readers cannot reach parity with `triton + bf16`, the format ships opt-in for long context
   or is reverted. It does not become default to make a table look good.

**Why:** the alternative was to spend the phase chasing a deficit that belongs to the backend, and to
report a compressed-KV number against a baseline it cannot physically touch. Naming the two costs
separately (9.1 % backend, then whatever the codec costs) is what makes each one fixable -- and the
first one is only fixable by teaching flashinfer a new dtype or by optimizing our own kernel, both of
which are their own decisions, not this one.

## D-018 — Do not charge an unowned dequant scratch buffer

**Date:** 2026-09-17 · **Status:** accepted · **Evidence:** `tests/engine/test_vram_ledger.py` (22 passed)

`modelled_reserves()` exposed a `dequant_scratch` argument and a reserve line, but the current
engine had no caller and no allocation matching it. Keeping that line made the 256K expert-slot
plan depend on an invented consumer. The dead parameter and charge are removed; any future
compressed-KV or expert-dequant workspace must be measured and charged by its owning allocator.

## D-019 — Validate TurboKV as two bounded kernels before QSA pruning

**Date:** 2026-09-18 · **Status:** accepted · **Evidence:** current QSA worktree and `LESSONS.md`

The fused TurboKV-deserialization plus sparse-attention Triton kernel triggered a `ptxas` host-RAM
failure of roughly 70 GiB. The current implementation therefore keeps decompression in a separate
Triton kernel and writes only the bounded page workspace consumed by the existing dense QSA
attention kernel. This split is the validation target; the fused path is not to be revived as an
optimization experiment before the separate path has correctness and A/B measurements.

The order is binding: compile and correctness pins, compare against `triton + bf16` at 16K, measure
TurboKV capacity and throughput at 128K/256K, and only then implement QSA block pruning or MoE/PLE
prefetch. Until those gates pass, TurboKV has capacity evidence but no certified PP/TG result.

## D-020 — Keep the first Turbo4 row as discovery evidence, not a certification gate

**Date:** 2026-09-18 · **Status:** accepted · **Evidence:** real-host run `turbo4-16k-eager-r086`

The split decompression path is proven executable on Flash-Next at 16K: PP 1713.7 / TG 24.72,
14.86 GiB VRAM and 99% GPU utilisation. The run used eager mode and disabled overlap after the
CUDA-graph path stalled, so it is not comparable enough to replace the 1857.7 / 28.685 baseline.
Keep it as discovery evidence until the graph/overlap behavior, matched `triton + bf16` arm and
repeat count are resolved.

The same probe reached MTP=1 speculation and accepted one draft token, then found a one-page
finished-request cache leak. Cleanup now covers the allocated `device_len` tail; this fix must be
tested before any MTP+Turbo4 throughput number is reported.

## D-021 — Diagnose page-accounting bugs by bisecting to the real mechanism, not by patching the
symptom's call site again

**Date:** 2026-09-18 · **Status:** accepted · **Evidence:** EXP-034

D-020's committed `cache.py` fix for the one-page leak (`alloc_end = max(cached_len, device_len)`)
was itself wrong: `cached_len < device_len` is a standing invariant of this codebase (the next,
not-yet-written slot), so the fix silently over-freed one page on every finished request at
`page_size=1`, regressing 11 pre-existing, unrelated scheduler tests
(`test_abort_inflight_prefill.py` and friends) that a scoped `pytest tests/scheduler/` run would
have caught immediately -- the original fix was validated only against the two spec-specific test
files it touched. **Decision: any fix to shared cache/page-accounting code runs the FULL
`tests/scheduler/` suite (not just the files the diff touched) before being trusted, and a bug
reproduced only at a specific numeric alignment (here: prompt length an exact multiple of
`page_size`) is diagnosed by re-deriving the allocation arithmetic from the actual call sites
(`allocate_paged`, `_prepare_batch`, `free_spec_reject`'s own contract) instead of adjusting the
one line the traceback points at.** The real bugs were three, all in `spec.py`, none touching
`cache.py`'s finished-tail logic at all: `_commit_spec_tokens`'s mid-window finish never
reclaiming the verify's surplus before `table_idx` recycling; `free_spec_reject`'s `keep_len`
argument using the wrong (lagged) convention; and the GDN-state replay's `_prepare_batch` call
re-running `allocate_paged` over an already-allocated range, orphaning the original page at a
page-boundary crossing. See EXP-034.

## D-022 — Fused verify MTP+TurboKV: TreeWY delta-rule pseudo-values and direct compressed tile attention

**Date:** 2026-09-18 · **Status:** accepted · **Evidence:** EXP-036, EXP-037, EXP-040, and A2/A8 audits

FreeToken's speculative verification under MTP currently executes target model verify over a $k+1$ token window and, upon partial/full rejection ($m \le k$ accepted), rolls back the GDN linear recurrent state and runs `gdn_replay` (a separate forward pass) to re-derive the seed residual for the next draft step. Concurrently, TurboKV currently decompresses historical 64-token tiles into a dedicated intermediate page workspace (`_ws_k`, `_ws_v`) before QSA sparse attention.

Line 12 of ROADMAP specifies fusing MTP verification with TurboKV attention to eliminate both the `gdn_replay` forward pass and the intermediate tile materialization penalty (-13% to -15% TG).

### 1. Mathematical Formulation: TreeWY Delta-Rule Pseudo-Values

The GDN (Gated Delta Net) layer update follows the linear recurrent delta rule:
$$S_t = S_{t-1} (I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

In chunked/parallel representation, the cumulative state update across a sequence of $k$ speculative tokens is expressed in semi-separable form using the lower-triangular kernel:
$$S_t = S_0 U_{1:t} + V_{1:t}^* K_{1:t}^T$$
where $U_{1:t} = \prod_{i=1}^t (I - \beta_i k_i k_i^T)$, and the pseudo-values $V^*$ satisfy the strictly lower-triangular system:
$$(I + L_{k, \beta}) V^* = V$$
with $L_{i, j} = \beta_j k_i^T k_j$ for $i > j$, and $0$ otherwise.

Because $(I + L_{k, \beta})$ is strictly lower-triangular with unit diagonal, the pseudo-values $v_1^*, \dots, v_m^*$ for any accepted prefix of length $m \le k$ are mathematically identical to the pseudo-values that would be obtained by running a forward pass of length $m$. Causal masking in attention ensures that the residual output `hidden_states[m-1]` computed during the verify forward has received no attention from rejected speculative positions $> m-1$.

**Decision:**
1. Eliminate `gdn_replay` entirely by capturing `hidden_states[m-1]` directly from the verify forward as the MTP draft seed residual.
2. In the GDN recurrent state update, checkpoint the per-step recurrent state during the small $k$-step verify window ($k \le 4$ costs only $k \times 128 \times 128 \times 2$ bytes = 128 KiB), or apply the prefix slice of the triangular pseudo-value accumulation $S_m = S_0 U_{1:m} + V_{1:m}^* K_{1:m}^T$. This eliminates the secondary prefill forward on spec reject, reducing rejection latency to 0 ms.

### 2. Direct Compressed Tile Attention (Zero-Materialization TurboKV)

During speculative verify, historical context ($t < d$) resides as compressed Turbo4 tiles (4-bit QSA coded blocks with Hadamard rotation and per-tile scale), while the speculative window ($d \le t < d+k$) is uncompressed FP16/BF16.

**Decision:**
The fused verify kernel loads compressed Turbo4 tiles directly into SRAM, dequantizes and applies the inverse Hadamard transformation on the fly within the thread-block tile accumulator, and evaluates attention queries against the uncompressed speculative window in local registers. No intermediate FP16 workspace (`_ws_k`, `_ws_v`) is materialized in DRAM, preserving the full memory savings of Turbo4 and removing the LTO memory bandwidth penalty.
