"""Authoritative VRAM ledger: one object owns the byte account for the whole engine.

Upstream v0.1.3 has no single memory owner. The engine takes ``mem_get_info`` snapshots in a
fixed order, ``net_cache_budget_bytes`` turns one of them into a budget that TWO independent
decisions spend (MoE expert slots first, KV pages from the residue), and everything the base
does not size -- CUDA-graph pools, attention-backend workspaces, Triton autotune scratch, the
gated-delta-net prefill workspace -- survives on the unmodelled ``(1 - memory_ratio)`` gap.
That gap is why Flash-Next OOMs at ``--memory-ratio 0.9`` (autotune asked for 256 MiB while
209 MiB were free, EXPERIMENTS.md EXP-001b) and only "works" at 0.86: the ratio was silently
running reserve policy, and the price of the silence is the ~1.5 GiB nobody may touch.

The ledger makes the reserve explicit and additive:

* every consumer is a named line item with a :class:`Kind`;
* :meth:`VramLedger.pool_budget_bytes` is what the two negotiable consumers (expert slots and
  KV pages) may split after everything knowable is charged;
* ``memory_ratio`` becomes a *cap* on the account instead of the policy: the ceiling is
  ``min(ratio x baseline, baseline - modelled_reserve)``, so lowering the ratio still tightens
  the plan (byte-compatible with the old formula when nothing is modelled) while raising it
  toward 1.0 stops being a gamble -- the plan refuses to promise the peak it cannot fund.

``report()`` prints the account so any MiB that disappeared is attributable, which is also the
honesty check on :meth:`unit_bytes`: a pool whose reported per-token cost differs from what it
allocated is a bug the report makes visible.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable

from freetoken.utils import div_ceil, init_logger

logger = init_logger(__name__)

_KIB = 1 << 10
_MIB = 1 << 20
_GIB = 1 << 30

# The FLA chunk width the gated-delta-net prefill kernels chunk by (kernel/fla CHUNK_SIZE);
# it sets how many ``h`` rows a T-token prefill materialises.
GDN_CHUNK_SIZE = 64

# ---- constants measured on this host (rtx5080, SM120); PERFORMANCE.md §2, EXP-001b -------
# Triton benchmarks autotune candidates with an L2-flush arena handed out by the CUDA driver's
# ``get_empty_cache_for_benchmark`` (triton/testing.py:152), 256 MiB on this card -- the exact
# allocation that OOM'd in EXP-001b. Freed after tuning, but it must be *fundable* while live,
# and it is live during a warmup forward, i.e. on top of that forward's own transients.
TRITON_AUTOTUNE_ARENA = 256 * _MIB
# The CUDA-graph pool is a private mempool holding every captured intermediate, so it is not
# reachable from a tensor walk: each captured shape adds its own copy of the forward's
# temporaries. Measured on this host as ~0.2 GB for one bs=1 graph and ~0.15 GB per additional
# batch size (A9 §3.1); the graph runner's static input buffers are the measured counterpart.
GRAPH_POOL_FIRST_SHAPE = 200 * _MIB
GRAPH_POOL_EXTRA_SHAPE = 150 * _MIB
# The CUDA-graph capture peak above the steady-state pool, PER captured shape. Measured as
# ~0.2 GB for a single bs=1 graph set on this host (EXP-001b); engine/graph.py:101-171
# captures one graph per batch size in the set and accounted for none of it.
GRAPH_CAPTURE_PEAK = 256 * _MIB
# Attention-backend plan/workspace memory allocated blind at first forward (attention/fi.py
# and the JIT probes); the radix scheduler's per-request CUDA-graph pool slots ride it too.
BACKEND_WORKSPACE = 128 * _MIB
# One image's vision transient: a 1024x1024 image is a 128 MiB uint8 pixel blob
# (mm/processor.py:519 resizes to 1024x1024 with factor-32 padding) plus ~3136 pixel tokens of
# tower activations (A9 §2). Priced per image, not per plausible burst: a multi-image request
# scales linearly and its real bound is the encoder cache, which today caps by TOKENS and image
# COUNT but never by BYTES (mm/encoder_cache.py:38) -- so the burst is a named gap in the
# account, not a number to average away here. Reserving the whole 0.4-1.2 GB burst band on a
# text deployment cost the 35B-A3B prefill guard ~0.3 %, which is the wrong trade while the
# encoder cache has no byte cap to price it against.
MM_ENCODER_PEAK = 192 * _MIB
# Allocator fragmentation and cross-rank drift: real, not modelable per-consumer, so it is a
# named reserve rather than a hole under the ratio.
FRAGMENTATION_RESERVE = 128 * _MIB
# Each extra captured shape beyond the first keeps its own (batch-sized) buffers; the bs=1
# graph dominates and the tail grows slowly, so the marginal shape is priced well below the
# first. Calibrated against the measured bs<=1 capture peak, not against a multi-shape run --
# widen it if a large --cuda-graph-max-bs run reports a capture peak above this line.
GRAPH_CAPTURE_EXTRA_SHAPE = 160 * _MIB
# The live activation stream of one forward: the residual stream plus the per-layer temporaries
# that are simultaneously reachable. Measured as the dominant transient on this host (A9 §3.2):
# ~1.1 GiB at a 16 384-token forward, ~0.27 GiB at the default 8 192-token chunk. `live_tensors`
# is the number kept reachable at once (A9 estimates 4-6); the decode side is batch-major
# instead, so the line is the larger of the two shapes.
LIVE_ACTIVATION_TENSORS = 6


def activation_peak_bytes(
    hidden_size: int, *, prefill_tokens: int = 0, batch: int = 1, decode_tokens: int = 1,
    itemsize: int = 2, live_tensors: int = LIVE_ACTIVATION_TENSORS,
) -> int:
    """Peak of the residual/temporary stream for the worst forward of each kind.

    Prefill is token-major (``T x hidden``) and decode is batch-major (``B x generated``
    bounded by the context), so the account takes the max: a big-batch decode and a long-chunk
    prefill are different consumers of the same pool.
    """
    prefill = int(prefill_tokens) * int(hidden_size) * itemsize * live_tensors
    decode = int(batch) * int(decode_tokens) * int(hidden_size) * itemsize * live_tensors
    return max(prefill, decode)


def tensor_bytes(obj) -> int:
    """Bytes of every CUDA tensor reachable in one hop from ``obj``.

    The measured counterpart to a formula: a consumer whose size we cannot derive (the offload
    cache's plan buffers, a graph runner's static inputs) still reports what it holds, so the
    account can be honest about it instead of leaving it to the fragmentation reserve. Only
    one hop deep on purpose -- a walk that finds another owner's tensors would double-count.

    Overlapping byte ranges are merged, because a view IS the same memory: the offload cache
    keeps ``prefill_bank_buffers`` as views into the first ``2 * num_experts`` slots of its own
    ``bank_caches``, and counting both billed the pool for those slots twice (which is exactly
    the 2x over-report this function used to have, and it looked like a plan that under-priced
    experts by 2x -- the account was wrong, not the plan).
    """
    import torch

    spans: list[tuple[int, int]] = []

    def visit(value) -> None:
        if isinstance(value, torch.Tensor):
            if value.is_cuda and value.nbytes:
                spans.append((value.data_ptr(), value.data_ptr() + value.nbytes))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in vars(obj).values():
        visit(value)
    total = 0
    cursor = 0
    for start, end in sorted(spans):
        if end <= cursor:
            continue
        total += end - max(start, cursor)
        cursor = end
    return total


def tensor_breakdown(obj, top: int = 4) -> str:
    """The largest tensor attributes of ``obj`` as ``name=GiB`` text, for a line's note.

    When a measured line comes in bigger than the formula that sized the same consumer, the
    name of the attribute is the whole diagnosis; printing it at startup beats reasoning about
    it later.
    """
    import torch

    sizes: list[tuple[int, str]] = []
    for name, value in vars(obj).items():
        if isinstance(value, dict):
            items = [(f"{name}.{key}", item) for key, item in value.items()]
        elif isinstance(value, (list, tuple)):
            items = [(f"{name}[{i}]", item) for i, item in enumerate(value)]
        else:
            items = [(name, value)]
        for label, item in items:
            if isinstance(item, torch.Tensor) and item.is_cuda:
                sizes.append((int(item.numel()) * int(item.element_size()), label))
    sizes.sort(reverse=True)
    return ", ".join(f"{name}={nbytes / _GIB:.3f}" for nbytes, name in sizes[:top])


def graph_capture_shapes(cuda_graph_max_bs: int | None) -> int:
    """How many graphs get captured: engine/graph.py:97 picks ``[1, 2, 4] + range(8, max+1, 8)``.

    Re-derived rather than imported because the ledger prices the account BEFORE the capture
    helper runs, and it must reserve for the set the capture will actually build.
    """
    if not cuda_graph_max_bs or cuda_graph_max_bs <= 0:
        return 0
    base = [1, 2, 4]
    if cuda_graph_max_bs < 8:
        return sum(1 for bs in base if bs <= cuda_graph_max_bs)
    return len(base) + (cuda_graph_max_bs - 8) // 8 + 1


def graph_capture_peak_bytes(cuda_graph_max_bs: int | None) -> int:
    shapes = graph_capture_shapes(cuda_graph_max_bs)
    if not shapes:
        return 0
    return GRAPH_CAPTURE_PEAK + GRAPH_CAPTURE_EXTRA_SHAPE * (shapes - 1)


def graph_pool_bytes(cuda_graph_max_bs: int | None) -> int:
    """The captured intermediates the graph pool holds for the whole session (semi-persistent)."""
    shapes = graph_capture_shapes(cuda_graph_max_bs)
    if not shapes:
        return 0
    return GRAPH_POOL_FIRST_SHAPE + GRAPH_POOL_EXTRA_SHAPE * (shapes - 1)


def page_table_bytes(max_running_req: int, max_seq_len: int, page_size: int,
                     itemsize: int = 4) -> int:
    """The engine's page table: one row per concurrent request (plus the dummy row) of
    32-aligned, page-rounded context. engine.py builds it after the KV solve, so the plan has
    to fund it from the formula or nothing else ever will."""
    from freetoken.utils import align_ceil

    columns = align_ceil(align_ceil(int(max_seq_len), int(page_size)), 32)
    return (int(max_running_req) + 1) * columns * itemsize


class Kind(str, enum.Enum):
    """How long a charge lives, which decides whether the planner may resize it.

    ``IMMUTABLE`` is settled by the checkpoint. ``PERSISTENT`` lives for the session but is
    re-solvable (that is what a cache rebuild does). ``SEMI_PERSISTENT`` is allocated once by
    the runtime at a size we do not negotiate. ``TRANSIENT`` peaks and falls but must be funded
    at its peak, because the peak arrives while the KV pool is already sitting on the memory.
    ``RESERVE`` is headroom we deliberately hand to nobody.
    """

    IMMUTABLE = "immutable"
    PERSISTENT = "persistent"
    SEMI_PERSISTENT = "semi-persistent"
    TRANSIENT = "transient"
    RESERVE = "reserve"
    MEASURED = "measured"


# The two consumers the planner actually decides between; everything else is charged first.
NEGOTIABLE = ("cache:expert", "cache:kv")
# The calibration reading is a report about the account, not a consumer of memory, so it is
# excluded from every total the planner acts on (ceiling, reserve, committed) and only ever
# compared against them.
MEASURED_LINE = "measured:allocator-held"
# What the allocator can be holding live at a quiet moment: everything the engine sized and
# keeps. TRANSIENT lines are excluded because they are gone by calibration time, RESERVE because
# it is by definition unallocated, MEASURED because it is the reading itself.
HELD_KINDS = (Kind.IMMUTABLE, Kind.PERSISTENT, Kind.SEMI_PERSISTENT)
# Calibration slack: the allocator's own rounding (a 2 MiB block minimum per segment,
# expandable_segments granularity) plus the temporaries of whichever forward ran last.
CALIBRATION_TOLERANCE = 256 * _MIB


@dataclass(frozen=True)
class Charge:
    name: str
    nbytes: int
    kind: Kind
    note: str = ""


def gdn_prefill_bytes(group, tokens: int, itemsize: int, *, batch: int = 1) -> int:
    """Peak workspace of ONE gated-delta-net layer's chunked-prefill forward over ``tokens``.

    The layers run sequentially, so this is a per-layer peak, not a per-model sum. Terms are
    every tensor the prefill branch of ``models/qwen4_exp/gdn.py:171-190`` keeps live at once:
    the fused projection (``conv_in`` = 2·key_dim + value_dim and ``z``), the q/k/v copies its
    ``reshape`` forces out of the split views, and inside ``kernel/fla`` the ``A`` slab
    (chunk_fwd.py:390), the ``w``/``u`` pair (wy_fast.py:130), ``h`` -- one [V, K] state per
    64-token chunk (chunk_delta_h.py:324) -- ``v_new`` and the output slab ``o`` (chunk_o.py:146,
    the 96 MiB allocation that OOM'd at --memory-ratio 0.9 in EXP-001b). The fp32 carried final
    state rides the concurrency, not the chunk, and is priced by the state-pool line.
    """
    tokens = max(int(tokens), 0)
    heads = group.num_value_heads
    k_heads = group.num_key_heads
    k_dim = group.key_head_dim
    v_dim = group.value_head_dim
    key_dim = k_heads * k_dim
    value_dim = heads * v_dim
    per_token = (
        2 * key_dim + value_dim  # conv_in
        + value_dim  # z
        + 2 * key_dim + value_dim  # q, k, v: reshaped copies of the split views
        + value_dim  # o
        + heads * k_dim * 2  # w + u
        + heads * GDN_CHUNK_SIZE  # A
        + v_dim * k_dim // GDN_CHUNK_SIZE  # h, one [V, K] state per chunk
        + value_dim  # v_new
        + 2 * heads * 4 // itemsize  # gate + beta, kept in fp32
    )
    return int(batch * tokens * per_token * itemsize)


def modelled_reserves(
    *,
    linear_group=None,
    prefill_tokens: int = 0,
    hidden_size: int = 0,
    dtype_itemsize: int = 2,
    batch: int = 1,
    autotune: bool = True,
    cuda_graph_max_bs: int | None = 1,
    backend_workspace: bool = True,
    mm_encoder: bool = False,
    dequant_scratch: int = 0,
    staging: int = 0,
) -> list[tuple[str, int, Kind, str]]:
    """The semi-persistent / transient / reserve lines the base allocates but never budgeted.

    Every entry is a *promise to fund a peak*, not an expectation of steady-state use: the
    consumer may come in under it, and the report shows the line either way. Summing the
    phases (instead of taking their max) is deliberate -- EXP-001b's OOM happened inside
    autotune while a prefill forward's own scratch was live, so they are not exclusive.
    """
    out: list[tuple[str, int, Kind, str]] = []
    if autotune:
        out.append(("transient:autotune", TRITON_AUTOTUNE_ARENA, Kind.TRANSIENT,
                    "Triton get_empty_cache_for_benchmark arena (measured)"))
    capture = graph_capture_peak_bytes(cuda_graph_max_bs)
    if capture:
        out.append((
            "graph:capture-peak", capture, Kind.TRANSIENT,
            f"CUDA graph capture peak, {graph_capture_shapes(cuda_graph_max_bs)} shape(s)",
        ))
    if backend_workspace:
        out.append(("workspace:attention", BACKEND_WORKSPACE, Kind.SEMI_PERSISTENT,
                    "attention backend plan + workspace + per-request pool slots"))
    pool = graph_pool_bytes(cuda_graph_max_bs)
    if pool:
        out.append(("graph:pool", pool, Kind.SEMI_PERSISTENT,
                    "CUDA-graph private pool, held for the session"))
    if hidden_size and prefill_tokens:
        out.append((
            "transient:activations",
            activation_peak_bytes(
                hidden_size, prefill_tokens=prefill_tokens, batch=batch,
                decode_tokens=prefill_tokens, itemsize=dtype_itemsize,
            ),
            Kind.TRANSIENT,
            f"live activation stream over a {prefill_tokens}-token forward "
            f"(hidden={hidden_size}, {LIVE_ACTIVATION_TENSORS} tensors)",
        ))
    if mm_encoder:
        out.append(("transient:mm-encoder", MM_ENCODER_PEAK, Kind.TRANSIENT,
                    "vision prefill peak (mm/processor.py resize + per-image pixel blob)"))
    if linear_group is not None and prefill_tokens:
        out.append((
            "transient:gdn-prefill",
            gdn_prefill_bytes(linear_group, prefill_tokens, dtype_itemsize, batch=batch),
            Kind.TRANSIENT,
            f"one GDN layer over a {prefill_tokens}-token chunk (heads={linear_group.num_value_heads}"
            f", k={linear_group.key_head_dim}, v={linear_group.value_head_dim})",
        ))
    if dequant_scratch:
        out.append(("transient:dequant-scratch", int(dequant_scratch), Kind.TRANSIENT,
                    "compressed-KV / expert dequant tile scratch"))
    if staging:
        out.append(("workspace:staging", int(staging), Kind.SEMI_PERSISTENT,
                    "H2D staging buffers"))
    out.append(("reserve:fragmentation", FRAGMENTATION_RESERVE, Kind.RESERVE,
                "allocator fragmentation + cross-rank drift (was the hidden 1-ratio gap)"))
    return out


@dataclass
class VramLedger:
    """Byte account for one device.

    ``baseline_free`` is the free-VRAM snapshot taken before any model allocation; the ceiling
    is ``min(memory_ratio x baseline_free, baseline_free - reserve_bytes)``, and
    :attr:`reserve_bytes` is the sum of the TRANSIENT / SEMI_PERSISTENT / RESERVE lines, i.e.
    the memory that must exist uncommitted for the peak to be survivable.
    """

    device_total_bytes: int
    baseline_free: int
    memory_ratio: float
    charges: dict[str, Charge] = field(default_factory=dict)

    # ---- charging ---------------------------------------------------------------

    def charge(self, name: str, nbytes: int, kind: Kind, note: str = "") -> int:
        """Record (or re-price) a line item; returns the delta. Re-charging a name replaces
        it, so a re-plan or a rebuild moves the account instead of double-counting."""
        nbytes = int(nbytes)
        assert nbytes >= 0, f"{name}: negative charge {nbytes}"
        previous = self.charges.get(name)
        self.charges[name] = Charge(name, nbytes, Kind(kind), note)
        return nbytes - (previous.nbytes if previous else 0)

    def charge_many(self, items: Iterable[tuple[str, int, Kind, str]]) -> None:
        for name, nbytes, kind, note in items:
            self.charge(name, nbytes, kind, note=note)

    def release(self, name: str) -> int:
        """Drop a line item, returning the bytes it gives back (0 when it was never charged)."""
        previous = self.charges.pop(name, None)
        return 0 if previous is None else previous.nbytes

    def bytes_of(self, name: str) -> int:
        charge = self.charges.get(name)
        return 0 if charge is None else charge.nbytes

    def total(self, kinds: Iterable[Kind] | None = None, exclude: Iterable[str] = ()) -> int:
        """Sum of the account. The calibration reading is never part of a total unless it is
        asked for by name: it reports what the allocator holds, so counting it would bill the
        engine for its own bookkeeping (and made ``headroom`` read -14 GiB on a healthy run)."""
        wanted = frozenset(kinds) if kinds is not None else None
        skip = set(exclude)
        if wanted is None or Kind.MEASURED not in wanted:
            skip.add(MEASURED_LINE)
        return sum(
            c.nbytes
            for c in self.charges.values()
            if c.name not in skip and (wanted is None or c.kind in wanted)
        )

    # ---- the policy -------------------------------------------------------------

    @property
    def reserve_bytes(self) -> int:
        """Memory that must stay EMPTY: the transient peaks and the named reserve.

        SEMI_PERSISTENT lines are NOT here -- the page table and the backend workspaces are
        real allocations the engine holds, so they belong in :meth:`engine_committed_bytes`;
        putting them in the reserve made the plan subtract them twice and report a zero pool
        budget on a card with 5.7 GiB unspent. What is left here is the memory that exists
        only as a peak: what ``--memory-ratio`` used to hide.
        """
        return self.total((Kind.TRANSIENT, Kind.RESERVE))

    @property
    def cap_bytes(self) -> int:
        """The user's cap, expressed in bytes: what ``memory_ratio`` alone would allow."""
        return int(self.memory_ratio * self.baseline_free)

    @property
    def ceiling_bytes(self) -> int:
        """What the engine may hold in total, from the ONE formula the budget arithmetic uses.

        ``memory_ratio`` caps the account and the modelled reserve floors it: the ratio's own
        ``(1 - ratio)`` hole already IS a reserve, so only a peak larger than that hole bites.
        With nothing modelled this is the pre-ledger ``ratio x baseline`` formula exactly.
        """
        from freetoken.engine.cache_budget import ceiling_bytes

        return ceiling_bytes(self.baseline_free, self.memory_ratio, self.reserve_bytes)

    def engine_committed_bytes(self) -> int:
        """What the engine itself will have allocated once every consumer is sized -- the
        reserve lines are excluded because they are the headroom that stays EMPTY, not memory
        the engine holds, and the calibration reading because it is not a consumer."""
        return self.total(exclude=(*NEGOTIABLE, MEASURED_LINE)) - self.reserve_bytes

    def held_bytes(self) -> int:
        """What the account claims the allocator must be holding at a quiet moment."""
        return self.total(HELD_KINDS, exclude=(MEASURED_LINE,))

    def engine_overhead_bytes(self) -> int:
        """Committed bytes that no pool or expert cost model prices.

        The KV families budget their own tiers and the engine adds the sibling GDN state pool,
        so the residue the ledger owns is the page table, the CUDA-graph pool, the backend
        workspaces and any PLE GPU residency. The engine folds exactly this set into the fixed
        term of the budget arithmetic -- which is why it is a distinct kind set and not simply
        ``total(PERSISTENT)``: double-subtracting a line is as wrong as omitting it.
        """
        return self.total(
            (Kind.PERSISTENT, Kind.SEMI_PERSISTENT),
            exclude=(*NEGOTIABLE, "cache:gdn-state", MEASURED_LINE),
        )

    def pool_budget_bytes(self, extra_fixed_bytes: int = 0) -> int:
        """Bytes the negotiable consumers (expert slots + KV pages) may split.

        ``extra_fixed_bytes`` is a cost the caller knows but the account cannot see yet -- a
        pool family's own fixed tier, which only exists once that pool is allocated. Passing it
        here is what makes the ledger the single place the split is decided instead of a
        reporter that describes a decision made elsewhere.

        Floored at zero on purpose: a negative remainder is a planning error, and handing the
        caller 0 makes the budget policy fail its own fit assert in arithmetic instead of
        OOMing inside a later CUDA allocation.
        """
        committed = self.engine_committed_bytes() + int(extra_fixed_bytes)
        return max(0, self.ceiling_bytes - committed)

    def headroom_bytes(self) -> int:
        """Unspent room under the ceiling: ``ceiling - held``.

        Negative is not a bug and not yet an OOM -- it says the consumers hold more than the
        ceiling promised, which on this host means the expert slot was priced below what the
        kernel's side tables actually cost. :meth:`uncommitted_bytes` is the physical version of
        the same question, and that one going negative is the OOM.
        """
        return self.ceiling_bytes - self.held_bytes()

    def uncommitted_bytes(self) -> int:
        """What the pre-model baseline leaves for nobody: ``baseline - held - reserve``.

        The account's real safety margin: it is what remains when every held byte is subtracted
        AND the modelled peak is still allowed to happen. ``--memory-ratio`` used to be the only
        thing standing between a filled pool and an OOM; this is the number it was guessing at.
        """
        return self.baseline_free - self.held_bytes() - self.reserve_bytes

    # ---- reporting --------------------------------------------------------------

    def report(self) -> str:
        """The account, one line per consumer, grouped by kind.

        Printed at startup and on every rebuild: subtracting the listed lines from the device
        total must land near what ``nvidia-smi`` reports free, so an unexplainable gap is a
        missing line item.
        """
        order = [k.value for k in Kind]
        rows = sorted(
            self.charges.values(),
            key=lambda c: (order.index(c.kind.value), -c.nbytes),
        )
        width = max((len(r.name) for r in rows), default=len("consumer"))
        lines = [f"  {'consumer'.ljust(width)}  {'GiB':>8}  kind               what"]
        for r in rows:
            lines.append(
                f"  {r.name.ljust(width)}  {r.nbytes / _GIB:8.3f}  {r.kind.value:<17} {r.note}"
            )
        lines.append(
            f"  {'-' * width}  {'-' * 8}  committed {self.total() / _GIB:.3f} GiB, ceiling "
            f"{self.ceiling_bytes / _GIB:.3f} GiB (ratio {self.memory_ratio} x baseline "
            f"{self.baseline_free / _GIB:.3f} GiB, reserve {self.reserve_bytes / _GIB:.3f} GiB)"
        )
        lines.append(
            f"  {'-' * width}  {'-' * 8}  headroom {self.headroom_bytes() / _GIB:+.3f} GiB "
            f"under the ceiling, uncommitted {self.uncommitted_bytes() / _GIB:+.3f} GiB "
            f"(baseline - held - reserve), pool budget "
            f"{self.pool_budget_bytes() / _GIB:.3f} GiB, account holds "
            f"{self.held_bytes() / _GIB:.3f} GiB, device "
            f"{self.device_total_bytes / _GIB:.2f} GiB"
        )
        return "VRAM ledger:\n" + "\n".join(lines)

    def log(self) -> None:
        logger.info_rank0(self.report())


def open_ledger(
    *,
    device_total_bytes: int,
    baseline_free: int,
    memory_ratio: float,
    weights_bytes: int,
    reserves: Iterable[tuple[str, int, Kind, str]] = (),
) -> VramLedger:
    """Start the account after the weights are resident: charge the immutable line and the
    modelled reserves, so the first thing a caller can ask is ``pool_budget_bytes()``."""
    ledger = VramLedger(
        device_total_bytes=int(device_total_bytes),
        baseline_free=int(baseline_free),
        memory_ratio=float(memory_ratio),
    )
    ledger.charge("weights:model", int(weights_bytes), Kind.IMMUTABLE, note="resident parameters")
    ledger.charge_many(reserves)
    return ledger


__all__ = [
    "BACKEND_WORKSPACE",
    "FRAGMENTATION_RESERVE",
    "GDN_CHUNK_SIZE",
    "GRAPH_CAPTURE_EXTRA_SHAPE",
    "GRAPH_CAPTURE_PEAK",
    "LIVE_ACTIVATION_TENSORS",
    "MM_ENCODER_PEAK",
    "NEGOTIABLE",
    "TRITON_AUTOTUNE_ARENA",
    "Charge",
    "Kind",
    "VramLedger",
    "activation_peak_bytes",
    "gdn_prefill_bytes",
    "graph_capture_peak_bytes",
    "graph_capture_shapes",
    "modelled_reserves",
    "open_ledger",
]
