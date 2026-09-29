"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.

    Supports mixed/non-uniform geometry across layers (Phase 7): for banks with
    varying layer shapes/dtypes, each distinct geometry allocates its own GPU slot
    cache of shape ``(cache_size, *shape[1:])``, so all distinct geometries must
    be summed to reflect actual VRAM occupancy per slot.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    total = 0
    for per_layer in sources.values():
        if not per_layer:
            continue
        unique_geoms = {
            (tuple(layer_t.shape[1:]), layer_t.dtype): layer_t[0].numel() * layer_t.element_size()
            for layer_t in per_layer
        }
        total += sum(unique_geoms.values())
    return total


# Byte alignment of each expert pool inside a bank's slot arena (keeps every view dtype- and
# vector-copy-aligned); the padding is part of the priced bytes.
POOL_ALIGN = 256


@dataclass(frozen=True)
class ExpertPool:
    """Layers whose rows share one geometry in every bank: they share one LRU slot range.

    ``row_bytes`` is one expert row per bank, in the sources' bank order."""

    layers: tuple[int, ...]
    row_bytes: tuple[int, ...]


def expert_pools(sources: dict[str, "list[torch.Tensor]"]) -> list[ExpertPool]:
    """Group layers by their per-bank row geometry, the largest group first.

    One geometry across all layers is one pool (the uniform case). A mixed GGUF (a few
    layers at a wider quant, an MTP draft's own Q8_0 bank) gets one pool per distinct
    geometry, so a small group never costs rows sized for every layer."""
    banks = [per_layer for per_layer in sources.values() if per_layer]
    groups: dict[tuple, list[int]] = {}
    for layer in range(len(banks[0]) if banks else 0):
        key = tuple((tuple(b[layer].shape[1:]), b[layer].dtype) for b in banks)
        groups.setdefault(key, []).append(layer)
    pools = [
        ExpertPool(
            tuple(layers),
            tuple(b[layers[0]][0].numel() * b[layers[0]].element_size() for b in banks),
        )
        for layers in groups.values()
    ]
    return sorted(pools, key=lambda p: (-len(p.layers), p.layers[0]))


def pool_capacities(
    pools: list[ExpertPool],
    num_experts: int,
    rows: int,
    min_rows: int,
    extra_lo: Sequence[int] | None = None,
) -> list[int]:
    """Split ``rows`` resident expert rows over ``pools``: an equal share per layer (what a
    global LRU converges to), each pool clamped to ``[min(min_rows, hi), hi]`` where
    ``hi = layers * num_experts`` is all it can ever hold. ``min_rows`` is the most distinct
    experts one decode step may route (the LRU needs them resident at once); 0 means one
    full layer. ``extra_lo`` raises per-pool minimums further (prefill front floors); they
    are part of the budget, so the equal share simply shrinks to pay for them. The
    remainder goes to unsaturated pools, largest first."""
    hi = [len(p.layers) * num_experts for p in pools]
    lo = [min(h, min_rows or num_experts) for h in hi]
    if extra_lo is not None:
        lo = [max(l, min(h, x)) for l, h, x in zip(lo, hi, extra_lo)]
    rows = min(rows, sum(hi))

    def caps(share: int) -> list[int]:
        return [min(h, max(l, len(p.layers) * share)) for p, l, h in zip(pools, lo, hi)]

    a, b = 0, num_experts  # largest share with sum(caps) <= rows
    while a < b:
        m = (a + b + 1) // 2
        a, b = (m, b) if sum(caps(m)) <= rows else (a, m - 1)
    out = caps(a)
    left = rows - sum(out)
    for i, h in enumerate(hi):
        add = max(0, min(left, h - out[i]))
        out[i] += add
        left -= add
    return out


def pool_layout(pools: list[ExpertPool], caps: list[int]) -> tuple[list[list[int]], list[int]]:
    """Byte offset of every pool in each bank's arena (``offsets[pool][bank]``) and each
    arena's total bytes; pools are laid out in order, each start ``POOL_ALIGN``-aligned."""
    num_banks = len(pools[0].row_bytes) if pools else 0
    ends = [0] * num_banks
    offsets = []
    for pool, cap in zip(pools, caps):
        offs = [div_ceil(end, POOL_ALIGN) * POOL_ALIGN for end in ends]
        offsets.append(offs)
        ends = [off + cap * rb for off, rb in zip(offs, pool.row_bytes)]
    return offsets, ends


def pool_staging_fits(
    pools: list[ExpertPool],
    caps: list[int],
    num_experts: int,
    arena_bytes: list[int],
    prefill_pools: set[int] | None = None,
) -> bool:
    """A pool smaller than one layer materializes a prefill layer into the front of each
    arena (overlaying resident rows), so every arena must hold ``num_experts`` of its rows."""
    return all(
        num_experts * rb <= size
        for p, (pool, cap) in enumerate(zip(pools, caps))
        if (prefill_pools is None or p in prefill_pools) and cap < num_experts
        for rb, size in zip(pool.row_bytes, arena_bytes)
    )


def expert_cache_bytes(pools: list[ExpertPool], num_experts: int, rows: int, min_rows: int) -> int:
    """GPU bytes of an expert cache holding ``rows`` resident rows (all arenas, padding
    included); what ``OffloadMoeCache`` allocates for the same arguments. No rows = no
    cache (0 bytes), not the pools' decode floors."""
    if rows <= 0:
        return 0
    _, ends = pool_layout(pools, pool_capacities(pools, num_experts, rows, min_rows))
    return sum(ends)


def expert_rows_bounds(
    pools: list[ExpertPool],
    num_experts: int,
    min_rows: int,
    prefill_pools: set[int] | None = None,
) -> tuple[int, int]:
    """Smallest and largest usable ``rows``: the floor is one full layer (the offload
    cache's historic minimum), raised until every sub-layer pool can stage a prefill layer
    (monotone in ``rows``, so a bisection finds it)."""
    top = sum(len(p.layers) for p in pools) * num_experts

    def fits(rows: int) -> bool:
        caps = pool_capacities(pools, num_experts, rows, min_rows)
        return pool_staging_fits(
            pools, caps, num_experts, pool_layout(pools, caps)[1], prefill_pools=prefill_pools
        )

    lo, hi = min(num_experts, top), top
    while lo < hi:
        m = (lo + hi) // 2
        lo, hi = (lo, m) if fits(m) else (m + 1, hi)
    return lo, top


def max_expert_rows(
    pools: list[ExpertPool], num_experts: int, budget_bytes: int, min_rows: int, lo: int, hi: int
) -> int:
    """Most rows in ``[lo, hi]`` whose cache fits ``budget_bytes`` (``lo`` if none does)."""
    while lo < hi:
        m = (lo + hi + 1) // 2
        if expert_cache_bytes(pools, num_experts, m, min_rows) <= budget_bytes:
            lo = m
        else:
            hi = m - 1
    return lo


def ceiling_bytes(baseline_free: int, memory_ratio: float, reserve_bytes: int = 0) -> int:
    """The most VRAM the engine may hold, given that ``reserve_bytes`` must stay free for the
    runtime's peaks (see engine/vram_ledger.py).

    ``(1 - memory_ratio) x baseline`` was always the de-facto reserve -- it is the hole between
    the promise and the card -- so the ledger only charges the SHORTFALL between the modelled
    peak and that hole. A low ratio is therefore untouched (byte-identical to the pre-ledger
    formula, which is what keeps this brick from resizing every existing checkpoint), a high
    ratio is clamped down to exactly what the peak needs, and ratio 1.0 becomes usable: the
    plan funds the transient instead of praying the allocator has room for it. That removes the
    hidden safety margin EXP-001b had to guess its way around (Flash-Next OOM'd at 0.9 and
    "worked" at 0.86)."""
    cap = int(memory_ratio * baseline_free)
    implicit_reserve = baseline_free - cap
    return cap - max(0, int(reserve_bytes) - implicit_reserve)


def net_cache_budget_bytes(
    memory_ratio: float,
    baseline_free: int,
    weights_bytes: int,
    fixed_cache_size: int,
    reserve_bytes: int = 0,
) -> int:
    """Net GPU bytes available for the MoE + KV pools: the ceiling minus weights and fixed
    (non-paged) cache. Single source of truth for startup auto-sizing and the runtime-rebuild
    fit check; ``reserve_bytes`` is the ledger's modelled peak, 0 for callers that have no
    ledger (and therefore keep the pre-ledger formula exactly)."""
    return (
        ceiling_bytes(baseline_free, memory_ratio, reserve_bytes) - weights_bytes - fixed_cache_size
    )


# Every KV pool allocates one page past the usable ones for padded / dummy rows to write
# into: create_kv_pool and every pool's rebuild pass ``num_pages + 1`` (mha, hybrid SWA,
# DSV4, dsa). The arithmetic below plans in USABLE pages, so it has to price the extra one
# or the plan over-commits the budget it was solved against.
DUMMY_PAGES = 1


def pool_pages(num_pages: int) -> int:
    """Pages a pool holding ``num_pages`` usable pages actually allocates."""
    return num_pages + DUMMY_PAGES


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies: MoE slots plus the
    ``num_pages`` usable KV pages and the pool's dummy page."""
    return moe_cache_size * per_expert_bytes + pool_pages(num_pages) * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    # The reserve is a promise about USABLE pages, so it costs the pool's dummy page too: the
    # greedy expert fill below must leave that many bytes behind or the num_pages floor will
    # over-commit the budget it was solved against (a 1.5 MiB miss that used to fail the plan).
    kv_reserve_bytes = pool_pages(kv_reserve_pages) * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    # One page of what ``remaining`` buys is spent on the pool's dummy page, so the plan hands
    # back usable pages and never promises bytes it did not price.
    num_pages = max(remaining // cache_per_page - DUMMY_PAGES, kv_reserve_pages)
    total = required_bytes(moe_cache_size, num_pages, per_expert_bytes, cache_per_page)
    # Experts filled greedily against the KV reserve in BYTES, but the reserve is a floor in
    # PAGES: when the byte fit leaves less than the floor, num_pages snaps up to it and the
    # plan is over-committed -- on this host by as little as 1.5 MiB, which used to be a hard
    # startup failure. Hand the excess back as expert slots (the greedy side) and re-solve.
    if total > budget_bytes and moe_cache_size > lo:
        give_back = -(-(total - budget_bytes) // per_expert_bytes)
        moe_cache_size = max(lo, moe_cache_size - give_back)
        remaining = budget_bytes - moe_cache_size * per_expert_bytes
        num_pages = max(remaining // cache_per_page - DUMMY_PAGES, kv_reserve_pages)
        total = required_bytes(moe_cache_size, num_pages, per_expert_bytes, cache_per_page)
        overlap = overlap and moe_cache_size >= 2 * num_experts
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
    reserve_bytes: int = 0,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.
    ``reserve_bytes`` is the VRAM ledger's modelled peak: the plan may not promise it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(
        memory_ratio, baseline_free, weights_bytes, fixed_cache_size, reserve_bytes
    )
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
