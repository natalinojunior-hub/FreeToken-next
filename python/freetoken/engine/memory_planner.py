"""Multi-stage automatic VRAM planner with physical feedback.

This module replaces the heuristic ledger-based planning with a physically-validated
multi-stage solver that:
1. Measures physical GPU memory at controlled synchronization points
2. Computes exact costs where possible
3. Measures runtime-dependent costs through controlled probes
4. Treats prefill chunk size as an independent solvable variable
5. Allocates expert cache only from genuine residual memory
6. Validates final configuration under real allocation state
7. Automatically recovers/replans if initial candidate doesn't fit
"""

from __future__ import annotations

import gc
import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from freetoken.attention import AttnType, attention_backend_info
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import BaseKVCachePool
from freetoken.kvcache.linear_state_pool import (
    _linear_pool_min_slots,
    _linear_pool_num_slots,
    spec_state_steps,
    state_pool_bytes,
)
from freetoken.models import create_model
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
from freetoken.utils import align_ceil, div_ceil, init_logger, mem_GB, torch_dtype

from .config import EngineConfig
from .cache_budget import (
    ExpertPool,
    expert_bytes_per_slot,
    expert_cache_bytes,
    expert_pools,
    expert_rows_bounds,
    max_expert_rows,
    pool_pages,
    required_bytes,
)
from .graph import GraphRunner, get_free_memory, verify_graph_tokens
from .vram_ledger import (
    CALIBRATION_TOLERANCE,
    Kind,
    Charge,
    modelled_reserves,
    open_ledger,
    page_table_bytes,
    tensor_bytes,
)

logger = init_logger(__name__)


class ContextInfeasible(RuntimeError):
    """The requested context does not fit this KV format (the auto KV ladder tries the next)."""

_MIB = 1 << 20
_GIB = 1 << 30
_MIN_CHUNK = 256
_MAX_PROBE_CHUNK = 8192
_CHUNK_LADDER = (256, 512, 1024, 2048, 4096, 8192)


class _AllocMeter:
    """Attribute allocator growth to consecutive constructors."""

    def __init__(self, device: torch.device):
        self.device = device
        self.mark = torch.cuda.memory_allocated(device)
        self.actual: Dict[str, int] = {}

    def took(self, owner: str) -> None:
        now = torch.cuda.memory_allocated(self.device)
        self.actual[owner] = now - self.mark
        self.mark = now


@dataclass(frozen=True)
class PhysicalMemorySnapshot:
    """Physical memory measurements at a synchronization point."""

    driver_free: int
    driver_total: int
    allocator_allocated: int
    allocator_reserved: int
    allocator_peak_allocated: int
    allocator_peak_reserved: int

    @property
    def driver_used(self) -> int:
        return self.driver_total - self.driver_free

    @property
    def non_allocator_visible(self) -> int:
        """Memory used by driver/non-PyTorch allocations."""
        return self.driver_used - self.allocator_reserved

    def __str__(self) -> str:
        return (
            f"PhysicalMemory(driver_free={mem_GB(self.driver_free)}, "
            f"allocated={mem_GB(self.allocator_allocated)}, "
            f"reserved={mem_GB(self.allocator_reserved)}, "
            f"peak_alloc={mem_GB(self.allocator_peak_allocated)}, "
            f"peak_reserved={mem_GB(self.allocator_peak_reserved)}, "
            f"non_alloc={mem_GB(self.non_allocator_visible)})"
        )


def take_physical_snapshot(
    device: torch.device, reset_peaks: bool = False
) -> PhysicalMemorySnapshot:
    """Take a synchronized physical memory snapshot."""
    torch.cuda.synchronize(device)
    if reset_peaks:
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return PhysicalMemorySnapshot(
        driver_free=free,
        driver_total=total,
        allocator_allocated=torch.cuda.memory_allocated(device),
        allocator_reserved=torch.cuda.memory_reserved(device),
        allocator_peak_allocated=torch.cuda.max_memory_allocated(device),
        allocator_peak_reserved=torch.cuda.max_memory_reserved(device),
    )


def log_reconciliation(stage: str, planned: Dict[str, int], actual: Dict[str, int]) -> None:
    """Log PLANNED | ACTUAL | DELTA per memory owner; warn on unexplained > 128 MiB."""
    logger.info_rank0(f"  [{stage}] owner | PLANNED | ACTUAL | DELTA")
    for owner in planned.keys() | actual.keys():
        p, a = planned.get(owner, 0), actual.get(owner, 0)
        line = f"  [{stage}] {owner} | {p / _MIB:.1f} MiB | {a / _MIB:.1f} MiB | {(a - p) / _MIB:+.1f} MiB"
        if abs(a - p) > 128 * _MIB:
            logger.warning_rank0(line)
        else:
            logger.info_rank0(line)


@dataclass(frozen=True)
class StaticCostModel:
    """Exactly calculable memory costs before any allocation."""

    weights_bytes: int
    expert_bytes_per_slot: int
    kv_bytes_per_page: int
    kv_fixed_bytes: int
    page_tokens: int
    dummy_page_bytes: int
    gdn_state_bytes_per_slot: int
    gdn_num_slots: int
    page_table_bytes: int
    expert_auxiliary_bytes: int
    quantization_side_tables: int
    staging_buffers: int
    attention_backend_fixed: int
    min_expert_slots: int
    max_expert_slots: int
    total_experts: int
    prefill_overlap: bool
    # Mixed-geometry banks: the cache's geometry pools, priced exactly (expert_cache_bytes);
    # empty = one geometry, linear in slots.
    expert_pools: tuple[ExpertPool, ...] = ()
    min_pool_rows: int = 0

    def kv_pages_for_context(self, tokens: int) -> int:
        return div_ceil(tokens, self.page_tokens)

    def kv_bytes_for_context(self, tokens: int) -> int:
        pages = self.kv_pages_for_context(tokens)
        return pool_pages(pages) * self.kv_bytes_per_page + self.kv_fixed_bytes

    def expert_bytes_for_slots(self, slots: int) -> int:
        if self.expert_pools:
            return expert_cache_bytes(
                list(self.expert_pools), self.total_experts, slots, self.min_pool_rows
            )
        return slots * self.expert_bytes_per_slot

    def expert_slots_for_bytes(self, budget: int) -> int:
        """Most slots (within the min/max bounds) whose expert cache fits ``budget``."""
        if self.expert_pools:
            return max_expert_rows(
                list(self.expert_pools),
                self.total_experts,
                budget,
                self.min_pool_rows,
                self.min_expert_slots,
                self.max_expert_slots,
            )
        return min(self.max_expert_slots, budget // self.expert_bytes_per_slot)

    def gdn_state_total_bytes(self) -> int:
        return self.gdn_state_bytes_per_slot * self.gdn_num_slots

    def fixed_overhead_bytes(self) -> int:
        """All non-negotiable fixed costs EXCEPT model weights and KV.

        KV's fixed cost and dummy page are priced by every `kv_bytes`
        computation (`kv_bytes_for_context` and its inline equivalents at
        each solver call site) -- including them here too double-charged the
        same bytes against the budget (confirmed: the joint-solver invariant
        tripped once the transient reserve below became accurate enough that
        this double-charge was the binding constraint, not slack it hid
        inside before). Weights are already accounted for in the physical
        budget (post-mandatory snapshot).
        """
        return (
            self.gdn_state_total_bytes()
            + self.page_table_bytes
            + self.expert_auxiliary_bytes
            + self.quantization_side_tables
            + self.staging_buffers
            + self.attention_backend_fixed
        )


@dataclass(frozen=True)
class RuntimeCalibration:
    """Measured runtime-dependent memory costs (Phase D)."""

    chunk_lo: int
    transient_lo: int
    chunk_hi: int
    transient_hi: int
    lazy_persistent: int
    graph_capture_peak: int
    graph_pool_size: int
    non_pytorch_growth: int

    def transient_at(self, chunk: int) -> int:
        """Prefill transient at ``chunk`` (<= chunk_hi): linear between the two
        measured points; below chunk_lo the smaller measurement bounds it."""
        assert chunk <= self.chunk_hi, (chunk, self.chunk_hi)
        if chunk <= self.chunk_lo:
            return self.transient_lo
        slope = max(0, self.transient_hi - self.transient_lo) / (self.chunk_hi - self.chunk_lo)
        return self.transient_lo + math.ceil(slope * (chunk - self.chunk_lo))


@dataclass(frozen=True)
class PlanCandidate:
    """A candidate memory plan."""

    prefill_chunk: int
    expert_slots: int
    kv_pages: int
    kv_tokens: int
    expert_bytes: int
    kv_bytes: int
    fixed_overhead: int
    transient_reserve: int
    total_committed: int
    physical_free_after: int
    validation_passed: bool = False
    validation_error: str = ""
    prefill_overlap: bool = False

    def __str__(self) -> str:
        return (
            f"Plan(chunk={self.prefill_chunk}, experts={self.expert_slots}, "
            f"kv_pages={self.kv_pages} ({self.kv_tokens} tokens), "
            f"expert={mem_GB(self.expert_bytes)}, kv={mem_GB(self.kv_bytes)}, "
            f"fixed={mem_GB(self.fixed_overhead)}, transient={mem_GB(self.transient_reserve)}, "
            f"total={mem_GB(self.total_committed)}, free_after={mem_GB(self.physical_free_after)}, "
            f"valid={self.validation_passed})"
        )


class MemoryPlanner:
    """Multi-stage automatic VRAM planner."""

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        model_config,
        dtype: torch.dtype,
        pool_cls: type[BaseKVCachePool],
        banks,
        expert_auxiliary_bytes: int = 0,
        quantization_side_tables: int = 0,
        staging_buffers: int = 0,
        method=None,
    ):
        self.config = config
        self.device = device
        self.model_config = model_config
        self.dtype = dtype
        self.pool_cls = pool_cls
        self.banks = banks
        self.banks_sources = banks.sources
        self.expert_auxiliary_bytes = expert_auxiliary_bytes
        self.quantization_side_tables = quantization_side_tables
        self.staging_buffers = staging_buffers
        self.method = method

        self.baseline_snapshot: Optional[PhysicalMemorySnapshot] = None
        self.post_weights_snapshot: Optional[PhysicalMemorySnapshot] = None
        self.post_mandatory_snapshot: Optional[PhysicalMemorySnapshot] = None
        self.static_model: Optional[StaticCostModel] = None
        self.runtime_calibration: Optional[RuntimeCalibration] = None
        self.final_plan: Optional[PlanCandidate] = None

        # Probe artifacts for cleanup
        self._probe_model = None
        self._probe_kv_pool = None
        self._probe_expert_cache = None
        self._probe_linear_pool = None
        self._probe_attn_backend = None
        self._probe_graph_runner = None
        self._orig_linear_state_pool = None
        self._orig_attn_backend = None

    # ======================= Phase A: Hardware Baseline =======================

    _host_pages = 0  # KV RAM tier pages; plan() sets it per run

    def _device_kv_pages(self, config: EngineConfig) -> int:
        """Device KV pages the context needs. With a KV RAM tier every token already has a RAM
        page, so the device keeps only the hot floor (``kv_reserve_tokens``); the rest of the
        budget goes to expert slots."""
        sm = self.static_model
        context_pages = sm.kv_pages_for_context(config.max_seq_len)
        if not self._host_pages:
            return context_pages
        hot = sm.kv_pages_for_context(min(config.max_seq_len, config.kv_reserve_tokens))
        return max(hot, context_pages - self._host_pages)

    def _create_kv_pool(self, config: EngineConfig, device_pages: int) -> BaseKVCachePool:
        """Planner pools carry the RAM tier too, so full-context probes address real slots."""
        return create_kv_pool(
            config,
            device_pages + self._host_pages,
            device=self.device,
            dtype=self.dtype,
            host_pages=self._host_pages,
        )

    def phase_a_hardware_baseline(self) -> PhysicalMemorySnapshot:
        """Measure physical GPU memory before any model allocation."""
        logger.info_rank0("Phase A: Measuring hardware baseline...")
        snapshot = take_physical_snapshot(self.device, reset_peaks=True)
        self.baseline_snapshot = snapshot
        logger.info_rank0(f"  Baseline: {snapshot}")
        return snapshot

    def phase_a_post_weights(self, model) -> PhysicalMemorySnapshot:
        """Measure after model weights are loaded."""
        logger.info_rank0("Phase A: Measuring post-weights memory...")
        snapshot = take_physical_snapshot(self.device, reset_peaks=False)
        self.post_weights_snapshot = snapshot
        logger.info_rank0(f"  Post-weights: {snapshot}")
        return snapshot

    def phase_a_post_mandatory(self) -> PhysicalMemorySnapshot:
        """Measure after all mandatory (non-negotiable) structures are allocated."""
        logger.info_rank0("Phase A: Measuring post-mandatory memory...")
        snapshot = take_physical_snapshot(self.device, reset_peaks=True)
        self.post_mandatory_snapshot = snapshot
        logger.info_rank0(f"  Post-mandatory: {snapshot}")
        return snapshot

    def min_pool_rows(self, num_experts: int) -> int:
        """Every geometry pool's LRU floor (see ``decode_pool_floor``)."""
        from freetoken.moe.offload_cache import decode_pool_floor

        return decode_pool_floor(
            num_experts, self.model_config.num_experts_per_tok, self.config.max_running_req
        )

    def _expert_bytes_per_slot(self) -> int:
        """Compute per-expert slot bytes from method layout (primary), banks.sources, or banks.expert_geometry."""
        # Primary: method layout (always available when method is set)
        if self.method is not None:
            try:
                layout = self.method.layout()
                total = 0
                for spec in layout.values():
                    if not getattr(spec, "resident", False):
                        elements = 1
                        for dim in spec.shape:
                            elements *= dim
                        total += elements * torch.tensor(0, dtype=spec.dtype).element_size()
                if total > 0:
                    return total
            except Exception:
                pass

        # Fallback: sources (non-streamed case)
        if self.banks_sources:
            result = expert_bytes_per_slot(self.banks_sources)
            if result > 0:
                return result

        # Fallback: expert_geometry (GGUF streamed case)
        if hasattr(self.banks, "expert_geometry") and self.banks.expert_geometry:
            total = 0
            for (bank_idx, role, quant_type), (shape, dtype) in self.banks.expert_geometry.items():
                elements = 1
                for dim in shape:
                    elements *= dim
                total += elements * torch.tensor(0, dtype=dtype).element_size()
            return total

        return 0

    # ======================= Phase B: Exact Static Cost Model =======================

    def phase_b_build_static_model(
        self,
        weights_bytes: int,
        num_experts: int,
        num_moe_layers: int,
        prefill_overlap: bool,
        method_slot_limit: Optional[int],
        max_running_req: int,
        max_seq_len: int,
        page_size: int,
    ) -> StaticCostModel:
        """Build exact static cost model from known geometries."""
        logger.info_rank0("Phase B: Building exact static cost model...")

        # Expert bytes per slot
        per_expert = self._expert_bytes_per_slot()

        # KV geometry from pool class
        kv_per_page, kv_fixed, page_tokens, min_reserve = self.pool_cls.kv_cost(self.config)
        dummy_page_bytes = kv_per_page  # +1 dummy page

        # GDN state pool: fixed_overhead must match what phase_h_construct_final_pools
        # actually builds (_linear_pool_num_slots, includes the cross-request snapshot
        # cache), not the bare non-evictable floor (_linear_pool_min_slots). The final
        # pool is always constructed at num_slots with no smaller fallback, so budgeting
        # at min_slots under-reserved every plan by (num_slots - min_slots) GDN slots --
        # a plan the solver approved as fitting could still OOM at final construction.
        gdn_slots = _linear_pool_num_slots(self.config)
        gdn_bytes_per_slot = state_pool_bytes(self.config, 1)
        gdn_total = gdn_bytes_per_slot * gdn_slots

        # Page table
        pt_bytes = page_table_bytes(max_running_req, max_seq_len, page_size)

        # Attention backend construction cost has no closed form: Phase C
        # measures it; nothing is guessed before that.
        attn_fixed = 0

        # Min/max expert slots
        min_slots = num_experts * (2 if prefill_overlap else 1)
        max_slots = num_moe_layers * num_experts
        if per_expert == 0:
            raise RuntimeError("Unable to determine expert slot geometry for automatic planning")
        if method_slot_limit is not None:
            max_slots = min(max_slots, method_slot_limit)
        # Mixed geometry: the cache splits its rows into per-geometry pools (OffloadMoeCache.
        # _alloc_bank_caches), so price those pools instead of one row of every geometry.
        pools = expert_pools(self.banks_sources) if self.banks_sources else []
        pool_floor = self.min_pool_rows(num_experts)
        if len(pools) > 1:
            lo, hi = expert_rows_bounds(pools, num_experts, pool_floor)
            min_slots, max_slots = max(min_slots, lo), min(max_slots, hi)
        else:
            pools = []

        self.static_model = StaticCostModel(
            weights_bytes=weights_bytes,
            expert_bytes_per_slot=per_expert,
            kv_bytes_per_page=kv_per_page,
            kv_fixed_bytes=kv_fixed,
            page_tokens=page_tokens,
            dummy_page_bytes=dummy_page_bytes,
            gdn_state_bytes_per_slot=gdn_bytes_per_slot,
            gdn_num_slots=gdn_slots,
            page_table_bytes=pt_bytes,
            expert_auxiliary_bytes=self.expert_auxiliary_bytes,
            quantization_side_tables=self.quantization_side_tables,
            staging_buffers=self.staging_buffers,
            attention_backend_fixed=attn_fixed,
            min_expert_slots=min_slots,
            max_expert_slots=max_slots,
            total_experts=num_experts,
            prefill_overlap=prefill_overlap,
            expert_pools=tuple(pools),
            min_pool_rows=pool_floor,
        )

        logger.info_rank0(
            f"  Static model: weights={mem_GB(weights_bytes)}, "
            f"expert/slot={mem_GB(per_expert)}, kv/page={mem_GB(kv_per_page)}, "
            f"page_tokens={page_tokens}, gdn_slots={gdn_slots}, "
            f"gdn/slot={mem_GB(gdn_bytes_per_slot)}, page_table={mem_GB(pt_bytes)}, "
            f"expert_aux={mem_GB(self.expert_auxiliary_bytes)}, "
            f"quant_tables={mem_GB(self.quantization_side_tables)}, "
            f"min_experts={min_slots}, max_experts={max_slots}"
        )

        return self.static_model

    # ======================= Phase C: Minimal Viable Configuration =======================

    def phase_c_build_minimal_config(
        self,
        config: EngineConfig,
        model,
    ) -> Tuple[BaseKVCachePool, OffloadMoeCache, Any]:
        """Build minimal runtime config for probing."""
        logger.info_rank0("Phase C: Building minimal viable configuration...")

        # Minimum expert cache (floor)
        min_experts = self.static_model.min_expert_slots

        # Minimum KV pages for requested context
        required_pages = self._device_kv_pages(config)

        # Create minimal KV pool
        logger.info_rank0(
            f"  Phase C: required_pages={required_pages} for max_seq_len={config.max_seq_len}"
        )
        meter = _AllocMeter(self.device)
        kv_pool = self._create_kv_pool(config, required_pages)
        meter.took("kv")

        # Create minimal expert cache
        method = self.method
        method_slot_limit = method.slot_limit() if method is not None else None
        expert_cache = OffloadMoeCache(
            num_layers=self.model_config.num_moe_layers,
            num_experts=self.model_config.num_experts,
            cache_size=min_experts,
            device=self.device,
            cache_policy=config.moe_cache_policy,
            prefill_overlap=config.moe_prefill_overlap,
            prefill_hit_d2d=config.moe_prefill_hit_d2d,
            quant_format=self.banks.quant_format,
            gguf_expert_types=getattr(self.model_config, "gguf_expert_types", None),
            decode_target="gpu",
            layout=None,
            max_slots=method_slot_limit,
            min_pool_rows=self.min_pool_rows(self.model_config.num_experts),
        )
        expert_cache.set_bank_sources(self.banks_sources)
        expert_cache.set_alphas(
            getattr(self.banks_sources, "gate_up_alpha", None),
            getattr(self.banks_sources, "down_alpha", None),
        )
        attach_offload_moe_cache(model, expert_cache)
        meter.took("experts")

        # Create linear state pool
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            linear_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_min_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
                slot_states=config.model_config.slot_states,
                spec_steps=spec_state_steps(config),
            )
        else:
            linear_pool = None
        meter.took("gdn")

        # Create attention backend - need to set probe KV pool on global context
        from freetoken.core import get_global_ctx
        from freetoken.attention import create_attention_backend

        global_ctx = get_global_ctx()
        original_kv_cache = getattr(global_ctx, "kv_cache", None)
        global_ctx.kv_cache = kv_pool
        try:
            attn_backend = create_attention_backend(config.attention_backend, config.model_config)
        finally:
            if original_kv_cache is not None:
                global_ctx.kv_cache = original_kv_cache
            else:
                delattr(global_ctx, "kv_cache")
        meter.took("attn_backend")

        # Reconcile the static model against what these constructors really
        # allocated. The attention backend's construction-time buffers have no
        # closed form, so the measurement replaces the placeholder term.
        sm = self.static_model
        planned = {
            "kv": sm.kv_bytes_for_context(required_pages * sm.page_tokens),
            "experts": sm.expert_bytes_for_slots(min_experts),
            "gdn": state_pool_bytes(config, 1) * _linear_pool_min_slots(config)
            if linear_group is not None
            else 0,
            "attn_backend": sm.attention_backend_fixed,
        }
        log_reconciliation("Phase C", planned, meter.actual)
        self.static_model = dataclasses.replace(
            sm, attention_backend_fixed=meter.actual["attn_backend"]
        )

        self._probe_kv_pool = kv_pool
        self._probe_expert_cache = expert_cache
        self._probe_linear_pool = linear_pool
        self._probe_attn_backend = attn_backend
        self._probe_model = model
        # Engine.__init__ already built production linear_state_pool/attn_backend
        # on global_ctx before invoking the planner; save them so cleanup can put
        # them back instead of leaving the engine permanently pointed at a probe
        # object or None.
        self._orig_linear_state_pool = getattr(global_ctx, "linear_state_pool", None)
        self._orig_attn_backend = getattr(global_ctx, "attn_backend", None)
        global_ctx.linear_state_pool = linear_pool
        global_ctx.attn_backend = attn_backend

        logger.info_rank0(f" Minimal config: experts={min_experts}, kv_pages={required_pages}")
        self._probe_required_pages = required_pages
        return kv_pool, expert_cache, linear_pool

    # ======================= Phase D: Runtime Calibration/Probe =======================

    def _measure_prefill_transient(self, model, config: EngineConfig, chunk: int) -> int:
        """Bytes the caching allocator must obtain from the driver, on top of
        everything already resident, to run the worst prefill step at ``chunk``
        (first chunk + context-final chunk, see _run_validation_prefill).

        Measured in reserved (not allocated) bytes: reserved is what the driver
        actually hands out, so allocator rounding/fragmentation is included
        instead of guessed. Lazily-created persistent tensors are excluded
        (they are priced once as ``lazy_persistent``)."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        pre_reserved = torch.cuda.memory_reserved(self.device)
        pre_alloc = torch.cuda.memory_allocated(self.device)
        self._run_validation_prefill(model, config, chunk, self._probe_kv_pool)
        torch.cuda.synchronize(self.device)
        kept = torch.cuda.memory_allocated(self.device) - pre_alloc
        transient = torch.cuda.max_memory_reserved(self.device) - pre_reserved - kept
        logger.info_rank0(
            f"  Prefill transient at chunk={chunk}: {mem_GB(transient)} (kept={mem_GB(kept)})"
        )
        return max(0, transient)

    def phase_d_runtime_calibration(self, config: EngineConfig, model) -> RuntimeCalibration:
        """Measure runtime costs no closed form prices.

        A warm-up forward first creates every lazily-built persistent tensor
        (autotune caches, backend workspaces). Then the worst prefill transient
        is measured at two chunk sizes, giving the per-token slope so the solver
        prices any chunk up to the largest one measured instead of
        extrapolating a single point.
        """
        c_hi = min(config.max_extend_tokens, config.max_seq_len, _MAX_PROBE_CHUNK)
        logger.info_rank0(f"Phase D: Runtime calibration up to chunk={c_hi}...")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        pre_alloc = torch.cuda.memory_allocated(self.device)
        pre_free = torch.cuda.mem_get_info(self.device)[0]
        pre_reserved = torch.cuda.memory_reserved(self.device)
        while True:
            c_lo = max(_MIN_CHUNK, c_hi // 2)
            try:
                self._run_validation_prefill(model, config, c_lo, self._probe_kv_pool)
                lazy_persistent = torch.cuda.memory_allocated(self.device) - pre_alloc
                t_lo = self._measure_prefill_transient(model, config, c_lo)
                t_hi = (
                    t_lo if c_hi == c_lo else self._measure_prefill_transient(model, config, c_hi)
                )
                break
            except torch.cuda.OutOfMemoryError:
                # Even the minimal pools leave no room for this chunk: halve it.
                # A genuine search step, bounded below by _MIN_CHUNK.
                gc.collect()
                torch.cuda.empty_cache()
                if c_hi <= _MIN_CHUNK:
                    raise
                logger.warning_rank0(f"  Probe OOM at chunk={c_hi}; halving")
                c_hi //= 2
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        non_pytorch = (pre_free - torch.cuda.mem_get_info(self.device)[0]) - (
            torch.cuda.memory_reserved(self.device) - pre_reserved
        )

        graph_peak, graph_pool = self._measure_graph_capture(config, model)

        self.runtime_calibration = RuntimeCalibration(
            chunk_lo=c_lo,
            transient_lo=t_lo,
            chunk_hi=c_hi,
            transient_hi=t_hi,
            lazy_persistent=lazy_persistent,
            graph_capture_peak=graph_peak,
            graph_pool_size=graph_pool,
            non_pytorch_growth=non_pytorch,
        )
        logger.info_rank0(f"  Calibration: {self.runtime_calibration}")
        return self.runtime_calibration

    def _measure_graph_capture(self, config: EngineConfig, model) -> Tuple[int, int]:
        """Measure CUDA graph capture memory cost."""
        if not config.cuda_graph_max_bs or config.cuda_graph_max_bs <= 0:
            return 0, 0

        from freetoken.core import Req

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        pre = take_physical_snapshot(self.device)

        # Create minimal graph runner for measurement
        dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,
            cache_handle=None,
        )

        try:
            runner = GraphRunner(
                stream=torch.cuda.current_stream(),
                device=self.device,
                model=model,
                attn_backend=self._probe_attn_backend,
                cuda_graph_bs=[1],
                cuda_graph_max_bs=config.cuda_graph_max_bs,
                free_memory=pre.driver_free,
                max_seq_len=config.max_seq_len,
                vocab_size=config.model_config.vocab_size,
                dummy_req=dummy_req,
                moe_offload_cache=self._probe_expert_cache,
                mrope=config.model_config.model_is_mrope,
                verify_tokens=verify_graph_tokens(config.spec_mtp),
            )
            self._probe_graph_runner = runner
        except Exception as e:
            logger.warning_rank0(f"Graph capture measurement failed: {e}")
            return 0, 0

        post = take_physical_snapshot(self.device)
        peak = post.allocator_peak_allocated - pre.allocator_allocated
        pool_size = tensor_bytes(runner) if runner else 0

        return peak, pool_size

    # ======================= Phase F/G: Canonical Ledger Solve =======================

    def ledger(
        self, config: EngineConfig, chunk: int, expert_slots: int, kv_pages: int
    ) -> Dict[str, int]:
        """The canonical VRAM ledger: every byte the plan commits beyond the
        measured budget, one term per owner.

        The budget (driver free after weights, CUDA context and probe cleanup)
        already excludes weights, CUDA context/modules and non-PyTorch runtime
        growth, so none of those appear here -- subtracting them again would
        double-count. ``transient`` is the larger of the prefill peak and the
        graph-capture peak: capture runs once at startup, never during a
        prefill, so the two never coexist and are not summed.
        """
        sm, rc = self.static_model, self.runtime_calibration
        return {
            "gdn_state": sm.gdn_state_total_bytes(),
            "page_table": sm.page_table_bytes,
            "expert_aux": sm.expert_auxiliary_bytes,
            "quant_tables": sm.quantization_side_tables,
            "staging": sm.staging_buffers,
            "attn_backend": sm.attention_backend_fixed,
            "experts": sm.expert_bytes_for_slots(expert_slots),
            "kv": pool_pages(kv_pages) * sm.kv_bytes_per_page + sm.kv_fixed_bytes,
            "lazy_persistent": rc.lazy_persistent if rc else 0,
            "graph_pool": rc.graph_pool_size if rc else 0,
            "transient": max(rc.transient_at(chunk), rc.graph_capture_peak) if rc else 0,
        }

    def infeasible(
        self, config: EngineConfig, budget: int, ledger: Dict[str, int]
    ) -> "ContextInfeasible":
        required = sum(ledger.values())
        # Largest context the same ledger funds: every non-KV term fixed, KV pages from what
        # remains (the RAM tier, when active, still covers its share of the context).
        sm = self.static_model
        spare = budget - (required - ledger.get("kv", 0)) - sm.kv_fixed_bytes
        device_pages = max(0, spare // sm.kv_bytes_per_page - 1)  # -1: the dummy page
        most = min(config.max_seq_len, (device_pages + self._host_pages) * sm.page_tokens)
        most = most // 1024 * 1024
        owners = ", ".join(
            f"{k}={mem_GB(v)}" for k, v in sorted(ledger.items(), key=lambda kv: -kv[1])[:5]
        )
        return ContextInfeasible(
            f"o contexto pedido de {config.max_seq_len} tokens não é possível nesse hardware, "
            f"o máximo possível é {most} tokens. "
            f"VRAM plan infeasible for max_seq_len={config.max_seq_len}: "
            f"required={required} bytes ({mem_GB(required)}), available={budget} bytes "
            f"({mem_GB(budget)}), shortfall={required - budget} bytes "
            f"({mem_GB(required - budget)}); largest owners: {owners}. "
            f"Use a compressed --kv-format (turbo3/turbo4), lower "
            f"--max-running-requests, or free VRAM."
        )

    def phase_fg_solve_chunk_and_experts(
        self, config: EngineConfig, budget: int
    ) -> Tuple[int, int, int]:
        """Solve once against the measured budget.

        Priority: the requested context is a hard floor (its KV pages are always
        funded), then the largest measured-safe prefill chunk that still funds
        the expert floor, then as many expert slots as fit; the sub-slot
        remainder becomes extra KV pages. Returns (chunk, expert_slots, kv_pages).
        """
        logger.info_rank0("Phase F/G: Solving canonical ledger...")
        sm, rc = self.static_model, self.runtime_calibration
        required_pages = self._device_kv_pages(config)
        min_bytes = sm.expert_bytes_for_slots(sm.min_expert_slots)

        def residual(chunk: int) -> int:
            return budget - sum(self.ledger(config, chunk, 0, required_pages).values())

        candidates = sorted(
            {c for c in (*_CHUNK_LADDER, rc.chunk_lo, rc.chunk_hi) if c <= rc.chunk_hi},
            reverse=True,
        )
        chunk = next((c for c in candidates if residual(c) >= min_bytes), None)
        if chunk is None:
            raise self.infeasible(
                config,
                budget,
                self.ledger(config, candidates[-1], sm.min_expert_slots, required_pages),
            )
        left = residual(chunk)
        expert_slots = sm.expert_slots_for_bytes(left)
        kv_pages = (
            required_pages
            + (left - sm.expert_bytes_for_slots(expert_slots)) // sm.kv_bytes_per_page
        )
        ledger = self.ledger(config, chunk, expert_slots, kv_pages)
        assert sum(ledger.values()) <= budget, (ledger, budget)
        logger.info_rank0(
            f"  chunk={chunk}, expert_slots={expert_slots}, kv_pages={kv_pages}, "
            f"committed={mem_GB(sum(ledger.values()))} of budget={mem_GB(budget)}"
        )
        for owner, nbytes in ledger.items():
            logger.info_rank0(f"    ledger {owner}: {nbytes / _MIB:.1f} MiB")
        return chunk, expert_slots, kv_pages

    # ======================= Phase H: Final Pool Construction =======================

    def phase_h_construct_final_pools(
        self,
        config: EngineConfig,
        model,
        expert_slots: int,
        kv_pages: int,
        prefill_overlap: bool,
    ) -> Tuple[BaseKVCachePool, OffloadMoeCache, Any]:
        """Build the solved pools exactly as sized (validation-only; engine.py
        builds the permanent ones with the same geometry) and reconcile each
        owner against the ledger."""
        logger.info_rank0(
            f"Phase H: Constructing final pools (experts={expert_slots}, kv_pages={kv_pages})..."
        )
        meter = _AllocMeter(self.device)
        kv_pool = self._create_kv_pool(config, kv_pages)
        meter.took("kv")

        method = self.method
        method_slot_limit = method.slot_limit() if method is not None else None
        expert_cache = OffloadMoeCache(
            num_layers=config.model_config.num_moe_layers,
            num_experts=config.model_config.num_experts,
            cache_size=expert_slots,
            device=self.device,
            cache_policy=config.moe_cache_policy,
            prefill_overlap=prefill_overlap,
            prefill_hit_d2d=config.moe_prefill_hit_d2d,
            quant_format=self.banks.quant_format,
            gguf_expert_types=getattr(self.model_config, "gguf_expert_types", None),
            decode_target="gpu",
            layout=None,
            max_slots=method_slot_limit,
            min_pool_rows=self.min_pool_rows(self.model_config.num_experts),
        )
        expert_cache.set_bank_sources(self.banks_sources)
        expert_cache.set_alphas(
            getattr(self.banks_sources, "gate_up_alpha", None),
            getattr(self.banks_sources, "down_alpha", None),
        )
        attach_offload_moe_cache(model, expert_cache)
        meter.took("experts")

        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            linear_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
                slot_states=config.model_config.slot_states,
                spec_steps=spec_state_steps(config),
            )
        else:
            linear_pool = None
        meter.took("gdn_state")

        ledger = self.ledger(config, 0, expert_slots, kv_pages)
        log_reconciliation(
            "Phase H", {k: ledger[k] for k in ("kv", "experts", "gdn_state")}, meter.actual
        )
        return kv_pool, expert_cache, linear_pool

    def _cleanup_probe_artifacts(self):
        """Destroy probe artifacts and verify memory reclamation.

        phase_c_build_minimal_config wires the probe pools into places that
        outlive the planner's own `_probe_*` attributes: the real model's MoE
        layers hold `layer.offload_cache` (via attach_offload_moe_cache), and
        the global context holds `linear_state_pool`/`attn_backend`. Deleting
        only the planner's references left those strong references alive, so
        the probe's ~2.6 GiB expert cache was never actually freed -- this is
        why post-cleanup driver_free previously showed no improvement over
        post-mandatory. Detach all of them before dropping the objects.
        """
        from freetoken.core import get_global_ctx

        if self._probe_model is not None:
            attach_offload_moe_cache(self._probe_model, None)

        global_ctx = get_global_ctx()
        global_ctx.linear_state_pool = self._orig_linear_state_pool
        global_ctx.attn_backend = self._orig_attn_backend
        self._orig_linear_state_pool = None
        self._orig_attn_backend = None

        for attr in [
            "_probe_graph_runner",
            "_probe_attn_backend",
            "_probe_linear_pool",
            "_probe_expert_cache",
            "_probe_kv_pool",
            "_probe_model",
        ]:
            obj = getattr(self, attr, None)
            if obj is not None:
                if hasattr(obj, "destroy_cuda_graphs"):
                    obj.destroy_cuda_graphs()
                del obj
                setattr(self, attr, None)

        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    # ======================= Phase I: Final In-Situ Validation =======================

    def phase_i_final_validation(
        self,
        config: EngineConfig,
        model,
        kv_pool: BaseKVCachePool,
        expert_cache: OffloadMoeCache,
        linear_pool: Any,
        prefill_chunk: int,
    ) -> Tuple[bool, str]:
        """Run the solved plan's worst prefill on the real final pools.

        Returns (valid, message). No headroom margin: the ledger already priced
        every owner, so an OOM here is an unpriced owner, reported, not absorbed.
        """
        logger.info_rank0("Phase I: Final in-situ validation...")

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        pre_val = take_physical_snapshot(self.device)

        # Phase H's pools are validation-only (engine.py builds the real, permanent
        # kv_cache/linear_state_pool/attn_backend/offload cache after plan() returns),
        # so wire them onto global_ctx only for the duration of this forward pass and
        # restore whatever engine.py had configured there beforehand.
        from freetoken.core import get_global_ctx
        from freetoken.attention import create_attention_backend

        global_ctx = get_global_ctx()
        orig_linear_state_pool = getattr(global_ctx, "linear_state_pool", None)
        orig_attn_backend = getattr(global_ctx, "attn_backend", None)
        orig_kv_cache = getattr(global_ctx, "kv_cache", None)
        global_ctx.linear_state_pool = linear_pool
        global_ctx.kv_cache = kv_pool
        global_ctx.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        try:
            try:
                # A single prefill can never exceed max_seq_len tokens regardless of
                # the scheduler's chunk cap -- validating at prefill_chunk itself
                # would overrun the KV page table when the requested context is
                # smaller than the chunk (e.g. small-context smoke tests), which is
                # not a real validation failure, just an oversized probe request.
                validation_chunk = min(prefill_chunk, config.max_seq_len)
                self._run_validation_prefill(model, config, validation_chunk, kv_pool)

                torch.cuda.synchronize(self.device)
                transient = torch.cuda.max_memory_reserved(self.device) - pre_val.allocator_reserved
                log_reconciliation(
                    "Phase I",
                    {"transient": self.runtime_calibration.transient_at(validation_chunk)},
                    {"transient": transient},
                )
                return True, "Validation passed"
            except torch.cuda.OutOfMemoryError as e:
                return False, f"Validation OOM: {e}"
        finally:
            global_ctx.linear_state_pool = orig_linear_state_pool
            global_ctx.attn_backend = orig_attn_backend
            if orig_kv_cache is not None:
                global_ctx.kv_cache = orig_kv_cache
            elif hasattr(global_ctx, "kv_cache"):
                delattr(global_ctx, "kv_cache")

    def _run_validation_prefill(
        self, model, config: EngineConfig, chunk_size: int, kv_pool: BaseKVCachePool
    ):
        """Run final validation as real serving would: one prefill forward per chunk,
        continuation chunks carrying the real GDN recurrent state forward.

        A single-chunk validation never exercises the ``has_initial_state=True``
        Triton specialization that any prompt longer than one chunk hits on its
        second forward (see docs/dev/LESSONS.md) -- validating only the first
        chunk can pass while that continuation path still OOMs in real serving.
        """
        from freetoken.core import Batch, Req, get_global_ctx
        from freetoken.attention.linear import build_fla_metadata

        global_ctx = get_global_ctx()
        aligned_max_seq_len = (
            (config.max_seq_len + config.page_size - 1) // config.page_size * config.page_size
        )
        validation_page_table = torch.zeros(
            config.max_running_req + 1, aligned_max_seq_len, dtype=torch.int32, device=self.device
        )
        global_ctx.page_table = validation_page_table
        if hasattr(kv_pool, "attach_page_table"):
            kv_pool.attach_page_table(validation_page_table)

        table_idx = config.max_running_req
        # Map the whole context to real KV slots up front so the final chunk's
        # attention reads a full-context page table, exactly as a real prompt would.
        validation_page_table[table_idx, : config.max_seq_len] = torch.arange(
            config.max_seq_len, dtype=torch.int32, device=self.device
        )

        def _run_chunk(cached_len: int, extend_len: int):
            device_len = cached_len + extend_len
            assert device_len <= config.max_seq_len, (
                f"validation chunk end {device_len} exceeds requested context "
                f"{config.max_seq_len}; kv_pool was not sized to hold it"
            )
            dummy_req = Req(
                input_ids=torch.zeros(device_len, dtype=torch.int32, device="cpu"),
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=1,
                uid=-1,
                sampling_params=None,
                cache_handle=None,
            )
            batch = Batch(reqs=[dummy_req], phase="prefill")
            batch.padded_reqs = batch.reqs
            batch.input_ids = torch.zeros(extend_len, dtype=torch.int32, device=self.device)
            batch.positions = torch.arange(
                cached_len, device_len, dtype=torch.int32, device=self.device
            )
            if config.model_config.model_is_mrope:
                batch.mrope_positions = batch.positions.unsqueeze(0).expand(3, -1).contiguous()

            batch.out_loc = validation_page_table[table_idx, cached_len:device_len].clone()

            if hasattr(global_ctx, "attn_backend") and global_ctx.attn_backend is not None:
                global_ctx.attn_backend.prepare_metadata(batch)
            fla = build_fla_metadata(batch, self.device)
            batch.fla_metadata = fla

            with torch.inference_mode(), global_ctx.forward_batch(batch):
                model.forward()

        # First chunk: fresh GDN state (cached_len=0). Final chunk: ends at
        # max_seq_len, carrying real GDN state (has_initial_state=True) and paying
        # the full-context attention/indexer workspace -- the worst prefill step
        # real serving can take, so probe and validation measure the same peak.
        _run_chunk(cached_len=0, extend_len=chunk_size)
        if config.max_seq_len > chunk_size:
            tail = min(chunk_size, config.max_seq_len - chunk_size)
            _run_chunk(cached_len=config.max_seq_len - tail, extend_len=tail)

        torch.cuda.synchronize(self.device)

    # ======================= Main Planning Entry Point =======================

    def plan(
        self,
        config: EngineConfig,
        model,
        num_experts: int,
        num_moe_layers: int,
        prefill_overlap: bool,
        method_slot_limit: Optional[int],
        max_running_req: int,
        max_seq_len: int,
        page_size: int,
        weights_bytes: Optional[int] = None,
        host_reserve_bytes: int = 0,
        host_pages: int = 0,
    ) -> PlanCandidate:
        """Execute multi-stage planning. ``host_pages`` is the KV RAM tier (0 = none)."""
        self._host_pages = host_pages
        logger.info_rank0("=" * 60)
        logger.info_rank0("Starting automatic VRAM planning")
        logger.info_rank0("=" * 60)

        # Phase A: Hardware baseline
        self.phase_a_hardware_baseline()
        baseline_free = self.baseline_snapshot.driver_free

        # model weights: use provided or measure
        if weights_bytes is not None:
            self.weights_bytes = weights_bytes
            # post-weights snapshot not available, use baseline with weights subtracted
            from dataclasses import replace

            self.post_weights_snapshot = replace(
                self.baseline_snapshot, driver_free=baseline_free - weights_bytes
            )
            logger.info_rank0(f" Phase A: Using provided weights={mem_GB(weights_bytes)}")
        else:
            self.phase_a_post_weights(model)
            weights_bytes = (
                self.baseline_snapshot.driver_free - self.post_weights_snapshot.driver_free
            )
            self.weights_bytes = weights_bytes

        # Phase B: Static cost model
        self.phase_b_build_static_model(
            weights_bytes=weights_bytes,
            num_experts=num_experts,
            num_moe_layers=num_moe_layers,
            prefill_overlap=prefill_overlap,
            method_slot_limit=method_slot_limit,
            max_running_req=max_running_req,
            max_seq_len=max_seq_len,
            page_size=page_size,
        )

        # Phase C: Minimal viable config
        # Reject before any pool exists if the context floor alone cannot fit.
        sm = self.static_model
        floor = self.ledger(config, 0, sm.min_expert_slots, self._device_kv_pages(config))
        if sum(floor.values()) + host_reserve_bytes > baseline_free:
            raise self.infeasible(config, baseline_free, floor)
        self.phase_c_build_minimal_config(config, model)

        # Phase A continued: post-mandatory measurement is diagnostic only (it
        # runs after Phase C's temporary probe pools -- min-experts expert
        # cache, minimal KV pool -- are already resident). Using its driver_free
        # as the solve budget double-counts: the probe's own footprint gets
        # subtracted once here, then Phase F/G would ask to fund the equivalent
        # final cache again out of what's left, even though _cleanup_probe_artifacts()
        # frees the probe before the final pools are ever built. The real budget
        # for sizing the final persistent pools is the free memory that existed
        # right after weights loaded, before any probe scaffolding was created.
        self.phase_a_post_mandatory()
        probe_overhead = (
            self.baseline_snapshot.driver_free - self.post_mandatory_snapshot.driver_free
        )
        logger.info_rank0(
            f"  Probe scaffolding overhead (reclaimed before final pools): {mem_GB(probe_overhead)}"
        )

        # Phase D: warm-up + two-point prefill transient + graph capture,
        # measured against the minimal probe pools.
        self.phase_d_runtime_calibration(config, model)

        # The budget is measured ONCE, after the probe scaffolding is gone: it
        # already reflects CUDA context, modules, lmem and every persistent
        # byte the probe left behind, so the ledger never re-subtracts them.
        self._cleanup_probe_artifacts()
        budget_snapshot = take_physical_snapshot(self.device)
        budget = budget_snapshot.driver_free
        # KV RAM tiering's device-side overhead (compressed index/rope rows + host_staging)
        # is not a pool the solve below ever builds, so take it off the top before solving
        # expert slots / device KV pages, same as engine.py's non-planner startup path.
        budget -= host_reserve_bytes
        logger.info_rank0(
            f"  Solve budget (post-probe): {budget_snapshot}, host tier reserve: {mem_GB(host_reserve_bytes)}"
        )

        # The ledger should price every owner; when validation still OOMs, an owner is unpriced
        # (e.g. an MTP draft path at long context). Report it and re-solve with a smaller
        # budget instead of refusing to serve: the plan must never OOM, and experts shrink first.
        for attempt in range(_VALIDATION_RETRIES + 1):
            chosen_chunk, expert_slots, kv_pages = self.phase_fg_solve_chunk_and_experts(
                config, budget
            )
            kv_pool, expert_cache, linear_pool = self.phase_h_construct_final_pools(
                config, model, expert_slots, kv_pages, prefill_overlap
            )
            try:
                valid, msg = self.phase_i_final_validation(
                    config, model, kv_pool, expert_cache, linear_pool, chosen_chunk
                )
            finally:
                # Phase H's pools only prove Phase I's forward fits; engine.py builds
                # the permanent ones from the returned sizes. Free them first or
                # the engine double-commits.
                attach_offload_moe_cache(model, None)
                del kv_pool, expert_cache, linear_pool
                gc.collect()
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            if valid:
                break
            ledger_text = self.ledger(config, chosen_chunk, expert_slots, kv_pages)
            if attempt == _VALIDATION_RETRIES:
                raise RuntimeError(
                    f"Final validation failed against the solved ledger "
                    f"({ledger_text}, budget={budget}): {msg}"
                )
            shrink = max(_VALIDATION_SHRINK_BYTES, budget // 20)
            logger.warning(
                f"Final validation OOM with an unpriced owner ({msg.splitlines()[0][:160]}); "
                f"re-solving with {mem_GB(shrink)} less budget (attempt {attempt + 2})"
            )
            budget -= shrink

        ledger = self.ledger(config, chosen_chunk, expert_slots, kv_pages)
        final_snapshot = take_physical_snapshot(self.device)
        self.final_plan = PlanCandidate(
            prefill_chunk=chosen_chunk,
            expert_slots=expert_slots,
            kv_pages=kv_pages,
            kv_tokens=kv_pages * self.static_model.page_tokens,
            expert_bytes=ledger["experts"],
            kv_bytes=ledger["kv"],
            fixed_overhead=sum(ledger.values())
            - ledger["experts"]
            - ledger["kv"]
            - ledger["transient"],
            transient_reserve=ledger["transient"],
            total_committed=sum(ledger.values()),
            physical_free_after=final_snapshot.driver_free,
            validation_passed=True,
            prefill_overlap=prefill_overlap,
        )

        logger.info_rank0("=" * 60)
        logger.info_rank0(f"PLANNING COMPLETE: {self.final_plan}")
        logger.info_rank0("=" * 60)

        return self.final_plan


_VALIDATION_RETRIES = 4
_VALIDATION_SHRINK_BYTES = 512 << 20


def create_memory_planner(
    config: EngineConfig,
    device: torch.device,
    model_config,
    dtype: torch.dtype,
    pool_cls: type[BaseKVCachePool],
    banks,
    expert_auxiliary_bytes: int = 0,
    quantization_side_tables: int = 0,
    staging_buffers: int = 0,
    method=None,
) -> MemoryPlanner:
    """Factory function for MemoryPlanner."""
    return MemoryPlanner(
        config=config,
        device=device,
        model_config=model_config,
        dtype=dtype,
        pool_cls=pool_cls,
        banks=banks,
        expert_auxiliary_bytes=expert_auxiliary_bytes,
        quantization_side_tables=quantization_side_tables,
        staging_buffers=staging_buffers,
        method=method,
    )
