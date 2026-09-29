from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat
from freetoken.moe.trace import MoeTracer, _capturing

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = int(os.getenv("FREETOKEN_SMALL_BANK_FEAT_BYTES", str(64 * 1024)))

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native GGUF experts of any MMVQ-capable ggml type (the generalization of "q4_0"):
    # gate_up [L*E, 2I, row_bytes(H, t_gate_up)], down [L*E, H, row_bytes(I, t_down)],
    # with the two types carried on ModelConfig.gguf_expert_types because they are a
    # property of the checkpoint, not of the format. Same two banks and the same
    # dequant-in-kernel grouped GEMV as q4_0 -- only the row stride is type-dependent.
    "gguf": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: (
        3 * I * H
        + (
            (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
            + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
        )
        * 2
    ),
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


# Most tokens a single-request prefill-phase batch may carry and still take the MoE decode
# path (speculative verify/replay windows); see layers/moe.py ``_use_decode_path``.
DECODE_PATH_MAX_TOKENS = 8


def decode_pool_floor(num_experts: int, top_k: int, max_running_req: int) -> int:
    """Most distinct experts one decode step can route: every geometry pool's LRU floor."""
    return min(num_experts, max(DECODE_PATH_MAX_TOKENS, max_running_req) * top_k)


@dataclass
class ExpertBank:
    weight: torch.Tensor
    quant_type: str
    shape: tuple[int, ...]


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # For quant_format == "gguf": the (gate_up, down) ggml types of the two banks.
    # Can be a single tuple (uniform) or a list of tuples per layer (Phase 7 exact geometry).
    gguf_expert_types: tuple[int, int] | list[tuple[int, int]] | None = None
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = int(os.getenv("FREETOKEN_HYBRID_MAX_FETCH", "4"))
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = float(os.getenv("FREETOKEN_HYBRID_FETCH_FRACTION", "0.0"))
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None
    # Mixed-geometry banks only: most distinct experts one decode step may route (decode-path
    # tokens * top_k), the floor of every geometry pool's LRU range; 0 = one full layer.
    min_pool_rows: int = 0
    # Mixed-geometry banks only: per-pool slot caps replacing the uniform layer-count split
    # (``pool_capacities``), in ``expert_pools`` order (largest layer group first). "" = off.
    pool_caps_override: str = ""
    # Resizable residency (CUDA VMM): bookkeeping, views and captured graphs are shaped for
    # ``vmm_rows`` slots while only the ``cache_size`` live prefix of each pool is physically
    # backed; :meth:`set_live` moves that boundary in place. 0 = fixed allocation.
    vmm_rows: int = 0
    # Prefill-eligible layer boundary: layers >= prefill_moe_layers are decode-only (e.g. MTP draft heads)
    # and never stage or materialize prefill rows. None = all num_layers run prefill.
    prefill_moe_layers: int | None = None

    def __post_init__(self) -> None:
        if self.prefill_moe_layers is None:
            self.prefill_moe_layers = self.num_layers
        if isinstance(self.gguf_expert_types, dict):
            try:
                self.gguf_expert_types = list(
                    zip(
                        self.gguf_expert_types["gate_up"],
                        self.gguf_expert_types["down"],
                        strict=True,
                    )
                )
            except KeyError as exc:
                raise ValueError("GGUF expert types require gate_up and down") from exc
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-expert LRU-3 reference history (g1, g2, g3 = the three most recent reference
        # steps), indexed by the flat layer * num_experts + expert id. Read by the
        # FREETOKEN_MOE_EVICT=lru3 victim-key path; the history is per EXPERT, so it
        # survives eviction (ghost) and cache_size rebuilds only clear it with the clock.
        self.ghost_hist = torch.zeros(
            (3, self.num_layers * self.num_experts), dtype=torch.int64, device=self.device
        )
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layout is not None:
            self.bank_schema = tuple(
                role for role, spec in self.layout.items() if not spec.resident
            )
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[Any, torch.Tensor] = {}
        # Phase 7 exact geometry keying: (bank_idx, role, quant_type) -> (shape, dtype)
        self.expert_geometry: dict[tuple[int, str, str], tuple[tuple[int, ...], torch.dtype]] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Geometry pools (see _alloc_bank_caches); until banks register, one pool = all slots.
        self.pools: list = []
        self.pool_caps = [self.cache_size]
        self.pool_of_layer = [0] * self.num_layers
        self._pool_starts = [0]
        self._pool_views: list[tuple[torch.Tensor, ...]] = []
        self._staging: dict[int, tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}
        self._arenas: list[torch.Tensor] = []
        self._vmm_arenas: list = []
        self._backing_cost_curve: tuple[int, ...] | None = None
        self.live_caps: list[int] = []
        self._staged: set[int] = set()
        self._bind_pool_state()
        # Fused multi-bank copy descriptors (built by set_bank_sources/_build_copy_plan).
        # Source, destination and row-byte tensors are per layer because mixed-geometry
        # pools give each layer its own destination view.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_dst_ptrs_by_layer: list[torch.Tensor] | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        self._copy_feat_bytes_by_layer: list[torch.Tensor] | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        # Decode hit/miss gather overlap stream state (lazily initialized on first use)
        self.decode_copy_stream: torch.cuda.Stream | None = None
        self.decode_begin_event: torch.cuda.Event | None = None
        self.decode_ready_event: torch.cuda.Event | None = None
        self.tracer = MoeTracer.from_env()
        self._trace_kind = "decode"
        self._trace_ids: list[int] | None = None
        self._trace_pool_id: int | None = None
        self._trace_evicted_ids: list[int] = []
        self._trace_resident_rows = 0

    def _get_layer_quant_type(self, layer_id: int, role: str) -> str:
        if self.quant_format == "gguf" and self.gguf_expert_types is not None:
            from freetoken.models.gguf.dequant import GGML_NAME

            types = self.gguf_expert_types
            if isinstance(types, list) or (
                isinstance(types, tuple) and isinstance(types[0], (tuple, list))
            ):
                t_gate_up, t_down = types[layer_id]
            else:
                t_gate_up, t_down = types
            qt = t_gate_up if "gate" in role or "up" in role else t_down
            return GGML_NAME.get(qt, str(qt))
        return self.quant_format.upper()

    def get_expert(self, bank: int, role: str, quant_type: str | int) -> ExpertBank:
        from freetoken.moe.expert_banks import validate_pool_key

        key = validate_pool_key((bank, role, quant_type))
        if key not in self.expert_geometry:
            raise KeyError(
                f"expert_geometry / geometria inválida: {key!r}. "
                f"MoE pool key mismatch: expected (bank, role, type)"
            )
        bank_idx, role_name, qt = key
        source = self.bank_sources[role_name]
        weight = source[bank_idx]
        return ExpertBank(weight=weight, quant_type=qt, shape=tuple(weight.shape))

    def put_expert(
        self,
        bank: int,
        role: str,
        quant_type: str | int,
        bank_data: torch.Tensor | ExpertBank,
    ) -> None:
        from freetoken.moe.expert_banks import validate_pool_key

        key = validate_pool_key((bank, role, quant_type))
        bank_idx, role_name, qt = key
        tensor = bank_data.weight if isinstance(bank_data, ExpertBank) else bank_data
        self.bank_sources[role_name][bank_idx] = tensor
        self.expert_geometry[key] = (tuple(tensor.shape), tensor.dtype)

    def prefetch_next(self, req: int | None = None) -> None:
        """Prefetch next layer buffer for prefill double-buffering."""
        if req is not None and self.prefill_overlap:
            self.prefetch_prefill_layer(req)

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        self._prepare_bank_sources(sources, layer_residency)
        self._allocate_prepared_bank_sources()

    def _prepare_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Validate and attach source bank geometry without allocating device arenas.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes). The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(i for i, r in enumerate(residency) if r != HostResidency.PINNED.value)
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        self.expert_geometry = {}
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            if self.layout is not None:
                spec = self.layout[name]
                if tuple(head.shape[1:]) != tuple(spec.shape) or head.dtype != spec.dtype:
                    raise ValueError(
                        f"bank {name!r} rows are {tuple(head.shape[1:])} {head.dtype} but the expert kernel's layout "
                        f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                    )
            is_uniform = all(s.shape == head.shape and s.dtype == head.dtype for s in per_layer)
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                if is_uniform:
                    assert source.shape == head.shape and source.dtype == head.dtype, (
                        name,
                        layer_id,
                        source.shape,
                        source.dtype,
                    )
                qt = self._get_layer_quant_type(layer_id, name)
                key = (layer_id, name, qt)
                self.expert_geometry[key] = (tuple(source.shape), source.dtype)
            self.bank_sources[name] = list(per_layer)

    def _allocate_prepared_bank_sources(self, vmm_plan=None) -> None:
        size = self._alloc_bank_caches(self.cache_size, precomputed_vmm_plan=vmm_plan)
        if size != self.cache_size:
            self._alloc_slot_state(size)
        self._bind_pool_state()
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()
        if self.tracer is not None:
            self.tracer.write_snapshot(self)

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            return
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            feat = math.prod(cache.shape[1:]) * cache.element_size()
            if feat % 128:
                raise RuntimeError(
                    f"MoE bank {name!r} rows are {feat} bytes (not a multiple of 128): "
                    f"only the fused multi-bank copy can move them, but it is disabled"
                )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_dst_ptrs_by_layer = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_feat_bytes_by_layer = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        layer_dst_ptrs = [[] for _ in range(self.num_layers)]
        layer_feat_bytes = [[] for _ in range(self.num_layers)]
        for bank_idx, (per_layer, _cache) in enumerate(self.banks):
            for layer_id, source in enumerate(per_layer):
                feat = math.prod(source.shape[1:]) * source.element_size()
                dst = self._layer_rows(layer_id, bank_idx, False)
                if feat % 16 != 0 or dst.data_ptr() % 16 != 0:
                    return  # leave fused disabled; copy_missing uses the per-bank path
                layer_dst_ptrs[layer_id].append(dst.data_ptr())
                layer_feat_bytes[layer_id].append(feat)
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
        self._copy_dst_ptrs_by_layer = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device) for ptrs in layer_dst_ptrs
        ]
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device) for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes_by_layer = [
            torch.tensor(feats, dtype=torch.int64, device=self.device) for feats in layer_feat_bytes
        ]
        # Keep the first-layer aliases for the pre-existing prefill overlap path, which is
        # disabled by the runtime when layer geometry is non-uniform.
        self._copy_dst_ptrs = self._copy_dst_ptrs_by_layer[0]
        self._copy_feat_bytes = self._copy_feat_bytes_by_layer[0]
        dst_ptrs = layer_dst_ptrs[0]
        feats = layer_feat_bytes[0]
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.max_slots is not None and cache_size > self.max_slots:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if (
            self.layout is None
            and self.quant_format == "nvfp4_marlin"
            and cache_size > MARLIN_MAX_CACHE_SIZE
        ):
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )
        if getattr(self, "bank_sources", None):  # unset while __post_init__ validates
            self._pool_plan(cache_size)

    def _alloc_slot_state(self, cache_size: int) -> None:
        """(Re)allocate the ``cache_size``-shaped LRU bookkeeping, all slots empty."""
        self.cache_size = cache_size
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)

    def _is_prefill_pool(self, pool) -> bool:
        return any(layer < self.prefill_moe_layers for layer in pool.layers)

    def _alloc_bank_caches(self, cache_size: int, precomputed_vmm_plan=None) -> int:
        """Allocate the GPU slot cache as one byte arena per bank, split into geometry pools.

        Layers with the same row geometry in every bank share a pool: a contiguous range of
        the global slot index space (its own LRU) and one aligned byte range per arena. The
        slot ids a layer's ``slot_for_id`` holds are local to its pool, so the kernels index
        the pool's views directly. One geometry = one pool of ``cache_size`` rows (the
        historic layout); several split ``cache_size`` rows by ``pool_capacities``, so a
        one-layer pool (an MTP draft's own bank) never costs rows sized for every layer.
        A pool smaller than one layer stages a prefill layer at the front of each arena
        (``_staging``), invalidating whatever resident rows it overlays. Returns the total
        resident rows allocated (``sum`` of the pool capacities)."""
        from freetoken.utils import div_ceil

        banks = {n: self.bank_sources[n] for n in self.bank_schema}
        self._release_vmm()
        plan = precomputed_vmm_plan
        if self.vmm_rows and plan is None:
            try:
                plan = self._vmm_plan(cache_size)
            except ValueError as e:
                logger.warning(f"expert cache stays fixed-size (no in-place residency): {e}")
                from freetoken.tuning import diagnostics

                diagnostics.log_event(
                    "fallback",
                    "expert_residency",
                    f"expert pool fell back to fixed-size full-back (VMM lazy plan rejected): {e}",
                    severity="warn",
                )
                self.vmm_rows = 0
        if plan is not None:
            pools, caps, offsets, ends, live = plan
            arenas = self._alloc_vmm_arenas(pools, caps, offsets, ends, live)
        else:
            pools, caps, offsets, ends = self._pool_plan(cache_size)
            arenas = [torch.empty((end,), dtype=torch.uint8, device=self.device) for end in ends]
            live = list(caps)
            self._staged = {
                p
                for p, c in enumerate(caps)
                if self._is_prefill_pool(pools[p]) and c < self.num_experts
            }
        self.live_caps = live
        starts = [sum(caps[:i]) for i in range(len(caps))]

        def view(arena: torch.Tensor, off: int, rows: int, head: torch.Tensor) -> torch.Tensor:
            rb = head[0].numel() * head.element_size()
            return arena[off : off + rows * rb].view(head.dtype).view(rows, *head.shape[1:])

        E = self.num_experts
        self.bank_caches = {}
        self.pool_caps = caps
        self.pool_of_layer = [0] * self.num_layers
        self._pool_views: list[tuple[torch.Tensor, ...]] = []
        self._staging: dict[int, tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}
        for p, (pool, cap) in enumerate(zip(pools, caps)):
            heads = [banks[n][pool.layers[0]] for n in self.bank_schema]
            views = tuple(view(a, off, cap, h) for a, off, h in zip(arenas, offsets[p], heads))
            self._pool_views.append(views)
            for layer_id in pool.layers:
                self.pool_of_layer[layer_id] = p
                for name, v in zip(self.bank_schema, views):
                    self.bank_caches[
                        (layer_id, name, self._get_layer_quant_type(layer_id, name))
                    ] = v
            if p in self._staged:
                window = [E * rb for rb in pool.row_bytes]  # staging bytes per arena
                overlaid = [
                    i
                    for q, (q_pool, q_cap) in enumerate(zip(pools, caps))
                    for i in range(
                        starts[q],
                        starts[q]
                        + min(
                            q_cap,
                            max(
                                div_ceil(max(0, w - off), rb)
                                for w, off, rb in zip(window, offsets[q], q_pool.row_bytes)
                            ),
                        ),
                    )
                ]
                self._staging[p] = (
                    tuple(view(a, 0, E, h) for a, h in zip(arenas, heads)),
                    torch.tensor(overlaid, dtype=torch.int64, device=self.device),
                )
        for name, v in zip(self.bank_schema, self._pool_views[0]):
            self.bank_caches[name] = v
        self.pools = pools
        self._pool_starts = starts
        self._arenas = arenas
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        if self._vmm_arenas:
            granule = self._vmm_arenas[0].g
            self._backing_cost_curve = tuple(
                self._backed_bytes_for_live(self._live_caps_for(pools, caps, size), granule)
                for size in range(cache_size + 1)
            )
        return sum(caps)

    def _override_caps(self, pools, cache_size: int) -> list[int]:
        """The ``--moe-pool-caps`` split of ``cache_size`` rows.

        The override names the SHAPE of the split (its entries are relative weights);
        ``cache_size`` stays the byte-budget authority, so the caps always sum to exactly
        ``cache_size`` (clamped to the pools' total row domain) - at build, where the
        planner sized the budget, and at every rebuild target, so a guard shrink/regrow
        keeps the operator's proportions instead of silently reverting to the uniform
        split. Largest-remainder rounding clamped to each pool's ``[floor, rows]``
        domain; deterministic for a given input.
        """
        want = [int(x) for x in self.pool_caps_override.split(",") if x.strip()]
        if len(want) != len(pools):
            raise ValueError(
                f"moe_pool_caps has {len(want)} entries but the banks have {len(pools)} "
                f"geometry pools (order: largest layer group first)"
            )
        if any(c < 0 for c in want) or sum(want) == 0:
            raise ValueError(f"moe_pool_caps entries must be >= 0 and sum > 0, got {want}")
        E = self.num_experts
        hi = [len(p.layers) * E for p in pools]
        lo = [min(h, self.min_pool_rows or E) for h in hi]
        for c, h in zip(want, hi):
            if c > h:
                raise ValueError(f"moe_pool_caps entry {c} exceeds its pool's {h} rows")
        total = sum(want)
        cache_size = min(cache_size, sum(hi))
        if cache_size < sum(lo):
            raise ValueError(
                f"moe_pool_caps cannot split {cache_size} rows: every pool needs its decode "
                f"floor (sum {sum(lo)})"
            )
        base = [max(l, min(h, w * cache_size // total)) for w, l, h in zip(want, lo, hi)]
        rem = cache_size - sum(base)
        order = sorted(range(len(want)), key=lambda i: (-(want[i] * cache_size % total), i))
        while rem > 0:
            moved = False
            for i in order:
                if rem == 0:
                    break
                if base[i] < hi[i]:
                    base[i] += 1
                    rem -= 1
                    moved = True
            if not moved:  # pragma: no cover - guarded by the sum(hi) clamp above
                raise ValueError(f"moe_pool_caps cannot place {cache_size} rows under {hi}")
        while rem < 0:
            moved = False
            for i in order:
                if rem == 0:
                    break
                if base[i] > lo[i]:
                    base[i] -= 1
                    rem += 1
                    moved = True
            if not moved:  # pragma: no cover - guarded by the sum(lo) check above
                raise ValueError(f"moe_pool_caps cannot shrink to {cache_size} rows above {lo}")
        return base

    def _vmm_plan(self, live_size: int):
        """Shape caps (``vmm_rows``), per-pool granule-aligned offsets and the live caps for
        ``live_size``; every pool gets its own virtual region so each can grow in place."""
        from freetoken.engine.cache_budget import expert_pools
        from freetoken.moe import vmm

        E = self.num_experts
        pools = expert_pools({n: self.bank_sources[n] for n in self.bank_schema})
        self.vmm_rows = max(self.vmm_rows, live_size)
        caps = self._caps_for(pools, self.vmm_rows)
        # a pool whose planned (prefill-phase) share is below one layer stages prefill layers,
        # whatever its decode-phase shape: that keeps the planned rows the whole prefill needs
        planned = self._caps_for(pools, min(live_size, sum(caps)))
        self._staged = {
            p
            for p, (c, cap) in enumerate(zip(planned, caps))
            if self._is_prefill_pool(pools[p]) and min(c, cap) < E
        }
        live = self._live_caps_for(pools, caps, live_size)
        g = vmm.granularity(self.device.index or 0)
        offsets, ends = [], [0] * len(self.bank_schema)
        for pool, cap in zip(pools, caps):
            offsets.append(list(ends))
            for b, rb in enumerate(pool.row_bytes):
                ends[b] += -(-cap * rb // g) * g
        return pools, caps, offsets, ends, live

    def _caps_for(self, pools, cache_size: int) -> list[int]:
        from freetoken.engine.cache_budget import pool_capacities

        if self.pool_caps_override and len(pools) > 1:
            return self._override_caps(pools, cache_size)
        if len(pools) == 1:
            return [cache_size]
        return pool_capacities(pools, self.num_experts, cache_size, self.min_pool_rows)

    def _live_caps_for(self, pools, shape_caps: list[int], size: int) -> list[int]:
        """Live rows per pool for ``size``: the usual split, inside each pool's shape and above
        the rows a prefill always writes -- a pool shaped for a whole layer materializes it at
        its front, a sub-layer pool stages one at the front of every arena (pool 0's region),
        and the prefill double buffers own the first ``2 * E`` rows."""
        E = self.num_experts
        caps = self._caps_for(pools, min(size, sum(shape_caps)))
        live = []
        floors = []
        for p, (c, cap) in enumerate(zip(caps, shape_caps)):
            lo = min(cap, self.min_pool_rows or E)
            if self._is_prefill_pool(pools[p]) and p not in self._staged:
                lo = max(lo, min(cap, E))
            if self.prefill_overlap and self._is_prefill_pool(pools[p]):
                lo = max(lo, 2 * E)
            if p == 0:
                for q, pool in enumerate(pools):
                    if q in self._staged:
                        for rb0, rb in zip(pools[0].row_bytes, pool.row_bytes):
                            lo = max(lo, -(-E * rb // rb0))
            if lo > cap:
                raise ValueError(f"pool {p} shape {cap} rows cannot hold its prefill front {lo}")
            floors.append(lo)
            live.append(min(cap, max(lo, c)))
        # Floors are part of the requested budget, not extra rows added after splitting it.
        excess = sum(live) - max(size, sum(floors))
        while excess > 0:
            movable = [p for p, (n, lo) in enumerate(zip(live, floors)) if n > lo]
            share = max(1, excess // len(movable))
            for p in movable:
                take = min(share, live[p] - floors[p], excess)
                live[p] -= take
                excess -= take
        return live

    def _effective_live_caps_for(self, size: int) -> list[int]:
        """Return the live pool caps ``set_live`` can apply at ``size`` rows.

        Shrinks clamp each pool to its current cap so largest-remainder rounding can never
        turn a physical shrink into a per-pool grow.
        """
        live = self._live_caps_for(self.pools, self.pool_caps, size)
        if size < sum(self.live_caps):
            live = [min(n, old) for n, old in zip(live, self.live_caps)]
        return live

    def backed_bytes_for(self, size: int) -> int:
        """Exact VMM backing bytes for the live pool geometry at ``size`` rows.

        Match ``VirtualArena.set_backed``: each pool/bank region rounds independently to
        the native CUDA granule. This prices the post-floor live split, not an average
        byte price per logical row.
        """
        if not self._vmm_arenas:
            raise RuntimeError("exact backing price requires VMM residency")
        granule = self._vmm_arenas[0].g
        if self._backing_cost_curve is not None and sum(self.live_caps) <= size < len(
            self._backing_cost_curve
        ):
            return self._backing_cost_curve[size]
        live = self._effective_live_caps_for(size)
        return self._backed_bytes_for_live(live, granule)

    def _backed_bytes_for_live(self, live: list[int], granule: int) -> int:
        """Exact bank-granule price for an already-computed live split."""
        return sum(
            -(-(rows * row_bytes) // granule) * granule
            for rows, pool in zip(live, self.pools)
            for row_bytes in pool.row_bytes
        )

    def _alloc_vmm_arenas(self, pools, caps, offsets, ends, live) -> list[torch.Tensor]:
        from freetoken.moe import vmm

        self._vmm_arenas = []
        try:
            for b in range(len(self.bank_schema)):
                regions = [
                    (
                        offsets[p][b],
                        (offsets[p + 1][b] if p + 1 < len(pools) else ends[b]) - offsets[p][b],
                    )
                    for p in range(len(pools))
                ]
                arena = vmm.VirtualArena(self.device, ends[b], regions)
                self._vmm_arenas.append(arena)
                for p, pool in enumerate(pools):
                    arena.set_backed(p, live[p] * pool.row_bytes[b])
            return [a.tensor for a in self._vmm_arenas]
        except Exception:
            self._release_vmm()
            raise

    def _release_vmm(self) -> None:
        self._backing_cost_curve = None
        if self._vmm_arenas:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            for arena in self._vmm_arenas:
                arena.release()
            self._vmm_arenas = []

    # usage of a slot with no physical backing: above every real LRU step, and small enough
    # that flashlib's packed ``usage << SLOT_BITS | slot`` key cannot overflow
    _BLOCKED_USAGE = 1 << 40

    @property
    def resident_rows(self) -> int:
        """Slots physically backed (== ``cache_size`` unless VMM residency is on)."""
        return sum(self.live_caps) if self._vmm_arenas else self.cache_size

    def _block_unbacked(self) -> None:
        """Park every slot past a pool's live prefix: no id, never an LRU victim."""
        if not self._vmm_arenas:
            return
        for (ids, usage), live in zip(self._pool_state, self.live_caps):
            ids[live:].fill_(-1)
            usage[live:].fill_(self._BLOCKED_USAGE)

    @torch.inference_mode()
    def set_live(self, size: int) -> int:
        """Move the backed boundary to ``size`` rows in place (VMM only); returns the rows now
        live. Shrinking forgets the experts held past the new boundary, then unmaps; growing
        maps first, then frees the new slots to the LRU. Slots inside both prefixes keep their
        experts, and no address or shape changes, so captured graphs stay valid. The caller
        guarantees no forward is in flight."""
        assert self._vmm_arenas, "set_live needs VMM residency"
        new = self._effective_live_caps_for(size)
        flat = self.slot_for_id.view(-1)
        for p, ((ids, usage), old, nxt) in enumerate(zip(self._pool_state, self.live_caps, new)):
            if nxt < old:
                held = ids[nxt:old]
                gone = held[held >= 0].to(torch.int64)
                flat[gone] = -1
                held.fill_(-1)
                usage[nxt:old].fill_(self._BLOCKED_USAGE)
        torch.cuda.synchronize(self.device)
        # rows whose bytes survived the re-map, per pool (min over banks)
        intact = list(new)
        # Free every shrinking region before mapping any growing one. Largest-remainder
        # redistribution can lower one pool's cap as total logical rows increase; applying
        # that transition shrink-first makes final-byte fit sufficient for the whole move.
        try:
            for growing in (False, True):
                for b, arena in enumerate(self._vmm_arenas):
                    for p, (pool, old, nxt) in enumerate(zip(self.pools, self.live_caps, new)):
                        if (nxt > old) != growing or nxt == old:
                            continue
                        rb = pool.row_bytes[b]
                        intact[p] = min(intact[p], arena.set_backed(p, nxt * rb) // rb)
        except Exception as error:
            from freetoken.moe.vmm import VMMResizeRollbackError

            rollback_error = error if isinstance(error, VMMResizeRollbackError) else None
            preserved = [min(old, nxt) for old, nxt in zip(self.live_caps, new)]
            safe = [min(preserved[p], intact[p]) for p in range(len(new))]
            for b, arena in enumerate(self._vmm_arenas):
                for p, pool in enumerate(self.pools):
                    safe[p] = min(safe[p], arena.backed[p] // pool.row_bytes[b])
            if safe != preserved and rollback_error is None:
                rollback_error = VMMResizeRollbackError(
                    f"VMM resize left backing below the retained prefix: {preserved} -> {safe}"
                )
            for b, arena in enumerate(self._vmm_arenas):
                for p, pool in enumerate(self.pools):
                    target = safe[p] * pool.row_bytes[b]
                    if arena.backed[p] > target:
                        try:
                            arena.set_backed(p, target)
                        except Exception as cleanup_error:
                            if rollback_error is None:
                                rollback_error = VMMResizeRollbackError(
                                    f"VMM resize failed ({error!r}); rollback failed "
                                    f"({cleanup_error!r})"
                                )
            actual = safe.copy()
            for b, arena in enumerate(self._vmm_arenas):
                for p, pool in enumerate(self.pools):
                    actual[p] = min(actual[p], arena.backed[p] // pool.row_bytes[b])
            if actual != preserved and rollback_error is None:
                rollback_error = VMMResizeRollbackError(
                    f"VMM rollback left backing below the retained prefix: {preserved} -> {actual}"
                )
            safe = actual
            for (ids, usage), old, keep in zip(self._pool_state, self.live_caps, safe):
                if keep < old:
                    held = ids[keep:old]
                    gone = held[held >= 0].to(torch.int64)
                    flat[gone] = -1
                    held.fill_(-1)
                    usage[keep:old].fill_(self._BLOCKED_USAGE)
            self.live_caps = safe
            self._block_unbacked()
            if rollback_error is not None:
                if rollback_error is error:
                    raise
                raise rollback_error from error
            raise
        for (ids, usage), old, nxt, keep in zip(self._pool_state, self.live_caps, new, intact):
            lost = min(keep, old)  # below this row the slot still holds its expert
            if nxt > lost:
                held = ids[lost:nxt]
                gone = held[held >= 0].to(torch.int64)
                flat[gone] = -1
                held.fill_(-1)
                usage[lost:nxt].zero_()
        self.live_caps = new
        return sum(new)

    def _pool_plan(self, cache_size: int):
        """``(pools, capacities, offsets, arena_bytes)`` for ``cache_size`` rows; raises
        ``ValueError`` when a sub-layer pool's prefill staging window exceeds an arena."""
        from freetoken.engine.cache_budget import (
            expert_pools,
            pool_capacities,
            pool_layout,
            pool_staging_fits,
        )

        pools = expert_pools({n: self.bank_sources[n] for n in self.bank_schema})
        if self.pool_caps_override and len(pools) > 1:
            caps = self._override_caps(pools, cache_size)
        else:
            caps = (
                [cache_size]
                if len(pools) == 1
                else pool_capacities(pools, self.num_experts, cache_size, self.min_pool_rows)
            )
        offsets, ends = pool_layout(pools, caps)
        prefill_pools = {p for p, pool in enumerate(pools) if self._is_prefill_pool(pool)}
        if not pool_staging_fits(pools, caps, self.num_experts, ends, prefill_pools=prefill_pools):
            raise ValueError(
                f"moe_cache_size={cache_size} is too small for the mixed expert geometry: a "
                f"pool below one layer ({caps}) cannot stage a prefill layer in its arenas"
            )
        return pools, caps, offsets, ends

    def _bind_pool_state(self) -> None:
        """Per-pool ``(id_of_slot, usage)`` views the LRU kernels run on."""
        self._pool_state = [
            (self.id_of_slot[s : s + c], self.usage[s : s + c])
            for s, c in zip(self._pool_starts, self.pool_caps)
        ]
        self._block_unbacked()

    def pool_state(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(id_of_slot, usage)`` of ``layer_id``'s pool; its slot ids index these."""
        return self._pool_state[self.pool_of_layer[layer_id]]

    @property
    def expert_pool_bytes(self) -> int:
        """GPU bytes the slot arenas hold (every pool, alignment padding included)."""
        if self._vmm_arenas:
            return sum(a.backed_bytes for a in self._vmm_arenas)
        return sum(a.nbytes for a in self._arenas)

    @property
    def unbacked_bytes(self) -> int:
        """Virtual arena bytes with no physical memory (views over them are not VRAM)."""
        return sum(a.nbytes - a.backed_bytes for a in self._vmm_arenas)

    def _layer_rows(self, layer_id: int, bank: int, whole_layer: bool) -> torch.Tensor:
        """The GPU rows a copy/GEMM of ``layer_id`` addresses in bank ``bank``: its pool,
        or the staging window when a whole layer does not fit the pool."""
        if not self._pool_views:  # banks attached by hand (benchbw), no pools
            return self.banks[bank][1]
        p = self.pool_of_layer[layer_id]
        if whole_layer and p in self._staging:
            return self._staging[p][0][bank]
        return self._pool_views[p][bank]

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self._arenas = []
        self._pool_views = []
        self._staging = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        cache_size = self._alloc_bank_caches(cache_size)
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self._alloc_slot_state(cache_size)
        self._bind_pool_state()
        self.step.zero_()
        self.ghost_hist.zero_()  # the LRU-3 history lives on the step clock: rebasing kills it
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        ids = self.pool_state(layer_id)[0]
        idx = layer_id * self.num_experts + (ids.clamp(min=0).long() % self.num_experts)
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def get_decode_copy_stream(
        self,
    ) -> tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]:
        """Lazy init of the dedicated copy stream + sync events for decode hit/miss overlap."""
        if self.decode_copy_stream is None:
            self.decode_copy_stream = torch.cuda.Stream(device=self.device)
            self.decode_begin_event = torch.cuda.Event()
            self.decode_ready_event = torch.cuda.Event()
        assert self.decode_begin_event is not None and self.decode_ready_event is not None
        return self.decode_copy_stream, self.decode_begin_event, self.decode_ready_event
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        ids = self.pool_state(layer_id)[0]
        idx = layer_id * self.num_experts + (ids.clamp(min=0).long() % self.num_experts)
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(
        self, n: int | None = None, layer_id: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the layer's whole geometry pool
        (decode; its slot ids are pool-local), or its first ``n`` rows (materialized layer;
        the staging window when the pool holds fewer than ``n``)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if layer_id is None:
            layer_id = self._pending_src_layer if self._pending_src_layer is not None else 0
        p = self.pool_of_layer[layer_id] if self._staging else None
        if n is not None and p in self._staging and n > self.live_caps[p]:
            return tuple(v[:n] for v in self._staging[p][0])
        views = []
        for name in self.bank_schema:
            qt = self._get_layer_quant_type(layer_id, name)
            key = (layer_id, name, qt)
            c = self.bank_caches.get(key, self.bank_caches.get(name))
            views.append(c if n is None else c[:n])
        return tuple(views)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # Check if bank sources have non-uniform geometry across layers
        self._is_uniform_geometry = all(
            all(s.shape == per_layer[0].shape and s.dtype == per_layer[0].dtype for s in per_layer)
            for per_layer in self.bank_sources.values()
        )
        if not self._is_uniform_geometry:
            # When geometry varies across layers (e.g. mixed Q4_K/Q3_K/Q8_0 GGUF MoE),
            # fixed-shape prefill overlap buffers cannot hold all layer geometries.
            # Disable prefill overlap and fall back to synchronous per-layer materialize.
            logger.info(
                "MoE banks have non-uniform layer geometry; disabling prefill overlap "
                "(falling back to per-layer materialize)"
            )
            self.prefill_overlap = False
            self.prefill_bank_buffers = []
            return

        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(
        self, layer_id: int, expert_ids: torch.Tensor, *, kind: str = "decode"
    ) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        trace_active = self.tracer is not None and not _capturing()
        if trace_active:
            self._trace_kind = kind
            self._trace_ids = [int(i) for i in expert_ids.reshape(-1).tolist()]
            self._trace_pool_id = self.pool_of_layer[layer_id]
            pool_ids, pool_usage = self.pool_state(layer_id)
            before = [int(i) for i in pool_ids.tolist()]
            before_usage = [int(i) for i in pool_usage.tolist()]
            self._trace_before_ids = before
            self._trace_before_usage = before_usage
            if not self.tracer.initial_written:
                self.tracer.write_initial_residency(
                    [int(i) for i in self.id_of_slot.tolist()],
                    [int(i) for i in self.usage.tolist()],
                )
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)
        if trace_active:
            pool_ids, pool_usage = self.pool_state(layer_id)
            after = [int(i) for i in pool_ids.tolist()]
            after_usage = [int(i) for i in pool_usage.tolist()]
            after_set = {i for i in after if i >= 0}
            self._trace_evicted_ids = sorted({i for i in before if i >= 0} - after_set)
            self._trace_resident_rows = len(after_set)
            self._trace_after_ids = after
            self._trace_after_usage = after_usage

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        self._block_unbacked()
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, (
            f"layer_id {layer_id} out of range [0, {self.num_layers})"
        )
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append(
                {
                    "layer": L,
                    "steps": s,
                    "active_per_step": (a / s) if s else 0.0,
                    "missing_per_step": (m / s) if s else 0.0,
                    "miss_rate": (m / a) if a else 0.0,
                    "fetched_per_step": (f / s) if s else 0.0,
                }
            )
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        trace_active = self.tracer is not None and self._trace_ids is not None and not _capturing()
        trace_start = time.perf_counter() if trace_active and self.device.type != "cuda" else None
        trace_event = None
        if trace_active and self.device.type == "cuda":
            trace_event = torch.cuda.Event(enable_timing=True)
            trace_event.record()
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            for i, name in enumerate(self.bank_schema):
                cache = self._layer_rows(layer_id, i, whole_layer=True)
                cache[: self.num_experts].copy_(self.bank_sources[name][layer_id])
            self._trace_copy_missing(layer_id, trace_event, trace_start)
            return
        # Whole-layer prefill may target a staging window whose base differs from the
        # decode pool view; keep the proven per-bank path for that one-shot copy.
        if self._copy_fused_ok and not self._pending_whole_layer:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            fast_index_copy_multi_jit(
                self._copy_dst_ptrs_by_layer[layer_id],
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes_by_layer[layer_id],
                self.evict_slots,
                self.src_indices,
                self.num_indices,
            )
            self._trace_copy_missing(layer_id, trace_event, trace_start)
            return

        from freetoken.kernel import fast_index_copy_jit

        for i, name in enumerate(self.bank_schema):
            source_layer = self.bank_sources[name][layer_id]
            fast_index_copy_jit(
                self._layer_rows(layer_id, i, self._pending_whole_layer),
                self.evict_slots,
                source_layer,
                self.src_indices,
                self.num_indices,
            )

        self._trace_copy_missing(layer_id, trace_event, trace_start)

    def _trace_copy_missing(
        self,
        layer_id: int,
        start_event: torch.cuda.Event | None,
        start_time: float | None,
    ) -> None:
        if self.tracer is None or self._trace_ids is None:
            return
        if start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            transfer_ms = start_event.elapsed_time(end_event)
        else:
            transfer_ms = (
                (time.perf_counter() - start_time) * 1000 if start_time is not None else None
            )
        bank_bytes = {
            name: self.bank_sources[name][layer_id][0].numel()
            * self.bank_sources[name][layer_id][0].element_size()
            for name in self.bank_schema
        }
        missing = int(self.num_indices.item())
        available_vram = (
            torch.cuda.mem_get_info(self.device)[0] if self.device.type == "cuda" else None
        )
        self.tracer.record(
            kind=self._trace_kind,
            layer_id=layer_id,
            pool_id=self._trace_pool_id
            if self._trace_pool_id is not None
            else self.pool_of_layer[layer_id],
            expert_ids=self._trace_ids,
            missing=missing,
            miss_bytes=missing * sum(bank_bytes.values()),
            bank_bytes=bank_bytes,
            evicted_ids=self._trace_evicted_ids,
            resident_rows=self._trace_resident_rows,
            transfer_ms=transfer_ms,
            available_vram_bytes=available_vram,
            requested_global_ids=[layer_id * self.num_experts + e for e in self._trace_ids],
            hit_global_ids=[
                layer_id * self.num_experts + e
                for e in self._trace_ids
                if layer_id * self.num_experts + e in set(getattr(self, "_trace_before_ids", []))
            ],
            miss_global_ids=[
                layer_id * self.num_experts + e
                for e in self._trace_ids
                if layer_id * self.num_experts + e
                not in set(getattr(self, "_trace_before_ids", []))
            ],
            before_ids=getattr(self, "_trace_before_ids", None),
            after_ids=getattr(self, "_trace_after_ids", None),
            before_usage=getattr(self, "_trace_before_usage", None),
            after_usage=getattr(self, "_trace_after_usage", None),
            victim_slots=[
                i
                for i, old in enumerate(self._trace_before_ids)
                if old in set(self._trace_evicted_ids)
            ]
            if hasattr(self, "_trace_before_ids")
            else None,
        )
        self._trace_ids = None


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
