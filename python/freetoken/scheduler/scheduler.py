from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.debug.token_trace import record as trace_token
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .mm import cut_image_spans, plan_mm_batch
from .prefill import ChunkedReq, PrefillManager
from .spec import SchedulerSpecMixin
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)
_HOST_TRACE = os.getenv("FREETOKEN_HOST_PATH_TRACE", "0") == "1"
_PROFILE_DECODE = os.getenv("FREETOKEN_PROFILE_DECODE", "0") == "1"
_host_trace_totals = [0, 0.0, 0.0, 0.0]

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


_runtime_env_mtime = 0.0


def _apply_runtime_env(path: str) -> None:
    """Measurement only (FREETOKEN_RUNTIME_ENV_FILE): re-apply a JSON env map when the file
    changes, so one boot can test many flag settings. null removes a variable. Flags baked into
    CUDA graphs at capture time are not affected; serve with --cuda-graph-max-bs 0 for those."""
    global _runtime_env_mtime
    try:
        mtime = os.stat(path).st_mtime
        if mtime == _runtime_env_mtime:
            return
        import json

        with open(path, encoding="utf-8") as stream:
            values = json.load(stream)
    except (OSError, ValueError):
        return
    _runtime_env_mtime = mtime
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin, SchedulerSpecMixin):
    def _mark_mtp_oom(self, depth: int | None = None) -> None:
        """Make a speculative OOM sticky for this process and discard its warm start."""
        if depth is None:
            depth = getattr(self, "_mtp_cycle_depth", 0)
        if depth > 0:
            limit = max(0, int(depth) - 1)
            if limit == 0 and not getattr(self, "_mtp_oom_at_k1", False):
                # the OOM already grew the learned decode reserve: k1 gets one more chance
                # before speculation is off for the process
                self._mtp_oom_at_k1 = True
                limit = 1
            current = getattr(self, "_mtp_unsafe_max_k", self.spec_mtp)
            self._mtp_unsafe_max_k = min(current, limit)
        self._mtp_profile_invalidated = True
        key = getattr(self, "_mtp_profile_key", None)
        if key is not None:
            try:
                from freetoken.tuning import mtp_profile

                mtp_profile.invalidate(key)
            except Exception as e:  # noqa: BLE001 -- cache invalidation never blocks serving
                logger.info_rank0(f"mtp depth profile invalidation skipped ({e})")
        controllers = [getattr(self, "_mtp_controller", None)] + list(
            getattr(self, "_mtp_controllers", {}).values()
        )
        for controller in controllers:
            if controller is not None:
                controller.limit_depth(getattr(self, "_mtp_unsafe_max_k", 0))
                controller.fallback_to_k0()

    def _load_mtp_depth_profile(self, cap: int) -> int | None:
        """Compute this serve's depth-profile key and load a previously learned optimal depth.

        Returns ``None`` (calibrate this run) when a measurement force-depth is pinned, when no
        profile matches the hardware+model+build+config fingerprint, or on ANY failure -- depth
        profiling is a warm-start optimization and must never block serving. The key is stashed
        on ``self`` so ``_save_mtp_depth_profile`` writes back to the same entry.
        """
        self._mtp_profile_key = None
        if os.getenv("FREETOKEN_MTP_FORCE_DEPTH"):
            return None  # a measurement pin owns the depth; do not read or write the cache
        try:
            from freetoken.tuning import mtp_profile

            eff_ctx = (
                getattr(self.engine, "max_seq_len", None)
                or getattr(self.config, "max_seq_len_override", None)
                or getattr(self.config, "max_seq_len", 0)
            )
            self._mtp_profile_key = mtp_profile.key_from_config(self.config, eff_ctx, cap)
            profiled = mtp_profile.load(self._mtp_profile_key)
            unsafe = getattr(self, "_mtp_unsafe_max_k", cap)
            return min(profiled, unsafe) if profiled is not None else None
        except Exception as e:  # noqa: BLE001 -- best-effort; calibrate this run on any failure
            logger.info_rank0(f"mtp depth profile load skipped ({e}); calibrating this run")
            self._mtp_profile_key = None
            return None

    def _save_mtp_depth_profile(self, depth: int) -> None:
        """Persist a freshly-learned depth so the next serve of this fingerprint warm-starts at
        it and skips the calibration probe. Best-effort; never blocks serving."""
        if self._mtp_profile_key is None or getattr(self, "_mtp_profile_invalidated", False):
            return
        try:
            from freetoken.tuning import mtp_profile

            mtp_profile.save(self._mtp_profile_key, depth)
        except Exception as e:  # noqa: BLE001
            logger.info_rank0(f"mtp depth profile save skipped ({e})")

    def _seed_vram_learned(self) -> None:
        """Warm-start the decode reserve from the persisted VRAM profile and stash the
        fingerprint key on the engine so future OOMs fold back into it. Best-effort."""
        try:
            from freetoken.tuning import vram_profile

            eff_ctx = (
                getattr(self.engine, "max_seq_len", None)
                or getattr(self.config, "max_seq_len_override", None)
                or getattr(self.config, "max_seq_len", 0)
            )
            self.engine._vram_profile_key = vram_profile.compute_key(
                self.config,
                eff_ctx,
                self.config.spec_mtp,
                bool(getattr(self.config, "active_encoders", None)),
                getattr(self.config.model_config, "mtp_layer_id", None) is not None,
            )
            learned = vram_profile.load(self.engine._vram_profile_key)
            if learned > self.engine._decode_reserve_learned:
                self.engine._decode_reserve_learned = learned
        except Exception as e:  # noqa: BLE001 -- warm start must never block serving
            logger.info_rank0(f"vram profile warm start skipped ({e!r})")

    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            )
            or getattr(self.engine.kv_cache, "sliding_window_size", None),
            host_pages=self.engine.host_pages,
        )
        self.cache_manager.recover_oom = self._recover_cache_oom
        self.decode_manager = DecodeManager(config.page_size)
        self._bidirectional_mm = any(
            getattr(g, "bidirectional_mm_blocks", False)
            for g in config.model_config.attention_groups
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self._bidirectional_mm,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._seed_vram_learned()  # reads self.config: an earlier call silently never keyed
        self.spec_mtp = config.spec_mtp
        self._mtp_controller = None
        self._mtp_profile_key: str | None = None
        if self.spec_mtp > 0 and os.getenv("FREETOKEN_MTP_FIXED_DEPTH", "0") != "1":
            from .adaptive_mtp import AdaptiveMtpController

            # Depth ceiling is the configured --spec-mtp itself (the single-head
            # tuning default stays at 4 by choice, not by clamp); verify/draft graphs
            # and mmvq row shapes scale with it in engine._maybe_enable_spec_mtp.
            cap = self.spec_mtp
            # Warm start from a previously learned depth (same hardware+model+build+config
            # fingerprint) so this serve skips the calibration probe; None -> calibrate.
            profiled = self._load_mtp_depth_profile(cap)
            self._mtp_controller = AdaptiveMtpController(cap, profiled_depth=profiled)
            self._mtp_profiled_depth = profiled
        self._spec_snapshot_slots: dict[int, int] = {}
        if self.spec_mtp > 0:
            if config.max_running_req != 1:
                from freetoken.scheduler.spec import (
                    MTP_BATCHED_ENV,
                    MTP_ROTATE_ENV,
                    batched_spec_enabled,
                    rotate_spec_enabled,
                )

                if not rotate_spec_enabled() and not batched_spec_enabled():
                    raise ValueError(
                        "--spec-mtp > 0 supports single-request serving only for now "
                        f"(EXPERIMENTAL opt-ins: {MTP_ROTATE_ENV}=1 rotates cycles across "
                        f"streams; {MTP_BATCHED_ENV}=1 verifies all streams in one window)."
                    )
            if getattr(self.engine.model, "mtp", None) is None:
                raise ValueError(
                    "--spec-mtp > 0 requires a checkpoint with a registered MTP layer."
                )
        self._model_is_mrope = config.model_config.model_is_mrope
        self._warned_cut_image = False
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)
        import gc

        gc.collect()
        # the startup heap (weights metadata, banks, tokenizer) is permanent: freeze it so a
        # runtime rebuild's gc.collect() scans only what was created since, not the whole heap
        gc.freeze()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        guard = getattr(getattr(self, "engine", None), "guard_vram_at_idle", None)
        if guard is not None:
            guard()
        if hasattr(self, "_spec_snapshot_slots"):
            assert len(self._spec_snapshot_slots) == 0, (
                f"leaked spec snapshot slots in idle: {self._spec_snapshot_slots}"
            )
        engine = getattr(self, "engine", None)
        cache = getattr(engine, "moe_offload_cache", None)
        if (
            getattr(getattr(engine, "config", None), "moe_collect_stats", False)
            and cache is not None
        ):
            logger.info_rank0(
                f"[moe-idle] totals={cache.decode_miss_stats()} "
                f"per_layer={cache.decode_miss_stats_per_layer()}"
            )
            logger.info_rank0(
                "[moe-idle] row_bytes="
                + str([sum(cache.pools[p].row_bytes) for p in cache.pool_of_layer])
            )
        if (
            _PROFILE_DECODE
            and getattr(self, "_profile_started", False)
            and not getattr(self, "_profile_stopped", False)
        ):
            torch.cuda.synchronize(self.device)
            # Nsight may terminate the process at this boundary. Emit final counters first.
            torch.cuda.profiler.stop()
            self._profile_stopped = True

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size,
            num_pages=num_pages,
            num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap
            else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if (
            self._pending_rebuild is not None
            and last_data is None
            and not (self.prefill_manager.runnable or self.decode_manager.runnable)
        ):
            self._execute_pending_rebuild()
        last_data = self._switch_residency_if_due(last_data)
        self._guard_vram_pressure()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        self._rebalance_kv_tiers()
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                out = self._forward_or_fail(forward_input)
                ongoing_data = (forward_input, out) if out is not None else None

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_oom()
        self._flush_abort_acks()
        return ongoing_data

    def _switch_residency_if_due(self, last_data: ForwardData | None) -> ForwardData | None:
        """Grow the expert cache while only decode runs, fold it back before a prefill. The
        in-flight batch is drained first (one step of overlap lost per transition)."""
        engine = getattr(self, "engine", None)
        if getattr(engine, "decode_residency_supported", False) is not True:
            return last_data
        grow = self.decode_manager.runnable and not self.prefill_manager.runnable
        if grow == (engine._expert_decode_slots is not None):
            return last_data
        if last_data is not None:
            self.stream.wait_stream(engine.stream)
            self._process_last_data(last_data)
            last_data = self._last_data = None
            grow = self.decode_manager.runnable and not self.prefill_manager.runnable
        engine.stream.synchronize()
        try:
            engine.set_decode_residency(grow, stream_drained=True)
        except CacheRebuildRejected as e:
            logger.warning(f"decode residency resize refused: {e}")
        return last_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        self._switch_residency_if_due(None)

        if (
            _PROFILE_DECODE
            and not getattr(self, "_profile_started", False)
            and self.decode_manager.running_reqs
        ):
            torch.cuda.profiler.start()
            self._profile_started = True

        if os.environ.get("FREETOKEN_RUNTIME_ENV_FILE"):
            _apply_runtime_env(os.environ["FREETOKEN_RUNTIME_ENV_FILE"])
        mtp_sample = self._begin_mtp_cycle() if getattr(self, "spec_mtp", 0) > 0 else None
        self._guard_vram_pressure()
        if getattr(self, "spec_mtp", 0) > 0 and self._spec_step_or_fail():
            self._finish_mtp_cycle(mtp_sample)
            self._flush_abort_acks()
            return

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            out = self._forward_or_fail(forward_input)
            ongoing_data = (forward_input, out) if out is not None else None

        self._process_last_data(ongoing_data)
        self._flush_oom()
        self._finish_mtp_cycle(mtp_sample)
        self._flush_abort_acks()

    def _guard_vram_pressure(self) -> None:
        guard = getattr(getattr(self, "engine", None), "guard_vram_before_forward", None)
        if guard is not None and guard(prefill=self.prefill_manager.runnable):
            for controller in [getattr(self, "_mtp_controller", None)] + list(
                getattr(self, "_mtp_controllers", {}).values()
            ):
                if controller is not None:
                    controller.fallback_to_k0()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        data = None
        normal = False
        while True:
            use_normal = ENV.DISABLE_OVERLAP_SCHEDULING or self._has_greedy_mtp_req(data)
            if use_normal:
                if data is not None:
                    self.stream.wait_stream(self.engine.stream)
                    self._last_data = data
                    self._process_last_data(data)
                    self._last_data = data = None
                    self._flush_oom()
                    self._flush_abort_acks()
                    use_normal = ENV.DISABLE_OVERLAP_SCHEDULING or self._has_greedy_mtp_req(None)
                    if not use_normal:
                        continue
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    while ENV.DISABLE_OVERLAP_SCHEDULING or self._has_greedy_mtp_req(None):
                        self.normal_loop()
                normal = True
            else:
                if normal:
                    self.stream.wait_stream(self.engine.stream)
                    with torch.cuda.stream(self.stream):
                        data = self.overlap_loop(data)
                else:
                    assert torch.cuda.current_stream() == self.stream
                    data = self.overlap_loop(data)
                normal = False

    def _has_greedy_mtp_req(self, pending_data: ForwardData | None) -> bool:
        if self.spec_mtp <= 0:
            return False
        reqs = self.decode_manager.running_reqs
        sampling = [req.sampling_params for req in reqs]
        sampling.extend(pending.sampling_params for pending in self.prefill_manager.pending_list)
        if pending_data is not None:
            sampling.extend(req.sampling_params for req in pending_data[0].batch.reqs)
        return any(params.is_greedy for params in sampling)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        sync_t0 = time.perf_counter() if _HOST_TRACE else 0.0
        copy_done.synchronize()
        sync_ms = (time.perf_counter() - sync_t0) * 1e3 if _HOST_TRACE else 0.0
        host_t0 = time.perf_counter() if _HOST_TRACE else 0.0
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                trace_token(
                    kind="decode",
                    uid=req.uid,
                    token_index=int(req.device_len),
                    token_id=next_token,
                    table_idx=req.table_idx,
                    linear_slot_idx=req.linear_slot_idx,
                )
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                # Overlap may already have advanced device_len for the following forward;
                # finish only after this drained token reaches the output budget.
                hit_length = req.input_ids.numel() >= req.max_device_len
                hit_eos = not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        host_ms = (time.perf_counter() - host_t0) * 1e3 if _HOST_TRACE else 0.0
        send_t0 = time.perf_counter() if _HOST_TRACE else 0.0
        self.send_result(reply)
        send_ms = (time.perf_counter() - send_t0) * 1e3 if _HOST_TRACE else 0.0
        if _HOST_TRACE:
            n = len(reply)
            _host_trace_totals[0] += n
            _host_trace_totals[1] += sync_ms
            _host_trace_totals[2] += host_ms
            _host_trace_totals[3] += send_ms
            if _host_trace_totals[0] and _host_trace_totals[0] % 64 == 0:
                print(
                    "[host-trace] tokens=%d sync_ms_per_token=%.4f host_ms_per_token=%.4f "
                    "send_ms_per_token=%.4f d2h_bytes_per_token=4"
                    % (
                        _host_trace_totals[0],
                        _host_trace_totals[1] / _host_trace_totals[0],
                        _host_trace_totals[2] / _host_trace_totals[0],
                        _host_trace_totals[3] / _host_trace_totals[0],
                    ),
                    flush=True,
                )

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            if msg.mm_items and self.engine.encoder_cache is None:
                # no encoder runtime: fail loudly instead of decoding unexpanded placeholders
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            error="image input is not supported by this server",
                        )
                    ]
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if (
                req_to_free is not None
                and req_to_free.mm_items
                and self.engine.encoder_cache is not None
            ):
                # drop the aborted request's claims; entries it held alone die here
                self.engine.encoder_cache.release(
                    msg.uid, [item.hash for item in req_to_free.mm_items]
                )
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id,
                    "unsupported",
                    "this model's cache does not support runtime rebuild",
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                if pool.has_slot_state("mtp_residual"):
                    residual = (
                        pool.slot_state("mtp_residual")[req.linear_slot_idx].unsqueeze(0).clone()
                    )
                    # Prefix hits restore the slot on the engine stream. Keep the model's
                    # scalar MTP seed in sync too; otherwise a recycled request can start
                    # its first draft from the previous request's residual.
                    model = getattr(getattr(self.engine, "model", None), "model", None)
                    if model is not None and getattr(self, "spec_mtp", 0) > 0:
                        model._last_residual = residual
                    self._mtp_prompt_carry = (
                        req.uid,
                        req.cached_len,
                        residual,
                    )
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _recover_cache_oom(self, error: torch.OutOfMemoryError) -> bool:
        """Fund a retry of metadata staging before it transfers request ownership."""
        cache = getattr(self.engine, "moe_offload_cache", None)
        before = getattr(cache, "resident_rows", None)
        shrink = getattr(self.engine, "shrink_after_oom", None)
        # Fixed backing would rebuild graph-bound state while this request is live.
        if before is None or shrink is None or not getattr(cache, "_vmm_arenas", None):
            return False
        note = getattr(self.engine, "note_decode_oom", None)
        if note is not None:
            note(error)
        torch.cuda.synchronize(self.device)
        shrink()
        torch.cuda.empty_cache()
        return cache.resident_rows < before

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        engine = getattr(self, "engine", None)
        kv = getattr(engine, "kv_cache", None) if engine is not None else None
        if kv is not None and hasattr(kv, "free_req"):
            kv.free_req(req.table_idx)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1
        # A request that stops being spec-eligible on its last token (remain_len <= 1, see
        # _spec_eligible_req) finishes through the plain decode path, never through
        # run_spec_step's own finished branch -- release here, universally, or the GDN
        # snapshot slot leaks on every request and exhausts LinearStatePool after a
        # handful of requests. Idempotent (pop with a default) if spec.py already freed it.
        # hasattr guard: some scheduler-unit-test stubs build a bare `self` without the
        # SchedulerSpecMixin (spec_mtp is always 0 for them, so there is nothing to free).
        if hasattr(self, "free_spec_snapshot_slot"):
            self.free_spec_snapshot_slot(req)

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _rebalance_kv_tiers(self) -> None:
        """Promote hot RAM-tier KV pages into the device slab. Runs on the scheduling stream after
        it waited for every prior forward and before this step's metadata, so placement changes
        between forwards only."""
        # ponytail: fixed cadence of one swap wave per 16 steps, 1/16 of the device slab per
        # wave; derive both from measured PCIe cost vs selection misses if this shows in TG.
        pool = self.engine.kv_cache
        if getattr(pool, "page_map", None) is None:
            return
        self._kv_steps = getattr(self, "_kv_steps", 0) + 1
        if self._kv_steps % 16 == 0:
            pool.rebalance(max(1, pool.num_device_pages // 16))

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size
            if eng.moe_offload_cache is not None
            else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1)
            if eng.linear_state_pool is not None
            else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch, *, skip_alloc: bool = False) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter
            )
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        # skip_alloc: SchedulerSpecMixin's GDN-state replay rewinds cached_len/device_len to an
        # already-verified window (its pages were allocated by that earlier verify's own
        # _prepare_batch call) purely to recompute positions/attention metadata over it --
        # allocate_paged has no memory of that prior call, so re-running it here on a rewound
        # (cached_len, device_len) pair that straddles a page boundary grabs a FRESH page and
        # overwrites the page_table row that already held the correct, real one (orphaning it).
        if not skip_alloc:
            self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [
                        r.linear_slot_idx if r.linear_slot_idx is not None else pool.padding_slot
                        for r in batch.padded_reqs
                    ]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Plan the chunk's encoder jobs, gather rows and scatter rows over the batch; the engine runs them before the LM forward."""
        jobs, plan, rows, block_ends = plan_mm_batch(batch.padded_reqs, self.engine.encoder_cache)
        if plan:
            batch.mm_encoder_jobs = jobs
            batch.mm_gather_plan = plan
            batch.mm_rows = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
                self.device, non_blocking=True
            )
            batch.mm_block_ends = torch.tensor(block_ends, dtype=torch.int32, pin_memory=True).to(
                self.device, non_blocking=True
            )
        if (
            self._bidirectional_mm
            and not self._warned_cut_image
            and (cut := cut_image_spans(batch.padded_reqs))
        ):
            # only a bidirectional image span loses context when cut, and only an image longer than the chunk still gets cut
            lo, hi = cut[0]
            self._warned_cut_image = True
            logger.warning_rank0(
                f"an image of {hi - lo} tokens does not fit one prefill chunk (--max-extend-tokens {self.prefill_budget}, or the sliding-window pool's share of it): its earlier rows attend within the first part only"
            )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward_or_fail(self, forward_input: ForwardInput) -> ForwardOutput | None:
        """``_forward``, except that a GPU out-of-memory fails this batch's requests (error reply,
        resources freed) and shrinks the expert cache instead of killing the process: the
        allocation fails on the host before the rest of the forward is launched, and the dropped
        requests' partial KV/GDN writes die with them. The failure is handled by ``_flush_oom``
        after the loop drained the previous (overlapped) batch, which may still hold these
        requests and must publish/free them first."""
        try:
            return self._forward(forward_input)
        except Exception as e:  # noqa: BLE001 -- only OOM is handled, everything else re-raised
            if not _is_oom(e):
                raise
            note = getattr(self.engine, "note_decode_oom", None)
            attempted = note(e) if note is not None else 0
            # One idempotent retry after paying the reserve back: a recomputed forward
            # writes identical KV/GDN values into the same slots, and sampling/commit
            # live outside forward, so re-issuing the exact batch is side-effect free.
            oom_retry = getattr(self, "_oom_retry", None)
            retry = oom_retry(forward_input, e, attempted) if oom_retry is not None else None
            if retry is not None:
                return retry
            self._oom_failed = ([r for r in forward_input.batch.reqs if r.uid >= 0], e)
            return None

    def _oom_retry(self, forward_input: ForwardInput, error: BaseException, attempted: int):
        if getattr(self, "_oom_retried_input", None) is forward_input:
            return None  # never loop: one retry per forward input
        self._oom_retried_input = forward_input
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001 -- sync is best-effort (stubs/cpu fallbacks)
            pass
        shrink = getattr(self.engine, "shrink_after_oom", None)
        if shrink is not None:
            try:
                shrink()
            except Exception:  # noqa: BLE001 -- shrink failure must not mask the retry
                pass
        try:
            out = self._forward(forward_input)
            logger.warning(f"OOM recovered: retried forward after reserving {attempted >> 20} MiB")
            return out
        except Exception as e2:  # noqa: BLE001 -- second OOM falls through to fail path
            if not _is_oom(e2):
                raise
            return None

    def _flush_oom(self) -> None:
        failed = getattr(self, "_oom_failed", None)
        if failed is None:
            return
        self._oom_failed = None
        reqs, error = failed
        # a request that finished in the drained batch is already freed and replied to
        self._fail_oom_reqs([r for r in reqs if r not in self.finished_reqs], error)

    def _spec_step_or_fail(self) -> bool:
        try:
            return self.run_spec_step()
        except Exception as e:  # noqa: BLE001 -- see _forward_or_fail
            if not _is_oom(e):
                import traceback

                logger.error("[spec-trace]\n" + traceback.format_exc())
                raise
            note = getattr(self.engine, "note_decode_oom", None)
            if note is not None:
                note(e)
            mark_oom = getattr(self, "_mark_mtp_oom", None)
            depth = getattr(self, "_mtp_cycle_depth", 0)
            rollback = getattr(self, "_spec_rollback", None)
            if rollback is not None:
                # Restore allocations need headroom too; release experts before undoing writes.
                torch.cuda.synchronize(self.device)
                cache = getattr(self.engine, "moe_offload_cache", None)
                rows = getattr(cache, "resident_rows", None)
                shrink = getattr(self.engine, "shrink_after_oom", None)
                if shrink is not None:
                    shrink()
                torch.cuda.empty_cache()
                try:
                    rollback()
                except Exception as restore_error:  # noqa: BLE001 -- only OOM permits retry
                    if not _is_oom(restore_error):
                        raise
                    if note is not None:
                        note(restore_error)
                    retry_rows = getattr(cache, "resident_rows", None)
                    if shrink is None:
                        raise
                    shrink()
                    torch.cuda.empty_cache()
                    if (
                        retry_rows is not None
                        and getattr(cache, "resident_rows", None) == retry_rows
                    ):
                        raise
                    # The preimage stays registered until this bounded retry succeeds.
                    rollback()
                self._spec_rollback = None
                if rows is None or getattr(cache, "resident_rows", None) == rows:
                    # nothing left to give back: only a shallower depth can fit
                    if mark_oom is not None:
                        mark_oom(depth)
                # else the next cycle retries this depth inside the reserve just learned
                # the RAW step that follows is not a sample of the refused depth
                self._mtp_cycle_observe = False
                running = getattr(getattr(self, "decode_manager", None), "running_reqs", ())
                if any(
                    getattr(req, "device_len", 0) - 1 > getattr(req, "cached_len", 0)
                    for req in running
                ):
                    # A protected flush still owes target rows; plain decode cannot skip them.
                    logger.warning(f"MTP replay OOM rolled back ({e}); replay will retry")
                    return True
                logger.warning(f"MTP cycle OOM rolled back ({e}); this step decodes RAW")
                return False
            if mark_oom is not None:
                mark_oom(depth)
            self._fail_oom_reqs(list(self.decode_manager.running_reqs), e)
            return True

    def _fail_oom_reqs(self, reqs, error: BaseException) -> None:
        logger.error(
            f"GPU out of memory in a forward ({error}); failing {len(reqs)} request(s) and "
            "shrinking the expert cache"
        )
        # the failed forward may have launched kernels before the allocation that failed
        torch.cuda.synchronize(self.device)
        for req in reqs:
            self.prefill_manager.abort_req(req.uid)
            self.decode_manager.abort_req(req.uid)
            if getattr(req, "table_idx", -1) >= 0:
                # Keep only the prefix it matched in the cache: return every page it allocated
                # past that (the failed chunk's never-committed pages would otherwise leak), and
                # insert nothing new -- its GDN slot may hold a half-advanced state.
                handle = getattr(req, "cache_handle", None)
                start = getattr(handle, "cached_len", 0) or 0
                page = self.cache_manager.page_size
                owned = max(req.device_len, getattr(req, "alloc_page_bound", 0) * page)
                self.cache_manager.free_spec_reject(req, keep_len=start, alloc_len=owned)
                req.cached_len = req.device_len = start
                self._free_req_resources(req)
        self.send_result(
            [ErrorReplyMsg(uid=r.uid, error="out of GPU memory; request dropped") for r in reqs]
        )
        torch.cuda.empty_cache()
        shrink = getattr(self.engine, "shrink_after_oom", None)
        if shrink is not None:
            shrink()

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        probe_residual = None
        controller = getattr(self, "_mtp_controller", None)
        if batch.is_decode and batch.size == 1 and controller is not None and controller.probing:
            residual = getattr(self.engine.model.model, "_last_residual", None)
            if residual is not None:
                probe_residual = residual[-1:].clone()
                probe_pos = batch.reqs[0].cached_len
        forward_output = self.engine.forward_batch(batch, sample_args)
        if probe_residual is not None:
            self._queue_mtp_fill(batch.reqs[0], probe_pos, probe_residual, batch.input_ids)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        if (
            getattr(self, "spec_mtp", 0) > 0
            and os.getenv("FREETOKEN_MTP_PROMPT_WARMUP", "1") == "1"
            and batch.is_prefill
            and batch.size == 1
        ):
            self.warmup_mtp_draft_kv(batch.reqs[0], batch)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e)


def _make_mrope_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """[3, N] rope rows: an image request's prompt tokens use their precomputed columns, everything else is sequence index + per-request delta."""
    needed = sum(r.extend_len for r in batch.padded_reqs)
    host = torch.empty((3, needed), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        out = host[:, offset : offset + length]
        full = req.mrope_positions_full
        if full is not None and req.device_len <= full.shape[1]:
            out.copy_(full[:, req.cached_len : req.device_len])
        else:
            row = torch.arange(
                req.cached_len + req.mrope_delta,
                req.device_len + req.mrope_delta,
                dtype=torch.int32,
            )
            out.copy_(row.unsqueeze(0).expand(3, -1))
        offset += length
    return host.to(device, non_blocking=True)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
