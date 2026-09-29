"""Native MTP (multi-token prediction) speculative decode: draft chain + verify + accept/reject.

Single-request only. Verification uses a spec-indexed prefill Batch and captured target windows
when available. Production chooses k0/k1/k2 from measured seconds per committed token.
Bypasses scheduler._forward /
_process_last_data entirely: their single-token-per-request write logic cannot carry more than
one accepted token per step. Design history: docs/freetoken-next/EXPERIMENTS.md EXP-023/024/025.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Callable, List

import torch
from freetoken.core import Batch, Req
from freetoken.engine.graph import DRAFT_GRAPH_CHECK_ENV, SPEC_DEFER_MAX
from freetoken.engine.spec import accept_drafts, spec_rollback_lengths
from freetoken.message import DetokenizeMsg
from freetoken.debug.token_trace import enabled as trace_enabled
from freetoken.debug.token_trace import record as trace_token
from freetoken.utils import init_logger, nvtx_annotate

if TYPE_CHECKING:
    from .scheduler import Scheduler  # noqa: F401  (self-typing only)

logger = init_logger(__name__)

SPEC_TIMING_ENV = "FREETOKEN_DEBUG_SPEC_TIMING"
# "1": run every graph-eligible verify eagerly first and assert the graph replay reproduces it
# bitwise (logits, sampled tokens, MTP residual, every state tensor the forward writes).
VERIFY_GRAPH_CHECK_ENV = "FREETOKEN_VERIFY_GRAPH_CHECK"
# "0" replays a k=1 rejection at once instead of re-feeding its tokens in the next verify window.
DEFER_REPLAY_ENV = "FREETOKEN_SPEC_DEFER_REPLAY"
_PROFILE_DECODE = os.getenv("FREETOKEN_PROFILE_DECODE", "0") == "1"
_REPLAY_GRAPH = os.getenv("FREETOKEN_SPEC_REPLAY_GRAPH", "1") != "0"


def _spec_mrope_positions(
    req: Req, cached_len: int, device_len: int, device: torch.device
) -> torch.Tensor:
    """Build the three-axis RoPE positions for a manually assembled draft batch."""
    full = req.mrope_positions_full
    if full is not None and device_len <= full.shape[1]:
        return full[:, cached_len:device_len].to(device, non_blocking=True)
    row = torch.arange(
        cached_len + req.mrope_delta,
        device_len + req.mrope_delta,
        dtype=torch.int32,
        device=device,
    )
    return row.unsqueeze(0).expand(3, -1)


class SchedulerSpecMixin:
    """Mixed into Scheduler. Needs self.spec_mtp/token_pool/cache_manager/engine/decode_manager
    /eos_token_ids/toolcall_anchor_id/finished_reqs/_prepare_batch (see scheduler.py)."""

    def warmup_mtp_draft_kv(self, req: Req, target_batch: Batch | None = None) -> None:
        """Fill prompt draft KV with h_i and token_(i+1), carrying one row across chunks.

        A radix hit has no saved target residual history: only its recomputed suffix can
        be warmed. Position zero has no preceding target residual and stays zero-filled.
        """
        if self.spec_mtp <= 0 or target_batch is None:
            return
        model = getattr(self.engine, "model", None)
        mtp = getattr(model, "mtp", None)
        if mtp is None:
            return
        r_last = getattr(getattr(model, "model", None), "_last_residual", None)
        if r_last is None or r_last.numel() == 0:
            return

        n = target_batch.input_ids.numel()
        if r_last.shape[0] != n:
            return
        end = req.cached_len  # engine.complete_one has consumed the original target window
        start = end - n
        carry = getattr(self, "_mtp_prompt_carry", None)
        self._mtp_prompt_carry = (req.uid, end, r_last[-1:].clone())
        if carry is not None and carry[0] == req.uid and carry[1] == start:
            start_pos = start
            r_window = torch.cat([carry[2], r_last[:-1]])
            tok_window = target_batch.input_ids
        else:
            start_pos = start + 1
            r_window = r_last[:-1]
            tok_window = target_batch.input_ids[1:]
        if tok_window.numel() == 0:
            return

        old_cached, old_device = req.cached_len, req.device_len
        from freetoken.moe.offload_cache import DECODE_PATH_MAX_TOKENS

        try:
            # Draft-only expert banks have no full-prefill staging pool. Keep each warmup
            # window on the fixed-size on-demand expert path, including on a cold prompt.
            for offset in range(0, tok_window.numel(), DECODE_PATH_MAX_TOKENS):
                stop = min(offset + DECODE_PATH_MAX_TOKENS, tok_window.numel())
                lo, hi = start_pos + offset, start_pos + stop
                req.cached_len, req.device_len = lo, hi
                wb = Batch(reqs=[req], phase="prefill")
                wb.padded_reqs = [req]
                wb.positions = torch.arange(lo, hi, dtype=torch.int32, device=self.device)
                if self._model_is_mrope:
                    wb.mrope_positions = _spec_mrope_positions(req, lo, hi, self.device)
                wb.out_loc = self.engine.page_table[req.table_idx, lo:hi]
                wb.input_ids = tok_window[offset:stop]
                self.engine.attn_backend.prepare_metadata(wb)
                with self.engine.ctx.forward_batch(wb):
                    mtp.forward(r_window[offset:stop], wb.input_ids, wb)
        finally:
            req.cached_len, req.device_len = old_cached, old_device

    def _spec_eligible_req(self) -> Req | None:
        if self.spec_mtp <= 0:
            return None
        running = self.decode_manager.running_reqs
        if len(running) != 1:
            return None
        (req,) = running
        if not req.sampling_params.is_greedy or req.remain_len <= 1:
            return None
        return req

    def _begin_mtp_cycle(self):
        controller = getattr(self, "_mtp_controller", None)
        running = self.decode_manager.running_reqs
        if controller is None or len(running) != 1:
            return None
        (req,) = running
        if not req.sampling_params.is_greedy:
            return None
        cache = getattr(self.engine, "moe_offload_cache", None)
        epoch = (
            req.device_len.bit_length(),
            getattr(cache, "cache_size", None),
            self.engine.num_pages,
        )
        controller.begin_request(req.uid, epoch)
        if req.remain_len <= 1:
            controller.fallback_to_k0()
        if getattr(self, "_mtp_request_uid", None) != req.uid:
            self._mtp_request_uid = req.uid
            self._mtp_distribution = {}
        self._mtp_cycle_depth = 0
        return req, req.input_ids.numel(), time.perf_counter()

    def _finish_mtp_cycle(self, sample) -> None:
        if sample is None:
            return
        req, before, started = sample
        committed = req.input_ids.numel() - before
        if committed <= 0:
            return
        elapsed = time.perf_counter() - started
        controller = self._mtp_controller
        depth = self._mtp_cycle_depth
        controller.observe(depth, elapsed, committed)
        # A calibration just converged -> persist the learned depth so the next serve of this
        # fingerprint warm-starts at it (no probe). None on every non-calibration cycle.
        learned = controller.consume_learned_depth()
        if learned is not None:
            self._save_mtp_depth_profile(learned)
        counts = self._mtp_distribution.setdefault(depth, [0, 0, 0.0])
        counts[0] += 1
        counts[1] += committed
        counts[2] += elapsed
        if req in self.finished_reqs:
            logger.info(
                f"[mtp-economics] uid={req.uid} distribution={self._mtp_distribution} "
                f"costs={controller.cost_summaries} selected={controller.selected_depth}"
            )

    def _spec_snapshot_slot(self, req: Req) -> int:
        slot = self._spec_snapshot_slots.get(req.uid)
        if slot is None:
            if self.cache_manager.is_hybrid:
                # Shares the free list with radix snapshots, which may have drained it.
                self.cache_manager.ensure_mamba_slots(1)
            slot = self.engine.linear_state_pool.alloc(1)[0]
            self._spec_snapshot_slots[req.uid] = slot
        return slot

    def free_spec_snapshot_slot(self, req: Req) -> None:
        slot = self._spec_snapshot_slots.pop(req.uid, None)
        if slot is not None:
            self.engine.linear_state_pool.free([slot])
        if hasattr(self, "_spec_qsa_snapshots"):
            self._spec_qsa_snapshots.pop(req.uid, None)
        rows = getattr(self, "_mtp_kv_rows", None)
        if rows is not None and rows[0] == req.uid:
            self._mtp_kv_rows = None
        carry = getattr(self, "_mtp_prompt_carry", None)
        if carry is not None and carry[0] == req.uid:
            self._mtp_prompt_carry = None

    def _snapshot_qsa_state(self, req: Req) -> tuple[torch.Tensor, torch.Tensor] | None:
        kv = getattr(self.engine, "kv_cache", None)
        ring_buf = getattr(kv, "_pending_ring", None)
        if ring_buf is None:
            return None
        if not hasattr(self, "_spec_qsa_snapshots"):
            self._spec_qsa_snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        snap = self._spec_qsa_snapshots.get(req.uid)
        ring_src = ring_buf[req.table_idx]
        scratch_base = getattr(kv, "_cmp_scratch_base", 0)
        scratch_src = kv._cmp_k_buffer[:, scratch_base + req.table_idx]
        if snap is None:
            snap_ring = ring_src.clone()
            snap_scratch = scratch_src.clone()
            self._spec_qsa_snapshots[req.uid] = (snap_ring, snap_scratch)
        else:
            snap_ring, snap_scratch = snap
            snap_ring.copy_(ring_src)
            snap_scratch.copy_(scratch_src)
        return snap_ring, snap_scratch

    def _restore_qsa_state(self, req: Req, *, keep_start: int = 0, keep_count: int = 0) -> None:
        if not hasattr(self, "_spec_qsa_snapshots"):
            return
        snap = self._spec_qsa_snapshots.get(req.uid)
        if snap is None:
            return
        kv = getattr(self.engine, "kv_cache", None)
        ring_buf = getattr(kv, "_pending_ring", None)
        if ring_buf is None:
            return
        snap_ring, snap_scratch = snap
        ring = ring_buf[req.table_idx]
        if keep_count:
            capacity = ring.shape[-2]
            positions = torch.arange(capacity, device=ring.device)
            keep = (positions - keep_start) % capacity < keep_count
            ring.copy_(torch.where(keep[None, :, None], ring, snap_ring))
        else:
            ring.copy_(snap_ring)
        scratch_base = getattr(kv, "_cmp_scratch_base", 0)
        kv._cmp_k_buffer[:, scratch_base + req.table_idx].copy_(snap_scratch)

    def _verify_state(self, req: Req) -> List[torch.Tensor]:
        """Every state tensor a verify forward can write: the request's linear-state slot
        (GDN conv/recurrent, PLE context/conv) and the whole KV cache (QSA K/V slabs, pending
        ring, compressed index slab) -- whole, so a write through a stale address shows too."""
        pool = self.engine.linear_state_pool
        slot = self._linear_slot(req)
        state = [pool.conv_states[:, slot], pool.recurrent_states[:, slot]]
        state += [t[:, slot] for t in pool.slot_states.values()]
        kv = self.engine.kv_cache
        # QSA's per-slot scratch rows sink the non-closing rows' racing writes and are never read
        scratch = getattr(kv, "_cmp_k_buffer", None)
        for owner in (kv, getattr(kv, "_pool", None)):
            for value in vars(owner).values() if owner is not None else ():
                values = value if isinstance(value, (list, tuple)) else (value,)
                state += [
                    t[:, : kv.cmp_scratch_base] if t is scratch else t
                    for t in values
                    if isinstance(t, torch.Tensor) and t.is_cuda
                ]
        return state

    def _checked_verify_forward(self, req: Req, vb: Batch, sample_args):
        """``VERIFY_GRAPH_CHECK_ENV``: eager verify, restore the pre-verify state, graph
        verify, and assert both agree bitwise. Every 8th check starts from a cold expert
        cache on one side, so hit/miss routing differs between the two runs."""
        engine, runner = self.engine, self.engine.graph_runner
        model = engine.model.model
        n = self._verify_checks = getattr(self, "_verify_checks", 0) + 1
        state = self._verify_state(req)
        before = [t.clone() for t in state]
        if n % 8 == 0 and engine.moe_offload_cache is not None:
            engine.moe_offload_cache.reset()
        graphs, runner.verify_graphs = runner.verify_graphs, {}
        try:
            eager = engine.forward_batch(vb, sample_args)
        finally:
            runner.verify_graphs = graphs
        expect = [t.clone() for t in state]
        expect += [engine.last_batch_logits.clone(), eager.next_tokens_gpu.clone()]
        expect.append(model._last_residual.clone())
        for t, b in zip(state, before):
            t.copy_(b)
        if n % 8 == 4 and engine.moe_offload_cache is not None:
            engine.moe_offload_cache.reset()
        out = engine.forward_batch(vb, sample_args)
        got = state + [engine.last_batch_logits, out.next_tokens_gpu, model._last_residual]
        bad = [i for i, (g, e) in enumerate(zip(got, expect)) if not torch.equal(g, e)]
        if bad:
            raise AssertionError(
                f"verify graph != eager at check {n} ({vb.input_ids.shape[0]} rows): tensors "
                f"{[(i, tuple(got[i].shape), got[i].dtype) for i in bad]} of {len(got)} "
                f"(last three: logits, tokens, residual)"
            )
        if n == 1 or n % 100 == 0:
            nbytes = sum(t.numel() * t.element_size() for t in state)
            logger.info(f"verify graph check {n} ok: {len(got)} tensors, {nbytes >> 20} MiB state")
        return out

    def _snapshot_ple_state(self, req: Req) -> torch.Tensor | None:
        pool = getattr(self.engine, "linear_state_pool", None)
        if pool is None or not pool.has_slot_state("ple_ngram_ctx"):
            return None
        state = pool.slot_state("ple_ngram_ctx")[self._linear_slot(req)]
        if not hasattr(self, "_spec_ple_snapshots"):
            self._spec_ple_snapshots: dict[int, torch.Tensor] = {}
        snapshot = state.clone()
        self._spec_ple_snapshots[req.uid] = snapshot
        return snapshot

    def _restore_ple_state(self, req: Req) -> None:
        snapshots = getattr(self, "_spec_ple_snapshots", None)
        snapshot = snapshots.get(req.uid) if snapshots is not None else None
        if snapshot is None:
            return
        pool = getattr(self.engine, "linear_state_pool", None)
        if pool is not None and pool.has_slot_state("ple_ngram_ctx"):
            pool.slot_state("ple_ngram_ctx")[self._linear_slot(req)].copy_(snapshot)

    @staticmethod
    def _linear_slot(req: Req) -> int:
        """GDN/PLE state slot: the hybrid-radix live slot when allocated, else table_idx
        (naive keeps the old keying -- same rule as build_fla_metadata's gdn_slot)."""
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def _commit_spec_tokens(
        self,
        req: Req,
        tokens: List[int],
        start_pos: int,
        spec_alloc_len: int,
        finish_state: Callable[[int], None] | None = None,
    ) -> int:
        """Append tokens one at a time, applying _process_last_data's per-token EOS/stop
        /length finish logic. ``start_pos`` is tokens[0]'s device-table position (== d in
        run_spec_step): cached_len/device_len advance ONE token at a time here, exactly like
        complete_one(), rather than being pre-set to the whole window's end -- otherwise
        can_decode reflects the window's final length for every token in it, not each
        token's own position, and hit_length fires (or doesn't) at the wrong offset.
        ``spec_alloc_len`` is the verify window's true allocation ceiling (d + k): a mid-window
        finish rewinds cached_len/device_len to the truncation point, which is BELOW that
        ceiling, so a finish here must reclaim [truncation, spec_alloc_len) itself, before
        _free_req_resources recycles table_idx -- the caller can no longer do it afterward
        (table_idx == -1 by then, and page_table[-1] silently frees a different row's pages).
        Returns how many actually count (stops at, and includes, the one that finishes the
        request -- exact-once-only completion)."""
        reply: List[DetokenizeMsg] = []
        committed = 0
        finished_now = False
        with self.cache_manager.lazy_free_region():
            for offset, next_token in enumerate(tokens):
                trace_token(
                    kind="spec_commit",
                    uid=req.uid,
                    token_index=int(start_pos + offset),
                    token_id=int(next_token),
                    table_idx=req.table_idx,
                    linear_slot_idx=req.linear_slot_idx,
                    accepted_length=int(committed),
                )
                req.append_host(torch.tensor([next_token], dtype=req.input_ids.dtype))
                committed += 1
                req.cached_len = start_pos + offset
                req.device_len = start_pos + offset + 1
                hit_length = not req.can_decode
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
                if finished:
                    finished_now = True
                    break
            self.send_result(reply)
            if finished_now:
                keep_cached, keep_device = spec_rollback_lengths(start_pos, committed)
                if finish_state is not None:
                    finish_state(committed)
                if keep_cached < spec_alloc_len:
                    self.cache_manager.free_spec_reject(
                        req, keep_len=keep_cached, alloc_len=spec_alloc_len
                    )
                # The final output may correct a rejected input; never cache its draft KV.
                # Reclaim from the same boundary, including a terminal page never donated.
                req.cached_len, req.device_len = keep_cached, keep_device
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished_reqs.add(req)
        return committed

    @nvtx_annotate("MTPReplay", enabled=_PROFILE_DECODE)
    def _replay(self, req: Req, start: int, n: int) -> None:
        """Re-run the target over tokens [start, start + n) from the restored state; the
        sampled output is discarded (those successors are already committed)."""
        rb = Batch(reqs=[req], phase="decode" if n == 1 else "prefill")
        if rb.is_decode:
            rb.padded_reqs = [req]
        elif _REPLAY_GRAPH:
            # A committed replay window uses the same row-wise GDN recurrence and captured
            # target shapes as verification; the general prefill chunk scan is redundant.
            rb.spec_logits_indices = torch.arange(n, device=self.device)
            rb.spec_host_ids = req.input_ids[start : start + n].tolist()
        req.cached_len = start
        req.device_len = start + n
        rfi = self._prepare_batch(rb, skip_alloc=True)
        rb.input_ids = self.token_pool[rfi.input_tuple]
        rout = self.engine.forward_batch(rb, rfi.sample_args)
        rout.copy_done_event.synchronize()
        if rb.spec_logits_indices is not None:
            # Spec-indexed forwards skip the engine's usual complete_one transition.
            # Preserve this helper's postcondition; its callers restore final live lengths.
            req.complete_one()

    def _flush_deferred_replays(self) -> None:
        """Replay deferred tokens before any non-spec forward, which expects one pending token."""
        for req in list(self.decode_manager.running_reqs):
            end = req.device_len - 1
            if end > req.cached_len:
                self._replay(req, req.cached_len, end - req.cached_len)
                req.cached_len, req.device_len = end, end + 1

    @nvtx_annotate("MTPDraft", enabled=_PROFILE_DECODE)
    def _draft_step(
        self, req: Req, pos: int, residual: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One MTP draft step at ``pos``: (next residual, logits, argmax token)."""
        db = Batch(reqs=[req], phase="prefill")
        db.padded_reqs = [req]
        n = residual.shape[0]  # fill rows + the draft row (the last)
        db.positions = torch.arange(pos, pos + n, dtype=torch.int32, device=self.device)
        if self._model_is_mrope:
            db.mrope_positions = _spec_mrope_positions(req, pos, pos + n, self.device)
        db.out_loc = self.engine.page_table[req.table_idx, pos : pos + n]
        db.input_ids = token
        db.spec_logits_indices = torch.arange(1, device=self.device)
        self.engine.attn_backend.prepare_metadata(db)
        runner = self.engine.graph_runner
        if runner is not None and n in runner.drafts:
            if os.getenv(DRAFT_GRAPH_CHECK_ENV, "0") != "1":
                return runner.replay_draft(db, residual, token)
            got = [t.clone() for t in runner.replay_draft(db, residual, token)]
        model = self.engine.model
        with self.engine.ctx.forward_batch(db):
            r = model.mtp.forward(residual, token, db)[-1:]
            from freetoken.engine.graph import mtp_draft_logits

            logits = mtp_draft_logits(model, r)
        out = (r, logits, torch.argmax(logits, dim=-1))
        if runner is not None and n in runner.drafts:
            bad = [i for i, (g, e) in enumerate(zip(got, out)) if not torch.equal(g, e)]
            if bad:
                raise AssertionError(
                    f"draft graph != eager: outputs {bad} (residual, logits, token)"
                )
        return out

    def _take_mtp_fill(self, req: Req):
        """(residual rows, tokens) for the committed positions the last draft chain skipped
        (see run_spec_step's commit), or None; they are prepended to the first draft step."""
        rows = getattr(self, "_mtp_kv_rows", None)
        self._mtp_kv_rows = None
        if rows is None or rows[0] != req.uid or rows[1] + rows[2].shape[0] != req.device_len - 1:
            return None
        return rows[2], rows[3]

    @nvtx_annotate("MTPCycle", enabled=_PROFILE_DECODE)
    def run_spec_step(self) -> bool:
        """Run one speculative decode step for the single eligible request. Returns True if
        it ran (the caller should skip its own _schedule_next_batch/_forward this iteration)."""
        req = self._spec_eligible_req()
        if req is None:
            self._flush_deferred_replays()
            return False
        from freetoken.scheduler.adaptive_mtp import resolve_adaptive_k

        controller = getattr(self, "_mtp_controller", None)
        requested_k = controller.next_depth() if controller is not None else self.spec_mtp
        k = resolve_adaptive_k(req, requested_k)
        if controller is not None and k != requested_k:
            controller.fallback_to_k0()
            k = 0
        self._mtp_cycle_depth = k
        if k <= 0:
            self._flush_deferred_replays()
            self._mtp_kv_rows = None
            return False
        # A k=1 rejection may defer target inputs. Flush before a wider depth would exceed
        # the pre-priced verify shapes or a model's per-row state buffers.
        if req.device_len - req.cached_len + k > self.spec_mtp + 1 + (
            SPEC_DEFER_MAX if self.spec_mtp == 1 else 0
        ):
            self._flush_deferred_replays()

        debug_timing = os.getenv(SPEC_TIMING_ENV, "0") == "1"
        _t0 = time.perf_counter()

        def mark(stage: str) -> None:
            if debug_timing:
                torch.cuda.synchronize(self.device)
                nonlocal _t0
                now = time.perf_counter()
                print(f"[spec-timing] k={k} {stage} {now - _t0:.4f}s", flush=True)
                _t0 = now

        d = req.device_len
        # Tokens [c0, d) are unprocessed: token d-1, plus the p tokens earlier steps deferred
        # instead of replaying them after a rejection. The verify window re-feeds them all.
        c0 = req.cached_len
        p = d - 1 - c0
        model = self.engine.model
        mtp = model.mtp

        # Snapshot QSA pending ring and scratch cmp buffer before draft chain mutates them
        self._snapshot_qsa_state(req)
        self._snapshot_ple_state(req)

        # ---- snapshot linear state before the draft chain mutates it ----
        pool = self.engine.linear_state_pool
        # Zero-replay: the draft never touches the linear state (a full-attention NextN
        # head) and the verify records every row's state, so no snapshot, no replay.
        zero_replay = getattr(pool, "spec_states", None) is not None
        snap_slot = None
        if pool is not None and not zero_replay:
            snap_slot = self._spec_snapshot_slot(req)
            pool.copy_from(self._linear_slot(req), snap_slot)
        residual_snapshot = model.model._last_residual[-1:].clone()

        fill = self._take_mtp_fill(req)

        # ---- draft chain: k autoregressive steps through the draft head's own QSA slot ----
        r_prev = model.model._last_residual[-1:].clone()
        tok_prev = self.token_pool[req.table_idx, d - 1 : d]

        for i in range(k):
            # prepare_metadata (e.g. qsa_sparse) reads req.cached_len/device_len for
            # seqlens_k/extend_len; at i >= 1 the draft's own query must see its own prior
            # draft-step KV, which needs these advanced per step, not left at the entry value.
            pos, r_in, tok_in = d - 1 + i, r_prev, tok_prev
            if i == 0 and fill is not None:  # NextN-KV fill rows ride the first draft step
                pos = d - 1 - fill[0].shape[0]
                r_in, tok_in = torch.cat([fill[0], r_prev]), torch.cat([fill[1], tok_prev])
            req.cached_len, req.device_len = pos, d + i
            r_prev, logits, tok_prev = self._draft_step(req, pos, r_in, tok_in)
            self.token_pool[req.table_idx, d + i] = tok_prev
            if trace_enabled():
                trace_token(
                    kind="spec_draft",
                    uid=req.uid,
                    cycle=int(d + i),
                    token_index=int(d - 1 + i),
                    token_id=int(tok_prev.item()),
                    draft_prob=float(torch.softmax(logits.float(), dim=-1).max()),
                    table_idx=req.table_idx,
                    linear_slot_idx=req.linear_slot_idx,
                )
        # The chain depends on device tokens, not host ids. Read the saved token-pool
        # slice once after every draft has been enqueued, instead of synchronizing k times.
        drafts = self.token_pool[req.table_idx, d : d + k].tolist()
        req.cached_len, req.device_len = c0, d
        mark("draft_chain")

        # ---- restore state before target verification ----
        if snap_slot is not None:
            pool.copy_from(snap_slot, self._linear_slot(req))
        model.model._last_residual = residual_snapshot
        self._restore_qsa_state(req)
        self._restore_ple_state(req)
        mark("snapshot")

        # ---- verify: one prefill-phase Batch over [d-1, d+k) ----
        req.device_len = d + k
        vb = Batch(reqs=[req], phase="prefill")
        # set before _prepare_batch: backends shape a spec window's metadata off it (triton)
        vb.spec_logits_indices = torch.arange(p + k + 1, device=self.device)
        fi = self._prepare_batch(vb)
        vb.input_ids = self.token_pool[fi.input_tuple]
        vb.spec_host_ids = [*req.input_ids[c0:d].tolist(), *drafts]
        mark("verify_prepare_batch")
        if os.getenv(VERIFY_GRAPH_CHECK_ENV, "0") == "1" and (
            self.engine.graph_runner.can_use_cuda_graph(vb)
        ):
            out = self._checked_verify_forward(req, vb, fi.sample_args)
        else:
            out = self.engine.forward_batch(vb, fi.sample_args)
        out.copy_done_event.synchronize()
        mark("verify_forward")
        # rows [0, p) re-feed deferred tokens whose successors are already committed
        sampled = out.next_tokens_cpu.tolist()[p:]
        accepted = accept_drafts(sampled, drafts)
        m = len(accepted)
        verify_scores = None
        verify_margins = None
        if trace_enabled():
            verify_logits = self.engine.last_batch_logits[p:].detach().float().cpu()
            top_values, top_ids = torch.topk(verify_logits, k=2, dim=-1)
            verify_scores = [
                float(row[int(token)]) for row, token in zip(verify_logits.tolist(), sampled)
            ]
            verify_margins = (top_values[:, 0] - top_values[:, 1]).tolist()
        trace_token(
            kind="spec_verify",
            uid=req.uid,
            cycle=int(d),
            speculative_position=int(p),
            drafts=drafts,
            sampled=sampled,
            accepted=accepted,
            accepted_length=int(m - 1),
            selected_scores=verify_scores,
            top2_ids=top_ids.tolist() if trace_enabled() else None,
            top2_margin=verify_margins,
            table_idx=req.table_idx,
            linear_slot_idx=req.linear_slot_idx,
        )
        logger.info(
            f"spec: k={k} p={p} m={m} accepted={m - 1}/{k} drafts={drafts} sampled={sampled}"
        )

        # ---- commit: only the tokens up to (and including) any finish reason count ----
        self.token_pool[req.table_idx, d : d + m] = out.next_tokens_gpu[p : p + m]

        def finish_state(count: int) -> None:
            n = p + count
            if n < vb.input_ids.shape[0]:
                if zero_replay:
                    pool.commit_spec_row(self._linear_slot(req), n - 1)
                    self._restore_qsa_state(req, keep_start=c0, keep_count=n)
                else:
                    if snap_slot is not None:
                        pool.copy_from(snap_slot, self._linear_slot(req))
                    self._restore_qsa_state(req)
                    self._replay(req, c0, n)

        committed = self._commit_spec_tokens(
            req, accepted, start_pos=d, spec_alloc_len=d + k, finish_state=finish_state
        )
        finished = committed < m or (committed == m and req in self.finished_reqs)
        mark("commit")

        if not finished and (committed < m or m <= k):
            # Return whatever the verify window allocated beyond what actually counts: the
            # target rejected part of the draft, request still live. A mid-window FINISH is
            # reclaimed inside _commit_spec_tokens itself, before table_idx is recycled.
            keep_cached, keep_device = spec_rollback_lengths(d, committed)
            self.cache_manager.free_spec_reject(req, keep_len=keep_device, alloc_len=d + k)
            req.cached_len = keep_cached
            req.device_len = keep_device
        mark("free_spec_reject")

        if finished:
            self.free_spec_snapshot_slot(req)
            # _commit_spec_tokens already removed and freed it (table_idx -1): filter_reqs([req])
            # would re-admit it whenever can_decode still holds (an EOS finish before
            # max_tokens), and the next spec step would verify a freed request.
            self.decode_manager.filter_reqs([])
            return True

        # Update _last_residual to last committed token for next draft chain
        # (verify_forward overwrites it; we need the residual of the last accepted token)
        last_res = getattr(model.model, "_last_residual", None)
        if last_res is not None and last_res.shape[0] >= p + committed:
            if committed > 1:
                # The draft chain only wrote the NextN layer's KV at its own positions, from
                # draft hiddens; positions d .. d+committed-2 need it from the target's hidden
                # (row j = h_{d-1+j}) and the committed token there, or later drafts attend
                # to holes / draft-state KV. Prepended to the next chain's first draft step (_take_mtp_fill).
                self._mtp_kv_rows = (
                    req.uid,
                    d,
                    last_res[p : p + committed - 1].clone(),
                    out.next_tokens_gpu[p : p + committed - 1].clone(),
                )
            model.model._last_residual = last_res[p + committed - 1 : p + committed].clone()

        if committed <= k:
            # The verify advanced every state past the rejected draft: restore the pre-verify
            # snapshot S0, then replay all but the deferred tail of the accepted tokens.
            if zero_replay:
                pool.commit_spec_row(self._linear_slot(req), p + committed - 1)
                self._restore_qsa_state(req, keep_start=c0, keep_count=p + committed)
                req.cached_len, req.device_len = keep_cached, keep_device
                mark("commit_spec_row")
                self.cache_manager.cache_req(req, finished=False)
                self.decode_manager.filter_reqs([req])
                return True
            if snap_slot is not None:
                pool.copy_from(snap_slot, self._linear_slot(req))
            self._restore_qsa_state(req)
            defer = SPEC_DEFER_MAX if k == 1 and os.getenv(DEFER_REPLAY_ENV, "1") != "0" else 0
            defer = min(defer, keep_cached - c0)
            n = keep_cached - c0 - defer
            if n > 0:
                next_residual = model.model._last_residual
                self._replay(req, c0, n)
                if defer:  # the replay stopped short of the last committed input
                    model.model._last_residual = next_residual
            req.cached_len = c0 + n
            req.device_len = keep_device
            mark("gdn_replay" if n > 0 else "deferred_replay")

        self.cache_manager.cache_req(req, finished=False)
        mark("cache_req")
        self.decode_manager.filter_reqs([req])
        return True
