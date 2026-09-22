"""Native MTP (multi-token prediction) speculative decode: draft chain + verify + accept/reject.

Single-request only. The verify step runs as its own phase="prefill" Batch (extend_len=k+1,
k = config.spec_mtp), which reuses the already extend_len-aware FLA/PLE prefill paths and never
touches CUDA graph capture (can_use_cuda_graph is decode-only). Bypasses scheduler._forward /
_process_last_data entirely: their single-token-per-request write logic cannot carry more than
one accepted token per step. Design history: docs/freetoken-next/EXPERIMENTS.md EXP-023/024/025.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, Req
from freetoken.engine.spec import accept_drafts, spec_rollback_lengths
from freetoken.scheduler.adaptive_mtp import (
    AdaptiveMTPController,
    AdaptiveMTPConfig,
    resolve_adaptive_k,
)
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .scheduler import Scheduler  # noqa: F401  (self-typing only)

logger = init_logger(__name__)

SPEC_TIMING_ENV = "FREETOKEN_DEBUG_SPEC_TIMING"


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

    def warmup_mtp_draft_kv(self, req: Req) -> None:
        """Populate the draft head's KV cache over the prefill window.

        Resolves EXP-025 / EXP-050 gap: warms up the draft KV slot over prompt tokens
        so initial draft steps have full context instead of starting cold with ~0% accept rate.
        """
        if self.spec_mtp <= 0:
            return
        model = getattr(self.engine, "model", None)
        mtp = getattr(model, "mtp", None)
        if mtp is None:
            return
        r_last = getattr(getattr(model, "model", None), "_last_residual", None)
        if r_last is None or r_last.numel() == 0:
            return

        d = req.device_len
        win_len = min(r_last.shape[0], d)
        if win_len <= 0:
            return

        start_pos = d - win_len
        tok_window = self.token_pool[req.table_idx, start_pos:d]
        r_window = r_last[-win_len:]

        old_cached, old_device = req.cached_len, req.device_len
        try:
            req.cached_len, req.device_len = start_pos, d
            wb = Batch(reqs=[req], phase="prefill")
            wb.padded_reqs = [req]
            wb.positions = torch.arange(start_pos, d, dtype=torch.int32, device=self.device)
            if self._model_is_mrope:
                wb.mrope_positions = _spec_mrope_positions(req, start_pos, d, self.device)
            wb.out_loc = self.engine.page_table[req.table_idx, start_pos:d]
            wb.input_ids = tok_window
            # Do NOT set spec_logits_indices: we want full prefill attention to populate
            # the draft head's QSA slot for ALL prefill positions, not just the last token.
            self.engine.attn_backend.prepare_metadata(wb)
            with self.engine.ctx.forward_batch(wb):
                mtp.forward(r_window, tok_window, wb)
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

    def _restore_qsa_state(self, req: Req) -> None:
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
        ring_buf[req.table_idx].copy_(snap_ring)
        scratch_base = getattr(kv, "_cmp_scratch_base", 0)
        kv._cmp_k_buffer[:, scratch_base + req.table_idx].copy_(snap_scratch)

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
        self, req: Req, tokens: List[int], start_pos: int, spec_alloc_len: int
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
                if keep_device < spec_alloc_len:
                    # free_spec_reject's keep_len is an EXCLUSIVE boundary (page_ceil(keep_len)
                    # must exclude the page containing it): keep_device respects the standing
                    # complete_one-style lag (keep_cached + 1 == keep_device).
                    self.cache_manager.free_spec_reject(
                        req, keep_len=keep_device, alloc_len=spec_alloc_len
                    )
                # cache_req(finished=True) (via _free_req_resources) uses req.cached_len as the
                # "KV already written" boundary, same as normal decode's complete_one lag -- but
                # that lag is only real for the k+1'th (bonus/correction) token, which is a fresh
                # sample never fed through the verify forward. Every other committed token WAS
                # fed as the verify batch's own input (drafts at positions [d, d+k)) and already
                # has written KV. Finishing on one of those (committed <= k) with cached_len left
                # at keep_cached excludes that token's own page from both insert_prefix and the
                # tail-free -- and free_spec_reject starts its range one page later -- so a page
                # landing exactly on that boundary is neither freed nor retained: orphaned. This
                # produced the live 'free_pages + cache_pages != num_pages' idle-check crash.
                k = spec_alloc_len - start_pos
                req.cached_len = start_pos + min(committed, k)
                req.device_len = req.cached_len + 1
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished_reqs.add(req)
        return committed

    def run_spec_step(self) -> bool:
        """Run one speculative decode step for the single eligible request. Returns True if
        it ran (the caller should skip its own _schedule_next_batch/_forward this iteration)."""
        req = self._spec_eligible_req()
        if req is None:
            return False
        from freetoken.scheduler.adaptive_mtp import resolve_adaptive_k

        cache = getattr(self.engine, "moe_offload_cache", None)
        slots_per_layer = None
        if cache is not None and getattr(cache, "num_layers", 0) > 0:
            slots_per_layer = cache.cache_size / cache.num_layers
        cfg = getattr(self.engine.model, "_config", None) or getattr(
            self.engine.model, "config", None
        )
        top_k = getattr(cfg, "num_experts_per_tok", 10)

        k = resolve_adaptive_k(
            req,
            self.spec_mtp,
            moe_slots_per_layer=slots_per_layer,
            top_k_experts=top_k,
        )
        if k <= 0:
            return False

        debug_timing = os.getenv(SPEC_TIMING_ENV, "0") == "1"
        _t0 = time.perf_counter()

        def mark(stage: str) -> None:
            if debug_timing:
                torch.cuda.synchronize(self.device)
                nonlocal _t0
                now = time.perf_counter()
                print(f"[spec-timing] k={k} {stage} {now - _t0:.4f}s", flush=True)
                _t0 = now

        d = req.device_len  # invariant: d == req.cached_len + 1
        model = self.engine.model
        mtp = model.mtp

        # Snapshot QSA pending ring and scratch cmp buffer before draft chain mutates them
        self._snapshot_qsa_state(req)
        self._snapshot_ple_state(req)

        # ---- snapshot linear state before the draft chain mutates it ----
        pool = self.engine.linear_state_pool
        snap_slot = None
        if pool is not None:
            snap_slot = self._spec_snapshot_slot(req)
            pool.copy_from(self._linear_slot(req), snap_slot)
        residual_snapshot = model.model._last_residual[-1:].clone()

        # ---- draft chain: k autoregressive steps through the draft head's own QSA slot ----
        r_prev = model.model._last_residual[-1:].clone()
        tok_prev = self.token_pool[req.table_idx, d - 1 : d]
        drafts: List[int] = []

        # Confidence gating: run first draft step, check top-1 prob, skip if low confidence
        db = Batch(reqs=[req], phase="prefill")
        db.padded_reqs = [req]
        db.positions = torch.tensor([d - 1], dtype=torch.int32, device=self.device)
        if self._model_is_mrope:
            db.mrope_positions = _spec_mrope_positions(req, d - 1, d, self.device)
        db.out_loc = self.engine.page_table[req.table_idx, d - 1 : d]
        db.input_ids = tok_prev
        db.spec_logits_indices = torch.arange(1, device=self.device)
        self.engine.attn_backend.prepare_metadata(db)
        with self.engine.ctx.forward_batch(db):
            r_prev = mtp.forward(r_prev, tok_prev, db)
            logits = model.lm_head.forward(mtp.to_head(r_prev))
        controller = getattr(self, "_adaptive_mtp_controller", None)
        if controller is not None:
            top1_prob = torch.softmax(logits, dim=-1).max().item()
            if top1_prob < controller.config.min_draft_prob:
                return False

        tok_prev = torch.argmax(logits, dim=-1)
        drafts.append(int(tok_prev.item()))
        self.token_pool[req.table_idx, d] = tok_prev

        # Continue draft chain for remaining k-1 steps
        for i in range(1, k):
            # prepare_metadata (e.g. qsa_sparse) reads req.cached_len/device_len for
            # seqlens_k/extend_len; at i >= 1 the draft's own query must see its own prior
            # draft-step KV, which needs these advanced per step, not left at the entry value.
            req.cached_len, req.device_len = d - 1 + i, d + i
            db = Batch(reqs=[req], phase="prefill")
            db.padded_reqs = [req]
            db.positions = torch.tensor([d - 1 + i], dtype=torch.int32, device=self.device)
            if self._model_is_mrope:
                db.mrope_positions = _spec_mrope_positions(req, d - 1 + i, d + i, self.device)
            db.out_loc = self.engine.page_table[req.table_idx, d - 1 + i : d + i]
            db.input_ids = tok_prev
            db.spec_logits_indices = torch.arange(1, device=self.device)
            self.engine.attn_backend.prepare_metadata(db)
            with self.engine.ctx.forward_batch(db):
                r_prev = mtp.forward(r_prev, tok_prev, db)
                logits = model.lm_head.forward(mtp.to_head(r_prev))
            tok_prev = torch.argmax(logits, dim=-1)
            drafts.append(int(tok_prev.item()))
            self.token_pool[req.table_idx, d + i] = tok_prev
        req.cached_len, req.device_len = d - 1, d
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
        fi = self._prepare_batch(vb)
        vb.spec_logits_indices = torch.arange(k + 1, device=self.device)
        vb.input_ids = self.token_pool[fi.input_tuple]
        mark("verify_prepare_batch")
        out = self.engine.forward_batch(vb, fi.sample_args)
        out.copy_done_event.synchronize()
        mark("verify_forward")
        sampled = out.next_tokens_cpu.tolist()
        accepted = accept_drafts(sampled, drafts)
        m = len(accepted)
        logger.info(f"spec: k={k} m={m} accepted={m - 1}/{k} drafts={drafts} sampled={sampled}")

        # ---- commit: only the tokens up to (and including) any finish reason count ----
        self.token_pool[req.table_idx, d : d + m] = out.next_tokens_gpu[:m]
        committed = self._commit_spec_tokens(req, accepted, start_pos=d, spec_alloc_len=d + k)
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
            self.decode_manager.filter_reqs([req])
            return True

        # Update _last_residual to last committed token for next draft chain
        # (verify_forward overwrites it; we need the residual of the last accepted token)
        last_res = getattr(model.model, "_last_residual", None)
        if last_res is not None and last_res.shape[0] >= committed:
            model.model._last_residual = last_res[committed - 1 : committed].clone()

        if committed <= k:
            checkpoints = None
            target_step = committed - 1
            if checkpoints is not None and target_step in checkpoints and pool is not None:
                # Phase 10 Pillar 2 Zero-Replay GDN: restore recurrent + conv states from verify checkpoint
                # ponytail: single-slot restore is O(num_layers); add batched restore if spec decode expands beyond batch=1.
                step_ckpts = checkpoints[target_step]
                slot = self._linear_slot(req)
                for li, (rec_s, conv_s) in step_ckpts.items():
                    pool.recurrent_states[li, slot].copy_(rec_s)
                    pool.conv_states[li, slot].copy_(conv_s)
                # Seed residual for the next draft chain from verify forward
                last_res = getattr(model.model, "_last_residual", None)
                if last_res is not None and last_res.shape[0] >= committed:
                    model.model._last_residual = last_res[committed - 1 : committed].clone()
                self._restore_qsa_state(req)
                req.cached_len = keep_cached
                req.device_len = keep_device
                mark("zero_replay_gdn")
            else:
                # Fallback replay path if checkpoints are not available
                if snap_slot is not None:
                    pool.copy_from(snap_slot, self._linear_slot(req))
                self._restore_qsa_state(req)
                start = d - 1
                phase = "decode" if committed == 1 else "prefill"
                rb = Batch(reqs=[req], phase=phase)
                if rb.is_decode:
                    rb.padded_reqs = [req]
                req.cached_len = start
                req.device_len = start + committed
                rfi = self._prepare_batch(rb, skip_alloc=True)
                rb.input_ids = self.token_pool[rfi.input_tuple]
                rout = self.engine.forward_batch(rb, rfi.sample_args)
                rout.copy_done_event.synchronize()
                req.cached_len = keep_cached
                req.device_len = keep_device
                mark("gdn_replay")

        self.cache_manager.cache_req(req, finished=False)
        mark("cache_req")
        self.decode_manager.filter_reqs([req])
        return True
