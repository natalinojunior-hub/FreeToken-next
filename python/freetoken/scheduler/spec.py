"""Native MTP (multi-token prediction) speculative decode: draft chain + verify + accept/reject.

Single-request only. The verify step runs as its own phase="prefill" Batch (extend_len=k+1,
k = config.spec_mtp), which reuses the already extend_len-aware FLA/PLE prefill paths and never
touches CUDA graph capture (can_use_cuda_graph is decode-only). Bypasses scheduler._forward /
_process_last_data entirely: their single-token-per-request write logic cannot carry more than
one accepted token per step. Design history: docs/freetoken-next/EXPERIMENTS.md EXP-023/024/025.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, Req
from freetoken.engine.spec import accept_drafts
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .scheduler import Scheduler  # noqa: F401  (self-typing only)

logger = init_logger(__name__)


def _spec_mrope_positions(req: Req, cached_len: int, device_len: int, device: torch.device) -> torch.Tensor:
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
            slot = self.engine.linear_state_pool.alloc(1)[0]
            self._spec_snapshot_slots[req.uid] = slot
        return slot

    def free_spec_snapshot_slot(self, req: Req) -> None:
        slot = self._spec_snapshot_slots.pop(req.uid, None)
        if slot is not None:
            self.engine.linear_state_pool.free([slot])

    @staticmethod
    def _linear_slot(req: Req) -> int:
        """GDN/PLE state slot: the hybrid-radix live slot when allocated, else table_idx
        (naive keeps the old keying -- same rule as build_fla_metadata's gdn_slot)."""
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def _commit_spec_tokens(self, req: Req, tokens: List[int], start_pos: int) -> int:
        """Append tokens one at a time, applying _process_last_data's per-token EOS/stop
        /length finish logic. ``start_pos`` is tokens[0]'s device-table position (== d in
        run_spec_step): cached_len/device_len advance ONE token at a time here, exactly like
        complete_one(), rather than being pre-set to the whole window's end -- otherwise
        can_decode reflects the window's final length for every token in it, not each
        token's own position, and hit_length fires (or doesn't) at the wrong offset.
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
                    if finished else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(DetokenizeMsg(
                    uid=req.uid, next_token=next_token, finished=finished,
                    finish_reason=finish_reason, matched_stop=matched_stop,
                    stop_strs=req.sampling_params.stop_strs or None,
                ))
                if finished:
                    finished_now = True
                    break
            self.send_result(reply)
            if finished_now:
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
        k = min(self.spec_mtp, req.remain_len - 1)
        if k <= 0:
            return False

        d = req.device_len  # invariant: d == req.cached_len + 1
        model = self.engine.model
        mtp = model.mtp

        # ---- draft chain: k autoregressive steps through the draft head's own QSA slot ----
        r_prev = model.model._last_residual[-1:].clone()
        tok_prev = self.token_pool[req.table_idx, d - 1 : d]
        drafts: List[int] = []
        for i in range(k):
            # prepare_metadata (e.g. qsa_sparse) reads req.cached_len/device_len for
            # seqlens_k/extend_len; at i >= 1 the draft's own query must see its own prior
            # draft-step KV, which needs these advanced per step, not left at the entry value.
            req.cached_len, req.device_len = d - 1 + i, d + i
            db = Batch(reqs=[req], phase="prefill")
            db.padded_reqs = [req]
            db.positions = torch.tensor([d - 1 + i], dtype=torch.int32, device=self.device)
            if self._model_is_mrope:
                db.mrope_positions = _spec_mrope_positions(
                    req, d - 1 + i, d + i, self.device
                )
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

        # ---- snapshot linear state (GDN conv+recurrent+PLE ctx) before verify mutates it ----
        pool = self.engine.linear_state_pool
        snap_slot = None
        if pool is not None:
            snap_slot = self._spec_snapshot_slot(req)
            pool.copy_from(self._linear_slot(req), snap_slot)

        # ---- verify: one prefill-phase Batch over [d-1, d+k) ----
        req.device_len = d + k
        vb = Batch(reqs=[req], phase="prefill")
        fi = self._prepare_batch(vb)
        vb.spec_logits_indices = torch.arange(k + 1, device=self.device)
        vb.input_ids = self.token_pool[fi.input_tuple]
        out = self.engine.forward_batch(vb, fi.sample_args)
        out.copy_done_event.synchronize()
        sampled = out.next_tokens_cpu.tolist()
        accepted = accept_drafts(sampled, drafts)
        m = len(accepted)
        logger.info(f"spec: k={k} m={m} accepted={m - 1}/{k} drafts={drafts} sampled={sampled}")

        # ---- commit: only the tokens up to (and including) any finish reason count ----
        self.token_pool[req.table_idx, d : d + m] = out.next_tokens_gpu[:m]
        committed = self._commit_spec_tokens(req, accepted, start_pos=d)
        finished = committed < m or (committed == m and req in self.finished_reqs)

        if committed < m or m <= k:
            # Return whatever the verify window allocated beyond what actually counts: either
            # a mid-window finish truncated it, or the target rejected part of the draft.
            keep_len = d - 1 + committed
            self.cache_manager.free_spec_reject(req, keep_len=keep_len, alloc_len=d + k)
            req.cached_len = keep_len
            req.device_len = keep_len + (0 if finished else 1)

        if finished:
            self.free_spec_snapshot_slot(req)
            self.decode_manager.filter_reqs([req])
            return True

        if committed <= k:
            # Partial/full reject, request still live: undo GDN/PLE state and re-forward the
            # committed tokens as a plain prefill to re-derive the next chain's seed residual
            # (the draft KV wrote garbage past keep_len; _last_residual only holds the last
            # forward's rows).
            if snap_slot is not None:
                pool.copy_from(snap_slot, self._linear_slot(req))
            start = d - 1
            rb = Batch(reqs=[req], phase="prefill")
            # The replay re-forwards ALREADY-COMMITTED positions, so it needs the pre-commit
            # window lengths -- but they must not survive it: keep_len/keep_len+1 (set above)
            # are this step's true post-commit state. Leaving the replay's lengths in place
            # rewinds the request one token behind input_ids, so the next step re-drafts the
            # previous token AND re-applies position start to a GDN state that already has it
            # (build_fla_metadata sets has_initial_state = cached_len > 0).
            keep_cached, keep_device = req.cached_len, req.device_len
            req.cached_len = start
            req.device_len = start + committed
            rfi = self._prepare_batch(rb)
            rb.input_ids = self.token_pool[rfi.input_tuple]
            self.engine.forward_batch(rb, rfi.sample_args)  # sampled token discarded
            req.cached_len, req.device_len = keep_cached, keep_device

        self.cache_manager.cache_req(req, finished=False)
        self.decode_manager.filter_reqs([req])
        return True
