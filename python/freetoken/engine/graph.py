from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.attention.linear import FLAMetadata
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)

# "0" forces the eager spec-verify forward (debug / correctness reference).
VERIFY_GRAPH_ENV = "FREETOKEN_VERIFY_GRAPH"

# "0" disables the captured MTP draft-step graph (falls back to the eager draft chain).
# Default ON: +3% k1 TG at 4K and 15.7K with identical output (ft-campaign2 LEDGER, campaign 3
# block 3); "0" is the fallback if a regression shows up.
DRAFT_GRAPH_ENV = "FREETOKEN_DRAFT_GRAPH"

# "1": every draft step also runs eagerly and asserts (residual, logits, token) match the
# graph replay bitwise -- same shape of check as FREETOKEN_VERIFY_GRAPH_CHECK.
DRAFT_GRAPH_CHECK_ENV = "FREETOKEN_DRAFT_GRAPH_CHECK"


def draft_graph_enabled(model: object) -> bool:
    """Draft graphs are currently supported for models with an MTP module."""
    return getattr(model, "mtp", None) is not None and os.getenv(DRAFT_GRAPH_ENV, "1") == "1"


# Most rejected-token replays a k=1 verify window re-feeds instead of replaying them alone.
SPEC_DEFER_MAX = 2


def verify_graph_tokens(spec_mtp: int) -> tuple[int, ...]:
    """Row counts of the captured spec-verify windows, empty for an eager verify: k+1 plus
    up to SPEC_DEFER_MAX deferred-replay tokens. Only k=1 is captured."""
    if spec_mtp != 1 or os.getenv(VERIFY_GRAPH_ENV, "1") == "0":
        return ()
    return tuple(range(spec_mtp + 1, spec_mtp + 2 + SPEC_DEFER_MAX))


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, bs, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _slice] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


@dataclass
class VerifyGraph:
    """The spec-verify forward of one request over a fixed ``tokens``-row window, captured
    once. Replay restages every input below plus the attention addressing, so it always
    consumes the current window; the GDN/PLE/KV/QSA state it updates lives in the pools."""

    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    mrope_positions: torch.Tensor | None
    logits_rows: torch.Tensor  # arange(tokens): the batch's spec_logits_indices
    fla: "FLAMetadata"
    logits: torch.Tensor
    # (owner, tensor): the captured MTP residual output, rebound on the model after replay
    residual: tuple | None = None

    @property
    def tokens(self) -> int:
        return self.input_ids.shape[0]

    def bind(self, batch: Batch) -> None:
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        batch.mrope_positions = self.mrope_positions
        batch.spec_logits_indices = self.logits_rows
        batch.fla_metadata = self.fla

    def copy_from(self, batch: Batch) -> None:
        self.input_ids.copy_(batch.input_ids)
        self.positions.copy_(batch.positions)
        self.out_loc.copy_(batch.out_loc)
        if self.mrope_positions is not None:
            self.mrope_positions.copy_(batch.mrope_positions)
        self.fla.cache_indices.copy_(batch.fla_metadata.cache_indices)


@dataclass
class DraftGraph:
    """One MTP draft step (draft layer, LM head, argmax) for one request, captured once.
    Replay restages the residual, token and addressing; outputs stay at fixed addresses."""

    graph: torch.cuda.CUDAGraph
    residual: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    mrope_positions: torch.Tensor | None
    logits_rows: torch.Tensor
    # Pre-allocated OUTSIDE torch.cuda.graph(), like VerifyGraph.logits / GraphCaptureBuffer.logits:
    # the capture writes into these via copy_() instead of returning fresh tensors from inside the
    # graph. A tensor first allocated *inside* torch.cuda.graph(..., pool=self._pool) only keeps a
    # stable address across replays as long as this graph is the last one captured in that shared
    # pool (decode + verify + draft all share it); pre-allocating removes that fragile ordering
    # dependency and matches every other captured buffer in this file.
    out_residual: torch.Tensor
    logits: torch.Tensor
    token: torch.Tensor

    def bind(self, batch: Batch) -> None:
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        batch.mrope_positions = self.mrope_positions
        batch.spec_logits_indices = self.logits_rows


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        verify_tokens: tuple[int, ...] = (),
        kv_replay_check: Callable[[Batch], bool] | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.kv_replay_check = kv_replay_check
        self.device = device
        self.verify_graphs: dict[int, VerifyGraph] = {}
        self.draft: DraftGraph | None = None
        self._capture_graphs(max_seq_len, vocab_size, model)
        if self.graph_map:
            for tokens in verify_tokens:
                self._capture_verify(model, tokens, vocab_size)
            if verify_tokens and draft_graph_enabled(model):
                self._capture_draft(model, vocab_size)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (
                self.dummy_req.linear_slot_idx
                if self.dummy_req.linear_slot_idx is not None
                else self.dummy_req.table_idx
            )
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
        self._pool = pool

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify(self, model: BaseLLMModel, tokens: int, vocab_size: int) -> None:
        """Capture the spec-verify forward on the dummy request/page (sharing the decode
        graphs' pool). Any failure leaves that window size eager."""
        from freetoken.attention.linear import FLAMetadata

        dummy = self.dummy_req
        req = Req(
            input_ids=torch.zeros(tokens + 1, dtype=torch.int32),
            table_idx=dummy.table_idx,
            cached_len=1,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        req.linear_slot_idx = dummy.linear_slot_idx
        slot = dummy.linear_slot_idx if dummy.linear_slot_idx is not None else dummy.table_idx
        device = self.device
        verify = VerifyGraph(
            graph=torch.cuda.CUDAGraph(),
            input_ids=torch.zeros(tokens, dtype=torch.int32, device=device),
            positions=torch.arange(1, tokens + 1, dtype=torch.int32, device=device),
            out_loc=get_global_ctx().page_table[dummy.table_idx, 1 : tokens + 1].clone(),
            mrope_positions=(
                torch.zeros(3, tokens, dtype=torch.int32, device=device) if self.mrope else None
            ),
            logits_rows=torch.arange(tokens, device=device),
            fla=FLAMetadata(
                cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
                cache_indices=torch.full((1,), slot, dtype=torch.int32, device=device),
                has_initial_state=torch.ones(1, dtype=torch.bool, device=device),
            ),
            logits=torch.empty(tokens, vocab_size, dtype=torch.float32, device=device),
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        verify.bind(batch)
        try:
            self.attn_backend.prepare_metadata(batch)
            self.attn_backend.stage_verify(batch)
            with get_global_ctx().forward_batch(batch):
                verify.logits.copy_(model.forward())
                with torch.cuda.graph(verify.graph, pool=self._pool, stream=self.stream):
                    verify.logits.copy_(model.forward())
        except Exception as e:
            logger.warning_rank0(f"Spec-verify CUDA graph capture failed, verify stays eager: {e}")
            return
        finally:
            self._reset_moe_offload_cache()
        owner = getattr(model, "model", None)
        residual = getattr(owner, "_last_residual", None)
        verify.residual = (owner, residual) if residual is not None else None
        self.verify_graphs[tokens] = verify
        logger.info_rank0(f"Captured spec-verify CUDA graph ({tokens} tokens)")

    def _capture_draft(self, model: BaseLLMModel, vocab_size: int) -> None:
        """Capture one MTP draft step on the dummy request/page (decode graphs' pool). Any
        failure leaves the draft chain eager."""
        dummy = self.dummy_req
        req = Req(
            input_ids=torch.zeros(2, dtype=torch.int32),
            table_idx=dummy.table_idx,
            cached_len=1,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        device = self.device
        ref = getattr(model.model, "_last_residual", None)  # set by the verify capture
        if ref is None:
            return
        draft = DraftGraph(
            graph=torch.cuda.CUDAGraph(),
            residual=torch.zeros(1, ref.shape[1], dtype=ref.dtype, device=device),
            input_ids=torch.zeros(1, dtype=torch.int32, device=device),
            positions=torch.ones(1, dtype=torch.int32, device=device),
            out_loc=get_global_ctx().page_table[dummy.table_idx, 1:2].clone(),
            mrope_positions=(
                torch.zeros(3, 1, dtype=torch.int32, device=device) if self.mrope else None
            ),
            logits_rows=torch.zeros(1, dtype=torch.int64, device=device),
            out_residual=torch.zeros(1, ref.shape[1], dtype=ref.dtype, device=device),
            logits=torch.empty(1, vocab_size, dtype=torch.float32, device=device),
            token=torch.zeros(1, dtype=torch.int64, device=device),
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        draft.bind(batch)

        def step():
            r = model.mtp.forward(draft.residual, draft.input_ids, batch)
            logits = model.lm_head.forward(model.mtp.to_head(r))
            # Copy into the pre-allocated, externally-referenced buffers (see DraftGraph's
            # out_residual/logits/token docstring) instead of returning fresh tensors -- keeps
            # this graph's outputs at addresses the shared capture pool cannot hand to a later
            # graph, exactly like VerifyGraph.logits.copy_(model.forward()) above.
            draft.out_residual.copy_(r)
            draft.logits.copy_(logits)
            draft.token.copy_(torch.argmax(logits, dim=-1))

        try:
            self.attn_backend.prepare_metadata(batch)
            self.attn_backend.stage_verify(batch)
            with get_global_ctx().forward_batch(batch):
                step()
                with torch.cuda.graph(draft.graph, pool=self._pool, stream=self.stream):
                    step()
        except Exception as e:
            logger.warning_rank0(f"MTP draft CUDA graph capture failed, draft stays eager: {e}")
            return
        finally:
            self._reset_moe_offload_cache()
        self.draft = draft
        logger.info_rank0("Captured MTP draft-step CUDA graph")

    def replay_draft(
        self, batch: Batch, residual: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One draft step on the prepared 1-row ``batch``: (residual, logits, token)."""
        d = self.draft
        assert d is not None
        d.residual.copy_(residual)
        d.input_ids.copy_(token)
        d.positions.copy_(batch.positions)
        d.out_loc.copy_(batch.out_loc)
        if d.mrope_positions is not None:
            d.mrope_positions.copy_(batch.mrope_positions)
        self.attn_backend.stage_verify(batch)
        d.graph.replay()
        return d.out_residual, d.logits, d.token

    def _is_verify(self, batch: Batch) -> bool:
        return (
            batch.is_prefill
            and batch.spec_logits_indices is not None
            and batch.size == 1
            and batch.input_ids.shape[0] in self.verify_graphs
            and batch.reqs[0].cached_len > 0
            and batch.mm_embeds is None
        )

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        check = getattr(self, "kv_replay_check", None)
        if check is not None and not check(batch):
            return False
        if batch.is_decode:
            return batch.size <= self.max_graph_bs
        return self._is_verify(batch)

    def attach_kv_replay_check(self, check: Callable[[Batch], bool] | None) -> None:
        """Attach an explicit page-residency gate; ``None`` preserves legacy behavior."""
        self.kv_replay_check = check

    def _replay_verify(self, batch: Batch) -> torch.Tensor:
        v = self.verify_graphs[batch.input_ids.shape[0]]
        v.copy_from(batch)
        self.attn_backend.stage_verify(batch)
        v.graph.replay()
        if v.residual is not None:
            owner, residual = v.residual
            owner._last_residual = residual
        return v.logits

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if not batch.is_decode:
            return self._replay_verify(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self.verify_graphs = {}
        self.draft = None
        self._pool = None
        gc.collect()
