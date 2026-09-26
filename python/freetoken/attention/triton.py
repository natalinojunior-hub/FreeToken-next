from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TritonCaptureData(BaseCaptureData):
    q_to_req: torch.Tensor
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    num_kv_splits: torch.Tensor
    swa_page_table: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        max_bs: int,
        max_seq_len: int,
        device: torch.device,
        *,
        num_q_heads: int,
        max_head_dim: int,
        max_kv_splits: int,
        **kwargs,
    ):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            q_to_req=torch.arange(max_bs, dtype=torch.int32, device=device),
            attn_logits=torch.empty(
                (max_bs, num_q_heads, max_kv_splits, max_head_dim),
                dtype=torch.float32,
                device=device,
            ),
            attn_lse=torch.empty(
                (max_bs, num_q_heads, max_kv_splits),
                dtype=torch.float32,
                device=device,
            ),
            num_kv_splits=torch.full(
                (max_bs,),
                max_kv_splits,
                dtype=torch.int32,
                device=device,
            ),
            **kwargs,
        )


@dataclass
class TritonMetadata(BaseAttnMetadata):
    cu_seqlens_q_gpu: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    q_to_req: torch.Tensor
    q_positions: torch.Tensor
    is_decode: bool
    prefix_lens: torch.Tensor
    max_q_len: int
    attn_logits: torch.Tensor | None = None
    attn_lse: torch.Tensor | None = None
    fi_prefill: bool = False  # planned on the FlashInfer fp8 prefill wrapper
    num_kv_splits: torch.Tensor | None = None
    swa_indices: torch.Tensor | None = None
    # (q lens, prefix lens, kv lens) per request on the host: the segmented FlashInfer prefill
    seg_lens: tuple[list[int], list[int], list[int]] | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


def _merge_lse(o1, lse1, o2, lse2):
    """Merge two attention partials over disjoint keys (FlashInfer's base-2 LSE), in fp32."""
    m = torch.maximum(lse1, lse2)
    w1, w2 = torch.exp2(lse1 - m), torch.exp2(lse2 - m)
    den = w1 + w2
    o = (o1.float() * (w1 / den).unsqueeze(-1) + o2.float() * (w2 / den).unsqueeze(-1)).to(o1.dtype)
    return o, m + torch.log2(den)


class TritonAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.capture: TritonCaptureData | None = None
        self._verify: Dict[int, dict] = {}  # spec rows -> static verify/draft buffers
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.max_kv_splits = 8
        self.prefill_tile_min_q = 128
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))
        kv_groups = list(getattr(config, "kv_cache_group_specs", lambda: ())())
        self.max_head_dim = max(
            (group.head_dim for group in kv_groups),
            default=int(getattr(config, "head_dim", 1)),
        )
        # Split-K decode must fill the GPU at batch 1: the grid is kv-head blocks x splits, so
        # 8 fixed splits left 16 CTAs on an 84-SM card walking 32K tokens each at 256K. Two
        # waves over the SMs per kv head; short sequences leave the extra splits empty.
        if self.device.type == "cuda":
            sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            kv_heads = min((g.num_kv_heads for g in kv_groups), default=1)
            self.max_kv_splits = max(self.max_kv_splits, 2 * sms // max(1, kv_heads))
        self._fi_kv_heads = min((g.num_kv_heads for g in kv_groups), default=1)
        self._fi_prefill = self._make_fi_fp8_prefill(kv_groups)
        self._seg_kv = self._make_segment_scratch(kv_groups)

    def _ensure_decode_scratch(
        self,
        metadata: TritonMetadata,
        bs: int,
        num_q_heads: int,
        head_dim: int,
    ) -> None:
        if (
            metadata.attn_logits is not None
            and metadata.attn_lse is not None
            and metadata.num_kv_splits is not None
            and metadata.attn_logits.shape[0] >= bs
            and metadata.attn_logits.shape[1] >= num_q_heads
            and metadata.attn_logits.shape[3] >= head_dim
        ):
            return
        scratch_head_dim = max(self.max_head_dim, head_dim)
        metadata.attn_logits = torch.empty(
            (bs, num_q_heads, self.max_kv_splits, scratch_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        metadata.attn_lse = torch.empty(
            (bs, num_q_heads, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        metadata.num_kv_splits = torch.full(
            (bs,),
            self.max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        from freetoken.kernel.triton import turbo_kv

        if not getattr(self.kvcache, "compressed", False) or not turbo_kv.is_rotated(
            self.kvcache.book
        ):
            return self._forward(q, k, v, layer_id, batch, attn_spec)
        # The slab holds rotated codes, so what enters the kernel rotates in and the accumulated
        # output rotates out. rotate is orthogonal, which makes that exact bookkeeping rather than
        # an approximation, and it keeps the per-tile work down to a byte gather and a multiply.
        from freetoken.kernel.triton.turbo_kv import inv_rotate, rotate

        dim = self.kvcache.head_dim

        def rot(t: torch.Tensor) -> torch.Tensor:
            return rotate(t.reshape(-1, dim)).reshape(t.shape)

        out = self._forward(rot(q), rot(k), rot(v), layer_id, batch, attn_spec)
        return inv_rotate(out.reshape(-1, dim)).reshape(out.shape)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        """Attention against the cache. ``forward`` owns the store and the rotation, so the slab
        handed to the kernels and the q/k/v they read are always in the same domain."""
        from freetoken.kernel.triton.attention import (
            decode_paged_attention,
            extend_paged_attention,
            paged_attention,
        )

        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonMetadata)

        turbo = None
        if getattr(self.kvcache, "compressed", False):
            from freetoken.kernel.triton.turbo_attn import BOOK_CODE

            k_raw, k_norm = self.kvcache.k_slab(layer_id)
            v_raw, v_norm = self.kvcache.v_slab(layer_id)
            turbo = {
                "k_norm": k_norm,
                "v_norm": v_norm,
                "cent": self.kvcache.cent_tensor,
                "book": BOOK_CODE[self.kvcache.book],
            }
            kv_heads, head_dim = k_raw.shape[1], self.kvcache.head_dim
            k_cache, v_cache = k_raw, v_raw  # element strides are the codes', not a bf16 slab's
        else:
            k_raw = self.kvcache.k_cache(layer_id)
            v_raw = self.kvcache.v_cache(layer_id)
            kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]
            k_cache = k_raw.view(-1, kv_heads, head_dim)
            v_cache = v_raw.view(-1, kv_heads, head_dim)
        assert head_dim == q.shape[-1]

        spec = attn_spec or AttentionSpec()
        block_ends = batch.mm_block_ends if spec.bidirectional_mm_blocks else None
        indices = metadata.indices
        if spec.sliding_window is not None and metadata.swa_indices is not None:
            indices = metadata.swa_indices
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5
        if metadata.is_decode and q.dtype in (torch.float16, torch.bfloat16):
            bs = metadata.indptr.numel() - 1
            self._ensure_decode_scratch(metadata, bs, q.shape[1], q.shape[-1])
            assert metadata.attn_logits is not None
            assert metadata.attn_lse is not None
            assert metadata.num_kv_splits is not None
            return decode_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                indptr=metadata.indptr,
                indices=indices,
                q_positions=metadata.q_positions,
                attn_logits=metadata.attn_logits[:bs],
                attn_lse=metadata.attn_lse[:bs],
                num_kv_splits=metadata.num_kv_splits[:bs],
                max_kv_splits=self.max_kv_splits,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                turbo=turbo,
            )
        if (
            metadata.fi_prefill
            and q.dtype == torch.bfloat16
            and spec.sliding_window is None
            and spec.sinks is None
            and spec.sm_scale is None
            and block_ends is None
        ):
            k8 = k_cache.view(torch.float8_e4m3fn).view(-1, 1, kv_heads, head_dim)
            v8 = v_cache.view(torch.float8_e4m3fn).view(-1, 1, kv_heads, head_dim)
            return self._fi_prefill.run(q, (k8, v8))
        if (
            metadata.seg_lens is not None
            and turbo is not None
            and q.dtype == torch.bfloat16
            and spec.sliding_window is None
            and spec.sinks is None
            and spec.sm_scale is None
            and block_ends is None
        ):
            return self._segmented_prefill(q, k, v, turbo, k_cache, v_cache, metadata, scale)
        if (
            (not metadata.is_decode)
            and q.dtype in (torch.float16, torch.bfloat16)
            and (q.shape[-1] <= 256 or metadata.max_q_len >= self.prefill_tile_min_q)
        ):
            return extend_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                qo_indptr=metadata.cu_seqlens_q_gpu,
                kv_indptr=metadata.indptr,
                kv_indices=indices,
                prefix_lens=metadata.prefix_lens,
                max_q_len=metadata.max_q_len,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                k_extend=k.view(q.shape[0], kv_heads, head_dim),
                v_extend=v.view(q.shape[0], kv_heads, head_dim),
                block_ends=block_ends,
                turbo=turbo,
            )
        if block_ends is not None:
            raise NotImplementedError("bidirectional multimodal blocks need the extend kernel path")
        if turbo is not None:
            raise NotImplementedError(
                "the non-grouped paged_attention kernel has no coded arm; a turbo KV slab needs "
                "the grouped decode or the extend path"
            )
        return paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            indptr=metadata.indptr,
            indices=indices,
            q_to_req=metadata.q_to_req,
            q_positions=metadata.q_positions,
            sm_scale=scale,
            sliding_window=spec.sliding_window,
            sinks=spec.sinks,
        )

    def _spec_rows_metadata(self, batch: Batch) -> bool:
        """A one-request MTP verify/draft window (<= 8 rows over a cached prefix) becomes a
        decode batch of one "request" per row, row i seeing prefix + i + 1 tokens (its own KV is
        stored first, so causality holds). The split-K decode kernel then spreads the long prefix
        over the SMs; the extend kernel runs one CTA per head over the whole prefix (24-32 CTAs
        on 84 SMs), which made the verify attention the slow part at long context."""
        reqs = batch.padded_reqs
        if (
            getattr(batch, "spec_logits_indices", None) is None
            or len(reqs) != 1
            or reqs[0].cached_len == 0
            or not 1 <= reqs[0].extend_len <= 8
            or getattr(self.kvcache, "swa_paged", False)
            or getattr(batch, "mm_block_ends", None) is not None
        ):
            return False
        req, device = reqs[0], self.device
        rows, c = req.extend_len, req.cached_len
        lens = [c + 1 + i for i in range(rows)]
        base = get_global_ctx().page_table[req.table_idx, : c + rows]
        positions = getattr(batch, "positions", None)
        if positions is None:
            positions = torch.arange(c, c + rows, dtype=torch.int64, device=device)
        indptr = torch.tensor([0, *lens], dtype=torch.int32, device=device).cumsum_(0)
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=torch.arange(rows + 1, dtype=torch.int32, device=device),
            indptr=indptr,
            indices=torch.cat([base[:n] for n in lens]),
            q_to_req=torch.arange(rows, dtype=torch.int32, device=device),
            q_positions=positions,
            is_decode=True,
            prefix_lens=indptr[1:] - 1,
            max_q_len=1,
        )
        return True

    def stage_verify(self, batch: Batch) -> None:
        """Bind a spec window's row-expanded metadata to static per-row-count buffers so a
        captured verify/draft graph reads the current window on every replay."""
        md = batch.attn_metadata
        assert isinstance(md, TritonMetadata) and md.is_decode
        rows = md.q_to_req.numel()
        v = self._verify.get(rows)
        if v is None:
            assert not torch.cuda.is_current_stream_capturing()
            width = get_global_ctx().page_table.shape[1]
            i32 = {"dtype": torch.int32, "device": self.device}
            v = self._verify[rows] = dict(
                cu_q=torch.arange(rows + 1, **i32),
                indptr=torch.zeros(rows + 1, **i32),
                indices=torch.zeros(rows * width, **i32),
                q_to_req=torch.arange(rows, **i32),
                positions=torch.zeros(rows, dtype=md.q_positions.dtype, device=self.device),
                prefix=torch.zeros(rows, **i32),
            )
            self._ensure_decode_scratch(md, rows, max(1, self.num_q_heads), self.max_head_dim)
            v.update(attn_logits=md.attn_logits, attn_lse=md.attn_lse, splits=md.num_kv_splits)
        v["indptr"].copy_(md.indptr)
        v["indices"][: md.indices.numel()].copy_(md.indices)
        v["positions"].copy_(md.q_positions)
        v["prefix"].copy_(md.prefix_lens)
        md.cu_seqlens_q_gpu, md.indptr, md.indices = v["cu_q"], v["indptr"], v["indices"]
        md.q_to_req, md.q_positions, md.prefix_lens = v["q_to_req"], v["positions"], v["prefix"]
        md.attn_logits, md.attn_lse, md.num_kv_splits = (
            v["attn_logits"],
            v["attn_lse"],
            v["splits"],
        )

    def _make_fi_fp8_prefill(self, kv_groups):
        """FlashInfer fa2 prefill over the fp8 KV codes (plain e4m3, unit norm, unrotated), built
        at construction so its workspace is in the startup VRAM account. Measured 1 layer, Tiel
        geometry, 8192 q over a 57K prefix: triton extend fp8 220 ms, FlashInfer fp8 106 ms."""
        if (
            self.device.type != "cuda"
            or getattr(self.kvcache, "book", None) != "fp8"
            or getattr(self.kvcache, "swa_paged", False)
            or self.max_head_dim not in (64, 128, 256)
            or len({g.head_dim for g in kv_groups}) > 1
        ):
            return None
        try:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        except ImportError:
            return None
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        return BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")

    # Bytes of the bf16 K+V scratch one prefix segment is dequantized into.
    _SEGMENT_SCRATCH_BYTES = 64 << 20

    def _make_segment_scratch(self, kv_groups):
        """Prefill over a coded (nvfp4 / turbo) prefix: each segment of it is dequantized into
        this bounded bf16 scratch and attended by FlashInfer, the chunk itself causally, and the
        partial outputs merge by log-sum-exp. Built with the backend, so it is in the startup
        VRAM account. The triton extend kernel it replaces ran at ~20% of tensor peak."""
        book = getattr(self.kvcache, "book", None)
        if (
            self.device.type != "cuda"
            or not getattr(self.kvcache, "compressed", False)
            or book in (None, "fp8")
            or getattr(self.kvcache, "swa_paged", False)
            or self.max_head_dim not in (64, 128, 256)
            or len({g.head_dim for g in kv_groups}) > 1
        ):
            return None
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            return None
        heads = max(g.num_kv_heads for g in kv_groups)
        per_token = 2 * heads * self.max_head_dim * 2
        rows = max(1024, self._SEGMENT_SCRATCH_BYTES // per_token // 1024 * 1024)
        shape = (rows, heads, self.max_head_dim)
        return (
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
        )

    def _segmented_prefill(self, q, k, v, turbo, k_raw, v_raw, metadata, scale):
        from flashinfer import single_prefill_with_kv_cache

        from freetoken.kernel.triton.turbo_attn import dequant_rows

        seg_k, seg_v = self._seg_kv
        rows = seg_k.shape[0]
        heads, dim = k_raw.shape[1], q.shape[-1]
        k = k.view(q.shape[0], heads, dim)
        v = v.view(q.shape[0], heads, dim)
        out = torch.empty_like(q)
        q0 = kv0 = 0
        for q_len, prefix, kv_len in zip(*metadata.seg_lens):
            if q_len:
                qr = q[q0 : q0 + q_len]
                o, lse = single_prefill_with_kv_cache(
                    qr, k[q0 : q0 + q_len], v[q0 : q0 + q_len],
                    causal=True, sm_scale=scale, return_lse=True,
                )  # fmt: skip
                for s0 in range(0, prefix, rows):
                    n = min(rows, prefix - s0)
                    slots = metadata.indices[kv0 + s0 : kv0 + s0 + n]
                    sk, sv = seg_k[:n, :heads], seg_v[:n, :heads]
                    dequant_rows(k_raw, turbo["k_norm"], turbo["cent"], slots, turbo["book"], sk)
                    dequant_rows(v_raw, turbo["v_norm"], turbo["cent"], slots, turbo["book"], sv)
                    o2, lse2 = single_prefill_with_kv_cache(
                        qr, sk, sv, causal=False, sm_scale=scale, return_lse=True
                    )
                    o, lse = _merge_lse(o, lse, o2, lse2)
                out[q0 : q0 + q_len] = o
            q0 += q_len
            kv0 += kv_len
        return out

    def reset_capture(self) -> None:
        super().reset_capture()
        self._verify = {}

    def prepare_metadata(self, batch: Batch) -> None:
        if self._spec_rows_metadata(batch):
            return
        reqs = batch.padded_reqs
        device = self.device
        ctx = get_global_ctx()
        page_table = ctx.page_table
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        num_query_tokens = sum(seqlens_q)
        is_decode = max(seqlens_q) == 1
        prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)

        indptr = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum_(0)
        if is_decode:
            cu_seqlens_q_gpu = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q_gpu = indptr
        else:
            cu_seqlens_q_gpu = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=device
            ).cumsum_(0)
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        swa_indices = None
        if getattr(self.kvcache, "swa_paged", False):
            # Global-paged SWA (naive + radix): the swa-layer gather reads swa-pool slots = full->swa
            # map of the full page-table slots (live for in-window tokens; out-of-window -> sentinel 0,
            # masked by the sliding window). Recomputed each step (incl. graph replay via
            # _point_to_capture) since the page table grows during decode.
            swa_indices = self.kvcache.translate_loc_from_full_to_swa(indices)

        q_to_req = torch.empty(num_query_tokens, dtype=torch.int32, device=device)
        offset = 0
        for req_idx, q_len in enumerate(seqlens_q):
            q_to_req[offset : offset + q_len].fill_(req_idx)
            offset += q_len

        q_positions = getattr(batch, "positions", None)
        if q_positions is None:
            q_positions = torch.zeros(num_query_tokens, dtype=torch.int64, device=device)

        fi_prefill = False
        if self._fi_prefill is not None and not is_decode and swa_indices is None:
            cpu = {"dtype": torch.int32, "device": "cpu"}
            qo_cpu = torch.tensor([0] + seqlens_q, **cpu).cumsum_(0)
            kv_cpu = torch.tensor([0] + seqlens_k, **cpu).cumsum_(0)
            self._fi_prefill.plan(
                qo_cpu,
                kv_cpu,
                indices,
                torch.ones(padded_size, **cpu),
                self.num_q_heads,
                self._fi_kv_heads,
                self.max_head_dim,
                1,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.float8_e4m3fn,
            )
            fi_prefill = True
        seg_lens = None
        if self._seg_kv is not None and not is_decode and swa_indices is None:
            seg_lens = (seqlens_q, cached_lens, seqlens_k)
        batch.attn_metadata = TritonMetadata(
            fi_prefill=fi_prefill,
            cu_seqlens_q_gpu=cu_seqlens_q_gpu,
            indptr=indptr,
            indices=indices,
            q_to_req=q_to_req,
            q_positions=q_positions,
            is_decode=is_decode,
            prefix_lens=prefix_lens,
            max_q_len=max(seqlens_q),
            swa_indices=swa_indices,
            seg_lens=seg_lens,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = TritonCaptureData.create(
            max_bs,
            max_seq_len,
            self.device,
            num_q_heads=max(1, self.num_q_heads),
            max_head_dim=max(1, self.max_head_dim),
            max_kv_splits=self.max_kv_splits,
        )
        if self._swa_capture_enabled():
            self.capture.swa_page_table = torch.zeros(
                (max_bs, max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max_bs

    def _swa_capture_enabled(self) -> bool:
        # A persistent swa-index capture buffer is needed for the global-paged SWA mode
        # (full->swa mapping translate, recomputed each replay).
        return getattr(self.kvcache, "swa_paged", False)

    def _capture_swa_indices(self) -> torch.Tensor | None:
        assert self.capture is not None
        if not self._swa_capture_enabled():
            return None
        assert self.capture.swa_page_table is not None
        return self.capture.swa_page_table.view(-1)

    def _point_to_capture(self, metadata: TritonMetadata, bs: int) -> None:
        assert self.capture is not None
        indices = self.capture.page_table.view(-1)
        self.capture.cu_seqlens_q[: bs + 1].copy_(metadata.cu_seqlens_q_gpu)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.indptr)
        total = metadata.indices.numel()
        indices[:total].copy_(metadata.indices)
        if metadata.swa_indices is not None:
            swa_indices = self._capture_swa_indices()
            assert swa_indices is not None
            swa_indices[:total].copy_(metadata.swa_indices)
            metadata.swa_indices = swa_indices
        else:
            metadata.swa_indices = None
        q_tokens = metadata.q_positions.numel()
        self.capture.positions[:q_tokens].copy_(metadata.q_positions)
        metadata.cu_seqlens_q_gpu = self.capture.cu_seqlens_q[: bs + 1]
        metadata.indptr = self.capture.cu_seqlens_k[: bs + 1]
        metadata.indices = indices
        metadata.q_to_req = self.capture.q_to_req[: metadata.q_to_req.numel()]
        metadata.q_positions = self.capture.positions[:q_tokens]
        metadata.attn_logits = self.capture.attn_logits[:bs]
        metadata.attn_lse = self.capture.attn_lse[:bs]
        metadata.num_kv_splits = self.capture.num_kv_splits[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        capture = self.capture
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=capture.cu_seqlens_q[: bs + 1],
            indptr=capture.cu_seqlens_k[: bs + 1],
            indices=capture.page_table.view(-1),
            q_to_req=capture.q_to_req[:bs],
            q_positions=capture.positions[:bs],
            is_decode=True,
            prefix_lens=capture.positions[:bs],
            max_q_len=1,
            attn_logits=capture.attn_logits[:bs],
            attn_lse=capture.attn_lse[:bs],
            num_kv_splits=capture.num_kv_splits[:bs],
            swa_indices=self._capture_swa_indices(),
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TritonMetadata)
        assert self.capture is not None and bs in self.capture_bs
        self._point_to_capture(metadata, bs)
