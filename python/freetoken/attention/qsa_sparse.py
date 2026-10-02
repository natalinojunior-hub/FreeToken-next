"""Qwen3.8-Flash-Next QSA compressed-block sparse attention backend.

Serves ``AttnType.QSA`` over ``kvcache/qsa_pool.py``: paged GQA K/V for the 12 full-attention
layers, a compressed index-key slab holding one key per ``index_ratio`` tokens, and a
per-request pending ring for the group a forward leaves open. The 36 GDN layers never reach
this backend, and the model has no dense attention layer, so :meth:`forward` is not served --
the only entry point is :meth:`qsa_forward` (``models/qwen4_exp/attention.py``).

One QSA layer's forward, all ragged over ``[T, ...]`` metadata:

1. store K/V at ``batch.out_loc``;
2. pool each row's closing group (members at positions >= ``cached_len`` come from this
   forward's raw index keys, the older ones from the pending ring), zero-centered rmsnorm it
   and rope it at the group's first position, then scatter it into the slab row
   ``out_loc // index_ratio`` (rows whose group does not close land on the request's scratch
   row and are never read);
3. store this forward's last ``ring_capacity`` raw index keys per request into the ring;
4. norm+rope the indexer queries at their own positions;
5. score every COMPLETE visible block (``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)``,
   clamped to ``kvlen // index_ratio`` -- slab rows are never cleared, so stale rows must stay
   unreachable), take the top ``index_budget // index_ratio`` blocks, expand them to token
   indices plus the causal tail of the open group;
6. attend to exactly those tokens.

Addressing: the engine pins ``page_size == 64`` (this backend's ``page_sizes``), so a group of
``index_ratio`` tokens never straddles a page and ``block_table[req, p] = page_table[req, p *
64] // 64`` names both the K/V page and, viewed as ``page_size // index_ratio`` compressed
rows, the block's slab page. Decode stages that table plus the live lengths and table_idx into
static buffers (``prepare_for_replay``) so the whole path is CUDA-graph capturable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Block-score transient budget (vLLM's number): the fp32 [rows, n_blocks] logits tile is
# 256 KB per row at a 1M-token context, so a long prefill must be scored in row chunks.
_LOGITS_WORKSPACE_BYTES = 128 << 20


TORCH_TOPK_ENV = "FREETOKEN_QSA_TORCH_TOPK"
QSA_TIMING_ENV = "FREETOKEN_DEBUG_QSA_TIMING"


# Up to this many query rows read the RAM tier zero-copy (decode, MTP verify, short spec
# windows); longer eager forwards stage it (see _plan_host_staging).
_ZERO_COPY_MAX_ROWS = 8


def _resolve_block_topk() -> Callable | None:
    """The in-repo Triton block top-k, or None to fall back on torch.topk."""
    if os.getenv(TORCH_TOPK_ENV, "0") == "1":
        logger.info(f"qsa_sparse block top-k: torch.topk ({TORCH_TOPK_ENV}=1)")
        return None
    try:
        from freetoken.kernel.triton.qsa import qsa_block_topk
    except Exception as exc:
        logger.info(f"qsa_sparse block top-k: torch.topk (triton unavailable: {exc})")
        return None
    logger.info("qsa_sparse block top-k: triton qsa_block_topk")
    return qsa_block_topk


_TXN_TIMING = os.environ.get("FREETOKEN_DEBUG_QSA_TXN_TIMING", "0") == "1"
_txn_acc: dict[str, list[float]] = {}
_NEVER_KEPT = 1 << 62  # logical position of rows that rollback always restores


def _txn_timed(fn):
    """Host wall time per call of a transaction method (includes its blocking syncs)."""
    if not _TXN_TIMING:
        return fn

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            acc = _txn_acc.setdefault(fn.__name__, [0.0, 0])
            acc[0] += time.perf_counter() - start
            acc[1] += 1
            if acc[1] % 64 == 0:
                print(
                    f"[qsa-txn-timing] {fn.__name__} mean {acc[0] / acc[1] * 1e3:.3f} ms "
                    f"calls {acc[1]}",
                    flush=True,
                )

    return wrapper


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:        bool
    last_indices:     torch.Tensor  # gpu
    qo_indptr_cpu:    torch.Tensor  # cpu pinned int32 [bs+1]
    kv_len_cpu:       torch.Tensor  # cpu pinned int32 [bs]
    # Ragged per-token / per-request addressing. Decode defers these to the static graph
    # buffers (prepare_for_replay) or to a lazy eager snapshot at the first QSA layer.
    token_to_req:     torch.Tensor | None = None  # [T] int32
    cu_seqlens:       torch.Tensor | None = None  # [bs+1] int32
    seq_lens:         torch.Tensor | None = None  # [bs] int32, device_len
    ring_slots:       torch.Tensor | None = None  # [bs] int32, Req.table_idx
    block_table:      torch.Tensor | None = None  # [bs, W//page_size] int32, physical page ids
    # Per-forward scatter plans, built once by the first QSA layer and reused by the rest.
    # positions is bound here (not in prepare_metadata) because a capture batch has none yet.
    cmp_rows:         torch.Tensor | None = None  # [T] int32, compressed slab destination
    ring_rows:        torch.Tensor | None = None  # [T] int32, flat ring row or -1
    last_slot:        int = -1  # QSA slot of the latest qsa_forward on this metadata
    positions:        torch.Tensor | None = None  # [T] int32, logical query positions
    # mrope only, built once per forward: row r of the caches is token r's [cos | sin] (queries) or its group start's (keys)
    rope_rows:        torch.Tensor | None = None  # [T] int32 arange
    q_rope_cache:     torch.Tensor | None = None  # [T, rotary_dim] float32
    k_rope_cache:     torch.Tensor | None = None  # [T, rotary_dim] float32
    # KV RAM tier, eager multi-row forwards only: RAM pages this forward reads (as tier
    # offsets) and the block table re-pointing them at the device staging slab.
    phys_loc:         torch.Tensor | None = None  # [T] out_loc through the RAM tier page map
    host_pages:       torch.Tensor | None = None  # [n] int64, sorted page - num_device_pages
    staged_table:     torch.Tensor | None = None  # [bs, W//page_size] int32
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None, "qsa_sparse backend needs ModelConfig.qwen4_args"
        self.head_dim = config.head_dim
        self.index_heads = args.index_n_heads
        self.token_topk = args.index_budget
        self.kvcache = get_global_ctx().kv_cache
        assert hasattr(self.kvcache, "cmp_k_cache"), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )
        self.device = self.kvcache.device
        self.dtype = self.kvcache.dtype
        self.index_head_dim = self.kvcache.index_head_dim
        self.ratio = self.kvcache.index_ratio
        self.ring_capacity = self.kvcache.ring_capacity
        self.page_size = get_global_ctx().page_size
        assert self.page_size % self.ratio == 0, (
            f"QSA needs page_size ({self.page_size}) divisible by index_ratio ({self.ratio})"
        )
        self.cmp_page_size = self.page_size // self.ratio
        self.block_topk = self.token_topk // self.ratio
        self.select_width = self.token_topk + self.ratio - 1
        assert self.token_topk % self.ratio == 0, "QSA budget must be a whole number of blocks"
        # The sparse attend kernel bakes 1/sqrt(head_dim) into its exp2 scale.
        assert config.attn_sm_scale in (None, self.head_dim**-0.5), (
            "qsa_sparse serves the default 1/sqrt(head_dim) attention scale only"
        )
        # QSA layer -> index slab slot, in sparse-layer order (the pool's own convention).
        group = self._qsa_group(config)
        self._idx_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self.rotary_config = group.rotary_config
        self._index_cos_sin: torch.Tensor | None = None
        self._section_table: torch.Tensor | None = None
        if group.rotary_config.mrope_section is not None:
            from freetoken.layers.rotary import build_section_table

            self._section_table = build_section_table(
                tuple(group.rotary_config.mrope_section), group.rotary_config.mrope_layout
            ).to(self.device)

        self._block_topk_kernel = _resolve_block_topk()
        self._turbo_consts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # decode staging (static buffers under CUDA graphs; eager decode snapshots per step)
        self._graph: dict[str, torch.Tensor] = {}
        # static addressing for the spec-verify graph (stage_verify), keyed by window length
        self._verify: dict = {}
        self.capture_bs: List[int] = []
        # Speculative QSA writes are transactional.  The scheduler already journals the
        # linear state and pending ring; this journal covers the persistent index/KV slabs
        # that a group-closing verify can otherwise overwrite with rejected draft tokens.
        self._spec_txns: dict[int, dict] = {}
        self._spec_preimages: dict[int, dict] = {}

    def begin_spec_txn(self, table_idx: int) -> None:
        table_idx = int(table_idx)
        if table_idx in self._spec_txns:
            raise RuntimeError(f"QSA speculative transaction already active for table {table_idx}")
        self._spec_txns[table_idx] = {"cmp": {}, "kv": {}, "rope": {}}

    @_txn_timed
    def commit_spec_txn(self, table_idx: int) -> None:
        """Drop a fully accepted journal before its table slot can be reused."""
        self._spec_txns.pop(int(table_idx), None)

    def spec_txn_active(self, table_idx: int) -> bool:
        return int(table_idx) in self._spec_txns

    def _journal_compressed(self, txn, table_idx, physical, logical_positions, live, activate):
        """Pre-journal for the compressed (turbo) pool, batched over layers, with no host sync
        unless a write lands in the RAM tier. Positions are distinct, so rows never repeat; the
        duplicate scratch rows of ``cmp_rows`` hold one preimage, so gather/scatter is exact."""
        pool = self.kvcache._pool
        device = physical.device
        scratch = txn["cmp_scratch"]
        closing = physical % self.ratio == self.ratio - 1
        cmp_rows = torch.where(closing, physical // self.ratio, scratch)
        txn["cmp_logical"] = {}
        txn["cmp_logical_t"] = torch.where(closing, logical_positions, _NEVER_KEPT)
        dense_t, slot_t = self._fast_layer_index(device)
        phys_live = physical[:live]
        dev, dev_mask = phys_live, None
        host_l: list[int] = []
        if getattr(self.kvcache, "page_map", None) is not None:
            device_tokens = int(self.kvcache.num_device_pages) * self.page_size
            dev_mask = phys_live < device_tokens
            dev = torch.where(dev_mask, phys_live, 0)  # host rows read row 0, never restored
            host_l = [p - device_tokens for p in phys_live.tolist() if p >= device_tokens]
        buf = self.kvcache._cmp_k_buffer
        txn["fast"] = {
            "dev": dev,
            "dev_mask": dev_mask,
            "kv": tuple(
                t[dense_t[:, None], dev]
                for t in (pool._k_codes, pool._k_norm, pool._v_codes, pool._v_norm)
            ),
            "cmp_rows": cmp_rows,
            "cmp": buf[slot_t[:, None], cmp_rows],
        }
        if host_l:
            host_cpu = torch.tensor(host_l, dtype=torch.long)
            kvh = getattr(self.kvcache, "_kv_host", None)
            if kvh is not None:
                dense_cpu = dense_t.cpu()[:, None]
                hk = kvh[0].reshape(kvh.shape[1], -1, *kvh.shape[4:])
                hv = kvh[1].reshape(kvh.shape[1], -1, *kvh.shape[4:])
                txn["bulk_kv"].append(
                    ("host", dense_cpu, host_cpu, hk[dense_cpu, host_cpu], hv[dense_cpu, host_cpu])
                )
            else:
                for layer_id in self._idx_slot:
                    host_kv = self.kvcache.host_kv(layer_id)
                    if host_kv is None:
                        continue
                    hk = host_kv[0].reshape(-1, *host_kv[0].shape[2:])
                    hv = host_kv[1].reshape(-1, *host_kv[1].shape[2:])
                    txn["bulk_kv"].append(
                        (
                            "host",
                            pool._dense(layer_id),
                            host_cpu,
                            hk.index_select(0, host_cpu),
                            hv.index_select(0, host_cpu),
                        )
                    )
        rope = getattr(self.kvcache, "_rope_positions", None)
        if rope is not None:
            txn["bulk_rope"].append((phys_live, rope.index_select(0, phys_live)))
        txn["graph_safe"] = True
        if not activate:
            self._spec_preimages[table_idx] = txn

    @_txn_timed
    def prepare_spec_txn(
        self,
        table_idx: int,
        out_loc: torch.Tensor,
        live_count: int | None = None,
        *,
        activate: bool = True,
        logical_positions: torch.Tensor | None = None,
    ) -> None:
        """Pre-journal the verify write-set with vectorized device copies."""
        table_idx = int(table_idx)
        if activate:
            self.begin_spec_txn(table_idx)
            txn = self._spec_txns[table_idx]
            preimage = self._spec_preimages.pop(table_idx, None)
            if preimage is not None:
                self._restore_bulk(preimage)
                self._spec_txns[table_idx] = preimage
                return
        else:
            txn = {"cmp": {}, "kv": {}, "rope": {}}
        txn["bulk_kv"], txn["bulk_cmp"], txn["bulk_rope"] = [], [], []
        physical = self._physical_loc(out_loc).long()
        if logical_positions is None:
            logical_positions = torch.arange(out_loc.numel(), device=out_loc.device)
        logical_positions = logical_positions.to(device=out_loc.device, dtype=torch.long)
        device = out_loc.device
        pool = self.kvcache._pool
        scratch = int(self.kvcache.cmp_scratch_base + table_idx)
        txn["cmp_scratch"] = scratch
        n = int(out_loc.numel())
        live = n if live_count is None else max(0, min(int(live_count), n))
        if getattr(pool, "compressed", False):
            self._journal_compressed(txn, table_idx, physical, logical_positions, live, activate)
            return
        no_ram_tier = getattr(self.kvcache, "page_map", None) is None and all(
            self.kvcache.host_kv(layer_id) is None for layer_id in self._idx_slot
        )
        if no_ram_tier:
            # Every row is on device and positions are distinct, so the whole write set is
            # derived on the device with no host sync; duplicate scratch rows hold the same
            # preimage, so gather/scatter over them is exact.
            closing = physical % self.ratio == self.ratio - 1
            cmp_rows = torch.where(closing, physical // self.ratio, scratch)
            cmp_logical_t = torch.where(closing, logical_positions, _NEVER_KEPT)
            txn["cmp_logical"] = {}
            phys_live = dev = physical[:live]
            host = physical[:0]
        else:
            cmp_logical: dict[int, int] = {}
            phys_list, logical_list = torch.stack((physical, logical_positions)).tolist()
            for phys, logical in zip(phys_list, logical_list):
                if phys % self.ratio == self.ratio - 1:
                    row = phys // self.ratio
                    cmp_logical[row] = min(cmp_logical.get(row, logical), logical)
            txn["cmp_logical"] = cmp_logical
            phys_all_l = sorted(set(phys_list))
            phys_live_l = phys_all_l if live == n else sorted(set(phys_list[:live]))
            device_tokens = int(self.kvcache.num_device_pages) * self.page_size
            dev_l = [p for p in phys_live_l if p < device_tokens]
            host_l = [p - device_tokens for p in phys_live_l if p >= device_tokens]
            cmp_rows_l = sorted(
                {p // self.ratio for p in phys_all_l if p % self.ratio == self.ratio - 1}
                | {scratch}
            )
            phys_live = torch.tensor(phys_live_l, dtype=torch.long, device=device)
            dev = torch.tensor(dev_l, dtype=torch.long, device=device)
            host = torch.tensor(host_l, dtype=torch.long, device=device)
            cmp_rows = torch.tensor(cmp_rows_l, dtype=torch.long, device=device)
            cmp_logical_t = torch.tensor(
                [
                    cmp_logical.get(r, _NEVER_KEPT) if r != scratch else _NEVER_KEPT
                    for r in cmp_rows_l
                ],
                dtype=torch.long,
                device=device,
            )
        txn["cmp_logical_t"] = cmp_logical_t
        for layer_id, slot in self._idx_slot.items():
            dense = pool._dense(layer_id)
            self._txn_save_rows(table_idx, layer_id, phys_live)
            slab = self.kvcache.cmp_k_cache(slot)
            txn["bulk_cmp"].append((slot, cmp_rows, slab.index_select(0, cmp_rows)))
        rope = getattr(self.kvcache, "_rope_positions", None)
        if rope is not None:
            txn["bulk_rope"].append((phys_live, rope.index_select(0, phys_live)))
        txn["graph_safe"] = bool(pool.compressed)
        if not activate:
            self._spec_preimages[table_idx] = txn

    def spec_txn_graph_safe(self, table_idx: int) -> bool:
        txn = self._spec_txns.get(int(table_idx))
        return bool(txn and txn.get("graph_safe"))

    def _restore_host(self, dense, rows, values) -> None:
        kvh = self.kvcache._kv_host
        if isinstance(dense, torch.Tensor):  # layer-batched entry: dense is [L, 1]
            hk = kvh[0].reshape(kvh.shape[1], -1, *kvh.shape[4:])
            hv = kvh[1].reshape(kvh.shape[1], -1, *kvh.shape[4:])
            hk[dense, rows] = values[0]
            hv[dense, rows] = values[1]
            return
        hk = kvh[0, dense].reshape(-1, *kvh.shape[4:])
        hv = kvh[1, dense].reshape(-1, *kvh.shape[4:])
        # CPU Float8 tensors do not implement index_copy_; indexed assignment does.
        hk[rows] = values[0]
        hv[rows] = values[1]

    def _fast_layer_index(self, device):
        cached = getattr(self, "_fast_layer_idx", None)
        if cached is None:
            pool = self.kvcache._pool
            cached = (
                torch.tensor(
                    [pool._dense(l) for l in self._idx_slot], dtype=torch.long, device=device
                ),
                torch.tensor(list(self._idx_slot.values()), dtype=torch.long, device=device),
            )
            self._fast_layer_idx = cached
        return cached

    def _restore_fast(self, txn: dict, keep_end: int | None = None) -> None:
        """Undo the layer-batched journal; with ``keep_end`` only compressed rows closed at or
        after it (KV/RoPE rows are left to the caller's page reclaim)."""
        fast = txn.get("fast")
        if fast is None:
            return
        rows = fast["cmp_rows"]
        dense_t, slot_t = self._fast_layer_index(rows.device)
        buf = self.kvcache._cmp_k_buffer
        saved = fast["cmp"]
        if keep_end is not None:
            mask = (txn["cmp_logical_t"] >= keep_end).view(1, -1, 1)
            saved = torch.where(mask, saved, buf[slot_t[:, None], rows])
        buf[slot_t[:, None], rows] = saved
        if keep_end is not None:
            return
        pool = self.kvcache._pool
        dev = fast["dev"]
        mask = fast["dev_mask"]
        for t, v in zip((pool._k_codes, pool._k_norm, pool._v_codes, pool._v_norm), fast["kv"]):
            if mask is not None:
                v = torch.where(
                    mask.view(1, -1, *([1] * (v.dim() - 2))), v, t[dense_t[:, None], dev]
                )
            t[dense_t[:, None], dev] = v

    def _restore_bulk(self, txn: dict) -> None:
        self._restore_fast(txn)
        for slot, rows, values in txn.get("bulk_cmp", ()):
            self.kvcache.cmp_k_cache(slot).index_copy_(0, rows, values)
        rope = getattr(self.kvcache, "_rope_positions", None)
        if rope is not None:
            for rows, values in txn.get("bulk_rope", ()):
                rope.index_copy_(0, rows, values)
        pool = self.kvcache._pool
        for entry in txn.get("bulk_kv", ()):
            kind, dense, rows, *values = entry
            if kind == "device":
                pool._k_codes[dense].index_copy_(0, rows, values[0])
                pool._k_norm[dense].index_copy_(0, rows, values[1])
                pool._v_codes[dense].index_copy_(0, rows, values[2])
                pool._v_norm[dense].index_copy_(0, rows, values[3])
            else:
                self._restore_host(dense, rows, values)

    @_txn_timed
    def snapshot_spec_preimage(
        self,
        table_idx: int,
        out_loc: torch.Tensor,
        live_count: int | None = None,
        logical_positions: torch.Tensor | None = None,
    ) -> None:
        self.prepare_spec_txn(
            table_idx,
            out_loc,
            live_count,
            activate=False,
            logical_positions=logical_positions,
        )

    @_txn_timed
    def rollback_spec_txn_partial(self, table_idx: int, keep_end: int) -> None:
        """Rollback only rejected QSA writes; preserve verify state before ``keep_end``."""
        txn = self._spec_txns.pop(int(table_idx), None)
        if txn is None:
            return
        cmp_logical = txn.get("cmp_logical", {})
        scratch = int(txn.get("cmp_scratch", self.kvcache.cmp_scratch_base + table_idx))
        restore = txn.get("cmp_logical_t")
        self._restore_fast(txn, keep_end)
        for slot, rows, values in txn.get("bulk_cmp", ()):
            restore_mask = (restore >= keep_end).view(-1, *([1] * (values.dim() - 1)))
            slab = self.kvcache.cmp_k_cache(slot)
            slab.index_copy_(0, rows, torch.where(restore_mask, values, slab.index_select(0, rows)))
        for (slot, row), value in txn.get("cmp", {}).items():
            logical = cmp_logical.get(int(row), keep_end)
            if int(row) == scratch or logical >= keep_end:
                self.kvcache.cmp_k_cache(slot)[row].copy_(value)
        # KV/RoPE rows before keep_end are the committed verify writes. Speculative rows
        # after it are newly allocated and are reclaimed by free_spec_reject.

    def _txn_save_rows(self, table_idx: int, layer_id: int, out_loc: torch.Tensor) -> None:
        txn = self._spec_txns.get(int(table_idx))
        if txn is None:
            return
        pool = self.kvcache._pool
        dense = pool._dense(layer_id)
        for loc in torch.unique(out_loc.detach()).tolist():
            loc = int(loc)
            key = (int(layer_id), loc)
            if key in txn["kv"]:
                continue
            if getattr(pool, "compressed", False):
                k_codes, k_norm = self.kvcache.k_slab(layer_id)
                v_codes, v_norm = self.kvcache.v_slab(layer_id)
                device_tokens = int(self.kvcache.num_device_pages) * self.page_size
                if loc < device_tokens:
                    txn["kv"][key] = (
                        "device",
                        int(layer_id),
                        dense,
                        loc,
                        k_codes[loc].clone(),
                        k_norm[loc].clone(),
                        v_codes[loc].clone(),
                        v_norm[loc].clone(),
                    )
                else:
                    host = self.kvcache.host_turbo(layer_id)
                    if host is None:
                        host_kv = self.kvcache.host_kv(layer_id)
                        if host_kv is None:
                            continue
                        hloc = loc - device_tokens
                        hk = host_kv[0].reshape(-1, *host_kv[0].shape[2:])
                        hv = host_kv[1].reshape(-1, *host_kv[1].shape[2:])
                        txn["kv"][key] = (
                            "host_fp8",
                            int(layer_id),
                            dense,
                            hloc,
                            hk[hloc].clone(),
                            hv[hloc].clone(),
                        )
                    else:
                        hloc = loc - device_tokens
                        txn["kv"][key] = (
                            "host_turbo",
                            int(layer_id),
                            dense,
                            hloc,
                            *(x[hloc].clone() for x in host),
                        )
            else:
                device_tokens = int(self.kvcache.num_device_pages) * self.page_size
                if loc >= device_tokens:
                    host_kv = self.kvcache.host_kv(layer_id)
                    if host_kv is None:
                        continue
                    hloc = loc - device_tokens
                    hk = host_kv[0].reshape(-1, *host_kv[0].shape[2:])
                    hv = host_kv[1].reshape(-1, *host_kv[1].shape[2:])
                    txn["kv"][key] = (
                        "bf16_host",
                        int(layer_id),
                        dense,
                        hloc,
                        hk[hloc].clone(),
                        hv[hloc].clone(),
                    )
                    continue
                k_page = self.kvcache.k_cache(layer_id)
                v_page = self.kvcache.v_cache(layer_id)
                page, offset = divmod(loc, self.page_size)
                txn["kv"][key] = (
                    "bf16_device",
                    int(layer_id),
                    dense,
                    loc,
                    k_page[page, offset].clone(),
                    v_page[page, offset].clone(),
                )

    def _txn_save_cmp(self, table_idx: int, slot: int, rows: torch.Tensor) -> None:
        txn = self._spec_txns.get(int(table_idx))
        if txn is None:
            return
        slab = self.kvcache.cmp_k_cache(slot)
        for row in torch.unique(rows[rows >= 0].detach()).tolist():
            key = (int(slot), int(row))
            if key not in txn["cmp"]:
                txn["cmp"][key] = slab[int(row)].clone()

    def _txn_save_rope(self, table_idx: int, out_loc: torch.Tensor) -> None:
        txn = self._spec_txns.get(int(table_idx))
        rope = getattr(self.kvcache, "_rope_positions", None)
        if txn is None or rope is None:
            return
        for loc in torch.unique(out_loc.detach()).tolist():
            loc = int(loc)
            if loc not in txn["rope"]:
                txn["rope"][loc] = rope[loc].clone()

    @_txn_timed
    def rollback_spec_txn(self, table_idx: int) -> None:
        txn = self._spec_txns.pop(int(table_idx), None)
        if txn is None:
            return
        self._restore_fast(txn)
        for slot, rows, values in txn.get("bulk_cmp", ()):
            self.kvcache.cmp_k_cache(slot).index_copy_(0, rows, values)
        rope = getattr(self.kvcache, "_rope_positions", None)
        if rope is not None:
            for rows, values in txn.get("bulk_rope", ()):
                rope.index_copy_(0, rows, values)
        pool = self.kvcache._pool
        for entry in txn.get("bulk_kv", ()):
            kind, dense, rows, *values = entry
            if kind == "device":
                pool._k_codes[dense].index_copy_(0, rows, values[0])
                pool._k_norm[dense].index_copy_(0, rows, values[1])
                pool._v_codes[dense].index_copy_(0, rows, values[2])
                pool._v_norm[dense].index_copy_(0, rows, values[3])
            else:
                self._restore_host(dense, rows, values)
        for (slot, row), value in txn["cmp"].items():
            self.kvcache.cmp_k_cache(slot)[row].copy_(value)
        rope = getattr(self.kvcache, "_rope_positions", None)
        if rope is not None:
            for row, value in txn["rope"].items():
                rope[row].copy_(value)
        for value in txn["kv"].values():
            kind, layer_id, dense, loc, *saved = value
            pool = self.kvcache._pool
            if kind == "device":
                pool._k_codes[dense][loc].copy_(saved[0])
                pool._k_norm[dense][loc].copy_(saved[1])
                pool._v_codes[dense][loc].copy_(saved[2])
                pool._v_norm[dense][loc].copy_(saved[3])
            elif kind == "host_fp8":
                hk = self.kvcache._kv_host[0, dense].reshape(-1, *self.kvcache._kv_host.shape[4:])
                hv = self.kvcache._kv_host[1, dense].reshape(-1, *self.kvcache._kv_host.shape[4:])
                hk[loc].copy_(saved[0])
                hv[loc].copy_(saved[1])
            elif kind == "host_turbo":
                host = self.kvcache.host_turbo(layer_id)
                for target, source in zip(host, saved):
                    target[loc].copy_(source)
            elif kind == "bf16_device":
                k_page = self.kvcache.k_cache(layer_id)
                v_page = self.kvcache.v_cache(layer_id)
                page, offset = divmod(loc, self.page_size)
                k_page[page, offset].copy_(saved[0])
                v_page[page, offset].copy_(saved[1])
            elif kind == "bf16_host":
                host_kv = self.kvcache.host_kv(layer_id)
                hk = host_kv[0].reshape(-1, *host_kv[0].shape[2:])
                hv = host_kv[1].reshape(-1, *host_kv[1].shape[2:])
                hk[loc].copy_(saved[0])
                hv[loc].copy_(saved[1])
            else:
                raise RuntimeError(f"unknown QSA transaction KV kind: {kind}")

    @staticmethod
    def _qsa_group(config: ModelConfig):
        from freetoken.models.config import FullAttentionGroupConfig

        groups = [
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1
        ]
        assert len(groups) == 1, f"expected one QSA attention group, got {len(groups)}"
        return groups[0]

    # ----- slab views ---------------------------------------------------------------------
    def _cmp_pages(self, slot: int) -> torch.Tensor:
        """The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's
        paged layout. The scratch rows past ``cmp_scratch_base`` stay out of the view."""
        rows = self.kvcache.cmp_k_cache(slot)[: self.kvcache.cmp_scratch_base]
        return rows.view(-1, self.cmp_page_size, 1, self.index_head_dim)

    def _index_rope_cache(self) -> torch.Tensor:
        """cos/sin table of the indexer rope: same rotary_dim and frequencies as the main
        attention, ``head_size`` 128 instead of 256, so it is a separate get_rope instance.

        The table itself (not RotaryEmbedding.forward) because the indexer's norm+rope is one
        fused kernel and the compressed keys rope at their group's position, not the query's."""
        if self._index_cos_sin is None:
            from freetoken.layers.rotary import get_rope

            rotary = self.rotary_config
            with torch.device(self.device):
                rope = get_rope(
                    head_dim=self.index_head_dim,
                    rotary_dim=rotary.rotary_dim,
                    max_position=rotary.max_position,
                    base=rotary.base,
                    rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
                )
            self._index_cos_sin = rope._cos_sin_cache.to(self.device)
        return self._index_cos_sin

    def _index_rope_rows(self, positions: torch.Tensor) -> torch.Tensor:
        """Per-token indexer cos/sin rows for [3, n] mrope positions."""
        from freetoken.layers.rotary import mrope_cos_sin_rows

        return mrope_cos_sin_rows(self._index_rope_cache(), positions, self._section_table)

    # ----- metadata -----------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )
        batch.attn_metadata = md
        if not is_decode:
            table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
            token_to_req = torch.repeat_interleave(
                torch.arange(len(reqs), dtype=torch.int32),
                torch.tensor(seqlens_q, dtype=torch.int32),
            ).pin_memory()
            md.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
            md.token_to_req = token_to_req.to(self.device, non_blocking=True)
            md.seq_lens = kv_len.to(self.device, non_blocking=True)
            md.ring_slots = table_idx.to(self.device, non_blocking=True)
            md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        # Decode addressing is DEFERRED: a graph-bound step stages it into the static
        # buffers (prepare_for_replay), an eager step snapshots at the first QSA layer.

    def _block_base_view(self) -> torch.Tensor:
        """Every-``page_size``-th column of the page table: the per-page base slots. A strided
        VIEW, so gathering rows through it materializes only [bs, W/page_size]."""
        return get_global_ctx().page_table[:, :: self.page_size]

    def _block_table(self, table_idx: torch.Tensor) -> torch.Tensor:
        return self._physical_pages(
            self._block_base_view().index_select(0, table_idx) // self.page_size
        ).to(torch.int32)

    def _physical_pages(self, pages: torch.Tensor) -> torch.Tensor:
        """Scheduler page ids -> where the RAM-tier rebalancer currently holds them."""
        page_map = getattr(self.kvcache, "page_map", None)
        if page_map is None:
            return pages
        return page_map[pages.long().clamp(0, page_map.shape[0] - 1)]

    def _physical_loc(self, out_loc: torch.Tensor) -> torch.Tensor:
        if getattr(self.kvcache, "page_map", None) is None:
            return out_loc
        loc = out_loc.long()
        pages = self._physical_pages(loc // self.page_size).long()
        return (pages * self.page_size + loc % self.page_size).to(out_loc.dtype)

    def _record_heat(self, md: QSASparseMetadata, indices: torch.Tensor) -> None:
        """Count this layer's selections per physical page (graph-capturable, no sync)."""
        heat = getattr(self.kvcache, "page_heat", None)
        if heat is None or indices is None or indices.shape[0] > _ZERO_COPY_MAX_ROWS:
            return  # decode/verify rows only: they are what zero-copy RAM reads slow down
        table = md.block_table
        valid = indices >= 0
        logical = (indices.clamp(min=0) // self.page_size).clamp(max=table.shape[1] - 1)
        rows = md.token_to_req.long()[:, None].expand_as(logical)
        pages = table[rows, logical.long()].long()
        sink = heat.shape[0] - 1
        pages = torch.where(valid & (pages >= 0) & (pages < sink), pages, sink)
        heat.index_add_(0, pages.flatten(), torch.ones_like(pages.flatten(), dtype=heat.dtype))

    def _stage_decode(self, md: QSASparseMetadata, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the metadata
        at them (restage-per-replay, m3/dsa precedent)."""
        self._graph["block_table"][:bs].copy_(
            self._physical_pages(
                self._block_base_view().index_select(0, table_idx) // self.page_size
            )
        )
        self._graph["kvlen"][:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        self._graph["table_idx"][:bs].copy_(table_idx)
        md.block_table = self._graph["block_table"][:bs]
        md.seq_lens = self._graph["kvlen"][:bs]
        md.ring_slots = self._graph["table_idx"][:bs]
        md.token_to_req = self._graph["token_to_req"][:bs]
        md.cu_seqlens = self._graph["cu_seqlens"][: bs + 1]

    def _snapshot_decode(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Eager decode (not graph-staged): this step's rows, once per forward. The live
        page-table row may mutate for the next batch while this one runs, so gather now."""
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        bs = len(reqs)
        table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
        md.ring_slots = table_idx.to(self.device, non_blocking=True)
        md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        md.seq_lens = md.kv_len_cpu.to(self.device, non_blocking=True)
        md.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
        md.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)

    # ----- dense layers -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "qsa_sparse serves QSA layers only (Qwen3.8-Flash-Next has no dense attention "
            "layer); the QSA layer calls qsa_forward"
        )

    # ----- QSA layers ---------------------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        debug_timing = os.getenv(QSA_TIMING_ENV, "0") == "1"
        started = time.perf_counter()

        def mark(stage: str) -> None:
            if debug_timing:
                capturing = torch.cuda.is_current_stream_capturing()
                if not capturing:
                    torch.cuda.synchronize(self.device)
                print(
                    f"[qsa-timing] layer={layer_id} T={q.shape[0]} {stage} "
                    f"{time.perf_counter() - started:.3f}s" + (" [capture]" if capturing else ""),
                    flush=True,
                )

        mark("start")

        md, slot = self.store_qsa_kv(k, v, index, layer_id, batch)
        mark("store_kv")
        mark("index_cache")
        indices = self._select(index, md, slot)
        self._record_heat(md, indices)
        mark("select")
        compressed = getattr(self.kvcache, "compressed", False) or getattr(
            self.kvcache._pool, "compressed", False
        )
        if compressed:
            from freetoken.kernel.triton.turbo_kv import is_rotated, rotate, inv_rotate

            book = self.kvcache._pool.book
            q_in = rotate(q.reshape(-1, self.head_dim)).reshape(q.shape) if is_rotated(book) else q
            k_codes, k_norm = self.kvcache.k_slab(layer_id)
            v_codes, v_norm = self.kvcache.v_slab(layer_id)
            k_cache, v_cache = k_codes, v_codes
            mark("coded_kv")
        else:
            q_in = q
            k_cache = self.kvcache.k_cache(layer_id)
            v_cache = self.kvcache.v_cache(layer_id)

        host_kv, block_table = self._host_tier(layer_id, md, indices)
        out = qsa_sparse_paged_attention(
            q_in,
            k_cache,
            v_cache,
            indices,
            block_table,
            md.token_to_req,
            torch.empty_like(q),
            host_kv=host_kv,
            kv_book=book if compressed else None,
            kv_norms=(k_norm, v_norm) if compressed else None,
            cent=self.kvcache.cent_tensor if compressed else None,
            page_size=self.page_size if compressed else None,
            row_invariant=getattr(batch, "spec_logits_indices", None) is not None
            and os.environ.get(
                "FREETOKEN_ROW_INVARIANT_QSA", os.environ.get("FREETOKEN_ROW_INVARIANT_LINEAR", "0")
            )
            == "1",
        )
        mark("attention")
        if compressed and is_rotated(book):
            out = inv_rotate(out.reshape(-1, self.head_dim)).reshape(out.shape)
            mark("inverse_rotate")
        return out

    def _decode_turbo_tier(
        self,
        md: QSASparseMetadata,
        indices: torch.Tensor | None,
        host_turbo: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the turbo RAM tokens this layer's selection reads (not their whole pages: a
        scattered top-k touches most pages) into the staging slab at their own page index and
        slot, so the unmodified block table addresses them. Fixed shapes (one entry per
        selected index, -1 = not in RAM): no host sync, graph-capturable."""
        from freetoken.kernel.triton.qsa.tiered import turbo_inverse_rotation, turbo_slots_to_bf16
        from freetoken.kernel.triton.turbo_kv import _book

        stage_k, stage_v = self.kvcache.host_staging
        pages = stage_k.shape[0]
        if indices is None:
            return stage_k, stage_v
        if self._turbo_consts is None:
            self._turbo_consts = (
                _book(self.device, self.kvcache.host_book)[0].float(),
                turbo_inverse_rotation(self.device),
            )
        cent, rotation = self._turbo_consts
        table = md.block_table
        safe = indices.clamp(min=0)
        logical = (safe // self.page_size).clamp(max=table.shape[1] - 1)
        rows = md.token_to_req.long()[:, None].expand_as(logical)
        host = table[rows, logical.long()].long() - self.kvcache.num_device_pages
        slot = host * self.page_size + safe % self.page_size
        slot = torch.where((indices >= 0) & (host >= 0) & (host < pages), slot, -1).flatten()
        tokens = pages * self.page_size
        if slot.numel() > tokens:
            # Many query rows (prefill) repeat the same tokens: decode each touched token once.
            hit = torch.zeros(tokens + 1, dtype=torch.bool, device=self.device)
            hit[torch.where(slot >= 0, slot, tokens)] = True
            slot = torch.where(hit[:tokens], torch.arange(tokens, device=self.device), -1)
        kc, kn, vc, vn = host_turbo
        book = self.kvcache.host_book
        turbo_slots_to_bf16(kc, kn, cent, rotation, slot, slot, stage_k.flatten(0, 1), book)
        turbo_slots_to_bf16(vc, vn, cent, rotation, slot, slot, stage_v.flatten(0, 1), book)
        return stage_k, stage_v

    def _plan_host_staging(self, md: QSASparseMetadata, rows: int) -> None:
        """Eager multi-row forwards (prefill) re-read every selected token once per query row;
        zero-copy RAM reads would cross PCIe per row, so the RAM pages this forward touches are
        copied once per layer into the device staging slab instead. Decode, verify and anything
        graph-bound read the RAM tier zero-copy: one pass, fixed addresses, no host sync."""
        md.host_pages = md.staged_table = None
        stage = getattr(self.kvcache, "host_staging", None)
        if (
            stage is None
            or getattr(self.kvcache, "host_book", None) is not None  # turbo decodes per layer
            or rows <= _ZERO_COPY_MAX_ROWS
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        device_pages = self.kvcache.num_device_pages
        table = md.block_table
        pages = torch.unique(table[table >= device_pages])  # sorted; host sync, eager only
        if pages.numel() > stage[0].shape[0]:
            return  # zero-copy stays correct, only slower
        if pages.numel() == 0:
            md.host_pages = pages  # nothing in RAM: the plain device kernel serves this forward
            return
        rank = torch.searchsorted(pages, table).to(torch.int32)
        md.staged_table = torch.where(table >= device_pages, device_pages + rank, table)
        md.host_pages = (pages - device_pages).to(torch.int64)

    def _host_tier(
        self, layer_id: int, md: QSASparseMetadata, indices: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        """The RAM tier view this layer's attention reads, and the block table addressing it."""
        host_turbo = (
            self.kvcache.host_turbo(layer_id) if hasattr(self.kvcache, "host_turbo") else None
        )
        if host_turbo is not None:
            return self._decode_turbo_tier(md, indices, host_turbo), md.block_table
        host_kv = self.kvcache.host_kv(layer_id) if hasattr(self.kvcache, "host_kv") else None
        if host_kv is None or md.host_pages is None:
            return host_kv, md.block_table
        from freetoken.kernel.triton.qsa.tiered import gather_pages

        n = md.host_pages.shape[0]
        if n == 0:
            return None, md.block_table
        stage_k, stage_v = (t[:n] for t in self.kvcache.host_staging)
        gather_pages(host_kv[0], md.host_pages, stage_k)
        gather_pages(host_kv[1], md.host_pages, stage_v)
        return (stage_k, stage_v), md.staged_table

    def store_qsa_kv(self, k, v, index, layer_id, batch, *, stage_host: bool = True):
        """Update only KV and index state; target-fed MTP prefix rows need no attention."""
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        new_forward = md.cmp_rows is None or slot <= md.last_slot
        if new_forward:
            md.phys_loc = self._physical_loc(batch.out_loc)
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        req_indices = None
        legacy_txn = self._spec_txns and any(
            not txn.get("graph_safe", False) for txn in self._spec_txns.values()
        )
        if legacy_txn:
            req_indices = md.token_to_req
            if req_indices is None:
                raise RuntimeError("QSA speculative transaction requires token_to_req metadata")
            reqs = getattr(batch, "padded_reqs", None) or batch.reqs
            for req_index in torch.unique(req_indices.detach()).tolist():
                mask = req_indices == int(req_index)
                table_idx = reqs[int(req_index)].table_idx
                if not self.spec_txn_graph_safe(table_idx):
                    self._txn_save_rows(table_idx, layer_id, md.phys_loc[mask])
                    self._txn_save_rope(table_idx, md.phys_loc[mask])
        self.kvcache.store_kv(k, v, md.phys_loc, layer_id)
        if new_forward:
            # Capture and warmup share metadata, but their addresses must be replanned.
            self._plan_index_writes(md, batch)
            if stage_host:
                self._plan_host_staging(md, k.shape[0])
        md.last_slot = slot
        if req_indices is not None:
            reqs = getattr(batch, "padded_reqs", None) or batch.reqs
            for req_index in torch.unique(req_indices.detach()).tolist():
                mask = req_indices == int(req_index)
                table_idx = reqs[int(req_index)].table_idx
                if not self.spec_txn_graph_safe(table_idx):
                    self._txn_save_cmp(table_idx, slot, md.cmp_rows[mask])
        self._update_index_cache(index, md, slot)
        return md, slot

    def _plan_index_writes(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Per-token slab row and ring row for this forward; the other QSA layers reuse it
        (it is layer-invariant). Pure device arithmetic: no host sync, graph-capturable."""
        md.positions = batch.positions
        out_loc = md.phys_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        if self._section_table is not None:
            rope_positions = batch.get_attn_positions()
            if rope_positions.dim() == 1:
                rope_positions = rope_positions.unsqueeze(0).expand(3, -1)
            rope_positions = rope_positions.to(torch.int32)
            self.kvcache.rope_positions.index_copy_(0, out_loc, rope_positions.t().contiguous())
            # groups are ratio-aligned within a page, so the group start is the slot rounded down
            first_pos = self.kvcache.rope_positions.index_select(
                0, out_loc - out_loc % self.ratio
            ).t()
            md.rope_rows = torch.arange(out_loc.numel(), dtype=torch.int32, device=self.device)
            md.q_rope_cache = self._index_rope_rows(rope_positions)
            md.k_rope_cache = self._index_rope_rows(first_pos)
        rows = torch.arange(out_loc.numel(), device=self.device)
        req = md.token_to_req.to(torch.int64)
        slots = md.ring_slots.to(torch.int64).index_select(0, req)
        # out_loc % page_size == position % page_size and index_ratio divides page_size, so a
        # group closes exactly on out_loc % index_ratio == index_ratio - 1.
        closing = out_loc % self.ratio == self.ratio - 1
        scratch = self.kvcache.cmp_scratch_base + slots
        md.cmp_rows = torch.where(closing, out_loc // self.ratio, scratch).to(torch.int32)
        # Only the last ring_capacity rows of a request survive to the next forward; the rest
        # are masked off instead of dumped somewhere (vLLM rule).
        ends = md.cu_seqlens.to(torch.int64).index_select(0, req + 1)
        keep = rows >= ends - self.ring_capacity
        ring_row = slots * self.ring_capacity + positions % self.ring_capacity
        md.ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(torch.int32)

    def _update_index_cache(self, index, md: QSASparseMetadata, slot: int) -> None:
        """Compress each closing group into the slab, then refresh the pending ring."""
        from freetoken.kernel.triton.qsa import (
            qsa_compress_groups,
            qsa_index_norm_rope,
            qsa_store_rows,
        )

        rows = index.k.shape[0]
        ring = self.kvcache.pending_ring(slot)
        pooled = self._scratch("pooled", rows, self.index_head_dim, dtype=self.dtype)
        first = self._scratch("first_pos", rows, dtype=torch.int32)
        qsa_compress_groups(
            index.k,
            ring,
            md.ring_slots,
            md.token_to_req,
            md.cu_seqlens,
            md.positions,
            self.ratio,
            pooled,
            first,
        )
        if md.k_rope_cache is None:
            rope_positions, rope_cache = first, self._index_rope_cache()
        else:
            rope_positions, rope_cache = md.rope_rows, md.k_rope_cache
        qsa_index_norm_rope(
            pooled,
            rope_positions,
            rope_cache,
            index.k_norm_weight,
            index.eps,
            self.kvcache.cmp_k_cache(slot),
            dest_rows=md.cmp_rows,
        )
        # After the compression read: the ring rows this forward overwrites are exactly the
        # ones a straddling group just consumed.
        qsa_store_rows(ring, md.ring_rows, index.k)

    def _select(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Score complete visible blocks, take the top-k, expand them to token indices."""
        from freetoken.kernel.triton.qsa import (
            expand_qsa_block_indices,
            qsa_index_norm_rope,
            qsa_mqa_paged,
        )

        rows = index.q.shape[0]
        positions = md.positions
        q_index = self._scratch(
            "q_index", rows, self.index_heads, self.index_head_dim, dtype=self.dtype
        )
        if md.q_rope_cache is None:
            rope_positions, rope_cache = positions, self._index_rope_cache()
        else:
            rope_positions, rope_cache = md.rope_rows, md.q_rope_cache
        qsa_index_norm_rope(
            index.q.view(-1, self.index_head_dim),
            rope_positions,
            rope_cache,
            index.q_norm_weight,
            index.eps,
            q_index.view(-1, self.index_head_dim),
            heads=self.index_heads,
        )
        cmp_pages = self._cmp_pages(slot)
        columns = md.block_table.shape[1] * self.cmp_page_size
        indices = self._scratch("indices", rows, self.select_width, dtype=torch.int32)
        rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = slice(start, end)
            logits = self._scratch("logits", end - start, columns, dtype=torch.float32)
            visible = self._scratch("visible", end - start, dtype=torch.int32)
            qsa_mqa_paged(
                q_index[chunk],
                cmp_pages,
                md.block_table,
                md.token_to_req[chunk],
                positions[chunk],
                md.seq_lens,
                self.ratio,
                logits,
                visible,
            )
            blocks = self._scratch("blocks", end - start, self.block_topk, dtype=torch.int32)
            self._top_blocks(logits, visible, blocks)
            expand_qsa_block_indices(
                blocks,
                positions[chunk],
                md.seq_lens,
                md.token_to_req[chunk],
                self.ratio,
                self.token_topk,
                indices[chunk],
            )
        return indices

    def _top_blocks(
        self,
        logits: torch.Tensor,
        visible: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        """Top ``block_topk`` complete blocks per row, row-relative, -1 padded."""
        assert blocks.shape == (logits.shape[0], self.block_topk), (
            f"qsa block top-k output must be [rows, {self.block_topk}], got {tuple(blocks.shape)}"
        )
        if self._block_topk_kernel is not None:
            scratch_width = self._topk_scratch_width(logits.shape[1])
            scratch = (
                self._scratch("topk_scratch", logits.shape[0], scratch_width, dtype=torch.int32)
                if scratch_width
                else None
            )
            self._block_topk_kernel(logits, visible, blocks, scratch)
            return
        # The score kernel only writes columns below visible_blocks; mask the rest so a
        # stale row cannot win a slot. Real block scores are relu sums, never -inf.
        columns = logits.shape[1]
        column = torch.arange(columns, dtype=torch.int32, device=logits.device)
        logits.masked_fill_(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
        width = min(self.block_topk, columns)
        values, chosen = torch.topk(logits, width, dim=-1)
        blocks[:, :width] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
        if width < self.block_topk:
            blocks[:, width:] = -1

    def _topk_scratch_width(self, columns: int) -> int:
        """int32 columns per row the block top-k wants as scratch, 0 when it wants none."""
        if self._block_topk_kernel is None:
            return 0
        from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

        return qsa_block_topk_scratch_width(columns, self.block_topk)

    # ----- scratch ------------------------------------------------------------------------
    def _scratch(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient: the static decode buffer when it is wide enough (so a
        captured graph keeps one address), otherwise a fresh allocation."""
        buffer = self._graph.get(name)
        if buffer is not None and rows <= buffer.shape[0] and buffer.shape[1:] == shape:
            return buffer[:rows]
        return torch.empty((rows, *shape), dtype=dtype, device=self.device)

    # ----- CUDA graph (decode) --------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        pages = -(-width // self.page_size)
        columns = pages * self.cmp_page_size
        chunk = max(1, min(max_bs, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        self._graph = {
            "block_table": torch.zeros((max_bs, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "token_to_req": torch.arange(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens": torch.arange(max_bs + 1, dtype=torch.int32, device=self.device),
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(max_bs, dtype=torch.int32),
            "blocks": empty(max_bs, self.block_topk, dtype=torch.int32),
            "indices": empty(max_bs, self.select_width, dtype=torch.int32),
            "pooled": empty(max_bs, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(max_bs, dtype=torch.int32),
            "q_index": empty(max_bs, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if topk_scratch:
            self._graph["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(md, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        self._stage_decode(md, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def stage_verify(self, batch: Batch) -> None:
        """Copy a spec-verify window's addressing (one request, fixed extend) into static
        buffers and point the metadata at them, so the verify graph reads current values.
        ``prepare_metadata`` must have run on ``batch`` first."""
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata) and not md.is_decode
        tokens = batch.input_ids.shape[0]
        # one buffer set per window size: each captured verify graph keeps its own addresses
        v = self._verify.setdefault(tokens, {})
        if not v:
            assert not torch.cuda.is_current_stream_capturing()
            v.update(
                block_table=torch.zeros_like(md.block_table),
                kvlen=torch.zeros_like(md.seq_lens),
                table_idx=torch.zeros_like(md.ring_slots),
                token_to_req=torch.zeros(tokens, dtype=torch.int32, device=self.device),
                cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=self.device),
            )
        for name, src in (
            ("block_table", md.block_table),
            ("kvlen", md.seq_lens),
            ("table_idx", md.ring_slots),
        ):
            v[name].copy_(src)
        md.block_table, md.seq_lens, md.ring_slots = v["block_table"], v["kvlen"], v["table_idx"]
        md.token_to_req, md.cu_seqlens = v["token_to_req"], v["cu_seqlens"]

    def reset_capture(self) -> None:
        super().reset_capture()
        self._graph = {}
        self._verify = {}


__all__ = ["QSASparseAttnBackend", "QSASparseMetadata"]
