"""Turbo3 / Turbo4 KV codecs: rotated-domain quantization over 128-element groups.

Byte layout, scale semantics and the tie rule follow llama-turbo-optimal
(``ggml-common.h:309-316`` / ``:356-361``, ``turbo-quant-cuda.cuh:617-669``, ``:845-1036``) as
recorded in docs/freetoken-next/audits/A8-turbo-codec-spec.md. Two deliberate deviations, both
byte-accounted: the per-group ``norm`` is stored once instead of in all four turbo3 sub-blocks
(the reference writes identical values there, so three of them are duplication), and the affine
mean-sub tap is not applied (K's is free but V's needs a graph-level add-back; neither is worth
its bookkeeping before the codec is measured).

Everything here stays in the ROTATED domain on purpose: attention consumes
``centroid[idx] * norm`` directly after pre-rotating Q, and the inverse rotation is applied once
to the accumulated output row instead of once per KV tile.
"""

from __future__ import annotations

import torch

QK_TURBO = 128  # one rotation group == one head_dim-128 row
FWHT_SCALE = 0.08838834764831845  # 1/sqrt(128), folded inside the butterfly

# RWHT sign arrays, taken from d_turbo_wht_s1 / d_turbo_wht_s2 in
# /models/servers/llama-turbo-optimal/ggml/src/ggml-cuda/turbo-wht.cu:4-17 (seed=42 rotation; the
# Metal table in ggml-metal/turbo-wht.h is identical). Length and parity pins live in
# tests/kvcache/test_turbo_kv.py: a hand-typed sign array is where a codec goes silently wrong,
# and the first transcription here was 127 long.
SIGNS1 = (
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    1,
)
SIGNS2 = (
    1,
    1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
)

# Lloyd-Max books for N(0, 1/128) coordinates after normalize+rotate (stock default, SKEW=0).
CENTROIDS_3 = (
    -0.190685,
    -0.117832,
    -0.065717,
    -0.021460,
    0.021460,
    0.065717,
    0.117832,
    0.190685,
)
MID_3 = (-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259)
CENTROIDS_4 = (
    -0.241556,
    -0.182907,
    -0.143047,
    -0.111065,
    -0.083317,
    -0.058069,
    -0.034311,
    -0.011353,
    0.011353,
    0.034311,
    0.058069,
    0.083317,
    0.111065,
    0.143047,
    0.182907,
    0.241556,
)
MID_4 = (
    -0.212232,
    -0.162977,
    -0.127056,
    -0.097191,
    -0.070693,
    -0.046190,
    -0.022832,
    0.0,
    0.022832,
    0.046190,
    0.070693,
    0.097191,
    0.127056,
    0.162977,
    0.212232,
)


# Packed payload bytes per 128-element group, excluding the fp16 norm.
CODE_BYTES = {"turbo3": 48, "turbo4": 64, "fp8": 128}
# bits per value including the deduped norm: (48*8 + 16) / 128, (64*8 + 16) / 128, (128*8 + 16) / 128
BPV = {"turbo3": 3.125, "turbo4": 4.125, "fp8": 8.125}
BOOKS = tuple(CODE_BYTES)

_CENT_TABLE = {"turbo3": CENTROIDS_3, "turbo4": CENTROIDS_4}
_MID_TABLE = {"turbo3": MID_3, "turbo4": MID_4}


def _book(device: torch.device, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), name)
    cache = _BOOK_CACHE
    hit = cache.get(key)
    if hit is None:
        cent = torch.tensor(_CENT_TABLE[name], device=device)
        mid = torch.tensor(_MID_TABLE[name], device=device)
        hit = (cent, mid)
        cache[key] = hit
    return hit


_BOOK_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _signs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    hit = _SIGNS_CACHE.get(key)
    if hit is None:
        hit = (
            torch.tensor(SIGNS1, device=device, dtype=torch.float32),
            torch.tensor(SIGNS2, device=device, dtype=torch.float32),
        )
        _SIGNS_CACHE[key] = hit
    return hit


_SIGNS_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def hadamard(group: int = QK_TURBO, device: torch.device | None = None) -> torch.Tensor:
    """Sylvester Hadamard matrix, fp32, exact: ``H[i][j] = (-1)**popcount(i & j)``.

    Equals the reference's in-place butterfly (pairing at distance 1, 2, ... group/2), which is
    what ``tests/kvcache/test_turbo_kv.py`` pins, because the two are the same transform up to the
    order in which the bit-stages are applied.
    """
    idx = torch.arange(group, device=device)
    anded = (idx[:, None] & idx[None, :]).to(torch.int32)
    parity = torch.zeros_like(anded)
    for bit in range(group.bit_length() - 1):
        parity = parity + ((anded >> bit) & 1)
    return torch.where(parity % 2 == 0, 1.0, -1.0)


def butterfly(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized in-place butterfly over the last axis (Sylvester/Walsh ordering, natural
    index order). Stages pair elements ``h = 1, 2, ..., 64`` apart, mirroring
    ``fwht128`` in turbo-quant-cuda.cuh:764-774."""
    n = x.shape[-1]
    if n != QK_TURBO:
        raise ValueError(f"butterfly needs a trailing axis of {QK_TURBO}, got {n}")
    v = x
    h = 1
    while h < n:
        # view as (..., n/(2h), 2, h): the axis of size 2 separates the paired elements
        shape = v.shape[:-1] + (n // (2 * h), 2, h)
        v = v.reshape(shape)
        lo = v.select(-2, 0)
        hi = v.select(-2, 1)
        v = torch.stack((lo + hi, lo - hi), dim=-2).reshape(x.shape)
        h *= 2
    return v


def rotate(x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Forward RWHT: ``x * s1 -> butterfly -> /sqrt(128) -> * s2`` (``turbo_rotate_forward_cuda``).

    Rows may hold several rotation groups (head_dim 256 is two): each 128-element group rotates on
    its own, which is what the encoder does per group and what the tile readers assume."""
    s1, s2 = _signs(x.device)
    flat_x = x.reshape(-1, x.shape[-1])
    if flat_x.shape[0] <= chunk_size:
        grouped = flat_x.reshape(*flat_x.shape[:-1], -1, QK_TURBO).float()
        out = butterfly(grouped * s1) * FWHT_SCALE * s2
        out = out.reshape(x.shape)
        return out.to(x.dtype) if x.dtype != torch.float32 else out

    out = torch.empty_like(flat_x)
    for i in range(0, flat_x.shape[0], chunk_size):
        c = flat_x[i : i + chunk_size]
        grouped = c.reshape(*c.shape[:-1], -1, QK_TURBO).float()
        c_out = (butterfly(grouped * s1) * FWHT_SCALE * s2).reshape(c.shape)
        out[i : i + chunk_size] = c_out.to(x.dtype) if x.dtype != torch.float32 else c_out
    return out.reshape(x.shape)


def inv_rotate(y: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Inverse RWHT: ``y * s2 -> butterfly -> /sqrt(128) -> * s1`` (the butterfly is an involution
    up to the scale, which is why the same 1/sqrt(128) is applied again)."""
    s1, s2 = _signs(y.device)
    flat_y = y.reshape(-1, y.shape[-1])
    if flat_y.shape[0] <= chunk_size:
        grouped = flat_y.reshape(*flat_y.shape[:-1], -1, QK_TURBO).float()
        out = butterfly(grouped * s2) * FWHT_SCALE * s1
        out = out.reshape(y.shape)
        return out.to(y.dtype) if y.dtype != torch.float32 else out

    out = torch.empty_like(flat_y)
    for i in range(0, flat_y.shape[0], chunk_size):
        c = flat_y[i : i + chunk_size]
        grouped = c.reshape(*c.shape[:-1], -1, QK_TURBO).float()
        c_out = (butterfly(grouped * s2) * FWHT_SCALE * s1).reshape(c.shape)
        out[i : i + chunk_size] = c_out.to(y.dtype) if y.dtype != torch.float32 else c_out
    return out.reshape(y.shape)


def _normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group L2 normalize; a group whose norm is <= 1e-10 encodes as all-zero scaled."""
    groups = x.reshape(-1, QK_TURBO).float()
    grp_norm = groups.norm(dim=-1)
    inv = torch.where(
        grp_norm > 1e-10, 1.0 / torch.clamp(grp_norm, min=1e-10), torch.zeros_like(grp_norm)
    )
    return groups * inv.unsqueeze(-1), grp_norm


def indices(y: torch.Tensor, book: str) -> torch.Tensor:
    """Nearest centroid by midpoint search; ties take the **higher** index (A8 sec.2 step 7:
    ``val == 0.0`` lands on index 4 for turbo3)."""
    _, mid = _book(y.device, book)
    return torch.bucketize(y.contiguous(), mid, right=True).to(torch.uint8)


def pack(idx: torch.Tensor, book: str) -> torch.Tensor:
    """Bit packing, group-major: [n_rows, groups * CODE_BYTES[book]].

    turbo4 -> 4-bit nibbles, low nibble = even element (``dst->qs[j/2] = (idx[j+1] << 4) | idx[j]``).
    turbo3 -> 32 B of low-two-bit pairs (4 per byte) then 16 B of the third bit (8 per byte),
    per 128-element group.
    """
    rows, elems = idx.shape
    if elems % QK_TURBO:
        raise ValueError(f"cannot pack {elems} elements into {QK_TURBO}-element groups")
    groups = elems // QK_TURBO
    if book == "turbo4":
        q = idx.reshape(rows, groups, QK_TURBO // 2, 2)
        return (q[:, :, :, 0] | (q[:, :, :, 1] << 4)).reshape(rows, -1).contiguous()
    two = (idx & 0x3).reshape(rows, groups, QK_TURBO // 4, 4)
    words = (
        two[:, :, :, 0] | (two[:, :, :, 1] << 2) | (two[:, :, :, 2] << 4) | (two[:, :, :, 3] << 6)
    )
    third = ((idx >> 2) & 1).reshape(rows, groups, QK_TURBO // 8, 8)
    # Shifts, not a host-built weight tensor: this runs inside CUDA graph capture.
    bits = third[:, :, :, 0]
    for b in range(1, 8):
        bits = bits | (third[:, :, :, b] << b)
    bits = bits.to(torch.uint8)
    out = torch.cat((words.to(torch.uint8), bits), dim=-1)
    return out.reshape(rows, -1).contiguous()


def unpack(codes: torch.Tensor, book: str) -> torch.Tensor:
    """Inverse of :func:`pack` -> centroid indices [n_rows, groups * 128]."""
    groups = codes.shape[1] // CODE_BYTES[book]
    if book == "turbo4":
        q = codes.reshape(-1, groups, QK_TURBO // 2, 1)
        even = q & 0xF
        odd = q >> 4
        return (
            torch.cat((even, odd), dim=-1)
            .reshape(codes.shape[0], groups * QK_TURBO)
            .to(torch.int64)
        )
    rows = codes.shape[0]
    # group-major: each group writes its 32 word bytes then its 16 third-bit bytes, so the slice
    # has to be per group. Slicing "all words, then all bits" is only right when groups == 1.
    grouped = codes.reshape(rows, groups, CODE_BYTES[book])
    words = grouped[:, :, :32].reshape(rows, groups, QK_TURBO // 4, 1)
    bits = grouped[:, :, 32:].reshape(rows, groups, QK_TURBO // 8, 1)
    sh4 = torch.arange(0, 8, 2, device=codes.device, dtype=torch.uint8)
    low = (words >> sh4) & 0x3
    sh8 = torch.arange(8, device=codes.device, dtype=torch.uint8)
    third = (bits >> sh8) & 0x1
    # both flatten to the same element order, so a reshape aligns the two bit-fields per element
    low = low.reshape(rows, groups, QK_TURBO // 8, 8)
    return (low | (third << 2)).reshape(rows, groups * QK_TURBO).to(torch.int64)


def quantize(x: torch.Tensor, book: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference encoder: normalize the group, rotate, nearest centroid, pack.

    Returns ``(codes, norm)`` where ``norm`` is the **corrected** fp16 scale
    ``group_norm / ||centroid[idx]||`` (the reference's ``norm`` is not a raw L2 norm), so that
    the decoded vector reproduces the original L2 exactly.
    """
    if book not in CODE_BYTES:
        raise ValueError(f"unknown turbo book {book!r}")
    if x.shape[-1] % QK_TURBO:
        raise ValueError(f"head_dim {x.shape[-1]} is not a multiple of {QK_TURBO}")
    if book in ELEMENT_BOOKS:
        return _element_quantize(x)
    rows = x.reshape(-1, QK_TURBO)
    normed, grp_norm = _normalize(rows)
    y = rotate(normed)
    idx = indices(y, book)
    cent, _ = _book(x.device, book)
    recon = cent[idx.long()].pow(2).sum(-1).sqrt()
    scale = torch.where(recon > 1e-10, grp_norm / recon, torch.zeros_like(recon))
    groups = x.shape[-1] // QK_TURBO
    return pack(idx.reshape(x.shape[0], -1), book), scale.reshape(x.shape[0], groups).to(
        torch.float16
    )


def decode_rotated(codes: torch.Tensor, norm: torch.Tensor, book: str) -> torch.Tensor:
    """Values in the rotated domain, ``centroid[idx] * norm`` -- what attention consumes."""
    if book in ELEMENT_BOOKS:
        return _element_decode(codes, norm, book)
    cent, _ = _book(codes.device, book)
    idx = unpack(codes, book)  # [rows, groups * 128]
    groups = idx.shape[1] // QK_TURBO
    scale = norm.reshape(-1, groups, 1).expand(-1, -1, QK_TURBO).reshape(idx.shape[0], -1).float()
    return cent[idx] * scale


def decode(codes: torch.Tensor, norm: torch.Tensor, book: str) -> torch.Tensor:
    """Materialized original-domain values. The slow arm, kept as the correctness oracle: the
    fused attention path never calls this."""
    groups = codes.shape[1] // CODE_BYTES[book]
    y = decode_rotated(codes, norm, book).reshape(-1, QK_TURBO)
    if is_rotated(book):
        y = inv_rotate(y)
    return y.reshape(codes.shape[0], groups * QK_TURBO)


def bytes_per_token(book: str, num_kv_heads: int, head_dim: int) -> int:
    """Packed bytes for one token, one KV head, one layer -- the number the ledger charges."""
    groups = head_dim // QK_TURBO
    return num_kv_heads * groups * (CODE_BYTES[book] + 2)


# ---- FP8 element format in the turbo slab layout ----------------------------------------------
# FP8: 128 e4m3 bytes per group of 128, a plain saturating cast of the KV values (no rotation, unit
# ``norm``). Measured against a rotated + per-group-scaled variant (campaign 18, Tiel 35B): same
# quality (usage 20/20 vs 19/20, needle 64K/256K pass) and faster at every context (TG 4K/64K/256K
# 128.3/101.6/84.7 vs 106.7/96.6/74.4), so the simple cast is the format.
ELEMENT_BOOKS = ("fp8",)
_E4M3_MAX = 448.0


def is_rotated(book: str) -> bool:
    """Whether ``book`` stores the rotated domain (the backend then rotates q/k/v and the output)."""
    return book not in ELEMENT_BOOKS


def _element_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.shape[0]
    y = x.reshape(-1, QK_TURBO).float().clamp(-_E4M3_MAX, _E4M3_MAX)
    codes = y.to(torch.float8_e4m3fn)
    norm = torch.ones(y.shape[0], device=x.device, dtype=torch.float16)
    return codes.view(torch.uint8).reshape(rows, -1), norm.reshape(rows, -1)


def _element_decode(codes: torch.Tensor, norm: torch.Tensor, book: str) -> torch.Tensor:
    grp = codes.reshape(-1, CODE_BYTES[book])
    return (grp.view(torch.float8_e4m3fn).float() * norm.reshape(-1, 1).float()).reshape(
        codes.shape[0], -1
    )
