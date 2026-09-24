"""CPU-only tests for GGML_TYPE_Q2_0 (id 42) support.

Covers the three purely-additive pieces that run without CUDA:
  1. The pure-torch reference dequantizer (:func:`dequant_q2_0`) against a bitwise port
     of upstream's ``dequantize_row_q2_0`` (ggml-quants.c).
  2. Type-table metadata (``BLOCK_SHAPE``, ``row_bytes``, membership in the MMVQ/dequant/
     moe-vec capability sets, exclusion from the MMQ/moe-mmq sets -- Q2_0 has no MMQ
     kernel, like the I-quants).
  3. ``gguf.GGUFReader`` actually parsing a file containing a Q2_0 tensor. The installed
     gguf-py release predates this type id, so without the compat patch in
     ``freetoken.models.gguf.reader._gguf_module`` this raises ``ValueError`` while
     parsing the tensor table, before any FreeToken code runs.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_NAME,
    GGML_Q2_0,
    MMQ_TYPES,
    MMVQ_TYPES,
    MOE_MMQ_TYPES,
    MOE_VEC_TYPES,
    dequant_q2_0,
    dequantize,
    row_bytes,
)
from freetoken.models.gguf.reader import _gguf_module

# GGUF value type tags (see tests/models/test_gguf_shards.py for the full raw-write pattern)
_UINT32, _STRING = 4, 8
_ALIGN = 32


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return _u64(len(raw)) + raw


def _kv_str(key: str, value: str) -> bytes:
    return _string(key) + _u32(_STRING) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + _u32(_UINT32) + _u32(value)


def _write_q2_0_gguf(path: Path, name: str, n_blocks: int, block_bytes: bytes) -> None:
    """Hand-write a single-tensor GGUF file with one Q2_0 tensor named ``name``."""
    n_elements = n_blocks * 64
    kvs = [_kv_str("general.architecture", "test"), _kv_u32("general.alignment", _ALIGN)]
    head = b"GGUF" + _u32(3) + _u64(1) + _u64(len(kvs))
    head += b"".join(kvs)
    info = _string(name) + _u32(1) + _u64(n_elements) + _u32(GGML_Q2_0) + _u64(0)
    body = head + info
    pad = (-len(body)) % _ALIGN
    body += b"\0" * pad
    body += block_bytes
    path.write_bytes(body)


def _reference_dequant_q2_0(raw: bytes) -> list[float]:
    """Bitwise port of upstream's dequantize_row_q2_0 (ggml-quants.c)."""
    out = []
    nb = len(raw) // 18
    for i in range(nb):
        (d,) = struct.unpack_from("<e", raw, i * 18)
        qs = raw[i * 18 + 2 : i * 18 + 18]
        for j in range(64):
            byte_index, bit_offset = j // 4, (j % 4) * 2
            q = (qs[byte_index] >> bit_offset) & 0x03
            out.append((q - 1) * d)
    return out


def test_dequant_q2_0_matches_upstream_bitwise_reference():
    random.seed(0)
    raw = bytearray()
    for _ in range(8):
        raw += struct.pack("<e", random.uniform(0.01, 2.0))
        raw += bytes(random.randint(0, 255) for _ in range(16))
    expected = torch.tensor(_reference_dequant_q2_0(bytes(raw)), dtype=torch.float32)

    out = dequant_q2_0(torch.frombuffer(bytearray(raw), dtype=torch.uint8), torch.float32)
    assert torch.equal(out, expected)

    # dequantize() dispatches to the same function for GGML_Q2_0.
    out2 = dequantize(torch.frombuffer(bytearray(raw), dtype=torch.uint8), GGML_Q2_0, torch.float32)
    assert torch.equal(out2, expected)


def test_q2_0_type_tables():
    assert BLOCK_SHAPE[GGML_Q2_0] == (64, 18)
    assert GGML_NAME[GGML_Q2_0] == "Q2_0"
    assert row_bytes(128, GGML_Q2_0) == 36

    # Has MMVQ + dequant + moe-vec kernels, like the I-quants -- no MMQ kernel.
    assert GGML_Q2_0 in DEQUANT_TYPES
    assert GGML_Q2_0 in MMVQ_TYPES
    assert GGML_Q2_0 in MOE_VEC_TYPES
    assert GGML_Q2_0 not in MMQ_TYPES
    assert GGML_Q2_0 not in MOE_MMQ_TYPES


def test_reader_parses_synthetic_q2_0_tensor(tmp_path: Path):
    gguf = _gguf_module()
    n_blocks = 3
    block_bytes = bytes((n * 7 + 1) % 256 for n in range(n_blocks * 18))
    path = tmp_path / "q2_0.gguf"
    _write_q2_0_gguf(path, "t", n_blocks, block_bytes)

    reader = gguf.GGUFReader(path)
    assert len(reader.tensors) == 1
    t = reader.tensors[0]
    assert t.name == "t"
    assert int(t.tensor_type) == GGML_Q2_0
    assert t.data.nbytes == n_blocks * 18
    assert bytes(t.data.tobytes()) == block_bytes
