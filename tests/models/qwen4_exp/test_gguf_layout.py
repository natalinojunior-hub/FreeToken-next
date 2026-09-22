"""GGUF layout transforms of the qwen4_exp loader (CPU-only).

llama.cpp writes Qwen3.8-Flash-Next with (1+w) folded into every plus-one RMSNorm and the GDN
V heads TILED; both were loaded raw, so the GGUF model produced nonsense text. The LM head must
also score every drafted row of a spec-decode verify, or MTP accepts nothing.
"""

from types import SimpleNamespace

import torch

from freetoken.layers.gguf import select_head_rows
from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_F32, GGML_Q8_0
from freetoken.models.qwen4_exp.gguf import _plus_one_norm, _ungroup_packed_cols


def _tile(x: torch.Tensor, dim: int, k: int, r: int, d: int) -> torch.Tensor:
    """llama.cpp _reorder_v_heads: grouped [K, R, D] -> tiled [R, K, D] along ``dim``."""
    shape = list(x.shape)
    view = shape[:dim] + [k, r, d] + shape[dim + 1 :]
    return x.reshape(view).transpose(dim, dim + 1).reshape(shape)


def _tensor(packed: torch.Tensor, ggml_type: int, rows: int):
    return SimpleNamespace(name="t", ggml_type=ggml_type, rows=rows, packed=lambda: packed)


def test_ungroup_packed_cols_inverts_llama_cpp_tiling_on_q8_0_bytes():
    k, r, d, rows = 2, 3, 64, 5  # one V head = two 32-element Q8_0 blocks
    block, type_size = BLOCK_SHAPE[GGML_Q8_0]
    head_bytes = d // block * type_size
    grouped = torch.randint(0, 255, (rows, k * r * head_bytes), dtype=torch.uint8)
    tiled = _tile(grouped, 1, k, r, head_bytes)
    assert not torch.equal(tiled, grouped)
    assert torch.equal(_ungroup_packed_cols(_tensor(tiled, GGML_Q8_0, rows), k, r, d), grouped)


def test_ungroup_packed_cols_refuses_a_head_that_straddles_blocks():
    t = _tensor(torch.zeros(1, 8, dtype=torch.uint8), 12, 1)  # Q4_K: 256-element blocks
    try:
        _ungroup_packed_cols(t, 2, 3, 128)
    except NotImplementedError:
        return
    raise AssertionError("a 128-wide head inside 256-element blocks must not be byte-permuted")


def test_plus_one_norm_unfolds_the_gguf_shift():
    w = torch.tensor([-0.25, 0.0, 0.5, 1.75], dtype=torch.float32)
    t = SimpleNamespace(ggml_type=GGML_F32, shape=(4,), packed=lambda: (w + 1.0).view(torch.uint8))
    assert torch.equal(_plus_one_norm(t), w.to(torch.bfloat16))


def test_head_rows_follow_spec_verify_indices():
    x = torch.arange(12.0).reshape(6, 2)
    last = SimpleNamespace(get_last_indices=lambda bs: torch.tensor([5]))
    prefill = SimpleNamespace(is_prefill=True, spec_logits_indices=None, attn_metadata=last, size=1)
    assert torch.equal(select_head_rows(x, prefill), x[[5]])
    prefill.spec_logits_indices = torch.tensor([3, 4, 5])
    assert torch.equal(select_head_rows(x, prefill), x[3:6])
    decode = SimpleNamespace(is_prefill=False)
    assert select_head_rows(x, decode) is x
