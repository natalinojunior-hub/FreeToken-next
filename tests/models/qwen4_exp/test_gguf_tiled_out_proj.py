"""A tiled ssm_out whose V head straddles quant blocks stays tiled; its input is permuted."""

import pytest
import torch

from freetoken.models.qwen3_5_moe.gguf import _ungroup_v
from freetoken.models.qwen4_exp.gguf import _straddles_blocks, _tiled_input_perm


def test_permuted_input_matches_ungrouped_weight():
    torch.manual_seed(0)
    k, r, d = 4, 3, 8
    w_tiled = torch.randn(5, k * r * d, dtype=torch.float64)
    w_grouped = _ungroup_v(w_tiled, 1, k, r, d)
    x = torch.randn(2, k * r * d, dtype=torch.float64)
    perm = _tiled_input_perm(k, r, d)
    assert torch.allclose(x.index_select(-1, perm) @ w_tiled.T, x @ w_grouped.T)


def test_straddle_detection():
    assert _straddles_blocks(23, 128)  # IQ4_XS: 256-wide blocks
    assert not _straddles_blocks(8, 128)  # Q8_0: 32-wide blocks


def test_ba_merge_decision():
    from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q8_0
    from freetoken.models.qwen4_exp.gguf import _ba_merge_ok

    assert _ba_merge_ok(GGML_BF16, GGML_BF16)
    assert not _ba_merge_ok(GGML_BF16, GGML_Q8_0)
    assert not _ba_merge_ok(GGML_Q8_0, GGML_Q8_0)  # quantized parts keep separate mmvq calls
    assert not _ba_merge_ok(None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS bf16 gemv bit-identity")
def test_ba_merged_gemv_bit_identical():
    # campaign-33: merging the two [num_v, hidden] bf16 ssm_beta/ssm_alpha GEMVs into one
    # [2*num_v, hidden] call must not move a single bit of the output cat layout.
    import torch.nn.functional as F

    gen = torch.Generator(device="cuda").manual_seed(7)
    for _ in range(8):
        wb = torch.randn(48, 2560, generator=gen, device="cuda", dtype=torch.bfloat16)
        wa = torch.randn(48, 2560, generator=gen, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(1, 2560, generator=gen, device="cuda", dtype=torch.bfloat16)
        split = torch.cat([F.linear(x, wb), F.linear(x, wa)], dim=-1)
        merged = F.linear(x, torch.cat([wb, wa], dim=0))
        assert torch.equal(merged, split)
