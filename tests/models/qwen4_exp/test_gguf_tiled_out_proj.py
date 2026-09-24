"""A tiled ssm_out whose V head straddles quant blocks stays tiled; its input is permuted."""

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
