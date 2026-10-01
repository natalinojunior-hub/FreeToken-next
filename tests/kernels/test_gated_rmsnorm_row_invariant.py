"""A decode row must normalise identically whatever the length of the MTP verify window."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_rows_per_block_pinned_for_small_windows():
    from freetoken.kernel.fla.layernorm_gated import calc_rows_per_block

    device = torch.device("cpu")
    assert {calc_rows_per_block(m, device) for m in (32, 192, 512, 1024)} == {1}


@cuda
@pytest.mark.parametrize("heads", [32, 48, 64])
def test_gated_rmsnorm_batched_matches_single_row_bitwise(heads):
    from freetoken.kernel.fla import rms_norm_gated

    torch.manual_seed(5)
    device, tokens, dim = torch.device("cuda"), 8, 128
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)

    def norm(x, z):
        return rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=z,
            eps=1e-6,
            is_rms_norm=True,
            norm_before_gate=True,
            activation="silu",
        )

    for _ in range(20):
        x = torch.randn(tokens * heads, dim, device=device, dtype=torch.bfloat16)
        z = torch.randn_like(x)
        single = torch.cat(
            [
                norm(x[t * heads : (t + 1) * heads], z[t * heads : (t + 1) * heads])
                for t in range(tokens)
            ]
        )
        for window in (2, 3, 4, 6, 8):
            assert torch.equal(
                norm(x[: window * heads], z[: window * heads]), single[: window * heads]
            )
