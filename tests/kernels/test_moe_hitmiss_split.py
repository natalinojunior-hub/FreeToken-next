"""Bit-exactness of the same-layer expert hit/miss gather-overlap split (C1).

``fused_experts_gguf_split`` computes the routed experts in two passes (hit routes,
then miss routes), each launched as the full fixed K-route grid with the other subset
sentineled to ``-1`` so masked routes leave their zero-init ``dst`` slot. The union
``down_hit + down_miss`` must be bit-identical to a single full ``fused_experts_gguf``
for EVERY hit/miss partition (``val + 0.0 == val``, same final weighted sum), so the
overlap cannot change generated output. Uses valid finite-scale Q4_0 banks so NaNs
never mask a real mismatch.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.timeout(300),
]

TOP_K = 8
SLOTS, H, I = 16, 512, 256


def _pack_q4_0(S: int, OUT: int, K: int, dev) -> torch.Tensor:
    nb = K // 32
    nib = torch.randint(0, 256, (S, OUT, nb, 16), dtype=torch.uint8)
    scale = (0.02 + 0.03 * torch.rand(S, OUT, nb)).to(torch.float16)
    sb = scale.view(torch.uint8).reshape(S, OUT, nb, 2)
    return torch.cat([sb, nib], dim=-1).reshape(S, OUT, nb * 18).contiguous().to(dev)


@pytest.fixture(scope="module")
def banks():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    return {
        "gate_up": _pack_q4_0(SLOTS, 2 * I, H, dev),
        "down": _pack_q4_0(SLOTS, H, I, dev),
        "ids": torch.randint(0, SLOTS, (1, TOP_K), dtype=torch.int32, device=dev),
        "w": torch.rand(1, TOP_K, device=dev, dtype=torch.float32),
        "x": (torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.5).contiguous(),
    }


def _ref(b):
    import freetoken.moe.fused_q4_0 as fq
    from freetoken.models.gguf.dequant import GGML_Q4_0

    saved, fq.DEQUANT_MIN_TOKENS = fq.DEQUANT_MIN_TOKENS, 1 << 30
    try:
        return fq.fused_experts_gguf(
            b["x"], b["gate_up"], b["down"], b["w"], b["ids"], "silu", GGML_Q4_0
        )
    finally:
        fq.DEQUANT_MIN_TOKENS = saved


def _split(b, hit_mask):
    import freetoken.moe.fused_q4_0 as fq
    from freetoken.models.gguf.dequant import GGML_Q4_0

    ids_hit = torch.where(hit_mask, b["ids"], torch.full_like(b["ids"], -1))
    ids_miss = torch.where(hit_mask, torch.full_like(b["ids"], -1), b["ids"])
    return fq.fused_experts_gguf_split(
        b["x"], b["gate_up"], b["down"], b["w"], ids_hit, ids_miss, "silu", GGML_Q4_0
    )


@pytest.mark.parametrize("seed", range(6))
def test_split_matches_full_for_random_partitions(banks, seed: int) -> None:
    """A random hit/miss partition of the K routes must reproduce the full GEMV exactly."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    hit = torch.randint(0, 2, (1, TOP_K), device="cuda", dtype=torch.bool, generator=g)
    ref = _ref(banks)
    got = _split(banks, hit)
    torch.cuda.synchronize()
    assert torch.equal(ref, got), (
        f"partition {hit.tolist()} diverged: max abs "
        f"{(ref.float() - got.float()).abs().max().item()}"
    )


def test_split_all_hit_all_miss_edges(banks) -> None:
    """All-hit (miss pass fully sentineled) and all-miss edges match the full GEMV."""
    ref = _ref(banks)
    all_hit = torch.ones(1, TOP_K, dtype=torch.bool, device="cuda")
    all_miss = torch.zeros(1, TOP_K, dtype=torch.bool, device="cuda")
    assert torch.equal(ref, _split(banks, all_hit))
    assert torch.equal(ref, _split(banks, all_miss))
