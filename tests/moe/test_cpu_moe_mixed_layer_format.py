"""Regression test for the per-layer GGUF format bug (C4).

Real mixed-format GGUF checkpoints (e.g. Qwen3.8-Flash-Next-Unsloth-IQ4_XS: IQ3_S
gate_up / IQ4_NL down on most layers, IQ4_XS gate_up / Q8_0 down on a minority of
layers) carry a *per-layer* ``gguf_expert_types`` list. Before this fix,
``CpuMoeExecutor`` resolved that to a single *dominant* (gate_up, down) format pair
(``_resolve_gguf_format`` / ``dominant_gguf_pair``) and applied it to every layer's CPU
GEMV -- misreading a minority-format layer's block geometry (wrong row stride) rather
than just computing it with the wrong numbers, so its decode output goes to NaN/garbage
(the "!!!!!..." token) instead of merely being inaccurate.

This builds a 2-layer cache with layer 0 in the dominant format and layer 1 in a
different (but still CPU-kernel-capable) format, and checks layer 1's CPU decode
against the CUDA dequant + bf16 GPU reference -- it must match, not NaN.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

GGML_Q8_0, GGML_IQ4_NL, GGML_IQ3_S, GGML_IQ4_XS = 8, 20, 21, 23
_BLOCK = {  # ggml_type -> (block elems, block bytes)
    GGML_Q8_0: (32, 34),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ4_XS: (256, 136),
}


def _random_rows(ggml_type: int, S: int, OUT: int, K: int, gen: torch.Generator) -> torch.Tensor:
    """[S, OUT, row_bytes] syntactically-valid packed blocks: every bit pattern decodes to
    *some* finite value for these i-quant/k-quant formats once the leading fp16 scale is
    pinned finite (see test_cpu_gguf_iq_formats.py's _random_block) -- unconstrained
    otherwise, since this test only needs CPU and GPU to agree on *some* interpretation
    of the bytes, not a semantically meaningful one."""
    qk, blk = _BLOCK[ggml_type]
    nb = K // qk
    raw = torch.randint(0, 256, (S, OUT, nb, blk), dtype=torch.uint8, generator=gen)
    scale = (0.01 + 0.02 * torch.rand(S, OUT, nb, generator=gen)).to(torch.float16)
    raw[..., 0:2] = scale.view(torch.uint8).reshape(S, OUT, nb, 2)  # fp16 -> little-endian bytes
    return raw.reshape(S, OUT, nb * blk).contiguous()


def _make_mixed_cache(H: int, I: int, E: int, seed: int = 0):
    """3-layer cache: layers 0 and 1 = dominant (IQ3_S/IQ4_NL, duplicated so the majority
    vote unambiguously resolves to it instead of tying 1-vs-1 against layer 2 and
    tie-breaking on ggml type id -- see dominant_gguf_pair), layer 2 = minority
    (IQ4_XS/Q8_0) -- the exact pair mismatch the real checkpoint has on its layer 2.
    gguf_expert_types is a list of (gate_up, down) per-layer pairs, the shape
    OffloadMoeCache actually carries for a non-uniform checkpoint (offload_cache.py /
    expert_banks.py), not the {"gate_up": [...], "down": [...]} dict shape."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gen = torch.Generator().manual_seed(seed)
    dominant = (GGML_IQ3_S, GGML_IQ4_NL)
    layer_types = [dominant, dominant, (GGML_IQ4_XS, GGML_Q8_0)]

    # Per-layer row_bytes differ (IQ3_S vs IQ4_XS, IQ4_NL vs Q8_0), so each layer's bank is
    # its own tensor (a real ragged-geometry checkpoint, per gguf_expert_specs' per-layer
    # spec list) rather than one uniformly-shaped [L*E, ...] stack.
    def per_layer_banks(elems, is_gate_up):
        out = []
        for gu_type, dn_type in layer_types:
            t = gu_type if is_gate_up else dn_type
            packed = _random_rows(t, E, elems[0], elems[1], gen)
            pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
            pinned.copy_(packed)
            out.append(pinned)
        return out

    gate_up_banks = per_layer_banks((2 * I, H), is_gate_up=True)
    down_banks = per_layer_banks((H, I), is_gate_up=False)

    return SimpleNamespace(
        quant_format="gguf",
        gguf_expert_types=layer_types,
        bank_sources={"gate_up": gate_up_banks, "down": down_banks},
        num_layers=3,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    ), layer_types


def _dequant_bank(packed: torch.Tensor, ggml_type: int, K: int, dev) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_dequantize

    S, OUT, row_bytes = packed.shape
    flat = ggml_dequantize(
        packed.reshape(-1, row_bytes).to(dev).contiguous(),
        ggml_type,
        S * OUT,
        K,
        torch.bfloat16,
    )
    return flat.reshape(S, OUT, K)


@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_minority_format_layer_matches_gpu(bs):
    """Layer 1 (IQ4_XS/Q8_0) must decode correctly even though layer 0's (IQ3_S/IQ4_NL)
    is the checkpoint's dominant pair. Pre-fix, CpuMoeExecutor applied the dominant pair
    to every layer and this assertion failed with NaN output (cosine NaN, not just low)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    H, I, E, top_k, layer = 256, 64, 4, 2, 2
    dev = torch.device("cuda")
    cache, layer_types = _make_mixed_cache(H, I, E, seed=100 + bs)
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    torch.manual_seed(400 + bs)
    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    assert not cpu_out.isnan().any(), "layer 1 decoded with the wrong (dominant) format"

    gu_type, dn_type = layer_types[layer]
    gu = _dequant_bank(cache.bank_sources["gate_up"][layer], gu_type, H, dev)
    dn = _dequant_bank(cache.bank_sources["down"][layer], dn_type, I, dev)
    gpu_out = fused_experts_decode_impl(hidden, gu, dn, w, ids.clone(), "silu", False).float()

    cos = torch.nn.functional.cosine_similarity(cpu_out.flatten(), gpu_out.flatten(), dim=0).item()
    assert cos > 0.999, f"bs={bs}: cosine {cos}"
