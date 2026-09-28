"""CPU MoE executor -- native GGUF i-quant experts (IQ3_S, IQ4_XS, IQ4_NL, Q8_0).

Companion to test_cpu_moe_kquant.py (Q4_K/Q6_K) and test_cpu_moe_q4_0.py (Q4_0), but
CPU-only: no CUDA context is created here (the GPU is reserved by the campaign
orchestrator while this test was written), so this does not go through
``CpuMoeExecutor`` (its ctor and decode() need a CUDA device/stream). Instead it
exercises the new scalar dot kernels directly through the ``*_dot_cpu`` pybind test
hooks added next to ``max_weight_format_id`` in cpu_moe_ext.cpp, and checks them
against ``gguf.quants.dequantize`` -- the same reference dequant the production GPU
decode path is checked against elsewhere -- rather than a hand-rolled Python decode.

Reference checkpoint layout (see C0-cpu-hybrid-audit.md): routed experts use IQ3_S
gate/up (IQ4_XS on one layer), IQ4_NL/Q8_0 down.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, quants

from freetoken.kernel import _cpu_moe
from freetoken.moe.cpu_executor import _GGML_TO_CPU_FMT, _GGUF_KQUANT_BLOCK, _WFMT_IDS

# (format name, block bytes, block elements, ggml type id, dot hook)
_CASES = [
    ("iq3_s", 110, 256, GGMLQuantizationType.IQ3_S, _cpu_moe.iq3_s_dot_cpu),
    ("iq4_xs", 136, 256, GGMLQuantizationType.IQ4_XS, _cpu_moe.iq4_xs_dot_cpu),
    ("iq4_nl", 18, 32, GGMLQuantizationType.IQ4_NL, _cpu_moe.iq4_nl_dot_cpu),
    ("q8_0", 34, 32, GGMLQuantizationType.Q8_0, _cpu_moe.q8_0_dot_cpu),
]


# Relative-error gate per format. IQ3_S needs a wider gate than the other three: its
# scalar reference (iq3_s_dot_f32_scalar, ported element-for-element from
# dequantize_row_iq3_s) already lands at ~1.4e-3 max relative error against
# gguf.quants.dequantize over a 500-trial sweep (not just the AVX-512 kernel added on
# top of it, which tracks the scalar kernel to ~2e-5) -- i.e. this is a real fp32
# dequant-then-FMA-order spread against the gguf reference's own IQ3_S dequant path,
# not a kernel bug, so 1e-4 was never reachable here. IQ4_XS/IQ4_NL/Q8_0 stay at the
# original tight gate.
_REL_TOL = {"iq3_s": 2e-3}
_DEFAULT_REL_TOL = 1e-4


def _random_block(rng: np.random.Generator, nbytes: int, scale: float) -> bytes:
    """A syntactically valid block: random quant/index/sign bytes (every bit pattern
    decodes to *some* value for these formats -- grid/LUT indices never go out of
    range) with the leading fp16 ``d`` field pinned to a small finite scale, matching
    the K-quant fixtures in test_cpu_moe_kquant.py."""
    raw = bytearray(rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes())
    raw[0:2] = np.float16(scale).tobytes()
    return bytes(raw)


@pytest.mark.parametrize("fmt,nbytes,K,ggml_type,dot_fn", _CASES, ids=[c[0] for c in _CASES])
def test_dot_kernel_matches_gguf_reference_dequant(fmt, nbytes, K, ggml_type, dot_fn):
    rng = np.random.default_rng(hash(fmt) & 0xFFFF)
    max_rel_err = 0.0
    for trial in range(20):
        raw = _random_block(rng, nbytes, scale=0.01 + 0.02 * rng.random())
        w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        ref = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), ggml_type).astype(np.float32)
        assert ref.shape == (K,)

        x_f32 = rng.standard_normal(K).astype(np.float32) * 0.5
        x_bf16 = torch.from_numpy(x_f32).to(torch.bfloat16)
        expected = float(np.dot(ref, x_bf16.float().numpy()))

        got = dot_fn(w, x_bf16)
        rel_err = abs(got - expected) / (abs(expected) + 1e-6)
        max_rel_err = max(max_rel_err, rel_err)

    # Both sides are W4A16 (bf16 activations, fp32 accumulate); the only spread is
    # activation bf16-rounding + fp32 reduction order, so this stays tight -- a wrong
    # block offset or table (e.g. iq3s_grid index math, kvalues_iq4nl LUT) blows this
    # up by orders of magnitude rather than nudging it. IQ3_S gets a wider gate; see
    # _REL_TOL above.
    tol = _REL_TOL.get(fmt, _DEFAULT_REL_TOL)
    assert max_rel_err < tol, f"{fmt}: max relative error {max_rel_err}"


@pytest.mark.parametrize("fmt,nbytes,K,ggml_type,dot_fn", _CASES, ids=[c[0] for c in _CASES])
def test_multi_block_row_matches_reference(fmt, nbytes, K, ggml_type, dot_fn):
    """A 2-block row (exercises the per-block loop / row-advance arithmetic, not just
    a single block)."""
    rng = np.random.default_rng((hash(fmt) & 0xFFFF) ^ 0x5EED)
    n_blocks = 2
    raw = b"".join(
        _random_block(rng, nbytes, scale=0.01 + 0.02 * rng.random()) for _ in range(n_blocks)
    )
    w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    ref = np.concatenate(
        [
            quants.dequantize(
                np.frombuffer(raw[i * nbytes : (i + 1) * nbytes], dtype=np.uint8), ggml_type
            )
            for i in range(n_blocks)
        ]
    ).astype(np.float32)

    x_f32 = rng.standard_normal(K * n_blocks).astype(np.float32) * 0.5
    x_bf16 = torch.from_numpy(x_f32).to(torch.bfloat16)
    expected = float(np.dot(ref, x_bf16.float().numpy()))
    got = dot_fn(w, x_bf16)
    rel_err = abs(got - expected) / (abs(expected) + 1e-6)
    tol = _REL_TOL.get(fmt, _DEFAULT_REL_TOL)
    assert rel_err < tol, f"{fmt}: rel error {rel_err}"


def _q8_0_quantize(x_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror cpu_moe_ext.cpp ``quant_q8_0``: per-32-block symmetric int8 + fp32 scale."""
    assert x_f32.size % 32 == 0
    blocks = x_f32.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    d = np.where(amax > 0.0, amax / 127.0, 1.0).astype(np.float32)
    inv = np.where(amax > 0.0, 1.0 / d, 0.0).astype(np.float32)
    aq = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int8)
    return aq.reshape(-1), d


@pytest.mark.skipif(
    not _cpu_moe.iq3_s_dot_i8_available(), reason="no AVX512-VNNI W4A8 IQ3_S dot on this CPU"
)
def test_iq3s_w4a8_simd_matches_scalar_reference():
    """The SIMD int8 kernel must track the scalar int8 reference: same integer math, so
    the only spread is fp accumulation order. A wrong iq3s_grid index, a mis-derived sign
    mask, or a bad sub-block scale shows up here as an order-1 error, not a nudge."""
    rng = np.random.default_rng(0x1035)
    for n_blocks in (1, 2, 10):
        K = 256 * n_blocks
        raw = b"".join(
            _random_block(rng, 110, scale=0.01 + 0.02 * rng.random()) for _ in range(n_blocks)
        )
        w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        aq, asb = _q8_0_quantize((rng.standard_normal(K).astype(np.float32) * 0.5))
        aq_t = torch.from_numpy(aq)
        asb_t = torch.from_numpy(asb.astype(np.float32))
        got_ref = _cpu_moe.iq3_s_dot_i8_scalar_cpu(w, aq_t, asb_t)
        got_simd = _cpu_moe.iq3_s_dot_i8_cpu(w, aq_t, asb_t)
        rel = abs(got_simd - got_ref) / (abs(got_ref) + 1e-9)
        assert rel < 1e-5, f"n_blocks={n_blocks}: simd {got_simd} vs scalar {got_ref} rel {rel}"


@pytest.mark.skipif(
    not _cpu_moe.iq3_s_dot_i8_available(), reason="no AVX512-VNNI W4A8 IQ3_S dot on this CPU"
)
def test_iq3s_w4a8_matches_gguf_dequant_reference():
    """Both int8 kernels against ``gguf.quants.dequantize`` weights dotted with the
    Q8_0-round-tripped activation, so activation quantization is not part of the error."""
    rng = np.random.default_rng(0x2035)
    max_rel = 0.0
    for n_blocks in (1, 4):
        K = 256 * n_blocks
        raw = b"".join(
            _random_block(rng, 110, scale=0.01 + 0.02 * rng.random()) for _ in range(n_blocks)
        )
        w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        ref_w = np.concatenate(
            [
                quants.dequantize(
                    np.frombuffer(raw[i * 110 : (i + 1) * 110], dtype=np.uint8),
                    GGMLQuantizationType.IQ3_S,
                )
                for i in range(n_blocks)
            ]
        ).astype(np.float32)
        assert ref_w.shape == (K,)
        x = rng.standard_normal(K).astype(np.float32) * 0.5
        aq, asb = _q8_0_quantize(x)
        expected = float(np.dot(ref_w, aq.astype(np.float32) * np.repeat(asb, 32)))
        for hook in (_cpu_moe.iq3_s_dot_i8_scalar_cpu, _cpu_moe.iq3_s_dot_i8_cpu):
            got = hook(w, torch.from_numpy(aq), torch.from_numpy(asb.astype(np.float32)))
            max_rel = max(max_rel, abs(got - expected) / (abs(expected) + 1e-6))
    assert max_rel < _REL_TOL["iq3_s"], f"W4A8 IQ3_S max relative error {max_rel}"


@pytest.mark.skipif(
    not _cpu_moe.iq3_s_dot_i8_available(), reason="no AVX512-VNNI W4A8 IQ3_S dot on this CPU"
)
def test_iq3s_w4a8_stays_close_to_w4a16_path():
    """Bound the numerics change the W4A8 switch introduces: same weights, same underlying
    fp32 activation, f32 (bf16-rounded) path versus Q8_0-quantized path. Q8_0 activation
    quantization is genuinely lossy, so this is a loose statistical gate -- it catches a
    systematically wrong scale, not last-bit drift."""
    rng = np.random.default_rng(0x3035)
    K = 2560  # the checkpoint's real gate_up K (H), 10 IQ3_S blocks
    raw = b"".join(_random_block(rng, 110, scale=0.01 + 0.02 * rng.random()) for _ in range(10))
    w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    x_f32 = rng.standard_normal(K).astype(np.float32) * 0.5
    x_bf16 = torch.from_numpy(x_f32).to(torch.bfloat16)
    aq, asb = _q8_0_quantize(x_bf16.float().numpy())
    f32 = _cpu_moe.iq3_s_dot_cpu(w, x_bf16)
    i8 = _cpu_moe.iq3_s_dot_i8_cpu(
        w, torch.from_numpy(aq), torch.from_numpy(asb.astype(np.float32))
    )
    rel = abs(i8 - f32) / (abs(f32) + 1e-6)
    assert rel < 2e-2, f"W4A8 {i8} drifted from W4A16 {f32} by rel {rel}"


def test_max_weight_format_id_covers_new_formats():
    assert _cpu_moe.max_weight_format_id() >= _WFMT_IDS["q8_0"]
    assert _WFMT_IDS["iq3_s"] == 7
    assert _WFMT_IDS["iq4_xs"] == 8
    assert _WFMT_IDS["iq4_nl"] == 9
    assert _WFMT_IDS["q8_0"] == 10


def test_block_geometry_matches_ggml():
    # Must agree with the row-byte arithmetic in cpu_moe_ext.cpp's ctor and with
    # freetoken.models.gguf.dequant's own (elements, bytes) table.
    assert _GGUF_KQUANT_BLOCK["iq3_s"] == (256, 110)
    assert _GGUF_KQUANT_BLOCK["iq4_xs"] == (256, 136)
    assert _GGUF_KQUANT_BLOCK["iq4_nl"] == (32, 18)
    assert _GGUF_KQUANT_BLOCK["q8_0"] == (32, 34)


def test_ggml_type_ids_resolve_to_cpu_formats():
    # GGML_IQ3_S=21, GGML_IQ4_XS=23, GGML_IQ4_NL=20, GGML_Q8_0=8
    # (freetoken/models/gguf/dequant.py).
    assert _GGML_TO_CPU_FMT[21] == "iq3_s"
    assert _GGML_TO_CPU_FMT[23] == "iq4_xs"
    assert _GGML_TO_CPU_FMT[20] == "iq4_nl"
    assert _GGML_TO_CPU_FMT[8] == "q8_0"


def test_mixed_gate_up_down_types_resolve_to_independent_formats():
    """The checkpoint under test mixes IQ3_S gate/up with IQ4_NL or Q8_0 down on most
    layers. The C++ executor now takes an independent weight_format per bank for the
    K-quant/I-quant family (gemm1_dot dispatches on gate_up's format, gemm2_dot on
    down's -- see fmt/fmt_dn and down_weight_format in cpu_moe_ext.cpp), so
    _resolve_gguf_format resolves this to a (gate_up, down) pair instead of refusing
    it; each bank's own dot kernel/block geometry is used, matching the previous
    single-format contract exactly when gate_up == down."""
    from types import SimpleNamespace

    from freetoken.moe.cpu_executor import _resolve_gguf_format

    cache = SimpleNamespace(quant_format="gguf", gguf_expert_types=(21, 20))  # IQ3_S, IQ4_NL
    assert _resolve_gguf_format(cache) == ("iq3_s", "iq4_nl")

    cache2 = SimpleNamespace(quant_format="gguf", gguf_expert_types=(21, 8))  # IQ3_S, Q8_0
    assert _resolve_gguf_format(cache2) == ("iq3_s", "q8_0")


def test_mixed_format_executor_runs_gemm1_act_gemm2_end_to_end():
    """A real CpuMoeExecutor, gate_up=IQ3_S / down=IQ4_NL, run on CPU (no CUDA device,
    no CUDA-stream decode() path -- construct with device="cpu" and drive the
    lower-level create_task/run_task entry directly, which is exactly what decode()
    does after its D2H/H2D). This is the actual coordinator-required proof: not just
    that _resolve_gguf_format names the right pair, but that gemm1 (IQ3_S dequant),
    the swiglu epilogue, and gemm2 (IQ4_NL dequant) all run correctly end-to-end with
    two different weight formats live in the same executor instance -- the exact
    combination test_mixed_gate_up_down_types_are_rejected_cleanly used to assert was
    impossible. Reference is the same *_dot_cpu kernels called row-by-row from Python
    plus a hand-rolled silu-swiglu + weighted-sum, not gguf.quants.dequantize, since
    this checks the executor's block-address/row-stride plumbing (right bank, right
    row, right per-bank format), not kernel numerics (already covered above)."""
    from types import SimpleNamespace

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    H, I, L, E, top_k, bs = 256, 256, 1, 4, 2, 3
    gu_qk, gu_blk = _GGUF_KQUANT_BLOCK["iq3_s"]
    dn_qk, dn_blk = _GGUF_KQUANT_BLOCK["iq4_nl"]
    rng = np.random.default_rng(7)

    def make_bank(S: int, OUT: int, K: int, qk: int, blk: int) -> torch.Tensor:
        nb = K // qk
        raw = rng.integers(0, 256, size=(S, OUT, nb, blk), dtype=np.uint8)
        d = (0.01 + 0.02 * rng.random((S, OUT, nb))).astype(np.float16)
        raw[..., 0:2] = d.view(np.uint8).reshape(S, OUT, nb, 2)
        return torch.from_numpy(raw.reshape(S, OUT, nb * blk).copy())

    gate_up_flat = make_bank(L * E, 2 * I, H, gu_qk, gu_blk)  # IQ3_S rows, K=H
    down_flat = make_bank(L * E, H, I, dn_qk, dn_blk)  # IQ4_NL rows, K=I
    gate_up = list(gate_up_flat.split(E))  # L tensors [E, 2I, row_bytes]
    down = list(down_flat.split(E))  # L tensors [E, H, row_bytes]

    cache = SimpleNamespace(
        quant_format="gguf",
        gguf_expert_types=(21, 20),  # IQ3_S, IQ4_NL
        bank_sources={"gate_up": gate_up, "down": down},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=torch.device("cpu"),
    )
    assert ex.quant_format == "iq3_s" and ex.quant_format_down == "iq4_nl"

    torch.manual_seed(11)
    hidden = (torch.randn(bs, H) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, dtype=torch.float32)
    y = torch.zeros(bs, H, dtype=torch.bfloat16)

    task = ex._ext.create_task(0, bs, hidden.data_ptr(), ids.data_ptr(), w.data_ptr(), y.data_ptr())
    ex._ext.run_task(task)

    def silu(x: float) -> float:
        return x / (1.0 + np.exp(-x))

    gate_up_layer, down_layer = gate_up[0], down[0]
    expected = torch.zeros(bs, H, dtype=torch.float32)
    for t in range(bs):
        acc = np.zeros(H, dtype=np.float64)
        for k in range(top_k):
            e = int(ids[t, k])
            wk = float(w[t, k])
            xb = hidden[t]
            g = np.empty(I, dtype=np.float64)
            for i in range(I):
                gate = _cpu_moe.iq3_s_dot_cpu(gate_up_layer[e, i].contiguous(), xb)
                up = _cpu_moe.iq3_s_dot_cpu(gate_up_layer[e, I + i].contiguous(), xb)
                g[i] = silu(gate) * up
            g_bf16 = torch.from_numpy(g.astype(np.float32)).to(torch.bfloat16)
            for h in range(H):
                acc[h] += _cpu_moe.iq4_nl_dot_cpu(down_layer[e, h].contiguous(), g_bf16) * wk
        expected[t] = torch.from_numpy(acc.astype(np.float32))

    got = y.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), expected.flatten(), dim=0).item()
    rel = ((got - expected).abs().max() / (expected.abs().max() + 1e-6)).item()
    assert cos > 0.999, f"mixed-format executor: cosine {cos} (rel {rel})"
    assert rel < 5e-2, f"mixed-format executor: rel {rel} (cosine {cos})"
