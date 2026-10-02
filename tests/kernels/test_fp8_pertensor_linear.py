"""Per-tensor FP8 linear: the W8A16 kernels against the dequant reference, and the W8A8
(torch._scaled_mm) path against a W8A8 reference.

Which of the two runs is fixed by the deployment -- an ``input_scale`` in the checkpoint plus
sm_89+ -- and never by M, so each is checked against the reference matching its own contract:
W8A16 keeps the activation in bf16 and matches the exact dequant reference, W8A8 quantizes it
to fp8 and is held to a reference that applies the same quantization.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.e4m3_compat import e4m3_native

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def _quant_parts(part_rows: list[int], K: int, seed: int = 0):
    """A fused per-tensor-FP8 weight: each part quantized under its own scalar, exactly how
    modelopt stores q/k/v (and how the loader concatenates them into a per-row vector)."""
    torch.manual_seed(seed)
    N = sum(part_rows)
    wf = torch.randn(N, K, device=DEV) * 0.05
    rows, qs, scales = 0, [], []
    for p in part_rows:
        block = wf[rows : rows + p]
        s = block.abs().max() / 448.0
        qs.append((block / s).clamp(-448, 448).to(FP8))
        scales.append(s.expand(p))
        rows += p
    return torch.cat(qs, 0), torch.cat(scales).contiguous().float()


def _dequant(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.to(torch.float32) * scale[:, None]


# 1 = split-K GEMV; 2..16 = decode batch (the CUDA-graph ladder); 64/300 = prefill.
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 64, 300])
# (5120, 14336) and (5120, 16384) are Qwen3.5-27B's fused qkv_proj / in_proj_qkvz;
# (1024, 6144) is a standalone o_proj shape; 6112 leaves a k-mask tail.
@pytest.mark.parametrize(
    "K,part_rows",
    [
        (5120, [12288, 1024, 1024]),
        (1024, [6144]),
        (6112, [512, 128]),
    ],
)
def test_w8a16_matches_dequant_reference(M: int, K: int, part_rows: list[int]):
    """Without an ``input_scale`` every M stays on the W8A16 kernels, activation exact."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = fp8_pertensor_linear(x, w8, scale)
    y_ref = (x.float() @ _dequant(w8, scale).t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.parametrize("M", [1, 2, 6])
def test_batched_gemv_rows_matches_sequential_gemv_bitwise(M: int):
    """Batched split-K GEMV preserves each row's M=1 arithmetic, including K tail masks."""
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
    from freetoken.kernel.triton.fp8_pertensor_linear import _gemv, _gemv_rows

    K = 259  # two full 128-wide chunks plus a masked tail
    w8, scale = _quant_parts([19, 18], K, seed=M)
    weight = e4m3_kernel_view(w8)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    batched = _gemv_rows(x, weight, scale, x.dtype)
    sequential = torch.stack([_gemv(row, weight, scale, x.dtype) for row in x])
    assert torch.equal(batched, sequential)


@pytest.mark.parametrize("M", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("N,K", [(8192, 2048), (2048, 4096), (12288, 2048), (48, 2048), (500, 777)])
def test_row_invariant_linear_bitwise_equals_m1(monkeypatch, M: int, N: int, K: int):
    """The verify-window linear returns, per row, the exact bits of the M==1 decode call."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    w8, scale = _quant_parts([N], K, seed=N + K)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 3
    bias = torch.randn(N, device=DEV, dtype=torch.bfloat16)
    monkeypatch.setenv("FREETOKEN_ROW_INVARIANT_LINEAR", "1")
    for b in (None, bias):
        rows = fp8_pertensor_linear(x, w8, scale, b)
        single = torch.cat([fp8_pertensor_linear(x[i : i + 1], w8, scale, b) for i in range(M)])
        assert torch.equal(rows.view(torch.int16), single.view(torch.int16))


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 2, 4, 16, 64])
@pytest.mark.parametrize(
    "part_rows,uniform",
    [
        ([12288, 1024, 1024], False),  # fused -> piecewise-constant scale -> row-wise
        ([6144], True),  # standalone -> one scalar -> tensor-wise
    ],
)
def test_w8a8_matches_w8a8_reference(M: int, part_rows: list[int], uniform: bool):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K = 2048
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    y = fp8_pertensor_linear(x, w8, scale, None, input_scale, uniform)

    xq = (x.float() / input_scale).clamp(-448, 448).to(FP8)
    y_ref = (xq.to(torch.float32) * input_scale) @ _dequant(w8, scale).t()
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-2, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_batch_size_does_not_change_the_numeric_scheme():
    """A deployment that can run W8A8 must run it at every M, so that a reply reproduces at
    bs=1 regardless of how many other requests shared its forward. Feeding the same row alone
    and as part of a batch must therefore agree bit-for-bit."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K, part_rows = 2048, [1024, 256]
    w8, scale = _quant_parts(part_rows, K)
    x = torch.randn(8, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    batched = fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    alone = fp8_pertensor_linear(x[:1], w8, scale, None, input_scale, False)
    assert torch.equal(alone, batched[:1])


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 4, 64, 300])
def test_per_part_path_matches_rowwise(M: int, monkeypatch):
    """Where row-wise ``_scaled_mm`` is unsafe a fused projection runs one tensor-wise GEMM per
    part instead. Same scheme, so the two paths agree up to accumulation order (~7e-4)."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    K, part_rows = 2048, [1024, 256, 256]
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: True)
    y_row = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: False)
    y_part = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    rel = ((y_part.float() - y_row.float()).norm() / y_row.float().norm()).item()
    assert rel < 2e-3, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_fused_layer_forward_on_a_side_stream_completes():
    """Regression for #182 / #72 / #220: on sm_89 with torch < 2.12 a fused FP8 projection's
    row-wise ``_scaled_mm`` issued from a non-default stream stalls the GPU (PyTorch's
    CUTLASS row-wise kernel ignored the current stream; fixed upstream in 2.12). The layer
    must take a path that completes on every supported build. Runs in a subprocess so a
    stall fails the test instead of hanging the session."""
    script = textwrap.dedent("""
        import os, time, torch
        from freetoken.kernel.triton.fp8_pertensor_linear import (
            FP8, fp8_pertensor_linear, rowwise_scaled_mm_ok, weight_scale_segments,
        )

        torch.manual_seed(0)
        K, parts = 2048, [8192, 512, 512]  # a prefill-sized fused qkv
        w8 = (torch.randn(sum(parts), K, device="cuda") * 8).clamp(-448, 448).to(FP8)
        scale = torch.cat([torch.full((p,), 0.01 * (i + 1), device="cuda")
                           for i, p in enumerate(parts)])
        input_scale = torch.tensor(0.02, device="cuda")
        # same load-time decisions the torch fp8_tensor kernel makes in finalize
        segments = weight_scale_segments(scale)
        rowwise_scaled_mm_ok()
        x = torch.randn(2010, K, device="cuda", dtype=torch.bfloat16)  # #182 shape
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        events = []
        with torch.cuda.stream(stream):
            for _ in range(128):
                fp8_pertensor_linear(x, w8, scale, None, input_scale, False, scale_segments=segments)
                ev = torch.cuda.Event()
                ev.record(stream)
                events.append(ev)
        deadline = time.monotonic() + 30
        done = 0
        while time.monotonic() < deadline:
            while done < len(events) and events[done].query():
                done += 1
            if done == len(events):
                print("completed", done, flush=True)
                os._exit(0)
            time.sleep(0.05)
        print("stalled at", done, "of", len(events), flush=True)
        os._exit(124)  # a normal exit would wait on the stuck kernel
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("part_rows", [[2048], [1024, 512, 512]])
def test_w8a8_rows_exact_probe_gates_batched_verify(monkeypatch, part_rows: list[int]):
    """A probed-exact shape serves the verify window in one GEMM with each row's M==1 bits."""
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        fp8_pertensor_linear,
        w8a8_rows_exact,
        weight_scale_segments,
    )

    K = 1024
    w8, scale = _quant_parts(part_rows, K, seed=7)
    uniform = len(part_rows) == 1
    segments = None if uniform else weight_scale_segments(scale)
    input_scale = torch.tensor(0.02, device=DEV)
    exact = w8a8_rows_exact(w8, scale, input_scale, uniform, segments)
    monkeypatch.setenv("FREETOKEN_ROW_INVARIANT_LINEAR", "1")
    x = torch.randn(6, K, device=DEV, dtype=torch.bfloat16)
    args = (w8, scale, None, input_scale, uniform, segments)
    rows = fp8_pertensor_linear(x, *args, rows_exact=exact)
    single = torch.cat([fp8_pertensor_linear(x[i : i + 1], *args) for i in range(6)])
    assert torch.equal(rows.view(torch.int16), single.view(torch.int16))
