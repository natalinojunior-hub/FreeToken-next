"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import atexit
import functools
import glob
import json
import logging
import os
import pathlib
import shutil
import time

import torch

import sys

venv_bin = os.path.join(sys.prefix, "bin")
if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

if "CUDA_HOME" not in os.environ and os.path.isdir("/models/outros/cuda-13.3"):
    os.environ["CUDA_HOME"] = "/models/outros/cuda-13.3"
if os.path.isdir(
    "/models/outros/cuda-13.3/bin"
) and "/models/outros/cuda-13.3/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/models/outros/cuda-13.3/bin:{os.environ.get('PATH', '')}"

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"

_TRACE_DIR = os.environ.get("FREETOKEN_GGUF_TRACE", "").strip()
_TRACE = {
    kind: {"calls": 0, "ms": 0.0, "host_ms": 0.0, "read": 0, "write": 0}
    for kind in ("a8_total", "quantize", "a8_prequant")
}
_TRACE_CALLS = 0


def _flush_trace() -> None:
    if _TRACE_DIR:
        path = pathlib.Path(_TRACE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"gguf-{os.getpid()}.json").write_text(json.dumps(_TRACE))


def _trace_call(kind: str, fn, read: int, write: int):
    global _TRACE_CALLS
    if not _TRACE_DIR or not torch.cuda.is_available():
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    row = _TRACE[kind]
    row["calls"] += 1
    row["ms"] += start.elapsed_time(end)
    row["host_ms"] += (time.perf_counter() - host_start) * 1000
    row["read"] += read
    row["write"] += write
    _TRACE_CALLS += 1
    if _TRACE_CALLS % 256 == 0:
        _flush_trace()
    return out


@atexit.register
def _write_trace() -> None:
    _flush_trace()


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc


_BUILD_LOCK_STALE_S = 60.0
_COMPILER_COMM = {
    "ninja",
    "nvcc",
    "cicc",
    "ptxas",
    "cc1plus",
    "cudafe++",
    "fatbinary",
    "gcc",
    "g++",
    "clang",
}


def _compiler_running() -> bool:
    for comm in glob.glob("/proc/[0-9]*/comm"):
        try:
            with open(comm) as stream:
                if stream.read().strip() in _COMPILER_COMM:
                    return True
        except OSError:
            continue
    return False


def _clear_stale_build_lock(name: str) -> None:
    """A builder killed mid-compile leaves ``lock`` behind and torch's FileBaton then waits on
    it forever (a silent boot hang). Stale = older than a minute with no compiler running."""
    from torch.utils.cpp_extension import _get_build_directory

    lock = os.path.join(_get_build_directory(name, verbose=False), "lock")
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return
    if age > _BUILD_LOCK_STALE_S and not _compiler_running():
        os.remove(lock)
        logging.getLogger(__name__).warning(f"removed stale JIT build lock {lock} ({age:.0f}s old)")


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    from freetoken.kernel._toolchain import check_nvcc_matches_torch

    check_nvcc_matches_torch()

    extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    _clear_stale_build_lock("freetoken_gguf_kernels")
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _trace_call(
        "a8_total",
        lambda: _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row),
        weight.numel() * weight.element_size() + x.numel() * x.element_size(),
        row * x.shape[0] * x.element_size(),
    )


def ggml_quantize_row_q8_1(x: torch.Tensor) -> torch.Tensor:
    """Quantize activation ``x`` -> q8_1 block buffer once, for reuse across MMVQ parts."""
    padded = (x.shape[1] + 511) // 512 * 512
    return _trace_call(
        "quantize",
        lambda: _module().ggml_quantize_row_q8_1(x),
        x.numel() * x.element_size(),
        x.shape[0] * (padded // 32 * 9) * 4,
    )


def ggml_mul_mat_vec_a8_prequant(
    weight: torch.Tensor,
    quant_x: torch.Tensor,
    quant_type: int,
    row: int,
    col: int,
    vecs: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """MMVQ against a pre-quantized q8_1 activation (bit-exact vs ggml_mul_mat_vec_a8)."""
    return _trace_call(
        "a8_prequant",
        lambda: _module().ggml_mul_mat_vec_a8_prequant(
            weight, quant_x, quant_type, row, col, vecs, out_dtype
        ),
        weight.numel() * weight.element_size() + quant_x.numel() * quant_x.element_size(),
        row * vecs * torch.tensor([], dtype=out_dtype).element_size(),
    )


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x,
        weight,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_quantize_row_q8_1",
    "ggml_mul_mat_vec_a8_prequant",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
