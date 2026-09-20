"""CUDA toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _nvcc_path() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME:
        return os.path.join(CUDA_HOME, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_cuda_major() -> int | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return int(cuda.split(".")[0]) if cuda else None


def _ensure_blackwell_arch_list() -> None:
    """Default TORCH_CUDA_ARCH_LIST and TVM_FFI_CUDA_ARCH_LIST to include the Blackwell
    "a" variant.

    Without this, JIT kernel builds launched outside `make` (bare `ft serve`,
    pytest, benchmark scripts) silently fall back to auto-detected plain `sm_120`
    (torch's own detection for the gguf.py path, `nvidia-smi --query-gpu=compute_cap`
    for the tvm-ffi/NVFP4 path), dropping the family-specific SASS that unlocks the
    NVFP4 tensor-core instructions (`code=[compute_120a,sm_120a]`).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return
    if major != 12:
        return
    if not os.getenv("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0;12.0a"
    if not os.getenv("TVM_FFI_CUDA_ARCH_LIST"):
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "12.0 12.0a"


@functools.cache
def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA majors.

    nvcc-built binaries link libcudart.so.<nvcc major>; at runtime only the
    torch wheel's own CUDA runtime is guaranteed to be loadable.
    """
    _ensure_blackwell_arch_list()
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )
    if release < (13, 3):
        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} is older than the required CUDA 13.3 for Blackwell SM120 (sm_120a). "
            f"Use CUDA_HOME=/models/outros/cuda-13.3."
        )
