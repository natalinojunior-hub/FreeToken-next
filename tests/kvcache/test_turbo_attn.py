"""GPU pins for the fused tile readers: what a Triton kernel reconstructs from the codes must be
the same values the torch codec says are stored, element for element.

This is the seam where a pointer-stride or bit-shift error would otherwise surface months later as
garbled generation, because everything above it (pool, backend) only ever sees the tile.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from freetoken.kernel.triton import turbo_kv as tk  # noqa: E402
from freetoken.kernel.triton.turbo_attn import turbo_k_tile, turbo_v_tile  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@triton.jit
def _probe_tiles(
    codes_ptr, norm_ptr, cent_ptr, k_out_ptr, v_out_ptr,
    stride_ct, stride_ch, stride_nt, stride_nh,
    D: tl.constexpr, N: tl.constexpr, CB: tl.constexpr, BOOK3: tl.constexpr, NREAL: tl.constexpr,
):
    offs_d = tl.arange(0, D)
    slots = tl.arange(0, N)
    mask = slots < NREAL
    kt = turbo_k_tile(codes_ptr, norm_ptr, cent_ptr, slots, 0, stride_ct, stride_ch,
                      stride_nt, stride_nh, offs_d, mask, BOOK3, tl.float32)
    vt = turbo_v_tile(codes_ptr, norm_ptr, cent_ptr, slots, 0, stride_ct, stride_ch,
                      stride_nt, stride_nh, offs_d, mask, BOOK3, tl.float32)
    tl.store(k_out_ptr + offs_d[:, None] * N + slots[None, :], kt)
    tl.store(v_out_ptr + slots[:, None] * D + offs_d[None, :], vt)


@pytest.mark.parametrize("book", list(tk.BOOKS))
@pytest.mark.parametrize("head_dim", [128, 256])
def test_tile_readers_reconstruct_exactly_what_the_codec_stored(book, head_dim):
    device = torch.device("cuda")
    groups = head_dim // tk.QK_TURBO
    tokens = 32
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(tokens, head_dim, generator=g, device=device, dtype=torch.bfloat16)
    codes, norm = tk.quantize(x, book)
    codes = codes.contiguous()
    norm = norm.contiguous()  # [tokens, groups]

    # pool layout is [tokens, heads=head_dim//head_dim... ] -- one head here, so head axis = 1
    codes4 = codes.reshape(tokens, 1, groups * tk.CODE_BYTES[book])
    norm2 = norm.reshape(tokens, 1, groups)
    cent = tk.CENTROIDS_3 if book == "turbo3" else tk.CENTROIDS_4
    cent_t = torch.tensor(cent, device=device, dtype=torch.float32)

    k_out = torch.empty(head_dim, tokens, device=device, dtype=torch.float32)
    v_out = torch.empty(tokens, head_dim, device=device, dtype=torch.float32)
    _probe_tiles[(1,)](
        codes4, norm2, cent_t, k_out, v_out,
        stride_ct=groups * tk.CODE_BYTES[book], stride_ch=groups * tk.CODE_BYTES[book],
        stride_nt=groups, stride_nh=groups,
        D=head_dim, N=tokens, CB=tk.CODE_BYTES[book], BOOK3=book == "turbo3", NREAL=tokens,
    )
    want = tk.decode_rotated(codes, norm, book)  # [tokens, head_dim]
    assert torch.equal(k_out.T.contiguous(), want), "K tile must equal the stored rotated values"
    assert torch.equal(v_out, want), "V tile must equal the stored rotated values"


@pytest.mark.parametrize("book", list(tk.BOOKS))
def test_masked_lanes_read_zero(book):
    """Padded slots (the dummy page, a split's tail) must contribute 0, not the first row's values:
    an unmasked byte gather would silently attend to a stale slot."""
    device = torch.device("cuda")
    tokens, real = 16, 5
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(tokens, tk.QK_TURBO, generator=g, device=device, dtype=torch.bfloat16)
    codes, norm = tk.quantize(x, book)
    codes4 = codes.reshape(tokens, 1, tk.CODE_BYTES[book]).contiguous()
    norm2 = norm.reshape(tokens, 1, 1).contiguous()
    cent_t = torch.tensor(tk.CENTROIDS_3 if book == "turbo3" else tk.CENTROIDS_4, device=device)
    k_out = torch.zeros(tk.QK_TURBO, tokens, device=device, dtype=torch.float32)
    v_out = torch.zeros(tokens, tk.QK_TURBO, device=device, dtype=torch.float32)
    _probe_tiles[(1,)](
        codes4, norm2, cent_t, k_out, v_out,
        stride_ct=tk.CODE_BYTES[book], stride_ch=tk.CODE_BYTES[book],
        stride_nt=1, stride_nh=1,
        D=tk.QK_TURBO, N=tokens, CB=tk.CODE_BYTES[book], BOOK3=book == "turbo3", NREAL=real,
    )
    want = tk.decode_rotated(codes, norm, book)
    assert torch.equal(k_out[:, :real], want[:real].T.contiguous())
    assert torch.equal(v_out[:real], want[:real])
    assert k_out[:, real:].eq(0).all(), "masked lanes leaked a real slot's value"
    assert v_out[real:].eq(0).all()
