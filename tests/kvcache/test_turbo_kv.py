"""Turbo3/Turbo4 KV codec pins: the transform, the bit packing, the scale semantics, and the
one property the fused attention read path stands on -- that scoring in the rotated domain equals
scoring against the materialized vector.

The reference (llama-turbo-optimal) has no faithful CPU encoder (``quantize_row_turbo3_0_ref`` is
a stub and the turbo4 ref rotates with a dense QR matrix instead of the RWHT), so nothing here can
be checked against llama.cpp bytes. What *is* pinned is the layout, the corrected-norm semantics,
the tie rule and the orthogonality that makes the rotated-domain dot product exact.
"""

import math

import pytest
import torch

from freetoken.kernel.triton import turbo_kv as tk

DEVICE = torch.device("cpu")
BOOKS = list(tk.BOOKS)


def _groups(n: int = 512, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(n, tk.QK_TURBO, generator=g, device=DEVICE)


def test_transcribed_constants_are_complete():
    """The sign arrays and books are hand-transcribed from the reference, so their shape and
    checksum are pinned: a dropped element shifts every rotation by one lane and nothing else
    complains. Sums are the ones the reference tables produce (d_turbo_wht_s1 / _s2)."""
    assert len(tk.SIGNS1) == len(tk.SIGNS2) == tk.QK_TURBO
    assert set(tk.SIGNS1) == {-1, 1} and set(tk.SIGNS2) == {-1, 1}
    assert sum(tk.SIGNS1) == 8
    assert sum(tk.SIGNS2) == -12
    assert len(tk.CENTROIDS_3) == 8 and len(tk.MID_3) == 7
    assert len(tk.CENTROIDS_4) == 16 and len(tk.MID_4) == 15
    for book, cent, mid in (("turbo3", tk.CENTROIDS_3, tk.MID_3), ("turbo4", tk.CENTROIDS_4, tk.MID_4)):
        assert sorted(cent) == list(cent), book
        assert sorted(mid) == list(mid), book
        assert all(cent[i] < mid[i] <= cent[i + 1] for i in range(len(mid))), book
    assert len({tk.CENTROIDS_3[i] + tk.CENTROIDS_3[7 - i] for i in range(8)}) == 1  # symmetric book


def test_butterfly_is_an_involution_up_to_scale():
    x = _groups(64)
    back = tk.butterfly(tk.butterfly(x))
    assert torch.allclose(back, x * tk.QK_TURBO, atol=1e-4)


def test_butterfly_matches_sylvester_hadamard():
    """Stage ordering pin: the distance-h loop must be the same transform as (-1)**popcount(i&j),
    otherwise the encode and the Q-side fold disagree about what "rotated" means."""
    x = _groups(128)
    h = tk.hadamard(tk.QK_TURBO, DEVICE)
    assert torch.allclose(tk.butterfly(x), x @ h, atol=1e-4)


def test_rotate_is_orthogonal_and_preserves_norm():
    x = _groups(128, seed=3)
    y = tk.rotate(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    assert torch.allclose(tk.inv_rotate(y), x, atol=1e-5)
    # the scaled butterfly is its own inverse: rotate and inv_rotate differ only in sign order
    assert torch.allclose(tk.inv_rotate(tk.rotate(x)), x, atol=1e-5)


def test_rotated_dot_equals_original_dot():
    """The fused read path never de-rotates a KV tile: it pre-rotates Q and dots against the
    stored values. This is that identity."""
    x = _groups(256, seed=5)
    q = _groups(64, seed=6)
    y = tk.rotate(x)
    left = q @ tk.inv_rotate(y).T
    right = tk.rotate(q) @ y.T
    assert torch.allclose(left, right, atol=1e-3)


def test_v_accumulation_can_stay_in_the_rotated_domain():
    """Sum p_i W^-1 v_i == W^-1 sum p_i v_i: the output transform happens once per query row."""
    v = _groups(256, seed=7)
    gp = torch.Generator(device=DEVICE).manual_seed(8)
    p = torch.softmax(torch.randn(64, 256, generator=gp, device=DEVICE), dim=-1)
    y = tk.rotate(v)
    left = p @ tk.inv_rotate(y)
    right = tk.inv_rotate(p @ y)
    assert torch.allclose(left, right, atol=1e-3)


@pytest.mark.parametrize("book", BOOKS)
def test_pack_unpack_round_trip(book):
    g = torch.Generator(device=DEVICE).manual_seed(11)
    levels = 8 if book == "turbo3" else 16
    idx = torch.randint(0, levels, (97, tk.QK_TURBO), generator=g, dtype=torch.uint8, device=DEVICE)
    codes = tk.pack(idx, book)
    assert codes.shape == (97, tk.CODE_BYTES[book])
    assert torch.equal(tk.unpack(codes, book), idx.long())


@pytest.mark.parametrize("book", BOOKS)
def test_deduped_norm_costs_the_stated_bytes(book):
    """turbo3 stores one fp16 norm per group where the reference stores four identical copies:
    that is 6 B/row, and the accounting has to say so."""
    ref_row = 14 * 4 if book == "turbo3" else 66
    ours = tk.CODE_BYTES[book] + 2
    assert ours == (50 if book == "turbo3" else 66)
    assert ours <= ref_row
    assert tk.bytes_per_token(book, 8, 128) == 8 * ours
    assert math.isclose(tk.BPV[book], (ours * 8) / 128)


@pytest.mark.parametrize("book", BOOKS)
def test_quantize_decode_recovers_the_group_energy(book):
    """The stored ``norm`` is the *corrected* scale ``||x|| / ||centroid[idx]||``, so the decoded
    vector reproduces the original L2 rather than just its shape."""
    x = _groups(512, seed=13)
    codes, norm = tk.quantize(x, book)
    assert norm.dtype is torch.float16
    back = tk.decode(codes, norm, book)
    assert back.shape == x.shape
    assert torch.allclose(back.norm(dim=-1), x.norm(dim=-1), rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize(
    ("book", "bound", "lloyd_max"),
    [("turbo3", 0.036, 0.0345), ("turbo4", 0.010, 0.0095)],
)
def test_gaussian_nmse_matches_the_lloyd_max_table(book, bound, lloyd_max):
    """The books are Lloyd-Max for a Gaussian source at 8 / 16 levels, whose normalized MSE is
    0.0345 and 0.0095. Measured lands slightly *under* the table because the corrected ``norm``
    projects the reconstruction back onto the input's sphere. If this ever regresses past the
    table, the encoder stopped being a nearest-centroid search on the rotated coordinates."""
    x = _groups(16384, seed=17)
    codes, norm = tk.quantize(x, book)
    err = (tk.decode(codes, norm, book) - x).pow(2).sum() / x.pow(2).sum()
    assert err.item() < bound, f"{book} NMSE {err.item():.5f} exceeds {bound}"
    assert err.item() > lloyd_max * 0.5, f"{book} suspiciously good: {err.item():.5f}"


def test_rotation_lands_coordinates_on_the_book_scale():
    """The books assume N(0, 1/128) coordinates after normalize+rotate. A sign-array or stage-order
    error shows up here first, because it changes the rotated marginal's spread."""
    u = torch.nn.functional.normalize(_groups(4096, seed=31), dim=-1)
    y = tk.rotate(u)
    assert abs(y.std().item() - 128**-0.5) < 1e-4
    assert abs(y.mean().item()) < 1e-3


def test_tie_rule_takes_the_higher_index():
    """A zero coordinate must land above the boundary, matching the reference's strict-< search
    (``val == 0.0`` -> index 4 for turbo3)."""
    zeros = torch.zeros(1, tk.QK_TURBO, device=DEVICE)
    assert set(tk.indices(zeros, "turbo3").tolist()[0]) == {4}
    assert set(tk.indices(zeros, "turbo4").tolist()[0]) == {8}


def test_extremes_clamp_to_the_book_ends():
    big = torch.full((1, tk.QK_TURBO), 9.0, device=DEVICE)
    assert set(tk.indices(big, "turbo3").tolist()[0]) == {7}
    assert set(tk.indices(-big, "turbo3").tolist()[0]) == {0}


def test_correlated_kv_is_not_degraded_by_the_rotation():
    """Real KV is far from white (low-rank, mean-shifted). Rotation is what makes a scalar
    quantizer tolerate that; check it removes the structure instead of amplifying it."""
    g = torch.Generator(device=DEVICE).manual_seed(19)
    base = torch.randn(8, tk.QK_TURBO, generator=g, device=DEVICE)
    x = base.repeat(64, 1) + 0.05 * torch.randn(512, tk.QK_TURBO, generator=g, device=DEVICE)
    codes, norm = tk.quantize(x, "turbo4")
    back = tk.decode(codes, norm, "turbo4")
    err = (back - x).pow(2).sum() / x.pow(2).sum()
    assert err.item() < 0.02, f"NMSE on correlated input {err.item():.5f}"


def test_encoding_is_deterministic():
    x = _groups(256, seed=23)
    for book in BOOKS:
        a = tk.quantize(x, book)
        b = tk.quantize(x, book)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _attend(q, k, v):
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    p = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
    return (p @ v).permute(0, 2, 1, 3)


def _compress(x, book):
    shape = x.shape
    codes, norm = tk.quantize(x.reshape(-1, shape[-1]), book)
    return tk.decode(codes, norm, book).reshape(shape)


def _synth(n=4, m=256, heads=4, d=tk.QK_TURBO, scale=1.0, seed=41):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    make = lambda: scale * torch.randn(n, m, heads, d, generator=g, device=DEVICE)
    return make(), make(), make()


@pytest.mark.parametrize(("book", "floor"), [("turbo3", 0.0345), ("turbo4", 0.0095)])
def test_value_side_error_is_a_floor_that_attention_sharpness_does_not_change(book, floor):
    """V is summed, so its compressed error averages exactly as fast as the output itself does:
    the relative error of the attention output equals the per-vector NMSE, whether the softmax is
    one-hot or uniform. That floor (~the book's Lloyd-Max number) is what no tier schedule can
    remove, and it is the number that decides whether V gets a higher-precision `(layer, side)`."""
    for scale in (0.05, 1.0, 4.0):  # flat -> peaked mixture weights
        q, k, v = _synth(scale=scale)
        ref = _attend(q, k, v)
        got = _attend(q, k, _compress(v, book))
        nmse = ((got - ref).pow(2).sum() / ref.pow(2).sum()).item()
        assert floor * 0.5 < nmse < floor * 2.0, f"{book} at scale {scale}: {nmse:.5f}"


@pytest.mark.parametrize("book", BOOKS)
def test_key_side_error_grows_with_logit_scale_and_only_with_it(book):
    """K enters through exp(), so its damage depends on how much the scores matter. Pinning the
    direction (not a magic constant) is what tells the VBR scheduler to protect peaked layers."""
    errs = []
    for scale in (0.05, 0.5, 2.0):
        q, k, v = _synth(scale=scale)
        ref = _attend(q, k, v)
        got = _attend(q, _compress(k, book), v)
        nmse = ((got - ref).pow(2).sum() / ref.pow(2).sum()).item()
        assert math.isfinite(nmse)
        errs.append(nmse)
    assert errs[0] < errs[1] < errs[2], f"{book} K-side error not monotone in logit scale: {errs}"


@pytest.mark.parametrize("book", BOOKS)
def test_both_sides_together_stay_bounded_and_finite(book):
    q, k, v = _synth(scale=1.0)
    ref = _attend(q, k, v)
    got = _attend(_compress(q, book), _compress(k, book), _compress(v, book))
    assert torch.isfinite(got).all()
    nmse = ((got - ref).pow(2).sum() / ref.pow(2).sum()).item()
    assert nmse < 0.10, f"{book} combined NMSE {nmse:.5f}"


@pytest.mark.parametrize("book", BOOKS)
def test_rotated_domain_scores_match_the_materialized_path(book):
    """The whole performance argument for this codec: scoring against the stored rotated values
    with a pre-rotated Q must give the *same numbers* as scoring against materialized K. If this
    drifts, the fused read path is silently a different model."""
    q, k, _ = _synth(scale=1.0)
    kq = _compress(k, book)
    qh = q.permute(0, 2, 1, 3)
    kh = kq.permute(0, 2, 1, 3)
    d = qh.shape[-1]
    direct = qh @ kh.transpose(-1, -2)
    rq = tk.rotate(qh.reshape(-1, d)).reshape(qh.shape)
    rk = tk.rotate(kh.reshape(-1, d)).reshape(kh.shape)
    assert torch.allclose(direct, rq @ rk.transpose(-1, -2), atol=2e-3)


def test_rejects_a_head_dim_that_is_not_a_group_multiple():
    with pytest.raises(ValueError, match="multiple of 128"):
        tk.quantize(torch.randn(4, 96, device=DEVICE), "turbo4")
    with pytest.raises(ValueError, match="unknown turbo book"):
        tk.quantize(torch.randn(4, 128, device=DEVICE), "turbo9")
