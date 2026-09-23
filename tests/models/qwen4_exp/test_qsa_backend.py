"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


@requires_cuda
def test_turbo4_split_path_runs():
    """Exercise the separate TurboKV decompression and QSA attention path."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=32, max_running_req=1, kv_format="turbo4")
    attn = fixture.layer(QSA_LAYER)
    length = 65
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)

    got = attn.forward(x, fixture.batch([req], "prefill"))

    assert got.shape == x.shape
    assert torch.isfinite(got).all()


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra,
            fixture.config.hidden_size,
            device=fixture.device,
            dtype=fixture.dtype,
            generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(q, k, v, layer_id, batch)
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize(
    "cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"]
)
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs,
        reqs=[dummy] * bs,
        phase="decode",
        size=bs,
        padded_size=bs,
        is_prefill=False,
        is_decode=True,
        positions=static["positions"],
        get_attn_positions=lambda: static["positions"],
        out_loc=static["out_loc"],
        attn_metadata=None,
        active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


def _kv_state(fixture: Fixture) -> list[torch.Tensor]:
    """Every device tensor of the QSA pool (K/V slabs, pending ring, compressed index slab)."""
    owners = (fixture.pool, getattr(fixture.pool, "_pool", None))
    values = [v for owner in owners if owner is not None for v in vars(owner).values()]
    flat = [t for v in values for t in (v if isinstance(v, (list, tuple)) else (v,))]
    return [t for t in flat if isinstance(t, torch.Tensor) and t.is_cuda]


def _verify_batch(fixture: Fixture, req, static: dict | None = None):
    """A spec-verify window (prefill phase, spec_logits_indices set); ``static`` rebinds the
    positions/out_loc to capture buffers after copying this window's values in."""
    batch = fixture.batch([req], "prefill")
    batch.spec_logits_indices = torch.arange(req.extend_len, device=fixture.device)
    batch.input_ids = torch.zeros(req.extend_len, dtype=torch.int32, device=fixture.device)
    if static is not None:
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        batch.positions, batch.out_loc = static["positions"], static["out_loc"]
        batch.get_attn_positions = lambda: static["positions"]
    return batch


@requires_cuda
@pytest.mark.parametrize("kv_format", ["auto", "turbo3"])
def test_verify_graph_replay_matches_eager(kv_format):
    """A captured 2-token spec-verify window (stage_verify) reproduces the eager verify
    bitwise -- output and every pool tensor -- over accepted and rejected windows whose draft
    rows change. Under capture the turbo path decompresses every page instead of the
    selected ones; the attended values must not change."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=256, kv_format=kv_format)
    attn = fixture.layer(QSA_LAYER)
    length, tokens = 300, 2
    accepts = [True, False, True, True, False, False, True]
    rows = _inputs(fixture, [length], extra=tokens * len(accepts))[0]
    req = fixture.req(0, 0, length)
    attn.forward(rows[:length], fixture.batch([req], "prefill"))

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[1])
    device, dtype = fixture.device, fixture.dtype
    static = {
        "x": torch.zeros(tokens, config.hidden_size, device=device, dtype=dtype),
        "positions": torch.zeros(tokens, dtype=torch.int32, device=device),
        "out_loc": torch.zeros(tokens, dtype=torch.int32, device=device),
    }
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=3, extend_len=tokens
    )
    capture_batch = _verify_batch(fixture, dummy, static)
    fixture.backend.stage_verify(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    start = length
    for window, accept in enumerate(accepts):
        fixture.allocate(req.table_idx, start, start + tokens)
        req.cached_len, req.device_len, req.extend_len = start, start + tokens, tokens
        x = rows[length + tokens * window : length + tokens * (window + 1)]
        state = _kv_state(fixture)
        before = [t.clone() for t in state]
        eager = attn.forward(x, _verify_batch(fixture, req))
        expect = [t.clone() for t in state]
        for t, b in zip(state, before):
            t.copy_(b)

        batch = _verify_batch(fixture, req, static)
        static["x"].copy_(x)
        fixture.backend.stage_verify(batch)
        # replay must stage into the captured buffers, never reallocate them
        staged = fixture.backend._verify[tokens]["block_table"]
        assert batch.attn_metadata.block_table.data_ptr() == staged.data_ptr()
        graph.replay()
        assert torch.equal(captured_out, eager), f"verify output diverged at window {window}"
        bad = [i for i, (t, e) in enumerate(zip(state, expect)) if not torch.equal(t, e)]
        assert not bad, f"pool tensors {bad} diverged at window {window}"
        start += tokens if accept else 1


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))
