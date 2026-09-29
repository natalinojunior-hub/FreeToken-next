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


def test_qwen4_last_forward_keeps_attention_rows_and_slices_mlp_rows():
    from freetoken.engine.graph import mtp_forward_last
    from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer

    class HyperConnection:
        def mix(self, x):
            return x * 0.5, x * 0.25

        def combine(self, residual, output, inject):
            return residual + output + inject

    class Attention:
        def __init__(self):
            self.rows = []

        def forward(self, x, batch):
            self.rows.append(x.shape[0])
            return x * 0.125

    class MLP:
        def __init__(self):
            self.rows = []

        def forward(self, x):
            self.rows.append(x.shape[0])
            return x * 0.25 + 0.5

    layer = object.__new__(Qwen4ExpDecoderLayer)
    layer._layer_id = 4
    layer._is_linear = False
    layer.ple = None
    layer.self_attn = Attention()
    layer.attn_hyper_connection = HyperConnection()
    layer.mlp_hyper_connection = HyperConnection()
    layer.mlp = MLP()
    hidden = torch.arange(20, dtype=torch.float32).view(5, 4)
    full = layer.forward(hidden, None)
    assert layer.self_attn.rows == [5]
    assert layer.mlp.rows == [5]

    layer.self_attn.rows.clear()
    layer.mlp.rows.clear()
    fast = layer.forward_last(hidden, None)
    assert layer.self_attn.rows == [5]
    assert layer.mlp.rows == [1]
    assert torch.equal(fast, full[-1:])

    class GenericMTP:
        def forward(self, residual, next_ids, batch):
            return residual + next_ids[:, None]

    residual = torch.arange(20, dtype=torch.float32).view(5, 4)
    tokens = torch.arange(5, dtype=torch.float32)
    expected = GenericMTP().forward(residual, tokens, None)[-1:]
    assert torch.equal(mtp_forward_last(GenericMTP(), residual, tokens, None), expected)


def test_mtp_forward_last_prefers_specialized_method():
    from freetoken.engine.graph import mtp_forward_last

    class MTP:
        def forward(self, residual, next_ids, batch):
            raise AssertionError("full forward should not be called")

        def forward_last(self, residual, next_ids, batch):
            return residual[-1:]

    residual = torch.arange(10).view(5, 2)
    assert torch.equal(mtp_forward_last(MTP(), residual, torch.empty(0), None), residual[-1:])


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


@requires_cuda
def test_kv_only_store_matches_normal_qsa_state():
    """MTP KV-only priming must write the same KV/index state as QSA attention."""
    config = parsed_config()
    normal = Fixture(config, num_pages=16, max_running_req=1)
    attn = normal.layer(QSA_LAYER)
    length = 70  # close one compressed group and leave a nonempty pending ring
    x = _inputs(normal, [length], seed=23)[0]
    normal_batch = normal.batch([normal.req(0, 0, length)], "prefill")
    q, k, v, index, _ = attn._project(x, normal_batch)
    normal.backend.qsa_forward(
        q.view(-1, attn.num_q, attn.head_dim), k, v, index, QSA_LAYER, normal_batch
    )

    # Fixture installs its context globally; finish normal metadata/forward before primed
    # becomes the global context used by QSA's block-table snapshot.
    primed = Fixture(config, num_pages=16, max_running_req=1)
    prime_batch = primed.batch([primed.req(0, 0, length)], "prefill")
    primed.backend.store_qsa_kv(k, v, index, QSA_LAYER, prime_batch, stage_host=False)

    torch.cuda.synchronize(normal.device)
    # out_loc is a flattened token slot; MHA-backed QSA caches expose [pages, page_size, ...].
    torch.testing.assert_close(
        normal.pool.k_cache(QSA_LAYER).flatten(0, 1).index_select(0, normal_batch.out_loc.long()),
        primed.pool.k_cache(QSA_LAYER).flatten(0, 1).index_select(0, prime_batch.out_loc.long()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        normal.pool.v_cache(QSA_LAYER).flatten(0, 1).index_select(0, normal_batch.out_loc.long()),
        primed.pool.v_cache(QSA_LAYER).flatten(0, 1).index_select(0, prime_batch.out_loc.long()),
        rtol=0,
        atol=0,
    )
    slot = normal.backend._idx_slot[QSA_LAYER]
    main_rows = normal.pool.cmp_scratch_base
    assert torch.equal(
        normal.pool.cmp_k_cache(slot)[:main_rows],
        primed.pool.cmp_k_cache(slot)[:main_rows],
    )
    assert torch.equal(normal.pool.pending_ring(slot), primed.pool.pending_ring(slot))


@requires_cuda
def test_mtp_prime_fill_matches_full_head_across_chunks_and_mrope():
    """Real MTP head parity, including its prompt warmup and the following decode."""
    from freetoken.models.config import with_mtp_layer
    from freetoken.models.qwen4_exp.model import Qwen4ExpMTP

    from .common import hf_config

    hf = hf_config(mtp={"num_hidden_layers": 1, "hybrid": True, "layer_types": ["full_attention"]})
    hf.text_config.rope_parameters.update(
        {"mrope_interleaved": True, "mrope_section": [11, 11, 10]}
    )
    hf.vision_config = SimpleNamespace(
        hidden_size=128,
        depth=1,
        num_heads=2,
        intermediate_size=256,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        out_hidden_size=128,
        in_channels=3,
    )
    hf.image_token_id = 511
    from freetoken.models.qwen4_exp.config import parse_config

    config = with_mtp_layer(parse_config(hf), 4)
    assert config.model_is_mrope
    length = 137  # cross pages and compressed groups; leave pending index rows
    device = torch.device("cuda")
    seed = 912
    gen = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(
        length,
        config.hidden_size * config.qwen4_args.hc_count,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    ids = torch.randint(config.vocab_size - 1, (length,), device=device, generator=gen)
    next_residual = torch.randn(
        1,
        config.hidden_size * config.qwen4_args.hc_count,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    next_id = torch.randint(config.vocab_size - 1, (1,), device=device, generator=gen)
    temporal = torch.arange(length + 1, dtype=torch.int32)
    mrope = torch.stack((temporal, temporal // 4, temporal // 16))

    # Build the real head while a QSA fixture owns the global context. Its toy MoE is
    # irrelevant to KV writes; preserve the real MTP preparation, HC, and attention path.
    owner = Fixture(config, num_pages=16, max_running_req=1)
    embedding = torch.nn.Embedding(
        config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device(device), torch_dtype(torch.bfloat16):
        mtp = Qwen4ExpMTP(config, config.mtp_layer_id, embedding=embedding)
    for index, tensor in enumerate(mtp.state_dict().values()):
        if tensor.is_floating_point():
            tensor.normal_(
                0.0,
                0.05,
                generator=torch.Generator(device=tensor.device).manual_seed(seed + index + 1),
            )
        else:
            tensor.zero_()
    with torch.no_grad():
        embedding.weight.normal_(
            0.0,
            0.05,
            generator=torch.Generator(device=device).manual_seed(seed + 2),
        )
    layer = mtp.layers.op_list[0]
    assert layer.ple is None  # MTP is synthetic layer num_layers; PLE belongs to target layers.
    layer.mlp.forward = lambda x: torch.zeros_like(x)

    class DynamicReq(SimpleNamespace):
        @property
        def extend_len(self):
            return self.device_len - self.cached_len

        @extend_len.setter
        def extend_len(self, value):
            pass  # Fixture.step mirrors the production Req tuple update; the property is derived.

    def make_run(kind: str, chunk: int | None):
        fixture = Fixture(config, num_pages=16, max_running_req=1)
        from freetoken.kvcache.linear_state_pool import LinearStatePool

        fixture.ctx.linear_state_pool = LinearStatePool(
            config.linear_attention_group(),
            fixture.num_req_slots,
            fixture.dtype,
            fixture.device,
            tp_size=1,
            slot_states=config.slot_states,
        )
        for state in fixture.ctx.linear_state_pool.slot_states.values():
            state.fill_(1 if not state.is_floating_point() else 0.125)
        req = DynamicReq(**vars(fixture.req(0, 0, length)))
        req.mrope_positions_full = mrope[:, :length].clone()
        req.mrope_delta = 0
        if kind == "full":
            full_chunk = chunk or length
            for offset in range(0, length, full_chunk):
                stop = min(offset + full_chunk, length)
                req.cached_len, req.device_len = offset, stop
                batch = fixture.batch([req], "prefill")
                batch.mrope_positions = mrope[:, offset:stop].to(device)
                batch.get_attn_positions = lambda: batch.mrope_positions
                with fixture.ctx.forward_batch(batch):
                    mtp.forward(residual[offset:stop], ids[offset:stop], batch)
            req.cached_len, req.device_len = 0, length
        elif kind == "auto":
            scheduler = SimpleNamespace(
                device=device,
                _model_is_mrope=True,
                engine=SimpleNamespace(
                    model=SimpleNamespace(mtp=mtp),
                    attn_backend=fixture.backend,
                    config=SimpleNamespace(max_extend_tokens=chunk),
                    page_table=fixture.page_table,
                    ctx=fixture.ctx,
                ),
            )
            SchedulerSpecMixin._fill_mtp_kv(scheduler, req, 0, residual, ids)
        else:
            for offset in range(0, length, chunk):
                stop = min(offset + chunk, length)
                req.cached_len, req.device_len = offset, stop
                batch = fixture.batch([req], "prefill")
                batch.mrope_positions = mrope[:, offset:stop].to(device)
                batch.get_attn_positions = lambda: batch.mrope_positions
                with fixture.ctx.forward_batch(batch):
                    mtp.prime_kv(residual[offset:stop], ids[offset:stop], batch)
            req.cached_len, req.device_len = 0, length
        return fixture, req

    from freetoken.scheduler.spec import SchedulerSpecMixin
    import freetoken.core as core

    def activate(fixture):
        core._GLOBAL_CTX = fixture.ctx

    runs = [
        make_run("full", 8),
        make_run("auto", 8),
        make_run("full", 128),
        make_run("prime", 128),
    ]

    def assert_same_state(reference, candidate):
        ref, ref_req = reference
        got, got_req = candidate
        extent = ref_req.device_len
        layer_id = config.mtp_layer_id
        slot = ref.backend._idx_slot[layer_id]
        torch.testing.assert_close(
            ref.pool.k_cache(layer_id)
            .flatten(0, 1)
            .index_select(0, ref.page_table[ref_req.table_idx, :extent].long()),
            got.pool.k_cache(layer_id)
            .flatten(0, 1)
            .index_select(0, got.page_table[got_req.table_idx, :extent].long()),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ref.pool.v_cache(layer_id)
            .flatten(0, 1)
            .index_select(0, ref.page_table[ref_req.table_idx, :extent].long()),
            got.pool.v_cache(layer_id)
            .flatten(0, 1)
            .index_select(0, got.page_table[got_req.table_idx, :extent].long()),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ref.pool.rope_positions.index_select(
                0, ref.page_table[ref_req.table_idx, :extent].long()
            ),
            got.pool.rope_positions.index_select(
                0, got.page_table[got_req.table_idx, :extent].long()
            ),
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            ref.pool.cmp_k_cache(slot)[: ref.pool.cmp_scratch_base],
            got.pool.cmp_k_cache(slot)[: got.pool.cmp_scratch_base],
        )
        assert torch.equal(ref.pool.pending_ring(slot), got.pool.pending_ring(slot))
        for name, state in ref.ctx.linear_state_pool.slot_states.items():
            assert torch.equal(state, got.ctx.linear_state_pool.slot_states[name])

    for full, prime in ((0, 1), (2, 3)):
        assert_same_state(runs[full], runs[prime])

    # A subsequent decode consumes the primed cache; verify both emitted head state and writes.
    decode_outputs = []
    decode_writes = []
    for fixture, req in runs:
        writes = []
        original_store = fixture.backend.store_qsa_kv

        def capture_store(*args, **kwargs):
            writes.append((args[0].clone(), args[1].clone()))
            return original_store(*args, **kwargs)

        fixture.backend.store_qsa_kv = capture_store
        activate(fixture)
        fixture.step(req)
        batch = fixture.batch([req], "decode")
        batch.mrope_positions = mrope[:, length : length + 1].to(device)
        batch.get_attn_positions = lambda: batch.mrope_positions
        with fixture.ctx.forward_batch(batch):
            decode_outputs.append(mtp.forward(next_residual, next_id, batch))
        decode_writes.append(writes)
    for full, prime in ((0, 1), (2, 3)):
        torch.testing.assert_close(
            decode_writes[prime][0][0], decode_writes[full][0][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            decode_writes[prime][0][1], decode_writes[full][0][1], rtol=0, atol=0
        )
        torch.testing.assert_close(decode_outputs[prime], decode_outputs[full], rtol=0, atol=0)
        assert_same_state(runs[full], runs[prime])

    real_mlp = layer.mlp.forward
    mlp_rows = []

    def track_real_mlp(x):
        mlp_rows.append(x.shape[0])
        return real_mlp(x)

    layer.mlp.forward = track_real_mlp

    def draft_run(use_last):
        fixture = Fixture(config, num_pages=16, max_running_req=1)
        req = DynamicReq(**vars(fixture.req(0, 0, 5)))
        req.mrope_positions_full = mrope[:, :6].clone()
        req.mrope_delta = 0
        batch = fixture.batch([req], "prefill")
        batch.mrope_positions = mrope[:, :5].to(device)
        batch.get_attn_positions = lambda: batch.mrope_positions
        activate(fixture)
        with fixture.ctx.forward_batch(batch):
            result = (
                mtp.forward_last(residual[:5], ids[:5], batch)
                if use_last
                else mtp.forward(residual[:5], ids[:5], batch)[-1:]
            )
        return fixture, req, result

    full_run = draft_run(False)
    last_run = draft_run(True)
    torch.testing.assert_close(full_run[2], last_run[2], rtol=0, atol=0)
    assert mlp_rows[:2] == [5, 1]

    def assert_draft_state_equal(reference, candidate):
        ref, ref_req, _ = reference
        got, got_req, _ = candidate
        extent = ref_req.device_len
        layer_id = config.mtp_layer_id
        slot = ref.backend._idx_slot[layer_id]
        for cache_name in ("k_cache", "v_cache"):
            ref_cache = getattr(ref.pool, cache_name)(layer_id)
            got_cache = getattr(got.pool, cache_name)(layer_id)
            torch.testing.assert_close(
                ref_cache.flatten(0, 1).index_select(
                    0, ref.page_table[ref_req.table_idx, :extent].long()
                ),
                got_cache.flatten(0, 1).index_select(
                    0, got.page_table[got_req.table_idx, :extent].long()
                ),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            ref.pool.rope_positions.index_select(
                0, ref.page_table[ref_req.table_idx, :extent].long()
            ),
            got.pool.rope_positions.index_select(
                0, got.page_table[got_req.table_idx, :extent].long()
            ),
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            ref.pool.cmp_k_cache(slot)[: ref.pool.cmp_scratch_base],
            got.pool.cmp_k_cache(slot)[: got.pool.cmp_scratch_base],
        )
        assert torch.equal(ref.pool.pending_ring(slot), got.pool.pending_ring(slot))

    assert_draft_state_equal(full_run, last_run)
    next_outputs = []
    for fixture, req, previous in (full_run, last_run):
        fixture.step(req)
        batch = fixture.batch([req], "decode")
        batch.mrope_positions = mrope[:, 5:6].to(device)
        batch.get_attn_positions = lambda: batch.mrope_positions
        activate(fixture)
        with fixture.ctx.forward_batch(batch):
            next_outputs.append(mtp.forward(previous, next_id, batch))
    torch.testing.assert_close(next_outputs[0], next_outputs[1], rtol=0, atol=0)
    assert mlp_rows == [5, 1, 1, 1]
    assert_draft_state_equal(
        (full_run[0], full_run[1], next_outputs[0]),
        (last_run[0], last_run[1], next_outputs[1]),
    )


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
def test_draft_window_graph_replay_matches_eager():
    """A captured 1-token draft window (stage_verify with ``tokens=1`` -- the MTP draft
    step's own window, never a size any spec-verify graph captures) reproduces the eager
    forward bitwise: repeated single-token replays crossing page boundaries, two requests
    in different KV slots taking turns through the one captured graph, and a step replayed
    twice in a row with no new input. Regression for the draft-graph illegal-memory-access
    (qsa_sparse.py qsa_forward, surfaced via ``indices[indices >= 0]``)."""
    # Two QSA layers, like the real model + its separate MTP layer: layer 3 stands in for the
    # main model (prefilled, decode/verify-captured -- its slab slot is warm), layer 7 stands
    # in for the MTP draft layer and is *never forwarded* before its own warmup+capture step,
    # exactly as graph.py._capture_draft is model.mtp's first-ever invocation in the process.
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    attn_main = fixture.layer(3)
    attn = fixture.layer(7, seed=7)  # the draft layer capture/replay below all use this one
    length, tokens, steps = 60, 1, 150  # length near the page_size=64 boundary
    rows = _inputs(fixture, [length, length], extra=steps + 5)
    req_a = fixture.req(0, 0, length)
    req_b = fixture.req(1, 0, length)
    attn_main.forward(rows[0][:length], fixture.batch([req_a], "prefill"))
    attn_main.forward(rows[1][:length], fixture.batch([req_b], "prefill"))

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[1])
    device, dtype = fixture.device, fixture.dtype
    static = {
        "x": torch.zeros(tokens, config.hidden_size, device=device, dtype=dtype),
        "positions": torch.zeros(tokens, dtype=torch.int32, device=device),
        "out_loc": torch.zeros(tokens, dtype=torch.int32, device=device),
    }
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=tokens
    )

    # Real capture order (graph.py GraphRunner): decode graph first (owns the shared pool),
    # then spec-verify window(s), then the draft graph LAST in that same pool -- the ordering
    # the C3b audit flags as the one place a fresh torch.empty_like(q) allocated *inside* a
    # capture could get handed an address a still-live earlier-graph tensor also claims.
    decode_static = {
        "x": torch.zeros(1, config.hidden_size, device=device, dtype=dtype),
        "positions": torch.zeros(1, dtype=torch.int32, device=device),
        "out_loc": torch.zeros(1, dtype=torch.int32, device=device),
    }
    decode_dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    decode_batch = SimpleNamespace(
        padded_reqs=[decode_dummy],
        reqs=[decode_dummy],
        phase="decode",
        size=1,
        padded_size=1,
        is_prefill=False,
        is_decode=True,
        positions=decode_static["positions"],
        get_attn_positions=lambda: decode_static["positions"],
        out_loc=decode_static["out_loc"],
        attn_metadata=None,
        active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(decode_batch)
    attn_main.forward(decode_static["x"], decode_batch)  # warmup, same metadata object as capture
    torch.cuda.synchronize()
    decode_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode_graph):
        attn_main.forward(decode_static["x"], decode_batch)
    torch.cuda.synchronize()
    pool = decode_graph.pool()

    verify2_static = {
        "x": torch.zeros(2, config.hidden_size, device=device, dtype=dtype),
        "positions": torch.zeros(2, dtype=torch.int32, device=device),
        "out_loc": torch.zeros(2, dtype=torch.int32, device=device),
    }
    verify2_dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=3, extend_len=2
    )
    verify2_batch = _verify_batch(fixture, verify2_dummy, verify2_static)
    fixture.backend.stage_verify(verify2_batch)
    attn_main.forward(verify2_static["x"], verify2_batch)  # warmup
    torch.cuda.synchronize()
    verify2_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(verify2_graph, pool=pool):
        attn_main.forward(verify2_static["x"], verify2_batch)
    torch.cuda.synchronize()

    capture_batch = _verify_batch(fixture, dummy, static)
    assert tokens not in fixture.backend._verify, "tokens=1 must not alias a verify window"
    fixture.backend.stage_verify(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    reqs = [req_a, req_b]
    for step in range(steps):
        req, slot = reqs[step % 2], step % 2
        fixture.step(req)
        x = rows[slot][req.cached_len : req.device_len]

        state = _kv_state(fixture)
        before = [t.clone() for t in state]
        eager = attn.forward(x, _verify_batch(fixture, req))
        expect = [t.clone() for t in state]
        for t, b in zip(state, before):
            t.copy_(b)

        batch = _verify_batch(fixture, req, static)
        static["x"].copy_(x)
        fixture.backend.stage_verify(batch)
        staged = fixture.backend._verify[tokens]["block_table"]
        assert batch.attn_metadata.block_table.data_ptr() == staged.data_ptr()
        graph.replay()
        assert torch.equal(captured_out, eager), f"draft output diverged at step {step}"
        bad = [i for i, (t, e) in enumerate(zip(state, expect)) if not torch.equal(t, e)]
        assert not bad, f"pool tensors {bad} diverged at step {step} (single replay)"
        # Replay the same step again with no new input: must be idempotent (same reqs, same
        # addressing, same pool state) -- a stale baked pointer/size or a workspace buffer the
        # capture didn't restage would show up as a second-replay-only divergence.
        graph.replay()
        bad2 = [i for i, (t, e) in enumerate(zip(state, expect)) if not torch.equal(t, e)]
        assert not bad2, f"pool tensors {bad2} diverged at step {step} (repeated replay)"


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
