"""GraphRunner's spec-verify graph: gating, per-replay restaging, residual rebinding, the eager
fallback and reset.

The real-model eager/graph parity (logits, sampled tokens, MTP residual and every GDN/PLE/QSA/KV
tensor, bitwise, with cold and warm expert caches) runs in-server under
FREETOKEN_VERIFY_GRAPH_CHECK=1; the attention and GDN halves have their own captured-vs-eager
tests (tests/models/qwen4_exp/test_qsa_backend.py, test_gdn.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.core import Batch, Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import VERIFY_GRAPH_ENV, GraphRunner, verify_graph_tokens

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB, SLOTS, WIDTH, TOKENS = 8, 4, 64, 2
SIZES = (TOKENS, TOKENS + 1, TOKENS + 2)  # k+1 rows plus up to two deferred-replay tokens
DEV = torch.device("cuda")


def test_verify_graph_tokens(monkeypatch):
    assert verify_graph_tokens(1) == SIZES
    assert verify_graph_tokens(0) == ()
    assert verify_graph_tokens(3) == (2, 3, 4)  # every adaptive k' + 1 rows; no defer extras
    monkeypatch.setenv(VERIFY_GRAPH_ENV, "0")
    assert verify_graph_tokens(1) == ()


def test_verify_gate():
    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs, runner.verify_graphs = 1, {TOKENS: SimpleNamespace(tokens=TOKENS)}
    ok = dict(
        is_decode=False,
        is_prefill=True,
        spec_logits_indices=torch.arange(TOKENS),
        size=1,
        input_ids=torch.zeros(TOKENS),
        reqs=[SimpleNamespace(cached_len=5)],
        mm_embeds=None,
    )
    assert runner.can_use_cuda_graph(SimpleNamespace(**ok))
    for change in (
        dict(spec_logits_indices=None),
        dict(size=2),
        dict(input_ids=torch.zeros(TOKENS + 1)),
        dict(reqs=[SimpleNamespace(cached_len=0)]),
        dict(mm_embeds=torch.zeros(1)),
    ):
        assert not runner.can_use_cuda_graph(SimpleNamespace(**{**ok, **change})), change
    # pad_batch asks before the verify window has input_ids or spec_logits_indices
    assert not runner.can_use_cuda_graph(
        SimpleNamespace(is_decode=False, is_prefill=True, spec_logits_indices=None)
    )
    assert runner.can_use_cuda_graph(SimpleNamespace(is_decode=True, size=1))
    runner.verify_graphs = {}
    assert not runner.can_use_cuda_graph(SimpleNamespace(**ok))


class _FakeModel:
    """Reads every input the verify graph restages (tokens, positions, out_loc, GDN slot, staged
    attention metadata), updates per-slot state in place and publishes an MTP residual."""

    def __init__(self, events: list) -> None:
        self.model = SimpleNamespace(_last_residual=None)
        self.state = torch.zeros(SLOTS, VOCAB, device=DEV)
        self.weight = torch.arange(1, VOCAB + 1, dtype=torch.float32, device=DEV)
        self.events = events

    def forward(self) -> torch.Tensor:
        self.events.append("forward")
        batch = get_global_ctx().batch
        slot = batch.fla_metadata.cache_indices.long()
        feat = (batch.input_ids * 3 + batch.positions * 5 + batch.out_loc * 7).float()
        if batch.is_prefill:
            feat = feat + batch.attn_metadata.kv.float()
        new = self.state.index_select(0, slot) * 0.5 + feat.sum()
        self.state.index_copy_(0, slot, new)
        hidden = feat.unsqueeze(1) * self.weight + new
        self.model._last_residual = hidden * 2
        return hidden[batch.spec_logits_indices] if batch.is_prefill else hidden


class _FakeBackend:
    def __init__(self, verify: bool = True) -> None:
        self.verify, self.static = verify, None

    def init_capture_graph(self, max_seq_len, bs_list) -> None: ...

    def prepare_for_capture(self, batch) -> None:
        batch.attn_metadata = SimpleNamespace()

    def prepare_for_replay(self, batch) -> None: ...

    def prepare_metadata(self, batch) -> None:
        kv = torch.tensor([r.device_len for r in batch.reqs], dtype=torch.int32, device=DEV)
        batch.attn_metadata = SimpleNamespace(kv=kv)

    def stage_verify(self, batch) -> None:
        if not self.verify:
            raise NotImplementedError("no verify graph")
        md = batch.attn_metadata
        if self.static is None:
            self.static = torch.zeros_like(md.kv)
        self.static.copy_(md.kv)
        md.kv = self.static


class _FakeCache:
    def __init__(self, events: list) -> None:
        self.events = events

    def reset(self) -> None:
        self.events.append("reset")


def _req(table_idx: int, cached_len: int, device_len: int) -> Req:
    return Req(
        input_ids=torch.zeros(device_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=table_idx,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
    )


def _runner(monkeypatch, verify: bool = True):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))
    ctx = get_global_ctx()
    ctx.page_table = torch.arange(SLOTS * WIDTH, dtype=torch.int32, device=DEV).view(SLOTS, WIDTH)
    events: list = []
    model = _FakeModel(events)
    stream = torch.cuda.Stream()  # capture needs a non-default stream
    with torch.cuda.stream(stream):
        runner = GraphRunner(
            stream=stream,
            device=DEV,
            model=model,
            attn_backend=_FakeBackend(verify),
            cuda_graph_bs=[1],
            cuda_graph_max_bs=1,
            free_memory=1 << 30,
            max_seq_len=WIDTH,
            vocab_size=VOCAB,
            dummy_req=_req(SLOTS - 1, 0, 1),
            moe_offload_cache=_FakeCache(events),
            verify_tokens=SIZES,
        )
    torch.cuda.synchronize()
    return ctx, runner, model, events


def _verify_batch(ctx, runner, req: Req, tokens: list[int]) -> Batch:
    from freetoken.attention.linear import FLAMetadata

    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    batch.input_ids = torch.tensor(tokens, dtype=torch.int32, device=DEV)
    batch.positions = torch.arange(req.cached_len, req.device_len, dtype=torch.int32, device=DEV)
    batch.out_loc = ctx.page_table[req.table_idx, req.cached_len : req.device_len]
    batch.spec_logits_indices = torch.arange(len(tokens), device=DEV)
    batch.fla_metadata = FLAMetadata(
        cu_seqlens=torch.tensor([0, len(tokens)], dtype=torch.int32, device=DEV),
        cache_indices=torch.tensor([req.table_idx], dtype=torch.int32, device=DEV),
    )
    runner.attn_backend.prepare_metadata(batch)
    return batch


def _decode_replay(runner, req: Req, token: int) -> None:
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.input_ids = torch.tensor([token], dtype=torch.int32, device=DEV)
    batch.positions = torch.tensor([req.device_len], dtype=torch.int32, device=DEV)
    batch.out_loc = batch.positions.clone()
    batch.linear_table_idx = torch.tensor([req.table_idx], dtype=torch.int32, device=DEV)
    runner.replay(batch)


@requires_cuda
def test_verify_replay_matches_eager(monkeypatch):
    """Changing tokens, positions and slots, accepted and rejected windows, a decode replay and
    an eager forward interleaved: each verify replay equals the eager forward (logits, slot state,
    residual) and leaves the model's residual bound to the captured output."""
    ctx, runner, model, events = _runner(monkeypatch)
    assert sorted(runner.verify_graphs) == list(SIZES) and runner.graph_map
    assert events[-2:] == ["forward", "reset"]  # expert cache reset after the verify capture

    windows = [
        (0, 10, [5, 9]),
        (1, 20, [7, 7, 2]),
        (0, 12, [3, 1]),
        (0, 13, [2, 8, 6]),
        (2, 30, [1, 4]),
        (2, 31, [4, 4, 1]),
        (1, 40, [6, 2, 5, 3]),
    ]
    for step, (slot, start, tokens) in enumerate(windows):
        verify = runner.verify_graphs[len(tokens)]
        req = _req(slot, start, start + len(tokens))
        state0 = model.state.clone()
        eager_batch = _verify_batch(ctx, runner, req, tokens)
        with ctx.forward_batch(eager_batch):
            eager = model.forward().clone()
        expect_state, expect_res = model.state.clone(), model.model._last_residual.clone()
        model.state.copy_(state0)

        batch = _verify_batch(ctx, runner, req, tokens)
        assert runner.can_use_cuda_graph(batch)
        got = runner.replay(batch)
        assert got.data_ptr() == verify.logits.data_ptr()  # static output, no reallocation
        assert torch.equal(got, eager), f"logits at window {step}"
        assert torch.equal(model.state, expect_state), f"slot state at window {step}"
        assert model.model._last_residual is verify.residual[1]
        assert torch.equal(model.model._last_residual, expect_res), f"residual at window {step}"

        _decode_replay(runner, req, token=tokens[-1])
        model.model._last_residual = None  # an eager forward rebinding it in between


@requires_cuda
def test_verify_capture_failure_falls_back_to_eager(monkeypatch):
    ctx, runner, _, events = _runner(monkeypatch, verify=False)
    assert not runner.verify_graphs and runner.graph_map  # decode graphs unaffected
    assert events[-1] == "reset"
    batch = _verify_batch(ctx, runner, _req(0, 10, 10 + TOKENS), [1, 2])
    assert not runner.can_use_cuda_graph(batch)


@requires_cuda
def test_verify_graph_env_off_and_destroy(monkeypatch):
    monkeypatch.setenv(VERIFY_GRAPH_ENV, "0")
    assert verify_graph_tokens(1) == ()
    monkeypatch.delenv(VERIFY_GRAPH_ENV)
    ctx, runner, _, _ = _runner(monkeypatch)
    batch = _verify_batch(ctx, runner, _req(0, 10, 10 + TOKENS), [1, 2])
    assert runner.can_use_cuda_graph(batch)
    runner.destroy_cuda_graphs()
    assert not runner.verify_graphs and not runner.can_use_cuda_graph(batch)


class _FakeMTP:
    """Draft layer reading the staged residual, token, positions and attention metadata."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def forward(self, residual, next_ids, batch):
        feat = (next_ids * 3 + batch.positions * 5 + batch.out_loc * 7).float()
        feat = feat + batch.attn_metadata.kv.float()
        return residual * 0.5 + feat.unsqueeze(1)

    def to_head(self, residual):
        return residual


class _FakeHead:
    def forward(self, x):
        return x * torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)


@requires_cuda
def test_draft_replay_matches_eager(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_DRAFT_GRAPH", "1")
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))
    ctx = get_global_ctx()
    ctx.page_table = torch.arange(SLOTS * WIDTH, dtype=torch.int32, device=DEV).view(SLOTS, WIDTH)
    events: list = []
    model = _FakeModel(events)
    model.mtp, model.lm_head = _FakeMTP(model), _FakeHead()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner = GraphRunner(
            stream=stream,
            device=DEV,
            model=model,
            attn_backend=_FakeBackend(),
            cuda_graph_bs=[1],
            cuda_graph_max_bs=1,
            free_memory=1 << 30,
            max_seq_len=WIDTH,
            vocab_size=VOCAB,
            dummy_req=_req(SLOTS - 1, 0, 1),
            moe_offload_cache=_FakeCache(events),
            verify_tokens=SIZES,
        )
    torch.cuda.synchronize()
    assert runner.draft is not None
    residual = torch.randn(1, VOCAB, device=DEV).to(model.model._last_residual.dtype)
    for slot, pos, token in [(0, 10, 3), (1, 20, 5), (0, 11, 7)]:
        req = _req(slot, pos, pos + 1)
        batch = _verify_batch(ctx, runner, req, [token])
        tok = torch.tensor([token], dtype=torch.int32, device=DEV)
        with ctx.forward_batch(batch):
            r = model.mtp.forward(residual, tok, batch)
            logits = model.lm_head.forward(r)
        got = runner.replay_draft(batch, residual, tok)
        assert torch.equal(got[0], r) and torch.equal(got[1], logits)
        assert torch.equal(got[2], torch.argmax(logits, dim=-1))
        residual = got[0].clone()
    runner.destroy_cuda_graphs()
    assert runner.draft is None
