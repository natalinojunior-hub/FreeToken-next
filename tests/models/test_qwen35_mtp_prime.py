"""Draft prompt priming preserves KV and the following decode, without expert work."""

from types import SimpleNamespace

import torch
import pytest

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer, Qwen3_5MTP
from freetoken.scheduler.spec import SchedulerSpecMixin


class _Norm:
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    def forward_add_residual(self, x, residual):
        residual = x + residual
        return self.forward(residual), residual


@pytest.mark.parametrize("prime_chunk", [8, 12, 32])
def test_prime_matches_full_kv_and_next_decode_without_attention_or_mlp(monkeypatch, prime_chunk):
    import freetoken.core as core

    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    gen = torch.Generator().manual_seed(37)
    linear = lambda rows, cols: SimpleNamespace(
        forward=lambda x, weight=torch.randn(rows, cols, generator=gen): x @ weight.T
    )
    norm = _Norm()
    attn = Qwen3_5Attention.__new__(Qwen3_5Attention)
    attn.layer_id = 4
    attn.num_q = attn.num_kv = 1
    attn.head_dim = attn.qo_attn_dim = attn.kv_attn_dim = 4
    attn._qkv_split = [8, 4, 4]
    attn.qkv_proj, attn.o_proj = linear(16, 4), linear(4, 4)
    attn.q_norm = attn.k_norm = norm
    attn.rotary = SimpleNamespace(forward=lambda positions, q, k: (q, k))
    calls = {"attention": 0, "mlp": 0}
    kv = {}

    def store(k, v, out_loc, layer_id):
        assert layer_id == attn.layer_id
        for row, pos in enumerate(out_loc.tolist()):
            kv[pos] = (k[row].clone(), v[row].clone())

    def attend(q, k, v, layer_id, batch):
        calls["attention"] += 1
        store(k, v, batch.out_loc, layer_id)
        out = []
        for row, pos in enumerate(batch.positions.tolist()):
            keys, values = zip(*(kv[i] for i in range(pos + 1)))
            scores = q[row, 0] @ torch.stack(keys).T / 2
            out.append(torch.softmax(scores, dim=-1) @ torch.stack(values))
        return torch.stack(out).unsqueeze(1)

    def mlp(x):
        calls["mlp"] += 1
        return torch.sin(x)

    ctx.kv_cache = SimpleNamespace(store_kv=store)
    ctx.attn_backend = SimpleNamespace(forward=attend)
    layer = Qwen3_5DecoderLayer.__new__(Qwen3_5DecoderLayer)
    layer._layer_id, layer._is_linear = 4, False
    layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm = attn, norm, norm
    layer.mlp = SimpleNamespace(forward=mlp)
    head = Qwen3_5MTP.__new__(Qwen3_5MTP)
    embedding = torch.randn(32, 4, generator=gen)
    head._embed_ref = SimpleNamespace(forward=lambda ids: embedding[ids])
    head.enorm = head.hnorm = head.shared_head_norm = norm
    head.eh_proj = linear(4, 8)
    head.layers = SimpleNamespace(op_list=[layer])
    residual = torch.randn(18, 4, generator=gen)
    tokens = torch.arange(18)

    def run(prime, chunk=8):
        kv.clear()
        calls.update(attention=0, mlp=0)
        for start in range(0, 17, chunk):
            stop = min(start + chunk, 17)
            batch = Batch(reqs=[], phase="prefill")
            batch.positions = batch.out_loc = torch.arange(start, stop, dtype=torch.int32)
            batch.mtp_fill = True
            with ctx.forward_batch(batch):
                (head.prime_kv if prime else head.forward)(
                    residual[start:stop], tokens[start:stop], batch
                )
        state = {pos: tuple(t.clone() for t in pair) for pos, pair in kv.items()}
        prompt_calls = dict(calls)
        batch = Batch(reqs=[], phase="decode")
        batch.positions = batch.out_loc = torch.tensor([17], dtype=torch.int32)
        with ctx.forward_batch(batch):
            out = head.forward(residual[-1:], tokens[-1:], batch)
        return state, out, prompt_calls

    reference, expected, full_calls = run(False)
    primed, actual, prime_calls = run(True, prime_chunk)
    assert full_calls == {"attention": 3, "mlp": 3}
    assert prime_calls == {"attention": 0, "mlp": 0}
    assert reference.keys() == primed.keys()
    for pos in reference:
        for a, b in zip(reference[pos], primed[pos]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    if prime_chunk == 8:
        assert all(
            torch.equal(a, b) for pos in reference for a, b in zip(reference[pos], primed[pos])
        )
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("mode", ["full_prime", "qsa_prime", "unknown_prime", "fallback"])
@pytest.mark.parametrize("cap", [None, 12, 32])
def test_scheduler_bounds_full_prime_by_resolved_prefill_cap(mode, cap):
    calls = []
    ctx = Context(page_size=1)
    scheduler = SchedulerSpecMixin()
    scheduler.device = torch.device("cpu")
    scheduler._model_is_mrope = False
    if cap is not None:
        scheduler.config = SimpleNamespace(max_extend_tokens=cap)
    req = Req(
        input_ids=torch.arange(19),
        table_idx=0,
        cached_len=18,
        output_len=4,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(cached_len=18),
    )
    lengths = req.cached_len, req.device_len

    def fail(*args):
        raise AssertionError("full draft forward must be skipped")

    def prime(hidden, tokens, batch):
        assert batch.mtp_fill
        calls.append((tokens.numel(), batch.positions.clone(), batch.out_loc.clone()))

    scheduler.engine = SimpleNamespace(
        model=SimpleNamespace(
            mtp=SimpleNamespace(
                forward=fail,
                prime_kv=prime,
                prime_kv_without_experts=mode in ("full_prime", "qsa_prime"),
            )
        ),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        kv_cache=SimpleNamespace(store_kv=lambda *args: None),
        ctx=ctx,
        page_table=torch.arange(32, dtype=torch.int32).reshape(1, -1),
    )
    if mode == "qsa_prime":
        scheduler.engine.attn_backend.store_qsa_kv = lambda *_args: None
    elif mode == "unknown_prime":
        del scheduler.engine.model.mtp.prime_kv_without_experts
    elif mode == "fallback":
        scheduler.engine.model.mtp = SimpleNamespace(forward=prime)
    scheduler._fill_mtp_kv(req, 1, torch.randn(17, 4), torch.arange(17))
    chunk = cap if mode == "full_prime" and cap is not None else 8
    assert [call[0] for call in calls] == [min(chunk, 17 - i) for i in range(0, 17, chunk)]
    assert torch.equal(torch.cat([call[1] for call in calls]), torch.arange(1, 18))
    assert all(torch.equal(pos, loc) for _, pos, loc in calls)
    assert (req.cached_len, req.device_len) == lengths
