"""qwen4_exp GatedDeltaNet op vs the pure-torch HF reference math.

The oracle is ``models/qwen4_exp/gdn_reference.py``, whose two delta rules and forward are
transcribed from the ``modeling_qwen4_exp.py`` snapshot, so no transformers build carrying
qwen4_exp is needed here. Covered: prefill at 128 and 1000 tokens, a decode step continuing
from the prefill state, ragged bs=3, both GQA head ratios, and the sigmoid output gate.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
from freetoken.models.qwen4_exp.gdn_reference import Qwen4ExpGatedDeltaNetReference
from freetoken.utils import torch_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HIDDEN, HEAD_DIM, CONV_K, EPS = 256, 128, 4, 1e-6
RTOL = ATOL = 2e-2
# (num_k_heads, num_v_heads) per value:key head ratio; 3:1 is the Qwen3.8-Flash-Next shape.
HEADS = {2: (8, 16), 3: (16, 48)}


def _bf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(DEV, torch.bfloat16)


def _state_dict(ref) -> dict[str, torch.Tensor]:
    """HF's four in_proj matrices fused into the op's single qkv|z|b|a GEMM. A_log / dt_bias
    stay fp32, as the weight loader keeps them."""
    return {
        "in_proj.weight": _bf(
            torch.cat(
                [
                    ref.in_proj_qkv.weight,
                    ref.in_proj_z.weight,
                    ref.in_proj_b.weight,
                    ref.in_proj_a.weight,
                ],
                dim=0,
            )
        ),
        "conv1d.weight": _bf(ref.conv1d.weight),
        "dt_bias": ref.dt_bias.detach().to(DEV, torch.float32),
        "A_log": ref.A_log.detach().to(DEV, torch.float32),
        "norm.weight": _bf(ref.norm.weight),
        "out_proj.weight": _bf(ref.out_proj.weight),
    }


def _make_layer(ratio: int, output_gate: str = "sigmoid", seed: int = 0):
    """fp32 reference + bf16 kernel op over one set of weights. The op is built on meta under
    the serving dtype, the way the engine builds a model, so load_state_dict's dtype check bites."""
    num_k, num_v = HEADS[ratio]
    torch.manual_seed(seed)
    ref = (
        Qwen4ExpGatedDeltaNetReference(
            hidden_size=HIDDEN,
            num_k_heads=num_k,
            num_v_heads=num_v,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            conv_kernel_size=CONV_K,
            rms_norm_eps=EPS,
            output_gate=output_gate,
        )
        .to(DEV)
        .float()
        .eval()
    )
    with torch.no_grad():
        # HF inits A_log = log(U(0.01, 16)); a zero dt_bias or a unit gate norm would hide sign errors.
        ref.A_log.uniform_(0.01, 16.0).log_()
        ref.dt_bias.uniform_(-1.0, 1.0)
        ref.norm.weight.normal_(1.0, 0.1)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = Qwen4ExpGatedDeltaNet(
            hidden_size=HIDDEN,
            num_k_heads=num_k,
            num_v_heads=num_v,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            conv_kernel_size=CONV_K,
            rms_norm_eps=EPS,
            layer_id=0,
            output_gate=output_gate,
        )
    op.load_state_dict(_state_dict(ref))
    return op, ref


def _ctx(ratio: int, num_slots: int = 8, spec_steps: int = 0) -> Context:
    import freetoken.core as core
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    num_k, num_v = HEADS[ratio]
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=num_k,
        num_value_heads=num_v,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_dim=CONV_K,
        output_gate="sigmoid",
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    ctx.linear_state_pool = LinearStatePool(
        group, num_slots, torch.bfloat16, DEV, tp_size=1, spec_steps=spec_steps
    )
    core.set_global_ctx(ctx)
    return ctx


def _prefill(op, ctx: Context, lengths: list[int], seed: int):
    """One ragged prefill batch, one state slot per request. Returns the per-request hidden
    states, the reqs (for a follow-up decode) and the packed output."""
    torch.manual_seed(seed)
    hidden = [torch.randn(n, HIDDEN, device=DEV, dtype=torch.bfloat16) for n in lengths]
    reqs = [
        Req(
            input_ids=torch.zeros(n, dtype=torch.int32),
            table_idx=i + 1,
            cached_len=0,
            output_len=1,
            uid=i,
            sampling_params=SamplingParams(),
            cache_handle=None,
        )
        for i, n in enumerate(lengths)
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    with ctx.forward_batch(batch):
        out = op.forward(torch.cat(hidden, dim=0))
    return hidden, reqs, out


def _decode(op, ctx: Context, reqs, hidden: torch.Tensor) -> torch.Tensor:
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.linear_table_idx = torch.tensor(
        [r.table_idx for r in reqs], dtype=torch.int32, device=DEV
    )
    with ctx.forward_batch(batch):
        return op.forward(hidden)


@torch.no_grad()
def _ref_out(ref, hidden: torch.Tensor, use_chunk_rule: bool = False) -> torch.Tensor:
    return ref(hidden.float().unsqueeze(0), use_chunk_rule=use_chunk_rule)[0]


@pytest.mark.parametrize("length", (1000,))
@pytest.mark.parametrize("ratio", (2, 3))
def test_prefill_matches_reference(ratio, length):
    op, ref = _make_layer(ratio, seed=ratio)
    hidden, _, out = _prefill(op, _ctx(ratio), [length], seed=11)
    torch.testing.assert_close(out.float(), _ref_out(ref, hidden[0]), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("ratio", (2, 3))
def test_ragged_prefill_then_decode(ratio):
    """bs=3 ragged prefill, then one decode step per request off the carried conv + recurrent
    state. The decode oracle is the whole (prefill + 1) sequence in one reference pass, so a
    state that did not survive the prefill shows up immediately."""
    op, ref = _make_layer(ratio, seed=ratio)
    ctx = _ctx(ratio)
    lengths = [128, 1000, 37]
    hidden, reqs, out = _prefill(op, ctx, lengths, seed=13)

    off = 0
    for h, n in zip(hidden, lengths):
        torch.testing.assert_close(
            out[off : off + n].float(), _ref_out(ref, h), rtol=RTOL, atol=ATOL
        )
        off += n

    nxt = torch.randn(len(lengths), HIDDEN, device=DEV, dtype=torch.bfloat16)
    dec = _decode(op, ctx, reqs, nxt)
    for i, h in enumerate(hidden):
        full = _ref_out(ref, torch.cat([h, nxt[i : i + 1]], dim=0))
        torch.testing.assert_close(dec[i].float(), full[-1], rtol=RTOL, atol=ATOL)


def test_chunk_and_recurrent_rules_agree():
    """The chunked form (what the fla prefill kernel implements) against the sequential
    definition, both fp32: the chunk oracle is only worth anything if it reproduces the
    recurrence to fp32 precision."""
    _, ref = _make_layer(3, seed=1)
    torch.manual_seed(17)
    hidden = torch.randn(1000, HIDDEN, device=DEV, dtype=torch.bfloat16)
    torch.testing.assert_close(
        _ref_out(ref, hidden, use_chunk_rule=True), _ref_out(ref, hidden), rtol=1e-4, atol=1e-4
    )


def test_output_gate_comes_from_the_config():
    """The gate activation is the group config's string, not a hardcoded silu. Both gates track
    their own reference, and the two are far apart -- so a stuck activation cannot pass."""
    op_silu, ref_silu = _make_layer(3, output_gate="silu", seed=2)
    hidden, _, out_silu = _prefill(op_silu, _ctx(3), [128], seed=19)
    torch.testing.assert_close(
        out_silu.float(), _ref_out(ref_silu, hidden[0]), rtol=RTOL, atol=ATOL
    )

    op_sig, ref_sig = _make_layer(3, output_gate="sigmoid", seed=2)
    _, _, out_sig = _prefill(op_sig, _ctx(3), [128], seed=19)
    torch.testing.assert_close(out_sig.float(), _ref_out(ref_sig, hidden[0]), rtol=RTOL, atol=ATOL)

    assert (out_sig.float() - out_silu.float()).abs().max().item() > 10 * ATOL


def test_decode_prefill_gdn_kernel_inequivalence():
    """Bug A documentation: gdn_decode_fla vs gdn_prefill_chunk_fla.
    Even at T=1 with identical initial states, the fused decode kernel and the chunked
    prefill kernel produce mathematically close but non-bit-identical recurrent states
    and outputs (diff ~1e-3), which can flip greedy argmax decisions."""
    ratio = 3
    op, _ = _make_layer(ratio, seed=42)
    ctx = _ctx(ratio)
    pool = ctx.linear_state_pool
    slot = 0
    N = 64
    torch.manual_seed(1234)
    tokens = torch.randn(N, HIDDEN, device=DEV, dtype=torch.bfloat16)

    # 1. Decode T=1
    pool.recurrent_states[0, slot].zero_()
    pool.conv_states[0, slot].zero_()
    req_dec = Req(
        input_ids=torch.zeros(N + 1, dtype=torch.int32),
        table_idx=slot,
        cached_len=0,
        output_len=1,
        uid=10,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    dec_outs = []
    for i in range(N):
        req_dec.cached_len = i
        req_dec.device_len = i + 1
        batch = Batch(reqs=[req_dec], phase="decode")
        batch.padded_reqs = [req_dec]
        batch.linear_table_idx = torch.tensor([slot], dtype=torch.int32, device=DEV)
        with ctx.forward_batch(batch):
            dec_outs.append(op.forward(tokens[i : i + 1]))
    dec_outs = torch.cat(dec_outs, dim=0)
    dec_rec = pool.recurrent_states[0, slot].clone()

    # 2. Prefill T=1
    pool.recurrent_states[0, slot].zero_()
    pool.conv_states[0, slot].zero_()
    req_pref = Req(
        input_ids=torch.zeros(N + 1, dtype=torch.int32),
        table_idx=slot,
        cached_len=0,
        output_len=1,
        uid=20,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    pref_outs = []
    for i in range(N):
        req_pref.cached_len = i
        req_pref.device_len = i + 1
        batch = Batch(reqs=[req_pref], phase="prefill")
        batch.padded_reqs = [req_pref]
        with ctx.forward_batch(batch):
            pref_outs.append(op.forward(tokens[i : i + 1]))
    pref_outs = torch.cat(pref_outs, dim=0)
    pref_rec = pool.recurrent_states[0, slot].clone()

    # Mathematically close
    torch.testing.assert_close(dec_outs.float(), pref_outs.float(), rtol=1e-2, atol=1e-2)
    # But not bit-identical (Bug A)
    diff_out = (dec_outs.float() - pref_outs.float()).abs().max().item()
    diff_rec = (dec_rec.float() - pref_rec.float()).abs().max().item()
    assert diff_out > 1e-4, f"Expected Bug A output inequivalence, got diff {diff_out}"
    assert diff_rec > 1e-4, f"Expected Bug A recurrent state inequivalence, got diff {diff_rec}"


@pytest.mark.parametrize("width", (16, 64, 128, 192, 256))
def test_chunk_checkpoint_tracks_fp32_exactly_without_changing_outputs(width):
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    torch.manual_seed(23)
    first, half, total = 64, 64, 192
    q = torch.randn(1, total, 2, width, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, 4, width, device=DEV, dtype=torch.bfloat16)
    g = -torch.rand(1, total, 4, device=DEV, dtype=torch.float32)
    beta = torch.rand_like(g)
    init = torch.randn(4, 4, width, width, device=DEV, dtype=torch.float32)
    init[3].copy_(init[1])
    indices = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    cu_seqlens = torch.tensor([0, first, total], dtype=torch.int64, device=DEV)

    plain_state = init.clone()
    plain = gdn_prefill_chunk_fla(
        q,
        k,
        v,
        g,
        beta,
        state_source=plain_state,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=width**-0.5,
    )
    tracked_state = init.clone()
    tracked = gdn_prefill_chunk_fla(
        q,
        k,
        v,
        g,
        beta,
        state_source=tracked_state,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=width**-0.5,
        track_indices=torch.tensor([2], dtype=torch.int32, device=DEV),
        track_h_rows=torch.tensor([2], dtype=torch.int32, device=DEV),
    )
    boundary_state = init.clone()
    gdn_prefill_chunk_fla(
        q[:, first : first + half],
        k[:, first : first + half],
        v[:, first : first + half],
        g[:, first : first + half],
        beta[:, first : first + half],
        state_source=boundary_state,
        indices=torch.tensor([3], dtype=torch.int32, device=DEV),
        cu_seqlens=torch.tensor([0, half], dtype=torch.int64, device=DEV),
        scale=width**-0.5,
    )

    assert torch.equal(tracked, plain)
    assert torch.equal(tracked_state[0], plain_state[0])
    assert torch.equal(tracked_state[1], plain_state[1])
    assert torch.equal(tracked_state[2], boundary_state[3]), (
        "tracked second-sequence state after row 64 differs from a separate 64-token prefill; "
        f"max_abs_diff={(tracked_state[2] - boundary_state[3]).abs().max().item():.8g}"
    )


@pytest.mark.parametrize("tokens", (2, 3, 4, 5, 6, 8))
def test_spec_verify_graph_matches_eager(tokens):
    """Captured multi-row recurrence and all row-checkpoint buffers match eager verify."""
    from freetoken.attention.linear import FLAMetadata

    op, _ = _make_layer(3, seed=3)
    ctx = _ctx(3, spec_steps=tokens)
    _, reqs, _ = _prefill(op, ctx, [128, 37], seed=13)
    pool = ctx.linear_state_pool

    def fla(slot: int) -> FLAMetadata:
        return FLAMetadata(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=DEV),
            cache_indices=torch.tensor([slot], dtype=torch.int32, device=DEV),
            has_initial_state=torch.ones(1, dtype=torch.bool, device=DEV),
        )

    def verify_batch(req, metadata: FLAMetadata) -> Batch:
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.spec_logits_indices = torch.arange(tokens, device=DEV)
        batch.fla_metadata = metadata
        return batch

    static_x = torch.zeros(tokens, HIDDEN, device=DEV, dtype=torch.bfloat16)
    static_fla = fla(0)  # slot 0 is scratch here
    with ctx.forward_batch(verify_batch(reqs[0], static_fla)):
        op.forward(static_x)  # warmup
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture = verify_batch(reqs[0], static_fla)
    with ctx.forward_batch(capture), torch.cuda.graph(graph):
        captured = op.forward(static_x)

    gen = torch.Generator(device=DEV).manual_seed(5)
    for step, req in enumerate([reqs[0], reqs[1], reqs[0], reqs[0], reqs[1]]):
        slot = req.table_idx
        x = torch.randn(tokens, HIDDEN, generator=gen, device=DEV, dtype=torch.bfloat16)
        state = [pool.conv_states[:, slot], pool.recurrent_states[:, slot]]
        before = [t.clone() for t in state]
        eager_batch = verify_batch(req, fla(slot))
        with ctx.forward_batch(eager_batch):
            eager = op.forward(x)
        expect = [t.clone() for t in state]
        expect_spec = [
            pool.spec_states[0, :tokens].clone(),
            pool.spec_conv_in[0, :tokens].clone(),
            pool.spec_conv_pre[0].clone(),
        ]
        for t, b in zip(state, before):
            t.copy_(b)
        static_x.copy_(x)
        static_fla.cache_indices.fill_(slot)
        graph.replay()
        assert torch.equal(captured, eager), f"verify output diverged at step {step}"
        assert all(torch.equal(t, e) for t, e in zip(state, expect)), f"state at step {step}"
        actual_spec = [
            pool.spec_states[0, :tokens],
            pool.spec_conv_in[0, :tokens],
            pool.spec_conv_pre[0],
        ]
        assert all(torch.equal(a, e) for a, e in zip(actual_spec, expect_spec)), (
            f"row checkpoints at step {step}"
        )


@pytest.mark.parametrize("tokens", (2, 3, 4, 5, 6, 8))
def test_gdn_row_checkpoints_equal_exact_prefix_recurrence(tokens):
    """Use identical precomputed recurrence inputs to isolate checkpoint exactness from
    shape-dependent input projection rounding. Rebuilding each conv window from its pre-window
    and captured conv inputs must also exactly match the sequential kernel state."""
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    ratio = 3
    num_k, num_v = HEADS[ratio]
    torch.manual_seed(103 + tokens)
    q = torch.randn(1, tokens, num_k, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, num_v, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(tokens, num_v, device=DEV, dtype=torch.float32)
    b = torch.randn_like(a)
    A_log = torch.randn(num_v, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn_like(A_log)
    initial = torch.randn(1, num_v, HEAD_DIM, HEAD_DIM, device=DEV, dtype=torch.float32)
    states = torch.empty(1, tokens, num_v, HEAD_DIM, HEAD_DIM, device=DEV, dtype=torch.float32)
    indices = torch.zeros(1, dtype=torch.int32, device=DEV)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=DEV)

    whole_state = initial.clone()
    gdn_decode_fla(
        q,
        k,
        v,
        a,
        b,
        A_log=A_log,
        dt_bias=dt_bias,
        state_source=whole_state,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=HEAD_DIM**-0.5,
        intermediate_states=states,
        intermediate_indices=indices,
    )
    for stop in range(1, tokens + 1):
        prefix_state = initial.clone()
        gdn_decode_fla(
            q[:, :stop],
            k[:, :stop],
            v[:, :stop],
            a[:stop],
            b[:stop],
            A_log=A_log,
            dt_bias=dt_bias,
            state_source=prefix_state,
            indices=indices,
            cu_seqlens=torch.tensor([0, stop], dtype=torch.int32, device=DEV),
            scale=HEAD_DIM**-0.5,
        )
        assert torch.equal(states[0, stop - 1], prefix_state[0]), (
            f"recurrent checkpoint after row {stop}"
        )

    conv_dim = 32
    conv_input = torch.randn(tokens, conv_dim, device=DEV, dtype=torch.bfloat16)
    conv_weight = torch.randn(conv_dim, CONV_K, device=DEV, dtype=torch.bfloat16)
    conv_pre = torch.randn(conv_dim, CONV_K - 1, device=DEV, dtype=torch.bfloat16)
    conv_state = conv_pre.unsqueeze(0).clone()
    conv_states = []
    for row in range(tokens):
        causal_conv1d_decode(
            conv_input[row : row + 1].clone(),
            conv_state,
            conv_weight,
            indices,
        )
        conv_states.append(conv_state[0].clone())
    for stop, actual in enumerate(conv_states, start=1):
        rebuilt = torch.cat([conv_pre, conv_input[:stop].transpose(0, 1)], dim=-1)[
            :, -(CONV_K - 1) :
        ]
        assert torch.equal(actual, rebuilt), f"conv window after row {stop}"


@pytest.mark.parametrize("state_dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize("compact", (False, True), ids=("row-cache", "compact-replay"))
def test_gdn_k4_commit_matches_raw_token_replay(state_dtype, compact, monkeypatch):
    """Verify and commit every prefix against RAW T=1 state persistence."""
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
    from freetoken.kvcache import linear_state_pool as pool_module
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    monkeypatch.setenv("FREETOKEN_MTP_COMPACT_STATE", "1" if compact else "0")
    monkeypatch.setattr(pool_module, "ssm_state_dtype", lambda: state_dtype)
    num_k, num_v = HEADS[3]
    torch.manual_seed(409)
    q = torch.randn(1, 5, num_k, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 5, num_v, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(5, num_v, device=DEV, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(num_v, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn_like(A_log)
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=num_k,
        num_value_heads=num_v,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_dim=CONV_K,
        output_gate="sigmoid",
    )
    pool = LinearStatePool(group, 2, torch.bfloat16, DEV, tp_size=1, spec_steps=5)
    slot = 1
    indices = torch.tensor([slot], dtype=torch.int32, device=DEV)
    initial = torch.randn(num_v, HEAD_DIM, HEAD_DIM, device=DEV).to(state_dtype)
    pool.recurrent_states[0, slot].copy_(initial)
    pool.spec_conv_pre.zero_()
    pool.spec_conv_in.zero_()
    if compact:
        pool.spec_gate_params[0] = (A_log, dt_bias)
        pool.spec_qkv[0].copy_(
            torch.cat([q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)], dim=1)
        )
        pool.spec_ba[0].copy_(torch.cat([b, a], dim=1))

    def run(start, stop, state, checkpoints=None, rollback_tape=None):
        return gdn_decode_fla(
            q[:, start:stop],
            k[:, start:stop],
            v[:, start:stop],
            a[start:stop],
            b[start:stop],
            A_log=A_log,
            dt_bias=dt_bias,
            state_source=state,
            indices=indices,
            cu_seqlens=torch.tensor([0, stop - start], dtype=torch.int32, device=DEV),
            scale=HEAD_DIM**-0.5,
            intermediate_states=checkpoints,
            intermediate_indices=pool.spec_index if checkpoints is not None else None,
            rollback_tape=rollback_tape,
        )

    if compact:
        tape = (pool.spec_states[0, 0], pool.spec_qkv[0], pool.spec_ba[0])
        verify_out = run(0, 5, pool.recurrent_states[0], rollback_tape=tape)
        saved_initial = pool.spec_states[0, 0].clone()
    else:
        verify_out = run(0, 5, pool.recurrent_states[0], pool.spec_states[0].unsqueeze(0))
        saved_initial = None

    raw_state = torch.zeros_like(pool.recurrent_states[0])
    raw_state[slot].copy_(initial)
    raw_prefix_states = []
    raw_outputs = []
    for row in range(5):
        raw_outputs.append(run(row, row + 1, raw_state))
        raw_prefix_states.append(raw_state[slot].clone())
    raw_outputs = torch.cat(raw_outputs, dim=0)
    assert torch.equal(verify_out, raw_outputs), (
        f"verify outputs differ from RAW rows; "
        f"max_abs={(verify_out.float() - raw_outputs.float()).abs().max().item():.8g}"
    )

    next_q = torch.randn(1, 1, num_k, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    next_k = torch.randn_like(next_q)
    next_v = torch.randn(1, 1, num_v, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    next_a = torch.randn(1, num_v, device=DEV, dtype=torch.bfloat16)
    next_b = torch.randn_like(next_a)
    decode_args = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        indices=indices,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=DEV),
        scale=HEAD_DIM**-0.5,
    )
    for row, raw_prefix in enumerate(raw_prefix_states):
        if compact:
            pool.spec_states[0, 0].copy_(saved_initial)
        pool.commit_spec_row(slot, row)
        committed_state = pool.recurrent_states[0, slot].clone()
        assert torch.equal(committed_state, raw_prefix), (
            f"committed prefix {row} differs from RAW; "
            f"max_abs={(committed_state.float() - raw_prefix.float()).abs().max().item():.8g}"
        )

        committed = torch.zeros_like(pool.recurrent_states[0])
        committed[slot].copy_(committed_state)
        raw = torch.zeros_like(pool.recurrent_states[0])
        raw[slot].copy_(raw_prefix)
        committed_out = gdn_decode_fla(
            next_q, next_k, next_v, next_a, next_b, state_source=committed, **decode_args
        )
        raw_out = gdn_decode_fla(
            next_q, next_k, next_v, next_a, next_b, state_source=raw, **decode_args
        )
        assert torch.equal(committed_out, raw_out), (
            f"next decode after prefix {row} differs; "
            f"max_abs={(committed_out.float() - raw_out.float()).abs().max().item():.8g}"
        )
