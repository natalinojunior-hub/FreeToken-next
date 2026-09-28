"""CPU checks for committing captured speculative state at an accepted prefix."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.spec import accept_drafts
from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    spec_state_bytes,
    ssm_state_dtype,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.models.qwen4_exp.ple import commit_ngram_context, short_conv_reference
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.spec import SchedulerSpecMixin
from tests.models.qwen4_exp.test_ple import _config, _make_layer, _meta


def test_ple_row_state_matches_prefix_reference():
    torch.manual_seed(29)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    tokens = [11, 12, 13, 14, 15]
    context = [21, 22]
    meta = _meta([tokens], [context])
    x = torch.randn(len(tokens), args.ple_state_width)
    initial = torch.randn(1, args.ple_state_width, args.ple_conv_state_len)
    states = initial.clone()
    spec_rows = torch.empty(len(tokens), args.ple_state_width, args.ple_conv_state_len)

    layer._prefill_conv(x, meta, states, spec_out=spec_rows)

    for end in range(1, len(tokens) + 1):
        prefix_state = initial.clone()
        prefix_meta = _meta([tokens[:end]], [context])
        short_conv_reference(
            x[:end], prefix_meta, prefix_state, layer.conv1d.weight, args.ple_conv_dilation
        )
        assert torch.allclose(spec_rows[end - 1], prefix_state[0], rtol=1e-5, atol=1e-6)
    assert torch.equal(states[0], spec_rows[-1])


def test_ngram_spec_rows_match_rolling_context():
    tokens = [31, 32, 33, 34]
    context = [7, 8]
    meta = _meta([tokens], [context])
    pool = torch.zeros((1, len(context)), dtype=torch.int32)
    spec_rows = torch.empty((len(tokens), len(context)), dtype=torch.int32)

    commit_ngram_context(meta, None, pool, spec_out=spec_rows)

    history = context.copy()
    for row, token in enumerate(tokens):
        history.append(token)
        assert spec_rows[row].tolist() == history[-len(context) :]
    assert torch.equal(pool[0], spec_rows[-1])


def _linear_group():
    return LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate="silu",
    )


def test_pool_row_commit_and_spec_state_byte_accounting(monkeypatch):
    steps, slot = 4, 1
    slot_states = (
        SlotStateSpec("ple_conv", (3, 4), layer_ids=(0, 1)),
        SlotStateSpec("ple_ngram_ctx", (2,), dtype=torch.int32),
    )
    group = _linear_group()
    pool = LinearStatePool(
        group,
        num_slots=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=slot_states,
        spec_steps=steps,
    )
    row_values = torch.arange(steps, dtype=pool.spec_states.dtype).view(1, steps, 1, 1, 1)
    pool.spec_states.copy_(row_values.expand_as(pool.spec_states))
    pool.spec_conv_pre.fill_(-1)
    for row in range(steps):
        pool.spec_conv_in[:, row].fill_(row + 1)
    for name, states in pool.spec_slot_states.items():
        for row in range(steps):
            states[:, row].fill_(row + 10)

    row = 2
    pool.commit_spec_row(slot, row)
    assert torch.equal(pool.recurrent_states[:, slot], pool.spec_states[:, row])
    expected_window = torch.cat(
        [pool.spec_conv_pre, pool.spec_conv_in[:, : row + 1].transpose(1, 2)], dim=2
    )[..., -pool.conv_states.shape[-1] :]
    assert torch.equal(pool.conv_states[:, slot], expected_window)
    for name, spec_rows in pool.spec_slot_states.items():
        assert torch.equal(pool.slot_states[name][:, slot], spec_rows[:, row])

    model_config = SimpleNamespace(
        mtp_row_state_commit=True,
        native_mtp_layers=0,
        slot_states=slot_states,
        linear_attention_group=lambda: group,
    )
    config = SimpleNamespace(
        spec_mtp=steps - 1,
        model_config=model_config,
        tp_info=SimpleNamespace(size=1),
        dtype=torch.bfloat16,
    )
    monkeypatch.setenv("FREETOKEN_MTP_ROW_COMMIT", "1")
    n_layers = len(group.layer_ids)
    conv_dim = (
        2 * group.num_key_heads * group.key_head_dim
        + group.num_value_heads * group.value_head_dim
    )
    recurrent_bytes = (
        group.num_value_heads
        * group.key_head_dim
        * group.value_head_dim
        * ssm_state_dtype().itemsize
    )
    conv_bytes = conv_dim * config.dtype.itemsize
    sibling_bytes = sum(
        max(1, len(spec.layer_ids))
        * torch.empty((), dtype=spec.dtype or config.dtype).element_size()
        * torch.tensor(spec.shape).prod().item()
        for spec in slot_states
    )
    expected_bytes = (
        n_layers
        * (
            steps * (recurrent_bytes + conv_bytes)
            + conv_dim * config.dtype.itemsize * (group.conv_kernel_dim - 1)
        )
        + steps * sibling_bytes
    )
    assert spec_state_bytes(config) == expected_bytes


def test_qsa_rejection_merge_keeps_wrapped_prefix_and_restores_scratch():
    capacity, start, count = 8, 6, 3
    snapshot = torch.arange(2 * capacity * 3, dtype=torch.float32).view(2, capacity, 3)
    scratch_snapshot = torch.arange(6, dtype=torch.float32).view(2, 3)
    ring = torch.full_like(snapshot, -1)
    ring[:, start % capacity] = 106
    ring[:, (start + 1) % capacity] = 107
    ring[:, (start + 2) % capacity] = 108
    ring[:, (start + count) % capacity] = 999  # rejected row
    scratch_base, table_idx = 5, 0
    scratch = torch.zeros(2, scratch_base + 3, 3)
    scratch[:, scratch_base + table_idx] = -5
    kv = SimpleNamespace(
        _pending_ring=torch.stack([ring]),
        _cmp_k_buffer=scratch,
        _cmp_scratch_base=scratch_base,
    )
    scheduler = SimpleNamespace(
        _spec_qsa_snapshots={4: (snapshot.clone(), scratch_snapshot.clone())},
        engine=SimpleNamespace(kv_cache=kv),
    )
    req = SimpleNamespace(uid=4, table_idx=table_idx)

    SchedulerSpecMixin._restore_qsa_state(
        scheduler, req, keep_start=start, keep_count=count
    )

    expected = snapshot.clone()
    for pos, value in zip((6, 7, 0), (106, 107, 108)):
        expected[:, pos] = value
    assert torch.equal(kv._pending_ring[0], expected)
    assert torch.equal(kv._cmp_k_buffer[:, scratch_base + table_idx], scratch_snapshot)


def _terminal_scheduler(prompt, *, page_size=4):
    page_table = torch.zeros((2, 32), dtype=torch.int32)
    cache = CacheManager(8, page_size, page_table, "radix")
    ids = torch.tensor(prompt, dtype=torch.int32)
    req = Req(
        input_ids=ids,
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=3,
        sampling_params=SamplingParams(),
        cache_handle=cache.prefix_cache.match_prefix(ids[:0]).cuda_handle,
    )
    cache.allocate_paged([req])
    verify_rows = 1 + 3  # pending token plus the k=3 drafts
    req.cached_len = len(prompt) - 1
    spec_alloc_len = req.device_len + 3
    req.device_len = spec_alloc_len
    cache.allocate_paged([req])

    row_states = [f"target-row-{i}" for i in range(verify_rows)]
    live = {"state": row_states[-1]}
    observed = []

    def finish_state(count):
        # Mirrors the native row-commit branch in run_spec_step: the final output remains
        # pending, so the last donated target state is one input row earlier.
        if count < verify_rows:
            live["state"] = row_states[count - 1]

    def free_req_resources(request):
        observed.append(
            (live["state"], request.cached_len, request.device_len,
             request.input_ids[: request.cached_len].tolist())
        )
        cache.cache_req(request, finished=True)

    scheduler = SimpleNamespace(
        cache_manager=cache,
        decode_manager=SimpleNamespace(remove_req=lambda request: None),
        eos_token_ids={99},
        toolcall_anchor_id=-1,
        finished_reqs=set(),
        send_result=lambda reply: None,
        _match_stop_str=lambda request: None,
        _free_req_resources=free_req_resources,
    )
    return scheduler, req, cache, row_states, live, observed, finish_state, spec_alloc_len


def _commit_terminal(scheduler, req, drafts, sampled, finish_state, spec_alloc_len):
    tokens = accept_drafts(sampled, drafts)
    committed = SchedulerSpecMixin._commit_spec_tokens(
        scheduler,
        req,
        tokens,
        start_pos=req.device_len - 3,
        spec_alloc_len=spec_alloc_len,
        finish_state=finish_state,
    )
    return tokens, committed


def test_eos_correction_excludes_rejected_kv_and_commits_state_before_free():
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = (
        _terminal_scheduler([1, 2, 3])
    )
    drafts = [10, 11, 12]
    sampled = [99, 20, 21, 22]  # EOS is the correction for rejected draft 10.

    tokens, committed = _commit_terminal(
        scheduler, req, drafts, sampled, finish_state, alloc_len
    )

    assert tokens == [99] and committed == 1
    assert observed == [(row_states[0], 3, 4, [1, 2, 3])]
    cache.check_integrity()


def test_accepted_eos_at_page_boundary_donates_only_valid_prefix_state():
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = (
        _terminal_scheduler([1, 2, 3])
    )
    drafts = [44, 99, 12]
    sampled = [44, 99, 13, 14]

    tokens, committed = _commit_terminal(
        scheduler, req, drafts, sampled, finish_state, alloc_len
    )

    assert tokens == [44, 99, 13] and committed == 2
    # The EOS is at position 4; cached_len 4 is the page boundary before it.
    assert observed == [(row_states[1], 4, 5, [1, 2, 3, 44])]
    cache.check_integrity()


def test_bonus_eos_keeps_full_accepted_verify_state_and_excludes_bonus():
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = (
        _terminal_scheduler([1, 2, 3])
    )
    drafts = [44, 45, 46]
    sampled = [44, 45, 46, 99]

    tokens, committed = _commit_terminal(
        scheduler, req, drafts, sampled, finish_state, alloc_len
    )

    assert tokens == sampled and committed == 4
    assert observed == [(row_states[-1], 6, 7, [1, 2, 3, 44, 45, 46])]
    cache.check_integrity()


@pytest.mark.parametrize("raise_on_call", [None, 2])
def test_mtp_draft_kv_warmup_chunks_aligned_rows_and_restores_request(
    monkeypatch, raise_on_call
):
    from contextlib import nullcontext

    import freetoken.scheduler.spec as spec_module

    monkeypatch.setattr(spec_module, "Batch", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("freetoken.moe.offload_cache.DECODE_PATH_MAX_TOKENS", 8)

    calls = []

    class MTP:
        def forward(self, hidden, tokens, batch):
            calls.append((hidden.clone(), tokens.clone(), batch.positions.clone()))
            if raise_on_call == len(calls):
                raise RuntimeError("synthetic warmup failure")

    class Backend:
        @staticmethod
        def prepare_metadata(batch):
            pass

    class ForwardBatch:
        @staticmethod
        def forward_batch(batch):
            return nullcontext()

    def make_scheduler(residual):
        model = SimpleNamespace(
            mtp=MTP(), model=SimpleNamespace(_last_residual=residual)
        )
        return SimpleNamespace(
            spec_mtp=4,
            engine=SimpleNamespace(
                model=model,
                page_table=torch.arange(256, dtype=torch.int32).view(1, 256),
                attn_backend=Backend(),
                ctx=ForwardBatch(),
            ),
            device=torch.device("cpu"),
            _model_is_mrope=False,
        )

    req = SimpleNamespace(uid=71, table_idx=0, cached_len=19, device_len=23)
    first_ids = torch.arange(19, dtype=torch.int32)
    first_residual = torch.arange(19, dtype=torch.float32).view(19, 1)
    scheduler = make_scheduler(first_residual)

    if raise_on_call is not None:
        with pytest.raises(RuntimeError, match="synthetic warmup failure"):
            SchedulerSpecMixin.warmup_mtp_draft_kv(
                scheduler, req, SimpleNamespace(input_ids=first_ids)
            )
        assert len(calls) == raise_on_call
        assert (req.cached_len, req.device_len) == (19, 23)
        return

    SchedulerSpecMixin.warmup_mtp_draft_kv(
        scheduler, req, SimpleNamespace(input_ids=first_ids)
    )
    assert [len(call[1]) for call in calls] == [8, 8, 2]
    assert [call[2].tolist() for call in calls] == [
        list(range(1, 9)), list(range(9, 17)), [17, 18]
    ]
    assert torch.equal(torch.cat([call[0] for call in calls]), first_residual[:-1])
    assert torch.equal(torch.cat([call[1] for call in calls]), first_ids[1:])
    assert (req.cached_len, req.device_len) == (19, 23)

    # The next contiguous target window consumes the saved final h_i with its token_(i+1).
    scheduler.engine.model.model._last_residual = torch.arange(
        19, 27, dtype=torch.float32
    ).view(8, 1)
    req.cached_len, req.device_len = 27, 31
    next_ids = torch.arange(19, 27, dtype=torch.int32)
    SchedulerSpecMixin.warmup_mtp_draft_kv(
        scheduler, req, SimpleNamespace(input_ids=next_ids)
    )
    hidden, tokens, positions = calls[-1]
    assert torch.equal(hidden[:, 0], torch.arange(18, 26, dtype=torch.float32))
    assert torch.equal(tokens, next_ids)
    assert torch.equal(positions, torch.arange(19, 27, dtype=torch.int32))
    assert (req.cached_len, req.device_len) == (27, 31)
