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
    # This test addresses distinct recurrent verify rows; compact-state mode stores only one
    # recurrent row and replays its raw-input tape on commit (covered separately).
    monkeypatch.setenv("FREETOKEN_MTP_COMPACT_STATE", "0")
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
        2 * group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
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


@pytest.mark.parametrize(("capacity", "start"), [(8, 6), (128, 126)])
def test_qsa_rejection_merge_keeps_wrapped_prefix_and_restores_scratch(capacity, start):
    count = 3
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

    SchedulerSpecMixin._restore_qsa_state(scheduler, req, keep_start=start, keep_count=count)

    expected = snapshot.clone()
    for pos, value in zip(((start + i) % capacity for i in range(count)), (106, 107, 108)):
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
            (
                live["state"],
                request.cached_len,
                request.device_len,
                request.input_ids[: request.cached_len].tolist(),
            )
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
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = _terminal_scheduler(
        [1, 2, 3]
    )
    drafts = [10, 11, 12]
    sampled = [99, 20, 21, 22]  # EOS is the correction for rejected draft 10.

    tokens, committed = _commit_terminal(scheduler, req, drafts, sampled, finish_state, alloc_len)

    assert tokens == [99] and committed == 1
    assert observed == [(row_states[0], 3, 4, [1, 2, 3])]


def test_accepted_eos_at_page_boundary_donates_only_valid_prefix_state():
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = _terminal_scheduler(
        [1, 2, 3]
    )
    drafts = [44, 99, 12]
    sampled = [44, 99, 13, 14]

    tokens, committed = _commit_terminal(scheduler, req, drafts, sampled, finish_state, alloc_len)

    assert tokens == [44, 99, 13] and committed == 2
    # The EOS is at position 4; cached_len 4 is the page boundary before it.
    assert observed == [(row_states[1], 4, 5, [1, 2, 3, 44])]
    cache.check_integrity()


def test_bonus_eos_keeps_full_accepted_verify_state_and_excludes_bonus():
    scheduler, req, cache, row_states, _, observed, finish_state, alloc_len = _terminal_scheduler(
        [1, 2, 3]
    )
    drafts = [44, 45, 46]
    sampled = [44, 45, 46, 99]

    tokens, committed = _commit_terminal(scheduler, req, drafts, sampled, finish_state, alloc_len)

    assert tokens == sampled and committed == 4
    assert observed == [(row_states[-1], 6, 7, [1, 2, 3, 44, 45, 46])]
    cache.check_integrity()


@pytest.mark.parametrize("raise_on_call", [None, 2])
def test_mtp_draft_kv_warmup_chunks_aligned_rows_and_restores_request(monkeypatch, raise_on_call):
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
        model = SimpleNamespace(mtp=MTP(), model=SimpleNamespace(_last_residual=residual))
        scheduler = SimpleNamespace(
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
        scheduler._fill_mtp_kv = lambda req, start_pos, r_window, tok_window: (
            SchedulerSpecMixin._fill_mtp_kv(scheduler, req, start_pos, r_window, tok_window)
        )
        return scheduler

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

    SchedulerSpecMixin.warmup_mtp_draft_kv(scheduler, req, SimpleNamespace(input_ids=first_ids))
    assert [len(call[1]) for call in calls] == [8, 8, 2]
    assert [call[2].tolist() for call in calls] == [list(range(1, 9)), list(range(9, 17)), [17, 18]]
    assert torch.equal(torch.cat([call[0] for call in calls]), first_residual[:-1])
    assert torch.equal(torch.cat([call[1] for call in calls]), first_ids[1:])
    assert (req.cached_len, req.device_len) == (19, 23)

    # The next contiguous target window consumes the saved final h_i with its token_(i+1).
    scheduler.engine.model.model._last_residual = torch.arange(19, 27, dtype=torch.float32).view(
        8, 1
    )
    req.cached_len, req.device_len = 27, 31
    next_ids = torch.arange(19, 27, dtype=torch.int32)
    SchedulerSpecMixin.warmup_mtp_draft_kv(scheduler, req, SimpleNamespace(input_ids=next_ids))
    hidden, tokens, positions = calls[-1]
    assert torch.equal(hidden[:, 0], torch.arange(18, 26, dtype=torch.float32))
    assert torch.equal(tokens, next_ids)
    assert torch.equal(positions, torch.arange(19, 27, dtype=torch.int32))
    assert (req.cached_len, req.device_len) == (27, 31)


@pytest.mark.parametrize(
    "has_prime,has_qsa_store", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mtp_kv_fill_dispatches_by_prime_and_qsa_capabilities(
    monkeypatch, has_prime, has_qsa_store
):
    from contextlib import nullcontext

    import freetoken.scheduler.spec as spec_module

    monkeypatch.setattr(spec_module, "Batch", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("freetoken.moe.offload_cache.DECODE_PATH_MAX_TOKENS", 8)
    calls = []

    class Backend:
        @staticmethod
        def prepare_metadata(batch):
            pass

    class ForwardBatch:
        @staticmethod
        def forward_batch(batch):
            return nullcontext()

    mtp = SimpleNamespace(
        forward=lambda hidden, tokens, batch: calls.append(
            ("forward", hidden.clone(), tokens.clone(), tokens.numel())
        )
    )
    if has_prime:
        mtp.prime_kv = lambda hidden, tokens, batch: calls.append(
            ("prime_kv", hidden.clone(), tokens.clone(), tokens.numel())
        )
    backend = Backend()
    if has_qsa_store:
        backend.store_qsa_kv = lambda *args, **kwargs: None
    scheduler = SimpleNamespace(
        device=torch.device("cpu"),
        _model_is_mrope=False,
        engine=SimpleNamespace(
            model=SimpleNamespace(mtp=mtp),
            page_table=torch.arange(32, dtype=torch.int32).view(1, 32),
            attn_backend=backend,
            config=SimpleNamespace(max_extend_tokens=1024),
            ctx=ForwardBatch(),
        ),
    )
    req = SimpleNamespace(uid=3, table_idx=0, cached_len=12, device_len=14)

    SchedulerSpecMixin._fill_mtp_kv(
        scheduler,
        req,
        start_pos=4,
        r_window=torch.arange(17, dtype=torch.float32).view(17, 1),
        tok_window=torch.arange(21, 38, dtype=torch.int32),
    )

    expected_method = "prime_kv" if has_prime and has_qsa_store else "forward"
    assert [call[0] for call in calls] == [expected_method] * 3
    assert [call[3] for call in calls] == [8, 8, 1]
    assert torch.equal(torch.cat([call[1][:, 0] for call in calls]), torch.arange(17))
    assert torch.equal(torch.cat([call[2] for call in calls]), torch.arange(21, 38))
    assert (req.cached_len, req.device_len) == (12, 14)


def test_qsa_first_draft_mtp_ring_survives_later_draft_rollback_and_target_merge():
    layers, capacity, width = 3, 8, 2
    scratch_base, table_idx = 5, 0
    ring = torch.arange(layers * capacity * width, dtype=torch.float32).view(
        1, layers, capacity, width
    )
    scratch = torch.arange(layers * (scratch_base + 1) * width, dtype=torch.float32).view(
        layers, scratch_base + 1, width
    )
    kv = SimpleNamespace(
        _pending_ring=ring.clone(),
        _cmp_k_buffer=scratch.clone(),
        _cmp_scratch_base=scratch_base,
        _mtp_slot=2,
    )
    scheduler = SimpleNamespace(engine=SimpleNamespace(kv_cache=kv))
    req = SimpleNamespace(uid=84, table_idx=table_idx)
    SchedulerSpecMixin._snapshot_qsa_state(scheduler, req)
    live_ring = kv._pending_ring
    initial_ring = live_ring[table_idx].clone()
    initial_scratch = kv._cmp_k_buffer.clone()

    # First MTP forward receives only accepted target-fed fill rows plus the current token.
    # Retain its exact MTP pending keys as the new rollback baseline.
    live_ring[table_idx, 2, torch.tensor([6, 7, 0])] = torch.tensor(
        [[106.0, 106.5], [107.0, 107.5], [108.0, 108.5]]
    )
    first_step_ring = live_ring[table_idx, 2].clone()
    SchedulerSpecMixin._retain_mtp_ring(scheduler, req)

    # Later draft steps may write speculative keys; verifier writes target-layer keys.
    live_ring[table_idx, 2].fill_(900)
    live_ring[table_idx, 0, torch.tensor([6, 7, 0])] = 200
    live_ring[table_idx, 1, torch.tensor([6, 7, 0])] = 300
    kv._cmp_k_buffer[:, scratch_base + table_idx].fill_(999)

    SchedulerSpecMixin._restore_qsa_state(scheduler, req, keep_start=6, keep_count=3)

    expected = initial_ring.clone()
    expected[0, torch.tensor([6, 7, 0])] = 200
    expected[1, torch.tensor([6, 7, 0])] = 300
    expected[2] = first_step_ring
    assert torch.equal(kv._pending_ring[table_idx], expected)
    assert torch.equal(
        kv._cmp_k_buffer[:, scratch_base + table_idx],
        initial_scratch[:, scratch_base + table_idx],
    )


def test_commit_mtp_residual_records_live_track_and_verify_rows(monkeypatch):
    import freetoken.models.qwen4_exp.model as model_module

    live = torch.zeros(4, 2)
    verify = torch.zeros(1, 3, 2)

    class Pool:
        spec_slot_states = {"mtp_residual": verify}

        @staticmethod
        def has_slot_state(name):
            return name == "mtp_residual"

        @staticmethod
        def slot_state(name):
            assert name == "mtp_residual"
            return live

    monkeypatch.setattr(
        model_module, "get_global_ctx", lambda: SimpleNamespace(linear_state_pool=Pool())
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    metadata = SimpleNamespace(
        cache_indices=torch.tensor([1]),
        cu_seqlens=torch.tensor([0, 3]),
        track_dst=torch.tensor([2]),
        track_boundary_row=torch.tensor([2]),
    )
    batch = SimpleNamespace(fla_metadata=metadata, spec_logits_indices=torch.arange(3))

    model_module.commit_mtp_residual(hidden, batch)

    assert torch.equal(live[1], hidden[-1])
    assert torch.equal(live[2], hidden[1])  # boundary row 2 is exclusive
    assert torch.equal(verify[0], hidden)


def test_restore_linear_prefix_primes_cloned_mtp_prompt_carry():
    from freetoken.scheduler.scheduler import Scheduler

    live = torch.zeros(6, 3)
    live[2] = torch.tensor([7.0, 8.0, 9.0])
    live[5] = torch.tensor([1.0, 2.0, 3.0])

    class Pool:
        @staticmethod
        def copy_from(src, dst):
            live[dst].copy_(live[src])

        @staticmethod
        def has_slot_state(name):
            return name == "mtp_residual"

        @staticmethod
        def slot_state(name):
            assert name == "mtp_residual"
            return live

    req = SimpleNamespace(uid=42, mamba_restore_src=5, linear_slot_idx=2, cached_len=16)
    model = SimpleNamespace(_last_residual=torch.full((1, 3), -1.0))
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(linear_state_pool=Pool(), model=SimpleNamespace(model=model)),
        _mtp_prompt_carry=None,
        spec_mtp=1,
    )
    Scheduler._restore_linear_states(scheduler, SimpleNamespace(is_prefill=True, reqs=[req]))

    assert req.mamba_restore_src is None
    assert live[2].tolist() == [1.0, 2.0, 3.0]
    uid, cached_len, carry = scheduler._mtp_prompt_carry
    assert (uid, cached_len) == (42, 16)
    assert carry.shape == (1, 3) and carry.tolist() == [[1.0, 2.0, 3.0]]
    assert model._last_residual.tolist() == [[1.0, 2.0, 3.0]]
    live[2].fill_(99)
    assert carry.tolist() == [[1.0, 2.0, 3.0]]


def test_mtp_residual_capture_and_restore_are_noops_without_declared_state(monkeypatch):
    import freetoken.models.qwen4_exp.model as model_module
    from freetoken.scheduler.scheduler import Scheduler

    class EmptyPool:
        @staticmethod
        def has_slot_state(name):
            return False

    monkeypatch.setattr(
        model_module, "get_global_ctx", lambda: SimpleNamespace(linear_state_pool=EmptyPool())
    )
    model_module.commit_mtp_residual(
        torch.ones(2, 3), SimpleNamespace(fla_metadata=None, spec_logits_indices=None)
    )

    req = SimpleNamespace(uid=9, mamba_restore_src=1, linear_slot_idx=0, cached_len=0)
    scheduler = SimpleNamespace(engine=SimpleNamespace(linear_state_pool=None))
    Scheduler._restore_linear_states(scheduler, SimpleNamespace(is_prefill=True, reqs=[req]))
    assert req.mamba_restore_src == 1
    assert not hasattr(scheduler, "_mtp_prompt_carry")


def test_mtp_fill_queue_merges_only_contiguous_rows_and_is_single_span():
    scheduler = SimpleNamespace(_mtp_kv_rows=None)
    req = SimpleNamespace(uid=17, device_len=13)
    residuals = [torch.tensor([[float(i), float(i + 1)]]) for i in range(6)]
    tokens = [torch.tensor([100 + i], dtype=torch.int32) for i in range(6)]

    SchedulerSpecMixin._queue_mtp_fill(scheduler, req, 8, residuals[0], tokens[0])
    SchedulerSpecMixin._queue_mtp_fill(scheduler, req, 9, residuals[1], tokens[1])
    SchedulerSpecMixin._queue_mtp_fill(scheduler, req, 10, residuals[2], tokens[2])
    uid, start, hidden, ids = scheduler._mtp_kv_rows
    assert (uid, start) == (17, 8)
    assert hidden.shape == (3, 2) and ids.tolist() == [100, 101, 102]
    residuals[0].fill_(-1)
    tokens[1].fill_(-1)
    assert hidden[0].tolist() == [0.0, 1.0] and ids.tolist() == [100, 101, 102]
    req.device_len = 12  # start 8 + 3 queued rows == device_len - 1
    taken = SchedulerSpecMixin._take_mtp_fill(scheduler, req)
    assert taken is not None
    assert torch.equal(taken[0], hidden) and taken[1].tolist() == [100, 101, 102]
    assert scheduler._mtp_kv_rows is None

    # A gap replaces the one pending span instead of growing a detached queue.
    SchedulerSpecMixin._queue_mtp_fill(scheduler, req, 12, residuals[3], tokens[3])
    assert scheduler._mtp_kv_rows[1] == 12
    assert scheduler._mtp_kv_rows[2].shape == (1, 2)
    assert scheduler._mtp_kv_rows[3].tolist() == [103]

    # Different owners also replace, and only an exact device_len boundary is consumable.
    other = SimpleNamespace(uid=18, device_len=40)
    SchedulerSpecMixin._queue_mtp_fill(scheduler, other, 13, residuals[4], tokens[4])
    assert scheduler._mtp_kv_rows[0] == 18
    assert SchedulerSpecMixin._take_mtp_fill(scheduler, other) is None
    assert scheduler._mtp_kv_rows is None


@pytest.mark.parametrize("accepted_drafts", range(5))
def test_run_spec_step_k4_commits_only_accepted_target_rows(accepted_drafts):
    """Exercise real k=4 acceptance, row commit, page rollback, and carry bookkeeping on CPU."""
    from freetoken.scheduler.table import TableManager

    page_table = torch.zeros((2, 32), dtype=torch.int32)
    cache = CacheManager(24, 1, page_table, "radix")
    req = Req(
        input_ids=torch.tensor(range(1, 9), dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=23,
        sampling_params=SamplingParams(max_tokens=16, temperature=0),
        cache_handle=cache.prefix_cache.match_prefix(torch.empty(0, dtype=torch.int32)).cuda_handle,
    )
    cache.allocate_paged([req])
    req.cached_len, req.device_len = 7, 8
    token_pool = torch.zeros((2, 32), dtype=torch.int64)
    token_pool[0, 7] = 8
    drafts = [50, 51, 52, 53]
    correction = 80 + accepted_drafts
    sampled = drafts[:accepted_drafts] + [correction]
    sampled += [90] * (5 - len(sampled))
    row_writes = []

    class Pool:
        spec_states = torch.zeros((1, 5, 1), dtype=torch.int64)

        def commit_spec_row(self, slot, row):
            row_writes.append((slot, row))

    pool = Pool()
    model_state = SimpleNamespace(_last_residual=torch.zeros((1, 1)))
    model = SimpleNamespace(
        mtp=object(), model=model_state, config=SimpleNamespace(num_experts_per_tok=1)
    )
    scheduler = SimpleNamespace(
        spec_mtp=4,
        device=torch.device("cpu"),
        engine=SimpleNamespace(model=model, linear_state_pool=pool, kv_cache=None),
        token_pool=token_pool,
        cache_manager=cache,
        decode_manager=SimpleNamespace(filter_reqs=lambda _reqs: None),
        finished_reqs=set(),
        _mtp_kv_rows=None,
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        _spec_eligible_req=lambda: req,
        _flush_deferred_replays=lambda: None,
        _snapshot_qsa_state=lambda _req: None,
        _snapshot_ple_state=lambda _req: None,
        _retain_mtp_ring=lambda _req: None,
        _spec_snapshot_slot=lambda _req: 9,
        _linear_slot=lambda _req: 0,
        _restore_qsa_state=lambda _req, **_kwargs: None,
        _restore_ple_state=lambda _req: None,
        _draft_step=lambda _req, pos, residual, _token: (
            residual,
            torch.zeros((1, 1)),
            torch.tensor([drafts[pos - 7]], dtype=torch.int64),
        ),
        _take_mtp_fill=lambda _req: None,
        _checked_verify_forward=lambda *_args: None,
        _replay=lambda *_args: pytest.fail("zero-replay row commit must not replay"),
        free_spec_snapshot_slot=lambda _req: None,
        _match_stop_str=lambda _req: None,
        send_result=lambda _messages: None,
    )
    allocated = []

    def prepare(batch):
        cache.allocate_paged([req])
        allocated[:] = page_table[0, 8:12].tolist()
        batch.positions = torch.arange(7, 12, dtype=torch.int32)
        return SimpleNamespace(
            input_tuple=(torch.zeros(5, dtype=torch.long), torch.arange(7, 12)),
            sample_args=None,
        )

    def forward(_batch, _sample_args):
        pool.spec_states[0, :, 0] = torch.arange(5)
        model_state._last_residual = torch.arange(5, dtype=torch.float32).view(5, 1)
        return SimpleNamespace(
            next_tokens_cpu=torch.tensor(sampled),
            next_tokens_gpu=torch.tensor(sampled),
            copy_done_event=SimpleNamespace(synchronize=lambda: None),
        )

    scheduler._prepare_batch = prepare
    scheduler.engine.forward_batch = forward
    scheduler._commit_spec_tokens = lambda r, tokens, **kwargs: (
        SchedulerSpecMixin._commit_spec_tokens(scheduler, r, tokens, **kwargs)
    )
    scheduler.table_manager = TableManager(max_running_reqs=2, page_table=page_table)

    assert SchedulerSpecMixin.run_spec_step(scheduler)

    committed = accepted_drafts + 1  # each rejected draft contributes its correction token
    keep = 7 + committed
    assert token_pool[0, 8 : 8 + committed].tolist() == sampled[:committed]
    assert req.input_ids.tolist() == list(range(1, 9)) + sampled[:committed]
    assert (req.cached_len, req.device_len) == (keep, keep + 1)
    assert row_writes == [(0, committed - 1)] if committed <= 4 else row_writes == []
    free = set(cache.free_slots.tolist())
    assert set(allocated[committed:]).issubset(free)
    assert set(allocated[:committed]).isdisjoint(free)
    expected_rows = committed - 1
    carry = scheduler._mtp_kv_rows
    assert (carry is None) == (expected_rows == 0)
    if carry is not None:
        assert carry[0:2] == (req.uid, 8)
        assert carry[2].shape == (expected_rows, 1)
        assert carry[3].tolist() == sampled[:expected_rows]
        assert (
            SchedulerSpecMixin._take_mtp_fill(scheduler, req)[1].tolist() == sampled[:expected_rows]
        )
