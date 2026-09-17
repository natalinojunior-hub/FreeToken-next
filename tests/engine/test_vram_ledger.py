"""Unit tests for the authoritative VRAM ledger (engine/vram_ledger.py).

Pure arithmetic on purpose: the ledger's whole contract is that a byte plan can be reasoned
about without a device, and the zero-regression property (a modelled reserve must not change
the plan while --memory-ratio is the tighter bound) is exactly the kind of invariant that is
cheap to pin here and expensive to discover on hardware.
"""

from types import SimpleNamespace

import pytest

from freetoken.engine.cache_budget import (
    net_cache_budget_bytes,
    pool_pages,
    required_bytes,
    resolve_moe_cache_auto,
)
from freetoken.engine.vram_ledger import (
    BACKEND_WORKSPACE,
    FRAGMENTATION_RESERVE,
    GRAPH_CAPTURE_EXTRA_SHAPE,
    GRAPH_CAPTURE_PEAK,
    TRITON_AUTOTUNE_ARENA,
    Kind,
    VramLedger,
    activation_peak_bytes,
    context_demand,
    context_feasibility,
    gdn_prefill_bytes,
    graph_capture_peak_bytes,
    graph_capture_shapes,
    graph_pool_bytes,
    modelled_reserves,
    open_ledger,
    tensor_bytes,
)

_GIB = 1 << 30
_MIB = 1 << 20


def _group(num_v_heads=48, num_k_heads=16, k_dim=128, v_dim=128):
    """A LinearGatedDeltaGroupConfig stand-in: only the four geometry fields are read."""
    return SimpleNamespace(
        num_key_heads=num_k_heads,
        num_value_heads=num_v_heads,
        key_head_dim=k_dim,
        value_head_dim=v_dim,
    )


def _ledger(baseline=16 * _GIB, ratio=0.9, weights=8 * _GIB, reserves=..., **kw):
    """A ledger with the default modelled reserve lines, like an engine that just opened one."""
    if reserves is ...:
        reserves = modelled_reserves(cuda_graph_max_bs=1)
    return open_ledger(
        device_total_bytes=baseline + 1 * _GIB,
        baseline_free=baseline,
        memory_ratio=ratio,
        weights_bytes=weights,
        reserves=reserves,
        **kw,
    )


# ---- the ceiling policy -----------------------------------------------------------------


def test_no_reserve_reproduces_the_pre_ledger_formula_byte_for_byte():
    # The base computed int(ratio * baseline) - weights - fixed. With nothing modelled the
    # ledger must be an identity, or landing this brick would silently resize every model.
    for ratio in (0.86, 0.9, 1.0):
        for baseline in (8 * _GIB, 16 * _GIB - 12345):
            assert net_cache_budget_bytes(ratio, baseline, 3 * _GIB, 512 * _MIB) == (
                int(ratio * baseline) - 3 * _GIB - 512 * _MIB
            )


def test_ratio_is_a_cap_and_the_reserve_is_a_floor():
    ledger = _ledger(baseline=16 * _GIB, ratio=0.9)
    # The gap (1 - 0.9) x 16 GiB = 1.6 GiB is bigger than the modelled reserve, so the cap
    # still binds and the plan is exactly what it was before the ledger existed.
    assert ledger.reserve_bytes < 1.6 * _GIB
    assert ledger.ceiling_bytes == int(0.9 * 16 * _GIB)
    # Push the cap to the sky and the floor takes over: the account, not the ratio, decides.
    ledger.memory_ratio = 1.0
    assert ledger.ceiling_bytes == 16 * _GIB - ledger.reserve_bytes


def test_reserve_only_counts_memory_that_must_stay_empty():
    ledger = _ledger()
    ledger.charge("cache:kv", 2 * _GIB, Kind.PERSISTENT)
    ledger.charge("cache:expert", 1 * _GIB, Kind.PERSISTENT)
    assert ledger.reserve_bytes == (
        TRITON_AUTOTUNE_ARENA + graph_capture_peak_bytes(1) + FRAGMENTATION_RESERVE
    )
    # The semi-persistent lines are HELD bytes, not headroom: they belong in the committed side
    # of the account, or the plan would subtract them twice.
    assert ledger.engine_overhead_bytes() == BACKEND_WORKSPACE + graph_pool_bytes(1)
    # A negotiable line never moves the ceiling, or sizing KV would shrink KV's own budget.
    before = ledger.ceiling_bytes
    ledger.charge("cache:kv", 4 * _GIB, Kind.PERSISTENT)
    assert ledger.ceiling_bytes == before


def test_pool_budget_is_the_ceiling_minus_everything_non_negotiable():
    ledger = _ledger(baseline=16 * _GIB, ratio=1.0, weights=8 * _GIB)
    ledger.charge("cache:gdn-state", 1 * _GIB, Kind.PERSISTENT)
    # The semi-persistent overhead is committed too: the page table and the graph pool are
    # memory the engine holds, so the split sees one fewer GiB than it would have an hour ago.
    assert ledger.pool_budget_bytes() == (
        ledger.ceiling_bytes - 8 * _GIB - 1 * _GIB - ledger.engine_overhead_bytes()
    )
    # A pool family's own fixed tier is handed in, because the account cannot see a pool that
    # has not been built yet and the split must still fund it.
    assert ledger.pool_budget_bytes(512 * _MIB) == ledger.pool_budget_bytes() - 512 * _MIB
    # Charging the negotiable consumers must not change what they were offered to split.
    offered = ledger.pool_budget_bytes()
    ledger.charge("cache:kv", 3 * _GIB, Kind.PERSISTENT)
    ledger.charge("cache:expert", offered - 3 * _GIB, Kind.PERSISTENT)
    assert ledger.pool_budget_bytes() == offered
    # headroom reads the ceiling; uncommitted reads the card. They differ by exactly the
    # reserve, and conflating them is what made --memory-ratio look like a safety policy.
    assert ledger.headroom_bytes() == ledger.ceiling_bytes - ledger.held_bytes()
    assert ledger.uncommitted_bytes() == (
        16 * _GIB - ledger.held_bytes() - ledger.reserve_bytes
    )
    assert ledger.headroom_bytes() - ledger.uncommitted_bytes() == (
        ledger.ceiling_bytes - ledger.baseline_free + ledger.reserve_bytes
    )


def test_pool_budget_floors_at_zero_instead_of_promising_negative_memory():
    ledger = _ledger(baseline=4 * _GIB, ratio=1.0, weights=4 * _GIB)
    assert ledger.pool_budget_bytes() == 0


def test_recharge_moves_the_account_instead_of_double_counting():
    ledger = _ledger()
    assert ledger.charge("cache:kv", 2 * _GIB, Kind.PERSISTENT) == 2 * _GIB
    assert ledger.charge("cache:kv", 5 * _GIB, Kind.PERSISTENT) == 3 * _GIB
    assert ledger.charge("cache:kv", 1 * _GIB, Kind.PERSISTENT) == -4 * _GIB
    assert ledger.release("cache:kv") == 1 * _GIB
    assert ledger.release("cache:kv") == 0
    assert ledger.bytes_of("cache:kv") == 0
    with pytest.raises(AssertionError, match="negative charge"):
        ledger.charge("cache:kv", -1, Kind.PERSISTENT)


# ---- the modelled consumers -------------------------------------------------------------


def test_graph_capture_shapes_follow_the_capture_helper():
    assert graph_capture_shapes(None) == 0
    assert graph_capture_shapes(0) == 0
    assert graph_capture_shapes(1) == 1
    assert graph_capture_shapes(3) == 2  # [1, 2]
    assert graph_capture_shapes(4) == 3  # [1, 2, 4]
    assert graph_capture_shapes(8) == 4  # + 8
    assert graph_capture_shapes(24) == 6  # 1, 2, 4, 8, 16, 24
    assert graph_capture_peak_bytes(1) == GRAPH_CAPTURE_PEAK
    assert graph_capture_peak_bytes(8) == (
        GRAPH_CAPTURE_PEAK + 3 * GRAPH_CAPTURE_EXTRA_SHAPE
    )


def test_gdn_prefill_peak_scales_with_the_chunk_and_the_state_geometry():
    group = _group()
    one = gdn_prefill_bytes(group, 8192, 2)
    # The dominant term is one [V, K] state per 64-token chunk, so doubling the chunk doubles
    # it; doubling head-dim width quadruples it. A plan that forgot this line OOM'd in
    # EXP-001b / EXP-003, which is why it is priced and not hand-waved.
    assert gdn_prefill_bytes(group, 16384, 2) == 2 * one
    assert gdn_prefill_bytes(_group(v_dim=256), 8192, 2) > one
    assert one > 400 * _MIB  # the ~451 MiB Flash-Next layer peak, not a rounding error
    assert gdn_prefill_bytes(group, 0, 2) == 0


def test_activation_peak_takes_the_worse_of_prefill_and_decode():
    prefill = activation_peak_bytes(2560, prefill_tokens=8192, batch=1, decode_tokens=1)
    decode = activation_peak_bytes(2560, prefill_tokens=0, batch=256, decode_tokens=4096)
    assert prefill == 8192 * 2560 * 2 * 6
    assert decode > prefill
    assert activation_peak_bytes(2560, prefill_tokens=8192, batch=256, decode_tokens=4096) == decode


def test_modelled_reserves_only_price_what_the_model_actually_has():
    names = lambda **kw: {e[0] for e in modelled_reserves(**kw)}
    plain = names(prefill_tokens=8192, hidden_size=2048)
    hybrid = names(linear_group=_group(), prefill_tokens=8192, hidden_size=2048)
    assert "transient:gdn-prefill" not in plain
    assert "transient:gdn-prefill" in hybrid
    assert "transient:mm-encoder" not in hybrid
    assert "transient:mm-encoder" in names(
        prefill_tokens=8192, hidden_size=2048, mm_encoder=True)
    # Every line is conditional except the named fragmentation reserve: an account with no
    # reserve at all is not an account, it is a promise the allocator cannot keep.
    assert names(prefill_tokens=0, hidden_size=0, cuda_graph_max_bs=None,
                 autotune=False, backend_workspace=False) == {"reserve:fragmentation"}
    assert "reserve:fragmentation" in plain


def test_tensor_bytes_measures_a_consumer_that_has_no_formula():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("measured line items need a device")
    holder = SimpleNamespace(
        bank_caches={"gate_up": torch.zeros(64, 128, device="cuda", dtype=torch.uint8)},
        host=torch.zeros(64, 128),  # pageable host memory is not a VRAM line
        scalars=torch.zeros(8, device="cuda", dtype=torch.int64),
    )
    assert tensor_bytes(holder) == 64 * 128 + 8 * 8


# ---- the planner consumes the reserve ---------------------------------------------------


def test_decide_splits_the_budget_and_prices_the_context_targets():
    ledger = _ledger(baseline=16 * _GIB, ratio=1.0, weights=8 * _GIB)
    ledger.charge("cache:gdn-state", 1 * _GIB, Kind.PERSISTENT)
    plan = ledger.decide(
        cache_per_page=4 * _MIB, page_tokens=64, per_expert_bytes=2 * _MIB,
        num_experts=64, total_experts=64, prefill_overlap=False, kv_reserve_tokens=4096,
        fixed_cache_bytes=256 * _MIB, contexts=(16384, 131072, 1048576),
    )
    assert plan.pool_budget_bytes == ledger.pool_budget_bytes(256 * _MIB)
    assert plan.expert_bytes == plan.moe_cache_size * 2 * _MIB
    assert plan.kv_budget_bytes == plan.pool_budget_bytes - plan.expert_bytes
    assert plan.usable_tokens == plan.num_pages * plan.page_tokens
    assert required_bytes(plan.moe_cache_size, plan.num_pages, 2 * _MIB, 4 * _MIB) <= (
        plan.pool_budget_bytes)
    rows = {c.tokens: c for c in plan.contexts}
    # A context has to fit in what the split LEFT, not in the whole pool budget: the expert
    # cache is not optional on a MoE model, it is where decode throughput lives. 16K costs
    # (256 + 1 dummy) x 4 MiB; 128K costs 8 GiB, which is more than the account has.
    assert rows[16384].kv_bytes == 257 * 4 * _MIB
    assert rows[16384].fits and rows[16384].kv_bytes <= plan.kv_budget_bytes
    assert not rows[131072].fits
    assert not rows[1048576].fits
    assert rows[1048576].shortfall_bytes == (
        rows[1048576].kv_bytes - plan.kv_budget_bytes)
    assert rows[131072].pages == 131072 // 64  # page_tokens 64, exact division
    text = plan.report()
    assert "128K" in text and "1M" in text and "expert slots" in text
    assert "left for KV" in text
    # kv_room_bytes is the same quantity read off a charged account instead of a plan.
    ledger.charge("cache:expert", plan.expert_bytes, Kind.PERSISTENT)
    assert ledger.kv_room_bytes(256 * _MIB) == plan.kv_budget_bytes


def test_a_compressed_kv_format_is_the_only_thing_that_makes_long_context_fit():
    # The economic gate for Phase 11 and Phase 3, stated as arithmetic: at 8 MiB a 4-bit page
    # 128K needs more than the card has, and the same context at 2 MiB fits -- without
    # tiering, without paging, and without touching the expert cache. This is why the plan
    # prints the rows: the answer is known before the money is spent.
    ledger = _ledger(baseline=16 * _GIB, ratio=1.0, weights=9 * _GIB)
    kwargs = dict(page_tokens=64, per_expert_bytes=2 * _MIB, num_experts=64,
                  total_experts=64, prefill_overlap=False, kv_reserve_tokens=0,
                  contexts=(131072,))
    bf16 = ledger.decide(cache_per_page=8 * _MIB, **kwargs)
    compressed = ledger.decide(cache_per_page=2 * _MIB, **kwargs)
    assert not bf16.contexts[0].fits
    assert compressed.contexts[0].fits
    assert compressed.moe_cache_size == bf16.moe_cache_size  # the KV half changed, not experts


def test_context_demand_prices_the_trade_the_other_way_round():
    # Feasibility says 128K does not fit what the MoE-first plan left; the demand row says what
    # it would cost to buy it instead -- the expert cache that survives funding the context.
    ledger = _ledger(baseline=16 * _GIB, ratio=1.0, weights=8 * _GIB)
    ledger.charge("cache:gdn-state", 1 * _GIB, Kind.PERSISTENT)
    plan = ledger.decide(
        cache_per_page=1 * _MIB, page_tokens=64, per_expert_bytes=2 * _MIB,
        num_experts=64, total_experts=2560, prefill_overlap=False, kv_reserve_tokens=4096,
        contexts=(131072,),
    )
    assert not plan.contexts[0].fits  # the MoE-first split starved KV
    demand = plan.demands[0]
    assert demand.tokens == 131072 and demand.pages == 2048
    assert demand.kv_bytes == 2049 * _MIB
    assert demand.feasible and demand.expert_slots > 0
    assert demand.expert_bytes + demand.kv_bytes <= plan.pool_budget_bytes
    # The MoE-first plan spent 5 GiB on experts; funding the context first has to cost slots.
    assert demand.expert_bytes < plan.expert_bytes
    assert "bought instead" in plan.report()


def test_context_demand_says_no_when_no_expert_cache_survives():
    # 1M of KV leaves nothing for experts: infeasible, not "0 slots and hope it is fast".
    demand = context_demand(
        pool_budget_bytes=8 * _GIB, cache_per_page=4 * _MIB, page_tokens=64,
        per_expert_bytes=2 * _MIB, tokens=1048576, expert_floor=64,
    )
    assert not demand.feasible and demand.expert_slots == 0
    assert "not fundable" in demand.describe()


def test_auto_plan_shrinks_only_once_the_reserve_beats_the_ratio_cap():
    # Modelled on a 16 GiB card holding an 8 GiB dense slice: 4 MiB per KV page, 2 MiB per
    # expert slot, and a 8192-token KV floor that is 128 pages at page_size 64.
    kwargs = dict(
        baseline_free=16 * _GIB, weights_bytes=8 * _GIB, cache_per_page=4 * _MIB,
        fixed_cache_size=0, per_expert_bytes=2 * _MIB, num_experts=64, total_experts=2560,
        prefill_overlap=True, kv_reserve_tokens=4096, page_size=64,
    )
    for ratio in (0.9, 0.95):
        # The cap binds at every shipped ratio, so the reserve must not move the plan.
        cap = resolve_moe_cache_auto(memory_ratio=ratio, **kwargs)
        with_reserve = resolve_moe_cache_auto(
            memory_ratio=ratio, reserve_bytes=TRITON_AUTOTUNE_ARENA, **kwargs)
        assert cap == with_reserve
    uncapped = resolve_moe_cache_auto(memory_ratio=1.0, **kwargs)
    reserve = 40 * 4 * _MIB  # 40 pages' worth of modelled peak
    capped = resolve_moe_cache_auto(memory_ratio=1.0, reserve_bytes=reserve, **kwargs)
    assert capped[1] < uncapped[1]
    # And whatever it returns, the plan has to be fundable at the shrunken ceiling.
    budget = net_cache_budget_bytes(1.0, 16 * _GIB, 8 * _GIB, 0, reserve)
    assert required_bytes(capped[0], capped[1], 2 * _MIB, 4 * _MIB) <= budget
    assert required_bytes(capped[0], capped[1] + 40, 2 * _MIB, 4 * _MIB) > budget


def test_required_bytes_still_prices_the_pool_dummy_page():
    # Phase-0 brick: the pool allocates num_pages + 1 and the plan pays for all of them.
    assert pool_pages(100) == 101
    assert required_bytes(0, 100, 0, 10) == 1010


def test_ledger_lines_add_up_to_the_physical_card():
    # The account's whole point is held + reserve + uncommitted == baseline. An aliased view
    # billed twice, or a kind moved between the two sides, breaks exactly this identity.
    ledger = _ledger(baseline=16 * _GIB, ratio=1.0, weights=6 * _GIB)
    ledger.charge("cache:expert", 4 * _GIB, Kind.PERSISTENT)
    ledger.charge("cache:kv", 2 * _GIB, Kind.PERSISTENT)
    overhead = ledger.engine_overhead_bytes()
    assert ledger.held_bytes() == 6 * _GIB + 4 * _GIB + 2 * _GIB + overhead
    assert ledger.held_bytes() + ledger.reserve_bytes + ledger.uncommitted_bytes() == 16 * _GIB
    # The overhead the planner folds into its fixed term is a subset of what is held, and the
    # GDN state pool is NOT in it (the engine subtracts that one itself, from the pool family).
    assert overhead > 0 and overhead < ledger.held_bytes()
    assert "cache:gdn-state" not in ledger.charges or overhead + 1 * _GIB <= ledger.held_bytes()


def test_calibration_reading_is_never_a_consumer():
    # The measured line reports what the allocator holds. If it entered a planner total, the
    # engine would bill itself for its own bookkeeping and every later budget would collapse.
    ledger = _ledger(ratio=1.0)
    before = (ledger.ceiling_bytes, ledger.pool_budget_bytes(), ledger.headroom_bytes(),
              ledger.reserve_bytes)
    ledger.charge("measured:allocator-held", 14 * _GIB, Kind.MEASURED, "torch holds this")
    assert (ledger.ceiling_bytes, ledger.pool_budget_bytes(), ledger.headroom_bytes(),
            ledger.reserve_bytes) == before
    assert ledger.held_bytes() == ledger.total((Kind.IMMUTABLE, Kind.PERSISTENT,
                                                Kind.SEMI_PERSISTENT))
    assert "measured:allocator-held" in ledger.report()


def test_report_prints_every_line_and_the_two_tightening_bounds():
    ledger = _ledger(ratio=1.0)
    ledger.charge("cache:kv", 3 * _GIB, Kind.PERSISTENT, note="usable pages + dummy")
    ledger.charge("cache:expert", 2 * _GIB, Kind.PERSISTENT, note="slots")
    text = ledger.report()
    for name in ("weights:model", "cache:kv", "cache:expert", "transient:autotune",
                 "graph:capture-peak", "workspace:attention", "reserve:fragmentation"):
        assert name in text, name
    assert "committed" in text and "pool budget" in text and "headroom" in text
    assert "usable pages + dummy" in text


def test_ledger_rejects_a_plan_that_exceeds_its_own_ceiling():
    ledger = VramLedger(
        device_total_bytes=8 * _GIB, baseline_free=8 * _GIB, memory_ratio=1.0,
    )
    ledger.charge("weights:model", 7 * _GIB, Kind.IMMUTABLE)
    ledger.charge("reserve:fragmentation", FRAGMENTATION_RESERVE, Kind.RESERVE)
    assert ledger.pool_budget_bytes() == 8 * _GIB - FRAGMENTATION_RESERVE - 7 * _GIB
    ledger.charge("cache:kv", 1 * _GIB, Kind.PERSISTENT)
    assert ledger.headroom_bytes() < 0  # overspent: the calibration warning catches this live
