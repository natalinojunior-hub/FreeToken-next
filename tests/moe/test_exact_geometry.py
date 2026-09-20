import pytest
import torch

from freetoken.moe.expert_banks import ExpertBanks, make_pool_key, validate_pool_key
from freetoken.moe.offload_cache import ExpertBank, OffloadMoeCache


def test_pool_key_and_validation():
    key = make_pool_key(3, "gate_up", "Q3_K")
    assert key == (3, "gate_up", "Q3_K")
    assert validate_pool_key(key) == (3, "gate_up", "Q3_K")

    # integer ggml type mapping
    key2 = make_pool_key(0, "down", 12)  # GGML_Q4_K = 12
    assert key2 == (0, "down", "Q4_K")

    # invalid key raises KeyError with expected message (MOE-003)
    with pytest.raises(KeyError, match="expert_geometry / geometria inválida"):
        validate_pool_key((3, "gate_up"))
    with pytest.raises(KeyError, match="MoE pool key mismatch"):
        validate_pool_key("gate_up")


def test_expert_banks_geometry_indexing():
    num_layers = 2
    num_experts = 4
    # Layer 0: 64 cols, Layer 1: 32 cols (mixed geometry)
    sources = {
        "gate_up": [
            torch.zeros(num_experts, 8, 64, dtype=torch.uint8),
            torch.ones(num_experts, 8, 32, dtype=torch.uint8),
        ],
        "down": [
            torch.zeros(num_experts, 4, 32, dtype=torch.uint8),
            torch.ones(num_experts, 4, 16, dtype=torch.uint8),
        ],
    }
    geometry = {
        (0, "gate_up", "Q4_K"): ((num_experts, 8, 64), torch.uint8),
        (1, "gate_up", "Q3_K"): ((num_experts, 8, 32), torch.uint8),
        (0, "down", "Q6_K"): ((num_experts, 4, 32), torch.uint8),
        (1, "down", "IQ4_XS"): ((num_experts, 4, 16), torch.uint8),
    }
    banks = ExpertBanks(
        quant_format="gguf",
        sources=sources,
        gguf_expert_types=[(12, 14), (11, 21)],
        expert_geometry=geometry,
    )
    # Exact geometry access
    t0 = banks[(0, "gate_up", "Q4_K")]
    assert t0.shape == (num_experts, 8, 64)
    t1 = banks[(1, "gate_up", "Q3_K")]
    assert t1.shape == (num_experts, 8, 32)
    assert (t1 == 1).all()

    # Legacy access
    assert len(banks["gate_up"]) == 2

    # KeyError on unregistered key
    with pytest.raises(KeyError, match="expert_geometry / geometria inválida"):
        _ = banks[(0, "gate_up", "UNKNOWN")]


def test_offload_cache_exact_geometry_views():
    num_layers = 2
    num_experts = 4
    cache_size = 8
    device = torch.device("cpu")

    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=device,
        quant_format="gguf",
        gguf_expert_types=[(12, 14), (11, 21)],  # Q4_K/Q6_K on L0, Q3_K/IQ4_XS on L1
    )

    sources = {
        "gate_up": [
            torch.zeros(num_experts, 8, 64, dtype=torch.uint8),
            torch.ones(num_experts, 8, 32, dtype=torch.uint8),
        ],
        "down": [
            torch.zeros(num_experts, 4, 32, dtype=torch.uint8),
            torch.ones(num_experts, 4, 16, dtype=torch.uint8),
        ],
    }
    cache.set_bank_sources(sources)

    # Verify expert_geometry populated
    assert (0, "gate_up", "Q4_K") in cache.expert_geometry
    assert (1, "gate_up", "Q3_K") in cache.expert_geometry

    # get_expert / put_expert
    exp = cache.get_expert(0, "gate_up", "Q4_K")
    assert isinstance(exp, ExpertBank)
    assert exp.shape == (num_experts, 8, 64)
    assert exp.quant_type == "Q4_K"

    with pytest.raises(KeyError, match="expert_geometry / geometria inválida"):
        cache.get_expert(0, "gate_up", "INVALID")

    # bank_views per layer
    views0 = cache.bank_views(layer_id=0)
    assert views0[0].shape == (cache_size, 8, 64)
    assert views0[1].shape == (cache_size, 4, 32)

    views1 = cache.bank_views(layer_id=1)
    assert views1[0].shape == (cache_size, 8, 32)
    assert views1[1].shape == (cache_size, 4, 16)


def test_spec_warmup_no_op_when_disabled():
    from unittest.mock import MagicMock
    from freetoken.scheduler.spec import SchedulerSpecMixin

    scheduler = SchedulerSpecMixin()
    scheduler.spec_mtp = 0
    scheduler.engine = MagicMock()
    req = MagicMock()
    # Safely no-op when spec_mtp <= 0
    scheduler.warmup_mtp_draft_kv(req)
