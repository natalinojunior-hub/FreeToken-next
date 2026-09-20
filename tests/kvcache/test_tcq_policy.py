import pytest
from freetoken.kvcache.tcq_policy import plan_vbr_schedule, TierSchedule


def test_vbr_schedule_uniform():
    sched = plan_vbr_schedule(num_layers=12, base_format="turbo4", vbr_policy="uniform")
    assert sched.num_layers == 12
    assert not sched.has_mixed_tiers
    for i in range(12):
        assert sched.get_tier(i, "k") == "turbo4"
        assert sched.get_tier(i, "v") == "turbo4"


def test_vbr_schedule_balanced():
    sched = plan_vbr_schedule(num_layers=10, base_format="turbo4", vbr_policy="balanced")
    assert sched.num_layers == 10
    assert sched.has_mixed_tiers
    # K is turbo4 across all layers
    for i in range(10):
        assert sched.get_tier(i, "k") == "turbo4"
    # V is turbo4 at ends (anchors), turbo3 in middle
    assert sched.get_tier(0, "v") == "turbo4"
    assert sched.get_tier(9, "v") == "turbo4"
    assert sched.get_tier(4, "v") == "turbo3"
    assert sched.get_tier(5, "v") == "turbo3"


def test_vbr_schedule_aggressive():
    sched = plan_vbr_schedule(num_layers=10, base_format="turbo4", vbr_policy="aggressive")
    assert sched.num_layers == 10
    for i in range(10):
        assert sched.get_tier(i, "v") == "turbo3"
    assert sched.get_tier(0, "k") == "turbo4"
    assert sched.get_tier(9, "k") == "turbo4"
    assert sched.get_tier(5, "k") == "turbo3"
