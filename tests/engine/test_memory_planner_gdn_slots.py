"""Regression for the GDN fixed-overhead slot-count mismatch.

``phase_b_build_static_model`` must budget the GDN state pool at the same slot
count ``phase_h_construct_final_pools`` actually builds (``_linear_pool_num_slots``,
which includes the cross-request snapshot cache). Budgeting the smaller
non-evictable floor (``_linear_pool_min_slots``) under-reserves the real pool by
the snapshot-cache slots, so a plan the solver approves as fitting could still
OOM when the final GDN pool is constructed.
"""

from __future__ import annotations

import inspect

from freetoken.engine import memory_planner
from freetoken.kvcache.linear_state_pool import _linear_pool_min_slots, _linear_pool_num_slots


class _FakeModelConfig:
    def linear_attention_group(self):
        return object()  # non-None is all _linear_pool_num_slots/min_slots need


class _FakeConfig:
    max_running_req = 4
    cache_type = "hybrid_radix"
    linear_state_cache_ratio = 2.0
    model_config = _FakeModelConfig()


def test_num_slots_exceeds_min_slots_for_hybrid_radix():
    """Sanity: the two slot counts genuinely differ, so budgeting the wrong one matters."""
    config = _FakeConfig()
    assert _linear_pool_num_slots(config) > _linear_pool_min_slots(config)


def test_static_model_budgets_gdn_at_num_slots_not_min_slots():
    """Regression guard: phase_b_build_static_model must size gdn_slots from
    _linear_pool_num_slots (matching the pool phase_h actually constructs), not
    _linear_pool_min_slots (a smaller floor that under-reserves the real pool)."""
    src = inspect.getsource(memory_planner.MemoryPlanner.phase_b_build_static_model)
    assert "_linear_pool_num_slots(self.config)" in src, (
        "gdn_slots in phase_b_build_static_model must use _linear_pool_num_slots to match "
        "the pool phase_h_construct_final_pools actually builds"
    )
    assert "gdn_slots = _linear_pool_min_slots(self.config)" not in src
