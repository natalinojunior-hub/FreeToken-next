"""cache_prompt: false (llama.cpp-style wire field) forces a cold prefill: the scheduler must
skip prefix-cache matching for that request even when a hit is available, and the OpenAI-wire
usage block must surface how many prompt tokens actually came from the cache. CPU-only, no
model/GPU involved -- exercises CacheManager.match_req and the two wire-layer helpers directly."""

from __future__ import annotations

import torch

from freetoken.core import SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq
from freetoken.server.generation import resolve_sampling
from freetoken.server.openai_api import _usage


def _pend(ids, *, skip_prefix_cache=False):
    t = torch.tensor(ids, dtype=torch.int32)
    return PendingReq(uid=0, input_ids=t, sampling_params=SamplingParams(skip_prefix_cache=skip_prefix_cache))


def test_cache_prompt_false_skips_prefix_match():
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "radix")

    # Seed the tree with a cached prefix, as a finished request would leave behind.
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    indices = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, indices)

    # Normal request: reuses the cached prefix.
    hit = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert hit.cuda_handle.cached_len == 4

    # cache_prompt: false -> zero reused tokens even though the same prefix is cached.
    cold = cm.match_req(_pend([1, 2, 3, 4, 5], skip_prefix_cache=True))
    assert cold.cuda_handle.cached_len == 0

    # The prefix is still there afterward -- skipping the match never evicts/disables the cache.
    still_cached = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert still_cached.cuda_handle.cached_len == 4


def test_resolve_sampling_maps_cache_prompt_false():
    assert resolve_sampling(
        temperature=0, top_k=-1, top_p=1.0, max_tokens=1, ignore_eos=False, model_sampling={},
        cache_prompt=False,
    ).skip_prefix_cache
    assert not resolve_sampling(
        temperature=0, top_k=-1, top_p=1.0, max_tokens=1, ignore_eos=False, model_sampling={},
        cache_prompt=None,
    ).skip_prefix_cache
    assert not resolve_sampling(
        temperature=0, top_k=-1, top_p=1.0, max_tokens=1, ignore_eos=False, model_sampling={},
        cache_prompt=True,
    ).skip_prefix_cache


def test_usage_reports_cached_tokens():
    usage = _usage(prompt_tokens=10, completion_tokens=3, cached_tokens=4)
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4}
    # No hit (e.g. the cache_prompt: false path) -> no details block at all.
    assert "prompt_tokens_details" not in _usage(prompt_tokens=10, completion_tokens=3, cached_tokens=0)
