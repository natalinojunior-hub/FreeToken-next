"""A zero temperature or top_k 1 is greedy whatever top_p a model default supplies: the
sampler takes argmax either way, and speculative decoding only runs for greedy requests."""

from freetoken.core import SamplingParams


def test_zero_temperature_is_greedy_under_model_default_top_p():
    assert SamplingParams(temperature=0.0, top_p=0.95).is_greedy


def test_top_k_one_is_greedy_under_top_p():
    assert SamplingParams(temperature=1.0, top_k=1, top_p=0.95).is_greedy


def test_sampling_request_is_not_greedy():
    assert not SamplingParams(temperature=1.0, top_k=20, top_p=0.95).is_greedy
