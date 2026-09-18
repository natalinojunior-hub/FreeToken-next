"""Pure MTP acceptance and rollback contracts."""

import torch

from freetoken.engine.spec import (
    SpecResult,
    accept_drafts,
    ngram_context_after,
    pack_spec_message,
    pages_to_free,
    rebuild_conv_state,
    spec_message_len,
    spec_rollback_lengths,
    unpack_spec_message,
)


def test_accept_drafts_keeps_matching_prefix_and_correction():
    assert accept_drafts([5, 6, 7, 8], [5, 6, 7]) == [5, 6, 7, 8]
    assert accept_drafts([5, 9, 7, 8], [5, 6, 7]) == [5, 9]
    assert accept_drafts([1], []) == [1]


def test_spec_message_round_trip_is_fixed_size():
    result = SpecResult(accepted=[11, 12], drafts=[13, 14, 15])
    message = pack_spec_message(result, 3)
    assert message.numel() == spec_message_len(3)
    assert unpack_spec_message(message, 3) == result


def test_rollback_helpers_keep_only_accepted_state():
    assert pages_to_free(keep_len=100, alloc_len=130, page_size=64) == (2, 3)
    previous = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    inputs = torch.tensor([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]])
    expected = torch.tensor([[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]])
    assert torch.equal(rebuild_conv_state(previous, inputs, 2), expected)
    assert ngram_context_after([1, 2, 3], [4, 5], 1, 4, 0) == [0, 1, 2, 3]


def test_spec_rollback_lengths():
    # start_pos=100, committed=2 (pos 100, 101 committed) -> last cached is 101, device is 102
    assert spec_rollback_lengths(100, 2) == (101, 102)
    # start_pos=64, committed=1 (pos 64 committed) -> last cached is 64, device is 65
    assert spec_rollback_lengths(64, 1) == (64, 65)
