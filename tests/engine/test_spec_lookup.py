import pytest
import torch

from freetoken.engine.spec import lookup_drafts


def test_lookup_drafts_finds_suffix_continuation() -> None:
    ids = torch.tensor([1, 2, 3, 4, 5, 2, 3])

    assert lookup_drafts(ids, 2) == [4, 5]


def test_lookup_drafts_returns_empty_without_match() -> None:
    assert lookup_drafts(torch.tensor([1, 2, 3]), 1) == []


@pytest.mark.parametrize("k", [0, -1])
def test_lookup_drafts_returns_empty_for_nonpositive_depth(k: int) -> None:
    assert lookup_drafts(torch.tensor([1, 2, 3]), k) == []


def test_lookup_drafts_respects_max_ngram() -> None:
    ids = torch.tensor([5, 6, 7, 8, 6, 7, 9, 5, 6, 7])

    assert lookup_drafts(ids, 1, max_ngram=3) == [8]
    assert lookup_drafts(ids, 1, max_ngram=2) == [9]


def test_lookup_drafts_uses_latest_matching_suffix() -> None:
    ids = torch.tensor([1, 2, 3, 9, 1, 2, 4, 1, 2])

    assert lookup_drafts(ids, 1) == [4]


def test_lookup_drafts_ignores_unaligned_byte_match() -> None:
    ids = torch.tensor([0x0100, 0x0302, 0x0504, 0x0201, 0x0403], dtype=torch.int16)

    assert lookup_drafts(ids, 1) == []


def test_lookup_drafts_allows_valid_periodic_overlap() -> None:
    ids = torch.tensor([1, 2, 1, 2, 1])

    assert lookup_drafts(ids, 2) == [2, 1]


@pytest.mark.parametrize(
    ("ids", "k"),
    [
        ([1, 2, 3, 4, 5, 2, 3], 2),
        ([1, 2, 1, 2, 1], 2),
        ([1, 2, 3, 9, 1, 2, 4, 1, 2], 1),
    ],
)
def test_lookup_drafts_returns_k_known_history_tokens(ids: list[int], k: int) -> None:
    history = torch.tensor(ids)
    drafts = lookup_drafts(history, k)

    assert len(drafts) == k
    assert any(history[i : i + k].tolist() == drafts for i in range(len(ids) - k + 1))
