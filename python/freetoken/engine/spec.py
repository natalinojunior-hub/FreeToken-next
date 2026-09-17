"""Pure host-side arithmetic for native MTP speculative decoding.

The runtime owns model execution and cache mutation; these helpers define the deterministic
acceptance and rollback contracts that those paths must share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class SpecResult:
    """Committed tokens from one verify window and drafts for the next window."""

    accepted: list[int]
    drafts: list[int] = field(default_factory=list)


def accept_drafts(sampled: Sequence[int], drafts: Sequence[int]) -> list[int]:
    """Keep the longest matching draft prefix and the target correction token."""
    accepted = 0
    while accepted < len(drafts) and accepted < len(sampled) - 1:
        if sampled[accepted] != drafts[accepted]:
            break
        accepted += 1
    return [int(token) for token in sampled[: accepted + 1]]


def pack_spec_message(result: SpecResult, k: int) -> torch.Tensor:
    """Pack a fixed-size rank message: count, accepted tokens, then next drafts."""
    count = len(result.accepted)
    assert 1 <= count <= k + 1, (count, k)
    accepted = list(result.accepted) + [-1] * (k + 1 - count)
    drafts = list(result.drafts)[:k]
    drafts += [-1] * (k - len(drafts))
    return torch.tensor([count, *accepted, *drafts], dtype=torch.int32)


def unpack_spec_message(message: torch.Tensor, k: int) -> SpecResult:
    values = message.tolist()
    count = int(values[0])
    assert 1 <= count <= k + 1, (count, k)
    accepted = [int(token) for token in values[1 : 1 + count]]
    drafts = [int(token) for token in values[2 + k : 2 + 2 * k] if token >= 0]
    return SpecResult(accepted=accepted, drafts=drafts)


def spec_message_len(k: int) -> int:
    return 2 * k + 2


def pages_to_free(keep_len: int, alloc_len: int, page_size: int) -> tuple[int, int]:
    """Return the half-open page range made unused by a speculative rollback."""
    first = -(-keep_len // page_size)
    last = -(-alloc_len // page_size)
    return first, max(first, last)


def rebuild_conv_state(prev_state: torch.Tensor, conv_in: torch.Tensor, accepted: int) -> torch.Tensor:
    """Rebuild a linear-attention convolution state after accepting ``accepted`` rows."""
    width = prev_state.shape[-1]
    combined = torch.cat(
        [prev_state, conv_in[:accepted].to(prev_state.dtype).transpose(0, 1)], dim=-1
    )
    return combined[..., -width:].contiguous()


def ngram_context_after(
    host_ids: Sequence[int], drafts: Sequence[int], accepted: int, ctx_len: int, boundary: int
) -> list[int]:
    """Return the PLE context after committing part of a speculative window."""
    sequence = list(host_ids) + list(drafts)
    end = len(host_ids) - 1 + accepted
    window = sequence[max(0, end - ctx_len) : end]
    return [boundary] * (ctx_len - len(window)) + [int(token) for token in window]


__all__ = [
    "SpecResult",
    "accept_drafts",
    "pack_spec_message",
    "unpack_spec_message",
    "spec_message_len",
    "pages_to_free",
    "rebuild_conv_state",
    "ngram_context_after",
]
