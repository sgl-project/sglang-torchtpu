"""TPU helpers for speculative decoding."""

from __future__ import annotations

from functools import cache

import torch


@cache
def _get_topk1_tree_constants(
    device: torch.device,
    bs: int,
    num_verify_tokens: int,
) -> tuple[torch.Tensor, ...]:
    """Return cached shape-only inputs for a linear draft tree."""
    retrieve_index = torch.arange(
        bs * num_verify_tokens, dtype=torch.long, device=device
    ).view(bs, num_verify_tokens)
    retrieve_next_token = torch.full(
        (bs, num_verify_tokens), -1, dtype=torch.long, device=device
    )
    if num_verify_tokens > 1:
        retrieve_next_token[:, :-1] = torch.arange(
            1, num_verify_tokens, dtype=torch.long, device=device
        )
    retrieve_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, dtype=torch.long, device=device
    )
    return (
        torch.empty((0,), dtype=torch.bool, device=device),
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        torch.arange(num_verify_tokens, dtype=torch.long, device=device),
    )


def build_topk1_tree(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_verify_tokens: int,
) -> tuple[torch.Tensor, ...]:
    """Build TPU verify inputs for a linear draft tree."""
    (
        tree_mask,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        token_offsets,
    ) = _get_topk1_tree_constants(seq_lens.device, seq_lens.numel(), num_verify_tokens)
    tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
    positions = (seq_lens[:, None] + token_offsets[None, :]).flatten()
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        tokens,
    )
