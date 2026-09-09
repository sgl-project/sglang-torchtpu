"""TPU helpers for EAGLE verify trees."""

import torch


def build_topk1_verify_chain_torch(
    bonus_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Build the fixed top-k=1 EAGLE verify tensors with pure torch ops.

    The fused TPU draft loop calls this before returning so tree construction
    stays in the same compiled program. The ordinary path also uses it as its
    torch-native fallback.

    Args:
        bonus_tokens: Bonus token IDs with shape ``[bs]``.
        draft_tokens: Draft token IDs with shape ``[bs, num_draft_tokens]``.
        seq_lens: Prefix lengths with shape ``[bs]``.

    Returns:
        An empty tree mask with shape ``[0]``, followed by positions, retrieve
        indices, next-token indices, next-sibling indices, and verification
        tokens, each with shape ``[bs, num_draft_tokens + 1]``.
    """
    bs = seq_lens.shape[0]
    num_verify_tokens = draft_tokens.shape[1] + 1
    device = seq_lens.device

    verify_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1)
    token_offsets = torch.arange(num_verify_tokens, device=device, dtype=torch.long)
    positions = seq_lens[:, None] + token_offsets[None, :]
    retrieve_index = torch.arange(
        bs * num_verify_tokens, device=device, dtype=torch.long
    ).view(bs, num_verify_tokens)
    next_token_row = torch.cat(
        (
            torch.arange(1, num_verify_tokens, device=device, dtype=torch.long),
            torch.full((1,), -1, device=device, dtype=torch.long),
        )
    )
    retrieve_next_token = next_token_row.expand(bs, -1).contiguous()
    retrieve_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    tree_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        verify_tokens,
    )


def build_topk1_verify_tree_tpu(
    *, bonus_tokens: torch.Tensor, draft_tokens: torch.Tensor, seq_lens: torch.Tensor
):
    tree_mask, positions, retrieve_index, next_token, next_sibling, verify_tokens = (
        build_topk1_verify_chain_torch(bonus_tokens, draft_tokens, seq_lens)
    )
    return (
        tree_mask,
        positions.flatten(),
        retrieve_index,
        next_token,
        next_sibling,
        verify_tokens.flatten(),
    )
