"""Dense DSpark greedy proposal traced into the TPU draft forward."""

from typing import Callable, Optional

import torch


def run_markov_block(
    markov_head,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: Callable[[torch.Tensor, int], torch.Tensor],
    collect_corrected: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Draft a [bs, gamma] block: each step adds the head's bias for the token
    drafted before it to its base logits, samples, and carries the head's state.

    Returns the drafted tokens and, with ``collect_corrected``, the corrected
    [bs, gamma, vocab] step logits.
    """
    bs, gamma = base_logits.shape[:2]
    if gamma == 0:
        empty = torch.empty(bs, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits
    state = markov_head.init_state(hidden_states)
    prev_tokens = first_prev_tokens.long()
    tokens, corrected = [], []
    for step in range(gamma):
        bias, state = markov_head.step_bias(
            prev_tokens,
            hidden_states=None if hidden_states is None else hidden_states[:, step],
            state=state,
        )
        step_logits = base_logits[:, step] + bias
        prev_tokens = sampler(step_logits, step)
        tokens.append(prev_tokens)
        if collect_corrected:
            corrected.append(step_logits.unsqueeze(1))
    return (
        torch.stack(tokens, dim=1),
        torch.cat(corrected, dim=1) if collect_corrected else None,
    )


def sample_draft_tokens(
    self, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Greedy markov draft chain [bs, gamma], traced into the TPU draft graph."""
    # Anchor drafts forward [r, m x (gamma - 1)] and read every row; the rest
    # forward [r, m x gamma] and drop the anchor row.
    width = self.gamma if self.sample_from_anchor else self.gamma + 1
    hidden_3d = hidden_states.view(-1, width, hidden_states.shape[-1])
    draft_hidden = hidden_3d if self.sample_from_anchor else hidden_3d[:, 1:]
    bs = hidden_3d.shape[0]
    base_logits, _ = self.compute_base_logits(draft_hidden.reshape(bs * self.gamma, -1))
    draft_tokens, _ = run_markov_block(
        self.markov_head,
        base_logits.view(bs, self.gamma, -1),
        first_prev_tokens=input_ids.view(bs, width)[:, 0],
        hidden_states=draft_hidden,
        sampler=lambda step_logits, step_idx: torch.argmax(step_logits, dim=-1),
        collect_corrected=False,
    )
    return draft_tokens
