"""Dense DSpark greedy proposal traced into the TPU draft forward."""

import torch


def sample_draft_tokens(
    self, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, None]:
    """Greedy markov draft chain [bs, gamma], traced into the TPU draft graph."""
    # Anchor drafts forward [r, m x (gamma - 1)] and read every row; the rest
    # forward [r, m x gamma] and drop the anchor row.
    width = self.gamma if self.sample_from_anchor else self.gamma + 1
    hidden_3d = hidden_states.view(-1, width, hidden_states.shape[-1])
    draft_hidden = hidden_3d if self.sample_from_anchor else hidden_3d[:, 1:]
    bs = hidden_3d.shape[0]
    base_logits, _ = self.compute_base_logits(draft_hidden.reshape(bs * self.gamma, -1))
    draft_tokens, _ = self.markov_head.sample_block(
        base_logits.view(bs, self.gamma, -1),
        first_prev_tokens=input_ids.view(bs, width)[:, 0],
        hidden_states=draft_hidden,
        sampler=lambda step_logits, step_idx: torch.argmax(step_logits, dim=-1),
        collect_corrected=False,
    )
    return draft_tokens, None
