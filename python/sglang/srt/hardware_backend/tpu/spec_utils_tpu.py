"""TPU speculative decoding helpers."""

import torch


def generate_simulated_accept_index_tpu(
    *,
    accept_index: torch.Tensor,
    predict: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: torch.Tensor | None,
    bs: int,
    spec_steps: int,
    simulate_acc_len: float,
    simulate_acc_method: str,
    simulate_acc_token_mode: str,
) -> torch.Tensor:
    """Simulate acceptance while keeping TPU bucket padding inactive.

    Padded roots are -1; adding offsets can write into another request's
    predictions. Use safe row-local indices and retain -1 counts so adding
    the bonus yields zero tokens for padded rows, including idle batches.
    """
    from sglang.srt.speculative.spec_utils import sample_simulated_acc_len

    assert simulate_acc_len > 0.0
    num_accept_tokens = sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    roots = accept_index[:, :1]
    valid_rows = roots >= 0
    # TPU bucket padding uses -1 roots; keep their scatter indices in their own
    # physical rows so they cannot overwrite another request's predictions.
    row_offsets = torch.arange(bs, device=accept_index.device).view(-1, 1)
    roots = torch.where(valid_rows, roots, row_offsets * candidates.shape[1])
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device=accept_index.device
    )
    sim_accept_index[:, :num_accept_tokens] = roots + torch.arange(
        num_accept_tokens, device=accept_index.device
    )
    num_correct_drafts.copy_(
        torch.where(valid_rows.flatten(), num_accept_tokens - 1, -1)
    )

    if simulate_acc_token_mode != "real-draft-token":
        predict.fill_(100)
        return torch.where(valid_rows, sim_accept_index, -1)

    if num_accept_tokens > 1:
        draft_indices = sim_accept_index[:, : num_accept_tokens - 1].long()
        predict[draft_indices] = torch.where(
            valid_rows,
            candidates[:, 1:num_accept_tokens].to(dtype=predict.dtype),
            predict[draft_indices],
        )
    bonus_indices = sim_accept_index[:, num_accept_tokens - 1].long()
    predict[bonus_indices] = torch.where(
        valid_rows.flatten(),
        target_predict[:, num_accept_tokens - 1].to(dtype=predict.dtype),
        predict[bonus_indices],
    )
    return torch.where(valid_rows, sim_accept_index, -1)
