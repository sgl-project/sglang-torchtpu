"""Host-side helpers shared by every TpuSpecWorker algorithm."""

from __future__ import annotations

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils.common import is_pin_memory_available


def apply_verify_logits_adjustments(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    num_verify_rows: int,
) -> None:
    """Apply per-request adjustments to [bs * num_verify_rows, V] request-major logits in place."""
    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            next_token_logits, sampling_info, num_tokens_in_batch=num_verify_rows
        )
    # Penalties come from the committed prefix only, same relaxation as EAGLE verify.
    # Overlap hands the forward precomputed acc_* penalties and no orchestrator.
    logits_3d = next_token_logits.view(-1, num_verify_rows, next_token_logits.shape[-1])
    orchestrator = sampling_info.penalizer_orchestrator
    if orchestrator is not None and orchestrator.is_required:
        orchestrator.apply(next_token_logits, repeat=num_verify_rows)
    else:
        if sampling_info.acc_additive_penalties is not None:
            logits_3d.add_(sampling_info.acc_additive_penalties[:, None, :])
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                logits_3d, sampling_info.acc_scaling_penalties[:, None, :]
            )
    if sampling_info.logit_bias is not None:
        logits_3d.add_(sampling_info.logit_bias[:, None, :])


def prompt_positions(*, prefix_lens: list[int], extend_lens: list[int]) -> torch.Tensor:
    """CPU [sum(extend_lens)] positions prefix_len .. prefix_len + extend_len - 1 per request."""
    assert len(prefix_lens) == len(extend_lens), (len(prefix_lens), len(extend_lens))
    extend = torch.tensor(extend_lens, dtype=torch.int64)
    starts = torch.repeat_interleave(
        torch.tensor(prefix_lens, dtype=torch.int64), extend
    )
    row_starts = torch.repeat_interleave(torch.cumsum(extend, 0) - extend, extend)
    return starts + torch.arange(int(extend.sum())) - row_starts


def pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero-pad dim 0 up to rows (a compile bucket); identity when already there."""
    pad = rows - tensor.shape[0]
    assert pad >= 0, (tensor.shape, rows)
    if pad == 0:
        return tensor
    return torch.nn.functional.pad(tensor, [0, 0] * (tensor.dim() - 1) + [0, pad])


def min_p_renorm_prob_tpu(probs: torch.Tensor, min_ps: torch.Tensor) -> torch.Tensor:
    """Sort-free min-p renormalization of [rows, V] probs, like top_p_renorm_prob_tpu."""
    from sglang_tpu.sampling_mask import tpu_filter_probs

    rows = probs.shape[0]
    kept = tpu_filter_probs(
        probs,
        torch.ones(rows, dtype=torch.int32, device=probs.device),
        torch.ones(rows, dtype=torch.float32, device=probs.device),
        min_ps,
        need_top_k=False,
        need_top_p=False,
        need_min_p=True,
    )
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def padded_h2d(rows_cpu: torch.Tensor, rows: int, device: str) -> torch.Tensor:
    """Zero-pad host rows to rows (a batch bucket) in pinned memory, then upload."""
    padded = torch.zeros(
        (rows, *rows_cpu.shape[1:]),
        dtype=rows_cpu.dtype,
        pin_memory=is_pin_memory_available(device),
    )
    padded[: rows_cpu.shape[0]] = rows_cpu
    return padded.to(device, non_blocking=True)
