"""TPU-specific forward-batch preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def prepare_decode_as_extend(batch: ForwardBatch, batch_size: int) -> None:
    """Continue live decode rows while a DP peer prefills, preserving cached state."""
    batch.extend_num_tokens = batch_size
    batch.extend_seq_lens = torch.full_like(batch.seq_lens, 1, dtype=torch.int32)
    batch.extend_prefix_lens = (batch.seq_lens - 1).to(torch.int32)
    batch.extend_start_loc = torch.arange(
        batch_size, dtype=torch.int32, device=batch.seq_lens.device
    )
    # Reuse host lengths to avoid synchronizing the TPU for metadata.
    if batch.seq_lens_cpu is not None:
        batch.extend_prefix_lens_cpu = (batch.seq_lens_cpu - 1).tolist()
        batch.extend_seq_lens_cpu = [1] * batch_size
    else:
        batch.extend_prefix_lens_cpu = batch.extend_prefix_lens.cpu().tolist()
        batch.extend_seq_lens_cpu = batch.extend_seq_lens.cpu().tolist()
    batch.extend_logprob_start_lens_cpu = batch.extend_prefix_lens_cpu
