"""TPU batch rows rebuilt from their host copies when requests join or leave."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def rows_on_host_tpu(
    batch: ScheduleBatch, other: Optional[ScheduleBatch] = None
) -> bool:
    """Whether merge/filter rebuilds the batch rows from their host copies.

    torch_tpu compiles one program per tensor shape, so joining or slicing the
    rows on the device compiles again for every batch composition. Speculative
    decoding can advance seq_lens on the device ahead of its host copy.
    """
    return (
        _is_tpu_device(batch.device)
        and batch.spec_algorithm.is_none()
        and batch.seq_lens_cpu is not None
        and (other is None or other.seq_lens_cpu is not None)
    )


def _is_tpu_device(device) -> bool:
    return str(device).split(":")[0] == "tpu"


def upload_rows_tpu(
    *,
    req_pool_indices_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upload req_pool_indices, seq_lens and orig_seq_lens, one copy each."""
    return (
        req_pool_indices_cpu.to(device, non_blocking=True),
        seq_lens_cpu.to(device, non_blocking=True),
        # No TPU consumer reads orig_seq_lens values; decode rows equal seq_lens.
        seq_lens_cpu.to(torch.int32).to(device, non_blocking=True),
    )
