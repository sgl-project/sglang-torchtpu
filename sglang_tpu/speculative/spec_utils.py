"""Host-side helpers shared by every TpuSpecWorker algorithm."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


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


def padded_h2d(rows_cpu: torch.Tensor, rows: int, device: str) -> torch.Tensor:
    """Zero-pad host rows to rows (a batch bucket) in pinned memory, then upload."""
    padded = torch.zeros(
        (rows, *rows_cpu.shape[1:]),
        dtype=rows_cpu.dtype,
        pin_memory=is_pin_memory_available(device),
    )
    padded[: rows_cpu.shape[0]] = rows_cpu
    return padded.to(device, non_blocking=True)


@contextlib.contextmanager
def overridden_fields(
    batch: ScheduleBatch, **overrides: Any
) -> Iterator[ScheduleBatch]:
    """batch with overrides set, restored on exit."""
    live = {name: getattr(batch, name) for name in overrides}
    for name, value in overrides.items():
        setattr(batch, name, value)
    try:
        yield batch
    finally:
        for name, value in live.items():
            setattr(batch, name, value)
