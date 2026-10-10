"""Power-of-two row buckets for eager TPU programs.

An eager op compiles once per argument shape, so a batch draining one request
at a time would compile at every size; bucketing rows caps that at log2(bs).
"""

from __future__ import annotations

import torch


def row_bucket(rows: int) -> int:
    """Smallest power of two >= rows (1 for an empty batch)."""
    return 1 << max(rows - 1, 0).bit_length()


def pad_rows(x: torch.Tensor, rows: int, fill: float = 0.0) -> torch.Tensor:
    """``x`` with its leading dim grown to ``rows``; the new rows hold ``fill``."""
    pad = rows - x.shape[0]
    if pad <= 0:
        return x
    return torch.cat([x, x.new_full((pad, *x.shape[1:]), fill)])
