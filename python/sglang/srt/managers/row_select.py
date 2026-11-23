"""Row selection for the batch filters and merges: the seams a backend hooks
to bound the shapes of the device programs they run."""

from __future__ import annotations

import torch


def select_rows(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``tensor[index]`` for a 1-D row index on the tensor's device.

    Callers reach it as ``row_select.select_rows`` so a hook on this name
    applies to every filter."""
    return tensor[index]


def concat_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """``torch.cat([first, second])`` along rows, reached as
    ``row_select.concat_rows`` like ``select_rows``."""
    return torch.cat([first, second])
