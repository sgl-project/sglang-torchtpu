"""Bound the shapes of the batch filters' and merges' device programs.

Why: a filter that keeps N of M requests indexes M-row tensors with an N-row
index, and a merge that adds K requests to M joins M rows to K rows. TorchTPU
makes one program per (M, N) or (M, K) pair, so each new pair loads (or
compiles) several programs at serving time, 3-7 ms each.
How: AROUND sglang.srt.managers.row_select.select_rows and concat_rows. Pad the
rows to --max-running-requests, then gather (a merge first joins two padded
tensors): each pad's program depends only on its own row count and the
gather's only on the result's. A merge that adds one row keeps torch.cat,
whose program depends only on M. A plugin that runs its own fixed-capacity
filter and merge programs can replace these seams.
Upstream seam issue: not filed (local seam).
Remove when: the batch keeps its rows padded to a bucket on the device.
"""

from __future__ import annotations

import torch

from sglang.srt.runtime_context import get_schedule

# Device positions of the real rows in two joined padded tensors, per
# (rows, added rows, padded rows, device): built on the host once per pair and
# kept as (host source, device copy). The device copy is non-blocking from
# pageable memory, and nothing proves TorchTPU has read the source when .to()
# returns, so the source lives as long as the copy.
_JOIN_INDEX: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
# Merges that added one row and also ran the padded join once, per
# (rows, dtypes, trailing shape, device).
_WARMED: set[tuple] = set()


def _on_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "cpu"


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    padding = (0, 0) * (tensor.dim() - 1) + (0, rows - tensor.shape[0])
    return torch.nn.functional.pad(tensor, padding)


def select_rows(original, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    capacity = get_schedule().max_running_requests
    if (
        not _on_device(tensor)
        or not capacity
        or tensor.shape[0] > capacity
        or index.shape[0] > capacity
    ):
        return original(tensor, index)
    if tensor.shape[0] < capacity:
        tensor = _pad_rows(tensor, capacity)
    return tensor.index_select(0, index)


def _join_padded(
    first: torch.Tensor, second: torch.Tensor, capacity: int
) -> torch.Tensor:
    rows, added = first.shape[0], second.shape[0]
    # torch.cat promotes mixed dtypes (int32 and int64 lengths) the same way.
    dtype = torch.promote_types(first.dtype, second.dtype)
    if first.dtype != dtype:
        first = first.to(dtype)
    if second.dtype != dtype:
        second = second.to(dtype)
    joined = torch.cat(
        [
            first if rows == capacity else _pad_rows(first, capacity),
            second if added == capacity else _pad_rows(second, capacity),
        ]
    )
    key = (rows, added, capacity, first.device)
    cached = _JOIN_INDEX.get(key)
    if cached is None:
        # Non-blocking like the batch filter's index: a blocking copy waits
        # for the queued device work.
        source = torch.tensor(
            [*range(rows), *range(capacity, capacity + added)], dtype=torch.int64
        )
        cached = (source, source.to(first.device, non_blocking=True))
        _JOIN_INDEX[key] = cached
    return joined.index_select(0, cached[1])


def concat_rows(original, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    capacity = get_schedule().max_running_requests
    rows, added = first.shape[0], second.shape[0]
    if (
        not _on_device(first)
        or not _on_device(second)
        or not capacity
        or rows + added > capacity
        or first.shape[1:] != second.shape[1:]
    ):
        return original(first, second)
    if added == 1:
        # A (rows, 1) program depends only on rows, and the decode warm-up adds
        # one request at every count, so plain torch.cat loads nothing new.
        # The first such merge at a count also runs the padded join once, so
        # the programs of merges that add several rows load in the warm-up too.
        key = (rows, first.dtype, second.dtype, first.shape[1:], first.device)
        if key not in _WARMED:
            _WARMED.add(key)
            _join_padded(first, second, capacity)
        return original(first, second)
    return _join_padded(first, second, capacity)
