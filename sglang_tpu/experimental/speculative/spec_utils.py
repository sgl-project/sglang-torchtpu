"""Host-side helpers shared by every TpuSpecWorker algorithm."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


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


def grammar_verify_bitmask(
    reqs: list[Req], verify_tokens_cpu: torch.Tensor, vocab_size: int, device: str
) -> torch.Tensor | None:
    """Packed int32 grammar mask [pbs * W, ceil(V / 32)] over verify rows [r, d_1..d_{W-1}].

    Row i * W + j allows req i's next tokens after d_1..d_j. Rows of padding, of
    requests without a live grammar, and past a draft the grammar rejects allow
    every token. Each grammar is rolled back, so its state is unchanged.
    """
    live = [
        (i, req.grammar)
        for i, req in enumerate(reqs)
        if req.grammar is not None
        and not req.grammar.finished
        and not req.grammar.is_terminated()
    ]
    if not live:
        return None
    pbs, width = verify_tokens_cpu.shape
    assert len(reqs) <= pbs, (len(reqs), pbs)
    bitmask = torch.full(
        (pbs * width, (vocab_size + 31) // 32),
        -1,
        dtype=torch.int32,
        pin_memory=is_pin_memory_available(device),
    )
    # A chain: node j's only child is j + 1; node 0 is r, already accepted.
    next_token = torch.full((width,), -1, dtype=torch.int64)
    next_token[:-1] = torch.arange(1, width)
    next_sibling = torch.full((width,), -1, dtype=torch.int64)
    for i, grammar in live:
        traverse_tree(
            next_token,
            next_sibling,
            verify_tokens_cpu[i],
            grammar,
            bitmask[i * width : (i + 1) * width],
            vocab_size=vocab_size,
        )
    return bitmask


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
