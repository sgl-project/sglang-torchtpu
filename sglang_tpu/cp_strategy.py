# SPDX-License-Identifier: Apache-2.0
"""Prefill context-parallel strategy of the TPU program."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TpuZigzagCPStrategy(ZigzagCPStrategy):
    """Zigzag strategy of the TPU context-parallel program.

    The TPU layout (``cp_layout``) also places requests shorter than
    ``2 * cp_size`` tokens, so one short request, such as a new turn after a
    prefix-cache hit, keeps the rest of its batch context-parallel. A batch
    needs one request of at least ``cp_size`` tokens, which gives every rank
    an attention row.
    """

    def __init__(self, cp_size: int) -> None:
        super().__init__(cp_size=cp_size)
        # Set by draft_cp_context while a speculative draft runs.
        self.suspended = False

    def can_apply(self, num_tokens: int, forward_batch: Any) -> bool:
        if self.suspended or self.cp_size <= 1 or num_tokens < self.cp_size * 2:
            return False
        forward_mode = forward_batch.forward_mode
        # A mixed batch carries its running requests as one-token extends and
        # runs the plain program: the context-parallel family warms extends only.
        if (
            not forward_mode.is_context_parallel_extend()
            or forward_mode == ForwardMode.MIXED
        ):
            return False
        extend_lens = forward_batch.extend_seq_lens_cpu
        if extend_lens is None:
            return True
        lengths = [int(length) for length in extend_lens]
        return min(lengths) > 0 and max(lengths) >= self.cp_size


@contextlib.contextmanager
def draft_cp_context() -> Iterator[None]:
    """Run speculative draft forwards as plain TP: the draft extend reads the
    target's prompt hidden states already gathered in global token order."""
    strategy = get_cp_strategy()
    if not isinstance(strategy, TpuZigzagCPStrategy) or strategy.suspended:
        yield
        return
    strategy.suspended = True
    try:
        yield
    finally:
        strategy.suspended = False
