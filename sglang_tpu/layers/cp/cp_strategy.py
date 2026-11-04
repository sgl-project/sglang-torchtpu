# SPDX-License-Identifier: Apache-2.0
"""Prefill context-parallel strategy of the TPU program."""

from __future__ import annotations

from typing import Any

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

    def can_apply(self, num_tokens: int, forward_batch: Any) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size * 2:
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
