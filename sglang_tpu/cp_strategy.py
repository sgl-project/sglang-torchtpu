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
        if not (min(lengths) > 0 and max(lengths) >= self.cp_size):
            return False
        if self.ring_max_local_rows is None:
            return True
        # Ring mode reserves two page-aligned blocks per request on every rank,
        # so many short requests can overflow the largest token bucket; such a
        # batch runs the striped (DCP) extend program instead. The decision is
        # made on the host before the layout is attached; the compiled forward
        # only re-reads it (see pinned_extend_seq_lens_cpu).
        from sglang.srt.runtime_context import get_schedule
        from sglang_tpu.cp_forward import get_tpu_cp_layout
        from sglang_tpu.cp_layout import ring_local_rows

        if get_tpu_cp_layout(forward_batch) is not None:
            return True
        rows = ring_local_rows(lengths, self.cp_size, get_schedule().page_size)
        return rows <= self.ring_max_local_rows

    # Ring mode (SGLANG_TPU_PCP_MODE=ring) only: the largest local token bucket
    # its layout can use, set by the compile policy on every rank alike.
    ring_max_local_rows: int | None = None
