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
        if self._ring_page_size is None:
            return True
        return self._ring_fits(forward_batch, lengths)

    # Ring mode (SGLANG_TPU_PCP_MODE=ring), set by ``configure_ring``: the KV
    # page size and the largest usable local token bucket. None in gather mode.
    _ring_page_size: int | None = None
    _ring_max_local_rows: int | None = None

    def configure_ring(self, *, page_size: int, max_local_rows: int) -> None:
        """Switch ``can_apply`` to the ring-mode layout's capacity rule.

        Called by the compile policy on every rank with identical values, so
        the ranks keep admitting the same batches to the same program.
        """
        self._ring_page_size = int(page_size)
        self._ring_max_local_rows = int(max_local_rows)

    def _ring_fits(self, forward_batch: Any, lengths: list[int]) -> bool:
        """Whether the ring-mode layout of this batch fits a token bucket.

        Ring mode reserves two page-aligned blocks per request on every rank,
        so a batch of many short requests can need more local rows than the
        largest bucket holds; it then runs the striped (DCP) extend program
        like any other batch the zigzag strategy does not take.
        """
        from sglang_tpu.cp_forward import get_tpu_cp_layout
        from sglang_tpu.cp_layout import ring_local_rows

        # Decided on the host before the layout was attached; the compiled
        # forward only re-reads that decision (see pinned_extend_seq_lens_cpu).
        if get_tpu_cp_layout(forward_batch) is not None:
            return True
        rows = ring_local_rows(lengths, self.cp_size, self._ring_page_size)
        return rows <= self._ring_max_local_rows
