"""Hand each request's KV on this pipeline stage to the Raiden pipeline transfer
before the stage's result point.

The PP prefill loop reports each forward it launches and, on the last stage,
each microbatch slot's sampled tokens once its deferred output send has copied
them to the host. For every request whose final chunk a forward computes:
- with stage groups (SGLANG_RAIDEN_PIPELINE_STAGE_PARTS), this stage's share
  registers once a host copy of one element of the forward's output lands
  (PipelineTransfer.make_fence);
- with SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE, this stage's share of the closing
  transfer registers at launch on stages before the last, and at the token
  copy on the last stage. That copy completes after the last stage's forward,
  and so after every earlier stage's forward of the batch, which it consumed.
  The last stage's share carries the first-token metadata unless the result
  point must send it (_first_token_row).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, NamedTuple, Optional

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import read_aux_blob

logger = logging.getLogger(__name__)


class _HeldShare(NamedTuple):
    room: int
    # The request's position in the launched batch, and its first token's.
    index: int
    full: list[int]
    swa: list[int]
    # Fills the request's metadata row from its first token; None when the
    # result point does.
    write_row: Optional[Callable[[int], Any]]
    aux_index: int


class _HeldSlot(NamedTuple):
    batch_size: int
    shares: list[_HeldShare]


class FinalChunks:
    """This stage's final chunks between launch and hand-off."""

    def __init__(self, transfer):
        self.transfer = transfer
        # Last stage, early close: microbatch slot -> shares awaiting its tokens.
        self.held: dict[int, _HeldSlot] = {}
        self._size_mismatch_logged = False

    def launched(self, scheduler, mb_id: int, batch, result) -> None:
        """This stage launched ``batch`` in microbatch slot ``mb_id``."""
        transfer = self.transfer
        if not (transfer.stage_parts or transfer.early_close):
            return
        # Register stage parts whose fences completed since the last launch.
        transfer.advance_stages()
        finals = [
            (index, req)
            for index, req in enumerate(batch.reqs)
            # Health checks carry a fake sender with no Raiden room.
            if isinstance(req.disagg_kv_sender, RaidenKVSender)
            and req.extend_end >= len(req.origin_input_ids)
        ]
        if not finals:
            return
        mgr = transfer.manager
        last_stage = mgr.pp_rank == mgr.pp_size - 1
        fence = None
        if transfer.stage_parts:
            output = (
                result.next_token_ids
                if last_stage
                else next(iter(result.pp_hidden_states_proxy_tensors.tensors.values()))
            )
            # One completion fence per forward, shared by its requests.
            fence = transfer.make_fence(output)
        held = []
        for index, req in finals:
            room = req.disagg_kv_sender.bootstrap_room
            full, swa = _pages(scheduler, mgr, req)
            if transfer.stage_parts:
                transfer.stage_ready(room, full, swa, fence)
            elif not last_stage:
                transfer.close_early(room, full, swa)
            else:
                write_row, aux_index = _first_token_row(scheduler, req)
                held.append(_HeldShare(room, index, full, swa, write_row, aux_index))
        if held:
            self.held[mb_id] = _HeldSlot(len(batch.reqs), held)

    def tokens_copied(self, mb_id: int, next_token_ids) -> None:
        """Last stage: the PP loop copied slot ``mb_id``'s sampled tokens to the
        host, so every stage's forward of its final chunks has completed."""
        slot = self.held.pop(mb_id, None)
        if slot is None:
            return
        tokens = next_token_ids.reshape(-1).tolist()
        if len(tokens) != slot.batch_size:
            if not self._size_mismatch_logged:
                self._size_mismatch_logged = True
                logger.warning(
                    "Raiden pipeline: the last stage sampled %d tokens for %d "
                    "requests; their first-token metadata waits for the result point",
                    len(tokens),
                    slot.batch_size,
                )
            tokens = None
        mgr = self.transfer.manager
        for share in slot.shares:
            aux = b""
            if tokens is not None and share.write_row is not None:
                share.write_row(tokens[share.index])
                aux = read_aux_blob(
                    mgr.kv_args.aux_data_ptrs,
                    mgr.kv_args.aux_item_lens,
                    share.aux_index,
                )
            self.transfer.close_early(share.room, share.full, share.swa, aux)


def _pages(scheduler, mgr, req) -> tuple[list[int], list[int]]:
    """The request's full-attention pages and its sliding-window pages."""
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    end = len(req.origin_input_ids)
    # Host-resident tables: reading page ids does not touch the device.
    row = scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
    full = kv_to_page_indices(row[:end], page_size).tolist()
    swa = []
    if StateType.SWA in mgr.kv_args.state_types:
        window_start = max(0, end - scheduler.sliding_window_size)
        window_start = window_start // page_size * page_size
        swa = kv_to_page_indices(
            scheduler.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
                row[window_start:end]
            ),
            page_size,
        ).tolist()
    return full, swa


def _first_token_row(scheduler, req) -> tuple[Optional[Callable[[int], Any]], int]:
    """How the request's metadata row gets its first token before the result
    point, or (None, -1) when the row also needs logprobs, a sampling mask, a
    grammar's verdict or a KV checksum, or the request is still bootstrapping."""
    if (
        req.return_logprob
        or req.return_sampling_mask
        or req.grammar is not None
        or req.pending_bootstrap
        or req.metadata_buffer_index < 0
        or scheduler.kv_checksum_computer is not None
    ):
        return None, -1
    write_row = functools.partial(
        scheduler.disagg_metadata_buffers.set_buf_with_first_token, req
    )
    return write_row, req.metadata_buffer_index
