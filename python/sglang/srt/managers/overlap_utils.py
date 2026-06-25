from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


# Device types whose backend bakes a Python-int index as a compile-time constant
# (e.g. torch_tpu / XLA), so indexing a buffer with a slice or int recompiles the
# graph per distinct value. On these the FutureMap keeps its circular-buffer write
# offset as a device tensor and writes with index_copy_; other backends index
# eagerly, where the cheaper slice assignment is fine.
_DYNAMIC_INDEX_DEVICE_TYPES = ("tpu",)


def _device_use_dynamic_index(device: torch.device) -> bool:
    """Whether ``device``'s backend bakes Python-int indices as graph constants.

    True for the backends in ``_DYNAMIC_INDEX_DEVICE_TYPES`` (which would otherwise
    recompile a slice/int-indexed write per value); those keep the future-map
    offsets dynamic (device tensor + ``index_copy_``) instead.
    """
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    return any(dev_type == known for known in _DYNAMIC_INDEX_DEVICE_TYPES)


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        # On backends that bake Python-int indices as HLO constants (e.g. TPU),
        # keep the circular-buffer write head on-device (`future_ct_dev`, mirrors
        # `future_ct`) so future-map indices stay DATA rather than baked constants
        # that recompile the store every step. See `alloc_future_indices` / `_store`.
        self._use_dynamic_index = _device_use_dynamic_index(device)
        if self._use_dynamic_index:
            self.future_ct_dev = torch.zeros((), dtype=torch.int64, device=device)

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_token0 = draft_input.bonus_tokens[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.future_buffer_len, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        if self._use_dynamic_index:
            # base is a device scalar (== start) carried on-device, so the offset
            # is a runtime operand: `arange(bs)` bakes only the bucketed bs, not
            # the position. future_ct_dev mirrors future_ct (same update below).
            base = self.future_ct_dev + 1
            indices = base + torch.arange(bs, dtype=torch.int64, device=self.device)
            self.future_ct_dev = torch.remainder(
                self.future_ct_dev + bs, self.future_limit
            )
        else:
            indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # batch.spec_info), so the caching allocator (torch GC) could
            # reclaim the memory before the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def _store(self, buf: torch.Tensor, fi: FutureIndices, values: torch.Tensor):
        """Write ``values`` into circular buffer ``buf`` at future indices ``fi``.

        On backends that avoid static indices (``_use_dynamic_index``) use a
        tensor-indexed ``index_copy_`` so the write offset stays DATA: a
        Python-slice assign (``buf[fi.interval] = values``) bakes the buffer
        position as an HLO compile-time constant, recompiling the future-map store
        every overlap step. ``fi.indices`` is the on-device equivalent of the
        ``fi.interval`` slice (see ``alloc_future_indices``). Elsewhere keep the
        cheaper slice assign.
        """
        if self._use_dynamic_index:
            buf.index_copy_(0, fi.indices, values)
        else:
            buf[fi.interval] = values

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            # idle indices in dp attention do not need store info
            if self.is_empty_slice(future_indices.interval):
                return
            self._store(
                self.token_ids_buf, future_indices, batch_result.next_token_ids
            )
        else:
            self.store_to_map_for_new_batch(
                future_indices, batch_result.next_draft_input
            )

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        # idle indices in dp attention do not need store info
        if self.is_empty_slice(future_indices.interval):
            return
        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)
        self._store(self.topk_p_buf, future_indices, draft_input.topk_p)
        self._store(self.topk_index_buf, future_indices, draft_input.topk_index)
        self._store(self.bonus_tokens_buf, future_indices, draft_input.bonus_tokens)
        self._store(self.new_seq_lens_buf, future_indices, draft_input.new_seq_lens)
        if spec_need_hidden_states():
            self._store(
                self.hidden_states_buf, future_indices, draft_input.hidden_states
            )
