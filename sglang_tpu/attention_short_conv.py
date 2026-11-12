# SPDX-License-Identifier: Apache-2.0
"""Causal ShortConv and its per-step state metadata for TPU.

Inputs and outputs are token-major [tokens, channels]. History is stored oldest
first in [slots, width - 1, 1, channels], and weight[:, -1] multiplies the current
token. Arithmetic and history use FP32; outputs retain the input dtype before
the model applies its output gate. All loops depend only on compiled shapes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.runtime_context import get_exec, get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


class ShortConvTracedMetadata(NamedTuple):
    """Per-step tensors lifted into direct_compile as graph inputs."""

    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor
    has_initial_state: Optional[torch.Tensor] = None
    token_request_indices: Optional[torch.Tensor] = None
    valid_tokens: Optional[torch.Tensor] = None


class TpuShortConvMetadata(NamedTuple):
    """Per-layer cache handle plus the shared per-step device metadata."""

    layer_cache: Any
    cache_indices: torch.Tensor
    query_start_loc: torch.Tensor
    has_initial_state: Optional[torch.Tensor]
    token_request_indices: Optional[torch.Tensor]
    valid_tokens: Optional[torch.Tensor]


class TpuShortConvAttnBackend(MambaAttnBackendBase):
    """State sidecar for causal LFM2, using the existing TPU Mamba pool."""

    needs_cpu_seq_lens = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        if not isinstance(model_runner.model_config.hf_config, Lfm2Config):
            raise NotImplementedError(
                "The TPU ShortConv sidecar currently implements causal LFM2 only."
            )
        if get_spec().speculative_algorithm is not None:
            raise NotImplementedError(
                "TPU LFM2 ShortConv does not implement speculation."
            )
        if get_exec().mamba.enable_mamba_extra_buffer:
            raise NotImplementedError(
                "TPU LFM2 ShortConv does not implement radix-cache state snapshots. "
                "Use --disable-radix-cache."
            )
        self._metadata: Optional[ShortConvTracedMetadata] = None
        self._decode_query_start_loc: dict[int, torch.Tensor] = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Do not call the GPU base metadata builder: its tracked-row selection
        # uses variable-size indexing and device-to-host mask reads.
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            raise NotImplementedError(
                "TPU LFM2 ShortConv does not implement speculation."
            )
        req_rows = forward_batch.req_pool_indices
        slots = self._translate_mamba_indices(
            self.req_to_token_pool.get_mamba_indices(req_rows.clamp(min=0))
        ).to(torch.int64)
        slots = torch.where(req_rows > 0, slots.clamp(min=0), 0)
        batch_size = slots.shape[0]
        device = slots.device

        if mode.is_decode_or_idle():
            query_start_loc = self._decode_query_start_loc.get(batch_size)
            if query_start_loc is None:
                query_start_loc = torch.arange(
                    batch_size + 1, dtype=torch.int64, device=device
                )
                self._decode_query_start_loc[batch_size] = query_start_loc
            self._metadata = ShortConvTracedMetadata(slots, query_start_loc)
        else:
            # Padded extend_start_loc rows are zero, not cumulative. Build
            # monotonic boundaries from lengths so zero-length requests and
            # trailing token padding do not change real request boundaries.
            query_start_loc = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(forward_batch.extend_seq_lens, 0, dtype=torch.int64),
                )
            )
            positions = torch.arange(
                forward_batch.input_ids.shape[0], dtype=torch.int64, device=device
            )
            request_indices = torch.searchsorted(
                query_start_loc[1:].contiguous(), positions, right=True
            )
            valid_tokens = positions < query_start_loc[-1]
            request_indices = torch.where(valid_tokens, request_indices, 0)
            self._metadata = ShortConvTracedMetadata(
                slots,
                query_start_loc,
                forward_batch.extend_prefix_lens > 0,
                request_indices,
                valid_tokens,
            )
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc, mamba_cache_indices=slots
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.init_forward_metadata(forward_batch)

    def traced_metadata(self) -> ShortConvTracedMetadata:
        return self._metadata

    @contextmanager
    def bind_traced_metadata(self, metadata: ShortConvTracedMetadata) -> Iterator[None]:
        previous = self._metadata
        self._metadata = metadata
        try:
            yield
        finally:
            self._metadata = previous

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> TpuShortConvMetadata:
        # List-index the per-layer buffer, preserving its storage identity for
        # compiled in-place writes. Radix-state offloading is disabled here.
        pool = self.req_to_token_pool
        layer_cache = pool.mamba_pool.mamba2_layer_cache(pool.mamba_map[layer_id])
        return TpuShortConvMetadata(layer_cache, *self._metadata)

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("ShortConv layers call conv_state_metadata directly.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("ShortConv layers call conv_state_metadata directly.")


def _write_history(
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    history: torch.Tensor,
    active: torch.Tensor,
) -> None:
    # Several padded requests can share slot 0. A batched index_copy_ would
    # have duplicate destinations, so serialize the fixed-size batch rows.
    # Inactive rows preserve their destination, including a dirty null slot.
    history = history.unsqueeze(2).to(conv_state.dtype)
    for row in range(slots.shape[0]):
        index = slots[row : row + 1]
        previous = conv_state.index_select(0, index)
        update = torch.where(
            active[row : row + 1].reshape(1, 1, 1, 1),
            history[row : row + 1],
            previous,
        )
        conv_state.index_copy_(0, index, update)


def short_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """One token per request, including padded requests mapped to slot 0."""
    history = conv_state.index_select(0, slots).squeeze(2)
    window = torch.cat((history, x.float().unsqueeze(1)), dim=1)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)
    for k in range(weight.shape[1]):
        output = output + window[:, k] * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    active = slots > 0
    _write_history(conv_state, slots, window[:, 1:], active)
    return torch.where(active[:, None], output, 0).to(x.dtype)


def short_conv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
    token_request_indices: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Packed prefill, with zero-length requests and trailing token padding.

    The caller builds token_request_indices once per forward, shared by all
    layers. Invalid tokens use request 0 and are masked before returning.
    """
    history = conv_state.index_select(0, slots).squeeze(2)
    history = torch.where(has_initial_state[:, None, None], history, 0)
    batch_size, history_len, channels = history.shape
    history_rows = batch_size * history_len
    bank = torch.cat((history.reshape(-1, channels), x.float()), dim=0)
    starts = query_start_loc[:-1]
    lengths = query_start_loc[1:] - starts
    positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    positions = torch.where(valid_tokens, positions, 0)
    token_starts = starts.index_select(0, token_request_indices)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)

    for k in range(weight.shape[1]):
        relative = positions - token_starts - history_len + k
        indices = torch.where(
            relative < 0,
            token_request_indices * history_len + relative + history_len,
            history_rows + positions - history_len + k,
        )
        output = output + bank.index_select(0, indices) * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    tail_positions = (
        lengths[:, None]
        - history_len
        + torch.arange(history_len, dtype=torch.int64, device=x.device)
    )
    request_base = (
        torch.arange(batch_size, dtype=torch.int64, device=x.device)[:, None]
        * history_len
    )
    tail_indices = torch.where(
        tail_positions < 0,
        request_base + tail_positions + history_len,
        history_rows + starts[:, None] + tail_positions,
    )
    next_history = bank.index_select(0, tail_indices.reshape(-1)).reshape(
        batch_size, history_len, channels
    )
    _write_history(conv_state, slots, next_history, (slots > 0) & (lengths > 0))
    active_tokens = valid_tokens & (slots.index_select(0, token_request_indices) > 0)
    return torch.where(active_tokens[:, None], output, 0).to(x.dtype)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    metadata: TpuShortConvMetadata,
    forward_mode: ForwardMode,
) -> torch.Tensor:
    """Platform kernel entry point used between LFM2's two gates."""
    conv_state = metadata.layer_cache.conv[0]
    if forward_mode.is_decode():
        return short_conv_decode(x, weight, bias, conv_state, metadata.cache_indices)
    return short_conv_prefill(
        x,
        weight,
        bias,
        conv_state,
        metadata.cache_indices,
        metadata.query_start_loc,
        metadata.has_initial_state,
        metadata.token_request_indices,
        metadata.valid_tokens,
    )
