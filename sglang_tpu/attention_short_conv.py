# SPDX-License-Identifier: Apache-2.0
"""TPU ShortConv state metadata, shared by every convolution layer in a step."""

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
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
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
