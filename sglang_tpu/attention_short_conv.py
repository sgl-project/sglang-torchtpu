# SPDX-License-Identifier: Apache-2.0
"""TPU ShortConv metadata and dispatch."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.short_conv_backend import ShortConvMetadata
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
                "TPU LFM2 ShortConv does not implement extra_buffer radix-cache "
                "snapshots. Use --mamba-radix-cache-strategy no_buffer or "
                "--disable-radix-cache."
            )
        self._metadata: Optional[ShortConvTracedMetadata] = None
        self._decode_query_start_loc: dict[int, torch.Tensor] = {}
        # Register before torch.compile traces the model.
        from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op

        build_short_conv_op()

    def init_forward_metadata(self, forward_batch: ForwardBatch):
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
                    batch_size + 1, dtype=torch.int32, device=device
                )
                self._decode_query_start_loc[batch_size] = query_start_loc
            self._metadata = ShortConvTracedMetadata(slots, query_start_loc)
        else:
            # Rebuild boundaries from lengths: extend_start_loc has zeroed padding.
            # libtpu's non-associative scan rejects int64 cumsum, so use int32.
            query_start_loc = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(forward_batch.extend_seq_lens, 0, dtype=torch.int32),
                )
            )
            self._metadata = ShortConvTracedMetadata(
                slots,
                query_start_loc,
                forward_batch.extend_prefix_lens > 0,
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
    ) -> ShortConvMetadata:
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return ShortConvMetadata(layer_cache, *self._metadata)

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("ShortConv layers call conv_state_metadata directly.")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("ShortConv layers call conv_state_metadata directly.")


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    metadata: ShortConvMetadata,
    forward_mode: ForwardMode,
) -> torch.Tensor:
    """Run the TPU ShortConv kernel between LFM2's two gates."""
    from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op

    return build_short_conv_op()(
        x,
        weight,
        bias,
        metadata.layer_cache.conv[0],
        metadata.cache_indices,
        metadata.query_start_loc,
        metadata.has_initial_state,
        forward_mode.is_decode(),
    )
