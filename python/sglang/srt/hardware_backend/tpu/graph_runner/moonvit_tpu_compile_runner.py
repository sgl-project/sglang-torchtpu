"""MoonViT (Kimi-K2.5) encoder runner for TPU torch.compile.

MoonViT diverges from Qwen3-VL's ViT (complex freqs_cis RoPE, a temporal-pool
merger that returns a list, no spatial_merge_size/deepstack), so it gets its
own runner/policy rather than being contorted onto VisionTpuCompilePolicy.
The policy half lives in tpu/compilation/encoder_compile_policy.py.

Compiled boundary = the encoder **block loop + final_layernorm** only. The
eager preamble (patch embed, pos embed, RoPE precompute, cu_seqlens) and the
eager postamble (tpool_patch_merger temporal pooling) stay outside the graph.
RoPE is passed as a kimi_k25.RealRoPE (cos, sin) pair so no
complex op enters the graph; per-image boundaries + padding are carried by a
fixed-shape segment_ids instead of a variable-length cu_seqlens.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    BlockedEncoder,
    EncoderTpuRunner,
    _cu_seqlens_to_segment_ids,
)


class MoonVitBlockEncoder(BlockedEncoder, Protocol):
    """What MoonViT3dEncoder must expose for this runner."""

    final_layernorm: nn.Module


class MoonVitTpuCompileRunner(EncoderTpuRunner[MoonVitBlockEncoder]):
    """Runs MoonViT's encoder blocks + final_layernorm on TPU.

    Eager until MoonVitTpuCompilePolicy.install replaces run with the compiled
    boundary. The merge (tpool_patch_merger) is applied by the caller after
    this runner, so the runner never touches it. A False supports_segment_ids
    means the caller must not use this runner at all; nothing here would mask.
    """

    def __init__(self, encoder: MoonVitBlockEncoder) -> None:
        super().__init__(
            encoder,
            unsupported_warning=(
                "MoonVitTpuCompileRunner: MoonViT attention does not support "
                "segment_ids; the caller must keep the eager cu_seqlens loop."
            ),
        )

        # Local import: kimi_k25 imports this module while still executing, so a
        # top-level import back into it would be order-dependent.
        from sglang.srt.models.kimi_k25 import RealRoPE

        self._real_rope = RealRoPE

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled MoonViT boundary over a padded patch bucket.

        x is [S_bucket, D], cos/sin are [S_bucket, head_dim/2], segment_ids is
        [1, S_bucket] int32; padding carries a distinct trailing id so it never
        attends to real tokens. Returns [1, S_bucket, D].
        """
        encoder = self.encoder
        for block in encoder.blocks:
            x = block(
                x,
                None,  # cu_seqlens unused: segment_ids drives TPU attention
                None,  # max_seqlen unused
                rope_freqs_cis=self._real_rope(cos, sin),
                segment_ids=segment_ids,
            )
        return encoder.final_layernorm(x)

    def run_eager(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Eager boundary used until the policy installs the compiled fn.

        Builds segment_ids from cu_seqlens so eager and compiled paths feed
        the segment-aware TPU attention identically.
        """
        seq_len = x.shape[0]
        segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)
        return self.forward(x, cos, sin, segment_ids)
