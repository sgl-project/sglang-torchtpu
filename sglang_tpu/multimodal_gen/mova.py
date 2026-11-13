# SPDX-License-Identifier: Apache-2.0
"""MOVA on TPU workers: real-valued RoPE tables.

Runtime patch, tracked debt:
- Why: MOVA's video and audio DiTs rotate queries and keys by complex128 RoPE
  tables, moved to the device every step. torch_tpu cannot move complex128
  tensors to the device: stock main fails every request at that move
  ("failed to inline functions in MLIR module", CDK j-bc9336da).
- How: :func:`install` stores each DiT's RoPE tables as float32 ``(cos, sin)``
  pairs in the complex layout's place (the last dim interleaves them), from
  the same float64 frequencies; the rotation then runs in float32 (bfloat16
  with the pair-swap form of ``sglang_tpu.multimodal_gen.rope``), rounding
  once to the activation dtype.
- Upstream PR or issue: none.
- Remove when: the MOVA DiTs take real RoPE tables on every platform.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import base as rope_base
from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils as rope_utils
from sglang.multimodal_gen.runtime.models.dits import mova_audio_dit as audio
from sglang.multimodal_gen.runtime.models.dits import mova_video_dit as video
from sglang_tpu.multimodal_gen.rope import apply_rotary_embedding_pair_swap

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK_COMPLEX_ROPE = rope_utils._apply_rotary_emb_complex
_STOCK_VIDEO_INIT_FREQS = video.WanModel._init_freqs
_STOCK_AUDIO_INIT_FREQS = audio.WanAudioModel._init_freqs


def real_freqs(freqs: torch.Tensor) -> torch.Tensor:
    """A complex table ``[..., n]`` as float32 ``[..., 2 n]``: cos, sin interleaved."""
    return torch.view_as_real(freqs).float().flatten(-2)


def apply_rotary_emb_complex(x, freqs, dtype=torch.float32):
    """``_apply_rotary_emb_complex``; real tables (:func:`real_freqs`) rotate in float32."""
    if freqs.is_complex():
        return _STOCK_COMPLEX_ROPE(x, freqs, dtype)
    # [s, 1, d] interleaved (cos, sin) -> [s, d // 2] each.
    pairs = freqs.squeeze(-2).unflatten(-1, (-1, 2))
    cos, sin = pairs[..., 0], pairs[..., 1]
    if x.dtype == torch.bfloat16:
        return apply_rotary_embedding_pair_swap(x, cos, sin, interleaved=True)
    # The native form runs in its input dtype: rotate float32, round once.
    return apply_rotary_embedding_native(x.float(), cos, sin, interleaved=True).to(
        x.dtype
    )


def _real_init_freqs(stock):
    def init_freqs(self):
        stock(self)
        if self.freqs is not None and any(f.is_complex() for f in self.freqs):
            self.freqs = tuple(real_freqs(f) for f in self.freqs)

    init_freqs.__wrapped_for_tpu__ = True
    return init_freqs


def install() -> None:
    rope_utils._apply_rotary_emb_complex = apply_rotary_emb_complex
    rope_base._apply_rotary_emb_complex = apply_rotary_emb_complex
    for cls, stock in (
        (video.WanModel, _STOCK_VIDEO_INIT_FREQS),
        (audio.WanAudioModel, _STOCK_AUDIO_INIT_FREQS),
    ):
        if not getattr(cls._init_freqs, "__wrapped_for_tpu__", False):
            cls._init_freqs = _real_init_freqs(stock)
