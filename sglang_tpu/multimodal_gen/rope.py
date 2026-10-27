# SPDX-License-Identifier: Apache-2.0
"""Rotary embedding without stride-2 lane access, for TPU workers.

Runtime patch, tracked debt:
- Why: ``apply_rotary_embedding_native`` reads the last dim with stride 2, and
  torch_tpu re-lays out those lane slices (~0.13 s per Wan2.2-A14B step).
- How: :func:`install` rebinds ``apply_rotary_embedding`` in
  ``sglang.multimodal_gen.runtime.layers.rotary_embedding.utils``, the name the
  non-neox ``_apply_rotary_emb`` path calls.
- Upstream PR or issue: none.
- Remove when: the diffusion kernel registry accepts a platform-provided rotary
  implementation, or torch_tpu lowers stride-2 lane slices without a relayout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
)


def _build_pair_swap_matrix(dim: int, dtype: torch.dtype, device) -> Tensor:
    idx = torch.arange(0, dim, 2, device=device)
    rot = torch.zeros(dim, dim, dtype=dtype, device=device)
    rot[idx + 1, idx] = -1
    rot[idx, idx + 1] = 1
    return rot


_PAIR_SWAP_MATRICES: dict[tuple, Tensor] = {}


def _pair_swap_matrix(dim: int, dtype: torch.dtype, device) -> Tensor:
    """R with (x @ R)[2i] = -x[2i + 1] and (x @ R)[2i + 1] = x[2i]."""
    # A compiled graph folds the construction; eager builds it once.
    if torch.compiler.is_compiling():
        return _build_pair_swap_matrix(dim, dtype, device)
    key = (dim, dtype, torch.device(device))
    if key not in _PAIR_SWAP_MATRICES:
        _PAIR_SWAP_MATRICES[key] = _build_pair_swap_matrix(dim, dtype, device)
    return _PAIR_SWAP_MATRICES[key]


def apply_rotary_embedding_pair_swap(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """``apply_rotary_embedding_native`` without stride-2 access to the last dim.

    Rotates adjacent pairs as ``x * cos + (x @ R) * sin`` with R a signed pair
    swap. In bfloat16 each output of ``x @ R`` is a single +-1 product
    accumulated in float32, so it is exact. The rotation itself runs in float32
    with float32 tables and rounds once, as the CUDA kernel does; the bf16
    strided form rounds the tables and each product (38% of outputs differ from
    the CUDA kernel at Wan2.2-A14B shapes, against 0.002% here). TPU float16 and
    float32 matmuls are not exact; other dtypes keep the strided form. A non-finite input spreads NaN
    (0 * inf) across its whole head through ``x @ R``, where the strided form
    touches only its pair.
    """
    if x.dtype != torch.bfloat16:
        return apply_rotary_embedding_native(x, cos, sin, interleaved)
    if interleaved and cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    rot = _pair_swap_matrix(x.shape[-1], x.dtype, x.device)
    return (x.float() * cos + torch.matmul(x, rot).float() * sin).to(x.dtype)


def install() -> None:
    """Route diffusion RoPE through :func:`apply_rotary_embedding_pair_swap`."""
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils

    utils.apply_rotary_embedding = apply_rotary_embedding_pair_swap
