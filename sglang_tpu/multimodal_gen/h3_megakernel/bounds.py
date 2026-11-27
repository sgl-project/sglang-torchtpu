# SPDX-License-Identifier: Apache-2.0
"""Which MiniMax-H3 layers the megakernel's fixed-max softmax covers, from the
q/k norm weights alone.

``kernel.stack`` runs a layer's attention as ``exp2(s - 17)`` without a running
max when its q and k norm weights bound every score by 109 log2 units, after a
rounding slack of 1.025 (``kernel.fixed_max_layers``; the kernel derives that set
itself from its packed weights). RMSNorm leaves q and k at norm ``sqrt(D)``
before their weights and RoPE turns dims ``d`` and ``d + rope_dims / 2``
together, so Cauchy-Schwarz bounds ``|q . k|`` by ``D`` times the largest product
of a RoPE pair's largest q weight and largest k weight (or of an unrotated dim's
two weights). The bound holds for every input. A layer's heads share its norm
weights, so it is the same for each of them.

This module computes that set on the host with NumPy, without JAX, on the bf16
values the kernel packs, with ``kernel.fixed_max_layers``' float32 operations in
its order, and gives each layer's bound for logs. Under ``jit`` XLA folds the
kernel's three constant factors into one, which can move a bound by an ulp; the
set cannot move: a peak is a product of two bf16 weights, and at H3's head width
none of those lies within 7e-5 (relative) of the threshold.

``stack(..., fixed=None, fixed_heads=None)`` runs exactly these layers at a
fixed max and every other layer with the online softmax; passing
``fixed=result.layers`` and ``fixed_heads=result.heads(56)`` instead changes
nothing.
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# kernel._FIXED_BOUND and kernel._BOUND_SLACK (the tests pin them equal).
FIXED_BOUND = 109.0
BOUND_SLACK = 1.025
# MiniMax-H3's head and RoPE widths (reference.Config's defaults).
HEAD_DIM = 128
ROPE_DIMS = 96


@dataclasses.dataclass(frozen=True)
class FixedMaxLayers:
    """The layers whose softmax runs at a fixed max, from the norm weights.

    Attributes:
      layers: ``[L]`` bool, the layers the norm weights prove.
      bounds: ``[L]`` float32, each layer's score bound times the slack, in
        log2 units; a layer is proven when it is at most ``FIXED_BOUND``.
    """

    layers: np.ndarray
    bounds: np.ndarray

    @property
    def online(self) -> list[int]:
        """The layers that keep the online softmax."""
        return [int(i) for i in np.flatnonzero(~self.layers)]

    def heads(self, heads: int) -> np.ndarray:
        """``[L, heads]`` bool: every head of a proven layer (the weight bound
        is the same for each head of a layer)."""
        return np.repeat(self.layers[:, None], heads, axis=1)

    def summary(self) -> str:
        """E.g. ``41/50 layers at a fixed max (online: 36, 37, 43-49)``."""
        return (
            f"{int(self.layers.sum())}/{self.layers.size} layers at a fixed max"
            f" (online: {_ranges(self.online) or 'none'})"
        )


def fixed_max_layers(
    q_norm: Any,
    k_norm: Any,
    *,
    head_dim: int = HEAD_DIM,
    rope_dims: int = ROPE_DIMS,
) -> FixedMaxLayers:
    """The layers ``kernel.stack`` runs at a fixed max for these norm weights.

    Args:
      q_norm: Every block's ``attn.q_norm.weight``, in block order: a sequence
        of ``[D]`` tensors or arrays (torch tensors on any device work), or one
        ``[L, D]`` / ``[L, 1, D]`` array.
      k_norm: The blocks' ``attn.k_norm.weight``, likewise.
      head_dim: Attention head width ``D``.
      rope_dims: Head dims RoPE rotates, as two halves.

    Returns:
      The proven layers and their bounds, from the weights rounded to bf16 as
      ``reference.pack_weights`` stores them.
    """
    q, k = (_stacked(w, head_dim) for w in (q_norm, k_norm))
    if q.shape != k.shape:
        raise ValueError(f"q_norm {q.shape} and k_norm {k.shape} differ")
    half = rope_dims // 2
    q, k = np.abs(q), np.abs(k)

    def pairs(w: np.ndarray) -> np.ndarray:
        return np.maximum(w[:, :half], w[:, half : 2 * half])

    # kernel.fixed_max_layers, operation for operation, in float32.
    rotated = (pairs(q) * pairs(k)).max(axis=1)
    plain = (q[:, 2 * half :] * k[:, 2 * half :]).max(axis=1, initial=np.float32(0.0))
    peak = np.maximum(rotated, plain)
    bound = peak * np.float32(math.sqrt(head_dim)) * np.float32(math.log2(math.e))
    bounds = bound * np.float32(BOUND_SLACK)
    return FixedMaxLayers(layers=bounds <= np.float32(FIXED_BOUND), bounds=bounds)


def _stacked(weights: Any, head_dim: int) -> np.ndarray:
    """``[L, D]`` float32 holding the bf16 rounding of each layer's weights."""
    if isinstance(weights, Sequence) and not isinstance(weights, np.ndarray):
        array = np.stack([_host(w).reshape(-1) for w in weights])
    else:
        array = _host(weights)
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[1] != head_dim:
        raise ValueError(f"norm weights {array.shape} are not [layers, {head_dim}]")
    return _round_bf16(array)


def _host(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):  # torch: bf16 has no NumPy dtype
        tensor = tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, np.float32)


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to bf16, as float32 (NaN and inf aside)."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & (
        np.uint32(0xFFFF0000)
    )
    return bits.view(np.float32)


def _ranges(values: list[int]) -> str:
    """``[36, 37, 43, 44, 45]`` -> ``"36, 37, 43-45"``."""
    out, start = [], None
    for i, value in enumerate(values):
        if start is None:
            start = value
        if i + 1 == len(values) or values[i + 1] != value + 1:
            if value - start >= 2:
                out.append(f"{start}-{value}")
            else:
                out.extend(str(v) for v in range(start, value + 1))
            start = None
    return ", ".join(out)
