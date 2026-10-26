# SPDX-License-Identifier: Apache-2.0
"""Shifted mHC boundaries. Layer RMSNorm stays in the SGLang decoder.

The JAX implementation and three-part weight packing follow
vllm-torchtpu bdb38700dd25959c029c7d77f00cee91ccd3a0bc, common/mhc.py.
"""

from functools import cache
from typing import NamedTuple

import torch
import torch.nn.functional as F


def hc_combine(x_flat, pre, hc, out_dtype):
    """Unfused mHC combine; the model still owns shifted-pre ordering."""
    return (
        (x_flat.unflatten(-1, (hc, -1)).float() * pre.unsqueeze(-1))
        .sum(-2)
        .to(out_dtype)
    )


class ShiftedCarry(NamedTuple):
    """An uncommitted sublayer result and the four inputs to its next boundary."""

    output: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    pre: torch.Tensor


def pack_mix_weight(weight: torch.Tensor) -> torch.Tensor:
    """Store an FP32 [mixes,width] matrix as BF16 [width,128] hi/mid/lo."""
    if weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("mHC packing expects an FP32 [mixes,width] matrix")
    if weight.shape[0] * 3 > 128:
        raise ValueError("mHC mix weight does not fit three parts in 128 columns")
    values = weight.detach().cpu().contiguous()

    def truncate(x):
        return (x.view(torch.int32) & -65536).view(torch.float32)

    high = truncate(values)
    middle = truncate(values - high)
    low = values - high - middle
    parts = torch.cat((high, middle, low)).to(torch.bfloat16)
    return F.pad(parts.T, (0, 128 - parts.shape[0])).contiguous().to(weight.device)


@cache
def register_ops():
    """Register once during model construction, before Dynamo sees the loop."""
    from sglang_tpu.dsv41.kernels.mhc import (
        _post_jax,
        delayed_post_pre_flat_jax,
        delayed_pre_flat_jax,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def outputs(flat, scale, previous_pre):
        tokens, mult = previous_pre.shape
        return (
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    pre = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_shifted_pre", delayed_pre_flat_jax
    )
    pre.register_fake(
        lambda flat, fn, scale, base, previous_pre, *_: outputs(
            flat, scale, previous_pre
        )
    )
    seam = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_shifted_post_pre", delayed_post_pre_flat_jax
    )
    seam.register_fake(
        lambda x, flat, post, comb, fn, scale, base, previous_pre, *_: (
            torch.empty_like(flat),
            *outputs(flat, scale, previous_pre),
        )
    )
    post = cpu_export_jax_op("sglang_tpu_pallas::dsv41_shifted_post", _post_jax)
    post.register_fake(lambda x, flat, post, comb: torch.empty_like(flat))
    return pre, seam, post


class ShiftedMhc:
    """One sublayer's parameters and its delayed pre-mix contract."""

    def __init__(self, layer, prefix):
        self.layer = layer
        self.prefix = prefix
        self.constants = (
            float(layer.rms_norm_eps),
            float(layer.hc_eps),
            float(layer.hc_eps),
            2.0,
            int(layer.hc_sinkhorn_iters),
        )
        self.pre_op, self.seam_op, self.post_op = register_ops()

    def weights(self):
        return tuple(
            getattr(self.layer, f"hc_{self.prefix}_{suffix}")
            for suffix in ("fn", "scale", "base")
        )

    def start(self, residual, previous_pre=None):
        flat = residual.flatten(1)
        if previous_pre is None:
            # The first sublayer reads stream zero; its new pre gate belongs
            # to the *next* sublayer, including across Engram boundaries.
            first = torch.arange(self.layer.hc_mult, device=flat.device) == 0
            previous_pre = first.to(torch.float32).expand(flat.shape[0], -1)
        return self.pre_op(flat, *self.weights(), previous_pre, *self.constants)

    def advance(self, carry: ShiftedCarry):
        flat, post, comb, value, pre = self.seam_op(
            carry.output,
            carry.residual.flatten(1),
            carry.post,
            carry.comb,
            *self.weights(),
            carry.pre,
            *self.constants,
        )
        return flat.view_as(carry.residual), post, comb, value, pre

    def commit(self, carry: ShiftedCarry):
        """Materialize before Engram or final head; keep carry.pre unchanged."""
        return self.post_op(
            carry.output, carry.residual.flatten(1), carry.post, carry.comb
        ).view_as(carry.residual)


def pack_layer_weights(layers):
    """Run after checkpoint loading; core names and loader ownership stay intact."""
    for layer in layers:
        for prefix in ("attn", "ffn"):
            name = f"hc_{prefix}_fn"
            weight = getattr(layer, name)
            if weight.dtype == torch.float32:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(pack_mix_weight(weight), requires_grad=False),
                )
