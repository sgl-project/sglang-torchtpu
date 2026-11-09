# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode parameters stored in the form the TPU graph consumes.

A parameter the forward converts or broadcasts the same way every step costs
an op per layer per step; XLA hoists those ops into idle TensorCore time, but
their small parameter copies queue behind large weight prefetches and the
TensorCore waits on them (K3.1 DSA layers: ~10 us per layer behind the
sparse MLA gather window at TP8).

- KDA o_norm (FusedRMSNormGated, TPU forward_native): the [head_dim] bf16
  weight was converted to f32 and broadcast over the local heads each step.
  It is stored as f32 [local_heads, head_dim]; the values are the same and
  the forward broadcasts it over the batch axes for free.
"""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def _expand_onorm_weight(attn: nn.Module) -> bool:
    w = attn.o_norm.weight
    if w is None or w.dim() != 1 or w.shape[0] != attn.head_dim:
        return False
    w.data = w.data.float().expand(attn.local_num_heads, w.shape[0]).contiguous()
    return True


def store_decode_params(model: nn.Module) -> None:
    """Runs post-load, before compilation; idempotent (a weight already
    expanded is skipped)."""
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention

    onorm = 0
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if isinstance(layer.self_attn, KimiK3DeltaAttention):
            onorm += _expand_onorm_weight(layer.self_attn)
    if onorm:
        logger.info(
            "Kimi-K3 KDA o_norm weights stored as f32 [heads, head_dim]: %d layers",
            onorm,
        )
