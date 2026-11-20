# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""SGLang top-k output to `moe_tp_decode` kernel inputs (pure JAX)."""

import jax
import jax.numpy as jnp

from sglang_tpu.moe.pallas.moe_tp_decode.kernel import ROW_BUCKETS


def kernel_inputs(x, topk_ids, topk_weights, num_experts: int):
    """(x, wtab, n_valid) padded to the smallest admitting row bucket.

    TopK marks padded tokens with id -1. Their rows get no routing weight
    (one_hot(-1) is all zeros, and their weights are masked so a NaN cannot
    reach the table), and they must follow the real rows: n_valid counts the
    rows with any live id.
    """
    m = x.shape[0]
    bucket = next(b for b in ROW_BUCKETS if b >= m)
    if bucket != m:
        pad = bucket - m
        x = jnp.pad(x, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)), constant_values=-1)
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
    valid = topk_ids >= 0
    weights = jnp.where(valid, topk_weights.astype(jnp.float32), 0.0)
    wtab = jnp.einsum(
        "rk,rke->re",
        weights,
        jax.nn.one_hot(topk_ids, num_experts, dtype=jnp.float32),
    )
    n_valid = jnp.sum(jnp.any(valid, axis=1)).astype(jnp.int32)
    return x, wtab, n_valid
