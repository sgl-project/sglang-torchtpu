# SPDX-License-Identifier: Apache-2.0
"""Tile arithmetic shared by the short and long query schedules."""

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _hadamard128(value):
    """Sylvester-order butterfly in f32; rounding is checked at the accuracy gate."""
    lane = jnp.arange(128, dtype=jnp.int32)[None, :]
    for step in (1, 2, 4, 8, 16, 32, 64):
        low = (lane & step) == 0
        partner = jnp.where(
            low,
            pltpu.roll(value, 128 - step, axis=1),
            pltpu.roll(value, step, axis=1),
        )
        value = partner + jnp.where(low, value, -value)
    return value


def pool_rows(raw, ape):
    """Pool a small aligned raw tile without expanding the live set."""
    pools = raw.shape[0] // 4
    raw = raw.astype(jnp.float32)
    keys = raw[:, :128].reshape((pools, 4, 128))
    scores = raw[:, 128:].reshape((pools, 4, 128)) + ape[None, :, :]
    scores -= jnp.max(scores, axis=1, keepdims=True)
    weight = jnp.exp(scores)
    weight /= jnp.sum(weight, axis=1, keepdims=True)
    return jnp.sum(weight * keys, axis=1).astype(jnp.bfloat16)


def pack_pools(pooled):
    """Pool BF16 -> rotate F32 -> FP8 keys and scale bytes."""
    rotated = _hadamard128(pooled.astype(jnp.float32)) * (128**-0.5)
    # Quantize the F32 rotation directly. Upstream's Triton writer rounds it to
    # BF16 first; that lowered GLM-5.3-Flash MTP acceptance on TPU, so it is omitted.
    amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0))
    q = jnp.clip(rotated / jnp.exp2(exponent), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    scale_bytes = jnp.where(
        jnp.arange(128)[None, :] == 0,
        (exponent + 127).astype(jnp.uint8),
        jnp.uint8(0),
    )
    # Each four adjacent pool rows belong to one physical destination quad.
    return jnp.concatenate(
        (pltpu.bitcast(q, jnp.uint32), pltpu.bitcast(scale_bytes, jnp.uint32)),
        axis=1,
    )
