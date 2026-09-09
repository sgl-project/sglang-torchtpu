# SPDX-License-Identifier: Apache-2.0

"""Shared attention-entry QKVOG epilogue semantics.

This module is deliberately free of sharding and communication policy.  It is
used by the pure-JAX oracle and by transitional schedule implementations so
all paths agree on component boundaries, Q/K normalization, query scaling,
RoPE, and final output dtypes.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ._contract import (
    HEAD_DIM,
    KEY_COLUMNS,
    KV_HEADS_PER_DESTINATION,
    OUTPUT_GATE_HEADS_PER_DESTINATION,
    PACKED_COLUMNS,
    QUERY_COLUMNS,
    QUERY_HEADS_PER_DESTINATION,
    VALUE_COLUMNS,
)


def _broadcast_rope_table(table: jax.Array, target_ndim: int) -> jax.Array:
    """Insert head/group axes immediately before the RoPE lane dimension."""
    while table.ndim < target_ndim:
        table = jnp.expand_dims(table, axis=-2)
    return table


def interleaved_rope(
    value: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply adjacent-pair RoPE to an array ending in ``[..., heads, 64]``."""
    if value.shape[-1] != HEAD_DIM:
        raise ValueError(f"RoPE requires head dimension {HEAD_DIM}, got {value.shape}")
    if cos.shape[-1] != HEAD_DIM or sin.shape != cos.shape:
        raise ValueError(
            f"cos and sin must have equal shapes ending in {HEAD_DIM}, "
            f"got {cos.shape} and {sin.shape}"
        )

    cos = _broadcast_rope_table(cos.astype(jnp.float32), value.ndim)
    sin = _broadcast_rope_table(sin.astype(jnp.float32), value.ndim)
    lane = jax.lax.broadcasted_iota(jnp.int32, value.shape, value.ndim - 1)
    rotated = jnp.where(
        lane % 2 == 0,
        -jnp.roll(value, -1, axis=-1),
        jnp.roll(value, 1, axis=-1),
    )
    return value * cos + rotated * sin


def gathered_interleaved_rope(
    value: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply adjacent-pair RoPE through one lane gather.

    This spelling is intended for Pallas kernels: XORing the lane index with
    one directly selects each lane's adjacent partner, avoiding the two rolls
    and full-value select used by the portable JAX reference.
    """
    if value.shape[-1] != HEAD_DIM:
        raise ValueError(f"RoPE requires head dimension {HEAD_DIM}, got {value.shape}")
    if cos.shape[-1] != HEAD_DIM or sin.shape != cos.shape:
        raise ValueError(
            f"cos and sin must have equal shapes ending in {HEAD_DIM}, "
            f"got {cos.shape} and {sin.shape}"
        )

    cos = _broadcast_rope_table(cos.astype(jnp.float32), value.ndim)
    sin = _broadcast_rope_table(sin.astype(jnp.float32), value.ndim)
    lane = jax.lax.broadcasted_iota(
        jnp.int32,
        value.shape,
        value.ndim - 1,
    )
    adjacent = jnp.take_along_axis(
        value,
        jax.lax.bitwise_xor(lane, jnp.int32(1)),
        axis=-1,
    )
    sin_lane = jax.lax.broadcasted_iota(
        jnp.int32,
        sin.shape,
        sin.ndim - 1,
    )
    signed_sin = sin * jnp.where(
        sin_lane % 2 == 0,
        jnp.float32(-1.0),
        jnp.float32(1.0),
    )
    return value * cos + adjacent * signed_sin


def qkvog_epilogue(
    projected: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps_qk: float,
    scale_query_by: float,
    use_qk_norm: bool,
    use_rope: bool,
    kv_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Split a packed destination group and produce attention-ready outputs.

    ``projected`` may be ``[tokens, 1152]`` or contain additional group axes,
    for example ``[local_tokens, 16, 1152]``.  ``cos`` and ``sin`` contain one
    full-width, pair-repeated RoPE row per token; missing group/head axes are
    broadcast automatically.

    Casting the projection to BF16 before Q/K normalization intentionally
    preserves the numerical boundary of the unfused JAX model path.
    """
    if projected.shape[-1] != PACKED_COLUMNS:
        raise ValueError(
            f"packed QKVOG width must be {PACKED_COLUMNS}, got {projected.shape}"
        )
    kv_dtype = jnp.dtype(kv_dtype)
    if kv_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float8_e4m3fn)):
        raise ValueError(f"unsupported K/V output dtype: {kv_dtype}")

    projected = projected.astype(jnp.bfloat16)
    q_end = QUERY_COLUMNS
    k_end = q_end + KEY_COLUMNS
    v_end = k_end + VALUE_COLUMNS
    prefix = projected.shape[:-1]

    q = projected[..., :q_end].reshape(*prefix, QUERY_HEADS_PER_DESTINATION, HEAD_DIM)
    k = projected[..., q_end:k_end].reshape(*prefix, KV_HEADS_PER_DESTINATION, HEAD_DIM)
    v = projected[..., k_end:v_end].reshape(*prefix, KV_HEADS_PER_DESTINATION, HEAD_DIM)
    og = projected[..., v_end:].reshape(
        *prefix, OUTPUT_GATE_HEADS_PER_DESTINATION, HEAD_DIM
    )

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    if use_qk_norm:
        q_inverse_rms = jax.lax.rsqrt(jnp.mean(q * q, axis=-1, keepdims=True) + eps_qk)
        k_inverse_rms = jax.lax.rsqrt(jnp.mean(k * k, axis=-1, keepdims=True) + eps_qk)
        q = q * q_inverse_rms * scale_query_by
        k = k * k_inverse_rms
    if use_rope:
        q = interleaved_rope(q, cos, sin)
        k = interleaved_rope(k, cos, sin)

    return (
        q.astype(jnp.bfloat16),
        k.astype(kv_dtype),
        v.astype(kv_dtype),
        og.astype(jnp.bfloat16),
    )


def split_epilogued_qkvog(
    projected: jax.Array,
    *,
    kv_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Split packed BF16 values after Q/K math has already been applied."""
    if projected.shape[-1] != PACKED_COLUMNS:
        raise ValueError(
            f"packed QKVOG width must be {PACKED_COLUMNS}, got {projected.shape}"
        )
    kv_dtype = jnp.dtype(kv_dtype)
    if kv_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float8_e4m3fn)):
        raise ValueError(f"unsupported K/V output dtype: {kv_dtype}")

    projected = projected.astype(jnp.bfloat16)
    prefix = projected.shape[:-1]
    k_start = QUERY_COLUMNS
    v_start = k_start + KEY_COLUMNS
    og_start = v_start + VALUE_COLUMNS
    return (
        projected[..., :k_start].reshape(
            *prefix, QUERY_HEADS_PER_DESTINATION, HEAD_DIM
        ),
        projected[..., k_start:v_start]
        .reshape(*prefix, KV_HEADS_PER_DESTINATION, HEAD_DIM)
        .astype(kv_dtype),
        projected[..., v_start:og_start]
        .reshape(*prefix, KV_HEADS_PER_DESTINATION, HEAD_DIM)
        .astype(kv_dtype),
        projected[..., og_start:].reshape(
            *prefix, OUTPUT_GATE_HEADS_PER_DESTINATION, HEAD_DIM
        ),
    )
