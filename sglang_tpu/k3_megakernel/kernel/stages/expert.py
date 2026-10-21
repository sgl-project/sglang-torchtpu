# SPDX-License-Identifier: Apache-2.0
"""Routed-expert math on resident VMEM weights (one expert, one lane's channels).

gate/up: e4m3 ``[K, 2C]`` with one power-of-two scale per column, so a
single bf16 x e4m3 MXU product and a column multiply give the exact sum.
down: packed MXFP4 ``[C/2, N]`` (sglang's gmm layout) with E8M0 exponents
per 32 K-rows. The activation is copied once per 32-row group with the other
groups masked, so one MXU pass yields one partial per group; the partials
are scaled and summed on the VPU.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

_DN = (((1,), (0,)), ((), ()))


def situ_and_mul(gate_up: jax.Array, *, beta: float, linear_beta: float) -> jax.Array:
    """``[R, 2C]`` bf16 gate|up -> ``[R, C]`` bf16, in f32 like the reference."""
    c = gate_up.shape[-1] // 2
    g = gate_up[:, :c].astype(jnp.float32)
    u = gate_up[:, c:].astype(jnp.float32)
    a = beta * jnp.tanh(g / beta) * jax.nn.sigmoid(g)
    u = linear_beta * jnp.tanh(u / linear_beta)
    return (a * u).astype(jnp.bfloat16)


def gate_up_dot(x: jax.Array, w_ref, scale_ref) -> jax.Array:
    """``x`` bf16 ``[R, K]``, ``w_ref`` e4m3 ``[K, 2C]``, ``scale_ref`` f32 ``[1, 2C]``
    -> bf16 ``[R, 2C]`` (rounded like a bf16 linear layer)."""
    acc = jax.lax.dot_general(x, w_ref[...], _DN, preferred_element_type=jnp.float32)
    return (acc * scale_ref[...]).astype(jnp.bfloat16)


def _e8m0_to_f32(exponents: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(exponents.astype(jnp.uint32) << 23, jnp.float32)


def _group_rows(a: jax.Array) -> jax.Array:
    """``[R, C]`` -> ``[R * C/32, C]``: row ``(r, g)`` keeps only group ``g``'s 32 channels."""
    rows, c = a.shape
    groups = c // 32
    group_of = jax.lax.broadcasted_iota(jnp.int32, (groups, c), 1) // 32
    mask = group_of == jax.lax.broadcasted_iota(jnp.int32, (groups, c), 0)
    left = jnp.where(mask[None], a[:, None, :], jnp.zeros((), a.dtype))
    return left.reshape(rows * groups, c)


def _scaled_group_sum(part: jax.Array, rows: int, exp: jax.Array) -> jax.Array:
    groups = exp.shape[0]
    part = part[: rows * groups].reshape(rows, groups, -1)
    return jnp.sum(part * _e8m0_to_f32(exp)[None], axis=1)


def down_dot_pair(
    a0: jax.Array, a1: jax.Array, packed_ref, exp_ref
) -> tuple[jax.Array, jax.Array]:
    """``down_dot`` of two experts in one MXU pass: ``packed_ref`` u8 ``[2, C/2, N]`` and
    ``exp_ref`` u8 ``[2, C/32, N]`` hold the two experts' weights back to back; the left
    operand is block-diagonal, so each expert's rows see only its own K rows."""
    rows, c = a0.shape
    w = pltpu.bitcast(
        packed_ref.bitcast(jnp.uint32)[...], jnp.float4_e2m1fn
    )  # [2, C, N]
    w = w.astype(jnp.float8_e4m3fn).reshape(2 * c, -1)
    l0, l1 = _group_rows(a0), _group_rows(a1)
    z = jnp.zeros(l0.shape, l0.dtype)
    left = jnp.concatenate(
        [jnp.concatenate([l0, z], axis=1), jnp.concatenate([z, l1], axis=1)], axis=0
    )
    pad = (-left.shape[0]) % 16
    if pad:
        left = jnp.concatenate([left, jnp.zeros((pad, 2 * c), left.dtype)], axis=0)
    part = jax.lax.dot_general(left, w, _DN, preferred_element_type=jnp.float32)
    n0 = l0.shape[0]
    exp = exp_ref[...]
    return _scaled_group_sum(part, rows, exp[0]), _scaled_group_sum(
        part[n0:], rows, exp[1]
    )


def down_dot(a: jax.Array, packed_ref, exp_ref) -> jax.Array:
    """``a`` bf16 ``[R, C]``, ``packed_ref`` u8 ``[C/2, N]``, ``exp_ref`` u8
    ``[C/32, N]`` -> f32 ``[R, N]`` partial (this lane's C channels)."""
    rows, c = a.shape
    groups = c // 32
    w = pltpu.bitcast(packed_ref.bitcast(jnp.uint32)[...], jnp.float4_e2m1fn)
    w = w.astype(jnp.float8_e4m3fn)  # exact: every E2M1 value is an e4m3 value
    group_of = jax.lax.broadcasted_iota(jnp.int32, (groups, c), 1) // 32
    mask = group_of == jax.lax.broadcasted_iota(jnp.int32, (groups, c), 0)
    left = jnp.where(mask[None], a[:, None, :], jnp.zeros((), a.dtype))
    left = left.reshape(rows * groups, c)
    pad = (-left.shape[0]) % 16
    if pad:
        left = jnp.concatenate([left, jnp.zeros((pad, c), left.dtype)], axis=0)
    part = jax.lax.dot_general(left, w, _DN, preferred_element_type=jnp.float32)
    part = part[: rows * groups].reshape(rows, groups, -1)
    return jnp.sum(part * _e8m0_to_f32(exp_ref[...])[None], axis=1)


def down_dot_dequant(a: jax.Array, packed_ref, exp_ref) -> jax.Array:
    """``down_dot`` with the e8m0 group scales applied to the weights (bf16, exact: FP4 values times
    powers of two) and one M=R dot; the f32 sum over K runs in the MXU's order instead of per group."""
    rows, c = a.shape
    groups = c // 32
    w = pltpu.bitcast(packed_ref.bitcast(jnp.uint32)[...], jnp.float4_e2m1fn)
    w = w.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    scale = _e8m0_to_f32(exp_ref[...]).astype(jnp.bfloat16)  # [groups, N]
    n = scale.shape[-1]
    scale = jnp.broadcast_to(scale[:, None, :], (groups, 32, n)).reshape(c, n)
    return jax.lax.dot_general(a, w * scale, _DN, preferred_element_type=jnp.float32)
