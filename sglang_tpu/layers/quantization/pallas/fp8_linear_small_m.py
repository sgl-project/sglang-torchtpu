# SPDX-License-Identifier: Apache-2.0
"""Single-kernel W8A8 dense linear for decode-sized batches.

Replaces apply_tpu_fp8_linear's XLA chain (row pad, per-row amax, scale, FP8
cast, f32 scaled_mm, row/column rescale, narrow, cast; ~6 ops per linear) with
one Pallas call. Same math: per-row scale amax / fp8_max floored at 1e-12, an
unsaturated FP8 cast, an f32-accumulated FP8 dot, then x_scale * w_scale.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows handled by the kernel; larger batches keep the XLA path.
MAX_TOKENS = 16
_TILE_N = (1024, 512, 256, 128)
# Double-buffered FP8 weight tile bytes; larger shapes keep the XLA path.
_WEIGHT_TILE_BUDGET = 24 * 1024 * 1024


def _tile_n(k: int, n: int):
    return next(
        (t for t in _TILE_N if n % t == 0 and 2 * k * t <= _WEIGHT_TILE_BUDGET), None
    )


def supports(tokens: int, k: int, n: int) -> bool:
    return 0 < tokens <= MAX_TOKENS and k % 128 == 0 and _tile_n(k, n) is not None


def _kernel(x_ref, w_ref, ws_ref, o_ref, *, fp8_dtype, fp8_max):
    x = x_ref[...].astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True) / fp8_max, 1e-12)
    x_q = (x / x_scale).astype(fp8_dtype)
    acc = jax.lax.dot_general(
        x_q,
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    o_ref[...] = (acc * x_scale * ws_ref[...]).astype(o_ref.dtype)


def fp8_linear_small_m(
    x: jax.Array, weight: jax.Array, weight_scale: jax.Array
) -> jax.Array:
    """x [M, K] (M <= MAX_TOKENS), weight [K, N] FP8, weight_scale [N] or scalar."""
    tokens, k = x.shape
    n = weight.shape[1]
    tile_n = _tile_n(k, n)
    # The row block spans the whole token dim, so no pad op.
    ws = jnp.broadcast_to(weight_scale.astype(jnp.float32).reshape(-1), (n,)).reshape(
        1, n
    )
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            fp8_dtype=weight.dtype,
            fp8_max=float(jnp.finfo(weight.dtype).max),
        ),
        grid=(n // tile_n,),
        in_specs=[
            pl.BlockSpec((tokens, k), lambda j: (0, 0)),
            pl.BlockSpec((k, tile_n), lambda j: (0, j)),
            pl.BlockSpec((1, tile_n), lambda j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tokens, tile_n), lambda j: (0, j)),
        out_shape=jax.ShapeDtypeStruct((tokens, n), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"fp8_linear_small_m_{k}x{n}",
    )(x, weight, ws)
    return out
