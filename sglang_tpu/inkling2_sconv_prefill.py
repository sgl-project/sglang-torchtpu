# SPDX-License-Identifier: Apache-2.0
"""Pallas prefill kernel for the in-forward taps of an Inkling short convolution (static weights).

y[t] = scale * (x[t] * use_residual + sum_lag w[lag] * x[t - lag] * (pos[t] >= lag)) in one pass over x.
Taps that reach before a sequence's first row (pos[t] < lag) are zero here; the caller adds the
conv-state contribution for those rows.
"""

# PEP 563 must remain disabled so `cpu_export_jax_op` sees `jax.Array` type
# objects rather than stringified annotations at registration time.

import functools
from threading import Lock

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_TILE_TOKENS = 256
_HALO = 8  # rows of the previous tile; covers kernel_size - 1 <= 8
MIN_TOKENS = 512
_VMEM_LIMIT = 48 * 1024 * 1024


def _kernel(
    halo_ref,
    x_ref,
    pos_ref,
    w_ref,
    out_ref,
    *,
    kernel_size: int,
    use_residual: bool,
    scale: float,
):
    x = x_ref[...].astype(jnp.float32)
    ext = jnp.concatenate([halo_ref[...].astype(jnp.float32), x], axis=0)
    pos = pos_ref[...]
    w = w_ref[...]
    acc = x * (w[0:1, :] + (1.0 if use_residual else 0.0))
    for lag in range(1, kernel_size):
        past = pltpu.roll(ext, shift=lag, axis=0)[_HALO:]
        acc = acc + w[lag : lag + 1, :] * jnp.where(pos >= lag, past, 0.0)
    out_ref[...] = (acc * scale).astype(out_ref.dtype)


def sconv_taps_jax(
    x: jax.Array, pos: jax.Array, w_lag: jax.Array, *, use_residual: bool, scale: float
) -> jax.Array:
    """x bf16 [T, D]; pos int32 [T, 1] (row index inside its sequence); w_lag fp32 [K, D] (lag 0 first)."""
    tokens, dim = x.shape
    tile = _TILE_TOKENS
    halo_blocks = tile // _HALO
    return pl.pallas_call(
        functools.partial(
            _kernel, kernel_size=w_lag.shape[0], use_residual=use_residual, scale=scale
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, dim), x.dtype),
        grid=(tokens // tile,),
        in_specs=[
            # The 8 rows before this tile; tile 0 reads its own first rows, which pos masks.
            pl.BlockSpec(
                (_HALO, dim), lambda i: (jnp.maximum(i * halo_blocks - 1, 0), 0)
            ),
            pl.BlockSpec((tile, dim), lambda i: (i, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
            pl.BlockSpec(w_lag.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tile, dim), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT
        ),
        name=f"inkling2_sconv_taps_k{w_lag.shape[0]}",
    )(x, x, pos, w_lag)


_OP = None
_OP_LOCK = Lock()


def register_sconv_taps_op():
    """Register the op before the model is traced; returns it."""
    global _OP
    with _OP_LOCK:
        if _OP is not None:
            return _OP
        # Lazy: the kernel itself must import without torch_tpu (raw-JAX unit test).
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _op(
            x: jax.Array,
            pos: jax.Array,
            w_lag: jax.Array,
            use_residual: bool,
            scale: float,
        ) -> jax.Array:
            return sconv_taps_jax(x, pos, w_lag, use_residual=use_residual, scale=scale)

        _OP = cpu_export_jax_op("sglang_tpu_pallas::inkling2_sconv_taps", _op)
        return _OP


def supports_sconv_taps(tokens: int, kernel_size: int) -> bool:
    return (
        tokens >= MIN_TOKENS and tokens % _TILE_TOKENS == 0 and kernel_size - 1 <= _HALO
    )


def sconv_taps_tpu(
    x: torch.Tensor,
    pos: torch.Tensor,
    w_lag: torch.Tensor,
    *,
    use_residual: bool,
    scale: float,
):
    return register_sconv_taps_op()(x, pos, w_lag, use_residual, scale)
