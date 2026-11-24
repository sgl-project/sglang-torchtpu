# SPDX-License-Identifier: Apache-2.0
"""The MiniMax H3 ViT decoder's fp16 split linear as a JAX function and a TPU op.

Used by minimax_h3_decode (SGLANG_TPU_OPT_H3_VAE_FP16_SPLIT_MATMUL): an fp16
linear as four bf16 x bf16 products with float32 results, one MXU pass each
whatever the float32 matmul precision.
"""

# NOTE: no `from __future__ import annotations` here: torch_tpu's
# _infer_static_argnums identifies tensor arguments by `annotation is
# jax.Array`, which needs evaluated annotations on the registered function.

import threading

import jax
import jax.numpy as jnp
from jax import lax

_NT = (((1,), (1,)), ((), ()))
_OP = None
_OP_LOCK = threading.Lock()


def split_linear_jax(
    x: jax.Array, weight_hi: jax.Array, weight_lo: jax.Array, bias: jax.Array
) -> jax.Array:
    """fp16 [M, K] x (bf16 hi + bf16 lo) [N, K]^T + fp16 bias -> fp16 [M, N].

    x splits exactly into bf16 hi + lo (an infinity into (inf, 0)); the four
    products hi*hi + hi*lo + lo*hi + lo*lo (each exact) accumulate in float32
    (an infinite input's cross term is 0, as in its fp16 product), the bias is
    added in float32 and the sum rounded to fp16 once. The parts are plain
    bf16 arrays behind an optimization barrier: fused into the dots as
    operand computations, XLA's tiling of one dot degrades (compile-only
    tpu7x: 2.1M against 0.1M estimated cycles at the to_qkv shape).
    """
    finite = jnp.logical_not(jnp.isinf(x))
    hi = x.astype(jnp.bfloat16)
    lo = jnp.where(finite, x.astype(jnp.float32) - hi.astype(jnp.float32), 0.0)
    lo = lo.astype(jnp.bfloat16)
    hi_cross = jnp.where(finite, hi, jnp.zeros_like(hi))
    hi, hi_cross, lo = lax.optimization_barrier((hi, hi_cross, lo))

    def dot(a, b):
        return lax.dot_general(a, b, _NT, preferred_element_type=jnp.float32)

    acc = (dot(hi, weight_hi) + dot(hi_cross, weight_lo)) + (
        dot(lo, weight_hi) + dot(lo, weight_lo)
    )
    return (acc + bias.astype(jnp.float32)).astype(jnp.float16)


def op():
    """:func:`split_linear_jax` as a torch op on TPU, registered once."""
    global _OP
    if _OP is None:
        with _OP_LOCK:
            if _OP is None:
                from sglang_tpu.pallas_jax_op import cpu_export_jax_op

                _OP = cpu_export_jax_op(
                    "sglang_tpu::h3_fp16_split_linear", split_linear_jax
                )
    return _OP
