# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas router top-k for small batches.

Sigmoid scores, the biased choice, k rounds of argmax-and-mask, the weight
gather and the renormalization in one VMEM-resident kernel. Ties pick the
lowest expert id. The op registration and the ``biased_topk_impl`` dispatch
live in sglang_tpu/layers/moe/router_topk_tpu.py.
"""

import functools


def _topk_kernel(
    logits_ref,
    bias_ref,
    w_ref,
    ids_ref,
    *,
    k: int,
    renormalize: bool,
    eps: float,
    scale: float,
):
    import jax
    import jax.numpy as jnp

    logits = logits_ref[...]
    # sigmoid in f32, rounded to the logits dtype like torch's bf16 sigmoid.
    scores = (
        (1.0 / (1.0 + jnp.exp(-logits.astype(jnp.float32))))
        .astype(logits.dtype)
        .astype(jnp.float32)
    )
    choice = scores + bias_ref[...].astype(jnp.float32)
    tokens, experts = choice.shape
    lane = jnp.broadcast_to(
        jnp.arange(experts, dtype=jnp.int32)[None, :], (tokens, experts)
    )
    col = jnp.broadcast_to(jnp.arange(k, dtype=jnp.int32)[None, :], (tokens, k))
    ids = jnp.zeros((tokens, k), jnp.int32)
    weights = jnp.zeros((tokens, k), jnp.float32)
    neg = jnp.float32(-jnp.inf)
    packed = logits.dtype == jnp.bfloat16
    if packed:
        # bf16 scores fit in 16 bits: key = expert * 2^16 + score bits, so the
        # min over the tied keys gives the lowest tied expert and its weight
        # in one reduction.
        bits = jax.lax.bitcast_convert_type(scores, jnp.int32) >> 16
        key = lane * 65536 + (bits & 0xFFFF)
        no_key = jnp.int32(experts * 65536)
    # Router logits are finite here (padded rows hold finite values and are
    # masked by the caller afterwards). A NaN logit would make `choice == m`
    # false across its row, so every id there would come out as `experts`.
    for r in range(k):
        m = jnp.max(choice, axis=-1, keepdims=True)
        if packed:
            best = jnp.min(jnp.where(choice == m, key, no_key), axis=-1, keepdims=True)
            idx = best >> 16
            w = jax.lax.bitcast_convert_type((best & 0xFFFF) << 16, jnp.float32)
        else:
            idx = jnp.min(jnp.where(choice == m, lane, experts), axis=-1, keepdims=True)
            w = jnp.sum(jnp.where(lane == idx, scores, 0.0), axis=-1, keepdims=True)
        ids = jnp.where(col == r, idx, ids)
        weights = jnp.where(col == r, w, weights)
        choice = jnp.where(lane == idx, neg, choice)
    if renormalize:
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + eps)
        weights = weights * scale
    w_ref[...] = weights
    ids_ref[...] = ids


def router_topk_jax(
    logits,
    bias,
    k: int,
    renormalize: bool,
    eps: float,
    scale: float,
    interpret: bool = False,
):
    """logits: [T, E]; bias: [E] f32. Returns (f32 weights [T, k], i32 ids [T, k])."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, experts = logits.shape
    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    return pl.pallas_call(
        functools.partial(
            _topk_kernel, k=k, renormalize=renormalize, eps=eps, scale=scale
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, k), jnp.float32),
            jax.ShapeDtypeStruct((tokens, k), jnp.int32),
        ),
        grid=(1,),
        in_specs=[full((tokens, experts)), full((1, experts))],
        out_specs=(full((tokens, k)), full((tokens, k))),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=4 * 1024 * 1024,
            allow_input_fusion=[True, False],
        ),
        interpret=interpret,
        name=f"router_topk_{k}_of_{experts}",
    )(logits, bias.reshape(1, experts).astype(jnp.float32))
