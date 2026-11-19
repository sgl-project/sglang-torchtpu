# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas router top-k.

Sigmoid scores, the biased choice, k rounds of argmax-and-mask, the weight
gather and the renormalization in one kernel, gridded over tiles of tokens.
Ties pick the lowest expert id. A single token (decode at batch 1) takes the
same top-k by rank instead (_topk_kernel_1tok): the k rounds are a serial
chain of reductions, ~0.2 us each on tpu7x, while the ranks are independent
compares. The op registration and the ``biased_topk_impl`` dispatch live in
sglang_tpu/moe/router_topk_tpu.py.
"""

import functools

# One token with at most this many experts (a multiple of 128) takes the rank
# formulation (_topk_kernel_1tok); its compares grow as experts^2.
_RANK_MAX_EXPERTS = 1024
# Tokens per grid step, measured on tpu7x over 128-1024 for 256, 288 and 896
# experts.
_BLOCK_TOKENS = 512


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
    """One tile of tokens: logits [bt, E] in, weights and ids [k, bt] out.

    Experts go down the sublanes and tokens across the lanes, so each round's
    max and lowest-tied-id are elementwise reductions over the expert vregs
    plus one sublane reduce, rather than a cross-lane reduce per token row.
    The expert iota is f32 (exact below 2^24): its min is cheaper than int32's.
    """
    import jax
    import jax.numpy as jnp

    logits = logits_ref[...]
    # sigmoid in f32, rounded to the logits dtype like torch's bf16 sigmoid.
    scores = (
        (1.0 / (1.0 + jnp.exp(-logits.astype(jnp.float32))))
        .astype(logits.dtype)
        .astype(jnp.float32)
    ).T
    experts, tokens = scores.shape
    # bias [E] (lanes) to a column [E, 1] (sublanes).
    bias = jnp.transpose(
        jnp.broadcast_to(bias_ref[...].reshape(1, experts), (8, experts))
    )[:, 0:1]
    choice = scores + bias
    expert = jax.lax.broadcasted_iota(jnp.int32, (experts, tokens), 0).astype(
        jnp.float32
    )
    total = jnp.zeros((1, tokens), jnp.float32)
    # Router logits are finite here (rows past the last token in the final tile
    # are never written back). A NaN logit would make `choice == m` false down
    # its column, so every id there would come out as `experts`.
    for r in range(k):
        m = jnp.max(choice, axis=0, keepdims=True)
        idx = jnp.min(
            jnp.where(choice == m, expert, jnp.float32(experts)),
            axis=0,
            keepdims=True,
        )
        hit = expert == idx
        w = jnp.sum(jnp.where(hit, scores, 0.0), axis=0, keepdims=True)
        w_ref[r : r + 1, :] = w
        ids_ref[r : r + 1, :] = idx.astype(jnp.int32)
        total = total + w
        choice = jnp.where(hit, -jnp.inf, choice)
    if renormalize:
        w_ref[...] = w_ref[...] * (scale / (total + eps))


def _topk_kernel_1tok(
    logits_ref, bias_ref, w_ref, ids_ref, *, k, renormalize, eps, scale
):
    """One token: the top-k by rank instead of k rounds of argmax-and-mask.

    rank[e] = #{j : choice[j] > choice[e] or (choice[j] == choice[e] and j < e)},
    i.e. e's position in (choice desc, expert asc) order, which is exactly the
    order the round loop picks experts in (ties -> lowest expert id). The k
    rounds are a serial chain of two cross-lane reductions each; the rank is
    one batch of independent compares, then slot r takes the expert of rank r.
    Bit-identical to the round loop for finite logits."""
    import jax
    import jax.numpy as jnp

    experts = logits_ref.shape[1]
    tiles = experts // 128
    logits = logits_ref[...]
    scores = (
        (1.0 / (1.0 + jnp.exp(-logits.astype(jnp.float32))))
        .astype(logits.dtype)
        .astype(jnp.float32)
    )
    choice = scores + bias_ref[...].reshape(1, -1).astype(jnp.float32)
    # choice[j] down the sublanes, broadcast across the lanes: [E, 128].
    col = jnp.transpose(jnp.broadcast_to(choice, (8, experts)))[:, 0:1]
    col = jnp.broadcast_to(col, (experts, 128))
    j_iota = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    e_iota = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    ranks = []
    for t in range(tiles):
        ce = jnp.broadcast_to(choice[:, t * 128 : (t + 1) * 128], (128, 128))
        cnt = jnp.zeros((8, 128), jnp.int32)
        for b in range(tiles):
            cj = col[b * 128 : (b + 1) * 128, :]
            if b < t:
                before = cj >= ce
            elif b > t:
                before = cj > ce
            else:
                before = (cj > ce) | ((cj == ce) & (j_iota < e_iota))
            cnt = cnt + jnp.sum(before.astype(jnp.int32).reshape(16, 8, 128), axis=0)
        ranks.append(jnp.sum(cnt, axis=0, keepdims=True))
    rank = jnp.concatenate(ranks, axis=1)  # [1, E]
    kp = max(8, -(-k // 8) * 8)
    slot = jax.lax.broadcasted_iota(jnp.int32, (kp, experts), 0)
    hit = jnp.broadcast_to(rank, (kp, experts)) == slot
    expert = jax.lax.broadcasted_iota(jnp.int32, (kp, experts), 1)
    ids_c = jnp.sum(jnp.where(hit, expert, 0), axis=1, keepdims=True)  # [kp, 1]
    w_c = jnp.sum(
        jnp.where(hit, jnp.broadcast_to(scores, (kp, experts)), 0.0),
        axis=1,
        keepdims=True,
    )
    ids = jnp.transpose(jnp.broadcast_to(ids_c, (kp, 128)))[0:1, 0:k]
    weights = jnp.transpose(jnp.broadcast_to(w_c, (kp, 128)))[0:1, 0:k]
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
    params = dict(k=k, renormalize=renormalize, eps=eps, scale=scale)
    # The bias stays 1-D: reshaping it to [1, E] outside the kernel is a
    # relayout op per call (f32 [E] and [1, E] tile differently).
    bias = bias.reshape(experts).astype(jnp.float32)
    bias_spec = pl.BlockSpec((experts,), lambda i: (0,))
    if tokens == 1 and experts % 128 == 0 and k <= experts <= _RANK_MAX_EXPERTS:
        out_spec = pl.BlockSpec((1, k), lambda i: (0, 0))
        return pl.pallas_call(
            functools.partial(_topk_kernel_1tok, **params),
            out_shape=(
                jax.ShapeDtypeStruct((1, k), jnp.float32),
                jax.ShapeDtypeStruct((1, k), jnp.int32),
            ),
            grid=(1,),
            in_specs=[pl.BlockSpec((1, experts), lambda i: (0, 0)), bias_spec],
            out_specs=(out_spec, out_spec),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=4 * 1024 * 1024,
                allow_input_fusion=[True, False],
            ),
            interpret=interpret,
            name=f"router_topk_{k}_of_{experts}",
        )(logits, bias)
    block = min(_BLOCK_TOKENS, tokens)
    out_spec = pl.BlockSpec((k, block), lambda i: (0, i))
    weights, ids = pl.pallas_call(
        functools.partial(_topk_kernel, **params),
        out_shape=(
            jax.ShapeDtypeStruct((k, tokens), jnp.float32),
            jax.ShapeDtypeStruct((k, tokens), jnp.int32),
        ),
        grid=(pl.cdiv(tokens, block),),
        in_specs=[pl.BlockSpec((block, experts), lambda i: (i, 0)), bias_spec],
        out_specs=(out_spec, out_spec),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=32 * 1024 * 1024,
            allow_input_fusion=[True, False],
        ),
        interpret=interpret,
        name=f"router_topk_{k}_of_{experts}",
    )(logits, bias)
    return weights.T, ids.T
