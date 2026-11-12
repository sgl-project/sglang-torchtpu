# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas router top-k for small batches.

Sigmoid scores, the biased choice, k rounds of argmax-and-mask, the weight
gather and the renormalization in one VMEM-resident kernel. Ties pick the
lowest expert id. A single token (decode at batch 1) takes the same top-k by
rank instead (_topk_kernel_1tok): the k rounds are a serial chain of
reductions, ~0.2 us each on tpu7x, while the ranks are independent compares.
The op registration and the ``biased_topk_impl`` dispatch live in
sglang_tpu/moe/router_topk_tpu.py.
"""

import functools

# One token with at most this many experts (a multiple of 128) takes the rank
# formulation (_topk_kernel_1tok); its compares grow as experts^2.
_RANK_MAX_EXPERTS = 1024


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

    tokens, experts = logits_ref.shape
    if tokens == 1 and experts % 128 == 0 and k <= experts <= _RANK_MAX_EXPERTS:
        return _topk_kernel_1tok(
            logits_ref,
            bias_ref,
            w_ref,
            ids_ref,
            k=k,
            renormalize=renormalize,
            eps=eps,
            scale=scale,
        )
    logits = logits_ref[...]
    # sigmoid in f32, rounded to the logits dtype like torch's bf16 sigmoid.
    scores = (
        (1.0 / (1.0 + jnp.exp(-logits.astype(jnp.float32))))
        .astype(logits.dtype)
        .astype(jnp.float32)
    )
    choice = scores + bias_ref[...].reshape(1, -1).astype(jnp.float32)
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
        # The bias stays 1-D: reshaping it to [1, E] outside the kernel is a
        # relayout op per call (f32 [E] and [1, E] tile differently).
        in_specs=[full((tokens, experts)), full((experts,))],
        out_specs=(full((tokens, k)), full((tokens, k))),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=4 * 1024 * 1024,
            allow_input_fusion=[True, False],
        ),
        interpret=interpret,
        name=f"router_topk_{k}_of_{experts}",
    )(logits, bias.reshape(experts).astype(jnp.float32))


_ROWMAX_NEG = -3.0e38
_ROWMAX_MAX_BLOCK_ROWS = 512


def _rowmax_select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """One block of rows: top ``topk`` by repeated max-and-mask (ties pick lowest id)."""
    import jax
    import jax.numpy as jnp

    x = jnp.where(x_ref[...] > _ROWMAX_NEG, x_ref[...], _ROWMAX_NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        w_ref[:, j] = m[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], masked, x)
    w_ref[...] = jnp.where(w_ref[:, :1] <= _ROWMAX_NEG, jnp.nan, w_ref[...])


def rowmax_select_jax(scores, topk: int, interpret: bool = False):
    """Sort-free top-k over ``[rows, experts]`` f32 scores, returning (f32, i32)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    rows, n = scores.shape
    block = min(_ROWMAX_MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(_rowmax_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
        name=f"router_rowmax_topk_{topk}_of_{n}",
    )(scores)
