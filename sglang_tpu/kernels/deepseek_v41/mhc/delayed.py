# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/models/vllm/deepseek_v41/common/mhc.py
# Revision: bdb38700dd25959c029c7d77f00cee91ccd3a0bc
# SPDX-License-Identifier: Apache-2.0
"""Shifted mHC kernels; attention/FFN RMSNorm remains a separate call."""

import functools

import jax
import jax.numpy as jnp


def _pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    V4.1 keeps Sinkhorn in a compact device loop.
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)

    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


def delayed_pre_jax(
    residual: jax.Array,
    x: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mult, dim = residual.shape[-2:]
    if fn.dtype == jnp.bfloat16:
        # Load-time packed weight (see `pack_mix_weight`).
        mixes = _packed_mixes(x, fn, mult * (mult + 2))
    else:
        mixes = jnp.matmul(
            x.astype(jnp.float32), fn.T, precision=jax.lax.Precision.HIGHEST
        )
    x = x.astype(jnp.float32)
    # Upstream's first-layer broadcast optimization folds x/fn to hidden_size.
    # Gate normalization must still use the actual projection input width.
    squares = jnp.mean(jnp.square(x), -1, keepdims=True) * (mult * dim)
    new_pre, post, comb = _pre_gates(
        mixes,
        squares,
        mult,
        dim,
        scale,
        base,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    value = (previous_pre[..., None] * residual.astype(jnp.float32)).sum(1)
    return post[..., None], comb, value.astype(residual.dtype), new_pre


def _truncate_bf16(values: jax.Array) -> jax.Array:
    # A mask, not an f32->bf16->f32 round trip: XLA's excess-precision
    # rewrite may drop the round trip and zero the residual parts.
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _pack_mix_weight(fn: jax.Array) -> jax.Array:
    """FP32 ``fn [n, K]`` as BF16 ``[K, 128]`` columns ``hi | mid | lo | 0``.

    ``hi + mid + lo == fn`` exactly, so one BF16 matmul against BF16 inputs
    matches the FP32 projection up to accumulation order.
    """
    high = _truncate_bf16(fn)
    middle = _truncate_bf16(fn - high)
    low = fn - high - middle
    parts = jnp.concatenate([high, middle, low]).astype(jnp.bfloat16)
    return jnp.pad(parts.T, ((0, 0), (0, 128 - parts.shape[0])))


def _packed_mixes(x: jax.Array, weight: jax.Array, mixes: int) -> jax.Array:
    """Mix logits from a packed weight, summed in `_flat_pre_sums` order."""
    parts = jnp.dot(x, weight, preferred_element_type=jnp.float32)
    return (
        parts[:, :mixes] + parts[:, mixes : 2 * mixes] + parts[:, 2 * mixes : 3 * mixes]
    )


def _flat_pre_sums(flat, weight, previous_pre, mixes, tile):
    """One pass over ``flat [T, M*D]``: mix logits, square sums, collapse."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(x_ref, w_ref, pre_ref, mix_ref, square_ref, value_ref):
        pre = pre_ref[...]
        terms, sums = [], []
        for m in range(mult):
            x = x_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            terms.append(pre[:, m : m + 1] * x)
            sums.append(jnp.sum(x * x, axis=1, keepdims=True))
        value_ref[...] = functools.reduce(jnp.add, terms).astype(value_ref.dtype)
        square_ref[...] = functools.reduce(jnp.add, sums)
        parts = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
    )(flat, weight, previous_pre)


def _collapse_flat(flat: jax.Array, previous_pre: jax.Array) -> jax.Array:
    """``(previous_pre[..., None] * flat.view(T, M, D).f32).sum(1)`` from slices.

    The ``[T, M, D]`` FP32 view costs XLA a relayout copy of the whole
    residual (``reshape`` to ``T(4,128)``) before the reduce. Summing the
    ``D``-wide column slices avoids it. For ``M == 4`` the sum uses the order
    XLA's reduce applies to that view, ``(s0 + s2) + (s1 + s3)``, so the result
    is bitwise equal. Other ``M`` keep the view.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    if mult != 4:
        residual = flat.reshape(tokens, mult, dim).astype(jnp.float32)
        return (previous_pre[..., None] * residual).sum(1)
    terms = [
        previous_pre[:, m : m + 1]
        * flat[:, m * dim : (m + 1) * dim].astype(jnp.float32)
        for m in range(mult)
    ]
    return (terms[0] + terms[2]) + (terms[1] + terms[3])


def _delayed_pre_flat_small(
    flat: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    *constants,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """XLA `delayed_pre_jax` on a flat residual, collapsed by `_collapse_flat`.

    The value `delayed_pre_jax` derives from the ``[T, M, D]`` view is unused
    and dead-code eliminated, so the view's relayout copy disappears.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    post, comb, _, new_pre = delayed_pre_jax(
        flat.reshape(tokens, mult, width // mult),
        flat,
        fn,
        scale,
        base,
        previous_pre,
        *constants,
    )
    value = _collapse_flat(flat, previous_pre).astype(flat.dtype)
    return post, comb, value, new_pre


def delayed_pre_flat_jax(
    flat: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """`delayed_pre_jax` with ``x = residual = flat [T, M*D]``.

    Prefill-sized batches use one Pallas pass; viewing the flat residual as
    ``[T, M, D]`` costs XLA a relayout copy plus an FP32 copy for the
    projection. Small (decode) batches keep the XLA path.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    if 3 * mixes > 128 or (
        tokens not in (64, 96, 128, 256) and (tokens < 1024 or tokens % 64 != 0)
    ):
        return _delayed_pre_flat_small(flat, fn, scale, base, previous_pre, *constants)
    tile = tokens if tokens <= 128 else (128 if tokens == 256 else 64)
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    mixes, square, value = _flat_pre_sums(flat, weight, previous_pre, mixes, tile)
    new_pre, post, comb = _pre_gates(mixes, square, mult, dim, scale, base, *constants)
    return post[..., None], comb, value, new_pre


def _post_streams_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> list[jax.Array]:
    """XLA 2D stream recombine matching `_post_kernel` accumulation order."""
    mult = comb_res_mix.shape[-1]
    dim = flat.shape[-1] // mult
    x_f32 = x.reshape(-1, dim).astype(flat.dtype).astype(jnp.float32)
    post = post_layer_mix.reshape(-1, mult)
    comb = comb_res_mix.reshape(-1, mult, mult)
    streams = [
        flat[:, i * dim : (i + 1) * dim].astype(jnp.float32) for i in range(mult)
    ]
    out_streams = []
    for j in range(mult):
        acc = post[:, j : j + 1] * x_f32
        for i in range(mult):
            acc = acc + comb[:, i, j : j + 1] * streams[i]
        out_streams.append(acc.astype(flat.dtype))
    return out_streams


def _post_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Concatenated `[T, M*D]` residual from `_post_streams_jax`."""
    return jnp.concatenate(
        _post_streams_jax(x, flat, post_layer_mix, comb_res_mix), axis=1
    )


def _post_and_collapse_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    previous_pre: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused stream recombine and collapse before `optimization_barrier`.

    Collapsing directly from the recombined stream slices avoids re-reading
    `flat_new [T, M*D]` from HBM in a separate `_collapse_flat` fusion after
    the barrier while preserving `_collapse_flat`'s `(s0 + s2) + (s1 + s3)`
    accumulation order.
    """
    out_streams = _post_streams_jax(x, flat, post_layer_mix, comb_res_mix)
    flat_new = jnp.concatenate(out_streams, axis=1)
    terms = [
        previous_pre[:, m : m + 1] * s.astype(jnp.float32)
        for m, s in enumerate(out_streams)
    ]
    value = (
        (terms[0] + terms[2]) + (terms[1] + terms[3])
        if len(terms) == 4
        else functools.reduce(jnp.add, terms)
    )
    return flat_new, value.astype(flat.dtype)


def _flat_post_pre_sums(
    sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre, mixes, tile
):
    """One Pallas pass fusing `mhc_post_2d` with `_flat_pre_sums`."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(
        sub_x_ref,
        res_ref,
        post_ref,
        comb_ref,
        w_ref,
        pre_ref,
        out_res_ref,
        mix_ref,
        square_ref,
        value_ref,
    ):
        sub_x_f32 = sub_x_ref[...].astype(jnp.float32)
        post = post_ref[...]
        comb = comb_ref[...]
        old_bf = [res_ref[:, i * dim : (i + 1) * dim] for i in range(mult)]
        for j in range(mult):
            acc = post[:, j : j + 1] * sub_x_f32
            for i in range(mult):
                k = i * mult + j
                acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
            out_res_ref[:, j * dim : (j + 1) * dim] = acc.astype(out_res_ref.dtype)
        pre = pre_ref[...]
        terms, sums = [], []
        for m in range(mult):
            x = out_res_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            terms.append(pre[:, m : m + 1] * x)
            sums.append(jnp.sum(x * x, axis=1, keepdims=True))
        value_ref[...] = functools.reduce(jnp.add, terms).astype(value_ref.dtype)
        square_ref[...] = functools.reduce(jnp.add, sums)
        parts = jnp.dot(
            out_res_ref[...], w_ref[...], preferred_element_type=jnp.float32
        )
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, dim), row),
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mult), row),
            pl.BlockSpec((tile, mult * mult), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, width), flat.dtype),
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
    )(sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre)


def delayed_post_pre_flat_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused `post` + `delayed_pre_flat_jax` seam for `shifted_post_pre`."""
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    if 3 * mixes > 128 or (
        tokens not in (64, 96, 128, 256) and (tokens < 1024 or tokens % 64 != 0)
    ):
        flat_new, value = jax.lax.optimization_barrier(
            _post_and_collapse_jax(
                x,
                flat,
                post_layer_mix,
                comb_res_mix,
                previous_pre,
            )
        )
        post, comb, _, new_pre = delayed_pre_jax(
            flat_new.reshape(tokens, mult, dim),
            flat_new,
            fn,
            scale,
            base,
            previous_pre,
            *constants,
        )
        return flat_new, post, comb, value, new_pre
    tile = tokens if tokens <= 128 else (128 if tokens == 256 else 64)
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    flat_new, mixes_out, square, value = _flat_post_pre_sums(
        x.reshape(tokens, dim).astype(flat.dtype),
        flat,
        post_layer_mix.reshape(tokens, mult),
        comb_res_mix.reshape(tokens, mult * mult),
        weight,
        previous_pre,
        mixes,
        tile,
    )
    new_pre, post, comb = _pre_gates(
        mixes_out, square, mult, dim, scale, base, *constants
    )
    return flat_new, post[..., None], comb, value, new_pre
