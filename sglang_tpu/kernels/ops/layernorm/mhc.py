# SPDX-License-Identifier: Apache-2.0
"""Fused mHC (manifold-constrained hyper-connection) site kernel.

One Pallas call per mHC site replaces the torch reference's ~80 small XLA ops:
the previous sublayer's hc_post (stream remix) and the next sublayer's hc_pre
(RMS-scaled 4*d -> 24 projection, sigmoid gates, softmax plus Sinkhorn on the
4x4 mix, stream collapse). Gate and Sinkhorn math follows sglang-jax
sgl_jax/srt/kernels/mhc/mhc.py (63584e276f4b); the post/pre seam follows
tpu-inference kernels/experimental/deepseek_v4/mhc/fused_post_pre_kernel.py.

Layouts: streams are [T, hc * d] bf16 (stream-major), post is [T, hc] f32 and
comb is [T, hc * hc] f32 row-major in (i, j), matching
y[t, j] = post[t, j] * x[t] + sum_i comb[t, i, j] * res[t, i].
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# [1, VEC_WIDTH] f32 parameter row: hc_base in [0, mix), hc_scale in [mix, mix + 3).
VEC_WIDTH = 128
# Prefill token block; smaller batches run as one block.
_MAX_BLOCK_TOKENS = 128
_PREFILL_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def param_row(hc_base, hc_scale):
    """The [1, VEC_WIDTH] row of one site's hc_base and hc_scale."""
    row = jnp.concatenate([hc_base.astype(jnp.float32), hc_scale.astype(jnp.float32)])
    return jnp.pad(row, (0, VEC_WIDTH - row.shape[0])).reshape(1, VEC_WIDTH)


def _remix(x_ref, res_ref, post_ref, comb_ref, new_res_ref, *, hc, d):
    """hc_post: returns the new streams as f32 values of their bf16 rounding."""
    x = x_ref[...].astype(jnp.float32)
    post = post_ref[...]
    comb = comb_ref[...]
    old = [res_ref[:, i * d : (i + 1) * d].astype(jnp.float32) for i in range(hc)]
    streams = []
    for j in range(hc):
        acc = post[:, j : j + 1] * x
        for i in range(hc):
            acc = acc + comb[:, i * hc + j : i * hc + j + 1] * old[i]
        rounded = acc.astype(new_res_ref.dtype)
        new_res_ref[:, j * d : (j + 1) * d] = rounded
        streams.append(rounded.astype(jnp.float32))
    return streams


def _gates_and_collapse(
    streams,
    fn_ref,
    vec_ref,
    norm_ref,
    layer_ref,
    post_ref,
    comb_ref,
    *,
    hc,
    d,
    iters,
    rms_eps,
    hc_eps,
    norm_eps,
):
    """hc_pre on f32 streams: writes the collapsed layer input (RMS-normalized
    by the sublayer's input norm when norm_ref is given), post and comb."""
    mix = (2 + hc) * hc
    sq = functools.reduce(
        jnp.add, [jnp.sum(s * s, axis=-1, keepdims=True) for s in streams]
    )
    rms = jax.lax.rsqrt(sq / (hc * d) + rms_eps)
    dims = (((1,), (1,)), ((), ()))

    def project(s, w):
        return jax.lax.dot_general(s, w, dims, preferred_element_type=jnp.float32)

    # The streams are bf16-exact; the f32 weight is split into bf16 hi + lo, so
    # two bf16 passes with f32 accumulation track the f32 projection.
    terms = []
    for i, s in enumerate(streams):
        s = s.astype(jnp.bfloat16)
        w = fn_ref[:, i * d : (i + 1) * d]
        hi = w.astype(jnp.bfloat16)
        lo = (w - hi.astype(jnp.float32)).astype(jnp.bfloat16)
        terms += [project(s, hi), project(s, lo)]
    mixes = functools.reduce(jnp.add, terms) * rms
    vec = vec_ref[...]
    base = vec[:, :mix]
    s0, s1, s2 = (
        vec[:, mix : mix + 1],
        vec[:, mix + 1 : mix + 2],
        vec[:, mix + 2 : mix + 3],
    )

    pre = jax.nn.sigmoid(mixes[:, :hc] * s0 + base[:, :hc]) + hc_eps
    post_ref[...] = 2.0 * jax.nn.sigmoid(
        mixes[:, hc : 2 * hc] * s1 + base[:, hc : 2 * hc]
    )

    # Tokens on lanes: row and column sums are elementwise adds of hc rows.
    logits = (mixes[:, 2 * hc :] * s2 + base[:, 2 * hc :]).T  # [hc * hc, bt]
    c = [[logits[i * hc + j : i * hc + j + 1] for j in range(hc)] for i in range(hc)]

    def row_sum(r):
        return functools.reduce(jnp.add, r)

    for i in range(hc):
        top = functools.reduce(jnp.maximum, c[i])
        e = [jnp.exp(v - top) for v in c[i]]
        total = row_sum(e)
        c[i] = [v / total + hc_eps for v in e]

    def col_norm(c):
        cols = [row_sum([c[i][j] for i in range(hc)]) + hc_eps for j in range(hc)]
        return [[c[i][j] / cols[j] for j in range(hc)] for i in range(hc)]

    c = col_norm(c)
    for _ in range(iters - 1):
        c = [[v / (row_sum(c[i]) + hc_eps) for v in c[i]] for i in range(hc)]
        c = col_norm(c)
    comb_ref[...] = jnp.concatenate([v for r in c for v in r], axis=0).T

    layer = functools.reduce(
        jnp.add, [pre[:, i : i + 1] * s for i, s in enumerate(streams)]
    ).astype(layer_ref.dtype)
    if norm_ref is not None:
        # Matches RMSNorm.forward_native on the bf16 collapse.
        x = layer.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + norm_eps)
        layer = (x * norm_ref[...]).astype(layer_ref.dtype)
    layer_ref[...] = layer


def _site_kernel(
    *refs, hc, d, has_post, has_pre, has_norm, iters, rms_eps, hc_eps, norm_eps
):
    refs = list(refs)
    if has_post:
        x_ref, res_ref, post_in_ref, comb_in_ref = refs[:4]
        refs = refs[4:]
    else:
        res_ref = refs.pop(0)
    norm_ref = None
    if has_pre:
        fn_ref, vec_ref = refs[:2]
        refs = refs[2:]
        if has_norm:
            norm_ref = refs.pop(0)
    outs = refs
    if has_post:
        new_res_ref = outs.pop(0)
        streams = _remix(
            x_ref, res_ref, post_in_ref, comb_in_ref, new_res_ref, hc=hc, d=d
        )
    else:
        streams = [
            res_ref[:, i * d : (i + 1) * d].astype(jnp.float32) for i in range(hc)
        ]
    if has_pre:
        layer_ref, post_ref, comb_ref = outs
        _gates_and_collapse(
            streams,
            fn_ref,
            vec_ref,
            norm_ref,
            layer_ref,
            post_ref,
            comb_ref,
            hc=hc,
            d=d,
            iters=iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        )


def mhc_site(
    x,
    res,
    post,
    comb,
    fn,
    vec,
    *,
    hc: int,
    iters: int,
    rms_eps: float,
    hc_eps: float,
    norm_weight=None,
    norm_eps: float = 0.0,
):
    """One mHC site. x/post/comb are None for the first hc_pre; fn/vec are None
    for the final hc_post. norm_weight ([d], the next sublayer's RMSNorm) makes
    layer_input the normalized sublayer input. Returns the outputs present among
    (new_res [T, hc*d], layer_input [T, d], post [T, hc], comb [T, hc*hc])."""
    has_post = x is not None
    has_pre = fn is not None
    tokens, width = res.shape
    d = width // hc
    mix = (2 + hc) * hc
    # A partial last block is safe: every reduction is within a token, and
    # Pallas drops the out-of-bounds rows' writes.
    bt = tokens if tokens <= _MAX_BLOCK_TOKENS else _MAX_BLOCK_TOKENS

    def rows_spec(cols):
        return pl.BlockSpec((bt, cols), lambda t: (t, 0))

    def full_spec(shape):
        return pl.BlockSpec(shape, lambda t: (0, 0))

    operands, in_specs, out_shape, out_specs = [], [], [], []
    if has_post:
        operands += [x, res, post.astype(jnp.float32), comb.astype(jnp.float32)]
        in_specs += [rows_spec(d), rows_spec(width), rows_spec(hc), rows_spec(hc * hc)]
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), res.dtype))
        out_specs.append(rows_spec(width))
    else:
        operands.append(res)
        in_specs.append(rows_spec(width))
    if has_pre:
        operands += [fn.astype(jnp.float32), vec]
        in_specs += [full_spec((mix, width)), full_spec((1, VEC_WIDTH))]
        if norm_weight is not None:
            operands.append(norm_weight.astype(jnp.float32).reshape(1, d))
            in_specs.append(full_spec((1, d)))
        out_shape += [
            jax.ShapeDtypeStruct((tokens, d), res.dtype),
            jax.ShapeDtypeStruct((tokens, hc), jnp.float32),
            jax.ShapeDtypeStruct((tokens, hc * hc), jnp.float32),
        ]
        out_specs += [rows_spec(d), rows_spec(hc), rows_spec(hc * hc)]

    params = {"dimension_semantics": ("parallel",)}
    if bt > 16:
        params["vmem_limit_bytes"] = _PREFILL_VMEM_LIMIT_BYTES
    outs = pl.pallas_call(
        functools.partial(
            _site_kernel,
            hc=hc,
            d=d,
            has_post=has_post,
            has_pre=has_pre,
            has_norm=has_pre and norm_weight is not None,
            iters=iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        ),
        grid=(pl.cdiv(tokens, bt),),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(**params),
        name=f"mhc_{'post' if has_post else ''}{'pre' if has_pre else ''}",
    )(*operands)
    return tuple(outs)
