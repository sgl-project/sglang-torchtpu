# SPDX-License-Identifier: Apache-2.0
"""Fused DeepSeek-V4.1 mHC post-pre kernel (one sublayer boundary).

One Pallas call per sublayer boundary, in V4.1 ordering
(``hc_pre_from_prev_sublayer``): the previous sublayer's post rebuilds the
streams, the streams collapse with that sublayer's own pre (an input), the next
sublayer's input RMSNorm runs, and the next sublayer's coefficients (pre, post,
comb) are computed from the new streams (outputs). This is the token-tiled site
kernel of sglang_tpu/kernels/mhc.py with the lagged pre; its gate and Sinkhorn
math is unchanged.

The projection is exact for an f32 hc_fn: the caller splits it once into three
bf16 parts (hi + mid + lo == fn in f32, see ``split_fn``) and the bf16 streams
are exact in bf16, so MXU passes with f32 accumulation reproduce the f32
product up to summation order. The parts are packed once (``pack_fn``) into one
[hc * d, 128] matrix, rows of [hi; mid; lo] (3 * mix = 72 columns, zero-padded
to the 128 lanes) transposed, so each stream takes one NN dot for all three
parts and the kernel adds the three mix-wide column groups.

Layouts: streams [T, hc * d] bf16 (stream-major), y [T, d] bf16, pre/post
[T, hc] f32, comb [T, hc * hc] f32 row-major in (i, j), fn_packed
[hc * d, 128] bf16, so
R'[t, j] = post[t, j] * y[t] + sum_i comb[t, i, j] * R[t, i].
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.mhc import VEC_WIDTH, param_row

__all__ = ["pack_fn", "param_row", "split_fn", "v41_post_pre"]

_BLOCK_TOKENS = 128
_LANES = 128
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def split_fn(fn):
    """f32 [mix, hc * d] -> bf16 [3, mix, hc * d] whose f32 sum is exactly fn."""
    fn = fn.astype(jnp.float32)
    hi = fn.astype(jnp.bfloat16)
    r = fn - hi.astype(jnp.float32)
    mid = r.astype(jnp.bfloat16)
    lo = (r - mid.astype(jnp.float32)).astype(jnp.bfloat16)
    return jnp.stack([hi, mid, lo])


def pack_fn(parts):
    """bf16 [3, mix, hc * d] -> [hc * d, 128]: (hi; mid; lo) rows stacked, zero
    rows up to the 128 lanes, transposed (the kernel's NN projection operand)."""
    three, mix, width = parts.shape
    stacked = parts.reshape(three * mix, width)
    stacked = jnp.pad(stacked, ((0, _LANES - three * mix), (0, 0)))
    return stacked.T


def _kernel(
    y_ref,
    res_ref,
    pre_ref,
    post_ref,
    comb_ref,
    fn_ref,
    vec_ref,
    norm_ref,
    new_res_ref,
    u_ref,
    pre_out_ref,
    post_out_ref,
    comb_out_ref,
    *scratch,
    nsteps,
    batch,
    hc,
    d,
    iters,
    rms_eps,
    hc_eps,
    norm_eps,
):
    # Post: the new streams, rounded to bf16 before the collapse and the stats,
    # as the torch path stores them.
    y = y_ref[...].astype(jnp.float32)
    post = post_ref[...]
    comb = comb_ref[...]
    old = [res_ref[:, i * d : (i + 1) * d].astype(jnp.float32) for i in range(hc)]
    streams = []
    for j in range(hc):
        acc = post[:, j : j + 1] * y
        for i in range(hc):
            acc = acc + comb[:, i * hc + j : i * hc + j + 1] * old[i]
        rounded = acc.astype(new_res_ref.dtype)
        new_res_ref[:, j * d : (j + 1) * d] = rounded
        streams.append(rounded.astype(jnp.float32))

    # Collapse with the previous sublayer's pre, then the next input RMSNorm.
    pre_in = pre_ref[...]
    layer = functools.reduce(
        jnp.add, [pre_in[:, i : i + 1] * s for i, s in enumerate(streams)]
    )
    x = layer.astype(u_ref.dtype).astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + norm_eps)
    u_ref[...] = (x * norm_ref[...]).astype(u_ref.dtype)

    # Next coefficients from the new streams: RMS over hc * d, exact 3-pass
    # projection, gates, Sinkhorn.
    mix = (2 + hc) * hc
    sq = functools.reduce(
        jnp.add, [jnp.sum(s * s, axis=-1, keepdims=True) for s in streams]
    )
    rms = jax.lax.rsqrt(sq / (hc * d) + rms_eps)
    acc = None
    for i, s in enumerate(streams):
        term = jnp.dot(
            s.astype(jnp.bfloat16),
            fn_ref[i * d : (i + 1) * d, :],
            preferred_element_type=jnp.float32,
        )
        acc = term if acc is None else acc + term
    mixes = (acc[:, :mix] + acc[:, mix : 2 * mix] + acc[:, 2 * mix : 3 * mix]) * rms
    vec = vec_ref[...]
    base = vec[:, :mix]
    s0, s1, s2 = (
        vec[:, mix : mix + 1],
        vec[:, mix + 1 : mix + 2],
        vec[:, mix + 2 : mix + 3],
    )
    two = 2 * hc
    if not batch:
        pre_out_ref[...] = jax.nn.sigmoid(mixes[:, :hc] * s0 + base[:, :hc]) + hc_eps
        post_out_ref[...] = 2.0 * jax.nn.sigmoid(
            mixes[:, hc:two] * s1 + base[:, hc:two]
        )
        logits = (mixes[:, two:] * s2 + base[:, two:]).T  # tokens on lanes
        c = [
            [logits[i * hc + j : i * hc + j + 1] for j in range(hc)]
            for i in range(hc)
        ]
        c = _sinkhorn(c, hc, iters, hc_eps)
        comb_out_ref[...] = jnp.concatenate([v for r in c for v in r], axis=0).T
        return

    # Several row tiles: park the 24 gate/Sinkhorn pre-activations of this tile
    # (tokens on lanes, same mul-then-add per element) and run the gates and
    # Sinkhorn once, at the last step, on [nsteps, bt] arrays. Sinkhorn is a
    # serial chain of small ops, so one pass over all tiles replaces one pass
    # per tile. The coefficient outputs are resident blocks written at the end.
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, mix), 1)
    scale = jnp.where(lane < hc, s0, jnp.where(lane < two, s1, s2))
    zt = (mixes * scale + base).T  # [mix, bt]
    scr = scratch[0]
    t = pl.program_id(0)
    bt = zt.shape[1]
    scr[pl.ds(pl.multiple_of(t * mix, 8), mix), :] = zt

    @pl.when(t == nsteps - 1)
    def _():
        def ld(k):
            return scr[pl.ds(k, nsteps, stride=mix), :]

        def st(k, v):
            scr[pl.ds(k, nsteps, stride=mix), :] = v

        c = [[ld(two + i * hc + j) for j in range(hc)] for i in range(hc)]
        c = _sinkhorn(c, hc, iters, hc_eps)
        for i in range(hc):
            for j in range(hc):
                st(two + i * hc + j, c[i][j])
        for k in range(two):
            g = jax.nn.sigmoid(ld(k))
            st(k, g + hc_eps if k < hc else 2.0 * g)
        for step in range(nsteps):
            blk = scr[step * mix : (step + 1) * mix, :].T  # [bt, mix]
            r0, r1 = step * bt, (step + 1) * bt
            pre_out_ref[r0:r1, :] = blk[:, :hc]
            post_out_ref[r0:r1, :] = blk[:, hc:two]
            comb_out_ref[r0:r1, :] = blk[:, two:]


def _sinkhorn(c, hc, iters, hc_eps):
    """Sinkhorn on c[i][j] arrays of one shape (softmax rows, then iters column
    and row normalizations), in the torch path's op order."""

    def row_sum(r):
        return functools.reduce(jnp.add, r)

    c = [list(r) for r in c]
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
    return c


def v41_post_pre(
    y,
    res,
    pre,
    post,
    comb,
    fn_packed,
    vec,
    norm_weight,
    *,
    hc: int,
    iters: int,
    rms_eps: float,
    hc_eps: float,
    norm_eps: float,
    block_tokens: int = _BLOCK_TOKENS,
    interpret: bool = False,
):
    """One V4.1 sublayer boundary: returns (R', u, pre', post', comb') with R' [T, hc * d],
    u [T, d] in res.dtype and the next coefficients in f32."""
    tokens, width = res.shape
    d = width // hc
    # Balanced row tiles: as few steps as block_tokens allows, rows split evenly
    # (a multiple of 16), so 192 rows run as 2 x 96, not 128 + a padded 128.
    nsteps = -(-tokens // block_tokens)
    bt = -(-(-(-tokens // nsteps)) // 16) * 16
    nsteps = -(-tokens // bt)
    mix = (2 + hc) * hc
    batch = nsteps > 1 and tokens % bt == 0

    def rows(cols):
        return pl.BlockSpec((bt, cols), lambda t: (t, 0))

    def full(shape):
        return pl.BlockSpec(shape, lambda t: (0,) * len(shape))

    coef = (lambda c: full((tokens, c))) if batch else rows

    return pl.pallas_call(
        functools.partial(
            _kernel,
            nsteps=nsteps,
            batch=batch,
            hc=hc,
            d=d,
            iters=iters,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        ),
        grid=(nsteps,),
        in_specs=[
            rows(d),
            rows(width),
            rows(hc),
            rows(hc),
            rows(hc * hc),
            full((width, _LANES)),
            full((1, VEC_WIDTH)),
            full((1, d)),
        ],
        out_specs=[rows(width), rows(d), coef(hc), coef(hc), coef(hc * hc)],
        scratch_shapes=[pltpu.VMEM((nsteps * mix, bt), jnp.float32)] if batch else [],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, width), res.dtype),
            jax.ShapeDtypeStruct((tokens, d), res.dtype),
            jax.ShapeDtypeStruct((tokens, hc), jnp.float32),
            jax.ShapeDtypeStruct((tokens, hc), jnp.float32),
            jax.ShapeDtypeStruct((tokens, hc * hc), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary" if batch else "parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="mhc_v41_post_pre",
    )(
        y,
        res,
        pre.astype(jnp.float32),
        post.astype(jnp.float32),
        comb.astype(jnp.float32),
        fn_packed,
        vec,
        norm_weight.astype(jnp.float32).reshape(1, d),
    )
