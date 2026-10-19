"""Phase helpers for the GLM-5.3-Flash decode megakernel.

Every phase is a plain function over JAX arrays that also lowers inside a Pallas
kernel, so each one can be wrapped in a standalone `pallas_call` and tested on its
own.  When the assembled kernel disagrees with the reference, the failing equation
is identifiable instead of being somewhere inside a 45-layer program.

Lowering constraints to verify with focused probes under the target compiler:

  * `jax.lax.top_k` and `jnp.cumsum` do not lower -- selections are built from
    reductions only;
  * storing a scalar into a VMEM ref raises, so scalars go to SMEM or are
    broadcast in with `jnp.where`;
  * a matmul at `highest` precision accepts float32 operands only, so the fp8 path
    must run at default precision;
  * a traced index into a stacked `[N, rows, cols]` array DMAs correctly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

F32 = jnp.float32
BF16 = jnp.bfloat16


# --------------------------------------------------------------- reductions
def rms_norm(x, weight, eps, out_dtype=None):
    """Weighted RMSNorm with the reduction in float32, as the model specifies.

    `out_dtype` exists because fidelity and accuracy point different ways here.
    The published implementation computes the norm in float32 and then casts back
    to the model dtype, so the value that reaches the next matmul has been through
    a bfloat16 round trip.  Keeping float32 is more accurate and a different
    function that can change downstream routing. Call sites that mirror the
    model pass bfloat16; accuracy against an oracle is a separate check.
    """
    xf = x.astype(F32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    out = weight.astype(F32) * (xf * jax.lax.rsqrt(var + eps))
    return out if out_dtype is None else out.astype(out_dtype)


def unweighted_rms_norm(x, eps):
    xf = x.astype(F32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return xf * jax.lax.rsqrt(var + eps)


def layer_norm(x, weight, bias, eps):
    xf = x.astype(F32)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mu), axis=-1, keepdims=True)
    return weight.astype(F32) * ((xf - mu) * jax.lax.rsqrt(var + eps)) + bias.astype(F32)


def l2norm(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def swiglu_clamped(gate, up, limit):
    return jax.nn.silu(jnp.minimum(gate, limit)) * jnp.clip(up, -limit, limit)


# ------------------------------------------------------------------ matmul
SUBLANE = 8


def _pad_rows(x):
    """Pad a short leading axis up to a sublane multiple.

    A two-row bfloat16 operand is not a legal vector shape here: Mosaic fails
    with "'vector.broadcast' op ... source operand and result have same element
    type" on the implicit promotion inside the matmul.  One row works and eight
    work; two do not.  Pad, multiply, slice back.
    """
    if x.ndim < 2:
        return x, 0
    r = x.shape[0]
    if r % SUBLANE == 0:
        return x, 0
    pad = SUBLANE - (r % SUBLANE)
    return jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1)), r


def matvec_bf16(x, w):
    """[R, in] x [out, in] -> [R, out], contracting the trailing axes.

    A narrow output with more than one input row trips a Mosaic verification
    failure -- it emits a `vector.broadcast` that changes element type, which is
    not a legal op -- so those go through float32 operands instead.  The
    projections this applies to are tiny (`b_proj` is 64 wide before sharding),
    so the cost of not using the bfloat16 path there is nil.
    """
    xp, keep = _pad_rows(x)
    # Unconditional on the row count: making it depend on Q would mean the
    # one-query and two-query kernels compute beta differently, and the
    # first query of a two-query step must match a one-query step exactly.
    narrow = w.shape[0] < SUBLANE * 16
    if narrow:
        out = jax.lax.dot_general(
            xp.astype(F32), w.astype(F32), (((xp.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=F32)
    else:
        out = jax.lax.dot_general(
            xp.astype(BF16), w.astype(BF16), (((xp.ndim - 1,), (1,)), ((), ())),
            preferred_element_type=F32)
    return out[:keep] if keep else out


def matvec_fp8_split(x, w_fp8, scale, block):
    """[1, in] x fp8[out, in] with blockwise scales -> [1, out], float32.

    One dot per 128-wide input block, with the fp8 tile going to the MXU as-is so
    no dequantised copy is materialised.  The catch is the *number* of dots: a
    4096-wide contraction is 32 of them, and with eight experts in four unrolled
    layers that is over a thousand `dot_general`s in one loop body, which the
    compiler takes a very long time to chew through.  `matvec_fp8` below is the
    form actually used; this one is kept because it is the obvious reading of the
    quantisation scheme and the two are checked against each other.
    """
    bo, bi = block
    out_dim, in_dim = w_fp8.shape
    n_i = -(-in_dim // bi)
    acc = None
    for j in range(n_i):
        lo, hi = j * bi, min((j + 1) * bi, in_dim)
        part = jax.lax.dot_general(
            x[..., lo:hi].astype(BF16), w_fp8[:, lo:hi],
            (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=F32,
        )
        s = jnp.repeat(scale[:, j], bo)[:out_dim]
        acc = part * s if acc is None else acc + part * s
    return acc


def matvec_fp8(x, w_fp8, scale, block):
    """[1, in] x fp8[out, in] with blockwise scales -> [1, out], float32.

    The block scale s[a, b] depends on both the output block a and the input
    block b, so the contraction has to be broken somewhere.  Two ways:

      * **per input block** -- one dot of the full output against a 128-wide
        slice, and the partial is scaled afterwards.  `in / 128` dots.
      * **per output block** -- scale the *activation* by s[a, :] expanded over
        the input axis, then one dot of the full input against a 128-row slice.
        `out / 128` dots.  Valid because
        `sum_i w[o,i] s[a, i/128] x[i] = sum_i w[o,i] (s[a, i/128] x[i])`.

    Both move the same `out * in / 128` elements elementwise; they differ only in
    how many `dot_general`s they emit, so take whichever is fewer.  That matters
    twice over: a 4096-wide expert contraction is 32 dots the first way and 4 the
    second. Record compile time as well as steady-state cost.

    Expanding the scale grid to the full weight shape permits a single dot but
    adds elementwise work and live storage: a [512, 4096] multiplier contains
    2,097,152 elements per expert. Compare this candidate in situ, including
    its effect on the complete VMEM live set.
    """
    bo, bi = block
    out_dim, in_dim = w_fp8.shape
    n_o = -(-out_dim // bo)
    n_i = -(-in_dim // bi)

    xp, keep = _pad_rows(x)

    if n_o <= n_i:
        parts = []
        for a in range(n_o):
            lo, hi = a * bo, min((a + 1) * bo, out_dim)
            sx = xp.astype(F32) * jnp.repeat(scale[a : a + 1], bi, axis=1)[:, :in_dim]
            parts.append(jax.lax.dot_general(
                sx.astype(BF16), w_fp8[lo:hi],
                (((xp.ndim - 1,), (1,)), ((), ())), preferred_element_type=F32))
        out = jnp.concatenate(parts, axis=-1)
        return out[:keep] if keep else out

    acc = None
    for j in range(n_i):
        lo, hi = j * bi, min((j + 1) * bi, in_dim)
        part = jax.lax.dot_general(
            xp[..., lo:hi].astype(BF16), w_fp8[:, lo:hi],
            (((xp.ndim - 1,), (1,)), ((), ())), preferred_element_type=F32)
        sc = jnp.repeat(scale[:, j : j + 1], bo, axis=0).reshape(1, -1)[:, :out_dim]
        acc = part * sc if acc is None else acc + part * sc
    return acc[:keep] if keep else acc


# --------------------------------------------------------------- selection
def select_top_k_mask(scores, k, *, score_iters=40, index_iters=24):
    """Exact top-k as a boolean mask, using only reductions.

    `lax.top_k` does not lower and `cumsum` does not either, so this bisects twice:
    once on the score to find the cut value, then on the *index* among the values
    that tie at the cut, so the tie-break is deterministic (lowest index wins) and
    does not depend on two layouts of the same number agreeing bitwise.

    scores: [R, P] float32, already masked with -inf where ineligible.
    returns: [R, P] bool with exactly k set per row (fewer if fewer are eligible).
    """
    idx = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    eligible = jnp.isfinite(scores)
    n_elig = jnp.sum(eligible.astype(jnp.int32), axis=1, keepdims=True)
    want = jnp.minimum(n_elig, k)

    finite = jnp.where(eligible, scores, 0.0)
    lo = jnp.min(jnp.where(eligible, scores, jnp.inf), axis=1, keepdims=True)
    hi = jnp.max(finite, axis=1, keepdims=True)
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0) - 1.0
    hi = hi + 1.0

    def score_step(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        cnt = jnp.sum((eligible & (scores > mid)).astype(jnp.int32), axis=1, keepdims=True)
        return (jnp.where(cnt >= want, mid, lo), jnp.where(cnt >= want, hi, mid))

    lo, hi = jax.lax.fori_loop(0, score_iters, score_step, (lo, hi))

    strict = eligible & (scores > hi)
    n_strict = jnp.sum(strict.astype(jnp.int32), axis=1, keepdims=True)
    tie = eligible & (scores > lo) & (~strict)
    need = want - n_strict

    jlo = jnp.zeros_like(need)
    jhi = jnp.full_like(need, scores.shape[1])

    def index_step(_, b):
        jlo, jhi = b
        jmid = (jlo + jhi) // 2
        cnt = jnp.sum((tie & (idx < jmid)).astype(jnp.int32), axis=1, keepdims=True)
        return (jnp.where(cnt < need, jmid, jlo), jnp.where(cnt < need, jhi, jmid))

    jlo, jhi = jax.lax.fori_loop(0, index_iters, index_step, (jlo, jhi))
    return strict | (tie & (idx < jhi))


def select_top_k_mask_2d(scores, k, *, score_iters=40, index_iters=24):
    """`select_top_k_mask` over every element of a [T, P] array at once: the
    flat index of element (t, p) is t * P + p, so the tie-break (lowest index
    wins) matches a [1, T*P] row.  Lets the pool scores stay tiled [tiles, 128]."""
    T, P = scores.shape
    idx = (jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) * P
           + jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1))
    ax = (0, 1)
    eligible = jnp.isfinite(scores)
    n_elig = jnp.sum(eligible.astype(jnp.int32), axis=ax, keepdims=True)
    want = jnp.minimum(n_elig, k)

    finite = jnp.where(eligible, scores, 0.0)
    lo = jnp.min(jnp.where(eligible, scores, jnp.inf), axis=ax, keepdims=True)
    hi = jnp.max(finite, axis=ax, keepdims=True)
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0) - 1.0
    hi = hi + 1.0

    def score_step(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        cnt = jnp.sum((eligible & (scores > mid)).astype(jnp.int32), axis=ax, keepdims=True)
        return (jnp.where(cnt >= want, mid, lo), jnp.where(cnt >= want, hi, mid))

    lo, hi = jax.lax.fori_loop(0, score_iters, score_step, (lo, hi))

    strict = eligible & (scores > hi)
    n_strict = jnp.sum(strict.astype(jnp.int32), axis=ax, keepdims=True)
    tie = eligible & (scores > lo) & (~strict)
    need = want - n_strict

    jlo = jnp.zeros_like(need)
    jhi = jnp.full_like(need, T * P)

    def index_step(_, b):
        jlo, jhi = b
        jmid = (jlo + jhi) // 2
        cnt = jnp.sum((tie & (idx < jmid)).astype(jnp.int32), axis=ax, keepdims=True)
        return (jnp.where(cnt < need, jmid, jlo), jnp.where(cnt < need, jhi, jmid))

    jlo, jhi = jax.lax.fori_loop(0, index_iters, index_step, (jlo, jhi))
    return strict | (tie & (idx < jhi))


def select_top_k_mask_3d(scores, k, *, score_iters=40, index_iters=24):
    """`select_top_k_mask_2d` for B independent [T, P] score arrays at once
    ([B, T, P]).  The bisections are serial chains of full reductions, so their
    latency, not their arithmetic, is the cost: one chain over B rows costs about
    what one chain over a single row does."""
    B, T, P = scores.shape
    idx = (jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1) * P
           + jax.lax.broadcasted_iota(jnp.int32, scores.shape, 2))
    ax = (1, 2)
    eligible = jnp.isfinite(scores)
    n_elig = jnp.sum(eligible.astype(jnp.int32), axis=ax, keepdims=True)
    want = jnp.minimum(n_elig, k)

    finite = jnp.where(eligible, scores, 0.0)
    lo = jnp.min(jnp.where(eligible, scores, jnp.inf), axis=ax, keepdims=True)
    hi = jnp.max(finite, axis=ax, keepdims=True)
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0) - 1.0
    hi = hi + 1.0

    def score_step(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        cnt = jnp.sum((eligible & (scores > mid)).astype(jnp.int32), axis=ax, keepdims=True)
        return (jnp.where(cnt >= want, mid, lo), jnp.where(cnt >= want, hi, mid))

    lo, hi = jax.lax.fori_loop(0, score_iters, score_step, (lo, hi))

    strict = eligible & (scores > hi)
    n_strict = jnp.sum(strict.astype(jnp.int32), axis=ax, keepdims=True)
    tie = eligible & (scores > lo) & (~strict)
    need = want - n_strict

    jlo = jnp.zeros_like(need)
    jhi = jnp.full_like(need, T * P)

    def index_step(_, b):
        jlo, jhi = b
        jmid = (jlo + jhi) // 2
        cnt = jnp.sum((tie & (idx < jmid)).astype(jnp.int32), axis=ax, keepdims=True)
        return (jnp.where(cnt < need, jmid, jlo), jnp.where(cnt < need, jhi, jmid))

    jlo, jhi = jax.lax.fori_loop(0, index_iters, index_step, (jlo, jhi))
    return strict | (tie & (idx < jhi))


# --------------------------------------------------------------------- mHC
def sinkhorn_step(c, hc_eps):
    """One row-then-column normalisation of the mHC combination matrix [N, N], or
    of a stack of them [B, N, N]."""
    c = c / (jnp.sum(c, axis=-1, keepdims=True) + hc_eps)
    return c / (jnp.sum(c, axis=-2, keepdims=True) + hc_eps)


def hyper_connection(streams, fn, base, scale, *, hc_mult, hc_eps, iters, rms_eps,
                     collapsed_dtype=BF16, defer_sinkhorn=False):
    """streams [N, D] (one token) -> (post [N, 1], comb [N, N], collapsed [1, D]).

    Everything stays rank-2.  A rank-1 `reshape(N*N) -> (N, N)` does not lower here
    ("infer-vector-layout: unsupported shape cast"), and the small coefficient
    vectors are exactly where that bites, so the N x N matrix is assembled by
    concatenating rows instead.

    `base` is [1, (2+N)*N] and `scale` is [1, 3]; passing them rank-1 reintroduces
    the same problem one level up.
    """
    n = hc_mult
    flat = unweighted_rms_norm(streams.reshape(1, -1), rms_eps)          # [1, N*D]
    # HIGHEST precision here, deliberately.  A float32 matmul at default precision
    # on this target runs through bf16 passes, and these 24 coefficients go into a
    # sigmoid and a softmax before scaling the entire residual stream.
    # Validate the [1, N*D] x [N*D, 24] projection against a float64 oracle
    # before changing precision; small coefficient errors can affect routing.
    coef = jax.lax.dot_general(flat, fn.astype(F32), (((1,), (1,)), ((), ())),
                               preferred_element_type=F32,
                               precision=jax.lax.Precision.HIGHEST)      # [1, (2+N)*N]

    pre_w, post_w = coef[:, :n], coef[:, n : 2 * n]
    pre_b, post_b = base[:, :n], base[:, n : 2 * n]

    pre = jax.nn.sigmoid(pre_w * scale[0, 0] + pre_b) + hc_eps           # [1, N]
    post = 2.0 * jax.nn.sigmoid(post_w * scale[0, 1] + post_b)           # [1, N]

    cw = jnp.concatenate(
        [coef[:, 2 * n + i * n : 2 * n + (i + 1) * n] * scale[0, 2]
         + base[:, 2 * n + i * n : 2 * n + (i + 1) * n] for i in range(n)],
        axis=0)                                                          # [N, N]
    comb = jax.nn.softmax(cw, axis=-1) + hc_eps

    # One column normalisation, then (iters-1) row/column pairs.  The asymmetric
    # first step is observable in the result; folding it into the loop changes the
    # model.
    comb = comb / (jnp.sum(comb, axis=0, keepdims=True) + hc_eps)

    # With `defer_sinkhorn` the caller runs the (iters-1) row/column pairs itself
    # (`sinkhorn_step`), e.g. interleaved with the block that runs meanwhile: only
    # the residual combine after the block needs `comb`.
    if not defer_sinkhorn:
        comb = jax.lax.fori_loop(0, iters - 1, lambda _, c: sinkhorn_step(c, hc_eps),
                                 comb, unroll=True)

    pre_col = jnp.concatenate([pre[:, i : i + 1] for i in range(n)], axis=0)  # [N,1]
    # Cast to the stream dtype, as the published implementation does: the collapsed
    # vector is the block input, and it reaches the next matmul rounded.
    collapsed = jnp.sum(pre_col * streams.astype(F32), axis=0,
                        keepdims=True).astype(collapsed_dtype)
    post_col = jnp.concatenate([post[:, i : i + 1] for i in range(n)], axis=0)
    return post_col, comb, collapsed


def hyper_connection_q(streams_list, fn, base, scale, *, hc_mult, hc_eps, iters,
                       rms_eps, collapsed_dtype=BF16, defer_sinkhorn=False):
    """`hyper_connection` for several tokens at once: one coefficient projection
    for all of them.  `fn` is bfloat16 in the slab, so the f32 x f32 HIGHEST dot is
    reproduced by three single-pass dots of the f32 operand's bf16 pieces against
    it (hi + mid + lo carries the f32 value; the bf16 weight is exact)."""
    n = hc_mult
    flats = [unweighted_rms_norm(st.reshape(1, -1), rms_eps) for st in streams_list]
    flat = jnp.concatenate(flats, axis=0) if len(flats) > 1 else flats[0]
    fnb = fn.astype(BF16)
    hi = flat.astype(BF16)
    r1 = flat - hi.astype(F32)
    mid = r1.astype(BF16)
    lo = (r1 - mid.astype(F32)).astype(BF16)
    coef_all = sum(jax.lax.dot_general(p_, fnb, (((1,), (1,)), ((), ())),
                                       preferred_element_type=F32) for p_ in (hi, mid, lo))
    outs = []
    for qi, streams in enumerate(streams_list):
        coef = coef_all[qi : qi + 1]
        pre_w, post_w = coef[:, :n], coef[:, n : 2 * n]
        pre_b, post_b = base[:, :n], base[:, n : 2 * n]
        pre = jax.nn.sigmoid(pre_w * scale[0, 0] + pre_b) + hc_eps
        post = 2.0 * jax.nn.sigmoid(post_w * scale[0, 1] + post_b)
        cw = jnp.concatenate(
            [coef[:, 2 * n + i * n : 2 * n + (i + 1) * n] * scale[0, 2]
             + base[:, 2 * n + i * n : 2 * n + (i + 1) * n] for i in range(n)], axis=0)
        comb = jax.nn.softmax(cw, axis=-1) + hc_eps
        comb = comb / (jnp.sum(comb, axis=0, keepdims=True) + hc_eps)
        if not defer_sinkhorn:
            comb = jax.lax.fori_loop(0, iters - 1, lambda _, c: sinkhorn_step(c, hc_eps),
                                     comb, unroll=True)
        pre_col = jnp.concatenate([pre[:, i : i + 1] for i in range(n)], axis=0)
        collapsed = jnp.sum(pre_col * streams.astype(F32), axis=0,
                            keepdims=True).astype(collapsed_dtype)
        post_col = jnp.concatenate([post[:, i : i + 1] for i in range(n)], axis=0)
        outs.append((post_col, comb, collapsed))
    return outs


def hyper_combine_vpu(post, comb, block_out, residual, out_dtype=BF16):
    """`hyper_combine` with the N x N mixing as N*N scalar-times-row adds on the VPU
    instead of a HIGHEST-precision MXU dot of a [N, N] by [N, D]."""
    n = comb.shape[0]
    res = residual.astype(F32)
    rows = []
    for j in range(n):
        acc = post[j : j + 1] * block_out.reshape(1, -1)
        for m in range(n):
            acc = acc + comb[m : m + 1, j : j + 1] * res[m : m + 1]
        rows.append(acc)
    return jnp.concatenate(rows, axis=0).astype(out_dtype)


def hyper_combine(post, comb, block_out, residual, out_dtype=BF16):
    """out[n, d] = post[n] * block_out[d] + sum_m comb[m, n] * residual[m, d].

    `post` is [N, 1] so the product broadcasts without a rank-1 reshape.

    The result is rounded to `out_dtype`, bfloat16 by default, because the
    published implementation carries `hidden_states` in the model dtype and rounds
    there.  Keeping the residual stream in float32 is numerically better but is a
    different function that can change routing and the token stream. Preserve
    the model's rounding points and check oracle accuracy separately.
    """
    expanded = post * block_out.reshape(1, -1)
    mixed = jax.lax.dot_general(comb, residual.astype(F32),
                                (((0,), (0,)), ((), ())),
                                preferred_element_type=F32,
                                precision=jax.lax.Precision.HIGHEST)
    return (expanded + mixed).astype(out_dtype)


# ------------------------------------------------------------- MoE routing
def select_top_k_mask_small(scores, k):
    """`select_top_k_mask` for a small k over finite scores: k rounds of max, then
    the lowest index holding it, then mask it out -- 2k reductions instead of the
    two bisections' 64.  Same selection: the k largest, ties to the lowest index."""
    return _top_k_small(scores, k)[0]


def _top_k_small(scores, k):
    idx = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    big = scores.shape[1]
    mask = jnp.zeros(scores.shape, bool)
    rem = scores
    picks = []
    for _ in range(k):
        m = jnp.max(rem, axis=1, keepdims=True)
        first = jnp.min(jnp.where(rem == m, idx, big), axis=1, keepdims=True)
        hit = idx == first
        mask = mask | hit
        rem = jnp.where(hit, -jnp.inf, rem)
        picks.append(first[0, 0])
    return mask, picks


# Batcher's 19-comparator network for 8 keys.
_SORT8 = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2),
          (5, 6), (0, 4), (3, 7), (1, 5), (2, 6), (1, 4), (3, 6), (2, 4), (3, 5),
          (3, 4))


def sort_ascending(xs):
    """A short list of scalars, ascending (a sorting network for 8)."""
    xs = list(xs)
    if len(xs) != 8:
        raise ValueError(f"sort_ascending is the 8-key network, got {len(xs)}")
    for a, b in _SORT8:
        lo, hi = jnp.minimum(xs[a], xs[b]), jnp.maximum(xs[a], xs[b])
        xs[a], xs[b] = lo, hi
    return jnp.stack(xs)


def route_experts_ids(x, gate_w, bias, *, top_k, norm_topk_prob, routed_scaling_factor):
    """`route_experts` for one token, plus the chosen ids ascending (what
    `mask_to_indices` returns) straight from the top-k rounds."""
    logits = matvec_bf16(x, gate_w)
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias.reshape(1, -1)
    mask, picks = _top_k_small(choice, top_k)
    w = jnp.where(mask, scores, 0.0)
    if norm_topk_prob:
        w = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-20)
    return mask, w * routed_scaling_factor, sort_ascending(picks)


def route_experts(x, gate_w, bias, *, top_k, norm_topk_prob, routed_scaling_factor):
    """One token -> (mask [1, E] bool, weights [1, E] float32 zero off-mask).

    Returning a mask rather than indices keeps the result in a lane layout that the
    expert loop can consume without a gather, and the weights carry the selection.
    """
    logits = matvec_bf16(x, gate_w)                 # [1, E] float32
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias.reshape(1, -1)
    mask = select_top_k_mask_small(choice, top_k)
    w = jnp.where(mask, scores, 0.0)
    if norm_topk_prob:
        w = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-20)
    return mask, w * routed_scaling_factor


def mask_to_indices(mask, k):
    """[1, E] bool -> [k] int32 expert ids, ascending; `E` where a slot is unused.

    Built from reductions: slot j takes the expert whose rank among the set bits
    is j.  `cumsum` does not lower, so the rank comes from a masked comparison
    matrix, which is cheap at E = 288.

    A slot with no expert returns the out-of-range value `E`, not 0.  Returning 0
    is the obvious thing and it silently applies expert 0 a second time whenever
    the mask has fewer than `k` bits -- which happens as soon as two tokens share
    an expert and the caller asked for the union of both their top-k sets.
    """
    e = mask.shape[1]
    idx = jax.lax.broadcasted_iota(jnp.int32, (e, e), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (e, e), 1)
    m = mask.reshape(1, e)
    before = jnp.sum(jnp.where((col < idx) & (m != 0), 1, 0), axis=1)     # [E]
    out = []
    ar = jnp.arange(e, dtype=jnp.int32)
    for j in range(k):
        hit = (mask[0] != 0) & (before == j)
        n_hit = jnp.sum(hit.astype(jnp.int32))
        out.append(jnp.where(n_hit > 0, jnp.sum(jnp.where(hit, ar, 0)), e))
    return jnp.stack(out)


def mask_to_indices_mxu(mask, k):
    """`mask_to_indices` without its serial chain of 2k reductions: each set bit's
    rank comes from one [1, E] x [E, E] strictly-upper-triangular 0/1 product
    (exact: bf16 0/1 operands, f32 sums <= E), and all k slots are picked by one
    [k, E] compare and one reduction.  Same ids, same `E` sentinel."""
    e = mask.shape[1]
    row = jax.lax.broadcasted_iota(jnp.int32, (e, e), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (e, e), 1)
    tri = (row < col).astype(BF16)                          # tri[c, i] = c < i
    m = mask.reshape(1, e)
    rank = jnp.dot(m.astype(BF16), tri, preferred_element_type=F32)   # [1, E]
    slot = jax.lax.broadcasted_iota(jnp.int32, (k, 1), 0).astype(F32)
    hit = (rank == slot) & (m != 0)                         # [k, E]
    ar = jax.lax.broadcasted_iota(jnp.int32, (k, e), 1)
    idx = jnp.sum(jnp.where(hit, ar, 0), axis=1)
    n_hit = jnp.sum(hit.astype(jnp.int32), axis=1)
    return jnp.where(n_hit > 0, idx, e)


# --------------------------------------------------------------------- KDA
def kda_step(state, q, k, v, g_log, beta):
    """One token of the Kimi delta rule.

    state [H, Dk, Dv], q/k [H, Dk], v [H, Dv], g_log [H, Dk], beta [H].
    Returns (new_state, out [H, Dv]).
    """
    dk = q.shape[-1]
    qn = l2norm(q) * (dk ** -0.5)
    kn = l2norm(k)
    state = state * jnp.exp(g_log)[..., None]
    kv_mem = jnp.sum(state * kn[..., None], axis=-2)
    delta = (v - kv_mem) * beta[:, None]
    state = state + kn[..., None] * delta[:, None, :]
    out = jnp.sum(state * qn[..., None], axis=-2)
    return state, out


def conv_step_cl(conv_state, x_new, weight):
    """`conv_step`, channels-last: conv_state [K-1, C], x_new [1, C], weight [K, C].
    Returns (out [1, C], new_state [K-1, C])."""
    window = jnp.concatenate([conv_state, x_new], axis=0)                   # [K, C]
    out = jnp.sum(window * weight, axis=0, keepdims=True)
    return jax.nn.silu(out), window[1:]


def conv_step(conv_state, x_new, weight):
    """Causal depthwise conv over [state, x_new] for one token, then SiLU.

    conv_state [C, K-1], x_new [1, C], weight [C, K].
    Returns (out [1, C], new_state [C, K-1]).
    """
    k = weight.shape[-1]
    window = jnp.concatenate([conv_state, x_new.reshape(-1, 1)], axis=1)   # [C, K]
    out = jnp.sum(window * weight, axis=1).reshape(1, -1)
    return jax.nn.silu(out), window[:, 1:]


# --------------------------------------------------------------------- DSA
def absorb_q(q_heads, kv_b, *, n_heads, qk_dim, v_dim, kv_lora):
    """Fold the key half of kv_b into the query: q_abs[h] = W_kc[h]^T @ q[h].

    Expanding the latent cache into per-head keys and values would cost
    `L * kv_lora * n_heads * (qk+v)` MACs per layer per step, which at a few
    thousand cached tokens is larger than everything else in the step put together.
    Absorbing is the same function by reassociation, and turns the per-step cost
    into a dot product against the 512-wide latent.
    """
    outs = []
    per = qk_dim + v_dim
    for h in range(n_heads):
        w_kc = kv_b[h * per : h * per + qk_dim]          # [qk_dim, kv_lora]
        outs.append(jax.lax.dot_general(
            q_heads[h : h + 1], w_kc, (((1,), (0,)), ((), ())),
            preferred_element_type=F32)[0])
    return jnp.stack(outs)                                # [n_heads, kv_lora]


def project_out(latent_out, kv_b, *, n_heads, qk_dim, v_dim):
    """out[h] = W_vc[h] @ latent_out[h], the other half of the absorption."""
    outs = []
    per = qk_dim + v_dim
    for h in range(n_heads):
        w_vc = kv_b[h * per + qk_dim : (h + 1) * per]    # [v_dim, kv_lora]
        outs.append(jax.lax.dot_general(
            latent_out[h : h + 1], w_vc, (((1,), (1,)), ((), ())),
            preferred_element_type=F32)[0])
    return jnp.stack(outs)                                # [n_heads, v_dim]


def latent_attention(q_abs, kv_cache, allowed, scaling):
    """Softmax attention directly against the compressed latent cache.

    q_abs [H, kv_lora], kv_cache [L, kv_lora], allowed **[1, L]** bool.
    Returns [H, kv_lora].

    `allowed` is rank-2 on purpose: a rank-1 reshape into the lane axis does not
    lower here, and this is one of the places that would need one.
    """
    logits = jax.lax.dot_general(q_abs, kv_cache, (((1,), (1,)), ((), ())),
                                 preferred_element_type=F32) * scaling   # [H, L]
    logits = jnp.where(allowed, logits, -jnp.inf)
    m = jnp.max(logits, axis=1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(logits - m)
    p = jnp.where(allowed, p, 0.0)
    p = p / (jnp.sum(p, axis=1, keepdims=True) + 1e-30)
    return jax.lax.dot_general(p, kv_cache, (((1,), (0,)), ((), ())),
                               preferred_element_type=F32)


def indexer_pool_keys(ik, ig, ape, pos, kpool):
    """Compress the cached indexer keys into pool keys.

    ik, ig [L, Dh]; ape [kpool, Dh]; `pos` the last valid cache index.  L must be a
    multiple of kpool; the caller sizes the cache so it is.
    Returns (pool_keys [P, Dh], pool_valid [1, P]).

    The published implementation masks invalid slots to -inf inside the pool
    softmax.  That mask is not applied here, and it cannot change any result that
    is used: a pool with any invalid slot is not fully valid, `pool_valid` is false
    for it, and the selection masks it to -inf before the top-k.  Dropping it
    avoids a `[P, kpool] -> [P, kpool, 1]` shape cast, which does not lower on this
    toolchain.  `tests/test_kernel_phases.py` compares only the valid pools, and
    says so.
    """
    L, dh = ik.shape
    p = L // kpool
    gk = ik.reshape(p, kpool, dh).astype(F32)
    gg = ig.reshape(p, kpool, dh).astype(F32)
    logits = gg + jnp.broadcast_to(ape.astype(F32), (p, kpool, dh))
    m = jnp.max(logits, axis=1, keepdims=True)
    e = jnp.exp(logits - m)
    probs = e / (jnp.sum(e, axis=1, keepdims=True) + 1e-30)
    pool_keys = jnp.sum(probs * gk, axis=1)

    pj = jax.lax.broadcasted_iota(jnp.int32, (1, p), 1)
    pool_valid = (pj + 1) * kpool - 1 <= pos
    return pool_keys, pool_valid


def indexer_scores(iq, head_w, pool_keys, scale):
    """iq [Hi, Dh], head_w [Hi], pool_keys [P, Dh] -> [1, P].  With head_w a
    [B, Hi] matrix (block-diagonal over B stacked queries' heads) -> [B, P]."""
    sc = jax.lax.dot_general(iq, pool_keys, (((1,), (1,)), ((), ())),
                             preferred_element_type=F32)
    sc = jax.nn.relu(sc * scale)
    hw = head_w if head_w.ndim == 2 else head_w.reshape(1, -1)
    return jax.lax.dot_general(hw, sc, (((1,), (0,)), ((), ())),
                               preferred_element_type=F32)
