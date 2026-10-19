"""Decode-megakernel phases: plain JAX functions that also lower inside a Pallas kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp

F32 = jnp.float32
BF16 = jnp.bfloat16


def rms_norm(x, weight, eps, out_dtype=None):
    """Weighted RMSNorm, reduced in f32; pass out_dtype=bf16 to match the model's rounding."""
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


SUBLANE = 8


def _pad_rows(x):
    """Pad rows to a sublane multiple: Mosaic rejects e.g. a two-row bf16 matmul operand."""
    if x.ndim < 2:
        return x, 0
    r = x.shape[0]
    if r % SUBLANE == 0:
        return x, 0
    pad = SUBLANE - (r % SUBLANE)
    return jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1)), r


def matvec_bf16(x, w):
    """[R, in] x [out, in] -> [R, out]; narrow outputs use f32 (Mosaic rejects the bf16 form)."""
    xp, keep = _pad_rows(x)
    # Independent of the row count, so a two-query step's first query matches a one-query step.
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
    """Reference form of `matvec_fp8`, one dot per input block (too many dots to compile)."""
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
    """[1, in] x fp8[out, in], block scales -> [1, out] f32; splits the axis with fewer blocks."""
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


def select_top_k_mask(scores, k, *, score_iters=40, index_iters=24):
    """Exact top-k mask over [R, P] (-inf = ineligible), ties to the lowest index; no top_k."""
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
    """`select_top_k_mask` over a [T, P] array as one flat row of T * P."""
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
    """`select_top_k_mask_2d` for B independent [T, P] arrays ([B, T, P])."""
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


def sinkhorn_step(c, hc_eps):
    c = c / (jnp.sum(c, axis=-1, keepdims=True) + hc_eps)
    return c / (jnp.sum(c, axis=-2, keepdims=True) + hc_eps)


def hyper_connection(streams, fn, base, scale, *, hc_mult, hc_eps, iters, rms_eps,
                     collapsed_dtype=BF16, defer_sinkhorn=False):
    """streams [N, D] -> (post [N, 1], comb [N, N], collapsed [1, D]); all rank-2 to lower."""
    n = hc_mult
    flat = unweighted_rms_norm(streams.reshape(1, -1), rms_eps)
    # HIGHEST: default-precision f32 runs as bf16 passes, too coarse for these coefficients.
    coef = jax.lax.dot_general(flat, fn.astype(F32), (((1,), (1,)), ((), ())),
                               preferred_element_type=F32,
                               precision=jax.lax.Precision.HIGHEST)

    pre_w, post_w = coef[:, :n], coef[:, n : 2 * n]
    pre_b, post_b = base[:, :n], base[:, n : 2 * n]

    pre = jax.nn.sigmoid(pre_w * scale[0, 0] + pre_b) + hc_eps
    post = 2.0 * jax.nn.sigmoid(post_w * scale[0, 1] + post_b)

    cw = jnp.concatenate(
        [coef[:, 2 * n + i * n : 2 * n + (i + 1) * n] * scale[0, 2]
         + base[:, 2 * n + i * n : 2 * n + (i + 1) * n] for i in range(n)],
        axis=0)
    comb = jax.nn.softmax(cw, axis=-1) + hc_eps

    # The model's asymmetric first step: one column normalisation, then (iters-1) pairs.
    comb = comb / (jnp.sum(comb, axis=0, keepdims=True) + hc_eps)

    # With defer_sinkhorn the caller runs the (iters-1) pairs via `sinkhorn_step`.
    if not defer_sinkhorn:
        comb = jax.lax.fori_loop(0, iters - 1, lambda _, c: sinkhorn_step(c, hc_eps),
                                 comb, unroll=True)

    pre_col = jnp.concatenate([pre[:, i : i + 1] for i in range(n)], axis=0)
    # Rounded to the stream dtype, as the model rounds the block input.
    collapsed = jnp.sum(pre_col * streams.astype(F32), axis=0,
                        keepdims=True).astype(collapsed_dtype)
    post_col = jnp.concatenate([post[:, i : i + 1] for i in range(n)], axis=0)
    return post_col, comb, collapsed


def hyper_connection_q(streams_list, fn, base, scale, *, hc_mult, hc_eps, iters,
                       rms_eps, collapsed_dtype=BF16, defer_sinkhorn=False):
    """`hyper_connection` for several tokens; the f32 HIGHEST dot is three bf16 dots."""
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
    """`hyper_combine` with the N x N mixing as VPU adds instead of a HIGHEST MXU dot."""
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
    """out = post * block_out + comb^T @ residual, rounded to out_dtype as the model does."""
    expanded = post * block_out.reshape(1, -1)
    mixed = jax.lax.dot_general(comb, residual.astype(F32),
                                (((0,), (0,)), ((), ())),
                                preferred_element_type=F32,
                                precision=jax.lax.Precision.HIGHEST)
    return (expanded + mixed).astype(out_dtype)


def select_top_k_mask_small(scores, k):
    """`select_top_k_mask` for a small k over finite scores: k rounds of max."""
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
    xs = list(xs)
    if len(xs) != 8:
        raise ValueError(f"sort_ascending is the 8-key network, got {len(xs)}")
    for a, b in _SORT8:
        lo, hi = jnp.minimum(xs[a], xs[b]), jnp.maximum(xs[a], xs[b])
        xs[a], xs[b] = lo, hi
    return jnp.stack(xs)


def route_experts_ids(x, gate_w, bias, *, top_k, norm_topk_prob, routed_scaling_factor):
    """`route_experts` plus the chosen ids ascending, as `mask_to_indices` returns them."""
    logits = matvec_bf16(x, gate_w)
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias.reshape(1, -1)
    mask, picks = _top_k_small(choice, top_k)
    w = jnp.where(mask, scores, 0.0)
    if norm_topk_prob:
        w = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-20)
    return mask, w * routed_scaling_factor, sort_ascending(picks)


def route_experts(x, gate_w, bias, *, top_k, norm_topk_prob, routed_scaling_factor):
    """One token -> (mask [1, E] bool, weights [1, E] float32 zero off-mask)."""
    logits = matvec_bf16(x, gate_w)
    scores = jax.nn.sigmoid(logits)
    choice = scores + bias.reshape(1, -1)
    mask = select_top_k_mask_small(choice, top_k)
    w = jnp.where(mask, scores, 0.0)
    if norm_topk_prob:
        w = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-20)
    return mask, w * routed_scaling_factor


def mask_to_indices(mask, k):
    """[1, E] bool -> [k] int32 ids ascending; unused slots are `E`, never 0 (expert 0)."""
    e = mask.shape[1]
    idx = jax.lax.broadcasted_iota(jnp.int32, (e, e), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (e, e), 1)
    m = mask.reshape(1, e)
    before = jnp.sum(jnp.where((col < idx) & (m != 0), 1, 0), axis=1)
    out = []
    ar = jnp.arange(e, dtype=jnp.int32)
    for j in range(k):
        hit = (mask[0] != 0) & (before == j)
        n_hit = jnp.sum(hit.astype(jnp.int32))
        out.append(jnp.where(n_hit > 0, jnp.sum(jnp.where(hit, ar, 0)), e))
    return jnp.stack(out)


def mask_to_indices_mxu(mask, k):
    """`mask_to_indices` via one exact triangular 0/1 matmul; same ids and sentinel."""
    e = mask.shape[1]
    row = jax.lax.broadcasted_iota(jnp.int32, (e, e), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (e, e), 1)
    tri = (row < col).astype(BF16)
    m = mask.reshape(1, e)
    rank = jnp.dot(m.astype(BF16), tri, preferred_element_type=F32)
    slot = jax.lax.broadcasted_iota(jnp.int32, (k, 1), 0).astype(F32)
    hit = (rank == slot) & (m != 0)
    ar = jax.lax.broadcasted_iota(jnp.int32, (k, e), 1)
    idx = jnp.sum(jnp.where(hit, ar, 0), axis=1)
    n_hit = jnp.sum(hit.astype(jnp.int32), axis=1)
    return jnp.where(n_hit > 0, idx, e)


def kda_step(state, q, k, v, g_log, beta):
    """One token of the Kimi delta rule: state [H, Dk, Dv] -> (new_state, out [H, Dv])."""
    dk = q.shape[-1]
    qn = l2norm(q) * (dk ** -0.5)
    kn = l2norm(k)
    state = state * jnp.exp(g_log)[..., None]
    kv_mem = jnp.sum(state * kn[..., None], axis=-2)
    delta = (v - kv_mem) * beta[:, None]
    state = state + kn[..., None] * delta[:, None, :]
    out = jnp.sum(state * qn[..., None], axis=-2)
    return state, out


def conv_step(conv_state, x_new, weight):
    """One token of causal depthwise conv + SiLU over a channels-last [K-1, C] state."""
    window = jnp.concatenate([conv_state, x_new], axis=0)
    out = jnp.sum(window * weight, axis=0, keepdims=True)
    return jax.nn.silu(out), window[1:]


def absorb_q(q_heads, kv_b, *, n_heads, qk_dim, v_dim, kv_lora):
    """q_abs[h] = W_kc[h]^T @ q[h], so attention runs against the latent cache."""
    outs = []
    per = qk_dim + v_dim
    for h in range(n_heads):
        w_kc = kv_b[h * per : h * per + qk_dim]
        outs.append(jax.lax.dot_general(
            q_heads[h : h + 1], w_kc, (((1,), (0,)), ((), ())),
            preferred_element_type=F32)[0])
    return jnp.stack(outs)


def project_out(latent_out, kv_b, *, n_heads, qk_dim, v_dim):
    """out[h] = W_vc[h] @ latent_out[h], the other half of the absorption."""
    outs = []
    per = qk_dim + v_dim
    for h in range(n_heads):
        w_vc = kv_b[h * per + qk_dim : (h + 1) * per]
        outs.append(jax.lax.dot_general(
            latent_out[h : h + 1], w_vc, (((1,), (1,)), ((), ())),
            preferred_element_type=F32)[0])
    return jnp.stack(outs)


def latent_attention(q_abs, kv_cache, allowed, scaling):
    """Softmax attention against the latent cache; `allowed` is [1, L], as rank-1 does not lower."""
    logits = jax.lax.dot_general(q_abs, kv_cache, (((1,), (1,)), ((), ())),
                                 preferred_element_type=F32) * scaling
    logits = jnp.where(allowed, logits, -jnp.inf)
    m = jnp.max(logits, axis=1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(logits - m)
    p = jnp.where(allowed, p, 0.0)
    p = p / (jnp.sum(p, axis=1, keepdims=True) + 1e-30)
    return jax.lax.dot_general(p, kv_cache, (((1,), (0,)), ((), ())),
                               preferred_element_type=F32)


def indexer_pool_keys(ik, ig, ape, pos, kpool):
    """-> (pool_keys [P, Dh], pool_valid [1, P]); no slot mask: invalid pools are never selected."""
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
    """iq [Hi, Dh], head_w [Hi] or block-diagonal [B, Hi], pool_keys [P, Dh] -> [1 or B, P]."""
    sc = jax.lax.dot_general(iq, pool_keys, (((1,), (1,)), ((), ())),
                             preferred_element_type=F32)
    sc = jax.nn.relu(sc * scale)
    hw = head_w if head_w.ndim == 2 else head_w.reshape(1, -1)
    return jax.lax.dot_general(hw, sc, (((1,), (0,)), ((), ())),
                               preferred_element_type=F32)
