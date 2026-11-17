# SPDX-License-Identifier: Apache-2.0
"""Untiled attention of the decode megakernel: per-request paged attention and the o-proj."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common.consts import (
    FP4_PACK,
    LANE,
    LOG2E,
    NEG,
)
from sglang_tpu.muse_megakernel.common.control import (
    _dispatch,
    _in_window,
    _sel,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _heads_rms,
    _rope_roll,
)
from sglang_tpu.muse_megakernel.common.packing import (
    _pack_bf16x2,
    _quant_fp8_rows,
)
from sglang_tpu.muse_megakernel.common.scratch import (
    _attn_p_ysend_streams,
    _can_w2_attn_scratch,
    _use_attn_p_fp8,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx


def attention(k: KernelCtx, l, qkvg_fn, use_rope, is_global, nxt, sm):
    """Returns the all-reduced o-proj output, or with fold_o None: its all-gather is

    in flight (fold_pool_router consumes it).
    """
    Hl = k.Hl
    R = k.R
    S = k.S
    allgather_sum = k.allgather_sum
    allreduce = k.allreduce
    c = k.c
    coll = k.coll
    cos_x = k.cos_x
    hd = k.hd
    hkv = k.hkv
    hq = k.hq
    kv_prologue = k.kv_prologue
    n_live = k.n_live
    newkv_out = k.newkv_out
    o_rs_streamed = k.o_rs_streamed
    pd = k.pd
    per_geom = k.per_geom
    q = k.q
    request_attention_body = k.request_attention_body
    sin_x = k.sin_x
    ws = k.ws
    per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
    with _scope("qkvg_mv"):
        qkvg = qkvg_fn()
    qd, kd = c.q_dim, c.kv_dim
    qv, kv_, vv, og = (
        qkvg[:, :qd],
        qkvg[:, qd : qd + kd],
        qkvg[:, qd + kd : qd + 2 * kd],
        qkvg[:, qd + 2 * kd :],
    )
    qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
    kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
    qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_x, sin_x), qn)
    kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_x, sin_x), kn)
    # the pool's bf16 rounding of K/V, as in the native path (K after norm+rope)
    own_dt = c.kv_dtype
    kb = kn.astype(own_dt).astype(jnp.bfloat16)
    vb = vv.astype(own_dt).astype(jnp.bfloat16)
    zpad = jnp.zeros((R, pd - hd), jnp.bfloat16)
    # ---- this step's K/V rows go out to newkv (scattered into the pools by kv_write_call);
    # attention reads them from VMEM, the cache only for positions before them.
    if c.kv_sal:  # [slot * words + word, row]: an identity matmul transposes exactly
        eye = (
            jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 0)
            == jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 1)
        ).astype(jnp.bfloat16)
        S["kv_row"][...] = jnp.concatenate(
            [
                pltpu.bitcast(
                    jax.lax.dot_general(
                        eye,
                        x[:, h * hd : (h + 1) * hd],
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    ).astype(c.kv_dtype),
                    jnp.uint32,
                )
                for h in range(hkv)
                for x in (kb, vb)
            ],
            axis=0,
        )
    for r in range(0 if c.kv_sal else R):
        if c.kv_packed:  # the projection row already holds the heads side by side
            S["kv_row"][r] = jnp.concatenate(
                [kb[r : r + 1], vb[r : r + 1]], axis=0
            ).reshape(1, 2, pd)
            continue
        parts = []
        for h in range(hkv):
            parts.append(
                jnp.concatenate(
                    [kb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                    axis=1,
                )
            )
            parts.append(
                jnp.concatenate(
                    [vb[r : r + 1, h * hd : (h + 1) * hd], zpad[r : r + 1]],
                    axis=1,
                )
            )
        S["kv_row"][r] = jnp.concatenate(parts, axis=0).reshape(hkv, 2, pd)
    kv_out_cp = pltpu.make_async_copy(
        S["kv_row"], newkv_out.at[l], S["kv_row_sem"].at[0]
    )
    kv_out_cp.start()

    # ---- attention per live request (rows of one request share their page fetches), in a
    # runtime loop: the body is emitted once, not once per bucket slot
    S["qn_s"][...] = qn
    S["kb_s"][...] = kb.astype(jnp.float32)
    S["vb_s"][...] = vb.astype(jnp.float32)
    S["attn_buf"][...] = jnp.zeros((R, c.q_dim), jnp.float32)
    with _scope("attn_core"):
        n_req_live = jnp.minimum((n_live + q - 1) // q, c.n_req)

        def run_geom(geo):
            def req_loop(j, carry):
                request_attention_body(geo, l, j, n_req_live)
                return carry

            jax.lax.fori_loop(0, n_req_live, req_loop, 0)

        per_geom(is_global, run_geom)
        kv_out_cp.wait()
    attn = S["attn_buf"][...]
    attn = _heads_rms(attn, hq, hd, c.rms_eps).astype(jnp.bfloat16).astype(jnp.float32)
    attn = (jax.nn.sigmoid(og.astype(jnp.bfloat16).astype(jnp.float32)) * attn).astype(
        jnp.bfloat16
    )
    if o_rs_streamed:
        with _scope("o_mv"):
            send, finish = coll.reduce_scatter_cols_streamed(
                "o_rs", S["xpack"], Hl // 2
            )

            def ship(m, col):
                w2 = Hl // 2
                c0 = m * w2 if isinstance(m, int) else pl.multiple_of(m * w2, LANE)
                S["xpack"][:, pl.ds(c0, w2)] = _pack_bf16x2(col)
                send(m)

            ws.mv(
                attn,
                l,
                "wo",
                nxt,
                sm=sm,
                on_col=ship,
                sm_ref=S["smalls"].at[l % 2],
            )
        with _scope("o_allreduce"):
            with _scope("o_rs_wait"):
                pad = finish()
            return allgather_sum(pad)
    with _scope("o_mv"):
        o = ws.mv(attn, l, "wo", nxt, sm=sm)
    with _scope("o_allreduce"):
        return allreduce(o)


def request_attention_body(k: KernelCtx, geo, l, j, n_req_live):
    """Both local KV heads in one MXU op per step: the query rows of all heads are

    stacked ``[hkv*q*g, .]`` (rows ``(h, t, gi)``). Padded pools: scored
    against the
    chunk's interleaved key rows ``(token, h)``, scores of the other head
    masked. Packed
    pools (kv_packed): a token's one row holds every head side by side and a
    query row
    carries its q in its head's lanes, zeros elsewhere, so the op scores only
    same-head
    pairs: one key row per token and no head mask.
    """
    He = k.He
    KV = k.KV
    R = k.R
    S = k.S
    c = k.c
    chunk_dst = k.chunk_dst
    g = k.g
    hd = k.hd
    hkv = k.hkv
    kv_prologue = k.kv_prologue
    page_src = k.page_src
    pd = k.pd
    pf_ref = k.pf_ref
    ph = k.ph
    pos_ref = k.pos_ref
    ps = k.ps
    q = k.q
    req_geom = k.req_geom
    ring_words = k.ring_words
    rows_at = k.rows_at
    wait_chunk = k.wait_chunk
    yw = k.yw
    is_global, KR_g, cpt_g = geo.is_global, geo.kr, geo.cp * ps
    r0 = pl.multiple_of(j * q, q)
    qg = q * g
    p_lo, first_ch, n_ch = req_geom(geo, j)
    qrows = rows_at(S["qn_s"], r0, q)  # [q, q_dim]

    def qrow(h, t, gi):
        x = qrows[t : t + 1, (h * g + gi) * hd : (h * g + gi + 1) * hd]
        if not c.packed_attn:
            return x
        z = jnp.zeros((1, hd), x.dtype)
        return jnp.concatenate([x if hh == h else z for hh in range(hkv)], axis=1)

    q2_raw = jnp.concatenate(
        [qrow(h, t, gi) for h in range(hkv) for t in range(q) for gi in range(g)],
        axis=0,
    )  # [hkv*qg, hd or pd] f32
    if c.attn_exp2:  # scores in log2 units: p = exp2(score)
        q2, sc = (q2_raw * (c.attn_scale * LOG2E)).astype(jnp.bfloat16), 1.0
    elif c.attn_unshifted or c.fold_attn_scale:  # power-of-two scale: exact
        q2 = q2_raw.astype(jnp.bfloat16) * jnp.asarray(c.attn_scale, jnp.bfloat16)
        sc = 1.0
    else:
        q2, sc = q2_raw.astype(jnp.bfloat16), c.attn_scale
    rowpos = jnp.concatenate(
        [jnp.broadcast_to(pos_ref[r0 + t], (g, 1)) for t in range(q)] * hkv,
        axis=0,
    )  # [hkv*qg, 1]
    rowhead = jax.lax.broadcasted_iota(jnp.int32, (hkv * qg, 1), 0) // qg

    # unshifted: m is unused and the chunk loops keep [rows, LANE] lane partials of the
    # row sums (reduced across lanes once, before the request's own rows)
    init = (
        jnp.full((hkv * qg, 1), NEG, jnp.float32),
        jnp.zeros((hkv * qg, LANE if c.attn_unshifted else 1), jnp.float32),
        jnp.zeros((hkv * qg, c.attn_width), jnp.float32),
    )
    col = jax.lax.broadcasted_iota(jnp.int32, (1, cpt_g * ph), 1)
    colhead = col % ph
    coltok = col // ph
    kdim = pd if c.packed_attn else hd

    def pv(pexp, v, v_t):
        """P @ V; ``v_t``: v is [kv_dim, tokens] (SAL chunks), else [tokens, dim]."""
        lhs = (
            pexp
            if pexp.dtype in (jnp.bfloat16, jnp.float8_e4m3fn)
            else pexp.astype(jnp.bfloat16)
        )
        return jax.lax.dot_general(
            lhs,
            v,
            (((1,), (1 if v_t else 0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    def online(carry, s_, v, v_t=False):
        m, lsum, acc = carry
        if c.attn_unshifted:
            with _scope("softmax"):
                pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                if lsum.shape[1] == LANE:
                    for lt in range(pexp.shape[1] // LANE):
                        lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                else:
                    lsum = lsum + jnp.sum(pexp, axis=1, keepdims=True)
            with _scope("pv"):
                acc = acc[:, : v.shape[0 if v_t else 1]] + pv(pexp, v, v_t)
            return m, lsum, acc
        with _scope("softmax"):
            m_new = jnp.maximum(m, jnp.max(s_, axis=1, keepdims=True))
            corr = jnp.exp(m - m_new)
            pexp = jnp.exp(s_ - m_new)
            lsum = lsum * corr + jnp.sum(pexp, axis=1, keepdims=True)
        with _scope("pv"):
            acc = acc[:, : v.shape[0 if v_t else 1]] * corr + pv(pexp, v, v_t)
        return m_new, lsum, acc

    def sal_unpack(slot):
        """A ring chunk -> K, V as [kv_dim, cpt_g], rows h * hd + dim, tokens on lanes."""
        fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
        fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
        x8 = pltpu.bitcast(
            ring_words(geo, slot)[...], c.kv_dtype
        )  # [2 * kv_heads * hd, cpt_g]
        xb = None if (fp8_k and fp8_v) else x8.astype(jnp.bfloat16)
        xk, xv = (x8 if fp8_k else xb), (x8 if fp8_v else xb)
        kk = jnp.concatenate([xk[2 * h * hd : (2 * h + 1) * hd] for h in range(hkv)], 0)
        vk = jnp.concatenate(
            [xv[(2 * h + 1) * hd : (2 * h + 2) * hd] for h in range(hkv)], 0
        )
        return kk, vk

    def sal_unpack_k(slot):
        w_hd = hd // c.sal_pack
        fp8_k = c.kv_sal_fp8 and c.sal_fp8_dot >= 1
        parts = []
        for h in range(hkv):
            xh = pltpu.bitcast(
                ring_words(geo, slot, 2 * h * w_hd, w_hd)[...], c.kv_dtype
            )
            parts.append(xh if fp8_k else xh.astype(jnp.bfloat16))
        return jnp.concatenate(parts, axis=0)

    def sal_unpack_v(slot):
        w_hd = hd // c.sal_pack
        fp8_v = c.kv_sal_fp8 and c.sal_fp8_dot >= 2
        parts = []
        for h in range(hkv):
            xh = pltpu.bitcast(
                ring_words(geo, slot, (2 * h + 1) * w_hd, w_hd)[...], c.kv_dtype
            )
            parts.append(xh if fp8_v else xh.astype(jnp.bfloat16))
        return jnp.concatenate(parts, axis=0)

    def chunk_kv(slot):
        with _scope("kv_unpack"):
            if c.kv_sal:
                return sal_unpack(slot)
            u = (
                S["kv_chunk"].at[slot].bitcast(jnp.uint32).reshape(cpt_g * ph, pd)[...]
            )  # rows (token, h)
            # u32 -> u16 truncation keeps the low half
            kk = pltpu.bitcast(u.astype(jnp.uint16), jnp.bfloat16)
            vk = pltpu.bitcast((u >> 16).astype(jnp.uint16), jnp.bfloat16)
            return kk, vk

    def chunk_k(slot):
        if c.kv_sal and c.sal_kv_split:
            with _scope("kv_unpack"):
                return sal_unpack_k(slot)
        kk, _ = chunk_kv(slot)
        return kk

    def chunk_v(slot):
        if c.kv_sal and c.sal_kv_split:
            with _scope("kv_unpack"):
                return sal_unpack_v(slot)
        _, vk = chunk_kv(slot)
        return vk

    def qk(kk):
        s_ = jax.lax.dot_general(
            q2,
            kk if c.kv_sal else kk[:, :kdim],
            (((1,), (0 if c.kv_sal else 1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return s_ * sc if sc != 1.0 else s_

    def chunk_scores(i, valid, clean=False):
        """Masked scores and V of chunk i; ``clean`` chunks lie wholly in the past of

        every row and skip the position and window masks.
        """
        slot = i % KR_g
        ci = first_ch + i
        kk, vk = chunk_kv(slot)
        live = None if c.packed_attn else colhead == rowhead
        if not clean:
            keypos = coltok + ci * cpt_g
            if c.attn_col_mask and is_global is True:
                pos_live = (keypos < p_lo) & valid
            else:
                pos_live = (keypos < p_lo) & (keypos <= rowpos) & valid
                in_win = _in_window(is_global, rowpos - keypos < c.window)
                if in_win is not None:
                    pos_live = pos_live & in_win
            live = pos_live if live is None else live & pos_live
        with _scope("qk"):
            s_ = qk(kk)
        return (s_ if live is None else jnp.where(live, s_, NEG)), vk

    def refill(ref, v):
        """Chunk v (clamped to the last one) into ring slot v % KR_g."""
        slot = v % KR_g
        ci = first_ch + jnp.minimum(v, n_ch - 1)
        for pi in range(geo.cp):
            pg_no = ci * geo.cp + pi
            ent = jnp.minimum(
                j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                c.n_pages_entries - 1,
            )
            page = pf_ref[ent]  # a global layer's table
            pltpu.make_async_copy(
                page_src(ref, page),
                chunk_dst(geo, slot, pi),
                S["kv_chunk_sems"].at[slot],
            ).start()

    def body(k, carry2, ref=None, clean=False):
        """Chunks 2k, 2k+1 as two independent online-softmax streams (two dependency

        chains for the scheduler). With ref, the two slots just read take chunks
        2k + KR_g, 2k + 1 + KR_g, issued straight-line so the copies' scalar
        work
        overlaps the vector work.
        """
        i0, i1 = 2 * k, 2 * k + 1
        valid_b = i1 < n_ch
        with _scope("kv_wait"):
            wait_chunk(geo, i0 % KR_g)

            @pl.when(valid_b)  # an odd tail re-reads chunk i0, already waited for
            def _():
                wait_chunk(geo, i1 % KR_g)

        s_a, v_a = chunk_scores(i0, True, clean)
        if c.attn_skip_idle_streams and not clean and ref is None:
            carry_b = jax.lax.cond(
                valid_b,
                lambda cr: online(cr, *chunk_scores(i1, True, False), c.kv_sal),
                lambda cr: cr,
                carry2[1],
            )
            return online(carry2[0], s_a, v_a, c.kv_sal), carry_b
        s_b, v_b = chunk_scores(
            i1 if clean else jnp.minimum(i1, n_ch - 1), valid_b, clean
        )
        out = (
            online(carry2[0], s_a, v_a, c.kv_sal),
            online(carry2[1], s_b, v_b, c.kv_sal),
        )
        if ref is not None:
            refill(ref, i0 + KR_g)
            refill(ref, i1 + KR_g)
        return out

    # Global-layer chunks before a request's last one lie wholly in the past of every
    # verify row: chunks [0, n_clean) need only the cross-head mask.
    n_clean = _sel(is_global, jnp.maximum(n_ch - 1, 0), 0)
    NS = c.attn_streams
    gls = sorted(c.global_layers)
    hoist_rows = tpu_envs.SGLANG_TPU_MUSE_MK_ATTN_HOIST_MAX_ROWS.get()
    hoist_kv = (R <= 64 or c.attn_opt == 5) if hoist_rows is None else R <= hoist_rows
    n_alias_p = _attn_p_ysend_streams(c)

    if not isinstance(n_clean, int):
        # The first KR_g chunks come from kv_prologue. Global pairs [0, k_ref) refill the
        # rest: all real chunks but the very last refill when n_ch - KR_g is odd (a copy of
        # the last chunk, drained after the loops). Sliding-window layers never refill.
        k_ref = jnp.clip((n_ch - KR_g + 1) // 2, 0, n_clean // 2)
        extra = (k_ref > 0) & (2 * k_ref - 1 + KR_g >= n_ch)

        def refill_loop(ref):
            return jax.lax.fori_loop(
                0,
                k_ref,
                functools.partial(body, ref=ref, clean=True),
                (init, init),
            )

        def branch_layers(ids, fns):
            """fns[i]() for l == ids[i], through a balanced tree of two-way branches."""
            if len(fns) == 1:
                return fns[0]()
            mid = len(fns) // 2
            return jax.lax.cond(
                l < ids[mid],
                lambda: branch_layers(ids[:mid], fns[:mid]),
                lambda: branch_layers(ids[mid:], fns[mid:]),
            )

        def hoisted(step, k_full, init0):
            """(carry, steps done) of the pipelined loop's first k_full steps, run in layer l's leaf."""
            return branch_layers(
                gls,
                [
                    lambda r=KV[x]: jax.lax.fori_loop(
                        0, k_full, functools.partial(step, ref=r), init0
                    )
                    for x in gls
                ],
            ), k_full

        use_p_fp8 = _use_attn_p_fp8(c)
        M_p = hkv * qg
        M_p_fp8 = -(-M_p // 32) * 32

        def psc_get(par, t):
            v = S["attn_psc"][par, t]
            return v[:M_p] if v.shape[0] > M_p else v

        def psc_put(par, t, psc):
            m_sc = -(-M_p // 8) * 8
            v = jnp.broadcast_to(psc, (M_p, LANE))
            if m_sc > M_p:
                v = jnp.concatenate(
                    [v, jnp.zeros((m_sc - M_p, LANE), jnp.float32)], axis=0
                )
            S["attn_psc"][par, t] = v

        def p_get(par, t):
            if use_p_fp8:
                if t >= n_alias_p:
                    raw_p = S["attn_p"][par, t - n_alias_p]
                else:
                    ys_fp8 = (
                        S["ysend"]
                        .bitcast(jnp.float8_e4m3fn)
                        .reshape(R * 4 * c.top_k, yw)
                    )
                    if 2 * cpt_g <= yw:
                        c0 = (
                            par * cpt_g
                            if isinstance(par, int)
                            else pl.multiple_of(par * cpt_g, LANE)
                        )
                        raw_p = ys_fp8[pl.ds(t * M_p_fp8, M_p_fp8), pl.ds(c0, cpt_g)]
                    else:
                        r_p = (
                            (par * n_alias_p + t) * M_p_fp8
                            if isinstance(par, int)
                            else pl.multiple_of((par * n_alias_p + t) * M_p_fp8, 32)
                        )
                        raw_p = ys_fp8[pl.ds(r_p, M_p_fp8), pl.ds(0, cpt_g)]
                return raw_p[:M_p] if M_p_fp8 > M_p else raw_p
            if t >= n_alias_p:
                return S["attn_p"][par, t - n_alias_p]
            if 2 * cpt_g <= yw:
                c0 = (
                    par * cpt_g
                    if isinstance(par, int)
                    else pl.multiple_of(par * cpt_g, LANE)
                )
                return (
                    S["ysend"]
                    .bitcast(jnp.bfloat16)[pl.ds(t * q, q), :, pl.ds(c0, cpt_g)]
                    .reshape(hkv * qg, cpt_g)
                )
            ys_bf16 = S["ysend"].bitcast(jnp.bfloat16).reshape(R * 2 * c.top_k, yw)
            r_p = (
                (par * n_alias_p + t) * M_p
                if isinstance(par, int)
                else pl.multiple_of((par * n_alias_p + t) * M_p, 8)
            )
            return ys_bf16[pl.ds(r_p, M_p), pl.ds(0, cpt_g)]

        def p_put(par, t, pb):
            if use_p_fp8:
                pb_pad = (
                    jnp.concatenate(
                        [pb, jnp.zeros((M_p_fp8 - M_p, cpt_g), pb.dtype)],
                        axis=0,
                    )
                    if M_p_fp8 > M_p
                    else pb
                )
                if t >= n_alias_p:
                    S["attn_p"][par, t - n_alias_p] = pb_pad
                    return
                ys_fp8 = (
                    S["ysend"].bitcast(jnp.float8_e4m3fn).reshape(R * 4 * c.top_k, yw)
                )
                if 2 * cpt_g <= yw:
                    c0 = (
                        par * cpt_g
                        if isinstance(par, int)
                        else pl.multiple_of(par * cpt_g, LANE)
                    )
                    ys_fp8[pl.ds(t * M_p_fp8, M_p_fp8), pl.ds(c0, cpt_g)] = pb_pad
                else:
                    r_p = (
                        (par * n_alias_p + t) * M_p_fp8
                        if isinstance(par, int)
                        else pl.multiple_of((par * n_alias_p + t) * M_p_fp8, 32)
                    )
                    ys_fp8[pl.ds(r_p, M_p_fp8), pl.ds(0, cpt_g)] = pb_pad
                return
            if t >= n_alias_p:
                S["attn_p"][par, t - n_alias_p] = pb
                return
            if 2 * cpt_g <= yw:
                c0 = (
                    par * cpt_g
                    if isinstance(par, int)
                    else pl.multiple_of(par * cpt_g, LANE)
                )
                S["ysend"].bitcast(jnp.bfloat16)[
                    pl.ds(t * q, q), :, pl.ds(c0, cpt_g)
                ] = pb.reshape(q, 2 * c.top_k, cpt_g)
                return
            ys_bf16 = S["ysend"].bitcast(jnp.bfloat16).reshape(R * 2 * c.top_k, yw)
            r_p = (
                (par * n_alias_p + t) * M_p
                if isinstance(par, int)
                else pl.multiple_of((par * n_alias_p + t) * M_p, 8)
            )
            ys_bf16[pl.ds(r_p, M_p), pl.ds(0, cpt_g)] = pb

        def qk_exp(i, lsum):
            """Probabilities (bf16) of clean chunk i and the lane partials of their row sums."""
            kk = chunk_k(i % KR_g)
            with _scope("qk"):
                s_ = qk(kk)
            with _scope("softmax"):
                pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                for lt in range(cpt_g // LANE):
                    lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
            return pexp.astype(jnp.bfloat16), lsum

        def pv_chunk(i, pb):
            vk = chunk_v(i % KR_g)
            with _scope("pv"):
                return pv(pb, vk, c.kv_sal)

        def refill_global(v0):
            """Chunks [v0, v0 + NS) that exist into their ring slots: one walk over the global

            layers' pools, its leaf a small loop.
            """
            n = jnp.clip(n_ch - v0, 0, NS)

            def leaf(ref):
                def one(u, carry):
                    v = v0 + u
                    for pi in range(geo.cp):
                        ent = jnp.minimum(
                            j * c.pages_per_seq
                            + jnp.minimum(
                                (first_ch + v) * geo.cp + pi,
                                c.pages_per_seq - 1,
                            ),
                            c.n_pages_entries - 1,
                        )
                        pltpu.make_async_copy(
                            page_src(ref, pf_ref[ent]),
                            chunk_dst(geo, v % KR_g, pi),
                            S["kv_chunk_sems"].at[v % KR_g],
                        ).start()
                    return carry

                jax.lax.fori_loop(0, n, one, 0)

            @pl.when(n > 0)
            def _():
                _dispatch(l, KV, leaf, only=gls)

        def refill_slot(ref, v):
            """Chunk v (known < n_ch) into ring slot v % KR_g, straight-line."""
            slot = v % KR_g
            ci = first_ch + v
            for pi in range(geo.cp):
                pg_no = ci * geo.cp + pi
                ent = jnp.minimum(
                    j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                    c.n_pages_entries - 1,
                )
                pltpu.make_async_copy(
                    page_src(ref, pf_ref[ent]),
                    chunk_dst(geo, slot, pi),
                    S["kv_chunk_sems"].at[slot],
                ).start()

        def pipelined_opt(n_it):
            """Hoisted straight-line KV refill + 2-acc PV/QK pipeline (stack2 for q<=4)."""
            M = hkv * qg
            use_fp8_q = (
                (c.attn_opt == 4)
                and c.attn_exp2
                and c.kv_sal_fp8
                and c.sal_fp8_dot >= 1
            )
            if use_fp8_q:
                q_lhs, q_sc = _quant_fp8_rows(q2_raw * (c.attn_scale * LOG2E))
            else:
                q_lhs = q2
                q_sc = sc
            use_fp8_p = use_p_fp8

            def _scale_pv(dpv, psc_v):
                return (
                    dpv * psc_v
                    if psc_v.shape[1] == dpv.shape[1]
                    else dpv * psc_v[:, 0:1]
                )

            use_stack2 = M <= 64 and NS % 2 == 0
            if use_stack2:
                zq = jnp.zeros((M, q_lhs.shape[1]), q_lhs.dtype)
                q2_blk = jnp.concatenate(
                    [
                        jnp.concatenate([q_lhs, zq], axis=1),
                        jnp.concatenate([zq, q_lhs], axis=1),
                    ],
                    axis=0,
                )

            def qk_exp_opt(i, lsum):
                kk = chunk_k(i % KR_g)
                with _scope("qk"):
                    s_ = jax.lax.dot_general(
                        q_lhs,
                        kk,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    if use_fp8_q or q_sc != 1.0:
                        s_ = s_ * q_sc
                with _scope("softmax"):
                    pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                    for lt in range(cpt_g // LANE):
                        lsum = lsum + pexp[:, lt * LANE : (lt + 1) * LANE]
                    if use_fp8_p:
                        pq, psc = _quant_fp8_rows(pexp, nonneg=True)
                        return (pq, psc), lsum
                return pexp.astype(jnp.bfloat16), lsum

            def qk_pair(i_a, i_b, ls0, ls1):
                ka = chunk_k(i_a % KR_g)
                kb_ = chunk_k(i_b % KR_g)
                k_cat = jnp.concatenate([ka, kb_], axis=0)
                with _scope("qk"):
                    sb = jax.lax.dot_general(
                        q2_blk,
                        k_cat,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    sa, sb_ = sb[:M], sb[M:]
                    if use_fp8_q or q_sc != 1.0:
                        sa, sb_ = sa * q_sc, sb_ * q_sc
                with _scope("softmax"):
                    pa = jnp.exp2(sa) if c.attn_exp2 else jnp.exp(sa)
                    pb = jnp.exp2(sb_) if c.attn_exp2 else jnp.exp(sb_)
                    for lt in range(cpt_g // LANE):
                        ls0 = ls0 + pa[:, lt * LANE : (lt + 1) * LANE]
                        ls1 = ls1 + pb[:, lt * LANE : (lt + 1) * LANE]
                    if use_fp8_p:
                        pab_q, psc_ab = _quant_fp8_rows(
                            jnp.concatenate([pa, pb], axis=1), nonneg=True
                        )
                        return (
                            pab_q[:, :cpt_g],
                            pab_q[:, cpt_g:],
                            psc_ab,
                            ls0,
                            ls1,
                        )
                return (
                    pa.astype(jnp.bfloat16),
                    pb.astype(jnp.bfloat16),
                    None,
                    ls0,
                    ls1,
                )

            def pv_pair(i_a, i_b, pa_b, pb_b):
                va = chunk_v(i_a % KR_g)
                vb_ = chunk_v(i_b % KR_g)
                p_cat = jnp.concatenate([pa_b, pb_b], axis=1)
                v_cat = jnp.concatenate([va, vb_], axis=1)
                with _scope("pv"):
                    return jax.lax.dot_general(
                        p_cat,
                        v_cat,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )

            zl = jnp.zeros((M, LANE), jnp.float32)
            za = jnp.zeros((M, c.attn_width), jnp.float32)
            for t in range(NS):
                wait_chunk(geo, t % KR_g)
            if not use_stack2 and _can_w2_attn_scratch(c):
                w_hd = hd // c.sal_pack
                w2_rows = (c.expert_depth + 1) * (c.inter_local // FP4_PACK)
                w2_f32 = S["w2"].bitcast(jnp.float32).reshape(w2_rows, He)
                qh_list = []
                q_sch = []
                for h in range(hkv):
                    qr_h = q2_raw[h * qg : (h + 1) * qg]
                    if qg < 64:
                        qr_h = jnp.concatenate(
                            [
                                qr_h,
                                jnp.zeros((64 - qg, qr_h.shape[1]), jnp.float32),
                            ],
                            axis=0,
                        )
                    if use_fp8_q:
                        qh_q, qh_s = _quant_fp8_rows(qr_h * (c.attn_scale * LOG2E))
                        qh_list.append(qh_q)
                        q_sch.append(qh_s)
                    else:
                        qh_list.append(
                            (qr_h * (c.attn_scale * LOG2E)).astype(jnp.bfloat16)
                        )
                        q_sch.append(1.0)
                p_sub_dt = jnp.float8_e4m3fn if c.attn_head_fp8 >= 2 else jnp.bfloat16
                ys_sub_rows = R * (4 if c.attn_head_fp8 >= 2 else 2) * c.top_k

                def p_get_sub(par, t, h, s0):
                    if t >= n_alias_p:
                        return S["attn_p"][
                            par,
                            t - n_alias_p,
                            pl.ds(h * 64, 64),
                            pl.ds(s0, 512),
                        ]
                    ys_2d = S["ysend"].bitcast(p_sub_dt).reshape(ys_sub_rows, yw)
                    if 2 * cpt_g <= yw:
                        c0 = (
                            par * cpt_g + s0
                            if isinstance(par, int)
                            else pl.multiple_of(par * cpt_g + s0, LANE)
                        )
                        return ys_2d[pl.ds(t * 128 + h * 64, 64), pl.ds(c0, 512)]
                    r_sub = (par * n_alias_p + t) * 128 + h * 64
                    r_sub = r_sub if isinstance(par, int) else pl.multiple_of(r_sub, 64)
                    return ys_2d[pl.ds(r_sub, 64), pl.ds(s0, 512)]

                def p_put_sub(par, t, h, s0, pb_sub):
                    if t >= n_alias_p:
                        S["attn_p"][
                            par,
                            t - n_alias_p,
                            pl.ds(h * 64, 64),
                            pl.ds(s0, 512),
                        ] = pb_sub
                        return
                    ys_2d = S["ysend"].bitcast(p_sub_dt).reshape(ys_sub_rows, yw)
                    if 2 * cpt_g <= yw:
                        c0 = (
                            par * cpt_g + s0
                            if isinstance(par, int)
                            else pl.multiple_of(par * cpt_g + s0, LANE)
                        )
                        ys_2d[pl.ds(t * 128 + h * 64, 64), pl.ds(c0, 512)] = pb_sub
                        return
                    r_sub = (par * n_alias_p + t) * 128 + h * 64
                    r_sub = r_sub if isinstance(par, int) else pl.multiple_of(r_sub, 64)
                    ys_2d[pl.ds(r_sub, 64), pl.ds(s0, 512)] = pb_sub

                def psc_get_w2(par, t, h, s_idx):
                    r_sc = (
                        64 + par * 64
                        if isinstance(par, int)
                        else pl.multiple_of(64 + par * 64, 64)
                    )
                    c_sc = (4 * t + 2 * h + s_idx) * LANE
                    return w2_f32[pl.ds(r_sc, 64), pl.ds(c_sc, LANE)]

                def psc_put_w2(par, t, h, s_idx, v_sc):
                    r_sc = (
                        64 + par * 64
                        if isinstance(par, int)
                        else pl.multiple_of(64 + par * 64, 64)
                    )
                    c_sc = (4 * t + 2 * h + s_idx) * LANE
                    w2_f32[pl.ds(r_sc, 64), pl.ds(c_sc, LANE)] = v_sc

                z_kv = jnp.zeros((64, 512), jnp.float8_e4m3fn)

                def kh_half(slot, h, s0):
                    with _scope("kv_unpack"):
                        kr = pltpu.bitcast(
                            ring_words(geo, slot, 2 * h * w_hd, w_hd)[
                                :, pl.ds(s0, 512)
                            ],
                            jnp.float8_e4m3fn,
                        )
                        return jnp.concatenate(
                            [kr, z_kv] if h == 0 else [z_kv, kr], axis=0
                        )

                def vh_half(slot, h, s0):
                    with _scope("kv_unpack"):
                        vr = pltpu.bitcast(
                            ring_words(geo, slot, (2 * h + 1) * w_hd, w_hd)[
                                :, pl.ds(s0, 512)
                            ],
                            jnp.float8_e4m3fn,
                        )
                        return jnp.concatenate(
                            [vr, z_kv] if h == 0 else [z_kv, vr], axis=0
                        )

                def qk_all_h(i_start, par):
                    for h in range(hkv):
                        ls_h = w2_f32[pl.ds(0, 64), pl.ds((2 + h) * LANE, LANE)]
                        qh = qh_list[h]
                        qsc_h = q_sch[h]
                        for t in range(NS):
                            slot = (i_start + t) % KR_g
                            for s_idx, s0 in enumerate((0, 512)):
                                kp = kh_half(slot, h, s0)
                                with _scope("qk"):
                                    s_h = jax.lax.dot_general(
                                        qh,
                                        kp,
                                        (((1,), (0,)), ((), ())),
                                        preferred_element_type=jnp.float32,
                                    )
                                    if use_fp8_q:
                                        s_h = s_h * qsc_h
                                with _scope("softmax"):
                                    if c.attn_head_fp8 >= 2:
                                        pe = jnp.exp2(s_h)
                                        sm0 = jnp.maximum(pe[:, :128], pe[:, 128:256])
                                        sm1 = jnp.maximum(
                                            pe[:, 256:384], pe[:, 384:512]
                                        )
                                        amax = jnp.max(
                                            jnp.maximum(sm0, sm1),
                                            axis=1,
                                            keepdims=True,
                                        )
                                        psc = jnp.where(
                                            amax == 0,
                                            jnp.zeros_like(amax),
                                            amax * jnp.float32(1.0 / 448.0),
                                        )
                                        inv = jnp.where(
                                            amax == 0,
                                            jnp.zeros_like(amax),
                                            jnp.float32(448.0)
                                            / jnp.maximum(amax, jnp.float32(1e-30)),
                                        )
                                        pf = (pe * inv).astype(jnp.float8_e4m3fn)
                                        pf_f = pf.astype(jnp.float32)
                                        dd = (
                                            (pf_f[:, :128] + pf_f[:, 128:256])
                                            + (pf_f[:, 256:384] + pf_f[:, 384:512])
                                        ) * psc
                                        ls_h = ls_h + dd
                                        p_put_sub(par, t, h, s0, pf)
                                        psc_put_w2(
                                            par,
                                            t,
                                            h,
                                            s_idx,
                                            jnp.broadcast_to(psc, (64, LANE)),
                                        )
                                    else:
                                        pe = jnp.exp2(s_h)
                                        dd = (pe[:, :128] + pe[:, 128:256]) + (
                                            pe[:, 256:384] + pe[:, 384:512]
                                        )
                                        ls_h = ls_h + dd
                                        p_put_sub(
                                            par,
                                            t,
                                            h,
                                            s0,
                                            pe.astype(jnp.bfloat16),
                                        )
                        w2_f32[pl.ds(0, 64), pl.ds((2 + h) * LANE, LANE)] = ls_h

                def pv_all_h(i0, par):
                    for a_idx in (0, 1):
                        acc_cur = w2_f32[pl.ds(0, 64), pl.ds(a_idx * LANE, LANE)]
                        for t in range(a_idx, NS, 2):
                            slot = (i0 + t) % KR_g
                            for h in range(hkv):
                                for s_idx, s0 in enumerate((0, 512)):
                                    pb_s = p_get_sub(par, t, h, s0)
                                    vp_s = vh_half(slot, h, s0)
                                    with _scope("pv"):
                                        dpv = jax.lax.dot_general(
                                            pb_s,
                                            vp_s,
                                            (((1,), (1,)), ((), ())),
                                            preferred_element_type=jnp.float32,
                                        )
                                    if c.attn_head_fp8 >= 2:
                                        dpv = dpv * psc_get_w2(par, t, h, s_idx)
                                    acc_cur = acc_cur + dpv
                        w2_f32[pl.ds(0, 64), pl.ds(a_idx * LANE, LANE)] = acc_cur

                z64 = jnp.zeros((64, LANE), jnp.float32)
                for b_i in range(4):
                    w2_f32[pl.ds(0, 64), pl.ds(b_i * LANE, LANE)] = z64
                qk_all_h(0, 0)

                def step_h(k, dummy, ref=None):
                    i0 = NS * k
                    par, npar = k % 2, (k + 1) % 2
                    with _scope("kv_wait"):
                        for t in range(NS):
                            wait_chunk(geo, (i0 + NS + t) % KR_g)
                    pv_all_h(i0, par)
                    qk_all_h(i0 + NS, npar)
                    if ref is not None:
                        for t in range(NS):
                            refill_slot(ref, i0 + KR_g + t)
                    else:
                        refill_global(i0 + KR_g)
                    return dummy

                k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                if isinstance(l, int):
                    jax.lax.fori_loop(
                        0,
                        k_full,
                        functools.partial(step_h, ref=KV[l]),
                        jnp.int32(0),
                    )
                elif hoist_kv:
                    _, k_full = hoisted(step_h, k_full, jnp.int32(0))
                else:
                    k_full = 0
                jax.lax.fori_loop(k_full, n_it - 1, step_h, jnp.int32(0))
                pv_all_h(NS * (n_it - 1), (n_it - 1) % 2)
                refill_global(NS * (n_it - 1) + KR_g)
                acc_pk = (
                    w2_f32[pl.ds(0, 64), pl.ds(0, LANE)]
                    + w2_f32[pl.ds(0, 64), pl.ds(LANE, LANE)]
                )[:qg]
                col_i = jax.lax.broadcasted_iota(jnp.int32, (qg, LANE), 1)
                acc_out = jnp.concatenate(
                    [
                        jnp.where(col_i < hd, acc_pk, 0.0),
                        jnp.where(col_i >= hd, acc_pk, 0.0),
                    ],
                    axis=0,
                )
                ls_out = jnp.concatenate(
                    [
                        w2_f32[pl.ds(0, 64), pl.ds(2 * LANE, LANE)][:qg],
                        w2_f32[pl.ds(0, 64), pl.ds(3 * LANE, LANE)][:qg],
                    ],
                    axis=0,
                )
                return (init[0], ls_out, acc_out), init
            if not use_stack2 and c.attn_carry:
                n_acc = 1 if c.attn_carry == 1 else 2

                def qk_tree(i):
                    kk = chunk_k(i % KR_g)
                    with _scope("qk"):
                        s_ = jax.lax.dot_general(
                            q_lhs,
                            kk,
                            (((1,), (0,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        if use_fp8_q or q_sc != 1.0:
                            s_ = s_ * q_sc
                    with _scope("softmax"):
                        if c.attn_carry == 3:
                            pb = jnp.exp2(s_.astype(jnp.bfloat16))
                            parts = [
                                pb[:, lt * LANE : (lt + 1) * LANE]
                                for lt in range(cpt_g // LANE)
                            ]
                            while len(parts) > 1:
                                parts = [
                                    parts[j] + parts[j + 1]
                                    for j in range(0, len(parts), 2)
                                ]
                            return pb, parts[0].astype(jnp.float32)
                        pexp = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                        parts = [
                            pexp[:, lt * LANE : (lt + 1) * LANE]
                            for lt in range(cpt_g // LANE)
                        ]
                        while len(parts) > 1:
                            parts = [
                                parts[j] + parts[j + 1] for j in range(0, len(parts), 2)
                            ]
                        if use_fp8_p:
                            pf, psc = _quant_fp8_rows(pexp, nonneg=True)
                            return (pf, psc), parts[0]
                    return pexp.astype(jnp.bfloat16), parts[0]

                pair = c.attn_carry == 4
                if pair:
                    zq_p = jnp.zeros((M, q_lhs.shape[1]), q_lhs.dtype)
                    q_blk_p = jnp.concatenate(
                        [
                            jnp.concatenate([q_lhs, zq_p], axis=1),
                            jnp.concatenate([zq_p, q_lhs], axis=1),
                        ],
                        axis=0,
                    )

                def lane_tree(x):
                    parts = [
                        x[:, lt * LANE : (lt + 1) * LANE] for lt in range(cpt_g // LANE)
                    ]
                    while len(parts) > 1:
                        parts = [
                            parts[j] + parts[j + 1] for j in range(0, len(parts), 2)
                        ]
                    return parts[0]

                def qk_all(i_start, par):
                    d = zl
                    if not pair:
                        for t in range(NS):
                            pb, dd = qk_tree(i_start + t)
                            if use_fp8_p:
                                p_put(par, t, pb[0])
                                psc_put(par, t, pb[1])
                            else:
                                p_put(par, t, pb)
                            d = d + dd
                        return d
                    for u in range(NS // 2):
                        ka = chunk_k((i_start + 2 * u) % KR_g)
                        kb2 = chunk_k((i_start + 2 * u + 1) % KR_g)
                        with _scope("qk"):
                            s2 = jax.lax.dot_general(
                                q_blk_p,
                                jnp.concatenate([ka, kb2], axis=0),
                                (((1,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32,
                            )
                        for h, s_ in enumerate((s2[:M], s2[M:])):
                            if use_fp8_q or q_sc != 1.0:
                                s_ = s_ * q_sc
                            with _scope("softmax"):
                                pe = jnp.exp2(s_) if c.attn_exp2 else jnp.exp(s_)
                                d = d + lane_tree(pe)
                                if use_fp8_p:
                                    pf, psc = _quant_fp8_rows(pe, nonneg=True)
                                    p_put(par, 2 * u + h, pf)
                                    psc_put(par, 2 * u + h, psc)
                                else:
                                    p_put(par, 2 * u + h, pe.astype(jnp.bfloat16))
                    return d

                ls = qk_all(0, 0)

                def pv_all(i0, par, accs):
                    accs = list(accs)
                    if pair:
                        for u in range(NS // 2):
                            va = chunk_v((i0 + 2 * u) % KR_g)
                            vb2 = chunk_v((i0 + 2 * u + 1) % KR_g)
                            pst = jnp.concatenate(
                                [p_get(par, 2 * u), p_get(par, 2 * u + 1)],
                                axis=0,
                            )
                            with _scope("pv"):
                                o2 = jax.lax.dot_general(
                                    pst,
                                    jnp.concatenate([va, vb2], axis=0),
                                    (((1,), (1,)), ((), ())),
                                    preferred_element_type=jnp.float32,
                                )
                            W = c.attn_width
                            if use_fp8_p:
                                part_pv = _scale_pv(
                                    o2[:M, :W], psc_get(par, 2 * u)
                                ) + _scale_pv(
                                    o2[M:, W : 2 * W], psc_get(par, 2 * u + 1)
                                )
                            else:
                                part_pv = o2[:M, :W] + o2[M:, W : 2 * W]
                            accs[u % n_acc] = accs[u % n_acc] + part_pv
                        return accs
                    for t in range(NS):
                        pv_t = pv_chunk(i0 + t, p_get(par, t))
                        if use_fp8_p:
                            pv_t = _scale_pv(pv_t, psc_get(par, t))
                        accs[t % n_acc] = accs[t % n_acc] + pv_t
                    return accs

                def step_c(k, carry, ref=None):
                    ls, accs = carry
                    i0 = NS * k
                    par, npar = k % 2, (k + 1) % 2
                    with _scope("kv_wait"):
                        for t in range(NS):
                            wait_chunk(geo, (i0 + NS + t) % KR_g)
                    accs = pv_all(i0, par, accs)
                    ls = ls + qk_all(i0 + NS, npar)
                    if ref is not None:
                        for t in range(NS):
                            refill_slot(ref, i0 + KR_g + t)
                    else:
                        refill_global(i0 + KR_g)
                    return ls, accs

                k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                init_c = (ls, [za] * n_acc)
                if isinstance(l, int):
                    carry_c = jax.lax.fori_loop(
                        0, k_full, functools.partial(step_c, ref=KV[l]), init_c
                    )
                elif hoist_kv:
                    carry_c, k_full = hoisted(step_c, k_full, init_c)
                else:
                    carry_c, k_full = init_c, 0
                ls, accs = jax.lax.fori_loop(k_full, n_it - 1, step_c, carry_c)
                accs = pv_all(NS * (n_it - 1), (n_it - 1) % 2, accs)
                refill_global(NS * (n_it - 1) + KR_g)
                return (init[0], ls, sum(accs[1:], accs[0])), init
            if not use_stack2:
                lsums = []
                for t in range(NS):
                    pb, ls_ = qk_exp_opt(t, zl)
                    if use_fp8_p:
                        p_put(0, t, pb[0])
                        psc_put(0, t, pb[1])
                    else:
                        p_put(0, t, pb)
                    lsums.append(ls_)

                use_acc2 = R > 64

                def step_ns(k, carry, ref=None):
                    lsums, accs = carry
                    i0 = NS * k
                    par, npar = k % 2, (k + 1) % 2
                    with _scope("kv_wait"):
                        for t in range(NS):
                            wait_chunk(geo, (i0 + NS + t) % KR_g)
                    pbs = [p_get(par, t) for t in range(NS)]
                    if use_acc2:
                        ac0, ac1 = accs
                        for t, pb in enumerate(pbs):
                            dpv = pv_chunk(i0 + t, pb)
                            if use_fp8_p:
                                dpv = _scale_pv(dpv, psc_get(par, t))
                            if t % 2 == 0:
                                ac0 = ac0 + dpv
                            else:
                                ac1 = ac1 + dpv
                        accs = [ac0, ac1]
                    else:
                        new_accs = []
                        for t, (a, pb) in enumerate(zip(accs, pbs)):
                            dpv = pv_chunk(i0 + t, pb)
                            if use_fp8_p:
                                dpv = _scale_pv(dpv, psc_get(par, t))
                            new_accs.append(a + dpv)
                        accs = new_accs
                    nxt = [qk_exp_opt(i0 + NS + t, lsums[t]) for t in range(NS)]
                    for t in range(NS):
                        if use_fp8_p:
                            p_put(npar, t, nxt[t][0][0])
                            psc_put(npar, t, nxt[t][0][1])
                        else:
                            p_put(npar, t, nxt[t][0])
                    if ref is not None:
                        for t in range(NS):
                            refill_slot(ref, i0 + KR_g + t)
                    else:
                        refill_global(i0 + KR_g)
                    return [x[1] for x in nxt], accs

                k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
                init_ns = (lsums, [za, za] if use_acc2 else [za] * NS)
                hoist_ns = hoist_kv
                if isinstance(l, int):
                    carry_ns = jax.lax.fori_loop(
                        0,
                        k_full,
                        functools.partial(step_ns, ref=KV[l]),
                        init_ns,
                    )
                elif hoist_ns:
                    carry_ns, k_full = hoisted(step_ns, k_full, init_ns)
                else:
                    carry_ns = init_ns
                    k_full = 0
                lsums, accs = jax.lax.fori_loop(k_full, n_it - 1, step_ns, carry_ns)
                i0 = NS * (n_it - 1)
                last_par = (n_it - 1) % 2
                if use_acc2:
                    ac0, ac1 = accs
                    for t in range(NS):
                        dpv = pv_chunk(i0 + t, p_get(last_par, t))
                        if use_fp8_p:
                            dpv = _scale_pv(dpv, psc_get(last_par, t))
                        if t % 2 == 0:
                            ac0 = ac0 + dpv
                        else:
                            ac1 = ac1 + dpv
                    accs = [ac0, ac1]
                else:
                    new_accs = []
                    for t, a in enumerate(accs):
                        dpv = pv_chunk(i0 + t, p_get(last_par, t))
                        if use_fp8_p:
                            dpv = _scale_pv(dpv, psc_get(last_par, t))
                        new_accs.append(a + dpv)
                    accs = new_accs
                refill_global(i0 + KR_g)
                return (
                    init[0],
                    sum(lsums[1:], lsums[0]),
                    sum(accs[1:], accs[0]),
                ), init

            ls0, ls1 = zl, zl
            for u in range(NS // 2):
                pa, pb, psc_ab, ls0, ls1 = qk_pair(2 * u, 2 * u + 1, ls0, ls1)
                p_put(0, 2 * u, pa)
                p_put(0, 2 * u + 1, pb)
                if use_fp8_p:
                    psc_put(0, 2 * u, psc_ab)

            def step_body(k, carry, ref=None):
                ls0, ls1, ac0, ac1 = carry
                i0 = NS * k
                par, npar = k % 2, (k + 1) % 2
                with _scope("kv_wait"):
                    for t in range(NS):
                        wait_chunk(geo, (i0 + NS + t) % KR_g)
                for u in range(NS // 2):
                    dpv = pv_pair(
                        i0 + 2 * u,
                        i0 + 2 * u + 1,
                        p_get(par, 2 * u),
                        p_get(par, 2 * u + 1),
                    )
                    if use_fp8_p:
                        dpv = _scale_pv(dpv, psc_get(par, 2 * u))
                    if u % 2 == 0:
                        ac0 = ac0 + dpv
                    else:
                        ac1 = ac1 + dpv
                for u in range(NS // 2):
                    pa, pb, psc_ab, ls0, ls1 = qk_pair(
                        i0 + NS + 2 * u, i0 + NS + 2 * u + 1, ls0, ls1
                    )
                    p_put(npar, 2 * u, pa)
                    p_put(npar, 2 * u + 1, pb)
                    if use_fp8_p:
                        psc_put(npar, 2 * u, psc_ab)
                if ref is not None:
                    for t in range(NS):
                        refill_slot(ref, i0 + KR_g + t)
                else:
                    refill_global(i0 + KR_g)
                return ls0, ls1, ac0, ac1

            k_full = jnp.clip((n_ch - KR_g) // NS, 0, n_it - 1)
            init4 = (ls0, ls1, za, za)
            if isinstance(l, int):
                carry4 = jax.lax.fori_loop(
                    0, k_full, functools.partial(step_body, ref=KV[l]), init4
                )
            elif hoist_rows is None or hoist_kv:
                carry4, k_full = hoisted(step_body, k_full, init4)
            else:
                carry4, k_full = init4, 0
            ls0, ls1, ac0, ac1 = jax.lax.fori_loop(k_full, n_it - 1, step_body, carry4)
            i0 = NS * (n_it - 1)
            par = (n_it - 1) % 2
            for u in range(NS // 2):
                dpv = pv_pair(
                    i0 + 2 * u,
                    i0 + 2 * u + 1,
                    p_get(par, 2 * u),
                    p_get(par, 2 * u + 1),
                )
                if use_fp8_p:
                    dpv = _scale_pv(dpv, psc_get(par, 2 * u))
                if u % 2 == 0:
                    ac0 = ac0 + dpv
                else:
                    ac1 = ac1 + dpv
            refill_global(i0 + KR_g)
            return (init[0], ls0 + ls1, ac0 + ac1), init

        def pipelined(n_it):
            """Clean chunks [0, NS * n_it), n_it >= 1."""
            zl = jnp.zeros((hkv * qg, LANE), jnp.float32)
            for t in range(NS):
                wait_chunk(geo, t % KR_g)
            lsums = []
            for t in range(NS):
                pb, ls_ = qk_exp(t, zl)
                p_put(0, t, pb)
                lsums.append(ls_)

            def it(k, carry):
                lsums, accs = carry
                i0 = NS * k
                with _scope("kv_wait"):
                    for t in range(NS):
                        wait_chunk(geo, (i0 + NS + t) % KR_g)
                pbs = [p_get(k % 2, t) for t in range(NS)]
                accs = [
                    a + pv_chunk(i0 + t, pb) for t, (a, pb) in enumerate(zip(accs, pbs))
                ]
                nxt = [qk_exp(i0 + NS + t, lsums[t]) for t in range(NS)]
                for t in range(NS):
                    p_put((k + 1) % 2, t, nxt[t][0])
                refill_global(i0 + KR_g)
                return [x[1] for x in nxt], accs

            za = jnp.zeros((hkv * qg, c.attn_width), jnp.float32)
            lsums, accs = jax.lax.fori_loop(0, n_it - 1, it, (lsums, [za] * NS))
            i0 = NS * (n_it - 1)
            accs = [
                a + pv_chunk(i0 + t, p_get((n_it - 1) % 2, t))
                for t, a in enumerate(accs)
            ]
            refill_global(i0 + KR_g)
            return (
                init[0],
                sum(lsums[1:], lsums[0]),
                sum(accs[1:], accs[0]),
            ), init

    with _scope("chunk_loop"):
        if isinstance(n_clean, int):
            (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                0, (n_ch + 1) // 2, body, (init, init)
            )
        else:
            if (
                c.attn_pipe
                and c.attn_opt in (3, 4, 5)
                and c.kv_sal
                and (
                    c.row_tile > 0
                    or hkv * qg <= 64
                    or c.attn_carry > 0
                    or (c.attn_head_fp8 > 0)
                )
            ):
                if not gls or (isinstance(l, int) and l not in gls):
                    carry2, k0 = (init, init), 0
                else:
                    n_it = n_clean // NS
                    carry2 = jax.lax.cond(
                        n_it > 0,
                        lambda: pipelined_opt(n_it),
                        lambda: (init, init),
                    )
                    k0 = n_it * NS // 2
            elif c.attn_pipe:
                # global chunks before the last one are clean; SWA layers have none
                n_it = n_clean // NS
                carry2 = jax.lax.cond(
                    n_it > 0, lambda: pipelined(n_it), lambda: (init, init)
                )
                k0 = n_it * NS // 2
            elif isinstance(l, int):
                carry2 = refill_loop(KV[l]) if l in gls else (init, init)
                k0 = k_ref if l in gls else 0
            else:
                carry2 = jax.lax.cond(
                    is_global,
                    lambda: branch_layers(
                        gls,
                        [functools.partial(refill_loop, KV[x]) for x in gls],
                    ),
                    lambda: (init, init),
                )
                k0 = _sel(is_global, k_ref, 0)
            carry2 = jax.lax.fori_loop(
                k0, n_clean // 2, functools.partial(body, clean=True), carry2
            )
            (m_a, l_a, acc_a), (m_b, l_b, acc_b) = jax.lax.fori_loop(
                n_clean // 2, (n_ch + 1) // 2, body, carry2
            )

            if not c.attn_pipe:

                @pl.when(extra & is_global)
                def _():
                    wait_chunk(geo, (2 * k_ref - 1 + KR_g) % KR_g)

        if c.attn_unshifted:  # the streams just add
            carry = (
                m_a,
                jnp.sum(l_a + l_b, axis=1, keepdims=True),
                acc_a + acc_b,
            )
        else:
            # a stream that saw no live key keeps m = NEG: weight exp(NEG - m) = 0
            m_ab = jnp.maximum(m_a, m_b)
            w_a, w_b = jnp.exp(m_a - m_ab), jnp.exp(m_b - m_ab)
            carry = (m_ab, l_a * w_a + l_b * w_b, acc_a * w_a + acc_b * w_b)

    # every ring slot is read and every copy waited for: the next request's first
    # chunks fly during this request's own-row attention (request 0's were requested
    # before the qkv matvec)
    @pl.when(j + 1 < n_req_live)
    def _():
        with _scope("kv_prologue"):
            kv_prologue(geo, l, j + 1)

    # the request's own new rows (causal within a verify block), padded to a sublane
    # tile (a 1-row operand does not lower), filler keys masked. Padded pools: key rows
    # (h, t); packed: the projection rows as they are, heads side by side.
    with _scope("own_rows"):
        q8 = -(-q // 8) * 8
        krows = rows_at(S["kb_s"], r0, q).astype(jnp.bfloat16)
        vrows = rows_at(S["vb_s"], r0, q).astype(jnp.bfloat16)
        if c.packed_attn:
            zrows = (
                jnp.zeros((q8 - q, c.kv_dim), jnp.bfloat16) if q8 > q else None
            )  # Mosaic has no 0-row vectors
            kn = jnp.concatenate([krows, zrows], axis=0) if q8 > q else krows
            vn = jnp.concatenate([vrows, zrows], axis=0) if q8 > q else vrows
            nt = jax.lax.broadcasted_iota(jnp.int32, (1, q8), 1)
            head_ok = True
        else:
            zrows = jnp.zeros((q8 - q, hd), jnp.bfloat16) if q8 > q else None
            kn = jnp.concatenate(
                [
                    x
                    for h in range(hkv)
                    for x in [krows[:, h * hd : (h + 1) * hd]]
                    + ([zrows] if q8 > q else [])
                ],
                axis=0,
            )
            vn = jnp.concatenate(
                [
                    x
                    for h in range(hkv)
                    for x in [vrows[:, h * hd : (h + 1) * hd]]
                    + ([zrows] if q8 > q else [])
                ],
                axis=0,
            )
            ncol = jax.lax.broadcasted_iota(jnp.int32, (1, hkv * q8), 1)
            nt = ncol % q8
            head_ok = ncol // q8 == rowhead
        newpos = jnp.zeros(nt.shape, jnp.int32) + 2**30
        for t in range(q):
            newpos = jnp.where(nt == t, pos_ref[r0 + t], newpos)
        live_new = head_ok & (newpos <= rowpos)
        in_win = _in_window(is_global, rowpos - newpos < c.window)
        if in_win is not None:
            live_new = live_new & in_win
        s_ = jax.lax.dot_general(
            q2, kn, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if sc != 1.0:
            s_ = s_ * sc
        _, lsum, acc = online(carry, jnp.where(live_new, s_, NEG), vn)
        res = acc / jnp.maximum(lsum, 1e-30)  # [hkv*qg, hd or pd]
        lanes = (
            (lambda h: slice(h * hd, (h + 1) * hd))
            if c.packed_attn
            else (lambda h: slice(0, hd))
        )
        if "attn_stage" in S and res.shape[1] == LANE:
            # rows (t, gi) of each head: one strided [q, hd] load and store per
            # (head, group) instead of q row-by-row concatenations
            stage = S["attn_stage"]
            stage[...] = res
            for h in range(hkv):
                for gi in range(g):
                    S["attn_buf"][
                        pl.ds(pl.multiple_of(r0, 8), q),
                        pl.ds((h * g + gi) * hd, hd),
                    ] = stage[pl.ds(h * qg + gi, q, stride=g), :][:, lanes(h)]
        else:
            for t in range(q):
                S["attn_buf"][pl.ds(r0 + t, 1), :] = jnp.concatenate(
                    [
                        res[
                            h * qg + t * g + gi : h * qg + t * g + gi + 1,
                            lanes(h),
                        ]
                        for h in range(hkv)
                        for gi in range(g)
                    ],
                    axis=1,
                )
