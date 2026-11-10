# SPDX-License-Identifier: Apache-2.0
"""The PP prefill stage's fused Pallas kernels around attention and the experts: attn_in (qkvg
projection, qk norms, rope), route (gated o projection and residual, router top-k, expert
projection) and combine (post-expert norm per slot, weighted slot sum, unpooler, residual).
They read one slot's dense weights from the stage's stacks; nothing in them talks to another
device."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.kernel import _rms, _topk_iter
from sglang_tpu.muse_megakernel.prefill_pp.config import (
    BF16,
    F32,
    F_GLOBAL,
    F_VALID,
    FP8,
    H_STAGE,
    HDR,
    LANE,
    NF,
    V_ROWS,
    WEIGHT_ORDER,
    PPConfig,
    V,
    skipped_phases,
)


def _bf(x):
    return x.astype(BF16).astype(F32)


def _mm(a_ref, w_ref, k: int, kc: int = 512):
    """a [M, k] bf16 @ w [k, N] (bf16 or fp8 refs), f32, contracted in kc slices."""
    acc = None
    for k0 in range(0, k, kc):
        part = jnp.dot(
            a_ref[:, pl.ds(k0, kc)], w_ref[pl.ds(k0, kc), :], preferred_element_type=F32
        )
        acc = part if acc is None else acc + part
    return acc


def _head_mats(hmat, hd: int):
    """hmat[0]: block-diagonal 1/hd (per-head mean); hmat[1]: signed pair swap (rotary partner)."""
    shape = hmat.shape[1:]
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    hmat[0] = jnp.where(row // hd == col // hd, 1.0 / hd, 0.0).astype(BF16)
    even = (col % 2) == 0
    swap = jnp.where(even & (row == col + 1), -1.0, 0.0)
    hmat[1] = jnp.where(~even & (row == col - 1), 1.0, swap).astype(BF16)


def _heads_rms_mxu(x, hmat, eps: float, gain: float = 1.0):
    """Per-head RMS norm of [M, n * hd]; the head means come off the MXU, not lane reduces."""
    ms = jnp.dot((x * x).astype(BF16), hmat[0], preferred_element_type=F32)
    return x * (jax.lax.rsqrt(ms + eps) * gain)


def _rope_mxu(x, hmat, cos, sin):
    """Interleaved rotary on [M, n * hd]; the signed pair partner is an MXU product."""
    return x * cos + jnp.dot(x.astype(BF16), hmat[1], preferred_element_type=F32) * sin


SCRATCH = (
    "xbuf", "accbuf", "abuf", "gbuf", "pbuf", "wbuf", "vec_v", "wr_v", "rope_v", "rope8",
    "otile", "ids_v", "tw_v", "ones_v", "hmat", "misc_s", "sem_x", "sem_w", "sem_o", "sem_m",
)  # fmt: skip


def stage_calls(c: PPConfig):
    """The stage's fused calls, for the slot in ctl[2] (ctl [3] i32 = [cached tokens, active
    stage or -1, slot]); a device whose stage is not active, or a slot without a layer, returns
    at once (outputs unwritten):
      attn_in(ctl, slot table, x [T, H] f32, rope [max_position, 2 hd] f32, *weights) ->
        (q [T, q_dim] bf16, k, v [T, kv_dim] bf16, gate [T, q_dim] bf16)
      route(ctl, slot table, x, o [T, q_dim] bf16, gate, *weights) ->
        (x, expert input [T, He] bf16, ids [T, 128] i32, weights [T, 128] f32), lanes past
        top_k zero, x updated in place
      combine(ctl, slot table, x, y [T * top_k, He] bf16 (token-major slots), routing weights
        [T, 128] f32 (lanes past top_k zero), *weights) -> (x,), x updated in place"""
    c.validate()
    skip = skipped_phases()
    T, H, He, E, K = c.rows, c.hidden, c.expert_hidden, c.n_experts, c.top_k
    hd = c.head_dim
    qd, kvd, nq = c.q_dim, c.kv_dim, c.qkvg_dim
    tm, tn = c.tm, c.tn
    n_rt = T // tm
    sub = 16  # rows per vector pass over a [tm, H] tile
    n_qt, n_kt = qd // tn, kvd // tn
    ntc = nq // tn
    G = tm // K  # tokens per combine group: their K slot rows fill a tile
    LC = 512  # lanes per combine chunk
    wr_rows = max(H, qd, He)
    WU_DEPTH = 4  # unpool weight buffers: wbuf's two tiles hold two He-row tiles each
    assert wr_rows >= 2 * He

    def body(R, phases):
        ctl, slots, x_in = R["ctl"], R["slots"], R["x_in"]
        wqkvg, wo, wpool, wunpool, wrouter, vecs = (R[n] for n in WEIGHT_ORDER)
        xbuf, accbuf, abuf, gbuf, pbuf, wbuf = (R[n] for n in SCRATCH[:6])
        vec_v, wr_v, rope_v, rope8, otile, ids_v, tw_v = (R[n] for n in SCRATCH[6:13])
        misc_s, sem_x, sem_w, sem_o, sem_m = (R[n] for n in SCRATCH[15:])
        x_out = x_in  # the stream is aliased in place

        def vrow(name, n, off=0):
            return vec_v[pl.ds(V[name], 1), pl.ds(off, n)]

        def rows(i, n):
            return pl.ds(pl.multiple_of(i * n, n), n)

        def attn_in(sl, is_g, ctx):
            rope, q_h, k_h, v_h, g_h, hmat = (
                R[n] for n in ("rope", "q_h", "k_h", "v_h", "g_h", "hmat")
            )
            _head_mats(hmat, hd)
            total = n_rt * ntc

            def wcp(idx, b):
                return pltpu.make_async_copy(
                    wqkvg.at[sl, idx % ntc], wbuf.at[b, pl.ds(0, H)], sem_w.at[b]
                )

            def out_cp(b, dst):
                return pltpu.make_async_copy(otile.at[b], dst, sem_o.at[b])

            def emit(acc, dst):
                """A [tm, tn] output tile through the double-buffered otile."""
                n = misc_s[0]
                b = n % 2

                @pl.when(n >= 2)
                def _():
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

                otile[b] = acc.astype(BF16)
                out_cp(b, dst).start()
                misc_s[0] = n + 1

            misc_s[0] = 0
            wcp(0, 0).start()

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_in.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                cr = pltpu.make_async_copy(
                    rope.at[pl.ds(ctx + r0, tm)], rope_v, sem_x.at[1]
                )
                cx.start()
                cr.start()
                cx.wait()
                cr.wait()
                rp = rope_v[...]
                rope8[0] = jnp.concatenate([rp[:, :hd]] * (tn // hd), axis=1)
                rope8[1] = jnp.concatenate([rp[:, hd:]] * (tn // hd), axis=1)

                def norm(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    abuf[rr, :] = (
                        _rms(xbuf[rr, :], c.rms_eps) * vrow("pre_attn_w", H)
                    ).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, norm, 0)

                def tile(jt):
                    idx = i * ntc + jt
                    b = idx % 2
                    wcp(idx, b).wait()

                    @pl.when(idx + 1 < total)
                    def _():
                        wcp(idx + 1, 1 - b).start()

                    return _mm(abuf, wbuf.at[b], H) * vrow(
                        "wqkvg_s", tn, pl.multiple_of(jt * tn, tn)
                    )

                def roped(x):
                    return jnp.where(
                        is_g != 0, x, _rope_mxu(x, hmat, rope8[0], rope8[1])
                    )

                def q_tile(jt, cc):
                    q = _heads_rms_mxu(tile(jt), hmat, c.rms_eps, gain=c.scale_query_by)
                    emit(roped(q), q_h.at[pl.ds(r0, tm), rows(jt, tn)])
                    return cc

                jax.lax.fori_loop(0, n_qt, q_tile, 0)
                for kt in range(n_kt):
                    k = roped(_heads_rms_mxu(tile(n_qt + kt), hmat, c.rms_eps))
                    emit(k, k_h.at[pl.ds(r0, tm), rows(kt, tn)])
                for kt in range(n_kt):
                    emit(tile(n_qt + n_kt + kt), v_h.at[pl.ds(r0, tm), rows(kt, tn)])

                def g_tile(jt, cc):
                    emit(
                        tile(jt), g_h.at[pl.ds(r0, tm), rows(jt - n_qt - 2 * n_kt, tn)]
                    )
                    return cc

                jax.lax.fori_loop(n_qt + 2 * n_kt, ntc, g_tile, 0)
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)
            for b in range(2):

                @pl.when(misc_s[0] > b)
                def _(b=b):
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

        def route(sl):
            o_h, g_in, xe_h, ids_h, tw_h = (
                R[n] for n in ("o_h", "g_in", "xe_h", "ids_h", "tw_h")
            )
            n_wo, n_wp = H // tn, He // tn
            per = n_wo + n_wp
            total = n_rt * per

            def wcp(idx, b, wait):
                j = idx % per

                @pl.when(j < n_wo)
                def _():
                    cp = pltpu.make_async_copy(
                        wo.at[sl, j], wbuf.at[b, pl.ds(0, qd)], sem_w.at[b]
                    )
                    cp.wait() if wait else cp.start()

                @pl.when(j >= n_wo)
                def _():
                    cp = pltpu.make_async_copy(
                        wpool.at[sl, j - n_wo], wbuf.at[b, pl.ds(0, H)], sem_w.at[b]
                    )
                    cp.wait() if wait else cp.start()

            def take(idx):
                b = idx % 2
                wcp(idx, b, True)

                @pl.when(idx + 1 < total)
                def _():
                    wcp(idx + 1, 1 - b, False)

                return b

            cwr = pltpu.make_async_copy(wrouter.at[sl], wr_v, sem_m.at[1])
            cwr.start()
            wcp(0, 0, False)

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_out.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                ca = pltpu.make_async_copy(o_h.at[pl.ds(r0, tm)], abuf, sem_x.at[1])
                cg = pltpu.make_async_copy(g_in.at[pl.ds(r0, tm)], gbuf, sem_x.at[2])
                for cp in (cx, ca, cg):
                    cp.start()
                for cp in (cx, ca, cg):
                    cp.wait()

                def gated(r, cc):
                    # the attention kernel stores each head's output RMS-normalized
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    g = jax.nn.sigmoid(gbuf[rr, :].astype(F32))
                    abuf[rr, :] = (g * abuf[rr, :].astype(F32)).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, gated, 0)

                def wo_tile(j, cc):
                    b = take(i * per + j)
                    accbuf[:, rows(j, tn)] = _mm(abuf, wbuf.at[b], qd)
                    return cc

                jax.lax.fori_loop(0, n_wo, wo_tile, 0)

                def resid(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    ao = _bf(accbuf[rr, :] * vrow("wo_s", H))
                    x = vrow("attn_alpha", H) * xbuf[rr, :] + vrow(
                        "attn_beta", H
                    ) * _bf(_rms(ao, c.post_eps))
                    xbuf[rr, :] = x
                    abuf[rr, :] = (_rms(x, c.rms_eps) * vrow("pre_ffn_w", H)).astype(
                        BF16
                    )
                    return cc

                jax.lax.fori_loop(0, tm // sub, resid, 0)
                cxo = pltpu.make_async_copy(xbuf, x_out.at[pl.ds(r0, tm)], sem_x.at[0])
                cxo.start()

                @pl.when(i == 0)
                def _():
                    cwr.wait()

                logits = _mm(abuf, wr_v, H)
                sc = jax.nn.sigmoid(logits * c.router_mult)
                ids, picks = _topk_iter(sc + vrow("expert_bias", E), K)
                tws = [
                    jnp.sum(jnp.where(pk, sc, 0.0), axis=1, keepdims=True)
                    for pk in picks
                ]
                denom = sum(tws) + c.router_eps
                lane = jax.lax.broadcasted_iota(jnp.int32, (tm, LANE), 1)
                ids_t = jnp.zeros((tm, LANE), jnp.int32)
                tw_t = jnp.zeros((tm, LANE), F32)
                for k in range(K):
                    ids_t = jnp.where(lane == k, ids[k], ids_t)
                    tw_t = jnp.where(lane == k, tws[k] / denom * c.route_scale, tw_t)
                ids_v[...] = ids_t
                tw_v[...] = tw_t
                cid = pltpu.make_async_copy(ids_v, ids_h.at[pl.ds(r0, tm)], sem_m.at[2])
                ctw = pltpu.make_async_copy(tw_v, tw_h.at[pl.ds(r0, tm)], sem_m.at[3])
                cid.start()
                ctw.start()

                def wp_tile(j, cc):
                    b = take(i * per + n_wo + j)
                    pbuf[:, rows(j, tn)] = _bf(
                        _mm(abuf, wbuf.at[b], H)
                        * vrow("wpool_s", tn, pl.multiple_of(j * tn, tn))
                    )
                    return cc

                jax.lax.fori_loop(0, n_wp, wp_tile, 0)

                def xe_norm(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    gbuf[rr, pl.ds(0, He)] = (
                        _rms(pbuf[rr, :], c.rms_eps) * vrow("pre_expert_w", He)
                    ).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, xe_norm, 0)
                cxe = pltpu.make_async_copy(
                    gbuf.at[:, pl.ds(0, He)], xe_h.at[pl.ds(r0, tm)], sem_x.at[1]
                )
                cxe.start()
                for cp in (cxo, cid, ctw, cxe):
                    cp.wait()
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)

        def combine(sl):
            y_h, tw_in, ones_v = R["y_h"], R["tw_in"], R["ones_v"]
            pew = vrow("post_expert_w", He)
            ones_v[...] = jnp.transpose(jnp.broadcast_to(pew * pew, (LANE, He))).astype(
                BF16
            )
            n_wu = H // tn
            total = n_rt * n_wu
            n_grp = tm // G

            def wcp(idx):
                """Unpool weight tiles rotate through wbuf's four He-row quarters."""
                q = idx % WU_DEPTH
                return pltpu.make_async_copy(
                    wunpool.at[sl, idx % n_wu],
                    wbuf.at[q // 2, pl.ds((q % 2) * He, He)],
                    sem_w.at[q],
                )

            def ycp(i, gi, yb):
                """A group's K slot rows per token (token-major) into a staging half."""
                return pltpu.make_async_copy(
                    y_h.at[pl.ds((i * tm + gi * G) * K, tm)],
                    gbuf.at[:, pl.ds(yb * He, He)],
                    sem_m.at[4 + yb],
                )

            def xo_cp(i):
                return pltpu.make_async_copy(
                    xbuf, x_out.at[pl.ds(pl.multiple_of(i * tm, tm), tm)], sem_x.at[2]
                )

            for d in range(min(WU_DEPTH - 1, total)):
                wcp(d).start()
            ycp(0, 0, 0).start()

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_out.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                ct = pltpu.make_async_copy(tw_in.at[pl.ds(r0, tm)], tw_v, sem_x.at[1])
                ct.start()
                ct.wait()

                def group(gi, cc):
                    yb = gi % 2
                    ycp(i, gi, yb).wait()

                    @pl.when(gi + 1 < n_grp)
                    def _():
                        ycp(i, gi + 1, 1 - yb).start()

                    stage = gbuf.at[:, pl.ds(yb * He, He)]

                    # sum(k) w_k s_k (y_k * pew) = pew * (Wsel @ y), Wsel[t, Kt + k] = w_k s_k:
                    # the squares' pew^2 sums and the slot sum run on the MXU
                    tw = jnp.concatenate(
                        [
                            tw_v[pl.ds(pl.multiple_of(gi * G, G), G), :],
                            jnp.zeros((G, LANE), F32),
                        ],
                        axis=1,
                    )
                    ss = jnp.zeros((tm, LANE), F32)
                    for l0 in range(0, He, LC):
                        yc = stage[:, pl.ds(l0, LC)]
                        ss = ss + jnp.dot(
                            yc * yc,
                            ones_v[pl.ds(l0, LC), :],
                            preferred_element_type=F32,
                        )
                    scale = jax.lax.rsqrt(
                        jnp.transpose(ss)[:1] * (1.0 / He) + c.post_eps
                    )
                    wsel = pltpu.roll(tw, 0, 1, stride=K, stride_axis=0)
                    wsel = (wsel * scale).astype(BF16)
                    for l0 in range(0, He, LC):
                        accbuf[pl.ds(pl.multiple_of(gi * G, G), G), pl.ds(l0, LC)] = (
                            jnp.dot(
                                wsel,
                                stage[:, pl.ds(l0, LC)],
                                preferred_element_type=F32,
                            )
                            * pew[:, l0 : l0 + LC]
                        )
                    return cc

                jax.lax.fori_loop(0, n_grp, group, 0)

                # the next tile's first y group and this tile's x stream overlap the unpool
                @pl.when(i + 1 < n_rt)
                def _():
                    ycp(i + 1, 0, 0).start()

                @pl.when(i > 0)
                def _():
                    xo_cp(i - 1).wait()

                cx.start()

                def to_bf16(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    abuf[rr, pl.ds(0, He)] = accbuf[rr, pl.ds(0, He)].astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, to_bf16, 0)

                def wu_tile(j, cc):
                    idx = i * n_wu + j
                    q = idx % WU_DEPTH
                    wcp(idx).wait()

                    @pl.when(idx + WU_DEPTH - 1 < total)
                    def _():
                        wcp(idx + WU_DEPTH - 1).start()

                    f = _mm(
                        abuf.at[:, pl.ds(0, He)],
                        wbuf.at[q // 2, pl.ds((q % 2) * He, He)],
                        He,
                    )
                    accbuf[:, rows(j, tn)] = _bf(
                        f * vrow("wunpool_s", tn, pl.multiple_of(j * tn, tn))
                    )
                    return cc

                jax.lax.fori_loop(0, n_wu, wu_tile, 0)
                cx.wait()

                def resid(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    f = accbuf[rr, :] * vrow("post_ffn_w", H)
                    xbuf[rr, :] = vrow("ffn_alpha", H) * xbuf[rr, :] + vrow(
                        "ffn_beta", H
                    ) * _bf(_rms(f, c.post_eps))
                    return cc

                jax.lax.fori_loop(0, tm // sub, resid, 0)
                xo_cp(i).start()
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)
            xo_cp(n_rt - 1).wait()

        sl = ctl[2]
        base = HDR + sl * NF
        active = (ctl[1] < 0) | (ctl[1] == slots[H_STAGE])

        @pl.when(active & (slots[base + F_VALID] != 0))
        def _():
            is_g = slots[base + F_GLOBAL]
            cv = pltpu.make_async_copy(vecs.at[sl], vec_v, sem_m.at[0])
            cv.start()
            cv.wait()
            for name, run in (
                ("attn_in", lambda: attn_in(sl, is_g, ctl[0])),
                ("route", lambda: route(sl)),
                ("combine", lambda: combine(sl)),
            ):
                if name in phases and name not in skip:
                    run()

    vmem, smem_t = pltpu.VMEM, pltpu.SMEM
    dma = pltpu.SemaphoreType.DMA
    scratch = [
        vmem((tm, H), F32),  # xbuf: stream tile
        vmem((tm, H), F32),  # accbuf: projection outputs; the combined slots
        vmem((tm, H), BF16),  # abuf: matmul inputs
        vmem((tm, qd), BF16),  # gbuf: attention gate; expert input; slot staging halves
        vmem((tm, He), F32),  # pbuf: pooled rows
        vmem((2, wr_rows, tn), FP8),  # wbuf: weight tiles
        vmem((V_ROWS, c.vec_width), F32),
        vmem((H, E), BF16),
        vmem((tm, 2 * hd), F32),
        vmem((2, tm, tn), F32),  # rope8: lane-expanded cos / sin of a row tile
        vmem((2, tm, tn), BF16),  # otile
        vmem((tm, LANE), jnp.int32),  # ids_v
        vmem((tm, LANE), F32),  # tw_v
        vmem(
            (He, LANE), BF16
        ),  # ones_v: pew^2 down each column, sums the squares on the MXU
        vmem((2, tn, tn), BF16),  # hmat: per-head mean and rotary partner matrices
        smem_t((8,), jnp.int32),  # misc_s
        dma((3,)),
        dma((WU_DEPTH,)),
        dma((2,)),
        dma((8,)),
    ]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    sds = jax.ShapeDtypeStruct
    params = pltpu.CompilerParams(vmem_limit_bytes=64 * 2**20 - 2**19)
    tag = f"t{T}" + "".join(f"_no_{p}" for p in sorted(skip))

    def build(in_names, outs, phases, aliases, name):
        names = in_names + list(outs) + list(SCRATCH)

        def kernel(*refs):
            body(dict(zip(names, refs)), phases)

        return pl.pallas_call(
            kernel,
            out_shape=tuple(outs.values()),
            in_specs=[smem] * 2 + [hbm] * (len(in_names) - 2),
            out_specs=(hbm,) * len(outs),
            scratch_shapes=scratch,
            input_output_aliases=aliases,
            compiler_params=params,
            name=name,
        )

    attn_in_call = build(
        ["ctl", "slots", "x_in", "rope", *WEIGHT_ORDER],
        {
            "q_h": sds((T, qd), BF16),
            "k_h": sds((T, kvd), BF16),
            "v_h": sds((T, kvd), BF16),
            "g_h": sds((T, qd), BF16),
        },
        ("attn_in",),
        {},
        f"muse_prefill_pp_attn_in_{tag}",
    )
    route_call = build(
        ["ctl", "slots", "x_in", "o_h", "g_in", *WEIGHT_ORDER],
        {
            "x_out": sds((T, H), F32),
            "xe_h": sds((T, He), BF16),
            "ids_h": sds((T, LANE), jnp.int32),
            "tw_h": sds((T, LANE), F32),
        },
        ("route",),
        {2: 0},
        f"muse_prefill_pp_route_{tag}",
    )
    combine_call = build(
        ["ctl", "slots", "x_in", "y_h", "tw_in", *WEIGHT_ORDER],
        {"x_out": sds((T, H), F32)},
        ("combine",),
        {2: 0},
        f"muse_prefill_pp_combine_{tag}",
    )
    return attn_in_call, route_call, combine_call
