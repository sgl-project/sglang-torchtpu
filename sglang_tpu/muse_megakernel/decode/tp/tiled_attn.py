# SPDX-License-Identifier: Apache-2.0
"""Row-tiled routing and attention of the decode megakernel (MKConfig.row_tile)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.common.consts import (
    LANE,
    SMALL,
)
from sglang_tpu.muse_megakernel.common.hooks import (
    EPOProjStep,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _heads_rms,
    _rope_roll,
)
from sglang_tpu.muse_megakernel.common.packing import (
    _pack_bf16x2,
    _unpack_bf16x2,
)
from sglang_tpu.muse_megakernel.common.plan import (
    _fold_g,
)
from sglang_tpu.muse_megakernel.common.scratch import (
    _TS_LOCAL,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx


def route_tiled(
    k: KernelCtx,
):
    """_route_table from ids_v row tile by row tile: per-expert counts and keys, then

    each pair's slot in perm from a running per-expert count; temporaries stay
    [RT * top_k, n_experts] instead of [R * top_k, R * top_k].
    """
    K = k.K
    NBR = k.NBR
    NT = k.NT
    R = k.R
    RT = k.RT
    S = k.S
    bf16 = k.bf16
    c = k.c
    coll = k.coll
    f32 = k.f32
    me = k.me
    n_live = k.n_live
    rows_of = k.rows_of
    stream_pair_y = k.stream_pair_y
    use_pair_y = k.use_pair_y
    y_block = k.y_block
    Ep, NB, Rc = c.n_experts_padded, NBR, c.rows_per_dev
    big = float(c.tp + 1)
    i32 = jnp.int32

    def tile_ids(t):
        ids = S["ids_v"][rows_of(t), pl.ds(0, K)]  # [RT, K]
        r = jax.lax.broadcasted_iota(i32, (RT, 1), 0) + t * RT
        return ids, r

    def key_of(r):
        if stream_pair_y:
            return (3 - jnp.bitwise_xor(r // (2 * Rc), me // 2)).astype(f32)
        if not y_block or use_pair_y:
            return jnp.zeros(r.shape, f32)
        o = r // Rc
        if coll.order == "xor":
            return ((c.tp - 1) - jnp.bitwise_xor(o, me)).astype(f32)
        return jax.lax.rem(o - me - 1 + c.tp, c.tp).astype(f32)

    R_ct = (
        64
        if (c.topk_xpose and R % 64 == 0)
        else (32 if (c.topk_xpose and R % 32 == 0) else RT)
    )

    def count_tile(t, carry):
        count, key = carry
        ids = S["ids_v"][pl.ds(pl.multiple_of(t * R_ct, R_ct), R_ct), pl.ds(0, K)]
        r = jax.lax.broadcasted_iota(i32, (R_ct, 1), 0) + t * R_ct
        live, kr = r < n_live, key_of(r)
        e_lane = jax.lax.broadcasted_iota(i32, (R_ct, Ep), 1)
        for k in range(K):
            hit = (ids[:, k : k + 1] == e_lane) & live
            count = count + jnp.sum(jnp.where(hit, 1.0, 0.0), axis=0, keepdims=True)
            key = jnp.minimum(
                key, jnp.min(jnp.where(hit, kr, big), axis=0, keepdims=True)
            )
        return count, key

    with _scope("route_count"):
        count, key = jax.lax.fori_loop(
            0,
            R // R_ct,
            count_tile,
            (jnp.zeros((1, Ep), f32), jnp.full((1, Ep), big, f32)),
        )
    present = count > 0
    split = c.split_rows
    mult = 2 if split else 1
    if split:
        key = 2 * key + jnp.where(count > split, 1.0, 0.0)
    e_row = jax.lax.broadcasted_iota(i32, (1, Ep), 1).astype(f32)
    cols = jnp.transpose(
        jnp.concatenate(
            [
                key,
                e_row,
                count,
                jnp.where(present, 1.0, 0.0),
                jnp.zeros((LANE - 4, Ep), f32),
            ],
            axis=0,
        )
    )  # [Ep, LANE]
    key_c, id_c, cnt_c, pres_c = (
        cols[:, 0:1],
        cols[:, 1:2],
        cols[:, 2:3],
        cols[:, 3:4],
    )
    earlier = present & ((key < key_c) | ((key == key_c) & (e_row < id_c)))
    pos_c = jnp.sum(jnp.where(earlier, 1.0, 0.0), axis=1, keepdims=True).astype(i32)
    off_c = jnp.sum(jnp.where(earlier, count, 0.0), axis=1, keepdims=True)
    off_e = jnp.transpose(
        jnp.concatenate([off_c, jnp.zeros((Ep, LANE - 1), f32)], axis=1)
    )[0:1]
    rv = S["route_v"]
    rv[...] = jnp.zeros(rv.shape, i32)
    for b in range(NB):
        u_lane = jax.lax.broadcasted_iota(i32, (Ep, LANE), 1) + b * LANE
        m = (pres_c > 0) & (pos_c == u_lane)
        for f, v in enumerate((id_c, cnt_c, off_c)):
            rv[f * NB + b : f * NB + b + 1, :] = jnp.sum(
                jnp.where(m, v, 0.0), axis=0, keepdims=True
            ).astype(i32)
    n_u = jnp.sum(jnp.where(present, 1, 0), axis=1, keepdims=True)
    rv[4 * NB : 4 * NB + 1, :] = jnp.broadcast_to(n_u, (1, LANE))
    b_lane = jax.lax.broadcasted_iota(i32, (Ep, LANE), 1).astype(f32)
    rv[4 * NB + 1 : 4 * NB + 2, :] = jnp.sum(
        jnp.where((pres_c > 0) & (key_c <= mult * b_lane + mult - 1), 1, 0),
        axis=0,
        keepdims=True,
    )
    if split:
        rv[4 * NB + 2 : 4 * NB + 3, :] = jnp.sum(
            jnp.where((pres_c > 0) & (key_c <= 2 * b_lane), 1, 0),
            axis=0,
            keepdims=True,
        )
    PT = RT * K

    def perm_tile(t, running):
        ids, _ = tile_ids(t)
        col_e = jnp.concatenate([ids[:, k : k + 1] for k in range(K)], axis=0)
        q_ = jax.lax.broadcasted_iota(i32, (PT, 1), 0)
        r_of = q_ % RT + t * RT
        enc = r_of * K + q_ // RT
        live = r_of < n_live
        hit = (col_e == jax.lax.broadcasted_iota(i32, (PT, Ep), 1)) & live
        onehot = jnp.where(hit, 1.0, 0.0)
        same = jax.lax.dot_general(
            onehot.astype(bf16),
            onehot.astype(bf16),
            (((1,), (1,)), ((), ())),
            preferred_element_type=f32,
        )  # [PT, PT]
        q_row = jax.lax.broadcasted_iota(i32, (PT, PT), 0)
        q_col = jax.lax.broadcasted_iota(i32, (PT, PT), 1)
        live_col = (q_col % RT + t * RT) < n_live
        rank = jnp.sum(
            jnp.where((q_col < q_row) & live_col, same, 0.0),
            axis=1,
            keepdims=True,
        )
        dest = (
            jnp.sum(onehot * (off_e + running), axis=1, keepdims=True) + rank
        ).astype(i32)
        for b in range(NB):
            t_lane = jax.lax.broadcasted_iota(i32, (PT, LANE), 1) + b * LANE
            rv[3 * NB + b : 3 * NB + b + 1, :] = rv[
                3 * NB + b : 3 * NB + b + 1, :
            ] + jnp.sum(
                jnp.where((dest == t_lane) & live, enc, 0),
                axis=0,
                keepdims=True,
            )
        return running + jnp.sum(onehot, axis=0, keepdims=True)

    with _scope("route_perm"):
        jax.lax.fori_loop(0, NT, perm_tile, jnp.zeros((1, Ep), f32))
    with _scope("route_dma"):
        cp = pltpu.make_async_copy(rv, S["route_s"], S["route_sem"].at[0])
        cp.start()
        cp.wait()
    return S["route_s"][4 * NB, 0]


def attention_tiled(k: KernelCtx, l, use_rope, is_global, nxt, smr, pend=False):
    """Leaves the all-reduced o-proj output all-gathered in pad_o_ag (or in flight)."""
    H = k.H
    Hl = k.Hl
    R = k.R
    RT = k.RT
    S = k.S
    ag_tile_start = k.ag_tile_start
    bf16 = k.bf16
    c = k.c
    capture_tile = k.capture_tile
    coll = k.coll
    ep = k.ep
    f32 = k.f32
    f_blocks = k.f_blocks
    for_tiles = k.for_tiles
    gathered_tile = k.gathered_tile
    hag_start = k.hag_start
    hd = k.hd
    hkv = k.hkv
    hq = k.hq
    kv_prologue = k.kv_prologue
    me = k.me
    mv_tiled = k.mv_tiled
    n_live = k.n_live
    newkv_out = k.newkv_out
    pad_view = k.pad_view
    pd = k.pd
    per_geom = k.per_geom
    q = k.q
    rcopy = k.rcopy
    request_attention_body = k.request_attention_body
    tile_pass = k.tile_pass
    tiled_fold_lhs = k.tiled_fold_lhs
    tsem = k.tsem
    use_pair_o_rs = k.use_pair_o_rs
    vpads_tiles = k.vpads_tiles
    ws = k.ws
    per_geom(is_global, lambda geo: kv_prologue(geo, l, 0))
    qd, kd = c.q_dim, c.kv_dim
    if pend:
        psmr = S["smalls"].at[(l + 1) % 2]
        g_pa = smr[SMALL["pre_attn_w"] : SMALL["pre_attn_w"] + 1, :H]
        p_alpha = psmr[SMALL["ffn_alpha"] : SMALL["ffn_alpha"] + 1, :H]
        p_beta = psmr[SMALL["ffn_beta"] : SMALL["ffn_beta"] + 1, :H]
        p_pfw = psmr[SMALL["post_ffn_w"] : SMALL["post_ffn_w"] + 1, :H]

        def _fold_lhs(rs, t):
            S["lhs"][rs] = (p_alpha * S["h_stream"][rs] * g_pa).astype(bf16)

        vpads_tiles(_fold_lhs)
        S["fold_v"][0:1, :] = p_beta * p_pfw * g_pa
        wait_block, finish = f_blocks()
        with _scope("qkvg_fold"):
            acc_q = ws.mv_fold(
                l,
                ("wqkvg",),
                nxt,
                _fold_g(c, ("wqkvg",)),
                Hl,
                wait_block,
                tiled_fold_lhs("pad_f"),
            )["wqkvg"]
        finish()
        wq_sc = smr[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
        if c.tile_loops:
            # per-row scales from one loop; the partial sums combine as values after it
            def _fold_q(rs, t):
                y = gathered_tile(S["pad_f"].at[:, rs]) * p_pfw
                s_f = jax.lax.rsqrt(
                    jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
                )
                s = p_alpha * S["h_stream"][rs] + p_beta * (y * s_f).astype(
                    bf16
                ).astype(f32)
                capture_tile(l - 1, rs, s)
                S["h_stream"][rs] = s
                r_inv = jax.lax.rsqrt(
                    jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                )
                S["fold_stage"][rs, pl.ds(0, LANE)] = jnp.broadcast_to(s_f, (RT, LANE))
                S["fold_stage"][rs, pl.ds(LANE, LANE)] = jnp.broadcast_to(
                    r_inv, (RT, LANE)
                )

            for_tiles(_fold_q)
            s_f_all = S["fold_stage"][:, pl.ds(0, 1)]
            r_inv_all = S["fold_stage"][:, pl.ds(LANE, 1)]
            S["acc"][:, pl.ds(0, c.qkvg_dim)] = (
                r_inv_all * (acc_q[:R] + s_f_all * acc_q[R:]) * wq_sc
            )
        else:
            for t in range(R // RT):
                r0, r1 = t * RT, (t + 1) * RT
                rs = pl.ds(r0, RT)
                y = gathered_tile(S["pad_f"].at[:, rs]) * p_pfw
                s_f = jax.lax.rsqrt(
                    jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
                )
                s = p_alpha * S["h_stream"][rs] + p_beta * (y * s_f).astype(
                    bf16
                ).astype(f32)
                capture_tile(l - 1, rs, s)
                S["h_stream"][rs] = s
                r_inv = jax.lax.rsqrt(
                    jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                )
                S["acc"][rs, pl.ds(0, c.qkvg_dim)] = (
                    r_inv * (acc_q[r0:r1] + s_f * acc_q[R + r0 : R + r1]) * wq_sc
                )
    elif c.fold_f:
        with _scope("qkvg_mv"):
            acc_q = ws.mv_fold(
                l,
                ("wqkvg",),
                nxt,
                _fold_g(c, ("wqkvg",)),
                Hl,
                lambda k: None,
                tiled_fold_lhs(None),
            )["wqkvg"]
        wq_sc = smr[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
        S["acc"][:, pl.ds(0, c.qkvg_dim)] = acc_q * wq_sc
    else:
        with _scope("qkvg_mv"):
            mv_tiled(l, "wqkvg", nxt, smr)

    def qkv_tile(rows, t):
        qkvg = S["acc"][rows, pl.ds(0, c.qkvg_dim)]
        qv, kv_, vv = (
            qkvg[:, :qd],
            qkvg[:, qd : qd + kd],
            qkvg[:, qd + kd : qd + 2 * kd],
        )
        rt_ = S["rtab"][rows, :]
        cos_t, sin_t = rt_[:, :hd], rt_[:, hd:]
        qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
        kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
        qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_t, sin_t), qn)
        kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_t, sin_t), kn)
        kb = kn.astype(c.kv_dtype).astype(bf16)
        vb = vv.astype(c.kv_dtype).astype(bf16)
        S["qn_s"][rows] = qn
        S["kb_s"][rows] = kb.astype(f32)
        S["vb_s"][rows] = vb.astype(f32)
        S["attn_buf"][rows] = jnp.zeros((RT, qd), f32)
        if c.kv_sal:
            return
        r0 = t * RT
        for r in range(RT):
            if c.kv_packed:
                S["kv_row"][r0 + r] = jnp.concatenate(
                    [kb[r : r + 1], vb[r : r + 1]], axis=0
                ).reshape(1, 2, pd)
                continue
            zp = jnp.zeros((1, pd - hd), bf16)
            parts = []
            for h in range(hkv):
                for x in (kb, vb):
                    parts.append(
                        jnp.concatenate(
                            [x[r : r + 1, h * hd : (h + 1) * hd], zp], axis=1
                        )
                    )
            S["kv_row"][r0 + r] = jnp.concatenate(parts, axis=0).reshape(hkv, 2, pd)

    if c.ex_opt and c.kv_sal and R <= c.untiled_max_rows:
        qkvg = S["acc"][:, pl.ds(0, c.qkvg_dim)]
        qv, kv_, vv = (
            qkvg[:, :qd],
            qkvg[:, qd : qd + kd],
            qkvg[:, qd + kd : qd + 2 * kd],
        )
        rt_ = S["rtab"][...]
        cos_t, sin_t = rt_[:, :hd], rt_[:, hd:]
        qn = _heads_rms(qv, hq, hd, c.rms_eps, gain=c.scale_query_by)
        kn = _heads_rms(kv_, hkv, hd, c.rms_eps)
        qn = jnp.where(use_rope, _rope_roll(qn, hq, hd, cos_t, sin_t), qn)
        kn = jnp.where(use_rope, _rope_roll(kn, hkv, hd, cos_t, sin_t), kn)
        kb = kn.astype(c.kv_dtype).astype(bf16)
        vb = vv.astype(c.kv_dtype).astype(bf16)
        S["qn_s"][...] = qn
        S["kb_s"][...] = kb.astype(f32)
        S["vb_s"][...] = vb.astype(f32)
        S["attn_buf"][...] = jnp.zeros((R, qd), f32)
    else:
        for_tiles(qkv_tile)
    if c.kv_sal:  # [slot * words + word, row]: an identity matmul transposes exactly
        if not (c.ex_opt and c.kv_sal and R <= c.untiled_max_rows):
            kb = S["kb_s"][...].astype(bf16)
            vb = S["vb_s"][...].astype(bf16)
        eye = (
            jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 0)
            == jax.lax.broadcasted_iota(jnp.int32, (hd, hd), 1)
        ).astype(bf16)
        S["kv_row"][...] = jnp.concatenate(
            [
                pltpu.bitcast(
                    jax.lax.dot_general(
                        eye,
                        x[:, h * hd : (h + 1) * hd],
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=f32,
                    ).astype(c.kv_dtype),
                    jnp.uint32,
                )
                for h in range(hkv)
                for x in (kb, vb)
            ],
            axis=0,
        )
    kv_out_cp = pltpu.make_async_copy(
        S["kv_row"], newkv_out.at[l], S["kv_row_sem"].at[0]
    )
    kv_out_cp.start()
    with _scope("attn_core"):
        n_req_live = jnp.minimum((n_live + q - 1) // q, c.n_req)

        def run_geom(geo):
            def req_loop(j, carry):
                request_attention_body(geo, l, j, n_req_live)
                return carry

            jax.lax.fori_loop(0, n_req_live, req_loop, 0)

        per_geom(is_global, run_geom)
        kv_out_cp.wait()

    if (c.ex_opt or c.tiled_vmem_pads) and R <= c.untiled_max_rows:
        attn_all = (
            _heads_rms(S["attn_buf"][...], hq, hd, c.rms_eps).astype(bf16).astype(f32)
        )
        og_all = S["acc"][:, pl.ds(qd + 2 * kd, qd)]
        S["lhs"][:, pl.ds(0, qd)] = (
            jax.nn.sigmoid(og_all.astype(bf16).astype(f32)) * attn_all
        ).astype(bf16)
    else:

        def o_in_tile(rows, t):
            attn = (
                _heads_rms(S["attn_buf"][rows], hq, hd, c.rms_eps)
                .astype(bf16)
                .astype(f32)
            )
            og = S["acc"][rows, pl.ds(qd + 2 * kd, qd)]
            S["lhs"][rows, pl.ds(0, qd)] = (
                jax.nn.sigmoid(og.astype(bf16).astype(f32)) * attn
            ).astype(bf16)

        for_tiles(o_in_tile)
    if ep is not None:
        with _scope("o_mv"):
            ep.o_proj(
                EPOProjStep(
                    c=c,
                    S=S,
                    smr=smr,
                    layer=l,
                    me=me,
                    n_live=n_live,
                    mv=lambda on_col: mv_tiled(
                        l, "wo", nxt, smr, col_off=lambda ni: 0, on_col=on_col
                    ),
                    W=k.W,
                )
            )
        return
    # o-proj column tile m is reduce-scatter block m; two (or four) send buffers alternate
    assert all(t[4] == Hl for t in ws.tiles if t[0] == "wo")
    obuf = S["o_buf"]
    ors_kinds = ("o_rs", "o_rs1", "o_rs2", "o_rs3")
    nb_o = 4 if (c.tiled_o_rot or use_pair_o_rs) else 2

    def o_rc(b, peer):
        return rcopy(
            obuf.at[b],
            S["pad_o_rs"].at[me],
            peer,
            "o_rs",
            send_kind=ors_kinds[b],
        )

    def o_lc(b):
        return pltpu.make_async_copy(obuf.at[b], S["pad_o_rs"].at[me], tsem(_TS_LOCAL))

    def o_wait(m):
        pl.when(me != m)(lambda: o_rc(m % 2, m).wait_send())
        if not c.tiled_vmem_pads:
            pl.when(me == m)(lambda: o_lc(m % 2).wait())

    def o_pair_cp(b):
        return pltpu.make_async_remote_copy(
            src_ref=obuf.at[b],
            dst_ref=S["pad_o_rs"].at[b],
            send_sem=S["pair_ssems"].at[6],
            recv_sem=S["pair_rsems"].at[6],
            device_id=(jnp.bitwise_xor(me, 1),),
        )

    def o_ship(ni, m):
        if use_pair_o_rs:
            b_p = ni // 2
            packed = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])

            @pl.when((ni % 2) == (me % 2))
            def _():
                S["xfull"][:, pl.ds(b_p * (Hl // 2), Hl // 2)] = packed

            @pl.when((ni % 2) != (me % 2))
            def _():
                obuf[b_p] = packed
                o_pair_cp(b_p).start()

            return

        b = ni % nb_o
        if c.tiled_o_rot:
            if ni >= nb_o and ni - nb_o < c.tp - 1:
                o_rc((ni - nb_o) % nb_o, ws.phys_col("wo", ni - nb_o)).wait_send()
        elif ni >= 2:
            o_wait(ni - 2)

        if c.tiled_o_rot or (
            (c.ex_opt or c.tiled_vmem_pads) and R <= c.untiled_max_rows
        ):
            obuf.at[b][...] = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])
        else:

            def pack_tile(rows, t):
                obuf.at[b][rows] = _pack_bf16x2(S["acc"][rows, pl.ds(0, Hl)])

            for_tiles(pack_tile)
        if c.tiled_o_rot:
            if ni < c.tp - 1:
                o_rc(b, m).start()
            elif c.tiled_vmem_pads:
                S["pad_o_rs"][me] = obuf[b]
            else:
                o_lc(b).start()
        else:
            pl.when(me != m)(lambda: o_rc(b, m).start())
            if c.tiled_vmem_pads:
                pl.when(me == m)(lambda: S["pad_o_rs"].__setitem__(me, obuf[b]))
            else:
                pl.when(me == m)(lambda: o_lc(b).start())

    with _scope("o_mv"):
        mv_tiled(l, "wo", nxt, smr, col_off=lambda ni: 0, on_col=o_ship)
    with _scope("o_allreduce"):
        if use_pair_o_rs:
            core_o = me % 2
            chip_o = me // 2
            for b_p in range(4):
                o_pair_cp(b_p).wait()
            if c.combine_loop and R > 64:
                for b_p in range(4):

                    def _tile(rs, t, b_p=b_p):
                        obuf.at[b_p][rs] = _pack_bf16x2(
                            _unpack_bf16x2(
                                S["xfull"][rs, pl.ds(b_p * (Hl // 2), Hl // 2)]
                            )
                            + _unpack_bf16x2(S["pad_o_rs"][b_p, rs])
                        )

                    vpads_tiles(_tile)
            else:
                for b_p in range(4):
                    obuf[b_p] = _pack_bf16x2(
                        _unpack_bf16x2(S["xfull"][:, pl.ds(b_p * (Hl // 2), Hl // 2)])
                        + _unpack_bf16x2(S["pad_o_rs"][b_p])
                    )
            S["pad_o_rs"][4 + chip_o] = obuf.at[chip_o][...]
            ors_cps = [
                rcopy(
                    obuf.at[jnp.bitwise_xor(chip_o, d)],
                    S["pad_o_rs"].at[4 + chip_o],
                    2 * jnp.bitwise_xor(chip_o, d) + core_o,
                    "o_rs",
                )
                for d in (1, 2, 3)
            ]
            for cp in ors_cps:
                cp.start()
            for cp in ors_cps:
                cp.wait_send()
            for cp in ors_cps:
                cp.wait_recv()

            def _tile(rs, t):
                mine = _unpack_bf16x2(S["pad_o_rs"][4, rs])
                for d in range(5, 8):
                    mine = mine + _unpack_bf16x2(S["pad_o_rs"][d, rs])
                S["o_send"][rs] = _pack_bf16x2(mine)
                if c.pipe_ag:
                    hag_start("o_ag", S["o_send"], S["pad_o_ag"], t, S["oag_tsems"])

            vpads_tiles(_tile)
            if c.pipe_ag:
                pass  # sent per tile above
            elif c.fold_o:
                coll.all_gather_blocks_start("o_ag", S["o_send"], me, S["o_rsems"])
            else:
                with _scope("o_ag"):
                    coll.all_gather("o_ag", S["o_send"], me)
        else:
            if c.tiled_o_rot:
                for ni in range(max(0, c.tp - nb_o), c.tp - 1):
                    o_rc(ni % nb_o, ws.phys_col("wo", ni)).wait_send()
                if not c.tiled_vmem_pads:
                    o_lc((c.tp - 1) % nb_o).wait()
            else:
                for m in range(max(0, c.tp - 2), c.tp):
                    o_wait(m)
            for d in range(1, c.tp):
                o_rc(0, coll._peer(d)).wait_recv()

            if c.tiled_vmem_pads:

                def _tile(rs, t):
                    mine = _unpack_bf16x2(S["pad_o_rs"][0, rs])
                    for d in range(1, c.tp):
                        mine = mine + _unpack_bf16x2(S["pad_o_rs"][d, rs])
                    S["o_send"][rs] = _pack_bf16x2(mine)
                    if c.pipe_ag:
                        hag_start(
                            "o_ag",
                            S["o_send"],
                            S["pad_o_ag"],
                            t,
                            S["oag_tsems"],
                        )

                vpads_tiles(_tile)
                if c.pipe_ag:
                    pass  # sent per tile above
                elif c.fold_o:
                    coll.all_gather_blocks_start("o_ag", S["o_send"], me, S["o_rsems"])
                else:
                    with _scope("o_ag"):
                        coll.all_gather("o_ag", S["o_send"], me)
            else:

                def o_sum_tile(rows, t, v):
                    mine = _unpack_bf16x2(v[0][0])
                    for d in range(1, c.tp):
                        mine = mine + _unpack_bf16x2(v[0][d])
                    S["o_send"][rows] = _pack_bf16x2(mine)
                    # pipelined all-gather: the post-attention pass waits per row tile
                    ag_tile_start(
                        "o_ag",
                        S["o_send"],
                        S["pad_o_ag"],
                        me,
                        t,
                        S["oag_tsems"],
                    )
                    return []

                tile_pass([(pad_view(S["pad_o_rs"]), S["ptile"])], o_sum_tile)
