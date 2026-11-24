# SPDX-License-Identifier: Apache-2.0
"""Row-tiled MoE of the decode megakernel (MKConfig.row_tile)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.common.consts import (
    FP4_GROUP,
    FP4_PACK,
    LANE,
    SMALL,
)
from sglang_tpu.muse_megakernel.common.control import (
    _select,
    for_layer_experts,
)
from sglang_tpu.muse_megakernel.common.dots import (
    _fp4_blockdiag_dot,
    _fp4_dot,
    _fp4_qblock_ffn,
    _fp8_dot,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _rms,
    _topk_iter,
    _topk_xpose,
)
from sglang_tpu.muse_megakernel.common.packing import (
    _pack_bf16x2,
    _pack_y,
    _scale_fp8_range,
    _unpack_bf16x2,
    _unpack_y,
)
from sglang_tpu.muse_megakernel.common.plan import (
    _fold_g,
    compact_buckets,
    expert_units,
    owner_window_rows,
)
from sglang_tpu.muse_megakernel.common.scratch import (
    _TC,
    _TS_LOCAL,
    Y_XFULL_ROW0,
    yrow_ring,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx


def expert_io_tiled(k: KernelCtx, l, kind, n_u):
    """Expert slab ring of layer l for the row-tiled expert loop (the base program's

    kind_fns DMAs, stacked groups included): starts the first depth experts'
    DMAs.
    """
    He = k.He
    Il = k.Il
    S = k.S
    c = k.c
    group_lo = k.group_lo
    kind_layers = k.kind_layers
    layer_ex = k.layer_ex
    rget = k.rget
    if kind == "fp4":
        nslot, W13, W2 = c.expert_depth + 1, S["w13"], S["w2"]
    else:
        nslot = (c.expert_depth + 1) // 2
        W13 = S["w13"].reshape(nslot, He // 4, 2 * Il)
        W2 = S["w2"].reshape(nslot, Il // 4, He)
    depth = nslot - 1
    S13, S2 = S["w13s"], S["w2s"]
    pack = FP4_PACK if kind == "fp4" else 4
    parts = (
        (W13, S["w13_sems"], He // pack, 2 * Il, False),
        (
            S13,
            S["w13s_sems"],
            He // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
            2 * Il,
            True,
        ),
        (W2, S["w2_sems"], Il // pack, He, False),
        (
            S2,
            S["w2s_sems"],
            Il // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
            He,
            True,
        ),
    )

    def slab_copy(refs, off, e, slot, j):
        dst, sem, rows_, cols, is_scale = parts[j]
        src = refs[j] if off is None else refs[j].at[off]
        src = src if is_scale else src.bitcast(jnp.uint32)
        return pltpu.make_async_copy(
            src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
            dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
            sem.at[slot],
        )

    for_layer = functools.partial(
        for_layer_experts, l, kind, layer_ex, kind_layers, group_lo
    )

    def issue_experts(lo, hi):
        def leaf(refs, off):
            def one(t, carry):
                e = rget("uniq", t)
                for j in range(4):
                    slab_copy(refs, off, e, t % nslot, j).start()
                return carry

            jax.lax.fori_loop(lo, hi, one, 0)

        for_layer(leaf)

    pairs = c.ex_opt and nslot >= 4 and len(expert_units(c)) <= 8
    pdepth = nslot - 2 if pairs else depth

    def start_expert(i):
        if pairs:
            pl.when((i % 2 == 0) & (i + pdepth < n_u))(
                lambda: issue_experts(i + pdepth, jnp.minimum(i + pdepth + 2, n_u))
            )
        else:

            @pl.when(i + depth < n_u)
            def _():
                e = rget("uniq", i + depth)
                slot = (i + depth) % nslot
                for_layer(
                    lambda refs, off: [
                        slab_copy(refs, off, e, slot, j).start() for j in range(4)
                    ]
                )

    def wait_expert(slot, j_parts=(0, 1, 2, 3)):
        tl = l if isinstance(l, int) else kind_layers[kind][0]
        trefs, toff = layer_ex[tl]
        for j in j_parts:
            slab_copy(trefs, toff, 0, slot, j).wait()

    issue_experts(0, jnp.minimum(pdepth, n_u))
    return nslot, W13, W2, S13, S2, start_expert, wait_expert


def moe_tiled(k: KernelCtx, l, smr, kind, nxt):
    """Leaves the ffn branch all-gathered in pad_f (or in flight)."""
    H = k.H
    He = k.He
    Hel = k.Hel
    Hl = k.Hl
    Il = k.Il
    K = k.K
    NBR = k.NBR
    R = k.R
    RT = k.RT
    S = k.S
    ag_start = k.ag_start
    ag_tile_start = k.ag_tile_start
    bf16 = k.bf16
    blk_owner = k.blk_owner
    c = k.c
    coll = k.coll
    divmod_nn = k.divmod_nn
    expert_io_tiled = k.expert_io_tiled
    f32 = k.f32
    for_tiles = k.for_tiles
    gathered_tile = k.gathered_tile
    hag_start = k.hag_start
    me = k.me
    mv_tiled = k.mv_tiled
    n_live = k.n_live
    o_blocks = k.o_blocks
    pad_view = k.pad_view
    pipe_u = k.pipe_u
    rcopy = k.rcopy
    rget = k.rget
    stream_pair_y = k.stream_pair_y
    tc_rmap = k.tc_rmap
    tc_smap = k.tc_smap
    tile_live = k.tile_live
    tile_pass = k.tile_pass
    tiled_fold_lhs = k.tiled_fold_lhs
    tp_rank = k.tp_rank
    tsem = k.tsem
    u_finish = k.u_finish
    u_start = k.u_start
    unique_experts = k.unique_experts
    use_pair_y = k.use_pair_y
    vpads_tiles = k.vpads_tiles
    ws = k.ws
    y_block = k.y_block
    yw = k.yw
    Rc = c.rows_per_dev
    Ep = c.n_experts_padded
    vy = c.tiled_vmem_y
    vpads = c.tiled_vmem_pads
    ws.hold = ws.ranges["wunpool"][0]
    if c.fold_o:
        g_pf = smr[SMALL["pre_ffn_w"] : SMALL["pre_ffn_w"] + 1, :H]
        a_alpha = smr[SMALL["attn_alpha"] : SMALL["attn_alpha"] + 1, :H]
        a_beta = smr[SMALL["attn_beta"] : SMALL["attn_beta"] + 1, :H]

        def _fold_lhs_o(rs, t):
            S["lhs"][rs] = (a_alpha * S["h_stream"][rs] * g_pf).astype(bf16)

        vpads_tiles(_fold_lhs_o)
        S["fold_v"][0:1, :] = a_beta * g_pf
        wait_block, finish = o_blocks()
        names = ("wpool", "wrouter")
        with _scope("pool_router_fold"):
            acc_pr = ws.mv_fold(
                l,
                names,
                nxt,
                _fold_g(c, names),
                Hl,
                wait_block,
                tiled_fold_lhs("pad_o_ag"),
            )
        finish()
        wp_sc = smr[SMALL["wpool_s"] : SMALL["wpool_s"] + 1, :Hel]
        if c.tile_loops:
            # per-row scales from one loop; the partial sums combine as values after it
            def _fold_pr(rs, t):
                a_val = gathered_tile(S["pad_o_ag"].at[:, rs])
                s_a = jax.lax.rsqrt(
                    jnp.mean(jnp.square(a_val), axis=-1, keepdims=True) + c.post_eps
                )
                s = a_alpha * S["h_stream"][rs] + a_beta * (a_val * s_a).astype(
                    bf16
                ).astype(f32)
                S["h_stream"][rs] = s
                r_inv = jax.lax.rsqrt(
                    jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                )
                S["fold_stage"][rs, pl.ds(0, LANE)] = jnp.broadcast_to(s_a, (RT, LANE))
                S["fold_stage"][rs, pl.ds(LANE, LANE)] = jnp.broadcast_to(
                    r_inv, (RT, LANE)
                )

            for_tiles(_fold_pr)
            s_a_all = S["fold_stage"][:, pl.ds(0, 1)]
            r_inv_all = S["fold_stage"][:, pl.ds(LANE, 1)]
            pooled = (
                (
                    r_inv_all
                    * (acc_pr["wpool"][:R] + s_a_all * acc_pr["wpool"][R:])
                    * wp_sc
                )
                .astype(bf16)
                .astype(f32)
            )
            S["pool_send"][...] = _pack_bf16x2(pooled)
            S["acc"][:, pl.ds(0, Ep)] = r_inv_all * (
                acc_pr["wrouter"][:R] + s_a_all * acc_pr["wrouter"][R:]
            )
        else:
            for t in range(R // RT):
                r0, r1 = t * RT, (t + 1) * RT
                rs = pl.ds(r0, RT)
                a_val = gathered_tile(S["pad_o_ag"].at[:, rs])
                s_a = jax.lax.rsqrt(
                    jnp.mean(jnp.square(a_val), axis=-1, keepdims=True) + c.post_eps
                )
                s = a_alpha * S["h_stream"][rs] + a_beta * (a_val * s_a).astype(
                    bf16
                ).astype(f32)
                S["h_stream"][rs] = s
                r_inv = jax.lax.rsqrt(
                    jnp.mean(jnp.square(s), axis=-1, keepdims=True) + c.rms_eps
                )
                pooled_t = (
                    (
                        r_inv
                        * (
                            acc_pr["wpool"][r0:r1]
                            + s_a * acc_pr["wpool"][R + r0 : R + r1]
                        )
                        * wp_sc
                    )
                    .astype(bf16)
                    .astype(f32)
                )
                S["pool_send"][rs] = _pack_bf16x2(pooled_t)
                S["acc"][rs, pl.ds(0, Ep)] = r_inv * (
                    acc_pr["wrouter"][r0:r1] + s_a * acc_pr["wrouter"][R + r0 : R + r1]
                )
        pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
    else:
        with _scope("pool_mv"):
            mv_tiled(l, "wpool", nxt, smr)
        if (c.ex_opt or vpads) and R <= c.untiled_max_rows:
            S["pool_send"][...] = _pack_bf16x2(
                S["acc"][:, pl.ds(0, Hel)].astype(bf16).astype(f32)
            )
        else:

            def pool_tile(rows, t):
                pooled = S["acc"][rows, pl.ds(0, Hel)].astype(bf16).astype(f32)
                S["pool_send"][rows] = _pack_bf16x2(pooled)

            for_tiles(pool_tile)
        if vpads:
            pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
        else:
            pool_wait = ag_start("pool", S["pool_send"], S["pad_pool"].at[tp_rank])
        with _scope("router_mv"):
            mv_tiled(l, "wrouter", nxt, smr)
    bias = smr[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :Ep]

    def topk_rows(rows):
        sc = jax.nn.sigmoid(S["acc"][rows, pl.ds(0, Ep)] * c.router_mult)
        ids_list, picks = _topk_iter(sc + bias, K)
        wsel = jnp.concatenate(
            [jnp.sum(jnp.where(p, sc, 0.0), axis=1, keepdims=True) for p in picks],
            axis=1,
        )
        wsel = (
            wsel / (jnp.sum(wsel, axis=1, keepdims=True) + c.router_eps) * c.route_scale
        )
        S["ids_v"][rows, pl.ds(0, K)] = jnp.concatenate(ids_list, axis=1)
        S["wsel_v"][rows, pl.ds(0, K)] = wsel

    with _scope("topk"):
        if c.topk_xpose and R >= 64:
            if R <= LANE:
                ids_f, wsel_f = _topk_xpose(
                    jax.nn.sigmoid(S["acc"][:, pl.ds(0, Ep)] * c.router_mult),
                    bias,
                    K,
                    c.router_eps,
                    c.route_scale,
                )
                S["ids_v"][...] = ids_f
                S["wsel_v"][...] = wsel_f
            else:
                br_tk = (
                    LANE
                    if R % LANE == 0
                    else (64 if R % 64 == 0 else (32 if R % 32 == 0 else RT))
                )

                def topk_xpose_tile(t, carry_):
                    rs_tk = pl.ds(pl.multiple_of(t * br_tk, br_tk), br_tk)
                    ids_f, wsel_f = _topk_xpose(
                        jax.nn.sigmoid(S["acc"][rs_tk, pl.ds(0, Ep)] * c.router_mult),
                        bias,
                        K,
                        c.router_eps,
                        c.route_scale,
                    )
                    S["ids_v"][rs_tk] = ids_f
                    S["wsel_v"][rs_tk] = wsel_f
                    return carry_

                jax.lax.fori_loop(0, R // br_tk, topk_xpose_tile, 0)
        elif c.ex_opt and R <= c.untiled_max_rows:
            topk_rows(pl.ds(0, R))
        else:
            for_tiles(lambda rows, t: topk_rows(rows))
    with _scope("unique_experts"):
        n_u = unique_experts(None, l)
    nslot, W13, W2, S13, S2, start_expert, wait_expert = expert_io_tiled(l, kind, n_u)
    with _scope("pool_ag"):
        pool_wait()
    pre_w = smr[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He]
    prequant_x = bool(c.ex_lhs_fp8 >= 1 and (kind != "fp4"))
    xsc_ref = S["ids_v"].bitcast(f32)

    with _scope("ex_setup"):
        if vpads:

            def _tile(rs, t):
                x = _rms(
                    gathered_tile(S["pad_pool"].at[:, rs]), c.rms_eps, pre_w
                ).astype(bf16)
                if prequant_x:
                    xs_f32, xsc = _scale_fp8_range(x)
                    S["xfull"][rs] = _pack_bf16x2(xs_f32)
                    xsc_ref[rs] = jnp.broadcast_to(xsc, (RT, LANE))
                else:
                    S["xfull"][rs] = _pack_bf16x2(x.astype(f32))

            vpads_tiles(_tile)
        else:

            def xfull_tile(rows, t, v):
                x = _rms(gathered_tile(v[0]), c.rms_eps, pre_w).astype(bf16)
                if prequant_x:
                    xs_f32, xsc = _scale_fp8_range(x)
                    S["xfull"][rows] = _pack_bf16x2(xs_f32)
                    xsc_ref[rows] = jnp.broadcast_to(xsc, (RT, LANE))
                else:
                    S["xfull"][rows] = _pack_bf16x2(x.astype(f32))
                return []

            tile_pass([(pad_view(S["pad_pool"]), S["pltile"])], xfull_tile)
        if vy >= 1:
            if c.combine_loop and R > 64:
                z_rt = jnp.zeros((RT, K, yw), jnp.uint32)
                for t in range(R // RT):
                    S["ysend"][pl.ds(t * RT, RT)] = z_rt
            else:
                S["ysend"][...] = jnp.zeros((R, K, yw), jnp.uint32)

    # QBLOCK fp4: pair indices by shift and mask (perm entries are >= 0)
    q_idx = bool(c.qblock and kind == "fp4" and c.vec_route)

    def perm_at(i):
        if not q_idx:
            return rget("perm", i)
        hi, lo = divmod_nn(i, LANE)
        return S["route_s"][3 * NBR + hi, lo]

    pad_y = S["pad_y"]
    if vy == 0:
        y_rsem = S["tc_rsems"].at[tc_rmap["y"]]
        for b in range(3):
            S["ypend"][b] = 0
        ystage = S["ystage"]

        def ydrain(b):
            """Waits for the row DMAs out of ystage[b] (local or remote: same bytes)."""
            tmpl = pltpu.make_async_copy(
                ystage.at[0, pl.ds(0, 1)],
                pad_y.at[0, 0, pl.ds(0, 1)],
                S["ystage_sems"].at[b],
            )

            def w(i, carry):
                tmpl.wait()
                return carry

            jax.lax.fori_loop(0, S["ypend"][b], w, 0)
            S["ypend"][b] = 0

        def yemit(yp, first, n_valid):
            n = yp.shape[0]
            b = S["ypend"][2] % 2
            ydrain(b)
            ystage.at[b][pl.ds(0, n), :] = yp

            def issue(j, carry):
                p = perm_at(first + j)
                if q_idx:
                    r, k = divmod_nn(p, K)
                    owner, rl = divmod_nn(r, Rc)
                else:
                    r, k = p // K, p % K
                    owner, rl = r // Rc, r % Rc
                src = ystage.at[b, pl.ds(j, 1)]
                dst = pad_y.at[me, rl, pl.ds(k, 1)]
                if q_idx:
                    # one remote copy per row, own rows too (y_rs counts them):
                    # a local/remote pair runs both copies' bounds checks
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=S["ystage_sems"].at[b],
                        recv_sem=y_rsem,
                        device_id=(owner,),
                    ).start()
                    return carry

                @pl.when(owner == me)
                def _():
                    pltpu.make_async_copy(src, dst, S["ystage_sems"].at[b]).start()

                @pl.when(owner != me)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=S["ystage_sems"].at[b],
                        recv_sem=y_rsem,
                        device_id=(owner,),
                    ).start()

                return carry

            jax.lax.fori_loop(0, n_valid, issue, 0)
            S["ypend"][b] = n_valid
            S["ypend"][2] = S["ypend"][2] + 1

    def y_block_cp(owner):
        return rcopy(
            S["ysend"].at[pl.ds(pl.multiple_of(owner * Rc, Rc), Rc)],
            pad_y.at[me],
            owner,
            "y",
        )

    if use_pair_y:
        core_y = me % 2
        chip_y = me // 2
        partner_y = jnp.bitwise_xor(me, 1)
        other_y = 1 - core_y

        def ysend_blk(rk):
            return S["ysend"].at[pl.ds(pl.multiple_of(rk * Rc, Rc), Rc)]

        yp_cps = [
            pltpu.make_async_remote_copy(
                src_ref=ysend_blk(2 * c_o + other_y),
                dst_ref=pad_y.at[c_o],
                send_sem=S["pair_ssems"].at[4],
                recv_sem=S["pair_rsems"].at[4],
                device_id=(partner_y,),
            )
            for c_o in range(4)
        ]

    def ystore(yp, pairs, first, n_valid):
        if vy >= 1:
            for j, p in enumerate(pairs):
                S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]
        else:
            yemit(yp, first, n_valid)

    split_w = c.ex_opt
    bd_kc = c.blockdiag_kc
    cw13_fp4 = 64 if c.ex_opt and c.fp4_chunk_words == 128 else c.fp4_chunk_words
    cw2_fp4 = min(Il // FP4_PACK, 64 if c.ex_opt else 16)
    cw2_fp8 = min(Il // 4, 128 if c.ex_opt else 32)

    def wait_w2_if_needed(slot, wait_w2):
        if split_w:
            with _scope("ex_wait"):
                if isinstance(wait_w2, bool):
                    if wait_w2:
                        wait_expert(slot, (2, 3))
                else:
                    pl.when(wait_w2)(lambda: wait_expert(slot, (2, 3)))

    def gather_x(pairs):
        if c.ex_opt:
            packed = jnp.concatenate(
                [S["xfull"][pl.ds(p // K, 1), :] for p in pairs], axis=0
            )
            return _unpack_bf16x2(packed).astype(bf16)
        return jnp.concatenate(
            [_unpack_bf16x2(S["xfull"][pl.ds(p // K, 1), :]) for p in pairs],
            axis=0,
        ).astype(bf16)

    def pq_sc_of(pairs):
        return jnp.concatenate(
            [xsc_ref[pl.ds(p // K, 1), :] for p in pairs],
            axis=0,
        )[:, 0:1]

    def run_ffn_and_store(
        nrows, pair_fn, slot, e, blockdiag, first, n_valid, wait_w2=True
    ):
        if c.qblock and kind == "fp4":
            # w2 waited for up front: a mid-arm DMA wait is a scheduling barrier
            # that drains the MXU between w13 and w2
            wait_w2_if_needed(slot, wait_w2)
            pairs = [pair_fn(j) for j in range(nrows)]
            rk = [divmod_nn(p, K) if q_idx else (p // K, p % K) for p in pairs]
            xs = jnp.concatenate([S["xfull"][pl.ds(r, 1), :] for r, _ in rk], axis=0)
            y = _fp4_qblock_ffn(
                _unpack_bf16x2(xs).astype(bf16),
                W13.at[slot],
                S["w13s"].at[slot],
                W2.at[slot],
                S["w2s"].at[slot],
                He,
                Il,
                c.qblock,
            )
            # full arms quantize bf16 y, as their VMEM-staged path did
            yp = _pack_y(y if blockdiag else y.astype(bf16), c)
            if vy >= 1:
                for j, (r, k) in enumerate(rk):
                    S["ysend"].at[r][pl.ds(k, 1), :] = yp[j : j + 1]
            else:
                yemit(yp, first, n_valid)
            return
        can_fp8_x = bool(
            prequant_x and not blockdiag and nrows >= 32 and nrows % 32 == 0
        )
        can_stage_ex = bool(
            c.ex_w2_stage
            and not blockdiag
            and nrows >= 16
            and nrows % 16 == 0
            and nrows <= R
            and He + Il <= H
            and 2 * Il <= S["acc"].shape[1]
        )
        if can_stage_ex:
            pq_chunks = []
            for s0 in range(0, nrows, 16):
                ps16 = [pair_fn(s0 + j) for j in range(16)]
                S["lhs"][pl.ds(s0, 16), pl.ds(0, He)] = gather_x(ps16)
                if prequant_x:
                    pq_chunks.append(pq_sc_of(ps16))
            pq_sc = jnp.concatenate(pq_chunks, axis=0) if prequant_x else None
            w13_prequant_sc = pq_sc if can_fp8_x else None
            w13_post_sc = pq_sc if (prequant_x and not can_fp8_x) else None

            def x_stage_chunk_fn(c_w, cw_w, out_fp8):
                w_x = S["lhs"][
                    pl.ds(0, nrows),
                    pl.ds(c_w * FP4_PACK, cw_w * FP4_PACK),
                ]
                return w_x.astype(jnp.float8_e4m3fn) if out_fp8 else w_x

            gu_stage = S["acc"].at[pl.ds(0, nrows), pl.ds(0, 2 * Il)]
            if kind == "fp4":
                _fp4_dot(
                    None,
                    W13.at[slot],
                    S["w13s"].at[slot],
                    He,
                    2 * Il,
                    cw13_fp4,
                    fast_bf16=c.ex_opt,
                    qblock=c.qblock,
                    via_fp8=c.ex_via_fp8,
                    n_tile=c.bd_n_tile,
                    lhs_chunk_fn=x_stage_chunk_fn,
                    prequant_sc=w13_prequant_sc,
                    out_stage_ref=gu_stage,
                    post_scale=w13_post_sc,
                    nrows=nrows,
                )
            else:
                _fp8_dot(
                    None,
                    W13.at[slot],
                    S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(f32),
                    He,
                    2 * Il,
                    c.fp4_chunk_words,
                    fp8_lhs=False,
                    lhs_chunk_fn=x_stage_chunk_fn,
                    prequant_sc=w13_prequant_sc,
                    out_stage_ref=gu_stage,
                    post_scale=w13_post_sc,
                    nrows=nrows,
                )
            quant_h = bool(
                c.ex_lhs_fp8 >= 2
                and (kind != "fp4")
                and Il == 512
                and nrows >= 32
                and nrows % 32 == 0
            )
            if quant_h:
                gt0 = S["acc"][pl.ds(0, nrows), pl.ds(0, 256)]
                up0 = S["acc"][pl.ds(0, nrows), pl.ds(Il, 256)]
                h0 = gt0 * jax.nn.sigmoid(gt0) * up0
                m0 = jnp.maximum(jnp.abs(h0[:, :128]), jnp.abs(h0[:, 128:]))
                S["acc"][pl.ds(0, nrows), pl.ds(0, 256)] = h0
                gt1 = S["acc"][pl.ds(0, nrows), pl.ds(256, 256)]
                up1 = S["acc"][pl.ds(0, nrows), pl.ds(Il + 256, 256)]
                h1 = gt1 * jax.nn.sigmoid(gt1) * up1
                m1 = jnp.maximum(jnp.abs(h1[:, :128]), jnp.abs(h1[:, 128:]))
                amax = jnp.max(jnp.maximum(m0, m1), axis=1, keepdims=True)
                h_pq_sc = jnp.where(
                    amax == 0,
                    jnp.zeros_like(amax),
                    amax * jnp.float32(1.0 / 448.0),
                )
                inv_h = jnp.where(
                    amax == 0,
                    jnp.zeros_like(amax),
                    jnp.float32(448.0) / jnp.maximum(amax, jnp.float32(1e-30)),
                )
                S["lhs"][pl.ds(0, nrows), pl.ds(He + 256, 256)] = (h1 * inv_h).astype(
                    bf16
                )
                h0_r = S["acc"][pl.ds(0, nrows), pl.ds(0, 256)]
                S["lhs"][pl.ds(0, nrows), pl.ds(He, 256)] = (h0_r * inv_h).astype(bf16)
            else:
                for c_h in range(0, Il, 256):
                    th = min(256, Il - c_h)
                    gt = S["acc"][pl.ds(0, nrows), pl.ds(c_h, th)]
                    up_ = S["acc"][pl.ds(0, nrows), pl.ds(Il + c_h, th)]
                    S["lhs"][pl.ds(0, nrows), pl.ds(He + c_h, th)] = (
                        gt * jax.nn.sigmoid(gt) * up_
                    ).astype(bf16)
                h_pq_sc = None
            wait_w2_if_needed(slot, wait_w2)

            def h_chunk_fn(c_w, cw_w, out_fp8):
                w_h = S["lhs"][
                    pl.ds(0, nrows),
                    pl.ds(He + c_w * FP4_PACK, cw_w * FP4_PACK),
                ]
                return w_h.astype(jnp.float8_e4m3fn) if out_fp8 else w_h

            y_stage = S["lhs"].at[pl.ds(0, nrows), pl.ds(0, He)]
            if kind == "fp4":
                _fp4_dot(
                    None,
                    W2.at[slot],
                    S["w2s"].at[slot],
                    Il,
                    He,
                    cw2_fp4,
                    fast_bf16=c.ex_opt,
                    qblock=c.qblock,
                    via_fp8=c.ex_via_fp8,
                    n_tile=c.bd_n_tile,
                    lhs_chunk_fn=h_chunk_fn,
                    prequant_sc=h_pq_sc,
                    out_stage_ref=y_stage,
                    post_scale=(None),
                    nrows=nrows,
                )
            else:
                _fp8_dot(
                    None,
                    W2.at[slot],
                    S2[slot, 0:1, :].astype(f32),
                    Il,
                    He,
                    cw2_fp8,
                    fp8_lhs=False,
                    lhs_chunk_fn=h_chunk_fn,
                    prequant_sc=h_pq_sc,
                    out_stage_ref=y_stage,
                    nrows=nrows,
                )
            for s0 in range(0, nrows, 16):
                yp = _pack_y(S["lhs"][pl.ds(s0, 16), pl.ds(0, He)], c)
                if vy >= 1:
                    for j in range(16):
                        p = pair_fn(s0 + j)
                        S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]
                else:
                    pl.when(s0 < n_valid)(
                        lambda s0=s0, yp=yp: yemit(
                            yp, first + s0, jnp.clip(n_valid - s0, 0, 16)
                        )
                    )
            return

        pairs = [pair_fn(j) for j in range(nrows)]
        pq_sc = pq_sc_of(pairs) if prequant_x else None
        w13_prequant_sc = pq_sc if can_fp8_x else None
        w13_post_sc = pq_sc if (prequant_x and not can_fp8_x) else None
        xc_raw = gather_x(pairs)
        xc = xc_raw.astype(jnp.float8_e4m3fn) if can_fp8_x else xc_raw
        if blockdiag:
            gu = _fp4_blockdiag_dot(
                xc,
                W13.at[slot],
                S["w13s"].at[slot],
                He,
                2 * Il,
                bd_kc,
                kperm=False,
                qblock=c.qblock,
                fp8_lhs=False,
                n_tile=c.bd_n_tile,
                prequant_sc=w13_prequant_sc,
                nrows=nrows,
                post_scale=w13_post_sc,
            )
        elif kind == "fp4":
            gu = _fp4_dot(
                xc,
                W13.at[slot],
                S["w13s"].at[slot],
                He,
                2 * Il,
                cw13_fp4,
                fast_bf16=c.ex_opt,
                qblock=c.qblock,
                via_fp8=c.ex_via_fp8,
                n_tile=c.bd_n_tile,
                prequant_sc=w13_prequant_sc,
                post_scale=w13_post_sc,
                nrows=nrows,
            )
        else:
            gu = _fp8_dot(
                xc,
                W13.at[slot],
                S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(f32),
                He,
                2 * Il,
                c.fp4_chunk_words,
                fp8_lhs=False,
                prequant_sc=w13_prequant_sc,
                post_scale=w13_post_sc,
                nrows=nrows,
            )
        gate, up = gu[:, :Il], gu[:, Il:]
        h = (gate * jax.nn.sigmoid(gate) * up).astype(bf16)
        wait_w2_if_needed(slot, wait_w2)
        h_pq = bool(
            c.ex_lhs_fp8 >= 2
            and (kind != "fp4")
            and not blockdiag
            and nrows >= 32
            and nrows % 32 == 0
        )
        if blockdiag:
            y = _fp4_blockdiag_dot(
                h,
                W2.at[slot],
                S["w2s"].at[slot],
                Il,
                He,
                bd_kc,
                kperm=False,
                qblock=c.qblock,
                fp8_lhs=(h_pq),
                n_tile=c.bd_n_tile,
            )
        elif kind == "fp4":
            y = _fp4_dot(
                h,
                W2.at[slot],
                S["w2s"].at[slot],
                Il,
                He,
                cw2_fp4,
                fast_bf16=c.ex_opt,
                qblock=c.qblock,
                via_fp8=c.ex_via_fp8,
                n_tile=c.bd_n_tile,
            )
        else:
            y = _fp8_dot(
                h,
                W2.at[slot],
                S2[slot, 0:1, :].astype(f32),
                Il,
                He,
                cw2_fp8,
                fp8_lhs=(h_pq),
            )
        ystore(_pack_y(y, c), pairs, first, n_valid)

    def pick_tree(arms, cnt):
        if len(arms) == 1:
            arms[0][1]()
            return
        mid = len(arms) // 2
        jax.lax.cond(
            cnt <= arms[mid - 1][0],
            lambda: pick_tree(arms[:mid], cnt),
            lambda: pick_tree(arms[mid:], cnt),
        )

    def ebody(i, carry, part=None):
        slot = i % nslot
        start_expert(i)
        with _scope("ex_wait"):
            wait_expert(slot, (0, 1) if split_w else (0, 1, 2, 3))
        e = rget("uniq", i)
        off, cnt = rget("ex_off", i), rget("ex_cnt", i)

        def compact(nrows):
            # bucket rows past the expert's count repeat its last pair (same write)
            with _scope(f"ex_c{nrows}"):
                run_ffn_and_store(
                    nrows,
                    lambda j: perm_at(off + jnp.minimum(j, cnt - 1)),
                    slot,
                    e,
                    True,
                    off,
                    cnt,
                )

        def full():
            can_bd = kind == "fp4"
            use_bd_chunk = c.ex_bd_all and can_bd
            CH = 8 if use_bd_chunk else min(c.expert_chunk, R)

            def full_one(nrows):
                with _scope(f"ex_f{nrows}"):
                    run_ffn_and_store(
                        nrows,
                        lambda j: perm_at(off + jnp.minimum(j, cnt - 1)),
                        slot,
                        e,
                        False,
                        off,
                        cnt,
                    )

            def chunk(ch, carry_):
                run_ffn_and_store(
                    CH,
                    lambda j: perm_at(off + jnp.minimum(ch * CH + j, cnt - 1)),
                    slot,
                    e,
                    use_bd_chunk,
                    off + ch * CH,
                    jnp.minimum(CH, cnt - ch * CH),
                    wait_w2=(ch == 0),
                )
                return carry_

            def run_chunks():
                with _scope("ex_fch"):
                    jax.lax.fori_loop(0, (cnt + CH - 1) // CH, chunk, 0)

            def bd_chunk_step(ch, carry_):
                run_ffn_and_store(
                    8,
                    lambda j: rget("perm", off + jnp.minimum(ch * 8 + j, cnt - 1)),
                    slot,
                    e,
                    True,
                    off + ch * 8,
                    jnp.minimum(8, cnt - ch * 8),
                    wait_w2=(ch == 0),
                )
                return carry_

            def run_bd_chunks():
                jax.lax.fori_loop(0, (cnt + 7) // 8, bd_chunk_step, 0)

            with _scope("ex_full"):
                # qblock: streamed-Y plans bucket busy experts too (a pass
                # costs its rows, not CH)
                if (
                    c.ex_opt
                    and (vy >= 1 or (c.qblock and kind == "fp4"))
                    and not use_bd_chunk
                ):
                    lo_b = c.compact_rows if kind == "fp4" else 0
                    far = []
                    if lo_b < R:
                        raw_fb = (8, 16, 32) if c.ex_w2_stage else (16, 24)
                        fb = tuple(b for b in raw_fb if lo_b < b <= R)
                        for b in fb:
                            far.append((b, functools.partial(full_one, b)))
                        if fb and fb[-1] >= R:
                            far[-1] = (None, far[-1][1])
                        else:
                            far.append((None, run_chunks))
                    else:
                        far[-1] = (None, far[-1][1])
                    pick_tree(far, cnt)
                else:
                    run_chunks()

        if kind == "fp4" and c.compact_rows:
            buckets = compact_buckets(c)
            arms = [(b, functools.partial(compact, b)) for b in buckets]
            if buckets[-1] < R:
                arms.append((None, full))
            else:
                arms[-1] = (None, arms[-1][1])
            if part == "few":
                arms = [a for a in arms if a[0] is not None and a[0] <= c.split_rows]
                arms[-1] = (None, arms[-1][1])
            elif part == "many":
                arms = [a for a in arms if a[0] is None or a[0] > c.split_rows]
            pick_tree(arms, cnt)
        else:
            full()
        if stream_pair_y:

            def ready_py(nb):
                end = rget("blk_end", jnp.minimum(nb, 2))
                return (nb < 3) & (end <= i + 1)

            def send_py(nb):
                c_o = jnp.bitwise_xor(chip_y, 3 - nb)
                _select(c_o, [0, 1, 2, 3], lambda co: yp_cps[co].start())
                return nb + 1

            return jax.lax.while_loop(ready_py, send_py, carry)
        if vy >= 1 and y_block and not use_pair_y:

            def ready(nb):
                end = rget("blk_end", jnp.minimum(nb, c.tp - 2))
                return (nb < c.tp - 1) & (end <= i + 1)

            def send_block(nb):
                owner = blk_owner(nb)
                pl.when(owner * Rc < n_live)(y_block_cp(owner).start)
                return nb + 1

            return jax.lax.while_loop(ready, send_block, carry)
        return carry

    with _scope("experts"):
        if c.split_rows and kind == "fp4":
            n_groups = (
                4 if stream_pair_y else (c.tp if (y_block and not use_pair_y) else 1)
            )

            def group(g_, carry):
                lo = jnp.where(g_ == 0, 0, rget("blk_end", jnp.maximum(g_ - 1, 0)))
                mid, hi = rget("blk_mid", g_), rget("blk_end", g_)
                carry = jax.lax.fori_loop(
                    lo, mid, functools.partial(ebody, part="few"), carry
                )
                return jax.lax.fori_loop(
                    mid, hi, functools.partial(ebody, part="many"), carry
                )

            nb_sent = jax.lax.fori_loop(0, n_groups, group, jnp.int32(0))
        else:
            nb_sent = jax.lax.fori_loop(0, n_u, ebody, jnp.int32(0))
    if vy == 0:
        ydrain(0)
        ydrain(1)
    ws.flush()
    my_live = me * Rc < n_live
    with _scope("y_rs"):
        if use_pair_y:
            if stream_pair_y:

                def send_rem_py(nb, carry_):
                    c_o = jnp.bitwise_xor(chip_y, 3 - nb)
                    _select(c_o, [0, 1, 2, 3], lambda co: yp_cps[co].start())
                    return carry_

                jax.lax.fori_loop(nb_sent, 4, send_rem_py, 0)
                for cp in yp_cps:
                    cp.wait()
            else:
                for cp in yp_cps:
                    cp.start()
                for cp in yp_cps:
                    cp.wait()
            if c.combine_loop:

                def pair_reduce_row(r, carry_):
                    for c_o in range(4):
                        r_same = 2 * c_o + core_y
                        S["ysend"].at[r_same * Rc + r][...] = _pack_y(
                            _unpack_y(S["ysend"].at[r_same * Rc + r][...], c)
                            + _unpack_y(pad_y.at[c_o, r][...], c),
                            c,
                        )
                    return carry_

                jax.lax.fori_loop(0, Rc, pair_reduce_row, 0)
            else:
                for c_o in range(4):
                    r_same = 2 * c_o + core_y
                    ysend_blk(r_same)[...] = _pack_y(
                        _unpack_y(ysend_blk(r_same)[...], c) + _unpack_y(pad_y[c_o], c),
                        c,
                    )
            if c.tiled_pair_y >= 2:
                m1_y = jnp.where(core_y == 0, 2, 1).astype(jnp.int32)
                m2_y = 3 - m1_y
                nb1_y = 2 * jnp.bitwise_xor(chip_y, m1_y) + core_y
                nb2_y = 2 * jnp.bitwise_xor(chip_y, m2_y) + core_y
                c2a = [
                    rcopy(ysend_blk(nb1_y), pad_y.at[4], nb1_y, "y"),
                    rcopy(
                        ysend_blk(2 * jnp.bitwise_xor(chip_y, 3) + core_y),
                        pad_y.at[5],
                        nb1_y,
                        "y",
                    ),
                ]
                for cp in c2a:
                    cp.start()
                for cp in c2a:
                    cp.wait_send()
                for cp in c2a:
                    cp.wait_recv()
                if c.combine_loop:

                    def pair_c2_row(r, carry_):
                        pad_y.at[7, r][...] = _pack_y(
                            _unpack_y(S["ysend"].at[nb2_y * Rc + r][...], c)
                            + _unpack_y(pad_y.at[5, r][...], c),
                            c,
                        )
                        return carry_

                    jax.lax.fori_loop(0, Rc, pair_c2_row, 0)
                else:
                    pad_y[7] = _pack_y(
                        _unpack_y(ysend_blk(nb2_y)[...], c) + _unpack_y(pad_y[5], c),
                        c,
                    )
                pad_y[5] = ysend_blk(me)[...]
                c2b = pltpu.make_async_remote_copy(
                    src_ref=pad_y.at[7],
                    dst_ref=pad_y.at[6],
                    send_sem=S["pair_ssems"].at[5],
                    recv_sem=S["pair_rsems"].at[5],
                    device_id=(nb2_y,),
                )
                c2b.start()
                c2b.wait()
            else:
                pad_y[4 + chip_y] = ysend_blk(me)[...]
                yc_cps = [
                    rcopy(
                        ysend_blk(2 * jnp.bitwise_xor(chip_y, d) + core_y),
                        pad_y.at[4 + chip_y],
                        2 * jnp.bitwise_xor(chip_y, d) + core_y,
                        "y",
                    )
                    for d in (1, 2, 3)
                ]
                for cp in yc_cps:
                    cp.start()
                for cp in yc_cps:
                    cp.wait_send()
                for cp in yc_cps:
                    cp.wait_recv()
        elif vy >= 1:
            own_slice = pl.ds(pl.multiple_of(me * Rc, Rc), Rc)
            if vy >= 2:

                @pl.when(my_live)
                def _():
                    pad_y[me] = S["ysend"][own_slice]

            else:
                ly_cp = pltpu.make_async_copy(
                    S["ysend"].at[own_slice], pad_y.at[me], tsem(_TS_LOCAL)
                )
                pl.when(my_live)(ly_cp.start)
            if not y_block:
                for d in range(1, c.tp):
                    peer = coll._peer(d)
                    pl.when(peer * Rc < n_live)(y_block_cp(peer).start)
            for d in range(1, c.tp):
                peer = coll._peer(d)
                pl.when(peer * Rc < n_live)(y_block_cp(peer).wait_send)
            if vy == 1:
                pl.when(my_live)(ly_cp.wait)

            @pl.when(my_live)
            def _():
                for d in range(1, c.tp):
                    y_block_cp(coll._peer(d)).wait_recv()

        else:
            # every peer's rows for this device's live rows: K slots of each, landed
            row_msg = pltpu.make_async_remote_copy(
                src_ref=pad_y.at[0, pl.ds(0, 1)],
                dst_ref=pad_y.at[me, pl.ds(0, 1)],
                send_sem=S["tc_ssems"].at[tc_smap["y"]],
                recv_sem=S["tc_rsems"].at[tc_rmap["y"]],
                device_id=(me,),
            )

            def recv_row(_, carry):
                row_msg.wait_recv()
                return carry

            jax.lax.fori_loop(
                0,
                (c.tp if q_idx else c.tp - 1) * jnp.clip(n_live - me * Rc, 0, Rc),
                recv_row,
                0,
            )

    with _scope("combine"):
        pw = smr[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
        # this device's routing weights: the 8-row-aligned window around its
        # rows
        win = owner_window_rows(c)
        if win == Rc:
            w0, w_off = pl.multiple_of(me * Rc, 8), 0
        else:
            w0 = pl.multiple_of(jnp.minimum((me * Rc) // 8 * 8, R - win), 8)
            w_off = me * Rc - w0
        wmine = S["wsel_v"][pl.ds(w0, win), pl.ds(0, K)]  # [win, K]
        wk = jnp.concatenate(
            [wmine[:, k : k + 1][:, :, None] for k in range(K)], axis=1
        )
        S["wk_s"][...] = jnp.broadcast_to(wk, (win, K, LANE))
        if vy >= 2:
            y_slabs = (
                (4, 5, 6)
                if (use_pair_y and c.tiled_pair_y >= 2)
                else ((4, 5, 6, 7) if use_pair_y else tuple(range(c.tp)))
            )
            if c.combine_loop:

                def combine_vmem_row(r, carry_):
                    ysum = _unpack_y(pad_y.at[y_slabs[0], r][...], c)
                    for d in y_slabs[1:]:
                        ysum = ysum + _unpack_y(pad_y.at[d, r][...], c)
                    ysum = (
                        jnp.where(me * Rc + r < n_live, ysum, 0.0)
                        .astype(bf16)
                        .astype(f32)
                    )
                    ysc = ysum * pw
                    nrm = jax.lax.rsqrt(
                        jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
                    )
                    ynorm = (ysc * nrm).astype(bf16).astype(f32)
                    u = jnp.sum(
                        ynorm * S["wk_s"][w_off + r, :, 0:1],
                        axis=0,
                        keepdims=True,
                    )
                    S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                    return carry_

                jax.lax.fori_loop(0, Rc, combine_vmem_row, 0)
            else:
                ysum = _unpack_y(pad_y[y_slabs[0]], c)  # [Rc, K, He]
                for d in y_slabs[1:]:
                    ysum = ysum + _unpack_y(pad_y[d], c)
                r_live = (
                    me * Rc + jax.lax.broadcasted_iota(jnp.int32, (Rc, 1, 1), 0)
                ) < n_live
                ysum = jnp.where(r_live, ysum, 0.0).astype(bf16).astype(f32)
                ysc = ysum * pw[None]
                nrm = jax.lax.rsqrt(
                    jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
                )
                ynorm = (ysc * nrm).astype(bf16).astype(f32)
                wk_rc = (
                    S["wk_s"][:, :, 0:1]
                    if win == Rc
                    else jnp.concatenate(
                        [S["wk_s"][w_off + r, :, 0:1][None] for r in range(Rc)],
                        axis=0,
                    )
                )
                u = jnp.sum(ynorm * wk_rc, axis=1)  # [Rc, He]
                S["u_send"][...] = _pack_bf16x2(u)
        else:
            # pad_y rows stream through a ring of NB buffers of YP parts per row (tp / YP
            # sources each), NB - 1 parts in flight: one row ahead left the DMA latency
            # exposed (y_row_wait 25.8 of combine's 40 us/layer at 256 rows of 8)
            YP, NB = yrow_ring(c)
            hs = c.tp // YP
            n_units = Rc * YP
            assert not c.vmem_trim or (
                Rc <= Y_XFULL_ROW0 and Y_XFULL_ROW0 + NB * hs * K <= R
            )

            def ybuf(b, d):
                """Source d's [K, yw] slot of ring buffer b: under vmem_trim in xfull rows
                past u_send (the expert inputs are dead during combine)."""
                if not c.vmem_trim:
                    return S["yrow"].at[b, d, 0]
                r0 = Y_XFULL_ROW0 + (b * hs + d) * K
                return S["xfull"].at[pl.ds(pl.multiple_of(r0, 8), K), pl.ds(0, yw)]

            def ycopies(u, b):
                r, part = u // YP, u % YP
                return [
                    pltpu.make_async_copy(
                        S["pad_y"].at[part * hs + d, r],
                        ybuf(b, d),
                        S["yrow_sems"].at[b],
                    )
                    for d in range(hs)
                ]

            def ystart(u):
                for cp in ycopies(u, u % NB):
                    cp.start()

            def ywait(u):
                with _scope("y_row_wait"):
                    for cp in ycopies(u, u % NB):
                        cp.wait()

            for u0 in range(NB - 1):
                pl.when(my_live & (u0 < n_units))(functools.partial(ystart, u0))

            def combine_row(r, carry):
                ysum = None
                for part in range(YP):
                    u = r * YP + part
                    pl.when(my_live)(functools.partial(ywait, u))
                    pl.when(my_live & (u + NB - 1 < n_units))(
                        functools.partial(ystart, u + NB - 1)
                    )
                    for d in range(hs):
                        y_d = _unpack_y(ybuf(u % NB, d)[...], c)  # [K, He]
                        ysum = y_d if ysum is None else ysum + y_d
                # rows past n_live received nothing (stale HBM): keep them zero
                ysum = (
                    jnp.where(my_live & (me * Rc + r < n_live), ysum, 0.0)
                    .astype(bf16)
                    .astype(f32)
                )
                ysc = ysum * pw
                nrm = jax.lax.rsqrt(
                    jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
                )
                ynorm = (ysc * nrm).astype(bf16).astype(f32)
                u = jnp.sum(ynorm * S["wk_s"][w_off + r, :, 0:1], axis=0, keepdims=True)
                S["u_send"][pl.ds(r, 1), :] = _pack_bf16x2(u)
                if pipe_u:
                    pl.when(r % 8 == 7)(functools.partial(u_start, r // 8))
                return carry

            with _scope("combine_rows"):
                jax.lax.fori_loop(0, Rc, combine_row, 0)

    if vpads:
        with _scope("u_ag"):
            if pipe_u:
                u_finish()
                padu = S["pad_u"]
            else:
                padu = coll.all_gather(
                    "u", S["u_send"], me, live=lambda m: m * Rc < n_live
                )
            if c.combine_loop and R % 16 == 0:

                def _store_u16(r0, raw16):
                    r_ok16 = (
                        r0 + jax.lax.broadcasted_iota(jnp.int32, (16, 1), 0)
                    ) < n_live
                    S["lhs"][pl.ds(r0, 16), pl.ds(0, He)] = jnp.where(
                        r_ok16, _unpack_bf16x2(raw16), 0.0
                    ).astype(bf16)

                if Rc % 16 == 0:

                    def u_ag_chunk16(ch, carry_):
                        r0 = pl.multiple_of(ch * 16, 16)
                        d = (ch * 16) // Rc
                        sub = pl.multiple_of((ch * 16) % Rc, 16)
                        _store_u16(r0, padu.at[d][pl.ds(sub, 16)])
                        return carry_

                    jax.lax.fori_loop(0, R // 16, u_ag_chunk16, 0)
                elif Rc == 8:

                    def u_ag_pair8(ch, carry_):
                        r0 = pl.multiple_of(ch * 16, 16)
                        raw16 = jnp.concatenate(
                            [padu.at[2 * ch][...], padu.at[2 * ch + 1][...]],
                            axis=0,
                        )
                        _store_u16(r0, raw16)
                        return carry_

                    jax.lax.fori_loop(0, R // 16, u_ag_pair8, 0)
                else:
                    S["xfull"][...] = jnp.concatenate(
                        [padu[d] for d in range(c.tp)], axis=0
                    )

                    def u_ag_xfull16(ch, carry_):
                        r0 = pl.multiple_of(ch * 16, 16)
                        _store_u16(r0, S["xfull"][pl.ds(r0, 16)])
                        return carry_

                    jax.lax.fori_loop(0, R // 16, u_ag_xfull16, 0)
            else:
                upk = jnp.concatenate([padu[d] for d in range(c.tp)], axis=0)
                r_ok = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0) < n_live
                S["lhs"][:, pl.ds(0, He)] = jnp.where(
                    r_ok, _unpack_bf16x2(upk), 0.0
                ).astype(bf16)
    else:
        # u all-gather, pipelined by owner: the unpool pass loads a row tile once
        # its owners' rows have landed (each owner signals its own semaphore)
        u_osems = S["uag_osems"]
        # an owner block off an 8-row boundary cannot be one HBM DMA window: send
        # it row by row (the waits below take one DMA each)
        if Rc % 8 == 0:
            u_parts = [
                (
                    S["u_send"],
                    S["pad_u"].at[pl.ds(pl.multiple_of(me * Rc, 8), Rc)],
                )
            ]
        else:
            u_parts = [
                (
                    S["u_send"].at[pl.ds(r, 1)],
                    S["pad_u"].at[pl.ds(me * Rc + r, 1)],
                )
                for r in range(Rc)
            ]

        @pl.when(my_live)
        def _():
            for src, dst in u_parts:
                pltpu.make_async_copy(src, dst, u_osems.at[me]).start()
                for d in range(1, c.tp):
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=S["tc_ssems"].at[_TC["u"]],
                        recv_sem=u_osems.at[me],
                        device_id=(coll._peer(d),),
                    ).start()

        def u_ready(t):

            def wait_owner(o):
                def wait():
                    src, dst = u_parts[0]
                    for _ in u_parts:
                        pltpu.make_async_copy(src, dst, u_osems.at[o]).wait()

                pl.when(o * Rc < n_live)(wait)

            if RT % Rc == 0:
                for j in range(RT // Rc):
                    wait_owner(t * (RT // Rc) + j)
            elif Rc % RT == 0:
                pl.when((t * RT) % Rc == 0)(lambda: wait_owner(t * RT // Rc))
            else:
                # owners whose rows tile t is the first to touch
                first = jnp.where(t == 0, 0, (t * RT - 1) // Rc + 1)
                last = ((t + 1) * RT - 1) // Rc
                for j in range(RT // Rc + 2):
                    pl.when(first + j <= last)(lambda j=j: wait_owner(first + j))

        def unpool_tile(rows, t, v):
            u = jnp.where(tile_live(t), _unpack_bf16x2(v[0][...]), 0.0)
            S["lhs"][rows, pl.ds(0, He)] = u.astype(bf16)
            return []

        with _scope("u_ag"):
            tile_pass(
                [(lambda rows: S["pad_u"].at[rows], S["utile"])],
                unpool_tile,
                ready=[u_ready],
            )

        @pl.when(my_live)
        def _():
            for d in range(1, c.tp):
                for src, dst in u_parts:
                    rcopy(src, dst, coll._peer(d), "u").wait_send()

    with _scope("unpool_mv"):
        mv_tiled(l, "wunpool", nxt, smr)

    if vpads:
        if c.combine_loop and R > 64:

            def _tile(rs, t):
                S["f_send"][rs] = _pack_bf16x2(
                    S["acc"][rs, pl.ds(0, Hl)].astype(bf16).astype(f32)
                )
                if c.pipe_ag:
                    hag_start("f", S["f_send"], S["pad_f"], t, S["fag_tsems"])

            vpads_tiles(_tile)
        else:
            assert not c.pipe_ag
            S["f_send"][...] = _pack_bf16x2(
                S["acc"][:, pl.ds(0, Hl)].astype(bf16).astype(f32)
            )
        if c.pipe_ag:
            pass  # sent per tile above
        elif c.fold_f and nxt is not None:
            coll.all_gather_blocks_start("f", S["f_send"], tp_rank, S["f_rsems"])
        else:
            with _scope("f_ag"):
                coll.all_gather("f", S["f_send"], tp_rank)
    else:

        def f_tile(rows, t):
            f = S["acc"][rows, pl.ds(0, Hl)].astype(bf16).astype(f32)
            S["f_send"][rows] = _pack_bf16x2(f)
            # pipelined all-gather: the ffn residual pass waits per row tile
            ag_tile_start("f", S["f_send"], S["pad_f"], tp_rank, t, S["fag_tsems"])

        for_tiles(f_tile)
