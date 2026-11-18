# SPDX-License-Identifier: Apache-2.0
"""Untiled MoE of the decode megakernel: routing, the expert loop and the combine."""

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
from sglang_tpu.muse_megakernel.common.control import for_layer_experts
from sglang_tpu.muse_megakernel.common.dots import (
    _fp4_blockdiag_dot,
    _fp4_dot,
    _fp4_qblock_ffn,
    _fp8_dot,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _rms,
    _topk_by_rank,
    _topk_iter,
    _topk_xpose,
)
from sglang_tpu.muse_megakernel.common.packing import (
    _pack_bf16x2,
    _pack_y,
    _unpack_bf16x2,
    _unpack_y,
)
from sglang_tpu.muse_megakernel.common.plan import (
    compact_buckets,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx


def unique_experts(k: KernelCtx, ids_list, l):
    """Distinct experts of the live rows' slots, in first-use order, into ``uniq`` (SMEM),

    with each expert's (row, slot) pairs grouped in ``perm``.
    """
    K = k.K
    Rc = k.Rc
    S = k.S
    blk_owner = k.blk_owner
    c = k.c
    n_live = k.n_live
    route_tiled = k.route_tiled
    unique_experts_vec = k.unique_experts_vec
    y_block = k.y_block
    if c.vec_route:
        # row-tiled steps route from ids_v, which their top-k fills tile by tile
        return route_tiled() if c.row_tile else unique_experts_vec(ids_list)
    S["ids_v"][:, 0:K] = jnp.concatenate(ids_list, axis=1)
    cp = pltpu.make_async_copy(S["ids_v"], S["ids_s"], S["ids_sem"].at[0])
    cp.start()
    cp.wait()
    stamp = l + 1  # "seen" is cleared once per call; one stamp per layer

    def count(slot, n):
        e = S["ids_s"][slot // K, slot % K]
        new = S["seen"][e] != stamp

        @pl.when(new)
        def _():
            S["seen"][e] = stamp
            S["uniq"][n] = e
            S["ex_of"][e] = n
            S["ex_cnt"][n] = 0

        u = S["ex_of"][e]
        S["ex_cnt"][u] = S["ex_cnt"][u] + 1
        return n + new.astype(jnp.int32)

    if y_block:
        # rows owner block by owner block (own rows last): uniq comes out grouped by
        # the block that first uses each expert, and blk_end[b] closes block b's group
        n_u = 0
        for b in range(c.tp):
            row0 = blk_owner(b) * Rc
            n_u = jax.lax.fori_loop(
                row0 * K,
                (row0 + jnp.clip(n_live - row0, 0, Rc)) * K,
                count,
                n_u,
            )
            S["blk_end"][b] = n_u
    else:
        n_u = jax.lax.fori_loop(0, n_live * K, count, 0)

    def offsets(u, off):
        S["ex_off"][u] = off
        S["ex_pos"][u] = off
        return off + S["ex_cnt"][u]

    jax.lax.fori_loop(0, n_u, offsets, 0)

    def place(slot, carry):
        u = S["ex_of"][S["ids_s"][slot // K, slot % K]]
        p = S["ex_pos"][u]
        S["perm"][p] = slot
        S["ex_pos"][u] = p + 1
        return carry

    jax.lax.fori_loop(0, n_live * K, place, 0)
    return n_u


def moe(k: KernelCtx, l, ffn_in32, sm, kind, nxt, pool_router=None):
    """pool_router(): (pooled, logits) from a fold, instead of from ffn_in32.

    Returns the ffn branch, or with fold_f and a next layer None: its
    all-gather is in flight (the next layer's fold_qkvg consumes it).
    """
    F = k.F
    He = k.He
    Hl = k.Hl
    Il = k.Il
    K = k.K
    R = k.R
    Rc = k.Rc
    S = k.S
    blk_owner = k.blk_owner
    c = k.c
    coll = k.coll
    group_lo = k.group_lo
    kind_layers = k.kind_layers
    layer_ex = k.layer_ex
    me = k.me
    n_live = k.n_live
    rget = k.rget
    row_live = k.row_live
    single_row = k.single_row
    tp_rank = k.tp_rank
    unique_experts = k.unique_experts
    ws = k.ws
    y_block = k.y_block
    yw = k.yw
    Ep = c.n_experts_padded
    ws.hold = ws.ranges["wunpool"][0]
    if pool_router is not None:
        pooled, logits = pool_router()
        S["pool_send"][...] = _pack_bf16x2(pooled)
        pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
    else:
        ffn_in = ffn_in32.astype(jnp.bfloat16)
        # pooler (column-parallel by TP rank) first: its gather flies during routing
        with _scope("pool_mv"):
            pooled = (
                ws.mv(ffn_in, l, "wpool", nxt, sm=sm)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        S["pool_send"][...] = _pack_bf16x2(pooled)
        pool_wait = coll.all_gather_start("pool", S["pool_send"], tp_rank)
        # router: one bf16 pass (as torch_tpu's DEFAULT-precision fp32 F.linear)
        with _scope("router_mv"):
            logits = ws.mv(ffn_in32, l, "wrouter", nxt) * c.router_mult
    with _scope("topk"):
        sc = jax.nn.sigmoid(logits)
        # padded lanes carry a -1e30 bias, so they are never selected
        bias = sm[SMALL["expert_bias"] : SMALL["expert_bias"] + 1, :Ep]
        if c.topk_xpose and c.topk_iter and 64 <= R <= LANE and R % 8 == 0:
            ids_full, wsel_full = _topk_xpose(sc, bias, K, c.router_eps, c.route_scale)
            ids_list = [ids_full[:, j : j + 1] for j in range(K)]
            wsel = wsel_full[:, :K]
        else:
            if c.topk_iter:
                ids_list, picks = _topk_iter(sc + bias, K)
            else:
                ids_list, picks = _topk_by_rank(sc + bias, K, n_live, S["rank"])
            wsel = jnp.concatenate(
                [jnp.sum(jnp.where(p, sc, 0.0), axis=1, keepdims=True) for p in picks],
                axis=1,
            )
            wsel = (
                wsel
                / (jnp.sum(wsel, axis=1, keepdims=True) + c.router_eps)
                * c.route_scale
            )  # [R, K]
    with _scope("unique_experts"):
        n_u = unique_experts(ids_list, l)

    def kind_fns(kind):
        """(issue the first expert DMAs, run the expert loop) for one expert kind."""
        if kind == "fp4":
            nslot, W13, W2 = c.expert_depth + 1, S["w13"], S["w2"]
        else:
            nslot = (c.expert_depth + 1) // 2
            W13 = S["w13"].reshape(nslot, He // 4, 2 * Il)
            W2 = S["w2"].reshape(nslot, Il // 4, He)
        depth = nslot - 1
        # scale slabs: e4m3 block scales (fp4) or bf16 (fp4 folded, fp8 channel)
        S13 = S["w13s"]
        S2 = S["w2s"]

        def copies(i, slot, e=None):
            e = rget("uniq", i) if e is None else e
            out = []
            pack = FP4_PACK if kind == "fp4" else 4
            kw13, kw2 = He // pack, Il // pack
            for j, (dst, sem, rows_, cols, is_scale) in enumerate(
                (
                    (W13, S["w13_sems"], kw13, 2 * Il, False),
                    (
                        S13,
                        S["w13s_sems"],
                        He // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
                        2 * Il,
                        True,
                    ),
                    (W2, S["w2_sems"], kw2, He, False),
                    (
                        S2,
                        S["w2s_sems"],
                        Il // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
                        He,
                        True,
                    ),
                )
            ):
                out.append((j, e, dst, sem, rows_, cols, is_scale))
            return out

        def expert_dmas(refs, i, slot, e=None, off=None):
            """The four slab DMAs of distinct expert ``i`` into ``slot``, from one layer's refs

            (``off``: the layer's index in stacked refs).
            """
            for j, e, dst, sem, rows_, cols, is_scale in copies(i, slot, e):
                src = refs[j] if off is None else refs[j].at[off]
                src = src if is_scale else src.bitcast(jnp.uint32)
                pltpu.make_async_copy(
                    src.at[e, pl.ds(0, rows_), pl.ds(0, cols)],
                    dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                    sem.at[slot],
                ).start()

        for_layer = functools.partial(
            for_layer_experts, l, kind, layer_ex, kind_layers, group_lo
        )

        def start_expert(i, slot):
            # expert id read before the walk: one SMEM load instead of one per leaf
            e = rget("uniq", i)
            for_layer(lambda refs, off: expert_dmas(refs, i, slot, e, off))

        def wait_expert(slot, parts=(0, 1, 2, 3)):
            pack = FP4_PACK if kind == "fp4" else 4
            kw13, kw2 = He // pack, Il // pack
            tl = l if isinstance(l, int) else kind_layers[kind][0]
            trefs, toff = layer_ex[tl]
            tmpl = [r if toff is None else r.at[toff] for r in trefs]
            for dst, sem, rows_, cols, j in (
                (W13, S["w13_sems"], kw13, 2 * Il, 0),
                (
                    S13,
                    S["w13s_sems"],
                    He // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
                    2 * Il,
                    1,
                ),
                (W2, S["w2_sems"], kw2, He, 2),
                (
                    S2,
                    S["w2s_sems"],
                    Il // (c.qblock or FP4_GROUP) if kind == "fp4" else 2,
                    He,
                    3,
                ),
            ):
                if j not in parts:
                    continue
                src = tmpl[j] if j in (1, 3) else tmpl[j].bitcast(jnp.uint32)
                pltpu.make_async_copy(
                    src.at[0, pl.ds(0, rows_), pl.ds(0, cols)],
                    dst.at[slot, pl.ds(0, rows_), pl.ds(0, cols)],
                    sem.at[slot],
                ).wait()

        def issue_experts(lo, hi):
            """Distinct experts [lo, hi) in one walk; the leaf is a small loop (code size:

            one set of descriptors per leaf).
            """

            def leaf(refs, off):
                def one(t, carry):
                    expert_dmas(refs, t, t % nslot, off=off)
                    return carry

                jax.lax.fori_loop(lo, hi, one, 0)

            for_layer(leaf)

        # pairs: even iterations issue two experts per layer walk (prefetch 1-2 deep, 2-3
        # right after the issue): half the walks
        pairs = c.ex_issue_pairs and nslot >= 4
        pdepth = nslot - 2 if pairs else depth

        def do_prologue():
            issue_experts(0, jnp.minimum(pdepth, n_u))

        def ebody(i, carry, part=None):
            slot = i % nslot

            if pairs:
                pl.when((i % 2 == 0) & (i + pdepth < n_u))(
                    lambda: issue_experts(i + pdepth, jnp.minimum(i + pdepth + 2, n_u))
                )
            else:

                @pl.when(i + depth < n_u)
                def _():
                    start_expert(i + depth, (i + depth) % nslot)

            # split: the w2 slabs land behind w13 (FIFO), so bd_w13 starts before they do
            with _scope("ex_wait"):
                wait_expert(slot, (0, 1))

            def w2_wait():
                with _scope("ex_wait"):
                    wait_expert(slot, (2, 3))

            # QBLOCK fp4 arms wait for w2 up front (a mid-arm DMA wait is a scheduling
            # barrier that drains the MXU) and run _fp4_qblock_ffn
            q_ffn = bool(c.qblock and kind == "fp4")

            def qb_ffn(xc):
                return _fp4_qblock_ffn(
                    xc,
                    W13.at[slot],
                    S["w13s"].at[slot],
                    W2.at[slot],
                    S["w2s"].at[slot],
                    He,
                    Il,
                    c.qblock,
                )

            e = rget("uniq", i)

            def gather_x_base(ps_):
                if c.xfull_packed:
                    packed = jnp.concatenate(
                        [S["xfull"][pl.ds(p // K, 1), :] for p in ps_], axis=0
                    )
                    return _unpack_bf16x2(packed).astype(jnp.bfloat16)
                return jnp.concatenate(
                    [S["xfull"][pl.ds(p // K, 1), :] for p in ps_], axis=0
                ).astype(jnp.bfloat16)

            def full_expert():
                with _scope("full_expert"):
                    full_expert_()

            def busy_bucket(nrows):
                off, cnt = rget("ex_off", i), rget("ex_cnt", i)
                ps_ = [
                    rget("perm", off + jnp.minimum(j, cnt - 1)) for j in range(nrows)
                ]
                with _scope("ex_gather"):
                    xc = gather_x_base(ps_)
                if q_ffn:
                    w2_wait()
                    yp = _pack_y(qb_ffn(xc), c)
                else:
                    cw13 = (
                        64
                        if c.ex_opt and c.fp4_chunk_words == 128
                        else c.fp4_chunk_words
                    )
                    if kind == "fp4":
                        gu = _fp4_dot(
                            xc,
                            W13.at[slot],
                            S["w13s"].at[slot],
                            He,
                            2 * Il,
                            cw13,
                            fast_bf16=c.ex_opt,
                            qblock=c.qblock,
                            via_fp8=c.ex_via_fp8,
                            n_tile=c.bd_n_tile,
                        )
                    else:
                        gu = _fp8_dot(
                            xc,
                            W13.at[slot],
                            S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                            fp8_lhs=(c.ex_lhs_fp8 >= 1),
                        )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    w2_wait()
                    if kind == "fp4":
                        y = _fp4_dot(
                            h,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            min(Il // FP4_PACK, 64 if c.ex_opt else 16),
                            fast_bf16=c.ex_opt,
                            qblock=c.qblock,
                            via_fp8=c.ex_via_fp8,
                            n_tile=c.bd_n_tile,
                        )
                    else:
                        y = _fp8_dot(
                            h,
                            W2.at[slot],
                            S2[slot, 0:1, :].astype(jnp.float32),
                            Il,
                            He,
                            min(Il // 4, 128 if c.ex_opt else 32),
                            fp8_lhs=(c.ex_lhs_fp8 >= 2),
                        )
                    yp = _pack_y(y, c)
                with _scope("ex_scatter"):
                    for j, p in enumerate(ps_):
                        S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]

            def full_expert_():
                if q_ffn:
                    w2_wait()
                    y = qb_ffn(x_full)
                else:
                    cw13 = (
                        64
                        if c.ex_opt and c.fp4_chunk_words == 128
                        else c.fp4_chunk_words
                    )
                    if kind == "fp4":
                        gu = _fp4_dot(
                            x_full,
                            W13.at[slot],
                            S["w13s"].at[slot],
                            He,
                            2 * Il,
                            cw13,
                            fast_bf16=c.ex_opt,
                            qblock=c.qblock,
                            via_fp8=c.ex_via_fp8,
                            n_tile=c.bd_n_tile,
                        )
                    else:
                        gu = _fp8_dot(
                            x_full,
                            W13.at[slot],
                            S13[slot, 0:1, pl.ds(0, 2 * Il)].astype(jnp.float32),
                            He,
                            2 * Il,
                            c.fp4_chunk_words,
                            fp8_lhs=(c.ex_lhs_fp8 >= 1),
                        )
                    gate, up = gu[:, :Il], gu[:, Il:]
                    h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    w2_wait()
                    if kind == "fp4":
                        y = _fp4_dot(
                            h,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            min(Il // FP4_PACK, 64 if c.ex_opt else 16),
                            fast_bf16=c.ex_opt,
                            qblock=c.qblock,
                            via_fp8=c.ex_via_fp8,
                            n_tile=c.bd_n_tile,
                        )
                    else:
                        y = _fp8_dot(
                            h,
                            W2.at[slot],
                            S2[slot, 0:1, :].astype(jnp.float32),
                            Il,
                            He,
                            min(Il // 4, 128 if c.ex_opt else 32),
                            fp8_lhs=(c.ex_lhs_fp8 >= 2),
                        )
                # slot space: row r, slot k gets y[r] where ids[r, k] == e
                with _scope("ex_accum"):
                    S["ysend"][...] = jnp.where(
                        ids3 == e, _pack_y(y, c)[:, None, :], S["ysend"][...]
                    )

            def compact_expert(nrows):
                # one variant per row bucket: the expanded input and the scale work scale
                # with the rows; bucket rows past the expert's count repeat its last pair
                bd_kc = c.blockdiag_kc
                off, cnt = rget("ex_off", i), rget("ex_cnt", i)
                ps_ = [
                    rget("perm", off + jnp.minimum(j, cnt - 1)) for j in range(nrows)
                ]
                with _scope("ex_gather"):
                    xc = gather_x_base(ps_)
                if q_ffn:
                    w2_wait()
                    yp = _pack_y(qb_ffn(xc), c)
                else:
                    with _scope("bd_w13"):
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
                        )
                        gate, up = gu[:, :Il], gu[:, Il:]
                        h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                    w2_wait()
                    with _scope("bd_w2"):
                        yc = _fp4_blockdiag_dot(
                            h,
                            W2.at[slot],
                            S["w2s"].at[slot],
                            Il,
                            He,
                            bd_kc,
                            kperm=False,
                            qblock=c.qblock,
                            fp8_lhs=False,
                            n_tile=c.bd_n_tile,
                        )
                        yp = _pack_y(yc, c)
                with _scope("ex_scatter"):
                    # a padded row repeats the last pair's row, so its write repeats that
                    # pair's identical write: no branch per row
                    for j, p in enumerate(ps_):
                        S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]

            def bd_chunks():
                bd_kc = c.blockdiag_kc
                off, cnt = rget("ex_off", i), rget("ex_cnt", i)

                def bd_step(ch, carry_):
                    ps_ = [
                        rget("perm", off + jnp.minimum(ch * 8 + j, cnt - 1))
                        for j in range(8)
                    ]
                    with _scope("ex_gather"):
                        xc = gather_x_base(ps_)
                    if q_ffn:
                        pl.when(ch == 0)(w2_wait)
                        yp = _pack_y(qb_ffn(xc), c)
                    else:
                        with _scope("bd_w13"):
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
                            )
                            gate, up = gu[:, :Il], gu[:, Il:]
                            h = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
                        pl.when(ch == 0)(w2_wait)
                        with _scope("bd_w2"):
                            yc = _fp4_blockdiag_dot(
                                h,
                                W2.at[slot],
                                S["w2s"].at[slot],
                                Il,
                                He,
                                bd_kc,
                                kperm=False,
                                qblock=c.qblock,
                                fp8_lhs=False,
                                n_tile=c.bd_n_tile,
                            )
                            yp = _pack_y(yc, c)
                    with _scope("ex_scatter"):
                        for j, p in enumerate(ps_):
                            S["ysend"].at[p // K][pl.ds(p % K, 1), :] = yp[j : j + 1]
                    return carry_

                jax.lax.fori_loop(0, (cnt + 7) // 8, bd_step, 0)

            if kind == "fp4" and c.compact_rows:
                cnt = rget("ex_cnt", i)
                # (upper count bound, variant) per bucket, picked by a compare tree
                # an expert has at most one pair per row: a bucket of R rows covers all
                buckets = compact_buckets(c)
                arms = [(b, functools.partial(compact_expert, b)) for b in buckets]
                if buckets[-1] < R:
                    if c.ex_bd_all:
                        arms.append((None, bd_chunks))
                    else:
                        lo_arm = buckets[-1]
                        if lo_arm < R:
                            if c.ex_opt and R <= 48:
                                bb_list = (
                                    (8, 16, 24, 32) if c.ex_w2_stage else (16, 24, 32)
                                )
                                for bb in bb_list:
                                    if lo_arm < bb < R:
                                        arms.append(
                                            (
                                                bb,
                                                functools.partial(busy_bucket, bb),
                                            )
                                        )
                                arms.append((None, functools.partial(busy_bucket, R)))
                            else:
                                arms.append((None, full_expert))
                else:
                    arms[-1] = (None, arms[-1][1])
                if part == "few":
                    arms = [
                        a for a in arms if a[0] is not None and a[0] <= c.split_rows
                    ]
                    arms[-1] = (None, arms[-1][1])
                elif part == "many":
                    arms = [a for a in arms if a[0] is None or a[0] > c.split_rows]

                def pick(arms):
                    if len(arms) == 1:
                        arms[0][1]()
                        return
                    mid = len(arms) // 2
                    jax.lax.cond(
                        cnt <= arms[mid - 1][0],
                        lambda: pick(arms[:mid]),
                        lambda: pick(arms[mid:]),
                    )

                pick(arms)
            else:
                full_expert()
            if y_block:
                # carry: the next owner block to send. Block nb is final once its
                # group's last expert is done; an empty group fires with the one before
                # (at expert 0 if no group before it has an expert).
                def ready(nb):
                    end = rget("blk_end", jnp.minimum(nb, c.tp - 2))
                    return (nb < c.tp - 1) & (end <= i + 1)

                def send_block(nb):
                    owner = blk_owner(nb)
                    pl.when(owner * Rc < n_live)(
                        coll.block_copy("y", S["ysend"], owner, Rc).start
                    )
                    return nb + 1

                return jax.lax.while_loop(ready, send_block, carry)
            return carry

        def do_loop():
            with _scope("experts"):
                if c.split_rows and kind == "fp4":

                    def group(g_, carry):
                        lo = jnp.where(
                            g_ == 0, 0, rget("blk_end", jnp.maximum(g_ - 1, 0))
                        )
                        mid, hi = rget("blk_mid", g_), rget("blk_end", g_)
                        carry = jax.lax.fori_loop(
                            lo, mid, functools.partial(ebody, part="few"), carry
                        )
                        return jax.lax.fori_loop(
                            mid,
                            hi,
                            functools.partial(ebody, part="many"),
                            carry,
                        )

                    jax.lax.fori_loop(0, c.tp if y_block else 1, group, jnp.int32(0))
                else:
                    jax.lax.fori_loop(0, n_u, ebody, jnp.int32(0))

        return do_prologue, do_loop

    kinds_here = (
        ([c.expert_kinds[l]] if isinstance(l, int) else sorted(kind_layers))
        if kind is None
        else [kind]
    )
    fns = {k: kind_fns(k) for k in kinds_here}

    def by_kind(j):
        if len(kinds_here) == 1:
            fns[kinds_here[0]][j]()
            return
        is_fp8 = False
        for x in kind_layers["fp8"]:
            is_fp8 = jnp.logical_or(is_fp8, l == x)
        jax.lax.cond(is_fp8, fns["fp8"][j], fns["fp4"][j])

    by_kind(0)

    with _scope("pool_ag"):
        pad = pool_wait()
    with _scope("ex_setup"):
        x_full = jnp.concatenate(
            [_unpack_bf16x2(pad[d]) for d in range(c.tp)], axis=1
        )  # [R, He]
        x_full = _rms(
            x_full,
            c.rms_eps,
            sm[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He],
        ).astype(jnp.bfloat16)
        ids3 = jnp.concatenate([i_[:, :, None] for i_ in ids_list], axis=1)  # [R, K, 1]
        # every live (row, slot) gets exactly one expert output, written bf16-packed;
        # rows past n_live stay zero
        S["ysend"][...] = jnp.zeros((R, K, yw), jnp.uint32)
        if c.xfull_packed:
            S["xfull"][...] = _pack_bf16x2(x_full.astype(jnp.float32))
        else:
            S["xfull"][...] = x_full.astype(jnp.float32)

    by_kind(1)
    # flushing inside the expert loop (once its prefetch ends) measured slower
    ws.flush()
    # ---- token-sharded reduce of the per-rank partials, then norm + combine
    with _scope("y_rs"):
        pad = coll.reduce_scatter_rows("y", S["ysend"], Rc, n_live, started=y_block)
    with _scope("combine"):
        ysum = _unpack_y(pad[0], c)
        for d in range(1, c.tp):
            ysum = ysum + _unpack_y(pad[d], c)  # [Rc, K, He]
        # rows past n_live received nothing: keep them finite
        ysum = (
            jnp.where(me * Rc < n_live, ysum, 0.0)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        pw = sm[SMALL["post_expert_w"] : SMALL["post_expert_w"] + 1, :He]
        ysc = ysum * pw[None]
        nrm = jax.lax.rsqrt(
            jnp.mean(jnp.square(ysc), axis=-1, keepdims=True) + c.post_eps
        )
        ynorm = (ysc * nrm).astype(jnp.bfloat16).astype(jnp.float32)
        # this device's rows' routing weights
        r0 = pl.multiple_of(me * Rc, Rc) if c.tp > 1 else 0
        wk = jnp.concatenate(
            [wsel[:, k : k + 1][:, :, None] for k in range(K)], axis=1
        )  # [R, K, 1]
        S["wk_s"][...] = jnp.broadcast_to(wk, (R, K, LANE))
        wmine = S["wk_s"][pl.ds(r0, Rc), :, 0:1]  # [Rc, K, 1]
        u = jnp.sum(ynorm * wmine, axis=1)  # [Rc, He]
        S["u_send"][...] = _pack_bf16x2(u)
    with _scope("u_ag"):
        padu = coll.all_gather("u", S["u_send"], me, live=lambda m: m * Rc < n_live)
    u_full = jnp.concatenate(
        [_unpack_bf16x2(padu[d]) for d in range(c.tp)], axis=0
    )  # [R, He]
    u_full = jnp.where(row_live, u_full, 0.0)
    with _scope("unpool_mv"):
        f = (
            ws.mv(u_full, l, "wunpool", nxt, sm=sm)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )

    def gather_rows(f):
        S["f_send"][...] = _pack_bf16x2(f)
        padf = coll.all_gather("f", S["f_send"], tp_rank)
        return jnp.concatenate(
            [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
        )  # [R, H]

    def gather_row(f):
        S["row1_send"][:, 0 : Hl // 2] = _pack_bf16x2(f[0:F, :])
        padf = coll.gather_few("f1", S["row1_send"].at[:, pl.ds(0, Hl // 2)], tp_rank)
        return jnp.concatenate(
            [_unpack_bf16x2(padf[d]) for d in range(c.tp)], axis=1
        )  # [F, H]

    if c.fold_f and nxt is not None:
        S["f_send"][...] = _pack_bf16x2(f)
        coll.all_gather_blocks_start("f", S["f_send"], tp_rank, S["f_rsems"])
        return None
    with _scope("f_ag"):
        return single_row(f, gather_rows, gather_row)
