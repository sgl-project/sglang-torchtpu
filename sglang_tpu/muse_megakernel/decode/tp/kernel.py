# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder Pallas megakernel for Muse Spark decode and DFlash verify on TPU.

One grid-less ``pallas_call`` per device per step takes the embedded rows and
runs all
decoder layers (paged attention, router, experts with the post-expert norm,
pooler/unpooler) and the final norm. Every tensor-parallel reduction is an
in-kernel
one-hop remote DMA. The LM head, logits and sampling stay in sglang.

It reads the weights sglang already loaded, in their serving layouts:

* experts: the fused-RS expert-TP layout (``w13 [E, H_e, 2*I_local]`` fp4 or
fp8,
    ``w2 [E, I_local, H_e]``); per-layer refs, addressed by a branch tree on the
    layer
    index so the layer loop can stay rolled;
* KV: the paged pools of the hybrid SWA allocator, one ref per layer, read in
place: fused
    ``[pages, page_size, kv_heads_x2/2, 2, 128]`` bf16, or SEQ_ALONG_LANE
    ``[pages, 2 * kv_heads, head_dim // pk, pk, page_size]`` bf16 / fp8 (tokens on
    lanes);
* attention / pooler / unpooler / router: stacked K-major copies made at load
time
    (``ops.MuseMegakernel.from_model``), because the matvec streams ``[K, N]``
    tiles faster
    than the checkpoint's ``[N, K]``; fp8 with per-output-channel scales, except
    the router.

Rows are ``num_requests * rows_per_request`` tokens padded to a multiple of the
TP size:
one per request at decode, a DFlash verify block per request at target-verify.
Each row
carries its own position. The step's own K/V rows join attention from VMEM,
causal within
a verify block, and come back as ``newkv`` for ``ops`` to scatter into the
pools.

The expert phase follows the fused-RS expert-TP contract: each rank evaluates
its
intermediate slice of every selected expert for all rows, the partial outputs
are
reduce-scattered by token, summed in f32, rounded to bf16, normalised by the
post-expert
norm, weighted and summed. Collective rounds per layer: o_proj reduce-scatter
and
all-gather, pooler all-gather, expert reduce-scatter, combined-output
all-gather, unpooler
all-gather.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common.coll import (
    Coll,
)
from sglang_tpu.muse_megakernel.common.config import (
    MKConfig,
)
from sglang_tpu.muse_megakernel.common.consts import (
    _COLLECTIVE_ID,
    LANE,
    SMALL,
)
from sglang_tpu.muse_megakernel.common.control import (
    _dispatch,
    _sel,
)
from sglang_tpu.muse_megakernel.common.hooks import (
    EPHook,
)
from sglang_tpu.muse_megakernel.common.numerics import (
    _rms,
)
from sglang_tpu.muse_megakernel.common.packing import (
    _pack_bf16x2,
    _unpack_bf16x2,
    y_words,
)
from sglang_tpu.muse_megakernel.common.plan import (
    _fold_g,
    dense_tiles,
    expert_units,
    few_rows,
    fold_groups,
    n_expert_refs,
    route_blocks,
)
from sglang_tpu.muse_megakernel.common.route import (
    _route_table,
)
from sglang_tpu.muse_megakernel.common.scratch import (
    _TC,
    _TC_VPADS_R,
    _TC_VPADS_S,
    _TS_AUX,
    _TS_LOCAL,
    WEIGHT_NAMES,
    _alias_oag_into_f,
    _alias_pads_into_y,
    _alias_pool_ysend,
    _alias_tiled_ids_kv,
    _alias_tiled_obuf_ysend,
    _alias_tiled_padu_attn,
    _alias_tiled_xfull_attn,
    _alias_tiled_xfull_q,
    _alias_u_xfull,
    scratch_plan,
)
from sglang_tpu.muse_megakernel.common.stream import (
    _WeightStream,
)
from sglang_tpu.muse_megakernel.common.tables import (
    TABLE_ALIGN,
    _AttnGeom,
    pad_table,
    swa_window_entries,
    table_entries_padded,
)
from sglang_tpu.muse_megakernel.common.trace import (
    _scope,
)
from sglang_tpu.muse_megakernel.decode.tp import attn as untiled_attn
from sglang_tpu.muse_megakernel.decode.tp import moe as untiled_moe
from sglang_tpu.muse_megakernel.decode.tp import tiled_attn, tiled_layer, tiled_moe
from sglang_tpu.muse_megakernel.decode.tp.ctx import KernelCtx

_PIPE_AG = {"o_ag": 0, "f": 1, "u": 2}


def build_kernel(cfg: MKConfig, axis: str = "model", ep: EPHook | None = None):
    """Per-device body: ``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)``.

    ``meta`` int32 SMEM: [n_live_rows, tp_rank]. ``emb`` [R, H] bf16 is the token
    embedding (gathered and reduced by sglang). ``experts`` is 4 refs per layer
    (w13, w13_scale, w2, w2_scale) in layer order. Outputs: hidden ``[R, H]``
    bf16, aux
    ``[R, ncap*H]`` bf16 (or ``[R, 128]`` dummy), then the aliased KV refs.
    """
    c = cfg.validate()
    R, K, H = c.rows, c.top_k, c.hidden
    hd, hq, hkv, g = c.head_dim, c.q_heads, c.kv_heads, c.g
    ph = c.pool_kv_heads
    He, Il, Hel, Hl = (
        c.expert_hidden,
        c.inter_local,
        c.expert_hidden_local,
        c.hidden_local,
    )
    L = c.n_layers
    q = c.rows_per_req
    ps = c.page_size
    pd = c.pool_padded_head_dim
    # HBM scratch (row-tiled steps) is passed in as extra pallas_call outputs: Mosaic only
    # allocates VMEM, SMEM and semaphore scratch
    plan = scratch_plan(c, ep)
    hbm_scratch = [(n, s) for n, s in plan if _is_hbm(s)]
    scratch = [(n, s) for n, s in plan if not _is_hbm(s)]
    hnames = [n for n, _ in hbm_scratch]
    snames = [n for n, _ in scratch]
    nw = len(WEIGHT_NAMES)
    ncap = len(c.capture_layers)
    fp4_layers = [l for l in range(L) if c.expert_kinds[l] == "fp4"]
    # rolled range: the contiguous block of fp4 layers
    roll_lo = fp4_layers[0] if fp4_layers else None
    roll_hi = fp4_layers[-1] + 1 if fp4_layers else None
    if fp4_layers and fp4_layers != list(range(roll_lo, roll_hi)):
        raise NotImplementedError("fp4 expert layers must be one contiguous range")
    gl = tuple(sorted(c.global_layers))
    kind_layers = {}
    for _l, _k in enumerate(c.expert_kinds):
        kind_layers.setdefault(_k, []).append(_l)

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, emb_ref, *rest):
        k = KernelCtx()
        k.H = H
        k.He = He
        k.Hel = Hel
        k.Hl = Hl
        k.Il = Il
        k.K = K
        k.L = L
        k.R = R
        k.c = c
        k.ep = ep
        k.g = g
        k.hd = hd
        k.hkv = hkv
        k.hq = hq
        k.kind_layers = kind_layers
        k.ncap = ncap
        k.pd = pd
        k.ph = ph
        k.ps = ps
        k.q = q
        k.roll_hi = roll_hi
        k.roll_lo = roll_lo
        k.pos_ref = pos_ref
        k.pf_ref = pf_ref
        k.emb_ref = emb_ref
        W = dict(zip(WEIGHT_NAMES, rest[:nw]))
        k.W = W
        rest = rest[nw:]
        NE = n_expert_refs(c)
        EX = rest[:NE]
        rest = rest[NE:]
        # layer -> (its 4 expert refs, its index in stacked refs or None)
        layer_ex = {}
        k.layer_ex = layer_ex
        for ui, unit in enumerate(expert_units(c)):
            refs = [EX[4 * ui + j] for j in range(4)]
            if unit[0] == "layer":
                layer_ex[unit[1]] = (refs, None)
            else:
                for li in range(unit[1], unit[2]):
                    layer_ex[li] = (refs, li - unit[1])
        group_lo = {lo: hi for lo, hi in c.expert_groups}
        k.group_lo = group_lo
        KV = rest[:L]  # read-only here; see kv_write_call
        k.KV = KV
        hid_out, aux_out, newkv_out = rest[L], rest[L + 1], rest[L + 2]
        k.aux_out = aux_out
        k.hid_out = hid_out
        k.newkv_out = newkv_out
        nh = len(hnames)
        S = dict(zip(hnames, rest[L + 3 : L + 3 + nh]))
        k.S = S
        S.update(zip(snames, rest[L + 3 + nh :]))
        yw = y_words(c)
        k.yw = yw
        if _alias_pads_into_y(c):
            py3 = S["pad_y"].reshape(c.tp, R, yw)
            S["pad_o_rs"] = py3.at[:, :, pl.ds(0, Hl // 2)]
            S["pad_o_ag"] = py3.at[:, :, pl.ds(Hl // 2, Hl // 2)]
        elif _alias_oag_into_f(c):
            S["pad_o_ag"] = S["pad_f"]
        if _alias_tiled_xfull_q(c):
            S["qn_s"] = S["xfull"].bitcast(jnp.float32).at[:, pl.ds(0, c.q_dim)]
        if _alias_tiled_padu_attn(c):
            S["attn_buf"] = (
                S["pad_u"]
                .reshape(R, He // 2)
                .bitcast(jnp.float32)
                .at[:, pl.ds(0, c.q_dim)]
            )
        if _alias_tiled_xfull_attn(c):
            S["attn_buf"] = (
                S["xfull"].bitcast(jnp.float32).at[:, pl.ds(c.q_dim, c.q_dim)]
            )
        if c.vmem_trim:
            assert not (c.fold_f or c.fold_o)
            S["final_norm"] = S["smalls"].at[1 - (c.n_layers - 1) % 2, pl.ds(0, 8)]
            S["f_send"] = S["o_send"]
            # lifetimes: o_ag, then the pool all-gather, then f_ag (u_send: after the experts)
            S["pool_send"] = S["o_send"].at[:, pl.ds(0, c.expert_hidden_local // 2)]
        if _alias_u_xfull(c):
            S["u_send"] = S["xfull"].at[pl.ds(0, c.rows_per_dev)]
        if _alias_tiled_ids_kv(c):
            S["kb_s"] = S["ids_v"].bitcast(jnp.float32)
            S["vb_s"] = S["wsel_v"]
        if _alias_tiled_obuf_ysend(c):
            ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
            S["o_buf"] = (
                S["ysend"]
                .reshape(ob_n, (R * K) // ob_n, yw)
                .at[:, pl.ds(0, R), pl.ds(0, Hl // 2)]
            )
        if _alias_pool_ysend(c):
            S["pad_pool"] = (
                S["ysend"]
                .reshape(c.tp, (R * K) // c.tp, yw)
                .at[:, pl.ds(0, R), pl.ds(0, c.expert_hidden_local // 2)]
            )
        n_live = meta_ref[0]
        k.n_live = n_live
        tp_rank = meta_ref[1]
        k.tp_rank = tp_rank
        me = jax.lax.axis_index(axis) if c.tp > 1 else 0
        k.me = me
        if c.swa_window_table:
            # every request's sliding window of pages_swa (HBM) into SMEM, once per step: the
            # SWA layers share one table, and a request's window spans few of its entries
            wn = swa_window_entries(c)
            swa_cpt = (c.swa_chunk_pages or c.chunk_pages) * ps

            def swa_win_copy(j):  # psw_base counts TABLE_ALIGN-entry units
                start = pl.multiple_of(S["psw_base"][j] * TABLE_ALIGN, TABLE_ALIGN)
                return pltpu.make_async_copy(
                    psw_ref.at[pl.ds(start, wn)],
                    S["psw_win"].at[pl.ds(j * wn, wn)],
                    S["psw_sem"].at[0],
                )

            def swa_win_start(j, carry):
                p_lo = pos_ref[j * q]
                first_pg = (
                    jnp.maximum(p_lo - c.window + 1, 0) // swa_cpt * (swa_cpt // ps)
                )
                ent = j * c.pages_per_seq + first_pg
                S["psw_base"][j] = jnp.minimum(
                    ent // TABLE_ALIGN, (table_entries_padded(c) - wn) // TABLE_ALIGN
                )
                swa_win_copy(j).start()
                return carry

            def swa_win_wait(j, carry):
                swa_win_copy(j).wait()
                return carry

            jax.lax.fori_loop(0, c.n_req, swa_win_start, 0)
            jax.lax.fori_loop(0, c.n_req, swa_win_wait, 0)

        def swa_page(j, ent):
            """pages_swa[ent], an entry of request j's window."""
            if not c.swa_window_table:
                return psw_ref[ent]
            wn = swa_window_entries(c)
            base = S["psw_base"][j] * TABLE_ALIGN
            return S["psw_win"][j * wn + jnp.clip(ent - base, 0, wn - 1)]

        can_pair_fold = bool(
            c.pair_ag >= 1
            and c.tp == 8
            and all(_fold_g(c, g) == 2 for g in fold_groups(c))
        )
        if c.row_tile and c.tiled_vmem_pads and c.trim_sems:
            coll_pads = {
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "u": S["pad_u"],
                "f": S["pad_f"],
            }
        else:
            coll_pads = {
                "o_rs": S["pad_o_rs"],
                "o_ag": S["pad_o_ag"],
                "pool": S["pad_pool"],
                "y": S["pad_y"],
                "u": S["pad_u"],
                "f": S["pad_f"],
                # absent in row-tiled steps, which run their own collectives
                "o1": S.get("pad_o1"),
                "f1": S.get("pad_f1"),
            }
        coll = Coll(
            c.tp,
            me,
            axis,
            coll_pads,
            S.get("coll_ssems"),
            S.get("coll_rsems"),
            order=c.coll_order,
            pair_ag=c.pair_ag,
            pair_ssems=S.get("pair_ssems"),
            pair_rsems=S.get("pair_rsems"),
            can_pair_fold=can_pair_fold,
        )
        k.coll = coll
        # rotated o-proj columns: this device's own reduce-scatter block streams last
        F = few_rows(c)
        k.F = F
        few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
        # o-proj column tile m is reduce-scatter block m: ship each as it finishes
        o_rs_streamed = (
            c.o_rs_streamed
            and c.tp > 1
            and not few
            and all(tn == Hl for name, _, _, _, tn, _ in dense_tiles(c) if name == "wo")
        )
        k.o_rs_streamed = o_rs_streamed
        use_pair_o_rs = bool(
            c.row_tile
            and c.pair_o_rs
            and c.tiled_vmem_pads
            and c.tp == 8
            and c.expert_hidden == 4 * c.hidden_local
        )
        k.use_pair_o_rs = use_pair_o_rs
        # (row-tiled steps ship o-proj column tile m to device m themselves)
        o_rot = (
            o_rs_streamed
            and c.o_rs_rotate
            and (not c.row_tile or (c.tiled_o_rot and not use_pair_o_rs))
        )
        k_rot = {}
        for grp in fold_groups(c):
            # f blocks sit in pad slots by TP rank, o blocks by mesh index
            own = me if can_pair_fold else (tp_rank if grp == ("wqkvg",) else me)
            for name, _, _, tk, _, _ in dense_tiles(c):
                if name in grp:
                    k_rot[name] = (own, _fold_g(c, grp), _fold_g(c, grp) * Hl // tk)
        ws = _WeightStream(
            W,
            S["ring"],
            S["ring_sems"],
            dense_tiles(c),
            c.stream_depth,
            col_rot={"wo": me + 1} if o_rot else None,
            k_rot=k_rot,
        )
        k.ws = ws
        msem = S.get("misc_sems")

        def clear_seen(e, carry):
            S["seen"][e] = 0
            return carry

        if not c.vec_route:
            jax.lax.fori_loop(0, c.n_experts_padded, clear_seen, 0)
        # Entry barrier: no peer's remote write lands in this device's VMEM before it has
        # entered the kernel. The call aliases nothing, so its collective_id costs no copy.
        if c.tp > 1:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(1, c.tp):
                pl.semaphore_signal(
                    barrier,
                    device_id=(jnp.mod(me + d, c.tp),),
                )
            pl.semaphore_wait(barrier, c.tp - 1)

        def single_row(x, multi, one):
            """``multi(x)`` -> [R, H], or in a step of at most few_rows_max rows ``one(x)``'s [F, H]

            rows (the rest zero).
            """
            if not few:
                return multi(x)
            y = one(x)
            return jnp.concatenate([y, jnp.zeros((R - F, y.shape[1]), y.dtype)], axis=0)

        k.single_row = single_row

        def allreduce_row(x):
            S["row1_send"][...] = _pack_bf16x2(x[0:F, :])
            pad = coll.gather_few("o1", S["row1_send"], me)
            tot = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                tot = tot + _unpack_bf16x2(pad[d])
            return _unpack_bf16x2(_pack_bf16x2(tot))

        def allreduce(x):
            return single_row(x, allreduce_rows, allreduce_row)

        k.allreduce = allreduce

        def allreduce_rows(x):
            """f32 ``[R, H]`` partials -> bf16-exact totals: column reduce-scatter + all-gather.

            Partials travel as bf16 and are summed in f32, as the native bf16
            all-reduce.
            """
            for d in range(c.tp):
                S["xpack"][:, d * Hl // 2 : (d + 1) * Hl // 2] = _pack_bf16x2(
                    x[:, d * Hl : (d + 1) * Hl]
                )
            return allgather_sum(coll.reduce_scatter_cols("o_rs", S["xpack"], Hl // 2))

        def allgather_sum(pad):
            """Reduce-scattered column blocks in pad -> summed, all-gathered [R, H] (fold_o:

            the all-gather is only started).
            """
            mine = _unpack_bf16x2(pad[0])
            for d in range(1, c.tp):
                mine = mine + _unpack_bf16x2(pad[d])
            S["o_send"][...] = _pack_bf16x2(mine)
            if c.fold_o:
                coll.all_gather_blocks_start("o_ag", S["o_send"], me, S["o_rsems"])
                return None
            with _scope("o_ag"):
                pad2 = coll.all_gather("o_ag", S["o_send"], me)
            return jnp.concatenate(
                [_unpack_bf16x2(pad2[d]) for d in range(c.tp)], axis=1
            )

        k.allgather_sum = allgather_sum

        def f_blocks():
            return coll.all_gather_blocks_wait("f", S["f_send"], tp_rank, S["f_rsems"])

        k.f_blocks = f_blocks

        def o_blocks():
            return coll.all_gather_blocks_wait("o_ag", S["o_send"], me, S["o_rsems"])

        k.o_blocks = o_blocks

        def fold_lhs(pad):
            """Per-block fold LHS: [stream part; arriving block * gains], bf16 [2R, Hl]."""

            def lhs(k):
                cols = pl.ds(pl.multiple_of(k * Hl, LANE), Hl)
                part = S["fold_x"][:, cols]
                if pad is None:
                    return part.astype(jnp.bfloat16)
                blk = _unpack_bf16x2(S[pad][k]) * S["fold_v"][0:1, cols]
                return jnp.concatenate([part, blk], axis=0).astype(jnp.bfloat16)

            return lhs

        def gathered(pad):
            return jnp.concatenate(
                [_unpack_bf16x2(S[pad][d]) for d in range(c.tp)], axis=1
            )

        def fold_qkvg(l, x, pslot, sm, nxt):
            """qkvg of layer l whose input stream is x plus layer l-1's ffn branch, its f still

            arriving; returns (qkvg, layer l's input stream, exact).
            """
            g = sm[SMALL["pre_attn_w"] : SMALL["pre_attn_w"] + 1, :]

            def prow(name):
                return S["smalls"][pslot, SMALL[name] : SMALL[name] + 1, :]

            S["fold_x"][...] = prow("ffn_alpha") * x * g
            S["fold_v"][0:1, :] = prow("ffn_beta") * prow("post_ffn_w") * g
            wait_block, finish = f_blocks()
            with _scope("qkvg_fold"):
                acc = ws.mv_fold(
                    l,
                    ("wqkvg",),
                    nxt,
                    _fold_g(c, ("wqkvg",)),
                    Hl,
                    wait_block,
                    fold_lhs("pad_f"),
                )["wqkvg"]
            finish()
            y = gathered("pad_f") * prow("post_ffn_w")
            s_f = jax.lax.rsqrt(
                jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
            )
            stream = prow("ffn_alpha") * x + prow("ffn_beta") * (y * s_f).astype(
                jnp.bfloat16
            ).astype(jnp.float32)
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(stream), axis=-1, keepdims=True) + c.rms_eps
            )
            sc = sm[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]
            return r * (acc[:R] + s_f * acc[R:]) * sc, stream

        def plain_qkvg(l, attn_in, sm, nxt):
            """qkvg through the fold tile order with nothing arriving (the first layer)."""
            S["fold_x"][...] = attn_in.astype(jnp.float32)
            acc = ws.mv_fold(
                l,
                ("wqkvg",),
                nxt,
                _fold_g(c, ("wqkvg",)),
                Hl,
                lambda k: None,
                fold_lhs(None),
            )["wqkvg"]
            return acc * sm[SMALL["wqkvg_s"] : SMALL["wqkvg_s"] + 1, : c.qkvg_dim]

        def fold_pool_router(l, x, sm, nxt):
            """(pooled, router logits, the stream after the attention residual) of layer l, the

            o-proj all-gather still arriving; x is the layer's input stream.
            """
            g = sm[SMALL["pre_ffn_w"] : SMALL["pre_ffn_w"] + 1, :]
            alpha = sm[SMALL["attn_alpha"] : SMALL["attn_alpha"] + 1, :]
            beta = sm[SMALL["attn_beta"] : SMALL["attn_beta"] + 1, :]
            S["fold_x"][...] = alpha * x * g
            S["fold_v"][0:1, :] = beta * g
            wait_block, finish = o_blocks()
            names = ("wpool", "wrouter")
            with _scope("pool_router_fold"):
                acc = ws.mv_fold(
                    l,
                    names,
                    nxt,
                    _fold_g(c, names),
                    Hl,
                    wait_block,
                    fold_lhs("pad_o_ag"),
                )
            finish()
            a = gathered("pad_o_ag")
            s_a = jax.lax.rsqrt(
                jnp.mean(jnp.square(a), axis=-1, keepdims=True) + c.post_eps
            )
            stream = alpha * x + beta * (a * s_a).astype(jnp.bfloat16).astype(
                jnp.float32
            )
            r = jax.lax.rsqrt(
                jnp.mean(jnp.square(stream), axis=-1, keepdims=True) + c.rms_eps
            )

            def comb(v):
                return r * (v[:R] + s_a * v[R:])

            pooled = (
                comb(acc["wpool"]) * sm[SMALL["wpool_s"] : SMALL["wpool_s"] + 1, :Hel]
            )
            logits = comb(acc["wrouter"]) * c.router_mult
            return pooled.astype(jnp.bfloat16).astype(jnp.float32), logits, stream

        if not c.row_tile:  # the row-tiled program streams these from HBM
            # ---------------- embedding: weightless RMS norm of the bf16 rows, kept in bf16 precision
            stream = (
                _rms(emb_ref[...].astype(jnp.float32), c.rms_eps)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )

            # ---------------- rotary rows for every row's position
            cps = []
            for b in range(R):
                p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
                cp = pltpu.make_async_copy(
                    W["rope"].at[pl.ds(pl.multiple_of((p // 8) * 8, 8), 8)],
                    S["rope_blk"].at[b],
                    msem.at[b],
                )
                cp.start()
                cps.append(cp)
            rr = []
            for b in range(R):
                cps[b].wait()
                p = jnp.clip(pos_ref[b], 0, c.max_position - 1)
                sel = jax.lax.broadcasted_iota(jnp.int32, (8, 2 * hd), 0) == (
                    p - (p // 8) * 8
                )
                rr.append(
                    jnp.sum(
                        jnp.where(sel, S["rope_blk"][b], 0.0), axis=0, keepdims=True
                    )
                )
            rtab = jnp.concatenate(rr, axis=0)
            cos_x, sin_x = rtab[:, :hd], rtab[:, hd:]
            k.cos_x = cos_x
            k.sin_x = sin_x
            row_live = jax.lax.broadcasted_iota(jnp.int32, (R, 1), 0) < n_live
            k.row_live = row_live

        def flags(l):
            if isinstance(l, int):
                return (l not in gl), (l in gl)
            is_g = False
            for x in gl:
                is_g = jnp.logical_or(is_g, l == x)
            return jnp.logical_not(is_g), is_g

        k.flags = flags

        def smalls_copy(l, slot):
            return pltpu.make_async_copy(
                W["smalls"].at[l], S["smalls"].at[slot], S["small_sems"].at[slot]
            )

        k.smalls_copy = smalls_copy

        # ------------------------------------------------------------ attention
        attention = functools.partial(untiled_attn.attention, k)

        def req_geom(geo, j):
            """First cache chunk and chunk count of request j (cached keys: [first_tok, p_lo))."""
            cpt_g = geo.cp * ps
            r0 = j * q
            p_lo = pos_ref[r0]
            first_tok = _sel(geo.is_global, 0, jnp.maximum(p_lo - c.window + 1, 0))
            first_ch = first_tok // cpt_g
            n_ch = jnp.where(
                jnp.logical_and(r0 < n_live, p_lo > 0),
                (p_lo - 1) // cpt_g - first_ch + 1,
                0,
            )
            return p_lo, first_ch, n_ch

        k.req_geom = req_geom

        KR = c.kv_ring
        KG = KR // 2
        geo_global = _AttnGeom(c.chunk_pages, KG, KR, True)
        if c.swa_chunk_pages:
            geo_swa = _AttnGeom(
                c.swa_chunk_pages, c.swa_kv_group, 2 * c.swa_kv_group, False
            )

        def per_geom(is_global, fn):
            """``fn(geometry)`` for a layer: one shared body, or with swa_chunk_pages a body per layer type."""
            if not c.swa_chunk_pages:
                fn(_AttnGeom(c.chunk_pages, KG, KR, is_global))
            elif isinstance(is_global, bool):
                fn(geo_global if is_global else geo_swa)
            else:
                pl.when(is_global)(lambda: fn(geo_global))
                pl.when(jnp.logical_not(is_global))(lambda: fn(geo_swa))

        k.per_geom = per_geom

        def ring_at(geo, slot, pi):
            """Ring entry and page of page ``pi`` of chunk slot ``slot``."""
            f = c.chunk_pages // geo.cp
            if f == 1:
                return slot, pi
            return slot // f, (slot % f) * geo.cp + pi

        def chunk_pages_of(geo, j, ci):
            pages = []
            for pi in range(geo.cp):
                pg_no = ci * geo.cp + pi
                ent = jnp.minimum(
                    j * c.pages_per_seq + jnp.minimum(pg_no, c.pages_per_seq - 1),
                    c.n_pages_entries - 1,
                )
                pages.append(_sel(geo.is_global, pf_ref[ent], swa_page(j, ent)))
            return pages

        def issue_range(geo, l, j, ci0, slot0, n):
            """Chunks ``ci0 ..

            ci0+n`` of request j into ring slots ``slot0 ..``: ONE layer branch-tree
            walk whose leaf is a small runtime loop, so the 62 leaves stay small
            (unrolled per-chunk code in every leaf bloats the kernel).
            """

            def leaf(ref):
                def one(u, carry):
                    slot = (slot0 + u) % geo.kr
                    for pi, page in enumerate(chunk_pages_of(geo, j, ci0 + u)):
                        pltpu.make_async_copy(
                            page_src(ref, page),
                            chunk_dst(geo, slot, pi),
                            S["kv_chunk_sems"].at[slot],
                        ).start()
                    return carry

                jax.lax.fori_loop(0, n, one, 0)

            only = None
            if isinstance(geo.is_global, bool) and c.swa_chunk_pages:
                only = [x for x in range(c.n_layers) if (x in gl) == geo.is_global]
            _dispatch(l, KV, leaf, only)

        def kv_ring_ref():
            """The KV chunk ring.

            SAL: the expert w13 slots, idle during attention (every expert DMA of the
            previous layer was waited, this layer's start after attention), as [chunk,
            sal_words, chunk tokens] u32 in the native tiling: the page DMA relayouts
            the pool's (pk, 128) tiles, so chunk loads are whole vregs.
            """
            if c.kv_sal:
                return S["w13"].reshape(c.kv_ring, c.sal_words, c.chunk_pages * ps)
            return S["kv_chunk"]

        def ring_words(geo, slot, off=0, n=None):
            """u32 rows [off, off + n) of a ring chunk (n: sal_words) as a ref."""
            n = c.sal_words if n is None else n
            e, p = ring_at(geo, slot, 0)
            if geo.cp == c.chunk_pages:
                if off == 0 and n == c.sal_words:
                    return kv_ring_ref().at[e]
                return kv_ring_ref().at[e, pl.ds(off, n), :]
            if off == 0 and n == c.sal_words:
                return kv_ring_ref().at[
                    e, :, pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps)
                ]
            return kv_ring_ref().at[
                e, pl.ds(off, n), pl.ds(pl.multiple_of(p * ps, LANE), geo.cp * ps)
            ]

        k.ring_words = ring_words

        def chunk_dst(geo, slot, pi):
            if c.kv_sal:
                e, p = ring_at(geo, slot, pi)
                return kv_ring_ref().at[e, :, pl.ds(p * ps, ps)]
            return S["kv_chunk"].at[slot, pi]

        k.chunk_dst = chunk_dst

        def page_src(ref, page):
            if c.kv_sal:  # [2 * kv_heads, hd // pk, pk, page] -> its 32-bit words, tokens on lanes
                return ref.at[page].bitcast(jnp.uint32).reshape(c.sal_words, ps)
            return ref.at[page]

        k.page_src = page_src

        def wait_chunk(geo, slot):
            if c.kv_sal:
                r = ring_words(geo, slot)
                pltpu.make_async_copy(r, r, S["kv_chunk_sems"].at[slot]).wait()
                return
            ring = kv_ring_ref()
            pltpu.make_async_copy(
                ring.at[slot], ring.at[slot], S["kv_chunk_sems"].at[slot]
            ).wait()

        k.wait_chunk = wait_chunk

        def kv_prologue(geo, l, j):
            # the request's first geo.kr chunks
            _, first_ch, n_ch = req_geom(geo, j)
            issue_range(geo, l, j, first_ch, 0, jnp.minimum(n_ch, geo.kr))

        k.kv_prologue = kv_prologue

        def rows_at(ref, r0, n):
            """``n`` rows from a dynamic start: one-row accesses, as a multi-row dynamic sublane

            window must start at a multiple of 8.
            """
            if n % 8 == 0:  # callers' r0 is a multiple of n
                return ref[pl.ds(pl.multiple_of(r0, 8), n), :]
            return (
                jnp.concatenate([ref[pl.ds(r0 + t, 1), :] for t in range(n)], axis=0)
                if n > 1
                else ref[pl.ds(r0, 1), :]
            )

        k.rows_at = rows_at

        request_attention_body = functools.partial(
            untiled_attn.request_attention_body, k
        )
        k.request_attention_body = request_attention_body

        # ------------------------------------------------------------ MoE
        Rc = c.rows_per_dev
        k.Rc = Rc
        y_block = c.y_block_send and c.tp > 1
        k.y_block = y_block

        def blk_owner(b):
            """Owner of expert block b: peers in send order, own rows last."""
            if isinstance(b, int) and b == c.tp - 1:
                return me
            if coll.order == "xor":
                return jnp.bitwise_xor(me, c.tp - 1 - b)
            return jax.lax.rem(me + 1 + b, c.tp)

        k.blk_owner = blk_owner

        NBR = route_blocks(c)
        k.NBR = NBR

        def route_get(field, i):
            """vec_route table entry (0 expert, 1 count, 2 offset, 3 pair, 5 y-block end, 6 blk_mid)."""
            if field in (5, 6):
                return S["route_s"][4 * NBR + 1 + (field == 6), i]
            return S["route_s"][field * NBR + i // LANE, i % LANE]

        def rget(name, i):
            if not c.vec_route:
                return S[name][i]
            return route_get(
                {
                    "uniq": 0,
                    "ex_cnt": 1,
                    "ex_off": 2,
                    "perm": 3,
                    "blk_end": 5,
                    "blk_mid": 6,
                }[name],
                i,
            )

        k.rget = rget

        def divmod_nn(i, d):
            """(i // d, i % d) for i >= 0: shift and mask when d is a power of two (signed //
            and % lower to ~10 scalar ops each)."""
            if d & (d - 1):
                return i // d, i % d
            return jax.lax.shift_right_logical(i, d.bit_length() - 1), jnp.bitwise_and(
                i, d - 1
            )

        k.divmod_nn = divmod_nn

        def unique_experts_vec(ids_list):
            P = R * K
            r_of = jax.lax.broadcasted_iota(jnp.int32, (P, 1), 0) % R
            if y_block:
                # a pair's key: the y block of its row's owner (own rows last)
                o = r_of // Rc
                key = (
                    (c.tp - 1) - jnp.bitwise_xor(o, me)
                    if coll.order == "xor"
                    else jax.lax.rem(o - me - 1 + c.tp, c.tp)
                )
            else:
                key = jnp.zeros((P, 1), jnp.int32)
            S["route_v"][...] = _route_table(
                ids_list, n_live, K, c.n_experts_padded, NBR, key, c.tp, c.split_rows
            )
            cp = pltpu.make_async_copy(S["route_v"], S["route_s"], S["route_sem"].at[0])
            cp.start()
            cp.wait()
            return S["route_s"][4 * NBR, 0]

        k.unique_experts_vec = unique_experts_vec

        unique_experts = functools.partial(untiled_moe.unique_experts, k)
        k.unique_experts = unique_experts

        moe = functools.partial(untiled_moe.moe, k)

        def prefetch_smalls(l, nxt):
            """The next layer's smalls into the slot layer l - 1 used."""
            if nxt is None:
                return
            if isinstance(l, int):
                smalls_copy(l + 1, (l + 1) % 2).start()
            else:
                pl.when(l + 1 < L)(smalls_copy(l + 1, (l + 1) % 2).start)

        k.prefetch_smalls = prefetch_smalls

        def capture(l, stream):
            for ci, cl in enumerate(c.capture_layers):
                if isinstance(l, int):
                    if l == cl:
                        aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)
                    continue

                @pl.when(l == cl)
                def _(ci=ci, stream=stream):
                    aux_out[:, ci * H : (ci + 1) * H] = stream.astype(jnp.bfloat16)

        def layer(l, carry, kind, next_layer, prefetched, pend=False):
            """carry: the layer's input stream, or with pend (fold_f) the previous

            layer's stream before its ffn residual, whose f is still arriving. Returns
            the
            output stream, or with fold_f and a next layer the stream before the ffn
            residual.
            """
            nxt = l + 1 if next_layer else None
            slot = l % 2
            if not prefetched:
                smalls_copy(l, slot).start()
                ws.prologue(l)
            with _scope("smalls_wait"):
                smalls_copy(l, slot).wait()
            sm = S["smalls"][slot]
            use_rope, is_global = flags(l)

            def row(name, n=H):
                return sm[SMALL[name] : SMALL[name] + 1, :n]

            hold = {}
            if pend:

                def qkvg_fn():
                    qkvg, hold["stream"] = fold_qkvg(l, carry, (l + 1) % 2, sm, nxt)
                    return qkvg

            else:
                stream = carry
                attn_in = _rms(stream, c.rms_eps, row("pre_attn_w")).astype(
                    jnp.bfloat16
                )
                if c.fold_f:
                    qkvg_fn = functools.partial(plain_qkvg, l, attn_in, sm, nxt)
                else:
                    qkvg_fn = functools.partial(ws.mv, attn_in, l, "wqkvg", nxt, sm=sm)
            with _scope("attention"):
                a = attention(l, qkvg_fn, use_rope, is_global, nxt, sm)
            prefetch_smalls(l, nxt)
            if pend:
                stream = hold["stream"]
                capture(l - 1, stream)
            if c.fold_o:

                def pool_router():
                    pooled, logits, hold["mid"] = fold_pool_router(l, stream, sm, nxt)
                    return pooled, logits

                with _scope("moe"):
                    f = moe(l, None, sm, kind, nxt, pool_router=pool_router)
                stream = hold["mid"]
            else:
                a = a.astype(jnp.bfloat16).astype(jnp.float32)
                stream = row("attn_alpha") * stream + row("attn_beta") * _rms(
                    a, c.post_eps
                ).astype(jnp.bfloat16).astype(jnp.float32)
                ffn_in32 = _rms(stream, c.rms_eps, row("pre_ffn_w"))
                with _scope("moe"):
                    f = moe(l, ffn_in32, sm, kind, nxt)
            if f is None:  # fold_f: the next layer finishes the ffn residual
                return stream
            stream = row("ffn_alpha") * stream + row("ffn_beta") * _rms(
                f * row("post_ffn_w"), c.post_eps
            ).astype(jnp.bfloat16).astype(jnp.float32)
            capture(l, stream)
            return stream

        # ============================================================ row-tiled program
        # Phase by phase over the whole step: each dense weight tile and expert slab is read
        # from HBM once and applied to every row; per-row passes run RT rows at a time.
        RT = c.row_tile
        k.RT = RT
        NT = R // RT if RT else 0
        k.NT = NT
        f32, bf16 = jnp.float32, jnp.bfloat16
        k.bf16 = bf16
        k.f32 = f32

        def tsem(i):
            return S["tile_sems"].at[i]

        k.tsem = tsem

        def rows_of(t):
            return pl.ds(pl.multiple_of(t * RT, RT), RT)

        k.rows_of = rows_of

        def for_tiles(fn):
            """fn(rows, t) for every row tile in one runtime loop (one emitted body)."""

            def body(t, carry):
                fn(rows_of(t), t)
                return carry

            jax.lax.fori_loop(0, NT, body, 0)

        k.for_tiles = for_tiles

        def vpads_tiles(fn):
            """fn(rows, t) per row tile of a VMEM-pads pass: one runtime loop under
            tile_loops (unrolled tiles keep their own VMEM temporaries), else unrolled."""
            if c.tile_loops:
                for_tiles(fn)
                return
            for t in range(NT):
                fn(pl.ds(t * RT, RT), t)

        k.vpads_tiles = vpads_tiles

        def dma(src, dst, i):
            cp = pltpu.make_async_copy(src, dst, tsem(i))
            cp.start()
            cp.wait()

        k.dma = dma

        def tile_pass(loads, compute, stores=(), ready=None):
            """for_tiles with the HBM traffic double-buffered.

            Tile t + 1's loads and tile t - 1's stores fly while tile t computes.
            loads / stores: (hbm_view_of(rows), buf [2, ...]); compute(rows, t, loaded
            views) returns the values to store. ready[i](t), if given, waits until
            load i's rows of tile t have landed (a pipelined all-gather) before that
            load is issued.
            """
            nl = len(loads)
            ready = ready or [None] * nl

            def ld_start(i, t, s_):
                if ready[i] is not None:
                    ready[i](t)
                ld(i, t, s_).start()

            def ld(i, t, s_):
                src, buf = loads[i]
                return pltpu.make_async_copy(
                    src(rows_of(t)), buf.at[s_], S["pass_sems"].at[s_, i]
                )

            def st(j, t, s_):
                dst, buf = stores[j]
                return pltpu.make_async_copy(
                    buf.at[s_], dst(rows_of(t)), S["pass_sems"].at[s_, nl + j]
                )

            for i in range(nl):
                ld_start(i, 0, 0)

            def body(t, carry):
                s_ = t % 2
                for i in range(nl):
                    ld(i, t, s_).wait()

                @pl.when(t + 1 < NT)
                def _():
                    for i in range(nl):
                        ld_start(i, t + 1, 1 - s_)

                vals = compute(rows_of(t), t, [buf.at[s_] for _, buf in loads])
                for j, v in enumerate(vals):
                    pl.when(t >= 2)(lambda j=j: st(j, t - 2, s_).wait())
                    stores[j][1].at[s_][...] = v
                    st(j, t, s_).start()
                return carry

            jax.lax.fori_loop(0, NT, body, 0)
            for j in range(len(stores)):
                for t in range(max(0, NT - 2), NT):
                    st(j, t, t % 2).wait()

        k.tile_pass = tile_pass

        def stream_view(rows):
            return S["h_stream"].at[rows]

        k.stream_view = stream_view

        def tile_live(t):
            return jax.lax.broadcasted_iota(jnp.int32, (RT, 1), 0) + t * RT < n_live

        k.tile_live = tile_live

        def pad_view(pad):
            return lambda rows: pad.at[:, rows]

        k.pad_view = pad_view

        def gathered_tile(v):
            """A row tile of an all-gathered [tp, R, w/2] pad -> f32 [RT, tp * w]."""
            return jnp.concatenate([_unpack_bf16x2(v[d]) for d in range(c.tp)], axis=1)

        k.gathered_tile = gathered_tile

        tc_smap = _TC_VPADS_S if (c.tiled_vmem_pads and c.trim_sems) else _TC
        k.tc_smap = tc_smap
        tc_rmap = _TC_VPADS_R if (c.tiled_vmem_pads and c.trim_sems) else _TC
        k.tc_rmap = tc_rmap
        use_pair_y = bool(
            c.row_tile and c.tiled_pair_y >= 1 and c.tp == 8 and c.tiled_vmem_y >= 2
        )
        k.use_pair_y = use_pair_y
        stream_pair_y = bool(use_pair_y and y_block and c.pair_y_stream)
        k.stream_pair_y = stream_pair_y

        def rcopy(src, dst, peer, kind, send_kind=None):
            return pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=S["tc_ssems"].at[tc_smap[send_kind or kind]],
                recv_sem=S["tc_rsems"].at[tc_rmap[kind]],
                device_id=(peer,),
            )

        k.rcopy = rcopy

        def ag_tile_start(kind, src, pad, slot, t, sems):
            """Sends row tile t of src to pad[slot] on every device once it is written.

            src is VMEM [R, w] and pad HBM [tp, R, w]; the receivers' sems[t] count
            the copies.
            """
            rows = rows_of(t)
            pltpu.make_async_copy(src.at[rows], pad.at[slot, rows], sems.at[t]).start()
            for d in range(1, c.tp):
                pltpu.make_async_remote_copy(
                    src_ref=src.at[rows],
                    dst_ref=pad.at[slot, rows],
                    send_sem=S["tc_ssems"].at[_TC[kind]],
                    recv_sem=sems.at[t],
                    device_id=(coll._peer(d),),
                ).start()

        k.ag_tile_start = ag_tile_start

        def ag_tile_ready(src, pad, sems):
            """ready(t): every device's row tile t of a pipelined all-gather has landed."""

            def ready(t):
                tmpl = pltpu.make_async_copy(
                    src.at[pl.ds(0, RT)], pad.at[0, pl.ds(0, RT)], sems.at[t]
                )
                for _ in range(c.tp):
                    tmpl.wait()

            return ready

        k.ag_tile_ready = ag_tile_ready

        def ag_tile_sent(kind, src, pad):
            """Waits for this device's sends of a pipelined all-gather (before src is reused)."""
            for d in range(1, c.tp):
                rcopy(src, pad.at[0], coll._peer(d), kind).wait_send()

        k.ag_tile_sent = ag_tile_sent

        # pipe_ag: pair_ag's two-stage all-gather per row tile. A tile goes to the same core
        # of the three other chips as it is written; the consumer forwards each chip's tile to
        # its pair partner one tile ahead of its own use.
        p_core, p_chip = me % 2, me // 2
        p_partner = jnp.bitwise_xor(me, 1)

        def hag_cross(kind, src, pad, t, tsems, d):
            rows = rows_of(t)
            return pltpu.make_async_remote_copy(
                src_ref=src.at[rows],
                dst_ref=pad.at[me, rows],
                send_sem=S["tc_ssems"].at[_TC[kind]],
                recv_sem=tsems.at[t],
                device_id=(2 * jnp.bitwise_xor(p_chip, d) + p_core,),
            )

        def hag_fwd(kind, pad, t, ptsems, ck):
            rows = rows_of(t)
            return pltpu.make_async_remote_copy(
                src_ref=pad.at[2 * ck + p_core, rows],
                dst_ref=pad.at[2 * ck + p_core, rows],
                send_sem=S["ag_fwd_sems"].at[_PIPE_AG[kind]],
                recv_sem=ptsems.at[t],
                device_id=(p_partner,),
            )

        def hag_start(kind, src, pad, t, tsems):
            """Producer: row tile t of src into pad[me] here and on the other chips."""
            pad[me, rows_of(t)] = src[rows_of(t)]
            for d in (1, 2, 3):
                hag_cross(kind, src, pad, t, tsems, d).start()

        k.hag_start = hag_start

        def hag_forward(kind, src, pad, tsems, t, ptsems):
            """Waits for tile t from the other chips, then forwards this core's four to the partner."""
            for d in (1, 2, 3):
                hag_cross(kind, src, pad, t, tsems, d).wait_recv()
            for ck in range(4):
                hag_fwd(kind, pad, t, ptsems, ck).start()

        k.hag_forward = hag_forward

        def hag_ready(kind, pad, t, ptsems):
            for ck in range(4):
                hag_fwd(kind, pad, t, ptsems, ck).wait_recv()

        k.hag_ready = hag_ready

        # pipe_u: the u all-gather in 8-row tiles of each owner block, sent as combine writes
        # them; the forwards and waits come after the combine loop
        pipe_u = bool(c.pipe_ag and c.tiled_vmem_y < 2 and c.rows_per_dev % 8 == 0)
        k.pipe_u = pipe_u
        n_ut = c.rows_per_dev // 8

        def u_rows(j):
            return pl.ds(pl.multiple_of(j * 8, 8), 8)

        def u_cross(j, d):
            return pltpu.make_async_remote_copy(
                src_ref=S["u_send"].at[u_rows(j)],
                dst_ref=S["pad_u"].at[me, u_rows(j)],
                send_sem=S["tc_ssems"].at[_TC["u"]],
                recv_sem=S["uag_tsems"].at[j],
                device_id=(2 * jnp.bitwise_xor(p_chip, d) + p_core,),
            )

        def u_fwd(j, ck):
            return pltpu.make_async_remote_copy(
                src_ref=S["pad_u"].at[2 * ck + p_core, u_rows(j)],
                dst_ref=S["pad_u"].at[2 * ck + p_core, u_rows(j)],
                send_sem=S["ag_fwd_sems"].at[_PIPE_AG["u"]],
                recv_sem=S["uag_ptsems"].at[j],
                device_id=(p_partner,),
            )

        def u_start(j):
            S["pad_u"][me, u_rows(j)] = S["u_send"][u_rows(j)]
            for d in (1, 2, 3):
                u_cross(j, d).start()

        k.u_start = u_start

        def u_finish():
            """Every owner block's tiles here: cross arrivals forwarded, forwards received."""
            for j in range(n_ut):
                for d in (1, 2, 3):
                    u_cross(j, d).wait_recv()
                for ck in range(4):
                    u_fwd(j, ck).start()
            for j in range(n_ut):
                for ck in range(4):
                    u_fwd(j, ck).wait_recv()
            for j in range(n_ut):
                for d in (1, 2, 3):
                    u_cross(j, d).wait_send()
                for ck in range(4):
                    u_fwd(j, ck).wait_send()

        k.u_finish = u_finish

        def hag_sent(kind, src, pad, tsems, ptsems):
            for t in range(NT):
                for d in (1, 2, 3):
                    hag_cross(kind, src, pad, t, tsems, d).wait_send()
                for ck in range(4):
                    hag_fwd(kind, pad, t, ptsems, ck).wait_send()

        k.hag_sent = hag_sent

        def ag_start(kind, src, dst, live=None):
            """src (VMEM) into dst, the same HBM region on every device; returns the wait.

            live(m): whether device m's part is needed (dead parts are neither sent
            nor waited).
            """
            local = pltpu.make_async_copy(src, dst, tsem(_TS_LOCAL))
            local.start()
            cps = [
                (coll._peer(d), rcopy(src, dst, coll._peer(d), kind))
                for d in range(1, c.tp)
            ]

            def start():
                for _, cp in cps:
                    cp.start()

            if live is None:
                start()
            else:
                pl.when(live(me))(start)

            def wait():
                local.wait()
                if live is None:
                    for _, cp in cps:
                        cp.wait()
                    return

                def wait_sends():
                    for _, cp in cps:
                        cp.wait_send()

                pl.when(live(me))(wait_sends)
                for peer, cp in cps:
                    pl.when(live(peer))(cp.wait_recv)

            return wait

        k.ag_start = ag_start

        route_tiled = functools.partial(tiled_attn.route_tiled, k)
        k.route_tiled = route_tiled

        def mv_tiled(l, name, nxt, smr, col_off=None, on_col=None):
            """S["lhs"] [R, K] @ W_l[name] -> S["acc"] columns (scaled fp8); on_col(ni, col) once

            column tile ni is final. One MXU pass per mv_rows rows per weight tile.
            """
            assert name not in ws.k_rot
            t0, t1, tk, tn, nn, dt = ws.ranges[name]
            nk = (t1 - t0) // nn
            is_fp8 = jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
            MR = min(R, c.mv_rows)
            assert R % MR == 0
            for t in range(t0, t1):
                with _scope("ring_wait"):
                    ws._copy(l, t).wait()
                _, ki, ni, _, _, _ = ws.tiles[t]
                col = ws.phys_col(name, ni)
                c0 = ni * tn if col_off is None else col_off(ni)
                w = ws.buf.at[t % ws.slots].bitcast(dt)[pl.ds(0, tk), pl.ds(0, tn)]
                for m0 in range(0, R, MR):
                    lhs_sub = S["lhs"][pl.ds(m0, MR), pl.ds(ki * tk, tk)]
                    part = jnp.dot(
                        lhs_sub,
                        w,
                        preferred_element_type=f32,
                    )
                    if ki > 0:
                        part = S["acc"][pl.ds(m0, MR), pl.ds(c0, tn)] + part
                    if ki == nk - 1 and is_fp8:
                        r = SMALL[name + "_s"]
                        sc_col = (
                            smr[r : r + 1, pl.ds(ni * tn, tn)]
                            if isinstance(col, int)
                            else smr[
                                r : r + 1, pl.ds(pl.multiple_of(col * tn, LANE), tn)
                            ]
                        )
                        part = part * sc_col
                    S["acc"][pl.ds(m0, MR), pl.ds(c0, tn)] = part
                ws.refill(l, t, nxt)
                if on_col is not None and ki == nk - 1:
                    on_col(ni, col)

        k.mv_tiled = mv_tiled

        def tiled_fold_lhs(pad):
            def lhs(k):
                cols = pl.ds(pl.multiple_of(k * Hl, LANE), Hl)
                part = S["lhs"][:, cols]
                if pad is None:
                    return part
                blk = (_unpack_bf16x2(S[pad][k]) * S["fold_v"][0:1, cols]).astype(bf16)
                return jnp.concatenate([part, blk], axis=0)

            return lhs

        k.tiled_fold_lhs = tiled_fold_lhs

        attention_tiled = functools.partial(tiled_attn.attention_tiled, k)
        k.attention_tiled = attention_tiled

        expert_io_tiled = functools.partial(tiled_moe.expert_io_tiled, k)
        k.expert_io_tiled = expert_io_tiled

        moe_tiled = functools.partial(tiled_moe.moe_tiled, k)
        k.moe_tiled = moe_tiled

        def capture_tile(l, rows, s):
            for ci, cl in enumerate(c.capture_layers):

                def write(ci=ci):
                    S["btile"][...] = s.astype(bf16)
                    dma(S["btile"], aux_out.at[rows, pl.ds(ci * H, H)], _TS_AUX)

                if isinstance(l, int):
                    if l == cl:
                        write()
                    continue
                pl.when(l == cl)(write)

        k.capture_tile = capture_tile

        layer_tiled = functools.partial(tiled_layer.layer_tiled, k)
        k.layer_tiled = layer_tiled

        tiled_main = functools.partial(tiled_layer.tiled_main, k)

        if c.row_tile:
            tiled_main()
            return

        # every layer but the last prefetches the next one's smalls and tiles
        state = {"pref": False, "pend": False}

        def emit(l, stream, kind, next_layer):
            stream = layer(l, stream, kind, next_layer, state["pref"], state["pend"])
            state["pref"] = next_layer
            state["pend"] = c.fold_f and next_layer
            return stream

        def rolled(i, v):
            # every rolled layer follows a layer that left its f in flight (with fold_f)
            assert state["pend"] == c.fold_f
            return layer(i, v, "fp4", True, True, c.fold_f)

        for l in range(L):
            if roll_lo is not None and l == roll_lo and l < roll_hi:
                lo, hi = l, roll_hi
                if not state["pref"]:
                    stream = emit(lo, stream, "fp4", next_layer=lo + 1 < L)
                    lo += 1
                if hi > lo:
                    if hi < L:
                        stream = jax.lax.fori_loop(lo, hi, rolled, stream)
                        state["pref"] = True
                    else:
                        if hi - 1 > lo:
                            stream = jax.lax.fori_loop(lo, hi - 1, rolled, stream)
                        state["pref"] = True
                        stream = emit(hi - 1, stream, "fp4", next_layer=False)
            elif roll_lo is None or not (roll_lo <= l < roll_hi):
                stream = emit(l, stream, c.expert_kinds[l], next_layer=l + 1 < L)
        # ---------------- final norm (on the bf16-cast stream, as the native path)
        PH_final = pltpu.make_async_copy(
            W["final_norm"], S["final_norm"], msem.at[R + 2]
        )
        PH_final.start()
        PH_final.wait()
        hidden = _rms(
            stream.astype(jnp.bfloat16).astype(jnp.float32),
            c.rms_eps,
            S["final_norm"][0:1, :],
        )
        hid_out[...] = jnp.where(row_live, hidden, 0.0).astype(jnp.bfloat16)
        if ncap == 0:
            aux_out[...] = jnp.zeros(aux_out.shape, aux_out.dtype)

    return kernel, scratch, hbm_scratch


def _is_hbm(ref) -> bool:
    return getattr(ref, "memory_space", None) == pltpu.HBM


def megakernel_call(cfg: MKConfig, axis: str = "model", ep: EPHook | None = None):
    """``f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv) -> (hidden, aux, newkv)``

    for one device (call inside ``shard_map``).
    """
    kernel, scratch, hbm_scratch = build_kernel(cfg, axis, ep)
    L = cfg.n_layers
    nw = len(WEIGHT_NAMES)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    ncap = len(cfg.capture_layers)
    aux_w = ncap * cfg.hidden if ncap else LANE

    def call(meta, pos, pages_full, pages_swa, emb, *rest):
        weights = rest[:nw]
        ne = n_expert_refs(cfg)
        experts = rest[nw : nw + ne]
        kv = rest[nw + ne :]
        assert len(kv) == L
        out_shape = [
            jax.ShapeDtypeStruct((cfg.rows, cfg.hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((cfg.rows, aux_w), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                newkv_shape(cfg), jnp.uint32 if cfg.kv_sal else jnp.bfloat16
            ),
        ] + [jax.ShapeDtypeStruct(r.shape, r.dtype) for _, r in hbm_scratch]
        # The pools are read-only here. With any remote DMA in the kernel XLA treats the call
        # as a collective and copies every aliased operand: the in-place write is kv_write_call.
        f = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=[smem] * 3
            + [hbm if cfg.swa_window_table else smem]
            + [hbm if cfg.row_tile else vmem]
            + [hbm]
            * (nw + n_expert_refs(cfg) + L),  # emb: bf16 rows, or the f32 stream
            out_specs=([hbm, hbm, hbm] if cfg.row_tile else [vmem, vmem, hbm])
            + [hbm] * len(hbm_scratch),
            scratch_shapes=[s for _, s in scratch],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                collective_id=_COLLECTIVE_ID if cfg.tp > 1 else None,
                disable_bounds_checks=ep is not None
                and tpu_envs.SGLANG_TPU_EP_NO_BOUNDS.get(),
            ),
            name=f"muse_{'ep_' if ep else ''}megakernel_r{cfg.rows}_q{cfg.rows_per_req}",
        )
        if cfg.swa_window_table:
            pages_swa = pad_table(cfg, pages_swa)
        out = f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv)
        return tuple(out[:3])

    return call


def newkv_shape(cfg: MKConfig):
    if cfg.kv_sal:  # new token columns of a page block, [slot * words + word, row]
        return (cfg.n_layers, cfg.sal_words, cfg.rows)
    return (
        cfg.n_layers,
        cfg.rows,
        cfg.pool_kv_heads,
        2,
        cfg.pool_padded_head_dim,
    )


def _kv_write_sal_call(cfg: MKConfig):
    """SAL pools: a token is one lane column of its page block, so each live request

    read-modify-writes the (at most two) pages its consecutive rows land in. Every
    layer's page
    is in flight at once; requests run one after another.
    """
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)
    nw = cfg.sal_words
    # with swa_window_table: a request's (at most two) pages_swa entries, from an aligned DMA
    KVW = 2 * TABLE_ALIGN

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        blk, new_v, rsem, wsem, nsem = rest[2 * L : 2 * L + 5]
        if cfg.swa_window_table:
            swin, sbase, ssem = rest[2 * L + 5 : 2 * L + 8]
        n_live = meta_ref[0]
        cp = pltpu.make_async_copy(newkv, new_v, nsem.at[0])
        cp.start()
        cp.wait()
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, ps), 1)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, R), 1)
        n_req_live = jnp.minimum((n_live + q - 1) // q, cfg.n_req)

        def page_of(l, j, pg_no):
            ent = jnp.minimum(j * cfg.pages_per_seq + pg_no, cfg.n_pages_entries - 1)
            if l in gl:
                return pf_ref[ent]
            if not cfg.swa_window_table:
                return psw_ref[ent]
            return swin[jnp.clip(ent - sbase[0] * TABLE_ALIGN, 0, KVW - 1)]

        def load_swa(j, first):
            unit = jnp.minimum(
                (j * cfg.pages_per_seq + first) // TABLE_ALIGN,
                (table_entries_padded(cfg) - KVW) // TABLE_ALIGN,
            )
            sbase[0] = unit  # in TABLE_ALIGN-entry units, the DMA offset's alignment
            start = pl.multiple_of(unit * TABLE_ALIGN, TABLE_ALIGN)
            cp = pltpu.make_async_copy(psw_ref.at[pl.ds(start, KVW)], swin, ssem.at[0])
            cp.start()
            cp.wait()

        def rmw(j, pg_no):
            for l in range(L):
                pltpu.make_async_copy(
                    kv_out[l].at[page_of(l, j, pg_no)], blk.at[l], rsem.at[l]
                ).start()

            def merge(l, carry):
                pltpu.make_async_copy(kv_out[0].at[0], blk.at[l], rsem.at[l]).wait()
                words = blk.at[l].bitcast(jnp.uint32).reshape(nw, ps)
                w = words[...]
                nv = jax.lax.bitcast_convert_type(new_v[l], jnp.int32)  # [nw, R]
                for t in range(q):
                    r = j * q + t
                    p = pos_ref[r]
                    newcol = jax.lax.bitcast_convert_type(
                        jnp.sum(jnp.where(col == r, nv, 0), axis=1, keepdims=True),
                        jnp.uint32,
                    )
                    hit = (lane == p % ps) & (p // ps == pg_no) & (r < n_live)
                    w = jnp.where(hit, newcol, w)
                words[...] = w
                return carry

            jax.lax.fori_loop(0, L, merge, 0)
            for l in range(L):
                pltpu.make_async_copy(
                    blk.at[l], kv_out[l].at[page_of(l, j, pg_no)], wsem.at[l]
                ).start()

            def drain(l, carry):
                pltpu.make_async_copy(blk.at[l], kv_out[0].at[0], wsem.at[l]).wait()
                return carry

            jax.lax.fori_loop(0, L, drain, 0)

        def req(j, carry):
            first = pos_ref[j * q] // ps
            last = pos_ref[jnp.minimum(j * q + q - 1, n_live - 1)] // ps
            if cfg.swa_window_table:
                load_swa(j, first)
            rmw(j, first)
            pl.when(last != first)(lambda: rmw(j, last))
            return carry

        jax.lax.fori_loop(0, n_req_live, req, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    blk_bytes = L * math.prod(cfg.sal_block) * jnp.dtype(cfg.kv_dtype).itemsize

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        if cfg.swa_window_table:
            pages_swa = pad_table(cfg, pages_swa)
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 3
            + [hbm if cfg.swa_window_table else smem]
            + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[
                pltpu.VMEM((L, *cfg.sal_block), cfg.kv_dtype),
                pltpu.VMEM((L, nw, R), jnp.uint32),
                pltpu.SemaphoreType.DMA((L,)),
                pltpu.SemaphoreType.DMA((L,)),
                pltpu.SemaphoreType.DMA((1,)),
            ]
            + (
                [
                    pltpu.SMEM((KVW,), jnp.int32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SemaphoreType.DMA((1,)),
                ]
                if cfg.swa_window_table
                else []
            ),
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=blk_bytes + 16 * 2**20
            ),
            name=f"muse_kv_write_sal_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call


def kv_write_call(cfg: MKConfig):
    """``f(meta, pos, pages_full, pages_swa, newkv, *kv) -> kv``: scatter this step's K/V rows

    into every layer's pool in place (no remote DMA, so XLA aliases the pools
    without a copy).
    """
    if cfg.kv_sal:
        return _kv_write_sal_call(cfg)
    L, R, ps, q = cfg.n_layers, cfg.rows, cfg.page_size, cfg.rows_per_req
    gl = set(cfg.global_layers)

    def kernel(meta_ref, pos_ref, pf_ref, psw_ref, newkv, *rest):
        kv_out = rest[L : 2 * L]
        sems = rest[2 * L]
        n_live = meta_ref[0]
        cps = []
        for l in range(L):
            tab = pf_ref if l in gl else psw_ref
            flat = kv_out[l].reshape(kv_out[l].shape[0] * ps, *kv_out[l].shape[2:])
            for r in range(R):
                p = pos_ref[r]
                ent = jnp.minimum(
                    (r // q) * cfg.pages_per_seq + p // ps, cfg.n_pages_entries - 1
                )
                tok = tab[ent] * ps + p % ps
                cp = pltpu.make_async_copy(
                    newkv.at[l, pl.ds(r, 1)], flat.at[pl.ds(tok, 1)], sems.at[l * R + r]
                )

                @pl.when(r < n_live)
                def _(cp=cp):
                    cp.start()

                cps.append((r, cp))
        for r, cp in cps:

            @pl.when(r < n_live)
            def _(cp=cp):
                cp.wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def call(meta, pos, pages_full, pages_swa, newkv, *kv):
        return pl.pallas_call(
            kernel,
            out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype) for k in kv],
            in_specs=[smem] * 4 + [hbm] * (1 + L),
            out_specs=[hbm] * L,
            scratch_shapes=[pltpu.SemaphoreType.DMA((L * R,))],
            input_output_aliases={5 + i: i for i in range(L)},
            compiler_params=pltpu.CompilerParams(),
            name=f"muse_kv_write_r{R}",
        )(meta, pos, pages_full, pages_swa, newkv, *kv)

    return call
