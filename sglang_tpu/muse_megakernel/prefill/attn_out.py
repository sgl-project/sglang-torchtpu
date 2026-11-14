# SPDX-License-Identifier: Apache-2.0
"""Prefill kernel between a layer's attention and its MoE, one row tile per pipeline step.

Fuses the o projection, its all-reduce (reduce-scatter + all-gather), the gated residual,
the router top-k and the expert input projection with its all-gather. The layer's weights
stay in VMEM; the stream and activations move through HBM one row tile at a time.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import (
    LANE,
    PREFILL_ATTN_OUT_COLLECTIVE_ID,
    SMALL,
)
from sglang_tpu.muse_megakernel.mk.numerics import _heads_rms, _rms, _topk_iter
from sglang_tpu.muse_megakernel.prefill.coll import (
    RS_KINDS,
    PairColl,
    ag_kinds,
    gathered,
    run_pipeline,
    to_fp8,
    wait_all,
)
from sglang_tpu.muse_megakernel.prefill.config import PFConfig

# One idle iteration after every stage: each stage's remote DMAs get two iterations to land.
_GAP = 1
# Pad slots per stage spacing; a peer runs at most this far ahead of the ranks it writes to.
_SLOTS = 4 * (1 + _GAP)


def attn_out_call(c: PFConfig, tm: int, axis: str = "model"):
    """(layer [1] i32, o [T, qd] bf16, gate [T, qd] bf16, x [T, H] f32, and the stacked
    wo [L, qd, H] fp8, wr [L, H, Ep] bf16, wp [L, H, He/tp] fp8, sm [L, 16, H] f32) ->
    (x' [T, H] f32, xe [T, He] bf16 expert input, ids [T, 128] i32, tw [T, 128] f32); lanes
    past top_k of ids / tw are 0."""
    T, H, tp = c.rows, c.hidden, c.tp
    qd, hd = c.q_dim, c.head_dim
    Hl, He = H // tp, c.expert_hidden
    Hel = He // tp
    E, K = c.n_experts, c.top_k
    Ep = -(-E // LANE) * LANE
    nchip = tp // 2
    nt = T // tm
    if T % tm or tm % 8:
        raise ValueError(f"{T} rows do not tile by {tm}")
    fp8 = c.fp8_gather
    kinds = (
        RS_KINDS + ag_kinds("ag") + ag_kinds("xe") + (ag_kinds("xsc") if fp8 else ())
    )

    def kernel(
        lyr,
        o_hbm,
        g_hbm,
        x_hbm,
        wo_hbm,
        wr_hbm,
        wp_hbm,
        sm_hbm,
        xn_hbm,
        xe_hbm,
        ids_hbm,
        tw_hbm,
        wo_v,
        wr_v,
        wp_v,
        sm_v,
        o_v,
        g_v,
        x_v,
        send_v,
        rsp_pad,
        ps_v,
        rsc_pad,
        ag_pad,
        xe_pad,
        xn_v,
        xe_v,
        ids_v,
        tw_v,
        w_sem,
        in_sem,
        x_sem,
        out_sem,
        *rest,
    ):
        xsc_pad, coll_sems = (rest[0], rest[1:]) if fp8 else (None, rest)
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = PairColl(
            tp, me, {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)}
        )
        col.barrier()

        l = lyr[0]
        w_cps = [
            pltpu.make_async_copy(s.at[l], d, w_sem.at[i])
            for i, (s, d) in enumerate(
                ((wo_hbm, wo_v), (wr_hbm, wr_v), (wp_hbm, wp_v), (sm_hbm, sm_v))
            )
        ]
        for cp in w_cps:
            cp.start()

        def rows(j):
            return pl.ds(pl.multiple_of(j * tm, tm), tm)

        def in_copies(j):
            b = j % 2
            return (
                pltpu.make_async_copy(o_hbm.at[rows(j)], o_v.at[b], in_sem.at[0, b]),
                pltpu.make_async_copy(g_hbm.at[rows(j)], g_v.at[b], in_sem.at[1, b]),
            )

        def x_copy(j):
            return pltpu.make_async_copy(
                x_hbm.at[rows(j)], x_v.at[j % 2], x_sem.at[j % 2]
            )

        def out_copies(j):
            """The stream and routing rows of tile j (written by s5)."""
            b = j % 2
            return tuple(
                pltpu.make_async_copy(v.at[b], h.at[rows(j)], out_sem.at[i, b])
                for i, (v, h) in enumerate(
                    ((xn_v, xn_hbm), (ids_v, ids_hbm), (tw_v, tw_hbm))
                )
            )

        def xe_out_copy(j):
            b = j % 2
            return pltpu.make_async_copy(
                xe_v.at[b], xe_hbm.at[rows(j)], out_sem.at[3, b]
            )

        for cp in in_copies(0):
            cp.start()
        for cp in w_cps:
            cp.wait()
        sm = sm_v[...]
        row = lambda name: sm[SMALL[name] : SMALL[name] + 1]
        ag_kp = (("ag", ag_pad),)
        xe_kp = (("xe", xe_pad),) + ((("xsc", xsc_pad),) if fp8 else ())

        def s1(j):
            """o projection of tile j; the partner core's column blocks go to the partner."""
            for cp in in_copies(j):
                cp.wait()

            @pl.when(j + 1 < nt)
            def _():
                for cp in in_copies(j + 1):
                    cp.start()

            b = j % 2
            o = _heads_rms(o_v[b], c.q_heads, hd, c.rms_eps)
            a = (jax.nn.sigmoid(g_v[b].astype(jnp.float32)) * o).astype(jnp.bfloat16)
            part = jnp.dot(a, wo_v[...], preferred_element_type=jnp.float32) * row(
                "wo_s"
            )
            send_v[b] = part.astype(jnp.bfloat16)
            for cp in col.rs_pair(j % _SLOTS, send_v.at[b], rsp_pad, Hl):
                cp.start()

        def s2(j):
            """Pair sums; RS phase 1."""
            s, b = j % _SLOTS, j % 2
            wait_all(col.rs_pair(s, send_v.at[b], rsp_pad, Hl))
            for ch in range(nchip):
                ps_v[s, ch] = col.rs_pair_sum(s, send_v.at[b], rsp_pad, Hl, ch).astype(
                    jnp.bfloat16
                )
            for cp in col.rs2_first(s, ps_v.at[s], rsc_pad):
                cp.start()

        def s3a(j):
            """RS phase 1 landed: the second neighbour's partial plus the relayed one go out."""
            s = j % _SLOTS
            _, m2 = col.masks()
            other = jnp.bitwise_xor(col.chip, m2)
            wait_all(col.rs2_first(s, ps_v.at[s], rsc_pad))
            rsc_pad[s, 3] = (
                ps_v[s, other].astype(jnp.float32) + rsc_pad[s, 1].astype(jnp.float32)
            ).astype(jnp.bfloat16)
            for cp in col.rs2_second(s, rsc_pad):
                cp.start()

        def s3(j):
            """This rank's reduced block of tile j; its all-gather starts."""
            s = j % _SLOTS
            wait_all(col.rs2_second(s, rsc_pad))
            red = (
                ps_v[s, col.chip].astype(jnp.float32)
                + rsc_pad[s, 0].astype(jnp.float32)
                + rsc_pad[s, 2].astype(jnp.float32)
            )
            ag_pad[s, me] = red.astype(jnp.bfloat16)
            col.ag_start(ag_kp, s)

        def s4(j):
            col.ag_to_partner(ag_kp, j % _SLOTS)
            x_copy(j).start()

        def s5(j):
            """Residual, router top-k and this rank's expert-projection columns of tile j."""
            s, b = j % _SLOTS, j % 2
            col.ag_wait(ag_kp, s)
            x_copy(j).wait()
            attn = gathered(ag_pad, None, s, tp)
            x = row("attn_alpha") * x_v[b] + row("attn_beta") * _rms(attn, c.post_eps)

            @pl.when(j >= 2)
            def _():
                for cp in out_copies(j - 2):
                    cp.wait()

            xn_v[b] = x
            r_in = (_rms(x, c.rms_eps) * row("pre_ffn_w")).astype(jnp.bfloat16)
            logits = jnp.dot(r_in, wr_v[...], preferred_element_type=jnp.float32)
            scores = jax.nn.sigmoid(logits * c.router_mult)
            lane = jax.lax.broadcasted_iota(jnp.int32, (tm, Ep), 1)
            scores = jnp.where(lane < E, scores, 0.0)
            ids, picks = _topk_iter(scores + row("expert_bias")[:, :Ep], K)
            tws = [
                jnp.sum(jnp.where(pk, scores, 0.0), axis=1, keepdims=True)
                for pk in picks
            ]
            denom = sum(tws) + c.router_eps
            lane8 = jax.lax.broadcasted_iota(jnp.int32, (tm, LANE), 1)
            ids_t = jnp.zeros((tm, LANE), jnp.int32)
            tw_t = jnp.zeros((tm, LANE), jnp.float32)
            for k in range(K):
                ids_t = jnp.where(lane8 == k, ids[k], ids_t)
                tw_t = jnp.where(lane8 == k, tws[k] / denom * c.route_scale, tw_t)
            ids_v[b] = ids_t
            tw_v[b] = tw_t
            for cp in out_copies(j):
                cp.start()
            xe = jnp.dot(r_in, wp_v[...], preferred_element_type=jnp.float32)
            xe = (xe * row("wpool_s")[:, :Hel]).astype(jnp.bfloat16)
            if fp8:
                xe_pad[s, me], xsc_pad[s, me] = to_fp8(xe.astype(jnp.float32))
            else:
                xe_pad[s, me] = xe
            col.ag_start(xe_kp, s)

        def s6(j):
            col.ag_to_partner(xe_kp, j % _SLOTS)

        def s7(j):
            """Gathered expert input of tile j, pre-expert normed."""
            s, b = j % _SLOTS, j % 2
            col.ag_wait(xe_kp, s)
            xe = gathered(xe_pad, xsc_pad, s, tp)
            gain = sm[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He]

            @pl.when(j >= 2)
            def _():
                xe_out_copy(j - 2).wait()

            xe_v[b] = (_rms(xe, c.rms_eps) * gain).astype(jnp.bfloat16)
            xe_out_copy(j).start()

        s4a = lambda j: col.ag_relay(ag_kp, j % _SLOTS)
        s6a = lambda j: col.ag_relay(xe_kp, j % _SLOTS)
        run_pipeline((s1, s2, s3a, s3, s4a, s4, s5, s6a, s6, s7), nt, gap=_GAP)
        for j in range(max(0, nt - 2), nt):
            for cp in (*out_copies(j), xe_out_copy(j)):
                cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    scratch = [
        vmem((qd, H), jnp.float8_e4m3fn),
        vmem((H, Ep), jnp.bfloat16),
        vmem((H, Hel), jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, H), jnp.bfloat16),
        vmem((_SLOTS, nchip, tm, Hl), jnp.bfloat16),
        vmem((_SLOTS, nchip, tm, Hl), jnp.bfloat16),
        vmem((_SLOTS, nchip, tm, Hl), jnp.bfloat16),
        vmem((_SLOTS, tp, tm, Hl), jnp.bfloat16),
        vmem((_SLOTS, tp, tm, Hel), jnp.float8_e4m3fn if fp8 else jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, He), jnp.bfloat16),
        vmem((2, tm, LANE), jnp.int32),
        vmem((2, tm, LANE), jnp.float32),
        sem((4,)),
        sem((2, 2)),
        sem((2,)),
        sem((4, 2)),
    ]
    scratch += [vmem((_SLOTS, tp, 8, LANE), jnp.float32)] if fp8 else []
    scratch += [sem((_SLOTS,)) for _ in range(2 * len(kinds))]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)

    def call(lyr, o, g, x, wo, wr, wp, sm):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((T, H), jnp.float32),
                jax.ShapeDtypeStruct((T, He), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, LANE), jnp.int32),
                jax.ShapeDtypeStruct((T, LANE), jnp.float32),
            ),
            in_specs=[smem] + [hbm] * 7,
            out_specs=(hbm,) * 4,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=PREFILL_ATTN_OUT_COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
            ),
            name=f"muse_prefill_attn_out_t{T}_m{tm}{'_fp8ag' if fp8 else ''}",
        )(lyr, o, g, x, wo, wr, wp, sm)

    return call
