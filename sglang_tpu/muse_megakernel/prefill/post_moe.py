# SPDX-License-Identifier: Apache-2.0
"""Prefill kernel between a layer's MoE and the next layer's attention.

Fuses the all-gather of the token-sharded MoE output, the post-expert projection and its
all-gather, the gated residual and the next layer's qkvg projection with its qk norms and
rope. Tile j holds rows j * tm / tp onward of every rank's token block, so every rank sends
in every tile.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import (
    LANE,
    PREFILL_POST_MOE_COLLECTIVE_ID,
    SMALL,
)
from sglang_tpu.muse_megakernel.mk.numerics import _heads_rms, _rms, _rope_roll
from sglang_tpu.muse_megakernel.prefill.coll import (
    PairColl,
    ag_kinds,
    gathered,
    run_pipeline,
    to_fp8,
)
from sglang_tpu.muse_megakernel.prefill.config import PFConfig

# Pad slots, checked against the peers' wait chains of the two-phase all-gathers.
_SLOTS = 5


def post_moe_call(c: PFConfig, tm: int, *, next_rope: bool | None, axis: str = "model"):
    """(layer [1] i32, y [T/tp, He] bf16 this rank's MoE rows, x [T, H] f32, rope [T, 2 hd]
    f32, and the stacked wu [L, He, H/tp] fp8, sm [L, 16, H], wqkvg [L, H, 2qd + 2kvd] fp8)
    -> (x' [T, H] f32, q [T, qd] bf16, kv [T, 2 kvd] bf16, gate [T, qd] bf16) of the next
    layer. next_rope None: the last layer, whose q, kv and gate are not written."""
    T, H, tp = c.rows, c.hidden, c.tp
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    Hl, He = H // tp, c.expert_hidden
    nq = c.qkvg_dim
    seg = tm // tp
    blk = T // tp
    nt = blk // seg
    with_next = next_rope is not None
    if blk % seg or seg % 16:
        raise ValueError(f"{T} rows do not tile by {tm} over tp {tp}")
    fp8 = c.fp8_gather
    kinds = ag_kinds("y") + ag_kinds("u") + (ag_kinds("usc") if fp8 else ())

    def kernel(
        lyr,
        y_hbm,
        x_hbm,
        rope_hbm,
        wu_hbm,
        sm_hbm,
        wq_hbm,
        xn_hbm,
        q_hbm,
        kv_hbm,
        g_hbm,
        wu_v,
        sm_v,
        wq_v,
        smn_v,
        y_pad,
        u_pad,
        x_v,
        rope_v,
        q_v,
        kv_v,
        g_v,
        w_sem,
        y_sem,
        x_sem,
        out_sem,
        *rest,
    ):
        usc_pad, coll_sems = (rest[0], rest[1:]) if fp8 else (None, rest)
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = PairColl(
            tp, me, {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)}
        )
        col.barrier()
        l = lyr[0]
        pairs = [(wu_hbm.at[l], wu_v), (sm_hbm.at[l], sm_v)]
        if with_next:
            pairs += [(wq_hbm.at[l + 1], wq_v), (sm_hbm.at[l + 1], smn_v)]
        w_cps = [
            pltpu.make_async_copy(a, b, w_sem.at[i]) for i, (a, b) in enumerate(pairs)
        ]
        for cp in w_cps:
            cp.start()

        def seg_rows(j, r):
            return pl.ds(pl.multiple_of(r * blk + j * seg, seg), seg)

        def tile_rows(r):
            return pl.ds(r * seg, seg)

        def y_load(j):
            s = j % _SLOTS
            return pltpu.make_async_copy(
                y_hbm.at[pl.ds(pl.multiple_of(j * seg, seg), seg)],
                y_pad.at[s, me],
                y_sem.at[s % 2],
            )

        def x_loads(j):
            b = j % 2
            srcs = [(x_hbm, x_v, 0)] + (
                [(rope_hbm, rope_v, 1)] if with_next and next_rope else []
            )
            return [
                pltpu.make_async_copy(
                    h.at[seg_rows(j, r)], v.at[b, tile_rows(r)], x_sem.at[i, b]
                )
                for h, v, i in srcs
                for r in range(tp)
            ]

        def out_writes(j):
            b = j % 2
            dsts = [(x_v, xn_hbm)] + (
                [(q_v, q_hbm), (kv_v, kv_hbm), (g_v, g_hbm)] if with_next else []
            )
            return [
                pltpu.make_async_copy(
                    v.at[b, tile_rows(r)], h.at[seg_rows(j, r)], out_sem.at[i, b]
                )
                for i, (v, h) in enumerate(dsts)
                for r in range(tp)
            ]

        y_load(0).start()
        for cp in w_cps:
            cp.wait()
        sm = sm_v[...]
        y_kp = (("y", y_pad),)
        u_kp = (("u", u_pad),) + ((("usc", usc_pad),) if fp8 else ())

        def s1(j):
            y_load(j).wait()

            @pl.when(j + 1 < nt)
            def _():
                y_load(j + 1).start()

            col.ag_start(y_kp, j % _SLOTS)

        def s3(j):
            """Post-expert projection of tile j: this rank's output columns."""
            s = j % _SLOTS
            col.ag_wait(y_kp, s)
            y = jnp.concatenate([y_pad[s, r] for r in range(tp)], axis=0)
            u = jnp.dot(y, wu_v[...], preferred_element_type=jnp.float32)
            u = (u * _row(sm, "wunpool_s")[:, :Hl]).astype(jnp.bfloat16)
            if fp8:
                u_pad[s, me], usc_pad[s, me] = to_fp8(u.astype(jnp.float32))
            else:
                u_pad[s, me] = u
            col.ag_start(u_kp, s)

        def s4(j):
            col.ag_to_partner(u_kp, j % _SLOTS)

            # x_v is double-buffered across s4..s5
            @pl.when(j >= 2)
            def _():
                for cp in out_writes(j - 2):
                    cp.wait()

            for cp in x_loads(j):
                cp.start()

        def s5(j):
            """Residual of tile j, then the next layer's attention inputs."""
            s, b = j % _SLOTS, j % 2
            col.ag_wait(u_kp, s)
            for cp in x_loads(j):
                cp.wait()
            u = gathered(u_pad, usc_pad, s, tp)
            u = _rms(u.astype(jnp.float32) * _row(sm, "post_ffn_w"), c.post_eps)
            x = _row(sm, "ffn_alpha") * x_v[b] + _row(sm, "ffn_beta") * u
            x_v[b] = x
            if with_next:
                q_v[b], kv_v[b], g_v[b] = attention_inputs(
                    c, x, smn_v[...], wq_v[...], rope_v[b] if next_rope else None
                )
            for cp in out_writes(j):
                cp.start()

        s2a = lambda j: col.ag_relay(y_kp, j % _SLOTS)
        s2 = lambda j: col.ag_to_partner(y_kp, j % _SLOTS)
        s4a = lambda j: col.ag_relay(u_kp, j % _SLOTS)
        run_pipeline((s1, s2a, s2, s3, s4a, s4, s5), nt)
        for j in range(max(0, nt - 2), nt):
            for cp in out_writes(j):
                cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    wq_shape = (H, nq) if with_next else (8, LANE)
    scratch = [
        vmem((He, Hl), jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem(wq_shape, jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem((_SLOTS, tp, seg, He), jnp.bfloat16),
        vmem((_SLOTS, tp, tm, Hl), jnp.float8_e4m3fn if fp8 else jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, 2 * hd), jnp.float32),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, 2 * kvd), jnp.bfloat16),
        vmem((2, tm, qd), jnp.bfloat16),
        sem((4,)),
        sem((2,)),
        sem((2, 2)),
        sem((4, 2)),
    ]
    scratch += [vmem((_SLOTS, tp, 8, LANE), jnp.float32)] if fp8 else []
    scratch += [sem((_SLOTS,)) for _ in range(2 * len(kinds))]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)

    def call(lyr, y, x, rope, wu, sm, wq):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((T, H), jnp.float32),
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
            ),
            in_specs=[smem] + [hbm] * 6,
            out_specs=(hbm,) * 4,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=PREFILL_POST_MOE_COLLECTIVE_ID,
                vmem_limit_bytes=60 * 2**20,
            ),
            name=(
                f"muse_prefill_post_moe_t{T}_m{tm}_r{next_rope}{'_fp8ag' if fp8 else ''}"
            ),
        )(lyr, y, x, rope, wu, sm, wq)

    return call


def _row(sm, name: str):
    return sm[SMALL[name] : SMALL[name] + 1]


def attention_inputs(c: PFConfig, x, smn, wq, rope):
    """A layer's (q, kv, gate) rows from its input stream x; rope None: a NoPE layer."""
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    a_in = (_rms(x, c.rms_eps) * _row(smn, "pre_attn_w")).astype(jnp.bfloat16)
    qkvg = jnp.dot(a_in, wq, preferred_element_type=jnp.float32)
    qkvg = qkvg * _row(smn, "wqkvg_s")[:, : c.qkvg_dim]
    q = _heads_rms(qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by)
    k = _heads_rms(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
    if rope is not None:
        q = _rope_roll(q, c.q_heads, hd, rope[:, :hd], rope[:, hd:])
        k = _rope_roll(k, c.kv_heads, hd, rope[:, :hd], rope[:, hd:])
    kv = jnp.concatenate([k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1)
    return (
        q.astype(jnp.bfloat16),
        kv.astype(jnp.bfloat16),
        qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16),
    )
