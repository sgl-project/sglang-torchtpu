# SPDX-License-Identifier: Apache-2.0
"""Pallas kernels of the DSpark draft (server/draft.py): the greedy Markov chain in one kernel.

At decode batch the draft is bound by per-op latency, so the 7 dependent Markov steps
(gather w1[prev], bias = w2 @ it, argmax over the vocab shard, exchange with the group) run
inside one kernel with the w2 shard resident in VMEM instead of as ~8 XLA ops per step.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import Topology
from sglang_tpu.k3_megakernel.kernel.collectives import (
    AllGather,
    AllGatherSpec,
    entry_barrier,
)
from sglang_tpu.k3_megakernel.mesh import AXIS

COLLECTIVE_ID = (
    10  # Pallas barrier id of the Markov kernel (draft AllReduce: 9, megakernel: 4)
)
_LANE = 128
_BIG = 3.0e38
_W1_ROWS = 16  # w1 rows per DMA: one bf16 sublane tile, aligned


def group_topology(group: int) -> Topology:
    return Topology(group // 8, 8) if group >= 8 else Topology(1, group)


def markov_chain(base, w2t, w1, anchor, *, group, ndraft):
    """Greedy Markov drafts (local view inside shard_map).

    base [B, Vs] bf16 (row j: draft j's base logits over this rank's vocab shard, B >= ndraft),
    w2t [R, Vs] bf16 (this rank's markov_w2 rows, transposed), w1 [V, R] bf16 (full), anchor int32.
    d_j = argmax(bf16(base_j + bf16(w2 @ w1[d_{j-1}]))) over the ``group`` ranks' shards, d_0 = anchor,
    lowest id on ties (XLA argmax order). -> [ndraft] int32.
    """
    B, vs = base.shape
    R = w2t.shape[0]
    topo = group_topology(group)
    ex = AllGatherSpec(topo, 1, jnp.float32, flat=True, cols=_LANE)

    def kernel(
        anchor_ref,
        base_ref,
        w2t_hbm,
        w1_hbm,
        out_ref,
        ebuf,
        sem,
        prev_ref,
        w2t_ref,
        *ex_scratch,
    ):
        me = jax.lax.axis_index(AXIS)
        w2cp = pltpu.make_async_copy(w2t_hbm, w2t_ref, sem.at[1])
        w2cp.start()
        ag = AllGather(ex, me, *ex_scratch)
        ag.init()
        entry_barrier(me, topo)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, vs), 1).astype(jnp.float32)
        shard0 = ((me % group) * vs).astype(jnp.float32)
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, _LANE), 1)
        sub = jax.lax.broadcasted_iota(jnp.int32, (_W1_ROWS, 1), 0)
        prev_ref[0] = anchor_ref[0]
        out = jnp.zeros((1, _LANE), jnp.int32)
        for j in range(ndraft):
            if j == 0:
                w2cp.wait()
            prev = prev_ref[0]
            start = pl.multiple_of((prev // _W1_ROWS) * _W1_ROWS, _W1_ROWS)
            cp = pltpu.make_async_copy(
                w1_hbm.at[pl.ds(start, _W1_ROWS)], ebuf, sem.at[0]
            )
            cp.start()
            cp.wait()
            e = jnp.sum(
                jnp.where(sub == prev % _W1_ROWS, ebuf[...].astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)  # [1, R], exact row pick
            bias = jax.lax.dot_general(
                e,
                w2t_ref[...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [1, vs]
            z = base_ref[pl.ds(j, 1), :].astype(jnp.float32) + bias.astype(
                jnp.bfloat16
            ).astype(jnp.float32)
            z = z.astype(jnp.bfloat16).astype(jnp.float32)
            m = jnp.max(z, axis=1, keepdims=True)
            idx = jnp.min(jnp.where(z == m, col, _BIG), axis=1, keepdims=True) + shard0
            packed = jnp.where(lane == 0, m, jnp.where(lane == 1, idx, 0.0))
            row = ag(packed)  # [1, group * 128], group-rank order
            rows = jnp.concatenate(
                [row[:, r * _LANE : (r + 1) * _LANE] for r in range(group)], axis=0
            )
            lanes = jax.lax.broadcasted_iota(jnp.int32, rows.shape, 1)
            vals = jnp.sum(jnp.where(lanes == 0, rows, 0.0), axis=1, keepdims=True)
            ids = jnp.sum(jnp.where(lanes == 1, rows, 0.0), axis=1, keepdims=True)
            tok = jnp.min(jnp.where(vals == jnp.max(vals), ids, _BIG)).astype(jnp.int32)
            prev_ref[0] = tok
            out = jnp.where(lane == j, tok, out)
        out_ref[...] = out

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    w2_bytes = R * vs * 2
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, _LANE), jnp.int32),
        in_specs=[smem, vmem, hbm, hbm],
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((_W1_ROWS, R), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SMEM((1,), jnp.int32),
            pltpu.VMEM((R, vs), jnp.bfloat16),
        ]
        + ex.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=int(w2_bytes * 1.5 + B * vs * 2 * 2 + (24 << 20)),
        ),
    )(jnp.reshape(anchor, (1,)).astype(jnp.int32), base, w2t, w1)
    return out[0, :ndraft]
