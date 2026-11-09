# SPDX-License-Identifier: Apache-2.0
"""The EP plan of one row-tiled step: whole experts per rank, I-chunks that fit the TP
expert rings, the routing table and the scratch the expert section adds."""

from typing import NamedTuple

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.experts import BD_KC, item_sizes
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE


class EPPlan(NamedTuple):
    experts: int  # EL: whole experts per rank
    inter: int  # an expert's whole intermediate
    chunk_fp4: int  # intermediate columns per fp4 chunk: one TP expert-ring slot
    chunk_fp8: int  # fp8 chunks take the same slot bytes
    item_rows: int  # rows of one work item; a busier expert takes several
    slots_pad: int  # routing-table width: an expert's slots <= live rows <= R
    max_items: int
    nslot: int  # weight-chunk ring slots (the TP expert ring's)

    def n_chunks(self, kind: str) -> int:
        return self.inter // (self.chunk_fp4 if kind == "fp4" else self.chunk_fp8)


def ep_plan(c: MKConfig, item_rows: int = 64) -> EPPlan:
    assert c.row_tile, "the EP expert section is row-tiled only"
    assert not c.y_quant and not c.scale_fp8 and c.n_experts % c.tp == 0
    assert c.inter_local % BD_KC == 0 and c.expert_hidden % BD_KC == 0
    item_sizes(item_rows)
    el = c.n_experts // c.tp
    return EPPlan(
        experts=el,
        inter=c.inter_local * c.tp,
        chunk_fp4=c.inter_local,
        chunk_fp8=c.inter_local // 2,
        item_rows=item_rows,
        slots_pad=-(-c.rows // LANE) * LANE,
        max_items=el + -(-c.rows * c.top_k // item_rows),
        nslot=c.expert_depth + 1,
    )


def ep_expert_shapes(
    c: MKConfig, kind: str
) -> tuple[tuple[tuple[int, ...], object], ...]:
    """(shape, dtype) of one layer's (w13, w13_scale, w2, w2_scale) for this rank's whole
    experts: the TP slab layout with E / tp experts of the whole intermediate."""
    el, he, i = c.n_experts // c.tp, c.expert_hidden, c.inter_local * c.tp
    if kind == "fp4":
        return (
            ((el, he // 8, 2 * i), jnp.int32),
            ((el, he // 16, 2 * i), jnp.bfloat16),
            ((el, i // 8, he), jnp.int32),
            ((el, i // 16, he), jnp.bfloat16),
        )
    return (
        ((el, he // 4, 2 * i), jnp.int32),
        ((el, 2, 2 * i), jnp.bfloat16),
        ((el, i // 4, he), jnp.int32),
        ((el, 2, he), jnp.bfloat16),
    )


def ep_scratch(c: MKConfig, p: EPPlan) -> list:
    """Buffers the EP expert section adds to the row-tiled plan."""
    ri, he = p.item_rows, c.expert_hidden
    vmem, smem = pltpu.VMEM, pltpu.SMEM
    dma = pltpu.SemaphoreType.DMA
    route_rows = -(-(p.experts + 1) // 8) * 8
    return [
        ("ep_x", vmem((2, ri, he // 2), jnp.uint32)),  # an item's x rows, bf16 pairs
        ("ep_acc", vmem((2, ri, he), jnp.float32)),
        ("ep_y", vmem((2, ri, he // 2), jnp.uint32)),  # outgoing y rows, bf16 pairs
        # slot j of local expert e on lane e
        ("ep_tab", vmem((p.slots_pad, LANE), jnp.int32)),
        # rows < EL: expert e's slots (row * K + slot); row EL: the slot counts
        ("ep_route_v", vmem((route_rows, p.slots_pad), jnp.int32)),
        ("ep_route_s", smem((route_rows, p.slots_pad), jnp.int32)),
        ("ep_items", smem((1 + 3 * p.max_items,), jnp.int32)),
        # outstanding y row DMAs per staging buffer; this rank's own rows it fed itself
        ("ep_cnt", smem((8,), jnp.int32)),
        ("ep_w_sems", dma((p.nslot, 6))),
        ("ep_x_sems", dma((2,))),
        ("ep_ys_sems", dma((2,))),
        ("ep_yr_sem", dma((1,))),
        ("ep_route_sem", dma((1,))),
    ]
