# SPDX-License-Identifier: Apache-2.0
"""The EP plan of one row-tiled step: whole experts per rank, I-chunks that fit the TP
expert rings, the routing table and the scratch the expert section adds."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.experts import BD_KC, item_sizes
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE


class EPBuffers(NamedTuple):
    """An item's VMEM rows: x (two items, bf16 pairs), the f32 accumulator, outgoing y."""

    x: Callable[[Any], Any]  # x(b): [item_rows, H_e / 2] u32 of item parity b
    acc: Any  # [item_rows, H_e] f32
    y: Any  # [item_rows, H_e / 2] u32


class EPPlan(NamedTuple):
    experts: int  # EL: whole experts per rank
    inter: int  # an expert's whole intermediate
    chunk_fp4: int  # intermediate columns per fp4 chunk: one TP expert-ring slot
    chunk_fp8: int  # an fp8 chunk takes two adjacent ring slots (as TP's fp8 slabs)
    item_rows: int  # rows of one work item; a busier expert takes several
    slots_pad: int  # routing-table width: an expert's slots <= live rows <= R
    max_items: int
    nslot: int  # weight-chunk ring slots (the TP expert ring's)

    def n_chunks(self, kind: str) -> int:
        return self.inter // self.chunk(kind)

    def chunk(self, kind: str) -> int:
        return self.chunk_fp4 if kind == "fp4" else self.chunk_fp8

    def slots(self, kind: str) -> int:
        return self.nslot if kind == "fp4" else self.nslot // 2


def ep_mk_config(c: MKConfig) -> MKConfig:
    """The step's MKConfig for the EP kernel: the TP expert section's VMEM y buffers (ysend,
    pad_y, their pair rounds) go unused, so the plan keeps them small (pad_y in HBM); no o-proj
    fold into the router, whose per-rank block order would let ranks route a near tie apart."""
    return msgspec.structs.replace(
        c, tiled_vmem_y=0, tiled_pair_y=0, pair_y_stream=False, fold_o=False
    )


def ep_plan(c: MKConfig, item_rows: int = 32) -> EPPlan:
    assert c.row_tile, "the EP expert section is row-tiled only"
    assert not c.scale_fp8 and c.n_experts % c.tp == 0
    assert c.inter_local % BD_KC == 0 and c.expert_hidden % BD_KC == 0
    item_sizes(item_rows)
    el = c.n_experts // c.tp
    return EPPlan(
        experts=el,
        inter=c.inter_local * c.tp,
        chunk_fp4=c.inter_local,
        chunk_fp8=c.inter_local,
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


def _item_rows_in_lhs(c: MKConfig, p: EPPlan) -> bool:
    """The TP expert phase stages its LHS in S["lhs"] ([R, H] bf16), free while experts run:
    the item buffers fit there from 128 rows on, where VMEM runs short."""
    return c.rows // 2 >= 2 * p.item_rows and c.hidden >= 2 * c.expert_hidden


def ep_buffers(c: MKConfig, p: EPPlan, S: dict) -> EPBuffers:
    ri, he = p.item_rows, c.expert_hidden
    if not _item_rows_in_lhs(c, p):
        return EPBuffers(lambda b: S["ep_x"].at[b], S["ep_acc"], S["ep_y"])
    lhs = S["lhs"].bitcast(jnp.uint32)  # [R / 2, H]
    return EPBuffers(
        x=lambda b: lhs.at[pl.ds(b * ri, ri), pl.ds(0, he // 2)],
        acc=lhs.bitcast(jnp.float32).at[pl.ds(0, ri), pl.ds(he // 2, he)],
        y=lhs.at[pl.ds(0, ri), pl.ds(he // 2 + he, he // 2)],
    )


def ep_scratch(c: MKConfig, p: EPPlan) -> list:
    """Buffers the EP expert section adds to the row-tiled plan."""
    ri, he = p.item_rows, c.expert_hidden
    vmem, smem = pltpu.VMEM, pltpu.SMEM
    dma = pltpu.SemaphoreType.DMA
    route_rows = (p.slots_pad // LANE) * (-(-p.experts // 8) * 8) + 8
    items = []
    if not _item_rows_in_lhs(c, p):
        items += [
            (
                "ep_x",
                vmem((2, ri, he // 2), jnp.uint32),
            ),  # an item's x rows, bf16 pairs
            ("ep_acc", vmem((ri, he), jnp.float32)),  # packed into ep_y before the next
            # outgoing y rows, bf16 pairs: an item's sends have long landed when the next packs
            ("ep_y", vmem((ri, he // 2), jnp.uint32)),
        ]
    return items + [
        # this rank's rows' y, slot by slot, as the experts' ranks send them (HBM: VMEM is full)
        ("ep_pad", pltpu.HBM((c.rows_per_dev, c.top_k, he // 2), jnp.uint32)),
        (
            "ep_yrow",
            vmem((2, c.top_k, he // 2), jnp.uint32),
        ),  # one row's slots, for the combine
        ("ep_yrow_sems", dma((2,))),
        # slot j of local expert e on lane e
        ("ep_tab", vmem((p.slots_pad, LANE), jnp.int32)),
        # route.local_slots' table: experts' (row * K + slot) per LANE-slot block, counts
        ("ep_route_v", vmem((route_rows, LANE), jnp.int32)),
        ("ep_route_s", smem((route_rows, LANE), jnp.int32)),
        ("ep_items", smem((1 + 3 * p.max_items,), jnp.int32)),
        # outstanding y row DMAs per staging buffer; this rank's own rows it fed itself
        ("ep_cnt", smem((8,), jnp.int32)),
        # y rows per owner rank: counted here, sent as one row each, received from each rank
        ("ep_cnt_s", smem((c.tp,), jnp.int32)),
        ("ep_cout", vmem((c.tp, LANE), jnp.int32)),
        ("ep_cin", vmem((c.tp, LANE), jnp.int32)),
        ("ep_cin_s", smem((c.tp, LANE), jnp.int32)),
        ("ep_cin_sem", dma((1,))),
        ("ep_cout_sem", dma((1,))),
        ("ep_w_sems", dma((p.nslot, 6))),
        ("ep_x_sems", dma((2,))),
        ("ep_ys_sems", dma((1,))),
        ("ep_yr_sem", dma((1,))),
        ("ep_route_sem", dma((1,))),
    ]
