# SPDX-License-Identifier: Apache-2.0
"""The EP plan of one row-tiled step: each rank's own rows (R / tp), 32 whole experts per
rank, the two ragged all-to-alls' buffers and the scratch the FFN section adds."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.experts import BD_KC, item_sizes
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE
from sglang_tpu.muse_megakernel.mk.plan import fit_ring_slots

MIN_ROWS = 64  # EP serves the large-row steps only
TAB_ROWS = 32  # route.slot_tables' rows
SEG = 8  # a2a segments start on 8-row (HBM tile) boundaries and move in 8-row DMAs
DN_TK, DN_TN = 2048, 1024  # a replicated pool / unpool tile: one fp4 expert-ring slot


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
    rows_own: int  # Rc: the rows this rank owns
    rows_own_padded: int  # Rc rounded up to SEG: one o-proj reduce-scatter block
    send_rows: (
        int  # the send / return buffer: own slots, each destination's segment padded
    )
    recv_rows: int  # the receive buffer: every row's slots could land on one rank
    max_items: int
    nslot: int  # weight-chunk ring slots (the TP expert ring's)

    def n_chunks(self, kind: str) -> int:
        return self.inter // self.chunk(kind)

    def chunk(self, kind: str) -> int:
        return self.chunk_fp4 if kind == "fp4" else self.chunk_fp8

    def slots(self, kind: str) -> int:
        return self.nslot if kind == "fp4" else self.nslot // 2


def ep_mk_config(c: MKConfig) -> MKConfig:
    """The step's MKConfig for the EP kernel: replicated pool / unpool stream outside the
    dense ring (refitted to the tiles left); no folds into the router or the next qkvg (the
    FFN section runs on own rows and ends in a row all-gather); the TP expert section's VMEM
    y buffers go unused (pad_y in HBM)."""
    c = msgspec.structs.replace(
        c,
        dense_skip=("wpool", "wunpool"),
        fold_o=False,
        fold_f=False,
        tiled_vmem_y=0,
        tiled_pair_y=0,
        pair_y_stream=False,
    )
    slots = fit_ring_slots(c, cap=c.ring_slots)
    return msgspec.structs.replace(
        c, ring_slots=slots, stream_depth=min(c.stream_depth, slots - 1)
    )


def ep_plan(c: MKConfig, item_rows: int = 32) -> EPPlan:
    assert c.row_tile and c.rows >= MIN_ROWS, "EP serves row-tiled steps of >= 64 rows"
    assert not c.scale_fp8 and c.n_experts % c.tp == 0 and c.rows % c.tp == 0
    assert c.inter_local % BD_KC == 0 and c.expert_hidden % BD_KC == 0
    assert c.n_experts_padded == c.n_experts and c.expert_hidden % DN_TN == 0
    assert c.hidden % DN_TK == 0 and c.expert_hidden % DN_TK == 0
    item_sizes(item_rows)
    el, rc, k = c.n_experts // c.tp, c.rows // c.tp, c.top_k
    pad = (SEG - 1) * c.tp
    # route.build_items packs a received row (12 bits), its source and its return row
    assert c.rows * k + pad <= 4096 and c.tp <= 8, (c.rows, c.tp)
    return EPPlan(
        experts=el,
        inter=c.inter_local * c.tp,
        chunk_fp4=c.inter_local,
        chunk_fp8=c.inter_local,
        item_rows=item_rows,
        rows_own=rc,
        rows_own_padded=-(-rc // SEG) * SEG,
        send_rows=-(-(rc * k + pad) // SEG) * SEG,
        recv_rows=-(-(c.rows * k + pad) // SEG) * SEG,
        max_items=el + -(-c.rows * k // item_rows),
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


def window_rows(c: MKConfig) -> int:
    """Rows of the 16-row aligned window (a bf16 row tile) around any rank's own rows."""
    rc = c.rows // c.tp
    reach = max((me * rc) % 16 for me in range(c.tp)) + rc
    return min(-(-reach // 16) * 16, c.rows)


def _item_rows_in_lhs(c: MKConfig, p: EPPlan) -> bool:
    """S["lhs"] ([R, H] bf16, every row's pre-FFN input) is free once the pool has read this
    rank's rows: the item buffers fit there from 128 rows on, where VMEM runs short."""
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
    """Buffers the EP FFN section adds to the row-tiled plan."""
    ri, he, tp, k = p.item_rows, c.expert_hidden, c.tp, c.top_k
    win = window_rows(c)  # own rows are computed in the aligned window around them
    vmem, smem, hbm = pltpu.VMEM, pltpu.SMEM, pltpu.HBM
    u32, i32, f32 = jnp.uint32, jnp.int32, jnp.float32
    dma = pltpu.SemaphoreType.DMA
    items = []
    if not _item_rows_in_lhs(c, p):
        items += [
            ("ep_x", vmem((2, ri, he // 2), u32)),  # an item's x rows, bf16 pairs
            ("ep_acc", vmem((ri, he), f32)),
            ("ep_y", vmem((ri, he // 2), u32)),  # an item's y rows, bf16 pairs
        ]
    if c.tiled_vmem_pads:
        items += [("ep_lhs_stage", vmem((c.row_tile, c.hidden), u32))]
    return items + [
        # window rows: pooled expert input (bf16 pairs), routing, slot places in ep_xs, the
        # pooled / combined f32, the unpool output and FFN branch
        ("ep_xo", vmem((win, he // 2), u32)),
        ("ep_ids", vmem((win, LANE), i32)),
        ("ep_wsel", vmem((win, LANE), f32)),
        ("ep_pos_v", vmem((8, LANE), i32)),  # [slot k, window row] (window <= LANE)
        ("ep_pos_s", smem((8, LANE), i32)),
        ("ep_u", vmem((win, he), f32)),
        ("ep_f", vmem((win, c.hidden), f32)),
        # row reduce-scatter of the o-proj: this rank's blocks of each wo column tile for every
        # owner (two tiles in flight), every rank's blocks of this rank's rows
        ("ep_osend", vmem((2, tp, p.rows_own_padded, c.hidden_local // 2), u32)),
        ("ep_orecv", vmem((tp * tp, p.rows_own_padded, c.hidden_local // 2), u32)),
        ("ep_o_sems", dma((3,))),  # received, sent from each send buffer
        # slots per (source rank, expert), every rank's row
        ("ep_csend", vmem((8, c.n_experts), i32)),
        ("ep_cnt_v", vmem((tp, c.n_experts), i32)),
        ("ep_tab_v", vmem((TAB_ROWS, LANE), i32)),  # route.slot_tables
        ("ep_tab_s", smem((TAB_ROWS, LANE), i32)),
        # the two ragged all-to-alls: own slots by destination (and expert), received slots
        # by source (and expert); y goes back row by row from the expert loop
        ("ep_xs", hbm((p.send_rows, he // 2), u32)),
        ("ep_xr", hbm((p.recv_rows, he // 2), u32)),
        ("ep_yb", hbm((p.send_rows, he // 2), u32)),
        ("ep_items", smem((1 + 3 * p.max_items,), i32)),
        ("ep_rows", smem((p.recv_rows,), i32)),  # received slots grouped by expert
        ("ep_yrow", vmem((2, k, he // 2), u32)),  # one own row's slots, for the combine
        ("ep_w_sems", dma((p.nslot, 6))),
        ("ep_dn_sems", dma((p.nslot,))),
        ("ep_x_sems", dma((2,))),
        ("ep_misc_sems", dma((4,))),  # local copies: xs fill, y scatter, tables
        ("ep_cnt_sems", dma((2,))),  # count rows: received, sent
        ("ep_a2a_sems", dma((4,))),  # dispatch received / sent, return received / sent
        ("ep_yrow_sems", dma((2,))),
        ("ep_ag_sems", dma((2,))),  # row all-gather: received, sent
        # the next layer's pre-attention input (bf16 pairs): every row's in HBM (own rows
        # staged in ep_f), its row tiles on the way into lhs (through sin[0] with HBM pads)
        ("ep_lhs_g", hbm((c.rows, c.hidden // 2), u32)),
        ("ep_lhs_sems", dma((5,))),  # (unused), sent, own copy, stage slots
        ("ep_lhs_rsems", dma((tp,))),  # each owner's rows received
    ]
