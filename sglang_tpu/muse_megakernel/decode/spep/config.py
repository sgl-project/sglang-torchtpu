# SPDX-License-Identifier: Apache-2.0
"""The EP plans of one row-tiled step, 32 whole experts per rank: a2a (own rows R / tp behind
two ragged all-to-alls) or gather (every row on every rank), and the scratch each adds."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.common.a2a import SEG
from sglang_tpu.muse_megakernel.common.config import MKConfig
from sglang_tpu.muse_megakernel.common.consts import LANE
from sglang_tpu.muse_megakernel.common.plan import fit_ring_slots
from sglang_tpu.muse_megakernel.decode.spep.experts import (
    BD_KC,
    item_sizes,
    scale_group,
)

ARMS = ("a2a", "gather")
TAB_ROWS = 32  # route.slot_tables' rows
DN_TK, DN_TN = 2048, 1024  # a replicated pool / unpool tile: one fp4 expert-ring slot
# TP's expert-section buffers the EP hooks drop (past SMEM capacity scalars spill into HBM)
TP_SCRATCH_UNUSED = frozenset(
    ("route_v", "route_s", "ids_s", "ystage", "ystage_sems", "yrow", "yrow_sems")
)


class EPBuffers(NamedTuple):
    """An item's VMEM rows: x (two items, bf16 pairs), the f32 accumulator, outgoing y."""

    x: Callable[[Any], Any]  # x(b): [item_rows, H_e / 2] u32 of item parity b
    acc: Any  # [item_rows, H_e] f32
    y: Any  # [item_rows, H_e / 2] u32


class EPPlan(NamedTuple):
    experts: int  # EL: whole experts per rank
    inter: int  # an expert's whole intermediate
    chunk: int  # I columns per chunk: one fp4 ring slot, two adjacent for fp8
    item_rows: int  # rows of one work item; a busier expert takes several
    rows_own: int  # Rc: the rows this rank owns
    rows_own_padded: int  # Rc rounded up to SEG: one o-proj reduce-scatter block
    send_rows: (
        int  # the send / return buffer: own slots, each destination's segment padded
    )
    recv_rows: int  # the receive buffer: every row's slots could land on one rank
    max_items: int
    nslot: int  # weight-chunk ring slots (the TP expert ring's)

    @property
    def n_chunks(self) -> int:
        return self.inter // self.chunk

    def slots(self, kind: str) -> int:
        return self.nslot if kind == "fp4" else self.nslot // 2


def ep_mk_config(c: MKConfig) -> MKConfig:
    """The step's MKConfig for the EP kernel."""
    cf = msgspec.structs.replace(
        c,
        dense_skip=("wpool", "wunpool"),
        single_row_coll=False,
        pair_o_rs=False,
        fold_o=False,
        fold_f=False,
        tiled_vmem_y=0,
        tiled_pair_y=0,
        pair_y_stream=False,
    )
    slots = fit_ring_slots(cf, cap=cf.ring_slots)
    return msgspec.structs.replace(
        cf, ring_slots=slots, stream_depth=min(cf.stream_depth, slots - 1)
    )


def ep_arm() -> str:
    """The EP decode paradigm, SGLANG_TPU_MUSE_MK_EP_ARM: gather (EP, every row on every rank)
    or a2a (SP + EP, own rows behind ragged all-to-alls). Read at trace time."""
    arm = tpu_envs.SGLANG_TPU_MUSE_MK_EP_ARM.get()
    if arm not in ARMS:
        raise ValueError(f"SGLANG_TPU_MUSE_MK_EP_ARM={arm!r}: a2a or gather")
    return arm


def ep_plan(c: MKConfig, item_rows: int = 32) -> EPPlan:
    if not c.row_tile:
        raise ValueError(
            "the EP a2a arm runs the row-tiled program (MuseMegakernel.moe_mode ep)"
        )
    assert c.n_experts % c.tp == 0 and c.rows % c.tp == 0
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
        chunk=c.inter_local,
        item_rows=item_rows,
        rows_own=rc,
        rows_own_padded=-(-rc // SEG) * SEG,
        send_rows=-(-(rc * k + pad) // SEG) * SEG,
        recv_rows=-(-(c.rows * k + pad) // SEG) * SEG,
        max_items=el + -(-c.rows * k // item_rows),
        nslot=c.expert_depth + 1,
    )


# The gather arm's replicated per-row VMEM buffers (~110-125 KiB/row) overflow VMEM above this.
GATHER_MAX_ROWS = 128


def gather_plan(c: MKConfig, item_rows: int = 32) -> EPPlan:
    """The gather arm's plan: every rank holds every row; the a2a buffers are unused."""
    if not c.row_tile:
        raise ValueError(
            "the EP gather arm runs the row-tiled program (MuseMegakernel.moe_mode ep)"
        )
    if c.rows > GATHER_MAX_ROWS:
        raise ValueError(
            f"the EP gather arm serves at most {GATHER_MAX_ROWS} rows, got {c.rows}: "
            "use SGLANG_TPU_MUSE_MK_EP_ARM=a2a"
        )
    assert c.n_experts % c.tp == 0
    assert c.inter_local % BD_KC == 0 and c.expert_hidden % BD_KC == 0
    assert c.n_experts_padded == c.n_experts and c.hidden // c.hidden_local == c.tp
    assert c.expert_hidden % (2 * LANE * c.tp) == 0 and c.hidden % DN_TK == 0
    assert c.expert_hidden % DN_TK == 0 and c.rows * c.top_k < 4096
    item_sizes(item_rows)
    el = c.n_experts // c.tp
    return EPPlan(
        experts=el,
        inter=c.inter_local * c.tp,
        chunk=c.inter_local,
        item_rows=item_rows,
        rows_own=c.rows,
        rows_own_padded=c.rows,
        send_rows=0,
        recv_rows=0,
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
        g = scale_group(kind)
        return (
            ((el, he // 8, 2 * i), jnp.int32),
            ((el, he // g, 2 * i), jnp.bfloat16),
            ((el, i // 8, he), jnp.int32),
            ((el, i // g, he), jnp.bfloat16),
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


def item_scratch(c: MKConfig, p: EPPlan) -> list:
    """An item's x / acc / y rows, unless they live in S["lhs"] (ep_buffers)."""
    if _item_rows_in_lhs(c, p):
        return []
    ri, he, vmem = p.item_rows, c.expert_hidden, pltpu.VMEM
    return [
        ("ep_x", vmem((2, ri, he // 2), jnp.uint32)),  # an item's x rows, bf16 pairs
        ("ep_acc", vmem((ri, he), jnp.float32)),
        ("ep_y", vmem((ri, he // 2), jnp.uint32)),  # an item's y rows, bf16 pairs
    ]


def ep_scratch(c: MKConfig, p: EPPlan) -> list:
    """Buffers the EP FFN section adds to the row-tiled plan."""
    he, tp, k = c.expert_hidden, c.tp, c.top_k
    win = window_rows(c)  # own rows are computed in the aligned window around them
    vmem, smem, hbm = pltpu.VMEM, pltpu.SMEM, pltpu.HBM
    u32, i32, f32 = jnp.uint32, jnp.int32, jnp.float32
    dma = pltpu.SemaphoreType.DMA
    items = item_scratch(c, p)
    if c.tiled_vmem_pads:
        items += [("ep_lhs_stage", vmem((c.row_tile, c.hidden), u32))]
    return (
        items
        + [
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
            (
                "ep_yrow",
                vmem((4, k, he // 2), u32),
            ),  # own rows' slots in flight (combine)
            ("ep_w_sems", dma((p.nslot, 4))),
            ("ep_dn_sems", dma((p.nslot,))),
            ("ep_x_sems", dma((2,))),
            ("ep_misc_sems", dma((4,))),  # local copies: xs fill, y scatter, tables
            ("ep_cnt_sems", dma((2,))),  # count rows: received, sent
            (
                "ep_a2a_sems",
                dma((4,)),
            ),  # dispatch received / sent, return received / sent
            ("ep_yrow_sems", dma((4,))),
            ("ep_ag_sems", dma((2,))),  # row all-gather: received, sent
            # the next layer's pre-attention input (bf16 pairs): every row's in HBM (own rows
            # staged in ep_f), its row tiles on the way into lhs (through sin[0] with HBM pads)
            ("ep_lhs_g", hbm((c.rows, c.hidden // 2), u32)),
            ("ep_lhs_sems", dma((5,))),  # (unused), sent, own copy, stage slots
            ("ep_lhs_rsems", dma((tp,))),  # each owner's rows received
        ]
        + (col_pool_scratch(c, p) if col_pool(c) else [])
    )


def col_pool(c: MKConfig) -> bool:
    """Column-sliced pool / unpool (a2a arm): SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS."""
    return c.rows <= tpu_envs.SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS.get()


def col_pool_scratch(c: MKConfig, p: EPPlan) -> list:
    """Column-sliced pool / unpool: this rank's column block of every row by owner (bf16
    pairs, each owner's rows at the top of its tile-aligned slot), every rank's block of own
    rows, every row's combined u (bf16 pairs) and the scale slices."""
    he, hc, tp, rcp = c.expert_hidden // c.tp, c.hidden // c.tp, c.tp, p.rows_own_padded
    vmem, hbm = pltpu.VMEM, pltpu.HBM
    u32 = jnp.uint32
    return [
        ("ep_pstage", vmem((tp, rcp, he // 2), u32)),
        ("ep_prcv", vmem((tp, rcp, he // 2), u32)),
        ("ep_ug", hbm((c.rows, c.expert_hidden // 2), u32)),
        ("ep_ustage", vmem((tp, rcp, hc // 2), u32)),
        ("ep_urcv", vmem((tp, rcp, hc // 2), u32)),
        ("ep_csc", vmem((8, max(he, hc)), jnp.float32)),
        ("ep_cx_sems", pltpu.SemaphoreType.DMA((3,))),  # received, sent, scale copy
    ]


O_BUFS_VMEM = 2**20


def o_bufs(c: MKConfig) -> int:
    """Send buffers of the gather arm's o-proj reduce-scatter: up to 8 within O_BUFS_VMEM."""
    fit = O_BUFS_VMEM // (c.rows * (c.hidden_local // 2) * 4)
    return max(2, min(8, c.hidden // c.hidden_local, fit))
