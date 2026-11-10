# SPDX-License-Identifier: Apache-2.0
"""Scratch plans (VMEM / SMEM / HBM buffers) and their byte budget."""

import math

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK, LANE, SMALL_ROWS
from sglang_tpu.muse_megakernel.mk.hooks import EPHook
from sglang_tpu.muse_megakernel.mk.packing import y_words
from sglang_tpu.muse_megakernel.mk.plan import (
    few_rows,
    owner_window_rows,
    ring_slot_shape,
    route_rows,
)
from sglang_tpu.muse_megakernel.mk.tables import swa_window_entries


# ----------------------------------------------------------------------------- scratch
def scratch_vmem_bytes(cfg: MKConfig) -> int:
    """VMEM bytes the scratch plan declares (Mosaic's own temporaries come on top)."""
    total = 0
    for _, ref in scratch_plan(cfg):
        if getattr(ref, "memory_space", None) == pltpu.VMEM:
            total += math.prod(ref.shape) * jnp.dtype(ref.dtype).itemsize
    return total


# row-tiled program: semaphores of tile_sems
_TS_AUX, _TS_OUT, _TS_LOCAL, _TS_ROPE = 0, 1, 2, 3
_TS_N = 3 + 8
# row-tiled collectives: (send, receive) semaphore index of each kind; o_rs sends alternate
# between two (or four, with tiled_o_rot) buffers, each with its own send semaphore
_TC = {
    "o_rs": 0,
    "o_rs1": 1,
    "o_ag": 2,
    "pool": 3,
    "y": 4,
    "u": 5,
    "f": 6,
    "o_rs2": 7,
    "o_rs3": 8,
}
_TC_VPADS_S = {"o_rs": 0, "o_rs1": 1, "o_rs2": 2, "o_rs3": 3, "y": 4}
_TC_VPADS_R = {"o_rs": 0, "y": 1}
# the weight rings, small slabs and attention buffers, the same in both programs
_SHARED_SCRATCH = (
    "ring",
    "ring_sems",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w13_sems",
    "w13s_sems",
    "w2_sems",
    "w2s_sems",
    "smalls",
    "small_sems",
    "final_norm",
    "kv_chunk",
    "kv_chunk_sems",
    "attn_p",
    "attn_psc",
    "kv_row",
    "kv_row_sem",
)


def _can_w2_attn_scratch(c: MKConfig) -> bool:
    return bool(
        c.attn_head_fp8 >= 1
        and c.attn_pipe
        and c.attn_opt in (3, 4, 5)
        and c.kv_sal
        and c.kv_sal_fp8
        and c.sal_fp8_dot >= 2
        and c.attn_exp2
        and not c.use_sal_preones
        and c.kv_heads == 2
        and c.head_dim == 64
        and 2 * c.kv_heads * c.rows_per_req * c.g > LANE
        and c.rows_per_req * c.g <= 64
        and c.chunk_pages * c.page_size == 1024
        and (c.expert_depth + 1) * (c.inter_local // FP4_PACK) >= 192
        and c.expert_hidden >= 4 * c.attn_streams * LANE
    )


def _use_attn_p_fp8(c: MKConfig) -> bool:
    return bool(
        (c.attn_p_fp8 or (c.attn_head_fp8 >= 2 and _can_w2_attn_scratch(c)))
        and c.attn_pipe
        and c.attn_opt in (3, 4, 5)
        and c.kv_sal
        and c.kv_sal_fp8
        and c.sal_fp8_dot >= 2
        and not (c.use_sal_preones and c.kv_heads * c.rows_per_req * c.g > 64)
    )


def _attn_p_rows(c: MKConfig) -> int:
    m_raw = c.kv_heads * c.rows_per_req * c.g
    if _can_w2_attn_scratch(c) and m_raw > 64:
        return 128
    if _use_attn_p_fp8(c):
        return -(-m_raw // 32) * 32
    return m_raw


def _attn_p_ysend_streams(c: MKConfig) -> int:
    """How many piped-attention streams can borrow ysend (disjoint lifetimes)."""
    cpt = c.chunk_pages * c.page_size
    yw = y_words(c)
    if not (
        c.alias_p
        and (c.row_tile == 0 or c.tiled_vmem_y >= 1)
        and c.attn_pipe
        and c.kv_sal
        and cpt <= yw
        and cpt % LANE == 0
        and c.kv_heads * c.g == 2 * c.top_k
    ):
        return 0
    m_pad = _attn_p_rows(c)
    if _use_attn_p_fp8(c):
        cap = (4 * c.top_k * c.rows) // m_pad
        return min(c.attn_streams, cap if 2 * cpt <= yw else cap // 2)
    cap = (2 * c.top_k * c.rows) // m_pad
    return min(c.attn_streams, cap if 2 * cpt <= yw else cap // 2)


def _alias_attn_p_ysend(c: MKConfig) -> bool:
    """Whether all piped-attention probabilities borrow ysend."""
    return c.attn_pipe and _attn_p_ysend_streams(c) == c.attn_streams


def _alias_pads_into_y(c: MKConfig) -> bool:
    """Whether pad_o_rs and pad_o_ag can borrow pad_y (pool_ag + f_ag separate lifetimes)."""
    few = c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max
    return (
        c.alias_oy
        and (
            (c.row_tile == 0 and c.ex_opt and not few)
            or (c.row_tile > 0 and c.tiled_vmem_pads and c.tiled_vmem_y >= 2)
        )
        and c.rows_per_dev * c.top_k == c.rows
        and c.top_k == 8
        and y_words(c) >= 2 * (c.hidden_local // 2)
    )


def _alias_oag_into_f(c: MKConfig) -> bool:
    """Whether pad_o_ag can borrow pad_f when pad_y is in HBM (vy < 2)."""
    return bool(
        c.alias_oag_f
        and c.alias_oy
        and c.row_tile > 0
        and c.tiled_vmem_pads
        and not _alias_pads_into_y(c)
    )


def _alias_tiled_xfull_q(c: MKConfig) -> bool:
    return bool(c.row_tile and c.alias_tiled_temps and c.q_dim <= c.expert_hidden // 2)


def _alias_tiled_padu_attn(c: MKConfig) -> bool:
    return bool(
        c.row_tile
        and c.alias_tiled_temps
        and c.tiled_vmem_pads
        and c.rows_per_dev % 8 == 0
        and c.tp * c.rows_per_dev == c.rows
        and c.q_dim <= c.expert_hidden // 2
    )


def _alias_tiled_xfull_attn(c: MKConfig) -> bool:
    """vmem_trim: attn_buf in the xfull columns past qn_s (dead before o_proj writes xfull)."""
    return bool(
        c.vmem_trim
        and _alias_tiled_xfull_q(c)
        and not _alias_tiled_padu_attn(c)
        and 2 * c.q_dim <= c.expert_hidden // 2
    )


def _alias_pool_ysend(c: MKConfig) -> bool:
    """vmem_trim: pad_pool in ysend. Peers write it only after this device's o all-gather
    (past attn_p and o_buf) and ex_setup reads it before zeroing ysend."""
    return bool(
        c.vmem_trim
        and c.row_tile
        and c.tiled_vmem_pads
        and c.tiled_vmem_y >= 1
        and (c.rows * c.top_k) % c.tp == 0
        and (c.rows * c.top_k) // c.tp >= c.rows
        and y_words(c) >= c.expert_hidden_local // 2
    )


def _alias_u_xfull(c: MKConfig) -> bool:
    """vmem_trim: u_send in xfull's first rows (an 8-row-aligned slice)."""
    return bool(c.vmem_trim and c.rows_per_dev % 8 == 0)


def _alias_tiled_ids_kv(c: MKConfig) -> bool:
    return bool(c.row_tile and c.alias_tiled_temps and c.kv_dim == LANE)


def _alias_tiled_obuf_ysend(c: MKConfig) -> bool:
    ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
    return bool(
        c.row_tile
        and c.alias_tiled_temps
        and c.tiled_vmem_y >= 1
        and (c.rows * c.top_k) % (ob_n * 8) == 0
        and (c.rows * c.top_k) // ob_n >= c.rows
        and y_words(c) >= c.hidden_local // 2
    )


def scratch_plan(cfg: MKConfig, ep: EPHook | None = None):
    plan = _scratch_plan_tiled(cfg) if cfg.row_tile else _scratch_plan_base(cfg)
    if ep is not None:
        assert cfg.row_tile, "the expert-section hook is row-tiled only"
        plan = ep.scratch_plan(cfg, plan)
    if cfg.swa_window_table:
        plan = plan + [
            ("psw_win", pltpu.SMEM((cfg.n_req * swa_window_entries(cfg),), jnp.int32)),
            ("psw_base", pltpu.SMEM((cfg.n_req,), jnp.int32)),
            ("psw_sem", pltpu.SemaphoreType.DMA((1,))),
        ]
    return plan


def _scratch_plan_tiled(c: MKConfig) -> list:
    """Row-tiled scratch: per-row VMEM is the matmul LHS, accumulators and small per-row

    vectors; everything [R, H]-sized or [R, top_k, He]-sized lives in HBM unless
    tiled_vmem_pads / tiled_vmem_y keep them in VMEM.
    """
    R, K, H, RT = c.rows, c.top_k, c.hidden, c.row_tile
    He, Hl, Hel, tp, Rc = (
        c.expert_hidden,
        c.hidden_local,
        c.expert_hidden_local,
        c.tp,
        c.rows_per_dev,
    )
    yw = y_words(c)
    hbm, vmem, smem = pltpu.HBM, pltpu.VMEM, pltpu.SMEM
    u32, f32 = jnp.uint32, jnp.float32
    nsc = 8 if c.vec_route else R * K + 8
    base = dict(_scratch_plan_base(c))
    use_p_fp8 = _use_attn_p_fp8(c)
    if c.vmem_trim:
        base["final_norm"] = vmem((8, LANE), f32)
    if _alias_attn_p_ysend(c):
        base["attn_p"] = vmem(
            (1, 1, 32 if use_p_fp8 else 8, LANE),
            jnp.float8_e4m3fn if use_p_fp8 else jnp.bfloat16,
        )
    vpads = c.tiled_vmem_pads
    vy = c.tiled_vmem_y
    alias_oy = _alias_pads_into_y(c)
    pmem = vmem if vpads else hbm
    ob_n = 4 if (c.tiled_o_rot or c.pair_o_rs) else 2
    items = [(n, base[n]) for n in _SHARED_SCRATCH] + [
        ("rtab", vmem((R, 2 * c.head_dim), f32)),
        # a request's attention output rows (h, t, gi), re-read strided per (head, group)
        *(
            [("attn_stage", vmem((c.kv_heads * c.rows_per_req * c.g, LANE), f32))]
            if c.rows_per_req % 8 == 0
            else []
        ),
        ("rope_blk", vmem((8, 8, 2 * c.head_dim), f32)),
        ("lhs", vmem((R, H), jnp.bfloat16)),
        ("acc", vmem((R, max(c.qkvg_dim, Hl, Hel, c.n_experts_padded)), f32)),
    ]
    if not _alias_tiled_xfull_q(c):
        items += [("qn_s", vmem((R, c.q_dim), f32))]
    if not (_alias_tiled_padu_attn(c) or _alias_tiled_xfull_attn(c)):
        items += [("attn_buf", vmem((R, c.q_dim), f32))]
    if not _alias_tiled_ids_kv(c):
        items += [
            ("kb_s", vmem((R, c.kv_dim), f32)),
            ("vb_s", vmem((R, c.kv_dim), f32)),
        ]
    items += [
        ("ids_v", vmem((R, LANE), jnp.int32)),
        ("wsel_v", vmem((R, LANE), f32)),
        ("wk_s", vmem((owner_window_rows(c), K, LANE), f32)),
        ("xfull", vmem((R, He // 2), u32)),
        # vmem_trim: pool_send is o_send's first columns
        *([] if c.vmem_trim else [("pool_send", vmem((R, Hel // 2), u32))]),
    ]
    if not _alias_tiled_obuf_ysend(c):
        items += [("o_buf", vmem((ob_n, R, Hl // 2), u32))]
    items += [
        ("o_send", vmem((R, Hl // 2), u32)),
        # vmem_trim: f_send is o_send (both all-gathers wait for their sends)
        *([] if c.vmem_trim else [("f_send", vmem((R, Hl // 2), u32))]),
        *([] if _alias_u_xfull(c) else [("u_send", vmem((Rc, He // 2), u32))]),
    ]
    if not vpads:
        items += [("bin", vmem((2, RT, H), jnp.bfloat16))]
    items += [
        ("btile", vmem((RT, H), jnp.bfloat16)),
        ("tile_sems", pltpu.SemaphoreType.DMA((_TS_N,))),
        # outstanding expert-output row DMAs per staging buffer, then the emit count
        ("ypend", smem((8,), jnp.int32)),
        # the scalar routing tables (vec_route builds route_s instead)
        ("uniq", smem((nsc,), jnp.int32)),
        ("ex_cnt", smem((nsc,), jnp.int32)),
        ("ex_off", smem((nsc,), jnp.int32)),
        ("ex_pos", smem((nsc,), jnp.int32)),
        ("perm", smem((nsc,), jnp.int32)),
        ("ex_of", smem((c.n_experts_padded,), jnp.int32)),
        ("ids_s", smem((8 if c.vec_route else R, LANE), jnp.int32)),
        ("seen", smem((c.n_experts_padded,), jnp.int32)),
        ("blk_end", smem((max(c.tp, 8),), jnp.int32)),
        ("route_v", vmem((route_rows(c), LANE), jnp.int32)),
        ("route_s", smem((route_rows(c), LANE), jnp.int32)),
        ("route_sem", pltpu.SemaphoreType.DMA((1,))),
        ("h_stream", pmem((R, H), f32)),
    ]
    if not (c.vec_route and c.trim_sems):
        items += [("ids_sem", pltpu.SemaphoreType.DMA((1,)))]
    if c.ex_gs_smem and (c.scale_fp8 or c.ex_kperm >= 3):
        items += [("gs_s", smem((8, c.n_experts_padded), f32))]
    if not alias_oy:
        items += [("pad_o_rs", pmem((tp, R, Hl // 2), u32))]
        if not _alias_oag_into_f(c):
            items += [("pad_o_ag", pmem((tp, R, Hl // 2), u32))]
    n_tc_s = len(_TC_VPADS_S) if (vpads and c.trim_sems) else len(_TC)
    n_tc_r = len(_TC_VPADS_R) if (vpads and c.trim_sems) else len(_TC)
    items += [
        *([] if _alias_pool_ysend(c) else [("pad_pool", pmem((tp, R, Hel // 2), u32))]),
        (
            "pad_y",
            (vmem if vy >= 2 else hbm)((tp, Rc, K, yw), u32),
        ),
        (
            "pad_u",
            vmem((tp, Rc, He // 2), u32) if vpads else hbm((R, He // 2), u32),
        ),
        ("pad_f", pmem((tp, R, Hl // 2), u32)),
        ("tc_ssems", pltpu.SemaphoreType.DMA((n_tc_s,))),
        ("tc_rsems", pltpu.SemaphoreType.DMA((n_tc_r,))),
    ]
    if not (vpads and c.trim_sems):
        items += [
            ("pass_sems", pltpu.SemaphoreType.DMA((2, 4))),
            # pipelined o / f all-gathers: per row tile, each sender's copy signals it
            ("oag_tsems", pltpu.SemaphoreType.DMA((R // RT,))),
            ("fag_tsems", pltpu.SemaphoreType.DMA((R // RT,))),
            # pipelined u all-gather: per owner
            ("uag_osems", pltpu.SemaphoreType.DMA((tp,))),
        ]
    if not vpads:
        items += [
            ("sin", vmem((2, RT, H), f32)),
            ("sout", vmem((2, RT, H), f32)),
            ("ptile", vmem((2, tp, RT, Hl // 2), u32)),
            ("utile", vmem((2, RT, He // 2), u32)),
            ("pltile", vmem((2, tp, RT, Hel // 2), u32)),
        ]
    else:
        n_coll_sems = 4 if c.trim_sems else 8
        items += [
            ("coll_ssems", pltpu.SemaphoreType.DMA((n_coll_sems,))),
            ("coll_rsems", pltpu.SemaphoreType.DMA((n_coll_sems,))),
        ]
        if c.tiled_pair_y or c.pair_ag or c.pair_o_rs:
            items += [
                ("pair_ssems", pltpu.SemaphoreType.DMA((7,))),
                ("pair_rsems", pltpu.SemaphoreType.DMA((7,))),
            ]
    if c.fold_f or c.fold_o:
        items += [
            ("fold_v", vmem((8, H), f32)),
            # tile_loops: the fold passes' per-row scales (s, r_inv)
            *([("fold_stage", vmem((R, 2 * LANE), f32))] if c.tile_loops else []),
            ("o_rsems", pltpu.SemaphoreType.DMA((tp,))),
            ("f_rsems", pltpu.SemaphoreType.DMA((tp,))),
        ]
    if vy >= 1:
        items += [("ysend", vmem((R, K, yw), u32))]
    else:
        items += [
            ("ystage", vmem((2, c.expert_chunk, yw), u32)),
            ("ystage_sems", pltpu.SemaphoreType.DMA((2,))),
        ]
    if vy < 2:
        items += [
            ("yrow", vmem((2, tp // 2 if c.vmem_trim else tp, 1, K, yw), u32)),
            ("yrow_sems", pltpu.SemaphoreType.DMA((2,))),
        ]
    return items


def _scratch_plan_base(cfg: MKConfig):
    c = cfg
    R, K, H = c.rows, c.top_k, c.hidden
    He, Il = c.expert_hidden, c.inter_local
    yw = y_words(c)
    use_p_fp8 = _use_attn_p_fp8(c)
    p_dt = jnp.float8_e4m3fn if use_p_fp8 else jnp.bfloat16
    p_rows = _attn_p_rows(c)
    nslot = c.expert_depth + 1
    # expert rings in u32 words sized for fp4 (8 K-rows per word); an fp8 slab (4 per word)
    # takes two adjacent slots
    w13_words = He // FP4_PACK
    w2_words = Il // FP4_PACK
    cpt = c.chunk_pages * c.page_size
    sdt = jnp.float8_e4m3fn if c.scale_fp8 else jnp.bfloat16
    # tile_loops QBLOCK slots hold the K / qblock scale rows the dots read
    slim = bool(c.qblock and c.tile_loops)
    s13_rows = max(He // c.qblock, 16) if slim else He // FP4_GROUP
    s2_rows = (
        max(Il // c.qblock, 16)
        if slim
        else max(Il // FP4_GROUP, 32 if c.scale_fp8 else 16)
    )
    items = [
        (
            "ring",
            pltpu.VMEM((c.ring_slots, *ring_slot_shape(c)), jnp.uint32),
        ),  # dense weight stream
        ("ring_sems", pltpu.SemaphoreType.DMA((c.ring_slots,))),
        ("w13", pltpu.VMEM((nslot, w13_words, 2 * Il), jnp.uint32)),
        ("w13s", pltpu.VMEM((nslot, s13_rows, 2 * Il), sdt)),
        ("w2", pltpu.VMEM((nslot, w2_words, He), jnp.uint32)),
        ("w2s", pltpu.VMEM((nslot, s2_rows, He), sdt)),
        ("w13_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w13s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("w2s_sems", pltpu.SemaphoreType.DMA((nslot,))),
        ("smalls", pltpu.VMEM((2, SMALL_ROWS, H), jnp.float32)),
        ("small_sems", pltpu.SemaphoreType.DMA((2,))),
        ("final_norm", pltpu.VMEM((8, H), jnp.float32)),
        ("rope_blk", pltpu.VMEM((R, 8, 2 * c.head_dim), jnp.float32)),
        ("misc_sems", pltpu.SemaphoreType.DMA((R + 4,))),
        ("xpack", pltpu.VMEM((R, H // 2), jnp.uint32)),
        ("o_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        ("ysend", pltpu.VMEM((R, K, yw), jnp.uint32)),
        ("attn_buf", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("qn_s", pltpu.VMEM((R, c.q_dim), jnp.float32)),
        ("kb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("vb_s", pltpu.VMEM((R, c.kv_dim), jnp.float32)),
        ("wk_s", pltpu.VMEM((R, K, LANE), jnp.float32)),
        ("u_send", pltpu.VMEM((c.rows_per_dev, He // 2), jnp.uint32)),
        ("pool_send", pltpu.VMEM((R, c.expert_hidden_local // 2), jnp.uint32)),
        ("f_send", pltpu.VMEM((R, c.hidden_local // 2), jnp.uint32)),
        (
            "kv_chunk",
            # SAL chunks live in the expert w13 slots (sal_kv_view)
            pltpu.VMEM((1, 8, LANE), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM(
                (
                    c.kv_ring,
                    c.chunk_pages,
                    c.page_size,
                    c.pool_kv_heads,
                    2,
                    c.pool_padded_head_dim,
                ),
                jnp.bfloat16,
            ),
        ),
        # one per (sub-)slot: a chunk's page copies all signal it, one wait takes the chunk
        (
            "kv_chunk_sems",
            pltpu.SemaphoreType.DMA(
                (
                    max(c.kv_ring, 2 * c.swa_kv_group)
                    if c.swa_chunk_pages
                    else c.kv_ring,
                )
            ),
        ),
        # attn_pipe: probabilities of the chunks in flight, double-buffered by loop iteration
        (
            "attn_p",
            pltpu.VMEM((1, 1, 32 if use_p_fp8 else 8, LANE), p_dt)
            if _alias_attn_p_ysend(c)
            else pltpu.VMEM(
                (
                    2,
                    (c.attn_streams - _attn_p_ysend_streams(c)) if c.attn_pipe else 1,
                    p_rows,
                    cpt if c.attn_pipe else LANE,
                ),
                p_dt,
            ),
        ),
        (
            "attn_psc",
            pltpu.VMEM(
                (
                    2,
                    c.attn_streams,
                    -(-(c.kv_heads * c.rows_per_req * c.g) // 8) * 8,
                    LANE,
                ),
                jnp.float32,
            )
            if (use_p_fp8 and not (c.attn_head_fp8 >= 2 and _can_w2_attn_scratch(c)))
            else pltpu.VMEM((1, 1, 8, LANE), jnp.float32),
        ),
        (
            "kv_row",
            pltpu.VMEM((c.sal_words, R), jnp.uint32)
            if c.kv_sal
            else pltpu.VMEM(
                (R, c.pool_kv_heads, 2, c.pool_padded_head_dim), jnp.bfloat16
            ),
        ),
        ("kv_row_sem", pltpu.SemaphoreType.DMA((1,))),
        ("uniq", pltpu.SMEM((R * K + 8,), jnp.int32)),
        # per distinct expert u: live (row, slot) count; its pairs, encoded row * K + slot, are
        # perm[ex_off[u] : ex_off[u] + ex_cnt[u]] (ex_pos is the fill cursor)
        ("ex_cnt", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_off", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_pos", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("perm", pltpu.SMEM((R * K + 8,), jnp.int32)),
        ("ex_of", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        (
            "xfull",
            pltpu.VMEM((R, He // 2), jnp.uint32)
            if c.xfull_packed
            else pltpu.VMEM((R, He), jnp.float32),
        ),
        ("ids_v", pltpu.VMEM((R, LANE), jnp.int32)),
        ("ids_s", pltpu.SMEM((R, LANE), jnp.int32)),
        ("ids_sem", pltpu.SemaphoreType.DMA((1,))),
        ("seen", pltpu.SMEM((c.n_experts_padded,), jnp.int32)),
        # y_block_send: uniq index past the last expert first used by owner block b
        ("blk_end", pltpu.SMEM((max(c.tp, 8),), jnp.int32)),
        ("route_v", pltpu.VMEM((route_rows(c), LANE), jnp.int32)),
        ("route_s", pltpu.SMEM((route_rows(c), LANE), jnp.int32)),
        ("route_sem", pltpu.SemaphoreType.DMA((1,))),
        # collective pads: bf16 pairs packed into u32 (_pack_bf16x2)
        (
            "pad_pool",
            pltpu.VMEM((c.tp, R, c.expert_hidden_local // 2), jnp.uint32),
        ),
        ("pad_y", pltpu.VMEM((c.tp, c.rows_per_dev, K, yw), jnp.uint32)),
        ("pad_u", pltpu.VMEM((c.tp, c.rows_per_dev, He // 2), jnp.uint32)),
        ("pad_f", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ("coll_ssems", pltpu.SemaphoreType.DMA((8,))),
        ("coll_rsems", pltpu.SemaphoreType.DMA((8,))),
    ]
    if not (c.ex_opt and c.topk_iter):
        items += [("rank", pltpu.VMEM((R, c.n_experts_padded), jnp.int32))]
    if not _alias_pads_into_y(c):
        items += [
            ("pad_o_rs", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
            ("pad_o_ag", pltpu.VMEM((c.tp, R, c.hidden_local // 2), jnp.uint32)),
        ]
    if not (
        c.ex_opt
        and not (c.single_row_coll and c.n_req * c.rows_per_req <= c.few_rows_max)
    ):
        items += [
            ("row1_send", pltpu.VMEM((few_rows(c), H // 2), jnp.uint32)),
            ("pad_o1", pltpu.VMEM((c.tp, few_rows(c), H // 2), jnp.uint32)),
            (
                "pad_f1",
                pltpu.VMEM((c.tp, few_rows(c), c.hidden_local // 2), jnp.uint32),
            ),
        ]
    if c.pair_ag >= 1:
        items += [
            ("pair_ssems", pltpu.SemaphoreType.DMA((8,))),
            ("pair_rsems", pltpu.SemaphoreType.DMA((8,))),
        ]
    if c.fold_f or c.fold_o:
        items += [
            # fold LHS: the stream part [R, H] and per-column gains (row 0)
            ("fold_x", pltpu.VMEM((R, H), jnp.float32)),
            ("fold_v", pltpu.VMEM((8, H), jnp.float32)),
            # one receive semaphore per sender of the folded all-gathers
            ("o_rsems", pltpu.SemaphoreType.DMA((c.tp,))),
            ("f_rsems", pltpu.SemaphoreType.DMA((c.tp,))),
        ]
    if c.scale_fp8:
        items += [
            # the fp8 layers' bf16 channel scales, which the e4m3 slabs cannot hold
            ("w13c", pltpu.VMEM((nslot, 2, 2 * Il), jnp.bfloat16)),
            ("w2c", pltpu.VMEM((nslot, 2, He), jnp.bfloat16)),
        ]
    if c.scale_fp8 or c.ex_kperm >= 3:
        items += [
            # this layer's (gate, up, down) globals, one expert per lane
            ("gs_v", pltpu.VMEM((8, c.n_experts_padded), jnp.float32)),
        ]
        if c.ex_gs_smem:
            items += [("gs_s", pltpu.SMEM((8, c.n_experts_padded), jnp.float32))]
    return items


# ----------------------------------------------------------------------------- the program
WEIGHT_NAMES = (
    "rope",
    "final_norm",
    "smalls",
    "wqkvg",
    "wo",
    "wpool",
    "wrouter",
    "wunpool",
)
