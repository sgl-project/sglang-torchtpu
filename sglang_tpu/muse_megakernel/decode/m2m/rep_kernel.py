# SPDX-License-Identifier: Apache-2.0
"""Replicated-stream Wave-9 `M2M` & `L0_A2M` Decode Megakernels (`TP = 8` & `TP = 16`).

Covers:
  - `TP = 16` (`2x2x2` multi-host) across `B in (16, 32, 64, 128)`, fusing ALL FIVE per-layer
    stages (`moe_to_attn_bridge` + in-VMEM `TP=16` QK-RMSNorm/RoPE/FP8-KV + `RPAd(tp16)` +
    in-VMEM head-RMSNorm/gate + `attn_to_moe_bridge`) into ONE zero-spill collective kernel!
  - `TP = 8` (`2x2x1`) small-batch `B in (16, 32)` (e.g. `Shared 64K/128K/256K` `c=16` SWA)
    as well as `B in (64, 128)` when `tok_sharded_stream=False`.
"""

# PEP 563 must stay off so cpu_export_jax_op sees concrete jax.Array annotations.

import functools

from sglang_tpu.attention.pallas.util import align_to
from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _COL_STRIP,
    _DOT_TN,
    _as_2d_f32_row,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _fori_col_blocks,
    _fori_strips,
    _gather_shard_xor1,
    _gather_token_shard_xor1_and_dot_post,
    _norm_pre_expert_slice,
    _pallas_call_maybe_interpret,
    _pick_strip,
    _prescale_s_buf_in_vmem,
    _reduce_col_blocks,
    _run_post_ffn_slice_epilogue,
    _signal_barrier,
    _use_dual_tree_xy,
    _wait_barrier,
)
from sglang_tpu.muse_megakernel.decode.m2m.kernel import (
    _M2M_CID_BASE,
    _NUM_KV_SLOTS,
    _llo_fence,
    _max_m2m_ring_pages,
    _prologue_prefetch_kv_pages,
    _run_in_vmem_qkv_epilogue_any,
    _run_in_vmem_rpad_and_gate,
)


def _run_rep_b1_from_vmem_core(
    s_buf,  # f32[rows, hidden] already resident in VMEM!
    gated_ref,  # bf16[rows, o_in] already resident in VMEM!
    wo_buf,  # fp8[2, o_in, half] DMA already started!
    wo_sem,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_buf,  # fp8[hidden, pre_cols] DMA already started!
    w_pre_sem,
    w_pre_scale_ref,
    w_router_buf,  # bf16/f32[hidden, n_exp] DMA already started!
    w_router_sem,
    pre_exp_gain_ref,
    out_stream_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    ffn_vmem,
    router_vmem,
    inv_pff_buf,
    send_half,
    mate_recv,
    own_half,
    scatter_ref,
    own_chunk,
    sq_buf,
    pre_shard,
    pre_half_a,
    pre_half_b,
    s_out_sem,
    ar_s1a_send,
    ar_s1a_recv,
    ar_s1b_send,
    ar_s1b_recv,
    ar_s2a_send,
    ar_s2a_recv,
    ar_s2b_send,
    ar_s2b_recv,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    *,
    me,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    need_wait_barrier: bool,
) -> None:
    """In-VMEM replicated-stream Bridge-1b (`TP=8` & `TP=16`) consuming resident `s_buf` + `gated`."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    half = wo_buf.shape[2]
    hidden = 2 * half
    chunk = hidden // world_size
    pre_cols = w_pre_buf.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_buf.shape[1]
    n_chips = world_size // 2
    bm = _pick_strip(rows, 32)
    bm64 = _pick_strip(rows, 64)
    chunk_strips = chunk // _COL_STRIP
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows
    is_r16 = w_router_buf.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)
    chip_c0 = pl.multiple_of(chip * chunk, 128)

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0, rows=rows, n_cols=half
    )

    if need_wait_barrier:
        _wait_barrier(world_size)
    c1a = pltpu.make_async_remote_copy(
        send_half,
        mate_recv,
        ar_s1a_send.at[0],
        ar_s1a_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c1a.start()

    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[1], wo_scale_ref, own_c0, own_half, 0, rows=rows, n_cols=half
    )
    c1a.wait()

    def _sum_s1(r0):
        rs = pl.ds(r0, bm64)

        def _col_step(c_sl):
            v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
            own_half[rs, c_sl] = v.astype(own_half.dtype)

        _fori_col_blocks(half, _col_step)

    _fori_strips(rows, bm64, _sum_s1)

    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    if use_xy:
        e_cols = chunk // 2
        e_strips = e_cols // _COL_STRIP
        cx = chip & jnp.int32(1)
        cy = chip >> jnp.int32(1)
        peer_x = me ^ jnp.int32(2)
        peer_y = me ^ jnp.int32(4)

        a_send1 = pl.multiple_of((jnp.int32(1) - cx) * chunk, 128)
        a_keep1 = pl.multiple_of(cx * chunk, 128)
        b_send1 = pl.multiple_of(2 * chunk + (jnp.int32(1) - cy) * chunk, 128)
        b_keep1 = pl.multiple_of(2 * chunk + cy * chunk, 128)

        rs1_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_send1, chunk)],
            scatter_ref.at[0],
            ar_s1b_send.at[0],
            ar_s1b_recv.at[0],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        rs1_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_send1, chunk)],
            scatter_ref.at[1],
            ar_s1b_send.at[1],
            ar_s1b_recv.at[1],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        rs1_a.start()
        rs1_b.start()
        rs1_a.wait()
        rs1_b.wait()

        def _sum_xy1(r0):
            rs = pl.ds(r0, bm64)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(off, _COL_STRIP)
                ca = pl.ds(pl.multiple_of(a_keep1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep1 + off, _COL_STRIP), _COL_STRIP)
                va = own_half[rs, ca].astype(jnp.float32) + scatter_ref[0, rs, c_l].astype(jnp.float32)
                vb = own_half[rs, cb].astype(jnp.float32) + scatter_ref[1, rs, c_l].astype(jnp.float32)
                own_half[rs, ca] = va.astype(own_half.dtype)
                own_half[rs, cb] = vb.astype(own_half.dtype)

        _fori_strips(rows, bm64, _sum_xy1)

        a_send2 = pl.multiple_of(a_keep1 + (jnp.int32(1) - cy) * e_cols, 128)
        a_keep2 = pl.multiple_of(a_keep1 + cy * e_cols, 128)
        b_send2 = pl.multiple_of(b_keep1 + (jnp.int32(1) - cx) * e_cols, 128)
        b_keep2 = pl.multiple_of(b_keep1 + cx * e_cols, 128)

        rs2_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_send2, e_cols)],
            scatter_ref.at[0, :, pl.ds(0, e_cols)],
            ar_s1b_send.at[2],
            ar_s1b_recv.at[2],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        rs2_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_send2, e_cols)],
            scatter_ref.at[1, :, pl.ds(0, e_cols)],
            ar_s1b_send.at[3],
            ar_s1b_recv.at[3],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        rs2_a.start()
        rs2_b.start()
        rs2_a.wait()
        rs2_b.wait()

        def _sum_xy2(r0):
            rs = pl.ds(r0, bm64)
            for s in range(e_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(off, _COL_STRIP)
                ca = pl.ds(pl.multiple_of(a_keep2 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep2 + off, _COL_STRIP), _COL_STRIP)
                va = own_half[rs, ca].astype(jnp.float32) + scatter_ref[0, rs, c_l].astype(jnp.float32)
                vb = own_half[rs, cb].astype(jnp.float32) + scatter_ref[1, rs, c_l].astype(jnp.float32)
                own_half[rs, ca] = va.astype(own_half.dtype)
                own_half[rs, cb] = vb.astype(own_half.dtype)

        _fori_strips(rows, bm64, _sum_xy2)

        ag1_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_keep2, e_cols)],
            own_half.at[:, pl.ds(a_keep2, e_cols)],
            ar_s2a_send.at[0],
            ar_s2a_recv.at[0],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        ag1_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_keep2, e_cols)],
            own_half.at[:, pl.ds(b_keep2, e_cols)],
            ar_s2a_send.at[1],
            ar_s2a_recv.at[1],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        ag1_a.start()
        ag1_b.start()
        _prescale_s_buf_in_vmem(s_buf, pa_alpha_ref, rows=rows, hidden=hidden, bm=bm64)
        ag1_a.wait()
        ag1_b.wait()

        a_other1 = pl.multiple_of((jnp.int32(1) - cx) * chunk, 128)
        b_other1 = pl.multiple_of(2 * chunk + (jnp.int32(1) - cy) * chunk, 128)
        ag2_a = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(a_keep1, chunk)],
            own_half.at[:, pl.ds(a_keep1, chunk)],
            ar_s2a_send.at[2],
            ar_s2a_recv.at[2],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        ag2_b = pltpu.make_async_remote_copy(
            own_half.at[:, pl.ds(b_keep1, chunk)],
            own_half.at[:, pl.ds(b_keep1, chunk)],
            ar_s2a_send.at[3],
            ar_s2a_recv.at[3],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        ag2_a.start()
        ag2_b.start()

        def _own_pa_sq_keep1(r0):
            rs = pl.ds(r0, bm64)
            acc = jnp.zeros((bm64, 128), dtype=jnp.float32)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                ca = pl.ds(pl.multiple_of(a_keep1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_keep1 + off, _COL_STRIP), _COL_STRIP)
                fa = own_half[rs, ca].astype(jnp.float32)
                fb = own_half[rs, cb].astype(jnp.float32)
                sq = fa * fa + fb * fb
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            sq_buf[rs, :] = acc

        _fori_strips(rows, bm64, _own_pa_sq_keep1)
        ag2_a.wait()
        ag2_b.wait()
    else:
        c1b = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            peer_chip = chip ^ jnp.int32(d)
            op = pltpu.make_async_remote_copy(
                own_half.at[:, pl.ds(pl.multiple_of(peer_chip * chunk, 128), chunk)],
                scatter_ref.at[d - 1],
                ar_s1b_send.at[d - 1],
                ar_s1b_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c1b.append(op)
        for op in c1b:
            op.wait()

        def _sum_s2(r0):
            rs = pl.ds(r0, bm64)
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                c_l = pl.ds(pl.multiple_of(off, _COL_STRIP), _COL_STRIP)
                c_h = pl.ds(pl.multiple_of(chip_c0 + off, _COL_STRIP), _COL_STRIP)
                tot = own_half[rs, c_h].astype(jnp.float32)
                for slot in range(n_chips - 1):
                    tot = tot + scatter_ref[slot, rs, c_l].astype(jnp.float32)
                vb = tot.astype(own_chunk.dtype)
                own_chunk[rs, c_l] = vb
                own_half[rs, c_h] = vb

        _fori_strips(rows, bm64, _sum_s2)

        c2a = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                own_chunk,
                own_half.at[:, pl.ds(chip_c0, chunk)],
                ar_s2a_send.at[d - 1],
                ar_s2a_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            c2a.append(op)
        _prescale_s_buf_in_vmem(s_buf, pa_alpha_ref, rows=rows, hidden=hidden, bm=bm64)
        for op in c2a:
            op.wait()

    c2b = pltpu.make_async_remote_copy(
        own_half,
        mate_recv,
        ar_s2b_send.at[0],
        ar_s2b_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c2b.start()

    if use_xy:
        def _own_pa_sq_rest(r0):
            rs = pl.ds(r0, bm64)
            acc = sq_buf[rs, :]
            for s in range(chunk_strips):
                off = s * _COL_STRIP
                ca = pl.ds(pl.multiple_of(a_other1 + off, _COL_STRIP), _COL_STRIP)
                cb = pl.ds(pl.multiple_of(b_other1 + off, _COL_STRIP), _COL_STRIP)
                fa = own_half[rs, ca].astype(jnp.float32)
                fb = own_half[rs, cb].astype(jnp.float32)
                sq = fa * fa + fb * fb
                acc = acc + (sq[:, 0:128] + sq[:, 128:256])
            sq_buf[rs, :] = acc

        _fori_strips(rows, bm64, _own_pa_sq_rest)
    else:
        def _own_pa_sq(r0):
            rs = pl.ds(r0, bm64)

            def _col(c_sl, acc):
                bf = own_half[rs, c_sl].astype(jnp.float32)
                sq = bf * bf
                return acc + (sq[:, 0:128] + sq[:, 128:256])

            sq_buf[rs, :] = _reduce_col_blocks(
                half, _col, jnp.zeros((bm64, 128), dtype=jnp.float32)
            )

        _fori_strips(rows, bm64, _own_pa_sq)
    c2b.wait()

    r_out_buf = None if (is_r16 or tile_r32) else router_vmem

    def _finish_pa_and_pre_ffn(r0):
        rs = pl.ds(r0, bm)

        def _oth_sq(c_sl, acc):
            bf = mate_recv[rs, c_sl].astype(jnp.float32)
            sq = bf * bf
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        oth_sq = _reduce_col_blocks(half, _oth_sq, jnp.zeros((bm, 128), dtype=jnp.float32))
        pa_sq = sq_buf[rs, :] + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )

        def _res_both_halves(c_l, acc):
            c_own = pl.ds(pl.multiple_of(own_c0 + c_l.start, _COL_STRIP), _COL_STRIP)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c_l.start, _COL_STRIP), _COL_STRIP)
            ns_own = s_buf[rs, c_own] + pa_beta_ref[0:1, c_own] * (
                own_half[rs, c_l].astype(jnp.float32) * inv_pa
            )
            ns_oth = s_buf[rs, c_oth] + pa_beta_ref[0:1, c_oth] * (
                mate_recv[rs, c_l].astype(jnp.float32) * inv_pa
            )
            s_buf[rs, c_own] = ns_own
            s_buf[rs, c_oth] = ns_oth
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            sq = v_own * v_own + v_oth * v_oth
            return acc + (sq[:, 0:128] + sq[:, 128:256])

        pff_sq = _reduce_col_blocks(
            half, _res_both_halves, jnp.zeros((bm, 128), dtype=jnp.float32)
        )
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        if tile_r32:
            inv_pff_buf[rs, :] = inv_pff

        def _pre_ffn_out(c_g):
            sv = s_buf[rs, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if r_out_buf is not None:
                r_out_buf[rs, c_g] = rn
            ffn_vmem[rs, c_g] = rn.astype(ffn_vmem.dtype)

        _fori_col_blocks(hidden, _pre_ffn_out)

    _fori_strips(rows, bm, _finish_pa_and_pre_ffn)
    if not tile_r32:
        pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    _dma_wait(w_pre_buf, w_pre_sem.at[0])
    _dot_fp8_scaled_to_ref(
        ffn_vmem, w_pre_buf, w_pre_scale_ref, 0, pre_shard, 0, rows=rows, n_cols=pre_cols
    )

    def _compute_router_under_ici():
        _dma_wait(w_router_buf, w_router_sem.at[0])
        bm_g = _pick_strip(rows, 64)
        rmul = jnp.float32(router_multiplier)

        def _r_step(r0):
            rs = pl.ds(r0, bm_g)
            if tile_r32:
                inv_r = inv_pff_buf[rs, :]

                def _fill_rtile(s, _):
                    c_g = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    sv = s_buf[rs, c_g]
                    gn = pre_ff_gain_ref[0:1, c_g]
                    router_vmem[:, c_g] = (
                        (sv * gn) * inv_r if pre_ff_scale_input else (sv * inv_r) * gn
                    )
                    return jnp.int32(0)

                lax.fori_loop(0, hidden // _COL_STRIP, _fill_rtile, jnp.int32(0))
                lhs_slice = router_vmem[:, :]
            else:
                r_lhs = ffn_vmem if is_r16 else router_vmem
                lhs_slice = r_lhs[rs, :]

            def _e_step(ce):
                c_sl = pl.ds(ce, _DOT_TN)
                lg = jnp.dot(lhs_slice, w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
                out_router_logits_ref[rs, c_sl] = lg * rmul

            _fori_strips(n_exp, _DOT_TN, _e_step)

        _fori_strips(rows, bm_g, _r_step)
        if tile_r32:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    align = 16 if use_halfrow else 8

    def _norm_pre_slice(src_buf, row_base, n_r):
        _norm_pre_expert_slice(
            src_buf,
            pre_exp_gain_ref,
            out_expert_in_ref,
            h_pre=h_pre,
            row_base=row_base,
            n_r=n_r,
            align=align,
            pre_exp_eps=pre_exp_eps,
            pre_exp_scale_input=pre_exp_scale_input,
        )

    eff_pre_a = send_half.at[pl.ds(0, pre_buf_rows), :] if h_pre == half else pre_half_a
    if use_halfrow:
        half_r = rows // 2
        bm_s1 = _pick_strip(half_r, 32)
        parity = me & jnp.int32(1)
        me_c0 = pl.multiple_of(me * pre_cols, 128)
        pair_cols = 2 * pre_cols
        chip_pre_c0 = pl.multiple_of(chip * pair_cols, 128)
        my_r0 = pl.multiple_of(parity * half_r, 16)
        mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half_r, 16)

        s1 = pltpu.make_async_remote_copy(
            pre_shard.at[pl.ds(mate_r0, half_r), :],
            eff_pre_a.at[:, pl.ds(me_c0, pre_cols)],
            ag_s1_send.at[0],
            ag_s1_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s1.start()
        _fori_strips(
            half_r,
            bm_s1,
            lambda r0: eff_pre_a.__setitem__(
                (pl.ds(r0, bm_s1), pl.ds(me_c0, pre_cols)),
                pre_shard[pl.ds(my_r0 + r0, bm_s1), :],
            ),
        )
        s1.wait()

        s2_ops = []
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op = pltpu.make_async_remote_copy(
                eff_pre_a.at[:, pl.ds(chip_pre_c0, pair_cols)],
                eff_pre_a.at[:, pl.ds(chip_pre_c0, pair_cols)],
                ag_s2_send.at[d - 1],
                ag_s2_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            s2_ops.append(op)
        _compute_router_under_ici()
        for op in s2_ops:
            op.wait()

        _norm_pre_slice(eff_pre_a, my_r0, half_r)
        s3 = pltpu.make_async_remote_copy(
            out_expert_in_ref.at[pl.ds(my_r0, half_r), :],
            out_expert_in_ref.at[pl.ds(my_r0, half_r), :],
            ag_s3_send.at[0],
            ag_s3_recv.at[0],
            device_id=(mate,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        s3.start()
        _dma_wait(out_stream_hbm, s_out_sem.at[0])
        s3.wait()
    else:
        eff_pre_b = mate_recv.at[pl.ds(0, pre_buf_rows), :] if h_pre == half else pre_half_b
        _gather_shard_xor1(
            pre_shard,
            eff_pre_a,
            eff_pre_b,
            ag_s1_send,
            ag_s1_recv,
            ag_s2_send,
            ag_s2_recv,
            ag_s3_send,
            ag_s3_recv,
            me=me,
            world_size=world_size,
            use_halfrow=False,
            during_ici_fn=_compute_router_under_ici,
            my_half_epilogue_fn=_norm_pre_slice,
            mate_half_epilogue_fn=_norm_pre_slice,
        )
        _dma_wait(out_stream_hbm, s_out_sem.at[0])


def _m2m_rep_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    *body_refs,
    world_size: int,
    sharded_moe_in: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    qkv_norm_eps: float,
    query_scale: float,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float,
    v_scale: float,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int,
    layer_offset_in_bundle: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_hbm = body_refs[idx]
    moe_out_ref = body_refs[idx + 1]
    w_post_hbm = body_refs[idx + 2]
    s_post_ref = body_refs[idx + 3]
    post_ff_gain_ref = body_refs[idx + 4]
    pf_alpha_ref = body_refs[idx + 5]
    pf_beta_ref = body_refs[idx + 6]
    next_in_gain_ref = body_refs[idx + 7]
    next_w_qkv_hbm = body_refs[idx + 8]
    next_s_qkv_ref = body_refs[idx + 9]
    next_w_gate_hbm = body_refs[idx + 10]
    next_s_gate_ref = body_refs[idx + 11]
    idx += 12
    if has_rope:
        cos_sin_ref = body_refs[idx]
        idx += 1
    else:
        cos_sin_ref = None
    kv_cache_ro_hbm = body_refs[idx]
    wo_hbm = body_refs[idx + 1]
    wo_scale_ref = body_refs[idx + 2]
    pa_alpha_ref = body_refs[idx + 3]
    pa_beta_ref = body_refs[idx + 4]
    pre_ff_gain_ref = body_refs[idx + 5]
    w_pre_hbm = body_refs[idx + 6]
    w_pre_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    out_stream_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kv_cache_hbm = body_refs[idx + 3]
    idx += 4

    slab_a = body_refs[idx]
    slab_b = body_refs[idx + 1]
    wo_buf = body_refs[idx + 2]
    w_router_buf = body_refs[idx + 3]
    bkv_vmem_ref = body_refs[idx + 4]
    s_buf = body_refs[idx + 5]
    next_in_vmem = body_refs[idx + 6]  # reused as ffn_vmem in Phase 3!
    qkv_vmem = body_refs[idx + 7]
    q_vmem_ref = body_refs[idx + 8]
    kv_merged_vmem_ref = body_refs[idx + 9]
    o_vmem_ref = body_refs[idx + 10]
    gate_vmem_ref = body_refs[idx + 11]
    gated_vmem_ref = body_refs[idx + 12]
    kvw_rows_vmem = body_refs[idx + 13]
    post_shard = body_refs[idx + 14]
    b2_half_a = body_refs[idx + 15]
    b2_half_b = body_refs[idx + 16]
    mate_tok_buf = body_refs[idx + 17]
    moe_full_buf = body_refs[idx + 18]
    router_vmem = body_refs[idx + 19]
    inv_pff_buf = body_refs[idx + 20]
    send_half = body_refs[idx + 21]
    mate_recv = body_refs[idx + 22]
    own_half = body_refs[idx + 23]
    scatter_ref = body_refs[idx + 24]
    own_chunk = body_refs[idx + 25]
    sq_buf = body_refs[idx + 26]
    pre_shard = body_refs[idx + 27]
    pre_half_a = body_refs[idx + 28]
    pre_half_b = body_refs[idx + 29]
    rpa_smem_ref = body_refs[idx + 30]
    idx += 31

    # Semaphores (`<= 59` on TP=16, `<= 41` on TP=8 — strictly < 64 hardware limit!):
    w_post_gate_sem = body_refs[idx]
    w_qkv_pre_sem = body_refs[idx + 1]
    wo_sem = body_refs[idx + 2]
    w_router_sem = body_refs[idx + 3]
    s_io_sem = body_refs[idx + 4]
    bkv_sems = body_refs[idx + 5]
    kvw_out_sem = body_refs[idx + 6]
    d2d_p0_send = body_refs[idx + 7]
    d2d_p0_recv = body_refs[idx + 8]
    d2d_p1_send = body_refs[idx + 9]
    d2d_p1_recv = body_refs[idx + 10]
    d2d_p2_send = body_refs[idx + 11]
    d2d_p2_recv = body_refs[idx + 12]
    ici_p0_send = body_refs[idx + 13]
    ici_p0_recv = body_refs[idx + 14]
    ici_p1_send = body_refs[idx + 15]
    ici_p1_recv = body_refs[idx + 16]
    ici_p2_send = body_refs[idx + 17]
    ici_p2_recv = body_refs[idx + 18]

    rows = stream_hbm.shape[0]
    hidden = stream_hbm.shape[1]
    half = hidden // 2
    post_cols = w_post_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    pre_cols = w_pre_hbm.shape[1]
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8
    bm_p = _pick_strip(rows, 64)

    me = lax.axis_index("model").astype(jnp.int32)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)

    w_post_buf = slab_b.at[pl.ds(0, w_post_hbm.shape[0]), pl.ds(0, post_cols)]
    w_qkv_buf = slab_a.at[:, pl.ds(0, qkv_cols)]
    w_gate_buf = slab_b.at[:, pl.ds(0, gate_cols)]
    w_pre_buf = slab_a.at[:, pl.ds(0, pre_cols)]

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_hbm, s_buf, s_io_sem.at[0]).start()
    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_gate_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_pre_sem.at[0]).start()
    _prologue_prefetch_kv_pages(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kv_cache_ro_hbm,
        bkv_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sliding_window=sliding_window,
        page_size=page_size,
        num_slots=bkv_vmem_ref.shape[0],
    )

    def _run_b2_post_dot():
        if sharded_moe_in:
            _gather_token_shard_xor1_and_dot_post(
                moe_out_ref,
                mate_tok_buf,
                moe_full_buf,
                w_post_buf,
                s_post_ref,
                post_shard,
                w_post_gate_sem,
                d2d_p0_send,
                d2d_p0_recv,
                ici_p0_send,
                ici_p0_recv,
                me=me,
                world_size=world_size,
            )
        else:
            _dma_wait(w_post_buf, w_post_gate_sem.at[0])
            _dot_fp8_scaled_to_ref(
                moe_out_ref,
                w_post_buf,
                s_post_ref,
                0,
                post_shard,
                0,
                rows=rows,
                n_cols=post_cols,
            )
            _wait_barrier(world_size)

    _llo_fence(distribution_ref, _run_b2_post_dot)

    # `w_post_buf` is dead -> reuse `slab_b` (`w_post_gate_sem`) for `W_gate` (defer `Wo`/`W_pre`/`W_router` past RPAd KV)!
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_post_gate_sem.at[0]).start()

    def _wait_and_prescale_stream():
        _dma_wait(s_buf, s_io_sem.at[0])
        _prescale_s_buf_in_vmem(s_buf, pf_alpha_ref, rows=rows, hidden=hidden, bm=bm_p)

    def _epi(buf, r_base, n_r):
        _run_post_ffn_slice_epilogue(
            s_buf,
            buf,
            post_ff_gain_ref,
            pf_alpha_ref,
            pf_beta_ref,
            next_in_gain_ref,
            s_buf,
            next_in_vmem,
            hidden=hidden,
            row_base=r_base,
            n_r=n_r,
            align=align,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps,
            next_in_scale_input=next_in_scale_input,
            with_next_norm=True,
            pre_scaled_alpha=True,
        )

    _gather_shard_xor1(
        post_shard,
        b2_half_a,
        b2_half_b,
        d2d_p1_send,
        d2d_p1_recv,
        ici_p1_send,
        ici_p1_recv,
        d2d_p2_send,
        d2d_p2_recv,
        me=me,
        world_size=world_size,
        use_halfrow=use_halfrow,
        during_ici_fn=_wait_and_prescale_stream,
        my_half_epilogue_fn=_epi,
        mate_half_epilogue_fn=_epi,
    )

    _dma_wait(w_qkv_buf, w_qkv_pre_sem.at[0])
    _llo_fence(
        distribution_ref,
        lambda: _dot_fp8_scaled_to_ref(
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_vmem, 0, rows=rows, n_cols=qkv_cols
        ),
    )

    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        kv_merged_vmem_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_rep():
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
        ).start()
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
        ).start()

    _run_in_vmem_rpad_and_gate(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        q_vmem_ref,
        kv_merged_vmem_ref,
        kv_cache_ro_hbm,
        out_kv_cache_hbm,
        bkv_vmem_ref,
        o_vmem_ref,
        next_in_vmem,
        w_gate_buf,
        w_post_gate_sem,
        next_s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        kvw_rows_vmem,
        rpa_smem_ref,
        bkv_sems,
        kvw_out_sem,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        out_norm_eps=qkv_norm_eps,
        kv_pack_dense2=kv_pack_dense2,
        layer_num_in_bundle=layer_num_in_bundle,
        layer_offset_in_bundle=layer_offset_in_bundle,
        early_prefetched=True,
        interpret_mode=interpret_mode,
        on_rpad_kv_done_fn=_start_wo_rep,
    )

    pltpu.make_async_copy(w_pre_hbm, w_pre_buf, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    # Phase 3: `ici_p1 + d2d_p2` causally closed `d2d_p0, d2d_p1, ici_p0` across all ranks,
    # and `ar_s2a (ici_p2) + ar_s2b (d2d_p1)` causally closes `ici_p1, d2d_p2, d2d_p0` before
    # `ag_s1..s3` starts!
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ar = 4 if use_xy else (world_size // 2 - 1)
    _run_rep_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        w_pre_buf,
        w_qkv_pre_sem,
        w_pre_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
        next_in_vmem,
        router_vmem,
        inv_pff_buf,
        send_half,
        mate_recv,
        own_half,
        scatter_ref,
        own_chunk,
        sq_buf,
        pre_shard,
        pre_half_a,
        pre_half_b,
        s_io_sem,
        d2d_p0_send,
        d2d_p0_recv,
        ici_p0_send.at[pl.ds(0, n_ar)],
        ici_p0_recv.at[pl.ds(0, n_ar)],
        ici_p2_send.at[pl.ds(0, n_ar)],
        ici_p2_recv.at[pl.ds(0, n_ar)],
        d2d_p1_send,
        d2d_p1_recv,
        d2d_p2_send,
        d2d_p2_recv,
        ici_p1_send,
        ici_p1_recv,
        d2d_p0_send,
        d2d_p0_recv,
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=False,
    )
    w_tot_cnt = rpa_smem_ref[0]

    @pl.when(w_tot_cnt > 0)
    def _wait_kv_scatter():
        src_kv = kvw_rows_vmem.at[pl.ds(0, w_tot_cnt)]
        pltpu.make_async_copy(src_kv, src_kv, kvw_out_sem.at[0]).wait()


def _l0_a2m_rep_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    *body_refs,
    world_size: int,
    qkv_norm_eps: float,
    query_scale: float,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float,
    v_scale: float,
    kv_pack_dense2: bool,
    layer_num_in_bundle: int,
    layer_offset_in_bundle: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_hbm = body_refs[idx]
    attn_in_ref = body_refs[idx + 1]
    w_qkv_hbm = body_refs[idx + 2]
    s_qkv_ref = body_refs[idx + 3]
    w_gate_hbm = body_refs[idx + 4]
    s_gate_ref = body_refs[idx + 5]
    idx += 6
    if has_rope:
        cos_sin_ref = body_refs[idx]
        idx += 1
    else:
        cos_sin_ref = None
    kv_cache_ro_hbm = body_refs[idx]
    wo_hbm = body_refs[idx + 1]
    wo_scale_ref = body_refs[idx + 2]
    pa_alpha_ref = body_refs[idx + 3]
    pa_beta_ref = body_refs[idx + 4]
    pre_ff_gain_ref = body_refs[idx + 5]
    w_pre_hbm = body_refs[idx + 6]
    w_pre_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    out_stream_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kv_cache_hbm = body_refs[idx + 3]
    idx += 4

    slab_a = body_refs[idx]
    w_gate_buf = body_refs[idx + 1]
    wo_buf = body_refs[idx + 2]
    w_router_buf = body_refs[idx + 3]
    bkv_vmem_ref = body_refs[idx + 4]
    s_buf = body_refs[idx + 5]
    ffn_vmem = body_refs[idx + 6]
    qkv_vmem = body_refs[idx + 7]
    q_vmem_ref = body_refs[idx + 8]
    kv_merged_vmem_ref = body_refs[idx + 9]
    o_vmem_ref = body_refs[idx + 10]
    gate_vmem_ref = body_refs[idx + 11]
    gated_vmem_ref = body_refs[idx + 12]
    kvw_rows_vmem = body_refs[idx + 13]
    router_vmem = body_refs[idx + 14]
    inv_pff_buf = body_refs[idx + 15]
    send_half = body_refs[idx + 16]
    mate_recv = body_refs[idx + 17]
    own_half = body_refs[idx + 18]
    scatter_ref = body_refs[idx + 19]
    own_chunk = body_refs[idx + 20]
    sq_buf = body_refs[idx + 21]
    pre_shard = body_refs[idx + 22]
    pre_half_a = body_refs[idx + 23]
    pre_half_b = body_refs[idx + 24]
    rpa_smem_ref = body_refs[idx + 25]
    idx += 26

    w_qkv_pre_sem = body_refs[idx]
    w_gate_sem = body_refs[idx + 1]
    wo_sem = body_refs[idx + 2]
    w_router_sem = body_refs[idx + 3]
    s_in_sem = body_refs[idx + 4]
    s_out_sem = body_refs[idx + 5]
    bkv_sems = body_refs[idx + 6]
    kvw_out_sem = body_refs[idx + 7]
    ar_s1a_send = body_refs[idx + 8]
    ar_s1a_recv = body_refs[idx + 9]
    ar_s1b_send = body_refs[idx + 10]
    ar_s1b_recv = body_refs[idx + 11]
    ar_s2a_send = body_refs[idx + 12]
    ar_s2a_recv = body_refs[idx + 13]
    ar_s2b_send = body_refs[idx + 14]
    ar_s2b_recv = body_refs[idx + 15]
    ag_s1_send = body_refs[idx + 16]
    ag_s1_recv = body_refs[idx + 17]
    ag_s2_send = body_refs[idx + 18]
    ag_s2_recv = body_refs[idx + 19]
    ag_s3_send = body_refs[idx + 20]
    ag_s3_recv = body_refs[idx + 21]

    rows = attn_in_ref.shape[0]
    hidden = stream_hbm.shape[1]
    half = hidden // 2
    qkv_cols = w_qkv_hbm.shape[1]
    pre_cols = w_pre_hbm.shape[1]
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows

    me = lax.axis_index("model").astype(jnp.int32)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)

    w_qkv_buf = slab_a.at[:, pl.ds(0, qkv_cols)]
    w_pre_buf = slab_a.at[:, pl.ds(0, pre_cols)]

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    _prologue_prefetch_kv_pages(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kv_cache_ro_hbm,
        bkv_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
        sliding_window=sliding_window,
        page_size=page_size,
        num_slots=bkv_vmem_ref.shape[0],
    )

    _dma_wait(w_qkv_buf, w_qkv_pre_sem.at[0])
    _llo_fence(
        distribution_ref,
        lambda: _dot_fp8_scaled_to_ref(
            attn_in_ref, w_qkv_buf, s_qkv_ref, 0, qkv_vmem, 0, rows=rows, n_cols=qkv_cols
        ),
    )

    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        kv_merged_vmem_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_l0_rep():
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
        ).start()
        pltpu.make_async_copy(
            wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
        ).start()

    _run_in_vmem_rpad_and_gate(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        q_vmem_ref,
        kv_merged_vmem_ref,
        kv_cache_ro_hbm,
        out_kv_cache_hbm,
        bkv_vmem_ref,
        o_vmem_ref,
        attn_in_ref,
        w_gate_buf,
        w_gate_sem,
        s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        kvw_rows_vmem,
        rpa_smem_ref,
        bkv_sems,
        kvw_out_sem,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        out_norm_eps=qkv_norm_eps,
        kv_pack_dense2=kv_pack_dense2,
        layer_num_in_bundle=layer_num_in_bundle,
        layer_offset_in_bundle=layer_offset_in_bundle,
        early_prefetched=True,
        interpret_mode=interpret_mode,
        on_rpad_kv_done_fn=_start_wo_l0_rep,
    )

    pltpu.make_async_copy(w_pre_hbm, w_pre_buf, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(s_buf, s_in_sem.at[0])

    _run_rep_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        w_pre_buf,
        w_qkv_pre_sem,
        w_pre_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
        ffn_vmem,
        router_vmem,
        inv_pff_buf,
        send_half,
        mate_recv,
        own_half,
        scatter_ref,
        own_chunk,
        sq_buf,
        pre_shard,
        pre_half_a,
        pre_half_b,
        s_out_sem,
        ar_s1a_send,
        ar_s1a_recv,
        ar_s1b_send,
        ar_s1b_recv,
        ar_s2a_send,
        ar_s2a_recv,
        ar_s2b_send,
        ar_s2b_recv,
        ag_s1_send,
        ag_s1_recv,
        ag_s2_send,
        ag_s2_recv,
        ag_s3_send,
        ag_s3_recv,
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=True,
    )
    w_tot_cnt = rpa_smem_ref[0]

    @pl.when(w_tot_cnt > 0)
    def _wait_kv_scatter():
        src_kv = kvw_rows_vmem.at[pl.ds(0, w_tot_cnt)]
        pltpu.make_async_copy(src_kv, src_kv, kvw_out_sem.at[0]).wait()


def m2m_rep_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream,
    moe_out,
    w_post_w8k,
    s_post,
    post_ff_gain,
    pf_alpha,
    pf_beta,
    next_in_gain,
    next_w_qkv_w8k,
    next_s_qkv,
    next_w_gate_w8k,
    next_s_gate,
    cos_sin,
    kv_cache_ro,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_w8k,
    w_pre_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int,
    sharded_moe_in: bool = False,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    sm_scale: float = 0.125,
    sliding_window: int | None = 2048,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_pack_dense2: bool = False,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE + 2,
    interpret: bool = False,
):
    """Per-layer `M2M` (`l >= 1`, replicated stream, `TP in (8, 16)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, hidden = stream.shape
    if rows < 8 or rows > 128 or rows % 8 != 0:
        raise ValueError(f"Unsupported m2m_rep rows={rows}")
    if sharded_moe_in and moe_out.shape[0] * world_size != rows:
        raise ValueError(f"sharded_moe_in shape mismatch: {moe_out.shape} vs rows={rows}")

    half = hidden // 2
    chunk = hidden // world_size
    post_cols = w_post_w8k.shape[1]
    qkv_cols = next_w_qkv_w8k.shape[1]
    gate_cols = next_w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_cols = w_pre_w8k.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_km.shape[1]
    n_chips = world_size // 2
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ici_p0 = max(n_chips - 1 if sharded_moe_in else 0, 4 if use_xy else (n_chips - 1))
    n_ici_p2 = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)
    is_r16 = w_router_km.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    bm_g = _pick_strip(rows, 64)
    use_halfrow = rows >= 32 and rows % 32 == 0
    buf_rows = rows // 2 if use_halfrow else rows
    reuse_pre_ab = h_pre == half
    has_rope = cos_sin is not None
    b_pad32 = align_to(rows, 32)
    r_per = rows // world_size if sharded_moe_in else 8

    phys_page_rows = (
        kv_cache_ro.shape[2] if kv_cache_ro.ndim == 6 else kv_cache_ro.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_ring_pages = _max_m2m_ring_pages(
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
    )

    vmem, smem, hbm, sem = pltpu.VMEM, pltpu.SMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    in_specs = [h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp]
    if has_rope:
        in_specs.append(v_sp)
    in_specs.extend([h_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp])

    kv_out_sd = (
        jax.ShapeDtypeStruct(kv_cache_ro.shape, kv_cache_ro.dtype)
        if interpret
        else pltpu.HBM(kv_cache_ro.shape, kv_cache_ro.dtype)
    )
    out_shape = (
        jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        kv_out_sd,
    )
    out_specs = (h_sp, v_sp, v_sp, h_sp)
    kv_in_idx = 5 + 12 + (1 if has_rope else 0)

    scratch_shapes = [
        vmem((hidden, max(qkv_cols, pre_cols)), next_w_qkv_w8k.dtype),  # slab_a
        vmem((hidden, max(post_cols, gate_cols)), next_w_gate_w8k.dtype),  # slab_b
        vmem((2, o_in, half), wo_w8k.dtype),  # wo_buf
        vmem((hidden, n_exp), w_router_km.dtype),  # w_router_buf
        vmem((_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128), kv_cache_ro.dtype),
        vmem((rows, hidden), jnp.float32),  # s_buf
        vmem((rows, hidden), jnp.bfloat16),  # next_in_vmem / ffn_vmem
        vmem((rows, qkv_cols), jnp.bfloat16),  # qkv_vmem
        vmem((rows, 16, 128), jnp.bfloat16),  # q_vmem_ref
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),  # kv_merged_vmem_ref
        vmem((rows, 16, 128), jnp.bfloat16),  # o_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gate_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gated_vmem_ref
        vmem((rows, 1, 4, 128), kv_cache_ro.dtype),  # kvw_rows_vmem
        vmem((rows, post_cols), jnp.bfloat16),  # post_shard
        vmem((buf_rows, hidden), jnp.bfloat16),  # b2_half_a
        vmem((buf_rows, hidden), jnp.bfloat16),  # b2_half_b
        vmem((r_per, w_post_w8k.shape[0]) if sharded_moe_in else (8, 128), jnp.bfloat16),
        vmem((rows, w_post_w8k.shape[0]) if sharded_moe_in else (8, 128), jnp.bfloat16),
        vmem((8, 128) if is_r16 else ((bm_g, hidden) if tile_r32 else (rows, hidden)), jnp.float32),
        vmem((rows, 1) if tile_r32 else (8, 1), jnp.float32),
        vmem((rows, half), jnp.bfloat16),  # send_half
        vmem((rows, half), jnp.bfloat16),  # mate_recv
        vmem((rows, half), jnp.bfloat16),  # own_half
        vmem((n_sc_slots, rows, chunk), jnp.bfloat16),  # scatter_ref
        vmem((8, 128) if use_xy else (rows, chunk), jnp.bfloat16),  # own_chunk
        vmem((rows, 128), jnp.float32),  # sq_buf
        vmem((rows, pre_cols), jnp.bfloat16),  # pre_shard
        vmem((8, 128) if reuse_pre_ab else (buf_rows, h_pre), jnp.bfloat16),
        vmem((8, 128) if (reuse_pre_ab or use_halfrow) else (buf_rows, h_pre), jnp.bfloat16),
        smem((2 + 2 * _NUM_KV_SLOTS,), jnp.int32),  # rpa_smem_ref
        sem((1,)),  # w_post_gate_sem
        sem((1,)),  # w_qkv_pre_sem
        sem((2,)),  # wo_sem
        sem((1,)),  # w_router_sem
        sem((1,)),  # s_io_sem
        sem((_NUM_KV_SLOTS,)),  # bkv_sems
        sem((1,)),  # kvw_out_sem
        sem((1,)),  # d2d_p0_send
        sem((1,)),  # d2d_p0_recv
        sem((1,)),  # d2d_p1_send
        sem((1,)),  # d2d_p1_recv
        sem((1,)),  # d2d_p2_send
        sem((1,)),  # d2d_p2_recv
        sem((n_ici_p0,)),  # ici_p0_send
        sem((n_ici_p0,)),  # ici_p0_recv
        sem((n_chips - 1,)),  # ici_p1_send
        sem((n_chips - 1,)),  # ici_p1_recv
        sem((n_ici_p2,)),  # ici_p2_send
        sem((n_ici_p2,)),  # ici_p2_recv
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(1,),
        scratch_shapes=scratch_shapes,
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=60 * 1024**2,
        collective_id=collective_id,
        disable_bounds_checks=True,
    )
    rope_args = (cos_sin.astype(jnp.float32),) if has_rope else ()
    kv_ro_arg = (
        kv_cache_ro
        if interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    sh_tag = "sh" if sharded_moe_in else "rep"
    name = (
        f"muse_m2m_rep_tp{world_size}_b{rows}_{sh_tag}_{sw_tag}_{d2_tag}_"
        f"{'rope' if has_rope else 'nope'}"
    )

    st_out, logits_out, exp_in_out, kv_cache_out = _pallas_call_maybe_interpret(
        functools.partial(
            _m2m_rep_kernel,
            world_size=world_size,
            sharded_moe_in=bool(sharded_moe_in),
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            sm_scale=float(sm_scale),
            sliding_window=sliding_window,
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            kv_pack_dense2=bool(kv_pack_dense2),
            layer_num_in_bundle=int(layer_num_in_bundle),
            layer_offset_in_bundle=int(layer_offset_in_bundle),
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            has_rope=has_rope,
            interpret_mode=interpret,
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases={kv_in_idx: 3},
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32_row(s_post, post_cols),
        _as_2d_f32_row(post_ff_gain, hidden),
        _as_2d_f32_row(pf_alpha, hidden),
        _as_2d_f32_row(pf_beta, hidden),
        _as_2d_f32_row(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_2d_f32_row(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_2d_f32_row(next_s_gate, gate_cols),
        *rope_args,
        kv_ro_arg,
        wo_w8k,
        _as_2d_f32_row(wo_scale, hidden),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_2d_f32_row(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32_row(pre_exp_gain, h_pre),
    )
    return st_out, logits_out, exp_in_out, kv_cache_out


def l0_a2m_rep_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream,
    attn_in,
    w_qkv_w8k,
    s_qkv,
    w_gate_w8k,
    s_gate,
    cos_sin,
    kv_cache_ro,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_w8k,
    w_pre_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    sm_scale: float = 0.125,
    sliding_window: int | None = 2048,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_pack_dense2: bool = False,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE + 3,
    interpret: bool = False,
):
    """Layer-0 fused `A2M` (`TP in (8, 16)`, replicated stream)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, hidden = stream.shape
    if rows < 8 or rows > 128 or rows % 8 != 0:
        raise ValueError(f"Unsupported l0_a2m_rep rows={rows}")

    half = hidden // 2
    chunk = hidden // world_size
    qkv_cols = w_qkv_w8k.shape[1]
    gate_cols = w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_cols = w_pre_w8k.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_km.shape[1]
    n_chips = world_size // 2
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ar_sems = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)
    is_r16 = w_router_km.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    bm_g = _pick_strip(rows, 64)
    use_halfrow = rows >= 32 and rows % 32 == 0
    buf_rows = rows // 2 if use_halfrow else rows
    reuse_pre_ab = h_pre == half
    has_rope = cos_sin is not None
    b_pad32 = align_to(rows, 32)

    phys_page_rows = (
        kv_cache_ro.shape[2] if kv_cache_ro.ndim == 6 else kv_cache_ro.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_ring_pages = _max_m2m_ring_pages(
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
    )

    vmem, smem, hbm, sem = pltpu.VMEM, pltpu.SMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    in_specs = [h_sp, v_sp, h_sp, v_sp, h_sp, v_sp]
    if has_rope:
        in_specs.append(v_sp)
    in_specs.extend([h_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp])

    kv_out_sd = (
        jax.ShapeDtypeStruct(kv_cache_ro.shape, kv_cache_ro.dtype)
        if interpret
        else pltpu.HBM(kv_cache_ro.shape, kv_cache_ro.dtype)
    )
    out_shape = (
        jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        kv_out_sd,
    )
    out_specs = (h_sp, v_sp, v_sp, h_sp)
    kv_in_idx = 5 + 6 + (1 if has_rope else 0)

    scratch_shapes = [
        vmem((hidden, max(qkv_cols, pre_cols)), w_qkv_w8k.dtype),
        vmem((hidden, gate_cols), w_gate_w8k.dtype),
        vmem((2, o_in, half), wo_w8k.dtype),
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem((_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128), kv_cache_ro.dtype),
        vmem((rows, hidden), jnp.float32),
        vmem((rows, hidden), jnp.bfloat16),
        vmem((rows, qkv_cols), jnp.bfloat16),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
        vmem((rows, 1, 4, 128), kv_cache_ro.dtype),
        vmem((8, 128) if is_r16 else ((bm_g, hidden) if tile_r32 else (rows, hidden)), jnp.float32),
        vmem((rows, 1) if tile_r32 else (8, 1), jnp.float32),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((n_sc_slots, rows, chunk), jnp.bfloat16),
        vmem((8, 128) if use_xy else (rows, chunk), jnp.bfloat16),
        vmem((rows, 128), jnp.float32),
        vmem((rows, pre_cols), jnp.bfloat16),
        vmem((8, 128) if reuse_pre_ab else (buf_rows, h_pre), jnp.bfloat16),
        vmem((8, 128) if (reuse_pre_ab or use_halfrow) else (buf_rows, h_pre), jnp.bfloat16),
        smem((2 + 2 * _NUM_KV_SLOTS,), jnp.int32),
        sem((1,)),
        sem((1,)),
        sem((2,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((_NUM_KV_SLOTS,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((n_ar_sems,)),
        sem((n_ar_sems,)),
        sem((n_ar_sems,)),
        sem((n_ar_sems,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((n_chips - 1,)),
        sem((n_chips - 1,)),
        sem((1,)),
        sem((1,)),
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(1,),
        scratch_shapes=scratch_shapes,
    )
    cp = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",),
        vmem_limit_bytes=60 * 1024**2,
        collective_id=collective_id,
        disable_bounds_checks=True,
    )
    rope_args = (cos_sin.astype(jnp.float32),) if has_rope else ()
    kv_ro_arg = (
        kv_cache_ro
        if interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    name = f"muse_l0_a2m_rep_tp{world_size}_b{rows}_{sw_tag}_{d2_tag}_{'rope' if has_rope else 'nope'}"

    st_out, logits_out, exp_in_out, kv_cache_out = _pallas_call_maybe_interpret(
        functools.partial(
            _l0_a2m_rep_kernel,
            world_size=world_size,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            sm_scale=float(sm_scale),
            sliding_window=sliding_window,
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            kv_pack_dense2=bool(kv_pack_dense2),
            layer_num_in_bundle=int(layer_num_in_bundle),
            layer_offset_in_bundle=int(layer_offset_in_bundle),
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
            has_rope=has_rope,
            interpret_mode=interpret,
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        input_output_aliases={kv_in_idx: 3},
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream.astype(jnp.float32),
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_2d_f32_row(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_2d_f32_row(s_gate, gate_cols),
        *rope_args,
        kv_ro_arg,
        wo_w8k,
        _as_2d_f32_row(wo_scale, hidden),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_2d_f32_row(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32_row(pre_exp_gain, h_pre),
    )
    return st_out, logits_out, exp_in_out, kv_cache_out
