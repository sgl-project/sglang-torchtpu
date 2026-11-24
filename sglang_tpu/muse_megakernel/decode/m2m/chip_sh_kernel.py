# SPDX-License-Identifier: Apache-2.0
"""Chip-Sharded (`R1`) Wave-9 `M2M` & `L0_A2M` Decode Megakernels (`TP = 8` & `TP = 16`).

Fuses `@w9-k-bridges` R1 chip-sharded stream (`[B // (TP // 2), 8192]` `f32`) directly with
`M2M v4` (`4`-sequence branchless unrolled `RPAd`, VREG-only `V` sanitizer, hoisted `new-KV`,
in-kernel overlapped 5D `kv_cache` scatter, and deferred Phase-3 weight DMAs):
  - `TP = 8` (`2x2x1`): `stream_chip [B // 4, 8192]` (`r_pair = 16` at `B = 64`), eliminating
    both `op_d2d1` in Phase-1 and `op_pair`/`op_rpair` + `concat` in Phase-3 (`0` stream HBM
    roundtrip between Bridge-2b and Bridge-1b).
  - `TP = 16` (`2x2x2`): `stream_chip [B // 8, 8192]` (`r_pair = 8` at `B = 64`, `8x` smaller
    stream than `m2m_rep_jax`), collapsing all 5+ per-layer `TP=16` launches into 1 Pallas call.
"""

# PEP 563 must stay off so cpu_export_jax_op sees concrete jax.Array annotations.

import functools

from sglang_tpu.attention.pallas.util import align_to
from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _COL_STRIP,
    _DOT_TN,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _fori_strips,
    _pick_strip,
    _signal_barrier,
    _wait_barrier,
)
from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    _ag_chip_sharded_half_col_xor1,
    _as_2d_f32,
    _dot_fp8_w8_slice_to_ref,
    _fold_cols_to_128,
    _pallas_call_maybe_interpret,
    _run_chip_sharded_post_fast_core,
    _strip_unroll,
)
from sglang_tpu.muse_megakernel.decode.m2m.kernel import (
    _M2M_CID_BASE,
    _NUM_KV_SLOTS,
    _kvw_1layer_jax,
    _llo_fence,
    _max_m2m_ring_pages,
    _prologue_prefetch_kv_pages,
    _run_in_vmem_qkv_epilogue_any,
    _run_in_vmem_rpad_and_gate,
)


def _run_chip_sharded_b1_from_vmem_core(
    s_buf,  # f32[r_pair, hidden] already resident in VMEM!
    gated_ref,  # bf16[rows, o_in] already resident in VMEM!
    wo_buf,  # fp8[2, o_in, half] DMA already started!
    wo_sem,
    wo_scale_ref,
    pa_alpha_ref,
    pa_beta_ref,
    pre_ff_gain_ref,
    w_pre_pair_buf,  # fp8[hidden, pre_pair_cols] DMA already started!
    w_pre_sem,
    w_pre_pair_scale_ref,
    w_router_buf,  # bf16/f32[hidden, n_exp] DMA already started!
    w_router_sem,
    pre_exp_gain_ref,
    out_stream_chip_hbm,
    out_router_logits_ref,
    out_expert_in_ref,
    send_half,
    mate_recv,
    own_half,
    rs1_a_or_z_recv,
    rs1_b_or_y_recv,
    rs2_a_or_x_recv,
    rs2_b_recv,
    my_pair_bf16,
    recv_pair_bf16,
    sq_pair_pa_buf,
    ffn_pair_buf,
    router_pair_buf,
    pre_my_half,
    pre_mate_half,
    sq_pair_pre_buf,
    exp_3d_buf,
    s_out_sem,
    c1a_send,
    c1a_recv,
    rs1_send,
    rs1_recv,
    rs2_send,
    rs2_recv,
    rs3_send,
    rs3_recv,
    c1c_send,
    c1c_recv,
    pre_send,
    pre_recv,
    ag_exp_send,
    ag_exp_recv,
    ag_log_send,
    ag_log_recv,
    *,
    me,
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    need_wait_barrier: bool = False,
    distribution_ref=None,
) -> None:
    """In-VMEM R1 chip-sharded Bridge-1b (`TP=8` & `TP=16`) consuming resident `s_buf` + `gated_ref`."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    r_pair, hidden = s_buf.shape
    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    half = hidden // 2
    q_cols = half // 2
    pre_pair_cols = w_pre_pair_buf.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_buf.shape[1]
    n_chips = world_size // 2
    r_align = 16 if (r_pair % 16 == 0) else 8

    bm_full = _pick_strip(rows, 32)
    bm_pair = _pick_strip(r_pair, 32)
    n_strips = hidden // _COL_STRIP
    half_strips = half // _COL_STRIP
    pre_half_strips = pre_pair_cols // _COL_STRIP
    u_full = _strip_unroll(n_strips, 4)
    u_half = _strip_unroll(half_strips, 4)
    u_pre = _strip_unroll(pre_half_strips, 4)
    q_step = 512 if (q_cols % 512 == 0) else _COL_STRIP
    is_r16 = w_router_buf.dtype == jnp.bfloat16

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)
    cx = chip & jnp.int32(1)
    cy = (chip >> jnp.int32(1)) & jnp.int32(1)
    cz = (chip >> jnp.int32(2)) & jnp.int32(1)
    peer_x = me ^ jnp.int32(2)
    peer_y = me ^ jnp.int32(4)
    peer_z = me ^ jnp.int32(8)

    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)
    chip_r0 = pl.multiple_of(chip * r_pair, r_align)
    own_pre_c0 = pl.multiple_of(side * pre_pair_cols, 128)
    other_pre_c0 = pl.multiple_of(other_side * pre_pair_cols, 128)

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0,
        rows=rows, n_cols=half,
    )

    if need_wait_barrier:
        _wait_barrier(world_size)
    op_c1a = pltpu.make_async_remote_copy(
        send_half, mate_recv, c1a_send.at[0], c1a_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1a.start()

    _dma_wait(wo_buf.at[1], wo_sem.at[1])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[1], wo_scale_ref, own_c0, own_half, 0,
        rows=rows, n_cols=half,
    )
    op_c1a.wait()

    if rows <= 128:
        for r0 in range(0, rows, 32):
            rs = pl.ds(r0, 32)
            for c0 in range(0, half, 512):
                c_sl = pl.ds(c0, 512)
                v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                own_half[rs, c_sl] = v.astype(own_half.dtype)
    else:

        def _sum_c1a(r0):
            rs = pl.ds(r0, bm_full)

            def _col_grp(g, _):
                base_s = g * u_half
                for sub in range(u_half):
                    c_sl = pl.ds(
                        pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                        _COL_STRIP,
                    )
                    v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                    own_half[rs, c_sl] = v.astype(own_half.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, half_strips // u_half, _col_grp, jnp.int32(0))

        _fori_strips(rows, bm_full, _sum_c1a)

    if world_size == 8 and r_pair >= 16:
        a_send_r1 = pl.multiple_of((jnp.int32(1) - cy) * two_pair, 32)
        a_keep_r1 = pl.multiple_of(cy * two_pair, 32)
        b_send_r1_0 = pl.multiple_of((jnp.int32(1) - cx) * r_pair, 16)
        b_send_r1_1 = pl.multiple_of(two_pair + (jnp.int32(1) - cx) * r_pair, 16)
        b_keep_r1_0 = pl.multiple_of(cx * r_pair, 16)
        b_keep_r1_1 = pl.multiple_of(two_pair + cx * r_pair, 16)

        op_rs1_a = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(a_send_r1, two_pair), pl.ds(0, q_cols)],
            rs1_a_or_z_recv,
            rs1_send.at[0],
            rs1_recv.at[0],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_b0 = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r1_0, r_pair), pl.ds(q_cols, q_cols)],
            rs1_b_or_y_recv.at[pl.ds(0, r_pair), :],
            rs1_send.at[1],
            rs1_recv.at[1],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_b1 = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r1_1, r_pair), pl.ds(q_cols, q_cols)],
            rs1_b_or_y_recv.at[pl.ds(r_pair, r_pair), :],
            rs1_send.at[2],
            rs1_recv.at[2],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs1_a.start()
        op_rs1_b0.start()
        op_rs1_b1.start()
        op_rs1_a.wait()
        op_rs1_b0.wait()
        op_rs1_b1.wait()

        for sub in range(2):
            r_loc = sub * r_pair
            ra = pl.ds(pl.multiple_of(a_keep_r1 + r_loc, 16), r_pair)
            rb = pl.ds(b_keep_r1_0 if sub == 0 else b_keep_r1_1, r_pair)
            rl = pl.ds(r_loc, r_pair)
            for off in range(0, q_cols, q_step):
                ca = pl.ds(off, q_step)
                cb = pl.ds(q_cols + off, q_step)
                va = own_half[ra, ca].astype(jnp.float32) + rs1_a_or_z_recv[rl, ca].astype(jnp.float32)
                vb = own_half[rb, cb].astype(jnp.float32) + rs1_b_or_y_recv[rl, ca].astype(jnp.float32)
                own_half[ra, ca] = va.astype(own_half.dtype)
                own_half[rb, cb] = vb.astype(own_half.dtype)

        a_send_r2 = pl.multiple_of(a_keep_r1 + (jnp.int32(1) - cx) * r_pair, 16)
        b_send_r2 = pl.multiple_of((jnp.int32(1) - cy) * two_pair + cx * r_pair, 16)

        op_rs2_a = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(a_send_r2, r_pair), pl.ds(0, q_cols)],
            rs2_a_or_x_recv,
            rs2_send.at[0],
            rs2_recv.at[0],
            device_id=(peer_x,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs2_b = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(b_send_r2, r_pair), pl.ds(q_cols, q_cols)],
            rs2_b_recv,
            rs2_send.at[1],
            rs2_recv.at[1],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs2_a.start()
        op_rs2_b.start()
        op_rs2_a.wait()
        op_rs2_b.wait()

        rc_chip = pl.ds(pl.multiple_of(chip * r_pair, 16), r_pair)
        for off in range(0, q_cols, q_step):
            ca = pl.ds(off, q_step)
            cb = pl.ds(q_cols + off, q_step)
            my_pair_bf16[:, ca] = (
                own_half[rc_chip, ca].astype(jnp.float32) + rs2_a_or_x_recv[:, ca].astype(jnp.float32)
            ).astype(jnp.bfloat16)
            my_pair_bf16[:, cb] = (
                own_half[rc_chip, cb].astype(jnp.float32) + rs2_b_recv[:, ca].astype(jnp.float32)
            ).astype(jnp.bfloat16)
    else:
        if world_size == 16:
            z_send_r = pl.multiple_of((jnp.int32(1) - cz) * four_pair, 16)
            z_keep_r = pl.multiple_of(cz * four_pair, 16)
            op_rs_z = pltpu.make_async_remote_copy(
                own_half.at[pl.ds(z_send_r, four_pair), :],
                rs1_a_or_z_recv,
                rs1_send.at[0],
                rs1_recv.at[0],
                device_id=(peer_z,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_z.start()
            op_rs_z.wait()
            bm_z = _pick_strip(four_pair, 32)
            if four_pair <= 64:
                for r0 in range(0, four_pair, bm_z):
                    rk = pl.ds(pl.multiple_of(z_keep_r + r0, 16), bm_z)
                    rl = pl.ds(r0, bm_z)
                    for c0 in range(0, half, 512):
                        cs = pl.ds(c0, 512)
                        v = own_half[rk, cs].astype(jnp.float32) + rs1_a_or_z_recv[rl, cs].astype(jnp.float32)
                        own_half[rl, cs] = v.astype(own_half.dtype)
            else:

                def _sum_z(r0):
                    rk = pl.ds(pl.multiple_of(z_keep_r + r0, 16), bm_z)
                    rl = pl.ds(r0, bm_z)

                    def _cz(g, _):
                        base_s = g * u_half
                        for sub in range(u_half):
                            cs = pl.ds(
                                pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                                _COL_STRIP,
                            )
                            v = own_half[rk, cs].astype(jnp.float32) + rs1_a_or_z_recv[rl, cs].astype(jnp.float32)
                            own_half[rl, cs] = v.astype(own_half.dtype)
                        return jnp.int32(0)

                    lax.fori_loop(0, half_strips // u_half, _cz, jnp.int32(0))

                _fori_strips(four_pair, bm_z, _sum_z)

        y_send_r = pl.multiple_of((jnp.int32(1) - cy) * two_pair, 16)
        y_keep_r = pl.multiple_of(cy * two_pair, 16)
        op_rs_y = pltpu.make_async_remote_copy(
            own_half.at[pl.ds(y_send_r, two_pair), :],
            rs1_b_or_y_recv,
            rs2_send.at[0],
            rs2_recv.at[0],
            device_id=(peer_y,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_rs_y.start()
        op_rs_y.wait()
        bm_y = _pick_strip(two_pair, 32)
        for r0 in range(0, two_pair, bm_y):
            rk = pl.ds(pl.multiple_of(y_keep_r + r0, 16), bm_y)
            rl = pl.ds(r0, bm_y)
            for c0 in range(0, half, 512):
                cs = pl.ds(c0, 512)
                v = own_half[rk, cs].astype(jnp.float32) + rs1_b_or_y_recv[rl, cs].astype(jnp.float32)
                own_half[rl, cs] = v.astype(own_half.dtype)

        c_step_x = min(half, 1024 if r_pair <= 16 else 512)
        if r_pair == 8:
            is_cx0 = cx == jnp.int32(0)
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                v16 = own_half[pl.ds(0, 16), cs]
                my_pair_bf16[:, cs] = lax.select(is_cx0, v16[0:8, :], v16[8:16, :])
                rs2_b_recv[:, cs] = lax.select(is_cx0, v16[8:16, :], v16[0:8, :])
            op_rs_x = pltpu.make_async_remote_copy(
                rs2_b_recv,
                rs2_a_or_x_recv,
                rs3_send.at[0],
                rs3_recv.at[0],
                device_id=(peer_x,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_x.start()
            op_rs_x.wait()
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                my_pair_bf16[:, cs] = (
                    my_pair_bf16[:, cs].astype(jnp.float32) + rs2_a_or_x_recv[:, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)
        else:
            x_send_r = pl.multiple_of((jnp.int32(1) - cx) * r_pair, 16)
            x_keep_r = pl.multiple_of(cx * r_pair, 16)
            op_rs_x = pltpu.make_async_remote_copy(
                own_half.at[pl.ds(x_send_r, r_pair), :],
                rs2_a_or_x_recv,
                rs3_send.at[0],
                rs3_recv.at[0],
                device_id=(peer_x,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_rs_x.start()
            op_rs_x.wait()
            rk_x = pl.ds(x_keep_r, r_pair)
            for c0 in range(0, half, c_step_x):
                cs = pl.ds(c0, c_step_x)
                my_pair_bf16[:, cs] = (
                    own_half[rk_x, cs].astype(jnp.float32) + rs2_a_or_x_recv[:, cs].astype(jnp.float32)
                ).astype(jnp.bfloat16)

    op_c1c = pltpu.make_async_remote_copy(
        my_pair_bf16,
        recv_pair_bf16,
        c1c_send.at[0],
        c1c_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    op_c1c.start()

    if r_pair <= 32:
        c_blk = min(half, 1024 if r_pair <= 16 else 512)
        h_blk = min(hidden, 1024 if r_pair <= 16 else 512)
        own_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = my_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        oth_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = recv_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
        pa_sq = own_sq + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )
        pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            c_l = pl.ds(c0, c_blk)
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            ns_own = (
                pa_alpha_ref[0:1, c_own] * s_buf[:, c_own]
                + pa_beta_ref[0:1, c_own] * (my_pair_bf16[:, c_l].astype(jnp.float32) * inv_pa)
            )
            ns_oth = (
                pa_alpha_ref[0:1, c_oth] * s_buf[:, c_oth]
                + pa_beta_ref[0:1, c_oth] * (recv_pair_bf16[:, c_l].astype(jnp.float32) * inv_pa)
            )
            s_buf[:, c_own] = ns_own
            s_buf[:, c_oth] = ns_oth
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            pff_sq = pff_sq + _fold_cols_to_128(v_own * v_own + v_oth * v_oth)
        pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        for c0 in range(0, hidden, h_blk):
            c_g = pl.ds(c0, h_blk)
            sv = s_buf[:, c_g]
            gn = pre_ff_gain_ref[0:1, c_g]
            rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
            if not is_r16:
                router_pair_buf[:, c_g] = rn
            ffn_pair_buf[:, c_g] = rn.astype(jnp.bfloat16)
    else:

        def _own_pa_sq(r0):
            rs = pl.ds(r0, bm_pair)

            def _col_grp(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    c_l = pl.ds(
                        pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                        _COL_STRIP,
                    )
                    bf = my_pair_bf16[rs, c_l].astype(jnp.float32)
                    sq = bf * bf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq_pair_pa_buf[rs, :] = lax.fori_loop(
                0, half_strips // u_half, _col_grp, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )

        _fori_strips(r_pair, bm_pair, _own_pa_sq)
        op_c1c.wait()

        def _finish_pa_and_pre_ffn_pair(r0):
            rs = pl.ds(r0, bm_pair)

            def _oth_sq(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    c_l = pl.ds(
                        pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                        _COL_STRIP,
                    )
                    bf = recv_pair_bf16[rs, c_l].astype(jnp.float32)
                    sq = bf * bf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            oth_sq = lax.fori_loop(
                0, half_strips // u_half, _oth_sq, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )
            inv_pa = lax.rsqrt(
                jnp.sum(sq_pair_pa_buf[rs, :] + oth_sq, axis=-1, keepdims=True)
                * jnp.float32(1.0 / hidden)
                + jnp.float32(pa_eps)
            )

            def _res_both(g, acc):
                base_s = g * u_half
                for sub in range(u_half):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    ns_own = (
                        pa_alpha_ref[0:1, c_own] * s_buf[rs, c_own]
                        + pa_beta_ref[0:1, c_own] * (my_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                    )
                    ns_oth = (
                        pa_alpha_ref[0:1, c_oth] * s_buf[rs, c_oth]
                        + pa_beta_ref[0:1, c_oth] * (recv_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                    )
                    s_buf[rs, c_own] = ns_own
                    s_buf[rs, c_oth] = ns_oth
                    v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
                    v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
                    sq = v_own * v_own + v_oth * v_oth
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            pff_sq = lax.fori_loop(
                0, half_strips // u_half, _res_both, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )
            inv_pff = lax.rsqrt(
                jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(pre_ff_eps)
            )

            def _pre_ffn_out(g, _):
                base_s = g * u_full
                for sub in range(u_full):
                    c_g = pl.ds(
                        pl.multiple_of((base_s + sub) * _COL_STRIP, _COL_STRIP),
                        _COL_STRIP,
                    )
                    sv = s_buf[rs, c_g]
                    gn = pre_ff_gain_ref[0:1, c_g]
                    rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
                    if not is_r16:
                        router_pair_buf[rs, c_g] = rn
                    ffn_pair_buf[rs, c_g] = rn.astype(jnp.bfloat16)
                return jnp.int32(0)

            lax.fori_loop(0, n_strips // u_full, _pre_ffn_out, jnp.int32(0))

        _fori_strips(r_pair, bm_pair, _finish_pa_and_pre_ffn_pair)
        pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()

    _dma_wait(w_pre_pair_buf, w_pre_sem.at[0])

    def _run_wpre_pair_dot():
        _dot_fp8_w8_slice_to_ref(
            ffn_pair_buf, 0, w_pre_pair_buf, w_pre_pair_scale_ref, 0, pre_my_half, 0, 0,
            n_r=r_pair, n_cols=pre_pair_cols, align=r_align,
        )

    if distribution_ref is not None:
        _llo_fence(distribution_ref, _run_wpre_pair_dot)
    else:
        _run_wpre_pair_dot()

    op_pre = pltpu.make_async_remote_copy(
        pre_my_half, pre_mate_half, pre_send.at[0], pre_recv.at[0],
        device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
    )
    op_pre.start()

    _dma_wait(w_router_buf, w_router_sem.at[0])
    rmul = jnp.float32(router_multiplier)
    r_lhs = ffn_pair_buf if is_r16 else router_pair_buf
    if r_pair <= 64 and n_exp == 256:
        lg = jnp.dot(r_lhs[...], w_router_buf[...], preferred_element_type=jnp.float32)
        out_router_logits_ref[pl.ds(chip_r0, r_pair), :] = lg * rmul
    else:
        for ce in range(0, n_exp, _DOT_TN):
            c_sl = pl.ds(ce, _DOT_TN)
            lg = jnp.dot(r_lhs[...], w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
            out_router_logits_ref[pl.ds(chip_r0, r_pair), c_sl] = lg * rmul

    ag_log_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op_l = pltpu.make_async_remote_copy(
            out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
            out_router_logits_ref.at[pl.ds(chip_r0, r_pair), :],
            ag_log_send.at[d - 1],
            ag_log_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_l.start()
        ag_log_ops.append(op_l)

    if r_pair <= 32:
        p_blk = min(pre_pair_cols, 1024 if r_pair <= 16 else 512)
        own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            xf = pre_my_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_own]
            own_pre_sq = own_pre_sq + _fold_cols_to_128(xf * xf)
        op_pre.wait()
        oth_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            xf = pre_mate_half[:, c_l].astype(jnp.float32)
            if pre_exp_scale_input:
                xf = xf * pre_exp_gain_ref[0:1, c_oth]
            oth_pre_sq = oth_pre_sq + _fold_cols_to_128(xf * xf)
        pre_sq = own_pre_sq + oth_pre_sq
        inv = lax.rsqrt(
            jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
            + jnp.float32(pre_exp_eps)
        )
        rs_g = pl.ds(chip_r0, r_pair)
        for c0 in range(0, pre_pair_cols, p_blk):
            c_l = pl.ds(c0, p_blk)
            c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
            c_oth = pl.ds(pl.multiple_of(other_pre_c0 + c0, 128), p_blk)
            x_own = pre_my_half[:, c_l].astype(jnp.float32)
            x_oth = pre_mate_half[:, c_l].astype(jnp.float32)
            g_own = pre_exp_gain_ref[0:1, c_own]
            g_oth = pre_exp_gain_ref[0:1, c_oth]
            y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
            y_oth = (x_oth * g_oth) * inv if pre_exp_scale_input else (x_oth * inv) * g_oth
            if r_pair == 8:
                exp_3d_buf[chip, :, c_own] = y_own.astype(out_expert_in_ref.dtype)
                exp_3d_buf[chip, :, c_oth] = y_oth.astype(out_expert_in_ref.dtype)
            else:
                out_expert_in_ref[rs_g, c_own] = y_own.astype(out_expert_in_ref.dtype)
                out_expert_in_ref[rs_g, c_oth] = y_oth.astype(out_expert_in_ref.dtype)
    else:

        def _own_pre_sq(r0):
            rs = pl.ds(r0, bm_pair)

            def _col_grp(g, acc):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    xf = pre_my_half[rs, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_own]
                    sq = xf * xf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            sq_pair_pre_buf[rs, :] = lax.fori_loop(
                0, pre_half_strips // u_pre, _col_grp, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )

        _fori_strips(r_pair, bm_pair, _own_pre_sq)
        op_pre.wait()

        def _finish_pre_exp_norm(r0):
            rs_l = pl.ds(r0, bm_pair)
            rs_g = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm_pair)

            def _oth_sq(g, acc):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    xf = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_oth]
                    sq = xf * xf
                    acc = acc + (sq[:, 0:128] + sq[:, 128:256])
                return acc

            oth_sq = lax.fori_loop(
                0, pre_half_strips // u_pre, _oth_sq, jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            )
            inv = lax.rsqrt(
                jnp.sum(sq_pair_pre_buf[rs_l, :] + oth_sq, axis=-1, keepdims=True)
                * jnp.float32(1.0 / h_pre)
                + jnp.float32(pre_exp_eps)
            )

            def _out_both(g, _):
                base_s = g * u_pre
                for sub in range(u_pre):
                    s = base_s + sub
                    c_l = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    c_oth = pl.ds(pl.multiple_of(other_pre_c0 + s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                    x_own = pre_my_half[rs_l, c_l].astype(jnp.float32)
                    x_oth = pre_mate_half[rs_l, c_l].astype(jnp.float32)
                    g_own = pre_exp_gain_ref[0:1, c_own]
                    g_oth = pre_exp_gain_ref[0:1, c_oth]
                    y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
                    y_oth = (x_oth * g_oth) * inv if pre_exp_scale_input else (x_oth * inv) * g_oth
                    out_expert_in_ref[rs_g, c_own] = y_own.astype(out_expert_in_ref.dtype)
                    out_expert_in_ref[rs_g, c_oth] = y_oth.astype(out_expert_in_ref.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, pre_half_strips // u_pre, _out_both, jnp.int32(0))

        _fori_strips(r_pair, bm_pair, _finish_pre_exp_norm)

    ag_exp_ops = []
    if r_pair == 8:
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op_e = pltpu.make_async_remote_copy(
                exp_3d_buf.at[chip],
                exp_3d_buf.at[chip],
                ag_exp_send.at[d - 1],
                ag_exp_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_e.start()
            ag_exp_ops.append(op_e)
        for op in ag_log_ops:
            op.wait()
        for op in ag_exp_ops:
            op.wait()
        e_w = min(h_pre, 1024)
        for p in range(n_chips // 2):
            r16 = pl.ds(16 * p, 16)
            for c0 in range(0, h_pre, e_w):
                c_sl = pl.ds(c0, e_w)
                out_expert_in_ref[r16, c_sl] = jnp.concatenate(
                    [exp_3d_buf[2 * p, :, c_sl], exp_3d_buf[2 * p + 1, :, c_sl]], axis=0
                )
    else:
        for d in range(1, n_chips):
            peer = me ^ jnp.int32(d << 1)
            op_e = pltpu.make_async_remote_copy(
                out_expert_in_ref.at[pl.ds(chip_r0, r_pair), :],
                out_expert_in_ref.at[pl.ds(chip_r0, r_pair), :],
                ag_exp_send.at[d - 1],
                ag_exp_recv.at[d - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_e.start()
            ag_exp_ops.append(op_e)
        for op in ag_log_ops:
            op.wait()
        for op in ag_exp_ops:
            op.wait()
    _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def _m2m_chip_sh_kernel(
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
    unroll_quad: bool = False,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_chip_hbm = body_refs[idx]
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
    w_pre_pair_hbm = body_refs[idx + 6]
    w_pre_pair_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    out_stream_chip_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kvw_rows_ref = body_refs[idx + 3]
    idx += 4

    slab_a = body_refs[idx]
    slab_b = body_refs[idx + 1]
    wo_buf = body_refs[idx + 2]
    w_router_buf = body_refs[idx + 3]
    bkv_vmem_ref = body_refs[idx + 4]
    s_buf = body_refs[idx + 5]
    next_in_pair = body_refs[idx + 6]
    next_in_vmem = body_refs[idx + 7]
    qkv_vmem = body_refs[idx + 8]
    q_vmem_ref = body_refs[idx + 9]
    kv_merged_vmem_ref = body_refs[idx + 10]
    o_vmem_ref = body_refs[idx + 11]
    gate_vmem_ref = body_refs[idx + 12]
    gated_vmem_ref = body_refs[idx + 13]
    mate_tok_buf = body_refs[idx + 14]
    moe_full_buf = body_refs[idx + 15]
    post_shard = body_refs[idx + 16]
    mate_post_shard = body_refs[idx + 17]
    chip_cols_buf = body_refs[idx + 18]
    post_pair_full = body_refs[idx + 19]  # reused as ffn_pair_buf in Phase 3!
    cols_3d_buf = body_refs[idx + 20]
    ag_exp_3d_buf = body_refs[idx + 21]  # reused across ag_3d_buf (P1) & exp_3d_buf (P3)!
    send_half = body_refs[idx + 22]
    mate_recv = body_refs[idx + 23]
    own_half = body_refs[idx + 24]
    rs1_a_or_z_recv = body_refs[idx + 25]
    rs1_b_or_y_recv = body_refs[idx + 26]
    rs2_a_or_x_recv = body_refs[idx + 27]
    rs2_b_recv = body_refs[idx + 28]
    my_pair_bf16 = body_refs[idx + 29]
    recv_pair_bf16 = body_refs[idx + 30]
    sq_pair_pa_buf = body_refs[idx + 31]
    router_pair_buf = body_refs[idx + 32]
    pre_my_half = body_refs[idx + 33]
    pre_mate_half = body_refs[idx + 34]
    sq_pair_pre_buf = body_refs[idx + 35]
    init_sv_vmem_ref = body_refs[idx + 36]
    rpa_smem_ref = body_refs[idx + 37]
    idx += 38

    # Semaphores (`46` on TP=8, `56` on TP=16 — strictly < 64 hardware limit!):
    w_post_gate_sem = body_refs[idx]
    w_qkv_pre_sem = body_refs[idx + 1]
    wo_sem = body_refs[idx + 2]
    w_router_sem = body_refs[idx + 3]
    s_io_sem = body_refs[idx + 4]
    bkv_sems = body_refs[idx + 5]
    tok_s1_send = body_refs[idx + 6]
    tok_s1_recv = body_refs[idx + 7]
    ici_grp_a_send = body_refs[idx + 8]   # tok_s2 -> in_ici -> ag_exp (causally closed!)
    ici_grp_a_recv = body_refs[idx + 9]
    post_d2d_send = body_refs[idx + 10]
    post_d2d_recv = body_refs[idx + 11]
    ici_grp_b_send = body_refs[idx + 12]  # post_ici -> ag_log (causally closed!)
    ici_grp_b_recv = body_refs[idx + 13]
    in_d2d2_send = body_refs[idx + 14]
    in_d2d2_recv = body_refs[idx + 15]
    c1a_send = body_refs[idx + 16]
    c1a_recv = body_refs[idx + 17]
    rs1_send = body_refs[idx + 18]
    rs1_recv = body_refs[idx + 19]
    rs2_send = body_refs[idx + 20]
    rs2_recv = body_refs[idx + 21]
    rs3_send = body_refs[idx + 22]
    rs3_recv = body_refs[idx + 23]
    c1c_send = body_refs[idx + 24]
    c1c_recv = body_refs[idx + 25]
    pre_send = body_refs[idx + 26]
    pre_recv = body_refs[idx + 27]

    r_pair, hidden = stream_chip_hbm.shape
    rows = (world_size // 2) * r_pair
    half = hidden // 2
    post_cols = w_post_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    phys_page_rows = (
        kv_cache_ro_hbm.shape[2] if kv_cache_ro_hbm.ndim == 6 else kv_cache_ro_hbm.shape[1]
    )
    page_size = phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows

    me = lax.axis_index("model").astype(jnp.int32)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    own_c0 = pl.multiple_of(side * half, 128)
    other_c0 = pl.multiple_of(other_side * half, 128)

    w_post_buf = slab_b.at[pl.ds(0, w_post_hbm.shape[0]), pl.ds(0, post_cols)]
    w_qkv_buf = slab_a.at[:, pl.ds(0, qkv_cols)]
    w_gate_buf = slab_b.at[:, pl.ds(0, gate_cols)]

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(stream_chip_hbm, s_buf, s_io_sem.at[0]).start()
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

    _llo_fence(
        distribution_ref,
        lambda: _run_chip_sharded_post_fast_core(
            moe_out_ref,
            w_post_buf,
            s_post_ref,
            s_buf,
            post_ff_gain_ref,
            pf_alpha_ref,
            pf_beta_ref,
            next_in_gain_ref,
            next_in_pair,
            mate_tok_buf,
            moe_full_buf,
            post_shard,
            mate_post_shard,
            chip_cols_buf,
            post_pair_full,
            cols_3d_buf,
            w_post_gate_sem,
            s_io_sem,
            tok_s1_send,
            tok_s1_recv,
            ici_grp_a_send,
            ici_grp_a_recv,
            post_d2d_send,
            post_d2d_recv,
            ici_grp_b_send,
            ici_grp_b_recv,
            me=me,
            world_size=world_size,
            sharded_moe_in=sharded_moe_in,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps,
            next_in_scale_input=next_in_scale_input,
            round_stream_bf16_for_norm=False,
            on_stream_updated_fn=None,
        ),
    )

    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_post_gate_sem.at[0]).start()

    _ag_chip_sharded_half_col_xor1(
        next_in_pair,
        next_in_vmem,
        ici_grp_a_send,
        ici_grp_a_recv,
        in_d2d2_send,
        in_d2d2_recv,
        me=me,
        world_size=world_size,
        ag_3d_buf=ag_exp_3d_buf,
        during_d2d2_fn=lambda: _dma_wait(w_qkv_buf, w_qkv_pre_sem.at[0]),
    )

    _llo_fence(
        distribution_ref,
        lambda: _dot_fp8_scaled_to_ref(
            next_in_vmem,
            w_qkv_buf,
            next_s_qkv_ref,
            0,
            qkv_vmem,
            0,
            rows=rows,
            n_cols=qkv_cols,
        ),
    )

    eff_kv_merged_ref = (
        out_kvw_rows_ref
        if (not kv_pack_dense2 and rows == kv_merged_vmem_ref.shape[0])
        else kv_merged_vmem_ref
    )
    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        eff_kv_merged_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_chip():
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
        eff_kv_merged_ref,
        kv_cache_ro_hbm,
        out_kvw_rows_ref,
        bkv_vmem_ref,
        o_vmem_ref,
        next_in_vmem,
        w_gate_buf,
        w_post_gate_sem,
        next_s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
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
        on_rpad_kv_done_fn=_start_wo_chip,
        init_sv_vmem_ref=init_sv_vmem_ref,
        unroll_quad=unroll_quad,
    )

    pltpu.make_async_copy(w_pre_pair_hbm, slab_a, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    eff_send_half = moe_full_buf if moe_full_buf.shape == (rows, half) else send_half
    _run_chip_sharded_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        slab_a,
        w_qkv_pre_sem,
        w_pre_pair_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_chip_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
        eff_send_half,
        mate_recv,
        own_half,
        rs1_a_or_z_recv,
        rs1_b_or_y_recv,
        rs2_a_or_x_recv,
        rs2_b_recv,
        my_pair_bf16,
        recv_pair_bf16,
        sq_pair_pa_buf,
        post_pair_full,
        router_pair_buf,
        pre_my_half,
        pre_mate_half,
        sq_pair_pre_buf,
        ag_exp_3d_buf,
        s_io_sem,
        c1a_send,
        c1a_recv,
        rs1_send,
        rs1_recv,
        rs2_send,
        rs2_recv,
        rs3_send,
        rs3_recv,
        c1c_send,
        c1c_recv,
        pre_send,
        pre_recv,
        ici_grp_a_send,
        ici_grp_a_recv,
        ici_grp_b_send,
        ici_grp_b_recv,
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=False,
        distribution_ref=distribution_ref,
    )


def _l0_a2m_chip_sh_kernel(
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
    unroll_quad: bool = False,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    idx = 0
    stream_chip_hbm = body_refs[idx]
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
    w_pre_pair_hbm = body_refs[idx + 6]
    w_pre_pair_scale_ref = body_refs[idx + 7]
    w_router_hbm = body_refs[idx + 8]
    pre_exp_gain_ref = body_refs[idx + 9]
    idx += 10

    out_stream_chip_hbm = body_refs[idx]
    out_router_logits_ref = body_refs[idx + 1]
    out_expert_in_ref = body_refs[idx + 2]
    out_kvw_rows_ref = body_refs[idx + 3]
    idx += 4

    slab_a = body_refs[idx]
    w_gate_buf = body_refs[idx + 1]
    wo_buf = body_refs[idx + 2]
    w_router_buf = body_refs[idx + 3]
    bkv_vmem_ref = body_refs[idx + 4]
    s_buf = body_refs[idx + 5]
    qkv_vmem = body_refs[idx + 6]
    q_vmem_ref = body_refs[idx + 7]
    kv_merged_vmem_ref = body_refs[idx + 8]
    o_vmem_ref = body_refs[idx + 9]
    gate_vmem_ref = body_refs[idx + 10]
    gated_vmem_ref = body_refs[idx + 11]
    send_half = body_refs[idx + 12]
    mate_recv = body_refs[idx + 13]
    own_half = body_refs[idx + 14]
    rs1_a_or_z_recv = body_refs[idx + 15]
    rs1_b_or_y_recv = body_refs[idx + 16]
    rs2_a_or_x_recv = body_refs[idx + 17]
    rs2_b_recv = body_refs[idx + 18]
    my_pair_bf16 = body_refs[idx + 19]
    recv_pair_bf16 = body_refs[idx + 20]
    sq_pair_pa_buf = body_refs[idx + 21]
    ffn_pair_buf = body_refs[idx + 22]
    router_pair_buf = body_refs[idx + 23]
    pre_my_half = body_refs[idx + 24]
    pre_mate_half = body_refs[idx + 25]
    sq_pair_pre_buf = body_refs[idx + 26]
    exp_3d_buf = body_refs[idx + 27]
    init_sv_vmem_ref = body_refs[idx + 28]
    rpa_smem_ref = body_refs[idx + 29]
    idx += 30

    w_qkv_pre_sem = body_refs[idx]
    w_gate_sem = body_refs[idx + 1]
    wo_sem = body_refs[idx + 2]
    w_router_sem = body_refs[idx + 3]
    s_in_sem = body_refs[idx + 4]
    s_out_sem = body_refs[idx + 5]
    bkv_sems = body_refs[idx + 6]
    c1a_send = body_refs[idx + 7]
    c1a_recv = body_refs[idx + 8]
    rs1_send = body_refs[idx + 9]
    rs1_recv = body_refs[idx + 10]
    rs2_send = body_refs[idx + 11]
    rs2_recv = body_refs[idx + 12]
    rs3_send = body_refs[idx + 13]
    rs3_recv = body_refs[idx + 14]
    c1c_send = body_refs[idx + 15]
    c1c_recv = body_refs[idx + 16]
    pre_send = body_refs[idx + 17]
    pre_recv = body_refs[idx + 18]
    ag_exp_send = body_refs[idx + 19]
    ag_exp_recv = body_refs[idx + 20]
    ag_log_send = body_refs[idx + 21]
    ag_log_recv = body_refs[idx + 22]

    rows = attn_in_ref.shape[0]
    qkv_cols = w_qkv_hbm.shape[1]
    hidden = wo_hbm.shape[1]
    half = hidden // 2
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

    _signal_barrier(me, world_size)
    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    pltpu.make_async_copy(stream_chip_hbm, s_buf, s_in_sem.at[0]).start()
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

    eff_kv_merged_ref = (
        out_kvw_rows_ref
        if (not kv_pack_dense2 and rows == kv_merged_vmem_ref.shape[0])
        else kv_merged_vmem_ref
    )
    _run_in_vmem_qkv_epilogue_any(
        qkv_vmem,
        cos_sin_ref,
        q_vmem_ref,
        eff_kv_merged_ref.at[pl.ds(0, rows)],
        norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        distribution_ref=distribution_ref,
    )

    def _start_wo_l0_chip():
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
        eff_kv_merged_ref,
        kv_cache_ro_hbm,
        out_kvw_rows_ref,
        bkv_vmem_ref,
        o_vmem_ref,
        attn_in_ref,
        w_gate_buf,
        w_gate_sem,
        s_gate_ref,
        gate_vmem_ref,
        gated_vmem_ref,
        rpa_smem_ref,
        bkv_sems,
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
        on_rpad_kv_done_fn=_start_wo_l0_chip,
        init_sv_vmem_ref=init_sv_vmem_ref,
    )

    pltpu.make_async_copy(w_pre_pair_hbm, slab_a, w_qkv_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(s_buf, s_in_sem.at[0])

    _run_chip_sharded_b1_from_vmem_core(
        s_buf,
        gated_vmem_ref,
        wo_buf,
        wo_sem,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        slab_a,
        w_qkv_pre_sem,
        w_pre_pair_scale_ref,
        w_router_buf,
        w_router_sem,
        pre_exp_gain_ref,
        out_stream_chip_hbm,
        out_router_logits_ref,
        out_expert_in_ref,
        send_half,
        mate_recv,
        own_half,
        rs1_a_or_z_recv,
        rs1_b_or_y_recv,
        rs2_a_or_x_recv,
        rs2_b_recv,
        my_pair_bf16,
        recv_pair_bf16,
        sq_pair_pa_buf,
        ffn_pair_buf,
        router_pair_buf,
        pre_my_half,
        pre_mate_half,
        sq_pair_pre_buf,
        exp_3d_buf,
        s_out_sem,
        c1a_send,
        c1a_recv,
        rs1_send,
        rs1_recv,
        rs2_send,
        rs2_recv,
        rs3_send,
        rs3_recv,
        c1c_send,
        c1c_recv,
        pre_send,
        pre_recv,
        ag_exp_send,
        ag_exp_recv,
        ag_log_send,
        ag_log_recv,
        me=me,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        need_wait_barrier=True,
        distribution_ref=distribution_ref,
    )


def m2m_chip_sh_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream_chip,
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
    w_pre_pair_w8k,
    w_pre_pair_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int = 8,
    sharded_moe_in: bool = True,
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
    kv_pack_dense2: bool = True,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE + 4,
    interpret: bool = False,
    unroll_quad: bool = False,
):
    """Per-layer R1 chip-sharded `M2M` (`l >= 1`, `stream_chip [B//(TP//2), 8192]`, `TP in (8, 16)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"m2m_chip_sh_jax requires world_size in (8, 16), got {world_size}")
    n_chips = world_size // 2
    r_pair, hidden = stream_chip.shape
    rows = n_chips * r_pair
    if rows not in (64, 128) or r_pair < 8 or r_pair % 8 != 0:
        raise ValueError(f"Unsupported m2m_chip_sh rows={rows}, r_pair={r_pair}, world_size={world_size}")
    in_r, k_in = moe_out.shape
    expected_in_r = rows // world_size if sharded_moe_in else rows
    if in_r != expected_in_r or in_r < 8 or in_r % 8 != 0:
        raise ValueError(
            f"Unsupported m2m_chip_sh moe_out={moe_out.shape}, rows={rows}, sharded_moe_in={sharded_moe_in}"
        )

    two_pair = 2 * r_pair
    half = hidden // 2
    q_cols = half // 2
    post_cols = w_post_w8k.shape[1]
    pair_cols = 2 * post_cols
    qkv_cols = next_w_qkv_w8k.shape[1]
    gate_cols = next_w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_pair_cols = w_pre_pair_w8k.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_km.shape[1]
    is_r16 = w_router_km.dtype == jnp.bfloat16
    use_xy = world_size == 8 and r_pair >= 16
    has_rope = cos_sin is not None
    b_pad32 = align_to(rows, 32)
    can_alias_send = sharded_moe_in and k_in == half

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

    is_interpret = bool(interpret)
    out_shape = (
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 1, 4, 128), kv_cache_ro.dtype),
    )
    out_specs = (h_sp, v_sp, v_sp, v_sp)

    scratch_shapes = [
        vmem((hidden, pre_pair_cols), next_w_qkv_w8k.dtype),  # slab_a
        vmem((hidden, max(post_cols, gate_cols)), next_w_gate_w8k.dtype),  # slab_b
        vmem((2, o_in, half), wo_w8k.dtype),  # wo_buf
        vmem((hidden, n_exp), w_router_km.dtype),  # w_router_buf
        vmem((_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128), kv_cache_ro.dtype),
        vmem((r_pair, hidden), jnp.float32),  # s_buf
        vmem((r_pair, hidden), jnp.bfloat16),  # next_in_pair
        vmem((rows, hidden), jnp.bfloat16),  # next_in_vmem
        vmem((rows, qkv_cols), jnp.bfloat16),  # qkv_vmem
        vmem((rows, 16, 128), jnp.bfloat16),  # q_vmem_ref
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),  # kv_merged_vmem_ref
        vmem((rows, 16, 128), jnp.bfloat16),  # o_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gate_vmem_ref
        vmem((rows, gate_cols), jnp.bfloat16),  # gated_vmem_ref
        vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),  # mate_tok_buf
        vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),  # moe_full_buf
        vmem((rows, post_cols), jnp.bfloat16),  # post_shard
        vmem((rows, post_cols), jnp.bfloat16),  # mate_post_shard
        vmem((rows, pair_cols), jnp.bfloat16),  # chip_cols_buf
        vmem((r_pair, hidden), jnp.bfloat16),  # post_pair_full (reused as ffn_pair_buf!)
        vmem((n_chips, 8, pair_cols) if r_pair == 8 else (8, 128), jnp.bfloat16),  # cols_3d_buf
        vmem((n_chips, 8, max(half, h_pre)) if r_pair == 8 else (8, 128), jnp.bfloat16),  # ag_exp_3d_buf
        vmem((8, 128) if can_alias_send else (rows, half), jnp.bfloat16),  # send_half
        vmem((rows, half), jnp.bfloat16),  # mate_recv
        vmem((rows, half), jnp.bfloat16),  # own_half
        vmem(
            (two_pair, q_cols)
            if use_xy
            else ((4 * r_pair, half) if world_size == 16 else (8, 128)),
            jnp.bfloat16,
        ),  # rs1_a_or_z_recv
        vmem((two_pair, q_cols) if use_xy else (two_pair, half), jnp.bfloat16),  # rs1_b_or_y_recv
        vmem((r_pair, q_cols) if use_xy else (r_pair, half), jnp.bfloat16),  # rs2_a_or_x_recv
        vmem(
            (r_pair, q_cols)
            if use_xy
            else ((r_pair, half) if r_pair == 8 else (8, 128)),
            jnp.bfloat16,
        ),  # rs2_b_recv
        vmem((r_pair, half), jnp.bfloat16),  # my_pair_bf16
        vmem((r_pair, half), jnp.bfloat16),  # recv_pair_bf16
        vmem((r_pair, 128), jnp.float32),  # sq_pair_pa_buf
        vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32),  # router_pair_buf
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),  # pre_my_half
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),  # pre_mate_half
        vmem((r_pair, 128), jnp.float32),  # sq_pair_pre_buf
        vmem((2, b_pad32 if sliding_window is None else 1, 16, 128), jnp.float32),  # init_sv_vmem_ref
        smem((2 + 2 * _NUM_KV_SLOTS,), jnp.int32),  # rpa_smem_ref
        # Semaphores (`46` on TP=8, `56` on TP=16 — strictly < 64!):
        sem((1,)),  # w_post_gate_sem
        sem((1,)),  # w_qkv_pre_sem
        sem((2,)),  # wo_sem
        sem((1,)),  # w_router_sem
        sem((1,)),  # s_io_sem
        sem((_NUM_KV_SLOTS,)),  # bkv_sems
        sem((1,)),  # tok_s1_send
        sem((1,)),  # tok_s1_recv
        sem((n_chips - 1,)),  # ici_grp_a_send
        sem((n_chips - 1,)),  # ici_grp_a_recv
        sem((1,)),  # post_d2d_send
        sem((1,)),  # post_d2d_recv
        sem((n_chips - 1,)),  # ici_grp_b_send
        sem((n_chips - 1,)),  # ici_grp_b_recv
        sem((1,)),  # in_d2d2_send
        sem((1,)),  # in_d2d2_recv
        sem((1,)),  # c1a_send
        sem((1,)),  # c1a_recv
        sem((3 if use_xy else 1,)),  # rs1_send
        sem((3 if use_xy else 1,)),  # rs1_recv
        sem((2 if use_xy else 1,)),  # rs2_send
        sem((2 if use_xy else 1,)),  # rs2_recv
        sem((1,)),  # rs3_send
        sem((1,)),  # rs3_recv
        sem((1,)),  # c1c_send
        sem((1,)),  # c1c_recv
        sem((1,)),  # pre_send
        sem((1,)),  # pre_recv
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
        if is_interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    sh_tag = "sh" if sharded_moe_in else "full"
    name = (
        f"muse_m2m_chip_tp{world_size}_b{rows}_{sh_tag}_{sw_tag}_{d2_tag}_"
        f"{'rope' if has_rope else 'nope'}"
    )

    st_out, logits_out, exp_in_out, kvw_rows_out = _pallas_call_maybe_interpret(
        functools.partial(
            _m2m_chip_sh_kernel,
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
            interpret_mode=is_interpret,
            unroll_quad=bool(unroll_quad),
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream_chip.astype(jnp.float32),
        moe_out.astype(jnp.bfloat16),
        w_post_w8k,
        _as_2d_f32(s_post, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_2d_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_2d_f32(next_s_gate, gate_cols),
        *rope_args,
        kv_ro_arg,
        wo_w8k,
        _as_2d_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_pair_w8k,
        _as_2d_f32(w_pre_pair_scale, pre_pair_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, h_pre),
    )
    kv_cache_out = _kvw_1layer_jax(
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        kvw_rows_out,
        kv_cache_ro,
        kv_pack_dense2=bool(kv_pack_dense2),
        layer_num_in_bundle=int(layer_num_in_bundle),
        layer_offset_in_bundle=int(layer_offset_in_bundle),
        interpret=interpret,
    )
    return st_out, logits_out, exp_in_out, kv_cache_out


def l0_a2m_chip_sh_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    stream_chip,
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
    w_pre_pair_w8k,
    w_pre_pair_scale,
    w_router_km,
    pre_exp_gain,
    *,
    world_size: int = 8,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    sm_scale: float = 0.125,
    sliding_window: int | None = 2048,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_pack_dense2: bool = True,
    layer_num_in_bundle: int = 1,
    layer_offset_in_bundle: int = 0,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    collective_id: int = _M2M_CID_BASE + 5,
    interpret: bool = False,
):
    """Layer-0 fused R1 chip-sharded `A2M` (`stream_chip [B//(TP//2), 8192]`, `TP in (8, 16)`)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"l0_a2m_chip_sh_jax requires world_size in (8, 16), got {world_size}")
    n_chips = world_size // 2
    r_pair, hidden = stream_chip.shape
    rows = attn_in.shape[0]
    if rows not in (64, 128) or rows != n_chips * r_pair or r_pair < 8 or r_pair % 8 != 0:
        raise ValueError(f"Unsupported l0_a2m_chip_sh rows={rows}, r_pair={r_pair}, world_size={world_size}")

    two_pair = 2 * r_pair
    half = hidden // 2
    q_cols = half // 2
    qkv_cols = w_qkv_w8k.shape[1]
    gate_cols = w_gate_w8k.shape[1]
    o_in = wo_w8k.shape[0]
    pre_pair_cols = w_pre_pair_w8k.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_km.shape[1]
    is_r16 = w_router_km.dtype == jnp.bfloat16
    use_xy = world_size == 8 and r_pair >= 16
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

    is_interpret = bool(interpret)
    out_shape = (
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
        jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 1, 4, 128), kv_cache_ro.dtype),
    )
    out_specs = (h_sp, v_sp, v_sp, v_sp)

    scratch_shapes = [
        vmem((hidden, pre_pair_cols), w_qkv_w8k.dtype),
        vmem((hidden, gate_cols), w_gate_w8k.dtype),
        vmem((2, o_in, half), wo_w8k.dtype),
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem((_NUM_KV_SLOTS, max_ring_pages, phys_page_rows, 1, 4, 128), kv_cache_ro.dtype),
        vmem((r_pair, hidden), jnp.float32),
        vmem((rows, qkv_cols), jnp.bfloat16),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((b_pad32, 1, 4, 128), kv_cache_ro.dtype),
        vmem((rows, 16, 128), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
        vmem((rows, gate_cols), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem(
            (two_pair, q_cols)
            if use_xy
            else ((4 * r_pair, half) if world_size == 16 else (8, 128)),
            jnp.bfloat16,
        ),
        vmem((two_pair, q_cols) if use_xy else (two_pair, half), jnp.bfloat16),
        vmem((r_pair, q_cols) if use_xy else (r_pair, half), jnp.bfloat16),
        vmem(
            (r_pair, q_cols)
            if use_xy
            else ((r_pair, half) if r_pair == 8 else (8, 128)),
            jnp.bfloat16,
        ),
        vmem((r_pair, half), jnp.bfloat16),
        vmem((r_pair, half), jnp.bfloat16),
        vmem((r_pair, 128), jnp.float32),
        vmem((r_pair, hidden), jnp.bfloat16),
        vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32),
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),
        vmem((r_pair, pre_pair_cols), jnp.bfloat16),
        vmem((r_pair, 128), jnp.float32),
        vmem((n_chips, 8, h_pre) if r_pair == 8 else (8, 128), jnp.bfloat16),
        vmem((2, b_pad32 if sliding_window is None else 1, 16, 128), jnp.float32),
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
        sem((3 if use_xy else 1,)),
        sem((3 if use_xy else 1,)),
        sem((2 if use_xy else 1,)),
        sem((2 if use_xy else 1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((1,)),
        sem((n_chips - 1,)),
        sem((n_chips - 1,)),
        sem((n_chips - 1,)),
        sem((n_chips - 1,)),
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
        if is_interpret
        else pltpu.with_memory_space_constraint(kv_cache_ro, pltpu.HBM)
    )
    sw_tag = f"sw{sliding_window}" if sliding_window is not None else f"full{max_ring_pages}p"
    d2_tag = "d2" if kv_pack_dense2 else "nod2"
    name = f"muse_l0_a2m_chip_tp{world_size}_b{rows}_{sw_tag}_{d2_tag}_{'rope' if has_rope else 'nope'}"

    st_out, logits_out, exp_in_out, kvw_rows_out = _pallas_call_maybe_interpret(
        functools.partial(
            _l0_a2m_chip_sh_kernel,
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
            interpret_mode=is_interpret,
            unroll_quad=bool(unroll_quad),
        ),
        interpret=interpret,
        grid_spec=grid_spec,
        out_shape=out_shape,
        compiler_params=cp,
        name=name,
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        cu_kv_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        stream_chip.astype(jnp.float32),
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_2d_f32(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_2d_f32(s_gate, gate_cols),
        *rope_args,
        kv_ro_arg,
        wo_w8k,
        _as_2d_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_pair_w8k,
        _as_2d_f32(w_pre_pair_scale, pre_pair_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, h_pre),
    )
    kv_cache_out = _kvw_1layer_jax(
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        kvw_rows_out,
        kv_cache_ro,
        kv_pack_dense2=bool(kv_pack_dense2),
        layer_num_in_bundle=int(layer_num_in_bundle),
        layer_offset_in_bundle=int(layer_offset_in_bundle),
        interpret=interpret,
    )
    return st_out, logits_out, exp_in_out, kv_cache_out
