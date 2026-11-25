# SPDX-License-Identifier: Apache-2.0
"""Wave-9 Step (a) Decode Superkernel: Chip-Sharded Bridge 1b + Sparse NVFP4 MoE.

Fuses ``attn_to_moe_chip_sharded_bridge_jax`` (``chip_b1``) directly with
``fused_decode_sparse_from_logits`` (``muse_sparse`` ``fb0``) in a single
``pallas_call`` across ``TP = EP in (8, 16)`` and ``B in (32, 64, 128)``:

  1. Zero ``logits [B, 256] f32`` & ``expert_in [B, 4096] bf16`` HBM roundtrip.
  2. Aliases ``wo_buf`` inside MoE ``w2_buf`` / ``w1_buf[0]`` (saves ``8 MiB`` VMEM).
  3. Hoists ``W_router + ag_log`` before ``W_pre_pair`` so router logits land early.
  4. Runs ``plan_and_prime_muse_sparse_from_vmem(..., zero_and_pack=False)``
     overlapped inside ``ag_exp`` ICI wait (hiding ``ROUTE_FAST + SMEM plan``
     and Expert-0's ``24 MiB`` DMA behind ``ag_exp``!) and
     ``zero_and_pack_muse_sparse_from_vmem`` inside ``ag_exp`` D2D2 wait.
  5. Enforces the ``#3706`` / ``#3707`` / ``#3708`` barrier-free non-aliasing contract:
     ``MoE`` RS receive buffers (``rs_pair_recv``, ``rs_chip_recv``, ``rs_recv_3d``)
     and receive semaphores (``rs_pair_rsem``, ``rs_chip_rsem``, ``rs_recv_sem``)
     plus ``exp_3d_buf`` are never shared with any ``chip_b1`` buffer or semaphore.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    _DOT_TN,
    SUPPORTED_WORLD_SIZES,
    _COL_STRIP,
    _as_2d_f32,
    _as_dot_scale_f32,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _dot_fp8_w8_slice_to_ref,
    _fold_cols_to_128,
    _fori_strips,
    _ag_chip_sharded_half_col_xor1,
    _pallas_call_maybe_interpret,
    _pick_strip,
    _signal_barrier,
    _wait_barrier,
    _W8_CID_BASE,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
    DenseRoutingScratch,
    RoutingScratch,
    SparseDecodeConfig,
    build_muse_sparse_scratch_shapes,
    default_sparse_decode_config,
    plan_and_prime_muse_sparse_from_vmem,
    run_muse_sparse_experts_and_rs_from_vmem,
    zero_and_pack_muse_sparse_from_vmem,
)

_E_TOTAL = 256
_H = 4096
_I = 4096
_N1 = 2 * _I
_FP4_PACK = 8


def _unpack_muse_sparse_scratch(
    it,
    *,
    ep: int,
    scale_block: int,
    num_replicas: int,
    return_sharded: bool,
    cfg: SparseDecodeConfig,
) -> dict:
    """Unpack ``build_muse_sparse_scratch_shapes`` refs in exact declaration order."""
    is_k256 = scale_block >= 256
    is_dense = cfg.row_mode == "dense"
    w_mode = (
        "stream3"
        if (cfg.weight_mode == "stream3_k2" and (_H // scale_block) < 16)
        else cfg.weight_mode
    )
    use_bd = (not is_dense) and (not is_k256) and bool(cfg.k16_blockdiag_rows)
    use_lhs_fp8 = is_k256 and cfg.lhs_fp8
    use_bd_fp8 = use_bd and cfg.k16_bd_fp8
    use_pair_rs = cfg.pair_rs and (ep in (8, 16))
    has_replicas = num_replicas > 0

    if is_dense:
        d_scratch = DenseRoutingScratch(
            *(next(it) for _ in range(len(DenseRoutingScratch._fields)))
        )
        r_scratch = None
    else:
        r_scratch = RoutingScratch(
            *(next(it) for _ in range(len(RoutingScratch._fields)))
        )
        d_scratch = None

    if has_replicas:
        owner_load_smem = next(it)
        moved_flag_smem = next(it)
        local_suppress_smem = next(it)
        holder_slot_expert_smem = next(it)
        holder_cand_smem = (
            next(it) if (not is_dense and not cfg.static_holder_mask) else None
        )
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = holder_cand_smem = None

    w1_buf = next(it)
    s1_buf = next(it)
    w2_buf = next(it)
    s2_buf = next(it)
    w1_sem = next(it)
    w2_sem = next(it)
    misc_sem = next(it)
    w1_acc_buf = next(it) if w_mode == "stream3_k2" else None

    if is_dense:
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it) if not return_sharded else None
        x_3d_buf = io_3d_buf = lhs_2d_buf = masked_x_buf = bd_sc_buf = None
    else:
        x_3d_buf = next(it)
        io_3d_buf = next(it)
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        masked_x_buf = next(it) if use_bd else None
        bd_sc_buf = next(it) if use_bd_fp8 else None
        gain_f32_buf = next(it)
        accum_3d = next(it)
        accum_2d = None
        fan_3d_buf = x_3d_buf if not return_sharded else None

    rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
    if use_pair_rs:
        rs_pair_recv = next(it)
        rs_chip_send = next(it)
        rs_chip_recv = next(it)
        rs_pair_ssem = next(it)
        rs_pair_rsem = next(it)
        rs_chip_ssem = next(it)
        rs_chip_rsem = next(it)
        rs_recv_3d = rs_send_sem = rs_recv_sem = None
    else:
        rs_recv_3d = next(it)
        rs_send_sem = next(it)
        rs_recv_sem = next(it)
        rs_pair_recv = rs_chip_send = rs_chip_recv = None
        rs_pair_ssem = rs_pair_rsem = rs_chip_ssem = rs_chip_rsem = None

    owner_3d = next(it)
    if not return_sharded:
        fan_send_sem = next(it)
        fan_recv_sem = next(it)
        fan_s2_send_sem = next(it) if cfg.bridge_ag else None
        fan_s2_recv_sem = next(it) if cfg.bridge_ag else None
    else:
        fan_send_sem = fan_recv_sem = fan_s2_send_sem = fan_s2_recv_sem = None

    return dict(
        w_mode=w_mode,
        d_scratch=d_scratch,
        r_scratch=r_scratch,
        owner_load_smem=owner_load_smem,
        moved_flag_smem=moved_flag_smem,
        local_suppress_smem=local_suppress_smem,
        holder_slot_expert_smem=holder_slot_expert_smem,
        holder_cand_smem=holder_cand_smem,
        w1_buf=w1_buf,
        s1_buf=s1_buf,
        w2_buf=w2_buf,
        s2_buf=s2_buf,
        w1_sem=w1_sem,
        w2_sem=w2_sem,
        misc_sem=misc_sem,
        w1_acc_buf=w1_acc_buf,
        x_3d_buf=x_3d_buf,
        io_3d_buf=io_3d_buf,
        lhs_2d_buf=lhs_2d_buf,
        act_2d_buf=act_2d_buf,
        y_stage_buf=y_stage_buf,
        lhs_fp8_buf=lhs_fp8_buf,
        lhs_sc_buf=lhs_sc_buf,
        masked_x_buf=masked_x_buf,
        bd_sc_buf=bd_sc_buf,
        gain_f32_buf=gain_f32_buf,
        accum_3d=accum_3d,
        accum_2d=accum_2d,
        fan_3d_buf=fan_3d_buf,
        rs_send_bf16=rs_send_bf16,
        rs_pair_recv=rs_pair_recv,
        rs_chip_send=rs_chip_send,
        rs_chip_recv=rs_chip_recv,
        rs_pair_ssem=rs_pair_ssem,
        rs_pair_rsem=rs_pair_rsem,
        rs_chip_ssem=rs_chip_ssem,
        rs_chip_rsem=rs_chip_rsem,
        rs_recv_3d=rs_recv_3d,
        rs_send_sem=rs_send_sem,
        rs_recv_sem=rs_recv_sem,
        owner_3d=owner_3d,
        fan_send_sem=fan_send_sem,
        fan_recv_sem=fan_recv_sem,
        fan_s2_send_sem=fan_s2_send_sem,
        fan_s2_recv_sem=fan_s2_recv_sem,
    )


def _alias_wo_buf(w1_buf, w2_buf, *, w_mode: str, o_in: int, half: int):
    """Alias ``(2, o_in, half) float8_e4m3fn`` inside dead MoE weight VMEM (0 extra MiB)."""
    from jax.experimental import pallas as pl

    assert half == 4096
    if w_mode in ("w1_2x_w2_2x", "half_n"):
        # w2_buf in w1_2x_w2_2x is (2, 512, 4096); in half_n it is (2, 512, 2048),
        # while w1_buf in half_n is (2, 512, 4096).
        base_u32 = w2_buf.at[0] if w_mode == "w1_2x_w2_2x" else w1_buf.at[0]
    else:
        base_u32 = w2_buf
    # base_u32 is (512, 4096) uint32 -> bitcast to (2048, 4096) fp8 (minor dim invariant).
    if o_in == 1024:
        return base_u32.bitcast(jnp.float8_e4m3fn).reshape((2, 1024, 4096))
    if o_in == 512:
        return (
            base_u32.at[pl.ds(0, 256), :]
            .bitcast(jnp.float8_e4m3fn)
            .reshape((2, 512, 4096))
        )
    raise ValueError(f"Unsupported o_in={o_in} for aliased wo_buf")


def _b1_moe_chip_sharded_kernel(
    *all_refs,
    world_size: int,
    scale_block: int,
    num_replicas: int,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    cfg: SparseDecodeConfig,
    has_spec_slot_arg: bool,
    interpret_mode: bool = False,
) -> None:
    """Fused Step (a) ``[chip_b1 + muse_sparse(fb0)]`` Pallas kernel body."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    it = iter(all_refs)
    # Scalar prefetch (matched to PrefetchScalarGridSpec).
    rank_ref = next(it)
    replica_meta_ref = next(it) if num_replicas > 0 else None
    spec_slot_ref = next(it) if has_spec_slot_arg else None

    # Bridge 1b + MoE inputs.
    stream_chip_hbm = next(it)
    gated_ref = next(it)
    wo_hbm = next(it)
    wo_scale_ref = next(it)
    pa_alpha_ref = next(it)
    pa_beta_ref = next(it)
    pre_ff_gain_ref = next(it)
    w_pre_pair_hbm = next(it)
    w_pre_pair_scale_ref = next(it)
    w_router_hbm = next(it)
    pre_exp_gain_ref = next(it)
    bias_ref = next(it)
    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    post_exp_norm_ref = next(it)
    if num_replicas > 0:
        rep_w1_ref = next(it)
        rep_s1_ref = next(it)
        rep_w2_ref = next(it)
        rep_s2_ref = next(it)
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

    # Outputs.
    out_stream_chip_hbm = next(it)
    out_moe_ref = next(it)

    # MoE scratch (first so build_muse_sparse_scratch_shapes layout stays untouched).
    msc = _unpack_muse_sparse_scratch(
        it,
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        cfg=cfg,
    )

    # Bridge 1b scratch.
    w_pre_pair_buf = next(it)
    w_router_buf = next(it)
    s_buf = next(it)
    send_half = next(it)
    mate_recv = next(it)
    own_half = next(it)
    rs1_a_or_z_recv = next(it)
    rs1_b_or_y_recv = next(it)
    rs2_a_or_x_recv = next(it)
    rs2_b_recv = next(it)
    my_pair_bf16 = next(it)
    recv_pair_bf16 = next(it)
    sq_pair_pa_buf = next(it)
    ffn_pair_buf = next(it)
    router_pair_buf = next(it)
    pre_my_half = next(it)
    pre_mate_half = next(it)
    sq_pair_pre_buf = next(it)
    exp_3d_buf = next(it)
    logits_vmem = next(it)
    expert_in_vmem = next(it)

    # Bridge 1b semaphores. Note ag_exp_send / ag_log_send reuse rs_pair_ssem /
    # rs_chip_ssem when pair_rs=True (send-only local completion before MoE starts);
    # rs_pair_rsem / rs_chip_rsem / rs_recv_sem stay 100% separate (#3706/#3707).
    w_pre_sem = next(it)
    w_router_sem = next(it)
    s_in_sem = next(it)
    s_out_sem = next(it)
    c1a_send = next(it)
    c1a_recv = next(it)
    rs1_send = next(it)
    rs1_recv = next(it)

    rows = gated_ref.shape[0]
    o_in = gated_ref.shape[1]
    r_pair = stream_chip_hbm.shape[0]
    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    q_cols = half // 2
    pre_pair_cols = w_pre_pair_hbm.shape[1]
    h_pre = 2 * pre_pair_cols
    n_exp = w_router_hbm.shape[1]
    n_chips = world_size // 2
    use_xy = world_size == 8 and r_pair >= 16

    if use_xy:
        rs2_send = next(it)
        rs2_recv = next(it)
        rs3_send = rs3_recv = None
    else:
        rs2_send, rs2_recv = rs1_send, rs1_recv
        rs3_send, rs3_recv = rs1_send, rs1_recv

    c1c_send = next(it)
    c1c_recv = next(it)
    pre_send = next(it)
    pre_recv = next(it)
    ag_exp_d2d_send = next(it)
    ag_exp_d2d_recv = next(it)
    ag_exp_recv = next(it)
    ag_log_recv = next(it)
    if msc["rs_pair_ssem"] is not None:
        ag_exp_send = msc["rs_pair_ssem"]
        ag_log_send = msc["rs_chip_ssem"]
    else:
        ag_exp_send = next(it)
        ag_log_send = next(it)

    r_align = 16 if (r_pair % 16 == 0) else 8
    bm_full = _pick_strip(rows, 32)
    bm_pair = _pick_strip(r_pair, 32)
    c_blk_half = 512 if (half % 512 == 0) else _COL_STRIP
    c_blk_full = 512 if (hidden % 512 == 0) else _COL_STRIP
    c_blk_pre = 512 if (pre_pair_cols % 512 == 0) else _COL_STRIP
    q_step = 512 if (q_cols % 512 == 0) else _COL_STRIP
    is_r16 = w_router_buf.dtype == jnp.bfloat16

    me = rank_ref[0]
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

    if interpret_mode:
        wo_buf = next(it)
    else:
        wo_buf = _alias_wo_buf(
            msc["w1_buf"],
            msc["w2_buf"],
            w_mode=msc["w_mode"],
            o_in=o_in,
            half=half,
        )
    wo_sem = msc["w1_sem"]

    # =========================================================================
    # Stage 1: Entry Barrier + W_o + Intra-chip Reduce
    # =========================================================================
    _signal_barrier(me, world_size)
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(other_c0, half)], wo_buf.at[0], wo_sem.at[0]
    ).start()
    pltpu.make_async_copy(
        wo_hbm.at[:, pl.ds(own_c0, half)], wo_buf.at[1], wo_sem.at[1]
    ).start()
    pltpu.make_async_copy(stream_chip_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(w_pre_pair_hbm, w_pre_pair_buf, w_pre_sem.at[0]).start()
    pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    _dma_wait(wo_buf.at[0], wo_sem.at[0])
    _dot_fp8_scaled_to_ref(
        gated_ref, wo_buf.at[0], wo_scale_ref, other_c0, send_half, 0,
        rows=rows, n_cols=half,
    )

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
            for c0 in range(0, half, c_blk_half):
                c_sl = pl.ds(c0, c_blk_half)
                v = own_half[rs, c_sl].astype(jnp.float32) + mate_recv[rs, c_sl].astype(jnp.float32)
                own_half[rs, c_sl] = v.astype(own_half.dtype)

        _fori_strips(rows, bm_full, _sum_c1a)

    # =========================================================================
    # Stage 2: Chip-sharded Token Reduce-Scatter + PostAttn + Stream Update + PreFFN
    # =========================================================================
    if use_xy:
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
                    for c0 in range(0, half, c_blk_half):
                        cs = pl.ds(c0, c_blk_half)
                        v = own_half[rk, cs].astype(jnp.float32) + rs1_a_or_z_recv[rl, cs].astype(jnp.float32)
                        own_half[rl, cs] = v.astype(own_half.dtype)

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

    if r_pair == 16:
        c_blk = min(half, 1024)
        own_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        own_chunks = []
        for c0 in range(0, half, c_blk):
            bf = my_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            own_chunks.append(bf)
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        _dma_wait(s_buf, s_in_sem.at[0])
        oth_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        oth_chunks = []
        for c0 in range(0, half, c_blk):
            bf = recv_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            oth_chunks.append(bf)
            oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
        pa_sq = own_sq + oth_sq
        inv_pa = lax.rsqrt(
            jnp.sum(pa_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pa_eps)
        )
        pff_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        ns_own_list = []
        ns_oth_list = []
        for idx_c, c0 in enumerate(range(0, half, c_blk)):
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            ns_own = (
                pa_alpha_ref[0:1, c_own] * s_buf[:, c_own]
                + pa_beta_ref[0:1, c_own] * (own_chunks[idx_c] * inv_pa)
            )
            ns_oth = (
                pa_alpha_ref[0:1, c_oth] * s_buf[:, c_oth]
                + pa_beta_ref[0:1, c_oth] * (oth_chunks[idx_c] * inv_pa)
            )
            s_buf[:, c_own] = ns_own
            s_buf[:, c_oth] = ns_oth
            ns_own_list.append(ns_own)
            ns_oth_list.append(ns_oth)
            v_own = ns_own * pre_ff_gain_ref[0:1, c_own] if pre_ff_scale_input else ns_own
            v_oth = ns_oth * pre_ff_gain_ref[0:1, c_oth] if pre_ff_scale_input else ns_oth
            pff_sq = pff_sq + _fold_cols_to_128(v_own * v_own + v_oth * v_oth)
        pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()
        inv_pff = lax.rsqrt(
            jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
            + jnp.float32(pre_ff_eps)
        )
        for idx_c, c0 in enumerate(range(0, half, c_blk)):
            c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk)
            c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk)
            go = pre_ff_gain_ref[0:1, c_own]
            gt = pre_ff_gain_ref[0:1, c_oth]
            ro = (
                (ns_own_list[idx_c] * go) * inv_pff
                if pre_ff_scale_input
                else (ns_own_list[idx_c] * inv_pff) * go
            )
            rt = (
                (ns_oth_list[idx_c] * gt) * inv_pff
                if pre_ff_scale_input
                else (ns_oth_list[idx_c] * inv_pff) * gt
            )
            if not is_r16:
                router_pair_buf[:, c_own] = ro
                router_pair_buf[:, c_oth] = rt
            ffn_pair_buf[:, c_own] = ro.astype(jnp.bfloat16)
            ffn_pair_buf[:, c_oth] = rt.astype(jnp.bfloat16)
    elif r_pair <= 32:
        c_blk = min(half, 1024 if r_pair <= 16 else 512)
        h_blk = min(hidden, 1024 if r_pair <= 16 else 512)
        own_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
        for c0 in range(0, half, c_blk):
            bf = my_pair_bf16[:, pl.ds(c0, c_blk)].astype(jnp.float32)
            own_sq = own_sq + _fold_cols_to_128(bf * bf)
        op_c1c.wait()
        _dma_wait(s_buf, s_in_sem.at[0])
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
            acc = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                bf = my_pair_bf16[rs, pl.ds(c0, c_blk_half)].astype(jnp.float32)
                acc = acc + _fold_cols_to_128(bf * bf)
            sq_pair_pa_buf[rs, :] = acc

        _fori_strips(r_pair, bm_pair, _own_pa_sq)
        op_c1c.wait()
        _dma_wait(s_buf, s_in_sem.at[0])

        def _finish_pa_and_pre_ffn_pair(r0):
            rs = pl.ds(r0, bm_pair)
            oth_sq = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                bf = recv_pair_bf16[rs, pl.ds(c0, c_blk_half)].astype(jnp.float32)
                oth_sq = oth_sq + _fold_cols_to_128(bf * bf)
            inv_pa = lax.rsqrt(
                jnp.sum(sq_pair_pa_buf[rs, :] + oth_sq, axis=-1, keepdims=True)
                * jnp.float32(1.0 / hidden)
                + jnp.float32(pa_eps)
            )
            pff_sq = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
            for c0 in range(0, half, c_blk_half):
                c_l = pl.ds(c0, c_blk_half)
                c_own = pl.ds(pl.multiple_of(own_c0 + c0, 128), c_blk_half)
                c_oth = pl.ds(pl.multiple_of(other_c0 + c0, 128), c_blk_half)
                ns_own = (
                    pa_alpha_ref[0:1, c_own] * s_buf[rs, c_own]
                    + pa_beta_ref[0:1, c_own]
                    * (my_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                )
                ns_oth = (
                    pa_alpha_ref[0:1, c_oth] * s_buf[rs, c_oth]
                    + pa_beta_ref[0:1, c_oth]
                    * (recv_pair_bf16[rs, c_l].astype(jnp.float32) * inv_pa)
                )
                s_buf[rs, c_own] = ns_own
                s_buf[rs, c_oth] = ns_oth
                v_own = (
                    ns_own * pre_ff_gain_ref[0:1, c_own]
                    if pre_ff_scale_input
                    else ns_own
                )
                v_oth = (
                    ns_oth * pre_ff_gain_ref[0:1, c_oth]
                    if pre_ff_scale_input
                    else ns_oth
                )
                pff_sq = pff_sq + _fold_cols_to_128(v_own * v_own + v_oth * v_oth)
            inv_pff = lax.rsqrt(
                jnp.sum(pff_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
                + jnp.float32(pre_ff_eps)
            )
            for c0 in range(0, hidden, c_blk_full):
                c_g = pl.ds(c0, c_blk_full)
                sv = s_buf[rs, c_g]
                gn = pre_ff_gain_ref[0:1, c_g]
                rn = (sv * gn) * inv_pff if pre_ff_scale_input else (sv * inv_pff) * gn
                if not is_r16:
                    router_pair_buf[rs, c_g] = rn
                ffn_pair_buf[rs, c_g] = rn.astype(jnp.bfloat16)

        _fori_strips(r_pair, bm_pair, _finish_pa_and_pre_ffn_pair)
        pltpu.make_async_copy(s_buf, out_stream_chip_hbm, s_out_sem.at[0]).start()

    # =========================================================================
    # Stage 3: Hoisted W_router + ag_log BEFORE W_pre_pair, then pre_exp_norm + ag_exp
    #          overlapped with plan_and_prime + zero_and_pack (#3640 §3)
    # =========================================================================
    rmul = jnp.float32(router_multiplier)
    r_lhs = ffn_pair_buf if is_r16 else router_pair_buf
    _dma_wait(w_router_buf, w_router_sem.at[0])
    if r_pair <= 64 and n_exp == 256:
        lg = jnp.dot(r_lhs[...], w_router_buf[...], preferred_element_type=jnp.float32)
        logits_vmem[pl.ds(chip_r0, r_pair), :] = lg * rmul
    else:
        for ce in range(0, n_exp, _DOT_TN):
            c_sl = pl.ds(ce, _DOT_TN)
            lg = jnp.dot(r_lhs[...], w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
            logits_vmem[pl.ds(chip_r0, r_pair), c_sl] = lg * rmul

    ag_log_ops = []
    for d in range(1, n_chips):
        peer = me ^ jnp.int32(d << 1)
        op_l = pltpu.make_async_remote_copy(
            logits_vmem.at[pl.ds(chip_r0, r_pair), :],
            logits_vmem.at[pl.ds(chip_r0, r_pair), :],
            ag_log_send.at[d - 1],
            ag_log_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        op_l.start()
        ag_log_ops.append(op_l)

    _dma_wait(w_pre_pair_buf, w_pre_sem.at[0])
    _dot_fp8_w8_slice_to_ref(
        ffn_pair_buf, 0, w_pre_pair_buf, w_pre_pair_scale_ref, 0, pre_my_half, 0, 0,
        n_r=r_pair, n_cols=pre_pair_cols, align=r_align,
    )

    w1_u32 = w1_ref if w1_ref.dtype == jnp.uint32 else w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref if w2_ref.dtype == jnp.uint32 else w2_ref.bitcast(jnp.uint32)
    rep_w1_u32 = (
        (rep_w1_ref if rep_w1_ref.dtype == jnp.uint32 else rep_w1_ref.bitcast(jnp.uint32))
        if num_replicas > 0
        else None
    )
    rep_w2_u32 = (
        (rep_w2_ref if rep_w2_ref.dtype == jnp.uint32 else rep_w2_ref.bitcast(jnp.uint32))
        if num_replicas > 0
        else None
    )

    plan_holder = []

    def _run_plan_prime(*, zero_and_pack: bool) -> None:
        # Ensure all router logits have landed from every chip before routing starts.
        for op in ag_log_ops:
            op.wait()
        plan_res = plan_and_prime_muse_sparse_from_vmem(
            me,
            logits_vmem,
            bias_ref,
            post_exp_norm_ref,
            w1_u32,
            s1_ref,
            w2_u32,
            s2_ref,
            d_scratch=msc["d_scratch"],
            r_scratch=msc["r_scratch"],
            w1_buf=msc["w1_buf"],
            s1_buf=msc["s1_buf"],
            w2_buf=msc["w2_buf"],
            s2_buf=msc["s2_buf"],
            w1_sem=msc["w1_sem"],
            w2_sem=msc["w2_sem"],
            misc_sem=msc["misc_sem"],
            gain_f32_buf=msc["gain_f32_buf"],
            accum_3d=msc["accum_3d"],
            io_3d_buf=msc["io_3d_buf"],
            accum_2d=msc["accum_2d"],
            rep_w1_u32=rep_w1_u32,
            rep_s1_ref=rep_s1_ref,
            rep_w2_u32=rep_w2_u32,
            rep_s2_ref=rep_s2_ref,
            replica_meta_ref=replica_meta_ref,
            owner_load_smem=msc["owner_load_smem"],
            moved_flag_smem=msc["moved_flag_smem"],
            local_suppress_smem=msc["local_suppress_smem"],
            holder_slot_expert_smem=msc["holder_slot_expert_smem"],
            holder_cand_smem=msc["holder_cand_smem"],
            spec_slot_ref=spec_slot_ref,
            ep=world_size,
            batch_size=rows,
            scale_block=scale_block,
            num_replicas=num_replicas,
            replica_cost_mode=replica_cost_mode,
            cfg=cfg,
            zero_and_pack=zero_and_pack,
        )
        plan_holder.append(plan_res)

    if r_pair == 8:
        op_pre = pltpu.make_async_remote_copy(
            pre_my_half, pre_mate_half, pre_send.at[0], pre_recv.at[0],
            device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
        )
        op_pre.start()

        p_blk = min(pre_pair_cols, 1024)
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
            exp_3d_buf[chip, :, c_own] = y_own.astype(expert_in_vmem.dtype)
            exp_3d_buf[chip, :, c_oth] = y_oth.astype(expert_in_vmem.dtype)

        ag_exp_ops = []
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

        # Overlap ag_log.wait + route + plan + Expert-0 prime + zero_and_pack under ag_exp ICI!
        # Safe per #3708 §3: exp_3d_buf does not alias accum_3d or io_3d_buf.
        _run_plan_prime(zero_and_pack=True)
        for op in ag_exp_ops:
            op.wait()
        e_w = min(h_pre, 1024)
        for p in range(n_chips // 2):
            r16 = pl.ds(16 * p, 16)
            for c0 in range(0, h_pre, e_w):
                c_sl = pl.ds(c0, e_w)
                expert_in_vmem[r16, c_sl] = jnp.concatenate(
                    [exp_3d_buf[2 * p, :, c_sl], exp_3d_buf[2 * p + 1, :, c_sl]], axis=0
                )
    else:
        rs_g = pl.ds(chip_r0, r_pair)
        if r_pair <= 32:
            p_blk = min(pre_pair_cols, 1024 if r_pair <= 16 else 512)
            own_pre_sq = jnp.zeros((r_pair, 128), dtype=jnp.float32)
            pre_f_chunks = []
            for c0 in range(0, pre_pair_cols, p_blk):
                c_l = pl.ds(c0, p_blk)
                c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
                xf = pre_my_half[:, c_l].astype(jnp.float32)
                if r_pair == 16:
                    pre_f_chunks.append(xf)
                xf_s = xf * pre_exp_gain_ref[0:1, c_own] if pre_exp_scale_input else xf
                own_pre_sq = own_pre_sq + _fold_cols_to_128(xf_s * xf_s)
            sq_pair_pre_buf[...] = own_pre_sq
            op_sq = pltpu.make_async_remote_copy(
                sq_pair_pre_buf, pre_mate_half, pre_send.at[0], pre_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_sq.start()
            op_sq.wait()
            pre_sq = own_pre_sq + pre_mate_half[...]
            inv = lax.rsqrt(
                jnp.sum(pre_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
                + jnp.float32(pre_exp_eps)
            )
            for idx_p, c0 in enumerate(range(0, pre_pair_cols, p_blk)):
                c_l = pl.ds(c0, p_blk)
                c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), p_blk)
                x_own = pre_f_chunks[idx_p] if r_pair == 16 else pre_my_half[:, c_l].astype(jnp.float32)
                g_own = pre_exp_gain_ref[0:1, c_own]
                y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
                expert_in_vmem[rs_g, c_own] = y_own.astype(expert_in_vmem.dtype)
        else:

            def _own_pre_sq(r0):
                rs = pl.ds(r0, bm_pair)
                acc = jnp.zeros((bm_pair, 128), dtype=jnp.float32)
                for c0 in range(0, pre_pair_cols, c_blk_pre):
                    c_l = pl.ds(c0, c_blk_pre)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), c_blk_pre)
                    xf = pre_my_half[rs, c_l].astype(jnp.float32)
                    if pre_exp_scale_input:
                        xf = xf * pre_exp_gain_ref[0:1, c_own]
                    acc = acc + _fold_cols_to_128(xf * xf)
                sq_pair_pre_buf[rs, :] = acc

            _fori_strips(r_pair, bm_pair, _own_pre_sq)
            op_sq = pltpu.make_async_remote_copy(
                sq_pair_pre_buf, pre_mate_half, pre_send.at[0], pre_recv.at[0],
                device_id=(mate,), device_id_type=pl.DeviceIdType.MESH,
            )
            op_sq.start()
            op_sq.wait()

            def _own_pre_exp_norm(r0):
                rs_l = pl.ds(r0, bm_pair)
                rs_go = pl.ds(pl.multiple_of(chip_r0 + r0, 16), bm_pair)
                inv = lax.rsqrt(
                    jnp.sum(sq_pair_pre_buf[rs_l, :] + pre_mate_half[rs_l, :], axis=-1, keepdims=True)
                    * jnp.float32(1.0 / h_pre)
                    + jnp.float32(pre_exp_eps)
                )
                for c0 in range(0, pre_pair_cols, c_blk_pre):
                    c_l = pl.ds(c0, c_blk_pre)
                    c_own = pl.ds(pl.multiple_of(own_pre_c0 + c0, 128), c_blk_pre)
                    x_own = pre_my_half[rs_l, c_l].astype(jnp.float32)
                    g_own = pre_exp_gain_ref[0:1, c_own]
                    y_own = (x_own * g_own) * inv if pre_exp_scale_input else (x_own * inv) * g_own
                    expert_in_vmem[rs_go, c_own] = y_own.astype(expert_in_vmem.dtype)

            _fori_strips(r_pair, bm_pair, _own_pre_exp_norm)

        def _during_ag_exp_ici() -> None:
            _run_plan_prime(zero_and_pack=False)

        def _during_ag_exp_d2d2() -> None:
            zero_and_pack_muse_sparse_from_vmem(
                plan_holder[0],
                r_scratch=msc["r_scratch"],
                accum_3d=msc["accum_3d"],
                io_3d_buf=msc["io_3d_buf"],
                accum_2d=msc["accum_2d"],
                batch_size=rows,
                cfg=cfg,
            )

        _ag_chip_sharded_half_col_xor1(
            None,
            expert_in_vmem,
            ag_exp_send,
            ag_exp_recv,
            ag_exp_d2d_send,
            ag_exp_d2d_recv,
            me=me,
            world_size=world_size,
            during_ici_fn=_during_ag_exp_ici,
            during_d2d2_fn=_during_ag_exp_d2d2,
            r_pair_in=r_pair,
            copy_local_from_pair=False,
        )

    # =========================================================================
    # Stage 4: Sparse NVFP4 MoE Experts + PAIR_RS (0 RS-recv alias, barrier-free!)
    # =========================================================================
    run_muse_sparse_experts_and_rs_from_vmem(
        me,
        expert_in_vmem,
        plan_holder[0],
        d_scratch=msc["d_scratch"],
        r_scratch=msc["r_scratch"],
        w1_buf=msc["w1_buf"],
        s1_buf=msc["s1_buf"],
        w2_buf=msc["w2_buf"],
        s2_buf=msc["s2_buf"],
        w1_acc_buf=msc["w1_acc_buf"],
        x_3d_buf=msc["x_3d_buf"],
        io_3d_buf=msc["io_3d_buf"],
        lhs_2d_buf=msc["lhs_2d_buf"],
        act_2d_buf=msc["act_2d_buf"],
        y_stage_buf=msc["y_stage_buf"],
        lhs_fp8_buf=msc["lhs_fp8_buf"],
        lhs_sc_buf=msc["lhs_sc_buf"],
        masked_x_buf=msc["masked_x_buf"],
        bd_sc_buf=msc["bd_sc_buf"],
        gain_f32_buf=msc["gain_f32_buf"],
        accum_3d=msc["accum_3d"],
        accum_2d=msc["accum_2d"],
        fan_3d_buf=msc["fan_3d_buf"],
        rs_send_bf16=msc["rs_send_bf16"],
        rs_pair_recv=msc["rs_pair_recv"],
        rs_chip_send=msc["rs_chip_send"],
        rs_chip_recv=msc["rs_chip_recv"],
        rs_pair_ssem=msc["rs_pair_ssem"],
        rs_pair_rsem=msc["rs_pair_rsem"],
        rs_chip_ssem=msc["rs_chip_ssem"],
        rs_chip_rsem=msc["rs_chip_rsem"],
        rs_recv_3d=msc["rs_recv_3d"],
        rs_send_sem=msc["rs_send_sem"],
        rs_recv_sem=msc["rs_recv_sem"],
        owner_3d=msc["owner_3d"],
        fan_send_sem=msc["fan_send_sem"],
        fan_recv_sem=msc["fan_recv_sem"],
        fan_s2_send_sem=msc["fan_s2_send_sem"],
        fan_s2_recv_sem=msc["fan_s2_recv_sem"],
        barrier_sem=None,
        output_ref=out_moe_ref,
        ep=world_size,
        batch_size=rows,
        scale_block=scale_block,
        return_sharded=return_sharded,
        ep_axis_name=ep_axis_name,
        cfg=cfg,
        x_is_3d=False,
        wait_entry_barrier=False,
        write_output_ref=True,
        on_experts_done_fn=None,
    )
    _dma_wait(out_stream_chip_hbm, s_out_sem.at[0])


def b1_moe_chip_sharded_jax(
    rank: jax.Array,
    stream_chip: jax.Array,
    gated_attn: jax.Array,
    wo_w8k: jax.Array,
    wo_scale: jax.Array,
    pa_alpha: jax.Array,
    pa_beta: jax.Array,
    pre_ff_gain: jax.Array,
    w_pre_pair_w8k: jax.Array,
    w_pre_pair_scale: jax.Array,
    w_router_km: jax.Array,
    pre_exp_gain: jax.Array,
    bias: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_exp_norm: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    spec_home_slot: jax.Array | None = None,
    *,
    world_size: int = 8,
    pa_eps: float = 1e-8,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    return_sharded: bool = True,
    replica_cost_mode: str = "active",
    ep_axis_name: str = "model",
    config: SparseDecodeConfig | None = None,
    collective_id: int = _W8_CID_BASE + 18,
    interpret: bool = False,
):
    """Fused Step (a) superkernel: ``(stream_chip, gated_attn, weights) -> (new_stream_chip, moe_out)``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_chip_sharded_jax requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    rows, _o_in = gated_attn.shape
    r_pair, hidden = stream_chip.shape
    n_chips = world_size // 2
    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    _, pre_pair_cols = w_pre_pair_w8k.shape
    h_pre = 2 * pre_pair_cols
    _, n_exp = w_router_km.shape
    valid_rows = (32, 64, 128) if world_size == 8 else (64, 128)
    if (
        rows not in valid_rows
        or rows != n_chips * r_pair
        or r_pair < 8
        or r_pair % 8 != 0
        or hidden != 8192
        or h_pre != _H
        or n_exp != _E_TOTAL
    ):
        raise ValueError(
            f"Unsupported b1_moe_chip_sharded shapes world_size={world_size}, "
            f"stream={stream_chip.shape}, gated={gated_attn.shape}, w_pre={w_pre_pair_w8k.shape}"
        )

    rows_per_rank = rows // world_size
    if not return_sharded and world_size == 16:
        # return_sharded=False on EP=16 adds 17-19 fan-out DMA sems (74 > 64 HW limit),
        # and rows_per_rank=4 on B=64 cannot sublane-shard into chip_b2 (#3740/#3770).
        raise ValueError(
            "b1_moe_chip_sharded_jax on world_size=16 requires return_sharded=True"
        )

    scale_block = _H // s1.shape[1]
    if scale_block not in (16, 256, 512):
        raise ValueError(f"Expected K16/K256/K512 scales, got s1.shape={s1.shape}")

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError(
                "All of rep_w1, rep_s1, rep_w2, rep_s2, replica_meta must be provided together"
            )
        num_replicas = int(rep_w1.shape[0])
    else:
        num_replicas = 0

    if config is None:
        import dataclasses

        base_cfg = default_sparse_decode_config(rows, world_size, scale_block)
        safe_wm = "stream3" if rows <= 64 else "half_n"
        config = dataclasses.replace(
            base_cfg,
            weight_mode=safe_wm,
            pair_rs=True,
            bridge_ag=(world_size == 16) or base_cfg.bridge_ag,
        )

    half = hidden // 2
    q_cols = half // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    use_xy = world_size == 8 and r_pair >= 16
    use_pair_rs = config.pair_rs and (world_size in (8, 16))

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)
    if isinstance(interpret, bool):
        interpret_arg = (
            pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
            if interpret
            else False
        )
    else:
        interpret_arg = interpret
    interpret_mode = interpret_arg is not False
    o_in = _o_in

    moe_scratch = build_muse_sparse_scratch_shapes(
        ep=world_size,
        batch_size=rows,
        scale_block=scale_block,
        s1_dtype=s1.dtype,
        s2_dtype=s2.dtype,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        config=config,
    )
    b1_scratch = [
        vmem((hidden, pre_pair_cols), w_pre_pair_w8k.dtype),
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem((r_pair, hidden), jnp.float32),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem(
            (two_pair, q_cols)
            if use_xy
            else ((four_pair, half) if world_size == 16 else (8, 128)),
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
        (
            vmem((r_pair, pre_pair_cols), jnp.bfloat16)
            if r_pair == 8
            else vmem((r_pair, 128), jnp.float32)
        ),
        vmem((r_pair, 128), jnp.float32),
        vmem((n_chips, 8, h_pre) if r_pair == 8 else (8, 128), jnp.bfloat16),
        vmem((rows, n_exp), jnp.float32),
        vmem((rows, h_pre), jnp.bfloat16),
        # Bridge 1b semaphores (rs2/rs3 share rs1 on not-use_xy).
        sem((1,)),  # w_pre_sem
        sem((1,)),  # w_router_sem
        sem((1,)),  # s_in_sem
        sem((1,)),  # s_out_sem
        sem((1,)),  # c1a_send
        sem((1,)),  # c1a_recv
        sem((3 if use_xy else 1,)),  # rs1_send
        sem((3 if use_xy else 1,)),  # rs1_recv
        *(([sem((2,)), sem((2,))]) if use_xy else []),
        sem((1,)),  # c1c_send
        sem((1,)),  # c1c_recv
        sem((1,)),  # pre_send
        sem((1,)),  # pre_recv
        sem((1,)),  # ag_exp_d2d_send
        sem((1,)),  # ag_exp_d2d_recv
        sem((n_chips - 1,)),  # ag_exp_recv
        sem((n_chips - 1,)),  # ag_log_recv
        *(() if use_pair_rs else [sem((n_chips - 1,)), sem((n_chips - 1,))]),
    ]
    if interpret_mode:
        # CPU Pallas Interpret's to_range() cannot DMA into BitcastTransform/ReshapeTransform
        # refs (#3722); keep non-aliased wo_buf only under CPU interpret.
        b1_scratch.append(vmem((2, o_in, half), wo_w8k.dtype))

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = post_exp_norm.reshape(1, _H)
    has_spec_slot_arg = spec_home_slot is not None
    scalar_args = [jnp.atleast_1d(rank).astype(jnp.int32)]
    if has_replicas:
        scalar_args.append(replica_meta)
    if has_spec_slot_arg:
        scalar_args.append(jnp.atleast_1d(spec_home_slot).astype(jnp.int32))

    base_in_specs = (
        h_sp,  # stream_chip
        v_sp,  # gated_attn
        h_sp,  # wo_w8k
        v_sp,  # wo_scale
        v_sp,  # pa_alpha
        v_sp,  # pa_beta
        v_sp,  # pre_ff_gain
        h_sp,  # w_pre_pair_w8k
        v_sp,  # w_pre_pair_scale
        h_sp,  # w_router_km
        v_sp,  # pre_exp_gain
        v_sp,  # bias
        h_sp,  # w1
        h_sp,  # s1
        h_sp,  # w2
        h_sp,  # s2
        v_sp,  # post_exp_norm
    )
    base_call_args = (
        stream_chip.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_dot_scale_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_pair_w8k,
        _as_dot_scale_f32(w_pre_pair_scale, pre_pair_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, h_pre),
        bias_2d,
        w1,
        s1,
        w2,
        s2,
        norm_2d,
    )
    if has_replicas:
        in_specs = base_in_specs + (h_sp, h_sp, h_sp, h_sp)
        call_args = (
            *scalar_args,
            *base_call_args,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
        )
    else:
        in_specs = base_in_specs
        call_args = (*scalar_args, *base_call_args)

    out_rows = rows_per_rank if return_sharded else rows
    out_shape = (
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((out_rows, _H), jnp.bfloat16),
    )

    kernel_fn = functools.partial(
        _b1_moe_chip_sharded_kernel,
        world_size=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=bool(return_sharded),
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        pa_eps=float(pa_eps),
        pre_ff_eps=float(pre_ff_eps),
        pre_ff_scale_input=bool(pre_ff_scale_input),
        router_multiplier=float(router_multiplier),
        pre_exp_eps=float(pre_exp_eps),
        pre_exp_scale_input=bool(pre_exp_scale_input),
        cfg=config,
        has_spec_slot_arg=has_spec_slot_arg,
        interpret_mode=interpret_mode,
    )

    # Note: fused_decode_sparse handles rank via PrefetchScalarGridSpec, so call
    # pallas_call directly (using InterpretParams when interpret=True).
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            grid=(1,),
            in_specs=in_specs,
            out_specs=(h_sp, v_sp),
            scratch_shapes=moe_scratch + b1_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(62 if rows > 64 else 60) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=interpret_arg,
        name=(
            f"muse_w9_b1_moe_b{rows}_tp{world_size}_k{scale_block}"
            f"_r{num_replicas}_{'r16' if is_r16 else 'r32'}"
        ),
    )(*call_args)
