# SPDX-License-Identifier: Apache-2.0
"""Wave-9 Step (b) Decode Superkernel: Chip-Sharded Bridge 1b + Sparse NVFP4 MoE + Bridge 2b.

Fuses an entire decode transformer layer transition ``attn_{L} -> MoE_{L} -> attn_{L+1}``
in a single ``pallas_call`` across ``TP = EP in (8, 16)`` and ``B in (32, 64, 128)``:

  1. ``_run_chip_b1_vmem_core`` (``skip_stream_store=True``, ``early_router_and_log_ag=True``):
     - Aliases ``wo_buf`` inside MoE ``w2_buf`` / ``w1_buf[0]``.
     - Keeps ``PostAttn``-updated ``s_buf`` ``[r_pair, 8192] f32`` live in VMEM across
       the whole MoE (eliminating both ``b1``'s ``512 KiB`` HBM stream store and ``b2``'s
       ``512 KiB`` HBM stream load).
     - Overlaps ``plan_and_prime_muse_sparse_from_vmem(..., zero_and_pack=False)``
       and ``zero_and_pack_muse_sparse_from_vmem`` inside ``ag_exp``.
  2. ``run_muse_sparse_experts_and_rs_from_vmem`` (``wait_entry_barrier=False``):
     - Uses ``on_experts_done_fn()`` to immediately launch ``34 MiB`` of ``chip_b2``
       weight DMAs (``w_post_tp2 -> w1_buf`` (``16 MiB``), ``w_qkv -> w_pre_pair_buf[:, :qkv_cols]``,
       ``w_gate -> w_pre_pair_buf[:, qkv_cols:]``) overlapped under ``PAIR_RS``.
  3. ``_run_chip_b2_vmem_core`` (``skip_barrier_signal=True``, ``skip_barrier_wait=True``,
     ``skip_stream_load=True``, ``prestarted_w_*_dma=True``):
     - Reads live ``s_buf`` and ``moe_out`` directly from VMEM (``sharded_moe_in=True``
       when ``B // EP >= 8``, ``False`` on ``TP=16 B=64`` / ``TP=8 B=32``).
     - Emits ``(new_stream_chip, q_5d, kv_merged, gate)`` (or ``(new_stream_chip, qkv, gate)``).
  4. Enforces the ``#3706`` / ``#3707`` / ``#3708`` barrier-free non-aliasing contract:
     ``MoE`` ``PAIR_RS`` receive buffers (``rs_pair_recv``, ``rs_chip_recv``, ``rs_recv_3d``)
     and receive semaphores (``rs_pair_rsem``, ``rs_chip_rsem``, ``rs_recv_sem``) are never
     shared with ``chip_b1`` or ``chip_b2``.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    SUPPORTED_WORLD_SIZES,
    _W8_CID_BASE,
    _as_2d_f32,
    _as_dot_scale_f32,
    _run_chip_b1_vmem_core,
    _run_chip_b2_vmem_core,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
    SparseDecodeConfig,
    build_muse_sparse_scratch_shapes,
    default_sparse_decode_config,
    plan_and_prime_muse_sparse_from_vmem,
    run_muse_sparse_experts_and_rs_from_vmem,
    zero_and_pack_muse_sparse_from_vmem,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
    _E_TOTAL,
    _H,
    _alias_wo_buf,
    _unpack_muse_sparse_scratch,
)


def _alias_b2_weights(
    w1_buf,
    w_pre_pair_buf,
    *,
    w_mode: str,
    qkv_cols: int,
    gate_cols: int,
):
    """Alias ``(w_post_tp2, w_qkv, w_gate)`` inside dead MoE + b1 weight buffers."""
    from jax.experimental import pallas as pl

    if w_mode == "stream3_k2":
        raise ValueError("b1_moe_b2 aliasing requires 4096-col w1_buf (not stream3_k2)")
    if w1_buf.shape != (2, 512, 4096):
        raise ValueError(f"Expected w1_buf shape (2, 512, 4096), got {w1_buf.shape}")
    w_post_buf = w1_buf.bitcast(jnp.float8_e4m3fn).reshape((4096, 4096))
    w_qkv_buf = w_pre_pair_buf.at[:, pl.ds(0, qkv_cols)]
    w_gate_buf = w_pre_pair_buf.at[:, pl.ds(qkv_cols, gate_cols)]
    return w_post_buf, w_qkv_buf, w_gate_buf


def _b1_moe_b2_chip_sharded_kernel(
    *all_refs,
    world_size: int,
    scale_block: int,
    num_replicas: int,
    sharded_moe_in: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    fuse_qkv_epi: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    cfg: SparseDecodeConfig,
    has_spec_slot_arg: bool,
    interpret_mode: bool,
    alias_mate_recv: bool,
) -> None:
    """Fused Step (b) ``[chip_b1 + muse_sparse(fb0) + chip_b2]`` Pallas kernel body."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    it = iter(all_refs)
    # Scalar prefetch.
    rank_ref = next(it)
    replica_meta_ref = next(it) if num_replicas > 0 else None
    spec_slot_ref = next(it) if has_spec_slot_arg else None

    # Bridge 1b inputs.
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

    # MoE inputs.
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

    # Bridge 2b inputs.
    w_post_hbm = next(it)
    s_post_ref = next(it)
    post_ff_gain_ref = next(it)
    pf_alpha_ref = next(it)
    pf_beta_ref = next(it)
    next_in_gain_ref = next(it)
    next_w_qkv_hbm = next(it)
    next_s_qkv_ref = next(it)
    next_w_gate_hbm = next(it)
    next_s_gate_ref = next(it)
    cos_sin_ref = next(it) if has_rope else None

    # Outputs.
    out_stream_chip_hbm = next(it)
    out_qkv_or_q_ref = next(it)
    out_kv_or_gate_ref = next(it)
    out_gate_epi_ref = next(it) if fuse_qkv_epi else None

    # MoE scratch.
    msc = _unpack_muse_sparse_scratch(
        it,
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=sharded_moe_in,
        cfg=cfg,
    )

    rows = gated_ref.shape[0]
    o_in = gated_ref.shape[1]
    r_pair = stream_chip_hbm.shape[0]
    rows_per_rank = rows // world_size
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    pre_pair_cols = w_pre_pair_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    n_chips = world_size // 2
    use_xy = world_size == 8 and r_pair >= 16

    # Shared Bridge 1b + Bridge 2b VMEM scratch.
    w_pre_pair_buf = next(it)
    w_router_buf = next(it)
    s_buf = next(it)
    next_in_vmem = next(it)
    if alias_mate_recv:
        mate_bf = msc["y_stage_buf"].bitcast(jnp.bfloat16)
        mate_recv = mate_bf if mate_bf.shape[0] == rows else mate_bf.at[pl.ds(0, rows), :]
    else:
        mate_recv = next(it)
    if interpret_mode:
        send_half = next(it)
        own_half = next(it)
    else:
        own_half = next_in_vmem.at[:, pl.ds(0, 4096)]
        send_half = next_in_vmem.at[:, pl.ds(4096, 4096)]
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
    if (not interpret_mode) and sharded_moe_in:
        expert_in_vmem = next_in_vmem.at[:, pl.ds(0, 4096)]
    else:
        expert_in_vmem = next(it)
    moe_sh_vmem = next(it) if sharded_moe_in else None
    b2_scalar_dummy_post = next(it)  # (8, 128)
    b2_scalar_dummy_cols = next(it)  # (8, 128)

    # Semaphores: keep RS receive sems (rs_pair_rsem, rs_chip_rsem) and Stage-4 fan_* 100%
    # disjoint from Stage-5 b2 sems! When TP=16 B=64 (not sharded_moe_in and world_size == 16):
    # - w_pre/w_router/s_in drain in Stages 1-3, so w_post/w_qkv/w_gate alias them sequentially
    #   (while s_out_sem stays dedicated because Stage-5 stream store overlaps w_gate DMA wait);
    # - ag_exp_recv is used ONLY in Stage-3 ag_exp (is_tp2_wpost=True skips Stage-5 post_ici) and
    #   is separated from Stage-4 fan_out by MoE PAIR_RS, so aliasing msc["fan_send_sem"] is safe;
    # - ag_log_recv (Stage-5 in_ici_recv!) stays strictly dedicated (7,) so total sems <= 63 <= 64.
    w_pre_sem = next(it)
    w_router_sem = next(it)
    s_in_sem = next(it)
    s_out_sem = next(it)
    if sharded_moe_in or world_size == 8:
        w_post_sem = next(it)
        w_qkv_sem = next(it)
        w_gate_sem = next(it)
    else:
        w_post_sem = w_pre_sem
        w_qkv_sem = w_router_sem
        w_gate_sem = s_in_sem
    c1a_send = next(it)
    c1a_recv = next(it)
    rs1_send = next(it)
    rs1_recv = next(it)

    if use_xy:
        rs2_send = next(it)
        rs2_recv = next(it)
    else:
        rs2_send, rs2_recv = rs1_send, rs1_recv
    rs3_send, rs3_recv = rs1_send, rs1_recv

    if sharded_moe_in or world_size == 8:
        c1c_send = next(it)
        c1c_recv = next(it)
        pre_send = next(it)
        pre_recv = next(it)
        ag_exp_d2d_send = next(it)
        ag_exp_d2d_recv = next(it)
        ag_exp_recv = next(it)
        ag_log_recv = next(it)
    else:
        c1c_send, c1c_recv = c1a_send, c1a_recv
        pre_send, pre_recv = rs1_send, rs1_recv
        ag_exp_d2d_send, ag_exp_d2d_recv = c1a_send, c1a_recv
        ag_exp_recv = msc["fan_send_sem"].at[pl.ds(0, n_chips - 1)]
        ag_log_recv = next(it)

    if msc["rs_pair_ssem"] is not None:
        ag_exp_send = msc["rs_pair_ssem"]
        ag_log_send = msc["rs_chip_ssem"]
    else:
        ag_exp_send = next(it)
        ag_log_send = next(it)

    if interpret_mode:
        wo_buf = next(it)
        w_pre_core_buf = w_pre_pair_buf
        w_post_buf = next(it)
        w_qkv_buf = next(it)
        w_gate_buf = next(it)
    else:
        wo_buf = _alias_wo_buf(
            msc["w1_buf"],
            msc["w2_buf"],
            w_mode=msc["w_mode"],
            o_in=o_in,
            half=half,
        )
        w_pre_core_buf = (
            w_pre_pair_buf
            if w_pre_pair_buf.shape[1] == pre_pair_cols
            else w_pre_pair_buf.at[:, pl.ds(0, pre_pair_cols)]
        )
        w_post_buf, w_qkv_buf, w_gate_buf = _alias_b2_weights(
            msc["w1_buf"],
            w_pre_pair_buf,
            w_mode=msc["w_mode"],
            qkv_cols=qkv_cols,
            gate_cols=gate_cols,
        )

    me = rank_ref[0]
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

    def _during_ag_exp_ici() -> None:
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
            zero_and_pack=(r_pair == 8),
        )
        plan_holder.append(plan_res)

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

    # =========================================================================
    # Stage 1-3: Bridge 1b -> live s_buf + logits_vmem + expert_in_vmem + MoE plan
    # =========================================================================
    _run_chip_b1_vmem_core(
        stream_chip_hbm,
        gated_ref,
        wo_hbm,
        wo_scale_ref,
        pa_alpha_ref,
        pa_beta_ref,
        pre_ff_gain_ref,
        w_pre_pair_hbm,
        w_pre_pair_scale_ref,
        w_router_hbm,
        pre_exp_gain_ref,
        out_stream_chip_hbm,
        logits_vmem,
        expert_in_vmem,
        wo_buf,
        w_pre_core_buf,
        w_router_buf,
        s_buf,
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
        msc["w1_sem"],
        w_pre_sem,
        w_router_sem,
        s_in_sem,
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
        ag_exp_d2d_send,
        ag_exp_d2d_recv,
        ag_log_send,
        ag_log_recv,
        world_size=world_size,
        pa_eps=pa_eps,
        pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        router_multiplier=router_multiplier,
        pre_exp_eps=pre_exp_eps,
        pre_exp_scale_input=pre_exp_scale_input,
        skip_barrier_signal=False,
        skip_barrier_wait=False,
        skip_stream_load=False,
        skip_stream_store=True,
        early_router_and_log_ag=True,
        during_ag_exp_ici_fn=_during_ag_exp_ici,
        during_ag_exp_d2d2_fn=None if r_pair == 8 else _during_ag_exp_d2d2,
    )

    # =========================================================================
    # Stage 4: Sparse NVFP4 MoE + PAIR_RS (launching b2 weight DMAs in parallel!)
    # =========================================================================
    moe_dst_vmem = moe_sh_vmem if sharded_moe_in else expert_in_vmem

    def _on_experts_done() -> None:
        # All w1/w2/spec_prime DMAs have completed; launch 34 MiB b2 weights under PAIR_RS!
        pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
        pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
        pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

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
        output_ref=moe_dst_vmem,
        ep=world_size,
        batch_size=rows,
        scale_block=scale_block,
        return_sharded=sharded_moe_in,
        ep_axis_name=ep_axis_name,
        cfg=cfg,
        x_is_3d=False,
        wait_entry_barrier=False,
        write_output_ref=True,
        on_experts_done_fn=_on_experts_done,
    )

    # =========================================================================
    # Stage 5: Bridge 2b from live VMEM (s_buf + moe_dst_vmem + prestarted weights)
    # =========================================================================
    # Reuse dead chip_b1 activation VMEM & sems cleanly (NEVER touch RS recv buffers/sems,
    # and never alias qkv_vmem into next_in_vmem since next_in_vmem feeds qkv dot!).
    qkv_vmem_slice = mate_recv.at[:, pl.ds(0, qkv_cols)]
    mate_tok_slice = send_half.at[pl.ds(0, rows_per_rank), :]
    moe_full_slice = mate_recv.at[pl.ds(0, r_pair), :]

    _run_chip_b2_vmem_core(
        stream_chip_hbm,
        moe_dst_vmem,
        w_post_hbm,
        s_post_ref,
        post_ff_gain_ref,
        pf_alpha_ref,
        pf_beta_ref,
        next_in_gain_ref,
        next_w_qkv_hbm,
        next_s_qkv_ref,
        next_w_gate_hbm,
        next_s_gate_ref,
        cos_sin_ref,
        out_stream_chip_hbm,
        out_qkv_or_q_ref,
        out_kv_or_gate_ref,
        out_gate_epi_ref,
        w_post_buf,
        w_qkv_buf,
        w_gate_buf,
        s_buf,
        ffn_pair_buf,  # next_in_pair (used only when r_pair == 8)
        next_in_vmem,
        qkv_vmem_slice,
        mate_tok_slice,
        moe_full_slice,
        b2_scalar_dummy_post,
        my_pair_bf16,  # mate_post_shard [r_pair, half]
        recv_pair_bf16,  # chip_cols_buf [r_pair, half]
        ffn_pair_buf,  # post_pair_full [r_pair, hidden]
        b2_scalar_dummy_cols,
        exp_3d_buf,  # ag_3d_buf [n_chips, 8, half] (r_pair == 8)
        w_post_sem,
        w_qkv_sem,
        w_gate_sem,
        s_in_sem,
        s_out_sem,
        c1a_send,  # tok_s1_send (1)
        c1a_recv,  # tok_s1_recv (1)
        ag_log_send,  # tok_s2_send (reuses rs_chip_ssem, shape (n_chips,))
        ag_log_recv,  # tok_s2_recv (reuses ag_log_recv)
        ag_exp_d2d_send,  # tok_d2d2_send
        ag_exp_d2d_recv,  # tok_d2d2_recv
        pre_send,  # post_d2d_send
        pre_recv,  # post_d2d_recv
        ag_exp_send,  # post_ici_send
        ag_exp_recv,  # post_ici_recv
        ag_log_send,  # in_ici_send
        ag_log_recv,  # in_ici_recv
        c1c_send,  # in_d2d2_send
        c1c_recv,  # in_d2d2_recv
        world_size=world_size,
        sharded_moe_in=sharded_moe_in,
        post_ff_eps=post_ff_eps,
        post_ff_scale_input=post_ff_scale_input,
        next_in_eps=next_in_eps,
        next_in_scale_input=next_in_scale_input,
        fuse_qkv_epi=fuse_qkv_epi,
        qkv_norm_eps=qkv_norm_eps,
        query_scale=query_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        has_rope=has_rope,
        interpret_mode=interpret_mode,
        skip_barrier_signal=True,
        skip_barrier_wait=True,
        skip_stream_load=True,
        skip_stream_store=False,
        prestarted_w_post_dma=True,
        prestarted_w_qkv_dma=True,
        prestarted_w_gate_dma=True,
    )


def b1_moe_b2_chip_sharded_jax(
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
    w_post_tp2_w8k: jax.Array,
    s_post_tp2: jax.Array,
    post_ff_gain: jax.Array,
    pf_alpha: jax.Array,
    pf_beta: jax.Array,
    next_in_gain: jax.Array,
    next_w_qkv_w8k: jax.Array,
    next_s_qkv: jax.Array,
    next_w_gate_w8k: jax.Array,
    next_s_gate: jax.Array,
    cos_sin_step: jax.Array | None = None,
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
    post_ff_eps: float = 1e-8,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    fuse_qkv_epi: bool = True,
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    replica_cost_mode: str = "active",
    ep_axis_name: str = "model",
    config: SparseDecodeConfig | None = None,
    collective_id: int = _W8_CID_BASE + 20,
    interpret: bool = False,
):
    """Fused Step (b) superkernel: ``attn_{L} -> MoE_{L} -> attn_{L+1}``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2_chip_sharded_jax requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    rows, o_in = gated_attn.shape
    r_pair, hidden = stream_chip.shape
    n_chips = world_size // 2
    two_pair = 2 * r_pair
    four_pair = 4 * r_pair
    _, pre_pair_cols = w_pre_pair_w8k.shape
    h_pre = 2 * pre_pair_cols
    _, n_exp = w_router_km.shape
    _, post_cols = w_post_tp2_w8k.shape
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape

    valid_rows = (32, 64, 128) if world_size == 8 else (64, 128)
    if (
        rows not in valid_rows
        or rows != n_chips * r_pair
        or r_pair < 8
        or r_pair % 8 != 0
        or hidden != 8192
        or h_pre != _H
        or n_exp != _E_TOTAL
        or post_cols != _H
        or w_post_tp2_w8k.shape[0] != _H
    ):
        raise ValueError(
            f"Unsupported b1_moe_b2 shapes world_size={world_size}, stream={stream_chip.shape}, "
            f"gated={gated_attn.shape}, w_pre={w_pre_pair_w8k.shape}, w_post={w_post_tp2_w8k.shape}"
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

    rows_per_rank = rows // world_size
    sharded_moe_in = rows_per_rank >= 8 and (rows_per_rank % 8 == 0)
    half = hidden // 2
    q_cols = half // 2
    is_r16 = w_router_km.dtype == jnp.bfloat16
    use_xy = world_size == 8 and r_pair >= 16
    use_pair_rs = config.pair_rs and (world_size in (8, 16))
    if world_size == 16 and not use_pair_rs:
        raise ValueError(
            "TP=16 b1_moe_b2_chip_sharded_jax requires config.pair_rs=True to stay <= 64 DMA semaphores"
        )
    has_rope = bool(fuse_qkv_epi and cos_sin_step is not None)
    cs_in = cos_sin_step[:rows] if (has_rope and cos_sin_step.shape[0] != rows) else cos_sin_step

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

    moe_scratch = build_muse_sparse_scratch_shapes(
        ep=world_size,
        batch_size=rows,
        scale_block=scale_block,
        s1_dtype=s1.dtype,
        s2_dtype=s2.dtype,
        num_replicas=num_replicas,
        return_sharded=sharded_moe_in,
        config=config,
    )
    w_pre_cols_buf = pre_pair_cols if interpret_mode else max(pre_pair_cols, qkv_cols + gate_cols)
    alias_mate_recv = (
        (not interpret_mode)
        and config.row_mode != "dense"
        and config.norm_f32_stage
        and (not config.loop32_in_tile)
        and (not (scale_block >= 256 and config.lhs_fp8))
        and (2 * config.tile_m >= rows)
    )
    shared_scratch = [
        vmem((hidden, w_pre_cols_buf), w_pre_pair_w8k.dtype),
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem((r_pair, hidden), jnp.float32),
        vmem((rows, hidden), jnp.bfloat16),  # next_in_vmem (holds own_half/send_half/expert_in earlier)
        *(() if alias_mate_recv else [vmem((rows, half), jnp.bfloat16)]),  # mate_recv
        *(([vmem((rows, half), jnp.bfloat16), vmem((rows, half), jnp.bfloat16)]) if interpret_mode else []),
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
        *(([vmem((rows, h_pre), jnp.bfloat16)]) if (interpret_mode or not sharded_moe_in) else []),
        *(([vmem((rows_per_rank, h_pre), jnp.bfloat16)]) if sharded_moe_in else []),
        vmem((8, 128), jnp.bfloat16),  # b2_scalar_dummy_post
        vmem((8, 128), jnp.bfloat16),  # b2_scalar_dummy_cols
        # Semaphores.
        sem((1,)),  # w_pre_sem
        sem((1,)),  # w_router_sem
        sem((1,)),  # s_in_sem
        sem((1,)),  # s_out_sem
        *(
            [
                sem((1,)),  # w_post_sem
                sem((1,)),  # w_qkv_sem
                sem((1,)),  # w_gate_sem
            ]
            if (sharded_moe_in or world_size == 8)
            else []
        ),
        sem((1,)),  # c1a_send
        sem((1,)),  # c1a_recv
        sem((3 if use_xy else 1,)),  # rs1_send
        sem((3 if use_xy else 1,)),  # rs1_recv
        *(([sem((2,)), sem((2,))]) if use_xy else []),
        *(
            [
                sem((1,)),  # c1c_send
                sem((1,)),  # c1c_recv
                sem((1,)),  # pre_send
                sem((1,)),  # pre_recv
                sem((1,)),  # ag_exp_d2d_send
                sem((1,)),  # ag_exp_d2d_recv
                sem((n_chips - 1,)),  # ag_exp_recv
                sem((n_chips - 1,)),  # ag_log_recv
            ]
            if (sharded_moe_in or world_size == 8)
            else [
                sem((n_chips - 1,)),  # ag_log_recv (disjoint from Stage-4 fan_*)
            ]
        ),
        *(() if use_pair_rs else [sem((n_chips - 1,)), sem((n_chips - 1,))]),
    ]
    if interpret_mode:
        shared_scratch.extend(
            [
                vmem((2, o_in, half), wo_w8k.dtype),
                vmem((_H, post_cols), w_post_tp2_w8k.dtype),
                vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
                vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            ]
        )

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = post_exp_norm.reshape(1, _H)
    has_spec_slot_arg = spec_home_slot is not None
    scalar_args = [jnp.atleast_1d(rank).astype(jnp.int32)]
    if has_replicas:
        scalar_args.append(replica_meta)
    if has_spec_slot_arg:
        scalar_args.append(jnp.atleast_1d(spec_home_slot).astype(jnp.int32))

    b1_in_specs = (
        h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp,
    )
    moe_in_specs = (v_sp, h_sp, h_sp, h_sp, h_sp, v_sp) + (
        (h_sp, h_sp, h_sp, h_sp) if has_replicas else ()
    )
    b2_in_specs = (
        h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp,
    ) + ((v_sp,) if has_rope else ())
    in_specs = b1_in_specs + moe_in_specs + b2_in_specs

    b1_args = (
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
    )
    moe_args = (bias_2d, w1, s1, w2, s2, norm_2d) + (
        (rep_w1, rep_s1, rep_w2, rep_s2) if has_replicas else ()
    )
    b2_args = (
        w_post_tp2_w8k,
        _as_dot_scale_f32(s_post_tp2, post_cols),
        _as_2d_f32(post_ff_gain, hidden),
        _as_2d_f32(pf_alpha, hidden),
        _as_2d_f32(pf_beta, hidden),
        _as_2d_f32(next_in_gain, hidden),
        next_w_qkv_w8k,
        _as_dot_scale_f32(next_s_qkv, qkv_cols),
        next_w_gate_w8k,
        _as_dot_scale_f32(next_s_gate, gate_cols),
    ) + ((cs_in,) if has_rope else ())
    call_args = (*scalar_args, *b1_args, *moe_args, *b2_args)

    if fuse_qkv_epi:
        out_shape = (
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        )
        out_specs = (h_sp, v_sp, v_sp, v_sp)
    else:
        out_shape = (
            jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        )
        out_specs = (h_sp, v_sp, v_sp)

    kernel_fn = functools.partial(
        _b1_moe_b2_chip_sharded_kernel,
        world_size=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        sharded_moe_in=bool(sharded_moe_in),
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        pa_eps=float(pa_eps),
        pre_ff_eps=float(pre_ff_eps),
        pre_ff_scale_input=bool(pre_ff_scale_input),
        router_multiplier=float(router_multiplier),
        pre_exp_eps=float(pre_exp_eps),
        pre_exp_scale_input=bool(pre_exp_scale_input),
        post_ff_eps=float(post_ff_eps),
        post_ff_scale_input=bool(post_ff_scale_input),
        next_in_eps=float(next_in_eps),
        next_in_scale_input=bool(next_in_scale_input),
        fuse_qkv_epi=bool(fuse_qkv_epi),
        qkv_norm_eps=float(qkv_norm_eps),
        query_scale=float(query_scale),
        k_scale=float(k_scale),
        v_scale=float(v_scale),
        has_rope=has_rope,
        cfg=config,
        has_spec_slot_arg=has_spec_slot_arg,
        interpret_mode=interpret_mode,
        alias_mate_recv=alias_mate_recv,
    )

    res = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            grid=(1,),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=moe_scratch + shared_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(64 if rows > 64 else 60) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=interpret_arg,
        name=(
            f"muse_w9_b1_moe_b2_b{rows}_tp{world_size}_k{scale_block}"
            f"_r{num_replicas}_{'r16' if is_r16 else 'r32'}"
            f"{'_epi_' + ('rope' if has_rope else 'nope') if fuse_qkv_epi else ''}"
        ),
    )(*call_args)
    if fuse_qkv_epi:
        st_ch, q_3d, kv_m, gate = res
        return st_ch, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate
    return res
