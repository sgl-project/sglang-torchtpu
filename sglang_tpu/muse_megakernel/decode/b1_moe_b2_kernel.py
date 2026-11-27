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

import dataclasses
import functools
import math
import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    SUPPORTED_WORLD_SIZES,
    _W8_CID_BASE,
    _as_2d_f32,
    _as_dot_scale_f32,
    _dma_wait,
    _run_chip_b1_vmem_core,
    _run_chip_b2_vmem_core,
    _signal_barrier,
    _wait_barrier,
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


def _unpack_rpad_outer_meta(it, msc_outer, *, world_size: int, rpa_cfg, rpa_skip_kv_write: bool):
    """Unpack outer Step (c) RPAd metadata SMEM and DMA semaphores."""
    from jax.experimental import pallas as pl

    written_cnt_smem_ref = next(it)
    if world_size == 16:
        rpad_bkv_sems = msc_outer["rs_pair_ssem"].at[pl.ds(0, rpa_cfg.num_slots)]
        rpad_cache_write_sem = (
            None
            if rpa_skip_kv_write
            else msc_outer["rs_pair_ssem"].at[pl.ds(rpa_cfg.num_slots, 1)]
        )
    else:
        rpad_bkv_sems = next(it)
        rpad_cache_write_sem = None if rpa_skip_kv_write else next(it)
    return written_cnt_smem_ref, rpad_bkv_sems, rpad_cache_write_sem


def _unpack_rpad_scope_refs(
    s_it,
    *,
    rpa_skip_kv_write: bool,
    rpa_kv_pack_dense2: bool,
    rpa_kv_pack_bd4: bool,
    rpa_cfg,
    rpa_use_grp_vec: bool,
) -> tuple:
    """Unpack Scope 2 / Scope 3 RPAd VMEM buffers from a scoped ref iterator."""
    rpad_q_vmem = next(s_it)
    rpad_kv_vmem = None if rpa_skip_kv_write else next(s_it)
    rpad_gate_vmem = next(s_it)
    bkv_vmem_ref = next(s_it)
    state_vmem_ref = next(s_it)
    o_vmem_ref = next(s_it)
    kv_write_vmem_ref = (
        next(s_it)
        if ((rpa_kv_pack_dense2 or rpa_kv_pack_bd4) and not rpa_skip_kv_write)
        else None
    )
    init_sv_vmem_ref = (
        next(s_it)
        if (getattr(rpa_cfg, "hoist_new_kv", False) or rpa_use_grp_vec)
        else None
    )
    return (
        rpad_q_vmem,
        rpad_kv_vmem,
        rpad_gate_vmem,
        bkv_vmem_ref,
        state_vmem_ref,
        o_vmem_ref,
        kv_write_vmem_ref,
        init_sv_vmem_ref,
    )


def _b1_moe_b2_chip_sharded_impl(
    all_refs,
    *,
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
    has_free_params_arg: bool,
    interpret_mode: bool,
    alias_mate_recv: bool,
    has_rpad: bool,
    # RPAd options (used only when has_rpad=True).
    rpa_sm_scale: float = 1.0,
    rpa_sliding_window: int | None = None,
    rpa_k_scale: float | None = None,
    rpa_v_scale: float | None = None,
    rpa_out_norm_eps: float = 1e-5,
    rpa_cfg=None,
    rpa_one_block_per_seq: bool = True,
    rpa_kv_pack_dense2: bool = False,
    rpa_kv_pack_bd4: bool = False,
    rpa_use_grp_vec: bool = False,
    rpa_skip_kv_write: bool = False,
    step_c_moe_is_scope1: tuple[bool, ...] = (),
    step_c_n_moe_outer: int = 0,
    step_c_n_moe_scope1: int = 0,
    step_c_n_b1_scope1: int = 0,
    step_c_scope1_shapes: tuple = (),
    step_c_scope2_shapes: tuple = (),
    step_b3_n_moe_outer: int = 0,
    step_b3_moe_is_scope2: tuple[bool, ...] = (),
    step_b3_scope1_shapes: tuple = (),
    step_b3_scope2_shapes: tuple = (),
    step_b3_scope3_shapes: tuple = (),
    use_3scope: bool = False,
) -> None:
    """Shared Pallas kernel body for Step (b) and 2-scope Step (c) (``+ RPAd``)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    it = iter(all_refs)
    # Scalar prefetch.
    rank_ref = next(it)
    replica_meta_ref = next(it) if num_replicas > 0 else None
    spec_slot_ref = next(it) if has_spec_slot_arg else None
    free_params_ref = next(it) if has_free_params_arg else None
    if has_rpad:
        kv_lens_ref = next(it)
        page_indices_ref = next(it)
        cu_q_lens_ref = next(it)
        cu_kv_lens_ref = next(it)
        distribution_ref = next(it)
    else:
        kv_lens_ref = page_indices_ref = cu_q_lens_ref = None
        cu_kv_lens_ref = distribution_ref = None

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
    kv_cache_hbm_ref = next(it) if has_rpad else None

    # Outputs.
    out_stream_chip_hbm = next(it)
    if has_rpad:
        out_gated_ref = next(it)
        if rpa_skip_kv_write:
            out_kv_merged_ref = next(it)
            out_kv_cache_hbm_ref = None
        else:
            out_kv_cache_hbm_ref = next(it)
            out_kv_merged_ref = None
        out_qkv_or_q_ref = out_kv_or_gate_ref = out_gate_epi_ref = None
    else:
        out_qkv_or_q_ref = next(it)
        out_kv_or_gate_ref = next(it)
        out_gate_epi_ref = next(it) if fuse_qkv_epi else None
        out_gated_ref = out_kv_cache_hbm_ref = out_kv_merged_ref = None

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
    use_3d_x = cfg.row_mode != "dense"
    need_b1_exp_in = (not use_3d_x) and (interpret_mode or not sharded_moe_in)

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

    def _unpack_sems(cur_it, *, rs_pair_ssem, rs_chip_ssem, fan_send_sem) -> dict:
        w_pre_sem = next(cur_it)
        w_router_sem = next(cur_it)
        s_in_sem = next(cur_it)
        s_out_sem = next(cur_it)
        if sharded_moe_in or world_size == 8:
            w_post_sem = next(cur_it)
            w_qkv_sem = next(cur_it)
            w_gate_sem = next(cur_it)
        else:
            w_post_sem, w_qkv_sem, w_gate_sem = w_pre_sem, w_router_sem, s_in_sem
        c1a_send = next(cur_it)
        c1a_recv = next(cur_it)
        rs1_send = next(cur_it)
        rs1_recv = next(cur_it)
        if use_xy:
            rs2_send = next(cur_it)
            rs2_recv = next(cur_it)
        else:
            rs2_send, rs2_recv = rs1_send, rs1_recv
        rs3_send, rs3_recv = rs1_send, rs1_recv
        if sharded_moe_in or world_size == 8:
            c1c_send = next(cur_it)
            c1c_recv = next(cur_it)
            pre_send = next(cur_it)
            pre_recv = next(cur_it)
            ag_exp_d2d_send = next(cur_it)
            ag_exp_d2d_recv = next(cur_it)
            ag_exp_recv = next(cur_it)
            ag_log_recv = next(cur_it)
        else:
            c1c_send, c1c_recv = c1a_send, c1a_recv
            pre_send, pre_recv = rs1_send, rs1_recv
            ag_exp_d2d_send, ag_exp_d2d_recv = c1a_send, c1a_recv
            ag_exp_recv = fan_send_sem.at[pl.ds(0, n_chips - 1)]
            ag_log_recv = next(cur_it)
        if rs_pair_ssem is not None:
            ag_exp_send = rs_pair_ssem
            ag_log_send = rs_chip_ssem
        else:
            ag_exp_send = next(cur_it)
            ag_log_send = next(cur_it)
        return dict(
            w_pre_sem=w_pre_sem, w_router_sem=w_router_sem, s_in_sem=s_in_sem, s_out_sem=s_out_sem,
            w_post_sem=w_post_sem, w_qkv_sem=w_qkv_sem, w_gate_sem=w_gate_sem,
            c1a_send=c1a_send, c1a_recv=c1a_recv, rs1_send=rs1_send, rs1_recv=rs1_recv,
            rs2_send=rs2_send, rs2_recv=rs2_recv, rs3_send=rs3_send, rs3_recv=rs3_recv,
            c1c_send=c1c_send, c1c_recv=c1c_recv, pre_send=pre_send, pre_recv=pre_recv,
            ag_exp_d2d_send=ag_exp_d2d_send, ag_exp_d2d_recv=ag_exp_d2d_recv,
            ag_exp_send=ag_exp_send, ag_exp_recv=ag_exp_recv,
            ag_log_send=ag_log_send, ag_log_recv=ag_log_recv,
        )

    def _run_stage1_4_b1_moe(
        *,
        msc: dict,
        w_pre_core_buf,
        wo_buf,
        w_router_buf,
        s_buf,
        send_half,
        mate_recv,
        own_half,
        b1_ephem: tuple,
        my_pair_bf16,
        recv_pair_bf16,
        ffn_pair_buf,
        exp_3d_buf,
        moe_dst_vmem,
        expert_in_vmem,
        w_post_buf,
        w_qkv_buf,
        w_gate_buf,
        sems: dict,
    ) -> None:
        (
            rs1_a_or_z_recv,
            rs1_b_or_y_recv,
            rs2_a_or_x_recv,
            rs2_b_recv,
            sq_pair_pa_buf,
            router_pair_buf,
            pre_my_half,
            pre_mate_half,
            sq_pair_pre_buf,
            logits_vmem,
        ) = b1_ephem
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
                free_params_ref=free_params_ref,
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
            None if use_3d_x else expert_in_vmem,
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
            sems["w_pre_sem"],
            sems["w_router_sem"],
            sems["s_in_sem"],
            sems["s_out_sem"],
            sems["c1a_send"],
            sems["c1a_recv"],
            sems["rs1_send"],
            sems["rs1_recv"],
            sems["rs2_send"],
            sems["rs2_recv"],
            sems["rs3_send"],
            sems["rs3_recv"],
            sems["c1c_send"],
            sems["c1c_recv"],
            sems["pre_send"],
            sems["pre_recv"],
            sems["ag_exp_send"],
            sems["ag_exp_recv"],
            sems["ag_exp_d2d_send"],
            sems["ag_exp_d2d_recv"],
            sems["ag_log_send"],
            sems["ag_log_recv"],
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
            out_expert_3d_ref=msc["x_3d_buf"] if use_3d_x else None,
        )

        def _on_experts_done() -> None:
            pltpu.make_async_copy(w_post_hbm, w_post_buf, sems["w_post_sem"].at[0]).start()
            pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, sems["w_qkv_sem"].at[0]).start()
            pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, sems["w_gate_sem"].at[0]).start()

        run_muse_sparse_experts_and_rs_from_vmem(
            me,
            None if use_3d_x else expert_in_vmem,
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
            x_is_3d=use_3d_x,
            wait_entry_barrier=False,
            write_output_ref=True,
            on_experts_done_fn=_on_experts_done,
        )

    def _run_stage5_6_b2_rpad(
        *,
        moe_dst_vmem,
        w_post_buf,
        w_qkv_buf,
        w_gate_buf,
        s_buf,
        next_in_vmem,
        send_half,
        mate_recv,
        my_pair_bf16,
        recv_pair_bf16,
        ffn_pair_buf,
        b2_scalar_dummy_post,
        b2_scalar_dummy_cols,
        exp_3d_buf,
        sems: dict,
        rpad_refs: tuple | None,
        written_cnt_smem_ref=None,
        rpad_bkv_sems=None,
        rpad_cache_write_sem=None,
        sync_entry_barrier: bool = False,
        prestarted_qkv_gate_dma: bool = True,
    ) -> None:
        qkv_vmem_slice = mate_recv.at[:, pl.ds(0, qkv_cols)]
        mate_tok_slice = send_half.at[pl.ds(0, rows_per_rank), :]
        moe_full_slice = mate_recv.at[pl.ds(0, r_pair), :]

        if rpad_refs is not None:
            from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
                _run_rpad_vmem_core,
                prepare_and_start_rpad_dmas,
            )

            (
                rpad_q_vmem,
                rpad_kv_vmem,
                rpad_gate_vmem,
                bkv_vmem_ref,
                state_vmem_ref,
                o_vmem_ref,
                kv_write_vmem_ref,
                init_sv_vmem_ref,
            ) = rpad_refs
            pre_state_holder = []

            def _after_qkv_epi() -> None:
                pre_state = prepare_and_start_rpad_dmas(
                    kv_lens_ref=kv_lens_ref,
                    page_indices_ref=page_indices_ref,
                    cu_q_lens_ref=cu_q_lens_ref,
                    cu_kv_lens_ref=cu_kv_lens_ref,
                    distribution_ref=distribution_ref,
                    kv_cache_hbm_ref=kv_cache_hbm_ref,
                    bkv_vmem_ref=bkv_vmem_ref,
                    bkv_sems=rpad_bkv_sems,
                    sliding_window=rpa_sliding_window,
                    cfg=rpa_cfg,
                    one_block_per_seq=rpa_one_block_per_seq,
                    decode_query_size=1,
                    kv_pack_dense2=rpa_kv_pack_dense2,
                    interpret_mode=interpret_mode,
                    b_tokens=rows,
                    init_sv_vmem_ref=init_sv_vmem_ref,
                    use_grp_vec=rpa_use_grp_vec,
                    kv_pack_bd4=rpa_kv_pack_bd4,
                )
                pre_state_holder.append(pre_state)

            b2_out_q = rpad_q_vmem
            b2_out_kv = out_kv_merged_ref if rpa_skip_kv_write else rpad_kv_vmem
            b2_out_gate_arg = rpad_gate_vmem
            b2_gate_dst = rpad_gate_vmem
            b2_after_qkv_fn = _after_qkv_epi
            defer_store_wait = True
        else:
            b2_out_q = out_qkv_or_q_ref
            b2_out_kv = out_kv_or_gate_ref
            b2_out_gate_arg = out_gate_epi_ref
            b2_gate_dst = None
            b2_after_qkv_fn = None
            defer_store_wait = False

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
            b2_out_q,
            b2_out_kv,
            b2_out_gate_arg,
            w_post_buf,
            w_qkv_buf,
            w_gate_buf,
            s_buf,
            ffn_pair_buf,
            next_in_vmem,
            qkv_vmem_slice,
            mate_tok_slice,
            moe_full_slice,
            b2_scalar_dummy_post,
            my_pair_bf16,
            recv_pair_bf16,
            ffn_pair_buf,
            b2_scalar_dummy_cols,
            exp_3d_buf,
            sems["w_post_sem"],
            sems["w_qkv_sem"],
            sems["w_gate_sem"],
            sems["s_in_sem"],
            sems["s_out_sem"],
            sems["c1a_send"],
            sems["c1a_recv"],
            sems["ag_log_send"],
            sems["ag_log_recv"],
            sems["ag_exp_d2d_send"],
            sems["ag_exp_d2d_recv"],
            sems["pre_send"],
            sems["pre_recv"],
            sems["ag_exp_send"],
            sems["ag_exp_recv"],
            sems["ag_log_send"],
            sems["ag_log_recv"],
            sems["c1c_send"],
            sems["c1c_recv"],
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
            skip_barrier_signal=not sync_entry_barrier,
            skip_barrier_wait=not sync_entry_barrier,
            skip_stream_load=True,
            skip_stream_store=False,
            prestarted_w_post_dma=True,
            prestarted_w_qkv_dma=prestarted_qkv_gate_dma,
            prestarted_w_gate_dma=prestarted_qkv_gate_dma,
            gate_dst_ref=b2_gate_dst,
            after_qkv_epi_fn=b2_after_qkv_fn,
            defer_stream_store_wait=defer_store_wait,
        )

        if rpad_refs is not None:
            if rpa_kv_pack_bd4:
                # TP=16 bd4 packs [K_even | K_odd] across lanes 0..63 and 64..127, so populate
                # sublanes 8..15 with [0, Q] matching the [Q, 0] in sublanes 0..7.
                q_rows_tot = rpad_q_vmem.shape[0]
                step_q = math.gcd(q_rows_tot, 8)
                for q0 in range(0, q_rows_tot, step_q):
                    q_sl = pl.ds(q0, step_q)
                    q_lo_2d = rpad_q_vmem[q_sl, pl.ds(0, 8), :].reshape(step_q * 8, 128)
                    q_hi_2d = (
                        jnp.roll(q_lo_2d, 64, axis=1)
                        if interpret_mode
                        else pltpu.roll(q_lo_2d, 64, axis=1)
                    )
                    rpad_q_vmem[q_sl, pl.ds(8, 8), :] = q_hi_2d.reshape(step_q, 8, 128)
                q_for_rpa = rpad_q_vmem
            elif world_size == 16:
                q_for_rpa = rpad_q_vmem.at[:, pl.ds(0, 8), :]
            else:
                q_for_rpa = rpad_q_vmem
            _run_rpad_vmem_core(
                kv_lens_ref=kv_lens_ref,
                page_indices_ref=page_indices_ref,
                cu_q_lens_ref=cu_q_lens_ref,
                cu_kv_lens_ref=cu_kv_lens_ref,
                distribution_ref=distribution_ref,
                q_vmem_ref=q_for_rpa,
                kv_merged_vmem_ref=b2_out_kv,
                gate_vmem_ref=rpad_gate_vmem,
                kv_cache_hbm_ref=kv_cache_hbm_ref,
                o_out_vmem_ref=out_gated_ref,
                updated_kv_cache_hbm_ref=out_kv_cache_hbm_ref,
                bkv_vmem_ref=bkv_vmem_ref,
                state_vmem_ref=state_vmem_ref,
                o_vmem_ref=o_vmem_ref,
                kv_write_vmem_ref=kv_write_vmem_ref,
                init_sv_vmem_ref=init_sv_vmem_ref,
                written_cnt_smem_ref=written_cnt_smem_ref,
                bkv_sems=rpad_bkv_sems,
                cache_write_sem=rpad_cache_write_sem,
                sm_scale=rpa_sm_scale,
                sliding_window=rpa_sliding_window,
                k_scale=rpa_k_scale,
                v_scale=rpa_v_scale,
                cfg=rpa_cfg,
                one_block_per_seq=rpa_one_block_per_seq,
                decode_query_size=1,
                in_kernel_pack2=True,
                has_gate=True,
                out_norm_eps=rpa_out_norm_eps,
                kv_pack_dense2=rpa_kv_pack_dense2,
                interpret_mode=interpret_mode,
                prestarted_bkv_state=(pre_state_holder[0] if pre_state_holder else None),
                use_grp_vec=rpa_use_grp_vec,
                skip_kv_write=rpa_skip_kv_write,
                kv_pack_bd4=rpa_kv_pack_bd4,
            )
            _dma_wait(out_stream_chip_hbm, sems["s_out_sem"].at[0])

    if (not has_rpad) and (not use_3scope):
        msc = _unpack_muse_sparse_scratch(
            it,
            ep=world_size,
            scale_block=scale_block,
            num_replicas=num_replicas,
            return_sharded=sharded_moe_in,
            cfg=cfg,
        )
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
        if use_3d_x:
            expert_in_vmem = None
        elif (not interpret_mode) and sharded_moe_in:
            expert_in_vmem = next_in_vmem.at[:, pl.ds(0, 4096)]
        else:
            expert_in_vmem = next(it)
        moe_sh_vmem = next(it) if sharded_moe_in else None
        moe_rep_vmem = (
            next(it) if ((not sharded_moe_in) and use_3d_x and interpret_mode) else None
        )
        b2_scalar_dummy_post = next(it)
        b2_scalar_dummy_cols = next(it)
        sems = _unpack_sems(
            it,
            rs_pair_ssem=msc["rs_pair_ssem"],
            rs_chip_ssem=msc["rs_chip_ssem"],
            fan_send_sem=msc["fan_send_sem"],
        )
        if interpret_mode:
            wo_buf = next(it)
            w_pre_core_buf = w_pre_pair_buf
            w_post_buf, w_qkv_buf, w_gate_buf = next(it), next(it), next(it)
        else:
            wo_buf = _alias_wo_buf(
                msc["w1_buf"], msc["w2_buf"], w_mode=msc["w_mode"], o_in=o_in, half=half,
            )
            w_pre_core_buf = (
                w_pre_pair_buf
                if w_pre_pair_buf.shape[1] == pre_pair_cols
                else w_pre_pair_buf.at[:, pl.ds(0, pre_pair_cols)]
            )
            w_post_buf, w_qkv_buf, w_gate_buf = _alias_b2_weights(
                msc["w1_buf"], w_pre_pair_buf, w_mode=msc["w_mode"], qkv_cols=qkv_cols, gate_cols=gate_cols,
            )
        if sharded_moe_in:
            moe_dst_vmem = moe_sh_vmem
        elif use_3d_x:
            moe_dst_vmem = moe_rep_vmem if interpret_mode else next_in_vmem.at[:, pl.ds(0, 4096)]
        else:
            moe_dst_vmem = expert_in_vmem

        _run_stage1_4_b1_moe(
            msc=msc,
            w_pre_core_buf=w_pre_core_buf,
            wo_buf=wo_buf,
            w_router_buf=w_router_buf,
            s_buf=s_buf,
            send_half=send_half,
            mate_recv=mate_recv,
            own_half=own_half,
            b1_ephem=(
                rs1_a_or_z_recv, rs1_b_or_y_recv, rs2_a_or_x_recv, rs2_b_recv,
                sq_pair_pa_buf, router_pair_buf, pre_my_half, pre_mate_half,
                sq_pair_pre_buf, logits_vmem,
            ),
            my_pair_bf16=my_pair_bf16,
            recv_pair_bf16=recv_pair_bf16,
            ffn_pair_buf=ffn_pair_buf,
            exp_3d_buf=exp_3d_buf,
            moe_dst_vmem=moe_dst_vmem,
            expert_in_vmem=expert_in_vmem,
            w_post_buf=w_post_buf,
            w_qkv_buf=w_qkv_buf,
            w_gate_buf=w_gate_buf,
            sems=sems,
        )
        _run_stage5_6_b2_rpad(
            moe_dst_vmem=moe_dst_vmem,
            w_post_buf=w_post_buf,
            w_qkv_buf=w_qkv_buf,
            w_gate_buf=w_gate_buf,
            s_buf=s_buf,
            next_in_vmem=next_in_vmem,
            send_half=send_half,
            mate_recv=mate_recv,
            my_pair_bf16=my_pair_bf16,
            recv_pair_bf16=recv_pair_bf16,
            ffn_pair_buf=ffn_pair_buf,
            b2_scalar_dummy_post=b2_scalar_dummy_post,
            b2_scalar_dummy_cols=b2_scalar_dummy_cols,
            exp_3d_buf=exp_3d_buf,
            sems=sems,
            rpad_refs=None,
        )
        return

    if use_3scope:
        # 3-Scope Step (b) for rows > 128 and 3-Scope Step (c) for rows > 64:
        # w_pre_pair stays in Scope 1 (-16 MiB outer/Scope 2) and w_qkv+w_gate live in
        # Scope 3 (alongside RPAd when has_rpad=True), keeping peak VMEM < 63.94 MiB.
        moe_outer_refs = [next(it) for _ in range(step_b3_n_moe_outer)]
        s_buf = next(it)
        my_pair_bf16 = next(it)
        recv_pair_bf16 = next(it)
        ffn_pair_buf = next(it)
        exp_3d_buf = next(it)
        logits_vmem = next(it)
        moe_sh_vmem = next(it) if sharded_moe_in else None
        b2_scalar_dummy_post = next(it)
        b2_scalar_dummy_cols = next(it)

        o_it_b3 = iter(moe_outer_refs)
        stitched_outer_b3 = [
            None if flg else next(o_it_b3) for flg in step_b3_moe_is_scope2
        ]
        msc_outer_b3 = _unpack_muse_sparse_scratch(
            iter(stitched_outer_b3),
            ep=world_size,
            scale_block=scale_block,
            num_replicas=num_replicas,
            return_sharded=sharded_moe_in,
            cfg=cfg,
        )
        sems = _unpack_sems(
            it,
            rs_pair_ssem=msc_outer_b3["rs_pair_ssem"],
            rs_chip_ssem=msc_outer_b3["rs_chip_ssem"],
            fan_send_sem=msc_outer_b3["fan_send_sem"],
        )
        if interpret_mode:
            w_post_buf = next(it)
        else:
            if msc_outer_b3["w_mode"] == "stream3_k2":
                raise ValueError("b1_moe_b2 aliasing requires 4096-col w1_buf (not stream3_k2)")
            if msc_outer_b3["w1_buf"].shape != (2, 512, 4096):
                raise ValueError(
                    f"Expected w1_buf shape (2, 512, 4096), got {msc_outer_b3['w1_buf'].shape}"
                )
            w_post_buf = (
                msc_outer_b3["w1_buf"].bitcast(jnp.float8_e4m3fn).reshape((4096, 4096))
            )
        if has_rpad:
            written_cnt_smem_ref, rpad_bkv_sems, rpad_cache_write_sem = (
                _unpack_rpad_outer_meta(
                    it,
                    msc_outer_b3,
                    world_size=world_size,
                    rpa_cfg=rpa_cfg,
                    rpa_skip_kv_write=rpa_skip_kv_write,
                )
            )
        else:
            written_cnt_smem_ref = None
            rpad_bkv_sems = None
            rpad_cache_write_sem = None

        plan_holder_b3 = []

        def _run_b3_scope1(*s1_refs) -> None:
            s1_it = iter(s1_refs)
            w_pre_core_buf = next(s1_it)
            w_router_buf = next(s1_it)
            own_half = next(s1_it)
            send_half = next(s1_it)
            mate_recv = next(s1_it)
            rs1_a_s1 = next(s1_it)
            rs1_b_s1 = next(s1_it)
            rs2_a_s1 = next(s1_it)
            rs2_b_recv = next(s1_it)
            sq_pair_pa_buf = next(s1_it)
            router_pair_buf = next(s1_it)
            pre_my_half = next(s1_it)
            pre_mate_half = next(s1_it)
            sq_pair_pre_buf = next(s1_it)
            wo_buf = (
                next(s1_it)
                if interpret_mode
                else _alias_wo_buf(
                    msc_outer_b3["w1_buf"],
                    msc_outer_b3["w2_buf"],
                    w_mode=msc_outer_b3["w_mode"],
                    o_in=o_in,
                    half=half,
                )
            )

            def _during_ag_exp_ici_b3() -> None:
                plan_res = plan_and_prime_muse_sparse_from_vmem(
                    me,
                    logits_vmem,
                    bias_ref,
                    post_exp_norm_ref,
                    w1_u32,
                    s1_ref,
                    w2_u32,
                    s2_ref,
                    d_scratch=msc_outer_b3["d_scratch"],
                    r_scratch=msc_outer_b3["r_scratch"],
                    w1_buf=msc_outer_b3["w1_buf"],
                    s1_buf=msc_outer_b3["s1_buf"],
                    w2_buf=msc_outer_b3["w2_buf"],
                    s2_buf=msc_outer_b3["s2_buf"],
                    w1_sem=msc_outer_b3["w1_sem"],
                    w2_sem=msc_outer_b3["w2_sem"],
                    misc_sem=msc_outer_b3["misc_sem"],
                    gain_f32_buf=msc_outer_b3["gain_f32_buf"],
                    accum_3d=None,
                    io_3d_buf=None,
                    accum_2d=None,
                    rep_w1_u32=rep_w1_u32,
                    rep_s1_ref=rep_s1_ref,
                    rep_w2_u32=rep_w2_u32,
                    rep_s2_ref=rep_s2_ref,
                    replica_meta_ref=replica_meta_ref,
                    owner_load_smem=msc_outer_b3["owner_load_smem"],
                    moved_flag_smem=msc_outer_b3["moved_flag_smem"],
                    local_suppress_smem=msc_outer_b3["local_suppress_smem"],
                    holder_slot_expert_smem=msc_outer_b3["holder_slot_expert_smem"],
                    holder_cand_smem=msc_outer_b3["holder_cand_smem"],
                    spec_slot_ref=spec_slot_ref,
                    free_params_ref=free_params_ref,
                    ep=world_size,
                    batch_size=rows,
                    scale_block=scale_block,
                    num_replicas=num_replicas,
                    replica_cost_mode=replica_cost_mode,
                    cfg=cfg,
                    zero_and_pack=False,
                )
                plan_holder_b3.append(plan_res)

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
                None,
                wo_buf,
                w_pre_core_buf,
                w_router_buf,
                s_buf,
                send_half,
                mate_recv,
                own_half,
                rs1_a_s1,
                rs1_b_s1,
                rs2_a_s1,
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
                msc_outer_b3["w1_sem"],
                sems["w_pre_sem"],
                sems["w_router_sem"],
                sems["s_in_sem"],
                sems["s_out_sem"],
                sems["c1a_send"],
                sems["c1a_recv"],
                sems["rs1_send"],
                sems["rs1_recv"],
                sems["rs2_send"],
                sems["rs2_recv"],
                sems["rs3_send"],
                sems["rs3_recv"],
                sems["c1c_send"],
                sems["c1c_recv"],
                sems["pre_send"],
                sems["pre_recv"],
                sems["ag_exp_send"],
                sems["ag_exp_recv"],
                sems["ag_exp_d2d_send"],
                sems["ag_exp_d2d_recv"],
                sems["ag_log_send"],
                sems["ag_log_recv"],
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
                during_ag_exp_ici_fn=_during_ag_exp_ici_b3,
                during_ag_exp_d2d2_fn=None,
                out_expert_3d_ref=msc_outer_b3["x_3d_buf"],
            )
            p_b3 = plan_holder_b3[0]
            return (
                p_b3.num_exec,
                jnp.int32(0) if p_b3.spec_slot is None else p_b3.spec_slot,
                p_b3.spec_enabled,
            )

        b3_num_exec, b3_spec_slot, b3_spec_en = pl.run_scoped(
            _run_b3_scope1, *step_b3_scope1_shapes
        )

        def _on_experts_done_b3() -> None:
            pltpu.make_async_copy(w_post_hbm, w_post_buf, sems["w_post_sem"].at[0]).start()

        def _run_b3_scope2(*s2_refs) -> None:
            o_it_s2 = iter(moe_outer_refs)
            s2_it = iter(s2_refs)
            stitched_s2 = [next(s2_it) if flg else next(o_it_s2) for flg in step_b3_moe_is_scope2]
            msc_s2 = _unpack_muse_sparse_scratch(
                iter(stitched_s2),
                ep=world_size,
                scale_block=scale_block,
                num_replicas=num_replicas,
                return_sharded=sharded_moe_in,
                cfg=cfg,
            )
            p_tmpl = plan_holder_b3[0]
            plan_s2 = dataclasses.replace(
                p_tmpl,
                num_exec=b3_num_exec,
                spec_slot=None if p_tmpl.spec_slot is None else b3_spec_slot,
                spec_enabled=b3_spec_en,
            )
            _signal_barrier(me, world_size)
            zero_and_pack_muse_sparse_from_vmem(
                plan_s2,
                r_scratch=msc_s2["r_scratch"],
                accum_3d=msc_s2["accum_3d"],
                io_3d_buf=msc_s2["io_3d_buf"],
                accum_2d=msc_s2["accum_2d"],
                batch_size=rows,
                cfg=cfg,
            )
            _wait_barrier(world_size)
            run_muse_sparse_experts_and_rs_from_vmem(
                me,
                None,
                plan_s2,
                d_scratch=msc_s2["d_scratch"],
                r_scratch=msc_s2["r_scratch"],
                w1_buf=msc_s2["w1_buf"],
                s1_buf=msc_s2["s1_buf"],
                w2_buf=msc_s2["w2_buf"],
                s2_buf=msc_s2["s2_buf"],
                w1_acc_buf=msc_s2["w1_acc_buf"],
                x_3d_buf=msc_s2["x_3d_buf"],
                io_3d_buf=msc_s2["io_3d_buf"],
                lhs_2d_buf=msc_s2["lhs_2d_buf"],
                act_2d_buf=msc_s2["act_2d_buf"],
                y_stage_buf=msc_s2["y_stage_buf"],
                lhs_fp8_buf=msc_s2["lhs_fp8_buf"],
                lhs_sc_buf=msc_s2["lhs_sc_buf"],
                masked_x_buf=msc_s2["masked_x_buf"],
                bd_sc_buf=msc_s2["bd_sc_buf"],
                gain_f32_buf=msc_s2["gain_f32_buf"],
                accum_3d=msc_s2["accum_3d"],
                accum_2d=msc_s2["accum_2d"],
                fan_3d_buf=msc_s2["fan_3d_buf"],
                rs_send_bf16=msc_s2["rs_send_bf16"],
                rs_pair_recv=msc_s2["rs_pair_recv"],
                rs_chip_send=msc_s2["rs_chip_send"],
                rs_chip_recv=msc_s2["rs_chip_recv"],
                rs_pair_ssem=msc_s2["rs_pair_ssem"],
                rs_pair_rsem=msc_s2["rs_pair_rsem"],
                rs_chip_ssem=msc_s2["rs_chip_ssem"],
                rs_chip_rsem=msc_s2["rs_chip_rsem"],
                rs_recv_3d=msc_s2["rs_recv_3d"],
                rs_send_sem=msc_s2["rs_send_sem"],
                rs_recv_sem=msc_s2["rs_recv_sem"],
                owner_3d=msc_s2["owner_3d"],
                fan_send_sem=msc_s2["fan_send_sem"],
                fan_recv_sem=msc_s2["fan_recv_sem"],
                fan_s2_send_sem=msc_s2["fan_s2_send_sem"],
                fan_s2_recv_sem=msc_s2["fan_s2_recv_sem"],
                barrier_sem=None,
                output_ref=moe_sh_vmem,
                ep=world_size,
                batch_size=rows,
                scale_block=scale_block,
                return_sharded=sharded_moe_in,
                ep_axis_name=ep_axis_name,
                cfg=cfg,
                x_is_3d=True,
                wait_entry_barrier=False,
                write_output_ref=True,
                on_experts_done_fn=_on_experts_done_b3,
            )

        pl.run_scoped(_run_b3_scope2, *step_b3_scope2_shapes)

        def _run_b3_scope3(*s3_refs) -> None:
            next_in_vmem, mate_recv_b2, mate_tok_b2, w_qkv_buf, w_gate_buf = s3_refs[:5]
            rpad_refs_b3 = (
                _unpack_rpad_scope_refs(
                    iter(s3_refs[5:]),
                    rpa_skip_kv_write=rpa_skip_kv_write,
                    rpa_kv_pack_dense2=rpa_kv_pack_dense2,
                    rpa_kv_pack_bd4=rpa_kv_pack_bd4,
                    rpa_cfg=rpa_cfg,
                    rpa_use_grp_vec=rpa_use_grp_vec,
                )
                if has_rpad
                else None
            )
            _run_stage5_6_b2_rpad(
                moe_dst_vmem=moe_sh_vmem,
                w_post_buf=w_post_buf,
                w_qkv_buf=w_qkv_buf,
                w_gate_buf=w_gate_buf,
                s_buf=s_buf,
                next_in_vmem=next_in_vmem,
                send_half=mate_tok_b2,
                mate_recv=mate_recv_b2,
                my_pair_bf16=my_pair_bf16,
                recv_pair_bf16=recv_pair_bf16,
                ffn_pair_buf=ffn_pair_buf,
                b2_scalar_dummy_post=b2_scalar_dummy_post,
                b2_scalar_dummy_cols=b2_scalar_dummy_cols,
                exp_3d_buf=exp_3d_buf,
                sems=sems,
                rpad_refs=rpad_refs_b3,
                written_cnt_smem_ref=written_cnt_smem_ref,
                rpad_bkv_sems=rpad_bkv_sems,
                rpad_cache_write_sem=rpad_cache_write_sem,
                sync_entry_barrier=True,
                prestarted_qkv_gate_dma=False,
            )

        scope3_shapes = step_b3_scope3_shapes + (
            tuple(step_c_scope2_shapes) if has_rpad else ()
        )
        pl.run_scoped(_run_b3_scope3, *scope3_shapes)
        return

    # 2-Scope Step (c): Outer (live + sems) -> Scope 1 (Stage 1..4) -> Scope 2 (b2 + RPAd)
    moe_outer_refs = [next(it) for _ in range(step_c_n_moe_outer)]
    w_pre_pair_buf = next(it)
    s_buf = next(it)
    next_in_vmem = next(it)
    mate_recv = next(it)
    if interpret_mode:
        send_half = next(it)
        own_half = next(it)
    else:
        own_half = next_in_vmem.at[:, pl.ds(0, 4096)]
        send_half = next_in_vmem.at[:, pl.ds(4096, 4096)]
    my_pair_bf16 = next(it)
    recv_pair_bf16 = next(it)
    ffn_pair_buf = next(it)
    exp_3d_buf = next(it)
    moe_sh_vmem = next(it) if sharded_moe_in else None
    moe_rep_vmem = (
        next(it) if ((not sharded_moe_in) and use_3d_x and interpret_mode) else None
    )
    b2_scalar_dummy_post = next(it)
    b2_scalar_dummy_cols = next(it)

    o_it_init = iter(moe_outer_refs)
    stitched_outer_only = [
        None if flg else next(o_it_init) for flg in step_c_moe_is_scope1
    ]
    msc_outer = _unpack_muse_sparse_scratch(
        iter(stitched_outer_only),
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=sharded_moe_in,
        cfg=cfg,
    )
    sems = _unpack_sems(
        it,
        rs_pair_ssem=msc_outer["rs_pair_ssem"],
        rs_chip_ssem=msc_outer["rs_chip_ssem"],
        fan_send_sem=msc_outer["fan_send_sem"],
    )
    if interpret_mode:
        w_post_outer, w_qkv_outer, w_gate_outer = next(it), next(it), next(it)
        w_pre_core_buf = w_pre_pair_buf
    else:
        w_pre_core_buf = (
            w_pre_pair_buf
            if w_pre_pair_buf.shape[1] == pre_pair_cols
            else w_pre_pair_buf.at[:, pl.ds(0, pre_pair_cols)]
        )
        w_mode_eff = (
            "stream3"
            if (cfg.weight_mode == "stream3_k2" and (_H // scale_block) < 16)
            else cfg.weight_mode
        )
        w_post_outer, w_qkv_outer, w_gate_outer = _alias_b2_weights(
            msc_outer["w1_buf"],
            w_pre_pair_buf,
            w_mode=w_mode_eff,
            qkv_cols=qkv_cols,
            gate_cols=gate_cols,
        )

    written_cnt_smem_ref, rpad_bkv_sems, rpad_cache_write_sem = (
        _unpack_rpad_outer_meta(
            it,
            msc_outer,
            world_size=world_size,
            rpa_cfg=rpa_cfg,
            rpa_skip_kv_write=rpa_skip_kv_write,
        )
    )

    if sharded_moe_in:
        moe_dst_vmem = moe_sh_vmem
    elif use_3d_x:
        moe_dst_vmem = moe_rep_vmem if interpret_mode else next_in_vmem.at[:, pl.ds(0, 4096)]
    else:
        moe_dst_vmem = None

    def _run_scope1(*s1_refs) -> None:
        moe_s1 = list(s1_refs[:step_c_n_moe_scope1])
        b1_s1 = s1_refs[step_c_n_moe_scope1 : step_c_n_moe_scope1 + step_c_n_b1_scope1]
        b1_i_s1 = s1_refs[step_c_n_moe_scope1 + step_c_n_b1_scope1 :]

        o_it = iter(moe_outer_refs)
        s1_it = iter(moe_s1)
        stitched_moe = [next(s1_it) if flg else next(o_it) for flg in step_c_moe_is_scope1]
        msc = _unpack_muse_sparse_scratch(
            iter(stitched_moe),
            ep=world_size,
            scale_block=scale_block,
            num_replicas=num_replicas,
            return_sharded=sharded_moe_in,
            cfg=cfg,
        )
        w_router_buf = b1_s1[0]
        b1_ephem = b1_s1[1:11]
        exp_in_s1 = b1_s1[11] if need_b1_exp_in else None
        if use_3d_x:
            expert_in_vmem = None
        elif (not interpret_mode) and sharded_moe_in:
            expert_in_vmem = next_in_vmem.at[:, pl.ds(0, 4096)]
        else:
            expert_in_vmem = exp_in_s1
        wo_buf = (
            b1_i_s1[0]
            if interpret_mode
            else _alias_wo_buf(
                msc["w1_buf"], msc["w2_buf"], w_mode=msc["w_mode"], o_in=o_in, half=half,
            )
        )
        _run_stage1_4_b1_moe(
            msc=msc,
            w_pre_core_buf=w_pre_core_buf,
            wo_buf=wo_buf,
            w_router_buf=w_router_buf,
            s_buf=s_buf,
            send_half=send_half,
            mate_recv=mate_recv,
            own_half=own_half,
            b1_ephem=b1_ephem,
            my_pair_bf16=my_pair_bf16,
            recv_pair_bf16=recv_pair_bf16,
            ffn_pair_buf=ffn_pair_buf,
            exp_3d_buf=exp_3d_buf,
            moe_dst_vmem=(moe_dst_vmem if moe_dst_vmem is not None else expert_in_vmem),
            expert_in_vmem=expert_in_vmem,
            w_post_buf=w_post_outer,
            w_qkv_buf=w_qkv_outer,
            w_gate_buf=w_gate_outer,
            sems=sems,
        )

    pl.run_scoped(_run_scope1, *step_c_scope1_shapes)

    def _run_scope2(*s2_refs) -> None:
        rpad_refs_s2 = _unpack_rpad_scope_refs(
            iter(s2_refs),
            rpa_skip_kv_write=rpa_skip_kv_write,
            rpa_kv_pack_dense2=rpa_kv_pack_dense2,
            rpa_kv_pack_bd4=rpa_kv_pack_bd4,
            rpa_cfg=rpa_cfg,
            rpa_use_grp_vec=rpa_use_grp_vec,
        )
        _run_stage5_6_b2_rpad(
            moe_dst_vmem=moe_dst_vmem,
            w_post_buf=w_post_outer,
            w_qkv_buf=w_qkv_outer,
            w_gate_buf=w_gate_outer,
            s_buf=s_buf,
            next_in_vmem=next_in_vmem,
            send_half=send_half,
            mate_recv=mate_recv,
            my_pair_bf16=my_pair_bf16,
            recv_pair_bf16=recv_pair_bf16,
            ffn_pair_buf=ffn_pair_buf,
            b2_scalar_dummy_post=b2_scalar_dummy_post,
            b2_scalar_dummy_cols=b2_scalar_dummy_cols,
            exp_3d_buf=exp_3d_buf,
            sems=sems,
            rpad_refs=rpad_refs_s2,
            written_cnt_smem_ref=written_cnt_smem_ref,
            rpad_bkv_sems=rpad_bkv_sems,
            rpad_cache_write_sem=rpad_cache_write_sem,
        )

    pl.run_scoped(_run_scope2, *step_c_scope2_shapes)


def _b1_moe_b2_chip_sharded_kernel(*all_refs, **kwargs) -> None:
    """Fused Step (b) ``[chip_b1 + muse_sparse(fb0) + chip_b2]`` Pallas kernel body."""
    _b1_moe_b2_chip_sharded_impl(all_refs, has_rpad=False, **kwargs)


def _b1_moe_b2_rpad_chip_sharded_kernel(*all_refs, **kwargs) -> None:
    """Fused Step (c) ``[chip_b1 + muse_sparse(3d) + chip_b2 + RPAd]`` Pallas kernel body."""
    _b1_moe_b2_chip_sharded_impl(all_refs, has_rpad=True, **kwargs)


def _prepare_chip_b1_moe_b2_setup(
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
    cos_sin_step: jax.Array | None,
    rep_w1: jax.Array | None,
    rep_s1: jax.Array | None,
    rep_w2: jax.Array | None,
    rep_s2: jax.Array | None,
    replica_meta: jax.Array | None,
    spec_home_slot: jax.Array | None,
    free_params: jax.Array | None,
    *,
    world_size: int,
    fuse_qkv_epi: bool,
    config: SparseDecodeConfig | None,
    interpret,
    has_rpad: bool = False,
) -> dict:
    """Validate shapes and construct shared specs/scratches for Step (b) and Step (c)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2 superkernel requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
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

    valid_rows = (32, 64, 128) if world_size == 8 else (64, 128, 256)
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
            "TP=16 b1_moe_b2 superkernel requires config.pair_rs=True to stay <= 64 DMA semaphores"
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
    use_3d_x = config.row_mode != "dense"
    use_3scope = (rows > 128) or (has_rpad and rows > 64)
    if use_3scope and not sharded_moe_in:
        raise ValueError(
            f"3-scope decode superkernel requires sharded_moe_in=True, got B={rows}, world_size={world_size}"
        )
    w_pre_cols_buf = pre_pair_cols if interpret_mode else max(pre_pair_cols, qkv_cols + gate_cols)
    alias_mate_recv = (
        (not has_rpad)
        and (not use_3scope)
        and (not interpret_mode)
        and config.row_mode != "dense"
        and config.norm_f32_stage
        and (not config.loop32_in_tile)
        and (not (scale_block >= 256 and config.lhs_fp8))
        and (2 * config.tile_m >= rows)
    )
    v_w_pre_pair_buf = vmem((hidden, w_pre_cols_buf), w_pre_pair_w8k.dtype)
    v_w_router_buf = vmem((hidden, n_exp), w_router_km.dtype)
    v_s_buf = vmem((r_pair, hidden), jnp.float32)
    v_next_in_vmem = vmem((rows, hidden), jnp.bfloat16)
    v_mate_recv = [] if alias_mate_recv else [vmem((rows, half), jnp.bfloat16)]
    v_interp_halves = (
        [vmem((rows, half), jnp.bfloat16), vmem((rows, half), jnp.bfloat16)]
        if interpret_mode
        else []
    )
    v_rs1_a = vmem(
        (two_pair, q_cols) if use_xy else ((four_pair, half) if world_size == 16 else (8, 128)),
        jnp.bfloat16,
    )
    v_rs1_b = vmem((two_pair, q_cols) if use_xy else (two_pair, half), jnp.bfloat16)
    v_rs2_a = vmem((r_pair, q_cols) if use_xy else (r_pair, half), jnp.bfloat16)
    v_rs_slice_shapes = [v_rs1_a, v_rs1_b, v_rs2_a]
    v_rs2_b = vmem(
        (r_pair, q_cols) if use_xy else ((r_pair, half) if r_pair == 8 else (8, 128)),
        jnp.bfloat16,
    )
    v_my_pair_bf16 = vmem((r_pair, half), jnp.bfloat16)
    v_recv_pair_bf16 = vmem((r_pair, half), jnp.bfloat16)
    v_sq_pair_pa = vmem((r_pair, 128), jnp.float32)
    v_ffn_pair = vmem((r_pair, hidden), jnp.bfloat16)
    v_router_pair = vmem((8, 128) if is_r16 else (r_pair, hidden), jnp.float32)
    v_pre_my = vmem((r_pair, pre_pair_cols), jnp.bfloat16)
    v_pre_mate = (
        vmem((r_pair, pre_pair_cols), jnp.bfloat16)
        if r_pair == 8
        else vmem((r_pair, 128), jnp.float32)
    )
    v_sq_pair_pre = vmem((r_pair, 128), jnp.float32)
    v_exp_3d = vmem((n_chips, 8, h_pre) if r_pair == 8 else (8, 128), jnp.bfloat16)
    v_logits = vmem((rows, n_exp), jnp.float32)
    v_exp_in = (
        [vmem((rows, h_pre), jnp.bfloat16)]
        if ((not use_3d_x) and (interpret_mode or not sharded_moe_in))
        else []
    )
    v_moe_sh = [vmem((rows_per_rank, h_pre), jnp.bfloat16)] if sharded_moe_in else []
    v_moe_rep = (
        [vmem((rows, h_pre), jnp.bfloat16)]
        if ((not sharded_moe_in) and use_3d_x and interpret_mode)
        else []
    )
    v_b2_dummy_post = vmem((8, 128), jnp.bfloat16)
    v_b2_dummy_cols = vmem((8, 128), jnp.bfloat16)

    outer_sems = [
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
    v_interp_wo = [vmem((2, o_in, half), wo_w8k.dtype)] if interpret_mode else []
    v_interp_b2 = (
        [
            vmem((_H, post_cols), w_post_tp2_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
        ]
        if interpret_mode
        else []
    )
    shared_scratch = [
        v_w_pre_pair_buf,
        v_w_router_buf,
        v_s_buf,
        v_next_in_vmem,
        *v_mate_recv,
        *v_interp_halves,
        *v_rs_slice_shapes,
        v_rs2_b,
        v_my_pair_bf16,
        v_recv_pair_bf16,
        v_sq_pair_pa,
        v_ffn_pair,
        v_router_pair,
        v_pre_my,
        v_pre_mate,
        v_sq_pair_pre,
        v_exp_3d,
        v_logits,
        *v_exp_in,
        *v_moe_sh,
        *v_moe_rep,
        v_b2_dummy_post,
        v_b2_dummy_cols,
        *outer_sems,
        *v_interp_wo,
        *v_interp_b2,
    ]

    b3_idx_map = _unpack_muse_sparse_scratch(
        iter(range(len(moe_scratch))),
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=sharded_moe_in,
        cfg=config,
    )
    b3_n_route_sc = 6 if config.row_mode == "dense" else 12
    b3_keep_outer = {
        *range(b3_n_route_sc),
        b3_idx_map["w1_buf"],
        b3_idx_map["s1_buf"],
        b3_idx_map["w2_buf"],
        b3_idx_map["s2_buf"],
        b3_idx_map["gain_f32_buf"],
        b3_idx_map["x_3d_buf"],
    }
    step_b3_moe_is_scope2 = tuple(
        (idx not in b3_keep_outer) and (getattr(sp, "memory_space", None) == vmem)
        for idx, sp in enumerate(moe_scratch)
    )
    step_b3_moe_outer = [
        sp for sp, flg in zip(moe_scratch, step_b3_moe_is_scope2) if not flg
    ]
    step_b3_moe_scope2 = tuple(
        sp for sp, flg in zip(moe_scratch, step_b3_moe_is_scope2) if flg
    )
    v_interp_b3_post = (
        [vmem((_H, post_cols), w_post_tp2_w8k.dtype)] if interpret_mode else []
    )
    step_b3_outer_scratch = [
        *step_b3_moe_outer,
        v_s_buf,
        v_my_pair_bf16,
        v_recv_pair_bf16,
        v_ffn_pair,
        v_exp_3d,
        v_logits,
        *v_moe_sh,
        v_b2_dummy_post,
        v_b2_dummy_cols,
        *outer_sems,
        *v_interp_b3_post,
    ]
    step_b3_scope1_shapes = tuple(
        [
            vmem((hidden, pre_pair_cols), w_pre_pair_w8k.dtype),
            v_w_router_buf,
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            *v_rs_slice_shapes,
            v_rs2_b,
            v_sq_pair_pa,
            v_router_pair,
            v_pre_my,
            v_pre_mate,
            v_sq_pair_pre,
            *v_interp_wo,
        ]
    )
    step_b3_scope3_shapes = (
        vmem((rows, hidden), jnp.bfloat16),
        vmem((rows, half), jnp.bfloat16),
        vmem((rows_per_rank, _H), jnp.bfloat16),
        vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
        vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
    )

    is_dense_cfg = config.row_mode == "dense"
    n_route_sc = 6 if is_dense_cfg else 12
    n_rep_sc = (
        0
        if num_replicas == 0
        else (4 if (is_dense_cfg or config.static_holder_mask) else 5)
    )
    w1_idx = n_route_sc + n_rep_sc
    moe_is_scope1 = tuple(
        (idx != w1_idx) and (getattr(sp, "memory_space", None) == vmem)
        for idx, sp in enumerate(moe_scratch)
    )
    moe_outer = [sp for sp, flg in zip(moe_scratch, moe_is_scope1) if not flg]
    moe_scope1 = [sp for sp, flg in zip(moe_scratch, moe_is_scope1) if flg]
    outer_shared_vmem = [
        v_w_pre_pair_buf,
        v_s_buf,
        v_next_in_vmem,
        *v_mate_recv,
        *v_interp_halves,
        v_my_pair_bf16,
        v_recv_pair_bf16,
        v_ffn_pair,
        v_exp_3d,
        *v_moe_sh,
        *v_moe_rep,
        v_b2_dummy_post,
        v_b2_dummy_cols,
    ]
    scope1_b1_vmem = [
        v_w_router_buf,
        v_rs1_a,
        v_rs1_b,
        v_rs2_a,
        v_rs2_b,
        v_sq_pair_pa,
        v_router_pair,
        v_pre_my,
        v_pre_mate,
        v_sq_pair_pre,
        v_logits,
        *v_exp_in,
    ]

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = post_exp_norm.reshape(1, _H)
    has_spec_slot_arg = spec_home_slot is not None
    has_free_params_arg = free_params is not None
    scalar_args = [jnp.atleast_1d(rank).astype(jnp.int32)]
    if has_replicas:
        scalar_args.append(replica_meta)
    if has_spec_slot_arg:
        scalar_args.append(jnp.atleast_1d(spec_home_slot).astype(jnp.int32))
    if has_free_params_arg:
        fp_arg = free_params.reshape(-1).astype(jnp.float32)
        if fp_arg.shape != (4,):
            raise ValueError(f"Expected free_params shape (4,), got {free_params.shape}")
        scalar_args.append(fp_arg)

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

    return dict(
        rows=rows,
        r_pair=r_pair,
        hidden=hidden,
        qkv_cols=qkv_cols,
        gate_cols=gate_cols,
        scale_block=scale_block,
        num_replicas=num_replicas,
        sharded_moe_in=sharded_moe_in,
        is_r16=is_r16,
        has_rope=has_rope,
        config=config,
        interpret_arg=interpret_arg,
        interpret_mode=interpret_mode,
        alias_mate_recv=alias_mate_recv,
        has_spec_slot_arg=has_spec_slot_arg,
        has_free_params_arg=has_free_params_arg,
        v_sp=v_sp,
        h_sp=h_sp,
        vmem=vmem,
        hbm=hbm,
        sem=sem,
        scalar_args=scalar_args,
        in_specs=in_specs,
        base_call_tensors=(*b1_args, *moe_args, *b2_args),
        scratch_shapes=moe_scratch + shared_scratch,
        use_3scope=use_3scope,
        step_b3_outer_scratch=step_b3_outer_scratch,
        step_b3_moe_is_scope2=step_b3_moe_is_scope2,
        step_b3_n_moe_outer=len(step_b3_moe_outer),
        step_b3_scope1_shapes=step_b3_scope1_shapes,
        step_b3_scope2_shapes=step_b3_moe_scope2,
        step_b3_scope3_shapes=step_b3_scope3_shapes,
        step_c_outer_scratch=moe_outer + outer_shared_vmem + outer_sems + v_interp_b2,
        step_c_moe_is_scope1=moe_is_scope1,
        step_c_n_moe_outer=len(moe_outer),
        step_c_n_moe_scope1=len(moe_scope1),
        step_c_n_b1_scope1=len(scope1_b1_vmem),
        step_c_scope1_shapes=tuple(moe_scope1 + scope1_b1_vmem + v_interp_wo),
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
    free_params: jax.Array | None = None,
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

    st = _prepare_chip_b1_moe_b2_setup(
        rank, stream_chip, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
        pre_ff_gain, w_pre_pair_w8k, w_pre_pair_scale, w_router_km, pre_exp_gain,
        bias, w1, s1, w2, s2, post_exp_norm,
        w_post_tp2_w8k, s_post_tp2, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
        cos_sin_step, rep_w1, rep_s1, rep_w2, rep_s2, replica_meta,
        spec_home_slot, free_params,
        world_size=world_size, fuse_qkv_epi=fuse_qkv_epi, config=config, interpret=interpret,
    )
    rows = st["rows"]
    r_pair = st["r_pair"]
    hidden = st["hidden"]
    qkv_cols = st["qkv_cols"]
    gate_cols = st["gate_cols"]
    v_sp, h_sp = st["v_sp"], st["h_sp"]
    scalar_args = st["scalar_args"]
    call_args = (*scalar_args, *st["base_call_tensors"])

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
        scale_block=st["scale_block"],
        num_replicas=st["num_replicas"],
        sharded_moe_in=bool(st["sharded_moe_in"]),
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
        has_rope=st["has_rope"],
        cfg=st["config"],
        has_spec_slot_arg=st["has_spec_slot_arg"],
        has_free_params_arg=st["has_free_params_arg"],
        interpret_mode=st["interpret_mode"],
        alias_mate_recv=st["alias_mate_recv"],
        use_3scope=st["use_3scope"],
        step_b3_n_moe_outer=st["step_b3_n_moe_outer"],
        step_b3_moe_is_scope2=st["step_b3_moe_is_scope2"],
        step_b3_scope1_shapes=st["step_b3_scope1_shapes"],
        step_b3_scope2_shapes=st["step_b3_scope2_shapes"],
        step_b3_scope3_shapes=st["step_b3_scope3_shapes"],
    )

    res = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            grid=(1,),
            in_specs=st["in_specs"],
            out_specs=out_specs,
            scratch_shapes=(
                st["step_b3_outer_scratch"] if st["use_3scope"] else st["scratch_shapes"]
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(63 if rows > 64 else 60) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=st["interpret_arg"],
        name=(
            f"muse_w9_b1_moe_b2_b{rows}_tp{world_size}_k{st['scale_block']}"
            f"_r{st['num_replicas']}_{'r16' if st['is_r16'] else 'r32'}"
            f"{'_dfp' if st['has_free_params_arg'] else ''}"
            f"{'_epi_' + ('rope' if st['has_rope'] else 'nope') if fuse_qkv_epi else ''}"
        ),
    )(*call_args)
    if fuse_qkv_epi:
        st_ch, q_3d, kv_m, gate = res
        return st_ch, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate
    return res


_STEP_C_BASE_MAX_BKV_ROWS = 80 * 512
_STEP_C_GROUP_MAX_BKV_ROWS = 80 * 512
_STEP_C_LARGE_B_MAX_BKV_ROWS = 16 * 512


def resolve_step_c_rpa_plan(
    rpa_mode_str: str,
    *,
    sliding_window: int | None,
    pages_per_seq: int,
    phys_page_rows: int,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    b_tokens: int | None = None,
    max_num_seqs: int | None = None,
    shared_group_size: int = 0,
    has_shared_prefix: bool = False,
    allow_large_b: bool | None = None,
):
    """Resolve Step (c) RPAd config, ``one_block_per_seq``, ``bkv_slot_pages``, and ``use_grp_vec``."""
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
        _uses_swabatch_group,
        get_shared_group_size,
        parse_decode_fast_config,
    )

    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_rows * 4
        if kv_pack_bd4
        else (phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows)
    )
    eff_b = int(b_tokens) if b_tokens is not None else 64
    eff_seqs = int(max_num_seqs) if max_num_seqs is not None else eff_b
    max_bkv_rows = _STEP_C_LARGE_B_MAX_BKV_ROWS if eff_b > 64 else _STEP_C_BASE_MAX_BKV_ROWS
    if allow_large_b is None:
        try:
            from sglang_tpu.envs import envs as tpu_envs

            allow_large_b = bool(tpu_envs.SGLANG_TPU_MUSE_MK_STEP_C_LARGE_B.get())
        except Exception:
            allow_large_b = False

    if shared_group_size > 0:
        eff_shared_grp = int(shared_group_size)
    elif has_shared_prefix:
        eff_shared_grp = get_shared_group_size(rpa_mode_str)
    else:
        eff_shared_grp = 0

    cfg_rpa = parse_decode_fast_config(
        rpa_mode_str,
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        decode_query_size=1,
        shared_group=eff_shared_grp,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
    )
    one_block_per_seq = (
        False
        if cfg_rpa.loop_kind == "ring4k"
        else (True if sliding_window is not None else (pages_per_seq <= cfg_rpa.max_ring_pages))
    )
    if eff_b > 64 and kv_pack_dense2 and not allow_large_b:
        return cfg_rpa, one_block_per_seq, cfg_rpa.max_ring_pages, False, False
    sh_grp = cfg_rpa.shared_group_size or (16 if cfg_rpa.loop_kind == "shared16" else 0)
    shared_bad = (
        (has_shared_prefix and shared_group_size <= 0)
        or (sh_grp > 0 and (shared_group_size <= 0 or eff_b < eff_seqs or (eff_seqs % sh_grp) != 0))
        or (sh_grp == 0 and cfg_rpa.loop_kind.startswith("shared"))
    )
    if (
        (not shared_bad)
        and sliding_window is None
        and sh_grp == 0
        and cfg_rpa.loop_kind in ("w7auto", "ring4k")
        and cfg_rpa.num_slots * cfg_rpa.max_ring_pages * phys_page_rows > max_bkv_rows
    ):
        cand_rings = ((3, 4096),)
        for c_slots, c_chunk in cand_rings:
            if c_chunk < page_size or (c_chunk % page_size) != 0:
                continue
            c_ring_pgs = max(1, c_chunk // page_size)
            c_blks_4k = (max(pages_per_seq, 1) + c_ring_pgs - 1) // c_ring_pgs
            c_smem_bytes = (
                eff_seqs * pages_per_seq + 2 + c_slots + 4 * eff_seqs * c_blks_4k
            ) * 4
            if (
                c_slots * c_ring_pgs * phys_page_rows <= max_bkv_rows
                and c_smem_bytes <= 960 * 1024
            ):
                cfg_rpa = dataclasses.replace(
                    cfg_rpa,
                    loop_kind="ring4k",
                    num_slots=c_slots,
                    chunk_sz=c_chunk,
                    max_ring_pages=c_ring_pgs,
                    lazy_slot_zero=True,
                    coalesce_dma=True,
                    swabatch_group_size=1,
                )
                one_block_per_seq = False
                break
    ring_smem_over = False
    if cfg_rpa.loop_kind == "ring4k" and sh_grp == 0:
        blks_4k = (max(pages_per_seq, 1) + max(cfg_rpa.max_ring_pages, 1) - 1) // max(
            cfg_rpa.max_ring_pages, 1
        )
        ring_smem_over = (
            eff_seqs * pages_per_seq + 2 + cfg_rpa.num_slots + 4 * eff_seqs * blks_4k
        ) * 4 > 960 * 1024
    if (
        shared_bad
        or ring_smem_over
        or cfg_rpa.num_slots * cfg_rpa.max_ring_pages * phys_page_rows > max_bkv_rows
    ):
        return cfg_rpa, one_block_per_seq, cfg_rpa.max_ring_pages, False, False

    eff_grp = 1
    use_grp_vec = False
    if cfg_rpa.swabatch_group_size > 1 and eff_b == eff_seqs and sh_grp == 0:
        grp_limit = _STEP_C_LARGE_B_MAX_BKV_ROWS if eff_b > 64 else _STEP_C_GROUP_MAX_BKV_ROWS
        for cand_grp in (4, 2):
            if cand_grp > cfg_rpa.swabatch_group_size or (eff_b % cand_grp) != 0:
                continue
            cand_cfg = dataclasses.replace(cfg_rpa, swabatch_group_size=cand_grp)
            if (
                _uses_swabatch_group(
                    cand_cfg,
                    one_block_per_seq=one_block_per_seq,
                    decode_query_size=1,
                    kv_pack_dense2=kv_pack_dense2,
                    probe=None,
                    b_tokens=eff_b,
                    max_num_seqs=eff_seqs,
                )
                and cand_cfg.num_slots * cand_grp * cand_cfg.max_ring_pages * phys_page_rows
                <= grp_limit
            ):
                cfg_rpa = cand_cfg
                eff_grp = cand_grp
                use_grp_vec = True
                break
        if eff_b > 64 and not use_grp_vec:
            return cfg_rpa, one_block_per_seq, cfg_rpa.max_ring_pages, False, False
    if not use_grp_vec:
        cfg_rpa = dataclasses.replace(cfg_rpa, swabatch_group_size=1)

    bkv_slot_pages = eff_grp * cfg_rpa.max_ring_pages
    return cfg_rpa, one_block_per_seq, bkv_slot_pages, use_grp_vec, True


def b1_moe_b2_rpad_chip_sharded_jax(
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
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    cos_sin_step: jax.Array | None = None,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    spec_home_slot: jax.Array | None = None,
    free_params: jax.Array | None = None,
    layer_idx: jax.Array | None = None,
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
    qkv_norm_eps: float = 1e-5,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    rpa_sm_scale: float = 1.0,
    rpa_sliding_window: int | None = None,
    rpa_k_scale: float = 1.0,
    rpa_v_scale: float = 1.0,
    rpa_out_norm_eps: float = 1e-5,
    rpa_mode_str: str | None = None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    replica_cost_mode: str = "active",
    ep_axis_name: str = "model",
    config: SparseDecodeConfig | None = None,
    collective_id: int = _W8_CID_BASE + 22,
    interpret: bool = False,
    skip_kv_write: bool = False,
    allow_large_b: bool | None = None,
):
    """Fused Step (c) superkernel: ``[chip_b1 + muse_sparse(3d) + chip_b2 + RPAd]``."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import align_to, cdiv

    if kv_pack_bd4:
        if world_size != 16:
            raise ValueError(f"Step (c) kv_pack_bd4 requires world_size=16, got {world_size}")
        kv_pack_dense2 = False

    st = _prepare_chip_b1_moe_b2_setup(
        rank, stream_chip, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
        pre_ff_gain, w_pre_pair_w8k, w_pre_pair_scale, w_router_km, pre_exp_gain,
        bias, w1, s1, w2, s2, post_exp_norm,
        w_post_tp2_w8k, s_post_tp2, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
        cos_sin_step, rep_w1, rep_s1, rep_w2, rep_s2, replica_meta,
        spec_home_slot, free_params,
        world_size=world_size, fuse_qkv_epi=True, config=config, interpret=interpret,
        has_rpad=True,
    )
    rows = st["rows"]
    r_pair = st["r_pair"]
    hidden = st["hidden"]
    if rows > 256:
        raise ValueError(f"Step (c) requires B <= 256 to fit VMEM, got B={rows}")

    eff_sw = None if (rpa_sliding_window is None or rpa_sliding_window <= 0) else int(rpa_sliding_window)
    kv_cache_flat = kv_cache_fused
    eff_page_indices = page_indices
    if layer_idx is not None:
        num_blocks, layer_num = kv_cache_fused.shape[:2]
        kv_cache_flat = kv_cache_fused.reshape(
            (num_blocks * layer_num,) + kv_cache_fused.shape[2:]
        )
        eff_page_indices = page_indices * jnp.int32(layer_num) + layer_idx.reshape(()).astype(jnp.int32)

    if rpa_mode_str is None:
        try:
            from sglang_tpu.envs import envs as tpu_envs

            rpa_mode_str = str(tpu_envs.SGLANG_TPU_RPA_V3_DECODE_FAST.get()).strip()
        except Exception:
            rpa_mode_str = ""
        if not rpa_mode_str:
            rpa_mode_str = "v2"

    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = eff_page_indices.shape[0] // max_num_seqs
    phys_page_rows = kv_cache_flat.shape[1]
    dist_len = int(distribution.shape[0])
    has_shared_prefix = dist_len > 3
    shared_group_size = (
        max_num_seqs // (dist_len - 3)
        if (has_shared_prefix and (max_num_seqs % (dist_len - 3)) == 0)
        else 0
    )
    cfg_rpa, one_block_per_seq, bkv_slot_pages, use_grp_vec, is_ok = resolve_step_c_rpa_plan(
        rpa_mode_str,
        sliding_window=eff_sw,
        pages_per_seq=pages_per_seq,
        phys_page_rows=phys_page_rows,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
        b_tokens=rows,
        max_num_seqs=max_num_seqs,
        shared_group_size=shared_group_size,
        has_shared_prefix=has_shared_prefix,
        allow_large_b=allow_large_b,
    )
    if not is_ok:
        raise ValueError(
            "Step (c) requires RPAd bkv within VMEM budget and compatible shared_group; "
            f"got num_slots={cfg_rpa.num_slots}, max_ring_pages={cfg_rpa.max_ring_pages}, "
            f"phys_page_rows={phys_page_rows}, loop_kind={cfg_rpa.loop_kind!r}, B={rows}"
        )
    if world_size == 16 and cfg_rpa.num_slots > 4:
        raise ValueError(
            f"TP=16 Step (c) aliases bkv_sems on rs_pair_ssem (8); requires num_slots <= 4, got {cfg_rpa.num_slots}"
        )

    q_heads = 8 if world_size == 16 else 16
    pack2_cols = 512 if world_size == 16 else 1024
    vmem, hbm, sem = st["vmem"], st["hbm"], st["sem"]
    v_sp, h_sp = st["v_sp"], st["h_sp"]
    interpret_mode = st["interpret_mode"]
    smem = pltpu.SMEM
    b_pad32 = align_to(max(rows, 1), 32)

    scalar_args = [
        *st["scalar_args"],
        kv_lens,
        eff_page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
    ]
    in_specs = st["in_specs"] + (h_sp,)
    kv_in_arg = (
        kv_cache_flat
        if interpret_mode
        else pltpu.with_memory_space_constraint(kv_cache_flat, hbm)
    )
    call_args = (*scalar_args, *st["base_call_tensors"], kv_in_arg)
    kv_cache_pallas_in = len(scalar_args) + len(st["base_call_tensors"])

    if skip_kv_write:
        kv_out_shape = jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn)
        kv_out_spec = v_sp
        pallas_aliases = {}
    else:
        kv_out_shape = (
            jax.ShapeDtypeStruct(kv_cache_flat.shape, kv_cache_flat.dtype)
            if interpret_mode
            else pltpu.HBM(kv_cache_flat.shape, kv_cache_flat.dtype)
        )
        kv_out_spec = h_sp
        pallas_aliases = {kv_cache_pallas_in: 2}

    out_shape = (
        jax.ShapeDtypeStruct((r_pair, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, pack2_cols), jnp.bfloat16),
        kv_out_shape,
    )
    out_specs = (h_sp, v_sp, kv_out_spec)
    pack_tag = "_bd4" if kv_pack_bd4 else ("_d2" if kv_pack_dense2 else "")

    def _invoke_step_c(run_cfg, run_one_blk: bool, run_slot_pg: int, run_grp_vec: bool, suffix: str = ""):
        max_blks_4k = (
            cdiv(max(pages_per_seq, 1), run_cfg.max_ring_pages)
            if run_cfg.loop_kind == "ring4k"
            else 0
        )
        sh_grp = run_cfg.shared_group_size or (16 if run_cfg.loop_kind == "shared16" else 0)
        max_blks_sh = (
            cdiv(max(pages_per_seq, 1), run_cfg.max_ring_pages) if sh_grp > 0 else 0
        )
        sh_smem_words = 4 * max(sh_grp * max_blks_sh, 0)
        if (max_num_seqs * pages_per_seq + 2 + run_cfg.num_slots + sh_smem_words) * 4 > 960 * 1024:
            sh_smem_words = 0
        smem_words = (
            2
            + run_cfg.num_slots
            + max(4 * max(max_num_seqs * max_blks_4k, 0), sh_smem_words)
        )
        state_rows = q_heads * sh_grp if sh_grp > 0 else q_heads
        scope2_rpad_vmem = (
            vmem((rows, 16, 128), jnp.bfloat16),  # rpad_q_vmem
            *(
                ()
                if skip_kv_write
                else (vmem((rows, 1, 4, 128), jnp.float8_e4m3fn),)  # rpad_kv_vmem
            ),
            vmem((rows, pack2_cols), jnp.bfloat16),  # rpad_gate_vmem
            vmem(
                (run_cfg.num_slots, run_slot_pg, phys_page_rows, 1, 4, 128),
                kv_cache_flat.dtype,
            ),
            vmem((3, state_rows, 128), jnp.float32),
            vmem((rows, q_heads, 128), jnp.bfloat16),
            *(
                (vmem((max(rows, 8), 8, 1, 4, 128), kv_cache_flat.dtype),)
                if ((kv_pack_dense2 or kv_pack_bd4) and not skip_kv_write)
                else ()
            ),
            *(
                (vmem((2, b_pad32, q_heads, 128), jnp.float32),)
                if (run_cfg.hoist_new_kv or run_grp_vec)
                else ()
            ),
        )
        outer_rpad_meta = [
            smem((smem_words,), jnp.int32),
            *(
                (
                    ([sem((run_cfg.num_slots,))])
                    if skip_kv_write
                    else ([sem((run_cfg.num_slots,)), sem((1,))])
                )
                if world_size == 8
                else []
            ),
        ]
        kernel_fn = functools.partial(
            _b1_moe_b2_rpad_chip_sharded_kernel,
            world_size=world_size,
            scale_block=st["scale_block"],
            num_replicas=st["num_replicas"],
            sharded_moe_in=bool(st["sharded_moe_in"]),
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
            fuse_qkv_epi=True,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            has_rope=st["has_rope"],
            cfg=st["config"],
            has_spec_slot_arg=st["has_spec_slot_arg"],
            has_free_params_arg=st["has_free_params_arg"],
            interpret_mode=interpret_mode,
            alias_mate_recv=st["alias_mate_recv"],
            rpa_sm_scale=float(rpa_sm_scale),
            rpa_sliding_window=eff_sw,
            rpa_k_scale=float(rpa_k_scale) if rpa_k_scale is not None else None,
            rpa_v_scale=float(rpa_v_scale) if rpa_v_scale is not None else None,
            rpa_out_norm_eps=float(rpa_out_norm_eps),
            rpa_cfg=run_cfg,
            rpa_one_block_per_seq=bool(run_one_blk),
            rpa_kv_pack_dense2=bool(kv_pack_dense2),
            rpa_kv_pack_bd4=bool(kv_pack_bd4),
            rpa_use_grp_vec=bool(run_grp_vec),
            rpa_skip_kv_write=bool(skip_kv_write),
            use_3scope=st["use_3scope"],
            step_b3_moe_is_scope2=st["step_b3_moe_is_scope2"],
            step_b3_n_moe_outer=st["step_b3_n_moe_outer"],
            step_b3_scope1_shapes=st["step_b3_scope1_shapes"],
            step_b3_scope2_shapes=st["step_b3_scope2_shapes"],
            step_b3_scope3_shapes=st["step_b3_scope3_shapes"],
            step_c_moe_is_scope1=st["step_c_moe_is_scope1"],
            step_c_n_moe_outer=st["step_c_n_moe_outer"],
            step_c_n_moe_scope1=st["step_c_n_moe_scope1"],
            step_c_n_b1_scope1=st["step_c_n_b1_scope1"],
            step_c_scope1_shapes=st["step_c_scope1_shapes"],
            step_c_scope2_shapes=scope2_rpad_vmem,
        )
        outer_scratch = (
            st["step_b3_outer_scratch"] if st["use_3scope"] else st["step_c_outer_scratch"]
        )
        return pl.pallas_call(
            kernel_fn,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_args),
                grid=(1,),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=outer_scratch + outer_rpad_meta,
            ),
            input_output_aliases=pallas_aliases,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=63 * 1024**2,
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            interpret=st["interpret_arg"],
            name=(
                f"muse_w9_b1_moe_b2_rpad_b{rows}_tp{world_size}_k{st['scale_block']}"
                f"_r{st['num_replicas']}_{'r16' if st['is_r16'] else 'r32'}"
                f"{'_dfp' if st['has_free_params_arg'] else ''}"
                f"{pack_tag}{suffix}"
                f"_{'rope' if st['has_rope'] else 'nope'}"
            ),
        )(*call_args)

    # Keep Step (c) Full-Attn unsplit (no jax.lax.cond(is_short_1k, ...)) so XLA dedups
    # all 45 SWA rope calls into 1 IMEM binary without per-position reloads (#4993/#4994).
    st_ch, next_gated, kv_out = _invoke_step_c(
        cfg_rpa, one_block_per_seq, bkv_slot_pages, use_grp_vec
    )
    return st_ch, next_gated, (kv_out if skip_kv_write else kv_out.reshape(kv_cache_fused.shape))


def _read_vmem_row_u32(rmw_vmem, idx, r_sub8, *, interpret_mode: bool):
    """Read row ``idx`` from a ``(..., 8, 1, 4, 128)`` FP8 VMEM buffer as ``(1, 128)`` uint32."""
    from jax import lax

    if interpret_mode:
        old_u8 = lax.bitcast_convert_type(rmw_vmem[idx, 0], jnp.uint8)
        return (
            old_u8[:, 0, :].astype(jnp.uint32)
            | (old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8))
            | (old_u8[:, 2, :].astype(jnp.uint32) << jnp.uint32(16))
            | (old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(24))
        )
    rmw_u32_ref = rmw_vmem.bitcast(jnp.uint32).reshape(-1, 8, 128)
    sub_u32 = rmw_u32_ref[idx]
    return jnp.sum(
        jnp.where(r_sub8 == 0, sub_u32.astype(jnp.int32), jnp.int32(0)),
        axis=0,
        keepdims=True,
    ).astype(jnp.uint32)


def _store_vmem_row_and_dma(
    w_vmem,
    w_idx,
    cache_out_ref,
    dst_row,
    packed_u32,
    out_sem,
    *,
    interpret_mode: bool,
):
    """Store packed ``(1, 128)`` uint32 into ``w_vmem[w_idx]`` and start 1-row async HBM DMA."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if interpret_mode:
        b0 = packed_u32.astype(jnp.uint8)
        b1 = (packed_u32 >> jnp.uint32(8)).astype(jnp.uint8)
        b2 = (packed_u32 >> jnp.uint32(16)).astype(jnp.uint8)
        b3 = (packed_u32 >> jnp.uint32(24)).astype(jnp.uint8)
        row_f8 = lax.bitcast_convert_type(
            jnp.stack([b0, b1, b2, b3], axis=1)[:, None, :, :],
            jnp.float8_e4m3fn,
        )
        w_vmem[w_idx] = jnp.broadcast_to(row_f8, (8, 1, 4, 128))
    else:
        w_u32_ref = w_vmem.bitcast(jnp.uint32).reshape(-1, 8, 128)
        w_u32_ref[w_idx] = jnp.broadcast_to(packed_u32, (8, 128))
    pltpu.make_async_copy(
        src_ref=w_vmem.at[w_idx, pl.ds(0, 1)],
        dst_ref=cache_out_ref.at[pl.ds(dst_row, 1)],
        sem=out_sem.at[0],
    ).start()


def _pack_kv_slot_rmw_u32(
    old_u32,
    u16_ev,
    slot,
    lane_lt_64,
    *,
    kv_pack_bd4: bool,
    interpret_mode: bool,
):
    """Merge ``u16_ev`` at non-zero sub-row ``slot`` into ``old_u32`` for bd4 or dense2."""
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import _roll_lanes_fast

    if kv_pack_bd4:
        u16_od = jnp.where(
            lane_lt_64,
            jnp.uint32(0),
            _roll_lanes_fast(u16_ev, 64, interpret_mode=interpret_mode),
        )
        keep_lo = jnp.where(
            (slot >= 2) | ((slot == 1) & lane_lt_64),
            jnp.uint32(0x0000FFFF),
            jnp.uint32(0),
        )
        keep_hi = jnp.where(
            (slot == 3) & lane_lt_64,
            jnp.uint32(0xFFFF0000),
            jnp.uint32(0),
        )
        new_v = jnp.where((slot & 1) != 0, u16_od, u16_ev)
        new_sh = jnp.where((slot & 2) != 0, new_v << jnp.uint32(16), new_v)
        return (old_u32 & (keep_lo | keep_hi)) | new_sh
    return (old_u32 & jnp.uint32(0x0000FFFF)) | (u16_ev << jnp.uint32(16))


def _packed_kv_write_fallback_eligible(
    t_tokens: int,
    max_num_seqs: int,
    kv_trailing_shape: tuple[int, ...],
    kv_dtype,
    *,
    kv_pack_dense2: bool,
    kv_pack_bd4: bool,
) -> bool:
    """Return True when packed VMEM fast KV writer constraints are NOT met."""
    is_packed_vmem = kv_pack_bd4 or kv_pack_dense2
    if t_tokens > 256 or max_num_seqs > 256:
        return True
    if is_packed_vmem and (
        t_tokens < 8
        or (t_tokens % 8) != 0
        or kv_trailing_shape != (1, 4, 128)
        or kv_dtype != jnp.float8_e4m3fn
    ):
        return True
    return False


def write_decode_kv_merged_jax(
    kv_new: jax.Array,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    interpret: bool | object = False,
) -> jax.Array:
    """Fast per-device KV append for Step (c) (single packed SMEM prefetch + VMEM scatter)."""
    if kv_pack_bd4:
        kv_pack_dense2 = False
    is_packed_vmem = kv_pack_bd4 or kv_pack_dense2
    if _packed_kv_write_fallback_eligible(
        kv_new.shape[0],
        kv_lens.shape[0],
        kv_new.shape[1:],
        kv_new.dtype,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
    ):
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            write_kv_only_rpa_v3,
        )

        return write_kv_only_rpa_v3(
            kv_new,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            interpret=bool(interpret),
            kv_pack_dense2=kv_pack_dense2,
            kv_pack_bd4=kv_pack_bd4,
        )

    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.util import align_to, cdiv, get_tpu_version

    if kv_new.shape[1:] != kv_cache_fused.shape[2:]:
        raise ValueError(
            f"kv_new trailing shape {kv_new.shape[1:]} != kv_cache trailing shape {kv_cache_fused.shape[2:]}"
        )
    if kv_new.dtype != kv_cache_fused.dtype:
        raise ValueError(f"kv_new dtype {kv_new.dtype} != kv_cache dtype {kv_cache_fused.dtype}")
    if kv_pack_bd4 and (kv_new.shape[1:] != (1, 4, 128) or kv_new.dtype != jnp.float8_e4m3fn):
        raise ValueError(
            f"kv_pack_bd4 requires kv_new trailing shape (1, 4, 128) fp8, got {kv_new.shape[1:]} {kv_new.dtype}"
        )

    phys_page_size = kv_cache_fused.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    flat_cache = kv_cache_fused.reshape(
        (kv_cache_fused.shape[0] * phys_page_size, *kv_cache_fused.shape[2:])
    )

    # Pack all 5 scalar-prefetch vectors into 1 contiguous SMEM buffer so XLA issues
    # a single HBM->SMEM transfer instead of 5 per layer (#4345/#4346).
    off_pg = max_num_seqs
    off_cq = off_pg + num_page_indices
    off_ckv = off_cq + (max_num_seqs + 1)
    off_dist = off_ckv + (max_num_seqs + 1)
    meta_packed = jnp.concatenate(
        [
            kv_lens.reshape(-1),
            page_indices.reshape(-1),
            cu_q_lens.reshape(-1),
            cu_kv_lens.reshape(-1),
            distribution.reshape(-1),
        ],
        axis=0,
    ).astype(jnp.int32)

    interpret_mode = interpret is not False
    tpu_version = 0 if interpret_mode else get_tpu_version()
    out_shape = (
        pltpu.HBM(shape=flat_cache.shape, dtype=flat_cache.dtype)
        if tpu_version >= 7
        else jax.ShapeDtypeStruct(shape=flat_cache.shape, dtype=flat_cache.dtype)
    )
    can_2page = phys_page_size >= kv_new.shape[0]

    def _kv_write_small_vmem_kernel(
        meta_ref,
        kv_new_ref,
        cache_in_ref,
        cache_out_ref,
        out_sem,
        wr_cnt_smem,
    ):
        del cache_in_ref
        n_seqs = jnp.clip(meta_ref[off_dist + 2], 0, max_num_seqs)

        def _seq_body(seq_idx, wr_cnt):
            q_start = meta_ref[off_cq + seq_idx]
            q_len = meta_ref[off_cq + seq_idx + 1] - q_start
            kv_len = meta_ref[seq_idx]
            active = (q_len > 0) & (kv_len >= q_len)

            if can_2page:
                @pl.when(active & (q_len == 1))
                def _write_single():
                    pos = kv_len - 1
                    seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], phys_page_size)
                    pg = meta_ref[
                        off_pg + jnp.minimum(seq_pg_base + (pos // phys_page_size), num_page_indices - 1)
                    ]
                    dst_row = pg * phys_page_size + (pos % phys_page_size)
                    pltpu.make_async_copy(
                        src_ref=kv_new_ref.at[pl.ds(q_start, 1)],
                        dst_ref=cache_out_ref.at[pl.ds(dst_row, 1)],
                        sem=out_sem.at[0],
                    ).start()

                @pl.when(active & (q_len > 1))
                def _write_multi_2page():
                    kv_start = kv_len - q_len
                    p0 = kv_start // phys_page_size
                    seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], phys_page_size)
                    tok_hi0 = jnp.minimum(kv_len, (p0 + 1) * phys_page_size)
                    n0 = tok_hi0 - kv_start
                    pg0 = meta_ref[off_pg + jnp.minimum(seq_pg_base + p0, num_page_indices - 1)]
                    dst_row0 = pg0 * phys_page_size + (kv_start - p0 * phys_page_size)
                    pltpu.make_async_copy(
                        src_ref=kv_new_ref.at[pl.ds(q_start, n0)],
                        dst_ref=cache_out_ref.at[pl.ds(dst_row0, n0)],
                        sem=out_sem.at[0],
                    ).start()

                    @pl.when(tok_hi0 < kv_len)
                    def _write_tail_page():
                        n1 = kv_len - tok_hi0
                        pg1 = meta_ref[off_pg + jnp.minimum(seq_pg_base + p0 + 1, num_page_indices - 1)]
                        pltpu.make_async_copy(
                            src_ref=kv_new_ref.at[pl.ds(q_start + n0, n1)],
                            dst_ref=cache_out_ref.at[pl.ds(pg1 * phys_page_size, n1)],
                            sem=out_sem.at[0],
                        ).start()
            else:
                @pl.when(active)
                def _write_seq():
                    kv_start = kv_len - q_len
                    p_start = kv_start // phys_page_size
                    p_end = cdiv(kv_len, phys_page_size)
                    seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], phys_page_size)

                    @pl.loop(p_start, p_end)
                    def _page(p):
                        tok_lo = jnp.maximum(kv_start, p * phys_page_size)
                        tok_hi = jnp.minimum(kv_len, (p + 1) * phys_page_size)
                        n = tok_hi - tok_lo
                        src_row = q_start + (tok_lo - kv_start)
                        pg = meta_ref[off_pg + jnp.minimum(seq_pg_base + p, num_page_indices - 1)]
                        dst_row = pg * phys_page_size + (tok_lo - p * phys_page_size)
                        pltpu.make_async_copy(
                            src_ref=kv_new_ref.at[pl.ds(src_row, n)],
                            dst_ref=cache_out_ref.at[pl.ds(dst_row, n)],
                            sem=out_sem.at[0],
                        ).start()

            return wr_cnt + jnp.where(active, q_len, jnp.int32(0))

        wr_cnt_smem[0] = lax.fori_loop(0, n_seqs, _seq_body, jnp.int32(0))
        w_cnt = wr_cnt_smem[0]

        @pl.when(w_cnt > 0)
        def _wait_all():
            src = kv_new_ref.at[pl.ds(0, w_cnt)]
            pltpu.make_async_copy(src, src, out_sem.at[0]).wait()

    def _kv_write_packed_vmem_kernel(
        meta_ref,
        kv_new_ref,
        cache_in_ref,
        cache_out_ref,
        rmw_vmem,
        w_vmem,
        in_sem,
        out_sem,
        cnt_smem,
    ):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _load_new_kv_1row_u16,
            _roll_lanes_fast,
        )

        pack_factor = 4 if kv_pack_bd4 else 2
        pack_mask = pack_factor - 1
        page_size = phys_page_size * pack_factor
        n_seqs = jnp.clip(meta_ref[off_dist + 2], 0, max_num_seqs)
        lane_lt_64 = (
            lax.broadcasted_iota(jnp.int32, (1, 128), 1) < 64
            if kv_pack_bd4
            else None
        )
        r_sub8 = lax.broadcasted_iota(jnp.int32, (8, 128), 0)

        def _rmw_prefetch_body(seq_idx, rmw_cnt):
            q_start = meta_ref[off_cq + seq_idx]
            q_len = meta_ref[off_cq + seq_idx + 1] - q_start
            kv_len = meta_ref[seq_idx]
            kv_start = kv_len - q_len
            need_rmw = (q_len > 0) & (kv_len >= q_len) & ((kv_start & pack_mask) != 0)

            @pl.when(need_rmw)
            def _fetch_row():
                p0 = kv_start // page_size
                seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], page_size)
                pg0 = meta_ref[off_pg + jnp.minimum(seq_pg_base + p0, num_page_indices - 1)]
                dst_row0 = pg0 * phys_page_size + ((kv_start % page_size) // pack_factor)
                pltpu.make_async_copy(
                    src_ref=cache_in_ref.at[pl.ds(dst_row0, 1)],
                    dst_ref=rmw_vmem.at[seq_idx, pl.ds(0, 1)],
                    sem=in_sem.at[0],
                ).start()

            return rmw_cnt + jnp.where(need_rmw, jnp.int32(1), jnp.int32(0))

        cnt_smem[0] = lax.fori_loop(0, n_seqs, _rmw_prefetch_body, jnp.int32(0))
        rmw_cnt = cnt_smem[0]

        @pl.when(rmw_cnt > 0)
        def _wait_rmw():
            w_ref = rmw_vmem.at[pl.ds(0, rmw_cnt), pl.ds(0, 1)]
            pltpu.make_async_copy(w_ref, w_ref, in_sem.at[0]).wait()

        def _write_packed_body(seq_idx, w_cur):
            q_start = meta_ref[off_cq + seq_idx]
            q_len = meta_ref[off_cq + seq_idx + 1] - q_start
            kv_len = meta_ref[seq_idx]
            active = (q_len > 0) & (kv_len >= q_len)
            seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], page_size)

            @pl.when(active & (q_len == 1))
            def _write_packed_single():
                pos = kv_len - 1
                pg = meta_ref[
                    off_pg + jnp.minimum(seq_pg_base + (pos // page_size), num_page_indices - 1)
                ]
                dst_row = pg * phys_page_size + ((pos % page_size) // pack_factor)
                slot = pos & pack_mask
                raw_u16 = _load_new_kv_1row_u16(
                    kv_new_ref, q_start, interpret_mode=interpret_mode
                )
                u16_ev = jnp.where(lane_lt_64, raw_u16, jnp.uint32(0)) if kv_pack_bd4 else raw_u16

                @pl.when(slot == 0)
                def _s0():
                    _store_vmem_row_and_dma(
                        w_vmem, w_cur, cache_out_ref, dst_row, u16_ev, out_sem,
                        interpret_mode=interpret_mode,
                    )

                @pl.when(slot != 0)
                def _s_rmw():
                    old_u32 = _read_vmem_row_u32(
                        rmw_vmem, seq_idx, r_sub8, interpret_mode=interpret_mode
                    )
                    packed = _pack_kv_slot_rmw_u32(
                        old_u32, u16_ev, slot, lane_lt_64,
                        kv_pack_bd4=kv_pack_bd4, interpret_mode=interpret_mode,
                    )
                    _store_vmem_row_and_dma(
                        w_vmem, w_cur, cache_out_ref, dst_row, packed, out_sem,
                        interpret_mode=interpret_mode,
                    )

            @pl.when(active & (q_len > 1))
            def _write_packed_multi():
                kv_start = kv_len - q_len

                @pl.when((kv_start & pack_mask) == 0)
                def _zero_unused_rmw():
                    rmw_vmem[seq_idx] = jnp.zeros((8, 1, 4, 128), kv_new_ref.dtype)

                old_u32 = _read_vmem_row_u32(
                    rmw_vmem, seq_idx, r_sub8, interpret_mode=interpret_mode
                )
                q_first = kv_start // pack_factor
                q_last = cdiv(kv_len, pack_factor)

                @pl.loop(q_first, q_last)
                def _packed_row(q_idx):
                    tok_base = q_idx * pack_factor
                    pg = meta_ref[
                        off_pg + jnp.minimum(seq_pg_base + (tok_base // page_size), num_page_indices - 1)
                    ]
                    dst_row = pg * phys_page_size + ((tok_base % page_size) // pack_factor)
                    w_vmem_idx = w_cur + (q_idx - q_first)
                    if kv_pack_bd4:
                        k_lo = jnp.where(
                            (tok_base + 1 < kv_start) | ((tok_base < kv_start) & lane_lt_64),
                            jnp.uint32(0x0000FFFF),
                            jnp.uint32(0),
                        )
                        k_hi = jnp.where(
                            (tok_base + 3 < kv_start) | ((tok_base + 2 < kv_start) & lane_lt_64),
                            jnp.uint32(0xFFFF0000),
                            jnp.uint32(0),
                        )
                        keep_mask = k_lo | k_hi
                    else:
                        keep_mask = jnp.uint32(0x0000FFFF)
                    acc_base = jnp.where(
                        tok_base < kv_start, old_u32 & keep_mask, jnp.uint32(0)
                    )

                    def _add_tok(s_off, acc_in):
                        tok_s = tok_base + s_off
                        tok_ok = (tok_s >= kv_start) & (tok_s < kv_len)
                        safe_row = q_start + jnp.clip(tok_s - kv_start, 0, q_len - 1)
                        raw_s = _load_new_kv_1row_u16(
                            kv_new_ref, safe_row, interpret_mode=interpret_mode
                        )
                        if kv_pack_bd4:
                            ev_s = jnp.where(lane_lt_64, raw_s, jnp.uint32(0))
                            if s_off & 1:
                                val_s = jnp.where(
                                    lane_lt_64,
                                    jnp.uint32(0),
                                    _roll_lanes_fast(ev_s, 64, interpret_mode=interpret_mode),
                                )
                            else:
                                val_s = ev_s
                            sh_s = (val_s << jnp.uint32(16)) if (s_off & 2) else val_s
                        else:
                            sh_s = (raw_s << jnp.uint32(16)) if s_off else raw_s
                        return jnp.where(tok_ok, acc_in | sh_s, acc_in)

                    packed = acc_base
                    for s_off in range(pack_factor):
                        packed = _add_tok(s_off, packed)
                    _store_vmem_row_and_dma(
                        w_vmem, w_vmem_idx, cache_out_ref, dst_row, packed, out_sem,
                        interpret_mode=interpret_mode,
                    )

            n_packed = jnp.where(
                active,
                jnp.where(
                    q_len == 1,
                    jnp.int32(1),
                    cdiv(kv_len, pack_factor) - ((kv_len - q_len) // pack_factor),
                ),
                jnp.int32(0),
            )
            return w_cur + n_packed

        cnt_smem[1] = lax.fori_loop(0, n_seqs, _write_packed_body, jnp.int32(0))
        w_cnt = cnt_smem[1]

        @pl.when(w_cnt > 0)
        def _wait_out():
            src = w_vmem.at[pl.ds(0, w_cnt), pl.ds(0, 1)]
            pltpu.make_async_copy(src, src, out_sem.at[0]).wait()

    if is_packed_vmem:
        scratches = [
            pltpu.VMEM((align_to(max(max_num_seqs, 1), 8), 8, 1, 4, 128), kv_new.dtype),
            pltpu.VMEM((align_to(max(kv_new.shape[0], 1), 8), 8, 1, 4, 128), kv_new.dtype),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SMEM((2,), jnp.int32),
        ]
        k_fn = _kv_write_packed_vmem_kernel
        k_name = (
            f"StepC_write_kv_only_bd4-p_{phys_page_size * 4}"
            if kv_pack_bd4
            else f"StepC_write_kv_only_d2-p_{phys_page_size * 2}"
        )
    else:
        scratches = [
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SMEM((1,), jnp.int32),
        ]
        k_fn = _kv_write_small_vmem_kernel
        k_name = f"StepC_write_kv_only-p_{phys_page_size}"

    pallas_kwargs = {
        "grid_spec": pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=scratches,
        ),
        "out_shape": out_shape,
        "input_output_aliases": {2: 0},
        "name": k_name,
    }
    if interpret_mode:
        pallas_kwargs["interpret"] = (
            pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
            if isinstance(interpret, bool)
            else interpret
        )
    else:
        pallas_kwargs["compiler_params"] = pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        )
    kernel = pl.pallas_call(k_fn, **pallas_kwargs)
    cache_in = (
        pltpu.with_memory_space_constraint(flat_cache, pltpu.HBM)
        if tpu_version >= 7
        else flat_cache
    )
    updated = kernel(meta_packed, kv_new, cache_in)
    return updated.reshape(kv_cache_fused.shape)


def _read_vmem_8_u32(rmw_vmem, seq_idx, *, interpret_mode: bool):
    """Read 8-sublane ``(8, 128)`` uint32 chunk from ``rmw_vmem[seq_idx]``."""
    from jax import lax

    if interpret_mode:
        old_u8 = lax.bitcast_convert_type(rmw_vmem[seq_idx, :, 0], jnp.uint8)
        return (
            old_u8[:, 0, :].astype(jnp.uint32)
            | (old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8))
            | (old_u8[:, 2, :].astype(jnp.uint32) << jnp.uint32(16))
            | (old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(24))
        )
    rmw_u32_ref = rmw_vmem.bitcast(jnp.uint32).reshape(rmw_vmem.shape[0], 8, 128)
    return rmw_u32_ref[seq_idx]


def _store_vmem_8_u32(w_vmem, w_idx, packed_u32, *, interpret_mode: bool):
    """Store 8-sublane ``(8, 128)`` uint32 chunk into ``w_vmem[w_idx]``."""
    from jax import lax

    if interpret_mode:
        b0 = packed_u32.astype(jnp.uint8)
        b1 = (packed_u32 >> jnp.uint32(8)).astype(jnp.uint8)
        b2 = (packed_u32 >> jnp.uint32(16)).astype(jnp.uint8)
        b3 = (packed_u32 >> jnp.uint32(24)).astype(jnp.uint8)
        w_vmem[w_idx] = lax.bitcast_convert_type(
            jnp.stack([b0, b1, b2, b3], axis=1)[:, None, :, :],
            jnp.float8_e4m3fn,
        )
        return
    w_u32_ref = w_vmem.bitcast(jnp.uint32).reshape(w_vmem.shape[0], 8, 128)
    w_u32_ref[w_idx] = packed_u32


def _load_layers_kv_u16(
    kv_chunk_vmem,
    q_start,
    n_wr_layers: int,
    t_tokens: int,
    r_sub8,
    *,
    interpret_mode: bool,
):
    """Pack ``kv_new`` low-16-bit entries for ``0..n_wr_layers-1`` into ``(8, 128)`` uint32."""
    from jax.experimental import pallas as pl
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import _load_new_kv_1row_u16

    acc_ev = jnp.zeros((8, 128), jnp.uint32)
    if interpret_mode:
        for li in range(n_wr_layers):
            v1 = _load_new_kv_1row_u16(
                kv_chunk_vmem, li * t_tokens + q_start, interpret_mode=True
            )
            acc_ev = jnp.where(r_sub8 == li, v1, acc_ev)
        return acc_ev

    kv_u32_ref = kv_chunk_vmem.bitcast(jnp.uint32).reshape(kv_chunk_vmem.shape[0], 128)
    aligned8 = (q_start // 8) * 8
    tok_mask = r_sub8 == (q_start - aligned8)
    for li in range(n_wr_layers):
        row_idx = li * t_tokens + aligned8
        blk = kv_u32_ref[pl.ds(pl.multiple_of(row_idx, 8), 8)] & jnp.uint32(0xFFFF)
        v1 = jnp.sum(
            jnp.where(tok_mask, blk.astype(jnp.int32), jnp.int32(0)),
            axis=0,
            keepdims=True,
        ).astype(jnp.uint32)
        acc_ev = jnp.where(r_sub8 == li, v1, acc_ev)
    return acc_ev


def write_decode_kv_multi_layer_jax(
    kv_new_layers: jax.Array,
    kv_caches: jax.Array | tuple[jax.Array, ...] | list[jax.Array],
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    layer_ids: jax.Array | None = None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    interpret: bool | object = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Batched multi-layer Step (c) KV append for 6D bundles or <=8 5D layer buffers."""
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.util import align_to, cdiv, get_tpu_version

    if kv_pack_bd4:
        kv_pack_dense2 = False
    is_packed = kv_pack_bd4 or kv_pack_dense2
    if kv_new_layers.ndim != 5:
        raise ValueError(
            f"Expected 5D kv_new_layers (L, T, 1, 4, 128), got {kv_new_layers.shape}"
        )

    n_wr_layers, t_tokens = kv_new_layers.shape[0], kv_new_layers.shape[1]
    is_bundle = layer_ids is not None
    if is_bundle:
        if not isinstance(kv_caches, jax.Array) or kv_caches.ndim != 6:
            raise ValueError(
                f"Expected single 6D kv_cache when layer_ids is provided, got {getattr(kv_caches, 'shape', type(kv_caches))}"
            )
        layer_ids_1d = layer_ids.reshape(-1).astype(jnp.int32)
        if layer_ids_1d.shape[0] != n_wr_layers:
            raise ValueError(
                f"layer_ids length {layer_ids_1d.shape[0]} != kv_new_layers.shape[0] {n_wr_layers}"
            )
        if kv_new_layers.shape[2:] != kv_caches.shape[3:]:
            raise ValueError(
                f"kv_new trailing shape {kv_new_layers.shape[2:]} != kv_cache trailing shape {kv_caches.shape[3:]}"
            )
        if kv_new_layers.dtype != kv_caches.dtype:
            raise ValueError(
                f"kv_new dtype {kv_new_layers.dtype} != kv_cache dtype {kv_caches.dtype}"
            )
        phys_page_size = kv_caches.shape[2]
        n_pool_layers = kv_caches.shape[1]
        cache_dtype = kv_caches.dtype
    else:
        caches_seq = (
            (kv_caches,) if isinstance(kv_caches, jax.Array) else tuple(kv_caches)
        )
        if len(caches_seq) != n_wr_layers or n_wr_layers == 0:
            raise ValueError(
                f"Expected {n_wr_layers} 5D kv_caches, got {len(caches_seq)}"
            )
        ref_shape = caches_seq[0].shape
        cache_dtype = caches_seq[0].dtype
        if len(ref_shape) != 5:
            raise ValueError(f"Expected 5D per-layer kv_cache, got {ref_shape}")
        for idx, c in enumerate(caches_seq):
            if c.shape != ref_shape or c.dtype != cache_dtype:
                raise ValueError(
                    f"kv_caches[{idx}] shape/dtype {(c.shape, c.dtype)} != {(ref_shape, cache_dtype)}"
                )
        if kv_new_layers.shape[2:] != ref_shape[2:]:
            raise ValueError(
                f"kv_new trailing shape {kv_new_layers.shape[2:]} != kv_cache trailing shape {ref_shape[2:]}"
            )
        if kv_new_layers.dtype != cache_dtype:
            raise ValueError(
                f"kv_new dtype {kv_new_layers.dtype} != kv_cache dtype {cache_dtype}"
            )
        phys_page_size = ref_shape[1]
        n_pool_layers = 1

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]

    def _fallback_per_layer(kv_in, caches_in):
        if is_bundle:
            n_blk = caches_in.shape[0]
            flat_2d = caches_in.reshape(
                (n_blk * n_pool_layers, phys_page_size, *caches_in.shape[3:])
            )

            def _loop_body(i, cur_flat):
                pg_i = page_indices * jnp.int32(n_pool_layers) + layer_ids_1d[i]
                return write_decode_kv_merged_jax(
                    kv_in[i],
                    cur_flat,
                    kv_lens,
                    pg_i,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    kv_pack_dense2=kv_pack_dense2,
                    kv_pack_bd4=kv_pack_bd4,
                    interpret=interpret,
                )

            return lax.fori_loop(0, n_wr_layers, _loop_body, flat_2d).reshape(
                caches_in.shape
            )

        out_list = []
        for i, c_i in enumerate(caches_in):
            out_list.append(
                write_decode_kv_merged_jax(
                    kv_in[i],
                    c_i,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    kv_pack_dense2=kv_pack_dense2,
                    kv_pack_bd4=kv_pack_bd4,
                    interpret=interpret,
                )
            )
        return tuple(out_list)

    if not is_bundle and n_wr_layers > 8:
        out_chunks: list[jax.Array] = []
        for lo in range(0, n_wr_layers, 8):
            hi = min(lo + 8, n_wr_layers)
            res = write_decode_kv_multi_layer_jax(
                kv_new_layers[lo:hi],
                caches_seq[lo:hi],
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                layer_ids=None,
                kv_pack_dense2=kv_pack_dense2,
                kv_pack_bd4=kv_pack_bd4,
                interpret=interpret,
            )
            out_chunks.extend(res)
        return tuple(out_chunks)

    if n_wr_layers <= 1 or _packed_kv_write_fallback_eligible(
        t_tokens,
        max_num_seqs,
        kv_new_layers.shape[2:],
        kv_new_layers.dtype,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
    ):
        return _fallback_per_layer(
            kv_new_layers, kv_caches if is_bundle else caches_seq
        )

    pack_factor = 4 if kv_pack_bd4 else (2 if kv_pack_dense2 else 1)
    pack_mask = pack_factor - 1
    page_size = phys_page_size * pack_factor
    use_sublane8 = (not is_bundle) and is_packed
    layer_chunk = (8 if t_tokens > 128 else 24) if is_bundle else n_wr_layers
    t_pad = align_to(max(t_tokens, 1), 8)
    seq_pad = align_to(max(max_num_seqs, 1), 8)
    rmw_rows = seq_pad if use_sublane8 else layer_chunk * seq_pad
    w_rows = t_pad if use_sublane8 else layer_chunk * t_pad

    off_pg = max_num_seqs
    off_cq = off_pg + num_page_indices
    off_ckv = off_cq + (max_num_seqs + 1)
    off_dist = off_ckv + (max_num_seqs + 1)
    off_lids = off_dist + distribution.shape[0]
    meta_parts = [
        kv_lens.reshape(-1),
        page_indices.reshape(-1),
        cu_q_lens.reshape(-1),
        cu_kv_lens.reshape(-1),
        distribution.reshape(-1),
    ]
    if is_bundle:
        meta_parts.append(layer_ids_1d)
    meta_packed = jnp.concatenate(meta_parts, axis=0).astype(jnp.int32)

    interpret_mode = interpret is not False
    tpu_version = 0 if interpret_mode else get_tpu_version()

    def _run_multi_fast(kv_in, caches_in):
        kv_in_hbm = kv_in.reshape((n_wr_layers * t_tokens, *kv_in.shape[2:]))
        if is_bundle:
            flat_caches = (
                caches_in.reshape(
                    (
                        caches_in.shape[0] * n_pool_layers * phys_page_size,
                        *caches_in.shape[3:],
                    )
                ),
            )
        else:
            flat_caches = tuple(
                c.reshape((c.shape[0] * phys_page_size, *c.shape[2:])) for c in caches_in
            )
        n_bufs = len(flat_caches)

        def _multi_kernel(*refs):
            meta_ref = refs[0]
            kv_all_ref = refs[1]
            cache_in_refs = refs[2 : 2 + n_bufs]
            cache_out_refs = refs[2 + n_bufs : 2 + 2 * n_bufs]
            scratch_refs = refs[2 + 2 * n_bufs :]
            if is_packed:
                (
                    kv_chunk_vmem,
                    rmw_vmem,
                    w_vmem,
                    kv_in_sem,
                    rmw_sem,
                    out_sem,
                    cnt_smem,
                ) = scratch_refs
            else:
                kv_chunk_vmem, kv_in_sem, out_sem, cnt_smem = scratch_refs
                del cache_in_refs

            n_seqs = jnp.clip(meta_ref[off_dist + 2], 0, max_num_seqs)
            lane_lt_64 = (
                lax.broadcasted_iota(jnp.int32, (1, 128), 1) < 64
                if kv_pack_bd4
                else None
            )
            r_sub8 = (
                lax.broadcasted_iota(jnp.int32, (8, 128), 0) if is_packed else None
            )

            def _seq_meta(seq_idx):
                q_start = meta_ref[off_cq + seq_idx]
                q_len = meta_ref[off_cq + seq_idx + 1] - q_start
                kv_len = meta_ref[seq_idx]
                active = (q_len > 0) & (kv_len >= q_len)
                pos = kv_len - 1
                seq_pg_base = cdiv(meta_ref[off_ckv + seq_idx], page_size)
                pg = meta_ref[
                    off_pg
                    + jnp.minimum(
                        seq_pg_base + (pos // page_size), num_page_indices - 1
                    )
                ]
                row_in_pg = (pos % page_size) // pack_factor
                return q_start, q_len, active, pos, pg, row_in_pg

            def _emit_chunk(ch_idx):
                l_lo = ch_idx * layer_chunk
                l_cnt = (
                    jnp.minimum(layer_chunk, n_wr_layers - l_lo)
                    if is_bundle
                    else n_wr_layers
                )
                pltpu.make_async_copy(
                    src_ref=kv_all_ref.at[pl.ds(l_lo * t_tokens, l_cnt * t_tokens)],
                    dst_ref=kv_chunk_vmem.at[pl.ds(0, l_cnt * t_tokens)],
                    sem=kv_in_sem.at[0],
                ).start()

                if is_packed:

                    def _rmw_seq(seq_idx, r_cnt):
                        _, q_len, active, pos, pg, row_in_pg = _seq_meta(seq_idx)
                        need_rmw = (
                            active & (q_len == 1) & ((pos & pack_mask) != 0)
                        )

                        @pl.when(need_rmw)
                        def _issue_layers():
                            if is_bundle:

                                @pl.loop(0, l_cnt)
                                def _li(li):
                                    lid = meta_ref[off_lids + l_lo + li]
                                    dst_row = (
                                        pg * n_pool_layers + lid
                                    ) * phys_page_size + row_in_pg
                                    pltpu.make_async_copy(
                                        src_ref=cache_in_refs[0].at[pl.ds(dst_row, 1)],
                                        dst_ref=rmw_vmem.at[
                                            li * seq_pad + seq_idx, pl.ds(0, 1)
                                        ],
                                        sem=rmw_sem.at[0],
                                    ).start()

                            else:
                                dst_row = pg * phys_page_size + row_in_pg
                                for li in range(n_wr_layers):
                                    pltpu.make_async_copy(
                                        src_ref=cache_in_refs[li].at[pl.ds(dst_row, 1)],
                                        dst_ref=(
                                            rmw_vmem.at[seq_idx, pl.ds(li, 1)]
                                            if use_sublane8
                                            else rmw_vmem.at[li * seq_pad + seq_idx, pl.ds(0, 1)]
                                        ),
                                        sem=rmw_sem.at[0],
                                    ).start()

                        r_inc = jnp.int32(1) if use_sublane8 else l_cnt
                        return r_cnt + jnp.where(need_rmw, r_inc, jnp.int32(0))

                    cnt_smem[0] = lax.fori_loop(
                        0, n_seqs, _rmw_seq, jnp.int32(0)
                    )
                    r_tot = cnt_smem[0]

                pltpu.make_async_copy(
                    kv_chunk_vmem.at[pl.ds(0, l_cnt * t_tokens)],
                    kv_chunk_vmem.at[pl.ds(0, l_cnt * t_tokens)],
                    kv_in_sem.at[0],
                ).wait()

                if is_packed:
                    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
                        _load_new_kv_1row_u16,
                    )

                    @pl.when(r_tot > 0)
                    def _wait_rmw():
                        w_ref = (
                            rmw_vmem.at[pl.ds(0, r_tot), pl.ds(0, n_wr_layers)]
                            if use_sublane8
                            else rmw_vmem.at[pl.ds(0, r_tot), pl.ds(0, 1)]
                        )
                        pltpu.make_async_copy(w_ref, w_ref, rmw_sem.at[0]).wait()

                def _emit_sublane8_seq(seq_idx, w_cur, q_start, pos, pg, row_in_pg):
                    dst_row = pg * phys_page_size + row_in_pg
                    slot = pos & pack_mask
                    raw_ev = _load_layers_kv_u16(
                        kv_chunk_vmem, q_start, n_wr_layers, t_tokens, r_sub8,
                        interpret_mode=interpret_mode,
                    )
                    u16_ev = (
                        jnp.where(lane_lt_64, raw_ev, jnp.uint32(0))
                        if kv_pack_bd4
                        else raw_ev
                    )

                    @pl.when(slot == 0)
                    def _s0_8():
                        _store_vmem_8_u32(
                            w_vmem, w_cur, u16_ev, interpret_mode=interpret_mode
                        )

                    @pl.when(slot != 0)
                    def _srmw_8():
                        old_u32 = _read_vmem_8_u32(
                            rmw_vmem, seq_idx, interpret_mode=interpret_mode
                        )
                        packed = _pack_kv_slot_rmw_u32(
                            old_u32,
                            u16_ev,
                            slot,
                            lane_lt_64,
                            kv_pack_bd4=kv_pack_bd4,
                            interpret_mode=interpret_mode,
                        )
                        _store_vmem_8_u32(
                            w_vmem, w_cur, packed, interpret_mode=interpret_mode
                        )

                    for li in range(n_wr_layers):
                        pltpu.make_async_copy(
                            src_ref=w_vmem.at[w_cur, pl.ds(li, 1)],
                            dst_ref=cache_out_refs[li].at[pl.ds(dst_row, 1)],
                            sem=out_sem.at[0],
                        ).start()

                def _emit_one_layer(seq_idx, li, w_cur, q_start, pos, pg, row_in_pg, dst_out_ref):
                    kv_row = li * t_tokens + q_start
                    if is_bundle:
                        lid = meta_ref[off_lids + l_lo + li]
                        dst_row = (pg * n_pool_layers + lid) * phys_page_size + row_in_pg
                    else:
                        dst_row = pg * phys_page_size + row_in_pg

                    if not is_packed:
                        pltpu.make_async_copy(
                            src_ref=kv_chunk_vmem.at[pl.ds(kv_row, 1)],
                            dst_ref=dst_out_ref.at[pl.ds(dst_row, 1)],
                            sem=out_sem.at[0],
                        ).start()
                        return

                    slot = pos & pack_mask
                    raw_u16 = _load_new_kv_1row_u16(
                        kv_chunk_vmem, kv_row, interpret_mode=interpret_mode
                    )
                    u16_ev = (
                        jnp.where(lane_lt_64, raw_u16, jnp.uint32(0))
                        if kv_pack_bd4
                        else raw_u16
                    )

                    @pl.when(slot == 0)
                    def _s0():
                        _store_vmem_row_and_dma(
                            w_vmem,
                            w_cur + li,
                            dst_out_ref,
                            dst_row,
                            u16_ev,
                            out_sem,
                            interpret_mode=interpret_mode,
                        )

                    @pl.when(slot != 0)
                    def _srmw():
                        old_u32 = _read_vmem_row_u32(
                            rmw_vmem,
                            li * seq_pad + seq_idx,
                            r_sub8,
                            interpret_mode=interpret_mode,
                        )
                        packed = _pack_kv_slot_rmw_u32(
                            old_u32,
                            u16_ev,
                            slot,
                            lane_lt_64,
                            kv_pack_bd4=kv_pack_bd4,
                            interpret_mode=interpret_mode,
                        )
                        _store_vmem_row_and_dma(
                            w_vmem,
                            w_cur + li,
                            dst_out_ref,
                            dst_row,
                            packed,
                            out_sem,
                            interpret_mode=interpret_mode,
                        )

                def _wr_seq(seq_idx, w_cur):
                    q_start, q_len, active, pos, pg, row_in_pg = _seq_meta(seq_idx)

                    @pl.when(active & (q_len == 1))
                    def _pack_all_layers():
                        if is_bundle:

                            @pl.loop(0, l_cnt)
                            def _li(li):
                                _emit_one_layer(
                                    seq_idx, li, w_cur, q_start, pos, pg, row_in_pg, cache_out_refs[0]
                                )

                        elif use_sublane8:
                            _emit_sublane8_seq(
                                seq_idx, w_cur, q_start, pos, pg, row_in_pg
                            )

                        else:
                            for li in range(n_wr_layers):
                                _emit_one_layer(
                                    seq_idx, li, w_cur, q_start, pos, pg, row_in_pg, cache_out_refs[li]
                                )

                    w_inc = jnp.int32(1) if use_sublane8 else l_cnt
                    return w_cur + jnp.where(
                        active & (q_len == 1), w_inc, jnp.int32(0)
                    )

                cnt_smem[1] = lax.fori_loop(0, n_seqs, _wr_seq, jnp.int32(0))
                w_tot = cnt_smem[1]

                @pl.when(w_tot > 0)
                def _wait_out():
                    if use_sublane8:
                        src = w_vmem.at[pl.ds(0, w_tot), pl.ds(0, n_wr_layers)]
                    elif is_packed:
                        src = w_vmem.at[pl.ds(0, w_tot), pl.ds(0, 1)]
                    else:
                        src = kv_chunk_vmem.at[pl.ds(0, w_tot)]
                    pltpu.make_async_copy(src, src, out_sem.at[0]).wait()

            if is_bundle:
                @pl.loop(0, cdiv(n_wr_layers, layer_chunk))
                def _chunk_loop(ch_idx):
                    _emit_chunk(ch_idx)
            else:
                _emit_chunk(jnp.int32(0))

        scratches = [
            pltpu.VMEM((layer_chunk * t_pad, *kv_new_layers.shape[2:]), cache_dtype),
            *(
                [
                    pltpu.VMEM((rmw_rows, 8, 1, 4, 128), cache_dtype),
                    pltpu.VMEM((w_rows, 8, 1, 4, 128), cache_dtype),
                    pltpu.SemaphoreType.DMA((1,)),
                ]
                if is_packed
                else []
            ),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SMEM((2,), jnp.int32),
        ]
        out_shape = tuple(
            pltpu.HBM(shape=fc.shape, dtype=fc.dtype)
            if tpu_version >= 7
            else jax.ShapeDtypeStruct(shape=fc.shape, dtype=fc.dtype)
            for fc in flat_caches
        )
        pallas_kwargs = {
            "grid_spec": pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * (1 + n_bufs),
                out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * n_bufs,
                grid=(1,),
                scratch_shapes=scratches,
            ),
            "out_shape": out_shape,
            "input_output_aliases": {2 + i: i for i in range(n_bufs)},
            "name": (
                f"StepC_write_kv_multi_{'bd4' if kv_pack_bd4 else ('d2' if kv_pack_dense2 else 'u')}"
                f"-p_{page_size}-m_{n_bufs}"
            ),
        }
        if interpret_mode:
            pallas_kwargs["interpret"] = (
                pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
                if isinstance(interpret, bool)
                else interpret
            )
        else:
            pallas_kwargs["compiler_params"] = pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                disable_bounds_checks=True,
            )
        kernel = pl.pallas_call(_multi_kernel, **pallas_kwargs)
        kv_arg = (
            kv_in_hbm
            if interpret_mode
            else pltpu.with_memory_space_constraint(kv_in_hbm, pltpu.HBM)
        )
        cache_args = tuple(
            fc
            if interpret_mode
            else pltpu.with_memory_space_constraint(fc, pltpu.HBM)
            for fc in flat_caches
        )
        updated = kernel(meta_packed, kv_arg, *cache_args)
        if is_bundle:
            return updated[0].reshape(caches_in.shape)
        return tuple(u.reshape(c.shape) for u, c in zip(updated, caches_in))

    q_lens_vec = cu_q_lens[1:] - cu_q_lens[:-1]
    n_s = jnp.clip(distribution[2], 0, max_num_seqs)
    has_multi_q = jnp.any(
        jnp.where(jnp.arange(max_num_seqs) < n_s, q_lens_vec > 1, False)
    )
    return lax.cond(
        has_multi_q,
        lambda args: _fallback_per_layer(*args),
        lambda args: _run_multi_fast(*args),
        (kv_new_layers, kv_caches if is_bundle else caches_seq),
    )
