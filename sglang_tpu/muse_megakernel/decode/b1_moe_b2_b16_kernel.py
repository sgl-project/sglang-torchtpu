# SPDX-License-Identifier: Apache-2.0
"""Wave-9 Replicated ``B=16`` Step (b) Superkernel: Bridge 1b + Muse Sparse/Dense MoE + Bridge 2b.

Fuses an entire ``B=16`` decode layer transition ``attn_{L} -> MoE_{L} -> attn_{L+1}``
on **replicated** residual stream ``[16, 8192] f32`` and standard TP-sharded bridge
weights (``_w7_wpre_w8k`` / ``_w7_wpost_w8k``) across ``TP = EP in (8, 16)``:

  1. ``_run_rep_b1_vmem_core`` (``skip_stream_store=True``, ``out_expert_3d_ref=msc["x_3d_buf"]``
     when ``row_mode=="sparse"``):
     - Aliases ``wo_buf`` inside MoE ``w2_buf``.
     - Aliases ``w_pre_buf``, ``w_qkv_buf``, ``w_gate_buf`` inside shared ``w_b_buf``.
     - Keeps ``s_buf [16, 8192] f32`` live in VMEM across MoE.
     - Overlaps ``plan_and_prime_muse_sparse_from_vmem(..., zero_and_pack=True)`` under Stage-2 ``ag_exp``.
  2. ``run_muse_sparse_experts_and_rs_from_vmem`` (``return_sharded=False``, ``wait_entry_barrier=False``):
     - Supports both ``DECODE_B16_SPARSE=1`` (``row_mode="sparse", tile_m=16, m_buckets=(8, 16)``)
       and default ``row_mode="dense"``, plus zero-compile ``free_params`` (``(4,) f32``).
     - Uses ``on_experts_done_fn()`` to overlap ``b2`` weight DMAs (``w_post -> w1_buf``,
       ``w_qkv + w_gate -> w_b_buf``) behind RS+AG.
  3. ``_run_rep_b2_vmem_core`` (``skip_barrier_signal=True``, ``skip_barrier_wait=True``,
     ``skip_stream_load=True``, ``prestarted_w_*_dma=True``):
     - Emits ``(new_stream, next_q_5d, next_kv_merged, next_gate)`` (or ``(new_stream, next_qkv, next_gate)``).
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import dataclasses
import functools
import math
import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges import _dma_wait, _run_rep_b1_vmem_core
from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    SUPPORTED_WORLD_SIZES,
    _W8_CID_BASE,
    _as_2d_f32,
    _as_dot_scale_f32,
    _run_rep_b2_vmem_core,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
    SparseDecodeConfig,
    build_muse_sparse_scratch_shapes,
    default_sparse_decode_config,
    plan_and_prime_muse_sparse_from_vmem,
    run_muse_sparse_experts_and_rs_from_vmem,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
    _E_TOTAL,
    _H,
    _alias_wo_buf,
    _unpack_muse_sparse_scratch,
)


def _b1_moe_b2_replicated_b16_impl(
    all_refs,
    *,
    world_size: int,
    scale_block: int,
    num_replicas: int,
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
    has_rpad: bool = False,
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
) -> None:
    """Shared Pallas kernel body for replicated ``B=8/16`` Step (b) and 2-scope Step (c)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.kernels.decode_collective_bridges import _use_dual_tree_xy

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
    stream_hbm = next(it)
    gated_ref = next(it)
    wo_hbm = next(it)
    wo_scale_ref = next(it)
    pa_alpha_ref = next(it)
    pa_beta_ref = next(it)
    pre_ff_gain_ref = next(it)
    w_pre_hbm = next(it)
    w_pre_scale_ref = next(it)
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
    out_stream_hbm = next(it)
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

    rows, o_in = gated_ref.shape
    hidden = stream_hbm.shape[1]
    half = hidden // 2
    pre_cols = w_pre_hbm.shape[1]
    post_cols = w_post_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    is_r16 = w_router_hbm.dtype == jnp.bfloat16
    use_3d_x = cfg.row_mode != "dense"
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_chips = world_size // 2

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

    def _unpack_rep_sems(cur_it, msc: dict) -> dict:
        b1_sems = next(cur_it)
        w_pre_sem = b1_sems.at[0:1]
        w_router_sem = b1_sems.at[1:2]
        s_in_sem = b1_sems.at[2:3]
        s_out_sem = b1_sems.at[3:4]
        w_post_sem = w_pre_sem
        w_qkv_sem = w_router_sem
        w_gate_sem = s_in_sem
        if world_size == 8:
            ar1a_s, ar1a_r = next(cur_it), next(cur_it)
            ar2b_s, ar2b_r = next(cur_it), next(cur_it)
            ag1_s, ag1_r = next(cur_it), next(cur_it)
            ag3_s, ag3_r = next(cur_it), next(cur_it)
            ar1b_s, ar1b_r = next(cur_it), next(cur_it)
            ar2a_s, ar2a_r = next(cur_it), next(cur_it)
            ag2_s, ag2_r = next(cur_it), next(cur_it)
            b2_ag1_s, b2_ag2_s, b2_ag3_s = ag1_s, ag2_s, ag3_s
            b2_ag2_r = ag2_r
        else:
            ar1a_r = next(cur_it)
            ar2b_r = next(cur_it)
            ag1_r = next(cur_it)
            ag3_r = next(cur_it)
            ar1b_r = next(cur_it)
            ar2a_r = next(cur_it)
            ag2_r = ar1b_r
            b2_ag2_r = ar1b_r
            ar1a_s = msc["rs_send_sem"].at[0:1]
            ar2b_s = msc["rs_send_sem"].at[1:2]
            ag1_s = msc["rs_send_sem"].at[2:3]
            ag3_s = msc["rs_send_sem"].at[3:4]
            ar1b_s = msc["rs_send_sem"].at[0 : n_chips - 1]
            ar2a_s = msc["rs_send_sem"].at[n_chips - 1 : 2 * (n_chips - 1)]
            ag2_s = msc["rs_send_sem"].at[0 : n_chips - 1]
            b2_ag1_s = msc["fan_send_sem"].at[n_chips - 1 : n_chips]
            b2_ag2_s = msc["fan_send_sem"].at[0 : n_chips - 1]
            b2_ag3_s = msc["fan_send_sem"].at[n_chips : n_chips + 1]
        return dict(
            w_pre_sem=w_pre_sem, w_router_sem=w_router_sem, s_in_sem=s_in_sem, s_out_sem=s_out_sem,
            w_post_sem=w_post_sem, w_qkv_sem=w_qkv_sem, w_gate_sem=w_gate_sem,
            ar1a_s=ar1a_s, ar1a_r=ar1a_r, ar2b_s=ar2b_s, ar2b_r=ar2b_r,
            ag1_s=ag1_s, ag1_r=ag1_r, ag3_s=ag3_s, ag3_r=ag3_r,
            ar1b_s=ar1b_s, ar1b_r=ar1b_r, ar2a_s=ar2a_s, ar2a_r=ar2a_r,
            ag2_s=ag2_s, ag2_r=ag2_r,
            b2_ag1_s=b2_ag1_s, b2_ag2_s=b2_ag2_s, b2_ag3_s=b2_ag3_s, b2_ag2_r=b2_ag2_r,
        )

    def _run_stage1_4_rep_b1_moe(
        *,
        msc: dict,
        wo_buf,
        w_pre_buf,
        w_router_buf,
        s_buf,
        ffn_vmem,
        own_half,
        b1_ephem: tuple,
        post_shard,
        dummy_bf16,
        dummy_f32,
        w_post_buf,
        w_qkv_buf,
        w_gate_buf,
        sems: dict,
    ) -> None:
        (
            send_half,
            mate_recv,
            scatter_ref,
            own_chunk,
            sq_buf,
            pre_sh,
            logits_vmem,
            router_vmem,
            expert_in_vmem,
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
                zero_and_pack=True,
            )
            plan_holder.append(plan_res)

        _run_rep_b1_vmem_core(
            stream_hbm,
            gated_ref,
            wo_hbm,
            wo_scale_ref,
            pa_alpha_ref,
            pa_beta_ref,
            pre_ff_gain_ref,
            w_pre_hbm,
            w_pre_scale_ref,
            w_router_hbm,
            pre_exp_gain_ref,
            out_stream_hbm,
            logits_vmem,
            None if use_3d_x else expert_in_vmem,
            wo_buf,
            w_pre_buf,
            w_router_buf,
            s_buf,
            router_vmem,
            dummy_f32,
            ffn_vmem,
            send_half,
            mate_recv,
            own_half,
            scatter_ref,
            own_chunk,
            sq_buf,
            pre_sh,
            dummy_bf16,
            dummy_bf16,
            dummy_bf16,
            dummy_f32,
            msc["w1_sem"],
            sems["w_pre_sem"],
            sems["w_router_sem"],
            sems["s_in_sem"],
            sems["s_out_sem"],
            sems["ar1a_s"],
            sems["ar1a_r"],
            sems["ar1b_s"],
            sems["ar1b_r"],
            sems["ar2a_s"],
            sems["ar2a_r"],
            sems["ar2b_s"],
            sems["ar2b_r"],
            sems["ag1_s"],
            sems["ag1_r"],
            sems["ag2_s"],
            sems["ag2_r"],
            sems["ag3_s"],
            sems["ag3_r"],
            world_size=world_size,
            pa_eps=pa_eps,
            pre_ff_eps=pre_ff_eps,
            pre_ff_scale_input=pre_ff_scale_input,
            router_multiplier=router_multiplier,
            pre_exp_eps=pre_exp_eps,
            pre_exp_scale_input=pre_exp_scale_input,
            use_w8a8=False,
            skip_barrier_signal=False,
            skip_barrier_wait=False,
            skip_stream_load=False,
            skip_stream_store=True,
            during_ag_exp_ici_fn=_during_ag_exp_ici,
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
            output_ref=own_half,
            ep=world_size,
            batch_size=rows,
            scale_block=scale_block,
            return_sharded=False,
            ep_axis_name=ep_axis_name,
            cfg=cfg,
            x_is_3d=use_3d_x,
            wait_entry_barrier=False,
            write_output_ref=True,
            on_experts_done_fn=_on_experts_done,
        )

    def _run_stage5_6_rep_b2_rpad(
        *,
        moe_out_vmem,
        w_post_buf,
        w_qkv_buf,
        w_gate_buf,
        s_buf,
        ffn_vmem,
        qkv_vmem_slice,
        post_shard,
        half_a,
        half_b,
        dummy_bf16,
        sems: dict,
        rpad_refs: tuple | None,
        written_cnt_smem_ref=None,
        rpad_bkv_sems=None,
        rpad_cache_write_sem=None,
    ) -> None:
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

        _run_rep_b2_vmem_core(
            stream_hbm,
            moe_out_vmem,
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
            out_stream_hbm,
            b2_out_q,
            b2_out_kv,
            b2_out_gate_arg,
            w_post_buf,
            w_qkv_buf,
            w_gate_buf,
            s_buf,
            ffn_vmem,  # next_in_vmem (ffn_vmem dead since Stage 3)
            qkv_vmem_slice,
            post_shard,
            half_a,
            half_b,
            dummy_bf16,
            dummy_bf16,
            sems["w_post_sem"],
            sems["w_qkv_sem"],
            sems["w_gate_sem"],
            sems["s_in_sem"],
            sems["s_out_sem"],
            sems["b2_ag1_s"],
            sems["ag1_r"],
            sems["b2_ag2_s"],
            sems["b2_ag2_r"],
            sems["b2_ag3_s"],
            sems["ag3_r"],
            sems["ar1a_s"],
            sems["ar1a_r"],
            sems["ar2b_s"],
            sems["ar2b_r"],
            world_size=world_size,
            post_ff_eps=post_ff_eps,
            post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps,
            next_in_scale_input=next_in_scale_input,
            sharded_moe_in=False,
            qkv_norm_eps=qkv_norm_eps,
            query_scale=query_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            has_rope=has_rope,
            norm_before_s3=False,
            interpret_mode=interpret_mode,
            skip_barrier_signal=True,
            skip_barrier_wait=True,
            skip_stream_load=True,
            skip_stream_store=False,
            prestarted_w_post_dma=True,
            prestarted_w_qkv_dma=True,
            prestarted_w_gate_dma=True,
            gate_dst_ref=b2_gate_dst,
            after_qkv_epi_fn=b2_after_qkv_fn,
            defer_stream_store_wait=defer_store_wait,
        )

        if rpad_refs is not None:
            if rpa_kv_pack_bd4:
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
            _dma_wait(out_stream_hbm, sems["s_out_sem"].at[0])

    if not has_rpad:
        msc = _unpack_muse_sparse_scratch(
            it,
            ep=world_size,
            scale_block=scale_block,
            num_replicas=num_replicas,
            return_sharded=False,
            cfg=cfg,
        )
        if interpret_mode:
            wo_buf = next(it)
            w_pre_buf = next(it)
            w_post_buf = next(it)
            w_qkv_buf = next(it)
            w_gate_buf = next(it)
        else:
            w_b_buf = next(it)
            wo_buf = _alias_wo_buf(
                msc["w1_buf"],
                msc["w2_buf"],
                w_mode=msc["w_mode"],
                o_in=o_in,
                half=half,
            )
            w_pre_buf = w_b_buf.at[:, pl.ds(0, pre_cols)]
            w_qkv_buf = w_b_buf.at[:, pl.ds(0, qkv_cols)]
            w_gate_buf = w_b_buf.at[:, pl.ds(qkv_cols, gate_cols)]
            w_post_buf = (
                msc["w1_buf"]
                .bitcast(jnp.float8_e4m3fn)
                .reshape((_H, _H))
                .at[:, pl.ds(0, post_cols)]
            )

        w_router_buf = next(it)
        s_buf = next(it)
        ffn_vmem = next(it)
        send_half = next(it)
        mate_recv = next(it)
        own_half = next(it)
        scatter_ref = next(it)
        own_chunk = next(it)
        sq_buf = next(it)
        pre_sh = next(it)
        logits_vmem = next(it)
        router_vmem = next(it) if not is_r16 else logits_vmem
        post_shard = next(it)
        half_a = next(it)
        half_b = next(it)
        dummy_bf16 = next(it)
        dummy_f32 = next(it)
        expert_in_vmem = next(it) if not use_3d_x else None
        sems = _unpack_rep_sems(it, msc)

        _run_stage1_4_rep_b1_moe(
            msc=msc,
            wo_buf=wo_buf,
            w_pre_buf=w_pre_buf,
            w_router_buf=w_router_buf,
            s_buf=s_buf,
            ffn_vmem=ffn_vmem,
            own_half=own_half,
            b1_ephem=(
                send_half,
                mate_recv,
                scatter_ref,
                own_chunk,
                sq_buf,
                pre_sh,
                logits_vmem,
                router_vmem,
                expert_in_vmem,
            ),
            post_shard=post_shard,
            dummy_bf16=dummy_bf16,
            dummy_f32=dummy_f32,
            w_post_buf=w_post_buf,
            w_qkv_buf=w_qkv_buf,
            w_gate_buf=w_gate_buf,
            sems=sems,
        )
        _run_stage5_6_rep_b2_rpad(
            moe_out_vmem=own_half,
            w_post_buf=w_post_buf,
            w_qkv_buf=w_qkv_buf,
            w_gate_buf=w_gate_buf,
            s_buf=s_buf,
            ffn_vmem=ffn_vmem,
            qkv_vmem_slice=mate_recv.at[:, pl.ds(0, qkv_cols)],
            post_shard=post_shard,
            half_a=half_a,
            half_b=half_b,
            dummy_bf16=dummy_bf16,
            sems=sems,
            rpad_refs=None,
        )
        return

    # 2-Scope Replicated Step (c): Outer (live + sems) -> Scope 1 (b1 + MoE) -> Scope 2 (b2 + RPAd).
    moe_outer_refs = [next(it) for _ in range(step_c_n_moe_outer)]
    o_it_init = iter(moe_outer_refs)
    stitched_outer = [
        None if flg else next(o_it_init) for flg in step_c_moe_is_scope1
    ]
    msc_outer = _unpack_muse_sparse_scratch(
        iter(stitched_outer),
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=False,
        cfg=cfg,
    )
    if interpret_mode:
        w_post_outer, w_qkv_outer, w_gate_outer = next(it), next(it), next(it)
        w_b_outer = None
    else:
        w_b_outer = next(it)
        w_qkv_outer = w_b_outer.at[:, pl.ds(0, qkv_cols)]
        w_gate_outer = w_b_outer.at[:, pl.ds(qkv_cols, gate_cols)]
        w_post_outer = (
            msc_outer["w1_buf"]
            .bitcast(jnp.float8_e4m3fn)
            .reshape((_H, _H))
            .at[:, pl.ds(0, post_cols)]
        )
    s_buf = next(it)
    ffn_vmem = next(it)
    own_half = next(it)
    post_shard = next(it)
    half_a = next(it)
    half_b = next(it)
    dummy_bf16 = next(it)
    dummy_f32 = next(it)
    sems = _unpack_rep_sems(it, msc_outer)
    written_cnt_smem_ref = next(it)
    if world_size == 16:
        # fan_send_sem has 16 entries; Bridge 2b uses 0..8, leaving 9..13 free for RPAd.
        rpad_bkv_sems = msc_outer["fan_send_sem"].at[pl.ds(9, rpa_cfg.num_slots)]
        rpad_cache_write_sem = (
            None
            if rpa_skip_kv_write
            else msc_outer["fan_send_sem"].at[pl.ds(13, 1)]
        )
    else:
        rpad_bkv_sems = next(it)
        rpad_cache_write_sem = None if rpa_skip_kv_write else next(it)

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
            return_sharded=False,
            cfg=cfg,
        )
        s1_b_it = iter(b1_s1)
        w_router_buf = next(s1_b_it)
        send_half = next(s1_b_it)
        mate_recv = next(s1_b_it)
        scatter_ref = next(s1_b_it)
        own_chunk = next(s1_b_it)
        sq_buf = next(s1_b_it)
        pre_sh = next(s1_b_it)
        logits_vmem = next(s1_b_it)
        router_vmem = next(s1_b_it) if not is_r16 else logits_vmem
        expert_in_vmem = next(s1_b_it) if not use_3d_x else None
        if interpret_mode:
            wo_buf, w_pre_buf = b1_i_s1
        else:
            wo_buf = _alias_wo_buf(
                msc["w1_buf"],
                msc["w2_buf"],
                w_mode=msc["w_mode"],
                o_in=o_in,
                half=half,
            )
            w_pre_buf = w_b_outer.at[:, pl.ds(0, pre_cols)]

        _run_stage1_4_rep_b1_moe(
            msc=msc,
            wo_buf=wo_buf,
            w_pre_buf=w_pre_buf,
            w_router_buf=w_router_buf,
            s_buf=s_buf,
            ffn_vmem=ffn_vmem,
            own_half=own_half,
            b1_ephem=(
                send_half,
                mate_recv,
                scatter_ref,
                own_chunk,
                sq_buf,
                pre_sh,
                logits_vmem,
                router_vmem,
                expert_in_vmem,
            ),
            post_shard=post_shard,
            dummy_bf16=dummy_bf16,
            dummy_f32=dummy_f32,
            w_post_buf=w_post_outer,
            w_qkv_buf=w_qkv_outer,
            w_gate_buf=w_gate_outer,
            sems=sems,
        )

    pl.run_scoped(_run_scope1, *step_c_scope1_shapes)

    def _run_scope2(*s2_refs) -> None:
        s2_it = iter(s2_refs)
        qkv_vmem_slice = next(s2_it)
        rpad_q_vmem = next(s2_it)
        rpad_kv_vmem = None if rpa_skip_kv_write else next(s2_it)
        rpad_gate_vmem = next(s2_it)
        bkv_vmem_ref = next(s2_it)
        state_vmem_ref = next(s2_it)
        o_vmem_ref = next(s2_it)
        kv_write_vmem_ref = (
            next(s2_it)
            if ((rpa_kv_pack_dense2 or rpa_kv_pack_bd4) and not rpa_skip_kv_write)
            else None
        )
        init_sv_vmem_ref = (
            next(s2_it)
            if (getattr(rpa_cfg, "hoist_new_kv", False) or rpa_use_grp_vec)
            else None
        )
        _run_stage5_6_rep_b2_rpad(
            moe_out_vmem=own_half,
            w_post_buf=w_post_outer,
            w_qkv_buf=w_qkv_outer,
            w_gate_buf=w_gate_outer,
            s_buf=s_buf,
            ffn_vmem=ffn_vmem,
            qkv_vmem_slice=qkv_vmem_slice,
            post_shard=post_shard,
            half_a=half_a,
            half_b=half_b,
            dummy_bf16=dummy_bf16,
            sems=sems,
            rpad_refs=(
                rpad_q_vmem,
                rpad_kv_vmem,
                rpad_gate_vmem,
                bkv_vmem_ref,
                state_vmem_ref,
                o_vmem_ref,
                kv_write_vmem_ref,
                init_sv_vmem_ref,
            ),
            written_cnt_smem_ref=written_cnt_smem_ref,
            rpad_bkv_sems=rpad_bkv_sems,
            rpad_cache_write_sem=rpad_cache_write_sem,
        )

    pl.run_scoped(_run_scope2, *step_c_scope2_shapes)


def _b1_moe_b2_replicated_b16_kernel(*all_refs, **kwargs) -> None:
    """Fused replicated ``B=8/16`` Step (b) ``[rep_b1 + muse_sparse + rep_b2]`` kernel body."""
    _b1_moe_b2_replicated_b16_impl(all_refs, has_rpad=False, **kwargs)


def _b1_moe_b2_rpad_replicated_b16_kernel(*all_refs, **kwargs) -> None:
    """Fused replicated ``B=8/16`` Step (c) ``[rep_b1 + muse_sparse + rep_b2 + RPAd]`` kernel body."""
    _b1_moe_b2_replicated_b16_impl(all_refs, has_rpad=True, **kwargs)


def _prepare_rep_b16_setup(
    rank: jax.Array,
    stream: jax.Array,
    gated_attn: jax.Array,
    wo_w8k: jax.Array,
    wo_scale: jax.Array,
    pa_alpha: jax.Array,
    pa_beta: jax.Array,
    pre_ff_gain: jax.Array,
    w_pre_w8k: jax.Array,
    w_pre_scale: jax.Array,
    w_router_km: jax.Array,
    pre_exp_gain: jax.Array,
    bias: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_exp_norm: jax.Array,
    w_post_w8k: jax.Array,
    s_post: jax.Array,
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
) -> dict:
    """Validate shapes and build shared specs/scratches for replicated ``B=8/16`` Step (b)/(c)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.kernels.decode_collective_bridges import _use_dual_tree_xy

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2_replicated_b16 requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if not fuse_qkv_epi:
        raise ValueError("b1_moe_b2_replicated_b16 currently requires fuse_qkv_epi=True")

    rows, o_in = gated_attn.shape
    s_rows, hidden = stream.shape
    _, pre_cols = w_pre_w8k.shape
    _, n_exp = w_router_km.shape
    _, post_cols = w_post_w8k.shape
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape
    expected_chunk = hidden // world_size
    if (
        rows not in (8, 16)
        or s_rows != rows
        or hidden != 8192
        or world_size * pre_cols != _H
        or world_size * post_cols != hidden
        or w_post_w8k.shape[0] != _H
        or n_exp != _E_TOTAL
        or o_in != expected_chunk
    ):
        raise ValueError(
            f"Unsupported replicated B=8/16 shapes world_size={world_size}, stream={stream.shape}, "
            f"gated={gated_attn.shape}, w_pre={w_pre_w8k.shape}, w_post={w_post_w8k.shape}"
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
        base_cfg = default_sparse_decode_config(rows, world_size, scale_block)
        # TP=8 uses pair_rs=True (total 60 sems); TP=16 uses flat RS+AG (38+22=60 <= 64).
        config = dataclasses.replace(
            base_cfg,
            weight_mode="stream3",
            pair_rs=(world_size == 8),
            bridge_ag=False,
        )
    else:
        if world_size == 16 and (config.pair_rs or config.bridge_ag):
            config = dataclasses.replace(config, pair_rs=False, bridge_ag=False)
    if rows == 8 and config.row_mode != "sparse":
        raise ValueError("b1_moe_b2_replicated_b16 with B=8 requires config.row_mode='sparse'")

    half = hidden // 2
    n_chips = world_size // 2
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_ar_sems = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)
    is_r16 = w_router_km.dtype == jnp.bfloat16
    has_rope = bool(cos_sin_step is not None)
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
        return_sharded=False,
        config=config,
    )
    use_3d_x = config.row_mode != "dense"
    weight_scratch = (
        [
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_cols), w_pre_w8k.dtype),
            vmem((_H, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
        ]
        if interpret_mode
        else [
            vmem((hidden, max(pre_cols, qkv_cols + gate_cols)), w_pre_w8k.dtype),
        ]
    )
    v_w_router_buf = vmem((hidden, n_exp), w_router_km.dtype)
    v_s_buf = vmem((rows, hidden), jnp.float32)
    v_ffn_vmem = vmem((rows, hidden), jnp.bfloat16)
    v_send_half = vmem((rows, half), jnp.bfloat16)
    v_mate_recv = vmem((rows, half), jnp.bfloat16)
    v_own_half = vmem((rows, half), jnp.bfloat16)
    v_scatter_ref = vmem((n_sc_slots, rows, post_cols), jnp.bfloat16)
    v_own_chunk = vmem((8, 128) if use_xy else (rows, post_cols), jnp.bfloat16)
    v_sq_buf = vmem((rows, 128), jnp.float32)
    v_pre_sh = vmem((rows, pre_cols), jnp.bfloat16)
    v_logits_vmem = vmem((rows, n_exp), jnp.float32)
    v_router_vmem_list = [] if is_r16 else [vmem((rows, hidden), jnp.float32)]
    v_post_shard = vmem((rows, post_cols), jnp.bfloat16)
    v_half_a = vmem((rows, hidden), jnp.bfloat16)
    v_half_b = vmem((8, 128), jnp.bfloat16)
    v_dummy_bf16 = vmem((8, 128), jnp.bfloat16)
    v_dummy_f32 = vmem((8, 128), jnp.float32)
    v_expert_in_list = [] if use_3d_x else [vmem((rows, _H), jnp.bfloat16)]
    rep_sems = [
        sem((4,)),  # b1_sems [w_pre/post, w_router/qkv, s_in/gate, s_out]
        *(
            [
                sem((1,)), sem((1,)),
                sem((1,)), sem((1,)),
                sem((1,)), sem((1,)),
                sem((1,)), sem((1,)),
                sem((n_ar_sems,)), sem((n_ar_sems,)),
                sem((n_ar_sems,)), sem((n_ar_sems,)),
                sem((n_chips - 1,)), sem((n_chips - 1,)),
            ]
            if world_size == 8
            else [
                sem((1,)), sem((1,)),
                sem((1,)), sem((1,)),
                sem((n_ar_sems,)), sem((n_ar_sems,)),
            ]
        ),
    ]
    shared_scratch = [
        *weight_scratch,
        v_w_router_buf,
        v_s_buf,
        v_ffn_vmem,
        v_send_half,
        v_mate_recv,
        v_own_half,
        v_scatter_ref,
        v_own_chunk,
        v_sq_buf,
        v_pre_sh,
        v_logits_vmem,
        *v_router_vmem_list,
        v_post_shard,
        v_half_a,
        v_half_b,
        v_dummy_bf16,
        v_dummy_f32,
        *v_expert_in_list,
        *rep_sems,
    ]

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
    w_outer_vmem = (
        [
            vmem((_H, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
        ]
        if interpret_mode
        else [
            vmem((hidden, max(pre_cols, qkv_cols + gate_cols)), w_pre_w8k.dtype),
        ]
    )
    w_scope1_vmem = (
        [
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_cols), w_pre_w8k.dtype),
        ]
        if interpret_mode
        else []
    )
    outer_shared_vmem = [
        v_s_buf,
        v_ffn_vmem,
        v_own_half,
        v_post_shard,
        v_half_a,
        v_half_b,
        v_dummy_bf16,
        v_dummy_f32,
    ]
    b1_scope1_vmem = [
        v_w_router_buf,
        v_send_half,
        v_mate_recv,
        v_scatter_ref,
        v_own_chunk,
        v_sq_buf,
        v_pre_sh,
        v_logits_vmem,
        *v_router_vmem_list,
        *v_expert_in_list,
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

    in_specs = (
        (h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp)
        + ((v_sp, h_sp, h_sp, h_sp, h_sp, v_sp) + ((h_sp, h_sp, h_sp, h_sp) if has_replicas else ()))
        + (h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp)
        + ((v_sp,) if has_rope else ())
    )
    b1_args = (
        stream.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_dot_scale_f32(wo_scale, hidden),
        _as_2d_f32(pa_alpha, hidden),
        _as_2d_f32(pa_beta, hidden),
        _as_2d_f32(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_dot_scale_f32(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32(pre_exp_gain, _H),
    )
    moe_args = (bias_2d, w1, s1, w2, s2, norm_2d) + (
        (rep_w1, rep_s1, rep_w2, rep_s2) if has_replicas else ()
    )
    b2_args = (
        w_post_w8k,
        _as_dot_scale_f32(s_post, post_cols),
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
        o_in=o_in,
        hidden=hidden,
        qkv_cols=qkv_cols,
        gate_cols=gate_cols,
        scale_block=scale_block,
        num_replicas=num_replicas,
        is_r16=is_r16,
        has_rope=has_rope,
        config=config,
        interpret_arg=interpret_arg,
        interpret_mode=interpret_mode,
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
        step_c_outer_scratch=moe_outer + w_outer_vmem + outer_shared_vmem + rep_sems,
        step_c_moe_is_scope1=moe_is_scope1,
        step_c_n_moe_outer=len(moe_outer),
        step_c_n_moe_scope1=len(moe_scope1),
        step_c_n_b1_scope1=len(b1_scope1_vmem),
        step_c_scope1_shapes=tuple(moe_scope1 + b1_scope1_vmem + w_scope1_vmem),
    )


def b1_moe_b2_replicated_b16_jax(
    rank: jax.Array,
    stream: jax.Array,
    gated_attn: jax.Array,
    wo_w8k: jax.Array,
    wo_scale: jax.Array,
    pa_alpha: jax.Array,
    pa_beta: jax.Array,
    pre_ff_gain: jax.Array,
    w_pre_w8k: jax.Array,
    w_pre_scale: jax.Array,
    w_router_km: jax.Array,
    pre_exp_gain: jax.Array,
    bias: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_exp_norm: jax.Array,
    w_post_w8k: jax.Array,
    s_post: jax.Array,
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
    collective_id: int = _W8_CID_BASE + 24,
    interpret: bool = False,
):
    """Fused replicated ``B=8/16`` Step (b) superkernel: ``attn_{L} -> MoE_{L} -> attn_{L+1}``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    st = _prepare_rep_b16_setup(
        rank, stream, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
        pre_ff_gain, w_pre_w8k, w_pre_scale, w_router_km, pre_exp_gain,
        bias, w1, s1, w2, s2, post_exp_norm,
        w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
        cos_sin_step, rep_w1, rep_s1, rep_w2, rep_s2, replica_meta,
        spec_home_slot, free_params,
        world_size=world_size, fuse_qkv_epi=fuse_qkv_epi, config=config, interpret=interpret,
    )
    rows = st["rows"]
    hidden = st["hidden"]
    gate_cols = st["gate_cols"]
    v_sp, h_sp = st["v_sp"], st["h_sp"]
    scalar_args = st["scalar_args"]
    call_args = (*scalar_args, *st["base_call_tensors"])

    out_shape = (
        jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, 16, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 1, 4, 128), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
    )
    out_specs = (h_sp, v_sp, v_sp, v_sp)

    kernel_fn = functools.partial(
        _b1_moe_b2_replicated_b16_kernel,
        world_size=world_size,
        scale_block=st["scale_block"],
        num_replicas=st["num_replicas"],
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
        interpret_mode=st["interpret_mode"],
    )

    st_out, q_3d, kv_m, gate = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            grid=(1,),
            in_specs=st["in_specs"],
            out_specs=out_specs,
            scratch_shapes=st["scratch_shapes"],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=60 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=st["interpret_arg"],
        name=(
            f"muse_w9_b1_moe_b2_rep_b{rows}_tp{world_size}_k{st['scale_block']}"
            f"_r{st['num_replicas']}_{'r16' if st['is_r16'] else 'r32'}"
            f"{'_dfp' if st['has_free_params_arg'] else ''}"
            f"_{'rope' if st['has_rope'] else 'nope'}"
        ),
    )(*call_args)
    return st_out, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate


def b1_moe_b2_rpad_replicated_b16_jax(
    rank: jax.Array,
    stream: jax.Array,
    gated_attn: jax.Array,
    wo_w8k: jax.Array,
    wo_scale: jax.Array,
    pa_alpha: jax.Array,
    pa_beta: jax.Array,
    pre_ff_gain: jax.Array,
    w_pre_w8k: jax.Array,
    w_pre_scale: jax.Array,
    w_router_km: jax.Array,
    pre_exp_gain: jax.Array,
    bias: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_exp_norm: jax.Array,
    w_post_w8k: jax.Array,
    s_post: jax.Array,
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
    collective_id: int = _W8_CID_BASE + 26,
    interpret: bool = False,
    skip_kv_write: bool = False,
):
    """Fused replicated ``B=8/16`` Step (c) superkernel: ``[rep_b1 + muse_sparse + rep_b2 + RPAd]``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import align_to, cdiv
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    if kv_pack_bd4:
        if world_size != 16:
            raise ValueError(
                f"Replicated Step (c) kv_pack_bd4 requires world_size=16, got {world_size}"
            )
        kv_pack_dense2 = False

    st = _prepare_rep_b16_setup(
        rank, stream, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
        pre_ff_gain, w_pre_w8k, w_pre_scale, w_router_km, pre_exp_gain,
        bias, w1, s1, w2, s2, post_exp_norm,
        w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta, next_in_gain,
        next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
        cos_sin_step, rep_w1, rep_s1, rep_w2, rep_s2, replica_meta,
        spec_home_slot, free_params,
        world_size=world_size, fuse_qkv_epi=True, config=config, interpret=interpret,
    )
    rows = st["rows"]
    hidden = st["hidden"]
    qkv_cols = st["qkv_cols"]

    eff_sw = (
        None
        if (rpa_sliding_window is None or rpa_sliding_window <= 0)
        else int(rpa_sliding_window)
    )
    kv_cache_flat = kv_cache_fused
    eff_page_indices = page_indices
    if layer_idx is not None:
        num_blocks, layer_num = kv_cache_fused.shape[:2]
        kv_cache_flat = kv_cache_fused.reshape(
            (num_blocks * layer_num,) + kv_cache_fused.shape[2:]
        )
        eff_page_indices = (
            page_indices * jnp.int32(layer_num) + layer_idx.reshape(()).astype(jnp.int32)
        )

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
    )
    if not is_ok:
        raise ValueError(
            "Replicated Step (c) requires RPAd bkv within VMEM budget and compatible shared_group; "
            f"got num_slots={cfg_rpa.num_slots}, max_ring_pages={cfg_rpa.max_ring_pages}, "
            f"phys_page_rows={phys_page_rows}, loop_kind={cfg_rpa.loop_kind!r}, B={rows}"
        )
    if world_size == 16 and cfg_rpa.num_slots > 4:
        raise ValueError(
            f"TP=16 Replicated Step (c) aliases bkv_sems on fan_send_sem[9:13]; requires num_slots <= 4, got {cfg_rpa.num_slots}"
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
        jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        jax.ShapeDtypeStruct((rows, pack2_cols), jnp.bfloat16),
        kv_out_shape,
    )
    out_specs = (h_sp, v_sp, kv_out_spec)
    pack_tag = "_bd4" if kv_pack_bd4 else ("_d2" if kv_pack_dense2 else "")

    def _invoke_step_c(run_cfg, run_one_blk: bool, run_slot_pg: int, run_grp_vec: bool):
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
        scope2_vmem = (
            vmem((rows, qkv_cols), jnp.bfloat16),  # qkv_vmem_slice
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
            _b1_moe_b2_rpad_replicated_b16_kernel,
            world_size=world_size,
            scale_block=st["scale_block"],
            num_replicas=st["num_replicas"],
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
            step_c_moe_is_scope1=st["step_c_moe_is_scope1"],
            step_c_n_moe_outer=st["step_c_n_moe_outer"],
            step_c_n_moe_scope1=st["step_c_n_moe_scope1"],
            step_c_n_b1_scope1=st["step_c_n_b1_scope1"],
            step_c_scope1_shapes=st["step_c_scope1_shapes"],
            step_c_scope2_shapes=scope2_vmem,
        )
        return pl.pallas_call(
            kernel_fn,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_args),
                grid=(1,),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=st["step_c_outer_scratch"] + outer_rpad_meta,
            ),
            input_output_aliases=pallas_aliases,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=63 * 1024**2,
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            interpret=st["interpret_arg"],
            name=(
                f"muse_w9_b1_moe_b2_rep_rpad_b{rows}_tp{world_size}_k{st['scale_block']}"
                f"_r{st['num_replicas']}_{'r16' if st['is_r16'] else 'r32'}"
                f"{'_dfp' if st['has_free_params_arg'] else ''}"
                f"{pack_tag}"
                f"_{'rope' if st['has_rope'] else 'nope'}"
            ),
        )(*call_args)

    # Adaptive 1K (lax.cond) is disabled: wrapping the 16 Full-Attn Step (c)
    # calls in HLO conditionals shatters XLA code deduplication across the 45
    # SWA rope calls into 4 separate IMEM binaries (+17..20 us/L cold reload).
    st_out, next_gated, kv_out = _invoke_step_c(
        cfg_rpa, one_block_per_seq, bkv_slot_pages, use_grp_vec
    )
    return st_out, next_gated, (kv_out if skip_kv_write else kv_out.reshape(kv_cache_fused.shape))
