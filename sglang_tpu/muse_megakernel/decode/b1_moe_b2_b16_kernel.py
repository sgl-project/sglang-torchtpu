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

import functools
import jax
import jax.numpy as jnp

from sglang_tpu.kernels.decode_collective_bridges import _run_rep_b1_vmem_core
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


def _b1_moe_b2_replicated_b16_kernel(
    *all_refs,
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
) -> None:
    """Fused replicated ``B=16`` Step (b) ``[rep_b1 + muse_sparse + rep_b2]`` kernel body."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.kernels.decode_collective_bridges import _use_dual_tree_xy

    it = iter(all_refs)
    # Scalar prefetch.
    rank_ref = next(it)
    replica_meta_ref = next(it) if num_replicas > 0 else None
    spec_slot_ref = next(it) if has_spec_slot_arg else None
    free_params_ref = next(it) if has_free_params_arg else None

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

    # Outputs.
    out_stream_hbm = next(it)
    out_qkv_or_q_ref = next(it)
    out_kv_or_gate_ref = next(it)
    out_gate_epi_ref = next(it) if fuse_qkv_epi else None

    # MoE scratch.
    msc = _unpack_muse_sparse_scratch(
        it,
        ep=world_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        return_sharded=False,
        cfg=cfg,
    )

    rows, o_in = gated_ref.shape
    hidden = stream_hbm.shape[1]
    half = hidden // 2
    pre_cols = w_pre_hbm.shape[1]
    post_cols = w_post_hbm.shape[1]
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    is_r16 = w_router_hbm.dtype == jnp.bfloat16
    use_3d_x = msc["x_3d_buf"] is not None
    use_xy = _use_dual_tree_xy(world_size, rows, hidden)
    n_chips = world_size // 2

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

    b1_sems = next(it)
    w_pre_sem = b1_sems.at[0:1]
    w_router_sem = b1_sems.at[1:2]
    s_in_sem = b1_sems.at[2:3]
    s_out_sem = b1_sems.at[3:4]
    w_post_sem = w_pre_sem
    w_qkv_sem = w_router_sem
    w_gate_sem = s_in_sem

    if world_size == 8:
        ar1a_s, ar1a_r = next(it), next(it)
        ar2b_s, ar2b_r = next(it), next(it)
        ag1_s, ag1_r = next(it), next(it)
        ag3_s, ag3_r = next(it), next(it)
        ar1b_s, ar1b_r = next(it), next(it)
        ar2a_s, ar2a_r = next(it), next(it)
        ag2_s, ag2_r = next(it), next(it)
        b2_ag1_s, b2_ag2_s, b2_ag3_s = ag1_s, ag2_s, ag3_s
        b2_ag2_r = ag2_r
    else:
        # TP=16 B=16: MoE uses flat RS+AG (38 sems); reuse local-drained send sems +
        # phase-separated ar1b_r for ag2_r so total semaphores = 60 <= 64.
        ar1a_r = next(it)
        ar2b_r = next(it)
        ag1_r = next(it)
        ag3_r = next(it)
        ar1b_r = next(it)
        ar2a_r = next(it)
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
        w_pre_sem,
        w_router_sem,
        s_in_sem,
        s_out_sem,
        ar1a_s,
        ar1a_r,
        ar1b_s,
        ar1b_r,
        ar2a_s,
        ar2a_r,
        ar2b_s,
        ar2b_r,
        ag1_s,
        ag1_r,
        ag2_s,
        ag2_r,
        ag3_s,
        ag3_r,
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
        pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
        pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
        pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    moe_out_vmem = own_half
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
        output_ref=moe_out_vmem,
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

    qkv_vmem_slice = mate_recv.at[:, pl.ds(0, qkv_cols)]
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
        out_qkv_or_q_ref,
        out_kv_or_gate_ref,
        out_gate_epi_ref,
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
        w_post_sem,
        w_qkv_sem,
        w_gate_sem,
        s_in_sem,
        s_out_sem,
        b2_ag1_s,
        ag1_r,
        b2_ag2_s,
        b2_ag2_r,
        b2_ag3_s,
        ag3_r,
        ar1a_s,
        ar1a_r,
        ar2b_s,
        ar2b_r,
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
    """Fused replicated ``B=16`` Step (b) superkernel: ``attn_{L} -> MoE_{L} -> attn_{L+1}``."""
    import dataclasses
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.kernels.decode_collective_bridges import _use_dual_tree_xy

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2_replicated_b16_jax requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if not fuse_qkv_epi:
        raise ValueError("b1_moe_b2_replicated_b16_jax currently requires fuse_qkv_epi=True")

    rows, o_in = gated_attn.shape
    s_rows, hidden = stream.shape
    _, pre_cols = w_pre_w8k.shape
    _, n_exp = w_router_km.shape
    _, post_cols = w_post_w8k.shape
    _, qkv_cols = next_w_qkv_w8k.shape
    _, gate_cols = next_w_gate_w8k.shape
    expected_chunk = hidden // world_size
    valid_rows = (8, 16) if world_size == 8 else (16,)
    if (
        rows not in valid_rows
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
        raise ValueError("b1_moe_b2_replicated_b16_jax with B=8 requires config.row_mode='sparse'")

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
    shared_scratch = [
        *weight_scratch,
        vmem((hidden, n_exp), w_router_km.dtype),
        vmem((rows, hidden), jnp.float32),  # s_buf
        vmem((rows, hidden), jnp.bfloat16),  # ffn_vmem / next_in_vmem
        vmem((rows, half), jnp.bfloat16),  # send_half
        vmem((rows, half), jnp.bfloat16),  # mate_recv (qkv_vmem_slice aliases [: , :qkv_cols])
        vmem((rows, half), jnp.bfloat16),  # own_half / moe_out_vmem
        vmem((n_sc_slots, rows, post_cols), jnp.bfloat16),  # scatter_ref
        vmem((8, 128) if use_xy else (rows, post_cols), jnp.bfloat16),  # own_chunk
        vmem((rows, 128), jnp.float32),  # sq_buf
        vmem((rows, pre_cols), jnp.bfloat16),  # pre_sh
        vmem((rows, n_exp), jnp.float32),  # logits_vmem
        *(() if is_r16 else [vmem((rows, hidden), jnp.float32)]),  # router_vmem
        vmem((rows, post_cols), jnp.bfloat16),  # post_shard
        vmem((rows, hidden), jnp.bfloat16),  # half_a (B=16 full gather in _gather_shard_xor1)
        vmem((8, 128), jnp.bfloat16),  # half_b (unused when use_halfrow=False)
        vmem((8, 128), jnp.bfloat16),  # dummy_bf16
        vmem((8, 128), jnp.float32),  # dummy_f32
        *(() if use_3d_x else [vmem((rows, _H), jnp.bfloat16)]),
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
    call_args = (*scalar_args, *b1_args, *moe_args, *b2_args)

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
        scale_block=scale_block,
        num_replicas=num_replicas,
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
        has_rope=has_rope,
        cfg=config,
        has_spec_slot_arg=has_spec_slot_arg,
        has_free_params_arg=has_free_params_arg,
        interpret_mode=interpret_mode,
    )

    st_out, q_3d, kv_m, gate = pl.pallas_call(
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
            vmem_limit_bytes=60 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        interpret=interpret_arg,
        name=(
            f"muse_w9_b1_moe_b2_rep_b{rows}_tp{world_size}_k{scale_block}"
            f"_r{num_replicas}_{'r16' if is_r16 else 'r32'}"
            f"{'_dfp' if has_free_params_arg else ''}"
            f"_{'rope' if has_rope else 'nope'}"
        ),
    )(*call_args)
    return st_out, q_3d.reshape(1, rows, 8, 2, 128), kv_m, gate
