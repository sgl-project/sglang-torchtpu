# SPDX-License-Identifier: Apache-2.0
"""Full-Layer Fuse-All Decode MoE Pallas Kernel for Muse Spark (Stage B + Core + Stage A).

Fuses the entire ``MuseSparkMoE.forward`` (and optionally the outer
``post_feedforward_layernorm`` + ``MuseSparkGatedResidual`` + next-layer
``input_layernorm``) into a SINGLE TPU7x Pallas launch across ``EP in {8, 16}``,
``B in {16, 32, 64, 128, 256}``, Compact K256 & Native K16 FP4 weights, and
optional §4.5b Preloaded Hot-Expert Replicas (``R in {0, 1, 2, 4}``):

  - **Stage B (Prologue)**:
    Supports two schedules:
      * ``"gate_first"`` (default): Streams ``router_input [B, 8192] f32`` +
        ``gate.weight`` first (staging ``hidden_states [B, 8192] bf16`` in dead
        ``post_ag_vmem``), immediately runs in-kernel tie-exact top-8 routing +
        SMEM §4.5b replica planning while ``pre_expert_proj`` weight streams into
        ``proj_w_buf``, and **primes Expert-0's 24-MiB FP4 weights BEFORE**
        ``pre_expert_proj`` + Column All-Gather + ``pre_expert_norm`` so 100% of
        Expert-0 cold-start HBM weight streaming and ``pre_expert_proj`` AG
        overlap!
      * ``"interleaved"``: Computes ``gate`` + ``pre_expert_proj`` in one pass,
        launches ``pre_expert_proj`` Column AG before routing, and overlaps
        routing + SMEM planning with ICI AG.
    Natively accepts both PyTorch ``(out, in)`` and pre-transposed ``(in, out)``
    projection weights with zero external XLA transposes.
  - **MoE Core**:
    Sparse (or ``B=16`` K256 dense-lowtok) expert execution reusing the
    verified Track-1 compute/gather/scatter/RS/bridge-AG primitives from
    ``fused_decode_sparse.py`` and ``fused_decode_routing.py``.
  - **Stage A (Epilogue)**:
    Prefetches ``post_expert_proj`` weight (and ``stream`` when fusing residual)
    DURING Expert Reduce-Scatter + All-Gather (0 us exposed HBM wait), computes
    ``moe_out @ w_post -> [B, 8192 // EP] bf16`` in VMEM, runs a zero-transpose
    in-VMEM 1-hop / 2-stage chipmate-bridge Column All-Gather into
    ``[EP, B, 8192 // EP]``, and either writes ``ffn_output [B, 8192] bf16``
    directly to HBM or fuses ``post_feedforward_layernorm`` +
    ``MuseSparkGatedResidual`` (+ optional ``next_input_layernorm``) in VMEM.

``fused_decode_sparse.py`` and ``fused_decode_routing.py`` remain 100% untouched.
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
    DenseRoutingScratch,
    RoutingScratch,
    build_default_exec_plan_smem,
    build_dense_exec_plan_smem,
    build_exec_plan_from_masks_smem,
    dense_routing_scratch_shapes,
    pack_active_tokens_smem,
    route_dense_and_counts_in_kernel,
    routing_scratch_shapes,
    run_replica_planner_smem,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
    SparseDecodeConfig,
    _build_masked_x_from_2d,
    _build_masked_x_from_3d,
    _fp4_blockdiag_hoisted_dual_strip,
    _fp4_blockdiag_hoisted_strip,
    _fp4_dual_gmm_chunked,
    _fp4_gmm_chunked,
    _pad_rows_to,
    _pick_bd_nc,
    _pick_k256_nc,
    default_sparse_decode_config,
)

_FP4_PACK = 8
_H_MODEL = 8192
_H = 4096
_I = 4096
_N1 = 8192
_E_TOTAL = 256
_TOPK = 8
_K_STAGE_B = 1024


@dataclasses.dataclass(frozen=True)
class FullLayerDecodeConfig:
    """Static compile-time configuration for ``fused_decode_full_layer``."""

    fuse_stage_b: bool = True  # Fuse gate + pre_expert_proj + col AG + pre_expert_norm
    fuse_stage_a: bool = True  # Fuse post_expert_proj + col AG
    fuse_post_ffn_res: bool = False  # Also fuse post_ffn_norm + gated residual
    fuse_next_norm: bool = False  # Also fuse next layer's input_layernorm
    stage_b_schedule: str = "gate_first"  # "gate_first" (#408 #6) or "interleaved"
    bridge_pre_ag: bool = False  # 2-stage chipmate-bridge AG on pre_expert_proj
    bridge_post_ag: bool = True  # 2-stage chipmate-bridge AG on post_expert_proj
    gate_highest_precision: bool = True  # IEEE FP32 3-pass MXU for router gate
    debug_stop_after: str = ""  # "", "route", "plan", "pack", "experts", or "rs"
    sparse_cfg: SparseDecodeConfig | None = None


def default_full_layer_decode_config(
    batch_size: int,
    ep: int,
    scale_block: int,
    *,
    fuse_stage_b: bool = True,
    fuse_stage_a: bool = True,
    fuse_post_ffn_res: bool = False,
    fuse_next_norm: bool = False,
    stage_b_schedule: str | None = None,
) -> FullLayerDecodeConfig:
    """Return the hardware-tuned default full-layer configuration for ``(B, ep, scale_block)``."""
    sp_cfg = default_sparse_decode_config(batch_size, ep, scale_block)
    if ep == 8 and batch_size == 64:
        # Wave-1 verified (#466): 2-stage XOR=1 chipmate-bridge AG + bf16 RS wire at B=64
        # saves -6.2 us/layer on K256 and -2.4..-3.6 us/layer on K16 across all regimes!
        sp_cfg = dataclasses.replace(sp_cfg, bridge_ag=True, rs_wire_bf16=True)
    sched = (
        ("gate_first" if batch_size <= 32 else "interleaved")
        if stage_b_schedule is None
        else stage_b_schedule
    )
    return FullLayerDecodeConfig(
        fuse_stage_b=fuse_stage_b,
        fuse_stage_a=fuse_stage_a,
        fuse_post_ffn_res=fuse_post_ffn_res,
        fuse_next_norm=fuse_next_norm,
        stage_b_schedule=sched,
        bridge_pre_ag=(ep == 8 and batch_size >= 64),
        bridge_post_ag=(ep == 8 and batch_size >= 64),
        gate_highest_precision=(batch_size <= 64),
        sparse_cfg=sp_cfg,
    )


def _col_all_gather_start(
    me: jax.Array,
    ep: int,
    ep_axis_name: str,
    shard_vmem,
    ag_vmem,
    send_sem,
    recv_sem,
    *,
    use_bridge: bool,
):
    """Start 1-hop or 2-stage chipmate-bridge Column All-Gather into ``ag_vmem[ep, B, C]``."""
    ag_vmem[me] = shard_vmem[...]
    if use_bridge:
        num_chips = ep // 2
        my_chip = me >> 1
        my_par = me & 1
        mate = jnp.bitwise_xor(me, jnp.int32(1))
        mate_op = pltpu.make_async_remote_copy(
            shard_vmem,
            ag_vmem.at[me],
            send_sem.at[0],
            recv_sem,
            device_id={ep_axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        mate_op.start()
        s1_ops = []
        for c in range(num_chips):
            dest = jnp.int32(c * 2) + my_par
            op = pltpu.make_async_remote_copy(
                shard_vmem,
                ag_vmem.at[me],
                send_sem.at[1 + c],
                recv_sem,
                device_id={ep_axis_name: dest},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s1_ops.append((c, op))

            @pl.when(my_chip != c)
            def _st(_op=op):
                _op.start()

        return ("bridge", mate_op, s1_ops)

    flat_ops = []
    for dest in range(ep):
        op = pltpu.make_async_remote_copy(
            shard_vmem,
            ag_vmem.at[me],
            send_sem.at[dest],
            recv_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        flat_ops.append((dest, op))

        @pl.when(me != dest)
        def _st_f(_op=op):
            _op.start()

    return ("flat", flat_ops)


def _col_all_gather_finish(
    me: jax.Array,
    ep: int,
    ep_axis_name: str,
    ag_vmem,
    recv_sem,
    s2_send_sem,
    s2_recv_sem,
    handle,
) -> list:
    """Wait for Column All-Gather completion (and run Stage-2 bridge hop if enabled)."""
    kind = handle[0]
    if kind == "bridge":
        _, mate_op, s1_ops = handle
        num_chips = ep // 2
        my_chip = me >> 1
        my_par = me & 1
        mate = jnp.bitwise_xor(me, jnp.int32(1))
        pltpu.make_async_copy(
            ag_vmem.at[pl.ds(0, num_chips)],
            ag_vmem.at[pl.ds(0, num_chips)],
            recv_sem,
        ).wait()
        s2_ops = []
        for c in range(num_chips):
            src_rank = jnp.int32(c * 2) + my_par
            op2 = pltpu.make_async_remote_copy(
                ag_vmem.at[src_rank],
                ag_vmem.at[src_rank],
                s2_send_sem.at[c],
                s2_recv_sem,
                device_id={ep_axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s2_ops.append((c, op2))

            @pl.when(my_chip != c)
            def _st2(_op=op2):
                _op.start()

        pltpu.make_async_copy(
            ag_vmem.at[pl.ds(0, num_chips - 1)],
            ag_vmem.at[pl.ds(0, num_chips - 1)],
            s2_recv_sem,
        ).wait()
        return [("bridge", my_chip, mate_op, s1_ops, s2_ops)]

    _, flat_ops = handle
    pltpu.make_async_copy(
        ag_vmem.at[pl.ds(0, ep - 1)],
        ag_vmem.at[pl.ds(0, ep - 1)],
        recv_sem,
    ).wait()
    return [("flat", me, flat_ops)]


def _col_all_gather_wait_sends(wait_records: list) -> None:
    for rec in wait_records:
        if rec[0] == "bridge":
            _, my_chip, mate_op, s1_ops, s2_ops = rec
            mate_op.wait_send()
            for c, op in s1_ops:

                @pl.when(my_chip != c)
                def _w1(_op=op):
                    _op.wait_send()

            for c, op2 in s2_ops:

                @pl.when(my_chip != c)
                def _w2(_op=op2):
                    _op.wait_send()

        else:
            _, me, flat_ops = rec
            for dest, op in flat_ops:

                @pl.when(me != dest)
                def _wf(_op=op):
                    _op.wait_send()


def _extract_holder_cand_eids_scalar(
    my_owner: jax.Array,
    num_replicas: int,
    replica_meta_smem,
) -> tuple:
    """Branchless scalar extraction of ``my_owner``'s ``R`` holder expert IDs (#544).

    Returns ``-2`` for unused replica slots so ``best == rep_vec`` (where
    ``best >= 0``) rejects unused slots with zero extra comparisons.
    """
    k_rep = replica_meta_smem.shape[1]
    out = []
    for s in range(num_replicas):
        acc = jnp.int32(0)
        for k in range(k_rep):
            eid_k = replica_meta_smem[0, k]
            mine_s = jnp.logical_and(
                eid_k >= 0,
                jnp.logical_and(
                    replica_meta_smem[2, k] == my_owner,
                    replica_meta_smem[3, k] == s,
                ),
            )
            acc = acc + jnp.where(mine_s, eid_k + jnp.int32(2), jnp.int32(0))
        out.append(acc - jnp.int32(2))
    return tuple(out)


def _route_topk_fast_in_kernel(
    logits_ref,
    bias_ref,
    scratch: RoutingScratch,
    dma_sem,
    *,
    my_owner: jax.Array,
    e_per_owner: int,
    holder_cand_eids: tuple = (),
    topk: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    route_chunk_rows: int = 32,
) -> None:
    """VPU-filtered top-k routing with hoisted per-chunk replica VREGs (#544)."""
    batch_size, num_experts = logits_ref.shape
    rb = min(batch_size, route_chunk_rows)
    if batch_size % rb != 0:
        rb = 8 if batch_size % 8 == 0 else batch_size

    first = my_owner * e_per_owner
    last_excl = first + e_per_owner
    bias_row = bias_ref[...].reshape(1, num_experts).astype(jnp.float32)
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)
    counts_init = jnp.zeros((1, num_experts), jnp.int32)
    neg_inf = jnp.float32(-jnp.inf)

    def _route_chunk(r0, counts_acc):
        row_s = pl.ds(r0, rb)
        lg_raw = logits_ref[row_s, :].astype(jnp.float32)
        lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
        scores = 1.0 / (1.0 + jnp.exp(-lg))
        choice = scores + bias_row
        pack_ids = jnp.zeros((rb, 128), dtype=jnp.int32)
        w_all_8 = jnp.zeros((rb, 128), dtype=jnp.float32)
        w_loc_raw = jnp.zeros((rb, 128), dtype=jnp.float32)
        loc_cnt = jnp.zeros((rb, 1), dtype=jnp.int32)
        hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)
        first_vec = jnp.broadcast_to(first, (rb, 1))
        last_vec = jnp.broadcast_to(last_excl, (rb, 1))
        rep_vecs = tuple(
            jnp.broadcast_to(eid, (rb, 1)) for eid in holder_cand_eids
        )

        for slot in range(topk):
            peak = jnp.max(choice, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(choice == peak, e_iota, num_experts),
                axis=1,
                keepdims=True,
            )
            pick = e_iota == best
            w_slot = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
            valid = w_slot > 0.0
            hit = jnp.logical_or(hit, jnp.logical_and(pick, scores > 0.0))
            w_all_8 = jnp.where(col128 == slot, w_slot, w_all_8)

            is_home = jnp.logical_and(
                valid, jnp.logical_and(best >= first_vec, best < last_vec)
            )
            cand_slot = jnp.where(is_home, best - first_vec, jnp.int32(-1))
            for r_slot, rep_vec in enumerate(rep_vecs):
                cand_slot = jnp.where(
                    jnp.logical_and(valid, best == rep_vec),
                    jnp.int32(e_per_owner + r_slot),
                    cand_slot,
                )

            is_cand = cand_slot >= 0
            id_col = jnp.int32(1) + loc_cnt
            w_col = jnp.int32(16) + loc_cnt
            pack_ids = jnp.where(
                jnp.logical_and(is_cand, col128 == id_col), cand_slot, pack_ids
            )
            w_loc_raw = jnp.where(
                jnp.logical_and(is_cand, col128 == w_col), w_slot, w_loc_raw
            )
            loc_cnt = loc_cnt + is_cand.astype(jnp.int32)
            choice = jnp.where(pick, neg_inf, choice)

        pack_ids = jnp.where(col128 == 0, loc_cnt, pack_ids)
        w_sum = jnp.sum(w_all_8, axis=1, keepdims=True)
        w_loc_norm = w_loc_raw / (w_sum + route_eps)
        if route_scale != 1.0:
            w_loc_norm = w_loc_norm * route_scale
        w_bits = lax.bitcast_convert_type(w_loc_norm, jnp.int32)
        scratch.route_pack_vmem[row_s, :] = jnp.where(
            col128 < 16, pack_ids, w_bits
        )
        return counts_acc + jnp.sum(
            hit.astype(jnp.int32), axis=0, keepdims=True
        )

    n_chunks = batch_size // rb
    if n_chunks == 1:
        counts_acc = _route_chunk(0, counts_init)
    else:
        counts_acc = lax.fori_loop(
            0,
            n_chunks,
            lambda ci, acc: _route_chunk(pl.multiple_of(ci * rb, rb), acc),
            counts_init,
        )

    scratch.counts_vmem[...] = jnp.broadcast_to(
        counts_acc, scratch.counts_vmem.shape
    )
    cp_counts = pltpu.make_async_copy(
        scratch.counts_vmem, scratch.counts_smem, dma_sem
    )
    cp_pack = pltpu.make_async_copy(
        scratch.route_pack_vmem, scratch.route_pack_smem, dma_sem
    )
    cp_counts.start()
    cp_pack.start()
    cp_counts.wait()
    cp_pack.wait()


def _full_layer_decode_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    replica_cost_mode: str,
    ep_axis_name: str,
    router_multiplier: float,
    pre_norm_eps: float,
    post_ffn_eps: float,
    next_norm_eps: float,
    has_separate_hs: bool,
    gate_is_out_in: bool,
    pre_is_out_in: bool,
    post_is_out_in: bool,
    fcfg: FullLayerDecodeConfig,
):
    cfg = fcfg.sparse_cfg
    assert cfg is not None
    has_replicas = num_replicas > 0
    is_k256 = scale_block == 256
    is_dense = cfg.row_mode == "dense"
    w_mode = cfg.weight_mode  # "stream3" or "half_n"
    use_bd = (not is_dense) and (not is_k256) and bool(cfg.k16_blockdiag_rows)
    tile_m = cfg.tile_m
    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep
    pre_cols = _H // ep
    post_cols = _H_MODEL // ep
    n_proj_bufs = 1 if batch_size >= 256 else 2
    is_gate_first = fcfg.fuse_stage_b and (fcfg.stage_b_schedule == "gate_first")
    prefetch_full_stream = fcfg.fuse_post_ffn_res and (batch_size <= 128)

    it = iter(all_refs)
    # --- Scalar prefetch ---
    rank_ref = next(it)
    replica_meta_ref = next(it) if has_replicas else None

    # --- Inputs ---
    if fcfg.fuse_stage_b:
        router_in_hbm = next(it)
        hs_in_hbm = next(it) if has_separate_hs else None
        gate_w_hbm = next(it)
        bias_ref = next(it)
        w_pre_hbm = next(it)
        pre_gain_ref = next(it)
        logits_in_ref = x_in_ref = None
    else:
        logits_in_ref = next(it)
        bias_ref = next(it)
        x_in_ref = next(it)
        router_in_hbm = hs_in_hbm = gate_w_hbm = w_pre_hbm = pre_gain_ref = None

    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    norm_ref = next(it)
    if has_replicas:
        rep_w1_ref = next(it)
        rep_s1_ref = next(it)
        rep_w2_ref = next(it)
        rep_s2_ref = next(it)
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None

    w_post_hbm = next(it) if fcfg.fuse_stage_a else None

    if fcfg.fuse_post_ffn_res:
        stream_in_hbm = next(it)
        post_ffn_gain_ref = next(it)
        alpha_ref = next(it)
        beta_ref = next(it)
        next_gain_ref = next(it) if fcfg.fuse_next_norm else None
    else:
        stream_in_hbm = post_ffn_gain_ref = alpha_ref = beta_ref = next_gain_ref = None

    # --- Outputs ---
    if fcfg.fuse_post_ffn_res and fcfg.fuse_next_norm:
        out_main_ref = next(it)  # stream_out [B, 8192] f32 (HBM)
        out_next_ref = next(it)  # next_norm [B, 8192] bf16 (HBM)
    else:
        out_main_ref = next(it)
        out_next_ref = None

    # --- Routing & Replica SMEM Scratch ---
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
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = None

    # --- Expert Weight Ring Scratch ---
    w1_buf = next(it)
    s1_buf = next(it)
    w2_buf = next(it)
    s2_buf = next(it)
    w1_sem = next(it)
    w2_sem = next(it)
    misc_sem = next(it)

    # --- Expert Compute Scratch ---
    if is_dense:
        x_dense_buf = next(it) if fcfg.fuse_stage_b else x_in_ref
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it)
        x_3d_buf = io_3d_buf = lhs_2d_buf = masked_x_buf = None
    else:
        x_3d_buf = next(it)
        io_3d_buf = next(it)
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        masked_x_buf = next(it) if use_bd else None
        gain_f32_buf = next(it)
        accum_3d = next(it)
        accum_2d = x_dense_buf = None
        fan_3d_buf = x_3d_buf

    # --- Expert RS + AG Scratch ---
    rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
    rs_recv_3d = next(it)
    rs_send_sem = next(it)
    rs_recv_sem = next(it)
    owner_3d = next(it)
    fan_send_sem = next(it)
    fan_recv_sem = next(it)
    fan_s2_send_sem = next(it) if cfg.bridge_ag else None
    fan_s2_recv_sem = next(it) if cfg.bridge_ag else None

    # --- Stage B & Stage A Projection / Collective Scratch ---
    proj_w_pre_buf = next(it) if fcfg.fuse_stage_b else None
    proj_w_post_buf = (
        proj_w_pre_buf
        if (fcfg.fuse_stage_b and fcfg.fuse_stage_a and (pre_is_out_in == post_is_out_in))
        else (next(it) if fcfg.fuse_stage_a else None)
    )
    proj_w_sem = next(it) if (fcfg.fuse_stage_b or fcfg.fuse_stage_a) else None

    if fcfg.fuse_stage_b:
        rin_buf = next(it)
        hs_buf = next(it) if has_separate_hs else None
        gw_buf = next(it)
        sb_dma_sem = next(it)
        logits_vmem = next(it)
        need_pre_acc = (not is_gate_first) or (n_proj_bufs == 1)
        pre_acc_vmem = next(it) if need_pre_acc else None
        pre_shard_vmem = next(it)
        pre_ag_vmem = next(it)
        pre_send_sem = next(it)
        pre_recv_sem = next(it)
        pre_s2_send_sem = next(it) if fcfg.bridge_pre_ag else None
        pre_s2_recv_sem = next(it) if fcfg.bridge_pre_ag else None
        hs_stage_vmem = next(it) if is_gate_first else None
    else:
        rin_buf = hs_buf = gw_buf = sb_dma_sem = None
        logits_vmem = logits_in_ref
        pre_acc_vmem = pre_shard_vmem = pre_ag_vmem = hs_stage_vmem = None
        pre_send_sem = pre_recv_sem = pre_s2_send_sem = pre_s2_recv_sem = None

    if fcfg.fuse_stage_a:
        post_shard_vmem = next(it)
        post_ag_vmem = next(it)
        post_send_sem = next(it)
        post_recv_sem = next(it)
        post_s2_send_sem = next(it) if fcfg.bridge_post_ag else None
        post_s2_recv_sem = next(it) if fcfg.bridge_post_ag else None
        out_dma_sem = next(it)
        if fcfg.fuse_post_ffn_res:
            if prefetch_full_stream:
                stream_full_vmem = next(it)  # (ep, B, post_cols) f32
                stream_in_sem = next(it)  # sem((ep,))
                stream_strip_buf = sout_strip_buf = None
            else:
                stream_full_vmem = stream_in_sem = None
                stream_strip_buf = next(it)  # (2, B, post_cols) f32
                sout_strip_buf = next(it)  # (1, B, post_cols) f32
            nout_strip_buf = post_shard_vmem if fcfg.fuse_next_norm else None
            nx_dma_sem = next(it) if fcfg.fuse_next_norm else None
        else:
            stream_full_vmem = stream_in_sem = None
            stream_strip_buf = sout_strip_buf = nout_strip_buf = nx_dma_sem = None
    else:
        post_shard_vmem = post_ag_vmem = None
        post_send_sem = post_recv_sem = post_s2_send_sem = post_s2_recv_sem = None
        out_dma_sem = stream_full_vmem = stream_in_sem = None
        stream_strip_buf = sout_strip_buf = nout_strip_buf = nx_dma_sem = None

    # =========================================================================
    # 1. Signal global entry readiness barrier immediately at t = 0.
    # =========================================================================
    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    if has_replicas:
        rep_w1_u32 = rep_w1_ref.bitcast(jnp.uint32)
        rep_w2_u32 = rep_w2_ref.bitcast(jnp.uint32)

    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    def _zero_expert_accumulators():
        if is_dense:
            for s_col in range(0, _H, 1024):
                accum_2d[:, pl.ds(s_col, 1024)] = jnp.zeros(
                    (batch_size, 1024), dtype=jnp.float32
                )
        else:
            zero_rc = min(batch_size, 8)
            zeros_acc = jnp.zeros((zero_rc, 32, 128), dtype=jnp.float32)

            def _zero_acc_chunk(ci, _):
                accum_3d[pl.ds(ci * zero_rc, zero_rc), :, :] = zeros_acc
                return _

            lax.fori_loop(0, batch_size // zero_rc, _zero_acc_chunk, jnp.int32(0))

            zeros_io = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)

            def _zero_io_chunk(ci, _):
                io_3d_buf[pl.ds(ci * 8, 8), :, :] = zeros_io
                return _

            lax.fori_loop(0, tile_m // 8, _zero_io_chunk, jnp.int32(0))

    # --- Expert Weight Ring DMA Helpers ---
    if w_mode == "stream3":

        def _issue_proj(w_src, s_src, part_id: int, slot):
            col = part_id * _I
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(col, _I)],
                w1_buf.at[part_id],
                w1_sem.at[part_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(col, _I)],
                s1_buf.at[part_id],
                w1_sem.at[part_id],
            ).start()

        def _issue_down(w_src, s_src, slot):
            pltpu.make_async_copy(w_src.at[slot], w2_buf, w2_sem.at[0]).start()
            pltpu.make_async_copy(s_src.at[slot], s2_buf, w2_sem.at[0]).start()

        def start_s3_part(part_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_proj(w1_u32, s1_ref, part_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_proj(rep_w1_u32, rep_s1_ref, part_id, slot)

            else:
                _issue_proj(w1_u32, s1_ref, part_id, slot)

        def start_s3_down(src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_down(w2_u32, s2_ref, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_down(rep_w2_u32, rep_s2_ref, slot)

            else:
                _issue_down(w2_u32, s2_ref, slot)

        def wait_s3_part(part_id: int):
            pltpu.make_async_copy(
                w1_buf.at[part_id], w1_buf.at[part_id], w1_sem.at[part_id]
            ).wait()
            pltpu.make_async_copy(
                s1_buf.at[part_id], s1_buf.at[part_id], w1_sem.at[part_id]
            ).wait()

        def wait_s3_down():
            pltpu.make_async_copy(w2_buf, w2_buf, w2_sem.at[0]).wait()
            pltpu.make_async_copy(s2_buf, s2_buf, w2_sem.at[0]).wait()

    else:
        half_i = _I // 2
        half_h = _H // 2

        def _issue_w1_hn(w_src, s_src, buf_id: int, slot):
            g_col = buf_id * half_i
            u_col = _I + buf_id * half_i
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(g_col, half_i)],
                w1_buf.at[buf_id, :, 0:half_i],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(u_col, half_i)],
                w1_buf.at[buf_id, :, half_i:_I],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(g_col, half_i)],
                s1_buf.at[buf_id, :, 0:half_i],
                w1_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(u_col, half_i)],
                s1_buf.at[buf_id, :, half_i:_I],
                w1_sem.at[buf_id],
            ).start()

        def _issue_w2_hn(w_src, s_src, buf_id: int, slot):
            c0 = buf_id * half_h
            pltpu.make_async_copy(
                w_src.at[slot, :, pl.ds(c0, half_h)],
                w2_buf.at[buf_id],
                w2_sem.at[buf_id],
            ).start()
            pltpu.make_async_copy(
                s_src.at[slot, :, pl.ds(c0, half_h)],
                s2_buf.at[buf_id],
                w2_sem.at[buf_id],
            ).start()

        def start_w1_half(buf_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w1_hn(w1_u32, s1_ref, buf_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w1_hn(rep_w1_u32, rep_s1_ref, buf_id, slot)

            else:
                _issue_w1_hn(w1_u32, s1_ref, buf_id, slot)

        def start_w2_half(buf_id: int, src_kind, slot):
            if has_replicas:

                @pl.when(src_kind == 0)
                def _h():
                    _issue_w2_hn(w2_u32, s2_ref, buf_id, slot)

                @pl.when(src_kind == 1)
                def _r():
                    _issue_w2_hn(rep_w2_u32, rep_s2_ref, buf_id, slot)

            else:
                _issue_w2_hn(w2_u32, s2_ref, buf_id, slot)

        def wait_w1_half(buf_id: int):
            pltpu.make_async_copy(
                w1_buf.at[buf_id], w1_buf.at[buf_id], w1_sem.at[buf_id]
            ).wait()
            pltpu.make_async_copy(
                s1_buf.at[buf_id], s1_buf.at[buf_id], w1_sem.at[buf_id]
            ).wait()

        def wait_w2_half(buf_id: int):
            pltpu.make_async_copy(
                w2_buf.at[buf_id], w2_buf.at[buf_id], w2_sem.at[buf_id]
            ).wait()
            pltpu.make_async_copy(
                s2_buf.at[buf_id], s2_buf.at[buf_id], w2_sem.at[buf_id]
            ).wait()

    # --- Stage B helpers ---
    def _start_w_pre_half(h_k: int):
        bslot = h_k % n_proj_bufs
        src_sl = (
            w_pre_hbm.at[:, pl.ds(h_k * _H, _H)]
            if pre_is_out_in
            else w_pre_hbm.at[pl.ds(h_k * _H, _H), :]
        )
        pltpu.make_async_copy(
            src_sl,
            proj_w_pre_buf.at[bslot],
            proj_w_sem.at[bslot],
        ).start()

    def _wait_w_pre_half(h_k: int):
        bslot = h_k % n_proj_bufs
        pltpu.make_async_copy(
            proj_w_pre_buf.at[bslot],
            proj_w_pre_buf.at[bslot],
            proj_w_sem.at[bslot],
        ).wait()

    def _start_w_pre_initial():
        for h_k in range(n_proj_bufs):
            _start_w_pre_half(h_k)

    def _matmul_maybe_out_in(lhs, rhs, *, rhs_is_out_in: bool, precision=None):
        if rhs_is_out_in:
            return lax.dot_general(
                lhs,
                rhs,
                (((1,), (1,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            )
        return jnp.dot(
            lhs, rhs, precision=precision, preferred_element_type=jnp.float32
        )

    def _get_wpre_slice(wpre_half: int, wpre_off: int):
        bslot = wpre_half % n_proj_bufs
        if pre_is_out_in:
            return proj_w_pre_buf[bslot, :, pl.ds(wpre_off, _K_STAGE_B)]
        return proj_w_pre_buf[bslot, pl.ds(wpre_off, _K_STAGE_B), :]

    def _run_pre_expert_norm_from_ag():
        rb_norm = min(batch_size, 16)
        shards_per_half = ep // 2  # 4 x 512 = 2048 (ep=8) or 8 x 256 = 2048 (ep=16)
        for r0 in range(0, batch_size, rb_norm):
            r_sl = pl.ds(r0, rb_norm)
            sq = jnp.zeros((rb_norm, 1), dtype=jnp.float32)
            for s_rk in range(ep):
                part = pre_ag_vmem[s_rk, r_sl, :].astype(jnp.float32)
                sq = sq + jnp.sum(part * part, axis=1, keepdims=True)
            inv = lax.rsqrt(sq * (1.0 / _H) + pre_norm_eps)
            for h_idx in range(2):
                normed_parts = []
                for sub_s in range(shards_per_half):
                    s_rk = h_idx * shards_per_half + sub_s
                    g_sl = pl.ds(s_rk * pre_cols, pre_cols)
                    part = pre_ag_vmem[s_rk, r_sl, :].astype(jnp.float32)
                    pg = pre_gain_ref[0:1, g_sl]
                    normed_parts.append((part * inv * pg).astype(jnp.bfloat16))
                half_2048 = jnp.concatenate(normed_parts, axis=1)
                if is_dense:
                    x_dense_buf[r_sl, pl.ds(h_idx * 2048, 2048)] = half_2048
                else:
                    x_3d_buf[r_sl, pl.ds(h_idx * 16, 16), :] = half_2048.reshape(
                        rb_norm, 16, 128
                    )

    # =========================================================================
    # 2. STAGE B PROLOGUE + ROUTING + REPLICA PLAN + EXPERT-0 WEIGHT PREFETCH
    # =========================================================================
    barrier_waited = False
    pre_ag_wait_recs: list = []

    if fcfg.fuse_stage_b:
        n_sb_bufs = rin_buf.shape[0]
        n_k_strips = _H_MODEL // _K_STAGE_B  # 8 strips of 1024 cols
        rb_proj = min(batch_size, 32)
        gate_prec = (
            lax.Precision.HIGHEST
            if fcfg.gate_highest_precision
            else lax.Precision.DEFAULT
        )

        def _start_sb_strip(ks_idx: int):
            bslot = ks_idx % n_sb_bufs
            col_k = pl.ds(ks_idx * _K_STAGE_B, _K_STAGE_B)
            pltpu.make_async_copy(
                router_in_hbm.at[:, col_k],
                rin_buf.at[bslot],
                sb_dma_sem.at[bslot],
            ).start()
            gw_src = (
                gate_w_hbm.at[:, col_k]
                if gate_is_out_in
                else gate_w_hbm.at[col_k, :]
            )
            pltpu.make_async_copy(
                gw_src,
                gw_buf.at[bslot],
                sb_dma_sem.at[bslot],
            ).start()
            if has_separate_hs:
                pltpu.make_async_copy(
                    hs_in_hbm.at[:, col_k],
                    hs_buf.at[bslot],
                    sb_dma_sem.at[bslot],
                ).start()

        def _wait_sb_strip(ks_idx: int):
            bslot = ks_idx % n_sb_bufs
            pltpu.make_async_copy(
                rin_buf.at[bslot], rin_buf.at[bslot], sb_dma_sem.at[bslot]
            ).wait()
            pltpu.make_async_copy(
                gw_buf.at[bslot], gw_buf.at[bslot], sb_dma_sem.at[bslot]
            ).wait()
            if has_separate_hs:
                pltpu.make_async_copy(
                    hs_buf.at[bslot], hs_buf.at[bslot], sb_dma_sem.at[bslot]
                ).wait()

        if is_gate_first:
            # Schedule 1 ("gate_first", #408 #6):
            #   (a) Stream router_input + gate_w ONLY -> compute logits FIRST
            #       (while staging hs [B, 8192] bf16 into dead post_ag_vmem!).
            #   (b) Start w_pre HBM DMA -> overlaps with in-kernel routing + SMEM plan!
            #   (c) Prime Expert-0's 24-MiB weights IMMEDIATELY!
            #   (d) Run pre_proj (VMEM) + pre_ag (overlapped with zero accum + pack!) + pre_norm
            #       UNDER Expert-0's 24-MiB HBM weight stream!
            hs_vmem_bank = hs_stage_vmem
            hs_shards_per_kb = _K_STAGE_B // hs_vmem_bank.shape[2]

            for init_s in range(min(n_sb_bufs, n_k_strips)):
                _start_sb_strip(init_s)

            for ks in range(n_k_strips):
                _wait_sb_strip(ks)
                bslot = ks % n_sb_bufs
                gw_sl = gw_buf[bslot, :, :]
                for r0 in range(0, batch_size, rb_proj):
                    r_sl = pl.ds(r0, rb_proj)
                    rin_c = rin_buf[bslot, r_sl, :]
                    hs_c = (
                        hs_buf[bslot, r_sl, :]
                        if has_separate_hs
                        else rin_c.astype(jnp.bfloat16)
                    )
                    for ps in range(hs_shards_per_kb):
                        sub_cols = hs_vmem_bank.shape[2]
                        hs_vmem_bank[
                            ks * hs_shards_per_kb + ps, r_sl, :
                        ] = hs_c[:, pl.ds(ps * sub_cols, sub_cols)]
                    g_part = _matmul_maybe_out_in(
                        rin_c,
                        gw_sl,
                        rhs_is_out_in=gate_is_out_in,
                        precision=gate_prec,
                    )
                    if ks == 0:
                        logits_vmem[r_sl, :] = g_part
                    elif ks == n_k_strips - 1:
                        logits_vmem[r_sl, :] = (
                            logits_vmem[r_sl, :] + g_part
                        ) * router_multiplier
                    else:
                        logits_vmem[r_sl, :] = logits_vmem[r_sl, :] + g_part
                if ks + n_sb_bufs < n_k_strips:
                    _start_sb_strip(ks + n_sb_bufs)

            # Kick off w_pre HBM DMA right as routing begins (idle HBM window)!
            _start_w_pre_initial()

        else:
            # Schedule 2 ("interleaved"): compute gate + pre_proj together, launch pre_ag before routing.
            _start_w_pre_initial()
            for init_s in range(min(n_sb_bufs, n_k_strips)):
                _start_sb_strip(init_s)
            _zero_expert_accumulators()

            for ks in range(n_k_strips):
                if ks == 0:
                    _wait_w_pre_half(0)
                elif ks == 4:
                    if n_proj_bufs == 1:
                        _start_w_pre_half(1)
                    _wait_w_pre_half(1)

                _wait_sb_strip(ks)

                bslot = ks % n_sb_bufs
                wpre_half = ks // 4
                wpre_off = (ks % 4) * _K_STAGE_B
                gw_sl = gw_buf[bslot, :, :]
                wp_sl = _get_wpre_slice(wpre_half, wpre_off)

                for r0 in range(0, batch_size, rb_proj):
                    r_sl = pl.ds(r0, rb_proj)
                    rin_c = rin_buf[bslot, r_sl, :]
                    hs_c = (
                        hs_buf[bslot, r_sl, :]
                        if has_separate_hs
                        else rin_c.astype(jnp.bfloat16)
                    )
                    g_part = _matmul_maybe_out_in(
                        rin_c,
                        gw_sl,
                        rhs_is_out_in=gate_is_out_in,
                        precision=gate_prec,
                    )
                    p_part = _matmul_maybe_out_in(
                        hs_c, wp_sl, rhs_is_out_in=pre_is_out_in
                    )
                    if ks == 0:
                        logits_vmem[r_sl, :] = g_part
                        pre_acc_vmem[r_sl, :] = p_part
                    elif ks == n_k_strips - 1:
                        logits_vmem[r_sl, :] = (
                            logits_vmem[r_sl, :] + g_part
                        ) * router_multiplier
                        pre_shard_vmem[r_sl, :] = (
                            pre_acc_vmem[r_sl, :] + p_part
                        ).astype(jnp.bfloat16)
                    else:
                        logits_vmem[r_sl, :] = logits_vmem[r_sl, :] + g_part
                        pre_acc_vmem[r_sl, :] = pre_acc_vmem[r_sl, :] + p_part

                if ks + n_sb_bufs < n_k_strips:
                    _start_sb_strip(ks + n_sb_bufs)

            pl.semaphore_wait(barrier_sem, ep)
            barrier_waited = True
            pre_ag_handle = _col_all_gather_start(
                me,
                ep,
                ep_axis_name,
                pre_shard_vmem,
                pre_ag_vmem,
                pre_send_sem,
                pre_recv_sem,
                use_bridge=fcfg.bridge_pre_ag,
            )

    # =========================================================================
    # 3. IN-KERNEL ROUTING + SMEM PLAN + EARLY EXPERT-0 WEIGHT PREFETCH!
    # =========================================================================
    if is_dense:
        route_dense_and_counts_in_kernel(
            logits_vmem,
            bias_ref,
            d_scratch,
            misc_sem.at[0],
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
        )
        counts_smem = d_scratch.counts_smem
    else:
        cand_eids = (
            _extract_holder_cand_eids_scalar(me, num_replicas, replica_meta_ref)
            if has_replicas
            else ()
        )
        _route_topk_fast_in_kernel(
            logits_vmem,
            bias_ref,
            r_scratch,
            misc_sem.at[0],
            my_owner=me,
            e_per_owner=e_local,
            holder_cand_eids=cand_eids,
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
        )
        counts_smem = r_scratch.counts_smem

    if has_replicas:
        run_replica_planner_smem(
            me,
            e_local,
            ep,
            num_replicas,
            counts_smem,
            replica_meta_ref,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=replica_cost_mode,
            tile_m=tile_m,
            moved_flag_smem=moved_flag_smem,
        )
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                d_scratch,
            )
        else:
            num_exec = build_exec_plan_from_masks_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                r_scratch,
            )
    else:
        if is_dense:
            num_exec = build_dense_exec_plan_smem(
                me, e_local, 0, None, None, d_scratch
            )
        else:
            num_exec = build_default_exec_plan_smem(me, e_local, r_scratch)

    exec_eid_smem = (
        d_scratch.exec_expert_id if is_dense else r_scratch.exec_expert_id
    )
    exec_src_smem = (
        d_scratch.exec_weight_src if is_dense else r_scratch.exec_weight_src
    )
    exec_slot_smem = (
        d_scratch.exec_weight_slot if is_dense else r_scratch.exec_weight_slot
    )

    stop_stage = fcfg.debug_stop_after
    if stop_stage in ("route", "plan"):
        assert not fcfg.fuse_stage_b and not fcfg.fuse_stage_a
        pl.semaphore_wait(barrier_sem, ep)
        sentinel = (
            counts_smem[0, 0] if stop_stage == "route" else num_exec
        ).astype(jnp.bfloat16)
        out_main_ref[0:16, 0:128] = jnp.broadcast_to(sentinel, (16, 128))
        return

    if is_gate_first:
        for h_k in range(n_proj_bufs):
            _wait_w_pre_half(h_k)

    # Prime Expert-0's 24-MiB FP4 weights IMMEDIATELY!
    if stop_stage != "pack":

        @pl.when(num_exec > 0)
        def _prime_ring():
            s0 = exec_src_smem[0]
            sl0 = exec_slot_smem[0]
            if w_mode == "stream3":
                start_s3_part(0, s0, sl0)
                start_s3_part(1, s0, sl0)
                start_s3_down(s0, sl0)
            else:
                start_w1_half(0, s0, sl0)
                start_w1_half(1, s0, sl0)
                start_w2_half(0, s0, sl0)
                start_w2_half(1, s0, sl0)

    # Finish Stage-B work completely overlapped with Expert-0's 24-MiB HBM weight stream!
    if is_gate_first:
        hs_vmem_bank = hs_stage_vmem
        hs_shards_per_kb = _K_STAGE_B // hs_vmem_bank.shape[2]
        rb_proj = min(batch_size, 32)
        if n_proj_bufs == 2:
            for r0 in range(0, batch_size, rb_proj):
                r_sl = pl.ds(r0, rb_proj)
                acc_pre = jnp.zeros((rb_proj, pre_cols), dtype=jnp.float32)
                for ks in range(8):
                    if hs_shards_per_kb == 1:
                        hs_c = hs_vmem_bank[ks, r_sl, :]
                    else:
                        hs_c = jnp.concatenate(
                            [
                                hs_vmem_bank[ks * hs_shards_per_kb + ps, r_sl, :]
                                for ps in range(hs_shards_per_kb)
                            ],
                            axis=1,
                        )
                    wp_sl = _get_wpre_slice(ks // 4, (ks % 4) * _K_STAGE_B)
                    acc_pre = acc_pre + _matmul_maybe_out_in(
                        hs_c, wp_sl, rhs_is_out_in=pre_is_out_in
                    )
                pre_shard_vmem[r_sl, :] = acc_pre.astype(jnp.bfloat16)
        else:
            # B == 256 1-slot (4 MiB) w_pre mode: stage Half 0 partial in pre_acc_vmem (0 VREGs across DMA wait!)
            for r0 in range(0, batch_size, rb_proj):
                r_sl = pl.ds(r0, rb_proj)
                acc_h0 = jnp.zeros((rb_proj, pre_cols), dtype=jnp.float32)
                for ks in range(4):
                    hs_c = (
                        hs_vmem_bank[ks, r_sl, :]
                        if hs_shards_per_kb == 1
                        else jnp.concatenate(
                            [
                                hs_vmem_bank[ks * hs_shards_per_kb + ps, r_sl, :]
                                for ps in range(hs_shards_per_kb)
                            ],
                            axis=1,
                        )
                    )
                    acc_h0 = acc_h0 + _matmul_maybe_out_in(
                        hs_c,
                        _get_wpre_slice(0, ks * _K_STAGE_B),
                        rhs_is_out_in=pre_is_out_in,
                    )
                pre_acc_vmem[r_sl, :] = acc_h0
            _start_w_pre_half(1)
            _wait_w_pre_half(1)
            for r0 in range(0, batch_size, rb_proj):
                r_sl = pl.ds(r0, rb_proj)
                acc_pre = pre_acc_vmem[r_sl, :]
                for ks2 in range(4):
                    ks = 4 + ks2
                    hs_c = (
                        hs_vmem_bank[ks, r_sl, :]
                        if hs_shards_per_kb == 1
                        else jnp.concatenate(
                            [
                                hs_vmem_bank[ks * hs_shards_per_kb + ps, r_sl, :]
                                for ps in range(hs_shards_per_kb)
                            ],
                            axis=1,
                        )
                    )
                    acc_pre = acc_pre + _matmul_maybe_out_in(
                        hs_c,
                        _get_wpre_slice(1, ks2 * _K_STAGE_B),
                        rhs_is_out_in=pre_is_out_in,
                    )
                pre_shard_vmem[r_sl, :] = acc_pre.astype(jnp.bfloat16)

        pl.semaphore_wait(barrier_sem, ep)
        barrier_waited = True
        pre_ag_handle = _col_all_gather_start(
            me,
            ep,
            ep_axis_name,
            pre_shard_vmem,
            pre_ag_vmem,
            pre_send_sem,
            pre_recv_sem,
            use_bridge=fcfg.bridge_pre_ag,
        )
        # Overlap accumulator zeroing + SMEM token packing WITH pre_expert_proj ICI AG!
        _zero_expert_accumulators()
        if not is_dense:
            pack_active_tokens_smem(
                batch_size, r_scratch, has_replicas=has_replicas
            )
        pre_ag_wait_recs = _col_all_gather_finish(
            me,
            ep,
            ep_axis_name,
            pre_ag_vmem,
            pre_recv_sem,
            pre_s2_send_sem,
            pre_s2_recv_sem,
            pre_ag_handle,
        )
        _run_pre_expert_norm_from_ag()

    elif fcfg.fuse_stage_b:
        if not is_dense:
            pack_active_tokens_smem(
                batch_size, r_scratch, has_replicas=has_replicas
            )
        pre_ag_wait_recs = _col_all_gather_finish(
            me,
            ep,
            ep_axis_name,
            pre_ag_vmem,
            pre_recv_sem,
            pre_s2_send_sem,
            pre_s2_recv_sem,
            pre_ag_handle,
        )
        _run_pre_expert_norm_from_ag()

    else:
        _zero_expert_accumulators()
        if not is_dense:
            for r0 in range(0, batch_size, 16):
                for h_idx in range(2):
                    x_3d_buf[pl.ds(r0, 16), pl.ds(h_idx * 16, 16), :] = x_in_ref[
                        pl.ds(r0, 16), pl.ds(h_idx * 2048, 2048)
                    ].reshape(16, 16, 128)
            pack_active_tokens_smem(
                batch_size, r_scratch, has_replicas=has_replicas
            )

    if stop_stage == "pack":
        assert not fcfg.fuse_stage_b and not fcfg.fuse_stage_a
        pl.semaphore_wait(barrier_sem, ep)
        pv = (
            d_scratch.dense_w_vmem[0, 0]
            if is_dense
            else (
                r_scratch.packed_tok_ids[0].astype(jnp.float32)
                + r_scratch.packed_tok_w[0]
            )
        ).astype(jnp.bfloat16)
        base_blk = (
            x_dense_buf[0:16, 0:128] if is_dense else x_3d_buf[0:16, 0, :]
        )
        out_main_ref[0:16, 0:128] = base_blk + pv
        return

    # =========================================================================
    # 4. MOE EXPERT COMPUTE LOOP
    # =========================================================================
    def _run_proj_only_slice(
        x_src_ref, w_ref, s_ref, width: int, m_rows: int, is_up: bool, act_buf=None
    ):
        act_dst = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        write_m = 16 if m_rows == 8 else m_rows

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            res = _fp4_gmm_chunked(
                x_src_ref,
                w_ref,
                s_ref,
                m_rows,
                _H,
                n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
            )
            if not is_up:
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    jax.nn.silu(res).astype(jnp.bfloat16), write_m
                )
            else:
                sg_read = act_dst[pl.ds(0, write_m), col_out]
                if m_rows == 8:
                    sg_read = sg_read[:8, :]
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    (sg_read.astype(jnp.float32) * res).astype(jnp.bfloat16),
                    write_m,
                )
            return carry

        n_strips = width // nc
        if n_strips == 1:
            _strip(0, 0)
        elif n_strips == 2:
            _strip(0, 0)
            _strip(1, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    def _run_gmm1_slice(
        x_src_ref, w_ref, s_ref, col_offset: int, width: int, m_rows: int, act_buf=None
    ):
        act_dst = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        dual_fits_vregs = (m_rows * nc) // 1024 <= 16
        use_dual = (
            is_k256
            and dual_fits_vregs
            and (
                cfg.k256_dual_mode == "dual"
                or (cfg.k256_dual_mode == "hybrid" and m_rows <= 16)
            )
        )
        write_m = 16 if m_rows == 8 else m_rows
        up_base = width

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            if use_dual:
                g, u = _fp4_dual_gmm_chunked(
                    x_src_ref,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    n0,
                    up_base + n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                )
                h = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
            else:
                g = _fp4_gmm_chunked(
                    x_src_ref,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                )
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    jax.nn.silu(g).astype(jnp.bfloat16), write_m
                )
                u = _fp4_gmm_chunked(
                    x_src_ref,
                    w_ref,
                    s_ref,
                    m_rows,
                    _H,
                    up_base + n0,
                    nc,
                    is_k256,
                    scale_block,
                    k16_dequant_mode=cfg.k16_dequant_mode,
                )
                sg_read = act_dst[pl.ds(0, write_m), col_out]
                if m_rows == 8:
                    sg_read = sg_read[:8, :]
                act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                    (sg_read.astype(jnp.float32) * u).astype(jnp.bfloat16),
                    write_m,
                )
            return carry

        n_strips = width // nc
        if n_strips == 1:
            _strip(0, 0)
        elif n_strips == 2:
            _strip(0, 0)
            _strip(1, 0)
        else:
            lax.fori_loop(0, n_strips, _strip, 0)

    def _run_gmm2_slice(
        w_ref, s_ref, col_offset: int, width: int, m_rows: int, sq_sum, act_buf=None
    ):
        act_src = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            if is_k256
            else min(cfg.k16_strip_nc, width)
        )
        write_m = m_rows

        def _strip(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            d = _fp4_gmm_chunked(
                act_src,
                w_ref,
                s_ref,
                m_rows,
                _I,
                n0,
                nc,
                is_k256,
                scale_block,
                k16_dequant_mode=cfg.k16_dequant_mode,
            ).astype(jnp.bfloat16)
            g_col = gain_f32_buf[0:1, col_out]
            u = d.astype(jnp.float32) * g_col
            sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
            y_stage_buf[pl.ds(0, write_m), col_out] = u
            return sq_acc

        n_strips = width // nc
        if n_strips == 1:
            return _strip(0, sq_sum)
        if n_strips == 2:
            sq_sum = _strip(0, sq_sum)
            return _strip(1, sq_sum)
        return lax.fori_loop(0, n_strips, _strip, sq_sum)

    if is_dense:
        lane_128 = lax.broadcasted_iota(jnp.int32, (batch_size, 128), 1)

        def _dense_expert_body(k, _):
            g_e = exec_eid_smem[k]
            has_next = (k + 1) < num_exec
            next_k = jnp.where(has_next, k + 1, jnp.int32(0))
            next_src = exec_src_smem[next_k]
            next_slot = exec_slot_smem[next_k]

            sq_sum = jnp.zeros((batch_size, 1), dtype=jnp.float32)
            if w_mode == "stream3":
                wait_s3_part(0)
                _run_proj_only_slice(
                    x_dense_buf, w1_buf.at[0], s1_buf.at[0], _I, batch_size, False
                )

                @pl.when(has_next)
                def _pf_g():
                    start_s3_part(0, next_src, next_slot)

                wait_s3_part(1)
                _run_proj_only_slice(
                    x_dense_buf, w1_buf.at[1], s1_buf.at[1], _I, batch_size, True
                )

                @pl.when(has_next)
                def _pf_u():
                    start_s3_part(1, next_src, next_slot)

                wait_s3_down()
                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, batch_size, sq_sum
                )

                @pl.when(has_next)
                def _pf_d():
                    start_s3_down(next_src, next_slot)

            else:
                wait_w1_half(0)
                _run_gmm1_slice(
                    x_dense_buf, w1_buf.at[0], s1_buf.at[0], 0, _I // 2, batch_size
                )

                @pl.when(has_next)
                def _pf10():
                    start_w1_half(0, next_src, next_slot)

                wait_w1_half(1)
                _run_gmm1_slice(
                    x_dense_buf,
                    w1_buf.at[1],
                    s1_buf.at[1],
                    _I // 2,
                    _I // 2,
                    batch_size,
                )

                @pl.when(has_next)
                def _pf11():
                    start_w1_half(1, next_src, next_slot)

                wait_w2_half(0)
                sq_sum = _run_gmm2_slice(
                    w2_buf.at[0], s2_buf.at[0], 0, _H // 2, batch_size, sq_sum
                )

                @pl.when(has_next)
                def _pf20():
                    start_w2_half(0, next_src, next_slot)

                wait_w2_half(1)
                sq_sum = _run_gmm2_slice(
                    w2_buf.at[1],
                    s2_buf.at[1],
                    _H // 2,
                    _H // 2,
                    batch_size,
                    sq_sum,
                )

                @pl.when(has_next)
                def _pf21():
                    start_w2_half(1, next_src, next_slot)

            blk = pl.multiple_of((g_e >> 7) << 7, 128)
            w_slice = d_scratch.dense_w_vmem[:, pl.ds(blk, 128)]
            w_col = jnp.sum(
                jnp.where((lane_128 + blk) == g_e, w_slice, 0.0),
                axis=1,
                keepdims=True,
            )
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            rc = min(batch_size, 8)
            for r0 in range(0, batch_size, rc):
                inv_r = inv_rms[r0 : r0 + rc, :]
                w_r = w_col[r0 : r0 + rc, :]
                for s_col in range(0, _H, 1024):
                    col_s = pl.ds(s_col, 1024)
                    u_c = y_stage_buf[pl.ds(r0, rc), col_s]
                    v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                    contrib = jnp.where(
                        w_r > 0.0, v_bf.astype(jnp.float32) * w_r, 0.0
                    )
                    accum_2d[pl.ds(r0, rc), col_s] = (
                        accum_2d[pl.ds(r0, rc), col_s] + contrib
                    )
            return _

        lax.fori_loop(0, num_exec, _dense_expert_body, jnp.int32(0))
        conv_rc = min(batch_size, 8)
        for r0 in range(0, batch_size, conv_rc):
            for h_idx in range(2):
                accum_3d[pl.ds(r0, conv_rc), pl.ds(h_idx * 16, 16), :] = (
                    accum_2d[
                        pl.ds(r0, conv_rc), pl.ds(h_idx * 2048, 2048)
                    ].reshape(conv_rc, 16, 128)
                )
    else:

        def _gather_rows_to_3d(base_pos, cur_rows):
            n8 = cur_rows >> 3

            def _gather_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                io_3d_buf[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                return _

            lax.fori_loop(0, n8, _gather_8, jnp.int32(0))

            def _gather_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                io_3d_buf[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                return _

            lax.fori_loop(n8 << 3, cur_rows, _gather_tail, jnp.int32(0))

        def _copy_io_3d_to_lhs_2d_at(r_base, m_rows: int):
            copy_m = 16 if m_rows == 8 else m_rows
            for r0 in range(0, copy_m, 16):
                rr = r_base + r0
                for h_idx in range(2):
                    lhs_2d_buf[pl.ds(rr, 16), pl.ds(h_idx * 2048, 2048)] = (
                        io_3d_buf[
                            pl.ds(rr, 16), pl.ds(h_idx * 16, 16), :
                        ].reshape(16, 2048)
                    )

        def _finish_rmsnorm_to_io_3d_at(r_base, m_rows: int, sq_sum: jax.Array):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            for r0 in range(0, m_rows, 8):
                inv_r = inv_rms[r0 : r0 + 8, :]
                for h_idx in range(2):
                    u_c = y_stage_buf[pl.ds(r0, 8), pl.ds(h_idx * 2048, 2048)]
                    v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                    io_3d_buf[pl.ds(r_base + r0, 8), pl.ds(h_idx * 16, 16), :] = (
                        v_bf.reshape(8, 16, 128)
                    )

        def _scatter_accumulate(base_pos, cur_rows):
            n8 = cur_rows >> 3

            def _acc_8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                ws = [r_scratch.packed_tok_w[p0 + s] for s in range(8)]
                y8 = io_3d_buf[pl.ds(i0, 8), :, :]
                for s in range(8):
                    ys = y8[s : s + 1, :, :].astype(jnp.float32) * ws[s]
                    accum_3d[pl.ds(ts[s], 1), :, :] = (
                        accum_3d[pl.ds(ts[s], 1), :, :] + ys
                    )
                return _

            lax.fori_loop(0, n8, _acc_8, jnp.int32(0))

            def _acc_tail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                w = r_scratch.packed_tok_w[base_pos + i]
                y_row = io_3d_buf[pl.ds(i, 1), :, :].astype(jnp.float32)
                accum_3d[pl.ds(tok, 1), :, :] = (
                    accum_3d[pl.ds(tok, 1), :, :] + y_row * w
                )
                return _

            lax.fori_loop(n8 << 3, cur_rows, _acc_tail, jnp.int32(0))

        def _run_bd_gmm1_slice(w_ref, s_ref, col_offset: int, width: int, bd_rows: int):
            nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)
            up_base = width
            use_dual_bd = (bd_rows * nc) // 128 <= 4

            def _strip(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(col_offset + n0, nc)
                if use_dual_bd:
                    g, u = _fp4_blockdiag_hoisted_dual_strip(
                        masked_x_buf, w_ref, s_ref, bd_rows, n0, up_base + n0, nc
                    )
                    h = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                    ).astype(jnp.bfloat16)
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(h, 16)
                else:
                    g = _fp4_blockdiag_hoisted_strip(
                        masked_x_buf, w_ref, s_ref, bd_rows, n0, nc
                    )
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        jax.nn.silu(g).astype(jnp.bfloat16), 16
                    )
                    u = _fp4_blockdiag_hoisted_strip(
                        masked_x_buf, w_ref, s_ref, bd_rows, up_base + n0, nc
                    )
                    sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                        jnp.float32
                    )
                    act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                        (sg * u).astype(jnp.bfloat16), 16
                    )
                return carry

            n_strips = width // nc
            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)

        def _run_bd_gmm2_slice(
            w_ref, s_ref, col_offset: int, width: int, bd_rows: int, sq_sum
        ):
            nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)
            pad_f32 = 16 if bd_rows > 8 else 8

            def _strip(j, sq_acc):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(col_offset + n0, nc)
                d = _fp4_blockdiag_hoisted_strip(
                    masked_x_buf, w_ref, s_ref, bd_rows, n0, nc
                ).astype(jnp.bfloat16)
                g_col = gain_f32_buf[0:1, col_out]
                u = d.astype(jnp.float32) * g_col
                sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                y_stage_buf[pl.ds(0, pad_f32), col_out] = _pad_rows_to(u, pad_f32)
                return sq_acc

            n_strips = width // nc
            if n_strips == 1:
                return _strip(0, sq_sum)
            if n_strips == 2:
                sq_sum = _strip(0, sq_sum)
                return _strip(1, sq_sum)
            return lax.fori_loop(0, n_strips, _strip, sq_sum)

        def _finish_bd_rmsnorm_to_io_3d(bd_rows: int, sq_sum: jax.Array):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            pad_r = 16 if bd_rows > 8 else 8
            inv_p = _pad_rows_to(inv_rms, pad_r)
            for r0 in range(0, pad_r, 8):
                inv_8 = inv_p[r0 : r0 + 8, :]
                for h_idx in range(2):
                    u_c = y_stage_buf[pl.ds(r0, 8), pl.ds(h_idx * 2048, 2048)]
                    v_bf = (u_c * inv_8).astype(jnp.bfloat16)
                    io_3d_buf[pl.ds(r0, 8), pl.ds(h_idx * 16, 16), :] = (
                        v_bf.reshape(8, 16, 128)
                    )

        bd_rows_tuple = cfg.k16_blockdiag_rows if use_bd else ()
        m_buckets = (
            tuple(b for b in cfg.m_buckets if b > bd_rows_tuple[-1])
            if bd_rows_tuple
            else cfg.m_buckets
        )
        if not m_buckets:
            m_buckets = (tile_m,)

        def _compute_tile_standard(
            m_rows: int,
            cur_rows,
            is_first_tile,
            is_last_tile,
            has_next,
            next_src,
            next_slot,
        ):
            do_pf_next = jnp.logical_and(is_last_tile, has_next)
            do_loop32 = cfg.loop32_in_tile and (m_rows == 32) and (tile_m > 32)

            if do_loop32:
                n_m32 = jnp.maximum(jnp.int32(1), (cur_rows + 31) // 32)

                def _l32_cp(mi, _):
                    _copy_io_3d_to_lhs_2d_at(pl.multiple_of(mi * 32, 32), 32)
                    return _

                lax.fori_loop(0, n_m32, _l32_cp, jnp.int32(0))
                if w_mode == "stream3":

                    @pl.when(is_first_tile)
                    def _l32_wg():
                        wait_s3_part(0)

                    def _l32_s3g(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_proj_only_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[0],
                            s1_buf.at[0],
                            _I,
                            32,
                            False,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_s3g, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_pfg():
                        start_s3_part(0, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_wu():
                        wait_s3_part(1)

                    def _l32_s3u(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_proj_only_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[1],
                            s1_buf.at[1],
                            _I,
                            32,
                            True,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_s3u, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_pfu():
                        start_s3_part(1, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_wd():
                        wait_s3_down()

                    def _l32_dn(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf,
                            s2_buf,
                            0,
                            _H,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_dn, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_pfd():
                        start_s3_down(next_src, next_slot)

                else:

                    @pl.when(is_first_tile)
                    def _l32_hn10():
                        wait_w1_half(0)

                    def _l32_hng10(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[0],
                            s1_buf.at[0],
                            0,
                            _I // 2,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_hng10, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hnp10():
                        start_w1_half(0, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_hn11():
                        wait_w1_half(1)

                    def _l32_hng11(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        _run_gmm1_slice(
                            lhs_2d_buf.at[pl.ds(r0, 32)],
                            w1_buf.at[1],
                            s1_buf.at[1],
                            _I // 2,
                            _I // 2,
                            32,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        return _

                    lax.fori_loop(0, n_m32, _l32_hng11, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hnp11():
                        start_w1_half(1, next_src, next_slot)

                    @pl.when(is_first_tile)
                    def _l32_hn2():
                        wait_w2_half(0)
                        wait_w2_half(1)

                    def _l32_hng2(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_s = _run_gmm2_slice(
                            w2_buf.at[0],
                            s2_buf.at[0],
                            0,
                            _H // 2,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        sq_s = _run_gmm2_slice(
                            w2_buf.at[1],
                            s2_buf.at[1],
                            _H // 2,
                            _H // 2,
                            32,
                            sq_s,
                            act_2d_buf.at[pl.ds(r0, 32)],
                        )
                        _finish_rmsnorm_to_io_3d_at(r0, 32, sq_s)
                        return _

                    lax.fori_loop(0, n_m32, _l32_hng2, jnp.int32(0))

                    @pl.when(do_pf_next)
                    def _l32_hnp2():
                        start_w2_half(0, next_src, next_slot)
                        start_w2_half(1, next_src, next_slot)

                return

            _copy_io_3d_to_lhs_2d_at(0, m_rows)
            sq_sum = jnp.zeros((m_rows, 1), dtype=jnp.float32)
            if w_mode == "stream3":

                @pl.when(is_first_tile)
                def _s3_wg():
                    wait_s3_part(0)

                _run_proj_only_slice(
                    lhs_2d_buf, w1_buf.at[0], s1_buf.at[0], _I, m_rows, False
                )

                @pl.when(do_pf_next)
                def _s3_pfg():
                    start_s3_part(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _s3_wu():
                    wait_s3_part(1)

                _run_proj_only_slice(
                    lhs_2d_buf, w1_buf.at[1], s1_buf.at[1], _I, m_rows, True
                )

                @pl.when(do_pf_next)
                def _s3_pfu():
                    start_s3_part(1, next_src, next_slot)

                @pl.when(is_first_tile)
                def _s3_wd():
                    wait_s3_down()

                sq_sum = _run_gmm2_slice(
                    w2_buf, s2_buf, 0, _H, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _s3_pfd():
                    start_s3_down(next_src, next_slot)

            else:

                @pl.when(is_first_tile)
                def _hn_w10():
                    wait_w1_half(0)

                _run_gmm1_slice(
                    lhs_2d_buf, w1_buf.at[0], s1_buf.at[0], 0, _I // 2, m_rows
                )

                @pl.when(do_pf_next)
                def _hn_pf10():
                    start_w1_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w11():
                    wait_w1_half(1)

                _run_gmm1_slice(
                    lhs_2d_buf,
                    w1_buf.at[1],
                    s1_buf.at[1],
                    _I // 2,
                    _I // 2,
                    m_rows,
                )

                @pl.when(do_pf_next)
                def _hn_pf11():
                    start_w1_half(1, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w20():
                    wait_w2_half(0)

                sq_sum = _run_gmm2_slice(
                    w2_buf.at[0], s2_buf.at[0], 0, _H // 2, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _hn_pf20():
                    start_w2_half(0, next_src, next_slot)

                @pl.when(is_first_tile)
                def _hn_w21():
                    wait_w2_half(1)

                sq_sum = _run_gmm2_slice(
                    w2_buf.at[1], s2_buf.at[1], _H // 2, _H // 2, m_rows, sq_sum
                )

                @pl.when(do_pf_next)
                def _hn_pf21():
                    start_w2_half(1, next_src, next_slot)

            _finish_rmsnorm_to_io_3d_at(0, m_rows, sq_sum)

        def _compute_tile_blockdiag(
            bd_rows: int, is_first_tile, is_last_tile, has_next, next_src, next_slot
        ):
            _build_masked_x_from_3d(io_3d_buf, masked_x_buf, bd_rows)
            sq_sum = jnp.zeros((bd_rows, 1), dtype=jnp.float32)
            do_pf_next = jnp.logical_and(is_last_tile, has_next)

            @pl.when(is_first_tile)
            def _bd_hn_w10():
                wait_w1_half(0)

            _run_bd_gmm1_slice(w1_buf.at[0], s1_buf.at[0], 0, _I // 2, bd_rows)

            @pl.when(do_pf_next)
            def _bd_hn_pf10():
                start_w1_half(0, next_src, next_slot)

            @pl.when(is_first_tile)
            def _bd_hn_w11():
                wait_w1_half(1)

            _run_bd_gmm1_slice(
                w1_buf.at[1], s1_buf.at[1], _I // 2, _I // 2, bd_rows
            )

            @pl.when(do_pf_next)
            def _bd_hn_pf11():
                start_w1_half(1, next_src, next_slot)

            _build_masked_x_from_2d(act_2d_buf, masked_x_buf, bd_rows)

            @pl.when(is_first_tile)
            def _bd_hn_w20():
                wait_w2_half(0)

            sq_sum = _run_bd_gmm2_slice(
                w2_buf.at[0], s2_buf.at[0], 0, _H // 2, bd_rows, sq_sum
            )

            @pl.when(do_pf_next)
            def _bd_hn_pf20():
                start_w2_half(0, next_src, next_slot)

            @pl.when(is_first_tile)
            def _bd_hn_w21():
                wait_w2_half(1)

            sq_sum = _run_bd_gmm2_slice(
                w2_buf.at[1], s2_buf.at[1], _H // 2, _H // 2, bd_rows, sq_sum
            )

            @pl.when(do_pf_next)
            def _bd_hn_pf21():
                start_w2_half(1, next_src, next_slot)

            _finish_bd_rmsnorm_to_io_3d(bd_rows, sq_sum)

        def _sparse_expert_body(k, _):
            has_next = (k + 1) < num_exec
            next_k = jnp.where(has_next, k + 1, jnp.int32(0))
            next_src = exec_src_smem[next_k]
            next_slot = exec_slot_smem[next_k]
            row_start = r_scratch.exec_start[k]
            row_end = r_scratch.exec_start[k + 1]
            r_e = row_end - row_start
            n_row_tiles = (r_e + tile_m - 1) // tile_m

            def _row_tile_body(tile_idx, _c):
                tile_off = tile_idx * tile_m
                base_pos = row_start + tile_off
                cur_rows = jnp.minimum(tile_m, r_e - tile_off)
                is_first_tile = tile_idx == 0
                is_last_tile = tile_idx == (n_row_tiles - 1)
                _gather_rows_to_3d(base_pos, cur_rows)

                bd_args = (is_first_tile, is_last_tile, has_next, next_src, next_slot)
                std_args = (cur_rows,) + bd_args
                if use_bd:
                    bd_idx = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bd_idx = bd_idx + (cur_rows > _r).astype(jnp.int32)
                    b_idx = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                    idx = jnp.where(
                        cur_rows <= bd_rows_tuple[-1],
                        bd_idx,
                        len(bd_rows_tuple) + b_idx,
                    )
                    branches = [
                        (lambda r=r: (_compute_tile_blockdiag(r, *bd_args), jnp.int32(0))[1])
                        for r in bd_rows_tuple
                    ] + [
                        (lambda b=b: (_compute_tile_standard(b, *std_args), jnp.int32(0))[1])
                        for b in m_buckets
                    ]
                    lax.switch(idx, branches)
                elif len(m_buckets) > 1:
                    b_idx = jnp.int32(0)
                    for _b in m_buckets[:-1]:
                        b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                    lax.switch(
                        b_idx,
                        [
                            (lambda b=b: (_compute_tile_standard(b, *std_args), jnp.int32(0))[1])
                            for b in m_buckets
                        ],
                    )
                else:
                    _compute_tile_standard(m_buckets[0], *std_args)

                _scatter_accumulate(base_pos, cur_rows)
                return _c

            lax.fori_loop(0, n_row_tiles, _row_tile_body, jnp.int32(0))
            return _

        lax.fori_loop(0, num_exec, _sparse_expert_body, jnp.int32(0))

    if stop_stage == "experts":
        assert not fcfg.fuse_stage_b and not fcfg.fuse_stage_a
        pl.semaphore_wait(barrier_sem, ep)
        for r0 in range(0, batch_size, 16):
            out_main_ref[pl.ds(r0, 16), :] = (
                accum_3d[pl.ds(r0, 16), :, :]
                .astype(jnp.bfloat16)
                .reshape(16, _H)
            )
        return

    # =========================================================================
    # 5. PREFETCH w_post (+ stream_in) DURING EXPERT REDUCE-SCATTER + ALL-GATHER!
    # =========================================================================
    if fcfg.fuse_stage_a:
        half_post_rows = post_cols // 2  # 512 at ep=8, 256 at ep=16
        for h_p in range(n_proj_bufs):
            src_post = (
                w_post_hbm.at[pl.ds(h_p * half_post_rows, half_post_rows), :]
                if post_is_out_in
                else w_post_hbm.at[:, pl.ds(h_p * half_post_rows, half_post_rows)]
            )
            pltpu.make_async_copy(
                src_post,
                proj_w_post_buf.at[h_p],
                proj_w_sem.at[h_p],
            ).start()
        if prefetch_full_stream:
            for s_rk in range(ep):
                pltpu.make_async_copy(
                    stream_in_hbm.at[:, pl.ds(s_rk * post_cols, post_cols)],
                    stream_full_vmem.at[s_rk],
                    stream_in_sem.at[s_rk],
                ).start()

    if cfg.rs_wire_bf16:
        cast_rc = min(batch_size, 8)

        def _cast_rs_chunk(ci, _):
            r0 = ci * cast_rc
            rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                pl.ds(r0, cast_rc), :, :
            ].astype(jnp.bfloat16)
            return _

        lax.fori_loop(0, batch_size // cast_rc, _cast_rs_chunk, jnp.int32(0))
        rs_src_ref = rs_send_bf16
    else:
        rs_src_ref = accum_3d

    if not barrier_waited:
        pl.semaphore_wait(barrier_sem, ep)

    rs_copies = []
    for d in range(ep):
        dst_slot = jnp.where(me < d, me, jnp.maximum(me - 1, 0))
        copy_op = pltpu.make_async_remote_copy(
            rs_src_ref.at[pl.ds(d * rows_per_rank, rows_per_rank), :, :],
            rs_recv_3d.at[dst_slot],
            rs_send_sem.at[d],
            rs_recv_sem,
            device_id={ep_axis_name: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
        rs_copies.append((d, copy_op))

        @pl.when(me != d)
        def _start_rs(_op=copy_op):
            _op.start()

    pltpu.make_async_copy(rs_recv_3d, rs_recv_3d, rs_recv_sem).wait()

    red_rc = min(rows_per_rank, 4)

    def _reduce_owner_chunk(ci, _):
        r0 = ci * red_rc
        acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
        for src in range(ep - 1):
            acc = acc + rs_recv_3d[src, pl.ds(r0, red_rc), :, :].astype(
                jnp.float32
            )
        owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
        return _

    lax.fori_loop(0, rows_per_rank // red_rc, _reduce_owner_chunk, jnp.int32(0))

    if stop_stage == "rs":
        assert not fcfg.fuse_stage_b and not fcfg.fuse_stage_a
        out_main_ref[0:16, 0:128] = jnp.broadcast_to(
            owner_3d[0, 0, :], (16, 128)
        )
        for d, copy_op in rs_copies:

            @pl.when(me != d)
            def _wrs_early(_op=copy_op):
                _op.wait_send()

        return

    # Expert All-Gather into fan_3d_buf [B, 32, 128] bf16
    my_row_off = me * rows_per_rank
    fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]
    if cfg.bridge_ag:
        num_chips = ep // 2
        my_chip = me >> 1
        my_par = me & 1
        mate = jnp.bitwise_xor(me, jnp.int32(1))
        mate_op = pltpu.make_async_remote_copy(
            owner_3d,
            fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
            fan_send_sem.at[0],
            fan_recv_sem,
            device_id={ep_axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        mate_op.start()
        s1_ops = []
        for c in range(num_chips):
            dest = jnp.int32(c * 2) + my_par
            op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[1 + c],
                fan_recv_sem,
                device_id={ep_axis_name: dest},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s1_ops.append((c, op))

            @pl.when(my_chip != c)
            def _st1(_op=op):
                _op.start()

        s1_recv_rows = num_chips * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s1_recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        s2_ops = []
        for c in range(num_chips):
            src_rank = jnp.int32(c * 2) + my_par
            r_off = src_rank * rows_per_rank
            op2 = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_s2_send_sem.at[c],
                fan_s2_recv_sem,
                device_id={ep_axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            s2_ops.append((c, op2))

            @pl.when(my_chip != c)
            def _st2(_op=op2):
                _op.start()

        s2_recv_rows = (num_chips - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, s2_recv_rows), :, :],
            fan_s2_recv_sem,
        ).wait()
        fan_wait_info = ("bridge", my_chip, mate_op, s1_ops, s2_ops)
    else:
        fan_copies = []
        for dest in range(ep):
            fan_op = pltpu.make_async_remote_copy(
                owner_3d,
                fan_3d_buf.at[pl.ds(my_row_off, rows_per_rank), :, :],
                fan_send_sem.at[dest],
                fan_recv_sem,
                device_id={ep_axis_name: jnp.int32(dest)},
                device_id_type=pl.DeviceIdType.MESH,
            )
            fan_copies.append((dest, fan_op))

            @pl.when(me != dest)
            def _stf(_op=fan_op):
                _op.start()

        recv_rows = (ep - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_recv_sem,
        ).wait()
        fan_wait_info = ("flat", me, fan_copies)

    # =========================================================================
    # 6. STAGE A EPILOGUE: post_expert_proj + Zero-Transpose Bridge Column AG
    #    (+ optional post_feedforward_layernorm + gated residual + next norm!)
    # =========================================================================
    post_ag_wait_recs: list = []
    if fcfg.fuse_stage_a:
        half_post_rows = post_cols // 2  # 512 at ep=8, 256 at ep=16
        rb_post = min(batch_size, 32)
        for h_p in range(2):
            bslot_p = h_p % n_proj_bufs
            if h_p == 1 and n_proj_bufs == 1:
                src_post1 = (
                    w_post_hbm.at[pl.ds(half_post_rows, half_post_rows), :]
                    if post_is_out_in
                    else w_post_hbm.at[:, pl.ds(half_post_rows, half_post_rows)]
                )
                pltpu.make_async_copy(
                    src_post1,
                    proj_w_post_buf.at[0],
                    proj_w_sem.at[0],
                ).start()
            pltpu.make_async_copy(
                proj_w_post_buf.at[bslot_p],
                proj_w_post_buf.at[bslot_p],
                proj_w_sem.at[bslot_p],
            ).wait()
            c_out = pl.ds(h_p * half_post_rows, half_post_rows)
            for r0 in range(0, batch_size, rb_post):
                r_sl = pl.ds(r0, rb_post)
                acc_p = jnp.zeros((rb_post, half_post_rows), dtype=jnp.float32)
                for ks4 in range(4):
                    m_strip = fan_3d_buf[
                        r_sl, pl.ds(ks4 * 8, 8), :
                    ].reshape(rb_post, 1024)
                    w_strip_sl = (
                        proj_w_post_buf[bslot_p, :, pl.ds(ks4 * 1024, 1024)]
                        if post_is_out_in
                        else proj_w_post_buf[bslot_p, pl.ds(ks4 * 1024, 1024), :]
                    )
                    acc_p = acc_p + _matmul_maybe_out_in(
                        m_strip, w_strip_sl, rhs_is_out_in=post_is_out_in
                    )
                post_shard_vmem[r_sl, c_out] = acc_p.astype(jnp.bfloat16)

        post_handle = _col_all_gather_start(
            me,
            ep,
            ep_axis_name,
            post_shard_vmem,
            post_ag_vmem,
            post_send_sem,
            post_recv_sem,
            use_bridge=fcfg.bridge_post_ag,
        )
        post_ag_wait_recs = _col_all_gather_finish(
            me,
            ep,
            ep_axis_name,
            post_ag_vmem,
            post_recv_sem,
            post_s2_send_sem,
            post_s2_recv_sem,
            post_handle,
        )

        if not fcfg.fuse_post_ffn_res:
            out_copies = []
            for s_rk in range(ep):
                cp = pltpu.make_async_copy(
                    post_ag_vmem.at[s_rk],
                    out_main_ref.at[:, pl.ds(s_rk * post_cols, post_cols)],
                    out_dma_sem.at[s_rk],
                )
                cp.start()
                out_copies.append(cp)
            for cp in out_copies:
                cp.wait()
        else:
            sq_pf = jnp.zeros((batch_size, 1), dtype=jnp.float32)
            rb_ep = min(batch_size, 32)
            for s_rk in range(ep):
                col_s = pl.ds(s_rk * post_cols, post_cols)
                pfg = post_ffn_gain_ref[0:1, col_s]
                sq_chunks = []
                for r0 in range(0, batch_size, rb_ep):
                    u_c = (
                        post_ag_vmem[s_rk, pl.ds(r0, rb_ep), :].astype(jnp.float32)
                        * pfg
                    )
                    sq_chunks.append(jnp.sum(u_c * u_c, axis=1, keepdims=True))
                sq_pf = sq_pf + jnp.concatenate(sq_chunks, axis=0)
            inv_pf = lax.rsqrt(sq_pf * (1.0 / _H_MODEL) + post_ffn_eps)

            if prefetch_full_stream:
                # B <= 128 fast path: stream_full_vmem arrived during Expert RS+AG!
                out_main_copies = []
                sq_nx = (
                    jnp.zeros((batch_size, 1), dtype=jnp.float32)
                    if fcfg.fuse_next_norm
                    else None
                )
                for s_rk in range(ep):
                    pltpu.make_async_copy(
                        stream_full_vmem.at[s_rk],
                        stream_full_vmem.at[s_rk],
                        stream_in_sem.at[s_rk],
                    ).wait()
                    col_s = pl.ds(s_rk * post_cols, post_cols)
                    pfg = post_ffn_gain_ref[0:1, col_s]
                    alp = alpha_ref[0:1, col_s]
                    bet = beta_ref[0:1, col_s]
                    nx_chunks = []
                    for r0 in range(0, batch_size, rb_ep):
                        r_sl = pl.ds(r0, rb_ep)
                        b_norm = (
                            post_ag_vmem[s_rk, r_sl, :].astype(jnp.float32)
                            * pfg
                            * inv_pf[r0 : r0 + rb_ep, :]
                        )
                        ns = alp * stream_full_vmem[s_rk, r_sl, :] + bet * b_norm
                        stream_full_vmem[s_rk, r_sl, :] = ns
                        if fcfg.fuse_next_norm:
                            nx_chunks.append(jnp.sum(ns * ns, axis=1, keepdims=True))
                    if fcfg.fuse_next_norm:
                        sq_nx = sq_nx + jnp.concatenate(nx_chunks, axis=0)
                    cp_out = pltpu.make_async_copy(
                        stream_full_vmem.at[s_rk],
                        out_main_ref.at[:, col_s],
                        out_dma_sem.at[s_rk],
                    )
                    cp_out.start()
                    out_main_copies.append(cp_out)

                if fcfg.fuse_next_norm:
                    inv_nx = lax.rsqrt(sq_nx * (1.0 / _H_MODEL) + next_norm_eps)
                    for s_rk in range(ep):
                        col_s = pl.ds(s_rk * post_cols, post_cols)
                        ng = next_gain_ref[0:1, col_s]
                        for r0 in range(0, batch_size, rb_ep):
                            r_sl = pl.ds(r0, rb_ep)
                            nout_strip_buf[r_sl, :] = (
                                stream_full_vmem[s_rk, r_sl, :]
                                * inv_nx[r0 : r0 + rb_ep, :]
                                * ng
                            ).astype(jnp.bfloat16)
                        cp_nx = pltpu.make_async_copy(
                            nout_strip_buf,
                            out_next_ref.at[:, col_s],
                            nx_dma_sem.at[0],
                        )
                        cp_nx.start()
                        cp_nx.wait()

                for cp_out in out_main_copies:
                    cp_out.wait()

            else:
                # B == 256 low-VMEM double-buffered async strip pipeline!
                def _start_st_in(s_idx: int):
                    sl = s_idx % 2
                    pltpu.make_async_copy(
                        stream_in_hbm.at[:, pl.ds(s_idx * post_cols, post_cols)],
                        stream_strip_buf.at[sl],
                        out_dma_sem.at[sl],
                    ).start()

                def _wait_st_in(s_idx: int):
                    sl = s_idx % 2
                    pltpu.make_async_copy(
                        stream_strip_buf.at[sl],
                        stream_strip_buf.at[sl],
                        out_dma_sem.at[sl],
                    ).wait()

                _start_st_in(0)
                if ep > 1:
                    _start_st_in(1)

                sq_nx = (
                    jnp.zeros((batch_size, 1), dtype=jnp.float32)
                    if fcfg.fuse_next_norm
                    else None
                )
                out_cp_prev = None
                for s_rk in range(ep):
                    _wait_st_in(s_rk)
                    sl = s_rk % 2
                    if out_cp_prev is not None:
                        out_cp_prev.wait()
                    col_s = pl.ds(s_rk * post_cols, post_cols)
                    pfg = post_ffn_gain_ref[0:1, col_s]
                    alp = alpha_ref[0:1, col_s]
                    bet = beta_ref[0:1, col_s]
                    nx_chunks = []
                    for r0 in range(0, batch_size, rb_ep):
                        r_sl = pl.ds(r0, rb_ep)
                        b_norm = (
                            post_ag_vmem[s_rk, r_sl, :].astype(jnp.float32)
                            * pfg
                            * inv_pf[r0 : r0 + rb_ep, :]
                        )
                        ns = alp * stream_strip_buf[sl, r_sl, :] + bet * b_norm
                        sout_strip_buf[0, r_sl, :] = ns
                        if fcfg.fuse_next_norm:
                            nx_chunks.append(jnp.sum(ns * ns, axis=1, keepdims=True))
                    if s_rk + 2 < ep:
                        _start_st_in(s_rk + 2)
                    if fcfg.fuse_next_norm:
                        sq_nx = sq_nx + jnp.concatenate(nx_chunks, axis=0)
                    cp_o = pltpu.make_async_copy(
                        sout_strip_buf.at[0],
                        out_main_ref.at[:, col_s],
                        out_dma_sem.at[2],
                    )
                    cp_o.start()
                    out_cp_prev = cp_o
                if out_cp_prev is not None:
                    out_cp_prev.wait()

                if fcfg.fuse_next_norm:
                    inv_nx = lax.rsqrt(sq_nx * (1.0 / _H_MODEL) + next_norm_eps)
                    for s_rk in range(ep):
                        col_s = pl.ds(s_rk * post_cols, post_cols)
                        cp_re = pltpu.make_async_copy(
                            out_main_ref.at[:, col_s],
                            sout_strip_buf.at[0],
                            out_dma_sem.at[0],
                        )
                        cp_re.start()
                        cp_re.wait()
                        ng = next_gain_ref[0:1, col_s]
                        for r0 in range(0, batch_size, rb_ep):
                            r_sl = pl.ds(r0, rb_ep)
                            nout_strip_buf[r_sl, :] = (
                                sout_strip_buf[0, r_sl, :]
                                * inv_nx[r0 : r0 + rb_ep, :]
                                * ng
                            ).astype(jnp.bfloat16)
                        cp_nx = pltpu.make_async_copy(
                            nout_strip_buf,
                            out_next_ref.at[:, col_s],
                            nx_dma_sem.at[0],
                        )
                        cp_nx.start()
                        cp_nx.wait()
    else:
        for r0 in range(0, batch_size, 16):
            out_main_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)

    # =========================================================================
    # 7. Cleanly await all outgoing ICI / bridge send semaphores before exit!
    # =========================================================================
    if pre_ag_wait_recs:
        _col_all_gather_wait_sends(pre_ag_wait_recs)
    if post_ag_wait_recs:
        _col_all_gather_wait_sends(post_ag_wait_recs)

    if fan_wait_info[0] == "bridge":
        _, my_chip, mate_op, s1_ops, s2_ops = fan_wait_info
        mate_op.wait_send()
        for c, op in s1_ops:

            @pl.when(my_chip != c)
            def _wf1(_op=op):
                _op.wait_send()

        for c, op2 in s2_ops:

            @pl.when(my_chip != c)
            def _wf2(_op=op2):
                _op.wait_send()

    else:
        _, _, fan_copies = fan_wait_info
        for dest, fan_op in fan_copies:

            @pl.when(me != dest)
            def _wff(_op=fan_op):
                _op.wait_send()

    for d, copy_op in rs_copies:

        @pl.when(me != d)
        def _wrs(_op=copy_op):
            _op.wait_send()


def _launch_full_layer_pallas(
    *,
    rank: jax.Array,
    stage_b_args: tuple,
    expert_args: tuple,
    rep_args: tuple,
    stage_a_args: tuple,
    res_args: tuple,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    replica_cost_mode: str,
    ep_axis_name: str,
    router_multiplier: float,
    pre_norm_eps: float,
    post_ffn_eps: float,
    next_norm_eps: float,
    has_separate_hs: bool,
    gate_is_out_in: bool,
    pre_is_out_in: bool,
    post_is_out_in: bool,
    fcfg: FullLayerDecodeConfig,
):
    cfg = fcfg.sparse_cfg
    assert cfg is not None
    if fcfg.stage_b_schedule not in ("gate_first", "interleaved"):
        raise ValueError(f"Unknown stage_b_schedule: {fcfg.stage_b_schedule!r}")

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep
    pre_cols = _H // ep
    post_cols = _H_MODEL // ep
    has_replicas = num_replicas > 0
    is_dense = cfg.row_mode == "dense"
    is_gate_first = fcfg.fuse_stage_b and (fcfg.stage_b_schedule == "gate_first")
    prefetch_full_stream = fcfg.fuse_post_ffn_res and (batch_size <= 128)
    max_exec = e_local + num_replicas

    scratch_shapes: list = []
    if is_dense:
        scratch_shapes.extend(
            dense_routing_scratch_shapes(
                batch_size, max_exec, num_experts=_E_TOTAL
            )
        )
    else:
        scratch_shapes.extend(
            routing_scratch_shapes(
                batch_size, max_exec, topk=_TOPK, num_experts=_E_TOTAL
            )
        )

    if has_replicas:
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((_E_TOTAL,), jnp.int32),
                smem((e_local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )

    nqb = _H // scale_block
    if cfg.weight_mode == "stream3":
        scratch_shapes.extend(
            [
                vmem((2, _H // _FP4_PACK, _I), jnp.uint32),
                vmem((2, nqb, _I), jnp.bfloat16),
                vmem((_I // _FP4_PACK, _H), jnp.uint32),
                vmem((nqb, _H), jnp.bfloat16),
                sem((2,)),
                sem((1,)),
                sem((1,)),
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((2, _H // _FP4_PACK, _I), jnp.uint32),
                vmem((2, nqb, _I), jnp.bfloat16),
                vmem((2, _I // _FP4_PACK, _H // 2), jnp.uint32),
                vmem((2, nqb, _H // 2), jnp.bfloat16),
                sem((2,)),
                sem((2,)),
                sem((1,)),
            ]
        )

    tile_m = cfg.tile_m
    use_bd = (not is_dense) and (scale_block == 16) and bool(cfg.k16_blockdiag_rows)
    y_stage_rows = (
        max(16, max(cfg.m_buckets)) if cfg.loop32_in_tile else tile_m
    )

    if is_dense:
        if fcfg.fuse_stage_b:
            scratch_shapes.append(vmem((batch_size, _H), jnp.bfloat16))
        scratch_shapes.extend(
            [
                vmem((batch_size, _I), jnp.bfloat16),
                vmem((batch_size, _H), jnp.float32),
                vmem((1, _H), jnp.float32),
                vmem((batch_size, _H), jnp.float32),
                vmem((batch_size, 32, 128), jnp.float32),
                vmem((batch_size, 32, 128), jnp.bfloat16),
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((batch_size, 32, 128), jnp.bfloat16),
                vmem((tile_m, 32, 128), jnp.bfloat16),
                vmem((tile_m, _H), jnp.bfloat16),
                vmem((tile_m, _I), jnp.bfloat16),
                vmem((y_stage_rows, _H), jnp.float32),
            ]
        )
        if use_bd:
            max_bd = max(cfg.k16_blockdiag_rows)
            scratch_shapes.append(vmem((max_bd * 16, _H), jnp.bfloat16))
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),
                vmem((batch_size, 32, 128), jnp.float32),
            ]
        )

    rs_wire_dtype = jnp.bfloat16 if cfg.rs_wire_bf16 else jnp.float32
    if cfg.rs_wire_bf16:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    scratch_shapes.extend(
        [
            vmem((ep - 1, rows_per_rank, 32, 128), rs_wire_dtype),
            sem((ep,)),
            sem,
            vmem((rows_per_rank, 32, 128), jnp.bfloat16),
        ]
    )
    if cfg.bridge_ag:
        num_chips = ep // 2
        scratch_shapes.extend([sem((1 + num_chips,)), sem, sem((num_chips,)), sem])
    else:
        scratch_shapes.extend([sem((ep,)), sem])

    n_proj_bufs = 1 if batch_size >= 256 else 2
    pre_buf_shape = (
        (n_proj_bufs, pre_cols, _H) if pre_is_out_in else (n_proj_bufs, _H, pre_cols)
    )
    post_buf_shape = (
        (n_proj_bufs, pre_cols, _H) if post_is_out_in else (n_proj_bufs, _H, pre_cols)
    )
    if fcfg.fuse_stage_b:
        scratch_shapes.append(vmem(pre_buf_shape, jnp.bfloat16))
    if fcfg.fuse_stage_a and (not fcfg.fuse_stage_b or pre_is_out_in != post_is_out_in):
        scratch_shapes.append(vmem(post_buf_shape, jnp.bfloat16))
    if fcfg.fuse_stage_b or fcfg.fuse_stage_a:
        scratch_shapes.append(sem((n_proj_bufs,)))

    if fcfg.fuse_stage_b:
        n_sb_bufs = 1 if batch_size >= 256 else 2
        scratch_shapes.append(
            vmem((n_sb_bufs, batch_size, _K_STAGE_B), jnp.float32)
        )
        if has_separate_hs:
            scratch_shapes.append(
                vmem((n_sb_bufs, batch_size, _K_STAGE_B), jnp.bfloat16)
            )
        gw_buf_shape = (
            (n_sb_bufs, _E_TOTAL, _K_STAGE_B)
            if gate_is_out_in
            else (n_sb_bufs, _K_STAGE_B, _E_TOTAL)
        )
        scratch_shapes.extend(
            [
                vmem(gw_buf_shape, jnp.float32),
                sem((n_sb_bufs,)),
                vmem((batch_size, _E_TOTAL), jnp.float32),
            ]
        )
        if (not is_gate_first) or (n_proj_bufs == 1):
            scratch_shapes.append(vmem((batch_size, pre_cols), jnp.float32))
        scratch_shapes.extend(
            [
                vmem((batch_size, pre_cols), jnp.bfloat16),
                vmem((ep, batch_size, pre_cols), jnp.bfloat16),
            ]
        )
        if fcfg.bridge_pre_ag:
            nc_p = ep // 2
            scratch_shapes.extend([sem((1 + nc_p,)), sem, sem((nc_p,)), sem])
        else:
            scratch_shapes.extend([sem((ep,)), sem])
        if is_gate_first:
            scratch_shapes.append(vmem((ep, batch_size, post_cols), jnp.bfloat16))

    if fcfg.fuse_stage_a:
        scratch_shapes.extend(
            [
                vmem((batch_size, post_cols), jnp.bfloat16),
                vmem((ep, batch_size, post_cols), jnp.bfloat16),
            ]
        )
        if fcfg.bridge_post_ag:
            nc_q = ep // 2
            scratch_shapes.extend([sem((1 + nc_q,)), sem, sem((nc_q,)), sem])
        else:
            scratch_shapes.extend([sem((ep,)), sem])
        scratch_shapes.append(sem((max(ep, 4),)))  # out_dma_sem
        if fcfg.fuse_post_ffn_res:
            if prefetch_full_stream:
                scratch_shapes.extend(
                    [
                        vmem((ep, batch_size, post_cols), jnp.float32),
                        sem((ep,)),
                    ]
                )
            else:
                scratch_shapes.extend(
                    [
                        vmem((2, batch_size, post_cols), jnp.float32),
                        vmem((1, batch_size, post_cols), jnp.float32),
                    ]
                )
            if fcfg.fuse_next_norm:
                scratch_shapes.append(sem((1,)))

    # Assemble input & output specs
    in_specs: list = []
    call_inputs: list = []

    if fcfg.fuse_stage_b:
        if has_separate_hs:
            router_in, hs_in, gate_w, bias_2d, w_pre, pre_gain_2d = stage_b_args
            in_specs.extend([h_spec, h_spec, h_spec, v_spec, h_spec, v_spec])
            call_inputs.extend(
                [router_in, hs_in, gate_w, bias_2d, w_pre, pre_gain_2d]
            )
        else:
            router_in, gate_w, bias_2d, w_pre, pre_gain_2d = stage_b_args
            in_specs.extend([h_spec, h_spec, v_spec, h_spec, v_spec])
            call_inputs.extend([router_in, gate_w, bias_2d, w_pre, pre_gain_2d])
    else:
        logits, bias_2d, x_2d = stage_b_args
        in_specs.extend([v_spec, v_spec, v_spec])
        call_inputs.extend([logits, bias_2d, x_2d])

    w1, s1, w2, s2, norm_2d = expert_args
    in_specs.extend([h_spec, h_spec, h_spec, h_spec, v_spec])
    call_inputs.extend([w1, s1, w2, s2, norm_2d])

    if has_replicas:
        rep_w1, rep_s1, rep_w2, rep_s2, replica_meta = rep_args
        in_specs.extend([h_spec, h_spec, h_spec, h_spec])
        call_inputs.extend([rep_w1, rep_s1, rep_w2, rep_s2])
        scalar_args = (rank, replica_meta)
        num_scalar_prefetch = 2
    else:
        scalar_args = (rank,)
        num_scalar_prefetch = 1

    if fcfg.fuse_stage_a:
        (w_post,) = stage_a_args
        in_specs.append(h_spec)
        call_inputs.append(w_post)

    if fcfg.fuse_post_ffn_res:
        if fcfg.fuse_next_norm:
            stream_in, post_ffn_g, alpha_2d, beta_2d, next_g = res_args
            in_specs.extend([h_spec, v_spec, v_spec, v_spec, v_spec])
            call_inputs.extend([stream_in, post_ffn_g, alpha_2d, beta_2d, next_g])
            out_shape = (
                jax.ShapeDtypeStruct((batch_size, _H_MODEL), jnp.float32),
                jax.ShapeDtypeStruct((batch_size, _H_MODEL), jnp.bfloat16),
            )
            out_specs = (h_spec, h_spec)
        else:
            stream_in, post_ffn_g, alpha_2d, beta_2d = res_args
            in_specs.extend([h_spec, v_spec, v_spec, v_spec])
            call_inputs.extend([stream_in, post_ffn_g, alpha_2d, beta_2d])
            out_shape = jax.ShapeDtypeStruct((batch_size, _H_MODEL), jnp.float32)
            out_specs = h_spec
    elif fcfg.fuse_stage_a:
        out_shape = jax.ShapeDtypeStruct((batch_size, _H_MODEL), jnp.bfloat16)
        out_specs = h_spec
    else:
        out_shape = jax.ShapeDtypeStruct((batch_size, _H), jnp.bfloat16)
        out_specs = v_spec

    kernel_fn = functools.partial(
        _full_layer_decode_kernel,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=router_multiplier,
        pre_norm_eps=pre_norm_eps,
        post_ffn_eps=post_ffn_eps,
        next_norm_eps=next_norm_eps,
        has_separate_hs=has_separate_hs,
        gate_is_out_in=gate_is_out_in,
        pre_is_out_in=pre_is_out_in,
        post_is_out_in=post_is_out_in,
        fcfg=fcfg,
    )
    sb_tag = (
        f"gf" if is_gate_first else ("il" if fcfg.fuse_stage_b else "0")
    )
    st_sfx = f"_stop_{fcfg.debug_stop_after}" if fcfg.debug_stop_after else ""
    tag = (
        f"muse_full_ep{ep}_b{batch_size}_k{scale_block}"
        f"_sb{sb_tag}_sa{int(fcfg.fuse_stage_a)}"
        f"_res{int(fcfg.fuse_post_ffn_res)}_nx{int(fcfg.fuse_next_norm)}"
        f"_r{num_replicas}{st_sfx}"
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=out_specs,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=tag,
    )(*scalar_args, *call_inputs)


def _validate_experts(ep: int, batch_size: int, w1, s1, w2, s2, rep_args):
    if ep not in (8, 16):
        raise ValueError(f"Expected ep in (8, 16), got {ep}")
    if batch_size not in (16, 32, 64, 128, 256):
        raise ValueError(f"Expected B in (16, 32, 64, 128, 256), got {batch_size}")
    e_local = _E_TOTAL // ep
    w_groups = w1.shape[0]
    if (
        not (e_local <= w_groups <= e_local + 8)
        or w1.shape[1:] != (_H, _N1)
        or w2.shape != (w_groups, _I, _H)
    ):
        raise ValueError(f"Unexpected FP4 weight shapes: w1={w1.shape}, w2={w2.shape}")
    scale_block = _H // s1.shape[1]
    if scale_block not in (16, 256):
        raise ValueError(f"Expected scale_block in (16, 256), got {scale_block}")
    has_rep = any(a is not None for a in rep_args)
    if has_rep:
        if any(a is None for a in rep_args):
            raise ValueError("All 5 replica bank arguments must be provided together")
        num_replicas = int(rep_args[0].shape[0])
    else:
        num_replicas = 0
    return scale_block, num_replicas


def _check_gate_orientation(gate_weight) -> bool:
    if gate_weight.shape == (_E_TOTAL, _H_MODEL):
        return True
    if gate_weight.shape == (_H_MODEL, _E_TOTAL):
        return False
    raise ValueError(
        f"Expected gate_weight shape ({_E_TOTAL}, {_H_MODEL}) or "
        f"({_H_MODEL}, {_E_TOTAL}), got {gate_weight.shape}"
    )


def _check_pre_orientation(pre_proj_weight, ep: int) -> bool:
    pre_cols = _H // ep
    if pre_proj_weight.shape == (pre_cols, _H_MODEL):
        return True
    if pre_proj_weight.shape == (_H_MODEL, pre_cols):
        return False
    raise ValueError(
        f"Expected pre_proj_weight shape ({pre_cols}, {_H_MODEL}) or "
        f"({_H_MODEL}, {pre_cols}), got {pre_proj_weight.shape}"
    )


def _check_post_orientation(post_proj_weight, ep: int) -> bool:
    post_cols = _H_MODEL // ep
    if post_proj_weight.shape == (post_cols, _H):
        return True
    if post_proj_weight.shape == (_H, post_cols):
        return False
    raise ValueError(
        f"Expected post_proj_weight shape ({post_cols}, {_H}) or "
        f"({_H}, {post_cols}), got {post_proj_weight.shape}"
    )


def fused_decode_core_ablation_from_logits(
    rank: jax.Array,
    logits: jax.Array,
    bias: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    norm: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    debug_stop_after: str = "",
    sparse_cfg: SparseDecodeConfig | None = None,
) -> jax.Array:
    """Core MoE decode with optional ``debug_stop_after in {'', 'route', 'plan', 'pack', 'experts', 'rs'}``."""
    batch_size = x.shape[0]
    rep_tuple = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    scale_block, num_replicas = _validate_experts(
        ep, batch_size, w1, s1, w2, s2, rep_tuple
    )
    sp = (
        default_sparse_decode_config(batch_size, ep, scale_block)
        if sparse_cfg is None
        else sparse_cfg
    )
    fcfg = FullLayerDecodeConfig(
        fuse_stage_b=False,
        fuse_stage_a=False,
        fuse_post_ffn_res=False,
        fuse_next_norm=False,
        debug_stop_after=debug_stop_after,
        sparse_cfg=sp,
    )
    return _launch_full_layer_pallas(
        rank=rank,
        stage_b_args=(logits, bias.reshape(1, _E_TOTAL), x.reshape(batch_size, _H)),
        expert_args=(w1, s1, w2, s2, norm.reshape(1, _H)),
        rep_args=rep_tuple if num_replicas > 0 else (),
        stage_a_args=(),
        res_args=(),
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=1.0,
        pre_norm_eps=1e-5,
        post_ffn_eps=1e-8,
        next_norm_eps=1e-5,
        has_separate_hs=False,
        gate_is_out_in=True,
        pre_is_out_in=True,
        post_is_out_in=True,
        fcfg=fcfg,
    )


def fused_decode_stage_b_moe(
    rank: jax.Array,
    hidden_states: jax.Array | None,
    router_input: jax.Array,
    gate_weight: jax.Array,
    bias: jax.Array,
    pre_proj_weight: jax.Array,
    pre_norm_gain: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_expert_norm: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    router_multiplier: float = 0.17677669529663687,
    pre_norm_eps: float = 1e-5,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    config: FullLayerDecodeConfig | None = None,
) -> jax.Array:
    """Stage B + Core MoE: ``[hidden_states, router_input] -> moe_out [B, 4096] bf16``.

    Composes directly with ``DECODE_NORM_LAYOUT=5`` (``self.post_expert_proj`` +
    ``tp_ag_res_norm``) while fusing ``gate + pre_expert_proj + pre_ag + pre_expert_norm``
    into the MoE Core launch!
    """
    batch_size = router_input.shape[0]
    rep_tuple = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    scale_block, num_replicas = _validate_experts(
        ep, batch_size, w1, s1, w2, s2, rep_tuple
    )
    gate_is_out_in = _check_gate_orientation(gate_weight)
    pre_is_out_in = _check_pre_orientation(pre_proj_weight, ep)
    if config is None:
        config = default_full_layer_decode_config(
            batch_size,
            ep,
            scale_block,
            fuse_stage_b=True,
            fuse_stage_a=False,
            fuse_post_ffn_res=False,
            fuse_next_norm=False,
        )
    elif config.sparse_cfg is None:
        config = dataclasses.replace(
            config,
            sparse_cfg=default_sparse_decode_config(batch_size, ep, scale_block),
        )

    has_sep = hidden_states is not None
    sb_args = (
        (
            router_input,
            hidden_states,
            gate_weight,
            bias.reshape(1, _E_TOTAL),
            pre_proj_weight,
            pre_norm_gain.reshape(1, _H).astype(jnp.float32),
        )
        if has_sep
        else (
            router_input,
            gate_weight,
            bias.reshape(1, _E_TOTAL),
            pre_proj_weight,
            pre_norm_gain.reshape(1, _H).astype(jnp.float32),
        )
    )
    return _launch_full_layer_pallas(
        rank=rank,
        stage_b_args=sb_args,
        expert_args=(w1, s1, w2, s2, post_expert_norm.reshape(1, _H)),
        rep_args=rep_tuple if num_replicas > 0 else (),
        stage_a_args=(),
        res_args=(),
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=router_multiplier,
        pre_norm_eps=pre_norm_eps,
        post_ffn_eps=1e-8,
        next_norm_eps=1e-5,
        has_separate_hs=has_sep,
        gate_is_out_in=gate_is_out_in,
        pre_is_out_in=pre_is_out_in,
        post_is_out_in=pre_is_out_in,
        fcfg=config,
    )


def fused_decode_stage_a_from_logits(
    rank: jax.Array,
    logits: jax.Array,
    bias: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    norm: jax.Array,
    post_proj_weight: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    config: FullLayerDecodeConfig | None = None,
) -> jax.Array:
    """Stage A only: ``[logits, x_norm] -> ffn_output [B, 8192] bf16`` in 1 Pallas call."""
    batch_size = x.shape[0]
    rep_tuple = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    scale_block, num_replicas = _validate_experts(
        ep, batch_size, w1, s1, w2, s2, rep_tuple
    )
    post_is_out_in = _check_post_orientation(post_proj_weight, ep)
    if config is None:
        config = default_full_layer_decode_config(
            batch_size,
            ep,
            scale_block,
            fuse_stage_b=False,
            fuse_stage_a=True,
            fuse_post_ffn_res=False,
            fuse_next_norm=False,
        )
    elif config.sparse_cfg is None:
        config = dataclasses.replace(
            config,
            sparse_cfg=default_sparse_decode_config(batch_size, ep, scale_block),
        )

    return _launch_full_layer_pallas(
        rank=rank,
        stage_b_args=(logits, bias.reshape(1, _E_TOTAL), x.reshape(batch_size, _H)),
        expert_args=(w1, s1, w2, s2, norm.reshape(1, _H)),
        rep_args=rep_tuple if num_replicas > 0 else (),
        stage_a_args=(post_proj_weight,),
        res_args=(),
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=1.0,
        pre_norm_eps=1e-5,
        post_ffn_eps=1e-8,
        next_norm_eps=1e-5,
        has_separate_hs=False,
        gate_is_out_in=True,
        pre_is_out_in=post_is_out_in,
        post_is_out_in=post_is_out_in,
        fcfg=config,
    )


def fused_decode_full_moe(
    rank: jax.Array,
    hidden_states: jax.Array | None,
    router_input: jax.Array,
    gate_weight: jax.Array,
    bias: jax.Array,
    pre_proj_weight: jax.Array,
    pre_norm_gain: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_expert_norm: jax.Array,
    post_proj_weight: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    router_multiplier: float = 0.17677669529663687,
    pre_norm_eps: float = 1e-5,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    config: FullLayerDecodeConfig | None = None,
) -> jax.Array:
    """Full ``MuseSparkMoE.forward`` in ONE Pallas launch -> ``[B, 8192] bf16``.

    Positional order ``(rank, hidden_states, router_input, ...)`` matches
    ``MuseSparkMoE.forward(hidden_states, router_input)`` and ``verify_w2_full_layer.py``.
    When ``hidden_states is None``, casts ``router_input`` to ``bfloat16`` inside VMEM
    (eliminating the extra ``[B, 8192] bf16`` HBM round-trip when ``hidden_states ==
    router_input.to(bfloat16)`` in ``MuseSparkDecoderLayer``).
    """
    batch_size = router_input.shape[0]
    rep_tuple = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    scale_block, num_replicas = _validate_experts(
        ep, batch_size, w1, s1, w2, s2, rep_tuple
    )
    if router_input.shape != (batch_size, _H_MODEL):
        raise ValueError(f"Unexpected router_input shape: {router_input.shape}")
    if hidden_states is not None and hidden_states.shape != (batch_size, _H_MODEL):
        raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
    gate_is_out_in = _check_gate_orientation(gate_weight)
    pre_is_out_in = _check_pre_orientation(pre_proj_weight, ep)
    post_is_out_in = _check_post_orientation(post_proj_weight, ep)

    if config is None:
        config = default_full_layer_decode_config(
            batch_size,
            ep,
            scale_block,
            fuse_stage_b=True,
            fuse_stage_a=True,
            fuse_post_ffn_res=False,
            fuse_next_norm=False,
        )
    elif config.sparse_cfg is None:
        config = dataclasses.replace(
            config,
            sparse_cfg=default_sparse_decode_config(batch_size, ep, scale_block),
        )

    has_sep = hidden_states is not None
    sb_args = (
        (
            router_input,
            hidden_states,
            gate_weight,
            bias.reshape(1, _E_TOTAL),
            pre_proj_weight,
            pre_norm_gain.reshape(1, _H).astype(jnp.float32),
        )
        if has_sep
        else (
            router_input,
            gate_weight,
            bias.reshape(1, _E_TOTAL),
            pre_proj_weight,
            pre_norm_gain.reshape(1, _H).astype(jnp.float32),
        )
    )
    return _launch_full_layer_pallas(
        rank=rank,
        stage_b_args=sb_args,
        expert_args=(w1, s1, w2, s2, post_expert_norm.reshape(1, _H)),
        rep_args=rep_tuple if num_replicas > 0 else (),
        stage_a_args=(post_proj_weight,),
        res_args=(),
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=router_multiplier,
        pre_norm_eps=pre_norm_eps,
        post_ffn_eps=1e-8,
        next_norm_eps=1e-5,
        has_separate_hs=has_sep,
        gate_is_out_in=gate_is_out_in,
        pre_is_out_in=pre_is_out_in,
        post_is_out_in=post_is_out_in,
        fcfg=config,
    )


def fused_decode_full_ffn_block(
    rank: jax.Array,
    stream: jax.Array,
    router_input: jax.Array,
    gate_weight: jax.Array,
    bias: jax.Array,
    pre_proj_weight: jax.Array,
    pre_norm_gain: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_expert_norm: jax.Array,
    post_proj_weight: jax.Array,
    post_ffn_gain: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    next_input_gain: jax.Array | None = None,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    router_multiplier: float = 0.17677669529663687,
    pre_norm_eps: float = 1e-5,
    post_ffn_eps: float = 1e-8,
    next_norm_eps: float = 1e-5,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    config: FullLayerDecodeConfig | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Entire FFN half of ``MuseSparkDecoderLayer.forward`` (+ optional next-layer norm) in 1 Pallas call."""
    batch_size = router_input.shape[0]
    rep_tuple = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    scale_block, num_replicas = _validate_experts(
        ep, batch_size, w1, s1, w2, s2, rep_tuple
    )
    gate_is_out_in = _check_gate_orientation(gate_weight)
    pre_is_out_in = _check_pre_orientation(pre_proj_weight, ep)
    post_is_out_in = _check_post_orientation(post_proj_weight, ep)
    has_next = next_input_gain is not None

    if config is None:
        config = default_full_layer_decode_config(
            batch_size,
            ep,
            scale_block,
            fuse_stage_b=True,
            fuse_stage_a=True,
            fuse_post_ffn_res=True,
            fuse_next_norm=has_next,
        )
    elif config.sparse_cfg is None:
        config = dataclasses.replace(
            config,
            sparse_cfg=default_sparse_decode_config(batch_size, ep, scale_block),
        )

    sb_args = (
        router_input,
        gate_weight,
        bias.reshape(1, _E_TOTAL),
        pre_proj_weight,
        pre_norm_gain.reshape(1, _H).astype(jnp.float32),
    )
    pfg_2d = post_ffn_gain.reshape(1, _H_MODEL).astype(jnp.float32)
    alp_2d = alpha.reshape(1, _H_MODEL).astype(jnp.float32)
    bet_2d = beta.reshape(1, _H_MODEL).astype(jnp.float32)
    res_args = (
        (
            stream.astype(jnp.float32),
            pfg_2d,
            alp_2d,
            bet_2d,
            next_input_gain.reshape(1, _H_MODEL).astype(jnp.float32),
        )
        if has_next
        else (stream.astype(jnp.float32), pfg_2d, alp_2d, bet_2d)
    )

    return _launch_full_layer_pallas(
        rank=rank,
        stage_b_args=sb_args,
        expert_args=(w1, s1, w2, s2, post_expert_norm.reshape(1, _H)),
        rep_args=rep_tuple if num_replicas > 0 else (),
        stage_a_args=(post_proj_weight,),
        res_args=res_args,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        router_multiplier=router_multiplier,
        pre_norm_eps=pre_norm_eps,
        post_ffn_eps=post_ffn_eps,
        next_norm_eps=next_norm_eps,
        has_separate_hs=False,
        gate_is_out_in=gate_is_out_in,
        pre_is_out_in=pre_is_out_in,
        post_is_out_in=post_is_out_in,
        fcfg=config,
    )
