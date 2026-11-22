# SPDX-License-Identifier: Apache-2.0
"""Wave-8 Novel Decode Bridge Kernels: Fused Router Top-K & Direct-Layout QKV/RoPE Overlap.

Provides three orthogonal, zero-spill TPU7x Pallas bridge upgrades gated behind opt-in env flags
(``_CARB_CID_BASE = 52``, zero collective-ID or file conflicts with ``decode_collective_bridges_w8``):

  1. ``attn_to_moe_bridge_router_topk_jax`` (``SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK=1``):
     - Fuses ``sigmoid(router_logits * router_multiplier) + e_score_correction_bias + exact
       lowest-ID Top-8 + selection-order weight normalization + global expert hit counts``
       directly inside Bridge 1b, overlapped underneath the ``pre_expert`` Stage-3 ``XOR=1``
       D2D copy and ``out_stream`` HBM DMA.
     - Losslessly packs ``(topk_weights [B, 8], topk_ids [B, 8], expert_counts [256])`` into
       the ``[B, 256] float32`` router output tensor so Bridge 1b's 3-tensor return contract
       and ``muse_sparse``'s 9/14-tensor input contract stay 100% unchanged while skipping
       all 256-wide ``sigmoid + bias + 8x argmax + count`` VPU passes inside ``muse_sparse``.

  2. ``moe_to_attn_direct_qkv_epi_bridge_jax`` & ``fp8_dual_direct_qkv_epi_gate_jax``
     (``SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI=1``):
     - Fuses ``decode_qkv_epilogue`` (per-head ``fp32`` Q/K RMSNorm, ``query_scale``,
       interleaved ``fp32`` RoPE via ``pltpu.roll``, ``KV_PACK_HEADS=1`` diagonal lane
       placement into ``[1, B, 8, 2, 128]``, and ``uint32``-packed ``fp8_e4m3fn[B, 1, 4, 128]``
       KV quantization) directly in VMEM inside Bridge 2b and L0 ``fp8_dual_qkv_gate``,
       interleaved on VPU alongside ``W_gate`` MXU tiles (eliminating 62 epilogue launches
       and all QKV ``as_strided`` sublane relayouts).

  3. ``overlap_mate_epi`` (``SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI=1``):
     - Interleaves ``my_half`` ``(W_qkv, W_gate)`` MXU dots + ``my_half`` ``out_stream`` async
       HBM DMA concurrently with ``mate_half`` ``post_ffn_ln + res + next_in_ln`` VPU epilogue
       in unrolled VLIW-schedulable blocks across both standard and direct-QKV Bridge 2b.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

import functools
from collections.abc import Callable
from threading import Lock

from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _COL_STRIP,
    _DOT_TN,
    _as_2d_f32_row,
    _dma_wait,
    _dot_fp8_scaled_to_ref,
    _fori_strips,
    _gather_shard_xor1,
    _gather_token_shard_xor1_and_dot_post,
    _get_tp_mesh,
    _pick_strip,
    _run_oproj_ar_and_post_attn_norm,
    _run_post_ffn_slice_epilogue,
    _signal_barrier,
    _wait_barrier,
)

_CARB_CID_BASE = 52
_CARB_LOCK = Lock()

_ATTN_TO_MOE_RTOPK_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_TO_ATTN_OVERLAP_OPS: dict[tuple[int, bool], Callable] = {}
_MOE_TO_ATTN_DIRECT_QKV_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_DUAL_DIRECT_QKV_OPS: dict[bool, Callable] = {}


def _align32(n: int) -> int:
    return ((max(n, 1) + 31) // 32) * 32


def _pallas_call_maybe_interpret(kernel_fn, *, interpret: bool, compiler_params, **kwargs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if interpret:
        return pl.pallas_call(
            kernel_fn,
            interpret=pltpu.InterpretParams(detect_races=False),
            **kwargs,
        )
    return pl.pallas_call(kernel_fn, compiler_params=compiler_params, **kwargs)


# =============================================================================
# 1. In-VMEM Router Top-K + Packed [B, 256] Encoder (Bridge 1b -> muse_sparse)
# =============================================================================
def _pack_router_topk_and_counts_in_vmem(
    logits_buf,
    bias_ref,
    out_packed_ref,
    *,
    rows: int,
    top_k: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
) -> None:
    """Zero-spill exact Muse Spark router Top-8 + expert hit counts packed into ``[B, 256] f32``.

    Packing layout inside ``out_packed_ref [rows, 256] float32`` (disjoint 128-lane tiles!):
      - ``[:, 0:128]``:
          * cols ``0 .. top_k-1``: normalized ``topk_weights [rows, top_k] float32``
          * cols ``8 .. 8+top_k-1``: exact ``topk_ids [rows, top_k]`` stored as ``float32``
            integers ``0.0 .. 255.0`` (24-bit significand is 0-ULP exact for all ints <= 2^24!)
          * cols ``16 .. 127``: ``0.0``
      - ``[0:8, 128:256]``:
          * row ``0``: ``expert_counts[0:128]`` as exact ``float32``
          * row ``1``: ``expert_counts[128:256]`` as exact ``float32``
          * rows ``2..7`` (and ``8..rows-1``): ``0.0``
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    num_experts = 256
    rb = _pick_strip(rows, 32)
    n_chunks = rows // rb
    bias_row = bias_ref[0:1, :]
    e_iota = lax.broadcasted_iota(jnp.int32, (rb, num_experts), 1)
    col128 = lax.broadcasted_iota(jnp.int32, (rb, 128), 1)
    neg_inf = jnp.float32(-jnp.inf)
    z_hi = jnp.zeros((rb, 128), dtype=jnp.float32)

    def _chunk_body(ci, counts_acc):
        r0 = pl.multiple_of(ci * rb, rb)
        rs = pl.ds(r0, rb)
        lg_raw = logits_buf[rs, :]
        lg = jnp.where(jnp.isnan(lg_raw), neg_inf, lg_raw)
        scores = jax.nn.sigmoid(lg)
        choice = scores + bias_row

        w_pack = jnp.zeros((rb, 128), dtype=jnp.float32)
        id_pack = jnp.zeros((rb, 128), dtype=jnp.float32)
        hit = jnp.zeros((rb, num_experts), dtype=jnp.bool_)

        for slot in range(top_k):
            peak = jnp.max(choice, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(choice == peak, e_iota, num_experts),
                axis=1,
                keepdims=True,
            )
            pick = e_iota == best
            w_slot = jnp.sum(jnp.where(pick, scores, 0.0), axis=1, keepdims=True)
            hit = jnp.logical_or(hit, jnp.logical_and(pick, scores > 0.0))
            w_pack = jnp.where(col128 == slot, w_slot, w_pack)
            id_pack = jnp.where(col128 == (8 + slot), best.astype(jnp.float32), id_pack)
            choice = jnp.where(pick, neg_inf, choice)

        w_sum = jnp.sum(w_pack, axis=1, keepdims=True)
        w_norm = w_pack / (w_sum + jnp.float32(route_eps))
        if route_scale != 1.0:
            w_norm = w_norm * jnp.float32(route_scale)

        out_packed_ref[rs, pl.ds(0, 128)] = jnp.where(col128 < 8, w_norm, id_pack)
        out_packed_ref[rs, pl.ds(128, 128)] = z_hi
        return counts_acc + jnp.sum(hit.astype(jnp.int32), axis=0, keepdims=True)

    counts_init = jnp.zeros((1, num_experts), dtype=jnp.int32)
    if n_chunks == 1:
        counts_tot = _chunk_body(jnp.int32(0), counts_init)
    else:
        counts_tot = lax.fori_loop(0, n_chunks, _chunk_body, counts_init)

    r8 = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    c_lo = jnp.broadcast_to(counts_tot[:, pl.ds(0, 128)].astype(jnp.float32), (8, 128))
    c_hi = jnp.broadcast_to(counts_tot[:, pl.ds(128, 128)].astype(jnp.float32), (8, 128))
    hdr_hi = jnp.where(r8 == 0, c_lo, jnp.where(r8 == 1, c_hi, jnp.float32(0.0)))
    out_packed_ref[pl.ds(0, 8), pl.ds(128, 128)] = hdr_hi


def _attn_to_moe_bridge_router_topk_kernel(
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
    router_bias_ref,
    pre_exp_gain_ref,
    out_stream_hbm,
    out_router_packed_ref,
    out_expert_in_ref,
    wo_buf,
    w_pre_buf,
    w_router_buf,
    s_buf,
    router_vmem,
    inv_pff_buf,
    ffn_vmem,
    logits_scratch,
    send_half,
    mate_recv,
    own_half,
    scatter_ref,
    own_chunk,
    sq_buf,
    pre_shard,
    pre_half_a,
    pre_half_b,
    wo_sem,
    w_pre_sem,
    w_router_sem,
    s_in_sem,
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
    world_size: int,
    pa_eps: float,
    pre_ff_eps: float,
    pre_ff_scale_input: bool,
    router_multiplier: float,
    top_k: int,
    route_eps: float,
    route_scale: float,
    pre_exp_eps: float,
    pre_exp_scale_input: bool,
) -> None:
    """11-op Bridge 1b fusing router ``sigmoid + bias + Top-8 + counts`` under ``ag_s3`` D2D!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = gated_ref.shape[0]
    hidden = wo_hbm.shape[1]
    half = hidden // 2
    pre_cols = w_pre_hbm.shape[1]
    h_pre = world_size * pre_cols
    n_exp = w_router_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows
    is_r16 = w_router_buf.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    def _prefetch_moe_weights():
        pltpu.make_async_copy(w_pre_hbm, w_pre_buf, w_pre_sem.at[0]).start()
        pltpu.make_async_copy(w_router_hbm, w_router_buf, w_router_sem.at[0]).start()

    r_buf_arg = None if (is_r16 or tile_r32) else router_vmem
    _run_oproj_ar_and_post_attn_norm(
        stream_hbm, gated_ref, wo_hbm, wo_scale_ref, pa_alpha_ref, pa_beta_ref,
        pre_ff_gain_ref, out_stream_hbm, wo_buf, s_buf, r_buf_arg,
        ffn_vmem, send_half, mate_recv, own_half, scatter_ref, own_chunk, sq_buf,
        wo_sem, s_in_sem, s_out_sem, ar_s1a_send, ar_s1a_recv, ar_s1b_send, ar_s1b_recv,
        ar_s2a_send, ar_s2a_recv, ar_s2b_send, ar_s2b_recv,
        me=me, world_size=world_size, pa_eps=pa_eps, pre_ff_eps=pre_ff_eps,
        pre_ff_scale_input=pre_ff_scale_input,
        start_extra_prefetch_fn=_prefetch_moe_weights,
        inv_pff_buf=inv_pff_buf if tile_r32 else None,
        defer_stream_store=tile_r32,
    )

    _dma_wait(w_pre_buf, w_pre_sem.at[0])
    _dot_fp8_scaled_to_ref(
        ffn_vmem, w_pre_buf, w_pre_scale_ref, 0, pre_shard, 0,
        rows=rows, n_cols=pre_cols,
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
                    router_vmem[:, c_g] = (sv * gn) * inv_r if pre_ff_scale_input else (sv * inv_r) * gn
                    return jnp.int32(0)

                lax.fori_loop(0, hidden // _COL_STRIP, _fill_rtile, jnp.int32(0))
                lhs_slice = router_vmem[:, :]
            else:
                r_lhs = ffn_vmem if is_r16 else router_vmem
                lhs_slice = r_lhs[rs, :]

            def _e_step(ce):
                c_sl = pl.ds(ce, _DOT_TN)
                lg = jnp.dot(lhs_slice, w_router_buf[:, c_sl], preferred_element_type=jnp.float32)
                logits_scratch[rs, c_sl] = lg * rmul

            _fori_strips(n_exp, _DOT_TN, _e_step)

        _fori_strips(rows, bm_g, _r_step)
        if tile_r32:
            pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()

    pre_strips = h_pre // _COL_STRIP

    def _norm_pre_slice(src_buf, row_base, n_r):
        bm_n = _pick_strip(n_r, 32)
        align = 16 if use_halfrow else 8

        def _strip(r_loc):
            rs_l = pl.ds(r_loc, bm_n)
            rs_g = pl.ds(pl.multiple_of(row_base + r_loc, align), bm_n)

            def _v_step(s, acc):
                c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                xf = src_buf[rs_l, c_sl].astype(jnp.float32)
                if pre_exp_scale_input:
                    xf = xf * pre_exp_gain_ref[0:1, c_sl]
                sq = xf * xf
                return acc + (sq[:, pl.ds(0, 128)] + sq[:, pl.ds(128, 128)])

            sq_s = lax.fori_loop(
                0, pre_strips, _v_step, jnp.zeros((bm_n, 128), dtype=jnp.float32)
            )
            inv = lax.rsqrt(
                jnp.sum(sq_s, axis=-1, keepdims=True) * jnp.float32(1.0 / h_pre)
                + jnp.float32(pre_exp_eps)
            )

            def _o_step(s, _):
                c_sl = pl.ds(pl.multiple_of(s * _COL_STRIP, _COL_STRIP), _COL_STRIP)
                xf = src_buf[rs_l, c_sl].astype(jnp.float32)
                g = pre_exp_gain_ref[0:1, c_sl]
                yn = (xf * g) * inv if pre_exp_scale_input else (xf * inv) * g
                out_expert_in_ref[rs_g, c_sl] = yn.astype(out_expert_in_ref.dtype)
                return jnp.int32(0)

            lax.fori_loop(0, pre_strips, _o_step, jnp.int32(0))

        _fori_strips(n_r, bm_n, _strip)

    # Overlap router sigmoid + bias + Top-8 + expert counts directly under Stage-3 D2D!
    def _my_half_norm_and_topk(src_buf, row_base, n_r):
        _norm_pre_slice(src_buf, row_base, n_r)
        _pack_router_topk_and_counts_in_vmem(
            logits_scratch,
            router_bias_ref,
            out_router_packed_ref,
            rows=rows,
            top_k=top_k,
            route_eps=route_eps,
            route_scale=route_scale,
        )

    eff_pre_a = send_half.at[pl.ds(0, pre_buf_rows), :] if h_pre == half else pre_half_a
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
        use_halfrow=use_halfrow,
        during_ici_fn=_compute_router_under_ici,
        my_half_epilogue_fn=_my_half_norm_and_topk,
        mate_half_epilogue_fn=_norm_pre_slice,
    )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def attn_to_moe_bridge_router_topk_jax(
    stream,
    gated_attn,
    wo_w8k,
    wo_scale,
    pa_alpha,
    pa_beta,
    pre_ff_gain,
    w_pre_w8k,
    w_pre_scale,
    w_router_km,
    router_bias,
    pre_exp_gain,
    *,
    world_size: int,
    pa_eps: float = 1e-5,
    pre_ff_eps: float = 1e-5,
    pre_ff_scale_input: bool = False,
    router_multiplier: float = 1.0,
    top_k: int = 8,
    route_eps: float = 1e-15,
    route_scale: float = 1.0,
    pre_exp_eps: float = 1e-5,
    pre_exp_scale_input: bool = False,
    interpret: bool = False,
    collective_id: int = _CARB_CID_BASE,
):
    """11-op Bridge 1b emitting ``(new_stream, packed_router_topk_counts [B, 256], expert_in)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    rows, o_in = gated_attn.shape
    wo_k, hidden = wo_w8k.shape
    wp_k, pre_cols = w_pre_w8k.shape
    wr_k, n_exp = w_router_km.shape
    h_pre = world_size * pre_cols
    if (
        rows < 8
        or rows > 384
        or rows % 8 != 0
        or wo_k != o_in
        or wp_k != hidden
        or wr_k != hidden
        or n_exp != 256
        or pre_cols % _DOT_TN != 0
        or top_k != 8
    ):
        raise ValueError(
            f"Unsupported attn_to_moe_router_topk shapes gated={gated_attn.shape}, "
            f"wo={wo_w8k.shape}, w_pre={w_pre_w8k.shape}, w_router={w_router_km.shape}"
        )
    half = hidden // 2
    chunk = hidden // world_size
    n_chips = world_size // 2
    use_xy = world_size == 8 and rows >= 32
    n_ar_sems = 4 if use_xy else (n_chips - 1)
    n_sc_slots = 2 if use_xy else (n_chips - 1)
    is_r16 = w_router_km.dtype == jnp.bfloat16
    tile_r32 = (not is_r16) and rows > 64
    bm_g = _pick_strip(rows, 64)
    use_halfrow = rows >= 32 and rows % 32 == 0
    pre_buf_rows = rows // 2 if use_halfrow else rows
    reuse_pre_ab = h_pre == half

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _attn_to_moe_bridge_router_topk_kernel,
            world_size=world_size,
            pa_eps=float(pa_eps),
            pre_ff_eps=float(pre_ff_eps),
            pre_ff_scale_input=bool(pre_ff_scale_input),
            router_multiplier=float(router_multiplier),
            top_k=int(top_k),
            route_eps=float(route_eps),
            route_scale=float(route_scale),
            pre_exp_eps=float(pre_exp_eps),
            pre_exp_scale_input=bool(pre_exp_scale_input),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, n_exp), jnp.float32),
            jax.ShapeDtypeStruct((rows, h_pre), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((2, o_in, half), wo_w8k.dtype),
            vmem((hidden, pre_cols), w_pre_w8k.dtype),
            vmem((hidden, n_exp), w_router_km.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((8, 128) if is_r16 else ((bm_g, hidden) if tile_r32 else (rows, hidden)), jnp.float32),
            vmem((rows, 1) if tile_r32 else (8, 1), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, n_exp), jnp.float32),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((rows, half), jnp.bfloat16),
            vmem((n_sc_slots, rows, chunk), jnp.bfloat16),
            vmem((8, 128) if use_xy else (rows, chunk), jnp.bfloat16),
            vmem((rows, 128), jnp.float32),
            vmem((rows, pre_cols), jnp.bfloat16),
            vmem((8, 128) if reuse_pre_ab else (pre_buf_rows, h_pre), jnp.bfloat16),
            vmem((8, 128) if reuse_pre_ab else (pre_buf_rows, h_pre), jnp.bfloat16),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
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
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=58 * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w8c_attn_to_moe_rtopk_b{rows}_tp{world_size}"
            f"_{'r16' if is_r16 else 'r32'}"
        ),
    )(
        stream.astype(jnp.float32),
        gated_attn.astype(jnp.bfloat16),
        wo_w8k,
        _as_2d_f32_row(wo_scale, hidden),
        _as_2d_f32_row(pa_alpha, hidden),
        _as_2d_f32_row(pa_beta, hidden),
        _as_2d_f32_row(pre_ff_gain, hidden),
        w_pre_w8k,
        _as_2d_f32_row(w_pre_scale, pre_cols),
        w_router_km,
        _as_2d_f32_row(router_bias, n_exp),
        _as_2d_f32_row(pre_exp_gain, h_pre),
    )


# =============================================================================
# 2. Pure-VMEM Fused QKV/RoPE Epilogue + Overlapped Bridge 2b & L0 Dual QKV/Gate
# =============================================================================
def _in_vmem_qkv_epilogue_tile(
    qkv_vmem_ref,  # bf16[B_pad32, 1280]
    cos_sin_vmem_ref,  # f32[B_pad32, 2, 128] or None
    q_out_vmem_ref,  # bf16[B_pad32, 16, 128]
    kv_out_vmem_ref,  # fp8[B_pad32, 1, 4, 128]
    *,
    b0,
    tile_r: int,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
    p_start: int = 0,
    p_end: int = 8,
    do_kv: bool = True,
) -> None:
    """Zero-spill in-VMEM QKV per-head RMSNorm + interleaved RoPE + FP8 merged-KV pack."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    fp8_min = float(jnp.finfo(jnp.float8_e4m3fn).min)
    b_pad = qkv_vmem_ref.shape[0]
    lane_iota = lax.broadcasted_iota(jnp.int32, (tile_r, 128), 1)
    mask_lo = lane_iota < 64
    even_lane = (lane_iota & 1) == 0
    inv_k_scale = 1.0 / float(k_scale)
    inv_v_scale = 1.0 / float(v_scale)
    rs = pl.ds(b0, tile_r)

    def _roll(x, shift: int):
        if interpret_mode:
            return jnp.roll(x, shift=shift, axis=1)
        return pltpu.roll(x, shift=shift, axis=1)

    def _norm_pair(x_bf16, scale: float):
        x = x_bf16.astype(jnp.float32)
        x2 = x * x
        ms_lo = jnp.broadcast_to(
            jnp.sum(jnp.where(mask_lo, x2, 0.0), axis=1, keepdims=True) * (1.0 / 64.0),
            (tile_r, 128),
        )
        ms_hi = jnp.broadcast_to(
            jnp.sum(jnp.where(mask_lo, 0.0, x2), axis=1, keepdims=True) * (1.0 / 64.0),
            (tile_r, 128),
        )
        inv_rms = jnp.where(
            mask_lo, lax.rsqrt(ms_lo + norm_eps), lax.rsqrt(ms_hi + norm_eps)
        )
        y = x * inv_rms
        if scale != 1.0:
            y = y * scale
        return y.astype(jnp.bfloat16)

    if has_rope:
        cos_128 = cos_sin_vmem_ref[rs, 0, :]
        sin_s128 = cos_sin_vmem_ref[rs, 1, :]
    else:
        cos_128 = None
        sin_s128 = None

    def _rope_pair(x_bf16):
        xv = x_bf16.astype(jnp.float32)
        partner = jnp.where(even_lane, _roll(xv, 127), _roll(xv, 1))
        return (xv * cos_128 + partner * sin_s128).astype(jnp.bfloat16)

    z_bf16 = jnp.zeros((tile_r, 128), jnp.bfloat16)
    for p_idx in range(p_start, p_end):
        qp = qkv_vmem_ref[rs, pl.ds(p_idx * 128, 128)]
        qp = _norm_pair(qp, query_scale)
        if has_rope:
            qp = _rope_pair(qp)
        qp_swapped = _roll(qp, 64)
        if p_idx < 4:
            h_even = jnp.where(mask_lo, qp, z_bf16)
            h_odd = jnp.where(mask_lo, qp_swapped, z_bf16)
        else:
            h_even = jnp.where(mask_lo, z_bf16, qp_swapped)
            h_odd = jnp.where(mask_lo, z_bf16, qp)
        q_out_vmem_ref[rs, 2 * p_idx, :] = h_even
        q_out_vmem_ref[rs, 2 * p_idx + 1, :] = h_odd

    if do_kv:
        kp = qkv_vmem_ref[rs, pl.ds(8 * 128, 128)]
        vp = qkv_vmem_ref[rs, pl.ds(9 * 128, 128)]
        kp = _norm_pair(kp, 1.0)
        if has_rope:
            kp = _rope_pair(kp)

        def _quant_u8(x_bf16, inv_sc: float):
            xf = x_bf16.astype(jnp.float32)
            if inv_sc != 1.0:
                xf = xf * inv_sc
            f8 = jnp.clip(xf, fp8_min, fp8_max).astype(jnp.float8_e4m3fn)
            if interpret_mode:
                return lax.bitcast_convert_type(f8, jnp.uint8)
            return pltpu.bitcast(f8, jnp.uint8)

        k_u8 = _quant_u8(kp, inv_k_scale)
        v_u8 = _quant_u8(vp, inv_v_scale)
        kv_u32 = k_u8.astype(jnp.uint32) | (v_u8.astype(jnp.uint32) << 8)
        if interpret_mode:
            z_u8 = jnp.zeros((tile_r, 128), jnp.uint8)
            packed_u8 = jnp.stack([k_u8, v_u8, z_u8, z_u8], axis=1)[:, None, :, :]
            kv_out_vmem_ref[rs] = lax.bitcast_convert_type(packed_u8, jnp.float8_e4m3fn)
        else:
            kv_out_u32_ref = kv_out_vmem_ref.bitcast(jnp.uint32).reshape(b_pad, 128)
            kv_out_u32_ref[rs] = kv_u32


def _interleaved_gate_and_qkv_epilogue(
    attn_in_ref,
    w_gate_buf,
    s_gate_ref,
    out_gate_ref,
    qkv_scratch,
    cos_sin_ref,
    out_q_ref,
    out_kv_ref,
    *,
    r_base,
    n_r: int,
    align: int,
    norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    has_rope: bool,
    interpret_mode: bool,
) -> None:
    """Overlap ``W_gate`` MXU (4 x 256-col tiles) with QKV RMSNorm + RoPE + FP8 KV on VPU!"""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    tile_r = 64 if (n_r >= 64 and n_r % 64 == 0) else 32
    n_tiles = n_r // tile_r
    for t_i in range(n_tiles):
        b0 = pl.multiple_of(r_base + t_i * tile_r, align)
        rs = pl.ds(b0, tile_r)
        for g_t in range(4):
            cn = g_t * _DOT_TN
            acc = jnp.dot(
                attn_in_ref[rs, :],
                w_gate_buf[:, pl.ds(cn, _DOT_TN)],
                preferred_element_type=jnp.float32,
            )
            out_gate_ref[rs, pl.ds(cn, _DOT_TN)] = (
                acc * s_gate_ref[0:1, pl.ds(cn, _DOT_TN)]
            ).astype(out_gate_ref.dtype)
            _in_vmem_qkv_epilogue_tile(
                qkv_scratch,
                cos_sin_ref,
                out_q_ref,
                out_kv_ref,
                b0=b0,
                tile_r=tile_r,
                norm_eps=norm_eps,
                query_scale=query_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                has_rope=has_rope,
                interpret_mode=interpret_mode,
                p_start=2 * g_t,
                p_end=2 * g_t + 2,
                do_kv=(g_t == 3),
            )


def _overlap_mate_epi_with_my_half_qkvg(
    s_buf,
    mate_buf,
    post_ff_gain_ref,
    pf_alpha_ref,
    pf_beta_ref,
    next_in_gain_ref,
    next_in_vmem,
    w_qkv_buf,
    next_s_qkv_ref,
    qkv_dst_ref,
    w_gate_buf,
    next_s_gate_ref,
    out_next_gate_ref,
    *,
    my_r0,
    mate_r0,
    half: int,
    hidden: int,
    qkv_cols: int,
    gate_cols: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
) -> None:
    """VLIW-interleaved ``mate_half`` VPU epilogue + ``my_half`` ``(W_qkv, W_gate)`` MXU dots!

    Avoids ``scf.for`` barriers across the MXU/VPU boundary by interleaving ``my_half`` 256-col
    MXU dots directly between unrolled 8-strip VPU chunks of ``mate_half`` variance/residual/norm
    (``<= 16`` live VREGs -> ``0 / 0`` LLO spills).
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    bm_epi = _pick_strip(half, 32)
    bm_mxu = _pick_strip(half, 64)
    n_strips = hidden // _COL_STRIP  # 32 for H=8192
    qkv_tiles = qkv_cols // _DOT_TN  # 5 for TP=8
    gate_tiles = gate_cols // _DOT_TN  # 4 for TP=8

    def _dot_my_slice(w_buf, s_ref, dst_ref, cn: int):
        for mr in range(0, half, bm_mxu):
            rs_m = pl.ds(pl.multiple_of(my_r0 + mr, 16), bm_mxu)
            acc = jnp.dot(
                next_in_vmem[rs_m, :],
                w_buf[:, pl.ds(cn, _DOT_TN)],
                preferred_element_type=jnp.float32,
            )
            dst_ref[rs_m, pl.ds(cn, _DOT_TN)] = (
                acc * s_ref[0:1, pl.ds(cn, _DOT_TN)]
            ).astype(dst_ref.dtype)

    # Interleave mate_half first strip (r_loc = 0) with my_half W_qkv + W_gate MXU tiles!
    rs_l0 = pl.ds(0, bm_epi)
    rs_g0 = pl.ds(pl.multiple_of(mate_r0, 16), bm_epi)

    b_sq = jnp.zeros((bm_epi, 128), dtype=jnp.float32)
    chunk_s = max(1, n_strips // max(qkv_tiles, 1))
    qt_issued = 0
    for s in range(n_strips):
        c_sl = pl.ds(s * _COL_STRIP, _COL_STRIP)
        bc = mate_buf[rs_l0, c_sl].astype(jnp.float32)
        if post_ff_scale_input:
            bc = bc * post_ff_gain_ref[0:1, c_sl]
        sq = bc * bc
        b_sq = b_sq + (sq[:, pl.ds(0, 128)] + sq[:, pl.ds(128, 128)])
        if qt_issued < qkv_tiles and (s + 1) % chunk_s == 0:
            _dot_my_slice(w_qkv_buf, next_s_qkv_ref, qkv_dst_ref, qt_issued * _DOT_TN)
            qt_issued += 1
    while qt_issued < qkv_tiles:
        _dot_my_slice(w_qkv_buf, next_s_qkv_ref, qkv_dst_ref, qt_issued * _DOT_TN)
        qt_issued += 1

    inv_b = lax.rsqrt(
        jnp.sum(b_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
        + jnp.float32(post_ff_eps)
    )

    n_sq = jnp.zeros((bm_epi, 128), dtype=jnp.float32)
    chunk_g = max(1, n_strips // max(gate_tiles, 1))
    gt_issued = 0
    for s in range(n_strips):
        c_sl = pl.ds(s * _COL_STRIP, _COL_STRIP)
        bc = mate_buf[rs_l0, c_sl].astype(jnp.float32)
        bg = post_ff_gain_ref[0:1, c_sl]
        b_n = (bc * bg) * inv_b if post_ff_scale_input else (bc * inv_b) * bg
        ns = pf_alpha_ref[0:1, c_sl] * s_buf[rs_g0, c_sl] + pf_beta_ref[0:1, c_sl] * b_n
        s_buf[rs_g0, c_sl] = ns
        nv = ns * next_in_gain_ref[0:1, c_sl] if next_in_scale_input else ns
        sq = nv * nv
        n_sq = n_sq + (sq[:, pl.ds(0, 128)] + sq[:, pl.ds(128, 128)])
        if gt_issued < gate_tiles and (s + 1) % chunk_g == 0:
            _dot_my_slice(w_gate_buf, next_s_gate_ref, out_next_gate_ref, gt_issued * _DOT_TN)
            gt_issued += 1
    while gt_issued < gate_tiles:
        _dot_my_slice(w_gate_buf, next_s_gate_ref, out_next_gate_ref, gt_issued * _DOT_TN)
        gt_issued += 1

    inv_n = lax.rsqrt(
        jnp.sum(n_sq, axis=-1, keepdims=True) * jnp.float32(1.0 / hidden)
        + jnp.float32(next_in_eps)
    )
    for s in range(n_strips):
        c_sl = pl.ds(s * _COL_STRIP, _COL_STRIP)
        sv = s_buf[rs_g0, c_sl]
        ng = next_in_gain_ref[0:1, c_sl]
        nn = (sv * ng) * inv_n if next_in_scale_input else (sv * inv_n) * ng
        next_in_vmem[rs_g0, c_sl] = nn.astype(next_in_vmem.dtype)

    if half > bm_epi:
        _run_post_ffn_slice_epilogue(
            s_buf, mate_buf.at[pl.ds(bm_epi, half - bm_epi), :],
            post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            s_buf, next_in_vmem,
            hidden=hidden, row_base=mate_r0 + bm_epi, n_r=half - bm_epi, align=16,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
            with_next_norm=True,
        )


def _moe_to_attn_overlap_bridge_kernel(
    stream_hbm,
    moe_out_ref,
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
    out_stream_hbm,
    out_next_qkv_ref,
    out_next_gate_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_vmem,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    sharded_moe_in: bool,
) -> None:
    """Bridge 2b overlapping ``mate_half`` VPU ``_epi`` with ``my_half`` ``(W_qkv, W_gate)`` MXU!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    if sharded_moe_in:
        _gather_token_shard_xor1_and_dot_post(
            moe_out_ref, mate_tok_buf, moe_full_buf, w_post_buf, s_post_ref, post_shard, w_post_sem,
            tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
            me=me, world_size=world_size,
        )
    else:
        _dma_wait(w_post_buf, w_post_sem.at[0])
        _dot_fp8_scaled_to_ref(
            moe_out_ref, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
        _wait_barrier(world_size)

    def _epi(buf, r_base, n_r):
        _run_post_ffn_slice_epilogue(
            s_buf, buf, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            s_buf, next_in_vmem,
            hidden=hidden, row_base=r_base, n_r=n_r, align=align,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
            with_next_norm=True,
        )

    if use_halfrow:
        half = rows // 2
        parity = me & jnp.int32(1)
        my_r0 = pl.multiple_of(parity * half, 16)
        mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half, 16)

        _gather_shard_xor1(
            post_shard, half_a, half_b,
            ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
            me=me, world_size=world_size, use_halfrow=True,
            during_ici_fn=lambda: _dma_wait(s_buf, s_in_sem.at[0]),
            my_half_epilogue_fn=_epi,
            mate_half_epilogue_fn=None,
        )
        # Kick off my_half updated stream HBM write immediately!
        pltpu.make_async_copy(
            s_buf.at[pl.ds(my_r0, half)], out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0]
        ).start()
        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
        _dma_wait(w_gate_buf, w_gate_sem.at[0])

        _overlap_mate_epi_with_my_half_qkvg(
            s_buf, half_b, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, out_next_qkv_ref,
            w_gate_buf, next_s_gate_ref, out_next_gate_ref,
            my_r0=my_r0, mate_r0=mate_r0, half=half, hidden=hidden,
            qkv_cols=qkv_cols, gate_cols=gate_cols,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        )
        pltpu.make_async_copy(
            s_buf.at[pl.ds(mate_r0, half)], out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1]
        ).start()
        bm_mxu = _pick_strip(half, 64)
        for mr in range(0, half, bm_mxu):
            rs_m = pl.ds(pl.multiple_of(mate_r0 + mr, 16), bm_mxu)
            for cn in range(0, qkv_cols, _DOT_TN):
                acc = jnp.dot(
                    next_in_vmem[rs_m, :], w_qkv_buf[:, pl.ds(cn, _DOT_TN)],
                    preferred_element_type=jnp.float32,
                )
                out_next_qkv_ref[rs_m, pl.ds(cn, _DOT_TN)] = (
                    acc * next_s_qkv_ref[0:1, pl.ds(cn, _DOT_TN)]
                ).astype(out_next_qkv_ref.dtype)
            for cn in range(0, gate_cols, _DOT_TN):
                acc = jnp.dot(
                    next_in_vmem[rs_m, :], w_gate_buf[:, pl.ds(cn, _DOT_TN)],
                    preferred_element_type=jnp.float32,
                )
                out_next_gate_ref[rs_m, pl.ds(cn, _DOT_TN)] = (
                    acc * next_s_gate_ref[0:1, pl.ds(cn, _DOT_TN)]
                ).astype(out_next_gate_ref.dtype)
        _dma_wait(out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0])
        _dma_wait(out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1])
        return

    _gather_shard_xor1(
        post_shard, half_a, half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=False,
        during_ici_fn=lambda: _dma_wait(s_buf, s_in_sem.at[0]),
        my_half_epilogue_fn=_epi,
        mate_half_epilogue_fn=None,
    )
    pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, out_next_qkv_ref, 0,
        rows=rows, n_cols=qkv_cols,
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_gate_buf, next_s_gate_ref, 0, out_next_gate_ref, 0,
        rows=rows, n_cols=gate_cols,
    )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def moe_to_attn_overlap_bridge_jax(
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
    *,
    world_size: int,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    sharded_moe_in: bool = False,
    interpret: bool = False,
    collective_id: int = _CARB_CID_BASE + 1,
):
    """Bridge 2b with overlapped ``mate_half`` VPU ``_epi`` + ``my_half`` ``(W_qkv, W_gate)`` MXU."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    wk, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    qkv_k, qkv_cols = next_w_qkv_w8k.shape
    gt_k, gate_cols = next_w_gate_w8k.shape
    if (
        in_r < 8
        or in_r % 8 != 0
        or rows > 384
        or stream.shape[0] != rows
        or wk != k_in
        or qkv_k != hidden
        or gt_k != hidden
        or post_cols % _DOT_TN != 0
        or qkv_cols % _DOT_TN != 0
        or gate_cols % _DOT_TN != 0
    ):
        raise ValueError("Unsupported moe_to_attn_overlap_bridge shapes")
    n_chips = world_size // 2
    half_rows = rows // 2 if (rows >= 32 and rows % 32 == 0) else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    return _pallas_call_maybe_interpret(
        functools.partial(
            _moe_to_attn_overlap_bridge_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((rows, qkv_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp],
        out_specs=(h_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(58 if rows > 256 else 56) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8c_moe_to_qkvg_ovl_b{rows}_tp{world_size}{'_sh' if sharded_moe_in else ''}",
    )(
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
    )


def _moe_to_attn_direct_qkv_epi_kernel(
    stream_hbm,
    moe_out_ref,
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
    out_next_q_ref,
    out_next_kv_ref,
    out_next_gate_ref,
    w_post_buf,
    w_qkv_buf,
    w_gate_buf,
    s_buf,
    next_in_vmem,
    qkv_scratch,
    post_shard,
    half_a,
    half_b,
    mate_tok_buf,
    moe_full_buf,
    w_post_sem,
    w_qkv_sem,
    w_gate_sem,
    s_in_sem,
    s_out_sem,
    ag_s1_send,
    ag_s1_recv,
    ag_s2_send,
    ag_s2_recv,
    ag_s3_send,
    ag_s3_recv,
    tok_s1_send,
    tok_s1_recv,
    tok_s2_send,
    tok_s2_recv,
    *,
    world_size: int,
    post_ff_eps: float,
    post_ff_scale_input: bool,
    next_in_eps: float,
    next_in_scale_input: bool,
    sharded_moe_in: bool,
    overlap_mate_epi: bool,
    has_rope: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    interpret_mode: bool,
) -> None:
    """Bridge 2b + pure-VMEM ``decode_qkv_epilogue`` emitting ``(stream, q_packed, kv_fp8, gate)``!"""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = world_size * moe_out_ref.shape[0] if sharded_moe_in else moe_out_ref.shape[0]
    b_pad32 = qkv_scratch.shape[0]
    post_cols = w_post_hbm.shape[1]
    hidden = world_size * post_cols
    qkv_cols = next_w_qkv_hbm.shape[1]
    gate_cols = next_w_gate_hbm.shape[1]
    use_halfrow = rows >= 32 and rows % 32 == 0
    align = 16 if use_halfrow else 8

    me = lax.axis_index("model").astype(jnp.int32)
    _signal_barrier(me, world_size)

    pltpu.make_async_copy(w_post_hbm, w_post_buf, w_post_sem.at[0]).start()
    pltpu.make_async_copy(stream_hbm, s_buf, s_in_sem.at[0]).start()
    pltpu.make_async_copy(next_w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(next_w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()

    if sharded_moe_in:
        _gather_token_shard_xor1_and_dot_post(
            moe_out_ref, mate_tok_buf, moe_full_buf, w_post_buf, s_post_ref, post_shard, w_post_sem,
            tok_s1_send, tok_s1_recv, tok_s2_send, tok_s2_recv,
            me=me, world_size=world_size,
        )
    else:
        _dma_wait(w_post_buf, w_post_sem.at[0])
        _dot_fp8_scaled_to_ref(
            moe_out_ref, w_post_buf, s_post_ref, 0, post_shard, 0,
            rows=rows, n_cols=post_cols,
        )
        _wait_barrier(world_size)

    def _epi(buf, r_base, n_r):
        _run_post_ffn_slice_epilogue(
            s_buf, buf, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            s_buf, next_in_vmem,
            hidden=hidden, row_base=r_base, n_r=n_r, align=align,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
            with_next_norm=True,
        )

    if overlap_mate_epi and use_halfrow and rows % 64 == 0:
        half = rows // 2
        parity = me & jnp.int32(1)
        my_r0 = pl.multiple_of(parity * half, 32)
        mate_r0 = pl.multiple_of((jnp.int32(1) - parity) * half, 32)

        _gather_shard_xor1(
            post_shard, half_a, half_b,
            ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
            me=me, world_size=world_size, use_halfrow=True,
            during_ici_fn=lambda: _dma_wait(s_buf, s_in_sem.at[0]),
            my_half_epilogue_fn=_epi,
            mate_half_epilogue_fn=None,
        )
        pltpu.make_async_copy(
            s_buf.at[pl.ds(my_r0, half)], out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0]
        ).start()
        _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
        _dma_wait(w_gate_buf, w_gate_sem.at[0])

        _overlap_mate_epi_with_my_half_qkvg(
            s_buf, half_b, post_ff_gain_ref, pf_alpha_ref, pf_beta_ref, next_in_gain_ref,
            next_in_vmem, w_qkv_buf, next_s_qkv_ref, qkv_scratch,
            w_gate_buf, next_s_gate_ref, out_next_gate_ref,
            my_r0=my_r0, mate_r0=mate_r0, half=half, hidden=hidden,
            qkv_cols=qkv_cols, gate_cols=gate_cols,
            post_ff_eps=post_ff_eps, post_ff_scale_input=post_ff_scale_input,
            next_in_eps=next_in_eps, next_in_scale_input=next_in_scale_input,
        )
        pltpu.make_async_copy(
            s_buf.at[pl.ds(mate_r0, half)], out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1]
        ).start()

        # Run my_half QKV epilogue + mate_half W_qkv + interleaved mate_half (W_gate + QKV epilogue)!
        tile_h = 64 if half % 64 == 0 else 32
        for t_i in range(half // tile_h):
            _in_vmem_qkv_epilogue_tile(
                qkv_scratch, cos_sin_ref, out_next_q_ref, out_next_kv_ref,
                b0=pl.multiple_of(my_r0 + t_i * tile_h, 32), tile_r=tile_h,
                norm_eps=qkv_norm_eps, query_scale=query_scale,
                k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
                interpret_mode=interpret_mode,
            )
        bm_mxu = _pick_strip(half, 64)
        for mr in range(0, half, bm_mxu):
            rs_m = pl.ds(pl.multiple_of(mate_r0 + mr, 32), bm_mxu)
            for cn in range(0, qkv_cols, _DOT_TN):
                acc = jnp.dot(
                    next_in_vmem[rs_m, :], w_qkv_buf[:, pl.ds(cn, _DOT_TN)],
                    preferred_element_type=jnp.float32,
                )
                qkv_scratch[rs_m, pl.ds(cn, _DOT_TN)] = (
                    acc * next_s_qkv_ref[0:1, pl.ds(cn, _DOT_TN)]
                ).astype(qkv_scratch.dtype)
        _interleaved_gate_and_qkv_epilogue(
            next_in_vmem, w_gate_buf, next_s_gate_ref, out_next_gate_ref,
            qkv_scratch, cos_sin_ref, out_next_q_ref, out_next_kv_ref,
            r_base=mate_r0, n_r=half, align=32,
            norm_eps=qkv_norm_eps, query_scale=query_scale,
            k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
        _dma_wait(out_stream_hbm.at[pl.ds(my_r0, half)], s_out_sem.at[0])
        _dma_wait(out_stream_hbm.at[pl.ds(mate_r0, half)], s_out_sem.at[1])
        return

    _gather_shard_xor1(
        post_shard, half_a, half_b,
        ag_s1_send, ag_s1_recv, ag_s2_send, ag_s2_recv, ag_s3_send, ag_s3_recv,
        me=me, world_size=world_size, use_halfrow=use_halfrow,
        during_ici_fn=lambda: _dma_wait(s_buf, s_in_sem.at[0]),
        my_half_epilogue_fn=_epi,
        mate_half_epilogue_fn=_epi,
    )
    pltpu.make_async_copy(s_buf, out_stream_hbm, s_out_sem.at[0]).start()
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    if rows < b_pad32:
        qkv_scratch[pl.ds(rows, b_pad32 - rows), :] = jnp.zeros(
            (b_pad32 - rows, qkv_cols), dtype=qkv_scratch.dtype
        )
    _dot_fp8_scaled_to_ref(
        next_in_vmem, w_qkv_buf, next_s_qkv_ref, 0, qkv_scratch, 0,
        rows=rows, n_cols=qkv_cols,
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    if rows == b_pad32:
        _interleaved_gate_and_qkv_epilogue(
            next_in_vmem, w_gate_buf, next_s_gate_ref, out_next_gate_ref,
            qkv_scratch, cos_sin_ref, out_next_q_ref, out_next_kv_ref,
            r_base=jnp.int32(0), n_r=b_pad32, align=32,
            norm_eps=qkv_norm_eps, query_scale=query_scale,
            k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    else:
        _dot_fp8_scaled_to_ref(
            next_in_vmem, w_gate_buf, next_s_gate_ref, 0, out_next_gate_ref, 0,
            rows=rows, n_cols=gate_cols,
        )
        for t_i in range(b_pad32 // 32):
            _in_vmem_qkv_epilogue_tile(
                qkv_scratch, cos_sin_ref, out_next_q_ref, out_next_kv_ref,
                b0=t_i * 32, tile_r=32,
                norm_eps=qkv_norm_eps, query_scale=query_scale,
                k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
                interpret_mode=interpret_mode,
            )
    _dma_wait(out_stream_hbm, s_out_sem.at[0])


def moe_to_attn_direct_qkv_epi_bridge_jax(
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
    cos_sin_step,
    *,
    world_size: int = 8,
    post_ff_eps: float = 1e-5,
    post_ff_scale_input: bool = True,
    next_in_eps: float = 1e-5,
    next_in_scale_input: bool = False,
    sharded_moe_in: bool = False,
    overlap_mate_epi: bool = False,
    qkv_norm_eps: float = 1e-6,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    interpret: bool = False,
    collective_id: int = _CARB_CID_BASE + 2,
):
    """Bridge 2b + QKV/RoPE epilogue -> ``(stream, q_5d [1, B, 8, 2, 128], kv [B, 1, 4, 128], gate)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        raise ValueError(f"Direct packed QKV/RoPE Bridge 2b requires TP=8 (16Q/2KV), got {world_size}")
    in_r, k_in = moe_out.shape
    rows = world_size * in_r if sharded_moe_in else in_r
    wk, post_cols = w_post_w8k.shape
    hidden = world_size * post_cols
    qkv_k, qkv_cols = next_w_qkv_w8k.shape
    gt_k, gate_cols = next_w_gate_w8k.shape
    if (
        in_r < 8
        or in_r % 8 != 0
        or rows > 384
        or stream.shape[0] != rows
        or wk != k_in
        or qkv_k != hidden
        or gt_k != hidden
        or qkv_cols != 1280
        or gate_cols != 1024
        or post_cols % _DOT_TN != 0
    ):
        raise ValueError("Unsupported moe_to_attn_direct_qkv_epi shapes")
    b_pad32 = _align32(rows)
    has_rope = cos_sin_step is not None
    cs_in = cos_sin_step
    if has_rope:
        if cos_sin_step.shape not in ((rows, 2, 128), (b_pad32, 2, 128)):
            raise ValueError(f"Expected cos_sin_step ({rows} or {b_pad32}, 2, 128), got {cos_sin_step.shape}")
        if cos_sin_step.shape[0] < b_pad32:
            cs_in = jnp.pad(cos_sin_step, ((0, b_pad32 - cos_sin_step.shape[0]), (0, 0), (0, 0)))
    n_chips = world_size // 2
    half_rows = rows // 2 if (rows >= 32 and rows % 32 == 0) else rows

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    in_specs = [
        h_sp, v_sp, h_sp, v_sp, v_sp, v_sp, v_sp, v_sp, h_sp, v_sp, h_sp, v_sp,
        v_sp if has_rope else None,
    ]
    out_stream, q_pad, kv_pad, out_gate = _pallas_call_maybe_interpret(
        functools.partial(
            _moe_to_attn_direct_qkv_epi_kernel,
            world_size=world_size,
            post_ff_eps=float(post_ff_eps),
            post_ff_scale_input=bool(post_ff_scale_input),
            next_in_eps=float(next_in_eps),
            next_in_scale_input=bool(next_in_scale_input),
            sharded_moe_in=bool(sharded_moe_in),
            overlap_mate_epi=bool(overlap_mate_epi),
            has_rope=has_rope,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            interpret_mode=bool(interpret),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
            jax.ShapeDtypeStruct((b_pad32, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((b_pad32, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=in_specs,
        out_specs=(h_sp, v_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((k_in, post_cols), w_post_w8k.dtype),
            vmem((hidden, qkv_cols), next_w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), next_w_gate_w8k.dtype),
            vmem((rows, hidden), jnp.float32),
            vmem((rows, hidden), jnp.bfloat16),
            vmem((b_pad32, qkv_cols), jnp.bfloat16),
            vmem((rows, post_cols), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((half_rows, hidden), jnp.bfloat16),
            vmem((in_r, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            vmem((rows, k_in) if sharded_moe_in else (8, 128), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((2,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((1,)),
            sem((n_chips - 1,)),
            sem((n_chips - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(58 if rows > 256 else 56) * 1024**2,
            collective_id=collective_id,
            disable_bounds_checks=True,
        ),
        name=(
            f"muse_w8c_moe_to_dqkv_b{rows}_tp{world_size}"
            f"{'_rope' if has_rope else '_nope'}"
            f"{'_sh' if sharded_moe_in else ''}"
            f"{'_ovl' if overlap_mate_epi else ''}"
        ),
    )(
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
        cs_in.astype(jnp.float32) if has_rope else None,
    )
    q_5d = (q_pad[:rows] if rows < b_pad32 else q_pad).reshape(1, rows, 8, 2, 128)
    kv_m = kv_pad[:rows] if rows < b_pad32 else kv_pad
    return out_stream, q_5d, kv_m, out_gate


def _fp8_dual_direct_qkv_epi_gate_kernel(
    attn_in_ref,
    w_qkv_hbm,
    s_qkv_ref,
    w_gate_hbm,
    s_gate_ref,
    cos_sin_ref,
    out_q_ref,
    out_kv_ref,
    out_gate_ref,
    w_qkv_buf,
    w_gate_buf,
    qkv_scratch,
    w_qkv_sem,
    w_gate_sem,
    *,
    has_rope: bool,
    qkv_norm_eps: float,
    query_scale: float,
    k_scale: float,
    v_scale: float,
    interpret_mode: bool,
) -> None:
    """L0 FP8 ``qkv_proj + gate_proj`` + pure-VMEM ``decode_qkv_epilogue`` overlapped w/ ``W_gate``!"""
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = attn_in_ref.shape[0]
    b_pad32 = qkv_scratch.shape[0]
    qkv_cols = w_qkv_hbm.shape[1]
    gate_cols = w_gate_hbm.shape[1]

    pltpu.make_async_copy(w_qkv_hbm, w_qkv_buf, w_qkv_sem.at[0]).start()
    pltpu.make_async_copy(w_gate_hbm, w_gate_buf, w_gate_sem.at[0]).start()
    if rows < b_pad32:
        qkv_scratch[pl.ds(rows, b_pad32 - rows), :] = jnp.zeros(
            (b_pad32 - rows, qkv_cols), dtype=qkv_scratch.dtype
        )
    _dma_wait(w_qkv_buf, w_qkv_sem.at[0])
    _dot_fp8_scaled_to_ref(
        attn_in_ref, w_qkv_buf, s_qkv_ref, 0, qkv_scratch, 0,
        rows=rows, n_cols=qkv_cols,
    )
    _dma_wait(w_gate_buf, w_gate_sem.at[0])
    if rows == b_pad32:
        _interleaved_gate_and_qkv_epilogue(
            attn_in_ref, w_gate_buf, s_gate_ref, out_gate_ref,
            qkv_scratch, cos_sin_ref, out_q_ref, out_kv_ref,
            r_base=jnp.int32(0), n_r=b_pad32, align=32,
            norm_eps=qkv_norm_eps, query_scale=query_scale,
            k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
            interpret_mode=interpret_mode,
        )
    else:
        _dot_fp8_scaled_to_ref(
            attn_in_ref, w_gate_buf, s_gate_ref, 0, out_gate_ref, 0,
            rows=rows, n_cols=gate_cols,
        )
        for t_i in range(b_pad32 // 32):
            _in_vmem_qkv_epilogue_tile(
                qkv_scratch, cos_sin_ref, out_q_ref, out_kv_ref,
                b0=t_i * 32, tile_r=32,
                norm_eps=qkv_norm_eps, query_scale=query_scale,
                k_scale=k_scale, v_scale=v_scale, has_rope=has_rope,
                interpret_mode=interpret_mode,
            )


def fp8_dual_direct_qkv_epi_gate_jax(
    attn_in,
    w_qkv_w8k,
    s_qkv,
    w_gate_w8k,
    s_gate,
    cos_sin_step,
    *,
    qkv_norm_eps: float = 1e-6,
    query_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    interpret: bool = False,
):
    """L0 ``fp8_dual_qkv_gate`` + ``decode_qkv_epilogue`` -> ``(q_5d, kv_merged, gate)``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, hidden = attn_in.shape
    qk, qkv_cols = w_qkv_w8k.shape
    gk, gate_cols = w_gate_w8k.shape
    if (
        rows < 8
        or rows > 384
        or rows % 8 != 0
        or qk != hidden
        or gk != hidden
        or qkv_cols != 1280
        or gate_cols != 1024
    ):
        raise ValueError("Unsupported fp8_dual_direct_qkv_epi_gate shapes")
    b_pad32 = _align32(rows)
    has_rope = cos_sin_step is not None
    cs_in = cos_sin_step
    if has_rope:
        if cos_sin_step.shape not in ((rows, 2, 128), (b_pad32, 2, 128)):
            raise ValueError(f"Expected cos_sin_step ({rows} or {b_pad32}, 2, 128), got {cos_sin_step.shape}")
        if cos_sin_step.shape[0] < b_pad32:
            cs_in = jnp.pad(cos_sin_step, ((0, b_pad32 - cos_sin_step.shape[0]), (0, 0), (0, 0)))

    vmem, hbm, sem = pltpu.VMEM, pltpu.HBM, pltpu.SemaphoreType.DMA
    v_sp, h_sp = pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=hbm)

    q_pad, kv_pad, out_gate = _pallas_call_maybe_interpret(
        functools.partial(
            _fp8_dual_direct_qkv_epi_gate_kernel,
            has_rope=has_rope,
            qkv_norm_eps=float(qkv_norm_eps),
            query_scale=float(query_scale),
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            interpret_mode=bool(interpret),
        ),
        interpret=interpret,
        out_shape=(
            jax.ShapeDtypeStruct((b_pad32, 16, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((b_pad32, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, gate_cols), jnp.bfloat16),
        ),
        in_specs=[v_sp, h_sp, v_sp, h_sp, v_sp, v_sp if has_rope else None],
        out_specs=(v_sp, v_sp, v_sp),
        scratch_shapes=[
            vmem((hidden, qkv_cols), w_qkv_w8k.dtype),
            vmem((hidden, gate_cols), w_gate_w8k.dtype),
            vmem((b_pad32, qkv_cols), jnp.bfloat16),
            sem((1,)),
            sem((1,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(48 if rows > 256 else 40) * 1024**2,
            disable_bounds_checks=True,
        ),
        name=f"muse_w8c_dual_dqkv_b{rows}_{'rope' if has_rope else 'nope'}",
    )(
        attn_in.astype(jnp.bfloat16),
        w_qkv_w8k,
        _as_2d_f32_row(s_qkv, qkv_cols),
        w_gate_w8k,
        _as_2d_f32_row(s_gate, gate_cols),
        cs_in.astype(jnp.float32) if has_rope else None,
    )
    q_5d = (q_pad[:rows] if rows < b_pad32 else q_pad).reshape(1, rows, 8, 2, 128)
    kv_m = kv_pad[:rows] if rows < b_pad32 else kv_pad
    return q_5d, kv_m, out_gate


# =============================================================================
# 3. TorchTPU `cpu_export_jax_op` Registration & Lock-Free Prewarming
# =============================================================================
def get_attn_to_moe_bridge_router_topk_op(
    world_size: int, *, bf16_router: bool = True
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(bf16_router))
    op = _ATTN_TO_MOE_RTOPK_OPS.get(key)
    if op is not None:
        return op
    with _CARB_LOCK:
        op = _ATTN_TO_MOE_RTOPK_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            t_specs = (
                P(),
                P(None, "model"),
                P("model", None),
                P("model"),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
            )
            in_specs = (*t_specs, None, None, None, None, None, None, None, None)
            out_specs = (P(), P(), P())

            def _wrapped(
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
                router_bias: jax.Array,
                pre_exp_gain: jax.Array,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                route_eps: float,
                route_scale: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                def _local(st, ga, wo, ws, a, b, pfg, wp, wps, wr, rb, peg):
                    ws_row = ws[0] if ws.ndim == 3 else ws
                    return attn_to_moe_bridge_router_topk_jax(
                        st, ga, wo, ws_row, a, b, pfg, wp, wps, wr, rb, peg,
                        world_size=world_size,
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        route_eps=float(route_eps),
                        route_scale=float(route_scale),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream, gated_attn, wo_w8k, wo_scale, pa_alpha, pa_beta,
                    pre_ff_gain, w_pre_w8k, w_pre_scale, w_router_km, router_bias, pre_exp_gain,
                )

            tag = "r16" if bf16_router else "r32"
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8c_attn_to_moe_rtopk_{tag}_tp{world_size}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _ATTN_TO_MOE_RTOPK_OPS[key] = op
    return op


def get_moe_to_attn_overlap_bridge_op(
    world_size: int, *, sharded_moe_in: bool = False
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in))
    op = _MOE_TO_ATTN_OVERLAP_OPS.get(key)
    if op is not None:
        return op
    with _CARB_LOCK:
        op = _MOE_TO_ATTN_OVERLAP_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            t_specs = (
                P(),
                moe_sp,
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
            )
            in_specs = (*t_specs, None, None, None, None)
            out_specs = (P(), P(None, "model"), P(None, "model"))

            def _wrapped(
                stream: jax.Array,
                moe_out: jax.Array,
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
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                fn = functools.partial(
                    moe_to_attn_overlap_bridge_jax,
                    world_size=world_size,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    sharded_moe_in=bool(sharded_moe_in),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                    next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                )

            sfx = "_sh" if sharded_moe_in else ""
            op = cpu_export_jax_op(
                f"sglang_tpu_pallas::muse_w8c_moe_to_qkvg_ovl_tp{world_size}{sfx}",
                _wrapped,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
            _MOE_TO_ATTN_OVERLAP_OPS[key] = op
    return op


def get_moe_to_attn_direct_qkv_epi_bridge_op(
    world_size: int = 8,
    *,
    has_rope: bool = True,
    sharded_moe_in: bool = False,
    overlap_mate_epi: bool = False,
) -> Callable:
    if world_size != 8:
        raise ValueError(f"Direct QKV/RoPE Bridge 2b requires TP=8, got {world_size}")
    key = (int(world_size), bool(has_rope), bool(sharded_moe_in), bool(overlap_mate_epi))
    op = _MOE_TO_ATTN_DIRECT_QKV_OPS.get(key)
    if op is not None:
        return op
    with _CARB_LOCK:
        op = _MOE_TO_ATTN_DIRECT_QKV_OPS.get(key)
        if op is None:
            import jax
            from jax.sharding import PartitionSpec as P

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            mesh = _get_tp_mesh(world_size)
            moe_sp = P("model", None) if sharded_moe_in else P()
            out_specs = (
                P(),
                P(None, None, "model", None, None),
                P(None, "model", None, None),
                P(None, "model"),
            )
            base_t_specs = (
                P(),
                moe_sp,
                P(None, "model"),
                P("model"),
                P(),
                P(),
                P(),
                P(),
                P(None, "model"),
                P("model"),
                P(None, "model"),
                P("model"),
            )
            sfx = f"{'_rope' if has_rope else '_nope'}{'_sh' if sharded_moe_in else ''}{'_ovl' if overlap_mate_epi else ''}"
            if has_rope:
                t_specs = (*base_t_specs, P())
                in_specs = (*t_specs, None, None, None, None, None, None, None, None)

                def _wrapped_rope(
                    stream: jax.Array,
                    moe_out: jax.Array,
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
                    cos_sin_step: jax.Array,
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    fn = functools.partial(
                        moe_to_attn_direct_qkv_epi_bridge_jax,
                        world_size=world_size,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        sharded_moe_in=bool(sharded_moe_in),
                        overlap_mate_epi=bool(overlap_mate_epi),
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )
                    return jax.shard_map(
                        fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                    )(
                        stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                        next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                        cos_sin_step,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8c_moe_to_dqkv_tp{world_size}{sfx}",
                    _wrapped_rope,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_specs,
                )
            else:
                t_specs = base_t_specs
                in_specs = (*t_specs, None, None, None, None, None, None, None, None)

                def _wrapped_nope(
                    stream: jax.Array,
                    moe_out: jax.Array,
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
                    post_ff_eps: float,
                    post_ff_scale_input: bool,
                    next_in_eps: float,
                    next_in_scale_input: bool,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    def _local_nope(st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg):
                        return moe_to_attn_direct_qkv_epi_bridge_jax(
                            st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, None,
                            world_size=world_size,
                            post_ff_eps=float(post_ff_eps),
                            post_ff_scale_input=bool(post_ff_scale_input),
                            next_in_eps=float(next_in_eps),
                            next_in_scale_input=bool(next_in_scale_input),
                            sharded_moe_in=bool(sharded_moe_in),
                            overlap_mate_epi=bool(overlap_mate_epi),
                            qkv_norm_eps=float(qkv_norm_eps),
                            query_scale=float(query_scale),
                            k_scale=float(k_scale),
                            v_scale=float(v_scale),
                        )

                    return jax.shard_map(
                        _local_nope, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                    )(
                        stream, moe_out, w_post_w8k, s_post, post_ff_gain, pf_alpha, pf_beta,
                        next_in_gain, next_w_qkv_w8k, next_s_qkv, next_w_gate_w8k, next_s_gate,
                    )

                op = cpu_export_jax_op(
                    f"sglang_tpu_pallas::muse_w8c_moe_to_dqkv_tp{world_size}{sfx}",
                    _wrapped_nope,
                    mesh=mesh,
                    input_partition_specs=in_specs,
                    output_partition_specs=out_specs,
                )
            _MOE_TO_ATTN_DIRECT_QKV_OPS[key] = op
    return op


def get_fp8_dual_direct_qkv_epi_gate_op(*, has_rope: bool = True) -> Callable:
    key = bool(has_rope)
    op = _DUAL_DIRECT_QKV_OPS.get(key)
    if op is not None:
        return op
    with _CARB_LOCK:
        op = _DUAL_DIRECT_QKV_OPS.get(key)
        if op is None:
            import jax

            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            if has_rope:

                def _wrapped_r(
                    attn_in: jax.Array,
                    w_qkv_w8k: jax.Array,
                    s_qkv: jax.Array,
                    w_gate_w8k: jax.Array,
                    s_gate: jax.Array,
                    cos_sin_step: jax.Array,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return fp8_dual_direct_qkv_epi_gate_jax(
                        attn_in, w_qkv_w8k, s_qkv, w_gate_w8k, s_gate, cos_sin_step,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )

                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::muse_w8c_dual_dqkv_rope", _wrapped_r
                )
            else:

                def _wrapped_n(
                    attn_in: jax.Array,
                    w_qkv_w8k: jax.Array,
                    s_qkv: jax.Array,
                    w_gate_w8k: jax.Array,
                    s_gate: jax.Array,
                    qkv_norm_eps: float,
                    query_scale: float,
                    k_scale: float,
                    v_scale: float,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return fp8_dual_direct_qkv_epi_gate_jax(
                        attn_in, w_qkv_w8k, s_qkv, w_gate_w8k, s_gate, None,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                    )

                op = cpu_export_jax_op(
                    "sglang_tpu_pallas::muse_w8c_dual_dqkv_nope", _wrapped_n
                )
            _DUAL_DIRECT_QKV_OPS[key] = op
    return op


def prewarm_carb_bridge_ops(
    world_size: int,
    *,
    bf16_router: bool = False,
    fuse_b1_topk: bool = False,
    fuse_b2_direct_qkv: bool = False,
    overlap_b2_epi: bool = False,
) -> None:
    """Eagerly register all opted-in Wave-8 carb bridge custom ops outside ``torch.compile``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        return
    if fuse_b1_topk:
        get_attn_to_moe_bridge_router_topk_op(world_size, bf16_router=bool(bf16_router))
    if overlap_b2_epi:
        for sh in (False, True):
            get_moe_to_attn_overlap_bridge_op(world_size, sharded_moe_in=sh)
    if fuse_b2_direct_qkv and world_size == 8:
        for hr in (False, True):
            get_fp8_dual_direct_qkv_epi_gate_op(has_rope=hr)
            for sh in (False, True):
                get_moe_to_attn_direct_qkv_epi_bridge_op(
                    world_size,
                    has_rope=hr,
                    sharded_moe_in=sh,
                    overlap_mate_epi=bool(overlap_b2_epi),
                )
