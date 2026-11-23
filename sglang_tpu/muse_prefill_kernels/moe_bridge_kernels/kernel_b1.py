# SPDX-License-Identifier: Apache-2.0
"""Kernel B1: Fused Post-Attention Norm + Gated Residual + Pre-FFN Norm + Router Gate + In-VMEM Top-8."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128


def exact_rsqrt(v: jax.Array) -> jax.Array:
    """Full 24-bit IEEE FP32 rsqrt via 1 Newton-Raphson step on Mosaic's 12-bit vrsqrt LUT."""
    y = jax.lax.rsqrt(v)
    return y * (jnp.float32(1.5) - jnp.float32(0.5) * v * (y * y))


def pallas_rsqrt_precision_probe(v_2d: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Step-1 diagnostic probe: returns (raw_lax_rsqrt, nr1_exact_rsqrt) from inside a Pallas TPU kernel."""
    m, n = v_2d.shape
    assert m % 8 == 0 and n % LANE == 0

    def _probe_kernel(v_ref, raw_out_ref, nr_out_ref):
        v = v_ref[:, :]
        raw_out_ref[:, :] = jax.lax.rsqrt(v)
        nr_out_ref[:, :] = exact_rsqrt(v)

    return pl.pallas_call(
        _probe_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, n), jnp.float32),
            jax.ShapeDtypeStruct((m, n), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[pl.BlockSpec((m, n), lambda i: (0, 0))],
            out_specs=[
                pl.BlockSpec((m, n), lambda i: (0, 0)),
                pl.BlockSpec((m, n), lambda i: (0, 0)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="pallas_rsqrt_precision_probe",
    )(v_2d)


def vpu_tree_sum_128(x: jax.Array) -> jax.Array:
    """Fold a 2D array [M, K] (where K % 128 == 0) into [M, 128] via a vertical VPU tree."""
    n_tiles = x.shape[1] // LANE
    tiles = [x[:, k * LANE : (k + 1) * LANE] for k in range(n_tiles)]
    while len(tiles) > 1:
        tiles = [
            (tiles[2 * i] + tiles[2 * i + 1])
            if (2 * i + 1 < len(tiles))
            else tiles[2 * i]
            for i in range((len(tiles) + 1) // 2)
        ]
    return tiles[0]


def vpu_tree_max_128(x: jax.Array) -> jax.Array:
    """Fold a non-negative 2D array [M, K] (where K % 128 == 0) into [M, 128] via a VPU max tree."""
    n_tiles = x.shape[1] // LANE
    tiles = [x[:, k * LANE : (k + 1) * LANE] for k in range(n_tiles)]
    while len(tiles) > 1:
        tiles = [
            jnp.maximum(tiles[2 * i], tiles[2 * i + 1])
            if (2 * i + 1 < len(tiles))
            else tiles[2 * i]
            for i in range((len(tiles) + 1) // 2)
        ]
    return tiles[0]


def prepare_b1_static_inputs(
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    e_score_bias: jax.Array,
    wrouter: jax.Array,
    *,
    router_mode: str = "bf16_hilo",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pack 1D norm/residual vectors into VMEM-aligned 2D tensors and split wrouter if requested."""
    if alpha_attn.ndim == 2 and alpha_attn.shape[0] == 8:
        vecs_8x8192 = alpha_attn
    else:
        h_dim = alpha_attn.shape[-1]
        a_1d = alpha_attn.reshape(1, h_dim).astype(jnp.float32)
        b_1d = beta_attn.reshape(1, h_dim).astype(jnp.float32)
        g_1d = pre_ffn_norm_w.reshape(1, h_dim).astype(jnp.float32) + jnp.float32(1.0)
        z_1d = jnp.zeros((1, h_dim), dtype=jnp.float32)
        vecs_8x8192 = jnp.concatenate(
            [a_1d, b_1d, g_1d, z_1d, z_1d, z_1d, z_1d, z_1d], axis=0
        )

    if e_score_bias.ndim == 2 and e_score_bias.shape[1] == LANE:
        num_experts = e_score_bias.shape[0]
        bias_t_256x128 = (
            e_score_bias
            if e_score_bias.dtype == jnp.float32
            else e_score_bias.astype(jnp.float32)
        )
    else:
        num_experts = e_score_bias.shape[-1]
        bias_t_256x128 = jnp.broadcast_to(
            e_score_bias.reshape(num_experts, 1).astype(jnp.float32),
            (num_experts, LANE),
        )

    if router_mode == "bf16_hilo_pre":
        w_hi = w_lo = wrouter
    elif router_mode == "fp32":
        w_hi = wrouter.astype(jnp.float32)
        w_lo = jnp.zeros((8, num_experts), dtype=jnp.bfloat16)
    elif router_mode == "bf16_hilo":
        w_f32 = wrouter.astype(jnp.float32)
        w_hi = w_f32.astype(jnp.bfloat16)
        w_lo = (w_f32 - w_hi.astype(jnp.float32)).astype(jnp.bfloat16)
    else:
        w_hi = wrouter.astype(jnp.bfloat16)
        w_lo = jnp.zeros((8, num_experts), dtype=jnp.bfloat16)

    return vecs_8x8192, bias_t_256x128, w_hi, w_lo


def pallas_postattn_res_prenorm_router_topk(
    attn_out_shard: jax.Array,
    stream_shard: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    *,
    bm: int = 128,
    router_mode: str = "bf16_hilo",
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
    topk: int = 8,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused Kernel B1 on token-sharded [T_local, 8192].

    Returns:
        router_in_bf16: bf16[T_local, 8192]
        stream_shard_mid: fp32[T_local, 8192]
        local_topk_ids_u8: uint8[T_local, topk]
        local_topk_weights: bf16[T_local, topk]
    """
    from sglang_tpu.envs import envs as _envs

    t_local, h_dim = attn_out_shard.shape
    num_experts = wrouter.shape[1]
    b1_fast_env = getattr(_envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
    use_b1_fast = bool(b1_fast_env is not None and b1_fast_env.get())
    # "bf16_hilo_pre": wrouter is the bf16 split already, [hi; lo] rows of [2 * H, E] (or [H, E] hi-only).
    hilo_pre = router_mode == "bf16_hilo_pre"
    kernel_name = router_mode
    prep_mode = "bf16_hilo_pre" if hilo_pre else router_mode
    router_mode = "bf16_hilo" if hilo_pre else router_mode
    skip_zero_wlo = hilo_pre and (use_b1_fast or wrouter.shape[0] == h_dim)
    assert t_local % bm == 0, f"t_local={t_local} must be divisible by bm={bm}"
    assert bm == LANE, (
        f"Kernel B1 requires bm == 128 for single-tile _topk_xpose, got bm={bm}"
    )
    assert h_dim == 8192 and num_experts % 8 == 0

    vecs_8x8192, bias_t_256x128, w_hi, w_lo = prepare_b1_static_inputs(
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        e_score_bias,
        wrouter,
        router_mode=prep_mode,
    )
    if skip_zero_wlo:
        w_lo = jnp.zeros((8, num_experts), dtype=jnp.bfloat16)

    num_steps = t_local // bm
    strip_w = 256 if use_b1_fast else 2048
    num_strips = h_dim // strip_w
    inv_h = 1.0 / float(h_dim)

    def _b1_kernel(
        attn_ref,
        stream_ref,
        vecs_ref,
        bias_t_ref,
        w_hi_ref,
        w_lo_ref,
        router_in_out_ref,
        stream_mid_out_ref,
        ids_out_ref,
        ws_out_ref,
    ):
        # Pass 1: Post-attention RMSNorm variance across strips (256-lane zero-spill when use_b1_fast)
        sq_a_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            sq_a_128 = sq_a_128 + vpu_tree_sum_128(a_s * a_s)

        inv_rms_a = jax.lax.rsqrt(
            jnp.sum(sq_a_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(post_norm_eps)
        )

        # Pass 2: Gated residual + Pre-FFN RMSNorm variance in a single sweep
        sq_m_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            # Exact Muse Spark rounding: (a * s_a).to(bf16).to(f32)
            branch_s = (a_s * inv_rms_a).astype(jnp.bfloat16).astype(jnp.float32)
            alpha_s = vecs_ref[0:1, c0:c1]
            beta_s = vecs_ref[1:2, c0:c1]
            smid_s = alpha_s * stream_ref[:, c0:c1] + beta_s * branch_s
            stream_mid_out_ref[:, c0:c1] = smid_s
            sq_m_128 = sq_m_128 + vpu_tree_sum_128(smid_s * smid_s)

        inv_rms_m = jax.lax.rsqrt(
            jnp.sum(sq_m_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(rms_eps)
        )

        # Pass 3: Normalize to router_in_bf16 + accumulate in-VMEM router dot [128, 256]
        logits = jnp.zeros((bm, num_experts), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            smid_s = stream_mid_out_ref[:, c0:c1]
            gain_s = vecs_ref[2:3, c0:c1]
            r_f32 = smid_s * inv_rms_m * gain_s
            r_bf16 = r_f32.astype(jnp.bfloat16)
            router_in_out_ref[:, c0:c1] = r_bf16

            if router_mode == "fp32":
                logits = logits + jnp.dot(
                    r_f32,
                    w_hi_ref[c0:c1, :],
                    precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                )
            elif router_mode == "bf16_hilo":
                r_hi = r_bf16
                r_lo = (r_f32 - r_hi.astype(jnp.float32)).astype(jnp.bfloat16)
                w_hi_sub = w_hi_ref[c0:c1, :]
                if skip_zero_wlo:
                    logits = (
                        logits
                        + jnp.dot(r_hi, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
                else:
                    w_lo_sub = w_lo_ref[c0:c1, :]
                    logits = (
                        logits
                        + jnp.dot(r_hi, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_hi, w_lo_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
            else:
                logits = logits + jnp.dot(
                    r_bf16,
                    w_hi_ref[c0:c1, :],
                    preferred_element_type=jnp.float32,
                )

        # In-VMEM sigmoid + single-transpose _topk_xpose
        sc = jax.nn.sigmoid(logits * jnp.float32(router_multiplier))
        sc_t = jnp.swapaxes(sc, 0, 1)  # [256, 128]
        cur = sc_t + bias_t_ref[:, :]  # [256, 128]
        iota = jax.lax.broadcasted_iota(jnp.int32, (num_experts, bm), 0)

        ids_list = []
        ws_list = []
        for _ in range(topk):
            top = jnp.max(cur, axis=0, keepdims=True)
            idx = jnp.min(
                jnp.where(cur == top, iota, jnp.int32(num_experts)),
                axis=0,
                keepdims=True,
            )
            m = iota == idx
            w_j = jnp.max(
                jnp.where(m, sc_t, jnp.float32(-jnp.inf)),
                axis=0,
                keepdims=True,
            )
            cur = jnp.where(m, jnp.float32(-jnp.inf), cur)
            ids_list.append(idx)
            ws_list.append(w_j)

        ids_k_bm = jnp.concatenate(ids_list, axis=0)  # [8, 128] int32
        ws_k_bm = jnp.concatenate(ws_list, axis=0)  # [8, 128] fp32
        ws_k_bm = (
            ws_k_bm / (jnp.sum(ws_k_bm, axis=0, keepdims=True) + jnp.float32(route_eps))
        ) * jnp.float32(route_scale)

        ids_pad = jnp.pad(ids_k_bm, ((0, LANE - topk), (0, 0)))
        ws_pad = jnp.pad(ws_k_bm.astype(jnp.bfloat16), ((0, LANE - topk), (0, 0)))
        ids_out_ref[:, :] = jnp.swapaxes(ids_pad, 0, 1)
        ws_out_ref[:, :] = jnp.swapaxes(ws_pad, 0, 1)

    w_lo_block_shape = (
        (h_dim, num_experts)
        if (router_mode == "bf16_hilo" and not skip_zero_wlo)
        else (8, num_experts)
    )
    w_lo_idx = 1 if (hilo_pre and not skip_zero_wlo) else 0
    fast_sfx = "_f256" if use_b1_fast else ""
    router_in_bf16, stream_shard_mid, ids_pad_out, ws_pad_out = pl.pallas_call(
        _b1_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t_local, h_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((t_local, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((t_local, LANE), jnp.int32),
            jax.ShapeDtypeStruct((t_local, LANE), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((8, h_dim), lambda i: (0, 0)),
                pl.BlockSpec((num_experts, LANE), lambda i: (0, 0)),
                pl.BlockSpec((h_dim, num_experts), lambda i: (0, 0)),
                pl.BlockSpec(w_lo_block_shape, lambda i, _idx=w_lo_idx: (_idx, 0)),
            ],
            out_specs=[
                pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
                pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"pallas_b1_postattn_router_{kernel_name}_bm{bm}{fast_sfx}",
    )(
        attn_out_shard,
        stream_shard,
        vecs_8x8192,
        bias_t_256x128,
        w_hi,
        w_lo,
    )

    ids_dtype = jnp.uint8 if num_experts <= 256 else jnp.uint16
    local_topk_ids_u8 = ids_pad_out[:, :topk].astype(ids_dtype)
    local_topk_weights = ws_pad_out[:, :topk]
    return router_in_bf16, stream_shard_mid, local_topk_ids_u8, local_topk_weights


def pallas_postattn_res_prenorm_router_topk_fp8(
    attn_out_shard: jax.Array,
    stream_shard: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    *,
    bm: int = 128,
    router_mode: str = "bf16_hilo",
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
    topk: int = 8,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused Kernel B1 with algebraic-cancel FP8 router_in output for native FP8xFP8 Kernel B2.

    Tracks row_amax of unnormalized (stream_mid * gain) in Pass 2 alongside sq_m_128, so Pass 3
    emits router_in_fp8: fp8_e4m3fn[T_local, 8192] + router_in_sc_pad: fp32[T_local, 128]
    in a single sweep with 50% less HBM write traffic.
    """
    from sglang_tpu.envs import envs as _envs

    t_local, h_dim = attn_out_shard.shape
    num_experts = wrouter.shape[1]
    b1_fast_env = getattr(_envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
    use_b1_fast = bool(b1_fast_env is not None and b1_fast_env.get())
    hilo_pre = router_mode == "bf16_hilo_pre"
    kernel_name = router_mode
    prep_mode = "bf16_hilo_pre" if hilo_pre else router_mode
    router_mode = "bf16_hilo" if hilo_pre else router_mode
    skip_zero_wlo = hilo_pre and (use_b1_fast or wrouter.shape[0] == h_dim)
    assert t_local % bm == 0 and bm == LANE
    assert h_dim == 8192 and num_experts % 8 == 0

    vecs_8x8192, bias_t_256x128, w_hi, w_lo = prepare_b1_static_inputs(
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        e_score_bias,
        wrouter,
        router_mode=prep_mode,
    )
    if skip_zero_wlo:
        w_lo = jnp.zeros((8, num_experts), dtype=jnp.bfloat16)

    num_steps = t_local // bm
    strip_w = 256 if use_b1_fast else 2048
    num_strips = h_dim // strip_w
    inv_h = 1.0 / float(h_dim)

    def _b1_fp8_kernel(
        attn_ref,
        stream_ref,
        vecs_ref,
        bias_t_ref,
        w_hi_ref,
        w_lo_ref,
        router_in_fp8_out_ref,
        router_in_sc_out_ref,
        stream_mid_out_ref,
        ids_out_ref,
        ws_out_ref,
    ):
        sq_a_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            sq_a_128 = sq_a_128 + vpu_tree_sum_128(a_s * a_s)

        inv_rms_a = jax.lax.rsqrt(
            jnp.sum(sq_a_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(post_norm_eps)
        )

        # Pass 2: Gated residual + Pre-FFN variance + unnormalized row_amax in ONE sweep
        sq_m_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        amax_unnorm_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            branch_s = (a_s * inv_rms_a).astype(jnp.bfloat16).astype(jnp.float32)
            alpha_s = vecs_ref[0:1, c0:c1]
            beta_s = vecs_ref[1:2, c0:c1]
            gain_s = vecs_ref[2:3, c0:c1]
            smid_s = alpha_s * stream_ref[:, c0:c1] + beta_s * branch_s
            stream_mid_out_ref[:, c0:c1] = smid_s
            sq_m_128 = sq_m_128 + vpu_tree_sum_128(smid_s * smid_s)
            amax_unnorm_128 = jnp.maximum(
                amax_unnorm_128,
                vpu_tree_max_128(jnp.abs(smid_s * gain_s)),
            )

        inv_rms_m = jax.lax.rsqrt(
            jnp.sum(sq_m_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(rms_eps)
        )
        row_amax_unnorm = jnp.max(amax_unnorm_128, axis=1, keepdims=True)
        row_amax_r = row_amax_unnorm * inv_rms_m
        r_sc = row_amax_r * jnp.float32(1.0 / 448.0)
        r_sc = jnp.where(row_amax_r == 0, jnp.zeros_like(r_sc), r_sc)
        inv_r_sc = jnp.where(
            row_amax_r == 0,
            jnp.zeros_like(row_amax_r),
            jnp.float32(448.0) / jnp.maximum(row_amax_r, jnp.float32(1e-30)),
        )
        router_in_sc_out_ref[:, :] = jnp.broadcast_to(r_sc, (bm, LANE))

        logits = jnp.zeros((bm, num_experts), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            smid_s = stream_mid_out_ref[:, c0:c1]
            gain_s = vecs_ref[2:3, c0:c1]
            r_f32 = smid_s * inv_rms_m * gain_s
            r_bf16 = r_f32.astype(jnp.bfloat16)
            router_in_fp8_out_ref[:, c0:c1] = jnp.clip(
                r_bf16.astype(jnp.float32) * inv_r_sc, -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)

            if router_mode == "fp32":
                logits = logits + jnp.dot(
                    r_f32,
                    w_hi_ref[c0:c1, :],
                    precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                )
            elif router_mode == "bf16_hilo":
                r_hi = r_bf16
                r_lo = (r_f32 - r_hi.astype(jnp.float32)).astype(jnp.bfloat16)
                w_hi_sub = w_hi_ref[c0:c1, :]
                if skip_zero_wlo:
                    logits = (
                        logits
                        + jnp.dot(r_hi, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
                else:
                    w_lo_sub = w_lo_ref[c0:c1, :]
                    logits = (
                        logits
                        + jnp.dot(r_hi, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_hi, w_lo_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
            else:
                logits = logits + jnp.dot(
                    r_bf16,
                    w_hi_ref[c0:c1, :],
                    preferred_element_type=jnp.float32,
                )

        sc = jax.nn.sigmoid(logits * jnp.float32(router_multiplier))
        sc_t = jnp.swapaxes(sc, 0, 1)
        cur = sc_t + bias_t_ref[:, :]
        iota = jax.lax.broadcasted_iota(jnp.int32, (num_experts, bm), 0)

        ids_list = []
        ws_list = []
        for _ in range(topk):
            top = jnp.max(cur, axis=0, keepdims=True)
            idx = jnp.min(
                jnp.where(cur == top, iota, jnp.int32(num_experts)),
                axis=0,
                keepdims=True,
            )
            m = iota == idx
            w_j = jnp.max(
                jnp.where(m, sc_t, jnp.float32(-jnp.inf)),
                axis=0,
                keepdims=True,
            )
            cur = jnp.where(m, jnp.float32(-jnp.inf), cur)
            ids_list.append(idx)
            ws_list.append(w_j)

        ids_k_bm = jnp.concatenate(ids_list, axis=0)
        ws_k_bm = jnp.concatenate(ws_list, axis=0)
        ws_k_bm = (
            ws_k_bm / (jnp.sum(ws_k_bm, axis=0, keepdims=True) + jnp.float32(route_eps))
        ) * jnp.float32(route_scale)

        ids_pad = jnp.pad(ids_k_bm, ((0, LANE - topk), (0, 0)))
        ws_pad = jnp.pad(ws_k_bm.astype(jnp.bfloat16), ((0, LANE - topk), (0, 0)))
        ids_out_ref[:, :] = jnp.swapaxes(ids_pad, 0, 1)
        ws_out_ref[:, :] = jnp.swapaxes(ws_pad, 0, 1)

    w_lo_block_shape = (
        (h_dim, num_experts)
        if (router_mode == "bf16_hilo" and not skip_zero_wlo)
        else (8, num_experts)
    )
    w_lo_idx = 1 if (hilo_pre and not skip_zero_wlo) else 0
    fast_sfx = "_f256" if use_b1_fast else ""
    router_in_fp8, router_in_sc_pad, stream_shard_mid, ids_pad_out, ws_pad_out = (
        pl.pallas_call(
            _b1_fp8_kernel,
            out_shape=(
                jax.ShapeDtypeStruct((t_local, h_dim), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((t_local, LANE), jnp.float32),
                jax.ShapeDtypeStruct((t_local, h_dim), jnp.float32),
                jax.ShapeDtypeStruct((t_local, LANE), jnp.int32),
                jax.ShapeDtypeStruct((t_local, LANE), jnp.bfloat16),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_steps,),
                in_specs=[
                    pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                    pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                    pl.BlockSpec((8, h_dim), lambda i: (0, 0)),
                    pl.BlockSpec((num_experts, LANE), lambda i: (0, 0)),
                    pl.BlockSpec((h_dim, num_experts), lambda i: (0, 0)),
                    pl.BlockSpec(w_lo_block_shape, lambda i, _idx=w_lo_idx: (_idx, 0)),
                ],
                out_specs=[
                    pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                    pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
                    pl.BlockSpec((bm, h_dim), lambda i: (i, 0)),
                    pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
                    pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=60 * 1024 * 1024,
            ),
            name=f"pallas_b1_fp8_postattn_router_{kernel_name}_bm{bm}{fast_sfx}",
        )(
            attn_out_shard,
            stream_shard,
            vecs_8x8192,
            bias_t_256x128,
            w_hi,
            w_lo,
        )
    )

    ids_dtype = jnp.uint8 if num_experts <= 256 else jnp.uint16
    local_topk_ids_u8 = ids_pad_out[:, :topk].astype(ids_dtype)
    local_topk_weights = ws_pad_out[:, :topk]
    return (
        router_in_fp8,
        router_in_sc_pad,
        stream_shard_mid,
        local_topk_ids_u8,
        local_topk_weights,
    )
