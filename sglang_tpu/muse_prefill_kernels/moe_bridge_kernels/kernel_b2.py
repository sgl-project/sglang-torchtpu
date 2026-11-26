# SPDX-License-Identifier: Apache-2.0
"""Kernel B2: Fused wpool (FP8xBF16) + Full-Row pre_expert_norm + 2D INT8 Quantization & 2-Half AG Dequant."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs

from .kernel_b1 import (
    LANE,
    prepare_b1_static_inputs,
    vpu_tree_max_128,
    vpu_tree_sum_128,
)


def prepare_b2_static_inputs(
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
) -> jax.Array:
    """Pack wpool_scale and pre_expert_gain into an (8, 4096) VMEM-aligned fp32 tensor."""
    if wpool_scale.ndim == 2 and wpool_scale.shape[0] == 8:
        return wpool_scale
    he_dim = wpool_scale.shape[-1]
    ws_1d = wpool_scale.reshape(1, he_dim).astype(jnp.float32)
    gn_1d = pre_expert_norm_w.reshape(1, he_dim).astype(jnp.float32) + jnp.float32(1.0)
    z_1d = jnp.zeros((1, he_dim), dtype=jnp.float32)
    return jnp.concatenate([ws_1d, gn_1d, z_1d, z_1d, z_1d, z_1d, z_1d, z_1d], axis=0)


def pallas_wpool_prenorm_int8_quant_nstream(
    router_in_bf16: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    *,
    m_pass: int = 1024,
    bn: int = 1024,
    rms_eps: float = 1e-5,
    pack_bf16_sc: bool = False,
    half_idx: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Layout B2-NStream: LHS-resident [m_pass, 8192], N-streamed wpool_fp8 [8192, bn].

    Exposes only the first 8 MiB weight slab DMA (~6.1 us) and hides the remaining 24 MiB
    behind MXU compute. Sub-tiles register accumulators to [128, 256] (32 VREGs, 0 spills).
    When `half_idx` is 0 or 1, reads the corresponding half of `router_in_bf16` directly via
    BlockSpec index offset without an HBM slice/copy (`%fusion.3`).
    When `pack_bf16_sc=True`, packs exact 16-bit bf16 row scales into columns `[4096:4224]`
    of `x_moe_i8`, eliminating separate sync scale `psum` collectives.

    Returns:
        x_moe_i8: int8[out_rows, 4096] (or int8[out_rows, 4224] when pack_bf16_sc=True)
        x_moe_sc_1d: bf16[out_rows] (or None when pack_bf16_sc=True)
    """
    t_rows, h_dim = router_in_bf16.shape
    w_k, he_dim = wpool_fp8.shape
    assert h_dim == w_k == 8192 and he_dim == 4096
    out_rows = (t_rows // 2) if half_idx is not None else t_rows
    m_tile = min(m_pass, out_rows)
    assert out_rows % m_tile == 0, (
        f"out_rows={out_rows} must be divisible by m_tile={m_tile}"
    )
    assert m_tile % LANE == 0, f"m_tile={m_tile} must be a multiple of 128"
    assert he_dim % bn == 0 and bn % 256 == 0

    num_m = out_rows // m_tile
    m_offset = (half_idx * num_m) if half_idx is not None else 0
    num_n = he_dim // bn
    if wpool_scale.ndim == 3 and wpool_scale.shape == (num_n, 8, bn):
        vecs_3d = wpool_scale
    else:
        vecs_8x4096 = prepare_b2_static_inputs(wpool_scale, pre_expert_norm_w)
        vecs_3d = vecs_8x4096.reshape(8, num_n, bn).swapaxes(0, 1)
    b2_fast_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
    use_b2_fast = bool(b2_fast_env is not None and b2_fast_env.get())
    sub_m = LANE
    sub_n = 256 if use_b2_fast else bn
    f256_sfx = "_f256" if use_b2_fast else ""
    inv_he = 1.0 / float(he_dim)
    out_cols = (he_dim + LANE) if pack_bf16_sc else he_dim

    def _b2_nstream_kernel(*refs):
        if pack_bf16_sc:
            x_in_ref, w_fp8_ref, vecs_ref, out_i8_ref, u_scratch_ref, sq_scratch_ref = (
                refs
            )
            out_sc_ref = None
        else:
            (
                x_in_ref,
                w_fp8_ref,
                vecs_ref,
                out_i8_ref,
                out_sc_ref,
                u_scratch_ref,
                sq_scratch_ref,
            ) = refs
        n_idx = pl.program_id(1)

        @pl.when(n_idx == 0)
        def _init_sq():
            sq_scratch_ref[:, :] = jnp.zeros((m_tile, LANE), dtype=jnp.float32)

        for rb in range(m_tile // sub_m):
            r0, r1 = rb * sub_m, (rb + 1) * sub_m
            sq_acc = sq_scratch_ref[r0:r1, :]
            for ns in range(bn // sub_n):
                cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                acc = jnp.dot(
                    x_in_ref[r0:r1, :],
                    w_fp8_ref[:, cn0:cn1],
                    preferred_element_type=jnp.float32,
                )
                w_sc = vecs_ref[n_idx, 0:1, cn0:cn1]
                # Exact Muse Spark rounding: (dot * wpool_scale).to(bf16).to(f32)
                u_bf16 = (acc * w_sc).astype(jnp.bfloat16)
                u_scratch_ref[n_idx, r0:r1, cn0:cn1] = u_bf16
                u_f32 = u_bf16.astype(jnp.float32)
                sq_acc = sq_acc + vpu_tree_sum_128(u_f32 * u_f32)
            sq_scratch_ref[r0:r1, :] = sq_acc

        @pl.when(n_idx == num_n - 1)
        def _epilogue_norm_and_quant():
            for rb in range(m_tile // sub_m):
                r0, r1 = rb * sub_m, (rb + 1) * sub_m
                sq_row = jnp.sum(sq_scratch_ref[r0:r1, :], axis=1, keepdims=True)
                inv_rms = jax.lax.rsqrt(
                    sq_row * jnp.float32(inv_he) + jnp.float32(rms_eps)
                )

                # Pass A: compute row_amax of BF16-rounded pre_expert_norm output
                amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                for s in range(num_n):
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        u_f32 = u_scratch_ref[s, r0:r1, cn0:cn1].astype(jnp.float32)
                        gain_s = vecs_ref[s, 1:2, cn0:cn1]
                        x_norm_bf16 = (u_f32 * inv_rms * gain_s).astype(jnp.bfloat16)
                        u_scratch_ref[s, r0:r1, cn0:cn1] = x_norm_bf16
                        amax_128 = jnp.maximum(
                            amax_128,
                            vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                        )

                row_amax = jnp.max(amax_128, axis=1, keepdims=True)
                row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
                row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
                inv_sc = jnp.where(
                    row_amax == 0,
                    jnp.zeros_like(row_amax),
                    jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
                ).astype(jnp.bfloat16)

                if pack_bf16_sc:
                    sc_u16 = jax.lax.bitcast_convert_type(row_sc, jnp.uint16).astype(
                        jnp.int32
                    )
                    lo_i8 = (sc_u16 & jnp.int32(0xFF)).astype(jnp.int8)
                    hi_i8 = ((sc_u16 >> 8) & jnp.int32(0xFF)).astype(jnp.int8)
                    lane = jax.lax.broadcasted_iota(jnp.int32, (sub_m, LANE), 1)
                    out_i8_ref[r0:r1, he_dim : he_dim + LANE] = jnp.where(
                        lane < 64,
                        jnp.broadcast_to(lo_i8, (sub_m, LANE)),
                        jnp.broadcast_to(hi_i8, (sub_m, LANE)),
                    )
                else:
                    out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))
                for s in range(num_n):
                    s_col0 = s * bn
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        out_i8_ref[r0:r1, s_col0 + cn0 : s_col0 + cn1] = jnp.rint(
                            u_scratch_ref[s, r0:r1, cn0:cn1] * inv_sc
                        ).astype(jnp.int8)

    if envs.SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE.get():
        def _b2_1d_kernel(*refs):
            if pack_bf16_sc:
                (
                    x_in_ref,
                    w_hbm_ref,
                    vecs_ref,
                    out_i8_ref,
                    u_scratch_ref,
                    sq_scratch_ref,
                    w_buf_ref,
                    w_sem_ref,
                ) = refs
                out_sc_ref = None
            else:
                (
                    x_in_ref,
                    w_hbm_ref,
                    vecs_ref,
                    out_i8_ref,
                    out_sc_ref,
                    u_scratch_ref,
                    sq_scratch_ref,
                    w_buf_ref,
                    w_sem_ref,
                ) = refs

            for _pf in range(min(2, num_n)):
                pltpu.make_async_copy(
                    src_ref=w_hbm_ref.at[:, pl.ds(_pf * bn, bn)],
                    dst_ref=w_buf_ref.at[_pf],
                    sem=w_sem_ref.at[_pf],
                ).start()

            sq_scratch_ref[:, :] = jnp.zeros((m_tile, LANE), dtype=jnp.float32)

            for n_idx in range(num_n):
                slot = n_idx % 2
                pltpu.make_async_copy(
                    src_ref=w_hbm_ref.at[:, pl.ds(n_idx * bn, bn)],
                    dst_ref=w_buf_ref.at[slot],
                    sem=w_sem_ref.at[slot],
                ).wait()
                for rb in range(m_tile // sub_m):
                    r0, r1 = rb * sub_m, (rb + 1) * sub_m
                    sq_acc = sq_scratch_ref[r0:r1, :]
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        acc = jnp.dot(
                            x_in_ref[r0:r1, :],
                            w_buf_ref[slot, :, cn0:cn1],
                            preferred_element_type=jnp.float32,
                        )
                        w_sc = vecs_ref[n_idx, 0:1, cn0:cn1]
                        u_bf16 = (acc * w_sc).astype(jnp.bfloat16)
                        u_scratch_ref[n_idx, r0:r1, cn0:cn1] = u_bf16
                        u_f32 = u_bf16.astype(jnp.float32)
                        sq_acc = sq_acc + vpu_tree_sum_128(u_f32 * u_f32)
                    sq_scratch_ref[r0:r1, :] = sq_acc
                if n_idx + 2 < num_n:
                    n_nxt = n_idx + 2
                    pltpu.make_async_copy(
                        src_ref=w_hbm_ref.at[:, pl.ds(n_nxt * bn, bn)],
                        dst_ref=w_buf_ref.at[slot],
                        sem=w_sem_ref.at[slot],
                    ).start()

            for rb in range(m_tile // sub_m):
                r0, r1 = rb * sub_m, (rb + 1) * sub_m
                sq_row = jnp.sum(sq_scratch_ref[r0:r1, :], axis=1, keepdims=True)
                inv_rms = jax.lax.rsqrt(
                    sq_row * jnp.float32(inv_he) + jnp.float32(rms_eps)
                )
                amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                for s in range(num_n):
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        u_f32 = u_scratch_ref[s, r0:r1, cn0:cn1].astype(jnp.float32)
                        gain_s = vecs_ref[s, 1:2, cn0:cn1]
                        x_norm_bf16 = (u_f32 * inv_rms * gain_s).astype(jnp.bfloat16)
                        u_scratch_ref[s, r0:r1, cn0:cn1] = x_norm_bf16
                        amax_128 = jnp.maximum(
                            amax_128,
                            vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                        )
                row_amax = jnp.max(amax_128, axis=1, keepdims=True)
                row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
                row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
                inv_sc = jnp.where(
                    row_amax == 0,
                    jnp.zeros_like(row_amax),
                    jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
                ).astype(jnp.bfloat16)
                if pack_bf16_sc:
                    sc_u16 = jax.lax.bitcast_convert_type(row_sc, jnp.uint16).astype(
                        jnp.int32
                    )
                    lo_i8 = (sc_u16 & jnp.int32(0xFF)).astype(jnp.int8)
                    hi_i8 = ((sc_u16 >> 8) & jnp.int32(0xFF)).astype(jnp.int8)
                    lane = jax.lax.broadcasted_iota(jnp.int32, (sub_m, LANE), 1)
                    out_i8_ref[r0:r1, he_dim : he_dim + LANE] = jnp.where(
                        lane < 64,
                        jnp.broadcast_to(lo_i8, (sub_m, LANE)),
                        jnp.broadcast_to(hi_i8, (sub_m, LANE)),
                    )
                else:
                    out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))
                for s in range(num_n):
                    s_col0 = s * bn
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        out_i8_ref[r0:r1, s_col0 + cn0 : s_col0 + cn1] = jnp.rint(
                            u_scratch_ref[s, r0:r1, cn0:cn1] * inv_sc
                        ).astype(jnp.int8)

        common_in_specs = [
            pl.BlockSpec((m_tile, h_dim), lambda m, m_off=m_offset: (m + m_off, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((num_n, 8, bn), lambda m: (0, 0, 0)),
        ]
        common_scratch = [
            pltpu.VMEM((num_n, m_tile, bn), jnp.bfloat16),
            pltpu.VMEM((m_tile, LANE), jnp.float32),
            pltpu.VMEM((2, h_dim, bn), jnp.float8_e4m3fn),
            pltpu.SemaphoreType.DMA((2,)),
        ]
        if pack_bf16_sc:
            x_moe_i8 = pl.pallas_call(
                _b2_1d_kernel,
                out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(num_m,),
                    in_specs=common_in_specs,
                    out_specs=pl.BlockSpec((m_tile, out_cols), lambda m: (m, 0)),
                    scratch_shapes=common_scratch,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    disable_bounds_checks=True,
                    vmem_limit_bytes=60 * 1024 * 1024,
                ),
                name=f"pallas_b2_wpool_nstream_1w_psc_m{m_tile}_bn{bn}{f256_sfx}",
            )(router_in_bf16, wpool_fp8, vecs_3d)
            return x_moe_i8, None

        x_moe_i8, x_moe_sc_pad = pl.pallas_call(
            _b2_1d_kernel,
            out_shape=(
                jax.ShapeDtypeStruct((out_rows, he_dim), jnp.int8),
                jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_m,),
                in_specs=common_in_specs,
                out_specs=[
                    pl.BlockSpec((m_tile, he_dim), lambda m: (m, 0)),
                    pl.BlockSpec((m_tile, LANE), lambda m: (m, 0)),
                ],
                scratch_shapes=common_scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                disable_bounds_checks=True,
                vmem_limit_bytes=60 * 1024 * 1024,
            ),
            name=f"pallas_b2_wpool_nstream_1w_m{m_tile}_bn{bn}{f256_sfx}",
        )(router_in_bf16, wpool_fp8, vecs_3d)
        return x_moe_i8, x_moe_sc_pad[:, 0]

    if pack_bf16_sc:
        x_moe_i8 = pl.pallas_call(
            _b2_nstream_kernel,
            out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_m, num_n),
                in_specs=[
                    pl.BlockSpec(
                        (m_tile, h_dim), lambda m, n, m_off=m_offset: (m + m_off, 0)
                    ),
                    pl.BlockSpec((h_dim, bn), lambda m, n: (0, n)),
                    pl.BlockSpec((num_n, 8, bn), lambda m, n: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((m_tile, out_cols), lambda m, n: (m, 0)),
                scratch_shapes=[
                    pltpu.VMEM((num_n, m_tile, bn), jnp.bfloat16),
                    pltpu.VMEM((m_tile, LANE), jnp.float32),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                disable_bounds_checks=True,
                vmem_limit_bytes=60 * 1024 * 1024,
            ),
            name=f"pallas_b2_wpool_nstream_psc_m{m_tile}_bn{bn}{f256_sfx}",
        )(router_in_bf16, wpool_fp8, vecs_3d)
        return x_moe_i8, None

    x_moe_i8, x_moe_sc_pad = pl.pallas_call(
        _b2_nstream_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((out_rows, he_dim), jnp.int8),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_m, num_n),
            in_specs=[
                pl.BlockSpec(
                    (m_tile, h_dim), lambda m, n, m_off=m_offset: (m + m_off, 0)
                ),
                pl.BlockSpec((h_dim, bn), lambda m, n: (0, n)),
                pl.BlockSpec((num_n, 8, bn), lambda m, n: (0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((m_tile, he_dim), lambda m, n: (m, 0)),
                pl.BlockSpec((m_tile, LANE), lambda m, n: (m, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((num_n, m_tile, bn), jnp.bfloat16),
                pltpu.VMEM((m_tile, LANE), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"pallas_b2_wpool_nstream_m{m_tile}_bn{bn}{f256_sfx}",
    )(router_in_bf16, wpool_fp8, vecs_3d)

    return x_moe_i8, x_moe_sc_pad[:, 0]


def pallas_wpool_prenorm_int8_quant_rowtile(
    router_in_bf16: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    *,
    bm: int = 512,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Layout B2-RowTile: Weight-resident wpool_fp8 [8192, 4096] (32 MiB), row-tiled bm."""
    t_rows, h_dim = router_in_bf16.shape
    w_k, he_dim = wpool_fp8.shape
    assert h_dim == w_k == 8192 and he_dim == 4096
    bm_eff = min(bm, t_rows)
    assert t_rows % bm_eff == 0 and bm_eff % LANE == 0

    vecs_8x4096 = prepare_b2_static_inputs(wpool_scale, pre_expert_norm_w)
    num_steps = t_rows // bm_eff
    sub_m = LANE
    inv_he = 1.0 / float(he_dim)

    def _b2_rowtile_kernel(
        x_in_ref,
        w_fp8_ref,
        vecs_ref,
        out_i8_ref,
        out_sc_ref,
        u_scratch_ref,
    ):
        for rb in range(bm_eff // sub_m):
            r0, r1 = rb * sub_m, (rb + 1) * sub_m
            sq_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                acc = jnp.dot(
                    x_in_ref[r0:r1, :],
                    w_fp8_ref[:, c0:c1],
                    preferred_element_type=jnp.float32,
                )
                u_bf16 = (acc * vecs_ref[0:1, c0:c1]).astype(jnp.bfloat16)
                u_scratch_ref[r0:r1, c0:c1] = u_bf16
                u_f32 = u_bf16.astype(jnp.float32)
                sq_128 = sq_128 + vpu_tree_sum_128(u_f32 * u_f32)

            inv_rms = jax.lax.rsqrt(
                jnp.sum(sq_128, axis=1, keepdims=True) * jnp.float32(inv_he)
                + jnp.float32(rms_eps)
            )
            amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                u_f32 = u_scratch_ref[r0:r1, c0:c1].astype(jnp.float32)
                x_norm_bf16 = (u_f32 * inv_rms * vecs_ref[1:2, c0:c1]).astype(
                    jnp.bfloat16
                )
                u_scratch_ref[r0:r1, c0:c1] = x_norm_bf16
                amax_128 = jnp.maximum(
                    amax_128,
                    vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                )

            row_amax = jnp.max(amax_128, axis=1, keepdims=True)
            row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
            row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
            inv_sc = jnp.where(
                row_amax == 0,
                jnp.zeros_like(row_amax),
                jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
            ).astype(jnp.bfloat16)
            out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                out_i8_ref[r0:r1, c0:c1] = jnp.rint(
                    u_scratch_ref[r0:r1, c0:c1] * inv_sc
                ).astype(jnp.int8)

    x_moe_i8, x_moe_sc_pad = pl.pallas_call(
        _b2_rowtile_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t_rows, he_dim), jnp.int8),
            jax.ShapeDtypeStruct((t_rows, LANE), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec((bm_eff, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((h_dim, he_dim), lambda i: (0, 0)),
                pl.BlockSpec((8, he_dim), lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((bm_eff, he_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm_eff, LANE), lambda i: (i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((bm_eff, he_dim), jnp.bfloat16),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"pallas_b2_wpool_rowtile_bm{bm_eff}",
    )(router_in_bf16, wpool_fp8, vecs_8x4096)

    return x_moe_i8, x_moe_sc_pad[:, 0]


def pallas_wpool_prenorm_int8_quant_fp8_nstream(
    router_in_fp8: jax.Array,
    router_in_sc_pad: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    *,
    m_pass: int = 1024,
    bn: int = 1024,
    rms_eps: float = 1e-5,
    pack_sc_in_i8: bool = False,
    half_idx: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Wave 2/4 Native FP8xFP8 (vmatmul.e4m3) Layout B2-NStream.

    When `pack_sc_in_i8=True`, packs the per-row `bf16` scale losslessly as Hi+Lo E4M3 bytes
    (bitcast to `int8`) into columns `[4096:4224]` of `x_moe_i8` (`int8[t_rows, 4224]`),
    eliminating separate scale `psum`/`all_gather` collectives.
    When `half_idx` is 0 or 1, reads the corresponding half of `router_in_fp8` and
    `router_in_sc_pad` directly via BlockSpec index offset without an HBM slice/copy.
    """
    t_rows, h_dim = router_in_fp8.shape
    w_k, he_dim = wpool_fp8.shape
    assert h_dim == w_k == 8192 and he_dim == 4096
    out_rows = (t_rows // 2) if half_idx is not None else t_rows
    m_tile = min(m_pass, out_rows)
    assert out_rows % m_tile == 0 and m_tile % LANE == 0
    assert he_dim % bn == 0 and bn % 256 == 0

    num_m = out_rows // m_tile
    m_offset = (half_idx * num_m) if half_idx is not None else 0
    num_n = he_dim // bn
    if wpool_scale.ndim == 3 and wpool_scale.shape == (num_n, 8, bn):
        vecs_3d = wpool_scale
    else:
        vecs_8x4096 = prepare_b2_static_inputs(wpool_scale, pre_expert_norm_w)
        vecs_3d = vecs_8x4096.reshape(8, num_n, bn).swapaxes(0, 1)
    b2_fast_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
    use_b2_fast = bool(b2_fast_env is not None and b2_fast_env.get())
    sub_m = LANE
    sub_n = 256 if use_b2_fast else bn
    f256_sfx = "_f256" if use_b2_fast else ""
    psc_sfx = "_psc" if pack_sc_in_i8 else ""
    inv_he = 1.0 / float(he_dim)
    out_cols = (he_dim + LANE) if pack_sc_in_i8 else he_dim

    def _b2_fp8_nstream_kernel(*refs):
        if pack_sc_in_i8:
            (
                x_fp8_ref,
                x_sc_ref,
                w_fp8_ref,
                vecs_ref,
                out_i8_ref,
                u_scratch_ref,
                sq_scratch_ref,
            ) = refs
            out_sc_ref = None
        else:
            (
                x_fp8_ref,
                x_sc_ref,
                w_fp8_ref,
                vecs_ref,
                out_i8_ref,
                out_sc_ref,
                u_scratch_ref,
                sq_scratch_ref,
            ) = refs
        n_idx = pl.program_id(1)

        @pl.when(n_idx == 0)
        def _init_sq():
            sq_scratch_ref[:, :] = jnp.zeros((m_tile, LANE), dtype=jnp.float32)

        for rb in range(m_tile // sub_m):
            r0, r1 = rb * sub_m, (rb + 1) * sub_m
            row_sc_x = x_sc_ref[r0:r1, 0:1]
            sq_acc = sq_scratch_ref[r0:r1, :]
            for ns in range(bn // sub_n):
                cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                acc = jnp.dot(
                    x_fp8_ref[r0:r1, :],
                    w_fp8_ref[:, cn0:cn1],
                    preferred_element_type=jnp.float32,
                )
                w_sc = vecs_ref[n_idx, 0:1, cn0:cn1]
                u_bf16 = (acc * (row_sc_x * w_sc)).astype(jnp.bfloat16)
                u_scratch_ref[n_idx, r0:r1, cn0:cn1] = u_bf16
                u_f32 = u_bf16.astype(jnp.float32)
                sq_acc = sq_acc + vpu_tree_sum_128(u_f32 * u_f32)
            sq_scratch_ref[r0:r1, :] = sq_acc

        @pl.when(n_idx == num_n - 1)
        def _epilogue_norm_and_quant():
            for rb in range(m_tile // sub_m):
                r0, r1 = rb * sub_m, (rb + 1) * sub_m
                sq_row = jnp.sum(sq_scratch_ref[r0:r1, :], axis=1, keepdims=True)
                inv_rms = jax.lax.rsqrt(
                    sq_row * jnp.float32(inv_he) + jnp.float32(rms_eps)
                )

                amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                for s in range(num_n):
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        u_f32 = u_scratch_ref[s, r0:r1, cn0:cn1].astype(jnp.float32)
                        gain_s = vecs_ref[s, 1:2, cn0:cn1]
                        x_norm_bf16 = (u_f32 * inv_rms * gain_s).astype(jnp.bfloat16)
                        u_scratch_ref[s, r0:r1, cn0:cn1] = x_norm_bf16
                        amax_128 = jnp.maximum(
                            amax_128,
                            vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                        )

                row_amax = jnp.max(amax_128, axis=1, keepdims=True)
                row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
                row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
                inv_sc = jnp.where(
                    row_amax == 0,
                    jnp.zeros_like(row_amax),
                    jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
                ).astype(jnp.bfloat16)

                if pack_sc_in_i8:
                    sc_u16 = jax.lax.bitcast_convert_type(row_sc, jnp.uint16).astype(
                        jnp.int32
                    )
                    lo_i8 = (sc_u16 & jnp.int32(0xFF)).astype(jnp.int8)
                    hi_i8 = jax.lax.shift_right_logical(sc_u16, 8).astype(jnp.int8)
                    lane = jax.lax.broadcasted_iota(jnp.int32, (sub_m, LANE), 1)
                    out_i8_ref[r0:r1, he_dim : he_dim + LANE] = jnp.where(
                        lane < 64,
                        jnp.broadcast_to(lo_i8, (sub_m, LANE)),
                        jnp.broadcast_to(hi_i8, (sub_m, LANE)),
                    )
                else:
                    out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))

                for s in range(num_n):
                    s_col0 = s * bn
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        out_i8_ref[r0:r1, s_col0 + cn0 : s_col0 + cn1] = jnp.rint(
                            u_scratch_ref[s, r0:r1, cn0:cn1] * inv_sc
                        ).astype(jnp.int8)

    if envs.SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE.get():
        def _b2_fp8_1d_kernel(*refs):
            if pack_sc_in_i8:
                (
                    x_fp8_ref,
                    x_sc_ref,
                    w_hbm_ref,
                    vecs_ref,
                    out_i8_ref,
                    u_scratch_ref,
                    sq_scratch_ref,
                    w_buf_ref,
                    w_sem_ref,
                ) = refs
                out_sc_ref = None
            else:
                (
                    x_fp8_ref,
                    x_sc_ref,
                    w_hbm_ref,
                    vecs_ref,
                    out_i8_ref,
                    out_sc_ref,
                    u_scratch_ref,
                    sq_scratch_ref,
                    w_buf_ref,
                    w_sem_ref,
                ) = refs

            for _pf in range(min(2, num_n)):
                pltpu.make_async_copy(
                    src_ref=w_hbm_ref.at[:, pl.ds(_pf * bn, bn)],
                    dst_ref=w_buf_ref.at[_pf],
                    sem=w_sem_ref.at[_pf],
                ).start()

            sq_scratch_ref[:, :] = jnp.zeros((m_tile, LANE), dtype=jnp.float32)

            for n_idx in range(num_n):
                slot = n_idx % 2
                pltpu.make_async_copy(
                    src_ref=w_hbm_ref.at[:, pl.ds(n_idx * bn, bn)],
                    dst_ref=w_buf_ref.at[slot],
                    sem=w_sem_ref.at[slot],
                ).wait()
                for rb in range(m_tile // sub_m):
                    r0, r1 = rb * sub_m, (rb + 1) * sub_m
                    row_sc_x = x_sc_ref[r0:r1, 0:1]
                    sq_acc = sq_scratch_ref[r0:r1, :]
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        acc = jnp.dot(
                            x_fp8_ref[r0:r1, :],
                            w_buf_ref[slot, :, cn0:cn1],
                            preferred_element_type=jnp.float32,
                        )
                        w_sc = vecs_ref[n_idx, 0:1, cn0:cn1]
                        u_bf16 = (acc * (row_sc_x * w_sc)).astype(jnp.bfloat16)
                        u_scratch_ref[n_idx, r0:r1, cn0:cn1] = u_bf16
                        u_f32 = u_bf16.astype(jnp.float32)
                        sq_acc = sq_acc + vpu_tree_sum_128(u_f32 * u_f32)
                    sq_scratch_ref[r0:r1, :] = sq_acc
                if n_idx + 2 < num_n:
                    n_nxt = n_idx + 2
                    pltpu.make_async_copy(
                        src_ref=w_hbm_ref.at[:, pl.ds(n_nxt * bn, bn)],
                        dst_ref=w_buf_ref.at[slot],
                        sem=w_sem_ref.at[slot],
                    ).start()

            for rb in range(m_tile // sub_m):
                r0, r1 = rb * sub_m, (rb + 1) * sub_m
                sq_row = jnp.sum(sq_scratch_ref[r0:r1, :], axis=1, keepdims=True)
                inv_rms = jax.lax.rsqrt(
                    sq_row * jnp.float32(inv_he) + jnp.float32(rms_eps)
                )
                amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                for s in range(num_n):
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        u_f32 = u_scratch_ref[s, r0:r1, cn0:cn1].astype(jnp.float32)
                        gain_s = vecs_ref[s, 1:2, cn0:cn1]
                        x_norm_bf16 = (u_f32 * inv_rms * gain_s).astype(jnp.bfloat16)
                        u_scratch_ref[s, r0:r1, cn0:cn1] = x_norm_bf16
                        amax_128 = jnp.maximum(
                            amax_128,
                            vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                        )
                row_amax = jnp.max(amax_128, axis=1, keepdims=True)
                row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
                row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
                inv_sc = jnp.where(
                    row_amax == 0,
                    jnp.zeros_like(row_amax),
                    jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
                ).astype(jnp.bfloat16)
                if pack_sc_in_i8:
                    sc_u16 = jax.lax.bitcast_convert_type(row_sc, jnp.uint16).astype(
                        jnp.int32
                    )
                    lo_i8 = (sc_u16 & jnp.int32(0xFF)).astype(jnp.int8)
                    hi_i8 = jax.lax.shift_right_logical(sc_u16, 8).astype(jnp.int8)
                    lane = jax.lax.broadcasted_iota(jnp.int32, (sub_m, LANE), 1)
                    out_i8_ref[r0:r1, he_dim : he_dim + LANE] = jnp.where(
                        lane < 64,
                        jnp.broadcast_to(lo_i8, (sub_m, LANE)),
                        jnp.broadcast_to(hi_i8, (sub_m, LANE)),
                    )
                else:
                    out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))
                for s in range(num_n):
                    s_col0 = s * bn
                    for ns in range(bn // sub_n):
                        cn0, cn1 = ns * sub_n, (ns + 1) * sub_n
                        out_i8_ref[r0:r1, s_col0 + cn0 : s_col0 + cn1] = jnp.rint(
                            u_scratch_ref[s, r0:r1, cn0:cn1] * inv_sc
                        ).astype(jnp.int8)

        fp8_1d_in_specs = [
            pl.BlockSpec(
                (m_tile, h_dim), lambda m, m_off=m_offset: (m + m_off, 0)
            ),
            pl.BlockSpec(
                (m_tile, LANE), lambda m, m_off=m_offset: (m + m_off, 0)
            ),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((num_n, 8, bn), lambda m: (0, 0, 0)),
        ]
        fp8_1d_scratch = [
            pltpu.VMEM((num_n, m_tile, bn), jnp.bfloat16),
            pltpu.VMEM((m_tile, LANE), jnp.float32),
            pltpu.VMEM((2, h_dim, bn), jnp.float8_e4m3fn),
            pltpu.SemaphoreType.DMA((2,)),
        ]
        if pack_sc_in_i8:
            x_moe_i8 = pl.pallas_call(
                _b2_fp8_1d_kernel,
                out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(num_m,),
                    in_specs=fp8_1d_in_specs,
                    out_specs=pl.BlockSpec((m_tile, out_cols), lambda m: (m, 0)),
                    scratch_shapes=fp8_1d_scratch,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("parallel",),
                    disable_bounds_checks=True,
                    vmem_limit_bytes=60 * 1024 * 1024,
                ),
                name=f"pallas_b2_fp8_wpool_nstream_1w{psc_sfx}_m{m_tile}_bn{bn}{f256_sfx}",
            )(router_in_fp8, router_in_sc_pad, wpool_fp8, vecs_3d)
            return x_moe_i8, None

        x_moe_i8, x_moe_sc_pad = pl.pallas_call(
            _b2_fp8_1d_kernel,
            out_shape=(
                jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
                jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_m,),
                in_specs=fp8_1d_in_specs,
                out_specs=[
                    pl.BlockSpec((m_tile, out_cols), lambda m: (m, 0)),
                    pl.BlockSpec((m_tile, LANE), lambda m: (m, 0)),
                ],
                scratch_shapes=fp8_1d_scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                disable_bounds_checks=True,
                vmem_limit_bytes=60 * 1024 * 1024,
            ),
            name=f"pallas_b2_fp8_wpool_nstream_1w_m{m_tile}_bn{bn}{f256_sfx}",
        )(router_in_fp8, router_in_sc_pad, wpool_fp8, vecs_3d)
        return x_moe_i8, x_moe_sc_pad[:, 0]

    fp8_2d_in_specs = [
        pl.BlockSpec(
            (m_tile, h_dim), lambda m, n, m_off=m_offset: (m + m_off, 0)
        ),
        pl.BlockSpec(
            (m_tile, LANE), lambda m, n, m_off=m_offset: (m + m_off, 0)
        ),
        pl.BlockSpec((h_dim, bn), lambda m, n: (0, n)),
        pl.BlockSpec((num_n, 8, bn), lambda m, n: (0, 0, 0)),
    ]
    fp8_2d_scratch = [
        pltpu.VMEM((num_n, m_tile, bn), jnp.bfloat16),
        pltpu.VMEM((m_tile, LANE), jnp.float32),
    ]
    if pack_sc_in_i8:
        x_moe_i8 = pl.pallas_call(
            _b2_fp8_nstream_kernel,
            out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_m, num_n),
                in_specs=fp8_2d_in_specs,
                out_specs=pl.BlockSpec((m_tile, out_cols), lambda m, n: (m, 0)),
                scratch_shapes=fp8_2d_scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                disable_bounds_checks=True,
                vmem_limit_bytes=60 * 1024 * 1024,
            ),
            name=f"pallas_b2_fp8_wpool_nstream{psc_sfx}_m{m_tile}_bn{bn}{f256_sfx}",
        )(router_in_fp8, router_in_sc_pad, wpool_fp8, vecs_3d)
        return x_moe_i8, None

    x_moe_i8, x_moe_sc_pad = pl.pallas_call(
        _b2_fp8_nstream_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_m, num_n),
            in_specs=fp8_2d_in_specs,
            out_specs=[
                pl.BlockSpec((m_tile, out_cols), lambda m, n: (m, 0)),
                pl.BlockSpec((m_tile, LANE), lambda m, n: (m, 0)),
            ],
            scratch_shapes=fp8_2d_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"pallas_b2_fp8_wpool_nstream_m{m_tile}_bn{bn}{f256_sfx}",
    )(router_in_fp8, router_in_sc_pad, wpool_fp8, vecs_3d)

    return x_moe_i8, x_moe_sc_pad[:, 0]


def _select_dequant_row_tile(rows: int, max_tile: int = 512) -> int:
    """Pick the largest 128-aligned row tile <= max_tile dividing rows to bound VMEM <= 16 MiB."""
    for cand in (512, 256, 128):
        if cand <= max_tile and cand <= rows and rows % cand == 0:
            return cand
    return rows


def dequant_half0_new_buffer(
    ag_i8_3d: jax.Array,
    ag_sc_2d: jax.Array,
) -> jax.Array:
    """Half 0 of `fused_moe._dequant_into_half` into a new `bf16[ep, 2, half_tok, hidden]`.

    Same math as `dequant_into_half_fast_0`, but it allocates the output instead of aliasing
    an `empty` input, which XLA materializes and copies. Half 1 rows stay unwritten.
    Row-tiled by `row_tile <= 512` so VMEM stays <= 16 MiB at T=16k..64k.
    """
    ep_sz, half_tok, hidden_sz = ag_i8_3d.shape
    row_tile = _select_dequant_row_tile(half_tok, 512)
    num_rt = half_tok // row_tile

    def _k0(i8_ref, sc_ref, dst_out_ref):
        r = pl.program_id(0)
        mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, 128), 0) == r
        for t_blk in range(row_tile // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            sc_1x128 = jnp.sum(
                jnp.where(mask_r, sc_ref[:, r0:r1].astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)
            sc_128x1 = jnp.swapaxes(jnp.broadcast_to(sc_1x128, (128, 128)), 0, 1)[:, :1]
            for c_blk in range(hidden_sz // 1024):
                c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                dst_out_ref[0, 0, r0:r1, c0:c1] = (
                    i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                )

    return pl.pallas_call(
        _k0,
        out_shape=jax.ShapeDtypeStruct((ep_sz, 2, half_tok, hidden_sz), jnp.bfloat16),
        grid=(ep_sz, num_rt),
        in_specs=[
            pl.BlockSpec((1, row_tile, hidden_sz), lambda r, rt: (r, rt, 0)),
            pl.BlockSpec((ep_sz, row_tile), lambda r, rt: (0, rt)),
        ],
        out_specs=pl.BlockSpec((1, 1, row_tile, hidden_sz), lambda r, rt: (r, 0, rt, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name="dequant_half0_new_buffer",
    )(ag_i8_3d, ag_sc_2d)


def dequant_half_gmm_layout(
    ag_i8_3d: jax.Array,
    ag_sc_2d: jax.Array | None = None,
    buf_5d: jax.Array | None = None,
    *,
    zero_spill_256: bool = False,
) -> jax.Array:
    """`dequant_half0_new_buffer` math into GMM's input layout `bf16[ep, 2, half_tok, hidden // 128, 128]`.

    With `buf_5d=None` it allocates the buffer and writes half 0; otherwise it writes half 1 into `buf_5d`
    in place. Merging the major dims then gives `[tok, hidden // 128, 128]` with no XLA relayout copy.
    When `ag_sc_2d=None`, unpacks exact 16-bit bf16 scales from columns `[hidden:hidden+128]` of `ag_i8_3d`.
    Row-tiled by `row_tile <= 512` so VMEM stays <= 16 MiB at T=28k/32k/48k/64k.
    """
    ep_sz, half_tok, raw_cols = ag_i8_3d.shape
    packed_sc = ag_sc_2d is None
    hidden_sz = (raw_cols - LANE) if packed_sc else raw_cols
    half_idx = 0 if buf_5d is None else 1
    col_step = 256 if zero_spill_256 else 1024
    blocks_per_step = col_step // 128
    row_tile = _select_dequant_row_tile(half_tok, 512)
    num_rt = half_tok // row_tile

    def _k(*refs):
        if packed_sc:
            i8_ref, dst_out_ref = refs[-2:]
            sc_ref = None
        else:
            i8_ref, sc_ref, dst_out_ref = refs[-3:]
            r = pl.program_id(0)
            mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, 128), 0) == r
        for t_blk in range(row_tile // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            if packed_sc:
                ps = i8_ref[0, r0:r1, hidden_sz : hidden_sz + LANE]
                lo_u16 = ps[:, 0:1].astype(jnp.int32) & jnp.int32(0xFF)
                hi_u16 = (ps[:, 64:65].astype(jnp.int32) & jnp.int32(0xFF)) << 8
                sc_128x1 = jax.lax.bitcast_convert_type(
                    (lo_u16 | hi_u16).astype(jnp.uint16), jnp.bfloat16
                )
            else:
                sc_1x128 = jnp.sum(
                    jnp.where(mask_r, sc_ref[:, r0:r1].astype(jnp.float32), 0.0),
                    axis=0,
                    keepdims=True,
                ).astype(jnp.bfloat16)
                sc_128x1 = jnp.swapaxes(jnp.broadcast_to(sc_1x128, (128, 128)), 0, 1)[
                    :, :1
                ]
            for c_blk in range(hidden_sz // col_step):
                c0, c1 = c_blk * col_step, (c_blk + 1) * col_step
                val = i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                dst_out_ref[0, 0, r0:r1, c0 // 128 : c1 // 128, :] = val.reshape(
                    128, blocks_per_step, 128
                )

    in_specs = [pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0))]
    args: tuple[jax.Array, ...] = (ag_i8_3d,)
    if not packed_sc:
        assert ag_sc_2d is not None
        in_specs.append(pl.BlockSpec((ep_sz, row_tile), lambda r, rt: (0, rt)))
        args = args + (ag_sc_2d,)
    if buf_5d is not None:  # aliased, never read: half 0 is already there
        in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] + in_specs
        args = (buf_5d,) + args
    suffix = ("_psc" if packed_sc else "") + ("_zs256" if zero_spill_256 else "")
    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct(
            (ep_sz, 2, half_tok, hidden_sz // 128, 128), jnp.bfloat16
        ),
        grid=(ep_sz, num_rt),
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (1, 1, row_tile, hidden_sz // 128, 128),
            lambda r, rt: (r, half_idx, rt, 0, 0),
        ),
        input_output_aliases={} if buf_5d is None else {0: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name=f"dequant_half{half_idx}_gmm_layout{suffix}",
    )(*args)


def dequant_both_halves_gmm_layout(
    ag_i8_0: jax.Array,
    ag_sc_0: jax.Array | None,
    ag_i8_1: jax.Array,
    ag_sc_1: jax.Array | None,
) -> jax.Array:
    """Single-pass Pallas dequant of both halves directly into `bf16[ep, 2, half_tok, hidden // 128, 128]`.

    Row-tiled over `(ep_sz, 2, half_tok // row_tile)` (`row_tile <= 256`) so VMEM stays <= 12 MiB at all T.
    """
    ep_sz, half_tok, raw_cols = ag_i8_0.shape
    packed_sc = ag_sc_0 is None
    hidden_sz = (raw_cols - LANE) if packed_sc else raw_cols
    row_tile = _select_dequant_row_tile(half_tok, 256)
    num_rt = half_tok // row_tile

    def _k_both(*refs):
        h = pl.program_id(1)
        if packed_sc:
            i8_0_ref, i8_1_ref, dst_out_ref = refs
            sc_0_ref = sc_1_ref = None
            mask_r = None
        else:
            i8_0_ref, sc_0_ref, i8_1_ref, sc_1_ref, dst_out_ref = refs
            r = pl.program_id(0)
            mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, 128), 0) == r

        def _deq_half(i8_ref, sc_ref):
            for t_blk in range(row_tile // 128):
                r0, r1 = t_blk * 128, (t_blk + 1) * 128
                if packed_sc:
                    ps = i8_ref[0, r0:r1, hidden_sz : hidden_sz + LANE]
                    lo_u16 = ps[:, 0:1].astype(jnp.int32) & jnp.int32(0xFF)
                    hi_u16 = (ps[:, 64:65].astype(jnp.int32) & jnp.int32(0xFF)) << 8
                    sc_128x1 = jax.lax.bitcast_convert_type(
                        (lo_u16 | hi_u16).astype(jnp.uint16), jnp.bfloat16
                    )
                else:
                    sc_1x128 = jnp.sum(
                        jnp.where(mask_r, sc_ref[:, r0:r1].astype(jnp.float32), 0.0),
                        axis=0,
                        keepdims=True,
                    ).astype(jnp.bfloat16)
                    sc_128x1 = jnp.swapaxes(
                        jnp.broadcast_to(sc_1x128, (128, 128)), 0, 1
                    )[:, :1]
                for c_blk in range(hidden_sz // 256):
                    c0, c1 = c_blk * 256, (c_blk + 1) * 256
                    val = i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                    dst_out_ref[0, 0, r0:r1, c0 // 128 : c1 // 128, :] = (
                        val.reshape(128, 2, 128)
                    )

        @pl.when(h == 0)
        def _h0():
            _deq_half(i8_0_ref, sc_0_ref)

        @pl.when(h == 1)
        def _h1():
            _deq_half(i8_1_ref, sc_1_ref)

    if packed_sc:
        in_specs = [
            pl.BlockSpec((1, row_tile, raw_cols), lambda r, h, rt: (r, rt, 0)),
            pl.BlockSpec((1, row_tile, raw_cols), lambda r, h, rt: (r, rt, 0)),
        ]
        args: tuple[jax.Array, ...] = (ag_i8_0, ag_i8_1)
    else:
        assert ag_sc_0 is not None and ag_sc_1 is not None
        in_specs = [
            pl.BlockSpec((1, row_tile, raw_cols), lambda r, h, rt: (r, rt, 0)),
            pl.BlockSpec((ep_sz, row_tile), lambda r, h, rt: (0, rt)),
            pl.BlockSpec((1, row_tile, raw_cols), lambda r, h, rt: (r, rt, 0)),
            pl.BlockSpec((ep_sz, row_tile), lambda r, h, rt: (0, rt)),
        ]
        args = (ag_i8_0, ag_sc_0, ag_i8_1, ag_sc_1)

    return pl.pallas_call(
        _k_both,
        out_shape=jax.ShapeDtypeStruct(
            (ep_sz, 2, half_tok, hidden_sz // 128, 128), jnp.bfloat16
        ),
        grid=(ep_sz, 2, num_rt),
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (1, 1, row_tile, hidden_sz // 128, 128), lambda r, h, rt: (r, h, rt, 0, 0)
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name="dequant_both_halves_gmm_layout" + ("_psc" if packed_sc else ""),
    )(*args)


def dequant_single_gmm_layout(
    ag_i8_3d: jax.Array,
) -> jax.Array:
    """Single-pass packed-bf16-scale dequant of `int8[ep, t_local, 4224]` -> `bf16[ep, t_local, 32, 128]`.

    Row-tiled over `(ep_sz, t_local // row_tile)` (`row_tile <= 512`, <= 12.1 MiB VMEM) so
    `FUSED_B1_B2=2` (1-pass `wp_fp8`) compiles cleanly without `e1001` ScopedVmemOom at T=16k..64k!
    """
    ep_sz, t_loc, raw_cols = ag_i8_3d.shape
    hidden_sz = raw_cols - LANE
    row_tile = _select_dequant_row_tile(t_loc, 512)
    num_rt = t_loc // row_tile

    def _k_single(i8_ref, dst_out_ref):
        for t_blk in range(row_tile // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            ps = i8_ref[0, r0:r1, hidden_sz : hidden_sz + LANE]
            lo_u16 = ps[:, 0:1].astype(jnp.int32) & jnp.int32(0xFF)
            hi_u16 = (ps[:, 64:65].astype(jnp.int32) & jnp.int32(0xFF)) << 8
            sc_128x1 = jax.lax.bitcast_convert_type(
                (lo_u16 | hi_u16).astype(jnp.uint16), jnp.bfloat16
            )
            for c_blk in range(hidden_sz // 256):
                c0, c1 = c_blk * 256, (c_blk + 1) * 256
                val = i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                dst_out_ref[0, r0:r1, c0 // 128 : c1 // 128, :] = val.reshape(
                    128, 2, 128
                )

    return pl.pallas_call(
        _k_single,
        out_shape=jax.ShapeDtypeStruct(
            (ep_sz, t_loc, hidden_sz // 128, 128), jnp.bfloat16
        ),
        grid=(ep_sz, num_rt),
        in_specs=[pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0))],
        out_specs=pl.BlockSpec(
            (1, row_tile, hidden_sz // 128, 128), lambda r, rt: (r, rt, 0, 0)
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name="dequant_single_gmm_layout_psc",
    )(ag_i8_3d)


def dequant_into_half_packed_1way(
    buf_4d: jax.Array,
    ag_i8_packed_3d: jax.Array,
    *,
    half_idx: int,
) -> jax.Array:
    """1-way write-only Pallas dequant consuming packed `ag_i8_packed_3d: int8[ep_sz, half_tok, 4224]`.

    Unpacks Hi+Lo E4M3 scale bytes from columns `[4096:4224]` directly in VREGs without any
    separate `ag_sc_2d` input or `mask_r` reduction.
    """
    ep_sz, _, half_tok, hidden_sz = buf_4d.shape
    packed_cols = ag_i8_packed_3d.shape[-1]
    assert packed_cols == hidden_sz + LANE

    def _k_packed(dst_in_ref, i8_ref, dst_out_ref, vmem_out_ref, dma_sem):
        del dst_in_ref
        r = pl.program_id(0)
        slot = r % 2

        @pl.when(r >= 2)
        def _wait_prev():
            r_prev = r - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(r_prev, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(0, half_tok),
                    pl.ds(0, hidden_sz),
                ],
                dma_sem.at[slot],
            ).wait()

        for t_blk in range(half_tok // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            ps = i8_ref[0, r0:r1, hidden_sz : hidden_sz + LANE]
            lo_u16 = ps[:, 0:1].astype(jnp.int32) & jnp.int32(0xFF)
            hi_u16 = (ps[:, 64:65].astype(jnp.int32) & jnp.int32(0xFF)) << 8
            sc_128x1 = jax.lax.bitcast_convert_type(
                (lo_u16 | hi_u16).astype(jnp.uint16), jnp.bfloat16
            )
            for c_blk in range(hidden_sz // 1024):
                c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                vmem_out_ref[slot, 0, 0, r0:r1, c0:c1] = (
                    i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                )

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(r, 1),
                pl.ds(half_idx, 1),
                pl.ds(0, half_tok),
                pl.ds(0, hidden_sz),
            ],
            dma_sem.at[slot],
        ).start()

        @pl.when(r == ep_sz - 1)
        def _wait_last():
            for s in range(2):
                r_s = (ep_sz - 2) + s
                slot_s = r_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(r_s, 1),
                        pl.ds(half_idx, 1),
                        pl.ds(0, half_tok),
                        pl.ds(0, hidden_sz),
                    ],
                    dma_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k_packed,
        out_shape=jax.ShapeDtypeStruct(buf_4d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, half_tok, packed_cols), lambda r: (r, 0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, 1, 1, half_tok, hidden_sz), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=62 * 1024 * 1024,
        ),
        name=f"dequant_into_half_packed_1way_{half_idx}",
    )(buf_4d, ag_i8_packed_3d)


def pallas_wpool_prenorm_int8_quant_fp8_rowtile(
    router_in_fp8: jax.Array,
    router_in_sc_pad: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    *,
    bm: int = 512,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Wave 2 Native FP8xFP8 (vmatmul.e4m3) Layout B2-RowTile (1D grid, weight-resident 32 MiB)."""
    t_rows, h_dim = router_in_fp8.shape
    w_k, he_dim = wpool_fp8.shape
    assert h_dim == w_k == 8192 and he_dim == 4096
    bm_eff = min(bm, t_rows)
    assert t_rows % bm_eff == 0 and bm_eff % LANE == 0

    vecs_8x4096 = prepare_b2_static_inputs(wpool_scale, pre_expert_norm_w)
    num_steps = t_rows // bm_eff
    sub_m = LANE
    inv_he = 1.0 / float(he_dim)

    def _b2_fp8_rowtile_kernel(
        x_fp8_ref,
        x_sc_ref,
        w_fp8_ref,
        vecs_ref,
        out_i8_ref,
        out_sc_ref,
        u_scratch_ref,
    ):
        for rb in range(bm_eff // sub_m):
            r0, r1 = rb * sub_m, (rb + 1) * sub_m
            row_sc_x = x_sc_ref[r0:r1, 0:1]
            sq_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                acc = jnp.dot(
                    x_fp8_ref[r0:r1, :],
                    w_fp8_ref[:, c0:c1],
                    preferred_element_type=jnp.float32,
                )
                u_bf16 = (acc * (row_sc_x * vecs_ref[0:1, c0:c1])).astype(jnp.bfloat16)
                u_scratch_ref[r0:r1, c0:c1] = u_bf16
                u_f32 = u_bf16.astype(jnp.float32)
                sq_128 = sq_128 + vpu_tree_sum_128(u_f32 * u_f32)

            inv_rms = jax.lax.rsqrt(
                jnp.sum(sq_128, axis=1, keepdims=True) * jnp.float32(inv_he)
                + jnp.float32(rms_eps)
            )
            amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                u_f32 = u_scratch_ref[r0:r1, c0:c1].astype(jnp.float32)
                x_norm_bf16 = (u_f32 * inv_rms * vecs_ref[1:2, c0:c1]).astype(
                    jnp.bfloat16
                )
                u_scratch_ref[r0:r1, c0:c1] = x_norm_bf16
                amax_128 = jnp.maximum(
                    amax_128,
                    vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
                )

            row_amax = jnp.max(amax_128, axis=1, keepdims=True)
            row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
            row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
            inv_sc = jnp.where(
                row_amax == 0,
                jnp.zeros_like(row_amax),
                jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
            ).astype(jnp.bfloat16)
            out_sc_ref[r0:r1, :] = jnp.broadcast_to(row_sc, (sub_m, LANE))
            for s in range(he_dim // 1024):
                c0, c1 = s * 1024, (s + 1) * 1024
                out_i8_ref[r0:r1, c0:c1] = jnp.rint(
                    u_scratch_ref[r0:r1, c0:c1] * inv_sc
                ).astype(jnp.int8)

    x_moe_i8, x_moe_sc_pad = pl.pallas_call(
        _b2_fp8_rowtile_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t_rows, he_dim), jnp.int8),
            jax.ShapeDtypeStruct((t_rows, LANE), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec((bm_eff, h_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm_eff, LANE), lambda i: (i, 0)),
                pl.BlockSpec((h_dim, he_dim), lambda i: (0, 0)),
                pl.BlockSpec((8, he_dim), lambda i: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((bm_eff, he_dim), lambda i: (i, 0)),
                pl.BlockSpec((bm_eff, LANE), lambda i: (i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((bm_eff, he_dim), jnp.bfloat16),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"pallas_b2_fp8_wpool_rowtile_bm{bm_eff}",
    )(router_in_fp8, router_in_sc_pad, wpool_fp8, vecs_8x4096)

    return x_moe_i8, x_moe_sc_pad[:, 0]


def pallas_fused_b1_b2_wpool_quant(
    attn_out_shard: jax.Array,
    stream_shard: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    *,
    router_mode: str = "bf16_hilo",
    fp8_wpool: bool = False,
    half_idx: int | None = None,
    row_range: tuple[int, int] | None = None,
    full_stream_mid: bool = False,
    stream_mid_prev: jax.Array | None = None,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
    topk: int = 8,
    emit_fp8_gmm: bool = False,
) -> tuple[jax.Array, ...]:
    """Single-launch fused Kernel B1 + VMEM-cached 32 MiB wpool_fp8 B2 + packed-bf16-scale quant.

    Eliminates the 32 MiB/L `router_in_bf16` HBM write+read roundtrip between B1 and B2,
    caches `wpool_fp8` [8192, 4096] (32 MiB) in VMEM across all `bm=128` tiles, and packs
    exact 16-bit bf16 row scales into columns [4096:4224] of `x_moe_i8` (0 sync `ag_sc` psums).
    """
    t_local, h_dim = attn_out_shard.shape
    w_k, he_dim = wpool_fp8.shape
    num_experts = wrouter.shape[1]
    bm = LANE
    if row_range is not None:
        row_start, row_end = int(row_range[0]), int(row_range[1])
        if row_start < 0 or row_end <= row_start or row_end > t_local:
            raise ValueError(f"Invalid row_range={row_range} for t_local={t_local}")
        out_rows = row_end - row_start
        if row_start % bm != 0 or out_rows % bm != 0:
            raise ValueError(f"row_range={row_range} must align to bm={bm}")
        m_offset = row_start // bm
        tag_half = f"_r{row_start}_{row_end}"
    else:
        out_rows = (t_local // 2) if half_idx is not None else t_local
        m_offset = (half_idx * (out_rows // bm)) if half_idx is not None else 0
        tag_half = f"_h{half_idx}" if half_idx is not None else "_full"
    assert h_dim == w_k == 8192 and he_dim == 4096
    assert out_rows % bm == 0 and num_experts % 8 == 0

    b12_fast_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
    use_b12_fast = bool(b12_fast_env is not None and b12_fast_env.get())
    hilo_pre = router_mode == "bf16_hilo_pre"
    prep_mode = "bf16_hilo_pre" if hilo_pre else router_mode
    k_rmode = "bf16_hilo" if hilo_pre else router_mode
    skip_zero_wlo = hilo_pre and (use_b12_fast or wrouter.shape[0] == h_dim)
    b1_vecs, bias_t_256x128, w_hi, w_lo = prepare_b1_static_inputs(
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        e_score_bias,
        wrouter,
        router_mode=prep_mode,
    )
    if skip_zero_wlo:
        w_lo = jnp.zeros((8, num_experts), dtype=jnp.bfloat16)
    b2_vecs = (
        wpool_scale.swapaxes(0, 1).reshape(8, he_dim)
        if wpool_scale.ndim == 3
        else prepare_b2_static_inputs(wpool_scale, pre_expert_norm_w)
    )

    num_steps = out_rows // bm
    strip_w = 256 if use_b12_fast else 2048
    num_strips = h_dim // strip_w
    sub_n = 256
    ep_sub_n = 256 if use_b12_fast else 1024
    f256_sfx = "_f256" if use_b12_fast else ""
    inv_h = 1.0 / float(h_dim)
    inv_he = 1.0 / float(he_dim)
    out_cols = he_dim + LANE
    r_dtype = jnp.float8_e4m3fn if fp8_wpool else jnp.bfloat16

    def _fused_b1_b2_kernel(
        attn_ref,
        stream_hbm_ref,
        b1_vecs_ref,
        bias_t_ref,
        w_hi_ref,
        w_lo_ref,
        wpool_ref,
        b2_vecs_ref,
        *tail_refs,
    ):
        if emit_fp8_gmm:
            (
                x_moe_fp8_out_ref,
                x_moe_sc_out_ref,
                stream_mid_hbm_ref,
                ids_out_ref,
                ws_out_ref,
                stream_buf_ref,
                r_scratch_ref,
                u_scratch_ref,
                dma_sems,
            ) = tail_refs
            x_moe_i8_out_ref = None
        else:
            (
                x_moe_i8_out_ref,
                stream_mid_hbm_ref,
                ids_out_ref,
                ws_out_ref,
                stream_buf_ref,
                r_scratch_ref,
                u_scratch_ref,
                dma_sems,
            ) = tail_refs
            x_moe_fp8_out_ref = x_moe_sc_out_ref = None
        step = pl.program_id(0)
        in_row0 = (step + m_offset) * bm
        smid_out_row0 = in_row0 if (full_stream_mid or stream_mid_prev is not None) else step * bm

        @pl.when(step > 0)
        def _wait_prev_smid():
            pltpu.make_async_copy(
                src_ref=stream_buf_ref,
                dst_ref=stream_mid_hbm_ref.at[pl.ds(smid_out_row0 - bm, bm), :],
                sem=dma_sems.at[1],
            ).wait()

        pltpu.make_async_copy(
            src_ref=stream_hbm_ref.at[pl.ds(in_row0, bm), :],
            dst_ref=stream_buf_ref,
            sem=dma_sems.at[0],
        ).start()

        # Pass 1 (overlaps with stream_in HBM->VMEM DMA): post-attention norm variance
        sq_a_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            sq_a_128 = sq_a_128 + vpu_tree_sum_128(a_s * a_s)

        inv_rms_a = jax.lax.rsqrt(
            jnp.sum(sq_a_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(post_norm_eps)
        )

        pltpu.make_async_copy(
            src_ref=stream_hbm_ref.at[pl.ds(in_row0, bm), :],
            dst_ref=stream_buf_ref,
            sem=dma_sems.at[0],
        ).wait()

        # Pass 2: Gated residual + Pre-FFN RMSNorm variance (+ algebraic gain*smid amax for fp8_wpool)
        sq_m_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        g_amax_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            a_s = attn_ref[:, c0:c1].astype(jnp.float32)
            branch_s = (a_s * inv_rms_a).astype(jnp.bfloat16).astype(jnp.float32)
            alpha_s = b1_vecs_ref[0:1, c0:c1]
            beta_s = b1_vecs_ref[1:2, c0:c1]
            smid_s = alpha_s * stream_buf_ref[:, c0:c1] + beta_s * branch_s
            stream_buf_ref[:, c0:c1] = smid_s
            sq_m_128 = sq_m_128 + vpu_tree_sum_128(smid_s * smid_s)
            if fp8_wpool:
                gain_s = b1_vecs_ref[2:3, c0:c1]
                g_amax_128 = jnp.maximum(
                    g_amax_128, vpu_tree_max_128(jnp.abs(smid_s * gain_s))
                )

        inv_rms_m = jax.lax.rsqrt(
            jnp.sum(sq_m_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(rms_eps)
        )
        if fp8_wpool:
            r_amax = jnp.max(g_amax_128, axis=1, keepdims=True) * inv_rms_m
            r_sc_x = jnp.where(
                r_amax == 0,
                jnp.zeros_like(r_amax),
                r_amax * jnp.float32(1.0 / 448.0),
            )
            r_inv_sc = jnp.where(
                r_amax == 0,
                jnp.zeros_like(r_amax),
                jnp.float32(448.0) / jnp.maximum(r_amax, jnp.float32(1e-12)),
            )
        else:
            r_sc_x = None
            r_inv_sc = None

        # Pass 3: Materialize router_in into VMEM r_scratch_ref + accumulate router logits
        logits = jnp.zeros((bm, num_experts), dtype=jnp.float32)
        for s in range(num_strips):
            c0, c1 = s * strip_w, (s + 1) * strip_w
            smid_s = stream_buf_ref[:, c0:c1]
            gain_s = b1_vecs_ref[2:3, c0:c1]
            r_f32 = smid_s * inv_rms_m * gain_s
            r_bf16 = r_f32.astype(jnp.bfloat16)
            if fp8_wpool:
                r_scratch_ref[:, c0:c1] = jnp.clip(
                    r_f32 * r_inv_sc, -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
            else:
                r_scratch_ref[:, c0:c1] = r_bf16

            if k_rmode == "fp32":
                logits = logits + jnp.dot(
                    r_f32,
                    w_hi_ref[c0:c1, :],
                    precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                )
            elif k_rmode == "bf16_hilo":
                r_lo = (r_f32 - r_bf16.astype(jnp.float32)).astype(jnp.bfloat16)
                w_hi_sub = w_hi_ref[c0:c1, :]
                if skip_zero_wlo:
                    logits = (
                        logits
                        + jnp.dot(r_bf16, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
                else:
                    w_lo_sub = w_lo_ref[c0:c1, :]
                    logits = (
                        logits
                        + jnp.dot(r_bf16, w_hi_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_bf16, w_lo_sub, preferred_element_type=jnp.float32)
                        + jnp.dot(r_lo, w_hi_sub, preferred_element_type=jnp.float32)
                    )
            else:
                logits = logits + jnp.dot(
                    r_bf16,
                    w_hi_ref[c0:c1, :],
                    preferred_element_type=jnp.float32,
                )

        # Async write stream_mid [128, 8192] to HBM overlapped under router top-8 + wpool MXU!
        pltpu.make_async_copy(
            src_ref=stream_buf_ref,
            dst_ref=stream_mid_hbm_ref.at[pl.ds(smid_out_row0, bm), :],
            sem=dma_sems.at[1],
        ).start()

        # In-VMEM router sigmoid + top-8
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
        ids_out_ref[:, :] = jnp.swapaxes(
            jnp.pad(ids_k_bm, ((0, LANE - topk), (0, 0))), 0, 1
        )
        ws_out_ref[:, :] = jnp.swapaxes(
            jnp.pad(ws_k_bm.astype(jnp.bfloat16), ((0, LANE - topk), (0, 0))), 0, 1
        )

        # Pass 4: wpool MXU sub-tiled [128, 256] (32 VREGs, 0 spills) + pre_expert_norm + packed int8
        sq_pe_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for ns in range(he_dim // sub_n):
            nc0, nc1 = ns * sub_n, (ns + 1) * sub_n
            acc = jnp.dot(
                r_scratch_ref[:, :],
                wpool_ref[:, nc0:nc1],
                preferred_element_type=jnp.float32,
            )
            w_sc = b2_vecs_ref[0:1, nc0:nc1]
            if fp8_wpool:
                u_bf16 = (acc * (r_sc_x * w_sc)).astype(jnp.bfloat16)
            else:
                u_bf16 = (acc * w_sc).astype(jnp.bfloat16)
            u_scratch_ref[:, nc0:nc1] = u_bf16
            u_f32 = u_bf16.astype(jnp.float32)
            sq_pe_128 = sq_pe_128 + vpu_tree_sum_128(u_f32 * u_f32)

        inv_rms_pe = jax.lax.rsqrt(
            jnp.sum(sq_pe_128, axis=1, keepdims=True) * jnp.float32(inv_he)
            + jnp.float32(rms_eps)
        )
        amax_128 = jnp.zeros((bm, LANE), dtype=jnp.float32)
        for ns in range(he_dim // ep_sub_n):
            nc0, nc1 = ns * ep_sub_n, (ns + 1) * ep_sub_n
            u_f32 = u_scratch_ref[:, nc0:nc1].astype(jnp.float32)
            x_norm_bf16 = (u_f32 * inv_rms_pe * b2_vecs_ref[1:2, nc0:nc1]).astype(
                jnp.bfloat16
            )
            u_scratch_ref[:, nc0:nc1] = x_norm_bf16
            amax_128 = jnp.maximum(
                amax_128,
                vpu_tree_max_128(jnp.abs(x_norm_bf16.astype(jnp.float32))),
            )

        row_amax = jnp.max(amax_128, axis=1, keepdims=True)
        if emit_fp8_gmm:
            row_sc_f32 = jnp.where(
                row_amax == 0,
                jnp.zeros_like(row_amax),
                row_amax * jnp.float32(1.0 / 448.0),
            )
            inv_fp8 = jnp.where(
                row_amax == 0,
                jnp.zeros_like(row_amax),
                jnp.float32(448.0) / jnp.maximum(row_amax, jnp.float32(1e-30)),
            ).astype(jnp.bfloat16)
            x_moe_sc_out_ref[:, :] = jnp.broadcast_to(row_sc_f32, (bm, LANE))
            for m0 in range(0, bm, 32):
                inv_m = inv_fp8[m0 : m0 + 32, :]
                q_parts = [
                    jnp.clip(
                        u_scratch_ref[m0 : m0 + 32, kh * 2048 : (kh + 1) * 2048]
                        * inv_m,
                        -448.0,
                        448.0,
                    )
                    .astype(jnp.float8_e4m3fn)
                    .reshape(32, 16, LANE)
                    for kh in range(he_dim // 2048)
                ]
                x_moe_fp8_out_ref[pl.ds(m0, 32), :, :] = (
                    q_parts[0] if len(q_parts) == 1 else jnp.concatenate(q_parts, axis=1)
                )
        else:
            row_sc = (row_amax * jnp.float32(1.0 / 127.0)).astype(jnp.bfloat16)
            row_sc = jnp.where(row_amax == 0, jnp.zeros_like(row_sc), row_sc)
            inv_sc = jnp.where(
                row_amax == 0,
                jnp.zeros_like(row_amax),
                jnp.float32(127.0) / jnp.maximum(row_amax, jnp.float32(1e-12)),
            ).astype(jnp.bfloat16)

            sc_u16 = jax.lax.bitcast_convert_type(row_sc, jnp.uint16).astype(jnp.int32)
            lo_i8 = (sc_u16 & jnp.int32(0xFF)).astype(jnp.int8)
            hi_i8 = ((sc_u16 >> 8) & jnp.int32(0xFF)).astype(jnp.int8)
            lane = jax.lax.broadcasted_iota(jnp.int32, (bm, LANE), 1)
            x_moe_i8_out_ref[:, he_dim : he_dim + LANE] = jnp.where(
                lane < 64,
                jnp.broadcast_to(lo_i8, (bm, LANE)),
                jnp.broadcast_to(hi_i8, (bm, LANE)),
            )
            for ns in range(he_dim // ep_sub_n):
                nc0, nc1 = ns * ep_sub_n, (ns + 1) * ep_sub_n
                x_moe_i8_out_ref[:, nc0:nc1] = jnp.rint(
                    u_scratch_ref[:, nc0:nc1] * inv_sc
                ).astype(jnp.int8)

        @pl.when(step == num_steps - 1)
        def _wait_last_smid():
            pltpu.make_async_copy(
                src_ref=stream_buf_ref,
                dst_ref=stream_mid_hbm_ref.at[pl.ds(smid_out_row0, bm), :],
                sem=dma_sems.at[1],
            ).wait()

    w_lo_block_shape = (
        (h_dim, num_experts)
        if (k_rmode == "bf16_hilo" and not skip_zero_wlo)
        else (8, num_experts)
    )
    w_lo_idx = 1 if (hilo_pre and not skip_zero_wlo) else 0
    tag_fp8 = "fp8" if fp8_wpool else "bf16"
    tag_out = "_outfp8" if emit_fp8_gmm else ""
    smid_rows = (
        t_local if (full_stream_mid or stream_mid_prev is not None) else out_rows
    )
    kernel_fn = _fused_b1_b2_kernel
    extra_in = ()
    extra_specs = []
    aliases = {}
    if stream_mid_prev is not None:
        # Operand 8 (after 8 base inputs) is stream_mid's aliased HBM buffer; skip its input ref.
        kernel_fn = lambda *r: _fused_b1_b2_kernel(*r[:8], *r[9:])  # noqa: E731
        extra_in = (stream_mid_prev,)
        extra_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]
        aliases = {8: (2 if emit_fp8_gmm else 1)}

    if emit_fp8_gmm:
        out_shape_b12: tuple[Any, ...] = (
            jax.ShapeDtypeStruct((out_rows, he_dim // LANE, LANE), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.float32),
            jax.ShapeDtypeStruct((smid_rows, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
        )
        out_specs_b12 = [
            pl.BlockSpec((bm, he_dim // LANE, LANE), lambda i: (i, 0, 0)),
            pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
            pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
        ]
    else:
        out_shape_b12 = (
            jax.ShapeDtypeStruct((out_rows, out_cols), jnp.int8),
            jax.ShapeDtypeStruct((smid_rows, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.int32),
            jax.ShapeDtypeStruct((out_rows, LANE), jnp.bfloat16),
        )
        out_specs_b12 = [
            pl.BlockSpec((bm, out_cols), lambda i: (i, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
            pl.BlockSpec((bm, LANE), lambda i: (i, 0)),
        ]

    res_b12 = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape_b12,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec((bm, h_dim), lambda i, m_off=m_offset: (i + m_off, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((8, h_dim), lambda i: (0, 0)),
                pl.BlockSpec((num_experts, LANE), lambda i: (0, 0)),
                pl.BlockSpec((h_dim, num_experts), lambda i: (0, 0)),
                pl.BlockSpec(w_lo_block_shape, lambda i, _idx=w_lo_idx: (_idx, 0)),
                pl.BlockSpec((h_dim, he_dim), lambda i: (0, 0)),
                pl.BlockSpec((8, he_dim), lambda i: (0, 0)),
                *extra_specs,
            ],
            out_specs=out_specs_b12,
            scratch_shapes=[
                pltpu.VMEM((bm, h_dim), jnp.float32),
                pltpu.VMEM((bm, h_dim), r_dtype),
                pltpu.VMEM((bm, he_dim), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=62 * 1024 * 1024,
        ),
        input_output_aliases=aliases,
        name=f"pallas_fused_b1_b2_{router_mode}_{tag_fp8}{tag_half}_bm{bm}{f256_sfx}{tag_out}",
    )(
        attn_out_shard,
        stream_shard,
        b1_vecs,
        bias_t_256x128,
        w_hi,
        w_lo,
        wpool_fp8,
        b2_vecs,
        *extra_in,
    )

    ids_dtype = jnp.uint8 if num_experts <= 256 else jnp.uint16
    if emit_fp8_gmm:
        x_moe_fp8, x_moe_sc_pad, stream_mid, ids_pad_out, ws_pad_out = res_b12
        local_topk_ids_u8 = ids_pad_out[:, :topk].astype(ids_dtype)
        local_topk_weights = ws_pad_out[:, :topk]
        return (
            x_moe_fp8,
            x_moe_sc_pad[:, 0],
            stream_mid,
            local_topk_ids_u8,
            local_topk_weights,
        )
    x_moe_i8, stream_mid, ids_pad_out, ws_pad_out = res_b12
    local_topk_ids_u8 = ids_pad_out[:, :topk].astype(ids_dtype)
    local_topk_weights = ws_pad_out[:, :topk]
    return x_moe_i8, stream_mid, local_topk_ids_u8, local_topk_weights
