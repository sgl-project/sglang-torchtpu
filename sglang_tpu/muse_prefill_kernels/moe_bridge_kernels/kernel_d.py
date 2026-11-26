# SPDX-License-Identifier: Apache-2.0
"""Kernel D: Fused Top-8 Combine + wunpool (FP8xBF16) + Post-FFN Norm + Gated Residual + Next Input Norm."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs

from .kernel_b1 import LANE, vpu_tree_max_128, vpu_tree_sum_128


def prepare_d_static_inputs(
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
) -> jax.Array:
    """Pack 1D vectors for Kernel D into an (8, 8192) VMEM-aligned fp32 tensor."""
    if wunpool_scale.ndim == 2 and wunpool_scale.shape[0] == 8:
        return wunpool_scale
    h_dim = wunpool_scale.shape[-1]
    ws_1d = wunpool_scale.reshape(1, h_dim).astype(jnp.float32)
    pfg_1d = post_ffn_norm_w.reshape(1, h_dim).astype(jnp.float32) + jnp.float32(1.0)
    a_1d = alpha_ffn.reshape(1, h_dim).astype(jnp.float32)
    b_1d = beta_ffn.reshape(1, h_dim).astype(jnp.float32)
    nig_1d = next_input_norm_w.reshape(1, h_dim).astype(jnp.float32) + jnp.float32(1.0)
    z_1d = jnp.zeros((1, h_dim), dtype=jnp.float32)
    return jnp.concatenate(
        [ws_1d, pfg_1d, a_1d, b_1d, nig_1d, z_1d, z_1d, z_1d], axis=0
    )


def pallas_combine_unpool_residual_nextnorm_4slab(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    stream_shard_mid: jax.Array,
    wunpool_fp8: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    tok_tile: int = 64,
    zero_spill: bool = False,
    sub_n_dot: int = 512,
    kmajor_buf: bool = False,
    loop_slabs: bool = False,
    unroll_dma: bool = True,
    p34_col_step: int = 512,
    t64_c_step: int = 8,
    t64_sub_n: int = 512,
    fp8_unpool: bool = False,
    fp8_unpool_qbs: int = 0,
    emit_fp8_attn_in: bool = False,
    emit_static_fp8_attn_in: bool = False,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    row_start: int = 0,
    num_rows: int | None = None,
    stream_out: jax.Array | None = None,
    salu_overlap: bool = False,
    scrub_mxu_nan: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Design D1 (4-Slab Pipelined VMEM Cache): Single-kernel fused combine + unpool + norms.

    Eliminates Stage 3 Output All-Gather and fuses Kernel 5 directly into _tc_combine.
    Loads wunpool_fp8 [4096, 8192] (32 MiB) once at step == 0 in 4 pipelined 8 MiB slabs
    overlapped with step 0's top-8 slot gathers, and reuses from VMEM on all step > 0.

    When `emit_fp8_attn_in=True`, uses algebraic cancellation in Phase 3/4 to emit
    `next_attn_in_packed_fp8: fp8_e4m3fn[chunk_size, 8192 + 128]` with Hi+Lo E4M3 scale
    packed in columns `[8192:8320]`, halving the HBM write of `next_attn_in` (`16 MiB -> 8 MiB`
    at 8K TP8) and eliminating `sp_input_layernorm_algebraic_fp8` + `ag_s` in `Kernel A1`!
    When `emit_static_fp8_attn_in=True`, skips `amax_next_128` and emits static `* 32.0`
    `fp8_e4m3fn[chunk_size, 8192]` directly matching `Kernel A1` (`a1_v6_prenormed` / `a1_v8_sp_prenormed`).

    ``row_start`` / ``num_rows`` (tok_tile multiples, non-zero-spill path): run only tokens
    [row_start, row_start + num_rows). new_stream keeps all chunk_size rows and is written in
    place into ``stream_out`` when given (rows outside the range keep its values; without it
    they are undefined); next_attn_in has num_rows rows.

    Returns:
        new_stream_shard: fp32[chunk_size, 8192]
        next_attn_in: bf16[chunk_size, 8192] (or fp8_e4m3fn[chunk_size, 8320 / 8192])
    """
    chunk_size, topk = inv_perm_c.shape
    n2_cols, lanes = out_buf_3d.shape[1], out_buf_3d.shape[2]
    he_dim = n2_cols * lanes
    w_k, h_dim = wunpool_fp8.shape
    assert he_dim == w_k == 4096 and h_dim == 8192
    in_buf_dtype = out_buf_3d.dtype
    in_buf_is_fp8 = jnp.dtype(in_buf_dtype) == jnp.float8_e4m3fn
    if in_buf_is_fp8:
        from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
            resolve_static_fp8_rs_scale,
        )

        _abs_sfp8, _ = resolve_static_fp8_rs_scale(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS.get()
        )
        if _abs_sfp8 > 0.0:
            local_topk_weights = (
                local_topk_weights.astype(jnp.float32) * jnp.float32(1.0 / _abs_sfp8)
            ).astype(local_topk_weights.dtype)
    env_tok_tile = envs.SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE.get()
    d_t128_fast_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_D_T128_FAST", None)
    use_d_t128_fast = bool(d_t128_fast_env is not None and d_t128_fast_env.get())
    if env_tok_tile not in (0, 64) and tok_tile == 64:
        tok_tile = env_tok_tile
    elif (
        use_d_t128_fast
        and tok_tile == 64
        and row_start % 128 == 0
        and (num_rows is None or num_rows % 128 == 0)
        and chunk_size >= 128
        and chunk_size % 128 == 0
    ):
        tok_tile = 128
    tok_tile_eff = min(tok_tile, chunk_size)
    assert chunk_size % tok_tile_eff == 0
    env_fp8_qbs = envs.SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL.get()
    eff_fp8_qbs = (
        fp8_unpool_qbs
        if fp8_unpool_qbs > 0
        else (env_fp8_qbs if env_fp8_qbs > 0 else (4096 if fp8_unpool else 0))
    )
    fp8_unpool = eff_fp8_qbs > 0
    use_blk_fp8_unpool = 0 < eff_fp8_qbs < he_dim
    num_fp8_kb = (he_dim // eff_fp8_qbs) if use_blk_fp8_unpool else 1
    use_salu_overlap = salu_overlap or envs.SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP.get()
    use_zero_spill = (
        zero_spill
        or envs.SGLANG_TPU_MUSE_KERNEL_D_ZERO_SPILL.get()
        or tok_tile_eff > 64
        or (tok_tile_eff >= 32 and chunk_size <= 32)
    ) and not (emit_fp8_attn_in or use_blk_fp8_unpool)
    if not use_zero_spill:
        # _d_4slab_kernel with t128+fp8q512+c16h+bf16 next_in exceeds 63 MiB (63.51M>63.00M),
        # while emit_static_fp8_attn_in=True fits cleanly at t128.
        if chunk_size <= 32 and chunk_size % 16 == 0:
            tok_tile_eff = min(tok_tile_eff, 16)
        elif not emit_static_fp8_attn_in:
            tok_tile_eff = min(tok_tile_eff, 64)

    num_rows = chunk_size - row_start if num_rows is None else num_rows
    assert row_start % tok_tile_eff == 0 and num_rows % tok_tile_eff == 0, (
        row_start,
        num_rows,
    )
    partial_rows = num_rows != chunk_size or stream_out is not None
    tile_off = row_start // tok_tile_eff
    # Token-indexed blocks of grid step i are tile i + tile_off (the default map traces unchanged).
    tok_map = (
        (lambda i, *_: (i, 0)) if tile_off == 0 else (lambda i, *_: (i + tile_off, 0))
    )

    num_steps = num_rows // tok_tile_eff
    d_opt_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_D_OPT", None)
    eff_d_opt = int(d_opt_env.get()) if d_opt_env is not None else 0
    d_fast_dma = (eff_d_opt & 1) != 0
    d_fuse_p34 = ((eff_d_opt & 2) != 0) and not emit_fp8_attn_in
    d_mxu_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE", None)
    eff_mxu_combine = (
        bool(d_mxu_env is not None and d_mxu_env.get())
        and topk == 8
        and LANE == 16 * topk
        and chunk_size % 16 == 0
        and tok_tile_eff % 16 == 0
        and num_rows % 16 == 0
        and not (use_zero_spill and kmajor_buf)
    )
    if eff_mxu_combine:
        eye16 = jnp.eye(16, dtype=jnp.bfloat16)[None, :, :, None]
        w_pad = (
            local_topk_weights.astype(jnp.bfloat16).reshape(
                chunk_size // 16, 16, 1, topk
            )
            * eye16
        ).reshape(chunk_size, LANE)
    else:
        w_pad = jnp.pad(
            local_topk_weights.astype(jnp.bfloat16), ((0, 0), (0, LANE - topk))
        )
    vecs_8x8192 = prepare_d_static_inputs(
        wunpool_scale, post_ffn_norm_w, alpha_ffn, beta_ffn, next_input_norm_w
    )

    slab_w = 2048
    num_slabs = h_dim // slab_w
    inv_h = 1.0 / float(h_dim)
    next_in_cols = (h_dim + LANE) if emit_fp8_attn_in else h_dim
    next_in_dtype = (
        jnp.float8_e4m3fn
        if (emit_fp8_attn_in or emit_static_fp8_attn_in)
        else jnp.bfloat16
    )

    def _combine_fp8_phase1(buf_2x_ref, slot, w_vmem_ref, u_scratch_ref):
        r_step = 16 if tok_tile_eff % 16 == 0 else 8
        c_step = 8 if (r_step == 16 and n2_cols % 8 == 0) else n2_cols
        ts_chunk = 4 if r_step == 16 else 2
        for r0 in range(0, tok_tile_eff, r_step):
            for c0 in range(0, n2_cols, c_step):
                parts = []
                for ts in range(0, r_step, ts_chunk):
                    m0 = (r0 + ts) * topk
                    m1 = (r0 + ts + ts_chunk) * topk
                    w_sub = w_vmem_ref[r0 + ts : r0 + ts + ts_chunk, :topk].reshape(
                        ts_chunk, topk, 1, 1
                    )
                    vals_sub = (
                        buf_2x_ref[slot, m0:m1, c0 : c0 + c_step, :]
                        .astype(jnp.bfloat16)
                        .reshape(ts_chunk, topk, c_step, lanes)
                    )
                    parts.append(
                        jnp.sum(vals_sub * w_sub, axis=1).reshape(
                            ts_chunk, c_step * lanes
                        )
                    )
                u_scratch_ref[r0 : r0 + r_step, c0 * lanes : (c0 + c_step) * lanes] = (
                    jnp.concatenate(parts, axis=0)
                )

    def _load_mxu_tile_128x1024(buf_2x_ref, slot, m0, m1, c0, c1, c_grp):
        vals_raw = buf_2x_ref[slot, m0:m1, c0:c1, :]
        if in_buf_is_fp8:
            vals_raw = vals_raw.astype(jnp.bfloat16)
            if scrub_mxu_nan:
                # float8_e4m3fn has no Inf (only 0x7F/0xFF NaN); bf16 self-compare avoids the f32 cast tax.
                vals_raw = jnp.where(vals_raw == vals_raw, vals_raw, jnp.bfloat16(0.0))
        elif scrub_mxu_nan:
            # Scrub NaN/Inf on bfloat16 inputs so 0.0 * NaN/Inf cannot propagate within a 16-row block.
            vals_raw = jnp.where(
                jnp.isfinite(vals_raw.astype(jnp.float32)), vals_raw, jnp.bfloat16(0.0)
            )
        return vals_raw.reshape(16 * topk, c_grp * lanes)

    if use_zero_spill:
        t_sub = min(32, tok_tile_eff)
        vecs_3d = vecs_8x8192.reshape(8, num_slabs, slab_w).swapaxes(0, 1)

        def _d_4slab_zs_kernel(
            inv_perm_ref,
            src_hbm_ref,
            wunpool_hbm_ref,
            stream_mid_hbm_ref,
            w_vmem_ref,
            vecs_3d_ref,
            new_stream_hbm_ref,
            next_in_hbm_ref,
            buf_2x_ref,
            sem_2x_ref,
            u_scratch_ref,
            u_fp8_scratch_ref,
            w_cache_ref,
            w_sem_ref,
            fw_buf_ref,
            smid_2x_ref,
            smid_sem_ref,
            next_2x_ref,
            next_sem_ref,
            ns_sem_ref,
        ):
            step = pl.program_id(0)
            base_tok = (step + tile_off) * tok_tile_eff if tile_off else step * tok_tile_eff
            next_tok = step * tok_tile_eff
            slot = step % 2
            eff_salu_ovlp_zs = (
                use_d_t128_fast
                and use_salu_overlap
                and (tok_tile_eff % num_slabs == 0)
                and not loop_slabs
            )
            use_c16_zs = (
                use_d_t128_fast
                and n2_cols == 32
                and not kmajor_buf
            )
            c_step_zs = 16 if use_c16_zs else 4

            def _start_step_range(s, sl, t0, t1):
                b_tok = (s + tile_off) * tok_tile_eff if tile_off else s * tok_tile_eff

                def _dma_tok(t, _):
                    tok = b_tok + t
                    idx_base = tok * topk
                    for k in range(topk):
                        idx_k = inv_perm_ref[idx_base + k]
                        dst_slice = (
                            buf_2x_ref.at[sl, k, pl.ds(t, 1), :, :]
                            if kmajor_buf
                            else buf_2x_ref.at[sl, pl.ds(t * topk + k, 1), :, :]
                        )
                        pltpu.make_async_copy(
                            src_ref=src_hbm_ref.at[pl.ds(idx_k, 1), :, :],
                            dst_ref=dst_slice,
                            sem=sem_2x_ref.at[sl],
                        ).start()
                    return None

                if unroll_dma:
                    for t in range(t0, t1):
                        _dma_tok(t, None)
                else:
                    jax.lax.fori_loop(t0, t1, _dma_tok, None)

            def _start_step(s, sl):
                _start_step_range(s, sl, 0, tok_tile_eff)

            def _wait_step(sl):
                pltpu.make_async_copy(
                    src_ref=buf_2x_ref.at[sl],
                    dst_ref=buf_2x_ref.at[sl],
                    sem=sem_2x_ref.at[sl],
                ).wait()

            @pl.when(step == 0)
            def _init_step0():
                if d_fast_dma:
                    for s in range(num_slabs):
                        pltpu.make_async_copy(
                            src_ref=wunpool_hbm_ref.at[:, pl.ds(s * slab_w, slab_w)],
                            dst_ref=w_cache_ref.at[s],
                            sem=w_sem_ref if loop_slabs else w_sem_ref.at[s],
                        ).start()
                    _start_step(0, 0)
                else:
                    _start_step(0, 0)
                    for s in range(num_slabs):
                        pltpu.make_async_copy(
                            src_ref=wunpool_hbm_ref.at[:, pl.ds(s * slab_w, slab_w)],
                            dst_ref=w_cache_ref.at[s],
                            sem=w_sem_ref if loop_slabs else w_sem_ref.at[s],
                        ).start()

            if not eff_salu_ovlp_zs:

                @pl.when(step + 1 < num_steps)
                def _prefetch_next():
                    _start_step(step + 1, 1 - slot)

            for s_pre in range(2):
                pltpu.make_async_copy(
                    src_ref=stream_mid_hbm_ref.at[
                        pl.ds(base_tok, tok_tile_eff), pl.ds(s_pre * slab_w, slab_w)
                    ],
                    dst_ref=smid_2x_ref.at[s_pre],
                    sem=smid_sem_ref.at[s_pre],
                ).start()

            _wait_step(slot)

            # Phase 1: Low-VREG top-8 combine into u_scratch_ref [tok_tile_eff, 4096]
            amax_u_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            if in_buf_is_fp8 and not kmajor_buf and not eff_mxu_combine:
                _combine_fp8_phase1(buf_2x_ref, slot, w_vmem_ref, u_scratch_ref)
            elif eff_mxu_combine:
                c_grp_zs = 8
                amax_groups_zs = []
                for b_16 in range(tok_tile_eff // 16):
                    r0, r1 = b_16 * 16, (b_16 + 1) * 16
                    m0, m1 = r0 * topk, r1 * topk
                    w_16x128 = w_vmem_ref[r0:r1, :]
                    amax_b_zs = jnp.zeros((16, LANE), dtype=jnp.float32)
                    for c_blk in range(n2_cols // c_grp_zs):
                        c0, c1 = c_blk * c_grp_zs, (c_blk + 1) * c_grp_zs
                        u0, u1 = c0 * lanes, c1 * lanes
                        vals_128x1024 = _load_mxu_tile_128x1024(
                            buf_2x_ref, slot, m0, m1, c0, c1, c_grp_zs
                        )
                        u_sub_bf16 = jnp.dot(
                            w_16x128,
                            vals_128x1024,
                            preferred_element_type=jnp.float32,
                        ).astype(jnp.bfloat16)
                        u_scratch_ref[r0:r1, u0:u1] = u_sub_bf16
                        if fp8_unpool:
                            amax_b_zs = jnp.maximum(
                                amax_b_zs,
                                vpu_tree_max_128(
                                    jnp.abs(u_sub_bf16.astype(jnp.float32))
                                ),
                            )
                    if fp8_unpool:
                        amax_groups_zs.append(amax_b_zs)
                if fp8_unpool:
                    amax_u_128 = jnp.concatenate(amax_groups_zs, axis=0)
            else:
                for tb in range(tok_tile_eff // t_sub):
                    t0, t1 = tb * t_sub, (tb + 1) * t_sub
                    for c_blk in range(n2_cols // c_step_zs):
                        c0, c1 = c_blk * c_step_zs, (c_blk + 1) * c_step_zs
                        if kmajor_buf:
                            w0 = w_vmem_ref[t0:t1, 0:1][:, :, None]
                            w1 = w_vmem_ref[t0:t1, 1:2][:, :, None]
                            s01 = (
                                buf_2x_ref[slot, 0, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w0
                                + buf_2x_ref[slot, 1, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w1
                            )
                            w2 = w_vmem_ref[t0:t1, 2:3][:, :, None]
                            w3 = w_vmem_ref[t0:t1, 3:4][:, :, None]
                            s23 = (
                                buf_2x_ref[slot, 2, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w2
                                + buf_2x_ref[slot, 3, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w3
                            )
                            s03 = s01 + s23
                            w4 = w_vmem_ref[t0:t1, 4:5][:, :, None]
                            w5 = w_vmem_ref[t0:t1, 5:6][:, :, None]
                            s45 = (
                                buf_2x_ref[slot, 4, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w4
                                + buf_2x_ref[slot, 5, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w5
                            )
                            w6 = w_vmem_ref[t0:t1, 6:7][:, :, None]
                            w7 = w_vmem_ref[t0:t1, 7:8][:, :, None]
                            s67 = (
                                buf_2x_ref[slot, 6, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w6
                                + buf_2x_ref[slot, 7, t0:t1, c0:c1, :].astype(jnp.bfloat16) * w7
                            )
                            u_sub = (s03 + (s45 + s67)).reshape(
                                t_sub, c_step_zs * lanes
                            )
                        else:
                            b0, b1 = tb * (t_sub * topk), (tb + 1) * (t_sub * topk)
                            w_tok = w_vmem_ref[t0:t1, :topk].reshape(t_sub, topk, 1, 1)
                            vals_sub = buf_2x_ref[slot, b0:b1, c0:c1, :].reshape(
                                t_sub, topk, c_step_zs, lanes
                            )
                            u_sub = jnp.sum(vals_sub * w_tok, axis=1).reshape(
                                t_sub, c_step_zs * lanes
                            )
                        u_scratch_ref[t0:t1, c0 * lanes : c1 * lanes] = u_sub

            if fp8_unpool:
                if not eff_mxu_combine:
                    for c_blk in range(4):
                        u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                        amax_u_128 = jnp.maximum(
                            amax_u_128,
                            vpu_tree_max_128(
                                jnp.abs(u_scratch_ref[:, u0:u1].astype(jnp.float32))
                            ),
                        )
                row_amax_u = jnp.max(amax_u_128, axis=1, keepdims=True)
                u_sc = jnp.where(
                    row_amax_u == 0,
                    jnp.zeros_like(row_amax_u),
                    row_amax_u * jnp.float32(1.0 / 448.0),
                )
                inv_u_sc = jnp.where(
                    row_amax_u == 0,
                    jnp.zeros_like(row_amax_u),
                    jnp.float32(448.0) / jnp.maximum(row_amax_u, jnp.float32(1e-30)),
                )
                for c_blk in range(4):
                    u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                    u_fp8_scratch_ref[:, u0:u1] = jnp.clip(
                        u_scratch_ref[:, u0:u1].astype(jnp.float32) * inv_u_sc,
                        -448.0,
                        448.0,
                    ).astype(jnp.float8_e4m3fn)
            else:
                u_sc = None

            if loop_slabs:

                @pl.when(step == 0)
                def _wait_w_all():
                    pltpu.make_async_copy(
                        src_ref=w_cache_ref,
                        dst_ref=w_cache_ref,
                        sem=w_sem_ref,
                    ).wait()

            if eff_salu_ovlp_zs and not d_fast_dma:

                @pl.when(step == 0)
                def _wait_w_all_zs():
                    for sw_i in range(num_slabs):
                        pltpu.make_async_copy(
                            src_ref=w_cache_ref.at[sw_i],
                            dst_ref=w_cache_ref.at[sw_i],
                            sem=w_sem_ref.at[sw_i],
                        ).wait()

            # Phase 2: 4-slab wunpool dot + post_feedforward_layernorm variance (no hoisted lhs_full!)
            def _run_slab_p2(s, sq_acc):
                if not loop_slabs and (not eff_salu_ovlp_zs or d_fast_dma):

                    @pl.when(step == 0)
                    def _wait_w_slab(s=s):
                        pltpu.make_async_copy(
                            src_ref=w_cache_ref.at[s],
                            dst_ref=w_cache_ref.at[s],
                            sem=w_sem_ref.at[s],
                        ).wait()

                for sub in range(slab_w // sub_n_dot):
                    cn0, cn1 = sub * sub_n_dot, (sub + 1) * sub_n_dot
                    lhs_ref = u_fp8_scratch_ref if fp8_unpool else u_scratch_ref
                    acc = jnp.dot(
                        lhs_ref[:, :],
                        w_cache_ref[s, :, cn0:cn1],
                        preferred_element_type=jnp.float32,
                    )
                    w_sc_sub = vecs_3d_ref[s, 0:1, cn0:cn1]
                    sc_prod = (u_sc * w_sc_sub) if fp8_unpool else w_sc_sub
                    y_sub = (acc * sc_prod).astype(jnp.bfloat16).astype(jnp.float32)
                    fw_sub = y_sub * vecs_3d_ref[s, 1:2, cn0:cn1]
                    fw_buf_ref[s, :, cn0:cn1] = fw_sub
                    sq_acc = sq_acc + vpu_tree_sum_128(fw_sub * fw_sub)
                if eff_salu_ovlp_zs:
                    next_s_zs = jnp.minimum(step + 1, num_steps - 1)
                    t_per_slab_zs = tok_tile_eff // num_slabs
                    _start_step_range(
                        next_s_zs,
                        1 - slot,
                        s * t_per_slab_zs,
                        (s + 1) * t_per_slab_zs,
                    )
                return sq_acc

            sq_ffn_init = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            if loop_slabs:
                sq_ffn_128 = jax.lax.fori_loop(0, num_slabs, _run_slab_p2, sq_ffn_init)
            else:
                sq_ffn_128 = sq_ffn_init
                for s in range(num_slabs):
                    sq_ffn_128 = _run_slab_p2(s, sq_ffn_128)

            inv_rms_ffn = jax.lax.rsqrt(
                jnp.sum(sq_ffn_128, axis=1, keepdims=True) * jnp.float32(inv_h)
                + jnp.float32(post_norm_eps)
            )

            # Phase 3: Double-buffered slab-streamed gated residual + next_in variance
            sq_next_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            for s in range(num_slabs):
                s_slot = s % 2
                pltpu.make_async_copy(
                    src_ref=smid_2x_ref.at[s_slot],
                    dst_ref=smid_2x_ref.at[s_slot],
                    sem=smid_sem_ref.at[s_slot],
                ).wait()

                for sub in range(slab_w // p34_col_step):
                    cn0, cn1 = sub * p34_col_step, (sub + 1) * p34_col_step
                    fw_s = fw_buf_ref[s, :, cn0:cn1]
                    branch_s = (
                        (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                    )
                    new_s = (
                        vecs_3d_ref[s, 2:3, cn0:cn1] * smid_2x_ref[s_slot, :, cn0:cn1]
                        + vecs_3d_ref[s, 3:4, cn0:cn1] * branch_s
                    )
                    fw_buf_ref[s, :, cn0:cn1] = new_s
                    sq_next_128 = sq_next_128 + (
                        new_s * new_s
                        if p34_col_step == LANE
                        else vpu_tree_sum_128(new_s * new_s)
                    )

                if s + 2 < num_slabs:
                    pltpu.make_async_copy(
                        src_ref=stream_mid_hbm_ref.at[
                            pl.ds(base_tok, tok_tile_eff),
                            pl.ds((s + 2) * slab_w, slab_w),
                        ],
                        dst_ref=smid_2x_ref.at[s_slot],
                        sem=smid_sem_ref.at[s_slot],
                    ).start()

                pltpu.make_async_copy(
                    src_ref=fw_buf_ref.at[s],
                    dst_ref=new_stream_hbm_ref.at[
                        pl.ds(base_tok, tok_tile_eff), pl.ds(s * slab_w, slab_w)
                    ],
                    sem=ns_sem_ref.at[s],
                ).start()

            inv_rms_next = jax.lax.rsqrt(
                jnp.sum(sq_next_128, axis=1, keepdims=True) * jnp.float32(inv_h)
                + jnp.float32(rms_eps)
            )
            inv_rms_next_eff = (
                inv_rms_next * jnp.float32(32.0)
                if emit_static_fp8_attn_in
                else inv_rms_next
            )

            # Phase 4: Double-buffered next_in norm & HBM write
            for s in range(num_slabs):
                s_slot = s % 2
                if s >= 2:
                    pltpu.make_async_copy(
                        src_ref=next_2x_ref.at[s_slot],
                        dst_ref=next_in_hbm_ref.at[
                            pl.ds(next_tok, tok_tile_eff),
                            pl.ds((s - 2) * slab_w, slab_w),
                        ],
                        sem=next_sem_ref.at[s_slot],
                    ).wait()
                for sub in range(slab_w // p34_col_step):
                    cn0, cn1 = sub * p34_col_step, (sub + 1) * p34_col_step
                    new_s = fw_buf_ref[s, :, cn0:cn1]
                    scaled_next = new_s * inv_rms_next_eff * vecs_3d_ref[s, 4:5, cn0:cn1]
                    if emit_static_fp8_attn_in:
                        scaled_next = jnp.clip(scaled_next, -448.0, 448.0)
                    next_2x_ref[s_slot, :, cn0:cn1] = scaled_next.astype(next_in_dtype)
                pltpu.make_async_copy(
                    src_ref=next_2x_ref.at[s_slot],
                    dst_ref=next_in_hbm_ref.at[
                        pl.ds(next_tok, tok_tile_eff), pl.ds(s * slab_w, slab_w)
                    ],
                    sem=next_sem_ref.at[s_slot],
                ).start()

            for s_rem in (num_slabs - 2, num_slabs - 1):
                pltpu.make_async_copy(
                    src_ref=next_2x_ref.at[s_rem % 2],
                    dst_ref=next_in_hbm_ref.at[
                        pl.ds(next_tok, tok_tile_eff), pl.ds(s_rem * slab_w, slab_w)
                    ],
                    sem=next_sem_ref.at[s_rem % 2],
                ).wait()
            for s in range(num_slabs):
                pltpu.make_async_copy(
                    src_ref=fw_buf_ref.at[s],
                    dst_ref=new_stream_hbm_ref.at[
                        pl.ds(base_tok, tok_tile_eff), pl.ds(s * slab_w, slab_w)
                    ],
                    sem=ns_sem_ref.at[s],
                ).wait()
            if eff_salu_ovlp_zs:

                @pl.when(step == num_steps - 1)
                def _drain_extra_pf_zs():
                    _wait_step(1 - slot)

        u_fp8_rows = tok_tile_eff if fp8_unpool else 8
        buf_2x_shape = (
            (2, topk, tok_tile_eff, n2_cols, lanes)
            if kmajor_buf
            else (2, tok_tile_eff * topk, n2_cols, lanes)
        )
        km_tag = "_km" if kmajor_buf else ""
        ls_tag = "_ls" if loop_slabs else ""
        ud_tag = "_ud" if unroll_dma else ""
        fp8_tag = "_fp8u" if fp8_unpool else ""
        fast_tag = "_c16so" if use_d_t128_fast else ""
        do_tag = f"_do{eff_d_opt}" if eff_d_opt else ""
        mxu_tag = (("_mxuc" if scrub_mxu_nan else "_mxuc_ns") if eff_mxu_combine else "")
        sfp8_tag = "_sfp8" if emit_static_fp8_attn_in else ""
        zs_in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((tok_tile_eff, LANE), tok_map),
            pl.BlockSpec((num_slabs, 8, slab_w), lambda i, *_: (0, 0, 0)),
        ]
        zs_call_args = [
            inv_perm_c.reshape(-1),
            out_buf_3d,
            wunpool_fp8,
            stream_shard_mid,
            w_pad,
            vecs_3d,
        ]
        zs_kernel_fn = _d_4slab_zs_kernel
        zs_aliases: dict[int, int] = {}
        if stream_out is not None:
            zs_aliases = {len(zs_call_args): 0}
            zs_in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
            zs_call_args.append(stream_out)
            # Operand 6 (after scalar prefetch + 5 base refs) is aliased stream_out HBM buffer.
            zs_kernel_fn = lambda *r: _d_4slab_zs_kernel(*r[:6], *r[7:])  # noqa: E731
        return pl.pallas_call(
            zs_kernel_fn,
            out_shape=(
                jax.ShapeDtypeStruct((chunk_size, h_dim), jnp.float32),
                jax.ShapeDtypeStruct((num_rows, next_in_cols), next_in_dtype),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(num_steps,),
                in_specs=zs_in_specs,
                out_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                scratch_shapes=[
                    pltpu.VMEM(buf_2x_shape, in_buf_dtype),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.VMEM((tok_tile_eff, he_dim), jnp.bfloat16),
                    pltpu.VMEM((u_fp8_rows, he_dim), jnp.float8_e4m3fn),
                    pltpu.VMEM((num_slabs, he_dim, slab_w), jnp.float8_e4m3fn),
                    pltpu.SemaphoreType.DMA(())
                    if loop_slabs
                    else pltpu.SemaphoreType.DMA((num_slabs,)),
                    pltpu.VMEM((num_slabs, tok_tile_eff, slab_w), jnp.float32),
                    pltpu.VMEM((2, tok_tile_eff, slab_w), jnp.float32),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.VMEM((2, tok_tile_eff, slab_w), next_in_dtype),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.SemaphoreType.DMA((num_slabs,)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=62 * 1024 * 1024,
            ),
            input_output_aliases=zs_aliases,
            name=f"pallas_d_combine_4slab_v3_t{tok_tile_eff}_sn{sub_n_dot}_p{p34_col_step}{km_tag}{ls_tag}{ud_tag}{fp8_tag}{sfp8_tag}{fast_tag}{do_tag}{mxu_tag}",
        )(*zs_call_args)

    def _d_4slab_kernel(
        inv_perm_ref,
        src_hbm_ref,
        wunpool_hbm_ref,
        w_vmem_ref,
        stream_mid_ref,
        vecs_ref,
        new_stream_out_ref,
        next_in_out_ref,
        buf_2x_ref,
        sem_2x_ref,
        u_scratch_ref,
        u_fp8_scratch_ref,
        *tail_refs,
    ):
        if use_blk_fp8_unpool:
            u_sc_blk_ref, w_cache_ref, w_sem_ref = tail_refs
        else:
            u_sc_blk_ref = None
            w_cache_ref, w_sem_ref = tail_refs
        step = pl.program_id(0)
        slot = step % 2
        eff_salu_ovlp = use_salu_overlap and (tok_tile_eff % num_slabs == 0)

        def _start_step_range(s, sl, t0, t1):
            base_tok = (s + tile_off) * tok_tile_eff if tile_off else s * tok_tile_eff
            for t in range(t0, t1):
                tok = base_tok + t
                dst_base = t * topk
                idx_base = tok * topk
                idx8 = [inv_perm_ref[idx_base + k] for k in range(topk)]
                for k in range(topk):
                    pltpu.make_async_copy(
                        src_ref=src_hbm_ref.at[pl.ds(idx8[k], 1), :, :],
                        dst_ref=buf_2x_ref.at[sl, pl.ds(dst_base + k, 1), :, :],
                        sem=sem_2x_ref.at[sl],
                    ).start()

        def _start_step(s, sl):
            _start_step_range(s, sl, 0, tok_tile_eff)

        def _wait_step(sl):
            pltpu.make_async_copy(
                src_ref=buf_2x_ref.at[sl],
                dst_ref=buf_2x_ref.at[sl],
                sem=sem_2x_ref.at[sl],
            ).wait()

        @pl.when(step == 0)
        def _init_step0():
            if d_fast_dma:
                for s in range(num_slabs):
                    pltpu.make_async_copy(
                        src_ref=wunpool_hbm_ref.at[:, pl.ds(s * slab_w, slab_w)],
                        dst_ref=w_cache_ref.at[s],
                        sem=w_sem_ref.at[s],
                    ).start()
                _start_step(0, 0)
            else:
                _start_step(0, 0)
                for s in range(num_slabs):
                    pltpu.make_async_copy(
                        src_ref=wunpool_hbm_ref.at[:, pl.ds(s * slab_w, slab_w)],
                        dst_ref=w_cache_ref.at[s],
                        sem=w_sem_ref.at[s],
                    ).start()

        if not eff_salu_ovlp:
            @pl.when(step + 1 < num_steps)
            def _prefetch_next():
                _start_step(step + 1, 1 - slot)

        _wait_step(slot)

        # Phase 1: Zero-spill combine into u_scratch_ref [tok_tile_eff, 4096]
        amax_u_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        use_c16_half = (
            envs.SGLANG_TPU_MUSE_KERNEL_D_T64_C16.get()
            and tok_tile_eff % 32 == 0
        )
        if in_buf_is_fp8 and not eff_mxu_combine:
            _combine_fp8_phase1(buf_2x_ref, slot, w_vmem_ref, u_scratch_ref)
            if fp8_unpool and not use_blk_fp8_unpool:
                for c_blk in range(4):
                    u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                    amax_u_128 = jnp.maximum(
                        amax_u_128,
                        vpu_tree_max_128(
                            jnp.abs(u_scratch_ref[:, u0:u1].astype(jnp.float32))
                        ),
                    )
        elif eff_mxu_combine:
            c_grp = 8
            amax_groups = []
            for b_16 in range(tok_tile_eff // 16):
                r0, r1 = b_16 * 16, (b_16 + 1) * 16
                m0, m1 = r0 * topk, r1 * topk
                w_16x128 = w_vmem_ref[r0:r1, :]
                amax_b = jnp.zeros((16, LANE), dtype=jnp.float32)
                for c_blk in range(n2_cols // c_grp):
                    c0, c1 = c_blk * c_grp, (c_blk + 1) * c_grp
                    u0, u1 = c0 * lanes, c1 * lanes
                    vals_128x1024 = _load_mxu_tile_128x1024(
                        buf_2x_ref, slot, m0, m1, c0, c1, c_grp
                    )
                    u_sub_bf16 = jnp.dot(
                        w_16x128,
                        vals_128x1024,
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                    u_scratch_ref[r0:r1, u0:u1] = u_sub_bf16
                    if fp8_unpool and not use_blk_fp8_unpool:
                        amax_b = jnp.maximum(
                            amax_b,
                            vpu_tree_max_128(
                                jnp.abs(u_sub_bf16.astype(jnp.float32))
                            ),
                        )
                if fp8_unpool and not use_blk_fp8_unpool:
                    amax_groups.append(amax_b)
            if fp8_unpool and not use_blk_fp8_unpool:
                amax_u_128 = jnp.concatenate(amax_groups, axis=0)
        elif use_c16_half:
            w_tok = w_vmem_ref[:, :topk].reshape(tok_tile_eff, topk, 1, 1)
            for h_t in range(tok_tile_eff // 32):
                r0, r1 = h_t * 32, (h_t + 1) * 32
                m0, m1 = r0 * topk, r1 * topk
                w_h = w_tok[r0:r1]
                for c_blk in range(2):
                    c0, c1 = c_blk * 16, (c_blk + 1) * 16
                    vals_sub = buf_2x_ref[slot, m0:m1, c0:c1, :].reshape(
                        32, topk, 16, lanes
                    )
                    u_sub = jnp.sum(vals_sub * w_h, axis=1).reshape(32, 16 * lanes)
                    u_scratch_ref[r0:r1, c0 * lanes : c1 * lanes] = u_sub
            if fp8_unpool and not use_blk_fp8_unpool:
                for c_blk in range(4):
                    u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                    amax_u_128 = jnp.maximum(
                        amax_u_128,
                        vpu_tree_max_128(
                            jnp.abs(u_scratch_ref[:, u0:u1].astype(jnp.float32))
                        ),
                    )
        else:
            w_tok = w_vmem_ref[:, :topk].reshape(tok_tile_eff, topk, 1, 1)
            for c_blk in range(n2_cols // t64_c_step):
                c0, c1 = c_blk * t64_c_step, (c_blk + 1) * t64_c_step
                vals_sub = buf_2x_ref[slot, :, c0:c1, :].reshape(
                    tok_tile_eff, topk, t64_c_step, lanes
                )
                u_sub = jnp.sum(vals_sub * w_tok, axis=1).reshape(
                    tok_tile_eff, t64_c_step * lanes
                )
                u_scratch_ref[:, c0 * lanes : c1 * lanes] = u_sub
                if fp8_unpool:
                    amax_u_128 = jnp.maximum(
                        amax_u_128, vpu_tree_max_128(jnp.abs(u_sub.astype(jnp.float32)))
                    )

        if use_blk_fp8_unpool:
            for kb in range(num_fp8_kb):
                k0, k1 = kb * eff_fp8_qbs, (kb + 1) * eff_fp8_qbs
                u_blk_f32 = u_scratch_ref[:, k0:k1].astype(jnp.float32)
                b_amax_128 = vpu_tree_max_128(jnp.abs(u_blk_f32))
                row_amax_b = jnp.max(b_amax_128, axis=1, keepdims=True)
                sc_b = jnp.where(
                    row_amax_b == 0,
                    jnp.zeros_like(row_amax_b),
                    row_amax_b * jnp.float32(1.0 / 448.0),
                )
                inv_sc_b = jnp.where(
                    row_amax_b == 0,
                    jnp.zeros_like(row_amax_b),
                    jnp.float32(448.0) / jnp.maximum(row_amax_b, jnp.float32(1e-30)),
                )
                u_fp8_scratch_ref[:, k0:k1] = jnp.clip(
                    u_blk_f32 * inv_sc_b, -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
                u_sc_blk_ref[kb, :, :] = jnp.broadcast_to(sc_b, (tok_tile_eff, LANE))
            u_sc = None
        elif fp8_unpool:
            row_amax_u = jnp.max(amax_u_128, axis=1, keepdims=True)
            u_sc = row_amax_u * jnp.float32(1.0 / 448.0)
            u_sc = jnp.where(row_amax_u == 0, jnp.zeros_like(u_sc), u_sc)
            inv_u_sc = jnp.where(
                row_amax_u == 0,
                jnp.zeros_like(row_amax_u),
                jnp.float32(448.0) / jnp.maximum(row_amax_u, jnp.float32(1e-30)),
            )
            for c_blk in range(4):
                u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                u_fp8_scratch_ref[:, u0:u1] = jnp.clip(
                    u_scratch_ref[:, u0:u1].astype(jnp.float32) * inv_u_sc,
                    -448.0,
                    448.0,
                ).astype(jnp.float8_e4m3fn)
        else:
            u_sc = None

        # Phase 2: Pipelined 4-slab wunpool dot + post_feedforward_layernorm (scale_input=True) variance
        # Temporarily store fw_sub in new_stream_out_ref[:, gn0:gn1] to save 2-4 MiB VMEM!
        if eff_salu_ovlp and not d_fast_dma:
            @pl.when(step == 0)
            def _wait_w_all_k():
                for sw_i in range(num_slabs):
                    pltpu.make_async_copy(
                        src_ref=w_cache_ref.at[sw_i],
                        dst_ref=w_cache_ref.at[sw_i],
                        sem=w_sem_ref.at[sw_i],
                    ).wait()

        next_s_k = jnp.minimum(step + 1, num_steps - 1)
        t_per_slab_k = tok_tile_eff // num_slabs
        subs_per_slab = slab_w // t64_sub_n
        total_subs = num_slabs * subs_per_slab
        finer_so = d_fast_dma and eff_salu_ovlp and (tok_tile_eff % total_subs == 0)
        t_per_sub = (tok_tile_eff // total_subs) if finer_so else 0
        sq_ffn_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        if d_fuse_p34:
            s_aa_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            s_ab_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            s_bb_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
        for s in range(num_slabs):
            if not eff_salu_ovlp or d_fast_dma:
                @pl.when(step == 0)
                def _wait_w_slab(s=s):
                    pltpu.make_async_copy(
                        src_ref=w_cache_ref.at[s],
                        dst_ref=w_cache_ref.at[s],
                        sem=w_sem_ref.at[s],
                    ).wait()

            s_col0 = s * slab_w
            for sub in range(subs_per_slab):
                cn0, cn1 = sub * t64_sub_n, (sub + 1) * t64_sub_n
                gn0, gn1 = s_col0 + cn0, s_col0 + cn1
                w_sc_sub = vecs_ref[0:1, gn0:gn1]
                if use_blk_fp8_unpool:
                    acc = jnp.zeros((tok_tile_eff, t64_sub_n), dtype=jnp.float32)
                    for kb in range(num_fp8_kb):
                        k0, k1 = kb * eff_fp8_qbs, (kb + 1) * eff_fp8_qbs
                        acc = acc + (
                            jnp.dot(
                                u_fp8_scratch_ref[:, k0:k1],
                                w_cache_ref[s, k0:k1, cn0:cn1],
                                preferred_element_type=jnp.float32,
                            )
                            * u_sc_blk_ref[kb, :, 0:1]
                        )
                    sc_prod = w_sc_sub
                else:
                    acc = jnp.dot(
                        u_fp8_scratch_ref[:, :] if fp8_unpool else u_scratch_ref[:, :],
                        w_cache_ref[s, :, cn0:cn1],
                        preferred_element_type=jnp.float32,
                    )
                    sc_prod = (u_sc * w_sc_sub) if fp8_unpool else w_sc_sub
                # Exact Muse Spark rounding: ((u @ wunpool) * scale).to(bf16).to(f32)
                y_sub = (acc * sc_prod).astype(jnp.bfloat16).astype(jnp.float32)
                # post_feedforward_layernorm has scale_input=True: multiply by gain before variance!
                fw_sub = y_sub * vecs_ref[1:2, gn0:gn1]
                new_stream_out_ref[:, gn0:gn1] = fw_sub
                sq_ffn_128 = sq_ffn_128 + vpu_tree_sum_128(fw_sub * fw_sub)
                if d_fuse_p34:
                    a_smid = vecs_ref[2:3, gn0:gn1] * stream_mid_ref[:, gn0:gn1]
                    b_fw = vecs_ref[3:4, gn0:gn1] * fw_sub
                    s_aa_128 = s_aa_128 + vpu_tree_sum_128(a_smid * a_smid)
                    s_ab_128 = s_ab_128 + vpu_tree_sum_128(a_smid * b_fw)
                    s_bb_128 = s_bb_128 + vpu_tree_sum_128(b_fw * b_fw)
                if finer_so:
                    sub_idx = s * subs_per_slab + sub
                    _start_step_range(
                        next_s_k,
                        1 - slot,
                        sub_idx * t_per_sub,
                        (sub_idx + 1) * t_per_sub,
                    )
            if eff_salu_ovlp and not finer_so:
                _start_step_range(
                    next_s_k,
                    1 - slot,
                    s * t_per_slab_k,
                    (s + 1) * t_per_slab_k,
                )

        inv_rms_ffn = jax.lax.rsqrt(
            jnp.sum(sq_ffn_128, axis=1, keepdims=True) * jnp.float32(inv_h)
            + jnp.float32(post_norm_eps)
        )

        if d_fuse_p34:
            sum_aa = jnp.sum(s_aa_128, axis=1, keepdims=True)
            sum_ab = jnp.sum(s_ab_128, axis=1, keepdims=True)
            sum_bb = jnp.sum(s_bb_128, axis=1, keepdims=True)
            sq_next_row = jnp.maximum(
                jnp.float32(0.0),
                sum_aa
                + jnp.float32(2.0) * inv_rms_ffn * sum_ab
                + (inv_rms_ffn * inv_rms_ffn) * sum_bb,
            )
            inv_rms_next = jax.lax.rsqrt(
                sq_next_row * jnp.float32(inv_h) + jnp.float32(rms_eps)
            )
            inv_rms_next_eff = (
                inv_rms_next * jnp.float32(32.0)
                if emit_static_fp8_attn_in
                else inv_rms_next
            )
            out_dt = jnp.float8_e4m3fn if emit_static_fp8_attn_in else jnp.bfloat16
            p_step = 256
            for c0 in range(0, h_dim, p_step):
                c1 = c0 + p_step
                fw_s = new_stream_out_ref[:, c0:c1]
                branch_s = (
                    (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                )
                new_s = (
                    vecs_ref[2:3, c0:c1] * stream_mid_ref[:, c0:c1]
                    + vecs_ref[3:4, c0:c1] * branch_s
                )
                new_stream_out_ref[:, c0:c1] = new_s
                scaled_next = new_s * inv_rms_next_eff * vecs_ref[4:5, c0:c1]
                if emit_static_fp8_attn_in:
                    scaled_next = jnp.clip(scaled_next, -448.0, 448.0)
                next_in_out_ref[:, c0:c1] = scaled_next.astype(out_dt)
        else:
            # Phase 3: Gated residual + next_layer.input_layernorm variance (+ algebraic amax) in a single sweep
            p_step = 256 if d_fast_dma else slab_w
            sq_next_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            amax_next_128 = jnp.zeros((tok_tile_eff, LANE), dtype=jnp.float32)
            for c0 in range(0, h_dim, p_step):
                c1 = c0 + p_step
                fw_s = new_stream_out_ref[:, c0:c1]
                branch_s = (
                    (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                )
                new_s = (
                    vecs_ref[2:3, c0:c1] * stream_mid_ref[:, c0:c1]
                    + vecs_ref[3:4, c0:c1] * branch_s
                )
                new_stream_out_ref[:, c0:c1] = new_s
                sq_next_128 = sq_next_128 + vpu_tree_sum_128(new_s * new_s)
                if emit_fp8_attn_in:
                    unscaled_next = new_s * vecs_ref[4:5, c0:c1]
                    amax_next_128 = jnp.maximum(
                        amax_next_128, vpu_tree_max_128(jnp.abs(unscaled_next))
                    )

            inv_rms_next = jax.lax.rsqrt(
                jnp.sum(sq_next_128, axis=1, keepdims=True) * jnp.float32(inv_h)
                + jnp.float32(rms_eps)
            )

            if emit_fp8_attn_in:
                row_amax_unscaled = jnp.max(amax_next_128, axis=1, keepdims=True)
                inv_q_sc = jnp.where(
                    row_amax_unscaled == 0,
                    jnp.zeros_like(row_amax_unscaled),
                    jnp.float32(448.0)
                    / jnp.maximum(row_amax_unscaled, jnp.float32(1e-30)),
                )
                eff_row_sc = (row_amax_unscaled * inv_rms_next) * jnp.float32(
                    1.0 / 448.0
                )
                s_scaled = jnp.clip(eff_row_sc * jnp.float32(8.0), 0.0, 448.0)
                s_hi_f8 = s_scaled.astype(jnp.float8_e4m3fn)
                s_hi_f32 = s_hi_f8.astype(jnp.float32)
                s_lo_f8 = jnp.clip(
                    (s_scaled - s_hi_f32) * jnp.float32(16.0), -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
                lane = jax.lax.broadcasted_iota(jnp.int32, (tok_tile_eff, LANE), 1)
                next_in_out_ref[:, h_dim : h_dim + LANE] = jnp.where(
                    lane < 64,
                    jnp.broadcast_to(s_hi_f8, (tok_tile_eff, LANE)),
                    jnp.broadcast_to(s_lo_f8, (tok_tile_eff, LANE)),
                )
                for c0 in range(0, h_dim, p_step):
                    c1 = c0 + p_step
                    new_s = new_stream_out_ref[:, c0:c1]
                    next_in_out_ref[:, c0:c1] = jnp.clip(
                        new_s * vecs_ref[4:5, c0:c1] * inv_q_sc, -448.0, 448.0
                    ).astype(jnp.float8_e4m3fn)
            elif emit_static_fp8_attn_in:
                inv_rms_next_s32 = inv_rms_next * jnp.float32(32.0)
                for c0 in range(0, h_dim, p_step):
                    c1 = c0 + p_step
                    new_s = new_stream_out_ref[:, c0:c1]
                    next_in_out_ref[:, c0:c1] = jnp.clip(
                        new_s * inv_rms_next_s32 * vecs_ref[4:5, c0:c1],
                        -448.0,
                        448.0,
                    ).astype(jnp.float8_e4m3fn)
            else:
                for c0 in range(0, h_dim, p_step):
                    c1 = c0 + p_step
                    new_s = new_stream_out_ref[:, c0:c1]
                    next_in_out_ref[:, c0:c1] = (
                        new_s * inv_rms_next * vecs_ref[4:5, c0:c1]
                    ).astype(jnp.bfloat16)

        if eff_salu_ovlp:
            @pl.when(step == num_steps - 1)
            def _drain_extra_pf_k():
                _wait_step(1 - slot)

    u_fp8_rows = tok_tile_eff if fp8_unpool else 8
    kernel, extra_in, extra_spec, aliases = _d_4slab_kernel, (), [], {}
    if stream_out is not None:
        # Operand 6 (scalar prefetch + 5 inputs) is new_stream's aliased buffer; skip its ref.
        kernel = lambda *r: _d_4slab_kernel(*r[:6], *r[7:])  # noqa: E731
        extra_in, extra_spec, aliases = (
            (stream_out,),
            [pl.BlockSpec(memory_space=pltpu.HBM)],
            {6: 0},
        )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((chunk_size, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((num_rows, next_in_cols), next_in_dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_tile_eff, LANE), tok_map),
                pl.BlockSpec((tok_tile_eff, h_dim), tok_map),
                pl.BlockSpec((8, h_dim), lambda i, *_: (0, 0)),
                *extra_spec,
            ],
            out_specs=[
                pl.BlockSpec((tok_tile_eff, h_dim), tok_map),
                pl.BlockSpec((tok_tile_eff, next_in_cols), lambda i, *_: (i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, tok_tile_eff * topk, n2_cols, lanes), in_buf_dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((tok_tile_eff, he_dim), jnp.bfloat16),
                pltpu.VMEM((u_fp8_rows, he_dim), jnp.float8_e4m3fn),
                *(
                    [pltpu.VMEM((num_fp8_kb, tok_tile_eff, LANE), jnp.float32)]
                    if use_blk_fp8_unpool
                    else []
                ),
                pltpu.VMEM((num_slabs, he_dim, slab_w), jnp.float8_e4m3fn),
                pltpu.SemaphoreType.DMA((num_slabs,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=63 * 1024 * 1024,
        ),
        name=(
            f"pallas_d_combine_unpool_4slab_t{tok_tile_eff}"
            + (f"_c{t64_c_step}" if t64_c_step != 8 else "")
            + (
                "_c16h"
                if (
                    envs.SGLANG_TPU_MUSE_KERNEL_D_T64_C16.get()
                    and tok_tile_eff % 32 == 0
                )
                else ""
            )
            + ("_so" if (use_salu_overlap and tok_tile_eff % num_slabs == 0) else "")
            + (f"_sn{t64_sub_n}" if t64_sub_n != 512 else "")
            + (f"_fp8q{eff_fp8_qbs}" if eff_fp8_qbs > 0 else "")
            + (f"_do{eff_d_opt}" if eff_d_opt else "")
            + (("_mxuc" if scrub_mxu_nan else "_mxuc_ns") if eff_mxu_combine else "")
        ),
        input_output_aliases=aliases,
    )(
        inv_perm_c.reshape(-1),
        out_buf_3d,
        wunpool_fp8,
        w_pad,
        stream_shard_mid,
        vecs_8x8192,
        *extra_in,
    )


def _tc_combine_bf16_4chunk(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    *,
    tok_tile: int = 128,
) -> jax.Array:
    """Zero-spill 4-chunk _tc_combine_bf16 producing u_bf16 [chunk_size, 4096]."""
    chunk_size, topk = inv_perm_c.shape
    n2_cols, lanes = out_buf_3d.shape[1], out_buf_3d.shape[2]
    hidden_size = n2_cols * lanes
    tok_tile_eff = min(tok_tile, chunk_size)
    num_steps = chunk_size // tok_tile_eff
    w_pad = jnp.pad(local_topk_weights.astype(jnp.bfloat16), ((0, 0), (0, LANE - topk)))

    def _kernel(inv_perm_ref, src_hbm_ref, w_vmem_ref, out_ref, buf_2x_ref, sem_2x_ref):
        step = pl.program_id(0)

        def _start_step(s, slot):
            base_tok = s * tok_tile_eff
            for t in range(tok_tile_eff):
                tok = base_tok + t
                dst_base = t * topk
                idx_base = tok * topk
                idx8 = [inv_perm_ref[idx_base + k] for k in range(topk)]
                for k in range(topk):
                    pltpu.make_async_copy(
                        src_ref=src_hbm_ref.at[pl.ds(idx8[k], 1), :, :],
                        dst_ref=buf_2x_ref.at[slot, pl.ds(dst_base + k, 1), :, :],
                        sem=sem_2x_ref.at[slot],
                    ).start()

        def _wait_step(slot):
            pltpu.make_async_copy(
                src_ref=buf_2x_ref.at[slot],
                dst_ref=buf_2x_ref.at[slot],
                sem=sem_2x_ref.at[slot],
            ).wait()

        @pl.when(step == 0)
        def _():
            _start_step(0, 0)

        @pl.when(step + 1 < num_steps)
        def _():
            _start_step(step + 1, 1 - (step % 2))

        slot = step % 2
        _wait_step(slot)
        w_tok = w_vmem_ref[:, :topk].reshape(tok_tile_eff, topk, 1, 1)
        for c_blk in range(n2_cols // 8):
            c0, c1 = c_blk * 8, (c_blk + 1) * 8
            vals_sub = buf_2x_ref[slot, :, c0:c1, :].reshape(
                tok_tile_eff, topk, 8, lanes
            )
            out_ref[:, c0 * lanes : c1 * lanes] = jnp.sum(
                vals_sub * w_tok, axis=1
            ).reshape(tok_tile_eff, 8 * lanes)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((chunk_size, hidden_size), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_tile_eff, LANE), lambda i, *_: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tok_tile_eff, hidden_size), lambda i, *_: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, tok_tile_eff * topk, n2_cols, lanes), in_buf_dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=60 * 1024 * 1024,
        ),
        name=f"tc_combine_bf16_4chunk_t{tok_tile_eff}",
    )(inv_perm_c.reshape(-1), out_buf_3d, w_pad)


def pallas_combine_unpool_residual_nextnorm_nstream(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    stream_shard_mid: jax.Array,
    wunpool_fp8: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    m_pass: int = 128,
    bn: int = 1024,
    fp8_unpool: bool = False,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Design D2 (N-Streamed wunpool_fp8): 4-chunk combine -> LHS-resident N-streamed unpool+norms."""
    u_bf16 = _tc_combine_bf16_4chunk(out_buf_3d, inv_perm_c, local_topk_weights)
    t_rows, he_dim = u_bf16.shape
    w_k, h_dim = wunpool_fp8.shape
    assert he_dim == w_k == 4096 and h_dim == 8192
    in_buf_dtype = out_buf_3d.dtype
    in_buf_is_fp8 = jnp.dtype(in_buf_dtype) == jnp.float8_e4m3fn
    if in_buf_is_fp8:
        from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
            resolve_static_fp8_rs_scale,
        )

        _abs_sfp8, _ = resolve_static_fp8_rs_scale(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS.get()
        )
        if _abs_sfp8 > 0.0:
            local_topk_weights = (
                local_topk_weights.astype(jnp.float32) * jnp.float32(1.0 / _abs_sfp8)
            ).astype(local_topk_weights.dtype)
    m_tile = min(m_pass, t_rows)
    assert t_rows % m_tile == 0 and m_tile % LANE == 0

    vecs_8x8192 = prepare_d_static_inputs(
        wunpool_scale, post_ffn_norm_w, alpha_ffn, beta_ffn, next_input_norm_w
    )
    num_m = t_rows // m_tile
    num_n = h_dim // bn
    vecs_3d = vecs_8x8192.reshape(8, num_n, bn).swapaxes(0, 1)
    sub_m = LANE
    sub_n = 512
    inv_h = 1.0 / float(h_dim)

    def _unpool_nstream_kernel(
        u_in_ref,
        w_fp8_ref,
        stream_mid_ref,
        vecs_ref,
        new_stream_out_ref,
        next_in_out_ref,
        fw_scratch_ref,
        sq_scratch_ref,
        u_fp8_scratch_ref,
        u_sc_scratch_ref,
    ):
        n_idx = pl.program_id(1)

        @pl.when(n_idx == 0)
        def _init_sq():
            sq_scratch_ref[:, :] = jnp.zeros((m_tile, LANE), dtype=jnp.float32)
            if fp8_unpool:
                for rb in range(m_tile // sub_m):
                    r0, r1 = rb * sub_m, (rb + 1) * sub_m
                    amax_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                    for c_blk in range(4):
                        u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                        amax_128 = jnp.maximum(
                            amax_128,
                            vpu_tree_max_128(
                                jnp.abs(u_in_ref[r0:r1, u0:u1].astype(jnp.float32))
                            ),
                        )
                    row_amax_u = jnp.max(amax_128, axis=1, keepdims=True)
                    u_sc = jnp.where(
                        row_amax_u == 0,
                        jnp.zeros_like(row_amax_u),
                        row_amax_u * jnp.float32(1.0 / 448.0),
                    )
                    inv_u_sc = jnp.where(
                        row_amax_u == 0,
                        jnp.zeros_like(row_amax_u),
                        jnp.float32(448.0)
                        / jnp.maximum(row_amax_u, jnp.float32(1e-30)),
                    )
                    u_sc_scratch_ref[r0:r1, :] = jnp.broadcast_to(u_sc, (sub_m, LANE))
                    for c_blk in range(4):
                        u0, u1 = c_blk * 1024, (c_blk + 1) * 1024
                        u_fp8_scratch_ref[r0:r1, u0:u1] = jnp.clip(
                            u_in_ref[r0:r1, u0:u1].astype(jnp.float32) * inv_u_sc,
                            -448.0,
                            448.0,
                        ).astype(jnp.float8_e4m3fn)

        for rb in range(m_tile // sub_m):
            r0, r1 = rb * sub_m, (rb + 1) * sub_m
            sq_acc = sq_scratch_ref[r0:r1, :]
            lhs_r = u_fp8_scratch_ref[r0:r1, :] if fp8_unpool else u_in_ref[r0:r1, :]
            u_sc_r = u_sc_scratch_ref[r0:r1, 0:1] if fp8_unpool else None
            for nb in range(bn // sub_n):
                cn0, cn1 = nb * sub_n, (nb + 1) * sub_n
                acc = jnp.dot(
                    lhs_r,
                    w_fp8_ref[:, cn0:cn1],
                    preferred_element_type=jnp.float32,
                )
                w_sc_sub = vecs_ref[n_idx, 0:1, cn0:cn1]
                sc_prod = (u_sc_r * w_sc_sub) if fp8_unpool else w_sc_sub
                y_sub = (acc * sc_prod).astype(jnp.bfloat16).astype(jnp.float32)
                fw_sub = y_sub * vecs_ref[n_idx, 1:2, cn0:cn1]
                fw_scratch_ref[n_idx, r0:r1, cn0:cn1] = fw_sub
                sq_acc = sq_acc + vpu_tree_sum_128(fw_sub * fw_sub)
            sq_scratch_ref[r0:r1, :] = sq_acc

        @pl.when(n_idx == num_n - 1)
        def _epilogue():
            for rb in range(m_tile // sub_m):
                r0, r1 = rb * sub_m, (rb + 1) * sub_m
                inv_rms_ffn = jax.lax.rsqrt(
                    jnp.sum(sq_scratch_ref[r0:r1, :], axis=1, keepdims=True)
                    * jnp.float32(inv_h)
                    + jnp.float32(post_norm_eps)
                )
                sq_next_128 = jnp.zeros((sub_m, LANE), dtype=jnp.float32)
                for s in range(num_n):
                    c0, c1 = s * bn, (s + 1) * bn
                    fw_s = fw_scratch_ref[s, r0:r1, :]
                    branch_s = (
                        (fw_s * inv_rms_ffn).astype(jnp.bfloat16).astype(jnp.float32)
                    )
                    new_s = (
                        vecs_ref[s, 2:3, :] * stream_mid_ref[r0:r1, c0:c1]
                        + vecs_ref[s, 3:4, :] * branch_s
                    )
                    new_stream_out_ref[r0:r1, c0:c1] = new_s
                    sq_next_128 = sq_next_128 + vpu_tree_sum_128(new_s * new_s)

                inv_rms_next = jax.lax.rsqrt(
                    jnp.sum(sq_next_128, axis=1, keepdims=True) * jnp.float32(inv_h)
                    + jnp.float32(rms_eps)
                )
                for s in range(num_n):
                    c0, c1 = s * bn, (s + 1) * bn
                    new_s = new_stream_out_ref[r0:r1, c0:c1]
                    next_in_out_ref[r0:r1, c0:c1] = (
                        new_s * inv_rms_next * vecs_ref[s, 4:5, :]
                    ).astype(jnp.bfloat16)

    u_fp8_rows = m_tile if fp8_unpool else 8
    return pl.pallas_call(
        _unpool_nstream_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t_rows, h_dim), jnp.float32),
            jax.ShapeDtypeStruct((t_rows, h_dim), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_m, num_n),
            in_specs=[
                pl.BlockSpec((m_tile, he_dim), lambda m, n: (m, 0)),
                pl.BlockSpec((he_dim, bn), lambda m, n: (0, n)),
                pl.BlockSpec((m_tile, h_dim), lambda m, n: (m, 0)),
                pl.BlockSpec((num_n, 8, bn), lambda m, n: (0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((m_tile, h_dim), lambda m, n: (m, 0)),
                pl.BlockSpec((m_tile, h_dim), lambda m, n: (m, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((num_n, m_tile, bn), jnp.float32),
                pltpu.VMEM((m_tile, LANE), jnp.float32),
                pltpu.VMEM((u_fp8_rows, he_dim), jnp.float8_e4m3fn),
                pltpu.VMEM((u_fp8_rows, LANE), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
            vmem_limit_bytes=62 * 1024 * 1024,
        ),
        name=f"pallas_d_unpool_nstream_m{m_tile}_bn{bn}",
    )(u_bf16, wunpool_fp8, stream_shard_mid, vecs_3d)
