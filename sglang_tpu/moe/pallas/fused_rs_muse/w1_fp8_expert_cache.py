# SPDX-License-Identifier: Apache-2.0
"""Expert-level N-outer / M-inner FP8 W1+W2 cache (`GMM_EXPERT_OUTER_FP8=3, 4`).

Processes up to `R` (`4` or `6`, up to `1,536` token-expert rows) consecutive
`tile_m=256` tiles of the same expert in a single unified code path
(`batch_n in {1..R}`, zero single-tile fallback duplication):

- **Mode 3 (`GMM_EXPERT_OUTER_FP8=3`, Scaled-FP8)**:
  Folds per-512-K `bf16` block scales into `float8_e4m3fn` weights + per-column
  `float32` scale once per batch (`94.8%` MXU utilization, `16–18%` VALU).
- **Mode 4 (`GMM_EXPERT_OUTER_FP8=4`, 0-ULP Direct-FP8 — Bit-Exact with `sw4`)**:
  Unpacks `FP4 -> float8_e4m3fn` directly via hardware `vcvt` once per expert
  batch (`-66%` expand VALU bundles vs Mode 3, `0` `bf16` scale folding loss,
  100% 0-ULP IEEE-754 bit-exact parity with `sw4`), and hides per-512-K
  `rhs_s_bf16` multiplication inside the `82%` idle VALU shadow of the `94%`-busy
  MXU matmul loop!
"""

from __future__ import annotations

from typing import Any

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0


def expand_fp4_chunk_direct_fp8(
    w_weight_ref: Any,
    w_half_fp8_ref: Any,
    *,
    src_start_n: int,
    chunk_cols: int,
    strip_step: int,
    tile_k: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
) -> None:
    """Bit-exact `FP4 -> float8_e4m3fn` direct `vcvt` into `w_half_fp8_ref[0:tile_k, 0:chunk_cols]`.

    Uses `(rhs_qbs, strip_step)` chunks matching `scoped_w4a8_strip_acc` bitcast tiling
    (`<= 16` VREGs, 0 spills, ~1,033 LLO bundles per `4096x2048` half, 0-ULP parity).
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    k_step = rhs_qbs if (rhs_qbs <= 512 and tile_k % rhs_qbs == 0) else min(tile_k, 256)
    n_step = strip_step
    for c0 in range(0, chunk_cols, n_step):
        cw = min(n_step, chunk_cols - c0)
        w_n0 = src_start_n + c0
        for rk0 in range(0, tile_k, k_step):
            raw_u32 = w_weight_ref[
                pl.ds(rk0 // packing, k_step // packing), pl.ds(w_n0, cw)
            ]
            w_fp8 = (
                pltpu.bitcast(raw_u32, rhs_quant_dtype) if packing > 1 else raw_u32
            ).astype(jnp.float8_e4m3fn)
            w_half_fp8_ref[pl.ds(rk0, k_step), pl.ds(c0, cw)] = w_fp8


def dot_unrolled_chunk_direct_w4a8(
    lhs_fp8_ref: Any,
    lhs_sc_ref: Any,
    w_half_fp8_ref: Any,
    w_scale_ref: Any,
    *,
    src_start_n: int,
    c0: int,
    col_w: int,
    m_rows: int,
    tile_k: int,
    rhs_qbs: int,
    global_scale_val: Any,
    use_bf16_inner: bool = False,
):
    """0-ULP bit-exact `FP8 x pre-unpacked FP8 -> F32` matmul with hoisted `[num_blocks, col_w]` RHS scales."""
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    m_step = 128 if (m_rows >= 256 and m_rows % 128 == 0) else m_rows
    num_r_blocks = tile_k // rhs_qbs
    w_n0 = src_start_n + c0
    scale_is_2d = len(w_scale_ref.shape) == 2
    if num_r_blocks >= 8 and num_r_blocks % 8 == 0:
        hoisted_rs = (
            w_scale_ref[:, pl.ds(w_n0, col_w)].astype(jnp.bfloat16)
            if scale_is_2d
            else w_scale_ref[:, :, pl.ds(w_n0, col_w)]
            .reshape(num_r_blocks, col_w)
            .astype(jnp.bfloat16)
        )
    else:
        hoisted_rs = None

    halves = []
    for m0 in range(0, m_rows, m_step):
        sub_h = None
        for rk0 in range(0, tile_k, rhs_qbs):
            r_bid = rk0 // rhs_qbs
            if hoisted_rs is not None:
                rs_bf16 = hoisted_rs[r_bid : r_bid + 1, :]
            elif scale_is_2d:
                rs_bf16 = w_scale_ref[pl.ds(r_bid, 1), pl.ds(w_n0, col_w)].astype(
                    jnp.bfloat16
                )
            else:
                rs_bf16 = (
                    w_scale_ref[pl.ds(r_bid, 1), :, pl.ds(w_n0, col_w)]
                    .reshape(1, col_w)
                    .astype(jnp.bfloat16)
                )
            x_sub = lhs_fp8_ref[pl.ds(m0, m_step), pl.ds(rk0, rhs_qbs)]
            w_sub = w_half_fp8_ref[pl.ds(rk0, rhs_qbs), pl.ds(c0, col_w)]
            pop = jnp.matmul(x_sub, w_sub, preferred_element_type=jnp.float32)
            if use_bf16_inner:
                sp = pop.astype(jnp.bfloat16) * rs_bf16
            else:
                sp = pop * rs_bf16.astype(jnp.float32)
            sub_h = sp if sub_h is None else sub_h + sp
        s128 = lhs_sc_ref[pl.ds(m0, m_step), :].astype(jnp.float32)
        if col_w > s128.shape[1] and col_w % s128.shape[1] == 0:
            lhs_s = jnp.concatenate([s128] * (col_w // s128.shape[1]), axis=1)
        else:
            lhs_s = s128[:, :col_w]
        halves.append((sub_h.astype(jnp.float32) * lhs_s) * global_scale_val)
    return halves[0] if len(halves) == 1 else jnp.concatenate(halves, axis=0)


def expand_fp4_chunk_to_scaled_fp8(
    w_weight_ref: Any,
    w_scale_ref: Any,
    w_half_fp8_ref: Any,
    w_colc_8_ref: Any,
    *,
    src_start_n: int,
    chunk_cols: int,
    strip_step: int,
    tile_k: int,
    rhs_qbs: int,
    packing: int,
    rhs_quant_dtype: Any,
    global_scale_val: Any,
) -> None:
    """Expand `w_weight_ref[:, src_start_n : src_start_n + chunk_cols]` into
    `w_half_fp8_ref[0:tile_k, 0:chunk_cols]` (`float8_e4m3fn`) and sublane-8-aligned
    `w_colc_8_ref[0:8, 0:chunk_cols]` (`float32`), strip-by-strip (`<= 32` VREGs, 0 spills).
    """
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    num_blocks = tile_k // rhs_qbs
    scale_is_2d = len(w_scale_ref.shape) == 2
    k_chunk = 256 if rhs_qbs >= 256 else 64
    groups_per_chunk = k_chunk // rhs_qbs

    for c0 in range(0, chunk_cols, strip_step):
        col_w = min(strip_step, chunk_cols - c0)
        w_n0 = src_start_n + c0
        if scale_is_2d:
            s_all = w_scale_ref[:, pl.ds(w_n0, col_w)].astype(jnp.float32)
        else:
            s_all = (
                w_scale_ref[:, :, pl.ds(w_n0, col_w)]
                .reshape(num_blocks, col_w)
                .astype(jnp.float32)
            )
        col_max = jnp.max(jnp.abs(s_all), axis=0, keepdims=True)
        col_c = col_max * jnp.float32(_FP4_MAX / _FP8_E4M3_MAX)
        inv_col_c = jnp.where(
            col_c == 0.0,
            jnp.zeros_like(col_c),
            jnp.float32(1.0) / jnp.maximum(col_c, jnp.float32(1e-30)),
        )
        rel_s_bf16 = (s_all * inv_col_c).astype(jnp.bfloat16)
        w_colc_8_ref[:, pl.ds(c0, col_w)] = jnp.broadcast_to(
            col_c * global_scale_val, (8, col_w)
        )

        for rk0 in range(0, tile_k, k_chunk):
            raw_u32 = w_weight_ref[
                pl.ds(rk0 // packing, k_chunk // packing), pl.ds(w_n0, col_w)
            ]
            w_bf16 = (
                pltpu.bitcast(raw_u32, rhs_quant_dtype) if packing > 1 else raw_u32
            ).astype(jnp.bfloat16)
            b_base = rk0 // rhs_qbs
            if groups_per_chunk <= 1:
                w_scaled = w_bf16 * rel_s_bf16[b_base : b_base + 1, :]
            else:
                s_grp = rel_s_bf16[b_base : b_base + groups_per_chunk, :]
                w_scaled = (
                    w_bf16.reshape(groups_per_chunk, rhs_qbs, col_w) * s_grp[:, None, :]
                ).reshape(k_chunk, col_w)
            w_half_fp8_ref[pl.ds(rk0, k_chunk), pl.ds(c0, col_w)] = jnp.clip(
                w_scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX
            ).astype(jnp.float8_e4m3fn)


def dot_unrolled_chunk_fp8(
    lhs_fp8_ref: Any,
    lhs_sc_ref: Any,
    w_half_fp8_ref: Any,
    w_colc_8_ref: Any,
    *,
    c0: int,
    col_w: int,
    m_rows: int,
    tile_k: int,
    k_chunk: int = 512,
):
    """Zero-spill chunked `FP8 x FP8 -> F32` MXU matmul for one strip `c0:c0+col_w`."""
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    m_step = 128 if (m_rows >= 256 and m_rows % 128 == 0) else m_rows
    eff_k = k_chunk if (tile_k > k_chunk and tile_k % k_chunk == 0) else tile_k
    col_c_8 = w_colc_8_ref[:, pl.ds(c0, col_w)]
    halves = []
    for m0 in range(0, m_rows, m_step):
        acc_m = None
        for rk0 in range(0, tile_k, eff_k):
            x_sub = lhs_fp8_ref[pl.ds(m0, m_step), pl.ds(rk0, eff_k)]
            w_sub = w_half_fp8_ref[pl.ds(rk0, eff_k), pl.ds(c0, col_w)]
            p = jnp.matmul(x_sub, w_sub, preferred_element_type=jnp.float32)
            acc_m = p if acc_m is None else acc_m + p
        s128 = lhs_sc_ref[pl.ds(m0, m_step), :].astype(jnp.float32)
        if col_w > s128.shape[1] and col_w % s128.shape[1] == 0:
            lhs_s = jnp.concatenate([s128] * (col_w // s128.shape[1]), axis=1)
        else:
            lhs_s = s128[:, :col_w]
        scaled = (acc_m.reshape(m_step // 8, 8, col_w) * col_c_8[None, :, :]).reshape(
            m_step, col_w
        ) * lhs_s
        halves.append(scaled)
    return halves[0] if len(halves) == 1 else jnp.concatenate(halves, axis=0)


def compute_eo3_batch_count(
    gm_id: Any,
    local_num_gm: Any,
    metadata_ref: Any,
    max_batch: int,
):
    """Return `batch_n in [1, max_batch]` consecutive active tiles of `expert_id[gm_id]`."""
    import jax.numpy as jnp

    last_valid = jnp.maximum(local_num_gm - 1, 0)
    cur_e = metadata_ref.gm_id_to_group_id[jnp.minimum(gm_id, last_valid)]
    batch_n = jnp.int32(1)
    still_same = jnp.bool_(True)
    for d in range(1, max_batch):
        in_b = (gm_id + d) < local_num_gm
        e_d = metadata_ref.gm_id_to_group_id[jnp.minimum(gm_id + d, last_valid)]
        still_same = jnp.logical_and(still_same, jnp.logical_and(in_b, e_d == cur_e))
        batch_n = batch_n + still_same.astype(jnp.int32)
    return batch_n


def run_eo3_expert_batch(
    *,
    gm_id: Any,
    batch_n: Any,
    local_num_gm: Any,
    gm_id_ref: Any,
    metadata_ref: Any,
    fused_metadata_ref: Any,
    hidden_states_ref: Any,
    gathered_lhs_2x_ref: Any,
    lhs_indices_ref: Any,
    gather_sem_ref: Any,
    gather_divisor: int,
    packed_u16_lhs: bool,
    dma_gather_gm_start_fn: Any,
    dma_gather_gm_wait_fn: Any,
    start_w1_dma_fn: Any,
    wait_w1_dma_fn: Any,
    start_w2_dma_fn: Any,
    wait_w2_dma_fn: Any,
    wait_staging_slot_writes_fn: Any,
    do_direct_write_fn: Any,
    maybe_start_deferred_cm_wt_send_fn: Any,
    enable_chipmate: bool,
    cm_defer_send: bool,
    defer_writes: bool,
    w1_buf_ref: Any,
    w1_scale_buf_ref: Any,
    w2_buf_ref: Any,
    w2_scale_buf_ref: Any,
    w1_gs_gate_ref: Any,
    w1_gs_up_ref: Any,
    w2_gs_ref: Any,
    has_w1_gs: bool,
    has_w2_gs: bool,
    pnw_ref: Any,
    in_kernel_post_norm_size: int,
    static_fp8_rs_scale: float,
    scatter_staging_3x_ref: Any,
    eo_w_half_fp8_ref: Any,
    eo_w_colc_ref: Any,
    eo_lhs_fp8_4x_ref: Any,
    eo_lhs_sc_4x_ref: Any,
    eo_buf_4x_ref: Any,
    num_n1: int,
    num_n2: int,
    tile_m: int,
    tile_k1: int,
    tile_n1: int,
    tile_k2: int,
    tile_n2: int,
    strip_step: int,
    rhs1_qbs: int,
    rhs2_qbs: int,
    packing1: int,
    packing2: int,
    rhs1_quant_dtype: Any,
    rhs2_quant_dtype: Any,
    out_dtype: Any,
    direct_fp8_mode: bool = False,
    use_bf16_inner: bool = False,
    k_chunk: int = 512,
    has_ext_pq_lhs: bool = False,
    lhs_row_scales_ref: Any = None,
    pq_lhs_mode: int = 0,
) -> None:
    """Execute `batch_n` (`1..R`) consecutive tiles of `expert_id[gm_id]` in N-outer/M-inner order."""
    import jax
    import jax.experimental.pallas as pl
    import jax.numpy as jnp

    from .prefill_gmm_opts import (
        fill_pq_lhs_scales_tree,
        quantize_2d_lhs_tile_to_fp8,
        quantize_3d_lhs_tile_to_fp8,
        unpack_3d_pq_lhs_to_2d_fp8,
    )

    if num_n1 > w1_buf_ref.shape[0] or num_n2 > w2_buf_ref.shape[0]:
        raise ValueError(
            "run_eo3_expert_batch requires full-expert W1/W2 buffer caching: "
            f"num_n1={num_n1} > w1_bufs={w1_buf_ref.shape[0]} or "
            f"num_n2={num_n2} > w2_bufs={w2_buf_ref.shape[0]}"
        )

    gm_id_ref[0] = gm_id
    expert_id = metadata_ref.gm_id_to_group_id[gm_id]
    fused_metadata_ref.gm_id_to_group_id[0] = expert_id
    prev_gm = jnp.maximum(gm_id - 1, 0)
    prev_e = metadata_ref.gm_id_to_group_id[prev_gm]
    is_new_e = jnp.logical_or(gm_id == 0, prev_e != expert_id)

    @pl.when(gm_id == 0)
    def _prologue_gm0():
        if (
            enable_chipmate
            and cm_defer_send
            and maybe_start_deferred_cm_wt_send_fn is not None
        ):
            maybe_start_deferred_cm_wt_send_fn()
        dma_gather_gm_start_fn(
            hidden_states_ref,
            gathered_lhs_2x_ref.at[0],
            lhs_indices_ref,
            gather_sem_ref.at[0],
            0,
            metadata_ref,
            divisor=gather_divisor,
            packed_u16=packed_u16_lhs,
        )
        for _i in range(num_n1):
            start_w1_dma_fn(_i, expert_id, _i, 0)
        for _i in range(num_n2):
            start_w2_dma_fn(_i, expert_id, _i, 0)

    # 1. Gather + 1-pass lmax_128 FP8 quantize for all batch_n tiles (prefetching cur_g + 1).
    @pl.loop(0, batch_n)
    def _gather_and_quant_lhs(t):
        cur_g = gm_id + t
        cur_sem = cur_g % 2
        nxt_sem = 1 - cur_sem

        @pl.when(cur_g + 1 < local_num_gm)
        def _prefetch_next_gather():
            dma_gather_gm_start_fn(
                hidden_states_ref,
                gathered_lhs_2x_ref.at[nxt_sem],
                lhs_indices_ref,
                gather_sem_ref.at[nxt_sem],
                cur_g + 1,
                metadata_ref,
                divisor=gather_divisor,
                packed_u16=packed_u16_lhs,
            )

        dma_gather_gm_wait_fn(
            gathered_lhs_2x_ref.at[cur_sem],
            gather_sem_ref.at[cur_sem],
            cur_g,
            metadata_ref,
        )
        if has_ext_pq_lhs:
            _n_sl_eo = metadata_ref.gm_id_to_group_id.shape[0]
            m_st_t = metadata_ref.gm_id_to_m_offset[cur_g]
            m_en_t = (
                metadata_ref.gm_id_to_m_offset[_n_sl_eo + cur_g]
                if metadata_ref.gm_id_to_m_offset.shape[0] == 2 * _n_sl_eo
                else metadata_ref.gm_id_to_m_offset[cur_g + 1]
            )
            unpack_3d_pq_lhs_to_2d_fp8(
                gathered_lhs_2x_ref.at[cur_sem],
                eo_lhs_fp8_4x_ref.at[t],
                tile_m=tile_m,
                tile_k=tile_k1,
            )
            fill_pq_lhs_scales_tree(
                eo_lhs_sc_4x_ref,
                lhs_row_scales_ref,
                lhs_indices_ref,
                m_st=m_st_t,
                m_en=m_en_t,
                num_valid=m_en_t - m_st_t,
                num_lanes=eo_lhs_sc_4x_ref.shape[-1],
                packed_u16_lhs=packed_u16_lhs,
                gather_divisor=gather_divisor,
                scale_b_id=t,
                unroll_m_rows=(tile_m if pq_lhs_mode >= 3 else None),
            )
        else:
            quantize_3d_lhs_tile_to_fp8(
                gathered_lhs_2x_ref.at[cur_sem],
                eo_lhs_fp8_4x_ref.at[t],
                eo_lhs_sc_4x_ref.at[t],
                tile_m=tile_m,
                tile_k=tile_k1,
            )

    gm_after = gm_id + batch_n
    has_after = gm_after < local_num_gm
    after_clamped = jnp.minimum(gm_after, jnp.maximum(local_num_gm - 1, 0))
    after_e = metadata_ref.gm_id_to_group_id[after_clamped]

    gs_g_val = (
        w1_gs_gate_ref[expert_id].astype(jnp.float32) if has_w1_gs else jnp.float32(1.0)
    )
    gs_u_val = (
        w1_gs_up_ref[expert_id].astype(jnp.float32) if has_w1_gs else jnp.float32(1.0)
    )
    max_w_cols = eo_w_half_fp8_ref.shape[1]

    # 2. GMM1: N-outer (expand W1_gate / W1_up ONCE per batch), M-inner (@pl.loop(0, batch_n)).
    for _n1 in range(num_n1):

        @pl.when(is_new_e)
        def _wait_w1(_b=_n1):
            wait_w1_dma_fn(_b)

        for c_off in range(0, tile_n1, max_w_cols):
            chunk_w = min(max_w_cols, tile_n1 - c_off)

            if direct_fp8_mode:
                expand_fp4_chunk_direct_fp8(
                    w1_buf_ref.at[_n1],
                    eo_w_half_fp8_ref,
                    src_start_n=c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k1,
                    rhs_qbs=rhs1_qbs,
                    packing=packing1,
                    rhs_quant_dtype=rhs1_quant_dtype,
                )
            else:
                expand_fp4_chunk_to_scaled_fp8(
                    w1_buf_ref.at[_n1],
                    w1_scale_buf_ref.at[_n1],
                    eo_w_half_fp8_ref,
                    eo_w_colc_ref,
                    src_start_n=c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k1,
                    rhs_qbs=rhs1_qbs,
                    packing=packing1,
                    rhs_quant_dtype=rhs1_quant_dtype,
                    global_scale_val=gs_g_val,
                )

            @pl.loop(0, batch_n)
            def _gmm1_gate_batch(t, _n=_n1, _co=c_off, _cw=chunk_w):
                for s_n in range(0, _cw, strip_step):
                    col_w = min(strip_step, _cw - s_n)
                    out_col = pl.ds(_n * tile_n1 + _co + s_n, col_w)
                    if direct_fp8_mode:
                        gate_acc = dot_unrolled_chunk_direct_w4a8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            w1_scale_buf_ref.at[_n],
                            src_start_n=_co,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k1,
                            rhs_qbs=rhs1_qbs,
                            global_scale_val=gs_g_val,
                            use_bf16_inner=use_bf16_inner,
                        )
                    else:
                        gate_acc = dot_unrolled_chunk_fp8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            eo_w_colc_ref,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k1,
                            k_chunk=k_chunk,
                        )
                    eo_buf_4x_ref[t, :, out_col] = jax.nn.silu(gate_acc).astype(
                        out_dtype
                    )

            if direct_fp8_mode:
                expand_fp4_chunk_direct_fp8(
                    w1_buf_ref.at[_n1],
                    eo_w_half_fp8_ref,
                    src_start_n=tile_n1 + c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k1,
                    rhs_qbs=rhs1_qbs,
                    packing=packing1,
                    rhs_quant_dtype=rhs1_quant_dtype,
                )
            else:
                expand_fp4_chunk_to_scaled_fp8(
                    w1_buf_ref.at[_n1],
                    w1_scale_buf_ref.at[_n1],
                    eo_w_half_fp8_ref,
                    eo_w_colc_ref,
                    src_start_n=tile_n1 + c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k1,
                    rhs_qbs=rhs1_qbs,
                    packing=packing1,
                    rhs_quant_dtype=rhs1_quant_dtype,
                    global_scale_val=gs_u_val,
                )

            @pl.loop(0, batch_n)
            def _gmm1_up_batch(t, _n=_n1, _co=c_off, _cw=chunk_w):
                for s_n in range(0, _cw, strip_step):
                    col_w = min(strip_step, _cw - s_n)
                    out_col = pl.ds(_n * tile_n1 + _co + s_n, col_w)
                    if direct_fp8_mode:
                        up_acc = dot_unrolled_chunk_direct_w4a8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            w1_scale_buf_ref.at[_n],
                            src_start_n=tile_n1 + _co,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k1,
                            rhs_qbs=rhs1_qbs,
                            global_scale_val=gs_u_val,
                            use_bf16_inner=use_bf16_inner,
                        )
                    else:
                        up_acc = dot_unrolled_chunk_fp8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            eo_w_colc_ref,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k1,
                            k_chunk=k_chunk,
                        )
                    gate_f32 = eo_buf_4x_ref[t, :, out_col].astype(jnp.float32)
                    eo_buf_4x_ref[t, :, out_col] = (gate_f32 * up_acc).astype(out_dtype)

        @pl.when(jnp.logical_and(has_after, after_e != expert_id))
        def _prefetch_next_w1(_b=_n1):
            start_w1_dma_fn(_b, after_e, _b, 0)

    # 3. Interlude: quantize GMM1 output (bf16 -> fp8) for all batch_n tiles.
    @pl.loop(0, batch_n)
    def _quant_gmm2_lhs(t):
        quantize_2d_lhs_tile_to_fp8(
            eo_buf_4x_ref.at[t],
            eo_lhs_fp8_4x_ref.at[t],
            eo_lhs_sc_4x_ref.at[t],
            tile_m=tile_m,
            tile_k=tile_k2,
        )

    # 4. GMM2: N-outer (expand W2 ONCE per batch), M-inner (@pl.loop(0, batch_n)).
    gs_w2_val = (
        w2_gs_ref[expert_id].astype(jnp.float32) if has_w2_gs else jnp.float32(1.0)
    )
    pnw_full_bf16 = pnw_ref[...].astype(out_dtype) if pnw_ref is not None else None

    for _n2 in range(num_n2):

        @pl.when(is_new_e)
        def _wait_w2(_b=_n2):
            wait_w2_dma_fn(_b)

        n_tile_off = _n2 * tile_n2
        for c_off in range(0, tile_n2, max_w_cols):
            chunk_w = min(max_w_cols, tile_n2 - c_off)

            if direct_fp8_mode:
                expand_fp4_chunk_direct_fp8(
                    w2_buf_ref.at[_n2],
                    eo_w_half_fp8_ref,
                    src_start_n=c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k2,
                    rhs_qbs=rhs2_qbs,
                    packing=packing2,
                    rhs_quant_dtype=rhs2_quant_dtype,
                )
            else:
                expand_fp4_chunk_to_scaled_fp8(
                    w2_buf_ref.at[_n2],
                    w2_scale_buf_ref.at[_n2],
                    eo_w_half_fp8_ref,
                    eo_w_colc_ref,
                    src_start_n=c_off,
                    chunk_cols=chunk_w,
                    strip_step=strip_step,
                    tile_k=tile_k2,
                    rhs_qbs=rhs2_qbs,
                    packing=packing2,
                    rhs_quant_dtype=rhs2_quant_dtype,
                    global_scale_val=gs_w2_val,
                )

            @pl.loop(0, batch_n)
            def _gmm2_batch(t, _n=_n2, _nt_off=n_tile_off, _co=c_off, _cw=chunk_w):
                for s_n in range(0, _cw, strip_step):
                    col_w = min(strip_step, _cw - s_n)
                    abs_n0 = _nt_off + _co + s_n
                    out_col = pl.ds(abs_n0, col_w)
                    if direct_fp8_mode:
                        acc_n = dot_unrolled_chunk_direct_w4a8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            w2_scale_buf_ref.at[_n],
                            src_start_n=_co,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k2,
                            rhs_qbs=rhs2_qbs,
                            global_scale_val=gs_w2_val,
                            use_bf16_inner=use_bf16_inner,
                        )
                    else:
                        acc_n = dot_unrolled_chunk_fp8(
                            eo_lhs_fp8_4x_ref.at[t],
                            eo_lhs_sc_4x_ref.at[t],
                            eo_w_half_fp8_ref,
                            eo_w_colc_ref,
                            c0=s_n,
                            col_w=col_w,
                            m_rows=tile_m,
                            tile_k=tile_k2,
                            k_chunk=k_chunk,
                        )
                    out_bf16 = acc_n.astype(out_dtype)
                    if pnw_full_bf16 is not None:
                        pnw_s = pnw_full_bf16[:, abs_n0 : abs_n0 + col_w]
                        out_bf16 = (
                            out_bf16.reshape(tile_m // 8, 8, col_w) * pnw_s[None, :, :]
                        ).reshape(tile_m, col_w)
                    eo_buf_4x_ref[t, :, out_col] = out_bf16

        @pl.when(jnp.logical_and(has_after, after_e != expert_id))
        def _prefetch_next_w2(_b=_n2):
            start_w2_dma_fn(_b, after_e, _b, 0)

    # 5. Post-norm + static FP8 / BF16 staging + async ICI reduce-scatter for all batch_n tiles.
    @pl.loop(0, batch_n)
    def _flush_batch_rs(t):
        g_t = gm_id + t
        stg_t = g_t % 4
        if defer_writes:
            wait_staging_slot_writes_fn(stg_t)
        y_bf16 = eo_buf_4x_ref[t][...]
        if in_kernel_post_norm_size > 0:
            y_f32 = y_bf16.astype(jnp.float32)
            var = jnp.sum(y_f32 * y_f32, axis=1, keepdims=True) * (
                1.0 / in_kernel_post_norm_size
            )
            if static_fp8_rs_scale > 0.0:
                inv_rms_s = (
                    jax.lax.rsqrt(var + 1e-8) * jnp.float32(static_fp8_rs_scale)
                ).astype(out_dtype)
                raw_out = jnp.clip(y_bf16 * inv_rms_s, -448.0, 448.0).astype(
                    jnp.float8_e4m3fn
                )
            else:
                raw_out = y_bf16 * jax.lax.rsqrt(var + 1e-8).astype(out_dtype)
        else:
            raw_out = y_bf16
        scatter_staging_3x_ref[stg_t] = raw_out.reshape(
            tile_m,
            scatter_staging_3x_ref.shape[2],
            scatter_staging_3x_ref.shape[3],
        )
        do_direct_write_fn(g_t, stg_t)
