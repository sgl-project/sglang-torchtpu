# SPDX-License-Identifier: Apache-2.0

"""Fused GMM kernel with ICI direct-write reduce-scatter.

Each chip ICI-sends its GMM2 output directly to the correct (token, topk_index)
position in the destination chip's output buffer. No recv buffer, no scatter-add,
no topk weighting inside the kernel. The reduction (topk_weight × sum) happens
as a trivial post-kernel JAX operation.

Key design principles:
1. All writes via DMA engine (direct to final position)
2. Static Python loop unrolling for routing
3. All routing pre-computed at JAX level
4. No recv buffer, no scatter-add, no periodic barriers
5. Post-kernel weighted reduction in JAX
"""

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
from jax import lax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import envs
from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
    Dimensions,
    FusedDims,
    GmmConfigs,
    InputConfigs,
    MetadataRef,
    TileSizes,
    WeightsRef,
    align_to,
    calculate_tiling,
    dma_gather_gm_start,
    dma_gather_gm_wait,
    get_maybe_quantize_lhs,
    inner_kernel,
    prequantize_rhs_to_vmem,
)
from sglang_tpu.moe.pallas.fused_rs_muse.tuned_block_sizes import (
    get_fused_rs_tuned_block_sizes,
)
from sglang_tpu.moe.pallas.gmm_v2 import recover_quant_block_size

# Decode tiles of at most 8 valid rows run NVFP4 matmuls block-diagonally (see
# _fp4_blockdiag_gmm), in row branches 1, 2, 4 and 8. Calls above _BLOCKDIAG_MAX_SIZE_M routed
# rows (prefill) keep the BF16-dequant path.
_BLOCKDIAG_ROWS = (1, 2, 4, 8)
_BLOCKDIAG_MAX_SIZE_M = 4096
_BLOCKDIAG_KC = 256
_FP4_GROUP = 16
_FP4_PACK = 8
# SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES applies from here up; same bound as the
# row-bucket gate (no buckets at prefill M), so decode tiles are never touched.
_PREFILL_TILES_MIN_SIZE_M = 16384
_PREFILL_TM256_MIN_SIZE_M = 32768
_LOCAL_ROWS_ALIGN = 128


def fused_rs_smem_rows(size_m: int, ep_size: int, expert_tp: bool) -> int:
    """Return the local SMEM routing window row count for ``gmm_v2_fused_rs``."""
    if expert_tp:
        return size_m // ep_size + _LOCAL_ROWS_ALIGN
    win_override = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_SMEM_WINDOW_ROWS.get() or 0)
    if win_override > 0:
        aligned = (
            (win_override + _LOCAL_ROWS_ALIGN - 1) // _LOCAL_ROWS_ALIGN
        ) * _LOCAL_ROWS_ALIGN
        return min(size_m, aligned)
    u16_env = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS.get() or 0)
    force_u16 = u16_env >= 1
    pq_lhs_on = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get() or 0) >= 1
    auto_ep16 = ep_size >= 16 and size_m >= 131_072
    auto_pq_win = (u16_env >= 0) and pq_lhs_on and size_m >= 131_072
    if ep_size >= 4 and (
        size_m > 131_072 or auto_ep16 or auto_pq_win or (force_u16 and size_m > 120_000)
    ):
        floor_rows = (
            131_072
            if size_m >= 262_144
            else (98_304 if not (force_u16 or auto_ep16) else 65_536)
        )
        target = max(floor_rows, (size_m // ep_size) * 3)
        aligned = (
            (target + _LOCAL_ROWS_ALIGN - 1) // _LOCAL_ROWS_ALIGN
        ) * _LOCAL_ROWS_ALIGN
        return min(size_m, aligned)
    return size_m


def _quant_lhs_fp8_block(x_bf16: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Fast 2-stage lane-aligned online FP8 e4m3fn quantizer along axis=1."""
    m_rows, k_cols = x_bf16.shape
    if k_cols > 128 and k_cols % 128 == 0:
        lane_max = jnp.max(jnp.abs(x_bf16.reshape(m_rows, k_cols // 128, 128)), axis=1)
        amax_f32 = jnp.max(lane_max, axis=1, keepdims=True).astype(jnp.float32)
    else:
        amax_f32 = jnp.max(jnp.abs(x_bf16), axis=1, keepdims=True).astype(jnp.float32)
    inv_sc = jnp.where(
        amax_f32 == 0.0,
        0.0,
        jnp.float32(448.0) / jnp.maximum(amax_f32, jnp.float32(1e-30)),
    ).astype(jnp.bfloat16)
    x_fp8 = jnp.clip(x_bf16 * inv_sc, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    x_sc = jnp.where(amax_f32 == 0.0, 0.0, amax_f32 * jnp.float32(1.0 / 448.0))
    return x_fp8, x_sc


def _fp4_k256_gmm(
    x_ref,
    w_ref,
    scale_ref,
    k,
    n0,
    nc,
    block256,
    scale_block=_FP4_GROUP,
    lhs_fp8_mode: int = 0,
):
    """``x_ref [A, k]`` bf16 @ columns ``n0:n0+nc`` of a packed NVFP4 tile ``w_ref
    [k/8, n]`` u32 with block-16 scales ``scale_ref [k/16, n]``, f32 ``[A, nc]``, one dot
    per K256 chunk. ``n0`` may be traced.

    ``block256`` (scales constant over each K256 block): the e2m1 codes reach the MXU as
    FP8, which holds them exactly, and the chunk's partial product takes its first scale
    row in f32 (``scale_block`` 256: compact block-256 scales ``[k/256, n]``). Otherwise each code is dequantized to BF16 times its group-16 scale (the
    W4A16 strip's rounding) before a BF16 dot.
    """
    kc = _BLOCKDIAG_KC
    gpc = kc // _FP4_GROUP
    acc = None
    x_row_fp8 = None
    x_row_sc = None
    if block256 and lhs_fp8_mode >= k:
        x_row_fp8, x_row_sc = _quant_lhs_fp8_block(x_ref[:, :k].astype(jnp.bfloat16))
    for c0 in range(0, k, kc):
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)],
            jnp.float4_e2m1fn,
        )
        if block256 and lhs_fp8_mode >= k:
            res = jnp.dot(
                x_row_fp8[:, c0 : c0 + kc],
                w.astype(jnp.float8_e4m3fn),
                preferred_element_type=jnp.float32,
            )
            s = scale_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0, nc)].astype(
                jnp.float32
            )
            res = res * s
        elif block256 and lhs_fp8_mode == kc:
            x_q, x_s = _quant_lhs_fp8_block(x_ref[:, c0 : c0 + kc].astype(jnp.bfloat16))
            res = jnp.dot(
                x_q, w.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            s = scale_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0, nc)].astype(
                jnp.float32
            )
            res = (res * x_s) * s
        else:
            x = x_ref[:, c0 : c0 + kc].astype(jnp.bfloat16)
            if block256:
                res = jnp.dot(
                    x, w.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
                )
                s = scale_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0, nc)].astype(
                    jnp.float32
                )
                res = res * s
            else:
                s = scale_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0, nc)].astype(
                    jnp.bfloat16
                )
                w = (
                    w.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc) * s[:, None, :]
                ).reshape(kc, nc)
                res = jnp.dot(x, w, preferred_element_type=jnp.float32)
        acc = res if acc is None else acc + res
    if block256 and lhs_fp8_mode >= k:
        acc = acc * x_row_sc
    return acc


def _fp4_k256_dual_gmm(
    x_ref, w_ref, scale_ref, k, n0_g, n0_u, nc, block256, scale_block=_FP4_GROUP
):
    """Interleaved gate + up K256 dots loading ``x`` once per 256-row K block."""
    kc = _BLOCKDIAG_KC
    gpc = kc // _FP4_GROUP
    acc_g = None
    acc_u = None
    for c0 in range(0, k, kc):
        x = x_ref[:, c0 : c0 + kc].astype(jnp.bfloat16)
        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0_g, nc)],
            jnp.float4_e2m1fn,
        )
        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0_u, nc)],
            jnp.float4_e2m1fn,
        )
        if block256:
            rg = jnp.dot(
                x, wg.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            sg = scale_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0_g, nc)].astype(
                jnp.float32
            )
            rg = rg * sg
            ru = jnp.dot(
                x, wu.astype(jnp.float8_e4m3fn), preferred_element_type=jnp.float32
            )
            su = scale_ref[pl.ds(c0 // scale_block, 1), pl.ds(n0_u, nc)].astype(
                jnp.float32
            )
            ru = ru * su
        else:
            sg = scale_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_g, nc)].astype(
                jnp.bfloat16
            )
            wg_bf = (
                wg.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc) * sg[:, None, :]
            ).reshape(kc, nc)
            rg = jnp.dot(x, wg_bf, preferred_element_type=jnp.float32)
            su = scale_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0_u, nc)].astype(
                jnp.bfloat16
            )
            wu_bf = (
                wu.astype(jnp.bfloat16).reshape(gpc, _FP4_GROUP, nc) * su[:, None, :]
            ).reshape(kc, nc)
            ru = jnp.dot(x, wu_bf, preferred_element_type=jnp.float32)
        acc_g = rg if acc_g is None else acc_g + rg
        acc_u = ru if acc_u is None else acc_u + ru
    return acc_g, acc_u


def _fp4_blockdiag_gmm(xc, w_ref, scale_ref, k, n):
    """``xc [A, k]`` bf16 @ a packed NVFP4 block-16 tile ``w_ref [k/8, n]`` u32 with
    scales ``scale_ref [k/16, n]``, f32 ``[A, n]``.

    The e2m1 codes reach the MXU as FP8, which holds them exactly, and each K256
    chunk expands ``xc`` block-diagonally, so row ``a * 16 + g`` is row ``a`` masked
    to scale group ``g``. Each group's partial product lands in its own result row
    and its scale multiplies that f32 row, so no weight-side BF16 dequant runs.
    Ported from the v9 fused-RS compact path (``_fp4_blockdiag_gmm``).
    """
    A = xc.shape[0]
    kc = _BLOCKDIAG_KC
    gpc = kc // _FP4_GROUP
    rows = A * gpc
    # acc folds to [A * 8, nc] f32, nc // 128 VREGs per row.
    nc = min(n, 256 if A > 2 else 512)
    diag = (
        lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % gpc
        == lax.broadcasted_iota(jnp.int32, (rows, kc), 1) // _FP4_GROUP
    )

    def masked_x(c0):
        # One broadcast per row: broadcasting through a middle axis and
        # reshaping costs a VPU permute.
        xs = jnp.concatenate(
            [
                jnp.broadcast_to(xc[a : a + 1, c0 : c0 + kc], (gpc, kc))
                for a in range(A)
            ],
            axis=0,
        )
        return jnp.where(diag, xs, 0)

    # Keep the masked rows across the n loop only when K has few chunks.
    hoisted = {c0: masked_x(c0) for c0 in range(0, k, kc)} if k // kc <= 2 else None
    cols = []
    for n0 in range(0, n, nc):
        acc = None
        for c0 in range(0, k, kc):
            w = pltpu.bitcast(
                w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
            res = jnp.dot(
                hoisted[c0] if hoisted else masked_x(c0),
                w,
                preferred_element_type=jnp.float32,
            )
            s = scale_ref[pl.ds(c0 // _FP4_GROUP, gpc), pl.ds(n0, nc)].astype(
                jnp.float32
            )
            # Folds 16 group rows to 8 per input row, without a cross-sublane move.
            part = (
                (res.reshape(A, 2, 8, nc) * s.reshape(1, 2, 8, nc))
                .sum(axis=1)
                .reshape(A * 8, nc)
            )
            acc = part if acc is None else acc + part
        cols.append(
            jnp.concatenate(
                [
                    jnp.sum(acc[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
                    for a in range(A)
                ],
                axis=0,
            )
        )
    return jnp.concatenate(cols, axis=1)


def _pad_rows_to(x, rows):
    """x [A, n] with zero rows appended up to ``rows``."""
    if x.shape[0] == rows:
        return x
    return jnp.concatenate(
        [x, jnp.zeros((rows - x.shape[0], x.shape[1]), x.dtype)], axis=0
    )


# =============================================================================
# Phase 1: Pre-kernel metadata computation (JAX level)
# =============================================================================


def compute_num_gm(group_sizes, tile_m, size_lhs_sublane, start_offset=0):
    """Compute number of gm tiles from group sizes (JAX level).

    Mirrors fill_local_metadata logic including sublane alignment offset.
    """

    start_offset = jnp.asarray(start_offset, dtype=jnp.int32)

    def _per_group(carry, group_size):
        num_gm, m_offset = carry
        local_offset = m_offset % size_lhs_sublane
        aligned = group_size + local_offset
        curr_gm = jnp.where(group_size > 0, pl.cdiv(aligned, tile_m), 0)
        return (num_gm + curr_gm, m_offset + group_size), None

    (total_gm, _), _ = lax.scan(_per_group, (jnp.int32(0), start_offset), group_sizes)
    return total_gm


def compute_send_routing(output_indices, chunk_size):
    """Pre-compute per-row routing: dest_chip and local_row for every slot.

    Args:
        output_indices: int32[size_m] — scatter indices (global token IDs)
        chunk_size: int — num_tokens // ep_size (tokens per chip)

    Returns:
        send_dest_chips: int32[size_m] — which chip each row goes to
        send_local_rows: int32[size_m] — token position within dest chip's shard
    """
    send_dest_chips = output_indices // chunk_size
    send_local_rows = output_indices % chunk_size
    return send_dest_chips, send_local_rows


@dataclasses.dataclass(frozen=True)
class FusedRsBlockSizes:
    """Immutable set of block sizes + post-pad dimensions for fused_rs.

    All fields are static Python ints picked before any JAX tracing; they
    are safe to use as ``@jax.jit`` static args or in shape computations.
    """

    tile_m: int
    tile_k1: int
    tile_n1: int
    tile_k2: int
    tile_n2: int
    num_w1_bufs: int
    num_w2_bufs: int
    # Padded / aligned dimensions (needed by both sites for shape math).
    padded_k1: int
    aligned_n1: int  # new size_n1 after 2*num_lanes alignment
    padded_k2: int  # new size_k2 after aligned_n1 bump (== aligned_n1 // 2)
    aligned_n2: int  # new size_n2 after num_lanes*out_sls alignment


def _select_fused_rs_block_sizes(
    *,
    size_m: int,
    size_k1: int,
    size_n1: int,
    size_k2: int,
    size_n2: int,
    size_group: int,
    size_lhs_group: int,
    ep_size: int,
    out_dtype: jnp.dtype,
    w1_dtype: jnp.dtype,
    w2_dtype: jnp.dtype,
    is_quantized: bool,
    quant_block_size: int | None,
    act_fn: str | None,
    vmem_limit_bytes: int | None = None,
    fp8_direct_write: bool = False,
    hot_interleave: bool = False,
) -> FusedRsBlockSizes:
    """Deterministic selection of all tile/block sizes for fused_rs.

    Pure Python — no JAX ops, no random state. Same inputs always return
    the same ``FusedRsBlockSizes``. Safe to call from multiple entry
    points (``run_gmm_fused_rs`` and ``gmm_v2_fused_rs``) for the same
    shape/dtype combination.

    Pipeline:
      1. Pad K1 / N1 / N2 to DMA and MXU alignment, updating K2 when N1
         changes (since GMM2's K is GMM1's intermediate size).
      2. VMEM budget accounting → ``fused_vmem`` for ``calculate_tiling``.
      3. ``calculate_tiling`` for GMM1 and GMM2 to produce *default* tiles.
      4. ``get_fused_rs_tuned_block_sizes`` dict lookup (overrides default).
      5. Alignment post-processing of tile_k1, tile_n1, tile_k2, tile_n2
         so they exactly divide the padded / original dims the kernel
         iterates over.
      6. Final divisibility assertions.
    """
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.95)
    del ep_size  # Reserved for future per-chip heuristics; unused here.
    # --- Step 1: Pad dimensions ---
    num_lanes = pltpu.get_tpu_info().num_lanes
    sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)

    tile_k_unit = num_lanes * sls
    padded_k1 = align_to(size_k1, tile_k_unit)

    n1_unit = 2 * num_lanes
    aligned_n1 = align_to(size_n1, n1_unit)
    if aligned_n1 != size_n1:
        new_k2 = aligned_n1 // 2
    else:
        new_k2 = size_k2
    padded_k2 = new_k2

    out_sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
    aligned_n2 = align_to(size_n2, num_lanes * out_sls)

    # --- Step 2: VMEM budget ---
    fixed_vmem = (
        2 * 256 * size_k1 * jax.dtypes.itemsize_bits(out_dtype) // 8
        + 2 * 256 * aligned_n2 * jax.dtypes.itemsize_bits(out_dtype) // 8
        + 256 * (aligned_n1 // 2 if act_fn else aligned_n1) * 4
        + 256 * padded_k2 * jax.dtypes.itemsize_bits(out_dtype) // 8
        + 256
        * (aligned_n2 // num_lanes)
        * num_lanes
        * jax.dtypes.itemsize_bits(out_dtype)
        // 8
    )
    spill_budget = 6 * 1024 * 1024
    available_vmem = vmem_limit_bytes - fixed_vmem - spill_budget
    fused_vmem = max(available_vmem // 2, 1024 * 1024)

    size_lhs_sublane = min(sls, size_m)

    # --- Step 3: calculate_tiling for defaults ---
    dims1 = Dimensions(
        size_m=size_m,
        size_k=size_k1,
        size_n=aligned_n1,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )
    rhs1_cfgs = InputConfigs(
        quant_dtype=w1_dtype if is_quantized else None,
        quant_block_size=quant_block_size if quant_block_size else size_k1,
        dtype=w1_dtype,
    )
    tiles1 = calculate_tiling(
        dims1,
        InputConfigs(quant_dtype=None, quant_block_size=size_k1, dtype=out_dtype),
        rhs1_cfgs,
        fused_vmem,
    )

    dims2 = Dimensions(
        size_m=size_m,
        size_k=padded_k2,
        size_n=aligned_n2,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )
    rhs2_cfgs = InputConfigs(
        quant_dtype=w2_dtype if is_quantized else None,
        quant_block_size=quant_block_size if quant_block_size else padded_k2,
        dtype=w2_dtype,
    )
    tiles2 = calculate_tiling(
        dims2,
        InputConfigs(quant_dtype=None, quant_block_size=padded_k2, dtype=out_dtype),
        rhs2_cfgs,
        fused_vmem,
    )

    default_tiles = (
        tiles1.tile_m,
        tiles1.tile_k,
        tiles1.tile_n,
        tiles2.tile_k,
        tiles2.tile_n,
        2,
        2,
    )

    # --- Step 4: Tuned lookup ---
    tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = (
        get_fused_rs_tuned_block_sizes(
            size_m,
            size_k1,
            aligned_n1,
            padded_k2,
            aligned_n2,
            size_group,
            out_dtype,
            w1_dtype,
            quant_block_size if quant_block_size else None,
            default_tiles,
            fuse_act=act_fn,
            fp8_direct_write=fp8_direct_write,
        )
    )
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get()
        and is_quantized
        and not fp8_direct_write
        and (size_k1, aligned_n1, padded_k2, aligned_n2) == (4096, 8192, 4096, 4096)
        and 4096 <= size_m <= 16384
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)
    ):
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = (
            128,
            4096,
            2048,
            4096,
            2048,
            2,
            2,
        )
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META.get()
        and is_quantized
        and not fp8_direct_write
        and (size_k1, aligned_n1, padded_k2, aligned_n2) == (4096, 8192, 4096, 4096)
        and 32768 <= size_m < 131072
        and (tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs)
        == default_tiles
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)
    ):
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = (
            128,
            4096,
            2048,
            4096,
            2048,
            2,
            2,
        )
    # --- Step 4a: prefill tile override (env, FP4/FP8 experts, large M) ---
    if (
        hot_interleave
        and is_quantized
        and not fp8_direct_write
        and (size_k1, aligned_n1, padded_k2, aligned_n2) == (4096, 8192, 4096, 4096)
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)
    ):
        # §4.2b: 1024-col double-buffered non-hot ring (32.25 MiB) + 20 MiB dedicated
        # full-expert hot W1 bank = 52.25 MiB VMEM (< 60.8 MiB limit).
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = (
            128,
            4096,
            1024,
            4096,
            1024,
            2,
            2,
        )
    # Both the caller's max_num_gm sizing and the kernel call this helper, so they agree.
    prefill_tiles = envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES.get()
    if (
        prefill_tiles
        and is_quantized
        and not fp8_direct_write
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)  # fp4, or fp4 packed in int32
    ):
        parsed = tuple(int(x) for x in prefill_tiles.split(","))
        if len(parsed) != 7:
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES needs 7 ints, got "
                f"{prefill_tiles!r}"
            )
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = parsed
        if size_m < _PREFILL_TM256_MIN_SIZE_M and tile_m > 128:
            # At 16K <= M < 32K (<= 96 rows/expert avg across 256 experts), tile_m=256
            # gets almost zero 256-row weight-unpack reuse while inflating <=64-row
            # experts to >=128 rows (+12.8% dynamic bundles, +35.9% gather/RS rows).
            # Stepping tile_m down to 128 keeps full-N (4096, 4096) wla=2 ping-pong.
            tile_m = 128
    prefill_fp8_tiles = envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES.get()
    if (
        prefill_fp8_tiles
        and is_quantized
        and not fp8_direct_write
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jnp.dtype(w1_dtype) == jnp.float8_e4m3fn
    ):
        parsed = tuple(int(x) for x in prefill_fp8_tiles.split(","))
        if len(parsed) != 7:
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES needs 7 ints, got "
                f"{prefill_fp8_tiles!r}"
            )
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = parsed
        if size_m > 120_000 and tile_m < 256:
            # At M=131072 (T=16384), tile_m=128 exceeds 1 MiB SMEM when block_scatter=True, and
            # tile_m=256 with tile_n=2048 exceeds 62 MiB VMEM on FP8 weights; step down tile_n to 1024.
            tile_m = 256
            tile_n1 = min(tile_n1, 1024)
            tile_n2 = min(tile_n2, 1024)
    _eo_pref = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0)
    if (
        _eo_pref >= 3
        and is_quantized
        and not fp8_direct_write
        and not hot_interleave
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)
        and (size_k1, aligned_n1, padded_k2, aligned_n2) == (4096, 8192, 4096, 4096)
        and (
            not prefill_tiles
            or tile_n1 > 2048
            or tile_n2 > 2048
            or num_w1_bufs < 2
            or num_w2_bufs < 2
        )
    ):
        _eo_tm = (
            tile_m
            if (prefill_tiles and tile_m <= 128)
            else (256 if size_m >= _PREFILL_TM256_MIN_SIZE_M else 128)
        )
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = (
            _eo_tm,
            4096,
            2048,
            4096,
            2048,
            2,
            2,
        )
    if (
        _eo_pref < 3
        and int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD.get() or 0) == 1
        and is_quantized
        and not fp8_direct_write
        and not hot_interleave
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)
        and (size_k1, aligned_n1, padded_k2, aligned_n2) == (4096, 8192, 4096, 4096)
        and (tile_k1, tile_n1, tile_k2, tile_n2) == (4096, 2048, 4096, 2048)
        and (-(-tile_k1 // (quant_block_size or size_k1))) % 8 == 0
        and (-(-tile_k2 // (quant_block_size or padded_k2))) % 8 == 0
    ):
        num_w1_bufs, num_w2_bufs = 3, 3
    # --- Step 4a': decode tile override (env, FP4 experts, decode M) ---
    decode_tiles = envs.SGLANG_TPU_MUSE_KERNEL_DECODE_GMM_TILES.get()
    if (
        decode_tiles
        and is_quantized
        and not fp8_direct_write
        and envs.SGLANG_TPU_MUSE_KERNEL_K256_DECODE_TILES.get() <= size_m
        and size_m <= _BLOCKDIAG_MAX_SIZE_M
        and jax.dtypes.itemsize_bits(w1_dtype) in (4, 32)  # fp4, or fp4 packed in int32
    ):
        parsed = tuple(int(x) for x in decode_tiles.split(","))
        if len(parsed) != 7:
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_DECODE_GMM_TILES needs 7 ints, got "
                f"{decode_tiles!r}"
            )
        tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = parsed
    # --- Step 4b: bound the W1/W2 VMEM buffers by their real element width ---
    # calculate_tiling sizes the RHS tiles for the quantized width. Unquantized
    # (bf16) experts at a large hidden size can otherwise get W1/W2 buffers that
    # exceed VMEM. Halve tile_n (then tile_k) until each buffer set fits 16 MiB;
    # quantized tiles already fit.
    if not is_quantized:
        w_bytes = jax.dtypes.itemsize_bits(w1_dtype) // 8
        budget = 16 * 1024 * 1024
        n1_cols = lambda: tile_n1 * 2 if act_fn else tile_n1
        while (
            num_w1_bufs * tile_k1 * n1_cols() * w_bytes > budget and tile_n1 > num_lanes
        ):
            tile_n1 //= 2
        while (
            num_w1_bufs * tile_k1 * n1_cols() * w_bytes > budget
            and tile_k1 > 2 * num_lanes
        ):
            tile_k1 //= 2
        while (
            num_w2_bufs * tile_k2 * tile_n2 * w_bytes > budget and tile_n2 > num_lanes
        ):
            tile_n2 //= 2
    # --- Step 5: Alignment post-processing ---
    k_align1 = num_lanes
    if tile_k1 % k_align1 != 0:
        tile_k1 = (tile_k1 // k_align1) * k_align1 or k_align1
    if size_k1 % tile_k1 != 0:
        tile_k1 = size_k1

    out_n1 = aligned_n1 // 2 if act_fn else aligned_n1
    if out_n1 % tile_n1 != 0:
        while tile_n1 > num_lanes and out_n1 % tile_n1 != 0:
            tile_n1 -= num_lanes

    # Adjust tile_n2 to divide size_n2 (not aligned_n2). The kernel
    # iterates num_n2 = original_n2 // tile_n2 real N-tiles only.
    if size_n2 % tile_n2 != 0:
        mxu_cols = pltpu.get_tpu_info().mxu_column_size
        nn2 = pl.cdiv(size_n2, tile_n2)
        adj_tn2 = (size_n2 // nn2 // mxu_cols) * mxu_cols
        if adj_tn2 > 0:
            tile_n2 = adj_tn2
    if size_n2 % tile_n2 != 0:
        while tile_n2 > num_lanes and size_n2 % tile_n2 != 0:
            tile_n2 -= num_lanes

    if tile_k2 % num_lanes != 0:
        tile_k2 = (tile_k2 // num_lanes) * num_lanes or num_lanes
    if padded_k2 % tile_k2 != 0:
        tk2_adj = tile_k2
        while tk2_adj > num_lanes and padded_k2 % tk2_adj != 0:
            tk2_adj -= num_lanes
        tile_k2 = tk2_adj if (tk2_adj > 0 and padded_k2 % tk2_adj == 0) else padded_k2

    # NOTE: fp8 direct-write tile_m sizing is now handled in Step 4
    # (get_fused_rs_tuned_block_sizes, called with fp8_direct_write): dedicated
    # fp8-comm entries carry VMEM-safe tiles, and shapes without one fall back
    # to the bf16 tile clamped to tile_m<=64. Both the host
    # max-gm calc and the kernel call this helper, so their loop bounds stay
    # aligned. (Replaces the former unconditional ``tile_m = min(tile_m, 64)``.)

    # --- Step 6: Assertions ---
    assert size_k1 % tile_k1 == 0, f"tile_k1={tile_k1} must divide size_k1={size_k1}"
    assert padded_k2 % tile_k2 == 0, (
        f"tile_k2={tile_k2} must divide padded_k2={padded_k2}"
    )
    assert aligned_n1 % tile_n1 == 0, (
        f"tile_n1={tile_n1} must divide aligned_n1={aligned_n1}"
    )
    # tile_n2 must divide original size_n2 (kernel only iterates over real N-tiles).
    assert size_n2 % tile_n2 == 0, f"tile_n2={tile_n2} must divide size_n2={size_n2}"

    return FusedRsBlockSizes(
        tile_m=tile_m,
        tile_k1=tile_k1,
        tile_n1=tile_n1,
        tile_k2=tile_k2,
        tile_n2=tile_n2,
        num_w1_bufs=num_w1_bufs,
        num_w2_bufs=num_w2_bufs,
        padded_k1=padded_k1,
        aligned_n1=aligned_n1,
        padded_k2=padded_k2,
        aligned_n2=aligned_n2,
    )


# =============================================================================
# Phase 2: kernel_main_fused_rs — Core Kernel (Direct-Write ICI)
# =============================================================================


def _scale_hbm_tile(scale_ref, expert_id, k_blocks, n_cols):
    """One expert's [k_blocks, n_cols] scale tile in either HBM scale layout.

    Scales are [E, K_blocks, N] (compact) or [E, K_blocks, 1, N]. The compact
    layout matches the parameter's stored HBM layout, so XLA does not insert a
    relayout copy of the whole scale tensor before every kernel call.
    """
    if len(scale_ref.shape) == 3:
        return scale_ref.at[expert_id, k_blocks, n_cols]
    return scale_ref.at[expert_id, k_blocks, :, n_cols]


def _scale_vmem_cols(scale_buf_ref, buf_id, n_cols):
    if len(scale_buf_ref.shape) == 3:
        return scale_buf_ref.at[buf_id, :, n_cols]
    return scale_buf_ref.at[buf_id, :, :, n_cols]


def _scale_vmem_tile_cols(scale_tile_ref, n_cols):
    if scale_tile_ref is None:
        return None
    if len(scale_tile_ref.shape) == 2:
        return scale_tile_ref.at[:, n_cols]
    return scale_tile_ref.at[:, :, n_cols]


def _precompute_fused_rs_metadata(
    group_sizes: jax.Array,
    group_offset: jax.Array,
    group_start: jax.Array,
    lhs_indices: jax.Array,
    *,
    size_group: int,
    tile_m: int,
    sublane: int,
    chunk_size: int,
    num_slots: int,
    perm_gm: jax.Array | None = None,
    hot_slot: jax.Array | None = None,
) -> jax.Array:
    """Vectorized version of the kernel's scalar routing prologue and gm tiling.

    Returns (local_start, local_end, total_recv, local_num_gm,
    gm_id_to_group_id[S], gm_id_to_m_offset[S + 1]) with fill_local_metadata's
    tiling: a group's first tile starts at its first row, later tiles at
    sublane-aligned multiples of tile_m; gm_id_to_m_offset[num_gm] = local_end.
    """
    go = group_offset[0]
    local_start = group_start[0]
    sizes = lax.dynamic_slice_in_dim(group_sizes, go, size_group).astype(jnp.int32)
    starts = local_start + jnp.cumsum(sizes) - sizes
    local_end = local_start + jnp.sum(sizes)
    tiles = jnp.where(sizes > 0, (sizes + tile_m - 1) // tile_m, 0)
    tiles_end = jnp.cumsum(tiles)
    num_gm = tiles_end[-1]
    gm = jnp.arange(num_slots, dtype=jnp.int32)
    group = jnp.minimum(
        jnp.sum((tiles_end[None, :] <= gm[:, None]).astype(jnp.int32), axis=1),
        size_group - 1,
    )
    onehot = group[:, None] == jnp.arange(size_group, dtype=jnp.int32)[None, :]

    def _select(values):
        return jnp.sum(jnp.where(onehot, values[None, :], 0), axis=1)

    tile_in_group = gm - (_select(tiles_end) - _select(tiles))
    first = _select(starts)
    m_offset = first + tile_in_group * tile_m
    valid = gm < num_gm
    gm_group = jnp.where(valid, group, 0)
    if group_start.shape[0] >= 2:
        total_recv = group_start[1]
    else:
        rows = jnp.arange(lhs_indices.shape[0], dtype=jnp.int32)
        my_id = go // size_group
        is_local = jnp.logical_and(rows >= local_start, rows < local_end)
        total_recv = jnp.sum(
            jnp.logical_and(
                lhs_indices // chunk_size == my_id, jnp.logical_not(is_local)
            )
        )
    if perm_gm is not None:
        m_end_each = jnp.where(
            valid, jnp.minimum(m_offset + tile_m, first + _select(sizes)), local_end
        )
        m_st_each = jnp.where(valid, m_offset, local_end)
        gm_group = gm_group[perm_gm]
        gm_m_offset = jnp.concatenate([m_st_each[perm_gm], m_end_each[perm_gm]])
        hs = jnp.int32(-1) if hot_slot is None else hot_slot.astype(jnp.int32)
        head = jnp.stack([local_start, local_end, total_recv, num_gm, hs]).astype(
            jnp.int32
        )
        return jnp.concatenate([head, gm_group, gm_m_offset]).astype(jnp.int32)
    gm_m_offset = jnp.concatenate(
        [jnp.where(valid, m_offset, local_end), local_end[None]]
    )
    head = jnp.stack([local_start, local_end, total_recv, num_gm]).astype(jnp.int32)
    return jnp.concatenate([head, gm_group, gm_m_offset]).astype(jnp.int32)


def _precompute_block_scatter_meta_closed_form(
    group_sizes: jax.Array,
    group_offset: jax.Array,
    group_start_raw: jax.Array,
    *,
    size_group: int,
    tile_m: int,
    num_slots: int,
    ep_size: int,
    perm_gm: jax.Array | None = None,
    rotate_rs: bool = False,
    inter32_min_kf: int = 0,
    smem_base_offset: jax.Array | int = 0,
) -> jax.Array:
    go = group_offset[0]
    my_id = go // size_group
    local_start = group_start_raw[0] - jnp.asarray(smem_base_offset, dtype=jnp.int32)
    sizes = lax.dynamic_slice_in_dim(group_sizes, go, size_group).astype(jnp.int32)
    exp_starts = jnp.cumsum(sizes) - sizes
    tiles = jnp.where(sizes > 0, (sizes + tile_m - 1) // tile_m, 0)
    tiles_end = jnp.cumsum(tiles)
    num_gm = tiles_end[-1]
    gm = jnp.arange(num_slots, dtype=jnp.int32)
    group = jnp.minimum(
        jnp.sum((tiles_end[None, :] <= gm[:, None]).astype(jnp.int32), axis=1),
        size_group - 1,
    )
    onehot = group[:, None] == jnp.arange(size_group, dtype=jnp.int32)[None, :]

    def _select(values):
        return jnp.sum(jnp.where(onehot, values[None, :], 0), axis=1)

    tile_in_group = gm - (_select(tiles_end) - _select(tiles))
    valid = gm < num_gm

    chip_before = lax.dynamic_slice_in_dim(group_start_raw, 2, ep_size, axis=0)
    counts_flat = lax.dynamic_slice_in_dim(
        group_start_raw, 2 + ep_size, size_group * ep_size, axis=0
    )
    counts_gc = counts_flat.reshape(size_group, ep_size)
    expert_chip_base = chip_before[None, :] + jnp.cumsum(counts_gc, axis=0) - counts_gc
    _blk = tile_m // ep_size
    if inter32_min_kf > 0 and rotate_rs and (tile_m % ep_size == 0):
        c_iota = jnp.arange(ep_size, dtype=jnp.int32)
        rot_order = jnp.mod(my_id + 1 + c_iota, ep_size)
        inv_rot = jnp.mod(c_iota + ep_size - 1 - my_id, ep_size)
        raw_kf = jnp.min(counts_gc // jnp.int32(_blk), axis=1)
        k_full_g = jnp.where(
            raw_kf >= jnp.int32(inter32_min_kf), raw_kf, jnp.int32(0)
        )
        counts_rot_full = counts_gc[:, rot_order]
        rot_smem_base = (
            local_start
            + exp_starts[:, None]
            + (jnp.cumsum(counts_rot_full, axis=1) - counts_rot_full)
        )
        rem_cnt_gc = counts_gc - k_full_g[:, None] * jnp.int32(_blk)
        rem_rot = rem_cnt_gc[:, rot_order]
        intra_start = (jnp.cumsum(rem_rot, axis=1) - rem_rot)[:, inv_rot]
        combined_table = jnp.concatenate(
            [
                expert_chip_base,
                intra_start,
                rem_cnt_gc,
                rot_smem_base,
                k_full_g[:, None],
            ],
            axis=1,
        )
        sel_33 = jnp.sum(
            jnp.where(onehot[:, :, None], combined_table[None, :, :], 0), axis=1
        )
        base_gc = sel_33[:, :ep_size]
        ist_gc = sel_33[:, ep_size : 2 * ep_size]
        cnt_gc = sel_33[:, 2 * ep_size : 3 * ep_size]
        r_smem_gm = sel_33[:, 3 * ep_size : 4 * ep_size]
        kf_gm = sel_33[:, 4 * ep_size]
        is_inter = jnp.logical_and(valid, tile_in_group < kf_gm)
        is_tail = jnp.logical_and(
            valid, jnp.logical_and(kf_gm > 0, tile_in_group >= kf_gm)
        )
        rem_ti = tile_in_group - kf_gm
        rel_st = rem_ti * tile_m
        rel_en = jnp.minimum(rel_st + tile_m, _select(sizes) - kf_gm * tile_m)
        in_st = jnp.clip(rel_st[:, None] - ist_gc, 0, cnt_gc)
        in_en = jnp.clip(rel_en[:, None] - ist_gc, 0, cnt_gc)
        dst_tail = jnp.where(
            valid[:, None], base_gc + kf_gm[:, None] * jnp.int32(_blk) + in_st, 0
        )
        len_tail = jnp.where(valid[:, None], in_en - in_st, 0)
        len_rot = len_tail[:, rot_order]
        src_tail = (jnp.cumsum(len_rot, axis=1) - len_rot)[:, inv_rot]
        len_c = jnp.where(is_inter[:, None], jnp.int32(_blk), len_tail)
        dst_off = jnp.where(
            is_inter[:, None],
            base_gc + tile_in_group[:, None] * jnp.int32(_blk),
            dst_tail,
        )
        src_off = jnp.where(
            is_inter[:, None], inv_rot[None, :] * jnp.int32(_blk), src_tail
        )
        m_st_plain = local_start + _select(exp_starts) + tile_in_group * tile_m
        smem_plain = m_st_plain[:, None] + c_iota[None, :] * jnp.int32(_blk)
        smem_inter = r_smem_gm + tile_in_group[:, None] * jnp.int32(_blk)
        in_st_rot = in_st[:, rot_order]
        smem_tail = r_smem_gm + kf_gm[:, None] * jnp.int32(_blk) + in_st_rot
        tile_mode = jnp.where(
            is_inter, jnp.int32(1), jnp.where(is_tail, jnp.int32(2), jnp.int32(0))
        )
        smem_rot_8 = jnp.where(
            is_inter[:, None],
            smem_inter,
            jnp.where(is_tail[:, None], smem_tail, smem_plain),
        )
        aux_body = jnp.concatenate([tile_mode[:, None], smem_rot_8], axis=1)
    else:
        if rotate_rs:
            c_iota = jnp.arange(ep_size, dtype=jnp.int32)
            rot_order = jnp.mod(my_id + 1 + c_iota, ep_size)
            inv_rot = jnp.mod(c_iota + ep_size - 1 - my_id, ep_size)
            counts_rot = counts_gc[:, rot_order]
            intra_start = (jnp.cumsum(counts_rot, axis=1) - counts_rot)[:, inv_rot]
        else:
            intra_start = jnp.cumsum(counts_gc, axis=1) - counts_gc
        combined_table = jnp.concatenate(
            [expert_chip_base, intra_start, counts_gc], axis=1
        )
        sel_24 = jnp.sum(
            jnp.where(onehot[:, :, None], combined_table[None, :, :], 0), axis=1
        )
        base_gc = sel_24[:, :ep_size]
        ist_gc = sel_24[:, ep_size : 2 * ep_size]
        cnt_gc = sel_24[:, 2 * ep_size : 3 * ep_size]
        rel_st = tile_in_group * tile_m
        rel_en = jnp.minimum(rel_st + tile_m, _select(sizes))
        in_st = jnp.clip(rel_st[:, None] - ist_gc, 0, cnt_gc)
        in_en = jnp.clip(rel_en[:, None] - ist_gc, 0, cnt_gc)
        dst_off = jnp.where(valid[:, None], base_gc + in_st, 0)
        len_c = jnp.where(valid[:, None], in_en - in_st, 0)
        if rotate_rs:
            len_rot = len_c[:, rot_order]
            src_off = (jnp.cumsum(len_rot, axis=1) - len_rot)[:, inv_rot]
        else:
            src_off = jnp.cumsum(len_c, axis=1) - len_c
        aux_body = None
    has_c = (len_c > 0).astype(jnp.int32)
    total_blks_gm = jnp.sum(has_c, axis=1, keepdims=True)
    is_my_chip = jnp.arange(ep_size, dtype=jnp.int32)[None, :] == my_id
    local_blks_gm = jnp.sum(jnp.where(is_my_chip, has_c, 0), axis=1, keepdims=True)
    send_blks_gm = total_blks_gm - local_blks_gm
    extra_delta = jnp.sum(counts_gc, axis=0, keepdims=True) - jnp.sum(
        has_c, axis=0, keepdims=True
    )
    trailer = jnp.concatenate(
        [extra_delta, jnp.zeros((1, 2 * ep_size + 2), dtype=jnp.int32)], axis=1
    )
    body = jnp.concatenate(
        [src_off, dst_off, len_c, send_blks_gm, local_blks_gm], axis=1
    )
    if perm_gm is not None:
        body = body[perm_gm]
        if aux_body is not None:
            aux_body = aux_body[perm_gm]
    res_flat = jnp.concatenate([body, trailer], axis=0).astype(jnp.int32)
    if aux_body is not None:
        return jnp.concatenate(
            [res_flat.reshape(-1), aux_body.astype(jnp.int32).reshape(-1)], axis=0
        )
    return res_flat


def _precompute_block_scatter_meta(
    gm_m_offset: jax.Array,
    lhs_indices: jax.Array,
    *,
    chunk_size: int,
    ep_size: int,
    my_id: jax.Array,
    num_slots: int | None = None,
    rotate_rs: bool = False,
) -> jax.Array:
    dest_chip = lhs_indices.astype(jnp.int32) // chunk_size
    chip_ids = jnp.arange(ep_size, dtype=jnp.int32)[None, :]
    chip_hits = (dest_chip[:, None] == chip_ids).astype(jnp.int32)
    chip_prefix = jnp.concatenate(
        [jnp.zeros((1, ep_size), dtype=jnp.int32), jnp.cumsum(chip_hits, axis=0)],
        axis=0,
    )
    if num_slots is not None and gm_m_offset.shape[0] == 2 * num_slots:
        m_st = gm_m_offset[:num_slots]
        m_en = gm_m_offset[num_slots:]
    else:
        m_st = gm_m_offset[:-1]
        m_en = gm_m_offset[1:]
    dst_off = chip_prefix[m_st]
    len_c = chip_prefix[m_en] - dst_off
    if rotate_rs:
        c_iota = jnp.arange(ep_size, dtype=jnp.int32)
        rot_order = jnp.mod(my_id + 1 + c_iota, ep_size)
        inv_rot = jnp.mod(c_iota + ep_size - 1 - my_id, ep_size)
        len_rot = len_c[:, rot_order]
        src_off = (jnp.cumsum(len_rot, axis=1) - len_rot)[:, inv_rot]
    else:
        src_off = jnp.cumsum(len_c, axis=1) - len_c
    has_c = (len_c > 0).astype(jnp.int32)
    total_blks_gm = jnp.sum(has_c, axis=1, keepdims=True)
    is_my_chip = chip_ids == my_id
    local_blks_gm = jnp.sum(jnp.where(is_my_chip, has_c, 0), axis=1, keepdims=True)
    send_blks_gm = total_blks_gm - local_blks_gm
    extra_delta = jnp.sum(len_c - has_c, axis=0, keepdims=True)
    trailer = jnp.concatenate(
        [extra_delta, jnp.zeros((1, 2 * ep_size + 2), dtype=jnp.int32)], axis=1
    )
    body = jnp.concatenate(
        [src_off, dst_off, len_c, send_blks_gm, local_blks_gm], axis=1
    )
    return jnp.concatenate([body, trailer], axis=0).astype(jnp.int32)


def kernel_main_fused_rs(
    # Scalar prefetch (10) — `output_indices` was dropped (== lhs_indices for EP).
    # `topk_indices_ref` is included but conditionally used:
    #   - pack_indices=False: lhs_indices_ref holds raw lhs_idx values, and
    #     topk_indices_ref holds raw topk_slot values (both size_m).
    #   - pack_indices=True:  lhs_indices_ref holds packed values
    #     `combined = lhs_idx * top_k + topk_slot`, and topk_indices_ref is a
    #     1-element dummy (saves ~512 KB of SMEM at large prefill).
    lhs_group_sizes_ref,
    group_offset_ref,
    group_start_ref,
    lhs_indices_ref,
    topk_indices_ref,
    max_num_gm_ref,  # (1,) int32 — max gm tiles across all chips
    w1_gs_gate_ref,  # (E,) GMM1 gate global_scale
    w1_gs_up_ref,  # (E,) GMM1 up global_scale
    w2_gs_ref,  # (E,) GMM2 global_scale
    # In (7)
    hidden_states_ref,
    w1_ref,
    w2_ref,
    w1_scale_ref,
    w2_scale_ref,
    w1_bias_ref,
    w2_bias_ref,
    # Out (1)
    out_buf_ref,  # HBM: (chunk_size * top_k, N2//NL, NL) — output & ICI DMA target
    # Scratch — compute pipeline (same as kernel_main_fused)
    metadata_ref,
    fused_metadata_ref,
    gathered_lhs_2x_ref,
    gmm1_out_ref,
    intermediate_ref,
    tiled_out_2x_ref,
    scatter_staging_3x_ref,  # VMEM: (3, tile_m, N2//NL, NL) triple-buffered, first-dim indexed
    partial_out1_ref,
    partial_out2_ref,
    shared_acc_ref,
    gather_sem_ref,
    staging_sem_ref,
    zero_sem_ref,
    gm_id_ref,
    w1_buf_ref,
    w2_buf_ref,
    w1_scale_buf_ref,
    w2_scale_buf_ref,
    w1_bias_buf_ref,
    w2_bias_buf_ref,
    w1_sem_ref,
    w2_sem_ref,
    output_sem_ref,
    # Scratch — ICI direct-write
    send_sems_ref,  # DMA(3,) per-staging-slot send sems
    local_write_sems_ref,  # DMA(3,) per-staging-slot local write sems
    recv_sem_ref,  # DMA(1,) for incoming remote writes
    scale_send_sems_ref=None,  # Optional DMA(3,) for FP8 row-scale sends.
    scale_local_write_sems_ref=None,  # Optional DMA(3,) for FP8 row-scale local writes.
    scale_recv_sem_ref=None,  # Optional DMA(1,) for incoming FP8 row-scale writes.
    scatter_fp8_staging_3x_ref=None,  # Optional VMEM staging for FP8 payload rows.
    scatter_scale_3x_ref=None,  # Optional VMEM staging for per-row FP8 scales.
    out_scale_ref=None,  # Optional HBM output for per-row FP8 activation scales.
    *,
    fused_dims: FusedDims,
    tile_m: int,
    tile_k1: int,
    tile_k2: int,
    tile_n1: int,
    tile_n2: int,
    num_w1_bufs: int,
    num_w2_bufs: int,
    act_fn: str,
    out_dtype: jnp.dtype,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    ep_size: int,
    chunk_size: int,
    ep_axis_name: str,
    top_k: int,
    pack_indices: bool = True,
    fp8_direct_write: bool = False,
    defer_writes: bool = False,
    in_kernel_post_norm_size: int = 0,
    has_w1_gs: bool = False,
    has_w2_gs: bool = False,
    block_scatter: bool = False,
    vectorized_dequant: bool = False,
    blockdiag_rows: tuple[int, ...] = _BLOCKDIAG_ROWS,
    block256_scales: bool = False,
    k256_tiles: bool = False,
    decode_strip_nc: int = 0,
    decode_dual_k256: int = 0,
    cm_refs=None,
    chipmate_xor: int = 1,
    hot_refs=None,
    packed_u16_lhs: bool = False,
    gather_2ahead_mode: int = 0,
    use_weight_lookahead: bool = False,
    rotate_rs_mode: int = 0,
    w8_refs=None,
    static_fp8_rs_scale: float = 0.0,
    cm_defer_send: bool = False,
    num_cm_slots: int = 1,
    eo_refs=None,
    pq_lhs_mode: int = 0,
    inter32_enabled: bool = False,
    tile_idx_smem_ref=None,
    static_fp8_lhs: bool = False,
):
    """Fused gather + GMM1 + act + GMM2 + ICI direct-write kernel.

    `pack_indices`: when True (default), `lhs_indices_ref` holds packed values
    `combined = lhs_idx * top_k + topk_slot`; `topk_indices_ref` is a dummy.
    When False, the two refs hold raw values separately. Packing saves
    ~512 KB SMEM at the cost of ~80 us extra scalar-pipe work at small N.

    Pipeline per gm tile:
      Steps 1-9: gather → GMM1 → act → GMM2
      Step 10: Per-row direct-write to out_buf on dest chip
    """
    dims = fused_dims
    num_lanes = pltpu.get_tpu_info().num_lanes
    my_rank = group_offset_ref[0] // dims.size_group
    my_id = my_rank
    max_num_gm = max_num_gm_ref[0]

    # Derive the receive count directly from the same scalar-prefetched routing refs
    # used by metadata construction and direct writes.
    # group_start_ref is either (local_start,) or the metadata precomputed by
    # _precompute_fused_rs_metadata: (local_start, local_end, total_recv,
    # local_num_gm, gm_id_to_group_id[S], gm_id_to_m_offset[S + 1]).
    has_precomputed_counts = group_start_ref.shape[0] >= 3
    precomputed_meta = group_start_ref.shape[0] > 3
    with jax.named_scope("routing_scalar_prologue"):
        group_offset = group_offset_ref[0]
        local_start = group_start_ref[0]

        def _sum_local_group_size(local_group_id, count):
            return count + lhs_group_sizes_ref[group_offset + local_group_id]

        if has_precomputed_counts:
            local_end = group_start_ref[1]
        else:
            local_end = local_start + lax.fori_loop(
                0,
                dims.size_group,
                _sum_local_group_size,
                jnp.int32(0),
            )

        def _count_remote_receive(assignment_row, count):
            gathered_index = lhs_indices_ref[assignment_row]
            if pack_indices:
                gathered_index = gathered_index // top_k
            destination_chip = gathered_index // chunk_size
            row_is_local = jnp.logical_and(
                assignment_row >= local_start, assignment_row < local_end
            )
            is_remote_receive = jnp.logical_and(
                destination_chip == my_id, jnp.logical_not(row_is_local)
            )
            return count + is_remote_receive.astype(jnp.int32)

        if has_precomputed_counts:
            total_recv = group_start_ref[2]
        else:
            total_recv = lax.fori_loop(
                0,
                dims.size_m,
                _count_remote_receive,
                jnp.int32(0),
            )

    @jax.named_scope("sync_barrier_signal")
    def sync_barrier_signal():
        barrier_sem = pltpu.get_barrier_semaphore()
        if inter32_enabled:
            lax.fori_loop(
                0,
                ep_size,
                lambda _i, _u: (
                    pl.semaphore_signal(
                        barrier_sem,
                        device_id={ep_axis_name: _i},
                        device_id_type=pl.DeviceIdType.MESH,
                    ),
                    _u,
                )[1],
                jnp.int32(0),
            )
            return
        for i in range(ep_size):
            pl.semaphore_signal(
                barrier_sem,
                device_id={ep_axis_name: jnp.int32(i)},
                device_id_type=pl.DeviceIdType.MESH,
            )

    @jax.named_scope("sync_barrier_wait")
    def sync_barrier_wait():
        barrier_sem = pltpu.get_barrier_semaphore()
        pl.semaphore_wait(barrier_sem, ep_size)

    def sync_barrier():
        sync_barrier_signal()
        sync_barrier_wait()

    # Zero-init the output buffer (which is ALSO the ICI DMA landing pad for
    # peers) and WAIT for it BEFORE signaling the entry barrier. A peer only
    # issues its remote writes after its barrier releases, guaranteeing
    # no remote write lands in an out_buf that is still being zeroed.
    # Every (local_row, topk_slot) in out_buf_ref is overwritten via direct DMA;
    # keep entry sync_barrier() (~2 us) and skip 67.1 MB zero_out_start_3d.
    _split_entry_barrier = gather_2ahead_mode in (4, 5, 6)
    if _split_entry_barrier:
        sync_barrier_signal()
    else:
        sync_barrier()

    # Build metadata.
    meta_dims = Dimensions(
        size_m=dims.size_m,
        size_k=dims.size_k1,
        size_n=dims.size_n1,
        size_group=dims.size_group,
        size_lhs_group=dims.size_lhs_group,
        size_lhs_sublane=dims.size_lhs_sublane,
    )
    meta_cfgs = GmmConfigs(
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k1, tile_n=dims.size_n1),
        dims=meta_dims,
        lhs_cfgs=InputConfigs(
            quant_dtype=None, quant_block_size=dims.size_k1, dtype=out_dtype
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=None, quant_block_size=dims.size_k1, dtype=out_dtype
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
    )

    enable_chipmate = cm_refs is not None
    hot_interleave = hot_refs is not None
    eff_cm_slots = num_cm_slots if enable_chipmate else 0
    if enable_chipmate:
        (
            cm_w1_pad,
            cm_s1_pad,
            cm_w2_pad,
            cm_s2_pad,
            cm_wt_send_sem_ref,
            cm_wt_recv_sem_ref,
            cm_awaited_ref,
        ) = cm_refs
        borrowed_slot = jnp.int32(dims.size_group - eff_cm_slots)
    else:
        cm_w1_pad = cm_s1_pad = cm_w2_pad = cm_s2_pad = None
        cm_wt_send_sem_ref = cm_wt_recv_sem_ref = cm_awaited_ref = None
        borrowed_slot = None
    eo3_enabled = False
    eo_w_half_fp8_ref = eo_w_colc_ref = eo_lhs_fp8_4x_ref = eo_lhs_sc_4x_ref = (
        eo_buf_4x_ref
    ) = None
    if eo_refs is not None:
        if len(eo_refs) == 5:
            (
                eo_w_half_fp8_ref,
                eo_w_colc_ref,
                eo_lhs_fp8_4x_ref,
                eo_lhs_sc_4x_ref,
                eo_buf_4x_ref,
            ) = eo_refs
            eo3_enabled = True
            eo_pair_enabled = False
            eo_w_strip_fp8_ref = eo_lhs_fp8_2x_ref = eo_lhs_sc_2x_ref = (
                eo_gmm1_2x_ref
            ) = None
        else:
            eo_w_strip_fp8_ref = eo_refs[0]
            if len(eo_refs) >= 4:
                eo_lhs_fp8_2x_ref, eo_lhs_sc_2x_ref, eo_gmm1_2x_ref = eo_refs[1:4]
                eo_pair_enabled = True
            else:
                eo_lhs_fp8_2x_ref = eo_lhs_sc_2x_ref = eo_gmm1_2x_ref = None
                eo_pair_enabled = False
    else:
        eo_w_strip_fp8_ref = eo_lhs_fp8_2x_ref = eo_lhs_sc_2x_ref = eo_gmm1_2x_ref = (
            None
        )
        eo_pair_enabled = False
    if hot_interleave:
        (
            hot_w1_buf_ref,
            hot_w1_scale_buf_ref,
            hot_w1_sem_ref,
            hot_w1_loaded_ref,
        ) = hot_refs
    else:
        hot_w1_buf_ref = hot_w1_scale_buf_ref = None
        hot_w1_sem_ref = hot_w1_loaded_ref = None

    @jax.named_scope("copy_local_metadata")
    def copy_local_metadata():
        num_slots = metadata_ref.gm_id_to_group_id.shape[0]
        num_gm = group_start_ref[3]
        if hot_interleave:

            def _copy_hi(gm, carry):
                metadata_ref.gm_id_to_group_id[gm] = group_start_ref[5 + gm]
                metadata_ref.gm_id_to_m_offset[gm] = group_start_ref[5 + num_slots + gm]
                metadata_ref.gm_id_to_m_offset[num_slots + gm] = group_start_ref[
                    5 + 2 * num_slots + gm
                ]
                return carry

            lax.fori_loop(0, num_gm, _copy_hi, jnp.int32(0))
            return num_gm

        def _copy(gm, carry):
            metadata_ref.gm_id_to_group_id[gm] = group_start_ref[4 + gm]
            metadata_ref.gm_id_to_m_offset[gm] = group_start_ref[4 + num_slots + gm]
            return carry

        lax.fori_loop(0, num_gm, _copy, jnp.int32(0))
        metadata_ref.gm_id_to_m_offset[num_gm] = group_start_ref[4 + num_slots + num_gm]
        return num_gm

    @jax.named_scope("fill_local_metadata")
    def fill_local_metadata():
        group_offset = group_offset_ref[0]
        metadata_ref.gm_id_to_m_offset[0] = group_start_ref[0]

        @jax.named_scope("inner_tm_loop")
        def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
            local_offset = curr_m_offset % meta_cfgs.dims.size_lhs_sublane
            tm_size = jnp.minimum(
                meta_cfgs.tiles.tile_m - local_offset,
                end_m_offset - curr_m_offset,
            )

            metadata_ref.gm_id_to_group_id[tm_id] = group_id

            next_m_offset = curr_m_offset + tm_size
            metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
            metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset
            return next_m_offset

        @jax.named_scope("outer_local_group_loop")
        def outer_local_group_loop(local_group_id, carry):
            num_gm, start_m_offset = carry
            lhs_group_id = group_offset + local_group_id
            group_size = lhs_group_sizes_ref[lhs_group_id]
            end_m_offset = start_m_offset + group_size

            local_offset = start_m_offset % meta_cfgs.dims.size_lhs_sublane
            aligned_group_size = group_size + local_offset
            curr_num_gm = pl.cdiv(aligned_group_size, meta_cfgs.tiles.tile_m)
            curr_num_gm = jnp.where(group_size > 0, curr_num_gm, 0)
            next_num_gm = num_gm + curr_num_gm

            tm_loop_fn = functools.partial(
                inner_tm_loop,
                end_m_offset=end_m_offset,
                group_id=local_group_id,
            )
            lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

            return next_num_gm, end_m_offset

        num_gm, _ = lax.fori_loop(
            0,
            meta_cfgs.dims.size_group,
            outer_local_group_loop,
            (0, group_start_ref[0]),
        )
        return num_gm

    if precomputed_meta:
        local_num_gm = copy_local_metadata()
    else:
        local_num_gm = fill_local_metadata()

    # (out_buf was zero-initialized and awaited above, before the entry-barrier
    # signal, so incoming ICI writes can never race the zeroing.)

    if cfgs1.rhs_cfgs.quant_dtype is not None and w1_ref.dtype != jnp.uint32:
        w1_packed = w1_ref.bitcast(jnp.uint32)
    else:
        w1_packed = w1_ref
    if cfgs2.rhs_cfgs.quant_dtype is not None and w2_ref.dtype != jnp.uint32:
        w2_packed = w2_ref.bitcast(jnp.uint32)
    else:
        w2_packed = w2_ref
    if w1_ref.ndim == 4:
        # Stacked layers (Muse megakernel expert groups): group_offset[1] is the layer.
        _layer_idx = group_offset_ref[1]
        w1_packed = w1_packed.at[_layer_idx]
        w2_packed = w2_packed.at[_layer_idx]
        if w1_scale_ref is not None:
            w1_scale_ref = w1_scale_ref.at[_layer_idx]
        if w2_scale_ref is not None:
            w2_scale_ref = w2_scale_ref.at[_layer_idx]

    num_k1 = dims.size_k1 // tile_k1
    num_k2 = dims.size_k2 // tile_k2
    out_n1 = dims.size_n1 // 2 if act_fn else dims.size_n1
    num_n1 = out_n1 // tile_n1
    # Use original (unpadded) N for GMM2 loop count, matching the DMA
    # gmm_v2 path which uses cfgs.out_size_n (not aligned_n).
    # This avoids iterating over padding N-tiles which can cause
    # incorrect results with fp8 weights.
    num_n2 = dims.original_n2 // tile_n2
    packing = cfgs1.rhs_cfgs.packing
    pk1 = tile_k1 // packing
    pk2 = tile_k2 // packing
    has_scale = cfgs1.rhs_cfgs.has_scale
    has_bias = cfgs1.rhs_cfgs.has_bias
    nqb1 = cfgs1.num_quant_blocks_per_tile_k if has_scale else 0
    nqb2 = cfgs2.num_quant_blocks_per_tile_k if has_scale else 0
    w1_dma_n = tile_n1 * 2 if act_fn else tile_n1

    def _scale_pad_tile(pad_ref, k_blocks, n_cols):
        if len(pad_ref.shape) == 2:
            return pad_ref.at[k_blocks, n_cols]
        return pad_ref.at[k_blocks, :, n_cols]

    def _cm_pad_view(pad_ref, slot_idx: int):
        return pad_ref if eff_cm_slots == 1 else pad_ref.at[slot_idx]

    _split_cm_wait = gather_2ahead_mode == 6
    if enable_chipmate:
        _cm_flag_slots = (2 if _split_cm_wait else 1) * eff_cm_slots
        for _ci in range(_cm_flag_slots):
            cm_awaited_ref[_ci] = jnp.int32(0)
        if cm_defer_send:
            cm_awaited_ref[_cm_flag_slots] = jnp.int32(0)
        _cm_slot_xors = (
            (1, 2, 4)
            if (eff_cm_slots == 3 and ep_size >= 8 and (ep_size & (ep_size - 1)) == 0)
            else tuple([chipmate_xor] * max(1, eff_cm_slots))
        )

        def _issue_all_cm_wt_sends():
            for _ci in range(eff_cm_slots):
                _cm_partner_ci = lax.bitwise_xor(my_id, jnp.int32(_cm_slot_xors[_ci]))
                _s_slot = max_num_gm_ref[1 + 2 * _ci]
                _sb = 4 * _ci
                _w1_dst = _cm_pad_view(cm_w1_pad, _ci)
                _s1_dst = _cm_pad_view(cm_s1_pad, _ci)
                _w2_dst = _cm_pad_view(cm_w2_pad, _ci)
                _s2_dst = _cm_pad_view(cm_s2_pad, _ci)

                @pl.when(_s_slot >= 0)
                def _start_one_cm_send(
                    _ss=_s_slot,
                    _b=_sb,
                    _dw1=_w1_dst,
                    _ds1=_s1_dst,
                    _dw2=_w2_dst,
                    _ds2=_s2_dst,
                    _pt=_cm_partner_ci,
                ):
                    pltpu.make_async_remote_copy(
                        src_ref=w1_packed.at[_ss],
                        dst_ref=_dw1,
                        send_sem=cm_wt_send_sem_ref.at[_b + 0],
                        recv_sem=cm_wt_recv_sem_ref.at[_b + 0],
                        device_id={ep_axis_name: _pt},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    if has_scale:
                        pltpu.make_async_remote_copy(
                            src_ref=w1_scale_ref.at[_ss],
                            dst_ref=_ds1,
                            send_sem=cm_wt_send_sem_ref.at[_b + 1],
                            recv_sem=cm_wt_recv_sem_ref.at[_b + 1],
                            device_id={ep_axis_name: _pt},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                    pltpu.make_async_remote_copy(
                        src_ref=w2_packed.at[_ss],
                        dst_ref=_dw2,
                        send_sem=cm_wt_send_sem_ref.at[_b + 2],
                        recv_sem=cm_wt_recv_sem_ref.at[_b + 2],
                        device_id={ep_axis_name: _pt},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    if has_scale:
                        pltpu.make_async_remote_copy(
                            src_ref=w2_scale_ref.at[_ss],
                            dst_ref=_ds2,
                            send_sem=cm_wt_send_sem_ref.at[_b + 3],
                            recv_sem=cm_wt_recv_sem_ref.at[_b + 3],
                            device_id={ep_axis_name: _pt},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

        def _maybe_start_deferred_cm_wt_send():
            if cm_defer_send:

                @pl.when(cm_awaited_ref[_cm_flag_slots] == 0)
                def _do_def_send():
                    _issue_all_cm_wt_sends()
                    cm_awaited_ref[_cm_flag_slots] = jnp.int32(1)

        if not cm_defer_send and not _split_entry_barrier:
            _issue_all_cm_wt_sends()

        def _ensure_cm_wt_ready(slot_idx: int = 0):
            if _split_cm_wait:
                _ensure_cm_w1_ready(slot_idx)
                _ensure_cm_w2_ready(slot_idx)
                return
            _sb = 4 * slot_idx
            _w1_dst = _cm_pad_view(cm_w1_pad, slot_idx)
            _s1_dst = _cm_pad_view(cm_s1_pad, slot_idx)
            _w2_dst = _cm_pad_view(cm_w2_pad, slot_idx)
            _s2_dst = _cm_pad_view(cm_s2_pad, slot_idx)

            @pl.when(cm_awaited_ref[slot_idx] == 0)
            def _wait_cm():
                pltpu.make_async_copy(
                    src_ref=_w1_dst,
                    dst_ref=_w1_dst,
                    sem=cm_wt_recv_sem_ref.at[_sb + 0],
                ).wait()
                if has_scale:
                    pltpu.make_async_copy(
                        src_ref=_s1_dst,
                        dst_ref=_s1_dst,
                        sem=cm_wt_recv_sem_ref.at[_sb + 1],
                    ).wait()
                pltpu.make_async_copy(
                    src_ref=_w2_dst,
                    dst_ref=_w2_dst,
                    sem=cm_wt_recv_sem_ref.at[_sb + 2],
                ).wait()
                if has_scale:
                    pltpu.make_async_copy(
                        src_ref=_s2_dst,
                        dst_ref=_s2_dst,
                        sem=cm_wt_recv_sem_ref.at[_sb + 3],
                    ).wait()
                cm_awaited_ref[slot_idx] = jnp.int32(1)

        def _ensure_cm_w1_ready(slot_idx: int = 0):
            if not _split_cm_wait:
                return _ensure_cm_wt_ready(slot_idx)
            _sb = 4 * slot_idx
            _w1_dst = _cm_pad_view(cm_w1_pad, slot_idx)
            _s1_dst = _cm_pad_view(cm_s1_pad, slot_idx)

            @pl.when(cm_awaited_ref[slot_idx] == 0)
            def _wait_cm_w1():
                pltpu.make_async_copy(
                    src_ref=_w1_dst,
                    dst_ref=_w1_dst,
                    sem=cm_wt_recv_sem_ref.at[_sb + 0],
                ).wait()
                if has_scale:
                    pltpu.make_async_copy(
                        src_ref=_s1_dst,
                        dst_ref=_s1_dst,
                        sem=cm_wt_recv_sem_ref.at[_sb + 1],
                    ).wait()
                cm_awaited_ref[slot_idx] = jnp.int32(1)

        def _ensure_cm_w2_ready(slot_idx: int = 0):
            if not _split_cm_wait:
                return _ensure_cm_wt_ready(slot_idx)
            _sb = 4 * slot_idx
            _w2_dst = _cm_pad_view(cm_w2_pad, slot_idx)
            _s2_dst = _cm_pad_view(cm_s2_pad, slot_idx)
            _w2_flag = eff_cm_slots + slot_idx

            @pl.when(cm_awaited_ref[_w2_flag] == 0)
            def _wait_cm_w2():
                pltpu.make_async_copy(
                    src_ref=_w2_dst,
                    dst_ref=_w2_dst,
                    sem=cm_wt_recv_sem_ref.at[_sb + 2],
                ).wait()
                if has_scale:
                    pltpu.make_async_copy(
                        src_ref=_s2_dst,
                        dst_ref=_s2_dst,
                        sem=cm_wt_recv_sem_ref.at[_sb + 3],
                    ).wait()
                cm_awaited_ref[_w2_flag] = jnp.int32(1)
    else:
        cm_partner = None
        _maybe_start_deferred_cm_wt_send = None
        _ensure_cm_w1_ready = None
        _ensure_cm_w2_ready = None
        _ensure_cm_wt_ready = None

    if hot_interleave:
        hot_slot = group_start_ref[4]
        hot_w1_loaded_ref[0] = jnp.int32(0)

        @pl.when(jnp.logical_and(local_num_gm > 0, hot_slot >= 0))
        def _prime_hot_w1_bank():
            for _hn in range(num_n1):
                if act_fn and num_n1 > 1:
                    _goff = _hn * tile_n1
                    _uoff = out_n1 + _hn * tile_n1
                    pltpu.make_async_copy(
                        src_ref=w1_packed.at[
                            hot_slot, pl.ds(0, pk1), pl.ds(_goff, tile_n1)
                        ],
                        dst_ref=hot_w1_buf_ref.at[_hn, :, :tile_n1],
                        sem=hot_w1_sem_ref.at[_hn],
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=w1_packed.at[
                            hot_slot, pl.ds(0, pk1), pl.ds(_uoff, tile_n1)
                        ],
                        dst_ref=hot_w1_buf_ref.at[_hn, :, tile_n1:],
                        sem=hot_w1_sem_ref.at[_hn],
                    ).start()
                    if has_scale:
                        pltpu.make_async_copy(
                            src_ref=_scale_hbm_tile(
                                w1_scale_ref,
                                hot_slot,
                                pl.ds(0, nqb1),
                                pl.ds(_goff, tile_n1),
                            ),
                            dst_ref=_scale_vmem_cols(
                                hot_w1_scale_buf_ref, _hn, slice(None, tile_n1)
                            ),
                            sem=hot_w1_sem_ref.at[_hn],
                        ).start()
                        pltpu.make_async_copy(
                            src_ref=_scale_hbm_tile(
                                w1_scale_ref,
                                hot_slot,
                                pl.ds(0, nqb1),
                                pl.ds(_uoff, tile_n1),
                            ),
                            dst_ref=_scale_vmem_cols(
                                hot_w1_scale_buf_ref, _hn, slice(tile_n1, None)
                            ),
                            sem=hot_w1_sem_ref.at[_hn],
                        ).start()
                else:
                    pltpu.make_async_copy(
                        src_ref=w1_packed.at[
                            hot_slot, pl.ds(0, pk1), pl.ds(_hn * w1_dma_n, w1_dma_n)
                        ],
                        dst_ref=hot_w1_buf_ref.at[_hn],
                        sem=hot_w1_sem_ref.at[_hn],
                    ).start()
                    if has_scale:
                        pltpu.make_async_copy(
                            src_ref=_scale_hbm_tile(
                                w1_scale_ref,
                                hot_slot,
                                pl.ds(0, nqb1),
                                pl.ds(_hn * w1_dma_n, w1_dma_n),
                            ),
                            dst_ref=hot_w1_scale_buf_ref.at[_hn],
                            sem=hot_w1_sem_ref.at[_hn],
                        ).start()

        def _ensure_hot_w1_ready():
            @pl.when(hot_w1_loaded_ref[0] == 0)
            def _wait_hot():
                for _hn in range(num_n1):
                    pltpu.make_async_copy(
                        src_ref=hot_w1_buf_ref.at[_hn],
                        dst_ref=hot_w1_buf_ref.at[_hn],
                        sem=hot_w1_sem_ref.at[_hn],
                    ).wait()
                    if has_scale:
                        pltpu.make_async_copy(
                            src_ref=hot_w1_scale_buf_ref.at[_hn],
                            dst_ref=hot_w1_scale_buf_ref.at[_hn],
                            sem=hot_w1_sem_ref.at[_hn],
                        ).wait()
                hot_w1_loaded_ref[0] = jnp.int32(1)
    else:
        hot_slot = None
        _ensure_hot_w1_ready = None

    # --- Weight DMA helpers ---
    def _start_w1_dma_local(buf_id, expert_id, n_id, k_id=0):
        if act_fn and num_n1 > 1:
            # Fix: with fuse_act, w1 is [all_gate | all_up] along N.
            # Load tile_n1 from gate + tile_n1 from up into buf as [gate|up].
            _gate_offset = n_id * tile_n1
            _up_offset = out_n1 + n_id * tile_n1
            # Gate half -> first tile_n1 cols of buf
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(_gate_offset, tile_n1)
                ],
                dst_ref=w1_buf_ref.at[buf_id, :, :tile_n1],
                sem=w1_sem_ref.at[buf_id],
            ).start()
            # Up half -> last tile_n1 cols of buf
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(_up_offset, tile_n1)
                ],
                dst_ref=w1_buf_ref.at[buf_id, :, tile_n1:],
                sem=w1_sem_ref.at[buf_id],
            ).start()
        else:
            pltpu.make_async_copy(
                src_ref=w1_packed.at[
                    expert_id, pl.ds(k_id * pk1, pk1), pl.ds(n_id * w1_dma_n, w1_dma_n)
                ],
                dst_ref=w1_buf_ref.at[buf_id],
                sem=w1_sem_ref.at[buf_id],
            ).start()
        if has_scale:
            if act_fn and num_n1 > 1:
                _gate_offset_s = n_id * tile_n1
                _up_offset_s = out_n1 + n_id * tile_n1
                pltpu.make_async_copy(
                    src_ref=_scale_hbm_tile(
                        w1_scale_ref,
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        pl.ds(_gate_offset_s, tile_n1),
                    ),
                    dst_ref=_scale_vmem_cols(
                        w1_scale_buf_ref, buf_id, slice(None, tile_n1)
                    ),
                    sem=w1_sem_ref.at[buf_id],
                ).start()
                pltpu.make_async_copy(
                    src_ref=_scale_hbm_tile(
                        w1_scale_ref,
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        pl.ds(_up_offset_s, tile_n1),
                    ),
                    dst_ref=_scale_vmem_cols(
                        w1_scale_buf_ref, buf_id, slice(tile_n1, None)
                    ),
                    sem=w1_sem_ref.at[buf_id],
                ).start()
            else:
                pltpu.make_async_copy(
                    src_ref=_scale_hbm_tile(
                        w1_scale_ref,
                        expert_id,
                        pl.ds(k_id * nqb1, nqb1),
                        pl.ds(n_id * w1_dma_n, w1_dma_n),
                    ),
                    dst_ref=w1_scale_buf_ref.at[buf_id],
                    sem=w1_sem_ref.at[buf_id],
                ).start()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w1_bias_ref.at[expert_id, :, pl.ds(n_id * w1_dma_n, w1_dma_n)],
                dst_ref=w1_bias_buf_ref,
                sem=w1_sem_ref.at[buf_id],
            ).start()

    @jax.named_scope("start_w1_dma")
    def start_w1_dma(buf_id, expert_id, n_id, k_id=0):
        if not enable_chipmate:
            return _start_w1_dma_local(buf_id, expert_id, n_id, k_id)

        for _ci in range(eff_cm_slots):
            _w1_pad_c = _cm_pad_view(cm_w1_pad, _ci)
            _s1_pad_c = _cm_pad_view(cm_s1_pad, _ci)

            @pl.when(expert_id == borrowed_slot + _ci)
            def _w1_from_cm(_idx=_ci, _pw1=_w1_pad_c, _ps1=_s1_pad_c):
                _ensure_cm_w1_ready(_idx)
                if act_fn and num_n1 > 1:
                    _goff = n_id * tile_n1
                    _uoff = out_n1 + n_id * tile_n1
                    pltpu.make_async_copy(
                        src_ref=_pw1.at[pl.ds(k_id * pk1, pk1), pl.ds(_goff, tile_n1)],
                        dst_ref=w1_buf_ref.at[buf_id, :, :tile_n1],
                        sem=w1_sem_ref.at[buf_id],
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=_pw1.at[pl.ds(k_id * pk1, pk1), pl.ds(_uoff, tile_n1)],
                        dst_ref=w1_buf_ref.at[buf_id, :, tile_n1:],
                        sem=w1_sem_ref.at[buf_id],
                    ).start()
                    if has_scale:
                        pltpu.make_async_copy(
                            src_ref=_scale_pad_tile(
                                _ps1, pl.ds(k_id * nqb1, nqb1), pl.ds(_goff, tile_n1)
                            ),
                            dst_ref=_scale_vmem_cols(
                                w1_scale_buf_ref, buf_id, slice(None, tile_n1)
                            ),
                            sem=w1_sem_ref.at[buf_id],
                        ).start()
                        pltpu.make_async_copy(
                            src_ref=_scale_pad_tile(
                                _ps1, pl.ds(k_id * nqb1, nqb1), pl.ds(_uoff, tile_n1)
                            ),
                            dst_ref=_scale_vmem_cols(
                                w1_scale_buf_ref, buf_id, slice(tile_n1, None)
                            ),
                            sem=w1_sem_ref.at[buf_id],
                        ).start()
                else:
                    pltpu.make_async_copy(
                        src_ref=_pw1.at[
                            pl.ds(k_id * pk1, pk1), pl.ds(n_id * w1_dma_n, w1_dma_n)
                        ],
                        dst_ref=w1_buf_ref.at[buf_id],
                        sem=w1_sem_ref.at[buf_id],
                    ).start()
                    if has_scale:
                        pltpu.make_async_copy(
                            src_ref=_scale_pad_tile(
                                _ps1,
                                pl.ds(k_id * nqb1, nqb1),
                                pl.ds(n_id * w1_dma_n, w1_dma_n),
                            ),
                            dst_ref=w1_scale_buf_ref.at[buf_id],
                            sem=w1_sem_ref.at[buf_id],
                        ).start()

        @pl.when(expert_id < borrowed_slot)
        def _w1_from_home():
            _start_w1_dma_local(buf_id, expert_id, n_id, k_id)

    def _start_w2_dma_local(buf_id, expert_id, n_id, k_id=0):
        pltpu.make_async_copy(
            src_ref=w2_packed.at[
                expert_id, pl.ds(k_id * pk2, pk2), pl.ds(n_id * tile_n2, tile_n2)
            ],
            dst_ref=w2_buf_ref.at[buf_id],
            sem=w2_sem_ref.at[buf_id],
        ).start()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=_scale_hbm_tile(
                    w2_scale_ref,
                    expert_id,
                    pl.ds(k_id * nqb2, nqb2),
                    pl.ds(n_id * tile_n2, tile_n2),
                ),
                dst_ref=w2_scale_buf_ref.at[buf_id],
                sem=w2_sem_ref.at[buf_id],
            ).start()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w2_bias_ref.at[expert_id, :, pl.ds(n_id * tile_n2, tile_n2)],
                dst_ref=w2_bias_buf_ref,
                sem=w2_sem_ref.at[buf_id],
            ).start()

    @jax.named_scope("start_w2_dma")
    def start_w2_dma(buf_id, expert_id, n_id, k_id=0):
        if not enable_chipmate:
            return _start_w2_dma_local(buf_id, expert_id, n_id, k_id)

        for _ci in range(eff_cm_slots):
            _w2_pad_c = _cm_pad_view(cm_w2_pad, _ci)
            _s2_pad_c = _cm_pad_view(cm_s2_pad, _ci)

            @pl.when(expert_id == borrowed_slot + _ci)
            def _w2_from_cm(_idx=_ci, _pw2=_w2_pad_c, _ps2=_s2_pad_c):
                _ensure_cm_w2_ready(_idx)
                pltpu.make_async_copy(
                    src_ref=_pw2.at[
                        pl.ds(k_id * pk2, pk2), pl.ds(n_id * tile_n2, tile_n2)
                    ],
                    dst_ref=w2_buf_ref.at[buf_id],
                    sem=w2_sem_ref.at[buf_id],
                ).start()
                if has_scale:
                    pltpu.make_async_copy(
                        src_ref=_scale_pad_tile(
                            _ps2,
                            pl.ds(k_id * nqb2, nqb2),
                            pl.ds(n_id * tile_n2, tile_n2),
                        ),
                        dst_ref=w2_scale_buf_ref.at[buf_id],
                        sem=w2_sem_ref.at[buf_id],
                    ).start()

        @pl.when(expert_id < borrowed_slot)
        def _w2_from_home():
            _start_w2_dma_local(buf_id, expert_id, n_id, k_id)

    @jax.named_scope("wait_w1_dma")
    def wait_w1_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w1_buf_ref.at[buf_id],
            dst_ref=w1_buf_ref.at[buf_id],
            sem=w1_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w1_scale_buf_ref.at[buf_id],
                dst_ref=w1_scale_buf_ref.at[buf_id],
                sem=w1_sem_ref.at[buf_id],
            ).wait()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w1_bias_buf_ref,
                dst_ref=w1_bias_buf_ref,
                sem=w1_sem_ref.at[buf_id],
            ).wait()

    @jax.named_scope("wait_w2_dma")
    def wait_w2_dma(buf_id):
        pltpu.make_async_copy(
            src_ref=w2_buf_ref.at[buf_id],
            dst_ref=w2_buf_ref.at[buf_id],
            sem=w2_sem_ref.at[buf_id],
        ).wait()
        if has_scale:
            pltpu.make_async_copy(
                src_ref=w2_scale_buf_ref.at[buf_id],
                dst_ref=w2_scale_buf_ref.at[buf_id],
                sem=w2_sem_ref.at[buf_id],
            ).wait()
        if has_bias:
            pltpu.make_async_copy(
                src_ref=w2_bias_buf_ref,
                dst_ref=w2_bias_buf_ref,
                sem=w2_sem_ref.at[buf_id],
            ).wait()

    # --- GMM compute helpers ---
    use_lhs_2d_cache = (
        num_k1 == 1
        and cfgs1.lhs_cfgs.quant_dtype is None
        and partial_out1_ref.ndim == 2
        and partial_out1_ref.dtype == out_dtype
        and partial_out1_ref.shape == (tile_m, tile_k1)
        and tile_m % 32 == 0
        and tile_k1 % 1024 == 0
    )
    use_lhs_fp8_vmem = (
        partial_out1_ref.ndim == 2
        and partial_out1_ref.dtype != out_dtype
        and partial_out2_ref.ndim == 3
    )
    _req_b_pq = max(
        dims.size_k1 // cfgs1.lhs_cfgs.quant_block_size,
        dims.size_k2 // cfgs2.lhs_cfgs.quant_block_size,
    )
    _g2_amax_slot = (
        _req_b_pq
        if (use_lhs_fp8_vmem and partial_out2_ref.shape[0] > _req_b_pq)
        else None
    )

    @jax.named_scope("blockdiag_gmm1_tile")
    def _blockdiag_gmm1_tile(buf_id, n_id, gm_id, rows):
        sem_id = gm_id % 2
        xc = jnp.concatenate(
            [
                gathered_lhs_2x_ref[sem_id, pl.ds(a, 1)].reshape(1, tile_k1)
                for a in range(rows)
            ],
            axis=0,
        )
        acc = _fp4_blockdiag_gmm(
            xc, w1_buf_ref.at[buf_id], w1_scale_buf_ref.at[buf_id], tile_k1, 2 * tile_n1
        )
        gate, up = acc[:, :tile_n1], acc[:, tile_n1:]
        if has_w1_gs:
            expert = fused_metadata_ref.gm_id_to_group_id[0]
            gate = gate * w1_gs_gate_ref[expert].astype(jnp.float32)
            up = up * w1_gs_up_ref[expert].astype(jnp.float32)
        # silu(gate) rounds to the output dtype first, as in the in-strip epilogue.
        h = jax.nn.silu(gate).astype(gmm1_out_ref.dtype).astype(jnp.float32) * up
        rows_out = dims.size_lhs_sublane
        gmm1_out_ref[pl.ds(0, rows_out), pl.ds(n_id * tile_n1, tile_n1)] = _pad_rows_to(
            h.astype(gmm1_out_ref.dtype), rows_out
        )

    k256_prefill_fp8 = (
        envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_FP8.get()
        if dims.size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    # K256 decode tiles (k256_tiles, implied by block256_scales): m_rows-row dots over
    # K256 chunks in one fori_loop over 256-column strips, LHS from the 2-D cache. The
    # unrolled W4A16 strips and blockdiag branches make a program large enough that its
    # instruction fetch, not the MXU or HBM, sets the decode time.
    b256_nc = decode_strip_nc if decode_strip_nc > 0 else 256
    # 256 when the caller passes compact block-256 scales (fp4_w4a16 + BLOCK256_SCALES).
    s1_block = cfgs1.rhs_cfgs.quant_block_size
    s2_block = cfgs2.rhs_cfgs.quant_block_size

    def _pick_strip_nc(m_rows: int, max_n: int) -> int:
        if decode_strip_nc > 0:
            return min(decode_strip_nc, max_n)
        if decode_strip_nc == -1:
            nc = 1024 if m_rows <= 16 else (512 if m_rows <= 64 else 256)
            return min(nc, max_n)
        if decode_strip_nc == -2:
            nc = 512 if m_rows <= 64 else 256
            return min(nc, max_n)
        if decode_strip_nc == -3:
            nc = (
                2048
                if m_rows <= 16
                else (1024 if m_rows <= 32 else (512 if m_rows <= 64 else 256))
            )
            return min(nc, max_n)
        if decode_strip_nc == -4:
            nc = 2048 if m_rows <= 32 else (1024 if m_rows <= 64 else 512)
            return min(nc, max_n)
        if decode_strip_nc == -5:
            nc = 2048 if m_rows <= 16 else (1024 if m_rows <= 64 else 512)
            return min(nc, max_n)
        return 256

    @jax.named_scope("block256_gmm1_tile")
    def _block256_gmm1_tile(buf_id, n_id, m_rows):
        nc_cur = _pick_strip_nc(m_rows, tile_n1)
        w_ref, s_ref = w1_buf_ref.at[buf_id], w1_scale_buf_ref.at[buf_id]
        if has_w1_gs:
            expert = fused_metadata_ref.gm_id_to_group_id[0]
            gs_gate = w1_gs_gate_ref[expert].astype(jnp.float32)
            gs_up = w1_gs_up_ref[expert].astype(jnp.float32)

        use_dual = (
            decode_dual_k256 == 1
            or (decode_dual_k256 == 3 and m_rows <= 32)
            or (decode_dual_k256 in (4, 6) and m_rows <= 16)
        )
        use_vmem_stage = (
            decode_dual_k256 in (2, 5)
            or (decode_dual_k256 == 3 and m_rows > 32)
            or (decode_dual_k256 in (4, 6) and m_rows > 16)
        )

        def _run_m_slice(r0):
            x_ref = partial_out1_ref.at[pl.ds(r0, m_rows)]
            row_out = pl.ds(r0, m_rows)

            def strip(j, carry):
                n0 = pl.multiple_of(j * nc_cur, nc_cur)
                col_out = pl.ds(n_id * tile_n1 + n0, nc_cur)
                if use_dual:
                    gate, up = _fp4_k256_dual_gmm(
                        x_ref,
                        w_ref,
                        s_ref,
                        tile_k1,
                        n0,
                        tile_n1 + n0,
                        nc_cur,
                        block256_scales,
                        s1_block,
                    )
                    if has_w1_gs:
                        gate, up = gate * gs_gate, up * gs_up
                    h = (
                        jax.nn.silu(gate).astype(gmm1_out_ref.dtype).astype(jnp.float32)
                        * up
                    )
                    gmm1_out_ref[row_out, col_out] = h.astype(gmm1_out_ref.dtype)
                elif use_vmem_stage:
                    gate = _fp4_k256_gmm(
                        x_ref,
                        w_ref,
                        s_ref,
                        tile_k1,
                        n0,
                        nc_cur,
                        block256_scales,
                        s1_block,
                        k256_prefill_fp8,
                    )
                    if has_w1_gs:
                        gate = gate * gs_gate
                    gmm1_out_ref[row_out, col_out] = jax.nn.silu(gate).astype(
                        gmm1_out_ref.dtype
                    )
                    up = _fp4_k256_gmm(
                        x_ref,
                        w_ref,
                        s_ref,
                        tile_k1,
                        tile_n1 + n0,
                        nc_cur,
                        block256_scales,
                        s1_block,
                        k256_prefill_fp8,
                    )
                    if has_w1_gs:
                        up = up * gs_up
                    sg = gmm1_out_ref[row_out, col_out].astype(jnp.float32)
                    gmm1_out_ref[row_out, col_out] = (sg * up).astype(
                        gmm1_out_ref.dtype
                    )
                else:
                    gate = _fp4_k256_gmm(
                        x_ref,
                        w_ref,
                        s_ref,
                        tile_k1,
                        n0,
                        nc_cur,
                        block256_scales,
                        s1_block,
                        k256_prefill_fp8,
                    )
                    up = _fp4_k256_gmm(
                        x_ref,
                        w_ref,
                        s_ref,
                        tile_k1,
                        tile_n1 + n0,
                        nc_cur,
                        block256_scales,
                        s1_block,
                        k256_prefill_fp8,
                    )
                    if has_w1_gs:
                        gate, up = gate * gs_gate, up * gs_up
                    # silu(gate) rounds to the output dtype first, as in the in-strip epilogue.
                    h = (
                        jax.nn.silu(gate).astype(gmm1_out_ref.dtype).astype(jnp.float32)
                        * up
                    )
                    gmm1_out_ref[row_out, col_out] = h.astype(gmm1_out_ref.dtype)
                return carry

            if decode_strip_nc != 0 and tile_n1 == nc_cur:
                strip(0, 0)
            elif decode_dual_k256 in (4, 5) and tile_n1 == 2 * nc_cur:
                strip(0, 0)
                strip(1, 0)
            else:
                lax.fori_loop(0, tile_n1 // nc_cur, strip, 0)

        if decode_dual_k256 == 6 and m_rows == 32 and tile_m > 32:
            num_v = fused_metadata_ref.gm_id_to_m_offset[1]
            n_m_chunks = jnp.maximum(jnp.int32(1), (num_v + 31) // 32)
            lax.fori_loop(
                0,
                n_m_chunks,
                lambda mi, c: (_run_m_slice(pl.multiple_of(mi * 32, 32)), c)[1],
                0,
            )
        else:
            _run_m_slice(0)

    @jax.named_scope("block256_gmm2_tile")
    def _block256_gmm2_tile(buf_id, n_id, gm_id, m_rows):
        nc_cur = _pick_strip_nc(m_rows, tile_n2)
        sem_id = gm_id % 2
        w_ref, s_ref = w2_buf_ref.at[buf_id], w2_scale_buf_ref.at[buf_id]
        if has_w2_gs:
            gs = w2_gs_ref[fused_metadata_ref.gm_id_to_group_id[0]].astype(jnp.float32)

        def _run_m2_slice(r0):
            x_ref = gmm1_out_ref.at[pl.ds(r0, m_rows)]
            row_out = pl.ds(r0, m_rows)

            def strip(j, carry):
                n0 = pl.multiple_of(j * nc_cur, nc_cur)
                out = _fp4_k256_gmm(
                    x_ref,
                    w_ref,
                    s_ref,
                    tile_k2,
                    n0,
                    nc_cur,
                    block256_scales,
                    s2_block,
                    k256_prefill_fp8,
                )
                if has_w2_gs:
                    out = out * gs
                out = out.astype(tiled_out_2x_ref.dtype)
                col = pl.ds(n_id * tile_n2 + n0, nc_cur)
                if in_kernel_post_norm_size > 0:
                    out = out * w2_bias_ref[0:1, col].astype(out.dtype)
                tiled_out_2x_ref[sem_id, row_out, col] = out
                return carry

            if decode_strip_nc != 0 and tile_n2 == nc_cur:
                strip(0, 0)
            elif decode_dual_k256 in (4, 5) and tile_n2 == 2 * nc_cur:
                strip(0, 0)
                strip(1, 0)
            else:
                lax.fori_loop(0, tile_n2 // nc_cur, strip, 0)

        if decode_dual_k256 == 6 and m_rows == 32 and tile_m > 32:
            num_v = fused_metadata_ref.gm_id_to_m_offset[1]
            n_m_chunks = jnp.maximum(jnp.int32(1), (num_v + 31) // 32)
            lax.fori_loop(
                0,
                n_m_chunks,
                lambda mi, c: (_run_m2_slice(pl.multiple_of(mi * 32, 32)), c)[1],
                0,
            )
        else:
            _run_m2_slice(0)

    if w8_refs is not None:
        lhs_row_scales_ref, w2_fp8_cache_ref, rhs_fp8_strip_ref = w8_refs
    else:
        lhs_row_scales_ref = w2_fp8_cache_ref = rhs_fp8_strip_ref = None
    has_ext_pq_lhs = (lhs_row_scales_ref is not None) or static_fp8_lhs

    @jax.named_scope("compute_gmm1_tile")
    def compute_gmm1_tile(buf_id, n_id, k_id, gm_id, m_rows, bd_rows=0):
        if k256_tiles:
            return _block256_gmm1_tile(buf_id, n_id, m_rows)
        if bd_rows:
            return _blockdiag_gmm1_tile(buf_id, n_id, gm_id, bd_rows)
        sem_id = gm_id % 2
        _cfgs1 = dataclasses.replace(
            cfgs1, tiles=dataclasses.replace(cfgs1.tiles, tile_m=m_rows)
        )
        k_cols = tile_k1 // num_lanes
        k_offset = k_id * k_cols
        _fast_lhs_q = (
            int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get() or 0)
            if dims.size_m > 4096
            else 0
        )
        _use_lhs_3d_ref = (
            _fast_lhs_q >= 1
            and use_lhs_fp8_vmem
            and not use_lhs_2d_cache
            and not has_ext_pq_lhs
        )
        lhs_data = (
            None
            if (use_lhs_2d_cache or _use_lhs_3d_ref or has_ext_pq_lhs)
            else gathered_lhs_2x_ref[
                sem_id, pl.ds(0, m_rows), pl.ds(k_offset, k_cols), :
            ].reshape(-1, dims.size_lhs_sublane, tile_k1)
        )

        if has_ext_pq_lhs and n_id == 0 and k_id == 0:
            from .prefill_gmm_opts import unpack_3d_pq_lhs_to_2d_fp8

            unpack_3d_pq_lhs_to_2d_fp8(
                gathered_lhs_2x_ref.at[
                    sem_id, pl.ds(0, m_rows), pl.ds(k_offset, k_cols), :
                ],
                partial_out1_ref.at[pl.ds(0, m_rows), pl.ds(0, tile_k1)],
                tile_m=m_rows,
                tile_k=tile_k1,
                via_bf16_xpose=(pq_lhs_mode == 2),
            )

        def _run_inner1(w1_tile_sel, w1_sc_sel):
            w1_bias_tile = w1_bias_buf_ref if has_bias else None
            w1_weights = WeightsRef(
                weight=w1_tile_sel, scale=w1_sc_sel, bias=w1_bias_tile
            )
            _acc1_slice = (
                shared_acc_ref
                if num_k1 == 1
                else shared_acc_ref.at[
                    pl.ds(0, m_rows), pl.ds(0, tile_n1 * 2 if act_fn else tile_n1)
                ]
            )
            inner_kernel(
                lhs_data,
                w1_weights,
                gmm1_out_ref.at[pl.ds(0, m_rows), pl.ds(n_id * tile_n1, tile_n1)],
                partial_out1_ref,
                _acc1_slice,
                fused_metadata_ref,
                cfgs=_cfgs1,
                gs_gate_ref=w1_gs_gate_ref if has_w1_gs else None,
                gs_up_ref=w1_gs_up_ref if has_w1_gs else None,
                scatter_mode=True,
                _k_id=k_id,
                _num_k=num_k1,
                _gm_id=0,
                _n_id=n_id,
                _num_n=num_n1,
                fused_g2_amax_slot=_g2_amax_slot,
                prequant_lhs_fp8_ref=(
                    partial_out1_ref.at[pl.ds(0, m_rows), pl.ds(0, tile_k1)]
                    if (use_lhs_fp8_vmem or has_ext_pq_lhs)
                    else None
                ),
                prequant_lhs_scale_ref=(
                    partial_out2_ref.at[:, pl.ds(0, m_rows), :]
                    if (use_lhs_fp8_vmem or has_ext_pq_lhs)
                    else None
                ),
                skip_row_mask=True,
                lhs_2d_vmem_ref=(
                    partial_out1_ref.at[pl.ds(0, m_rows), :]
                    if use_lhs_2d_cache
                    else None
                ),
                lhs_3d_vmem_ref=(
                    gathered_lhs_2x_ref.at[
                        sem_id, pl.ds(0, m_rows), pl.ds(k_offset, k_cols), :
                    ]
                    if _use_lhs_3d_ref
                    else None
                ),
                prequant_lhs_3d_fp8_ref=None,
                rhs_fp8_strip_ref=rhs_fp8_strip_ref,
                vectorized_dequant=vectorized_dequant,
                w_strip_fp8_ref=eo_w_strip_fp8_ref,
                prequant_lhs_ready=has_ext_pq_lhs,
                skip_lhs_scale_mul=static_fp8_lhs,
            )

        if hot_interleave:
            cur_eid = fused_metadata_ref.gm_id_to_group_id[0]
            use_hot_bank = (hot_slot >= 0) & (cur_eid == hot_slot)

            @pl.when(use_hot_bank)
            def _gmm1_hot():
                _run_inner1(
                    hot_w1_buf_ref.at[n_id],
                    hot_w1_scale_buf_ref.at[n_id] if has_scale else None,
                )

            @pl.when(~use_hot_bank)
            def _gmm1_std():
                _run_inner1(
                    w1_buf_ref.at[buf_id],
                    w1_scale_buf_ref.at[buf_id] if has_scale else None,
                )
        else:
            _run_inner1(
                w1_buf_ref.at[buf_id],
                w1_scale_buf_ref.at[buf_id] if has_scale else None,
            )

    @jax.named_scope("blockdiag_gmm2_tile")
    def _blockdiag_gmm2_tile(buf_id, n_id, gm_id, rows):
        sem_id = gm_id % 2
        rows_out = dims.size_lhs_sublane
        xc = gmm1_out_ref[pl.ds(0, rows_out), :][:rows]
        out = _fp4_blockdiag_gmm(
            xc, w2_buf_ref.at[buf_id], w2_scale_buf_ref.at[buf_id], tile_k2, tile_n2
        )
        if has_w2_gs:
            out = out * w2_gs_ref[fused_metadata_ref.gm_id_to_group_id[0]].astype(
                jnp.float32
            )
        out = out.astype(tiled_out_2x_ref.dtype)
        if in_kernel_post_norm_size > 0:
            out = out * w2_bias_ref[0:1, pl.ds(n_id * tile_n2, tile_n2)].astype(
                out.dtype
            )
        tiled_out_2x_ref[sem_id, pl.ds(0, rows_out), pl.ds(n_id * tile_n2, tile_n2)] = (
            _pad_rows_to(out, rows_out)
        )

    @jax.named_scope("compute_gmm2_tile")
    def compute_gmm2_tile(buf_id, n_id, k_id, gm_id, m_rows, bd_rows=0):
        if k256_tiles:
            return _block256_gmm2_tile(buf_id, n_id, gm_id, m_rows)
        if bd_rows:
            return _blockdiag_gmm2_tile(buf_id, n_id, gm_id, bd_rows)
        sem_id = gm_id % 2
        _cfgs2 = dataclasses.replace(
            cfgs2, tiles=dataclasses.replace(cfgs2.tiles, tile_m=m_rows)
        )
        use_lhs2_2d_ref = dims.size_k2 == gmm1_out_ref.shape[1]
        if use_lhs2_2d_ref:
            lhs_data = None
        else:
            lhs_data = intermediate_ref[
                pl.ds(0, m_rows // dims.size_lhs_sublane),
                :,
                pl.ds(k_id * tile_k2, tile_k2),
            ]
        w2_tile = w2_buf_ref.at[buf_id]
        w2_sc = w2_scale_buf_ref.at[buf_id] if has_scale else None
        _acc2_slice = (
            shared_acc_ref
            if num_k2 == 1
            else shared_acc_ref.at[pl.ds(0, m_rows), pl.ds(0, tile_n2)]
        )
        use_w2_cache = partial_out1_ref.ndim == 3
        inner_kernel(
            lhs_data,
            WeightsRef(
                weight=w2_tile, scale=w2_sc, bias=w2_bias_buf_ref if has_bias else None
            ),
            tiled_out_2x_ref.at[
                sem_id, pl.ds(0, m_rows), pl.ds(n_id * tile_n2, tile_n2)
            ],
            partial_out2_ref,
            _acc2_slice,
            fused_metadata_ref,
            cfgs=_cfgs2,
            gs_gate_ref=w2_gs_ref if has_w2_gs else None,
            gs_up_ref=w2_gs_ref if has_w2_gs else None,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k2,
            _gm_id=0,
            _n_id=n_id,
            _num_n=num_n2,
            fused_g2_amax_slot=_g2_amax_slot,
            prequant_rhs_fp8_ref=partial_out1_ref.at[buf_id] if use_w2_cache else None,
            prequant_rhs_scale_ref=partial_out2_ref.at[buf_id]
            if use_w2_cache
            else None,
            prequant_lhs_fp8_ref=(
                partial_out1_ref.at[pl.ds(0, m_rows), pl.ds(0, tile_k2)]
                if use_lhs_fp8_vmem
                else None
            ),
            prequant_lhs_scale_ref=(
                partial_out2_ref.at[:, pl.ds(0, m_rows), :]
                if use_lhs_fp8_vmem
                else None
            ),
            pnw_ref=w2_bias_ref if in_kernel_post_norm_size > 0 else None,
            skip_row_mask=True,
            lhs_2d_vmem_ref=(
                gmm1_out_ref.at[pl.ds(0, m_rows), pl.ds(k_id * tile_k2, tile_k2)]
                if use_lhs2_2d_ref
                else None
            ),
            w2_fp8_cache_ref=(
                w2_fp8_cache_ref.at[buf_id]
                if (
                    w2_fp8_cache_ref is not None
                    and not use_weight_lookahead
                    and n_id < w2_fp8_cache_ref.shape[0]
                )
                else None
            ),
            rhs_fp8_strip_ref=rhs_fp8_strip_ref,
            vectorized_dequant=vectorized_dequant,
            w_strip_fp8_ref=eo_w_strip_fp8_ref,
        )

    gathered_lhs_2x_ref[...] = jnp.zeros_like(gathered_lhs_2x_ref)
    # Initialize per-slot DMA counts to 0 (active staging slots).
    num_stg_slots = scatter_staging_3x_ref.shape[0]
    for _s in range(num_stg_slots):
        gm_id_ref[1 + _s] = jnp.int32(0)  # send counts
        gm_id_ref[9 + _s] = jnp.int32(0)  # local write counts
    if block_scatter:
        for _c in range(2 * ep_size):
            gm_id_ref[17 + _c] = jnp.int32(0)

    # --- Bucketed tile_m (option 1): compute each gm-tile over the smallest
    # static row-bucket >= its real rows, capped at the tuned tile_m. Trims the
    # MXU M-padding on partial (tail) tiles without shrinking the tile globally
    # (which regresses imbalanced routing). Buckets are multiples of the sublane
    # tiling; a single-bucket list (tile_m == sublane) disables the dispatch.
    #
    # TODO(rupengliu): explore host-side size-grouped gm-loops. Instead of a
    # per-tile lax.switch, partition gm-tiles by bucket in the scalar metadata
    # prologue and run one loop per static tile_m (full tiles in a tile_m=128
    # loop, small tails in a tile_m=32 loop, ...). Each loop has a single static
    # shape -> uniform control flow, pipelines cleanly, no per-tile branch, and
    # the small-bucket loop is ~empty at high concurrency (0 high-end cost). It
    # runs the same matmul shapes as unconditional bucketing (so no extra
    # low-token gain) but removes the ~0.5% switch overhead at large sizes.
    # Cost: extra loop body + scratch + metadata partitioning; a tail re-streams
    # its expert's weights once. Marginal vs the ~8% weight-streaming ceiling.
    _sls_b = dims.size_lhs_sublane
    _prefill_rb_mode = int(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get() or 0)
    if decode_dual_k256 == 6 and k256_tiles and tile_m > 32:
        M_BUCKETS = [b for b in (16, 32) if _sls_b <= b <= tile_m]
    elif dims.size_m >= 16384 and _prefill_rb_mode == 2 and tile_m <= 128:
        M_BUCKETS = [b for b in (32, 64, 128) if _sls_b <= b <= tile_m]
        if not M_BUCKETS or M_BUCKETS[-1] != tile_m:
            M_BUCKETS = sorted(set(M_BUCKETS) | {tile_m})
    elif dims.size_m >= 16384 and _prefill_rb_mode == 3:
        M_BUCKETS = [b for b in (64, 128, 256) if _sls_b <= b <= tile_m]
        if not M_BUCKETS or M_BUCKETS[-1] != tile_m:
            M_BUCKETS = sorted(set(M_BUCKETS) | {tile_m})
    elif dims.size_m >= 16384 and _prefill_rb_mode == 4:
        _rb4_set = (32, 64, 128) if tile_m <= 128 else (128, 256)
        M_BUCKETS = [b for b in _rb4_set if _sls_b <= b <= tile_m]
        if not M_BUCKETS or M_BUCKETS[-1] != tile_m:
            M_BUCKETS = sorted(set(M_BUCKETS) | {tile_m})
    else:
        M_BUCKETS = [b for b in (8, 16, 32, 64, 128, 256, 512) if _sls_b <= b <= tile_m]
        if not M_BUCKETS or M_BUCKETS[-1] != tile_m:
            M_BUCKETS = sorted(set(M_BUCKETS) | {tile_m})
    # fp4-only gate (plus coarse prefill mode 2 on FP8 when tile_n <= 1024 or tile_m <= 64).
    _w_qdt = cfgs1.rhs_cfgs.quant_dtype
    _weight_is_fp4 = _w_qdt is not None and jax.dtypes.itemsize_bits(_w_qdt) <= 4
    _weight_is_fp8_safe = (
        _w_qdt == jnp.float8_e4m3fn
        and dims.size_m >= 16384
        and _prefill_rb_mode == 2
        and tile_m <= 128
        and (tile_n1 <= 1024 or tile_m <= 64)
    )
    _short_prefill_nobuckets = (
        block_scatter
        and 4096 <= dims.size_m < 16384
        and envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get()
    )
    _USE_BUCKETS = (
        len(M_BUCKETS) > 1
        and (_weight_is_fp4 or _weight_is_fp8_safe)
        and not _short_prefill_nobuckets
        and (
            dims.size_m < 16384
            or _prefill_rb_mode in (1, 3, 4)
            or (_prefill_rb_mode == 2 and tile_m <= 128)
        )
    )
    # Block-diagonal NVFP4 decode tiles: block-16 scales in the compact rank-3 layout,
    # W4A16 on both GMMs, whole-K tiles, gated silu, the GMM1 output read in place by GMM2.
    fp4_whole_k_silu = (
        _w_qdt == jnp.float4_e2m1fn
        and cfgs1.lhs_cfgs.quant_dtype is None
        and cfgs2.lhs_cfgs.quant_dtype is None
        and len(w1_scale_buf_ref.shape) == 3
        and len(w2_scale_buf_ref.shape) == 3
        and act_fn == "silu"
        and not has_bias
        and not fp8_direct_write
        and num_k1 == 1
        and num_k2 == 1
        and tile_k1 % _BLOCKDIAG_KC == 0
        and tile_k2 % _BLOCKDIAG_KC == 0
        and dims.size_k2 == gmm1_out_ref.shape[1]
    )
    scale_blocks = (s1_block, s2_block)
    use_blockdiag = (
        bool(blockdiag_rows)
        and not has_ext_pq_lhs
        and _USE_BUCKETS
        and M_BUCKETS[0] == dims.size_lhs_sublane >= blockdiag_rows[-1]
        and dims.size_m <= _BLOCKDIAG_MAX_SIZE_M
        and fp4_whole_k_silu
        and scale_blocks == (_FP4_GROUP, _FP4_GROUP)
    )
    if k256_tiles and not (
        fp4_whole_k_silu
        and (
            scale_blocks == (_FP4_GROUP, _FP4_GROUP)
            or (block256_scales and scale_blocks == (256, 256))
        )
        and use_lhs_2d_cache
        and tile_n1 % b256_nc == 0
        and tile_n2 % b256_nc == 0
        and b256_nc % 128 == 0
        and dims.size_lhs_sublane % 16 == 0
    ):
        raise ValueError(
            "SGLANG_TPU_MUSE_KERNEL_K256_DECODE_TILES / K256_PREFILL_TILES / BLOCK256_SCALES need NVFP4 "
            f"whole-K W4A16 silu tiles and the 2-D LHS cache (tile_m={tile_m}, tile_k1={tile_k1}, "
            f"tile_n1={tile_n1}, tile_n2={tile_n2}, b256_nc={b256_nc}, size_m={dims.size_m})"
        )

    # --- Main padded gm loop ---
    @jax.named_scope("wait_staging_slot_writes")
    def wait_staging_slot_writes(slot):
        """Wait for and clear the row writes still reading staging ``slot``."""
        pending_local = gm_id_ref[9 + slot]
        pending_send = gm_id_ref[1 + slot]

        @pl.when(pending_local > 0)
        def _():
            pl.semaphore_wait(local_write_sems_ref.at[slot], pending_local)
            if fp8_direct_write:
                pl.semaphore_wait(scale_local_write_sems_ref.at[slot], pending_local)

        @pl.when(pending_send > 0)
        def _():
            pl.semaphore_wait(send_sems_ref.at[slot], pending_send)
            if fp8_direct_write:
                pl.semaphore_wait(scale_send_sems_ref.at[slot], pending_send)

        gm_id_ref[1 + slot] = jnp.int32(0)
        gm_id_ref[9 + slot] = jnp.int32(0)

    @jax.named_scope("reshape_gmm2_output")
    def _reshape_gmm2_output(t_sem, t_stg):
        y_bf16 = tiled_out_2x_ref[t_sem][...]
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
        scatter_staging_3x_ref[t_stg] = raw_out.reshape(
            tile_m,
            scatter_staging_3x_ref.shape[2],
            scatter_staging_3x_ref.shape[3],
        )

    @jax.named_scope("reshape_gmm2_output_rows")
    def _reshape_gmm2_output_rows(t_sem, t_stg, m_rows):
        y_bf16 = tiled_out_2x_ref[t_sem, pl.ds(0, m_rows), :]
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
        scatter_staging_3x_ref[t_stg, pl.ds(0, m_rows), :, :] = raw_out.reshape(
            m_rows,
            scatter_staging_3x_ref.shape[2],
            scatter_staging_3x_ref.shape[3],
        )

    @jax.named_scope("direct_write_rows")
    def _do_direct_write(t_gm, t_stg):
        if block_scatter:
            _bs_stride = 3 * ep_size + 2
            _bs_base = t_gm * _bs_stride
            if topk_indices_ref.ndim == 1:
                send_blks = topk_indices_ref[_bs_base + 3 * ep_size]
                local_blks = topk_indices_ref[_bs_base + 3 * ep_size + 1]
            else:
                send_blks = topk_indices_ref[t_gm, 3 * ep_size]
                local_blks = topk_indices_ref[t_gm, 3 * ep_size + 1]
            gm_id_ref[1 + t_stg] = gm_id_ref[1 + t_stg] + send_blks
            gm_id_ref[9 + t_stg] = gm_id_ref[9 + t_stg] + local_blks
            if rotate_rs_mode >= 1 and (ep_size & (ep_size - 1)) == 0:
                _xor_steps = (
                    tuple(range(2, ep_size, 2))
                    + tuple(range(3, ep_size, 2))
                    + ((1,) if ep_size >= 2 else ())
                )
                for _st in _xor_steps:
                    _tgt = lax.bitwise_xor(my_id, jnp.int32(_st))
                    if topk_indices_ref.ndim == 1:
                        _src_t = topk_indices_ref[_bs_base + _tgt]
                        _dst_t = topk_indices_ref[_bs_base + ep_size + _tgt]
                        _len_t = topk_indices_ref[_bs_base + 2 * ep_size + _tgt]
                    else:
                        _src_t = topk_indices_ref[t_gm, _tgt]
                        _dst_t = topk_indices_ref[t_gm, ep_size + _tgt]
                        _len_t = topk_indices_ref[t_gm, 2 * ep_size + _tgt]

                    @pl.when(_len_t > 0)
                    def _(_c=_tgt, _src=_src_t, _dst=_dst_t, _len=_len_t):
                        pltpu.make_async_remote_copy(
                            src_ref=scatter_staging_3x_ref.at[
                                t_stg, pl.ds(_src, _len), :, :
                            ],
                            dst_ref=out_buf_ref.at[pl.ds(_dst, _len), :, :],
                            send_sem=send_sems_ref.at[t_stg],
                            recv_sem=recv_sem_ref.at[0],
                            device_id={ep_axis_name: _c},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

                if topk_indices_ref.ndim == 1:
                    _src_self = topk_indices_ref[_bs_base + my_id]
                    _dst_self = topk_indices_ref[_bs_base + ep_size + my_id]
                    _len_self = topk_indices_ref[_bs_base + 2 * ep_size + my_id]
                else:
                    _src_self = topk_indices_ref[t_gm, my_id]
                    _dst_self = topk_indices_ref[t_gm, ep_size + my_id]
                    _len_self = topk_indices_ref[t_gm, 2 * ep_size + my_id]

                @pl.when(_len_self > 0)
                def _(_src=_src_self, _dst=_dst_self, _len=_len_self):
                    pltpu.make_async_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            t_stg, pl.ds(_src, _len), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(_dst, _len), :, :],
                        sem=local_write_sems_ref.at[t_stg],
                    ).start()

                return
            for c in range(ep_size):
                if topk_indices_ref.ndim == 1:
                    _src_c = topk_indices_ref[_bs_base + c]
                    _dst_c = topk_indices_ref[_bs_base + ep_size + c]
                    _len_c = topk_indices_ref[_bs_base + 2 * ep_size + c]
                else:
                    _src_c = topk_indices_ref[t_gm, c]
                    _dst_c = topk_indices_ref[t_gm, ep_size + c]
                    _len_c = topk_indices_ref[t_gm, 2 * ep_size + c]
                has_c = _len_c > 0
                is_loc_c = c == my_id

                @pl.when(jnp.logical_and(has_c, ~is_loc_c))
                def _(_c=c, _src=_src_c, _dst=_dst_c, _len=_len_c):
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            t_stg, pl.ds(_src, _len), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(_dst, _len), :, :],
                        send_sem=send_sems_ref.at[t_stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: jnp.int32(_c)},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                @pl.when(jnp.logical_and(has_c, is_loc_c))
                def _(_src=_src_c, _dst=_dst_c, _len=_len_c):
                    pltpu.make_async_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            t_stg, pl.ds(_src, _len), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(_dst, _len), :, :],
                        sem=local_write_sems_ref.at[t_stg],
                    ).start()

            return
        _ml = jnp.int32(0)
        _n_slots_dw = metadata_ref.gm_id_to_group_id.shape[0]
        t_m_st = metadata_ref.gm_id_to_m_offset[t_gm]
        t_m_en = (
            metadata_ref.gm_id_to_m_offset[_n_slots_dw + t_gm]
            if metadata_ref.gm_id_to_m_offset.shape[0] == 2 * _n_slots_dw
            else metadata_ref.gm_id_to_m_offset[t_gm + 1]
        )
        t_num_valid = t_m_en - t_m_st

        def _write_row(i, carry):
            send_sz, local_sz = carry
            row_idx = _ml + i
            if pack_indices:
                combined = lhs_indices_ref[t_m_st + i]
                cs_tk = chunk_size * top_k
                write_pos = combined % cs_tk
                dest_chip = combined // cs_tk
            else:
                packed_t = topk_indices_ref[t_m_st + i]
                dest_chip = lax.shift_right_logical(packed_t, 16)
                write_pos = packed_t & jnp.int32(0xFFFF)
            is_local = dest_chip == my_id

            if fp8_direct_write:
                row_f32 = scatter_staging_3x_ref[t_stg, row_idx, :, :].astype(
                    jnp.float32
                )
                valid_n2 = (
                    jnp.arange(row_f32.shape[0])[:, None] * row_f32.shape[1]
                    + jnp.arange(row_f32.shape[1])[None, :]
                    < dims.original_n2
                )
                fp8_max = jnp.array(jnp.finfo(jnp.float8_e4m3fn).max, dtype=jnp.float32)
                row_scale = (
                    jnp.maximum(
                        jnp.max(jnp.where(valid_n2, jnp.abs(row_f32), 0.0)),
                        jnp.array(1e-6, dtype=jnp.float32),
                    )
                    / fp8_max
                )
                scatter_scale_3x_ref[t_stg, row_idx, :] = row_scale + jnp.zeros(
                    (128,), dtype=jnp.float32
                )
                scatter_fp8_staging_3x_ref[t_stg, row_idx, :, :] = jnp.where(
                    valid_n2,
                    jnp.clip(row_f32 / row_scale, -fp8_max, fp8_max),
                    0.0,
                ).astype(jnp.float8_e4m3fn)

                @pl.when(~is_local)
                def _():
                    target_device_id = dest_chip
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_fp8_staging_3x_ref.at[
                            t_stg, pl.ds(row_idx, 1), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        send_sem=send_sems_ref.at[t_stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: target_device_id},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                    if fp8_direct_write:
                        pltpu.make_async_remote_copy(
                            src_ref=scatter_scale_3x_ref.at[
                                t_stg, pl.ds(row_idx, 1), :
                            ],
                            dst_ref=out_scale_ref.at[pl.ds(write_pos, 1), :],
                            send_sem=scale_send_sems_ref.at[t_stg],
                            recv_sem=scale_recv_sem_ref.at[0],
                            device_id={ep_axis_name: target_device_id},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

                @pl.when(is_local)
                def _():
                    pltpu.make_async_copy(
                        src_ref=scatter_fp8_staging_3x_ref.at[
                            t_stg, pl.ds(row_idx, 1), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        sem=local_write_sems_ref.at[t_stg],
                    ).start()
                    if fp8_direct_write:
                        pltpu.make_async_copy(
                            src_ref=scatter_scale_3x_ref.at[
                                t_stg, pl.ds(row_idx, 1), :
                            ],
                            dst_ref=out_scale_ref.at[pl.ds(write_pos, 1), :],
                            sem=scale_local_write_sems_ref.at[t_stg],
                        ).start()

            else:

                @pl.when(~is_local)
                def _():
                    target_device_id = dest_chip
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            t_stg, pl.ds(row_idx, 1), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        send_sem=send_sems_ref.at[t_stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: target_device_id},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                @pl.when(is_local)
                def _():
                    pltpu.make_async_copy(
                        src_ref=scatter_staging_3x_ref.at[
                            t_stg, pl.ds(row_idx, 1), :, :
                        ],
                        dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                        sem=local_write_sems_ref.at[t_stg],
                    ).start()

            return (
                send_sz + lax.select(~is_local, jnp.int32(1), jnp.int32(0)),
                local_sz + lax.select(is_local, jnp.int32(1), jnp.int32(0)),
            )

        if (not fp8_direct_write) and (not pack_indices):

            def _emit_one_write_local_only(r_idx, p_val):
                dc = lax.shift_right_logical(p_val, 16)
                wp = p_val & jnp.int32(0xFFFF)
                loc = dc == my_id

                @pl.when(~loc)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_staging_3x_ref.at[t_stg, pl.ds(r_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(wp, 1), :, :],
                        send_sem=send_sems_ref.at[t_stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: dc},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                @pl.when(loc)
                def _():
                    pltpu.make_async_copy(
                        src_ref=scatter_staging_3x_ref.at[t_stg, pl.ds(r_idx, 1), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(wp, 1), :, :],
                        sem=local_write_sems_ref.at[t_stg],
                    ).start()

                return loc.astype(jnp.int32)

            def _write_16_spillfree(h, l_sz):
                i0 = h * 16
                base = t_m_st + i0
                for sub in range(2):
                    off = sub * 8
                    p8 = [topk_indices_ref[base + off + k] for k in range(8)]
                    for k in range(8):
                        l_sz = l_sz + _emit_one_write_local_only(
                            _ml + i0 + off + k, p8[k]
                        )
                return l_sz

            def _write_tail_single(i, l_sz):
                return l_sz + _emit_one_write_local_only(
                    _ml + i, topk_indices_ref[t_m_st + i]
                )

            num_16_w = t_num_valid >> 4
            l0 = lax.fori_loop(0, num_16_w, _write_16_spillfree, jnp.int32(0))
            rem_start = num_16_w << 4

            def _write_mid_8(l_in):
                p8 = [topk_indices_ref[t_m_st + rem_start + k] for k in range(8)]
                for k in range(8):
                    l_in = l_in + _emit_one_write_local_only(_ml + rem_start + k, p8[k])
                return l_in

            has_mid_8 = (t_num_valid - rem_start) >= 8
            l1 = lax.cond(has_mid_8, _write_mid_8, lambda x: x, l0)
            tail_start = rem_start + jnp.where(has_mid_8, jnp.int32(8), jnp.int32(0))
            local_sz = lax.fori_loop(tail_start, t_num_valid, _write_tail_single, l1)
            send_sz = t_num_valid - local_sz
        else:
            send_sz, local_sz = lax.fori_loop(
                0, t_num_valid, _write_row, (jnp.int32(0), jnp.int32(0))
            )
        gm_id_ref[1 + t_stg] = gm_id_ref[1 + t_stg] + send_sz
        gm_id_ref[9 + t_stg] = gm_id_ref[9 + t_stg] + local_sz

        @pl.when(jnp.logical_and(local_sz > 0, not defer_writes))
        def _():
            pl.semaphore_wait(local_write_sems_ref.at[t_stg], local_sz)
            if fp8_direct_write:
                pl.semaphore_wait(scale_local_write_sems_ref.at[t_stg], local_sz)

        @pl.when(jnp.logical_and(send_sz > 0, not defer_writes))
        def _():
            pl.semaphore_wait(send_sems_ref.at[t_stg], send_sz)
            if fp8_direct_write:
                pl.semaphore_wait(scale_send_sems_ref.at[t_stg], send_sz)

    _early_rs = gather_2ahead_mode in (1, 2, 4, 6)
    _two_ahead_gather = (
        gather_2ahead_mode in (1, 3, 4, 5, 6) and not eo3_enabled
    )
    _gather_during_gmm1 = (
        gather_2ahead_mode in (4, 5, 6)
        and (use_lhs_fp8_vmem or has_ext_pq_lhs)
        and not use_lhs_2d_cache
        and num_k1 == 1
        and not k256_tiles
        and not eo_pair_enabled
        and not hot_interleave
    )
    _prologue_w_dma = (
        gather_2ahead_mode == 6
        and not hot_interleave
        and not eo3_enabled
        and not eo_pair_enabled
    )
    _wla1 = int(use_weight_lookahead) == 1
    _wla2 = int(use_weight_lookahead) == 2
    _gather_divisor_top = top_k if pack_indices else 1
    _inter32_gather_kw = (
        {
            "bs_meta_ref": topk_indices_ref,
            "ep_size": ep_size,
            "my_id": my_id,
            "tile_idx_smem_ref": tile_idx_smem_ref,
        }
        if inter32_enabled
        else {}
    )
    _w_base_idx = 17 + 2 * ep_size
    _la_started_idx = 18 + 2 * ep_size
    _next_act_grp_off = (1 + 2 * eff_cm_slots) if enable_chipmate else 1
    if _wla1 or _wla2:
        gm_id_ref[_w_base_idx] = jnp.int32(0)
        gm_id_ref[_la_started_idx] = jnp.int32(0)

    if _two_ahead_gather:

        @jax.named_scope("dma_gather_prologue_2ahead")
        @pl.when(local_num_gm > 0)
        def _():
            if inter32_enabled:

                def _pro_g(gi, _u):
                    dma_gather_gm_start(
                        hidden_states_ref,
                        gathered_lhs_2x_ref.at[gi],
                        lhs_indices_ref,
                        gather_sem_ref.at[gi],
                        gi,
                        metadata_ref,
                        divisor=_gather_divisor_top,
                        packed_u16=packed_u16_lhs,
                        **_inter32_gather_kw,
                    )
                    return _u

                lax.fori_loop(0, jnp.minimum(local_num_gm, 2), _pro_g, jnp.int32(0))
            else:
                dma_gather_gm_start(
                    hidden_states_ref,
                    gathered_lhs_2x_ref.at[0],
                    lhs_indices_ref,
                    gather_sem_ref.at[0],
                    0,
                    metadata_ref,
                    divisor=_gather_divisor_top,
                    packed_u16=packed_u16_lhs,
                )
            if _prologue_w_dma:
                _e0_pro = metadata_ref.gm_id_to_group_id[0]
                _total_w1_pro = num_n1 * num_k1
                _total_w2_pro = num_n2 * num_k2
                if enable_chipmate:

                    @pl.when(_e0_pro < borrowed_slot)
                    def _start_pro_w_home():
                        for _i in range(min(num_w1_bufs, _total_w1_pro)):
                            _start_w1_dma_local(
                                _i, _e0_pro, _i // num_k1, _i % num_k1
                            )
                        for _i in range(min(num_w2_bufs, _total_w2_pro)):
                            _start_w2_dma_local(
                                _i, _e0_pro, _i // num_k2, _i % num_k2
                            )
                else:
                    for _i in range(min(num_w1_bufs, _total_w1_pro)):
                        _start_w1_dma_local(_i, _e0_pro, _i // num_k1, _i % num_k1)
                    for _i in range(min(num_w2_bufs, _total_w2_pro)):
                        _start_w2_dma_local(_i, _e0_pro, _i // num_k2, _i % num_k2)

        if not inter32_enabled:

            @pl.when(local_num_gm > 1)
            def _():
                dma_gather_gm_start(
                    hidden_states_ref,
                    gathered_lhs_2x_ref.at[1],
                    lhs_indices_ref,
                    gather_sem_ref.at[1],
                    1,
                    metadata_ref,
                    divisor=_gather_divisor_top,
                    packed_u16=packed_u16_lhs,
                )

    if _split_entry_barrier:
        sync_barrier_wait()
        if enable_chipmate and not cm_defer_send:
            _issue_all_cm_wt_sends()

    @jax.named_scope("gm_loop_body")
    def gm_loop_body(gm_id, skip_cur):
        is_active = gm_id < local_num_gm
        eo_skip_enabled = eo_pair_enabled or eo3_enabled
        should_run = (
            jnp.logical_and(is_active, skip_cur == 0) if eo_skip_enabled else is_active
        )
        sem_id = gm_id % 2
        stg_id = gm_id % 4
        next_skip_out = jnp.int32(0)

        if eo3_enabled:
            from .w1_fp8_expert_cache import (
                compute_eo3_batch_count,
                run_eo3_expert_batch,
            )

            batch_n_eo3 = compute_eo3_batch_count(
                gm_id, local_num_gm, metadata_ref, max_batch=eo_buf_4x_ref.shape[0]
            )
            next_skip_out = jnp.where(
                should_run,
                batch_n_eo3 - jnp.int32(1),
                jnp.maximum(skip_cur - jnp.int32(1), jnp.int32(0)),
            )

            @pl.when(should_run)
            def _run_eo3():
                run_eo3_expert_batch(
                    gm_id=gm_id,
                    batch_n=batch_n_eo3,
                    local_num_gm=local_num_gm,
                    gm_id_ref=gm_id_ref,
                    metadata_ref=metadata_ref,
                    fused_metadata_ref=fused_metadata_ref,
                    hidden_states_ref=hidden_states_ref,
                    gathered_lhs_2x_ref=gathered_lhs_2x_ref,
                    lhs_indices_ref=lhs_indices_ref,
                    gather_sem_ref=gather_sem_ref,
                    gather_divisor=top_k if pack_indices else 1,
                    packed_u16_lhs=packed_u16_lhs,
                    dma_gather_gm_start_fn=dma_gather_gm_start,
                    dma_gather_gm_wait_fn=dma_gather_gm_wait,
                    start_w1_dma_fn=start_w1_dma,
                    wait_w1_dma_fn=wait_w1_dma,
                    start_w2_dma_fn=start_w2_dma,
                    wait_w2_dma_fn=wait_w2_dma,
                    wait_staging_slot_writes_fn=wait_staging_slot_writes,
                    do_direct_write_fn=_do_direct_write,
                    maybe_start_deferred_cm_wt_send_fn=(
                        _maybe_start_deferred_cm_wt_send
                        if (enable_chipmate and cm_defer_send)
                        else None
                    ),
                    enable_chipmate=enable_chipmate,
                    cm_defer_send=cm_defer_send,
                    defer_writes=defer_writes,
                    w1_buf_ref=w1_buf_ref,
                    w1_scale_buf_ref=w1_scale_buf_ref,
                    w2_buf_ref=w2_buf_ref,
                    w2_scale_buf_ref=w2_scale_buf_ref,
                    w1_gs_gate_ref=w1_gs_gate_ref,
                    w1_gs_up_ref=w1_gs_up_ref,
                    w2_gs_ref=w2_gs_ref,
                    has_w1_gs=has_w1_gs,
                    has_w2_gs=has_w2_gs,
                    pnw_ref=w2_bias_ref if in_kernel_post_norm_size > 0 else None,
                    in_kernel_post_norm_size=in_kernel_post_norm_size,
                    static_fp8_rs_scale=static_fp8_rs_scale,
                    scatter_staging_3x_ref=scatter_staging_3x_ref,
                    eo_w_half_fp8_ref=eo_w_half_fp8_ref,
                    eo_w_colc_ref=eo_w_colc_ref,
                    eo_lhs_fp8_4x_ref=eo_lhs_fp8_4x_ref,
                    eo_lhs_sc_4x_ref=eo_lhs_sc_4x_ref,
                    eo_buf_4x_ref=eo_buf_4x_ref,
                    num_n1=num_n1,
                    num_n2=num_n2,
                    tile_m=tile_m,
                    tile_k1=tile_k1,
                    tile_n1=tile_n1,
                    tile_k2=tile_k2,
                    tile_n2=tile_n2,
                    strip_step=min(
                        max(
                            num_lanes,
                            int(
                                envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N.get() or 256
                            ),
                        ),
                        tile_n1,
                        tile_n2,
                    ),
                    rhs1_qbs=cfgs1.rhs_cfgs.quant_block_size,
                    rhs2_qbs=cfgs2.rhs_cfgs.quant_block_size,
                    packing1=cfgs1.rhs_cfgs.packing,
                    packing2=cfgs2.rhs_cfgs.packing,
                    rhs1_quant_dtype=cfgs1.rhs_cfgs.quant_dtype,
                    rhs2_quant_dtype=cfgs2.rhs_cfgs.quant_dtype,
                    out_dtype=out_dtype,
                    direct_fp8_mode=(
                        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0)
                        >= 4
                    ),
                    use_bf16_inner=(
                        int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get() or 0)
                        in (1, 3)
                    ),
                    k_chunk=int(
                        envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP.get() or 512
                    ),
                    has_ext_pq_lhs=has_ext_pq_lhs,
                    lhs_row_scales_ref=lhs_row_scales_ref,
                    pq_lhs_mode=pq_lhs_mode,
                )

            return next_skip_out

        if eo_pair_enabled:
            _n_slots_eo = metadata_ref.gm_id_to_group_id.shape[0]
            _cur_e_eo = metadata_ref.gm_id_to_group_id[
                jnp.minimum(gm_id, jnp.maximum(local_num_gm - 1, 0))
            ]
            _nxt_e_eo = metadata_ref.gm_id_to_group_id[
                jnp.minimum(gm_id + 1, jnp.maximum(local_num_gm - 1, 0))
            ]
            can_pair_eo = jnp.logical_and(
                should_run,
                jnp.logical_and(gm_id + 1 < local_num_gm, _nxt_e_eo == _cur_e_eo),
            )
            next_skip_out = jnp.where(can_pair_eo, jnp.int32(1), jnp.int32(0))

            @pl.when(can_pair_eo)
            def _run_eo_pair():
                from .prefill_gmm_opts import (
                    compute_pair_gmm1_n_tile_strip_major,
                    compute_pair_gmm2_n_tile_strip_major,
                    quantize_2d_lhs_tile_to_fp8,
                    quantize_3d_lhs_tile_to_fp8,
                )

                gm_id_ref[0] = gm_id
                expert_id_p = metadata_ref.gm_id_to_group_id[gm_id]
                fused_metadata_ref.gm_id_to_group_id[0] = expert_id_p
                prev_gm_p = jnp.maximum(gm_id - 1, 0)
                prev_e_p = metadata_ref.gm_id_to_group_id[prev_gm_p]
                is_new_e_p = jnp.logical_or(gm_id == 0, prev_e_p != expert_id_p)
                is_same_w1_p = jnp.logical_not(is_new_e_p)
                is_same_w2_p = jnp.logical_not(is_new_e_p)
                _div_p = top_k if pack_indices else 1
                sem0 = sem_id
                sem1 = 1 - sem_id
                stg_prev = (gm_id + 3) % 4
                stg0 = stg_id

                @pl.when(gm_id == 0)
                def _():
                    dma_gather_gm_start(
                        hidden_states_ref,
                        gathered_lhs_2x_ref.at[sem0],
                        lhs_indices_ref,
                        gather_sem_ref.at[sem0],
                        0,
                        metadata_ref,
                        divisor=_div_p,
                    )
                    for _i in range(num_n1):
                        start_w1_dma(_i, expert_id_p, _i, 0)
                    for _i in range(num_n2):
                        start_w2_dma(_i, expert_id_p, _i, 0)

                # Start gather for gm_id + 1 into sem1 immediately.
                dma_gather_gm_start(
                    hidden_states_ref,
                    gathered_lhs_2x_ref.at[sem1],
                    lhs_indices_ref,
                    gather_sem_ref.at[sem1],
                    gm_id + 1,
                    metadata_ref,
                    divisor=_div_p,
                )

                dma_gather_gm_wait(
                    gathered_lhs_2x_ref.at[sem0],
                    gather_sem_ref.at[sem0],
                    gm_id,
                    metadata_ref,
                )
                quantize_3d_lhs_tile_to_fp8(
                    gathered_lhs_2x_ref.at[sem0],
                    eo_lhs_fp8_2x_ref.at[0],
                    eo_lhs_sc_2x_ref.at[0],
                    tile_m=tile_m,
                    tile_k=tile_k1,
                )

                dma_gather_gm_wait(
                    gathered_lhs_2x_ref.at[sem1],
                    gather_sem_ref.at[sem1],
                    gm_id + 1,
                    metadata_ref,
                )
                quantize_3d_lhs_tile_to_fp8(
                    gathered_lhs_2x_ref.at[sem1],
                    eo_lhs_fp8_2x_ref.at[1],
                    eo_lhs_sc_2x_ref.at[1],
                    tile_m=tile_m,
                    tile_k=tile_k1,
                )

                has_g2 = gm_id + 2 < local_num_gm
                g2_clamped = jnp.minimum(gm_id + 2, local_num_gm - 1)
                g2_e = metadata_ref.gm_id_to_group_id[g2_clamped]

                @pl.when(has_g2)
                def _():
                    dma_gather_gm_start(
                        hidden_states_ref,
                        gathered_lhs_2x_ref.at[sem0],
                        lhs_indices_ref,
                        gather_sem_ref.at[sem0],
                        gm_id + 2,
                        metadata_ref,
                        divisor=_div_p,
                    )

                if defer_writes:

                    @pl.when(gm_id > 0)
                    def _():
                        wait_staging_slot_writes(stg_prev)

                @pl.when(gm_id > 0)
                def _():
                    _reshape_gmm2_output(sem1, stg_prev)

                gs_g_val = (
                    w1_gs_gate_ref[expert_id_p].astype(jnp.float32)
                    if has_w1_gs
                    else jnp.float32(1.0)
                )
                gs_u_val = (
                    w1_gs_up_ref[expert_id_p].astype(jnp.float32)
                    if has_w1_gs
                    else jnp.float32(1.0)
                )
                eo_strip_w = eo_w_strip_fp8_ref.shape[1]

                for _n1 in range(num_n1):

                    @pl.when(jnp.logical_not(is_same_w1_p))
                    def _(_b=_n1):
                        wait_w1_dma(_b)

                    compute_pair_gmm1_n_tile_strip_major(
                        lhs_fp8_pair_ref=eo_lhs_fp8_2x_ref,
                        lhs_sc_pair_ref=eo_lhs_sc_2x_ref,
                        w1_tile_ref=w1_buf_ref.at[_n1],
                        w1_scale_tile_ref=w1_scale_buf_ref.at[_n1],
                        w_strip_fp8_ref=eo_w_strip_fp8_ref,
                        gmm1_pair_ref=eo_gmm1_2x_ref,
                        n_id=_n1,
                        tile_m=tile_m,
                        tile_k1=tile_k1,
                        tile_n1=tile_n1,
                        strip_step=eo_strip_w,
                        rhs_qbs=cfgs1.rhs_cfgs.quant_block_size,
                        packing=cfgs1.rhs_cfgs.packing,
                        rhs_quant_dtype=cfgs1.rhs_cfgs.quant_dtype,
                        gs_gate_val=gs_g_val,
                        gs_up_val=gs_u_val,
                        out_dtype=out_dtype,
                    )

                    @pl.when(jnp.logical_and(has_g2, g2_e != expert_id_p))
                    def _(_b=_n1):
                        start_w1_dma(_b, g2_e, _b, 0)

                @pl.when(gm_id > 0)
                def _():
                    _do_direct_write(gm_id - 1, stg_prev)

                if enable_chipmate and cm_defer_send:

                    @pl.when(gm_id == 0)
                    def _():
                        _maybe_start_deferred_cm_wt_send()

                for _pi in range(2):
                    quantize_2d_lhs_tile_to_fp8(
                        eo_gmm1_2x_ref.at[_pi],
                        eo_lhs_fp8_2x_ref.at[_pi],
                        eo_lhs_sc_2x_ref.at[_pi],
                        tile_m=tile_m,
                        tile_k=tile_k2,
                    )

                if defer_writes:
                    wait_staging_slot_writes(stg0)

                gs_w2_val = (
                    w2_gs_ref[expert_id_p].astype(jnp.float32)
                    if has_w2_gs
                    else jnp.float32(1.0)
                )
                for _n2 in range(num_n2):

                    @pl.when(jnp.logical_not(is_same_w2_p))
                    def _(_b=_n2):
                        wait_w2_dma(_b)

                    compute_pair_gmm2_n_tile_strip_major(
                        lhs_fp8_pair_ref=eo_lhs_fp8_2x_ref,
                        lhs_sc_pair_ref=eo_lhs_sc_2x_ref,
                        w2_tile_ref=w2_buf_ref.at[_n2],
                        w2_scale_tile_ref=w2_scale_buf_ref.at[_n2],
                        w_strip_fp8_ref=eo_w_strip_fp8_ref,
                        tiled_out_2x_ref=tiled_out_2x_ref,
                        sem0=sem0,
                        sem1=sem1,
                        n_id=_n2,
                        tile_m=tile_m,
                        tile_k2=tile_k2,
                        tile_n2=tile_n2,
                        strip_step=eo_strip_w,
                        rhs_qbs=cfgs2.rhs_cfgs.quant_block_size,
                        packing=cfgs2.rhs_cfgs.packing,
                        rhs_quant_dtype=cfgs2.rhs_cfgs.quant_dtype,
                        gs_w2_val=gs_w2_val,
                        pnw_ref=w2_bias_ref if in_kernel_post_norm_size > 0 else None,
                        out_dtype=out_dtype,
                    )

                    @pl.when(jnp.logical_and(has_g2, g2_e != expert_id_p))
                    def _(_b=_n2):
                        start_w2_dma(_b, g2_e, _b, 0)

                _reshape_gmm2_output(sem0, stg0)
                _do_direct_write(gm_id, stg0)

            should_run_single = jnp.logical_and(
                should_run, jnp.logical_not(can_pair_eo)
            )
        else:
            should_run_single = should_run

        # Steps 1-9: Only when this chip still has compute work.
        @pl.when(should_run_single)
        def _():
            gm_id_ref[0] = gm_id
            expert_id = metadata_ref.gm_id_to_group_id[gm_id]
            fused_metadata_ref.gm_id_to_group_id[0] = expert_id
            _n_slots_gm = metadata_ref.gm_id_to_group_id.shape[0]
            m_st = metadata_ref.gm_id_to_m_offset[gm_id]
            m_en = (
                metadata_ref.gm_id_to_m_offset[_n_slots_gm + gm_id]
                if metadata_ref.gm_id_to_m_offset.shape[0] == 2 * _n_slots_gm
                else metadata_ref.gm_id_to_m_offset[gm_id + 1]
            )
            num_valid = m_en - m_st
            fused_metadata_ref.gm_id_to_m_offset[0] = jnp.int32(0)
            fused_metadata_ref.gm_id_to_m_offset[1] = num_valid
            has_next = gm_id + 1 < local_num_gm
            next_gm_clamped = jnp.minimum(gm_id + 1, local_num_gm - 1)
            next_e = metadata_ref.gm_id_to_group_id[next_gm_clamped]
            is_hot_cur = (
                (hot_slot >= 0) & (expert_id == hot_slot)
                if hot_interleave
                else jnp.bool_(False)
            )
            is_hot_next = (
                (hot_slot >= 0) & (next_e == hot_slot)
                if hot_interleave
                else jnp.bool_(False)
            )

            # DMA gather. When indices are packed, set divisor=top_k so the
            # gather extracts the actual lhs_idx via integer division.
            _gather_divisor = top_k if pack_indices else 1

            if not _two_ahead_gather:

                @jax.named_scope("dma_gather_start")
                @pl.when(gm_id == 0)
                def _():
                    dma_gather_gm_start(
                        hidden_states_ref,
                        gathered_lhs_2x_ref.at[sem_id],
                        lhs_indices_ref,
                        gather_sem_ref.at[sem_id],
                        0,
                        metadata_ref,
                        divisor=_gather_divisor,
                        packed_u16=packed_u16_lhs,
                        **_inter32_gather_kw,
                    )

            # --- Weight DMA (overlapped with gather) ---
            # w1 prologue: load first num_w1_bufs tiles (gm==0 only;
            # for gm>0, tiles were prefetched during previous gm's GMM2).
            total_w1_steps = num_n1 * num_k1
            total_w2_steps = num_n2 * num_k2

            can_cache_w1 = num_w1_bufs >= total_w1_steps
            can_cache_w2 = num_w2_bufs >= total_w2_steps

            # Read previous expert id (clamp to gm_id=0 case, masked below).
            prev_gm_clamped = jnp.maximum(gm_id - 1, 0)
            prev_expert = metadata_ref.gm_id_to_group_id[prev_gm_clamped]
            # is_new_expert is True at gm_id=0 OR when expert changes.
            is_new_expert = jnp.logical_or(gm_id == 0, prev_expert != expert_id)
            # is_same_expert as previous gm. Used to skip prefetch+wait.
            is_same_w1 = jnp.logical_and(
                jnp.logical_not(is_new_expert), jnp.bool_(can_cache_w1)
            )
            is_same_w2 = jnp.logical_and(
                jnp.logical_not(is_new_expert), jnp.bool_(can_cache_w2)
            )

            if enable_chipmate and cm_defer_send:
                _cond_early_cm = (
                    (gm_id == 0)
                    if (_wla1 or _wla2)
                    else jnp.logical_and(
                        gm_id == 0,
                        jnp.logical_or(
                            expert_id >= borrowed_slot,
                            (has_next & (next_e >= borrowed_slot))
                            if can_cache_w1
                            else jnp.bool_(False),
                        ),
                    )
                )

                @pl.when(_cond_early_cm)
                def _():
                    _maybe_start_deferred_cm_wt_send()

            if _prologue_w_dma:
                if enable_chipmate:

                    @pl.when(jnp.logical_and(gm_id == 0, expert_id >= borrowed_slot))
                    def _():
                        for _i in range(min(num_w1_bufs, total_w1_steps)):
                            start_w1_dma(_i, expert_id, _i // num_k1, _i % num_k1)
                        for _i in range(min(num_w2_bufs, total_w2_steps)):
                            start_w2_dma(_i, expert_id, _i // num_k2, _i % num_k2)
            else:

                @pl.when(jnp.logical_and(gm_id == 0, jnp.logical_not(is_hot_cur)))
                def _():
                    for _i in range(min(num_w1_bufs, total_w1_steps)):
                        start_w1_dma(_i, expert_id, _i // num_k1, _i % num_k1)

                if hot_interleave:

                    @pl.when(is_hot_cur)
                    def _():
                        _ensure_hot_w1_ready()

                # Early w2 prefetch for gm_id == 0 (for gm_id > 0, prefetched in prev gm's GMM2).
                @pl.when(gm_id == 0)
                def _():
                    for _i in range(min(num_w2_bufs, total_w2_steps)):
                        start_w2_dma(_i, expert_id, _i // num_k2, _i % num_k2)

            if _wla1:

                @pl.when(jnp.logical_and(gm_id > 0, is_new_expert))
                def _advance_w_ring():
                    gm_id_ref[_w_base_idx] = (
                        gm_id_ref[_w_base_idx] + jnp.int32(2)
                    ) % jnp.int32(3)
                    gm_id_ref[_la_started_idx] = jnp.int32(0)

                _w_base = gm_id_ref[_w_base_idx]
                _slot_la = (_w_base + jnp.int32(2)) % jnp.int32(3)
                _e_after_cur = max_num_gm_ref[_next_act_grp_off + expert_id]
                _is_last_tile_of_e = jnp.logical_or(
                    jnp.logical_not(has_next), next_e != expert_id
                )

                @pl.when(
                    jnp.logical_and(_e_after_cur >= 0, gm_id_ref[_la_started_idx] == 0)
                )
                def _start_la_step0():
                    start_w1_dma(_slot_la, _e_after_cur, 0, 0)
                    start_w2_dma(_slot_la, _e_after_cur, 0, 0)
                    gm_id_ref[_la_started_idx] = jnp.int32(1)
            elif _wla2:

                @pl.when(jnp.logical_and(gm_id > 0, is_new_expert))
                def _advance_w_pingpong():
                    gm_id_ref[_w_base_idx] = jnp.int32(1) - gm_id_ref[_w_base_idx]
                    gm_id_ref[_la_started_idx] = jnp.int32(0)

                _w_base = gm_id_ref[_w_base_idx]
                _slot_la2 = jnp.int32(1) - _w_base
                _e_after_cur = max_num_gm_ref[_next_act_grp_off + expert_id]
                _is_last_tile_of_e = jnp.logical_or(
                    jnp.logical_not(has_next), next_e != expert_id
                )

                @pl.when(
                    jnp.logical_and(_e_after_cur >= 0, gm_id_ref[_la_started_idx] == 0)
                )
                def _start_la_pingpong():
                    start_w1_dma(_slot_la2, _e_after_cur, 0, 0)
                    if num_w2_bufs >= 2:
                        start_w2_dma(_slot_la2, _e_after_cur, 0, 0)
                    gm_id_ref[_la_started_idx] = jnp.int32(1)
            else:
                _w_base = None
                _e_after_cur = None
                _is_last_tile_of_e = None

            _prev_sem = 1 - sem_id
            _prev_stg = (gm_id + 3) % 4

            def _emit_prev_reshape():
                if decode_dual_k256 >= 4 and k256_tiles and tile_m > 32:
                    _p_gm = jnp.maximum(gm_id - 1, 0)
                    _p_val = (
                        metadata_ref.gm_id_to_m_offset[_p_gm + 1]
                        - metadata_ref.gm_id_to_m_offset[_p_gm]
                    )
                    lax.cond(
                        gm_id == 0,
                        lambda: jnp.int32(0),
                        lambda: lax.cond(
                            _p_val <= 16,
                            lambda: (
                                _reshape_gmm2_output_rows(_prev_sem, _prev_stg, 16),
                                jnp.int32(0),
                            )[1],
                            lambda: lax.cond(
                                _p_val <= 32,
                                lambda: (
                                    _reshape_gmm2_output_rows(_prev_sem, _prev_stg, 32),
                                    jnp.int32(0),
                                )[1],
                                lambda: lax.cond(
                                    _p_val <= 64,
                                    lambda: (
                                        _reshape_gmm2_output_rows(
                                            _prev_sem, _prev_stg, 64
                                        ),
                                        jnp.int32(0),
                                    )[1],
                                    lambda: (
                                        _reshape_gmm2_output(_prev_sem, _prev_stg),
                                        jnp.int32(0),
                                    )[1],
                                ),
                            ),
                        ),
                    )
                elif decode_dual_k256 == 3 and k256_tiles and tile_m > 32:
                    _p_gm = jnp.maximum(gm_id - 1, 0)
                    _p_val = (
                        metadata_ref.gm_id_to_m_offset[_p_gm + 1]
                        - metadata_ref.gm_id_to_m_offset[_p_gm]
                    )
                    lax.cond(
                        gm_id == 0,
                        lambda: jnp.int32(0),
                        lambda: lax.cond(
                            _p_val <= 32,
                            lambda: (
                                _reshape_gmm2_output_rows(_prev_sem, _prev_stg, 32),
                                jnp.int32(0),
                            )[1],
                            lambda: (
                                _reshape_gmm2_output(_prev_sem, _prev_stg),
                                jnp.int32(0),
                            )[1],
                        ),
                    )
                elif _USE_BUCKETS and _prefill_rb_mode in (2, 3, 4) and tile_m >= 128:
                    _p_gm = jnp.maximum(gm_id - 1, 0)
                    _p_en = (
                        metadata_ref.gm_id_to_m_offset[_n_slots_gm + _p_gm]
                        if metadata_ref.gm_id_to_m_offset.shape[0] == 2 * _n_slots_gm
                        else metadata_ref.gm_id_to_m_offset[_p_gm + 1]
                    )
                    _p_val = _p_en - metadata_ref.gm_id_to_m_offset[_p_gm]
                    lax.cond(
                        gm_id == 0,
                        lambda: jnp.int32(0),
                        lambda: lax.cond(
                            _p_val <= 32,
                            lambda: (
                                _reshape_gmm2_output_rows(_prev_sem, _prev_stg, 32),
                                jnp.int32(0),
                            )[1],
                            lambda: lax.cond(
                                _p_val <= 64,
                                lambda: (
                                    _reshape_gmm2_output_rows(_prev_sem, _prev_stg, 64),
                                    jnp.int32(0),
                                )[1],
                                lambda: (
                                    lax.cond(
                                        _p_val <= 128,
                                        lambda: (
                                            _reshape_gmm2_output_rows(
                                                _prev_sem, _prev_stg, 128
                                            ),
                                            jnp.int32(0),
                                        )[1],
                                        lambda: (
                                            _reshape_gmm2_output(_prev_sem, _prev_stg),
                                            jnp.int32(0),
                                        )[1],
                                    )
                                    if tile_m > 128
                                    else (
                                        _reshape_gmm2_output(_prev_sem, _prev_stg),
                                        jnp.int32(0),
                                    )[1]
                                ),
                            ),
                        ),
                    )
                else:

                    @pl.when(gm_id > 0)
                    def _():
                        _reshape_gmm2_output(_prev_sem, _prev_stg)

            if _early_rs:
                if defer_writes:

                    @pl.when(gm_id > 0)
                    def _():
                        wait_staging_slot_writes(_prev_stg)

                _emit_prev_reshape()

                @pl.when(gm_id > 0)
                def _():
                    _do_direct_write(gm_id - 1, _prev_stg)

            # Wait for gather (weight DMAs running in parallel).
            dma_gather_gm_wait(
                gathered_lhs_2x_ref.at[sem_id],
                gather_sem_ref.at[sem_id],
                gm_id,
                metadata_ref,
            )

            if lhs_row_scales_ref is not None:
                from .prefill_gmm_opts import fill_pq_lhs_scales_tree

                _inter32_scale_kw = (
                    {
                        "bs_meta_ref": topk_indices_ref,
                        "gm_id": gm_id,
                        "n_slots_gm": _n_slots_gm,
                        "ep_size": ep_size,
                        "tile_idx_smem_ref": tile_idx_smem_ref,
                    }
                    if inter32_enabled
                    else {}
                )
                fill_pq_lhs_scales_tree(
                    partial_out2_ref,
                    lhs_row_scales_ref,
                    lhs_indices_ref,
                    m_st=m_st,
                    m_en=m_en,
                    num_valid=num_valid,
                    num_lanes=num_lanes,
                    packed_u16_lhs=packed_u16_lhs,
                    gather_divisor=_gather_divisor,
                    scale_b_id=0,
                    unroll_m_rows=(tile_m if pq_lhs_mode >= 3 else None),
                    **_inter32_scale_kw,
                )

            _tile_real = num_valid
            if _USE_BUCKETS:
                # Index of the smallest bucket >= real rows (no captured array
                # constant: sum scalar comparisons against Python-int buckets).
                bucket_idx = jnp.int32(0)
                for _b in M_BUCKETS:
                    bucket_idx += (_tile_real > _b).astype(jnp.int32)
                bucket_idx = jnp.minimum(bucket_idx, len(M_BUCKETS) - 1)

            def _dispatch(fn, *args):
                if use_blockdiag and not k256_tiles:
                    # Tiles of at most 8 valid rows take the block-diagonal branch
                    # of the next row count up; larger ones the row buckets.
                    bd_idx = jnp.int32(0)
                    for _r in blockdiag_rows[:-1]:
                        bd_idx += (_tile_real > _r).astype(jnp.int32)
                    idx = jnp.where(
                        _tile_real <= blockdiag_rows[-1],
                        bd_idx,
                        len(blockdiag_rows) + bucket_idx,
                    )
                    branches = [
                        (
                            lambda r=r: (
                                fn(*args, M_BUCKETS[0], bd_rows=r),
                                jnp.int32(0),
                            )[1]
                        )
                        for r in blockdiag_rows
                    ] + [
                        (lambda b=b: (fn(*args, b), jnp.int32(0))[1]) for b in M_BUCKETS
                    ]
                    lax.switch(idx, branches)
                elif _USE_BUCKETS:
                    lax.switch(
                        bucket_idx,
                        [
                            (lambda b=b: (fn(*args, b), jnp.int32(0))[1])
                            for b in M_BUCKETS
                        ],
                    )
                else:
                    fn(*args, tile_m)

            def _copy_lhs_2d_cache():
                for _mc in range(tile_m // 32):
                    for _kc in range(tile_k1 // 1024):
                        partial_out1_ref[
                            _mc * 32 : (_mc + 1) * 32, _kc * 1024 : (_kc + 1) * 1024
                        ] = gathered_lhs_2x_ref[
                            sem_id,
                            _mc * 32 : (_mc + 1) * 32,
                            _kc * 8 : (_kc + 1) * 8,
                            :,
                        ].reshape(32, 1024)

            def _copy_lhs_2d_cache_rows(m_rows_c: int):
                if m_rows_c == 16:
                    for _kc in range(tile_k1 // 1024):
                        partial_out1_ref[0:16, _kc * 1024 : (_kc + 1) * 1024] = (
                            gathered_lhs_2x_ref[
                                sem_id, 0:16, _kc * 8 : (_kc + 1) * 8, :
                            ].reshape(16, 1024)
                        )
                else:
                    for _mc in range(m_rows_c // 32):
                        for _kc in range(tile_k1 // 1024):
                            partial_out1_ref[
                                _mc * 32 : (_mc + 1) * 32,
                                _kc * 1024 : (_kc + 1) * 1024,
                            ] = gathered_lhs_2x_ref[
                                sem_id,
                                _mc * 32 : (_mc + 1) * 32,
                                _kc * 8 : (_kc + 1) * 8,
                                :,
                            ].reshape(32, 1024)

            if (
                decode_dual_k256 >= 4
                and use_lhs_2d_cache
                and k256_tiles
                and tile_m > 32
            ):
                lax.cond(
                    _tile_real <= 16,
                    lambda: (_copy_lhs_2d_cache_rows(16), jnp.int32(0))[1],
                    lambda: lax.cond(
                        _tile_real <= 32,
                        lambda: (_copy_lhs_2d_cache_rows(32), jnp.int32(0))[1],
                        lambda: lax.cond(
                            _tile_real <= 64,
                            lambda: (_copy_lhs_2d_cache_rows(64), jnp.int32(0))[1],
                            lambda: (_copy_lhs_2d_cache(), jnp.int32(0))[1],
                        ),
                    ),
                )
            elif (
                decode_dual_k256 == 3
                and use_lhs_2d_cache
                and k256_tiles
                and tile_m > 32
            ):
                lax.cond(
                    _tile_real <= 32,
                    lambda: (_copy_lhs_2d_cache_rows(32), jnp.int32(0))[1],
                    lambda: (_copy_lhs_2d_cache(), jnp.int32(0))[1],
                )
            elif use_lhs_2d_cache and use_blockdiag and not k256_tiles:
                # Block-diagonal tiles read the gathered rows directly.
                pl.when(_tile_real > blockdiag_rows[-1])(_copy_lhs_2d_cache)
            elif use_lhs_2d_cache:
                _copy_lhs_2d_cache()

            if defer_writes and not _early_rs:

                @pl.when(gm_id > 0)
                def _():
                    wait_staging_slot_writes(_prev_stg)

            # GMM1 loop.
            for step in range(total_w1_steps):
                _n1 = step // num_k1
                _k1 = step % num_k1
                buf_id = (
                    _w_base
                    if _wla2
                    else (
                        (_w_base + jnp.int32(step)) % jnp.int32(3)
                        if _wla1
                        else (step % num_w1_bufs)
                    )
                )

                @pl.when(
                    jnp.logical_and(
                        jnp.logical_not(is_hot_cur), jnp.logical_not(is_same_w1)
                    )
                )
                def _(_b=buf_id):
                    wait_w1_dma(_b)

                if step == 0 and not _early_rs:
                    _emit_prev_reshape()

                _dispatch(compute_gmm1_tile, buf_id, _n1, _k1, gm_id)
                if _gather_during_gmm1 and step == 0:

                    @jax.named_scope("dma_gather_prefetch_2ahead_gmm1")
                    @pl.when(gm_id + 2 < local_num_gm)
                    def _():
                        dma_gather_gm_start(
                            hidden_states_ref,
                            gathered_lhs_2x_ref.at[sem_id],
                            lhs_indices_ref,
                            gather_sem_ref.at[sem_id],
                            gm_id + 2,
                            metadata_ref,
                            divisor=_gather_divisor,
                            packed_u16=packed_u16_lhs,
                            **_inter32_gather_kw,
                        )

                if step + num_w1_bufs < total_w1_steps:
                    ns = step + num_w1_bufs

                    @pl.when(jnp.logical_not(is_hot_cur))
                    def _(_ns=ns):
                        start_w1_dma(
                            _ns % num_w1_bufs, expert_id, _ns // num_k1, _ns % num_k1
                        )

                if step == 0 and _wla1:

                    @pl.when(jnp.logical_and(_is_last_tile_of_e, _e_after_cur >= 0))
                    def _():
                        start_w1_dma(_w_base, _e_after_cur, 1 // num_k1, 1 % num_k1)
                elif step == 0 and can_cache_w1 and not _wla2:

                    @pl.when(
                        jnp.logical_and(
                            has_next,
                            jnp.logical_and(
                                jnp.logical_not(is_hot_next), next_e != expert_id
                            ),
                        )
                    )
                    def _():
                        start_w1_dma(0, next_e, 0, 0)

            if not _early_rs:

                @pl.when(gm_id > 0)
                def _():
                    _do_direct_write(gm_id - 1, _prev_stg)

            if enable_chipmate and cm_defer_send and not (_wla1 or _wla2):

                @pl.when(gm_id == 0)
                def _():
                    _maybe_start_deferred_cm_wt_send()

            if _two_ahead_gather:

                @jax.named_scope("dma_gather_prefetch_2ahead")
                @pl.when(gm_id + 2 < local_num_gm)
                def _():
                    if not _gather_during_gmm1:
                        dma_gather_gm_start(
                            hidden_states_ref,
                            gathered_lhs_2x_ref.at[sem_id],
                            lhs_indices_ref,
                            gather_sem_ref.at[sem_id],
                            gm_id + 2,
                            metadata_ref,
                            divisor=_gather_divisor,
                            packed_u16=packed_u16_lhs,
                            **_inter32_gather_kw,
                        )
            else:

                @jax.named_scope("dma_gather_prefetch")
                @pl.when(has_next)
                def _():
                    dma_gather_gm_start(
                        hidden_states_ref,
                        gathered_lhs_2x_ref.at[1 - sem_id],
                        lhs_indices_ref,
                        gather_sem_ref.at[1 - sem_id],
                        gm_id + 1,
                        metadata_ref,
                        divisor=_gather_divisor,
                        packed_u16=packed_u16_lhs,
                        **_inter32_gather_kw,
                    )

            if not _wla1 and not _wla2:

                @pl.when(has_next)
                def _():
                    next_same = jnp.logical_and(
                        next_e == expert_id, jnp.bool_(can_cache_w1)
                    )

                    @pl.when(
                        jnp.logical_and(
                            jnp.logical_not(is_hot_next), jnp.logical_not(next_same)
                        )
                    )
                    def _():
                        for _i in range(
                            1 if can_cache_w1 else 0, min(num_w1_bufs, total_w1_steps)
                        ):
                            start_w1_dma(_i, next_e, _i // num_k1, _i % num_k1)

            if dims.size_k2 != gmm1_out_ref.shape[1]:
                with jax.named_scope("interlude"):

                    def _interlude(m_rows):
                        gmm1_result = gmm1_out_ref[pl.ds(0, m_rows)]
                        k2_pad = dims.size_k2 - gmm1_result.shape[1]
                        if k2_pad > 0:
                            gmm1_result = jnp.concatenate(
                                [
                                    gmm1_result,
                                    jnp.zeros(
                                        (m_rows, k2_pad), dtype=gmm1_result.dtype
                                    ),
                                ],
                                axis=1,
                            )
                        m_sub = m_rows // dims.size_lhs_sublane
                        intermediate_ref[pl.ds(0, m_sub)] = gmm1_result.astype(
                            out_dtype
                        ).reshape(m_sub, dims.size_lhs_sublane, dims.size_k2)

                    _dispatch(_interlude)

            # GMM2 loop.
            for step in range(total_w2_steps):
                _n2 = step // num_k2
                _k2 = step % num_k2
                buf_id = (
                    (_w_base if num_w2_bufs >= 2 else 0)
                    if _wla2
                    else (
                        (_w_base + jnp.int32(step)) % jnp.int32(3)
                        if _wla1
                        else (step % num_w2_bufs)
                    )
                )

                # Skip wait if W2 was cached (same expert as prev + buffers fit all).
                @pl.when(jnp.logical_not(is_same_w2))
                def _(_b2=buf_id):
                    wait_w2_dma(_b2)
                    if partial_out1_ref.ndim == 3:
                        prequantize_rhs_to_vmem(
                            w2_buf_ref.at[_b2],
                            w2_scale_buf_ref.at[_b2],
                            partial_out1_ref.at[_b2],
                            partial_out2_ref.at[_b2],
                            cfgs=cfgs2,
                        )
                    elif (
                        w2_fp8_cache_ref is not None
                        and not use_weight_lookahead
                        and _n2 < w2_fp8_cache_ref.shape[0]
                    ):
                        _w2c_lim = w2_fp8_cache_ref.shape[-1]
                        _k_w2c = 256 if tile_k2 % 256 == 0 else min(tile_k2, 1024)
                        _n_w2c = 512 if _w2c_lim % 512 == 0 else min(_w2c_lim, 256)
                        for _ck in range(0, tile_k2, _k_w2c):
                            _ck_len = min(_k_w2c, tile_k2 - _ck)
                            for _cn in range(0, _w2c_lim, _n_w2c):
                                _cn_len = min(_n_w2c, _w2c_lim - _cn)
                                _raw_c = w2_buf_ref[
                                    _b2,
                                    pl.ds(
                                        _ck // cfgs2.rhs_cfgs.packing,
                                        _ck_len // cfgs2.rhs_cfgs.packing,
                                    ),
                                    pl.ds(_cn, _cn_len),
                                ]
                                w2_fp8_cache_ref[
                                    _b2, pl.ds(_ck, _ck_len), pl.ds(_cn, _cn_len)
                                ] = pltpu.bitcast(
                                    _raw_c, cfgs2.rhs_cfgs.quant_dtype
                                ).astype(jnp.float8_e4m3fn)

                # See GMM1 loop: consume the slot before refilling it (WAR).
                _dispatch(compute_gmm2_tile, buf_id, _n2, _k2, gm_id)
                if step + num_w2_bufs < total_w2_steps:
                    ns = step + num_w2_bufs
                    start_w2_dma(ns % num_w2_bufs, expert_id, ns // num_k2, ns % num_k2)
                if step == 0 and _wla1:

                    @pl.when(jnp.logical_and(_is_last_tile_of_e, _e_after_cur >= 0))
                    def _():
                        start_w2_dma(_w_base, _e_after_cur, 1 // num_k2, 1 % num_k2)
                elif step == 0 and can_cache_w2 and not (_wla2 and num_w2_bufs >= 2):

                    @pl.when(jnp.logical_and(has_next, next_e != expert_id))
                    def _():
                        start_w2_dma(0, next_e, 0, 0)

            if not _wla1 and not _wla2:

                @pl.when(
                    jnp.logical_and(
                        has_next,
                        jnp.logical_not(
                            jnp.logical_and(
                                next_e == expert_id, jnp.bool_(can_cache_w2)
                            )
                        ),
                    )
                )
                def _():
                    for _i in range(
                        1 if can_cache_w2 else 0, min(num_w2_bufs, total_w2_steps)
                    ):
                        start_w2_dma(_i, next_e, _i // num_k2, _i % num_k2)

        return next_skip_out if eo_skip_enabled else skip_cur

    lax.fori_loop(
        0,
        local_num_gm,
        gm_loop_body,
        jnp.int32(0) if (eo_pair_enabled or eo3_enabled) else None,
    )

    if enable_chipmate and cm_defer_send:
        _maybe_start_deferred_cm_wt_send()

    @pl.when(local_num_gm > 0)
    def _flush_last_tile():
        if eo3_enabled:
            return
        last_gm = local_num_gm - 1
        last_sem = last_gm % 2
        last_stg = last_gm % 4
        if defer_writes:
            wait_staging_slot_writes(last_stg)
        if decode_dual_k256 >= 4 and k256_tiles and tile_m > 32:
            last_val = (
                metadata_ref.gm_id_to_m_offset[last_gm + 1]
                - metadata_ref.gm_id_to_m_offset[last_gm]
            )
            lax.cond(
                last_val <= 16,
                lambda: (
                    _reshape_gmm2_output_rows(last_sem, last_stg, 16),
                    jnp.int32(0),
                )[1],
                lambda: lax.cond(
                    last_val <= 32,
                    lambda: (
                        _reshape_gmm2_output_rows(last_sem, last_stg, 32),
                        jnp.int32(0),
                    )[1],
                    lambda: lax.cond(
                        last_val <= 64,
                        lambda: (
                            _reshape_gmm2_output_rows(last_sem, last_stg, 64),
                            jnp.int32(0),
                        )[1],
                        lambda: (
                            _reshape_gmm2_output(last_sem, last_stg),
                            jnp.int32(0),
                        )[1],
                    ),
                ),
            )
        elif (
            _USE_BUCKETS
            and _prefill_rb_mode in (2, 3, 4)
            and tile_m >= 128
            and not inter32_enabled
        ):
            _n_sl_last = metadata_ref.gm_id_to_group_id.shape[0]
            _last_en = (
                metadata_ref.gm_id_to_m_offset[_n_sl_last + last_gm]
                if metadata_ref.gm_id_to_m_offset.shape[0] == 2 * _n_sl_last
                else metadata_ref.gm_id_to_m_offset[last_gm + 1]
            )
            last_val = _last_en - metadata_ref.gm_id_to_m_offset[last_gm]
            lax.cond(
                last_val <= 32,
                lambda: (
                    _reshape_gmm2_output_rows(last_sem, last_stg, 32),
                    jnp.int32(0),
                )[1],
                lambda: lax.cond(
                    last_val <= 64,
                    lambda: (
                        _reshape_gmm2_output_rows(last_sem, last_stg, 64),
                        jnp.int32(0),
                    )[1],
                    lambda: (
                        lax.cond(
                            last_val <= 128,
                            lambda: (
                                _reshape_gmm2_output_rows(last_sem, last_stg, 128),
                                jnp.int32(0),
                            )[1],
                            lambda: (
                                _reshape_gmm2_output(last_sem, last_stg),
                                jnp.int32(0),
                            )[1],
                        )
                        if tile_m > 128
                        else (_reshape_gmm2_output(last_sem, last_stg), jnp.int32(0))[1]
                    ),
                ),
            )
        else:
            _reshape_gmm2_output(last_sem, last_stg)
        _do_direct_write(last_gm, last_stg)

    if block_scatter:
        _n_sl_bs = metadata_ref.gm_id_to_group_id.shape[0]
        _meta_last = (
            _n_sl_bs * (3 * ep_size + 2)
            if topk_indices_ref.ndim == 1
            else topk_indices_ref.shape[0] - 1
        )
        if rotate_rs_mode >= 1 and (ep_size & (ep_size - 1)) == 0:
            if inter32_enabled:

                def _sig_ex(_step, _u):
                    _tgt_ex = lax.bitwise_xor(my_id, _step)
                    _extra = (
                        topk_indices_ref[_meta_last + _tgt_ex]
                        if topk_indices_ref.ndim == 1
                        else topk_indices_ref[_meta_last, _tgt_ex]
                    )

                    @pl.when(_extra > 0)
                    def _():
                        pl.semaphore_signal(
                            recv_sem_ref.at[0],
                            _extra,
                            device_id={ep_axis_name: _tgt_ex},
                            device_id_type=pl.DeviceIdType.MESH,
                        )

                    return _u

                lax.fori_loop(1, ep_size, _sig_ex, jnp.int32(0))
            else:
                for _step in range(1, ep_size):
                    _tgt_ex = lax.bitwise_xor(my_id, jnp.int32(_step))
                    _extra = (
                        topk_indices_ref[_meta_last + _tgt_ex]
                        if topk_indices_ref.ndim == 1
                        else topk_indices_ref[_meta_last, _tgt_ex]
                    )

                    @pl.when(_extra > 0)
                    def _(_target=_tgt_ex, _delta=_extra):
                        pl.semaphore_signal(
                            recv_sem_ref.at[0],
                            _delta,
                            device_id={ep_axis_name: _target},
                            device_id_type=pl.DeviceIdType.MESH,
                        )
        else:
            for _c in range(ep_size):
                _extra = (
                    topk_indices_ref[_meta_last + _c]
                    if topk_indices_ref.ndim == 1
                    else topk_indices_ref[_meta_last, _c]
                )

                @pl.when(jnp.logical_and(_c != my_id, _extra > 0))
                def _(_target=_c, _delta=_extra):
                    pl.semaphore_signal(
                        recv_sem_ref.at[0],
                        _delta,
                        device_id={ep_axis_name: jnp.int32(_target)},
                        device_id_type=pl.DeviceIdType.MESH,
                    )

    if defer_writes:
        # Drain the active staging slots before the cross-rank barrier.
        if inter32_enabled:
            lax.fori_loop(
                0,
                num_stg_slots,
                lambda _s, _u: (wait_staging_slot_writes(_s), _u)[1],
                jnp.int32(0),
            )
        else:
            for _s in range(num_stg_slots):
                wait_staging_slot_writes(_s)

    if hot_interleave:

        @pl.when(jnp.logical_and(local_num_gm > 0, hot_slot >= 0))
        def _drain_hot_w1():
            _ensure_hot_w1_ready()

    if enable_chipmate:
        for _ci in range(eff_cm_slots):
            _cm_partner_d = lax.bitwise_xor(my_id, jnp.int32(_cm_slot_xors[_ci]))
            _s_slot_d = max_num_gm_ref[1 + 2 * _ci]
            _r_flag_d = max_num_gm_ref[2 + 2 * _ci]
            _sb_d = 4 * _ci
            _w1_d = _cm_pad_view(cm_w1_pad, _ci)
            _s1_d = _cm_pad_view(cm_s1_pad, _ci)
            _w2_d = _cm_pad_view(cm_w2_pad, _ci)
            _s2_d = _cm_pad_view(cm_s2_pad, _ci)

            @pl.when(_s_slot_d >= 0)
            def _drain_cm_wt_send(
                _ss=_s_slot_d,
                _b=_sb_d,
                _dw1=_w1_d,
                _ds1=_s1_d,
                _dw2=_w2_d,
                _ds2=_s2_d,
                _pt=_cm_partner_d,
            ):
                pltpu.make_async_remote_copy(
                    src_ref=w1_packed.at[_ss],
                    dst_ref=_dw1,
                    send_sem=cm_wt_send_sem_ref.at[_b + 0],
                    recv_sem=cm_wt_recv_sem_ref.at[_b + 0],
                    device_id={ep_axis_name: _pt},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                if has_scale:
                    pltpu.make_async_remote_copy(
                        src_ref=w1_scale_ref.at[_ss],
                        dst_ref=_ds1,
                        send_sem=cm_wt_send_sem_ref.at[_b + 1],
                        recv_sem=cm_wt_recv_sem_ref.at[_b + 1],
                        device_id={ep_axis_name: _pt},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()
                pltpu.make_async_remote_copy(
                    src_ref=w2_packed.at[_ss],
                    dst_ref=_dw2,
                    send_sem=cm_wt_send_sem_ref.at[_b + 2],
                    recv_sem=cm_wt_recv_sem_ref.at[_b + 2],
                    device_id={ep_axis_name: _pt},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                if has_scale:
                    pltpu.make_async_remote_copy(
                        src_ref=w2_scale_ref.at[_ss],
                        dst_ref=_ds2,
                        send_sem=cm_wt_send_sem_ref.at[_b + 3],
                        recv_sem=cm_wt_recv_sem_ref.at[_b + 3],
                        device_id={ep_axis_name: _pt},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait_send()

            @pl.when(_r_flag_d > 0)
            def _drain_cm_wt_recv(_idx=_ci):
                _ensure_cm_wt_ready(_idx)

    # Post-Scatter Cross-Rank Optical Barrier
    sync_barrier()

    # Drain incoming receive semaphores using in-kernel computed total_recv
    @pl.when(total_recv > 0)
    def _():
        pl.semaphore_wait(recv_sem_ref.at[0], total_recv)
        if fp8_direct_write:
            pl.semaphore_wait(scale_recv_sem_ref.at[0], total_recv)

    # All ICI DMA writes go directly to out_buf_ref (the pallas_call output).
    # No input_output_aliases needed — avoids XLA tiling copy on weight inputs.


def kernel_main_fused_rs_fp8(
    lhs_group_sizes_ref,
    group_offset_ref,
    group_start_ref,
    lhs_indices_ref,
    topk_indices_ref,
    max_num_gm_ref,
    w1_gs_gate_ref,
    w1_gs_up_ref,
    w2_gs_ref,
    hidden_states_ref,
    w1_ref,
    w2_ref,
    w1_scale_ref,
    w2_scale_ref,
    w1_bias_ref,
    w2_bias_ref,
    out_buf_ref,
    out_scale_ref,
    metadata_ref,
    fused_metadata_ref,
    gathered_lhs_2x_ref,
    gmm1_out_ref,
    intermediate_ref,
    tiled_out_2x_ref,
    scatter_staging_3x_ref,
    partial_out1_ref,
    partial_out2_ref,
    shared_acc_ref,
    gather_sem_ref,
    staging_sem_ref,
    zero_sem_ref,
    gm_id_ref,
    w1_buf_ref,
    w2_buf_ref,
    w1_scale_buf_ref,
    w2_scale_buf_ref,
    w1_bias_buf_ref,
    w2_bias_buf_ref,
    w1_sem_ref,
    w2_sem_ref,
    output_sem_ref,
    send_sems_ref,
    local_write_sems_ref,
    recv_sem_ref,
    scale_send_sems_ref,
    scale_local_write_sems_ref,
    scale_recv_sem_ref,
    scatter_fp8_staging_3x_ref,
    scatter_scale_3x_ref,
    *,
    fused_dims: FusedDims,
    tile_m: int,
    tile_k1: int,
    tile_k2: int,
    tile_n1: int,
    tile_n2: int,
    num_w1_bufs: int,
    num_w2_bufs: int,
    act_fn: str,
    out_dtype: jnp.dtype,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    ep_size: int,
    chunk_size: int,
    ep_axis_name: str,
    top_k: int,
    pack_indices: bool = True,
    defer_writes: bool = False,
):
    """FP8 direct-write wrapper with output refs ordered for pallas_call."""
    return kernel_main_fused_rs(
        lhs_group_sizes_ref,
        group_offset_ref,
        group_start_ref,
        lhs_indices_ref,
        topk_indices_ref,
        max_num_gm_ref,
        w1_gs_gate_ref,
        w1_gs_up_ref,
        w2_gs_ref,
        hidden_states_ref,
        w1_ref,
        w2_ref,
        w1_scale_ref,
        w2_scale_ref,
        w1_bias_ref,
        w2_bias_ref,
        out_buf_ref,
        metadata_ref,
        fused_metadata_ref,
        gathered_lhs_2x_ref,
        gmm1_out_ref,
        intermediate_ref,
        tiled_out_2x_ref,
        scatter_staging_3x_ref,
        partial_out1_ref,
        partial_out2_ref,
        shared_acc_ref,
        gather_sem_ref,
        staging_sem_ref,
        zero_sem_ref,
        gm_id_ref,
        w1_buf_ref,
        w2_buf_ref,
        w1_scale_buf_ref,
        w2_scale_buf_ref,
        w1_bias_buf_ref,
        w2_bias_buf_ref,
        w1_sem_ref,
        w2_sem_ref,
        output_sem_ref,
        send_sems_ref,
        local_write_sems_ref,
        recv_sem_ref,
        scale_send_sems_ref,
        scale_local_write_sems_ref,
        scale_recv_sem_ref,
        scatter_fp8_staging_3x_ref,
        scatter_scale_3x_ref,
        out_scale_ref,
        fused_dims=fused_dims,
        tile_m=tile_m,
        tile_k1=tile_k1,
        tile_k2=tile_k2,
        tile_n1=tile_n1,
        tile_n2=tile_n2,
        num_w1_bufs=num_w1_bufs,
        num_w2_bufs=num_w2_bufs,
        act_fn=act_fn,
        out_dtype=out_dtype,
        cfgs1=cfgs1,
        cfgs2=cfgs2,
        ep_size=ep_size,
        chunk_size=chunk_size,
        ep_axis_name=ep_axis_name,
        top_k=top_k,
        pack_indices=pack_indices,
        fp8_direct_write=True,
        defer_writes=defer_writes,
    )


# =============================================================================
# Phase 3: gmm_v2_fused_rs — Public API
# =============================================================================


@jax.jit(
    static_argnames=[
        "act_fn",
        "output_size",
        "vmem_limit_bytes",
        "ep_size",
        "ep_axis_name",
        "top_k",
        "fp8_direct_write",
        "caller_tile_m",
        "expert_tp",
        "fp4_w4a16",
        "block_scatter",
        "chipmate_xor",
        "hot_interleave",
        "static_fp8_lhs_scale",
    ],
)
def gmm_v2_fused_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    lhs_indices: jax.Array,
    output_indices: jax.Array,
    *,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w1_global_scale: jax.Array | None = None,  # (E, 2) gate/up per-expert
    w2_global_scale: jax.Array | None = None,  # (E,) per-expert
    w1_bias: jax.Array | None = None,
    w2_bias: jax.Array | None = None,
    act_fn: str = "silu",
    group_offset: jax.Array | None = None,
    group_start: jax.Array | None = None,
    output_size: int,
    vmem_limit_bytes: int | None = None,
    topk_indices: jax.Array,
    ep_size: int,
    ep_axis_name: str,
    max_num_gm: jax.Array,
    caller_tile_m: int | None = None,
    top_k: int,
    fp8_direct_write: bool = False,
    expert_tp: bool = False,
    fp4_w4a16: bool = False,
    post_expert_norm_weight: jax.Array | None = None,
    block_scatter: bool = False,
    chipmate_plan: jax.Array | None = None,
    chipmate_xor: int = 1,
    hot_interleave: bool = False,
    hidden_states_scale: jax.Array | None = None,
    static_fp8_lhs_scale: float = 0.0,
) -> jax.Array:
    """Fused gather + GMM1 + act + GMM2 + ICI direct-write.

    fp4_w4a16: block-256 fp4 experts take bf16 rows, each block's scale applied after its
    dot (requantized NVFP4), instead of fp8-quantized rows.

    Output: (chunk_size * top_k, size_n2) per chip — raw expert contributions.
    Post-kernel reduction applies topk_weights and sums over top_k dim.
    """
    stacked = w1.ndim == 4
    size_group_phys, size_k1, size_n1 = w1.shape[1:] if stacked else w1.shape
    enable_chipmate = chipmate_plan is not None
    num_cm_slots = (int(chipmate_plan.shape[0]) // 2) if enable_chipmate else 0
    size_group = size_group_phys + num_cm_slots
    _, size_k2, size_n2 = w2.shape[1:] if stacked else w2.shape
    size_m = lhs_indices.shape[0]
    chunk_size = output_size // ep_size

    is_quantized = w1_scale is not None
    if is_quantized:
        assert w2_scale is not None
        probe_qbs_cfg = int(envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK.get() or 0)
        if (
            probe_qbs_cfg < 0
            and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        ):
            target_qbs = -probe_qbs_cfg
            s_ax = 2 if stacked else 1
            cur_qbs1 = recover_quant_block_size(size_k1, w1_scale.shape[s_ax])
            cur_qbs2 = recover_quant_block_size(size_k2, w2_scale.shape[s_ax])
            if (
                target_qbs in (512, 1024, 2048, 4096)
                and target_qbs > cur_qbs1
                and target_qbs % cur_qbs1 == 0
                and target_qbs > cur_qbs2
                and target_qbs % cur_qbs2 == 0
            ):
                st1 = target_qbs // cur_qbs1
                st2 = target_qbs // cur_qbs2
                w1_scale = w1_scale[:, :, ::st1] if stacked else w1_scale[:, ::st1]
                w2_scale = w2_scale[:, :, ::st2] if stacked else w2_scale[:, ::st2]
        w1s_shape = w1_scale.shape[1:] if stacked else w1_scale.shape
        w2s_shape = w2_scale.shape[1:] if stacked else w2_scale.shape
        rhs1_quant_block_size = recover_quant_block_size(size_k1, w1s_shape[1])
        rhs2_quant_block_size = recover_quant_block_size(size_k2, w2s_shape[1])
        quant_block_size = rhs1_quant_block_size
        # Scales stay in their stored dtype when it is bf16 or fp32: the matmul
        # casts each tile slice, while a full-tensor cast here would re-copy
        # every expert's scales on each call (group-16 scales are large).
        scale_dtype = (
            w1_scale.dtype
            if w1_scale.dtype in (jnp.bfloat16, jnp.float32)
            else jnp.dtype(jnp.float32)
        )
        w1_scale = w1_scale.astype(scale_dtype)
        w2_scale = w2_scale.astype(scale_dtype)
        if len(w1s_shape) != len(w2s_shape) or len(w1s_shape) not in (3, 4):
            raise ValueError(
                "fused-RS scales must both be [E, K_blocks, N] or "
                f"[E, K_blocks, 1, N], got {w1_scale.shape} and {w2_scale.shape}"
            )
        scale_has_unit_dim = len(w1s_shape) == 4
    else:
        rhs1_quant_block_size = size_k1
        rhs2_quant_block_size = size_k2
        quant_block_size = None
        scale_dtype = jnp.dtype(jnp.float32)
        scale_has_unit_dim = True
    # Per-expert global_scale as scalar prefetch (SMEM): one scalar per expert,
    # indexed inside inner_kernel via metadata_ref.gm_id_to_group_id.
    if w1_global_scale is not None:
        if w1_global_scale.ndim == 2:
            _w1_gs_gate = w1_global_scale[:, 0].astype(jnp.float32)
            _w1_gs_up = w1_global_scale[:, 1].astype(jnp.float32)
        else:
            _w1_gs_gate = w1_global_scale.astype(jnp.float32)
            _w1_gs_up = _w1_gs_gate
    else:
        _w1_gs_gate = jnp.ones((size_group,), dtype=jnp.float32)
        _w1_gs_up = _w1_gs_gate
    if w2_global_scale is not None:
        _w2_gs = w2_global_scale.astype(jnp.float32)
    else:
        _w2_gs = jnp.ones((size_group,), dtype=jnp.float32)

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    if group_start is None:
        group_start = jnp.array([0], dtype=jnp.int32)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.95)

    num_lanes = pltpu.get_tpu_info().num_lanes
    from .prefill_gmm_opts import resolve_static_fp8_lhs_scale

    lhs_in_dtype = jnp.dtype(hidden_states.dtype)
    has_ext_lhs_scales = (
        hidden_states_scale is not None and lhs_in_dtype == jnp.float8_e4m3fn
    )
    _raw_static_fp8_lhs = (
        resolve_static_fp8_lhs_scale(None, static_fp8_lhs_scale)
        if static_fp8_lhs_scale != 0.0
        else resolve_static_fp8_lhs_scale(
            envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get(),
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS.get(),
        )
    )
    eff_static_fp8_lhs = (
        _raw_static_fp8_lhs
        if (
            _raw_static_fp8_lhs > 0.0
            and not has_ext_lhs_scales
            and lhs_in_dtype == jnp.float8_e4m3fn
            and is_quantized
            and size_m > _BLOCKDIAG_MAX_SIZE_M
            and not fp8_direct_write
            and not fp4_w4a16
            and (static_fp8_lhs_scale != 0.0 or size_m >= _PREFILL_TILES_MIN_SIZE_M)
        )
        else 0.0
    )
    has_static_fp8_lhs = eff_static_fp8_lhs > 0.0
    if has_static_fp8_lhs:
        _inv_static_fp8_lhs = jnp.float32(1.0 / eff_static_fp8_lhs)
        _w1_gs_gate = _w1_gs_gate * _inv_static_fp8_lhs
        _w1_gs_up = _w1_gs_up * _inv_static_fp8_lhs
    out_dtype = (
        jnp.dtype(jnp.bfloat16)
        if (has_ext_lhs_scales or has_static_fp8_lhs)
        else lhs_in_dtype
    )
    sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
    intermediate_size = size_n1 // 2

    eff_hot_interleave = (
        bool(hot_interleave)
        and is_quantized
        and not fp8_direct_write
        and not expert_tp
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jax.dtypes.itemsize_bits(w1.dtype) in (4, 32)
    )

    # --- Single source of truth for all block/tile sizes ---
    # This helper is pure-Python and deterministic.
    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=size_k1,
        size_n1=size_n1,
        size_k2=size_k2,
        size_n2=size_n2,
        size_group=size_group,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=out_dtype,
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        is_quantized=is_quantized,
        quant_block_size=quant_block_size,
        act_fn=act_fn,
        vmem_limit_bytes=vmem_limit_bytes,
        fp8_direct_write=fp8_direct_write,
        hot_interleave=eff_hot_interleave,
    )
    tile_m = block_sizes.tile_m
    # The caller sizes max_num_gm from an independent block-size lookup. A
    # divergent lookup key can otherwise undercount the loop and leave the
    # final gather DMA unawaited. Both values are static, so fail during trace
    # instead of surfacing later as a leaked semaphore or device hang.
    if caller_tile_m is not None and caller_tile_m != tile_m:
        raise ValueError(
            f"caller computed max_num_gm with tile_m={caller_tile_m}, but the "
            f"kernel selected tile_m={tile_m}; check that quantization, "
            "activation, variant, and FP8 direct-write lookup keys agree"
        )
    tile_k1 = block_sizes.tile_k1
    tile_n1 = block_sizes.tile_n1
    tile_k2 = block_sizes.tile_k2
    tile_n2 = block_sizes.tile_n2
    num_w1_bufs = block_sizes.num_w1_bufs
    num_w2_bufs = block_sizes.num_w2_bufs
    padded_k1 = block_sizes.padded_k1
    aligned_n1 = block_sizes.aligned_n1
    aligned_n2 = block_sizes.aligned_n2
    size_lhs_sublane = min(sls, size_m)

    # --- Apply the padding the helper chose ---
    # Pad K1 for DMA gather alignment.  A caller may pass a pre-tiled
    # [M, K // lanes, lanes] activation when the producer already emitted the
    # GMM-ready layout; keep it tiled instead of flattening and reshaping again.
    hidden_states_is_tiled = hidden_states.ndim == 3
    if hidden_states_is_tiled:
        if hidden_states.shape[2] != num_lanes:
            raise ValueError(
                "tiled hidden_states must have the TPU lane dimension last, got "
                f"{hidden_states.shape} with num_lanes={num_lanes}"
            )
        tiled_k = hidden_states.shape[1] * num_lanes
        if tiled_k < padded_k1:
            k_pad = padded_k1 - tiled_k
            if k_pad % num_lanes:
                raise ValueError(
                    f"cannot pad tiled hidden_states by {k_pad}; num_lanes={num_lanes}"
                )
            hidden_states = jnp.pad(
                hidden_states, ((0, 0), (0, k_pad // num_lanes), (0, 0))
            )
        elif tiled_k > padded_k1:
            if padded_k1 % num_lanes:
                raise ValueError(
                    f"padded_k1 must be divisible by {num_lanes}, got {padded_k1}"
                )
            hidden_states = hidden_states[:, : padded_k1 // num_lanes, :]
    elif padded_k1 != size_k1:
        k_pad = padded_k1 - size_k1
        hidden_states = jnp.pad(hidden_states, ((0, 0), (0, k_pad)))

    # Pad N1 (and propagate to K2 since GMM2's K is GMM1's intermediate size).
    lead_pads = ((0, 0),) if stacked else ()
    s_axis = 2 if stacked else 1
    if aligned_n1 != size_n1:
        n1_pad = aligned_n1 - size_n1
        w1 = jnp.pad(w1, lead_pads + ((0, 0), (0, 0), (0, n1_pad)))
        if is_quantized:
            w1_scale = jnp.pad(
                w1_scale, ((0, 0),) * (w1_scale.ndim - 1) + ((0, n1_pad),)
            )
        new_k2 = aligned_n1 // 2
        if new_k2 != size_k2:
            w2 = jnp.pad(w2, lead_pads + ((0, 0), (0, new_k2 - size_k2), (0, 0)))
            size_k2 = new_k2
        size_n1 = aligned_n1

    # N2 alignment note: aligned_n2 is used for OUTPUT buffer sizing (DMA
    # scatter requires 3D refs at aligned width), but w2 weights do NOT need
    # padding — the kernel iterates num_n2 = original_n2 // tile_n2 tiles,
    # so DMA only accesses w2 columns [0, original_n2).  Removing w2 pad
    # eliminates a large copy+pad (~33% of w2) on every forward pass.

    # Pad scales.
    if is_quantized:
        nqb1 = -(-tile_k1 // rhs1_quant_block_size)
        total_scale_blocks1 = (size_k1 // tile_k1) * nqb1
        if w1_scale.shape[s_axis] < total_scale_blocks1:
            w1_scale = jnp.concatenate(
                [
                    w1_scale,
                    jnp.repeat(
                        w1_scale[:, :, -1:] if stacked else w1_scale[:, -1:],
                        total_scale_blocks1 - w1_scale.shape[s_axis],
                        axis=s_axis,
                    ),
                ],
                axis=s_axis,
            )
        nqb2 = -(-tile_k2 // rhs2_quant_block_size)
        total_scale_blocks2 = (size_k2 // tile_k2) * nqb2
        if w2_scale.shape[s_axis] < total_scale_blocks2:
            w2_scale = jnp.concatenate(
                [
                    w2_scale,
                    jnp.repeat(
                        w2_scale[:, :, -1:] if stacked else w2_scale[:, -1:],
                        total_scale_blocks2 - w2_scale.shape[s_axis],
                        axis=s_axis,
                    ),
                ],
                axis=s_axis,
            )
    else:
        total_scale_blocks1 = 0
        total_scale_blocks2 = 0

    has_bias = w1_bias is not None
    if has_bias:
        assert w2_bias is not None
        if w1_bias.shape[-1] != size_n1:
            w1_bias = jnp.pad(
                w1_bias, ((0, 0), (0, 0), (0, size_n1 - w1_bias.shape[-1]))
            )
        if w2_bias.shape[-1] != size_n2:
            w2_bias = jnp.pad(
                w2_bias, ((0, 0), (0, 0), (0, size_n2 - w2_bias.shape[-1]))
            )
        w1_bias = w1_bias.astype(jnp.float32)
        w2_bias = w2_bias.astype(jnp.float32)

    if hidden_states_is_tiled:
        hidden_3d = hidden_states
    else:
        hidden_3d = hidden_states.reshape(
            hidden_states.shape[0], padded_k1 // num_lanes, num_lanes
        )
    fused_dims = FusedDims(
        size_m=size_m,
        size_k1=size_k1,
        padded_k1=padded_k1,
        size_n1=size_n1,
        size_k2=size_k2,
        size_n2=aligned_n2,
        original_n2=size_n2,
        size_group=size_group,
        size_lhs_group=group_sizes.shape[0],
        size_lhs_sublane=size_lhs_sublane,
        intermediate_size=intermediate_size,
        has_bias=has_bias,
        quant_block_size=quant_block_size,
        num_scale_blocks1=total_scale_blocks1,
        num_scale_blocks2=total_scale_blocks2,
    )

    rhs1_packing = (32 // jax.dtypes.itemsize_bits(w1.dtype)) if is_quantized else 1
    rhs2_packing = (32 // jax.dtypes.itemsize_bits(w2.dtype)) if is_quantized else 1
    lhs1_quant_block_size = min(256 if rhs1_quant_block_size < 512 else 512, size_k1)
    lhs2_quant_block_size = min(256 if rhs2_quant_block_size < 512 else 512, size_k2)
    fp8_w8a16 = False
    if is_quantized and jnp.dtype(w1.dtype) == jnp.float8_e4m3fn:
        fp8_lhs_qbs = envs.SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS.get()
        if fp8_lhs_qbs > 0:
            lhs1_quant_block_size = min(fp8_lhs_qbs, size_k1)
            lhs2_quant_block_size = min(fp8_lhs_qbs, size_k2)
        elif fp8_lhs_qbs < 0:
            fp8_w8a16 = True
            lhs1_quant_block_size = size_k1
            lhs2_quant_block_size = size_k2
    fp4_lhs_qbs = (
        envs.SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS.get()
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    if (
        is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs1_quant_block_size >= 256
        and rhs2_quant_block_size >= 256
        and fp4_lhs_qbs != 0
    ):
        if fp4_lhs_qbs > 0:
            lhs1_quant_block_size = min(fp4_lhs_qbs, size_k1)
            lhs2_quant_block_size = min(fp4_lhs_qbs, size_k2)
        else:
            fp8_w8a16 = True
            lhs1_quant_block_size = size_k1
            lhs2_quant_block_size = size_k2
    if fp4_w4a16 and not (
        is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs1_quant_block_size == lhs1_quant_block_size == 256
        and rhs2_quant_block_size == lhs2_quant_block_size == 256
    ):
        raise ValueError(
            "fp4_w4a16 takes block-256 fp4 experts; got "
            f"{w1.dtype} / {w2.dtype}, rhs blocks {rhs1_quant_block_size} / "
            f"{rhs2_quant_block_size}, lhs blocks {lhs1_quant_block_size} / "
            f"{lhs2_quant_block_size}"
        )

    compact_k256 = (
        (
            size_m <= _BLOCKDIAG_MAX_SIZE_M
            or envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES.get()
        )
        and envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES.get()
        and envs.SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES.get()
        and is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs1_quant_block_size == 256
        and rhs2_quant_block_size == 256
    )
    k256_prefill_w4a16 = (
        size_m > _BLOCKDIAG_MAX_SIZE_M
        and envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES.get()
        and is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs1_quant_block_size == 256
        and rhs2_quant_block_size == 256
    )
    fp4_w4a16_eff = fp4_w4a16 or compact_k256 or k256_prefill_w4a16

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=size_m,
            size_k=size_k1,
            size_n=size_n1,
            size_group=size_group,
            size_lhs_group=group_sizes.shape[0],
            size_lhs_sublane=size_lhs_sublane,
        ),
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k1, tile_n=tile_n1),
        lhs_cfgs=InputConfigs(
            quant_dtype=(
                jnp.float8_e4m3fn.dtype
                if is_quantized
                and not fp4_w4a16_eff
                and not fp8_w8a16
                and get_maybe_quantize_lhs(
                    w1.dtype, rhs1_quant_block_size, lhs1_quant_block_size
                )
                else None
            ),
            quant_block_size=lhs1_quant_block_size,
            dtype=out_dtype,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=w1.dtype if is_quantized else None,
            quant_block_size=rhs1_quant_block_size,
            dtype=w1.dtype,
            has_bias=has_bias,
            has_scale=is_quantized,
            packing=rhs1_packing,
            num_quant_blocks=total_scale_blocks1 if is_quantized else 1,
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=act_fn,
    )

    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=size_m,
            size_k=size_k2,
            size_n=aligned_n2,
            size_group=size_group,
            size_lhs_group=group_sizes.shape[0],
            size_lhs_sublane=size_lhs_sublane,
        ),
        tiles=TileSizes(tile_m=tile_m, tile_k=tile_k2, tile_n=tile_n2),
        lhs_cfgs=InputConfigs(
            quant_dtype=(
                jnp.float8_e4m3fn.dtype
                if is_quantized
                and not fp4_w4a16_eff
                and not fp8_w8a16
                and get_maybe_quantize_lhs(
                    w2.dtype, rhs2_quant_block_size, lhs2_quant_block_size
                )
                else None
            ),
            quant_block_size=lhs2_quant_block_size,
            dtype=out_dtype,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=w2.dtype if is_quantized else None,
            quant_block_size=rhs2_quant_block_size,
            dtype=w2.dtype,
            has_bias=has_bias,
            has_scale=is_quantized,
            packing=rhs2_packing,
            num_quant_blocks=total_scale_blocks2 if is_quantized else 1,
        ),
        out_dtype=out_dtype,
        acc_dtype=jnp.float32,
        zero_init=False,
    )

    # Scratch shapes.
    smem_rows = fused_rs_smem_rows(size_m, ep_size, expert_tp)
    max_num_gm_static = size_group + pl.cdiv(smem_rows, tile_m) - 1
    n2_cols = aligned_n2 // num_lanes
    _wla_cfg = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD.get() or 0)
    eff_weight_lookahead = (
        _wla_cfg == 1
        and is_quantized
        and not fp8_direct_write
        and not eff_hot_interleave
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and num_w1_bufs == 3
        and num_w2_bufs == 3
        and (size_k1 // tile_k1) * ((size_n1 // 2 if act_fn else size_n1) // tile_n1)
        == 2
        and (size_k2 // tile_k2) * (size_n2 // tile_n2) == 2
        and (-(-tile_k1 // rhs1_quant_block_size)) % 8 == 0
        and (-(-tile_k2 // rhs2_quant_block_size)) % 8 == 0
    )
    eff_wla2 = (
        _wla_cfg == 2
        and is_quantized
        and not fp8_direct_write
        and not eff_hot_interleave
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and num_w1_bufs >= 2
        and (size_k1 // tile_k1) * ((size_n1 // 2 if act_fn else size_n1) // tile_n1)
        == 1
        and (size_k2 // tile_k2) * (size_n2 // tile_n2) == 1
    )
    use_w2_prequant_cache = (
        is_quantized
        and jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and cfgs2.lhs_cfgs.quant_dtype is not None
        and rhs2_quant_block_size < 256
        and rhs2_quant_block_size < lhs2_quant_block_size
        and (tile_k2 // cfgs2.lhs_cfgs.quant_block_size) % 8 == 0
    )
    use_dedicated_w2_fp8_cache = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE.get() or 0) >= 1
        and is_quantized
        and not fp8_direct_write
        and not use_w2_prequant_cache
        and size_m >= _PREFILL_TILES_MIN_SIZE_M
        and jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs2_quant_block_size >= 256
        and cfgs2.lhs_cfgs.quant_dtype is not None
        and (
            tile_m <= 128
            or (
                tile_m == 256
                and num_w1_bufs <= 2
                and num_w2_bufs <= 2
                and not eff_hot_interleave
                and int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0)
                == 0
                and int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2.get() or 0) == 0
            )
        )
    )
    use_lhs_fp8_cache = (
        is_quantized
        and not use_w2_prequant_cache
        and (
            has_ext_lhs_scales
            or has_static_fp8_lhs
            or size_m >= _PREFILL_TILES_MIN_SIZE_M
        )
        and (
            has_ext_lhs_scales
            or has_static_fp8_lhs
            or bool(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES.get())
            or envs.SGLANG_TPU_MUSE_KERNEL_FP8_LAYERS.get()
            or envs.SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS.get() > 0
            or envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get() > 0
            or envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N.get() > 0
            or int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8.get() or 0) > 0
            or int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0) > 0
        )
        and cfgs1.lhs_cfgs.quant_dtype is not None
        and cfgs2.lhs_cfgs.quant_dtype is not None
        and cfgs1.lhs_cfgs.quant_dtype == cfgs2.lhs_cfgs.quant_dtype
    )
    _flq_val = (
        1
        if (
            size_m > _BLOCKDIAG_MAX_SIZE_M
            and int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get() or 0) >= 1
        )
        else 0
    )
    _g2a_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD.get() or 0)
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _rot_rs_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS.get() or 0)
        if (block_scatter and size_m > _BLOCKDIAG_MAX_SIZE_M)
        else 0
    )
    _dsilu_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU.get() or 0)
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _tlp_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2.get() or 0)
        if (
            size_m > _BLOCKDIAG_MAX_SIZE_M
            and is_quantized
            and jnp.dtype(w1.dtype) == jnp.float4_e2m1fn
            and rhs1_quant_block_size in (256, 512)
        )
        else 0
    )
    _tlp_kstep = int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP.get() or 1024)
    _sn_for_strip = max(
        num_lanes,
        int(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N.get() or num_lanes),
    )
    use_rhs_fp8_strip_buf = (
        _tlp_val >= 1 and not fp8_direct_write and _tlp_kstep >= 2048
    )
    use_in_kernel_pn = (
        (post_expert_norm_weight is not None) and (not expert_tp) and (not has_bias)
    )
    in_kernel_pn_size = int(post_expert_norm_weight.shape[0]) if use_in_kernel_pn else 0
    from .prefill_gmm_opts import resolve_static_fp8_rs_scale

    raw_static_fp8_rs, keep_static_fp8_out = (
        resolve_static_fp8_rs_scale(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS.get()
        )
        if size_m >= _PREFILL_TILES_MIN_SIZE_M
        else (0.0, False)
    )
    eff_static_fp8_rs = (
        raw_static_fp8_rs
        if (raw_static_fp8_rs > 0.0 and use_in_kernel_pn and not fp8_direct_write)
        else 0.0
    )
    stg_payload_dtype = jnp.float8_e4m3fn if eff_static_fp8_rs > 0.0 else out_dtype
    max_k_pq = max(padded_k1, size_k2)
    max_b_pq = max(
        padded_k1 // lhs1_quant_block_size,
        size_k2 // lhs2_quant_block_size,
    )

    scratch_shapes = [
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm_static,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM(
                (
                    2 * max_num_gm_static
                    if eff_hot_interleave
                    else max_num_gm_static + 1,
                ),
                jnp.int32,
            ),
        ),
        # fused_metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((1,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((2,), jnp.int32),
        ),
        # gathered_lhs_2x_ref
        pltpu.VMEM((2, tile_m, padded_k1 // num_lanes, num_lanes), lhs_in_dtype),
        # gmm1_out_ref (out_dtype avoids f32->bf16 VMEM roundtrip in interlude)
        pltpu.VMEM((tile_m, size_n1 // 2 if act_fn else size_n1), out_dtype),
        # intermediate_ref (unused when size_k2 == gmm1_out cols)
        pltpu.VMEM(
            (1, size_lhs_sublane, num_lanes)
            if size_k2 == (size_n1 // 2 if act_fn else size_n1)
            else (tile_m // size_lhs_sublane, size_lhs_sublane, size_k2),
            out_dtype,
        ),
        # tiled_out_2x_ref
        pltpu.VMEM((2, tile_m, aligned_n2), out_dtype),
        # scatter_staging_3x_ref — 4D: 4-slot staging buffer (saves 4 MiB VMEM; 4 MiB more when static FP8 RS)
        pltpu.VMEM((4, tile_m, n2_cols, num_lanes), stg_payload_dtype),
        # partial_out1_ref: 2D (tile_m, padded_k1) LHS VMEM cache for W4A16, FP8 LHS cache for W8A8, or FP8 W2 cache
        pltpu.VMEM(
            (num_w2_bufs, tile_k2, tile_n2),
            cfgs2.lhs_cfgs.quant_dtype,
        )
        if use_w2_prequant_cache
        else (
            pltpu.VMEM((tile_m, max_k_pq), cfgs1.lhs_cfgs.quant_dtype)
            if use_lhs_fp8_cache
            else pltpu.VMEM((tile_m, padded_k1), out_dtype)
        ),
        # partial_out2_ref: prequantized FP32 scale cache for W2 or LHS W8A8
        pltpu.VMEM(
            (num_w2_bufs, tile_k2 // cfgs2.lhs_cfgs.quant_block_size, tile_n2),
            jnp.float32,
        )
        if use_w2_prequant_cache
        else (
            pltpu.VMEM(
                (
                    max_b_pq
                    + (
                        1
                        if (
                            size_m > _BLOCKDIAG_MAX_SIZE_M
                            and int(
                                envs.SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW.get() or 0
                            )
                            >= 2
                            and int(
                                envs.SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU.get() or 0
                            )
                            in (0, 1)
                        )
                        else 0
                    ),
                    tile_m,
                    num_lanes,
                ),
                jnp.float32,
            )
            if use_lhs_fp8_cache
            else pltpu.VMEM((1, num_lanes), jnp.float32)
        ),
        # shared_acc_ref (unused when single K-tile for both GMM1 and GMM2)
        pltpu.VMEM(
            (1, num_lanes)
            if (padded_k1 == tile_k1 and size_k2 == tile_k2)
            else (tile_m, max(tile_n1 * 2 if act_fn else tile_n1, tile_n2)),
            jnp.float32,
        ),
        # gather_sem_ref
        pltpu.SemaphoreType.DMA((2,)),
        # staging_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # zero_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # gm_id_ref: [0]=gm_id, [1..8]=send counts, [9..16]=local counts (+2 for w_ring)
        pltpu.SMEM(
            ((19 if (eff_weight_lookahead or eff_wla2) else 17) + 2 * ep_size,),
            jnp.int32,
        ),
        # w1_buf_ref
        pltpu.VMEM(
            (
                num_w1_bufs,
                (
                    tile_k1 // (32 // jax.dtypes.itemsize_bits(w1.dtype))
                    if is_quantized
                    else tile_k1
                ),
                tile_n1 * 2 if act_fn else tile_n1,
            ),
            jnp.uint32 if is_quantized else w1.dtype,
        ),
        # w2_buf_ref
        pltpu.VMEM(
            (
                num_w2_bufs,
                (
                    tile_k2 // (32 // jax.dtypes.itemsize_bits(w2.dtype))
                    if is_quantized
                    else tile_k2
                ),
                tile_n2,
            ),
            jnp.uint32 if is_quantized else w2.dtype,
        ),
        # w1_scale_buf_ref: [bufs, k_blocks, (1,) n] matching the HBM scale rank.
        pltpu.VMEM(
            (
                num_w1_bufs,
                (-(-tile_k1 // rhs1_quant_block_size) if is_quantized else 1),
                *((1,) if scale_has_unit_dim else ()),
                tile_n1 * 2 if act_fn else tile_n1,
            ),
            scale_dtype,
        ),
        # w2_scale_buf_ref
        pltpu.VMEM(
            (
                num_w2_bufs,
                (-(-tile_k2 // rhs2_quant_block_size) if is_quantized else 1),
                *((1,) if scale_has_unit_dim else ()),
                tile_n2,
            ),
            scale_dtype,
        ),
        # w1_bias_buf_ref
        pltpu.VMEM((1, tile_n1 * 2 if act_fn else tile_n1), jnp.float32),
        # w2_bias_buf_ref
        pltpu.VMEM((1, tile_n2), jnp.float32),
        # w1_sem_ref
        pltpu.SemaphoreType.DMA((num_w1_bufs,)),
        # w2_sem_ref
        pltpu.SemaphoreType.DMA((num_w2_bufs,)),
        # output_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # --- ICI direct-write buffers ---
        # send_sems_ref (per-staging-slot, 8 slots)
        pltpu.SemaphoreType.REGULAR((8,)),
        # local_write_sems_ref (per-staging-slot, 8 slots)
        pltpu.SemaphoreType.REGULAR((8,)),
        # recv_sem_ref (for incoming remote writes, 1 slot)
        pltpu.SemaphoreType.REGULAR((1,)),
    ]

    # Keep the ID shared so identical MoE layers reuse one Mosaic compilation.
    # The wrapper's post-kernel TP all-gather orders successive layer calls.
    compiler_params = pltpu.CompilerParams(
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
        collective_id=0,
    )

    w1_scale_input = w1_scale if is_quantized else jnp.zeros((1, 1, 1, 1), jnp.float32)
    w2_scale_input = w2_scale if is_quantized else jnp.zeros((1, 1, 1, 1), jnp.float32)
    w1_bias_input = w1_bias if has_bias else jnp.zeros((1, 1, 1), jnp.float32)
    if use_in_kernel_pn:
        pnw_1d = post_expert_norm_weight.astype(jnp.float32) + 1.0
        if aligned_n2 > in_kernel_pn_size:
            pnw_1d = jnp.concatenate(
                [pnw_1d, jnp.zeros(aligned_n2 - in_kernel_pn_size, dtype=jnp.float32)]
            )
        w2_bias_input = jnp.broadcast_to(pnw_1d[None, :], (8, aligned_n2))
    else:
        w2_bias_input = w2_bias if has_bias else jnp.zeros((1, 1, 1), jnp.float32)
    if enable_chipmate:
        max_num_gm_arr = jnp.concatenate(
            [
                jnp.array([max_num_gm], dtype=jnp.int32),
                chipmate_plan[: 2 * num_cm_slots].astype(jnp.int32),
            ]
        )
    else:
        max_num_gm_arr = jnp.array([max_num_gm], dtype=jnp.int32)
    if eff_weight_lookahead or eff_wla2:
        _go_la = group_offset[0]
        _sz_la = lax.dynamic_slice_in_dim(group_sizes, _go_la, size_group).astype(
            jnp.int32
        )
        _g_idx = jnp.arange(size_group, dtype=jnp.int32)
        _later_act = (_g_idx[:, None] < _g_idx[None, :]) & (_sz_la[None, :] > 0)
        _next_act_grp = jnp.where(
            jnp.any(_later_act, axis=1),
            jnp.argmax(_later_act, axis=1).astype(jnp.int32),
            jnp.int32(-1),
        )
        max_num_gm_arr = jnp.concatenate([max_num_gm_arr, _next_act_grp])

    # Output buffer — also serves as ICI DMA target. In the FP8 direct-write
    # path this stores quantized payload rows; row scales are returned as a
    # second output and dequantized immediately after the kernel.
    payload_dtype = (
        jnp.float8_e4m3fn
        if (fp8_direct_write or eff_static_fp8_rs > 0.0)
        else out_dtype
    )
    out_buf_init = jax.ShapeDtypeStruct(
        (chunk_size * top_k, n2_cols, num_lanes), payload_dtype
    )
    # TPU VMEM vector slices must be tile-aligned. Store the scalar row scale
    # as a padded 128-wide row and use column 0 when dequantizing.
    out_scale_init = jax.ShapeDtypeStruct((chunk_size * top_k, 128), jnp.float32)

    u16_lhs_env = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS.get() or 0)
    force_u16_lhs = u16_lhs_env >= 1
    auto_ep16_u16 = (u16_lhs_env >= 0) and (ep_size >= 16 and size_m >= 131_072)
    flatten_bs_meta = bool(envs.SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META.get()) or (
        block_scatter and (size_m > 131_072 or force_u16_lhs or auto_ep16_u16)
    )
    skip_pack_for_bs = block_scatter and (
        flatten_bs_meta
        or envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_MOE_TOKENS.get()
        or envs.SGLANG_TPU_MUSE_KERNEL_FP8_LAYERS.get()
        or bool(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES.get())
    )
    pack_indices = (smem_rows > 120_000) and (not skip_pack_for_bs)
    pack_u16_lhs = (
        block_scatter
        and not pack_indices
        and not fp8_direct_write
        and hidden_3d.shape[0] <= 65_536
        and (smem_rows % 2 == 0)
        and (size_m >= 262_144 or force_u16_lhs or auto_ep16_u16)
    )

    perm_gm_opt = None
    hot_slot_opt = None
    if eff_hot_interleave:
        from .moe_imbalance_planner import interleave_hot_expert_tiles_jax

        _go_hi = group_offset[0]
        _sz_hi = lax.dynamic_slice_in_dim(group_sizes, _go_hi, size_group).astype(
            jnp.int32
        )
        _tl_hi = jnp.where(_sz_hi > 0, (_sz_hi + tile_m - 1) // tile_m, 0)
        _tl_end_hi = jnp.cumsum(_tl_hi)
        _gm_hi = jnp.arange(max_num_gm_static, dtype=jnp.int32)
        _grp_hi = jnp.minimum(
            jnp.sum((_tl_end_hi[None, :] <= _gm_hi[:, None]).astype(jnp.int32), axis=1),
            size_group - 1,
        )
        _grp_hi = jnp.where(_gm_hi < _tl_end_hi[-1], _grp_hi, 0)
        perm_gm_opt, hot_slot_opt = interleave_hot_expert_tiles_jax(
            _grp_hi,
            _tl_hi,
            num_slots=max_num_gm_static,
            cand_groups=size_group_phys,
        )

    # Expert TP builds the routing metadata ahead of the kernel in XLA and
    # leaves each tile's output writes in flight until its staging slot is reused.
    group_start_raw = group_start
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PRECOMPUTED_METADATA.get()
        and not expert_tp
        and not block_scatter
        and not eff_hot_interleave
        and smem_rows == size_m
        and group_start_raw.shape == (3,)
        and envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_ROUTER.get() >= 2
    ):
        group_start = group_start_raw
    else:
        group_start = _precompute_fused_rs_metadata(
            group_sizes,
            group_offset,
            group_start_raw,
            lhs_indices,
            size_group=size_group,
            tile_m=tile_m,
            sublane=size_lhs_sublane,
            chunk_size=chunk_size,
            num_slots=max_num_gm_static,
            perm_gm=perm_gm_opt,
            hot_slot=hot_slot_opt,
        )
        if (
            envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PRECOMPUTED_METADATA.get()
            and not expert_tp
            and not block_scatter
            and not eff_hot_interleave
            and smem_rows == size_m
        ):
            # Pass only (local_start, local_end, total_recv): the kernel tiles in its scalar
            # prologue as stock fused_rs does, and XLA drops the unused vector tiling.
            group_start = group_start[:3]
    if smem_rows < size_m:
        _raw_base_w = (group_start[0] // _LOCAL_ROWS_ALIGN) * _LOCAL_ROWS_ALIGN
        _base_w = jnp.minimum(_raw_base_w, jnp.int32(size_m - smem_rows))
    else:
        _base_w = jnp.int32(0)

    _inter32_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS.get() or 0)
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _use_inter32_cf = bool(
        _inter32_val > 0
        and block_scatter
        and group_start_raw.shape[0] >= 2 + ep_size + size_group * ep_size
        and flatten_bs_meta
        and not pack_indices
        and not pack_u16_lhs
        and tile_m in (128, 256)
        and tile_m % ep_size == 0
        and (tile_m // ep_size) in (16, 32)
        and _rot_rs_val >= 2
        and (ep_size & (ep_size - 1)) == 0
        and not eff_hot_interleave
        and not fp8_direct_write
        and int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0) <= 0
    )
    if block_scatter:
        if group_start_raw.shape[0] >= 2 + ep_size + size_group * ep_size:
            secondary_idx_ref = _precompute_block_scatter_meta_closed_form(
                group_sizes,
                group_offset,
                group_start_raw,
                size_group=size_group,
                tile_m=tile_m,
                num_slots=max_num_gm_static,
                ep_size=ep_size,
                perm_gm=perm_gm_opt,
                rotate_rs=(_rot_rs_val >= 2),
                inter32_min_kf=(_inter32_val if _use_inter32_cf else 0),
                smem_base_offset=_base_w,
            )
        else:
            _hdr_len = 5 if eff_hot_interleave else 4
            gm_m_offset_vec = group_start[_hdr_len + max_num_gm_static :]
            secondary_idx_ref = _precompute_block_scatter_meta(
                gm_m_offset_vec,
                lhs_indices,
                chunk_size=chunk_size,
                ep_size=ep_size,
                my_id=group_offset[0] // size_group,
                num_slots=max_num_gm_static if eff_hot_interleave else None,
                rotate_rs=(_rot_rs_val >= 2),
            )
        if flatten_bs_meta:
            secondary_idx_ref = secondary_idx_ref.reshape(-1)

    if smem_rows < size_m:
        _hdr_len_w = 5 if eff_hot_interleave else 4
        if group_start.shape[0] > 3:
            _shift_w = (
                jnp.zeros_like(group_start)
                .at[0:2]
                .set(_base_w)
                .at[_hdr_len_w + max_num_gm_static :]
                .set(_base_w)
            )
            group_start = group_start - _shift_w
        elif group_start.shape[0] >= 2:
            group_start = group_start.at[0:2].add(-_base_w)
        else:
            group_start = group_start - _base_w
        window_lhs = lax.dynamic_slice_in_dim(lhs_indices, _base_w, smem_rows)
        window_tk = (
            lax.dynamic_slice_in_dim(topk_indices, _base_w, smem_rows)
            if topk_indices.shape[0] == size_m
            else topk_indices
        )
    else:
        window_lhs = lhs_indices
        window_tk = topk_indices

    if pack_u16_lhs:
        _u_pairs = window_lhs.astype(jnp.uint32).reshape(smem_rows // 2, 2)
        _packed_u16 = (
            (_u_pairs[:, 0] & jnp.uint32(0xFFFF)) | (_u_pairs[:, 1] << jnp.uint32(16))
        ).astype(jnp.int32)
        primary_idx_ref = jnp.pad(_packed_u16, (0, 1))
    elif pack_indices:
        # combined = lhs_idx * top_k + topk_slot
        primary_idx_ref = window_lhs.astype(jnp.int32) * top_k + window_tk.astype(
            jnp.int32
        )
        secondary_idx_ref = jnp.zeros((1,), dtype=jnp.int32)  # dummy
    elif block_scatter:
        primary_idx_ref = window_lhs
    else:
        primary_idx_ref = window_lhs
        _oi_i32 = window_lhs.astype(jnp.int32)
        _dest_chip_vec = _oi_i32 // chunk_size
        _write_pos_vec = window_tk.astype(jnp.int32) * chunk_size + (
            _oi_i32 % chunk_size
        )
        secondary_idx_ref = lax.shift_left(_dest_chip_vec, 16) | (
            _write_pos_vec & jnp.int32(0xFFFF)
        )

    scalar_prefetch_inputs = (
        (
            group_sizes,
            group_offset,
            group_start,
            primary_idx_ref,
            secondary_idx_ref,
            max_num_gm_arr,
            _w1_gs_gate,
            _w1_gs_up,
            _w2_gs,
            hidden_states_scale.reshape(-1).astype(jnp.float32),
        )
        if has_ext_lhs_scales
        else (
            group_sizes,
            group_offset,
            group_start,
            primary_idx_ref,
            secondary_idx_ref,
            max_num_gm_arr,
            _w1_gs_gate,
            _w1_gs_up,
            _w2_gs,
        )
    )
    pallas_inputs = (
        *scalar_prefetch_inputs,
        hidden_3d,
        w1,
        w2,
        w1_scale_input,
        w2_scale_input,
        w1_bias_input,
        w2_bias_input,
    )
    vectorized_dequant = envs.SGLANG_TPU_MUSE_KERNEL_VECTORIZED_DEQUANT.get()
    blockdiag_max_rows = envs.SGLANG_TPU_MUSE_KERNEL_BLOCKDIAG_MAX_ROWS.get()
    if blockdiag_max_rows not in (0, *_BLOCKDIAG_ROWS):
        raise ValueError(
            f"SGLANG_TPU_MUSE_KERNEL_BLOCKDIAG_MAX_ROWS must be 0 or one of {_BLOCKDIAG_ROWS}, "
            f"got {blockdiag_max_rows}"
        )
    blockdiag_rows = tuple(r for r in _BLOCKDIAG_ROWS if r <= blockdiag_max_rows)
    is_fp4_k256 = (
        is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and rhs1_quant_block_size == 256
        and rhs2_quant_block_size == 256
    )
    if size_m <= _BLOCKDIAG_MAX_SIZE_M:
        k256_min_m = envs.SGLANG_TPU_MUSE_KERNEL_K256_DECODE_TILES.get()
        block256_scales = (
            envs.SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES.get() and is_fp4_k256
        )
        # Block-256 scales arrive compact (fp4_w4a16 / COMPACT_K256_SCALES) or repeated (probe).
        if block256_scales and not (
            fp4_w4a16_eff or envs.SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES.get()
        ):
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES reads block-256 scales; set "
                "SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES=1 or "
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES=1 so the experts are requantized"
            )
        k256_tiles = block256_scales or (is_fp4_k256 and 0 < k256_min_m <= size_m)
        if k256_tiles:
            # Gate only: every decode tile takes the K256 dots.
            blockdiag_rows = _BLOCKDIAG_ROWS
    else:
        # Prefill keeps the W4A16 strips unless K256_PREFILL_TILES is set.
        k256_tiles = (
            is_fp4_k256 and envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES.get()
        )
        block256_scales = (
            envs.SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES.get()
            and is_fp4_k256
            and k256_tiles
        )
        if block256_scales and not (
            fp4_w4a16_eff or envs.SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES.get()
        ):
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES reads block-256 scales; set "
                "SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES=1 or "
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES=1 so the experts are requantized"
            )
    prefill_strip_nc = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_NC.get()
        if (size_m > _BLOCKDIAG_MAX_SIZE_M and k256_tiles)
        else 0
    )
    decode_strip_nc = (
        envs.SGLANG_TPU_MUSE_KERNEL_DECODE_STRIP_NC.get()
        if (size_m <= _BLOCKDIAG_MAX_SIZE_M and k256_tiles)
        else prefill_strip_nc
    )
    decode_dual_k256 = (
        envs.SGLANG_TPU_MUSE_KERNEL_DECODE_DUAL_K256.get()
        if (size_m <= _BLOCKDIAG_MAX_SIZE_M and k256_tiles)
        else 0
    )
    short_prefill_tag = (
        "-spt"
        if (
            envs.SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES.get()
            and 4096 <= size_m <= 16384
        )
        else ""
    )
    _nc_tag = (
        f"-nc{decode_strip_nc}"
        if decode_strip_nc > 0
        else (f"-nca{-decode_strip_nc}" if decode_strip_nc < 0 else "")
    )
    _sn_val = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N.get()
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _kp_fp8 = (
        envs.SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_FP8.get()
        if (size_m > _BLOCKDIAG_MAX_SIZE_M and k256_tiles)
        else 0
    )
    _fp4_am = (
        envs.SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE.get()
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _rb_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get() or 0)
        if size_m >= 16384
        else 0
    )
    _rb_active = _rb_val in (1, 3, 4) or (_rb_val == 2 and tile_m <= 128)
    _sw_val = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8.get() or 0)
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _eo_raw = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() or 0)
        if size_m > _BLOCKDIAG_MAX_SIZE_M
        else 0
    )
    _out_n1_full = size_n1 // 2 if act_fn else size_n1
    _can_cache_full_expert = num_w1_bufs >= (
        _out_n1_full // tile_n1
    ) and num_w2_bufs >= (size_n2 // tile_n2)
    eff_eo_mode = 0
    if (
        _eo_raw >= 1
        and is_quantized
        and jnp.dtype(w1.dtype) == jnp.dtype(w2.dtype) == jnp.float4_e2m1fn
        and use_lhs_fp8_cache
        and not k256_tiles
        and not has_static_fp8_lhs
        and not eff_hot_interleave
        and not fp8_direct_write
        and padded_k1 == tile_k1
        and size_k2 == tile_k2 == _out_n1_full
        and cfgs1.lhs_cfgs.quant_block_size == tile_k1
        and cfgs2.lhs_cfgs.quant_block_size == tile_k2
    ):
        eff_eo_mode = (
            min(_eo_raw, 4)
            if (
                _eo_raw >= 3
                and _can_cache_full_expert
                and act_fn == "silu"
                and not has_bias
            )
            else (
                2
                if (
                    _eo_raw >= 2
                    and _can_cache_full_expert
                    and act_fn == "silu"
                    and not has_bias
                )
                else 1
            )
        )
    _eo_strip_w = min(_sn_val if _sn_val > 0 else num_lanes, tile_n1, tile_n2)
    if _eo_strip_w % num_lanes != 0:
        _eo_strip_w = num_lanes
    eff_cm_defer = bool(
        enable_chipmate and envs.SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND.get()
    )
    _pq_lhs_raw = int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get() or 0)
    _pq_lhs_env = (
        6
        if has_static_fp8_lhs
        else (max(1, _pq_lhs_raw or 1) if has_ext_lhs_scales else 0)
    )
    _slh_tag = (
        f"-slh{int(eff_static_fp8_lhs) if eff_static_fp8_lhs.is_integer() else str(eff_static_fp8_lhs).replace('.', 'p')}"
        if has_static_fp8_lhs
        else ""
    )
    _sfp8_tag = (
        f"-sfp8_{int(eff_static_fp8_rs) if eff_static_fp8_rs.is_integer() else str(eff_static_fp8_rs).replace('.', 'p')}"
        if eff_static_fp8_rs > 0.0
        else ""
    )
    pallas_name = (
        f"gmm_v2_fused_rs-E_{size_group}-M_{size_m}"
        f"-K1_{size_k1}-N1_{size_n1}-K2_{size_k2}-N2_{size_n2}"
        f"-EP_{ep_size}-TK_{top_k}"
        f"{'-packed' if pack_indices else ''}"
        f"{'-u16w' if pack_u16_lhs else ''}"
        f"{'-fbs' if (flatten_bs_meta and block_scatter) else ''}"
        f"{short_prefill_tag}"
        f"{'-vdq' if vectorized_dequant else ''}"
        f"{'' if blockdiag_rows == _BLOCKDIAG_ROWS else f'-bd{blockdiag_max_rows}'}"
        f"{'-b256' if block256_scales else '-k256' if k256_tiles else ''}"
        f"{'-ck256' if compact_k256 else ''}"
        f"{_nc_tag}"
        f"{f'-dk{decode_dual_k256}' if decode_dual_k256 > 0 else ''}"
        f"{f'-fq4_{fp4_lhs_qbs}' if fp4_lhs_qbs != 0 else ''}"
        f"{f'-fa{_fp4_am}' if _fp4_am > 0 else ''}"
        f"{f'-sn{_sn_val}' if _sn_val > 0 else ''}"
        f"{f'-kf{_kp_fp8}' if _kp_fp8 > 0 else ''}"
        f"{f'-rb{_rb_val}' if _rb_active else ''}"
        f"{f'-flq{_flq_val}' if _flq_val > 0 else ''}"
        f"{f'-g2a{_g2a_val}' if _g2a_val > 0 else ''}"
        f"{'-wla2' if eff_wla2 else ('-wla1' if eff_weight_lookahead else '')}"
        f"{'-w2c1' if use_dedicated_w2_fp8_cache else ''}"
        f"{f'-rrs{_rot_rs_val}' if _rot_rs_val > 0 else ''}"
        f"{f'-i32_{_inter32_val}' if _use_inter32_cf else ''}"
        f"{f'-ds{_dsilu_val}' if _dsilu_val > 0 else ''}"
        f"{f'-pql{_pq_lhs_env}' if _pq_lhs_env > 0 else ''}"
        f"{_slh_tag}"
        f"{f'-tlp{_tlp_val}k{_tlp_kstep}' if _tlp_val > 0 else ''}"
        f"{f'-sw{_sw_val}' if _sw_val > 0 else ''}"
        f"{f'-eo{eff_eo_mode}' if eff_eo_mode > 0 else ''}"
        f"{_sfp8_tag}"
    )
    kernel_kwargs = dict(
        fused_dims=fused_dims,
        tile_m=tile_m,
        tile_k1=tile_k1,
        tile_k2=tile_k2,
        tile_n1=tile_n1,
        tile_n2=tile_n2,
        num_w1_bufs=num_w1_bufs,
        num_w2_bufs=num_w2_bufs,
        act_fn=act_fn,
        out_dtype=out_dtype,
        cfgs1=cfgs1,
        cfgs2=cfgs2,
        ep_size=ep_size,
        chunk_size=chunk_size,
        ep_axis_name=ep_axis_name,
        top_k=top_k,
        pack_indices=pack_indices,
        defer_writes=True,
        in_kernel_post_norm_size=in_kernel_pn_size,
        has_w1_gs=(w1_global_scale is not None) or has_static_fp8_lhs,
        has_w2_gs=w2_global_scale is not None,
        block_scatter=block_scatter,
        vectorized_dequant=vectorized_dequant,
        blockdiag_rows=blockdiag_rows,
        block256_scales=block256_scales,
        k256_tiles=k256_tiles,
        decode_strip_nc=decode_strip_nc,
        decode_dual_k256=decode_dual_k256,
        packed_u16_lhs=pack_u16_lhs,
        gather_2ahead_mode=(0 if eff_eo_mode >= 3 else _g2a_val),
        use_weight_lookahead=(2 if eff_wla2 else (1 if eff_weight_lookahead else 0)),
        rotate_rs_mode=_rot_rs_val,
        static_fp8_rs_scale=eff_static_fp8_rs,
        cm_defer_send=eff_cm_defer,
        num_cm_slots=max(1, num_cm_slots),
        pq_lhs_mode=_pq_lhs_env,
        inter32_enabled=_use_inter32_cf,
        static_fp8_lhs=has_static_fp8_lhs,
    )
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states_3d
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1
        pl.BlockSpec(memory_space=pltpu.HBM),  # w2
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1_scale
        pl.BlockSpec(memory_space=pltpu.HBM),  # w2_scale
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1_bias
        pl.BlockSpec(
            memory_space=pltpu.VMEM if use_in_kernel_pn else pltpu.HBM
        ),  # w2_bias or pnw_8
    ]

    if fp8_direct_write:
        fp8_scratch_shapes = scratch_shapes + [
            pltpu.SemaphoreType.REGULAR((3,)),
            pltpu.SemaphoreType.REGULAR((3,)),
            pltpu.SemaphoreType.REGULAR((1,)),
            pltpu.VMEM((3, tile_m, n2_cols, num_lanes), jnp.float8_e4m3fn),
            pltpu.VMEM((3, tile_m, 128), jnp.float32),
        ]
        payload, row_scales = pl.pallas_call(
            functools.partial(kernel_main_fused_rs_fp8, **kernel_kwargs),
            out_shape=(out_buf_init, out_scale_init),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetch_inputs),
                in_specs=in_specs,
                out_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                scratch_shapes=fp8_scratch_shapes,
            ),
            compiler_params=compiler_params,
            name=f"{pallas_name}-fp8-direct-write",
        )(*pallas_inputs)
        payload_2d = payload.reshape(chunk_size * top_k, aligned_n2).astype(jnp.float32)
        return (payload_2d * row_scales[:, :1]).astype(out_dtype)[:, :size_n2]

    has_w8_ext = (
        has_ext_lhs_scales or use_dedicated_w2_fp8_cache or use_rhs_fp8_strip_buf
    )
    if (
        enable_chipmate
        or eff_hot_interleave
        or has_w8_ext
        or eff_eo_mode >= 1
        or _use_inter32_cf
    ):
        assert not fp8_direct_write, (
            "prefill chipmate/hot_interleave/w8_ext/eo does not support fp8_direct_write"
        )
        n_base_scratch = len(scratch_shapes)
        ext_scratch = list(scratch_shapes)
        if enable_chipmate:
            _cm_scr_slots = (
                2 if (0 if eff_eo_mode >= 3 else _g2a_val) == 6 else 1
            ) * num_cm_slots
            ext_scratch.extend(
                [
                    pltpu.SemaphoreType.DMA((4 * num_cm_slots,)),
                    pltpu.SemaphoreType.DMA((4 * num_cm_slots,)),
                    pltpu.SMEM(
                        (_cm_scr_slots + (1 if eff_cm_defer else 0),), jnp.int32
                    ),
                ]
            )
        if eff_hot_interleave:
            _out_n1_hi = size_n1 // 2 if act_fn else size_n1
            _num_n1_hi = _out_n1_hi // tile_n1
            _w1_dma_n_hi = tile_n1 * 2 if act_fn else tile_n1
            ext_scratch.extend(
                [
                    pltpu.VMEM(
                        (
                            _num_n1_hi,
                            (
                                tile_k1 // (32 // jax.dtypes.itemsize_bits(w1.dtype))
                                if is_quantized
                                else tile_k1
                            ),
                            _w1_dma_n_hi,
                        ),
                        jnp.uint32 if is_quantized else w1.dtype,
                    ),
                    pltpu.VMEM(
                        (
                            _num_n1_hi,
                            (
                                -(-tile_k1 // rhs1_quant_block_size)
                                if is_quantized
                                else 1
                            ),
                            *((1,) if scale_has_unit_dim else ()),
                            _w1_dma_n_hi,
                        ),
                        scale_dtype,
                    ),
                    pltpu.SemaphoreType.DMA((_num_n1_hi,)),
                    pltpu.SMEM((1,), jnp.int32),
                ]
            )
        if use_dedicated_w2_fp8_cache:
            _w1_vmem_b = (
                num_w1_bufs
                * tile_k1
                * (tile_n1 * 2 if act_fn else tile_n1)
                * jax.dtypes.itemsize_bits(w1.dtype)
                // 8
            )
            if tile_m <= 128 and _w1_vmem_b <= 16 * 1024 * 1024:
                _n_w2c_slots = num_w2_bufs
                _w2c_cols = tile_n2
            else:
                _n_w2c_slots = 1
                # Leave >= 1.9 MiB TPU7x VMEM safety margin under 63.94 MiB usable cap:
                # base VMEM at (K1=4096, N1=8192, K2=4096, N2=4096) is ~50 MiB at m=128
                # and ~56 MiB at m=256 -> budget 12 MiB (3072 cols) at m<=128, 6 MiB (1536 cols) at m>=256.
                _w2c_budget_b = (8 if tile_m <= 128 else 3) * 1024 * 1024
                if tile_k2 * tile_n2 <= _w2c_budget_b:
                    _w2c_cols = tile_n2
                else:
                    _w2c_cols = max(256, ((_w2c_budget_b // tile_k2) // 256) * 256)
            ext_scratch.append(
                pltpu.VMEM((_n_w2c_slots, tile_k2, _w2c_cols), jnp.float8_e4m3fn)
            )
        if use_rhs_fp8_strip_buf:
            _max_strip_w = min(
                max(tile_n1, tile_n2),
                ((_sn_for_strip + num_lanes - 1) // num_lanes) * num_lanes,
            )
            ext_scratch.append(
                pltpu.VMEM((max(tile_k1, tile_k2), _max_strip_w), jnp.float8_e4m3fn)
            )
        n_eo_scratch = 0
        if eff_eo_mode == 1:
            ext_scratch.append(
                pltpu.VMEM((max(tile_k1, tile_k2), _eo_strip_w), jnp.float8_e4m3fn)
            )
            n_eo_scratch = 1
        elif eff_eo_mode == 2:
            ext_scratch.extend(
                [
                    pltpu.VMEM((max(tile_k1, tile_k2), _eo_strip_w), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, tile_m, max_k_pq), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, tile_m, num_lanes), jnp.float32),
                    pltpu.VMEM((2, tile_m, _out_n1_full), out_dtype),
                ]
            )
            n_eo_scratch = 4
        elif eff_eo_mode >= 3:
            # Shrink unused single-tile VMEM buffers (gmm1_out_ref, tiled_out_2x_ref,
            # partial_out1_ref, partial_out2_ref) to 8 rows, freeing ~8 MiB VMEM.
            ext_scratch[3] = pltpu.VMEM((8, _out_n1_full), out_dtype)
            ext_scratch[5] = pltpu.VMEM((2, 8, aligned_n2), out_dtype)
            ext_scratch[7] = pltpu.VMEM((8, max_k_pq), cfgs1.lhs_cfgs.quant_dtype)
            ext_scratch[8] = pltpu.VMEM((1, 8, num_lanes), jnp.float32)
            _eo3_w_cols = min(2048, max(tile_n1, tile_n2))
            _w1_eo_b = (
                num_w1_bufs
                * tile_k1
                * (tile_n1 * 2 if act_fn else tile_n1)
                * jax.dtypes.itemsize_bits(w1.dtype)
                // 8
            )
            _eo3_r = 5 if (tile_m <= 256 and _w1_eo_b <= 16 * 1024 * 1024) else 4
            ext_scratch.extend(
                [
                    pltpu.VMEM((max(tile_k1, tile_k2), _eo3_w_cols), jnp.float8_e4m3fn),
                    pltpu.VMEM(
                        (8, num_lanes if eff_eo_mode >= 4 else _eo3_w_cols),
                        jnp.float32,
                    ),
                    pltpu.VMEM((_eo3_r, tile_m, max_k_pq), jnp.float8_e4m3fn),
                    pltpu.VMEM((_eo3_r, tile_m, num_lanes), jnp.float32),
                    pltpu.VMEM((_eo3_r, tile_m, _out_n1_full), out_dtype),
                ]
            )
            n_eo_scratch = 5
        if _use_inter32_cf:
            ext_scratch.append(pltpu.SMEM((4, tile_m), jnp.int32))

        if enable_chipmate:
            _w1_p_rows = (
                size_k1 // (32 // jax.dtypes.itemsize_bits(w1.dtype))
                if is_quantized
                else size_k1
            )
            _w2_p_rows = (
                size_k2 // (32 // jax.dtypes.itemsize_bits(w2.dtype))
                if is_quantized
                else size_k2
            )
            _cm_lead = (num_cm_slots,) if num_cm_slots > 1 else ()
            cm_w1_init = jax.ShapeDtypeStruct(
                (*_cm_lead, _w1_p_rows, size_n1),
                jnp.uint32 if is_quantized else w1.dtype,
            )
            cm_s1_init = jax.ShapeDtypeStruct(
                (*_cm_lead, *w1_scale_input.shape[2 if stacked else 1 :]),
                w1_scale_input.dtype,
            )
            cm_w2_init = jax.ShapeDtypeStruct(
                (*_cm_lead, _w2_p_rows, size_n2),
                jnp.uint32 if is_quantized else w2.dtype,
            )
            cm_s2_init = jax.ShapeDtypeStruct(
                (*_cm_lead, *w2_scale_input.shape[2 if stacked else 1 :]),
                w2_scale_input.dtype,
            )
            ext_out_shape = (
                out_buf_init,
                cm_w1_init,
                cm_s1_init,
                cm_w2_init,
                cm_s2_init,
            )
            ext_out_specs = tuple(
                pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(5)
            )
        else:
            ext_out_shape = out_buf_init
            ext_out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        _cm_sfx = (
            f"-cm{chipmate_xor}{f's{num_cm_slots}' if num_cm_slots > 1 else ''}{'d' if eff_cm_defer else ''}"
            if enable_chipmate
            else ""
        )
        ext_name = f"{pallas_name}{_cm_sfx}{'-hi' if eff_hot_interleave else ''}"

        def _ext_kernel_entry(*all_refs):
            if has_ext_lhs_scales:
                sp_in = (*all_refs[:9], *all_refs[10:17])
                lhs_row_sc_r = all_refs[9]
                out_r = all_refs[17]
                cur = 18
            else:
                sp_in = all_refs[:16]
                lhs_row_sc_r = None
                out_r = all_refs[16]
                cur = 17
            if enable_chipmate:
                cm_pads = all_refs[cur : cur + 4]
                cur += 4
            else:
                cm_pads = None
            base_scr = all_refs[cur : cur + n_base_scratch]
            cur += n_base_scratch
            if enable_chipmate:
                cm_scr = all_refs[cur : cur + 3]
                cur += 3
                cm_bundle = (*cm_pads, *cm_scr)
            else:
                cm_bundle = None
            if eff_hot_interleave:
                hot_bundle = all_refs[cur : cur + 4]
                cur += 4
            else:
                hot_bundle = None
            if use_dedicated_w2_fp8_cache:
                w2_fp8_r = all_refs[cur]
                cur += 1
            else:
                w2_fp8_r = None
            if use_rhs_fp8_strip_buf:
                rhs_strip_r = all_refs[cur]
                cur += 1
            else:
                rhs_strip_r = None
            w8_bundle = (lhs_row_sc_r, w2_fp8_r, rhs_strip_r) if has_w8_ext else None
            if n_eo_scratch > 0:
                eo_bundle = all_refs[cur : cur + n_eo_scratch]
                cur += n_eo_scratch
            else:
                eo_bundle = None
            if _use_inter32_cf:
                tile_idx_smem_r = all_refs[cur]
                cur += 1
            else:
                tile_idx_smem_r = None
            return kernel_main_fused_rs(
                *sp_in,
                out_r,
                *base_scr,
                cm_refs=cm_bundle,
                chipmate_xor=chipmate_xor,
                hot_refs=hot_bundle,
                w8_refs=w8_bundle,
                eo_refs=eo_bundle,
                tile_idx_smem_ref=tile_idx_smem_r,
                **kernel_kwargs,
            )

        ext_res = pl.pallas_call(
            _ext_kernel_entry,
            out_shape=ext_out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetch_inputs),
                in_specs=in_specs,
                out_specs=ext_out_specs,
                scratch_shapes=ext_scratch,
            ),
            compiler_params=compiler_params,
            name=ext_name,
        )(*pallas_inputs)
        result = ext_res[0] if enable_chipmate else ext_res
        if use_in_kernel_pn:
            if eff_static_fp8_rs > 0.0 and not keep_static_fp8_out:
                return result.astype(out_dtype) * jnp.asarray(
                    1.0 / eff_static_fp8_rs, dtype=out_dtype
                )
            return result
        return result.reshape(chunk_size * top_k, aligned_n2)[:, :size_n2]

    result = pl.pallas_call(
        functools.partial(kernel_main_fused_rs, **kernel_kwargs),
        out_shape=out_buf_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=9,
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=pallas_name,
    )(*pallas_inputs)

    if use_in_kernel_pn:
        if eff_static_fp8_rs > 0.0 and not keep_static_fp8_out:
            return result.astype(out_dtype) * jnp.asarray(
                1.0 / eff_static_fp8_rs, dtype=out_dtype
            )
        return result
    return result.reshape(chunk_size * top_k, aligned_n2)[:, :size_n2]


# =============================================================================
# Phase 4: run_gmm_fused_rs — High-level shard_map entry point
# =============================================================================

EXPERT = "model"
MLP_DATA = "data"


@functools.partial(
    jax.jit,
    static_argnames=(
        "act_fn",
        "mesh",
        "top_k",
        "ep_axis_name",
        "tile_m",
        "ep_size",
    ),
    compiler_options={
        # "xla_enable_transpose_trace": True,
    },
)
def run_gmm_fused_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    lhs_indices: jax.Array,
    output_indices: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    ep_size: int,
    mesh: jax.sharding.Mesh,
    act_fn: str = "silu",
    top_k: int = 1,
    ep_axis_name: str = EXPERT,
    tile_m: int = 128,
) -> jax.Array:
    """Run the fused GMM + ICI direct-write kernel via shard_map.

    Includes post-kernel topk_weights reduction and all_gather.

    Args:
        hidden_states: bf16[num_tokens, hidden_size] — ungathered input.
        w1: bf16[num_experts, hidden_size, intermediate_size*2] — gate+up weights.
        w2: bf16[num_experts, intermediate_size, hidden_size] — down weights.
        group_sizes: int32[num_experts] — rows per expert.
        lhs_indices: int32[size_m] — gather indices into hidden_states.
        output_indices: int32[size_m] — scatter indices (global token IDs).
        topk_weights: float32[num_tokens, top_k] — topk weights for post-kernel reduction.
        topk_indices: int32[size_m] — per-row topk slot (0..top_k-1).
        ep_size: number of expert-parallel chips.
        mesh: JAX Mesh.
        act_fn: activation function ("silu", "gelu", etc.).
        top_k: number of topk selections per token.
        ep_axis_name: mesh axis name for expert parallelism.
        tile_m: tile size for M dimension.

    Returns:
        (num_tokens, hidden_size) — post-reduction output with all_gather.
    """
    del tile_m  # ignored; see docstring. See _select_fused_rs_block_sizes.
    from jax.sharding import PartitionSpec as P

    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    hidden_size = w2.shape[-1]
    size_m = lhs_indices.shape[0]
    sls = min(pltpu.get_tpu_info().get_sublane_tiling(hidden_states.dtype), size_m)

    # Single source of truth for tile_m. Using the same helper the kernel
    # calls internally guarantees that ``compute_num_gm`` / ``max_num_gm``
    # (outer loop bound) matches the kernel's ``fill_local_metadata``-derived
    # ``local_num_gm``. Mismatch here causes the kernel's final prefetch
    # gather DMA to be unawaited and leaks ``gather_sem`` on kernel exit.
    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=w1.shape[1],
        size_n1=w1.shape[2],
        size_k2=w2.shape[1],
        size_n2=w2.shape[2],
        size_group=num_experts_per_shard,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=hidden_states.dtype,
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        # This entry point does not pipe scales through to gmm_v2_fused_rs,
        # so gmm_v2_fused_rs sees ``is_quantized=False`` internally.
        is_quantized=False,
        quant_block_size=None,
        act_fn=act_fn,
    )
    kernel_tile_m = block_sizes.tile_m

    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    def _run(h, w1l, w2l, gs, go, li, oi, tw, ti):
        num_tokens = h.shape[0]
        chunk_size = num_tokens // ep_size

        num_local_experts = w1l.shape[0]
        group_ids = jnp.arange(gs.shape[0], dtype=jnp.int32)
        go_val = go[0]
        local_start = jnp.sum(jnp.where(group_ids < go_val, gs, 0))
        local_group_sizes = lax.dynamic_slice(gs, (go[0],), (num_local_experts,))
        local_num_gm = compute_num_gm(
            local_group_sizes, kernel_tile_m, sls, start_offset=local_start
        )
        max_num_gm = lax.pmax(local_num_gm, axis_name=ep_axis_name)

        out_buf = gmm_v2_fused_rs(
            h,
            w1l,
            w2l,
            gs,
            li,
            oi,
            act_fn=act_fn,
            output_size=num_tokens,
            group_offset=go,
            group_start=jnp.array([local_start], dtype=jnp.int32),
            topk_indices=ti,
            ep_size=ep_size,
            ep_axis_name=ep_axis_name,
            max_num_gm=max_num_gm,
            caller_tile_m=kernel_tile_m,
            top_k=top_k,
        )

        # Post-kernel reduction: apply topk_weights and sum over top_k.
        # topk_weights arrives pre-sharded [chunk_size, top_k] via shard_map in_specs.
        my_weights = tw
        out_3d = out_buf.reshape(chunk_size, top_k, hidden_size).astype(jnp.float32)
        token_hidden = jnp.sum(
            out_3d * my_weights.astype(jnp.float32)[:, :, None], axis=1
        ).astype(h.dtype)

        return lax.all_gather(token_hidden, axis_name=ep_axis_name, axis=0, tiled=True)

    ep_p_spec = P(ep_axis_name)
    replicated = P()

    return jax.shard_map(
        _run,
        mesh=mesh,
        in_specs=(
            replicated,  # hidden_states (replicated on all chips)
            ep_p_spec,  # w1 (expert-sharded)
            ep_p_spec,  # w2 (expert-sharded)
            replicated,  # group_sizes (replicated)
            ep_p_spec,  # group_offset (expert-sharded)
            replicated,  # lhs_indices (replicated)
            replicated,  # output_indices (replicated)
            P(ep_axis_name, None),  # topk_weights (EP-sharded, local chunk only)
            replicated,  # topk_indices (replicated)
        ),
        out_specs=replicated,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w2,
        group_sizes,
        group_offset,
        lhs_indices,
        output_indices,
        topk_weights,
        topk_indices,
    )
