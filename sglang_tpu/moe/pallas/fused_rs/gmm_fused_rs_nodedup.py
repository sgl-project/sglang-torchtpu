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

from sglang_tpu.envs import envs as tpu_envs

from sglang_tpu.moe.pallas.fused_rs.gmm_v2_gather_scatter import (
    Dimensions,
    FusedDims,
    FusedWeightsRef,
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
    zero_out_end_3d,
    zero_out_start_3d,
)
from sglang_tpu.moe.pallas.fused_rs.tuned_block_sizes import (
    get_fused_rs_tuned_block_sizes,
)
from sglang_tpu.moe.pallas.gmm_v2 import recover_quant_block_size

# Ring slots for the sibling's partial tiles in pair exchange; a sender waits for
# a credit before reusing a slot, so 2 is enough for the cores' lockstep tiles.
_PAIR_RING = 2

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
    tiles = jnp.where(sizes > 0, (sizes + starts % sublane + tile_m - 1) // tile_m, 0)
    tiles_end = jnp.cumsum(tiles)
    num_gm = tiles_end[-1]
    gm = jnp.arange(num_slots, dtype=jnp.int32)
    # Gather-free: searchsorted and indexed gathers lower to serial gather loops
    # on TPU; compare-reduce over the small [S, E] grid fuses.
    group = jnp.minimum(
        jnp.sum((tiles_end[None, :] <= gm[:, None]).astype(jnp.int32), axis=1),
        size_group - 1,
    )
    onehot = group[:, None] == jnp.arange(size_group, dtype=jnp.int32)[None, :]

    def _select(values):
        return jnp.sum(jnp.where(onehot, values[None, :], 0), axis=1)

    tile_in_group = gm - (_select(tiles_end) - _select(tiles))
    first = _select(starts)
    m_offset = jnp.where(
        tile_in_group == 0, first, first - first % sublane + tile_in_group * tile_m
    )
    valid = gm < num_gm
    gm_group = jnp.where(valid, group, 0)
    gm_m_offset = jnp.concatenate(
        [jnp.where(valid, m_offset, local_end), local_end[None]]
    )
    rows = jnp.arange(lhs_indices.shape[0], dtype=jnp.int32)
    my_id = go // size_group
    is_local = jnp.logical_and(rows >= local_start, rows < local_end)
    total_recv = jnp.sum(
        jnp.logical_and(lhs_indices // chunk_size == my_id, jnp.logical_not(is_local))
    )
    head = jnp.stack([local_start, local_end, total_recv, num_gm]).astype(jnp.int32)
    return jnp.concatenate([head, gm_group, gm_m_offset]).astype(jnp.int32)


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
    skip_out_zero: bool = False,
    row_loop_unroll: int = 1,
    staging_slots: int = 3,
    pair_exchange: bool = False,
    pair_meta_base: int = 0,  # group_start offset of per-(tile, owner) run lengths [S * ep]
    pair_meta_stride: int = 0,  # run ranks follow the lengths at this offset
    pair_ring_ref=None,  # VMEM (_PAIR_RING, tile_m, N2//NL, NL): the sibling's partial tiles
    pair_sum_ref=None,  # VMEM (staging_slots, tile_m, N2//NL, NL): own + sibling partials per slot
    pair_recv_sem_ref=None,  # REGULAR (_PAIR_RING,): sibling pieces landed in a ring slot
    pair_credit_ref=None,  # REGULAR (1,): ring slots the sibling has freed
):
    """Fused gather + GMM1 + act + GMM2 + ICI direct-write kernel.

    `row_loop_unroll`: rows per iteration of the per-row gather and write loops.

    `staging_slots`: output staging buffers a tile's deferred writes may stay in
    flight across before the slot is reused.

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
    precomputed_meta = group_start_ref.shape[0] > 1
    with jax.named_scope("routing_scalar_prologue"):
        group_offset = group_offset_ref[0]
        local_start = group_start_ref[0]

        def _sum_local_group_size(local_group_id, count):
            return count + lhs_group_sizes_ref[group_offset + local_group_id]

        if precomputed_meta:
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

        if precomputed_meta:
            total_recv = group_start_ref[2]
        else:
            total_recv = lax.fori_loop(
                0,
                dims.size_m,
                _count_remote_receive,
                jnp.int32(0),
            )

    if pair_exchange:
        sibling = jnp.bitwise_xor(my_id, 1)

    @jax.named_scope("sync_barrier_signal")
    def sync_barrier_signal():
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(ep_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id={ep_axis_name: jnp.int32(i)},
                device_id_type=pltpu.DeviceIdType.MESH,
            )

    @jax.named_scope("sync_barrier_wait")
    def sync_barrier_wait():
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_wait(barrier_sem, ep_size)

    def sync_barrier():
        sync_barrier_signal()
        sync_barrier_wait()

    # Zero-init the output buffer (which is ALSO the ICI DMA landing pad for
    # peers) and WAIT for it BEFORE signaling the entry barrier. A peer only
    # issues its remote writes after its barrier releases, guaranteeing
    # no remote write lands in an out_buf that is still being zeroed.
    zero_src = (
        scatter_fp8_staging_3x_ref.at[0]
        if fp8_direct_write
        else scatter_staging_3x_ref.at[0]
    )
    # Expert TP writes every out_buf row exactly once: each device sends its
    # slice of every routed (token, slot) row, padding tokens included.
    if not skip_out_zero:
        zero_size = zero_out_start_3d(out_buf_ref, zero_src, zero_sem_ref)
        zero_out_end_3d(out_buf_ref, zero_sem_ref, zero_size)

    # Entry barrier: synchronize across all cores after output buffer zeroing
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

    @jax.named_scope("copy_local_metadata")
    def copy_local_metadata():
        num_slots = metadata_ref.gm_id_to_group_id.shape[0]
        num_gm = group_start_ref[3]

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

    if cfgs1.rhs_cfgs.quant_dtype is not None:
        w1_packed = w1_ref.bitcast(jnp.uint32)
    else:
        w1_packed = w1_ref
    if cfgs2.rhs_cfgs.quant_dtype is not None:
        w2_packed = w2_ref.bitcast(jnp.uint32)
    else:
        w2_packed = w2_ref

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

    # --- Weight DMA helpers ---
    @jax.named_scope("start_w1_dma")
    def start_w1_dma(buf_id, expert_id, n_id, k_id=0):
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

    @jax.named_scope("start_w2_dma")
    def start_w2_dma(buf_id, expert_id, n_id, k_id=0):
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
    @jax.named_scope("compute_gmm1_tile")
    def compute_gmm1_tile(buf_id, n_id, k_id, gm_id, m_rows):
        sem_id = gm_id % 2
        _cfgs1 = dataclasses.replace(
            cfgs1, tiles=dataclasses.replace(cfgs1.tiles, tile_m=m_rows)
        )
        k_cols = tile_k1 // num_lanes
        k_offset = k_id * k_cols
        lhs_data = gathered_lhs_2x_ref[
            sem_id, pl.ds(0, m_rows), pl.ds(k_offset, k_cols), :
        ].reshape(-1, dims.size_lhs_sublane, tile_k1)
        w1_tile = w1_buf_ref.at[buf_id]
        w1_sc = w1_scale_buf_ref.at[buf_id] if has_scale else None
        w1_bias_tile = w1_bias_buf_ref if has_bias else None
        w1_weights = WeightsRef(weight=w1_tile, scale=w1_sc, bias=w1_bias_tile)
        if cfgs1.fuse_act is not None:
            w1_up = WeightsRef(
                weight=w1_tile.at[:, tile_n1:],
                scale=_scale_vmem_tile_cols(w1_sc, slice(tile_n1, None)),
                bias=w1_bias_tile.at[:, pl.ds(tile_n1, tile_n1)] if has_bias else None,
            )
            w1_gate = WeightsRef(
                weight=w1_tile.at[:, :tile_n1],
                scale=_scale_vmem_tile_cols(w1_sc, slice(None, tile_n1)),
                bias=w1_bias_tile.at[:, pl.ds(0, tile_n1)] if has_bias else None,
            )
            w1_weights = FusedWeightsRef(gate=w1_gate, up=w1_up)
        inner_kernel(
            lhs_data,
            w1_weights,
            gmm1_out_ref.at[pl.ds(0, m_rows), pl.ds(n_id * tile_n1, tile_n1)],
            partial_out1_ref,
            shared_acc_ref.at[
                pl.ds(0, m_rows), pl.ds(0, tile_n1 * 2 if act_fn else tile_n1)
            ],
            fused_metadata_ref,
            cfgs=_cfgs1,
            gs_gate_ref=w1_gs_gate_ref,
            gs_up_ref=w1_gs_up_ref,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k1,
            _gm_id=0,
            _n_id=n_id,
        )

    @jax.named_scope("compute_gmm2_tile")
    def compute_gmm2_tile(buf_id, n_id, k_id, gm_id, m_rows):
        sem_id = gm_id % 2
        _cfgs2 = dataclasses.replace(
            cfgs2, tiles=dataclasses.replace(cfgs2.tiles, tile_m=m_rows)
        )
        lhs_data = intermediate_ref[
            pl.ds(0, m_rows // dims.size_lhs_sublane), :, pl.ds(k_id * tile_k2, tile_k2)
        ]
        w2_tile = w2_buf_ref.at[buf_id]
        w2_sc = w2_scale_buf_ref.at[buf_id] if has_scale else None
        inner_kernel(
            lhs_data,
            WeightsRef(
                weight=w2_tile, scale=w2_sc, bias=w2_bias_buf_ref if has_bias else None
            ),
            tiled_out_2x_ref.at[
                sem_id, pl.ds(0, m_rows), pl.ds(n_id * tile_n2, tile_n2)
            ],
            partial_out2_ref,
            shared_acc_ref.at[pl.ds(0, m_rows), pl.ds(0, tile_n2)],
            fused_metadata_ref,
            cfgs=_cfgs2,
            gs_gate_ref=w2_gs_ref,
            gs_up_ref=w2_gs_ref,
            scatter_mode=True,
            _k_id=k_id,
            _num_k=num_k2,
            _gm_id=0,
            _n_id=n_id,
        )

    # Initialize per-slot DMA counts to 0.
    for _s in range(staging_slots):
        gm_id_ref[1 + _s] = jnp.int32(0)  # send counts
        gm_id_ref[1 + staging_slots + _s] = jnp.int32(0)  # local write counts

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
    M_BUCKETS = [b for b in (8, 16, 32, 64, 128, 256, 512) if _sls_b <= b <= tile_m]
    if not M_BUCKETS or M_BUCKETS[-1] != tile_m:
        M_BUCKETS = sorted(set(M_BUCKETS) | {tile_m})
    # fp4-only gate. Bucketing materializes a separate f32 matmul accumulator
    # (~0.7MB) per lax.switch branch. That headroom only exists with fp4 weights
    # (0.5 B/param); fp8 (1 B) / bf16 (2 B) weight buffers are 2-4x larger and, on
    # tight configs (e.g. EP8 E_32 M_256), leave less than the bucket transient of
    # VMEM free -> scoped-VMEM OOM. All measured bucketing gains are on fp4 (the
    # prod decode dtype) anyway, so only bucket when the weights are fp4.
    _w_qdt = cfgs1.rhs_cfgs.quant_dtype
    _weight_is_fp4 = _w_qdt is not None and jax.dtypes.itemsize_bits(_w_qdt) <= 4
    if (only := tpu_envs.SGLANG_TPU_FUSED_RS_M_BUCKETS.get()) is not None:
        # every bucket variant is a full copy of the GMM code; fewer keeps it in
        # instruction memory
        keep = {int(b) for b in only.split(",") if b}
        M_BUCKETS = [b for b in M_BUCKETS if b in keep or b == tile_m]
    _USE_BUCKETS = len(M_BUCKETS) > 1 and _weight_is_fp4

    # --- Main padded gm loop ---
    @jax.named_scope("wait_staging_slot_writes")
    def wait_staging_slot_writes(slot):
        """Wait for and clear the row writes still reading staging ``slot``."""
        pending_local = gm_id_ref[1 + staging_slots + slot]
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
        gm_id_ref[1 + staging_slots + slot] = jnp.int32(0)

    pair_pieces = [b for b in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if b <= tile_m]
    _sls_pair = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)

    def _pair_count_pieces(num_valid):
        n = jnp.int32(0)
        for b in pair_pieces:
            n = n + ((num_valid & b) != 0).astype(jnp.int32)
        return n

    @jax.named_scope("pair_send")
    def _pair_send(gm_id, stg_id, row0, num_valid):
        """Ship tile gm_id's valid partial rows to the sibling's ring slot (power-of-two DMAs)."""
        ring_slot = gm_id % _PAIR_RING

        @pl.when(gm_id >= _PAIR_RING)
        def _():
            pl.semaphore_wait(pair_credit_ref.at[0], 1)  # the sibling freed this slot

        off = jnp.int32(0)
        for b in pair_pieces:
            has = (num_valid & b) != 0

            @pl.when(has)
            def _(b=b, off=off):
                pltpu.make_async_remote_copy(
                    src_ref=scatter_staging_3x_ref.at[stg_id, pl.ds(row0 + off, b), :, :],
                    dst_ref=pair_ring_ref.at[ring_slot, pl.ds(row0 + off, b), :, :],
                    send_sem=send_sems_ref.at[stg_id],
                    recv_sem=pair_recv_sem_ref.at[ring_slot],
                    device_id={ep_axis_name: sibling},
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            off = off + jnp.where(has, b, 0)
        gm_id_ref[1 + stg_id] = gm_id_ref[1 + stg_id] + _pair_count_pieces(num_valid)

    @jax.named_scope("pair_finish")
    def _pair_finish(g):
        """Tile g: add the sibling's partial rows, free its ring slot, write each owner's run.

        Rows of a tile are token-sorted, so the rows of each owner form one run whose ranks
        among that owner's rows are consecutive. A run goes to the owner's receive block
        for this source: own rows (block 0) and the sibling's (block 1) carry one core's
        partial; another chip's run carries the pair sum from the core of the owner's parity.
        """
        stg = g % staging_slots
        ring_slot = g % _PAIR_RING
        m_st = metadata_ref.gm_id_to_m_offset[g]
        m_en = metadata_ref.gm_id_to_m_offset[g + 1]
        row0 = m_st % _sls_pair
        n_pieces = _pair_count_pieces(m_en - m_st)

        @pl.when(n_pieces > 0)
        def _():
            pl.semaphore_wait(pair_recv_sem_ref.at[ring_slot], n_pieces)

        pair_sum_ref[stg] = (
            scatter_staging_3x_ref[stg].astype(jnp.float32)
            + pair_ring_ref[ring_slot].astype(jnp.float32)
        ).astype(pair_sum_ref.dtype)
        pltpu.semaphore_signal(
            pair_credit_ref.at[0],
            device_id={ep_axis_name: sibling},
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        my_chip = my_id // 2
        num_chips = ep_size // 2
        rows_per_block = out_buf_ref.shape[0] // (num_chips + 1)

        def _owner(d, carry):
            src_off, sent = carry
            run_len = group_start_ref[pair_meta_base + g * ep_size + d]
            rank = group_start_ref[pair_meta_base + pair_meta_stride + g * ep_size + d]
            d_chip = d // 2
            same_chip = d_chip == my_chip
            forward = jnp.logical_and(~same_chip, (d % 2) == (my_id % 2))
            block = jnp.where(
                d == my_id,
                0,
                jnp.where(same_chip, 1, 1 + (my_chip - d_chip + num_chips) % num_chips),
            )
            dst0 = block * rows_per_block + rank
            send = jnp.logical_and(same_chip | forward, run_len > 0)
            off = jnp.int32(0)
            for b in pair_pieces:
                has = jnp.logical_and(send, (run_len & b) != 0)

                @pl.when(jnp.logical_and(has, ~forward))
                def _(b=b, off=off):
                    pltpu.make_async_remote_copy(
                        src_ref=scatter_staging_3x_ref.at[stg, pl.ds(row0 + src_off + off, b), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(dst0 + off, b), :, :],
                        send_sem=send_sems_ref.at[stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: d},
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                @pl.when(jnp.logical_and(has, forward))
                def _(b=b, off=off):
                    pltpu.make_async_remote_copy(
                        src_ref=pair_sum_ref.at[stg, pl.ds(row0 + src_off + off, b), :, :],
                        dst_ref=out_buf_ref.at[pl.ds(dst0 + off, b), :, :],
                        send_sem=send_sems_ref.at[stg],
                        recv_sem=recv_sem_ref.at[0],
                        device_id={ep_axis_name: d},
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                off = off + jnp.where((run_len & b) != 0, b, 0)
                sent = sent + has.astype(jnp.int32)
            return src_off + run_len, sent

        _, sent = lax.fori_loop(0, ep_size, _owner, (jnp.int32(0), jnp.int32(0)))
        gm_id_ref[1 + stg] = gm_id_ref[1 + stg] + sent

    @jax.named_scope("gm_loop_body")
    def gm_loop_body(gm_id, _):
        is_active = gm_id < local_num_gm
        sem_id = gm_id % 2
        stg_id = gm_id % staging_slots

        # Steps 1-9: Only when this chip still has compute work.
        @pl.when(is_active)
        def _():
            gm_id_ref[0] = gm_id
            fused_metadata_ref.gm_id_to_group_id[0] = metadata_ref.gm_id_to_group_id[
                gm_id
            ]
            fused_metadata_ref.gm_id_to_m_offset[0] = metadata_ref.gm_id_to_m_offset[
                gm_id
            ]
            fused_metadata_ref.gm_id_to_m_offset[1] = metadata_ref.gm_id_to_m_offset[
                gm_id + 1
            ]
            expert_id = fused_metadata_ref.gm_id_to_group_id[0]

            # DMA gather. When indices are packed, set divisor=top_k so the
            # gather extracts the actual lhs_idx via integer division.
            _gather_divisor = top_k if pack_indices else 1

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
                    unroll=row_loop_unroll,
                )

            @jax.named_scope("dma_gather_prefetch")
            @pl.when(gm_id + 1 < local_num_gm)
            def _():
                dma_gather_gm_start(
                    hidden_states_ref,
                    gathered_lhs_2x_ref.at[1 - sem_id],
                    lhs_indices_ref,
                    gather_sem_ref.at[1 - sem_id],
                    gm_id + 1,
                    metadata_ref,
                    divisor=_gather_divisor,
                    unroll=row_loop_unroll,
                )

            # --- Weight DMA (overlapped with gather) ---
            # w1 prologue: load first num_w1_bufs tiles (gm==0 only;
            # for gm>0, tiles were prefetched during previous gm's GMM2).
            total_w1_steps = num_n1 * num_k1
            total_w2_steps = num_n2 * num_k2
            # Expert ping-pong: W1 and W2 of expert e live in buffer e % 2, and the next
            # expert's are fetched at the start of this tile unless it shares the parity
            # (an expert without rows in between), when they follow the matmuls instead.
            pingpong = (
                tpu_envs.SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG.get()
                and total_w1_steps == 1
                and total_w2_steps == 1
                and num_w1_bufs >= 2
                and num_w2_bufs >= 2
            )
            wbuf = expert_id % 2

            # ---- Same-expert weight caching (Idea 1) ----
            # When num_w*_bufs >= total_w*_steps, ALL of the current expert's
            # W1/W2 weight tiles fit simultaneously in VMEM. If the next gm
            # tile uses the same expert, we can SKIP the cross-gm prefetch
            # entirely — the buffers already hold the right data. This is
            # very common: at M=65536, E=32, tile_m=128 each expert spans
            # ~16 consecutive gm tiles → up to 16x fewer weight DMAs.
            #
            # When buffers can't hold all tiles (bf16 case where total_steps>num_bufs),
            # the buffer rotation already overwrites earlier tiles within a
            # single gm — same-expert caching is unsafe there. The static
            # checks below disable the optimization in that case.
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

            def _next_expert_fetch(fetch, *, same_parity):
                # fetch(next_e) when the next tile starts a new expert whose parity
                # matches this one's exactly when ``same_parity``.
                @pl.when(gm_id + 1 < local_num_gm)
                def _():
                    next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
                    parity_match = (next_e % 2 == wbuf) == same_parity

                    @pl.when(jnp.logical_and(next_e != expert_id, parity_match))
                    def _():
                        fetch(next_e)

            if pingpong:

                @pl.when(gm_id == 0)
                def _():
                    start_w1_dma(wbuf, expert_id, 0, 0)
                    start_w2_dma(wbuf, expert_id, 0, 0)

                def _fetch_both(next_e):
                    start_w1_dma(1 - wbuf, next_e, 0, 0)
                    start_w2_dma(1 - wbuf, next_e, 0, 0)

                _next_expert_fetch(_fetch_both, same_parity=False)
            else:

                @pl.when(gm_id == 0)
                def _():
                    for _i in range(min(num_w1_bufs, total_w1_steps)):
                        start_w1_dma(_i, expert_id, _i // num_k1, _i % num_k1)

                # Early w2 prefetch — skip if cache hit (same expert as prev gm).
                @pl.when(jnp.logical_not(is_same_w2))
                def _():
                    for _i in range(min(num_w2_bufs, total_w2_steps)):
                        start_w2_dma(_i, expert_id, _i // num_k2, _i % num_k2)

            # Wait for gather (weight DMAs running in parallel).
            dma_gather_gm_wait(
                gathered_lhs_2x_ref.at[sem_id],
                gather_sem_ref.at[sem_id],
                gm_id,
                metadata_ref,
            )

            # Pick the smallest row-bucket >= this tile's real rows (capped at
            # tile_m), then dispatch each matmul over that bucket via lax.switch.
            # NOTE: this is applied UNCONDITIONALLY to every gm-tile. An earlier
            # attempt to guard it with lax.cond(is_last_tile, ...) — only
            # bucketing an expert's partial last tile — regressed >=2048 tokens
            # ~4x: the per-tile data-dependent branch diverges once experts span
            # multiple tiles, which breaks the gm-loop's DMA/compute pipelining.
            # Uniform control flow (same switch structure every tile) pipelines
            # cleanly and only costs ~0.5% at the largest sizes.
            _m_s = metadata_ref.gm_id_to_m_offset[gm_id]
            _m_e = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            _tile_real = _m_e - (_m_s - _m_s % dims.size_lhs_sublane)
            if _USE_BUCKETS:
                # Index of the smallest bucket >= real rows (no captured array
                # constant: sum scalar comparisons against Python-int buckets).
                bucket_idx = jnp.int32(0)
                for _b in M_BUCKETS:
                    bucket_idx += (_tile_real > _b).astype(jnp.int32)
                bucket_idx = jnp.minimum(bucket_idx, len(M_BUCKETS) - 1)

            def _dispatch(fn, *args):
                if _USE_BUCKETS:
                    lax.switch(
                        bucket_idx,
                        [
                            (lambda b=b: (fn(*args, b), jnp.int32(0))[1])
                            for b in M_BUCKETS
                        ],
                    )
                else:
                    fn(*args, tile_m)

            # GMM1 loop.
            for step in range(total_w1_steps):
                _n1 = step // num_k1
                _k1 = step % num_k1
                buf_id = wbuf if pingpong else step % num_w1_bufs

                # Skip wait if W1 was cached from previous gm tile
                # (same expert + all tiles fit in buffers).
                @pl.when(jnp.logical_not(is_same_w1))
                def _():
                    wait_w1_dma(buf_id)

                # Consume this rotating slot on the MXU BEFORE refilling it.
                # ns % num_w1_bufs == buf_id, so starting the next DMA first
                # would overwrite the tile this matmul is still reading (WAR
                # race between the refill DMA and compute -> nondeterminism).
                _dispatch(compute_gmm1_tile, buf_id, _n1, _k1, gm_id)
                if step + num_w1_bufs < total_w1_steps:
                    ns = step + num_w1_bufs
                    start_w1_dma(ns % num_w1_bufs, expert_id, ns // num_k1, ns % num_k1)
                if pingpong:
                    _next_expert_fetch(
                        lambda next_e: start_w1_dma(wbuf, next_e, 0, 0),
                        same_parity=True,
                    )
                elif step == 0 and can_cache_w1:

                    @pl.when(gm_id + 1 < local_num_gm)
                    def _():
                        next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
                        # Skip cross-prefetch when next gm reuses same expert
                        # (caching only makes sense when buffers fit all tiles).
                        next_same = jnp.logical_and(
                            next_e == expert_id, jnp.bool_(can_cache_w1)
                        )

                        @pl.when(jnp.logical_not(next_same))
                        def _():
                            start_w1_dma(0, next_e, 0, 0)

            @pl.when(gm_id + 1 < local_num_gm)
            def _():
                next_e = metadata_ref.gm_id_to_group_id[gm_id + 1]
                next_same = jnp.logical_and(
                    next_e == expert_id, jnp.bool_(can_cache_w1)
                )

                @pl.when(jnp.logical_not(next_same))
                def _():
                    # Rotating W1 tiles still need buffer zero until GMM1 finishes.
                    for _i in range(
                        1 if can_cache_w1 else 0, min(num_w1_bufs, total_w1_steps)
                    ):
                        start_w1_dma(_i, next_e, _i // num_k1, _i % num_k1)

            with jax.named_scope("interlude"):
                # Copy gmm1_out (f32) to intermediate_ref (bf16, scatter layout).
                # Activation was already applied inside inner_kernel via fuse_act.
                def _interlude(m_rows):
                    gmm1_result = gmm1_out_ref[pl.ds(0, m_rows)]
                    k2_pad = dims.size_k2 - gmm1_result.shape[1]
                    if k2_pad > 0:
                        gmm1_result = jnp.concatenate(
                            [
                                gmm1_result,
                                jnp.zeros((m_rows, k2_pad), dtype=gmm1_result.dtype),
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
                buf_id = wbuf if pingpong else step % num_w2_bufs

                # Skip wait if W2 was cached (same expert as prev + buffers fit all).
                @pl.when(jnp.logical_not(is_same_w2))
                def _():
                    wait_w2_dma(buf_id)

                # See GMM1 loop: consume the slot before refilling it (WAR).
                _dispatch(compute_gmm2_tile, buf_id, _n2, _k2, gm_id)
                if step + num_w2_bufs < total_w2_steps:
                    ns = step + num_w2_bufs
                    start_w2_dma(ns % num_w2_bufs, expert_id, ns // num_k2, ns % num_k2)
                if pingpong:
                    _next_expert_fetch(
                        lambda next_e: start_w2_dma(wbuf, next_e, 0, 0),
                        same_parity=True,
                    )

            m_st = metadata_ref.gm_id_to_m_offset[gm_id]
            m_en = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            _sls = pltpu.get_tpu_info().get_sublane_tiling(out_dtype)
            _ml = m_st % _sls

            if defer_writes:
                wait_staging_slot_writes(stg_id)

            with jax.named_scope("reshape_gmm2_output"):
                # Only the bucket's rows: the rows sent below all lie under it, and the
                # packed-row relayout costs ~1 us per 256 rows.
                def _relayout(m_rows):
                    scatter_staging_3x_ref[stg_id, pl.ds(0, m_rows)] = tiled_out_2x_ref[
                        sem_id, pl.ds(0, m_rows)
                    ].reshape(
                        m_rows,
                        scatter_staging_3x_ref.shape[2],
                        scatter_staging_3x_ref.shape[3],
                    )

                _dispatch(_relayout)

            # Step 10: Direct-write — single pass over valid rows only.
            # Each row either ICI-sends (remote) or DMA-copies (local) to
            # direct_write_buf[local_row * top_k + topk_idx].
            # Loop bound is m_en - m_st (valid rows), not tile_m, eliminating
            # wasted iterations on padding rows.
            num_valid = m_en - m_st

            if pair_exchange:
                # tile g's rows go to the sibling now; tile g-1 completes while they fly
                _pair_send(gm_id, stg_id, _ml, num_valid)

                @pl.when(gm_id >= 1)
                def _():
                    _pair_finish(gm_id - 1)

            @jax.named_scope("direct_write_rows")
            def _do_direct_write():
                def _write_row(i, carry):
                    send_sz, local_sz = carry
                    row_idx = _ml + i
                    # Read indices. Either packed (one ref) or separate.
                    if pack_indices:
                        # combined = lhs_idx * top_k + topk_slot
                        #          = (dest_chip * chunk_size + local_row) * top_k + topk_slot
                        # So:
                        #   write_pos = local_row * top_k + topk_slot
                        #             = combined % (chunk_size * top_k)
                        #   dest_chip = combined // (chunk_size * top_k)
                        # When chunk_size * top_k is a power of 2 (typical),
                        # XLA lowers these to a shift + mask — essentially free.
                        # No need to compute oi/topk_idx/local_row separately.
                        combined = lhs_indices_ref[m_st + i]
                        cs_tk = chunk_size * top_k
                        write_pos = combined % cs_tk
                        dest_chip = combined // cs_tk
                    else:
                        oi = lhs_indices_ref[m_st + i]
                        topk_idx = topk_indices_ref[m_st + i]
                        dest_chip = oi // chunk_size
                        local_row = oi % chunk_size
                        write_pos = local_row * top_k + topk_idx
                    is_local = dest_chip == my_id

                    if fp8_direct_write:
                        # Quantize each completed expert row before the direct
                        # write. Remote ICI then moves FP8 payload plus one
                        # fp32 row scale instead of the full bf16 activation row.
                        row_f32 = scatter_staging_3x_ref[stg_id, row_idx, :, :].astype(
                            jnp.float32
                        )
                        valid_n2 = (
                            jnp.arange(row_f32.shape[0])[:, None] * row_f32.shape[1]
                            + jnp.arange(row_f32.shape[1])[None, :]
                            < dims.original_n2
                        )
                        fp8_max = jnp.array(
                            jnp.finfo(jnp.float8_e4m3fn).max, dtype=jnp.float32
                        )
                        row_scale = (
                            jnp.maximum(
                                jnp.max(jnp.where(valid_n2, jnp.abs(row_f32), 0.0)),
                                jnp.array(1e-6, dtype=jnp.float32),
                            )
                            / fp8_max
                        )
                        scatter_scale_3x_ref[stg_id, row_idx, :] = (
                            row_scale + jnp.zeros((128,), dtype=jnp.float32)
                        )
                        scatter_fp8_staging_3x_ref[stg_id, row_idx, :, :] = jnp.where(
                            valid_n2,
                            jnp.clip(row_f32 / row_scale, -fp8_max, fp8_max),
                            0.0,
                        ).astype(jnp.float8_e4m3fn)

                        @pl.when(~is_local)
                        def _():
                            target_device_id = dest_chip
                            pltpu.make_async_remote_copy(
                                src_ref=scatter_fp8_staging_3x_ref.at[
                                    stg_id, pl.ds(row_idx, 1), :, :
                                ],
                                dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                                send_sem=send_sems_ref.at[stg_id],
                                recv_sem=recv_sem_ref.at[0],
                                device_id={ep_axis_name: target_device_id},
                                device_id_type=pltpu.DeviceIdType.MESH,
                            ).start()
                            if fp8_direct_write:
                                pltpu.make_async_remote_copy(
                                    src_ref=scatter_scale_3x_ref.at[
                                        stg_id, pl.ds(row_idx, 1), :
                                    ],
                                    dst_ref=out_scale_ref.at[pl.ds(write_pos, 1), :],
                                    send_sem=scale_send_sems_ref.at[stg_id],
                                    recv_sem=scale_recv_sem_ref.at[0],
                                    device_id={ep_axis_name: target_device_id},
                                    device_id_type=pltpu.DeviceIdType.MESH,
                                ).start()

                        @pl.when(is_local)
                        def _():
                            pltpu.make_async_copy(
                                src_ref=scatter_fp8_staging_3x_ref.at[
                                    stg_id, pl.ds(row_idx, 1), :, :
                                ],
                                dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                                sem=local_write_sems_ref.at[stg_id],
                            ).start()
                            if fp8_direct_write:
                                pltpu.make_async_copy(
                                    src_ref=scatter_scale_3x_ref.at[
                                        stg_id, pl.ds(row_idx, 1), :
                                    ],
                                    dst_ref=out_scale_ref.at[pl.ds(write_pos, 1), :],
                                    sem=scale_local_write_sems_ref.at[stg_id],
                                ).start()

                    else:

                        @pl.when(~is_local)
                        def _():
                            target_device_id = dest_chip
                            pltpu.make_async_remote_copy(
                                src_ref=scatter_staging_3x_ref.at[
                                    stg_id, pl.ds(row_idx, 1), :, :
                                ],
                                dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                                send_sem=send_sems_ref.at[stg_id],
                                recv_sem=recv_sem_ref.at[0],
                                device_id={ep_axis_name: target_device_id},
                                device_id_type=pltpu.DeviceIdType.MESH,
                            ).start()

                        @pl.when(is_local)
                        def _():
                            pltpu.make_async_copy(
                                src_ref=scatter_staging_3x_ref.at[
                                    stg_id, pl.ds(row_idx, 1), :, :
                                ],
                                dst_ref=out_buf_ref.at[pl.ds(write_pos, 1), :, :],
                                sem=local_write_sems_ref.at[stg_id],
                            ).start()

                    return (
                        send_sz + lax.select(~is_local, jnp.int32(1), jnp.int32(0)),
                        local_sz + lax.select(is_local, jnp.int32(1), jnp.int32(0)),
                    )

                if row_loop_unroll == 1:
                    send_sz, local_sz = lax.fori_loop(
                        0, num_valid, _write_row, (jnp.int32(0), jnp.int32(0))
                    )
                else:

                    def _write_block(b, carry):
                        for u in range(row_loop_unroll):
                            carry = _write_row(b * row_loop_unroll + u, carry)
                        return carry

                    num_blocks = num_valid // row_loop_unroll
                    send_sz, local_sz = lax.fori_loop(
                        num_blocks * row_loop_unroll,
                        num_valid,
                        _write_row,
                        lax.fori_loop(
                            0,
                            num_blocks,
                            _write_block,
                            (jnp.int32(0), jnp.int32(0)),
                        ),
                    )
                gm_id_ref[1 + stg_id] = gm_id_ref[1 + stg_id] + send_sz
                gm_id_ref[1 + staging_slots + stg_id] = (
                    gm_id_ref[1 + staging_slots + stg_id] + local_sz
                )

                @pl.when(jnp.logical_and(local_sz > 0, not defer_writes))
                def _():
                    pl.semaphore_wait(local_write_sems_ref.at[stg_id], local_sz)
                    if fp8_direct_write:
                        pl.semaphore_wait(
                            scale_local_write_sems_ref.at[stg_id], local_sz
                        )

                @pl.when(jnp.logical_and(send_sz > 0, not defer_writes))
                def _():
                    pl.semaphore_wait(send_sems_ref.at[stg_id], send_sz)
                    if fp8_direct_write:
                        pl.semaphore_wait(scale_send_sems_ref.at[stg_id], send_sz)

            if not pair_exchange:
                _do_direct_write()

        return _

    lax.fori_loop(0, max_num_gm, gm_loop_body, None)
    if pair_exchange:

        @pl.when(local_num_gm > 0)
        def _():
            _pair_finish(local_num_gm - 1)

    if defer_writes:
        # A kernel must not exit with DMAs or semaphores outstanding.
        for _s in range(staging_slots):
            wait_staging_slot_writes(_s)
    if pair_exchange:
        # one credit per tile the sibling consumed; the first _PAIR_RING were never waited
        @pl.when(local_num_gm > 0)
        def _():
            pl.semaphore_wait(
                pair_credit_ref.at[0], jnp.minimum(local_num_gm, _PAIR_RING)
            )

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
    skip_out_zero: bool = False,
    row_loop_unroll: int = 1,
    staging_slots: int = 3,
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
        staging_slots=staging_slots,
    )


def _pair_exchange_metadata(
    group_start: jax.Array,
    local_idx: jax.Array,
    *,
    group_offset: jax.Array,
    size_group: int,
    pack_indices: bool,
    top_k: int,
    chunk_size: int,
    ep_size: int,
    num_slots: int,
    num_blocks: int,
) -> tuple[jax.Array, int, int]:
    """Append per-(tile, owner) run lengths and ranks to the local-rows metadata.

    A local row's owner is its token's rank; rows are expert-sorted then token-sorted,
    so each tile holds one consecutive run per owner. The receive count becomes the
    DMA pieces arriving from every source block (self, sibling, one core per other chip).
    """
    tokens = local_idx // top_k if pack_indices else local_idx
    owner = tokens // chunk_size
    onehot = (owner[:, None] == jnp.arange(ep_size, dtype=jnp.int32)[None, :]).astype(jnp.int32)
    counts = jnp.concatenate([jnp.zeros((1, ep_size), jnp.int32), jnp.cumsum(onehot, axis=0)])
    m_offset = group_start[4 + num_slots : 4 + 2 * num_slots + 1]
    run_rank = counts[m_offset[:num_slots]]
    run_len = counts[m_offset[1:]] - run_rank
    my_rank = group_offset[0] // size_group
    my_len = jnp.take(run_len, my_rank, axis=1)
    pieces = sum(((my_len & b) != 0).astype(jnp.int32) for b in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))
    total_recv = num_blocks * jnp.sum(pieces)
    head = group_start.at[2].set(total_recv.astype(jnp.int32))
    base = head.shape[0]
    stride = num_slots * ep_size
    meta = jnp.concatenate([head, run_len.reshape(-1), run_rank.reshape(-1)]).astype(jnp.int32)
    return meta, base, stride


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
        "local_rows",
        "tiled_output",
        "skip_out_zero",
        "row_loop_unroll",
        "staging_slots",
        "pair_exchange",
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
    local_rows: bool = False,
    tiled_output: bool = False,
    skip_out_zero: bool = False,
    row_loop_unroll: int = 1,
    staging_slots: int = 3,
    pair_exchange: bool = False,
) -> jax.Array:
    """Fused gather + GMM1 + act + GMM2 + ICI direct-write.

    ``pair_exchange`` (expert TP, logical ranks 2c and 2c+1 on one chip): for a row
    owned on another chip, the core of the owner's parity adds its sibling's partial
    (sent on-chip into a VMEM ring) to its own and sends the pair sum; the other
    core sends nothing for that row. out_buf is then zero-initialized.

    ``local_rows`` (expert TP only): the scalar-prefetched row indices hold just
    this device's contiguous block of routed rows instead of all ``ep_size`` blocks.
    ``skip_out_zero`` (expert TP): do not zero out_buf before the writes.
    ``tiled_output``: return the bf16 result in the kernel's own
    (rows, size_n2 // 128, 128) layout when N2 is unpadded, so a row-split
    reduction can read it without a relayout copy.

    Output: (chunk_size * top_k, size_n2) per chip — raw expert contributions.
    Post-kernel reduction applies topk_weights and sums over top_k dim.
    """
    size_group, size_k1, size_n1 = w1.shape
    _, size_k2, size_n2 = w2.shape
    size_m = lhs_indices.shape[0]
    chunk_size = output_size // ep_size
    # Rows the kernel indexes; tile selection keeps the global size_m to match
    # the caller's lookup.
    kernel_size_m = size_m // ep_size if local_rows else size_m

    is_quantized = w1_scale is not None
    if is_quantized:
        assert w2_scale is not None
        rhs1_quant_block_size = recover_quant_block_size(size_k1, w1_scale.shape[1])
        rhs2_quant_block_size = recover_quant_block_size(size_k2, w2_scale.shape[1])
        quant_block_size = rhs1_quant_block_size
        # Scales stay in their stored dtype when it is bf16 or fp32: the matmul
        # casts each tile slice, while a full-tensor cast here would re-copy
        # every expert's scales on each call (group-16 scales are large).
        scale_dtype = (
            w1_scale.dtype
            if w1_scale.dtype in (jnp.bfloat16, jnp.float32, jnp.float8_e4m3fn)
            else jnp.dtype(jnp.float32)
        )
        w1_scale = w1_scale.astype(scale_dtype)
        w2_scale = w2_scale.astype(scale_dtype)
        if w1_scale.ndim != w2_scale.ndim or w1_scale.ndim not in (3, 4):
            raise ValueError(
                "fused-RS scales must both be [E, K_blocks, N] or "
                f"[E, K_blocks, 1, N], got {w1_scale.shape} and {w2_scale.shape}"
            )
        scale_has_unit_dim = w1_scale.ndim == 4
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
    sls = pltpu.get_tpu_info().get_sublane_tiling(hidden_states.dtype)
    out_dtype = hidden_states.dtype
    intermediate_size = size_n1 // 2

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
    if aligned_n1 != size_n1:
        n1_pad = aligned_n1 - size_n1
        w1 = jnp.pad(w1, ((0, 0), (0, 0), (0, n1_pad)))
        if is_quantized:
            w1_scale = jnp.pad(
                w1_scale, ((0, 0),) * (w1_scale.ndim - 1) + ((0, n1_pad),)
            )
        new_k2 = aligned_n1 // 2
        if new_k2 != size_k2:
            w2 = jnp.pad(w2, ((0, 0), (0, new_k2 - size_k2), (0, 0)))
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
        if w1_scale.shape[1] < total_scale_blocks1:
            w1_scale = jnp.concatenate(
                [
                    w1_scale,
                    jnp.repeat(
                        w1_scale[:, -1:],
                        total_scale_blocks1 - w1_scale.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
            )
        nqb2 = -(-tile_k2 // rhs2_quant_block_size)
        total_scale_blocks2 = (size_k2 // tile_k2) * nqb2
        if w2_scale.shape[1] < total_scale_blocks2:
            w2_scale = jnp.concatenate(
                [
                    w2_scale,
                    jnp.repeat(
                        w2_scale[:, -1:],
                        total_scale_blocks2 - w2_scale.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
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
        size_m=kernel_size_m,
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

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=kernel_size_m,
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
            size_m=kernel_size_m,
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
    max_num_gm_static = size_group + pl.cdiv(kernel_size_m, tile_m) - 1
    n2_cols = aligned_n2 // num_lanes

    scratch_shapes = [
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm_static,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm_static + 1,), jnp.int32),
        ),
        # fused_metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((1,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((2,), jnp.int32),
        ),
        # gathered_lhs_2x_ref
        pltpu.VMEM((2, tile_m, padded_k1 // num_lanes, num_lanes), out_dtype),
        # gmm1_out_ref
        pltpu.VMEM((tile_m, size_n1 // 2 if act_fn else size_n1), jnp.float32),
        # intermediate_ref
        pltpu.VMEM((tile_m // size_lhs_sublane, size_lhs_sublane, size_k2), out_dtype),
        # tiled_out_2x_ref
        pltpu.VMEM((2, tile_m, aligned_n2), out_dtype),
        # scatter_staging_3x_ref — 4D: first-dim selects the staging slot,
        # avoids dynamic offset that triggers Mosaic VMEM tiling alignment error.
        pltpu.VMEM((staging_slots, tile_m, n2_cols, num_lanes), out_dtype),
        # partial_out1_ref
        pltpu.VMEM((size_lhs_sublane, tile_n1 * 2 if act_fn else tile_n1), jnp.float32),
        # partial_out2_ref
        pltpu.VMEM((size_lhs_sublane, tile_n2), jnp.float32),
        # shared_acc_ref
        pltpu.VMEM(
            (tile_m, max(tile_n1 * 2 if act_fn else tile_n1, tile_n2)), jnp.float32
        ),
        # gather_sem_ref
        pltpu.SemaphoreType.DMA((2,)),
        # staging_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # zero_sem_ref
        pltpu.SemaphoreType.DMA((1,)),
        # gm_id_ref: [0]=gm_id, then per-slot send counts, then per-slot local counts
        pltpu.SMEM((1 + 2 * staging_slots,), jnp.int32),
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
        # send_sems_ref (per staging slot)
        pltpu.SemaphoreType.REGULAR((staging_slots,)),
        # local_write_sems_ref (per staging slot)
        pltpu.SemaphoreType.REGULAR((staging_slots,)),
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
    w2_bias_input = w2_bias if has_bias else jnp.zeros((1, 1, 1), jnp.float32)
    max_num_gm_arr = jnp.array([max_num_gm], dtype=jnp.int32)

    # Output buffer — also serves as ICI DMA target. In the FP8 direct-write
    # path this stores quantized payload rows; row scales are returned as a
    # second output and dequantized immediately after the kernel.
    payload_dtype = jnp.float8_e4m3fn if fp8_direct_write else out_dtype
    out_buf_init = jax.ShapeDtypeStruct(
        (chunk_size * top_k, n2_cols, num_lanes), payload_dtype
    )
    # TPU VMEM vector slices must be tile-aligned. Store the scalar row scale
    # as a padded 128-wide row and use column 0 when dequantizing.
    out_scale_init = jax.ShapeDtypeStruct((chunk_size * top_k, 128), jnp.float32)

    # Decide whether to pack lhs_indices + topk_slot_indices into a single
    # SMEM scalar prefetch. Packing's per-row unpack (1 mod + 1 div via the
    # math identity) is cheap (~bit ops for power-of-2 chunk_size*top_k) but
    # NOT free — costs ~130-200 us at size_m=65K. So we only pack when needed
    # to fit SMEM budget (1 MB).
    #
    # Two int32[size_m] arrays + internal scratch (~10-20 KB) need to fit in
    # 1 MB SMEM. Pack when 2 separate refs would exceed ~960 KB (2 * size_m * 4).
    # That's size_m > 120K (e.g., prefill_16384 with top_k=8 has size_m=131K).
    pack_indices = kernel_size_m > 120_000
    if pack_indices:
        # combined = lhs_idx * top_k + topk_slot
        primary_idx_ref = lhs_indices.astype(jnp.int32) * top_k + topk_indices.astype(
            jnp.int32
        )
        secondary_idx_ref = jnp.zeros((1,), dtype=jnp.int32)  # dummy
    else:
        primary_idx_ref = lhs_indices
        secondary_idx_ref = topk_indices

    # Expert TP builds the routing metadata ahead of the kernel in XLA and
    # leaves each tile's output writes in flight until its staging slot is reused.
    if expert_tp:
        group_start = _precompute_fused_rs_metadata(
            group_sizes,
            group_offset,
            group_start,
            lhs_indices,
            size_group=size_group,
            tile_m=tile_m,
            sublane=size_lhs_sublane,
            chunk_size=chunk_size,
            num_slots=max_num_gm_static,
        )
        if local_rows:
            # Expert TP orders rows shard-major, so this device's rows are the
            # contiguous block at local_start; rebase the row offsets onto it.
            local_start = group_start[0]
            primary_idx_ref = lax.dynamic_slice_in_dim(
                primary_idx_ref, local_start, kernel_size_m
            )
            if not pack_indices:
                secondary_idx_ref = lax.dynamic_slice_in_dim(
                    secondary_idx_ref, local_start, kernel_size_m
                )
            num_slots = max_num_gm_static
            row_offset = jnp.concatenate(
                [
                    jnp.stack([local_start, local_start, 0, 0]),
                    jnp.zeros((num_slots,), jnp.int32),
                    jnp.full((num_slots + 1,), local_start, jnp.int32),
                ]
            ).astype(jnp.int32)
            group_start = group_start - row_offset

    pair_meta_base = pair_meta_stride = 0
    num_pair_blocks = ep_size // 2 + 1
    if pair_exchange:
        group_start, pair_meta_base, pair_meta_stride = _pair_exchange_metadata(
            group_start,
            primary_idx_ref,
            group_offset=group_offset,
            size_group=size_group,
            pack_indices=pack_indices,
            top_k=top_k,
            chunk_size=chunk_size,
            ep_size=ep_size,
            num_slots=max_num_gm_static,
            num_blocks=num_pair_blocks,
        )
        out_buf_init = jax.ShapeDtypeStruct(
            (num_pair_blocks * chunk_size * (top_k // ep_size), n2_cols, num_lanes),
            payload_dtype,
        )

    pallas_inputs = (
        group_sizes,
        group_offset,
        group_start,
        primary_idx_ref,
        secondary_idx_ref,
        max_num_gm_arr,
        _w1_gs_gate,
        _w1_gs_up,
        _w2_gs,
        hidden_3d,
        w1,
        w2,
        w1_scale_input,
        w2_scale_input,
        w1_bias_input,
        w2_bias_input,
    )
    pallas_name = (
        f"gmm_v2_fused_rs-E_{size_group}-M_{size_m}"
        f"-K1_{size_k1}-N1_{size_n1}-K2_{size_k2}-N2_{size_n2}"
        f"-EP_{ep_size}-TK_{top_k}"
        f"{'-packed' if pack_indices else ''}"
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
        defer_writes=expert_tp or tpu_envs.SGLANG_TPU_FUSED_RS_EP_DEFER_WRITES.get(),
        # pair exchange writes every receive row exactly once
        skip_out_zero=(skip_out_zero and expert_tp and not fp8_direct_write) or pair_exchange,
        row_loop_unroll=row_loop_unroll,
        staging_slots=staging_slots,
        pair_exchange=pair_exchange,
        pair_meta_base=pair_meta_base,
        pair_meta_stride=pair_meta_stride,
    )
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states_3d
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1
        pl.BlockSpec(memory_space=pltpu.HBM),  # w2
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1_scale
        pl.BlockSpec(memory_space=pltpu.HBM),  # w2_scale
        pl.BlockSpec(memory_space=pltpu.HBM),  # w1_bias
        pl.BlockSpec(memory_space=pltpu.HBM),  # w2_bias
    ]

    if fp8_direct_write:
        fp8_scratch_shapes = scratch_shapes + [
            pltpu.SemaphoreType.REGULAR((staging_slots,)),
            pltpu.SemaphoreType.REGULAR((staging_slots,)),
            pltpu.SemaphoreType.REGULAR((1,)),
            pltpu.VMEM((staging_slots, tile_m, n2_cols, num_lanes), jnp.float8_e4m3fn),
            pltpu.VMEM((staging_slots, tile_m, 128), jnp.float32),
        ]
        payload, row_scales = pl.pallas_call(
            functools.partial(
                kernel_main_fused_rs_fp8,
                **{k: v for k, v in kernel_kwargs.items() if k != "pair_exchange"},
            ),
            out_shape=(out_buf_init, out_scale_init),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=9,
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

    kernel_fn = functools.partial(kernel_main_fused_rs, **kernel_kwargs)
    if pair_exchange:
        # Appended after the base scratch; passed by keyword since the fp8 refs
        # occupy the optional positional slots.
        scratch_shapes = scratch_shapes + [
            pltpu.VMEM((_PAIR_RING, tile_m, n2_cols, num_lanes), out_dtype),
            pltpu.VMEM((staging_slots, tile_m, n2_cols, num_lanes), out_dtype),
            pltpu.SemaphoreType.REGULAR((_PAIR_RING,)),
            pltpu.SemaphoreType.REGULAR((1,)),
        ]

        def kernel_fn(*refs, _base=kernel_fn):
            *base, ring, pair_sum, pair_recv, credit = refs
            _base(*base, pair_ring_ref=ring, pair_sum_ref=pair_sum,
                  pair_recv_sem_ref=pair_recv, pair_credit_ref=credit)

    result = pl.pallas_call(
        kernel_fn,
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

    if tiled_output and aligned_n2 == size_n2:
        return result
    return result.reshape(result.shape[0], aligned_n2)[:, :size_n2]


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
