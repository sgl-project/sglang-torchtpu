# SPDX-License-Identifier: Apache-2.0
"""Unified MoE Bridge + Fused GMM-RS + Combine/Unpool Pipeline & Reference for Muse Spark."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sglang_tpu.envs import envs
from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import (
    EXPERT,
    _dequant_into_half,
    fully_fused_moe_func_rs,
    get_mesh_shape_product,
    get_moe_expert_axis,
    moe_gmm_local_rs_nodedup,
)
from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    _select_fused_rs_block_sizes,
)
from sglang_tpu.moe.pallas.fused_rs_muse.v2_fused_rs_tuned_block_sizes import (
    FUSED_RS_TUNED_BLOCK_SIZES,
)
from sglang_tpu.moe.pallas.gmm_v2 import recover_quant_block_size
from sglang_tpu.muse_prefill_kernels.eplb import remap_zeroglue_virtual_ids

from .kernel_b1 import (
    pallas_postattn_res_prenorm_router_topk,
    pallas_postattn_res_prenorm_router_topk_fp8,
)
from .kernel_b2 import (
    dequant_both_halves_gmm_layout,
    dequant_half0_new_buffer,
    dequant_half_gmm_layout,
    dequant_into_half_packed_1way,
    dequant_single_gmm_layout,
    pallas_fused_b1_b2_wpool_quant,
    pallas_wpool_prenorm_int8_quant_fp8_nstream,
    pallas_wpool_prenorm_int8_quant_fp8_rowtile,
    pallas_wpool_prenorm_int8_quant_nstream,
    pallas_wpool_prenorm_int8_quant_rowtile,
)
from .kernel_d import (
    _tc_combine_bf16_4chunk,
    pallas_combine_unpool_residual_nextnorm_4slab,
    pallas_combine_unpool_residual_nextnorm_nstream,
)
from .routing_pallas import (
    pallas_compact_local_packed_keys,
    pallas_global_local_pos,
    pallas_local_inv_perm_and_counts,
)


def register_ep16_tuned_block_sizes() -> None:
    """Ensure EP16 (g-16) and EP8 (g-32) for 8K & 16K have tuned block sizes that fit SMEM & VMEM.

    At M=131072 (T=16384, topk=8) with block_scatter=True, tile_m=128 creates max_num_gm_static=1055
    which pads secondary_idx_ref [1055, 26] to [1056, 128] = 528 KiB in SMEM + lhs_indices 512 KiB = 1040 KiB
    (> 1024 KiB SMEM limit). Using tile_m=256 at M=131072 halves max_num_gm_static to 543 (272 KiB SMEM,
    total 784 KiB < 1024 KiB SMEM) while keeping 50 MiB VMEM (< 60.8 MiB limit).
    """
    tpu_root = FUSED_RS_TUNED_BLOCK_SIZES.setdefault("TPU7x", {})
    tuned_128 = (128, 4096, 2048, 4096, 2048, 2, 2)
    tuned_256 = (256, 4096, 2048, 4096, 2048, 2, 2)
    tuned_ep_tp2_128 = (128, 4096, 2048, 2048, 4096, 2, 2)
    tuned_ep_tp2_256 = (256, 4096, 2048, 2048, 4096, 2, 2)
    tuned_fp8_256 = (256, 4096, 1024, 4096, 1024, 2, 2)
    finer_ms = (
        8192,
        16384,
        24576,
        32768,
        40960,
        49152,
        57344,
        65536,
        131072,
        163840,
        196608,
        229376,
        262144,
        327680,
        393216,
        524288,
    )
    all_fp4_bs = ("16", "256", "512", "1024", "2048", "4096")
    for dtype_key in ("lhs_bfloat16_rhs_float4_e2m1fn", "lhs_bfloat16_rhs_int32"):
        tpu_dict = tpu_root.setdefault(dtype_key, {})
        for g in (16, 32):
            for bs_sfx in all_fp4_bs:
                joint_key = (
                    f"k-4096_n-8192_g-{g}_lhs_quant_rhs_bs-{bs_sfx}+"
                    f"k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-{bs_sfx}"
                )
                entry = tpu_dict.setdefault(joint_key, {})
                for m in finer_ms:
                    entry[f"{m}_silu"] = tuned_256 if m >= 131072 else tuned_128
        # Hybrid EP4xTP2 (g=64, I_half=2048) & EP8xTP2 (g=32, I_half=2048):
        # full W1 (4096x4096) + full W2 (2048x4096) fit in VMEM simultaneously!
        for g in (32, 64):
            for bs_sfx in all_fp4_bs:
                tp2_key = (
                    f"k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-{bs_sfx}+"
                    f"k-2048_n-4096_g-{g}_lhs_quant_rhs_bs-{bs_sfx}"
                )
                tp2_entry = tpu_dict.setdefault(tp2_key, {})
                for m in (
                    8192,
                    16384,
                    32768,
                    49152,
                    65536,
                    81920,
                    98304,
                    114688,
                    131072,
                    262144,
                    393216,
                    524288,
                ):
                    tp2_entry[f"{m}_silu"] = (
                        tuned_ep_tp2_256 if m >= 131072 else tuned_ep_tp2_128
                    )
        for g in range(33, 41):
            for bs_sfx in all_fp4_bs:
                src_key = (
                    f"k-4096_n-8192_g-32_lhs_quant_rhs_bs-{bs_sfx}+"
                    f"k-4096_n-4096_g-32_lhs_quant_rhs_bs-{bs_sfx}"
                )
                if src_key in tpu_dict:
                    dst_key = src_key.replace("_g-32_", f"_g-{g}_")
                    tpu_dict[dst_key] = dict(tpu_dict[src_key])
            joint_key_16 = (
                f"k-4096_n-8192_g-{g}_lhs_quant_rhs_bs-16+"
                f"k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-16"
            )
            entry_g = tpu_dict.setdefault(joint_key_16, {})
            for m in finer_ms:
                entry_g[f"{m}_silu"] = tuned_256 if m >= 131072 else tuned_128
    fp8_dict = tpu_root.setdefault("lhs_bfloat16_rhs_float8_e4m3fn", {})
    for g in (16, 32):
        fp8_key = (
            f"k-4096_n-8192_g-{g}_lhs_quant_rhs_bs-4096+"
            f"k-4096_n-4096_g-{g}_lhs_quant_rhs_bs-4096"
        )
        fp8_entry = fp8_dict.setdefault(fp8_key, {})
        for m in finer_ms:
            fp8_entry[f"{m}_silu"] = tuned_fp8_256 if m >= 131072 else tuned_128
    for g in range(33, 41):
        for bs_sfx in ("256", "512", "2048", "4096"):
            src_fp8_key = (
                f"k-4096_n-8192_g-32_lhs_quant_rhs_bs-{bs_sfx}+"
                f"k-4096_n-4096_g-32_lhs_quant_rhs_bs-{bs_sfx}"
            )
            dst_fp8_key = src_fp8_key.replace("_g-32_", f"_g-{g}_")
            if src_fp8_key in fp8_dict:
                copied_fp8 = dict(fp8_dict[src_fp8_key])
                for m in finer_ms:
                    if m >= 131072:
                        copied_fp8[f"{m}_silu"] = tuned_fp8_256
                fp8_dict[dst_fp8_key] = copied_fp8
            elif bs_sfx == "4096":
                fp8_entry_g = fp8_dict.setdefault(dst_fp8_key, {})
                for m in finer_ms:
                    fp8_entry_g[f"{m}_silu"] = (
                        tuned_fp8_256 if m >= 131072 else tuned_128
                    )


register_ep16_tuned_block_sizes()


def _convert_both_halves_fp8_gmm_layout(
    ag_i8_0: jax.Array,
    ag_sc_0: jax.Array | None,
    ag_i8_1: jax.Array | None,
    ag_sc_1: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Row-pipelined Pallas VMEM i8 -> (fp8[T_global, H//128, 128], fp8_scale[T_global]) converter.

    Tiles over `grid=(ep_sz, half_tok // row_tile)` (`row_tile <= 256`, `<= 8.7 MiB` double-buffered
    VMEM, `< 1,500` static bundles, `0` inner branches) so HBM DMA overlaps VPU completely.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    ep_sz, half_tok, raw_cols = ag_i8_0.shape
    n_halves = 1 if ag_i8_1 is None else 2
    packed_sc = ag_sc_0 is None
    lane_w = 128
    hidden_sz = (raw_cols - lane_w) if packed_sc else raw_cols
    row_tile = 256 if (half_tok >= 512 and half_tok % 256 == 0) else 128
    num_rt = half_tok // row_tile

    def _k_fp8(*refs):
        if packed_sc:
            if n_halves == 1:
                i8_0_ref, fp8_out_ref, sc_out_ref = refs
                pairs = ((i8_0_ref, None),)
            else:
                i8_0_ref, i8_1_ref, fp8_out_ref, sc_out_ref = refs
                pairs = ((i8_0_ref, None), (i8_1_ref, None))
            mask_r = None
        else:
            r = pl.program_id(0)
            mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, lane_w), 0) == r
            if n_halves == 1:
                i8_0_ref, sc_0_ref, fp8_out_ref, sc_out_ref = refs
                pairs = ((i8_0_ref, sc_0_ref),)
            else:
                i8_0_ref, sc_0_ref, i8_1_ref, sc_1_ref, fp8_out_ref, sc_out_ref = refs
                pairs = ((i8_0_ref, sc_0_ref), (i8_1_ref, sc_1_ref))
        for half_idx, (i8_ref, sc_ref) in enumerate(pairs):
            for t_blk in range(row_tile // lane_w):
                r0, r1 = t_blk * lane_w, (t_blk + 1) * lane_w
                if packed_sc:
                    ps = i8_ref[0, r0:r1, hidden_sz : hidden_sz + lane_w]
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
                        jnp.broadcast_to(sc_1x128, (lane_w, lane_w)), 0, 1
                    )[:, :1]
                lmax_bf16 = None
                for c_blk in range(hidden_sz // 256):
                    c0, c1 = c_blk * 256, (c_blk + 1) * 256
                    val = i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                    m128 = jnp.maximum(jnp.abs(val[:, :128]), jnp.abs(val[:, 128:]))
                    lmax_bf16 = (
                        m128 if lmax_bf16 is None else jnp.maximum(lmax_bf16, m128)
                    )
                amax_f32 = jnp.max(lmax_bf16, axis=1, keepdims=True).astype(
                    jnp.float32
                )
                inv_fp8 = jnp.where(
                    amax_f32 == 0.0,
                    jnp.zeros_like(amax_f32),
                    jnp.float32(448.0) / jnp.maximum(amax_f32, jnp.float32(1e-30)),
                ).astype(jnp.bfloat16)
                row_sc = jnp.where(
                    amax_f32 == 0.0,
                    jnp.zeros_like(amax_f32),
                    amax_f32 * jnp.float32(1.0 / 448.0),
                )
                sc_out_ref[0, half_idx, r0:r1, :] = jnp.broadcast_to(
                    row_sc, (lane_w, lane_w)
                )
                for c_blk in range(hidden_sz // 256):
                    c0, c1 = c_blk * 256, (c_blk + 1) * 256
                    val = i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                    q_fp8 = jnp.clip(val * inv_fp8, -448.0, 448.0).astype(
                        jnp.float8_e4m3fn
                    )
                    fp8_out_ref[0, half_idx, r0:r1, c0 // lane_w : c1 // lane_w, :] = (
                        q_fp8.reshape(lane_w, 2, lane_w)
                    )

    if packed_sc:
        in_specs = [
            pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0))
        ] * n_halves
        args: tuple[jax.Array, ...] = (
            (ag_i8_0,) if n_halves == 1 else (ag_i8_0, ag_i8_1)
        )
    else:
        assert ag_sc_0 is not None
        if n_halves == 1:
            in_specs = [
                pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0)),
                pl.BlockSpec((ep_sz, row_tile), lambda r, rt: (0, rt)),
            ]
            args = (ag_i8_0, ag_sc_0)
        else:
            assert ag_sc_1 is not None
            in_specs = [
                pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0)),
                pl.BlockSpec((ep_sz, row_tile), lambda r, rt: (0, rt)),
                pl.BlockSpec((1, row_tile, raw_cols), lambda r, rt: (r, rt, 0)),
                pl.BlockSpec((ep_sz, row_tile), lambda r, rt: (0, rt)),
            ]
            args = (ag_i8_0, ag_sc_0, ag_i8_1, ag_sc_1)

    fp8_5d, sc_4d = pl.pallas_call(
        _k_fp8,
        out_shape=(
            jax.ShapeDtypeStruct(
                (ep_sz, n_halves, half_tok, hidden_sz // lane_w, lane_w),
                jnp.float8_e4m3fn,
            ),
            jax.ShapeDtypeStruct((ep_sz, n_halves, half_tok, lane_w), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(ep_sz, num_rt),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec(
                    (1, n_halves, row_tile, hidden_sz // lane_w, lane_w),
                    lambda r, rt: (r, 0, rt, 0, 0),
                ),
                pl.BlockSpec(
                    (1, n_halves, row_tile, lane_w),
                    lambda r, rt: (r, 0, rt, 0),
                ),
            ],
            scratch_shapes=[],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name=f"pallas_deq_{'single' if n_halves == 1 else 'both'}_fp8_gmm_ep{ep_sz}_ht{half_tok}_rt{row_tile}",
    )(*args)
    return (
        fp8_5d.reshape(ep_sz * n_halves * half_tok, hidden_sz // lane_w, lane_w),
        sc_4d[:, :, :, 0].reshape(-1),
    )


def _packed_sort_low(high: jax.Array, high_bound: int) -> jax.Array:
    """Low part of ``sort(high * n + iota)``: argsort by ``high``, ties by index, as one s32 sort."""
    n = high.shape[0]
    assert high_bound * n <= 2**31, (high_bound, n)
    if (n & (n - 1)) == 0 and envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get():
        shift = n.bit_length() - 1
        packed = (high.astype(jnp.int32) << shift) | jnp.arange(n, dtype=jnp.int32)
        return jnp.sort(packed) & jnp.int32(n - 1)
    packed = high.astype(jnp.int32) * n + jnp.arange(n, dtype=jnp.int32)
    return jax.lax.rem(jnp.sort(packed), jnp.int32(n))


def _stable_argsort_expert_ids(
    expert_ids: jax.Array, num_experts: int = 256
) -> jax.Array:
    """``jnp.argsort(expert_ids, stable=True)`` for ids in ``[0, num_experts)``."""
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS.get()
        or envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get()
    ):
        return _packed_sort_low(expert_ids, num_experts)
    return jnp.argsort(expert_ids, stable=True)


def _inverse_permutation(perm: jax.Array) -> jax.Array:
    """``jnp.argsort(perm)`` for a permutation of ``range(len(perm))``."""
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS.get()
        or envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get()
    ) and perm.shape[0] ** 2 <= 2**31:
        return _packed_sort_low(perm, perm.shape[0])
    return jnp.argsort(perm)


def _bucket_ranks(ids_2d: jax.Array, num_buckets: int) -> jax.Array:
    """``[T, topk]`` ids -> ``[T, num_buckets]``: start of bucket b plus count of b in rows before t.

    VALU one-hot (8-sum) + MXU strictly-lower L_256 prefix; needs ``T % 256 == 0``.
    """
    t_len, topk = ids_2d.shape
    iota_b = jnp.arange(num_buckets, dtype=jnp.int32)[None, :]
    tok_oh = (ids_2d[:, 0:1] == iota_b).astype(jnp.bfloat16)
    for k in range(1, topk):
        tok_oh = tok_oh + (ids_2d[:, k : k + 1] == iota_b).astype(jnp.bfloat16)
    oh_3d = tok_oh.reshape(t_len // 256, 256, num_buckets)
    l_256 = jnp.tril(jnp.ones((256, 256), dtype=jnp.bfloat16), k=-1)
    intra = jnp.matmul(l_256, oh_3d, preferred_element_type=jnp.float32)
    blk_sums = jnp.sum(oh_3d.astype(jnp.float32), axis=1)
    blk_excl = jnp.cumsum(blk_sums, axis=0) - blk_sums
    counts = jnp.sum(blk_sums, axis=0)
    starts = jnp.cumsum(counts) - counts
    return (
        (intra + blk_excl[:, None, :] + starts[None, None, :])
        .reshape(t_len, num_buckets)
        .astype(jnp.int32)
    )


def _compute_local_inv_perm(local_topk_ids: jax.Array, num_experts: int) -> jax.Array:
    """Compute ``argsort(argsort(local_topk_ids.reshape(-1), stable=True)).reshape(t_local, topk)``."""
    t_local, topk = local_topk_ids.shape
    zsr_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING", None)
    use_zsr = bool(zsr_env is not None and zsr_env.get())
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get() or use_zsr
    ) and t_local % 256 == 0:
        sf_mode = envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE.get()
        if sf_mode == 0 and use_zsr:
            sf_mode = 3
        if sf_mode == 2:
            # Lane-local vst.iar mark + idle-MXU U_256 prefix sum + lane-local vld.iar lookup.
            ids_t = local_topk_ids.T
            tok_oh_t = (
                jnp.zeros((num_experts, t_local), dtype=jnp.bfloat16)
                .at[ids_t, jnp.arange(t_local, dtype=jnp.int32)[None, :]]
                .set(jnp.bfloat16(1.0))
            )
            n_blk = t_local // 256
            oh_3d = tok_oh_t.reshape(num_experts, n_blk, 256)
            u_256 = jnp.triu(jnp.ones((256, 256), dtype=jnp.bfloat16), k=1)
            intra = jnp.matmul(oh_3d, u_256, preferred_element_type=jnp.float32)
            blk_sums = jnp.sum(oh_3d.astype(jnp.float32), axis=2)
            blk_excl = jnp.cumsum(blk_sums, axis=1) - blk_sums
            counts = jnp.sum(blk_sums, axis=1)
            starts = jnp.cumsum(counts) - counts
            rank_t = (
                (intra + blk_excl[:, :, None] + starts[:, None, None])
                .reshape(num_experts, t_local)
                .astype(jnp.int32)
            )
            return jnp.take_along_axis(rank_t, ids_t, axis=0).T
        if sf_mode == 3:
            rank_2d = _bucket_ranks(local_topk_ids, num_experts)
            return jnp.take_along_axis(rank_2d.T, local_topk_ids.T, axis=0).T
    perm_c = _stable_argsort_expert_ids(local_topk_ids.reshape(-1), num_experts)
    return _inverse_permutation(perm_c).reshape(t_local, topk).astype(jnp.int32)


def _sort_local_packed_lhs_indices(
    *,
    global_topk_ids: jax.Array,
    topk_indices_flat: jax.Array,
    tok_ids_flat: jax.Array,
    go_val: jax.Array,
    my_id: jax.Array,
    num_tokens: int,
    chunk_size: int,
    ep_size: int,
    num_local_experts: int,
    topk: int,
    rotate_rs: bool,
    compact_mode: int,
) -> jax.Array:
    """Sort local-expert `(bucket, token)` keys with optional VMEM/chunked compaction."""
    n_flat = topk_indices_flat.shape[0]
    tok_bits = max(1, (num_tokens - 1).bit_length())
    tok_mask = jnp.int32((1 << tok_bits) - 1)
    sentinel_key = jnp.int32(int(num_local_experts) << tok_bits)

    def _build_full_packed() -> jax.Array:
        rel_flat = topk_indices_flat.astype(jnp.int32) - go_val
        bucket_flat = jnp.where(
            (rel_flat >= 0) & (rel_flat < num_local_experts),
            rel_flat,
            jnp.int32(num_local_experts),
        )
        if rotate_rs:
            chip_r = jnp.mod(
                tok_ids_flat // chunk_size + ep_size - 1 - my_id, ep_size
            )
            sort_tok = chip_r * chunk_size + (tok_ids_flat % chunk_size)
        else:
            sort_tok = tok_ids_flat
        return (bucket_flat << tok_bits) | sort_tok

    def _decode_sorted(sorted_keys: jax.Array) -> jax.Array:
        if rotate_rs:
            is_valid = sorted_keys < sentinel_key
            rot_tok = sorted_keys & tok_mask
            orig_c = jnp.mod(rot_tok // chunk_size + my_id + 1, ep_size)
            raw_tok = jnp.where(
                is_valid,
                orig_c * chunk_size + (rot_tok % chunk_size),
                jnp.int32(0),
            )
        else:
            raw_tok = sorted_keys & tok_mask
        if sorted_keys.shape[0] < n_flat:
            return jnp.pad(raw_tok, (0, n_flat - sorted_keys.shape[0]))
        return raw_tok

    if (
        compact_mode == 1
        and topk == 8
        and n_flat >= 65536
        and n_flat % (128 * 128) == 0
        and 0 < num_local_experts <= 120
    ):
        o32, o_mid, tier = pallas_compact_local_packed_keys(
            global_topk_ids,
            go_val,
            my_id,
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            rotate_rs=rotate_rs,
        )
        return jax.lax.switch(
            tier,
            (
                lambda _: _decode_sorted(jnp.sort(o32)),
                lambda _: _decode_sorted(jnp.sort(o_mid)),
                lambda _: _decode_sorted(jnp.sort(_build_full_packed())),
            ),
            operand=None,
        )

    if compact_mode == 2 and n_flat >= 65536 and n_flat % 4096 == 0:
        packed_full = _build_full_packed()
        n_chunks = n_flat // 4096
        cap_c = min(2048, 32768 // n_chunks)
        s_chunk = jnp.sort(packed_full.reshape(n_chunks, 4096), axis=1)
        no_ovf = jnp.min(s_chunk[:, cap_c]) >= sentinel_key

        def _fast_32k(_):
            cand = s_chunk[:, :cap_c].reshape(-1)
            if cand.shape[0] < 32768:
                cand = jnp.pad(
                    cand,
                    (0, 32768 - cand.shape[0]),
                    constant_values=sentinel_key,
                )
            return _decode_sorted(jnp.sort(cand))

        return jax.lax.cond(
            no_ovf,
            _fast_32k,
            lambda _: _decode_sorted(jnp.sort(packed_full)),
            operand=None,
        )

    return _decode_sorted(jnp.sort(_build_full_packed()))


def compute_block_scatter_routing(
    global_topk_ids: jax.Array,
    go_val: jax.Array,
    my_id: jax.Array,
    *,
    ep_size: int,
    num_local_experts: int,
    topk: int = 8,
    stable_sort: bool = False,
    has_pad_sentinel: bool = False,
    precomputed_counts: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute block-scatter routing metadata (lhs_indices, group_sizes, group_start_arg, local_t_recv)."""
    num_experts = ep_size * num_local_experts
    num_tokens = global_topk_ids.shape[0]
    chunk_size = num_tokens // ep_size
    topk_indices_flat = global_topk_ids.reshape(-1)
    pallas_sf_mode = int(
        envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get() or 0
    )
    zsr_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING", None)
    use_zero_sort_rt = bool(zsr_env is not None and zsr_env.get())

    def _mxu_bucket_scatter_lhs_indices(bucket_2d: jax.Array, tok_ids_1d: jax.Array) -> jax.Array:
        n_flat = tok_ids_1d.shape[0]
        rank_2d = _bucket_ranks(bucket_2d, num_local_experts + 1)
        k_iota = jnp.arange(topk, dtype=jnp.int32)
        lt_mask = k_iota[None, :, None] > k_iota[None, None, :]
        intra_row = jnp.sum(
            ((bucket_2d[:, :, None] == bucket_2d[:, None, :]) & lt_mask).astype(
                jnp.int32
            ),
            axis=2,
        )
        pos = jnp.where(
            bucket_2d < num_local_experts,
            jnp.take_along_axis(rank_2d, bucket_2d, axis=1) + intra_row,
            jnp.int32(n_flat),
        )
        return (
            jnp.zeros((n_flat,), dtype=jnp.int32)
            .at[pos.reshape(-1)]
            .set(tok_ids_1d, mode="drop", unique_indices=True)
        )

    rotate_rs = (
        int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS.get() or 0) >= 2
        and num_tokens > (4096 // max(1, topk))
    )
    if (
        precomputed_counts is not None
        and (envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get() or use_zero_sort_rt)
        and (topk & (topk - 1)) == 0
        and num_tokens >= 256
    ):
        topk_bits = topk.bit_length() - 1
        tok_ids_flat = (
            jnp.arange(topk_indices_flat.shape[0], dtype=jnp.int32) >> topk_bits
        )
        sf_scatter = envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER.get()
        if pallas_sf_mode >= 2 and num_tokens % 128 == 0 and num_local_experts < 128:
            n_flat = topk_indices_flat.shape[0]
            pos_2d = pallas_global_local_pos(
                global_topk_ids,
                go_val,
                num_local_experts=num_local_experts,
                ep_size=ep_size,
                rotate_rs=rotate_rs,
            )
            lhs_indices = (
                jnp.zeros((n_flat,), dtype=jnp.int32)
                .at[pos_2d.reshape(-1)]
                .set(tok_ids_flat, mode="drop", unique_indices=True)
            )
        elif (sf_scatter == 2 or use_zero_sort_rt) and num_tokens % 256 == 0 and not rotate_rs:
            rel_flat = topk_indices_flat.astype(jnp.int32) - go_val
            bucket_2d = jnp.where(
                (rel_flat >= 0) & (rel_flat < num_local_experts),
                rel_flat,
                jnp.int32(num_local_experts),
            ).reshape(num_tokens, topk)
            lhs_indices = _mxu_bucket_scatter_lhs_indices(bucket_2d, tok_ids_flat)
        else:
            compact_mode = int(
                envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get() or 0
            )
            lhs_indices = _sort_local_packed_lhs_indices(
                global_topk_ids=global_topk_ids,
                topk_indices_flat=topk_indices_flat,
                tok_ids_flat=tok_ids_flat,
                go_val=go_val,
                my_id=my_id,
                num_tokens=num_tokens,
                chunk_size=chunk_size,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                topk=topk,
                rotate_rs=rotate_rs,
                compact_mode=compact_mode,
            )
        return (lhs_indices, *precomputed_counts)
    if (
        (envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get() or use_zero_sort_rt)
        and (topk & (topk - 1)) == 0
        and num_tokens >= 256
    ):
        # Map non-local experts (including sentinel == num_experts) to bin `num_local_experts`
        # and local experts to 0..num_local_experts-1 so local_start = 0 and all local experts
        # sit contiguously at lhs_indices[0:local_total].
        tok_bits = max(1, (num_tokens - 1).bit_length())
        tok_mask = jnp.int32((1 << tok_bits) - 1)
        topk_bits = topk.bit_length() - 1
        rel_flat = topk_indices_flat.astype(jnp.int32) - go_val
        bucket_flat = jnp.where(
            (rel_flat >= 0) & (rel_flat < num_local_experts),
            rel_flat,
            jnp.int32(num_local_experts),
        )
        tok_ids_flat = (
            jnp.arange(topk_indices_flat.shape[0], dtype=jnp.int32) >> topk_bits
        )
        packed = (bucket_flat << tok_bits) | tok_ids_flat
        topk_by_chip = global_topk_ids.reshape(ep_size, -1)
        chip_before_local = jnp.sum((topk_by_chip < go_val).astype(jnp.int32), axis=1)
        local_start = jnp.int32(0)
        local_expert_ids = go_val + jnp.arange(num_local_experts, dtype=jnp.int32)
        local_chip_counts = jnp.sum(
            (topk_by_chip[:, :, None] == local_expert_ids[None, None, :]).astype(
                jnp.int32
            ),
            axis=1,
        )
        sf_scatter = envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER.get()
        if (
            (sf_scatter == 2 or use_zero_sort_rt)
            and num_tokens % 256 == 0
            and not rotate_rs
        ):
            # Local entries land at [0, local_total) in (bucket, token, slot) order; the rest stay 0,
            # as the fast merge leaves them.
            bucket_2d = bucket_flat.reshape(num_tokens, topk)
            lhs_indices = _mxu_bucket_scatter_lhs_indices(bucket_2d, tok_ids_flat)
        elif (
            rotate_rs
            or int(envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get() or 0) >= 1
        ):
            lhs_indices = _sort_local_packed_lhs_indices(
                global_topk_ids=global_topk_ids,
                topk_indices_flat=topk_indices_flat,
                tok_ids_flat=tok_ids_flat,
                go_val=go_val,
                my_id=my_id,
                num_tokens=num_tokens,
                chunk_size=chunk_size,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                topk=topk,
                rotate_rs=rotate_rs,
                compact_mode=int(
                    envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get() or 0
                ),
            )
        elif sf_scatter == 1:
            # One flat sort measured cheaper than per-rank sort + merge (qp xprof: 30 vs 35 us).
            lhs_indices = jnp.sort(packed) & tok_mask
        else:
            # 2-stage pre-compacted sort: sort each rank's chunk along axis=1, then merge only the
            # leading `cap = (chunk_size * topk) // 4` entries (2x expected local count) when safe.
            s_chunk = jnp.sort(packed.reshape(ep_size, -1), axis=1)
            cap = max(256, (chunk_size * topk) // 4)
            max_c = jnp.max(jnp.sum(local_chip_counts, axis=1))

            def _fast_merge(_):
                merged = jnp.sort(s_chunk[:, :cap].reshape(-1)) & tok_mask
                return jnp.pad(merged, (0, packed.shape[0] - merged.shape[0]))

            def _full_merge(_):
                return jnp.sort(packed) & tok_mask

            lhs_indices = jax.lax.cond(
                max_c <= cap, _fast_merge, _full_merge, operand=None
            )
    else:
        use_stable = (
            stable_sort
            or has_pad_sentinel
            or envs.SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META.get()
        )
        if rotate_rs:
            _flat_iota = jnp.arange(topk_indices_flat.shape[0], dtype=jnp.int32)
            _tok_f = _flat_iota // topk
            _chip_rot = jnp.mod(_tok_f // chunk_size + ep_size - 1 - my_id, ep_size)
            _rot_key = topk_indices_flat.astype(jnp.int32) * ep_size + _chip_rot
            topk_argsort_indices = jnp.argsort(_rot_key, stable=True)
        elif envs.SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS.get():
            topk_argsort_indices = _packed_sort_low(
                topk_indices_flat, num_experts + (1 if has_pad_sentinel else 0)
            )
        elif use_stable:
            topk_argsort_indices = jnp.argsort(topk_indices_flat, stable=True)
        else:
            topk_argsort_indices = jnp.argsort(topk_indices_flat)
        lhs_indices = topk_argsort_indices // topk
        topk_by_chip = global_topk_ids.reshape(ep_size, -1)
        chip_before_local = jnp.sum((topk_by_chip < go_val).astype(jnp.int32), axis=1)
        local_start = jnp.sum(chip_before_local)
        local_expert_ids = go_val + jnp.arange(num_local_experts, dtype=jnp.int32)
        local_chip_counts = jnp.sum(
            (topk_by_chip[:, :, None] == local_expert_ids[None, None, :]).astype(
                jnp.int32
            ),
            axis=1,
        )
    local_group_sizes = jnp.sum(local_chip_counts, axis=0)
    group_sizes = jax.lax.dynamic_update_slice(
        jnp.zeros((num_experts,), dtype=jnp.int32), local_group_sizes, (go_val,)
    )
    if has_pad_sentinel:
        my_chip_ids = jax.lax.dynamic_index_in_dim(
            topk_by_chip, my_id, axis=0, keepdims=False
        )
        active_slots_my_id = jnp.sum((my_chip_ids < num_experts).astype(jnp.int32))
        my_local_hits = jnp.sum(
            jax.lax.dynamic_index_in_dim(
                local_chip_counts, my_id, axis=0, keepdims=False
            )
        )
        t_recv_val = active_slots_my_id - my_local_hits
    else:
        t_recv_val = jnp.int32(chunk_size * topk) - jnp.sum(local_chip_counts[my_id])
    group_start_arg = jnp.concatenate(
        [
            jnp.stack([local_start, t_recv_val]),
            chip_before_local,
            local_chip_counts.T.reshape(-1),
        ]
    ).astype(jnp.int32)
    return lhs_indices, group_sizes, group_start_arg, t_recv_val


def reference_moe_bridge_and_ffn(
    attn_out_sharded: jax.Array,
    stream_sharded: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    post_expert_norm_w: jax.Array,
    group_offset: jax.Array,
    wunpool_fp8: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    mesh: Mesh,
    topk: int = 8,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure-JAX Muse Spark reference (exact FP32/BF16 rounding match).

    Returns:
        new_stream_sharded: fp32[T, 8192] (token-sharded)
        next_attn_in_sharded: bf16[T, 8192] (token-sharded)
        router_in_sharded: bf16[T, 8192] (token-sharded)
        stream_mid_sharded: fp32[T, 8192] (token-sharded)
        local_topk_ids: int32[T, topk] (token-sharded)
        local_topk_weights: bf16[T, topk] (token-sharded)
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)

    def _ref_body(
        attn_s,
        stream_s,
        a_attn,
        b_attn,
        pffn_w,
        w_r,
        bias,
        wp_fp8,
        wp_sc,
        pe_w,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        pnw_l,
        go_l,
        wu_fp8,
        wu_sc,
        post_ffn_w,
        a_ffn,
        b_ffn,
        next_in_w,
    ):
        # 1. post_attention_layernorm (weightless, eps=1e-8) + gated residual
        # Use optimization_barrier before .astype(bf16) so XLA Bfloat16Propagation does not
        # demote the FP32 multiply (a_f32 * s_a) to bf16 (attn_s * s_a.astype(bf16))
        a_f32 = attn_s.astype(jnp.float32)
        s_a = jax.lax.rsqrt(
            jnp.mean(a_f32 * a_f32, axis=-1, keepdims=True) + post_norm_eps
        )
        branch_attn = (
            jax.lax.optimization_barrier(a_f32 * s_a)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        stream_mid = a_attn[None, :] * stream_s + b_attn[None, :] * branch_attn

        # 2. pre_feedforward_layernorm (gain_center=1.0, eps=1e-5)
        s_m = jax.lax.rsqrt(
            jnp.mean(stream_mid * stream_mid, axis=-1, keepdims=True) + rms_eps
        )
        router_in_f32 = stream_mid * s_m * (pffn_w.astype(jnp.float32) + 1.0)[None, :]
        router_in_bf16 = router_in_f32.astype(jnp.bfloat16)

        # 3. FP32 Router gate + sigmoid + biased top-8 + unbiased renormalize
        logits = jnp.dot(
            router_in_f32,
            w_r.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        scores = jax.nn.sigmoid(logits * router_multiplier)
        _, local_topk_ids = jax.lax.top_k(
            scores + bias.astype(jnp.float32)[None, :], topk
        )
        local_topk_ids = local_topk_ids.astype(jnp.int32)
        selected_unbiased = jnp.take_along_axis(scores, local_topk_ids, axis=-1)
        local_topk_weights = (
            selected_unbiased
            / (jnp.sum(selected_unbiased, axis=-1, keepdims=True) + route_eps)
            * route_scale
        ).astype(jnp.bfloat16)

        # 4. Option 3A wpool + pre_expert_norm + All-Gather
        u_pool = (
            (
                jnp.dot(router_in_bf16, wp_fp8, preferred_element_type=jnp.float32)
                * wp_sc.astype(jnp.float32)[None, :]
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        s_pe = jax.lax.rsqrt(
            jnp.mean(u_pool * u_pool, axis=-1, keepdims=True) + rms_eps
        )
        x_moe_local = (
            u_pool * s_pe * (pe_w.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)

        hidden_states = jax.lax.all_gather(
            x_moe_local, axis_name=expert_axis, axis=0, tiled=True
        )
        global_topk_ids = jax.lax.all_gather(
            local_topk_ids, axis_name=expert_axis, axis=0, tiled=True
        )

        # 5. Expert MoE (gmm_v2_fused_rs_nodedup + _tc_combine_bf16)
        moe_out_shard = moe_gmm_local_rs_nodedup(
            hidden_states,
            w1_l,
            w1s_l,
            None,
            w2_l,
            w2s_l,
            None,
            None,
            None,
            go_l,
            local_topk_weights,
            global_topk_ids,
            pnw_l,
            activation="silu",
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=expert_axis,
            has_post_norm=True,
            sp_enabled=True,
            fp8_direct_write=False,
            expert_tp=False,
            fp4_w4a16=False,
            block_scatter=True,
        )

        # 6. Option 3A wunpool + post_feedforward_layernorm (scale_input=True) + residual + next norm
        y_unpool = (
            (
                jnp.dot(moe_out_shard, wu_fp8, preferred_element_type=jnp.float32)
                * wu_sc.astype(jnp.float32)[None, :]
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        fw = y_unpool * (post_ffn_w.astype(jnp.float32) + 1.0)[None, :]
        s_ffn = jax.lax.rsqrt(jnp.mean(fw * fw, axis=-1, keepdims=True) + post_norm_eps)
        branch_ffn = (fw * s_ffn).astype(jnp.bfloat16).astype(jnp.float32)
        new_stream = a_ffn[None, :] * stream_mid + b_ffn[None, :] * branch_ffn

        s_next = jax.lax.rsqrt(
            jnp.mean(new_stream * new_stream, axis=-1, keepdims=True) + rms_eps
        )
        next_attn_in = (
            new_stream * s_next * (next_in_w.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)

        return (
            new_stream,
            next_attn_in,
            router_in_bf16,
            stream_mid,
            local_topk_ids,
            local_topk_weights,
        )

    return jax.shard_map(
        _ref_body,
        mesh=mesh,
        in_specs=(
            P(expert_axis, None),  # attn_out_sharded
            P(expert_axis, None),  # stream_sharded
            P(),  # alpha_attn
            P(),  # beta_attn
            P(),  # pre_ffn_norm_w
            P(),  # wrouter
            P(),  # e_score_bias
            P(),  # wpool_fp8
            P(),  # wpool_scale
            P(),  # pre_expert_norm_w
            P(expert_axis),  # w1
            P(expert_axis),  # w2
            P(expert_axis),  # w1_scale
            P(expert_axis),  # w2_scale
            P(),  # post_expert_norm_w
            P(expert_axis),  # group_offset
            P(),  # wunpool_fp8
            P(),  # wunpool_scale
            P(),  # post_ffn_norm_w
            P(),  # alpha_ffn
            P(),  # beta_ffn
            P(),  # next_input_norm_w
        ),
        out_specs=(
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
        ),
        check_vma=False,
    )(
        attn_out_sharded,
        stream_sharded,
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        wrouter,
        e_score_bias,
        wpool_fp8,
        wpool_scale,
        pre_expert_norm_w,
        w1,
        w2,
        w1_scale,
        w2_scale,
        post_expert_norm_w,
        group_offset,
        wunpool_fp8,
        wunpool_scale,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )


def baseline_unfused_sp_moe_layer(
    attn_out_sharded: jax.Array,
    stream_sharded: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_col_sharded: jax.Array,
    wpool_scale_col_sharded: jax.Array,
    pre_expert_norm_w: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    post_expert_norm_w: jax.Array,
    group_offset: jax.Array,
    wunpool_row_sharded: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    mesh: Mesh,
    topk: int = 8,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Unfused SP Baseline before Kernel B1 / B2 / Option 3A / Kernel D fusions.

    Runs XLA norms + wrouter, AG(router_in) + column-sharded wpool -> fully_fused_moe_func_rs
    (including Stage 1 psum/pmax + Stage 2 AG + GMM-RS + Stage 3 Output AG) -> row-sharded
    wunpool + ReduceScatter + XLA post-FFN norm & residual.
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)

    def _pre_moe(attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, wp_col, wp_sc_col):
        a_f32 = attn_s.astype(jnp.float32)
        s_a = jax.lax.rsqrt(
            jnp.mean(a_f32 * a_f32, axis=-1, keepdims=True) + post_norm_eps
        )
        branch_attn = (a_f32 * s_a).astype(jnp.bfloat16).astype(jnp.float32)
        stream_mid = a_attn[None, :] * stream_s + b_attn[None, :] * branch_attn

        s_m = jax.lax.rsqrt(
            jnp.mean(stream_mid * stream_mid, axis=-1, keepdims=True) + rms_eps
        )
        router_in_f32 = stream_mid * s_m * (pffn_w.astype(jnp.float32) + 1.0)[None, :]
        router_in_bf16 = router_in_f32.astype(jnp.bfloat16)
        router_logits = jnp.dot(
            router_in_bf16,
            w_r.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        router_in_full = jax.lax.all_gather(
            router_in_bf16, axis_name=expert_axis, axis=0, tiled=True
        )
        proj_col = (
            jnp.dot(router_in_full, wp_col, preferred_element_type=jnp.float32)
            * wp_sc_col.astype(jnp.float32)[None, :]
        ).astype(jnp.bfloat16)
        return stream_mid, proj_col, router_logits

    stream_mid_sharded, pre_proj_col_sharded, router_logits_tok_sharded = jax.shard_map(
        _pre_moe,
        mesh=mesh,
        in_specs=(
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
            P(),
            P(None, expert_axis),
            P(expert_axis),
        ),
        out_specs=(
            P(expert_axis, None),
            P(None, expert_axis),
            P(expert_axis, None),
        ),
        check_vma=False,
    )(
        attn_out_sharded,
        stream_sharded,
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        wrouter,
        wpool_col_sharded,
        wpool_scale_col_sharded,
    )

    moe_out_replicated = fully_fused_moe_func_rs(
        pre_proj_col_sharded,
        pre_expert_norm_w,
        router_logits_tok_sharded,
        e_score_bias,
        w1,
        w2,
        w1_scale,
        w2_scale,
        post_expert_norm_w,
        group_offset,
        mesh=mesh,
        topk=topk,
        activation="silu",
        router_multiplier=router_multiplier,
        route_scale=route_scale,
        route_eps=route_eps,
        rms_eps=rms_eps,
    )

    def _post_moe(
        moe_rep, stream_mid, wu_row, wu_sc, post_ffn_w, a_ffn, b_ffn, next_in_w
    ):
        ep_sz = jax.lax.psum(jnp.int32(1), axis_name=expert_axis)
        my_id = jax.lax.axis_index(expert_axis)
        he_chunk = wu_row.shape[0]
        moe_slice = jax.lax.dynamic_slice_in_dim(
            moe_rep, my_id * he_chunk, he_chunk, axis=1
        )
        partial_y = jnp.dot(moe_slice, wu_row, preferred_element_type=jnp.float32)
        y_shard = jax.lax.psum_scatter(
            partial_y, axis_name=expert_axis, scatter_dimension=0, tiled=True
        )
        y_unpool = (
            (y_shard * wu_sc.astype(jnp.float32)[None, :])
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        fw = y_unpool * (post_ffn_w.astype(jnp.float32) + 1.0)[None, :]
        s_ffn = jax.lax.rsqrt(jnp.mean(fw * fw, axis=-1, keepdims=True) + post_norm_eps)
        branch_ffn = (fw * s_ffn).astype(jnp.bfloat16).astype(jnp.float32)
        new_stream = a_ffn[None, :] * stream_mid + b_ffn[None, :] * branch_ffn
        s_next = jax.lax.rsqrt(
            jnp.mean(new_stream * new_stream, axis=-1, keepdims=True) + rms_eps
        )
        next_attn_in = (
            new_stream * s_next * (next_in_w.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)
        del ep_sz
        return new_stream, next_attn_in

    return jax.shard_map(
        _post_moe,
        mesh=mesh,
        in_specs=(
            P(),
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(expert_axis, None), P(expert_axis, None)),
        check_vma=False,
    )(
        moe_out_replicated,
        stream_mid_sharded,
        wunpool_row_sharded,
        wunpool_scale,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )


def _unpack_ag_i8_and_scale(
    ag_i8: jax.Array,
    sc_1d: jax.Array | None,
    *,
    expert_axis: str,
    b2_i8pack: bool,
    he_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Return `(ag_i8_data, ag_sc)` for 1-way B2 gather whether scales are separate or packed."""
    del b2_i8pack
    if sc_1d is not None:
        ag_sc = jax.lax.all_gather(sc_1d[:, None], axis_name=expert_axis, axis=0, tiled=True)
        return ag_i8, ag_sc
    ps = ag_i8[:, he_dim : he_dim + 128]
    lo_u16 = ps[:, 0:1].astype(jnp.int32) & jnp.int32(0xFF)
    hi_u16 = (ps[:, 64:65].astype(jnp.int32) & jnp.int32(0xFF)) << 8
    ag_sc = jax.lax.bitcast_convert_type(
        (lo_u16 | hi_u16).astype(jnp.uint16), jnp.bfloat16
    )
    return ag_i8[:, :he_dim], ag_sc


def fused_bridge_b1_b2_ag_routing(
    attn_out_sharded: jax.Array,
    stream_sharded: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    group_offset: jax.Array,
    *,
    mesh: Mesh,
    num_local_experts: int,
    topk: int = 8,
    router_mode: str = "bf16_hilo",
    b1_two_half: bool = False,
    b2_layout: str = "nstream",
    b2_fp8_lhs: bool = False,
    b2_two_half: bool = True,
    b2_pack_sc: bool = False,
    b2_i8pack: bool = False,
    b2_deq0_first: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Standalone benchmark for Kernel B1 + Overlapped Routing & Kernel B2 + 2-Half Input AG."""
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)

    def _bridge_body(
        attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, bias, wp_fp8, wp_sc, pe_w, go_l
    ):
        t_local = attn_s.shape[0]
        he_dim = wp_fp8.shape[1]
        go_val = go_l[0]
        my_id = go_val // num_local_experts
        rank_mask_2d = jnp.arange(ep_size, dtype=jnp.int32)[:, None] == my_id

        def _call_b1(a_slice, s_slice):
            if b2_fp8_lhs:
                r_fp8, r_sc, smid, tk_u8, tk_w = (
                    pallas_postattn_res_prenorm_router_topk_fp8(
                        a_slice,
                        s_slice,
                        a_attn,
                        b_attn,
                        pffn_w,
                        w_r,
                        bias,
                        router_mode=router_mode,
                        topk=topk,
                    )
                )
                return None, r_fp8, r_sc, smid, tk_u8, tk_w
            r_bf16, smid, tk_u8, tk_w = pallas_postattn_res_prenorm_router_topk(
                a_slice,
                s_slice,
                a_attn,
                b_attn,
                pffn_w,
                w_r,
                bias,
                router_mode=router_mode,
                topk=topk,
            )
            return r_bf16, None, None, smid, tk_u8, tk_w

        def _call_b2(r_bf16_in, r_fp8_in, sc_in, half_idx: int | None = None):
            if b2_fp8_lhs:
                if b2_layout == "rowtile":
                    if half_idx is not None:
                        ht = r_fp8_in.shape[0] // 2
                        sl = slice(0, ht) if half_idx == 0 else slice(ht, 2 * ht)
                        r_fp8_in, sc_in = r_fp8_in[sl], sc_in[sl]
                    return pallas_wpool_prenorm_int8_quant_fp8_rowtile(
                        r_fp8_in, sc_in, wp_fp8, wp_sc, pe_w
                    )
                return pallas_wpool_prenorm_int8_quant_fp8_nstream(
                    r_fp8_in,
                    sc_in,
                    wp_fp8,
                    wp_sc,
                    pe_w,
                    pack_sc_in_i8=b2_i8pack,
                    half_idx=half_idx,
                )
            if b2_layout == "rowtile":
                if half_idx is not None:
                    ht = r_bf16_in.shape[0] // 2
                    sl = slice(0, ht) if half_idx == 0 else slice(ht, 2 * ht)
                    r_bf16_in = r_bf16_in[sl]
                return pallas_wpool_prenorm_int8_quant_rowtile(
                    r_bf16_in, wp_fp8, wp_sc, pe_w
                )
            return pallas_wpool_prenorm_int8_quant_nstream(
                r_bf16_in, wp_fp8, wp_sc, pe_w, half_idx=half_idx
            )

        if b2_two_half:
            half_tok = t_local // 2
            if b1_two_half:
                r_bf16_0, r_fp8_0, r_sc_0, smid_0, tk_u8_0, tk_w_0 = _call_b1(
                    attn_s[:half_tok], stream_s[:half_tok]
                )
                i8_0, sc0_1d = _call_b2(r_bf16_0, r_fp8_0, r_sc_0, half_idx=None)
                if b2_i8pack:
                    i8_0, smid_0, tk_u8_0, tk_w_0 = jax.lax.optimization_barrier(
                        (i8_0, smid_0, tk_u8_0, tk_w_0)
                    )
                else:
                    i8_0, sc0_1d, smid_0, tk_u8_0, tk_w_0 = (
                        jax.lax.optimization_barrier(
                            (i8_0, sc0_1d, smid_0, tk_u8_0, tk_w_0)
                        )
                    )
                ag_i8_0 = jax.lax.all_gather(
                    i8_0[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )
                if not b2_pack_sc and not b2_i8pack:
                    ag_sc_0 = jax.lax.psum(
                        jnp.where(rank_mask_2d, sc0_1d[None, :], jnp.bfloat16(0.0)),
                        axis_name=expert_axis,
                    )
                a_1, s_1, _ = jax.lax.optimization_barrier(
                    (attn_s[half_tok:], stream_s[half_tok:], i8_0)
                )
                r_bf16_1, r_fp8_1, r_sc_1, smid_1, tk_u8_1, tk_w_1 = _call_b1(a_1, s_1)
                stream_shard_mid = jnp.concatenate([smid_0, smid_1], axis=0)
                local_topk_ids_u8 = jnp.concatenate([tk_u8_0, tk_u8_1], axis=0)
                local_topk_weights = jnp.concatenate([tk_w_0, tk_w_1], axis=0)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                )
                if b2_fp8_lhs:
                    r_fp8_1, r_sc_1, _ = jax.lax.optimization_barrier(
                        (r_fp8_1, r_sc_1, global_topk_u8)
                    )
                    i8_1, sc1_1d = _call_b2(None, r_fp8_1, r_sc_1, half_idx=None)
                else:
                    r_bf16_1, _ = jax.lax.optimization_barrier(
                        (r_bf16_1, global_topk_u8)
                    )
                    i8_1, sc1_1d = _call_b2(r_bf16_1, None, None, half_idx=None)
            else:
                (
                    router_in_bf16,
                    router_in_fp8,
                    router_in_sc_pad,
                    stream_shard_mid,
                    local_topk_ids_u8,
                    local_topk_weights,
                ) = _call_b1(attn_s, stream_s)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                )
                if b2_fp8_lhs:
                    r_fp8_0, r_sc_0, _ = jax.lax.optimization_barrier(
                        (router_in_fp8, router_in_sc_pad, global_topk_u8)
                    )
                    i8_0, sc0_1d = _call_b2(None, r_fp8_0, r_sc_0, half_idx=0)
                else:
                    r_bf16_0, _ = jax.lax.optimization_barrier(
                        (router_in_bf16, global_topk_u8)
                    )
                    i8_0, sc0_1d = _call_b2(r_bf16_0, None, None, half_idx=0)

                if b2_i8pack:
                    i8_0 = jax.lax.optimization_barrier(i8_0)
                else:
                    i8_0, sc0_1d = jax.lax.optimization_barrier((i8_0, sc0_1d))
                ag_i8_0 = jax.lax.all_gather(
                    i8_0[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )
                if not b2_pack_sc and not b2_i8pack:
                    ag_sc_0 = jax.lax.psum(
                        jnp.where(rank_mask_2d, sc0_1d[None, :], jnp.bfloat16(0.0)),
                        axis_name=expert_axis,
                    )

                if b2_fp8_lhs:
                    if b2_i8pack:
                        r_fp8_1, r_sc_1, _ = jax.lax.optimization_barrier(
                            (router_in_fp8, router_in_sc_pad, i8_0)
                        )
                        i8_1, _ = _call_b2(None, r_fp8_1, r_sc_1, half_idx=1)
                        sc1_1d = None
                    else:
                        r_fp8_1, r_sc_1, _, _ = jax.lax.optimization_barrier(
                            (router_in_fp8, router_in_sc_pad, i8_0, sc0_1d)
                        )
                        i8_1, sc1_1d = _call_b2(None, r_fp8_1, r_sc_1, half_idx=1)
                else:
                    r_bf16_1, _, _ = jax.lax.optimization_barrier(
                        (router_in_bf16, i8_0, sc0_1d)
                    )
                    i8_1, sc1_1d = _call_b2(r_bf16_1, None, None, half_idx=1)

            if b2_i8pack:
                i8_1 = jax.lax.optimization_barrier(i8_1)
            else:
                i8_1, sc1_1d = jax.lax.optimization_barrier((i8_1, sc1_1d))

            ag_i8_1 = jax.lax.all_gather(
                i8_1[None, ...], axis_name=expert_axis, axis=0, tiled=True
            )
            if not b2_i8pack:
                if b2_pack_sc:
                    sc01_1d = jnp.concatenate([sc0_1d, sc1_1d], axis=0)
                    ag_sc_01 = jax.lax.psum(
                        jnp.where(rank_mask_2d, sc01_1d[None, :], jnp.bfloat16(0.0)),
                        axis_name=expert_axis,
                    )
                    ag_sc_0 = ag_sc_01[:, :half_tok]
                    ag_sc_1 = ag_sc_01[:, half_tok:]
                else:
                    ag_sc_1 = jax.lax.psum(
                        jnp.where(rank_mask_2d, sc1_1d[None, :], jnp.bfloat16(0.0)),
                        axis_name=expert_axis,
                    )

            hs_4d = jnp.empty((ep_size, 2, half_tok, he_dim), dtype=jnp.bfloat16)
            local_topk_ids_u8_b, global_topk_u8_b, hs_4d, _ = (
                jax.lax.optimization_barrier(
                    (local_topk_ids_u8, global_topk_u8, hs_4d, i8_1)
                )
            )
            if b2_deq0_first:
                if b2_i8pack:
                    hs_4d = dequant_into_half_packed_1way(hs_4d, ag_i8_0, half_idx=0)
                else:
                    hs_4d = _dequant_into_half(hs_4d, ag_i8_0, ag_sc_0, half_idx=0)
                local_topk_ids = local_topk_ids_u8_b.astype(jnp.int32)
                perm_c = _stable_argsort_expert_ids(local_topk_ids.reshape(-1))
                inv_perm_c = (
                    _inverse_permutation(perm_c)
                    .reshape(t_local, topk)
                    .astype(jnp.int32)
                )
                global_topk_ids = global_topk_u8_b.astype(jnp.int32)
                lhs_indices, group_sizes, group_start_arg, _ = (
                    compute_block_scatter_routing(
                        global_topk_ids,
                        go_val,
                        my_id,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        topk=topk,
                    )
                )
                hs_4d, inv_perm_c, lhs_indices, group_sizes, group_start_arg = (
                    jax.lax.optimization_barrier(
                        (hs_4d, inv_perm_c, lhs_indices, group_sizes, group_start_arg)
                    )
                )
            else:
                local_topk_ids = local_topk_ids_u8_b.astype(jnp.int32)
                perm_c = _stable_argsort_expert_ids(local_topk_ids.reshape(-1))
                inv_perm_c = (
                    _inverse_permutation(perm_c)
                    .reshape(t_local, topk)
                    .astype(jnp.int32)
                )
                global_topk_ids = global_topk_u8_b.astype(jnp.int32)
                lhs_indices, group_sizes, group_start_arg, _ = (
                    compute_block_scatter_routing(
                        global_topk_ids,
                        go_val,
                        my_id,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        topk=topk,
                    )
                )
                hs_4d, inv_perm_c, lhs_indices, group_sizes, group_start_arg = (
                    jax.lax.optimization_barrier(
                        (hs_4d, inv_perm_c, lhs_indices, group_sizes, group_start_arg)
                    )
                )
                if b2_i8pack:
                    hs_4d = dequant_into_half_packed_1way(hs_4d, ag_i8_0, half_idx=0)
                else:
                    hs_4d = _dequant_into_half(hs_4d, ag_i8_0, ag_sc_0, half_idx=0)

            if b2_i8pack:
                hs_4d = dequant_into_half_packed_1way(hs_4d, ag_i8_1, half_idx=1)
            else:
                hs_4d = _dequant_into_half(hs_4d, ag_i8_1, ag_sc_1, half_idx=1)
            hidden_states_3d = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
        else:
            (
                router_in_bf16,
                router_in_fp8,
                router_in_sc_pad,
                stream_shard_mid,
                local_topk_ids_u8,
                local_topk_weights,
            ) = _call_b1(attn_s, stream_s)
            global_topk_u8 = jax.lax.all_gather(
                local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
            )
            i8_all, sc_1d = _call_b2(
                router_in_bf16, router_in_fp8, router_in_sc_pad, half_idx=None
            )
            ag_i8 = jax.lax.all_gather(
                i8_all, axis_name=expert_axis, axis=0, tiled=True
            )
            ag_i8_data, ag_sc = _unpack_ag_i8_and_scale(
                ag_i8, sc_1d, expert_axis=expert_axis, b2_i8pack=b2_i8pack, he_dim=he_dim
            )
            local_topk_ids = local_topk_ids_u8.astype(jnp.int32)
            perm_c = _stable_argsort_expert_ids(local_topk_ids.reshape(-1))
            inv_perm_c = (
                _inverse_permutation(perm_c).reshape(t_local, topk).astype(jnp.int32)
            )
            global_topk_ids = global_topk_u8.astype(jnp.int32)
            lhs_indices, group_sizes, group_start_arg, _ = (
                compute_block_scatter_routing(
                    global_topk_ids,
                    go_val,
                    my_id,
                    ep_size=ep_size,
                    num_local_experts=num_local_experts,
                    topk=topk,
                )
            )
            hidden_states_3d = (ag_i8_data.astype(jnp.bfloat16) * ag_sc).reshape(
                ep_size * t_local, he_dim // 128, 128
            )

        return (
            hidden_states_3d,
            stream_shard_mid,
            local_topk_weights,
            inv_perm_c,
            lhs_indices,
            group_sizes,
            group_start_arg,
        )

    return jax.shard_map(
        _bridge_body,
        mesh=mesh,
        in_specs=(
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(expert_axis),
        ),
        out_specs=(
            P(),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )(
        attn_out_sharded,
        stream_sharded,
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        wrouter,
        e_score_bias,
        wpool_fp8,
        wpool_scale,
        pre_expert_norm_w,
        group_offset,
    )


def _d_with_next_in_gather(
    *, d_fn, d_args, rows0: int, split: bool, axis_name, gather_static_fp8: bool = False
):
    """Kernel D with the first ``rows0`` rows of next_attn_in all-gathered: (new_stream, rest, gathered)."""
    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather

    # Split _d_with_next_in_gather overlaps AG with Kernel D Part 2 only when AG runs on
    # SparseCore (jax.lax.all_gather); XOR1_BRIDGE_AG=4 is MoE-only and keeps D on SparseCore.
    bridge_ag_mode = envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get()
    under_d_mode = envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D.get()
    use_xor1 = bool(under_d_mode) if split else (bridge_ag_mode in (1, 2, 3))
    xor1_xy_torus = (bridge_ag_mode >= 3) or (
        bool(under_d_mode) and bridge_ag_mode == 0
    )

    def _prep_part0(x_part):
        if gather_static_fp8 and x_part.dtype != jnp.float8_e4m3fn:
            return jnp.clip(
                x_part.astype(jnp.float32) * jnp.float32(32.0), -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
        return x_part

    def _ag_d(x_part):
        x_ag = _prep_part0(x_part)
        if use_xor1:
            return xor1_bridge_all_gather(
                x_ag, axis_name=axis_name, tiled=False, xy_torus=xor1_xy_torus
            )
        return jax.lax.all_gather(x_ag, axis_name, axis=0, tiled=False)


    if not split:  # rows0 not on a D tile boundary: same outputs, no overlap
        new_stream, next_in = d_fn(*d_args)
        gathered = _ag_d(next_in[:rows0])
        return new_stream, next_in[rows0:], gathered
    stream_0, next_in_0 = d_fn(*d_args, num_rows=rows0)
    # Issued before the second D call so the SparseCore all-gather runs under it.
    gathered = _ag_d(next_in_0)
    new_stream, next_in_rest = d_fn(*d_args, row_start=rows0, stream_out=stream_0)
    return new_stream, next_in_rest, gathered


def _moe_b2_all_gather(arr, *, axis_name: str):
    """MoE-internal B2 ag_i8 all-gather controlled by XOR1_AG_MOE or XOR1_BRIDGE_AG >= 2."""
    bridge_mode = envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get()
    moe_mode = envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE.get()
    eff_mode = (
        moe_mode if moe_mode > 0 else (bridge_mode if bridge_mode >= 2 else 0)
    )
    if eff_mode >= 2 or moe_mode >= 1:
        from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather

        return xor1_bridge_all_gather(
            arr, axis_name=axis_name, tiled=False, xy_torus=(eff_mode >= 3)
        )
    return jax.lax.all_gather(
        arr[None, ...], axis_name=axis_name, axis=0, tiled=True
    )


def _allow_pallas_sortfree_routing(
    pallas_sf_rt: int,
    *,
    sortfree_prefill: bool,
    topk: int,
    ep_size: int,
    t_local: int,
    n_loc_eff: int,
    has_pad_sentinel: bool,
    eff_hoist_routing: bool,
    eff_cm_pipe: bool,
) -> bool:
    """Gate PALLAS_SORTFREE_ROUTING; disabled on ep_size==16 until bounds-checked silicon repro passes."""
    return (
        pallas_sf_rt >= 1
        and sortfree_prefill
        and (topk & (topk - 1)) == 0
        and (ep_size * t_local) >= 256
        and (t_local % 128 == 0)
        and ((t_local * topk) % 128 == 0)
        and (n_loc_eff <= 120)
        and ep_size != 16
        and (eff_hoist_routing or has_pad_sentinel or eff_cm_pipe)
    )


def fused_moe_full_bridge_and_ffn(
    attn_out_sharded: jax.Array,
    stream_sharded: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_fp8: jax.Array,
    wpool_scale: jax.Array,
    pre_expert_norm_w: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    post_expert_norm_w: jax.Array,
    group_offset: jax.Array,
    wunpool_fp8: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    mesh: Mesh,
    topk: int = 8,
    router_mode: str = "bf16_hilo",
    b1_two_half: bool = False,
    b2_layout: str = "nstream",
    b2_fp8_lhs: bool = False,
    b2_two_half: bool = True,
    b2_pack_sc: bool = False,
    b2_i8pack: bool = False,
    b2_deq0_first: bool = False,
    hoist_routing: bool = True,
    d_layout: str = "4slab",
    d_tok_tile: int = 64,
    d_fp8_unpool: bool = False,
    d_emit_fp8_attn_in: bool = False,
    d_emit_static_fp8_attn_in: bool = False,
    d_two_half: bool = False,
    in_shard_map: bool = False,
    b2_deq0_new_buffer: bool = False,
    eplb_mode: str = "",
    valid_tokens: jax.Array | None = None,
    next_in_gather_rows: int = 0,
    next_in_gather_static_fp8: bool = False,
    ep_tp2_mode: bool = False,
    causal_chunk_stream: bool = False,
    causal_chunk_spec: str = "",
) -> tuple[jax.Array, ...]:
    """End-to-end unified shard_map: Kernel B1 + Overlapped B2/AG/Routing + Kernel C + Kernel D.

    ``in_shard_map``: the caller already runs inside a shard_map over ``mesh`` and passes
    every argument as this device's local block (the serving path's per-rank tensors).
    ``b2_deq0_new_buffer`` (with ``b2_deq0_first``, not ``b2_i8pack``): half 0's dequant
    allocates the AG buffer, instead of aliasing a copied ``jnp.empty``.
    ``next_in_gather_rows`` (in_shard_map, 4slab D; SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D): D runs
    its first that many tokens, then the rest, and the first rows' next_attn_in are all-gathered
    (tiled=False, ``[ep, rows, H]``) under the second part as the next layer's first A1 chunk.
    next_attn_in then holds the remaining rows only, and the gather is appended to the outputs.

    Returns:
        new_stream_sharded: fp32[T, 8192]
        next_attn_in_sharded: bf16[T, 8192] (or fp8_e4m3fn[T, 8320/8192])
        local_topk_ids_sharded: int32[T, topk]
        local_topk_weights_sharded: bf16[T, topk]
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    b2_fp8_lhs = b2_fp8_lhs or envs.SGLANG_TPU_MUSE_KERNEL_B2_FP8_LHS.get()

    def _full_body(
        attn_s,
        stream_s,
        a_attn,
        b_attn,
        pffn_w,
        w_r,
        bias,
        wp_fp8,
        wp_sc,
        pe_w,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        pnw_l,
        go_l,
        wu_fp8,
        wu_sc,
        post_ffn_w,
        a_ffn,
        b_ffn,
        next_in_w,
        *extra_vt,
    ):
        vt_arg = extra_vt[0] if extra_vt else valid_tokens
        gmm_layout_deq = (
            b2_deq0_new_buffer and envs.SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT.get()
        )
        dequant_1pass = (
            gmm_layout_deq and envs.SGLANG_TPU_MUSE_KERNEL_DEQUANT_1PASS.get()
        )
        b12_fast_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST", None)
        use_b12_fast = bool(b12_fast_env is not None and b12_fast_env.get())
        b2_pack_bf16_sc = (
            gmm_layout_deq
            and (not b2_fp8_lhs or use_b12_fast)
            and not b2_i8pack
            and b2_layout == "nstream"
            and envs.SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE.get()
        )
        fused_b1_b2_mode = envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get()
        t_local = attn_s.shape[0]
        he_dim = wp_fp8.shape[1]
        num_base_local = max(1, int(bias.shape[0]) // ep_size)
        num_local_experts = w1_l.shape[0]
        go_val = go_l[0]
        my_id = go_val // num_local_experts
        rank_mask_2d = jnp.arange(ep_size, dtype=jnp.int32)[:, None] == my_id
        eff_b2_two_half = b2_two_half and (t_local % 256 == 0)
        eff_hoist_routing = hoist_routing and not ep_tp2_mode

        eff_hi_pipe = (
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_HOT_INTERLEAVE.get()
            and w1_l.ndim == 3
        )
        eff_cm_pipe = (
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE.get()
            and (ep_size % 2 == 0)
            and w1_l.ndim == 3
        )
        _req_cm_slots_pipe = (
            int(envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS.get() or 1)
            if (
                eff_cm_pipe
                and (
                    not eplb_mode
                    or eplb_mode == "off"
                    or num_local_experts <= num_base_local
                )
            )
            else 0
        )
        eff_cm_slots_pipe = (
            3
            if (_req_cm_slots_pipe >= 3 and ep_size == 8)
            else (2 if _req_cm_slots_pipe >= 2 else (1 if eff_cm_pipe else 0))
        )
        eff_cm_mode_pipe = envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE.get()

        def _query_kernel_tile_m(n_groups_q: int) -> int:
            w1_shape = w1_l.shape[1:] if w1_l.ndim == 4 else w1_l.shape
            w2_shape = w2_l.shape[1:] if w2_l.ndim == 4 else w2_l.shape
            return _select_fused_rs_block_sizes(
                size_m=ep_size * t_local * topk,
                size_k1=w1_shape[1],
                size_n1=w1_shape[2],
                size_k2=w2_shape[1],
                size_n2=w2_shape[2],
                size_group=n_groups_q,
                size_lhs_group=ep_size * n_groups_q,
                ep_size=ep_size,
                out_dtype=jnp.bfloat16,
                w1_dtype=w1_l.dtype,
                w2_dtype=w2_l.dtype,
                is_quantized=w1s_l is not None,
                quant_block_size=(
                    recover_quant_block_size(w1_shape[1], w1s_l.shape[w1_l.ndim - 2])
                    if w1s_l is not None
                    else None
                ),
                act_fn="silu",
                fp8_direct_write=False,
                hot_interleave=eff_hi_pipe,
            ).tile_m

        def _maybe_remap_eplb(raw_global_ids, raw_local_ids):
            if not eplb_mode or eplb_mode == "off" or num_local_experts <= num_base_local:
                return raw_global_ids, raw_local_ids
            kernel_tile_m = _query_kernel_tile_m(
                num_local_experts + eff_cm_slots_pipe
            )
            return remap_zeroglue_virtual_ids(
                raw_global_ids,
                my_id,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                num_base_local=num_base_local,
                mode=eplb_mode,
                quantum=kernel_tile_m,
            )

        pallas_sf_rt = int(
            envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get() or 0
        )
        _pallas_counts_slot: list[tuple[jax.Array, jax.Array, jax.Array] | None] = [
            None
        ]

        def _can_use_pallas_sf(n_loc_eff: int) -> bool:
            return _allow_pallas_sortfree_routing(
                pallas_sf_rt,
                sortfree_prefill=bool(envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL.get()),
                topk=topk,
                ep_size=ep_size,
                t_local=t_local,
                n_loc_eff=n_loc_eff,
                has_pad_sentinel=vt_arg is not None,
                eff_hoist_routing=eff_hoist_routing,
                eff_cm_pipe=eff_cm_pipe,
            )

        def _route_ids(u8_local, u8_global, tw_local):
            cm_plan_local = None
            eff_n_loc = num_local_experts + eff_cm_slots_pipe
            eff_go_here = my_id * eff_n_loc if eff_cm_pipe else go_val
            _pallas_counts_slot[0] = None
            if vt_arg is None:
                if eplb_mode and eplb_mode != "off" and num_local_experts > num_base_local:
                    raw_local = u8_local.astype(jnp.int32)
                    raw_global = u8_global.astype(jnp.int32)
                    g_ids, l_ids = _maybe_remap_eplb(raw_global, raw_local)
                else:
                    raw_local = l_ids = u8_local.astype(jnp.int32)
                    g_ids = u8_global.astype(jnp.int32)
                if eff_cm_pipe:
                    if eff_cm_slots_pipe >= 2:
                        from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
                            plan_prefill_chipmate_2slot_jax,
                        )

                        _slot_xors_p = (1, 2, 4) if eff_cm_slots_pipe == 3 else (1, 1)
                        g_ids, _cm_all = plan_prefill_chipmate_2slot_jax(
                            g_ids,
                            ep_size=ep_size,
                            num_local_experts=num_local_experts,
                            tile_m=_query_kernel_tile_m(eff_n_loc),
                            chipmate_xor=1,
                            mode=eff_cm_mode_pipe,
                            slot_xors=_slot_xors_p,
                        )
                        _cm_entries_p = []
                        for _ci in range(eff_cm_slots_pipe):
                            _cm_entries_p.extend([_cm_all[my_id, 3 * _ci + 0], _cm_all[my_id, 3 * _ci + 1]])
                        cm_plan_local = jnp.stack(_cm_entries_p)
                    else:
                        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
                            plan_prefill_chipmate_jax,
                        )

                        g_ids, _cm_all = plan_prefill_chipmate_jax(
                            g_ids,
                            ep_size=ep_size,
                            num_local_experts=num_local_experts,
                            tile_m=_query_kernel_tile_m(eff_n_loc),
                            chipmate_xor=1,
                            mode=eff_cm_mode_pipe,
                        )
                        cm_plan_local = _cm_all[my_id, :2]
                    l_ids = jax.lax.dynamic_slice_in_dim(
                        g_ids, my_id * t_local, t_local, axis=0
                    )
                if _can_use_pallas_sf(eff_n_loc):
                    inv_p, _gs_p, _gstart_p, _trecv_p = pallas_local_inv_perm_and_counts(
                        l_ids,
                        g_ids,
                        eff_go_here,
                        my_id,
                        ep_size=ep_size,
                        num_local_experts=eff_n_loc,
                        total_id_buckets=ep_size * eff_n_loc,
                        has_pad_sentinel=False,
                        zero_local_start=True,
                    )
                    _pallas_counts_slot[0] = (_gs_p, _gstart_p, _trecv_p)
                else:
                    inv_p = _compute_local_inv_perm(l_ids, ep_size * eff_n_loc)
                return raw_local, l_ids, g_ids, inv_p, tw_local, None, cm_plan_local

            vt = vt_arg.reshape(-1)[0].astype(jnp.int32)
            raw_local = u8_local.astype(jnp.int32)
            raw_global = u8_global.astype(jnp.int32)
            if causal_chunk_stream:
                from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
                    chunk_major_global_token_ids_jax,
                    resolve_causal_chunks,
                )

                _ccs_chunks = resolve_causal_chunks(t_local, causal_chunk_spec)
                _tok_ids_2d = chunk_major_global_token_ids_jax(ep_size, _ccs_chunks)
                g_active = _tok_ids_2d.reshape(-1, 1) < vt
                l_active = (
                    jax.lax.dynamic_slice_in_dim(_tok_ids_2d, my_id, 1, axis=0).reshape(
                        t_local, 1
                    )
                    < vt
                )
            else:
                g_active = jnp.arange(ep_size * t_local, dtype=jnp.int32)[:, None] < vt
                l_active = (my_id * t_local + jnp.arange(t_local, dtype=jnp.int32))[
                    :, None
                ] < vt
            if eplb_mode and eplb_mode != "off" and num_local_experts > num_base_local:
                sentinel_base = jnp.int32(ep_size * num_base_local)
                masked_global = jnp.where(g_active, raw_global, sentinel_base)
                masked_local = jnp.where(l_active, raw_local, sentinel_base)
                g_ids, l_ids = _maybe_remap_eplb(masked_global, masked_local)
            else:
                sentinel = jnp.int32(ep_size * num_local_experts)
                g_ids = jnp.where(g_active, raw_global, sentinel)
                l_ids = jnp.where(l_active, raw_local, sentinel)
            if eff_cm_pipe:
                if eff_cm_slots_pipe >= 2:
                    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
                        plan_prefill_chipmate_2slot_jax,
                    )

                    _slot_xors_p = (1, 2, 4) if eff_cm_slots_pipe == 3 else (1, 1)
                    g_ids, _cm_all = plan_prefill_chipmate_2slot_jax(
                        g_ids,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        tile_m=_query_kernel_tile_m(eff_n_loc),
                        chipmate_xor=1,
                        mode=eff_cm_mode_pipe,
                        slot_xors=_slot_xors_p,
                    )
                    _cm_entries_p = []
                    for _ci in range(eff_cm_slots_pipe):
                        _cm_entries_p.extend([_cm_all[my_id, 3 * _ci + 0], _cm_all[my_id, 3 * _ci + 1]])
                    cm_plan_local = jnp.stack(_cm_entries_p)
                else:
                    from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
                        plan_prefill_chipmate_jax,
                    )

                    g_ids, _cm_all = plan_prefill_chipmate_jax(
                        g_ids,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        tile_m=_query_kernel_tile_m(eff_n_loc),
                        chipmate_xor=1,
                        mode=eff_cm_mode_pipe,
                    )
                    cm_plan_local = _cm_all[my_id, :2]
                l_ids = jax.lax.dynamic_slice_in_dim(
                    g_ids, my_id * t_local, t_local, axis=0
                )
            if _can_use_pallas_sf(eff_n_loc):
                inv_p, _gs_p, _gstart_p, _trecv_p = pallas_local_inv_perm_and_counts(
                    l_ids,
                    g_ids,
                    eff_go_here,
                    my_id,
                    ep_size=ep_size,
                    num_local_experts=eff_n_loc,
                    total_id_buckets=ep_size * eff_n_loc + 1,
                    has_pad_sentinel=True,
                    zero_local_start=True,
                )
                _pallas_counts_slot[0] = (_gs_p, _gstart_p, _trecv_p)
            else:
                inv_p = _compute_local_inv_perm(l_ids, ep_size * eff_n_loc + 1)
            safe_rows = min(1024, t_local * topk)
            pad_spread = (
                jnp.arange(t_local * topk, dtype=jnp.int32) % safe_rows
            ).reshape(t_local, topk)
            inv_p = jnp.where(l_active, inv_p, pad_spread)
            tw_masked = jnp.where(l_active, tw_local, jnp.bfloat16(0.0))
            active_cnt_local = jnp.sum(l_active.astype(jnp.int32)) * topk
            return (
                raw_local,
                l_ids,
                g_ids,
                inv_p,
                tw_masked,
                active_cnt_local,
                cm_plan_local,
            )

        def _call_b1(a_slice, s_slice):
            if b2_fp8_lhs:
                r_fp8, r_sc, smid, tk_u8, tk_w = (
                    pallas_postattn_res_prenorm_router_topk_fp8(
                        a_slice,
                        s_slice,
                        a_attn,
                        b_attn,
                        pffn_w,
                        w_r,
                        bias,
                        router_mode=router_mode,
                        topk=topk,
                    )
                )
                return None, r_fp8, r_sc, smid, tk_u8, tk_w
            r_bf16, smid, tk_u8, tk_w = pallas_postattn_res_prenorm_router_topk(
                a_slice,
                s_slice,
                a_attn,
                b_attn,
                pffn_w,
                w_r,
                bias,
                router_mode=router_mode,
                topk=topk,
            )
            return r_bf16, None, None, smid, tk_u8, tk_w

        def _call_b2(r_bf16_in, r_fp8_in, sc_in, half_idx: int | None = None):
            if b2_fp8_lhs:
                if b2_layout == "rowtile":
                    if half_idx is not None:
                        ht = r_fp8_in.shape[0] // 2
                        sl = slice(0, ht) if half_idx == 0 else slice(ht, 2 * ht)
                        r_fp8_in, sc_in = r_fp8_in[sl], sc_in[sl]
                    return pallas_wpool_prenorm_int8_quant_fp8_rowtile(
                        r_fp8_in, sc_in, wp_fp8, wp_sc, pe_w
                    )
                return pallas_wpool_prenorm_int8_quant_fp8_nstream(
                    r_fp8_in,
                    sc_in,
                    wp_fp8,
                    wp_sc,
                    pe_w,
                    pack_sc_in_i8=b2_i8pack or b2_pack_bf16_sc,
                    half_idx=half_idx,
                )
            if b2_layout == "rowtile":
                if half_idx is not None:
                    ht = r_bf16_in.shape[0] // 2
                    sl = slice(0, ht) if half_idx == 0 else slice(ht, 2 * ht)
                    r_bf16_in = r_bf16_in[sl]
                return pallas_wpool_prenorm_int8_quant_rowtile(
                    r_bf16_in, wp_fp8, wp_sc, pe_w
                )
            return pallas_wpool_prenorm_int8_quant_nstream(
                r_bf16_in,
                wp_fp8,
                wp_sc,
                pe_w,
                pack_bf16_sc=b2_pack_bf16_sc,
                half_idx=half_idx,
            )

        hidden_states_scale = None
        _pq_lhs_mode = int(
            envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get() or 0
        )
        _use_pq_lhs_fp8 = (
            _pq_lhs_mode >= 1
            and not ep_tp2_mode
            and (ep_size * t_local * topk) >= 16384
            and (
                (
                    jnp.dtype(w1_l.dtype) == jnp.float4_e2m1fn
                    and int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS.get() or 0)
                    >= he_dim
                )
                or (
                    jnp.dtype(w1_l.dtype) == jnp.float8_e4m3fn
                    and int(envs.SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS.get() or 0)
                    >= he_dim
                )
            )
        )
        _b2_direct_fp8_1pass = _use_pq_lhs_fp8 and _pq_lhs_mode == 4
        _b2_direct_fp8_2half = _use_pq_lhs_fp8 and _pq_lhs_mode == 5

        def _moe_b2_all_gather_local(arr):
            return _moe_b2_all_gather(arr, axis_name=expert_axis)
        with jax.named_scope("stage1_2_kernel_b1_b2_and_overlapped_ag_routing"):
            if (
                (fused_b1_b2_mode in (1, 2) and (t_local % 128 == 0))
                or (
                    fused_b1_b2_mode in (3, 4)
                    and (t_local % 128 == 0)
                    and (t_local % 256 != 0)
                )
                or (
                    _b2_direct_fp8_1pass
                    and fused_b1_b2_mode in (1, 2, 3, 4)
                    and (t_local % 128 == 0)
                )
            ):
                if _b2_direct_fp8_1pass:
                    (
                        fp8_all,
                        sc_local_1d,
                        stream_shard_mid,
                        local_topk_ids_u8,
                        local_topk_weights,
                    ) = pallas_fused_b1_b2_wpool_quant(
                        attn_s,
                        stream_s,
                        a_attn,
                        b_attn,
                        pffn_w,
                        w_r,
                        bias,
                        wp_fp8,
                        wp_sc,
                        pe_w,
                        router_mode=router_mode,
                        fp8_wpool=(fused_b1_b2_mode in (2, 4)),
                        half_idx=None,
                        topk=topk,
                        emit_fp8_gmm=True,
                    )
                    i8_all = fp8_all.reshape(t_local * (he_dim // 128), 128)
                else:
                    sc_local_1d = None
                    (
                        i8_all,
                        stream_shard_mid,
                        local_topk_ids_u8,
                        local_topk_weights,
                    ) = pallas_fused_b1_b2_wpool_quant(
                        attn_s,
                        stream_s,
                        a_attn,
                        b_attn,
                        pffn_w,
                        w_r,
                        bias,
                        wp_fp8,
                        wp_sc,
                        pe_w,
                        router_mode=router_mode,
                        fp8_wpool=(fused_b1_b2_mode in (2, 4)),
                        half_idx=None,
                        topk=topk,
                    )
                i8_all, local_topk_ids_u8 = jax.lax.optimization_barrier(
                    (i8_all, local_topk_ids_u8)
                )
                ag_i8 = _moe_b2_all_gather_local(i8_all)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                )
                if _b2_direct_fp8_1pass:
                    hidden_states_scale = jax.lax.all_gather(
                        sc_local_1d, axis_name=expert_axis, axis=0, tiled=True
                    ).reshape(-1)
                local_topk_ids_u8_b, global_topk_u8_b, _ = (
                    jax.lax.optimization_barrier(
                        (local_topk_ids_u8, global_topk_u8, i8_all)
                    )
                )
                eff_n_loc_rt = num_local_experts + eff_cm_slots_pipe
                eff_go_rt = my_id * eff_n_loc_rt if eff_cm_pipe else go_val
                (
                    raw_local_topk_ids,
                    local_topk_ids,
                    global_topk_ids,
                    inv_perm_c,
                    local_topk_weights,
                    active_cnt_local,
                    pipe_cm_plan,
                ) = _route_ids(
                    local_topk_ids_u8_b, global_topk_u8_b, local_topk_weights
                )
                if eff_hoist_routing or vt_arg is not None or eff_cm_pipe:
                    lhs_indices, group_sizes, group_start_arg, local_t_recv = (
                        compute_block_scatter_routing(
                            global_topk_ids,
                            eff_go_rt,
                            my_id,
                            ep_size=ep_size,
                            num_local_experts=eff_n_loc_rt,
                            topk=topk,
                            stable_sort=bool(
                                (eplb_mode and eplb_mode != "off") or eff_cm_pipe
                            ),
                            has_pad_sentinel=(vt_arg is not None),
                            precomputed_counts=_pallas_counts_slot[0],
                        )
                    )
                    (
                        inv_perm_c,
                        lhs_indices,
                        group_sizes,
                        group_start_arg,
                        local_t_recv,
                    ) = jax.lax.optimization_barrier(
                        (
                            inv_perm_c,
                            lhs_indices,
                            group_sizes,
                            group_start_arg,
                            local_t_recv,
                        )
                    )
                    precomputed_routing = (lhs_indices, group_sizes, group_start_arg)
                else:
                    local_t_recv = jnp.sum(
                        (
                            (local_topk_ids < go_val)
                            | (local_topk_ids >= go_val + num_local_experts)
                        ).astype(jnp.int32)
                    )
                    inv_perm_c, local_t_recv = jax.lax.optimization_barrier(
                        (inv_perm_c, local_t_recv)
                    )
                    precomputed_routing = None
                if _b2_direct_fp8_1pass:
                    hidden_states = ag_i8.reshape(
                        ep_size * t_local, he_dim // 128, 128
                    )
                elif _use_pq_lhs_fp8:
                    hidden_states, hidden_states_scale = (
                        _convert_both_halves_fp8_gmm_layout(ag_i8, None, None, None)
                    )
                else:
                    hs_4d = dequant_single_gmm_layout(ag_i8)
                    hidden_states = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
            elif fused_b1_b2_mode in (3, 4) and (t_local % 256 == 0):
                half_tok = t_local // 2
                fp8_wp = fused_b1_b2_mode == 4
                if _b2_direct_fp8_2half:
                    fp8_0, sc_0, smid_0, tk_u8_0, tk_w_0 = (
                        pallas_fused_b1_b2_wpool_quant(
                            attn_s,
                            stream_s,
                            a_attn,
                            b_attn,
                            pffn_w,
                            w_r,
                            bias,
                            wp_fp8,
                            wp_sc,
                            pe_w,
                            router_mode=router_mode,
                            fp8_wpool=fp8_wp,
                            half_idx=0,
                            full_stream_mid=True,
                            topk=topk,
                            emit_fp8_gmm=True,
                        )
                    )
                    i8_0 = fp8_0.reshape(half_tok * (he_dim // 128), 128)
                else:
                    sc_0 = None
                    i8_0, smid_0, tk_u8_0, tk_w_0 = pallas_fused_b1_b2_wpool_quant(
                        attn_s,
                        stream_s,
                        a_attn,
                        b_attn,
                        pffn_w,
                        w_r,
                        bias,
                        wp_fp8,
                        wp_sc,
                        pe_w,
                        router_mode=router_mode,
                        fp8_wpool=fp8_wp,
                        half_idx=0,
                        full_stream_mid=True,
                        topk=topk,
                    )
                i8_0, smid_0, tk_u8_0, tk_w_0 = jax.lax.optimization_barrier(
                    (i8_0, smid_0, tk_u8_0, tk_w_0)
                )
                ag_i8_0 = _moe_b2_all_gather_local(i8_0)
                attn_s_1, stream_s_1, smid_0_1, _ = jax.lax.optimization_barrier(
                    (attn_s, stream_s, smid_0, i8_0)
                )
                if _b2_direct_fp8_2half:
                    fp8_1, sc_1, stream_shard_mid, tk_u8_1, tk_w_1 = (
                        pallas_fused_b1_b2_wpool_quant(
                            attn_s_1,
                            stream_s_1,
                            a_attn,
                            b_attn,
                            pffn_w,
                            w_r,
                            bias,
                            wp_fp8,
                            wp_sc,
                            pe_w,
                            router_mode=router_mode,
                            fp8_wpool=fp8_wp,
                            half_idx=1,
                            stream_mid_prev=smid_0_1,
                            topk=topk,
                            emit_fp8_gmm=True,
                        )
                    )
                    i8_1 = fp8_1.reshape(half_tok * (he_dim // 128), 128)
                else:
                    sc_1 = None
                    i8_1, stream_shard_mid, tk_u8_1, tk_w_1 = (
                        pallas_fused_b1_b2_wpool_quant(
                            attn_s_1,
                            stream_s_1,
                            a_attn,
                            b_attn,
                            pffn_w,
                            w_r,
                            bias,
                            wp_fp8,
                            wp_sc,
                            pe_w,
                            router_mode=router_mode,
                            fp8_wpool=fp8_wp,
                            half_idx=1,
                            stream_mid_prev=smid_0_1,
                            topk=topk,
                        )
                    )
                local_topk_ids_u8 = jnp.concatenate([tk_u8_0, tk_u8_1], axis=0)
                local_topk_weights = jnp.concatenate([tk_w_0, tk_w_1], axis=0)
                i8_1 = jax.lax.optimization_barrier(i8_1)
                ag_i8_1 = _moe_b2_all_gather_local(i8_1)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                )
                if _b2_direct_fp8_2half:
                    sc_01_local = jnp.concatenate([sc_0, sc_1], axis=0)
                    hidden_states_scale = jax.lax.all_gather(
                        sc_01_local, axis_name=expert_axis, axis=0, tiled=True
                    ).reshape(-1)
                ovl_deq0_env = getattr(
                    envs, "SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG", None
                )
                use_ovl_deq0 = bool(ovl_deq0_env is not None and ovl_deq0_env.get())
                if use_ovl_deq0 and not (dequant_1pass or _use_pq_lhs_fp8):
                    ag_i8_0_b, _ = jax.lax.optimization_barrier((ag_i8_0, i8_1))
                    hs_4d = dequant_half_gmm_layout(ag_i8_0_b, None)
                    hs_4d, local_topk_ids_u8_b, global_topk_u8_b = (
                        jax.lax.optimization_barrier(
                            (hs_4d, local_topk_ids_u8, global_topk_u8)
                        )
                    )
                else:
                    local_topk_ids_u8_b, global_topk_u8_b, _ = (
                        jax.lax.optimization_barrier(
                            (local_topk_ids_u8, global_topk_u8, i8_1)
                        )
                    )
                    hs_4d = (
                        None
                        if (dequant_1pass or _use_pq_lhs_fp8)
                        else dequant_half_gmm_layout(ag_i8_0, None)
                    )
                eff_n_loc_rt = num_local_experts + eff_cm_slots_pipe
                eff_go_rt = my_id * eff_n_loc_rt if eff_cm_pipe else go_val
                (
                    raw_local_topk_ids,
                    local_topk_ids,
                    global_topk_ids,
                    inv_perm_c,
                    local_topk_weights,
                    active_cnt_local,
                    pipe_cm_plan,
                ) = _route_ids(
                    local_topk_ids_u8_b, global_topk_u8_b, local_topk_weights
                )
                if eff_hoist_routing or vt_arg is not None or eff_cm_pipe:
                    lhs_indices, group_sizes, group_start_arg, local_t_recv = (
                        compute_block_scatter_routing(
                            global_topk_ids,
                            eff_go_rt,
                            my_id,
                            ep_size=ep_size,
                            num_local_experts=eff_n_loc_rt,
                            topk=topk,
                            stable_sort=bool(
                                (eplb_mode and eplb_mode != "off") or eff_cm_pipe
                            ),
                            has_pad_sentinel=(vt_arg is not None),
                            precomputed_counts=_pallas_counts_slot[0],
                        )
                    )
                    if hs_4d is None:
                        (
                            inv_perm_c,
                            lhs_indices,
                            group_sizes,
                            group_start_arg,
                            local_t_recv,
                        ) = jax.lax.optimization_barrier(
                            (
                                inv_perm_c,
                                lhs_indices,
                                group_sizes,
                                group_start_arg,
                                local_t_recv,
                            )
                        )
                    else:
                        (
                            hs_4d,
                            inv_perm_c,
                            lhs_indices,
                            group_sizes,
                            group_start_arg,
                            local_t_recv,
                        ) = jax.lax.optimization_barrier(
                            (
                                hs_4d,
                                inv_perm_c,
                                lhs_indices,
                                group_sizes,
                                group_start_arg,
                                local_t_recv,
                            )
                        )
                    precomputed_routing = (lhs_indices, group_sizes, group_start_arg)
                else:
                    local_t_recv = jnp.sum(
                        (
                            (local_topk_ids < go_val)
                            | (local_topk_ids >= go_val + num_local_experts)
                        ).astype(jnp.int32)
                    )
                    precomputed_routing = None
                if _b2_direct_fp8_2half:
                    hidden_states = jnp.concatenate(
                        [
                            ag_i8_0.reshape(ep_size, half_tok, he_dim // 128, 128),
                            ag_i8_1.reshape(ep_size, half_tok, he_dim // 128, 128),
                        ],
                        axis=1,
                    ).reshape(ep_size * t_local, he_dim // 128, 128)
                elif _use_pq_lhs_fp8:
                    hidden_states, hidden_states_scale = (
                        _convert_both_halves_fp8_gmm_layout(
                            ag_i8_0, None, ag_i8_1, None
                        )
                    )
                elif dequant_1pass:
                    hs_4d = dequant_both_halves_gmm_layout(
                        ag_i8_0, None, ag_i8_1, None
                    )
                    hidden_states = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
                else:
                    hs_4d = dequant_half_gmm_layout(ag_i8_1, None, hs_4d)
                    hidden_states = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
            elif eff_b2_two_half:
                half_tok = t_local // 2
                if b1_two_half:
                    r_bf16_0, r_fp8_0, r_sc_0, smid_0, tk_u8_0, tk_w_0 = _call_b1(
                        attn_s[:half_tok], stream_s[:half_tok]
                    )
                    i8_0, sc0_1d = _call_b2(r_bf16_0, r_fp8_0, r_sc_0, half_idx=None)
                    if b2_i8pack:
                        i8_0, smid_0, tk_u8_0, tk_w_0 = jax.lax.optimization_barrier(
                            (i8_0, smid_0, tk_u8_0, tk_w_0)
                        )
                    else:
                        i8_0, sc0_1d, smid_0, tk_u8_0, tk_w_0 = (
                            jax.lax.optimization_barrier(
                                (i8_0, sc0_1d, smid_0, tk_u8_0, tk_w_0)
                            )
                        )
                    ag_i8_0 = _moe_b2_all_gather_local(i8_0)
                    if not b2_pack_sc and not b2_i8pack:
                        ag_sc_0 = jax.lax.psum(
                            jnp.where(rank_mask_2d, sc0_1d[None, :], jnp.bfloat16(0.0)),
                            axis_name=expert_axis,
                        )
                    a_1, s_1, _ = jax.lax.optimization_barrier(
                        (attn_s[half_tok:], stream_s[half_tok:], i8_0)
                    )
                    r_bf16_1, r_fp8_1, r_sc_1, smid_1, tk_u8_1, tk_w_1 = _call_b1(
                        a_1, s_1
                    )
                    stream_shard_mid = jnp.concatenate([smid_0, smid_1], axis=0)
                    local_topk_ids_u8 = jnp.concatenate([tk_u8_0, tk_u8_1], axis=0)
                    local_topk_weights = jnp.concatenate([tk_w_0, tk_w_1], axis=0)
                    global_topk_u8 = jax.lax.all_gather(
                        local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                    )
                    if b2_fp8_lhs:
                        r_fp8_1, r_sc_1, _ = jax.lax.optimization_barrier(
                            (r_fp8_1, r_sc_1, global_topk_u8)
                        )
                        i8_1, sc1_1d = _call_b2(None, r_fp8_1, r_sc_1, half_idx=None)
                    else:
                        r_bf16_1, _ = jax.lax.optimization_barrier(
                            (r_bf16_1, global_topk_u8)
                        )
                        i8_1, sc1_1d = _call_b2(r_bf16_1, None, None, half_idx=None)
                else:
                    (
                        router_in_bf16,
                        router_in_fp8,
                        router_in_sc_pad,
                        stream_shard_mid,
                        local_topk_ids_u8,
                        local_topk_weights,
                    ) = _call_b1(attn_s, stream_s)
                    global_topk_u8 = jax.lax.all_gather(
                        local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                    )
                    if b2_fp8_lhs:
                        r_fp8_0, r_sc_0, _ = jax.lax.optimization_barrier(
                            (router_in_fp8, router_in_sc_pad, global_topk_u8)
                        )
                        i8_0, sc0_1d = _call_b2(None, r_fp8_0, r_sc_0, half_idx=0)
                    else:
                        r_bf16_0, _ = jax.lax.optimization_barrier(
                            (router_in_bf16, global_topk_u8)
                        )
                        i8_0, sc0_1d = _call_b2(r_bf16_0, None, None, half_idx=0)

                    if b2_i8pack or b2_pack_bf16_sc:
                        i8_0 = jax.lax.optimization_barrier(i8_0)
                    else:
                        i8_0, sc0_1d = jax.lax.optimization_barrier((i8_0, sc0_1d))
                    ag_i8_0 = _moe_b2_all_gather_local(i8_0)
                    if b2_pack_bf16_sc:
                        ag_sc_0 = None
                    elif not b2_pack_sc and not b2_i8pack:
                        ag_sc_0 = jax.lax.psum(
                            jnp.where(rank_mask_2d, sc0_1d[None, :], jnp.bfloat16(0.0)),
                            axis_name=expert_axis,
                        )

                    if b2_fp8_lhs:
                        if b2_i8pack or b2_pack_bf16_sc:
                            r_fp8_1, r_sc_1, _ = jax.lax.optimization_barrier(
                                (router_in_fp8, router_in_sc_pad, i8_0)
                            )
                            i8_1, _ = _call_b2(None, r_fp8_1, r_sc_1, half_idx=1)
                            sc1_1d = None
                        else:
                            r_fp8_1, r_sc_1, _, _ = jax.lax.optimization_barrier(
                                (router_in_fp8, router_in_sc_pad, i8_0, sc0_1d)
                            )
                            i8_1, sc1_1d = _call_b2(None, r_fp8_1, r_sc_1, half_idx=1)
                    elif b2_pack_bf16_sc:
                        r_bf16_1, _ = jax.lax.optimization_barrier(
                            (router_in_bf16, i8_0)
                        )
                        i8_1, sc1_1d = _call_b2(r_bf16_1, None, None, half_idx=1)
                    else:
                        r_bf16_1, _, _ = jax.lax.optimization_barrier(
                            (router_in_bf16, i8_0, sc0_1d)
                        )
                        i8_1, sc1_1d = _call_b2(r_bf16_1, None, None, half_idx=1)

                if b2_deq0_first:
                    if b2_i8pack or b2_pack_bf16_sc:
                        i8_1 = jax.lax.optimization_barrier(i8_1)
                    else:
                        i8_1, sc1_1d = jax.lax.optimization_barrier((i8_1, sc1_1d))
                    ag_i8_1 = _moe_b2_all_gather_local(i8_1)
                    if b2_pack_bf16_sc:
                        ag_sc_1 = None
                    elif not b2_i8pack:
                        if b2_pack_sc:
                            sc01_1d = jnp.concatenate([sc0_1d, sc1_1d], axis=0)
                            ag_sc_01 = jax.lax.psum(
                                jnp.where(
                                    rank_mask_2d, sc01_1d[None, :], jnp.bfloat16(0.0)
                                ),
                                axis_name=expert_axis,
                            )
                            ag_sc_0 = ag_sc_01[:, :half_tok]
                            ag_sc_1 = ag_sc_01[:, half_tok:]
                        else:
                            ag_sc_1 = jax.lax.psum(
                                jnp.where(
                                    rank_mask_2d, sc1_1d[None, :], jnp.bfloat16(0.0)
                                ),
                                axis_name=expert_axis,
                            )
                    if b2_deq0_new_buffer:
                        assert not b2_i8pack
                        local_topk_ids_u8_b, global_topk_u8_b, _ = (
                            jax.lax.optimization_barrier(
                                (local_topk_ids_u8, global_topk_u8, i8_1)
                            )
                        )
                        if dequant_1pass:
                            hs_4d = None
                        elif gmm_layout_deq:
                            hs_4d = dequant_half_gmm_layout(ag_i8_0, ag_sc_0)
                        else:
                            hs_4d = dequant_half0_new_buffer(ag_i8_0, ag_sc_0)
                        if (
                            hs_4d is not None
                            and envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE.get() == 3
                        ):
                            # Order local_topk_ids_u8_b after hs_4d so MXU-based sort-free inv_perm_c
                            # cannot be hoisted ahead of SparseCore async all-gather start!
                            hs_4d, local_topk_ids_u8_b = jax.lax.optimization_barrier(
                                (hs_4d, local_topk_ids_u8_b)
                            )
                    else:
                        hs_4d = jnp.empty(
                            (ep_size, 2, half_tok, he_dim), dtype=jnp.bfloat16
                        )
                        local_topk_ids_u8_b, global_topk_u8_b, hs_4d, _ = (
                            jax.lax.optimization_barrier(
                                (local_topk_ids_u8, global_topk_u8, hs_4d, i8_1)
                            )
                        )
                        if b2_i8pack:
                            hs_4d = dequant_into_half_packed_1way(
                                hs_4d, ag_i8_0, half_idx=0
                            )
                        else:
                            hs_4d = _dequant_into_half(
                                hs_4d, ag_i8_0, ag_sc_0, half_idx=0
                            )
                    eff_n_loc_rt = num_local_experts + eff_cm_slots_pipe
                    eff_go_rt = my_id * eff_n_loc_rt if eff_cm_pipe else go_val
                    (
                        raw_local_topk_ids,
                        local_topk_ids,
                        global_topk_ids,
                        inv_perm_c,
                        local_topk_weights,
                        active_cnt_local,
                        pipe_cm_plan,
                    ) = _route_ids(
                        local_topk_ids_u8_b, global_topk_u8_b, local_topk_weights
                    )
                    if eff_hoist_routing or vt_arg is not None or eff_cm_pipe:
                        lhs_indices, group_sizes, group_start_arg, local_t_recv = (
                            compute_block_scatter_routing(
                                global_topk_ids,
                                eff_go_rt,
                                my_id,
                                ep_size=ep_size,
                                num_local_experts=eff_n_loc_rt,
                                topk=topk,
                                stable_sort=bool(
                                    (eplb_mode and eplb_mode != "off") or eff_cm_pipe
                                ),
                                has_pad_sentinel=(vt_arg is not None),
                                precomputed_counts=_pallas_counts_slot[0],
                            )
                        )
                        if dequant_1pass:
                            (
                                inv_perm_c,
                                lhs_indices,
                                group_sizes,
                                group_start_arg,
                                local_t_recv,
                            ) = jax.lax.optimization_barrier(
                                (
                                    inv_perm_c,
                                    lhs_indices,
                                    group_sizes,
                                    group_start_arg,
                                    local_t_recv,
                                )
                            )
                        else:
                            (
                                hs_4d,
                                inv_perm_c,
                                lhs_indices,
                                group_sizes,
                                group_start_arg,
                                local_t_recv,
                            ) = jax.lax.optimization_barrier(
                                (
                                    hs_4d,
                                    inv_perm_c,
                                    lhs_indices,
                                    group_sizes,
                                    group_start_arg,
                                    local_t_recv,
                                )
                            )
                        precomputed_routing = (
                            lhs_indices,
                            group_sizes,
                            group_start_arg,
                        )
                    else:
                        local_t_recv = jnp.sum(
                            (
                                (local_topk_ids < go_val)
                                | (local_topk_ids >= go_val + num_local_experts)
                            ).astype(jnp.int32)
                        )
                        if dequant_1pass:
                            inv_perm_c, local_t_recv = jax.lax.optimization_barrier(
                                (inv_perm_c, local_t_recv)
                            )
                        else:
                            hs_4d, inv_perm_c, local_t_recv = (
                                jax.lax.optimization_barrier(
                                    (hs_4d, inv_perm_c, local_t_recv)
                                )
                            )
                        precomputed_routing = None
                    if b2_i8pack:
                        hs_4d = dequant_into_half_packed_1way(
                            hs_4d, ag_i8_1, half_idx=1
                        )
                    elif dequant_1pass:
                        hs_4d = dequant_both_halves_gmm_layout(
                            ag_i8_0, ag_sc_0, ag_i8_1, ag_sc_1
                        )
                    elif gmm_layout_deq:
                        hs_4d = dequant_half_gmm_layout(ag_i8_1, ag_sc_1, hs_4d)
                    else:
                        hs_4d = _dequant_into_half(hs_4d, ag_i8_1, ag_sc_1, half_idx=1)
                    hidden_states = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
                else:
                    if b2_i8pack:
                        i8_1 = jax.lax.optimization_barrier(i8_1)
                    else:
                        i8_1, sc1_1d = jax.lax.optimization_barrier((i8_1, sc1_1d))
                    eff_n_loc_rt = num_local_experts + eff_cm_slots_pipe
                    eff_go_rt = my_id * eff_n_loc_rt if eff_cm_pipe else go_val
                    (
                        raw_local_topk_ids,
                        local_topk_ids,
                        global_topk_ids,
                        inv_perm_c,
                        local_topk_weights,
                        active_cnt_local,
                        pipe_cm_plan,
                    ) = _route_ids(
                        local_topk_ids_u8, global_topk_u8, local_topk_weights
                    )

                    if eff_hoist_routing or vt_arg is not None or eff_cm_pipe:
                        lhs_indices, group_sizes, group_start_arg, local_t_recv = (
                            compute_block_scatter_routing(
                                global_topk_ids,
                                eff_go_rt,
                                my_id,
                                ep_size=ep_size,
                                num_local_experts=eff_n_loc_rt,
                                topk=topk,
                                stable_sort=bool(
                                    (eplb_mode and eplb_mode != "off") or eff_cm_pipe
                                ),
                                has_pad_sentinel=(vt_arg is not None),
                                precomputed_counts=_pallas_counts_slot[0],
                            )
                        )
                        precomputed_routing = (
                            lhs_indices,
                            group_sizes,
                            group_start_arg,
                        )
                    else:
                        local_t_recv = jnp.sum(
                            (
                                (local_topk_ids < go_val)
                                | (local_topk_ids >= go_val + num_local_experts)
                            ).astype(jnp.int32)
                        )
                        precomputed_routing = None

                    ag_i8_1 = _moe_b2_all_gather_local(i8_1)
                    hs_4d = jnp.empty(
                        (ep_size, 2, half_tok, he_dim), dtype=jnp.bfloat16
                    )
                    if b2_i8pack:
                        hs_4d = dequant_into_half_packed_1way(
                            hs_4d, ag_i8_0, half_idx=0
                        )
                        hs_4d = dequant_into_half_packed_1way(
                            hs_4d, ag_i8_1, half_idx=1
                        )
                    else:
                        if b2_pack_sc:
                            sc01_1d = jnp.concatenate([sc0_1d, sc1_1d], axis=0)
                            ag_sc_01 = jax.lax.psum(
                                jnp.where(
                                    rank_mask_2d, sc01_1d[None, :], jnp.bfloat16(0.0)
                                ),
                                axis_name=expert_axis,
                            )
                            ag_sc_0 = ag_sc_01[:, :half_tok]
                            ag_sc_1 = ag_sc_01[:, half_tok:]
                        else:
                            ag_sc_1 = jax.lax.psum(
                                jnp.where(
                                    rank_mask_2d, sc1_1d[None, :], jnp.bfloat16(0.0)
                                ),
                                axis_name=expert_axis,
                            )
                        hs_4d = _dequant_into_half(hs_4d, ag_i8_0, ag_sc_0, half_idx=0)
                        hs_4d = _dequant_into_half(hs_4d, ag_i8_1, ag_sc_1, half_idx=1)
                    hidden_states = hs_4d.reshape(ep_size * t_local, he_dim // 128, 128)
            else:
                (
                    router_in_bf16,
                    router_in_fp8,
                    router_in_sc_pad,
                    stream_shard_mid,
                    local_topk_ids_u8,
                    local_topk_weights,
                ) = _call_b1(attn_s, stream_s)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids_u8, axis_name=expert_axis, axis=0, tiled=True
                )
                i8_all, sc_1d = _call_b2(
                    router_in_bf16, router_in_fp8, router_in_sc_pad, half_idx=None
                )
                ag_i8 = jax.lax.all_gather(
                    i8_all, axis_name=expert_axis, axis=0, tiled=True
                )
                ag_i8_data, ag_sc = _unpack_ag_i8_and_scale(
                    ag_i8,
                    sc_1d,
                    expert_axis=expert_axis,
                    b2_i8pack=b2_i8pack,
                    he_dim=he_dim,
                )
                eff_n_loc_rt = num_local_experts + eff_cm_slots_pipe
                eff_go_rt = my_id * eff_n_loc_rt if eff_cm_pipe else go_val
                (
                    raw_local_topk_ids,
                    local_topk_ids,
                    global_topk_ids,
                    inv_perm_c,
                    local_topk_weights,
                    active_cnt_local,
                    pipe_cm_plan,
                ) = _route_ids(local_topk_ids_u8, global_topk_u8, local_topk_weights)
                if vt_arg is not None or eff_cm_pipe:
                    lhs_indices, group_sizes, group_start_arg, local_t_recv = (
                        compute_block_scatter_routing(
                            global_topk_ids,
                            eff_go_rt,
                            my_id,
                            ep_size=ep_size,
                            num_local_experts=eff_n_loc_rt,
                            topk=topk,
                            stable_sort=bool(
                                (eplb_mode and eplb_mode != "off") or eff_cm_pipe
                            ),
                            has_pad_sentinel=(vt_arg is not None),
                        )
                    )
                    precomputed_routing = (lhs_indices, group_sizes, group_start_arg)
                else:
                    local_t_recv = jnp.sum(
                        (
                            (local_topk_ids < go_val)
                            | (local_topk_ids >= go_val + num_local_experts)
                        ).astype(jnp.int32)
                    )
                    precomputed_routing = None
                hidden_states = (ag_i8_data.astype(jnp.bfloat16) * ag_sc).reshape(
                    ep_size * t_local, he_dim // 128, 128
                )

        with jax.named_scope("stage3_kernel_c_gmm_fused_rs"):
            if ep_tp2_mode:
                from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
                    _ep_tp2_virtual_ids,
                )

                hs_by_rank = hidden_states.reshape(ep_size, t_local, he_dim)
                gids_by_rank = global_topk_ids.reshape(ep_size, t_local, topk)
                pnw_1d = pnw_l.reshape((he_dim,)).astype(jnp.float32) + 1.0
                max_c_local = max(128, 4096 // ep_size)
                v_parts = []
                for r_st in range(0, t_local, max_c_local):
                    c_loc = min(max_c_local, t_local - r_st)
                    hs_c = hs_by_rank[:, r_st : r_st + c_loc, :].reshape(
                        ep_size * c_loc, he_dim
                    )
                    gids_c = gids_by_rank[:, r_st : r_st + c_loc, :].reshape(
                        ep_size * c_loc, topk
                    )
                    virt_ids_c = _ep_tp2_virtual_ids(
                        gids_c, e_chip=num_local_experts, chipmate_xor=1
                    )
                    virt_w_c = jnp.ones(
                        (c_loc, topk * 2), dtype=hidden_states.dtype
                    )
                    raw_c = moe_gmm_local_rs_nodedup(
                        hs_c,
                        w1_l,
                        w1s_l,
                        None,
                        w2_l,
                        w2s_l,
                        None,
                        None,
                        None,
                        go_l,
                        virt_w_c,
                        virt_ids_c,
                        None,
                        activation="silu",
                        topk=topk * 2,
                        ep_size=ep_size,
                        ep_axis_name=expert_axis,
                        has_post_norm=False,
                        sp_enabled=True,
                        fp8_direct_write=False,
                        expert_tp=False,
                        fp4_w4a16=False,
                        block_scatter=False,
                        return_raw_out_buf=True,
                    )
                    out_4d = raw_c.reshape(2, topk, c_loc, -1)[:, :, :, :he_dim]
                    z_f32 = out_4d[0].astype(jnp.float32) + out_4d[1].astype(
                        jnp.float32
                    )
                    u_f32 = z_f32 * pnw_1d[None, None, :]
                    v_bf16 = (
                        u_f32
                        * jax.lax.rsqrt(
                            jnp.mean(u_f32 * u_f32, axis=-1, keepdims=True)
                            + jnp.float32(1e-8)
                        )
                    ).astype(jnp.bfloat16)
                    v_parts.append(jnp.swapaxes(v_bf16, 0, 1))
                v_all = (
                    v_parts[0]
                    if len(v_parts) == 1
                    else jnp.concatenate(v_parts, axis=0)
                )
                out_buf_3d = v_all.reshape(t_local * topk, he_dim // 128, 128)
                inv_perm_c = jnp.arange(t_local * topk, dtype=jnp.int32).reshape(
                    t_local, topk
                )
            else:
                go_l_eff = (
                    go_l.at[0].set(my_id * (num_local_experts + eff_cm_slots_pipe))
                    if eff_cm_pipe
                    else go_l
                )
                out_buf_3d = moe_gmm_local_rs_nodedup(
                    hidden_states,
                    w1_l,
                    w1s_l,
                    None,
                    w2_l,
                    w2s_l,
                    None,
                    None,
                    None,
                    go_l_eff,
                    local_topk_weights,
                    global_topk_ids,
                    pnw_l,
                    activation="silu",
                    topk=topk,
                    ep_size=ep_size,
                    ep_axis_name=expert_axis,
                    has_post_norm=True,
                    sp_enabled=True,
                    fp8_direct_write=False,
                    expert_tp=False,
                    fp4_w4a16=False,
                    local_t_recv=local_t_recv,
                    return_tiled=True,
                    block_scatter=True,
                    return_raw_out_buf=True,
                    precomputed_routing=precomputed_routing,
                    precomputed_chipmate_plan=pipe_cm_plan,
                    hidden_states_scale=hidden_states_scale,
                )
                if active_cnt_local is not None:
                    safe_rows = min(1024, out_buf_3d.shape[0])

                    def _zero_unwritten_head(b):
                        row_iota = jnp.arange(safe_rows, dtype=jnp.int32)[:, None, None]
                        return b.at[:safe_rows].set(
                            jnp.where(
                                row_iota >= active_cnt_local,
                                jnp.zeros((), dtype=b.dtype),
                                b[:safe_rows],
                            )
                        )

                    out_buf_3d = jax.lax.cond(
                        active_cnt_local < safe_rows,
                        _zero_unwritten_head,
                        lambda b: b,
                        out_buf_3d,
                    )

        with jax.named_scope("stage4_kernel_d_combine_unpool_norms"):
            d_opt_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_D_OPT", None)
            d_opt_val = int(d_opt_env.get()) if d_opt_env is not None else 0
            eff_emit_static_fp8 = d_emit_static_fp8_attn_in or (
                next_in_gather_static_fp8 and d_opt_val > 0
            )
            d_fn = (
                functools.partial(
                    pallas_combine_unpool_residual_nextnorm_4slab,
                    tok_tile=d_tok_tile,
                    fp8_unpool=d_fp8_unpool,
                    emit_fp8_attn_in=d_emit_fp8_attn_in,
                    emit_static_fp8_attn_in=eff_emit_static_fp8,
                )
                if d_layout == "4slab"
                else functools.partial(
                    pallas_combine_unpool_residual_nextnorm_nstream,
                    fp8_unpool=d_fp8_unpool,
                )
            )
            d_coal = bool(envs.SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER.get())
            d_ik_ag = bool(envs.SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG.get())
            w4_wu_qbs = int(envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get() or 0)
            has_w4_wu = w4_wu_qbs > 0 and jnp.dtype(wu_fp8.dtype) == jnp.float4_e2m1fn
            eff_w4_wu_qbs = w4_wu_qbs if has_w4_wu else 0
            if d_layout == "4slab" and (d_coal or d_ik_ag or has_w4_wu):
                from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
                    pallas_d_radical_4slab,
                )

                d_fn = functools.partial(
                    pallas_d_radical_4slab,
                    tok_tile=d_tok_tile,
                    fp8_unpool=d_fp8_unpool,
                    emit_fp8_attn_in=d_emit_fp8_attn_in,
                    emit_static_fp8_attn_in=d_emit_static_fp8_attn_in,
                    coal_gather=d_coal,
                    w4_unpool_qbs=eff_w4_wu_qbs,
                    wunpool_w4_scale=(wu_sc if has_w4_wu else None),
                )
            if (
                next_in_gather_rows
                and d_layout == "4slab"
                and d_ik_ag
                and 0 < next_in_gather_rows < t_local
                and next_in_gather_rows % d_tok_tile == 0
            ):
                new_stream, next_attn_in, next_in_gathered = d_fn(
                    out_buf_3d,
                    inv_perm_c,
                    local_topk_weights,
                    stream_shard_mid,
                    wu_fp8,
                    wu_sc,
                    post_ffn_w,
                    a_ffn,
                    b_ffn,
                    next_in_w,
                    emit_static_fp8_attn_in=(
                        d_emit_static_fp8_attn_in
                        or (next_in_gather_static_fp8 and causal_chunk_stream)
                    ),
                    in_kernel_ag_rows=next_in_gather_rows,
                    ag_axis_name=expert_axis,
                    ag_tp_deg=ep_size,
                    ag_part0_static_fp8=next_in_gather_static_fp8,
                )
            elif next_in_gather_rows:
                new_stream, next_attn_in, next_in_gathered = _d_with_next_in_gather(
                    d_fn=d_fn,
                    d_args=(
                        out_buf_3d,
                        inv_perm_c,
                        local_topk_weights,
                        stream_shard_mid,
                        wu_fp8,
                        wu_sc,
                        post_ffn_w,
                        a_ffn,
                        b_ffn,
                        next_in_w,
                    ),
                    rows0=next_in_gather_rows,
                    split=d_layout == "4slab" and next_in_gather_rows % d_tok_tile == 0,
                    axis_name=expert_axis,
                    gather_static_fp8=next_in_gather_static_fp8,
                )
            elif d_two_half:
                half_tok = t_local // 2
                ns_0, ni_0 = d_fn(
                    out_buf_3d,
                    inv_perm_c[:half_tok],
                    local_topk_weights[:half_tok],
                    stream_shard_mid[:half_tok],
                    wu_fp8,
                    wu_sc,
                    post_ffn_w,
                    a_ffn,
                    b_ffn,
                    next_in_w,
                )
                ns_0, ni_0 = jax.lax.optimization_barrier((ns_0, ni_0))
                ns_1, ni_1 = d_fn(
                    out_buf_3d,
                    inv_perm_c[half_tok:],
                    local_topk_weights[half_tok:],
                    stream_shard_mid[half_tok:],
                    wu_fp8,
                    wu_sc,
                    post_ffn_w,
                    a_ffn,
                    b_ffn,
                    next_in_w,
                )
                new_stream = jnp.concatenate([ns_0, ns_1], axis=0)
                next_attn_in = jnp.concatenate([ni_0, ni_1], axis=0)
            else:
                new_stream, next_attn_in = d_fn(
                    out_buf_3d,
                    inv_perm_c,
                    local_topk_weights,
                    stream_shard_mid,
                    wu_fp8,
                    wu_sc,
                    post_ffn_w,
                    a_ffn,
                    b_ffn,
                    next_in_w,
                )

        if next_in_gather_rows:
            return (
                new_stream,
                next_attn_in,
                raw_local_topk_ids,
                local_topk_weights,
                next_in_gathered,
            )
        return new_stream, next_attn_in, raw_local_topk_ids, local_topk_weights

    args = (
        attn_out_sharded,
        stream_sharded,
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        wrouter,
        e_score_bias,
        wpool_fp8,
        wpool_scale,
        pre_expert_norm_w,
        w1,
        w2,
        w1_scale,
        w2_scale,
        post_expert_norm_w,
        group_offset,
        wunpool_fp8,
        wunpool_scale,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )
    if in_shard_map:
        return _full_body(*args)
    assert not next_in_gather_rows, "next_in_gather_rows needs in_shard_map"
    base_in_specs = (
        P(expert_axis, None),  # attn_out_sharded
        P(expert_axis, None),  # stream_sharded
        P(),  # alpha_attn
        P(),  # beta_attn
        P(),  # pre_ffn_norm_w
        P(),  # wrouter
        P(),  # e_score_bias
        P(),  # wpool_fp8
        P(),  # wpool_scale
        P(),  # pre_expert_norm_w
        P(expert_axis),  # w1
        P(expert_axis),  # w2
        P(expert_axis),  # w1_scale
        P(expert_axis),  # w2_scale
        P(),  # post_expert_norm_w
        P(expert_axis),  # group_offset
        P(),  # wunpool_fp8
        P(),  # wunpool_scale
        P(),  # post_ffn_norm_w
        P(),  # alpha_ffn
        P(),  # beta_ffn
        P(),  # next_input_norm_w
    )
    if valid_tokens is not None:
        base_in_specs = base_in_specs + (P(),)
        args = args + (valid_tokens,)
    return jax.shard_map(
        _full_body,
        mesh=mesh,
        in_specs=base_in_specs,
        out_specs=(
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
        ),
        check_vma=False,
    )(*args)


def default_b1_b2_replicated_bf16(
    attn_out_sharded: jax.Array,
    stream_sharded: jax.Array,
    alpha_attn: jax.Array,
    beta_attn: jax.Array,
    pre_ffn_norm_w: jax.Array,
    wrouter: jax.Array,
    e_score_bias: jax.Array,
    wpool_col_sharded: jax.Array,
    wpool_scale_col_sharded: jax.Array,
    pre_expert_norm_w: jax.Array,
    group_offset: jax.Array,
    *,
    mesh: Mesh,
    num_local_experts: int,
    topk: int = 8,
    replicated_inputs: bool = False,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
    router_multiplier: float = 0.17677669529663687,
    route_scale: float = 1.0,
    route_eps: float = 1e-15,
):
    """True Default Replicated Muse Spark B1 + B2 + Stage 1 + Stage 2 Baseline (#50).

    Full-[T, 8192] replicated post_attention_layernorm + post_attention_residual_gate +
    full-[T, 8192] pre_feedforward_layernorm + full-[T, 8192] @ [8192, 256] wrouter +
    full-[T, 256] lax.top_k(8) + ColumnParallel pre_expert_proj ([T, 4096 // TP]) +
    Stage 1 (psum(local_sq) + pmax(local_max_w)) + Stage 2 (3D axis=1 column-sharded
    all_gather [T, 4096 // TP] -> [T, 4096] + dequant + global argsort).
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)

    def _body(
        attn_s,
        stream_s,
        a_attn,
        b_attn,
        pffn_w,
        w_r,
        bias,
        wp_col,
        wp_sc_col,
        pe_w,
        go_l,
    ):
        if replicated_inputs:
            attn_rep = attn_s
            stream_rep = stream_s
        else:
            attn_rep = jax.lax.all_gather(
                attn_s, axis_name=expert_axis, axis=0, tiled=True
            )
            stream_rep = jax.lax.all_gather(
                stream_s, axis_name=expert_axis, axis=0, tiled=True
            )

        a_f32 = attn_rep.astype(jnp.float32)
        s_a = jax.lax.rsqrt(
            jnp.mean(a_f32 * a_f32, axis=-1, keepdims=True) + post_norm_eps
        )
        branch_attn = (a_f32 * s_a).astype(jnp.bfloat16).astype(jnp.float32)
        stream_mid_rep = a_attn[None, :] * stream_rep + b_attn[None, :] * branch_attn

        s_m = jax.lax.rsqrt(
            jnp.mean(stream_mid_rep * stream_mid_rep, axis=-1, keepdims=True) + rms_eps
        )
        router_in_rep = (
            stream_mid_rep * s_m * (pffn_w.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)

        logits_rep = jnp.dot(router_in_rep.astype(jnp.float32), w_r.astype(jnp.float32))
        sc_rep = jax.nn.sigmoid(logits_rep * router_multiplier)
        biased_rep = sc_rep + bias[None, :]
        _, topk_ids_rep = jax.lax.top_k(biased_rep, topk)
        topk_w_rep = jnp.take_along_axis(sc_rep, topk_ids_rep, axis=-1)
        topk_w_rep = (
            topk_w_rep
            / (jnp.sum(topk_w_rep, axis=-1, keepdims=True) + route_eps)
            * route_scale
        ).astype(jnp.bfloat16)

        # ColumnParallel pre_expert_proj [T, 4096 // TP]
        pre_proj_col = (
            jnp.dot(router_in_rep, wp_col, preferred_element_type=jnp.float32)
            * wp_sc_col.astype(jnp.float32)[None, :]
        ).astype(jnp.bfloat16)

        # Stage 1: Distributed pre_expert_norm via psum(local_sq) + pmax(local_max_w)
        he_local = pre_proj_col.shape[1]
        he_full = he_local * ep_size
        my_id = go_l[0] // num_local_experts
        pe_w_local = jax.lax.dynamic_slice_in_dim(
            pe_w, my_id * he_local, he_local, axis=0
        )
        x_f32 = pre_proj_col.astype(jnp.float32)
        local_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        global_sq = jax.lax.psum(local_sq, axis_name=expert_axis)
        inv_rms = jax.lax.rsqrt(global_sq * (1.0 / he_full) + rms_eps)
        x_norm_col = (
            x_f32 * inv_rms * (pe_w_local.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)

        local_amax = jnp.max(
            jnp.abs(x_norm_col.astype(jnp.float32)), axis=-1, keepdims=True
        )
        global_amax = jax.lax.pmax(local_amax, axis_name=expert_axis)
        row_sc = (global_amax * (1.0 / 127.0)).astype(jnp.bfloat16)
        inv_sc = jnp.where(
            global_amax == 0, 0.0, 127.0 / jnp.maximum(global_amax, 1e-12)
        ).astype(jnp.bfloat16)
        x_i8_col = jnp.rint(x_norm_col * inv_sc).astype(jnp.int8)

        # Stage 2: 3D axis=1 column-sharded all_gather [T, 4096 // TP] -> [T, 4096] + dequant + global argsort
        x_i8_full = jax.lax.all_gather(
            x_i8_col, axis_name=expert_axis, axis=1, tiled=True
        )
        hidden_states_3d = (x_i8_full.astype(jnp.bfloat16) * row_sc).reshape(
            router_in_rep.shape[0], he_full // 128, 128
        )
        t_local = router_in_rep.shape[0] // ep_size
        local_topk_ids = jax.lax.dynamic_slice_in_dim(
            topk_ids_rep, my_id * t_local, t_local, axis=0
        )
        perm_c = jnp.argsort(local_topk_ids.reshape(-1), stable=True)
        inv_perm_c = jnp.argsort(perm_c).reshape(t_local, topk).astype(jnp.int32)
        lhs_indices, group_sizes, group_start_arg, _ = compute_block_scatter_routing(
            topk_ids_rep.astype(jnp.int32),
            go_l[0],
            my_id,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            topk=topk,
        )
        stream_mid_local = jax.lax.dynamic_slice_in_dim(
            stream_mid_rep, my_id * t_local, t_local, axis=0
        )
        local_topk_w = jax.lax.dynamic_slice_in_dim(
            topk_w_rep, my_id * t_local, t_local, axis=0
        )
        return (
            hidden_states_3d,
            stream_mid_local,
            local_topk_w,
            inv_perm_c,
            lhs_indices,
            group_sizes,
            group_start_arg,
        )

    in_spec_tok = P() if replicated_inputs else P(expert_axis, None)
    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(
            in_spec_tok,
            in_spec_tok,
            P(),
            P(),
            P(),
            P(),
            P(),
            P(None, expert_axis),
            P(expert_axis),
            P(),
            P(expert_axis),
        ),
        out_specs=(
            P(),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )(
        attn_out_sharded,
        stream_sharded,
        alpha_attn,
        beta_attn,
        pre_ffn_norm_w,
        wrouter,
        e_score_bias,
        wpool_col_sharded,
        wpool_scale_col_sharded,
        pre_expert_norm_w,
        group_offset,
    )


def default_d_replicated_bf16(
    out_buf_3d_sharded: jax.Array,
    inv_perm_c_sharded: jax.Array,
    topk_weights_sharded: jax.Array,
    stream_mid_sharded: jax.Array,
    wunpool_col_sharded: jax.Array,
    wunpool_scale_col_sharded: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    mesh: Mesh,
    replicated_stream_mid: bool = False,
    post_norm_eps: float = 1e-8,
    rms_eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """True Default Replicated Muse Spark D Baseline (#50).

    Stage 3 (_tc_combine_bf16 + INT8 quant + 2-half Output All-Gather [T // TP, 4096] -> [T, 4096] + dequant) +
    ColumnParallel post_expert_proj ([T, 8192 // TP]) + full [T, 8192] All-Gather (128 MiB at 8K, 256 MiB at 16K) +
    full-[T, 8192] replicated post_feedforward_layernorm (scale_input=True) +
    full-[T, 8192] post_feedforward_residual_gate + next layer's full-[T, 8192] input_layernorm.
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)

    def _body(
        out_3d,
        inv_p,
        tw,
        stream_mid,
        wu_col,
        wu_sc_col,
        post_ffn_w,
        a_ffn,
        b_ffn,
        next_in_w,
    ):
        # Stage 3: _tc_combine + INT8 Output All-Gather [T // TP, 4096] -> [T, 4096]
        u_local = _tc_combine_bf16_4chunk(out_3d, inv_p, tw)
        u_amax = jnp.max(jnp.abs(u_local.astype(jnp.float32)), axis=-1, keepdims=True)
        u_sc = (u_amax * (1.0 / 127.0)).astype(jnp.bfloat16)
        u_inv_sc = jnp.where(
            u_amax == 0, 0.0, 127.0 / jnp.maximum(u_amax, 1e-12)
        ).astype(jnp.bfloat16)
        u_i8 = jnp.rint(u_local * u_inv_sc).astype(jnp.int8)
        u_i8_rep = jax.lax.all_gather(u_i8, axis_name=expert_axis, axis=0, tiled=True)
        u_sc_rep = jax.lax.all_gather(u_sc, axis_name=expert_axis, axis=0, tiled=True)
        u_rep = u_i8_rep.astype(jnp.bfloat16) * u_sc_rep

        # ColumnParallel post_expert_proj [T, 8192 // TP] + full [T, 8192] All-Gather (128/256 MiB)
        y_col = (
            jnp.dot(u_rep, wu_col, preferred_element_type=jnp.float32)
            * wu_sc_col.astype(jnp.float32)[None, :]
        ).astype(jnp.bfloat16)
        y_rep = jax.lax.all_gather(
            y_col, axis_name=expert_axis, axis=1, tiled=True
        ).astype(jnp.float32)

        # Full-[T, 8192] replicated post_feedforward_layernorm + residual + next input_layernorm
        stream_mid_rep = (
            stream_mid
            if replicated_stream_mid
            else jax.lax.all_gather(
                stream_mid, axis_name=expert_axis, axis=0, tiled=True
            )
        )
        fw_rep = y_rep * (post_ffn_w.astype(jnp.float32) + 1.0)[None, :]
        s_ffn = jax.lax.rsqrt(
            jnp.mean(fw_rep * fw_rep, axis=-1, keepdims=True) + post_norm_eps
        )
        branch_ffn = (fw_rep * s_ffn).astype(jnp.bfloat16).astype(jnp.float32)
        new_stream_rep = a_ffn[None, :] * stream_mid_rep + b_ffn[None, :] * branch_ffn
        s_next = jax.lax.rsqrt(
            jnp.mean(new_stream_rep * new_stream_rep, axis=-1, keepdims=True) + rms_eps
        )
        next_in_rep = (
            new_stream_rep * s_next * (next_in_w.astype(jnp.float32) + 1.0)[None, :]
        ).astype(jnp.bfloat16)

        my_id = jax.lax.axis_index(expert_axis)
        t_local = inv_p.shape[0]
        new_stream_loc = jax.lax.dynamic_slice_in_dim(
            new_stream_rep, my_id * t_local, t_local, axis=0
        )
        next_in_loc = jax.lax.dynamic_slice_in_dim(
            next_in_rep, my_id * t_local, t_local, axis=0
        )
        return new_stream_loc, next_in_loc

    smid_spec = P() if replicated_stream_mid else P(expert_axis, None)
    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(
            P(expert_axis, None, None),
            P(expert_axis, None),
            P(expert_axis, None),
            smid_spec,
            P(None, expert_axis),
            P(expert_axis),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(expert_axis, None), P(expert_axis, None)),
        check_vma=False,
    )(
        out_buf_3d_sharded,
        inv_perm_c_sharded,
        topk_weights_sharded,
        stream_mid_sharded,
        wunpool_col_sharded,
        wunpool_scale_col_sharded,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )


def fused_d_standalone(
    out_buf_3d_sharded: jax.Array,
    inv_perm_c_sharded: jax.Array,
    topk_weights_sharded: jax.Array,
    stream_mid_sharded: jax.Array,
    wunpool_fp8: jax.Array,
    wunpool_scale: jax.Array,
    post_ffn_norm_w: jax.Array,
    alpha_ffn: jax.Array,
    beta_ffn: jax.Array,
    next_input_norm_w: jax.Array,
    *,
    mesh: Mesh,
    d_layout: str = "4slab",
    d_tok_tile: int = 64,
    d_fp8_unpool: bool = False,
    d_emit_fp8_attn_in: bool = False,
    d_emit_static_fp8_attn_in: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Standalone benchmark for fused Kernel D (Zero Stage 3 Output All-Gather!)."""
    expert_axis = get_moe_expert_axis(mesh, EXPERT)

    def _body(
        out_3d,
        inv_p,
        tw,
        stream_mid,
        wu_fp8,
        wu_sc,
        post_ffn_w,
        a_ffn,
        b_ffn,
        next_in_w,
    ):
        if d_layout == "4slab":
            return pallas_combine_unpool_residual_nextnorm_4slab(
                out_3d,
                inv_p,
                tw,
                stream_mid,
                wu_fp8,
                wu_sc,
                post_ffn_w,
                a_ffn,
                b_ffn,
                next_in_w,
                tok_tile=d_tok_tile,
                fp8_unpool=d_fp8_unpool,
                emit_fp8_attn_in=d_emit_fp8_attn_in,
                emit_static_fp8_attn_in=d_emit_static_fp8_attn_in,
            )
        return pallas_combine_unpool_residual_nextnorm_nstream(
            out_3d,
            inv_p,
            tw,
            stream_mid,
            wu_fp8,
            wu_sc,
            post_ffn_w,
            a_ffn,
            b_ffn,
            next_in_w,
            fp8_unpool=d_fp8_unpool,
        )

    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(
            P(expert_axis, None, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(expert_axis, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(expert_axis, None), P(expert_axis, None)),
        check_vma=False,
    )(
        out_buf_3d_sharded,
        inv_perm_c_sharded,
        topk_weights_sharded,
        stream_mid_sharded,
        wunpool_fp8,
        wunpool_scale,
        post_ffn_norm_w,
        alpha_ffn,
        beta_ffn,
        next_input_norm_w,
    )
