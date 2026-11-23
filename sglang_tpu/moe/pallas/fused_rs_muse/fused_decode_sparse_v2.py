# SPDX-License-Identifier: Apache-2.0
"""Exploratory v2 Sparse & Dense-Lowtok Fuse-All Decode MoE Kernel for Muse Spark.

Advances ``fused_decode_sparse.py`` with four architectural mechanisms:
  1. True ``moe_lowtok``-style homogeneous circular VMEM weight ring
     (``ring3_proj``, ``ring3_dual``, ``ring4_proj``, ``ring4_dual``):
     streams uniform 8-MiB ``(512, 4096)`` units (Gate/Up or Dual-Half0 /
     Dual-Half1 / unstrided full W2) in strict FIFO unit order with zero
     division/modulo SALU overhead, starting compute after a single 8-MiB
     cold-start and issuing prefetches OUTSIDE ``lax.switch`` before waits.
  2. Flattened SMEM tile schedule + double-buffered async VMEM->VMEM token
     gather (``async_gather=True``) + single-pass fused ``inv_rms`` BF16
     normalization & weighted scatter-accumulate (``fused_norm_scatter=True``),
     eliminating inter-expert VPU gather stalls and ``io_3d_buf`` roundtrips.
  3. Pipelined W4A16 unpack & multi-row weight reuse:
     - Interleaved ``wg`` unpack -> MXU push -> ``wu`` unpack -> MXU push
       halving peak live unpacked-weight VREG pressure (0 LLO spills).
     - Selectable ``fp4_unpack_mode`` (``"via_fp8"`` native 2-op v7x hardware
       path, ``"lut_u4"`` branchless IEEE-754 nibble unpack, or ``"bf16"``)
       plus ``kc_step in (128, 256)``.
     - Eliminates ``loop32_in_tile`` redundant K16 FP4 re-dequantization via
       either unlooped ``m=64``/``128`` systolic passes or ``vmem_deq_reuse``
       (unpacking each strip once into VMEM and reusing across all 32-row slices).
     - Optional lossless ``float8_e4m3fn`` K16 block scales + per-expert ``f32``
       global scale (``w1_gs``, ``w2_gs``), cutting K16 scale HBM DMA by 50%.
  4. Direct peer-VMEM ``Coll`` collectives ported from ``muse_megakernel``:
     - ``pair_rs=True``: 2-stage intra-chip paired Reduce-Scatter over the
       733 GB/s die-to-die bridge, halving off-chip ICI Reduce-Scatter bytes.
     - ``pair_ag_relay=True``: per-source-block semaphore All-Gather relaying
       1-hop remote blocks across the intra-chip bridge immediately on arrival.
"""

# PEP 563 stays off so TorchTPU / cpu_export_jax_op sees concrete annotations.

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
    DecodeRoutingPruneConfig,
    read_sparse_v2_env_overrides,
    route_dense_and_counts_v2_in_kernel,
    route_topk_and_counts_v2_in_kernel,
    scale_free_expert_for_batch,
    unpack_w3_u32_chunk_pallas,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
    DenseRoutingScratch,
    RoutingScratch,
    build_default_exec_plan_smem,
    build_dense_exec_plan_smem,
    build_exec_plan_from_masks_smem,
    dense_routing_scratch_shapes,
    extract_my_holder_candidates_smem,
    extract_my_holder_candidates_tuple,
    pack_active_tokens_smem,
    route_dense_and_counts_in_kernel,
    route_topk_and_counts_in_kernel,
    routing_scratch_shapes,
    run_replica_planner_smem,
    select_non_replicated_home_slot,
)

_FP4_PACK = 8
_FP4_GROUP = 16
_H = 4096
_I = 4096
_N1 = 8192
_E_TOTAL = 256
_TOPK = 8
_VALID_RING_MODES = (
    "ring3_proj",
    "ring3_dual",
    "ring4_proj",
    "ring4_dual",
    "ring6_proj",
    "ring6_dual",
    "ring3_khalf",
    "ring4_khalf",
)
_VALID_UNPACK_MODES = ("via_fp8", "lut_u4", "bf16")


@dataclasses.dataclass(frozen=True)
class SparseDecodeV2Config:
    """Static compile-time configuration for ``fused_decode_sparse_v2_from_logits``."""

    row_mode: str = "sparse"  # "sparse", "dense", or "mxu_sparse"
    # "ring3_proj", "ring3_dual", "ring4_proj", "ring4_dual", "ring6_proj",
    # "ring6_dual", "ring3_khalf", "ring4_khalf"
    ring_mode: str = "ring3_dual"
    tile_m: int = 32  # 16, 32, 64, or 128
    m_buckets: tuple[int, ...] = (16, 32)
    # "none" (unlooped m_buckets), "loop32" (baseline re-unpack), "vmem_deq" (unpack once to VMEM!)
    multi_m32_mode: str = "none"
    k16_blockdiag_rows: tuple[int, ...] = (1, 2, 4, 8)
    k16_bd_narrow_nc: bool = False
    # "via_fp8" (f4->f8e4m3->bf16), "lut_u4" (uint4 IEEE754), or "bf16"
    fp4_unpack_mode: str = "via_fp8"
    kc_step: int = 256  # 128 or 256 contraction step
    kperm_k16: bool = False  # True: weights/x in kperm(4096) order -> zero repeat(s,16) shuffle!
    lhs_fp8: bool = False  # True: online K256 W4A8/W3A8 FP8 activation quant (2x MXU)
    k16_strip_nc: int = 512
    k16_dual_mode: str = "hybrid"  # "dual", "vstage", "hybrid" (dual at m<=32, vstage at m>=64)
    k256_strip_mode: str = "adaptive"  # "adaptive" (1024 at m<=16, 512 at m>=32: 0 expert spills!), "wide"
    k256_dual_mode: str = "hybrid"  # "dual", "vstage", "hybrid" (dual at m<=16, vstage at m>=32)
    async_gather: bool = True  # double-buffered async DMA token gather overlapped w/ W2
    fused_norm_scatter: bool = True  # single-pass y_stage -> inv_rms -> accum_3d scatter
    rs_wire_bf16: bool = False
    pair_rs: bool = False  # 2-stage chipmate-paired Reduce-Scatter (halves RS ICI!)
    pair_ag_relay: bool = False  # per-block relayed 2-stage All-Gather (Coll.pair_all_gather)
    norm_f32_stage: bool = True
    route_chunk_rows: int = 32
    static_holder_mask: bool = False
    spec_prime_parts: int = 0
    spec_skip_replicated: bool = True
    hoist_scales: bool = False
    contig_parts_mode: int = 0  # 0=std, 1=khalf, 2=part_major_proj, 3=part_major_dual
    routing_prune: DecodeRoutingPruneConfig = DecodeRoutingPruneConfig()


def _apply_v2_env_overrides(
    base: SparseDecodeV2Config, batch_size: int, ep: int, scale_block: int
) -> SparseDecodeV2Config:
    ov = read_sparse_v2_env_overrides()
    ring_mode = base.ring_mode
    contig_mode = base.contig_parts_mode
    if ov.contig_parts == 1:
        contig_mode = 1
        if not ov.ring_mode:
            ring_mode = "ring4_khalf"
    elif ov.contig_parts == 2:
        contig_mode = 2
        if not ov.ring_mode:
            ring_mode = "ring4_proj"
    elif ov.contig_parts == 3:
        contig_mode = 3
        if not ov.ring_mode:
            ring_mode = "ring4_dual"

    if ov.ring_mode:
        if ov.ring_mode not in _VALID_RING_MODES:
            raise ValueError(
                f"Invalid SGLANG_TPU_MUSE_KERNEL_DECODE_V2_RING={ov.ring_mode!r}; "
                f"expected one of {_VALID_RING_MODES}"
            )
        ring_mode = ov.ring_mode
        if ring_mode.endswith("_khalf") and scale_block > 256:
            ring_mode = ring_mode.replace("_khalf", "_proj")
        elif ring_mode.endswith("_khalf") and contig_mode == 0:
            contig_mode = 1

    unpack_mode = base.fp4_unpack_mode
    if ov.fp4_unpack_mode:
        if ov.fp4_unpack_mode not in _VALID_UNPACK_MODES:
            raise ValueError(
                f"Invalid SGLANG_TPU_MUSE_KERNEL_DECODE_FP4_UNPACK={ov.fp4_unpack_mode!r}; "
                f"expected one of {_VALID_UNPACK_MODES}"
            )
        unpack_mode = ov.fp4_unpack_mode

    m_buckets = base.m_buckets
    tile_m = base.tile_m
    row_mode = base.row_mode
    if ov.m_buckets:
        if row_mode == "dense" and batch_size > max(ov.m_buckets):
            row_mode = "sparse"
        if row_mode != "dense":
            m_buckets = ov.m_buckets
            tile_m = max(32, max(m_buckets))
    elif ov.m8 and row_mode != "dense" and batch_size <= 64:
        m_buckets = (8, 32)
        tile_m = max(tile_m, 32)

    multi_m32 = base.multi_m32_mode
    if ov.vmem_deq == 1:
        multi_m32 = "vmem_deq"
        if row_mode != "dense" and batch_size >= 64 and tile_m <= 32:
            tile_m = 64
            if 32 not in m_buckets:
                m_buckets = tuple(sorted(set(m_buckets) | {32}))
    elif ov.vmem_deq == 2:
        multi_m32 = "loop32"
        if row_mode != "dense" and batch_size >= 64 and tile_m <= 32:
            tile_m = 64
            if 32 not in m_buckets:
                m_buckets = tuple(sorted(set(m_buckets) | {32}))

    pair_rs = base.pair_rs if ov.pair_rs < 0 else (bool(ov.pair_rs) and ep in (8, 16))
    async_gather = (
        base.async_gather if ov.async_gather < 0 else bool(ov.async_gather)
    )
    fused_ns = (
        base.fused_norm_scatter
        if ov.fused_norm_scatter < 0
        else bool(ov.fused_norm_scatter)
    )
    rs_bf16 = base.rs_wire_bf16 or (
        ov.rs_bridge64 and ep in (8, 16) and batch_size == 64
    )
    spec_eff = ov.spec_prime_parts if batch_size <= 128 else 0
    lhs_fp8 = base.lhs_fp8 or (ov.lhs_fp8 and scale_block >= 256)
    prune = ov.routing_prune
    if prune.free_expert_rows > 0:
        f_rows_eff, f_rank_eff = scale_free_expert_for_batch(
            prune.free_expert_rows, prune.free_min_rank, batch_size
        )
        prune = dataclasses.replace(
            prune,
            free_expert_rows=f_rows_eff,
            free_min_rank=max(1, min(7, f_rank_eff)),
        )

    return dataclasses.replace(
        base,
        row_mode=row_mode,
        ring_mode=ring_mode,
        tile_m=tile_m,
        m_buckets=m_buckets,
        multi_m32_mode=multi_m32,
        fp4_unpack_mode=unpack_mode,
        lhs_fp8=lhs_fp8,
        async_gather=async_gather,
        fused_norm_scatter=fused_ns,
        rs_wire_bf16=rs_bf16,
        pair_rs=pair_rs,
        static_holder_mask=ov.static_holder_mask,
        spec_prime_parts=spec_eff,
        hoist_scales=ov.hoist_scales,
        contig_parts_mode=contig_mode,
        routing_prune=prune,
    )


def default_sparse_decode_v2_config(
    batch_size: int, ep: int, scale_block: int
) -> SparseDecodeV2Config:
    """Return the tuned default v2 configuration for ``(batch_size, ep, scale_block)``."""
    use_paired_ep = ep in (8, 16)
    if scale_block == 16:
        if batch_size <= 64:
            base = SparseDecodeV2Config(
                row_mode="sparse",
                ring_mode="ring3_dual",
                tile_m=32,
                m_buckets=(16, 32),
                multi_m32_mode="none",
                k16_blockdiag_rows=(1, 2, 4, 8),
                fp4_unpack_mode="bf16",
                kc_step=256,
                k16_strip_nc=512,
                k16_dual_mode="dual",
                async_gather=False,
                fused_norm_scatter=True,
                rs_wire_bf16=(batch_size >= 64),
                pair_rs=(use_paired_ep and batch_size >= 32),
                pair_ag_relay=(use_paired_ep and batch_size >= 32),
                route_chunk_rows=min(batch_size, 32),
            )
            return _apply_v2_env_overrides(base, batch_size, ep, scale_block)
        # B >= 128: unloop64 + ring4_dual + via_fp8 + pair_rs + pair_ag_relay
        base = SparseDecodeV2Config(
            row_mode="sparse",
            ring_mode="ring3_dual" if batch_size > 256 else "ring4_dual",
            tile_m=64,
            m_buckets=(16, 32, 64),
            multi_m32_mode="none",
            k16_blockdiag_rows=(1, 2, 4, 8),
            fp4_unpack_mode="via_fp8",
            kc_step=256,
            k16_strip_nc=512,
            k16_dual_mode="hybrid",
            async_gather=False,
            fused_norm_scatter=True,
            rs_wire_bf16=True,
            pair_rs=use_paired_ep,
            pair_ag_relay=use_paired_ep,
            route_chunk_rows=32,
        )
        return _apply_v2_env_overrides(base, batch_size, ep, scale_block)

    # Compact K256 / K512
    if batch_size <= 16:
        base = SparseDecodeV2Config(
            row_mode="dense",
            ring_mode="ring3_proj",
            tile_m=batch_size,
            m_buckets=(batch_size,),
            multi_m32_mode="none",
            k16_blockdiag_rows=(),
            fp4_unpack_mode="via_fp8",
            kc_step=256,
            k256_strip_mode="adaptive",
            k256_dual_mode="vstage",
            async_gather=False,
            fused_norm_scatter=True,
            rs_wire_bf16=False,
            pair_rs=False,
            pair_ag_relay=False,
            route_chunk_rows=batch_size,
        )
        return _apply_v2_env_overrides(base, batch_size, ep, scale_block)
    if batch_size <= 64:
        base = SparseDecodeV2Config(
            row_mode="sparse",
            ring_mode="ring3_proj",
            tile_m=32,
            m_buckets=(16, 32),
            multi_m32_mode="none",
            k16_blockdiag_rows=(),
            fp4_unpack_mode="via_fp8",
            kc_step=256,
            k256_strip_mode="adaptive",
            k256_dual_mode="hybrid",
            async_gather=False,
            fused_norm_scatter=True,
            rs_wire_bf16=(batch_size >= 64),
            pair_rs=(ep == 16 and batch_size >= 32) or (ep == 8 and batch_size >= 64),
            pair_ag_relay=(ep == 16 and batch_size >= 32) or (ep == 8 and batch_size >= 64),
            route_chunk_rows=32,
        )
        return _apply_v2_env_overrides(base, batch_size, ep, scale_block)
    base = SparseDecodeV2Config(
        row_mode="sparse",
        ring_mode="ring4_proj",
        tile_m=64,
        m_buckets=(16, 32, 64),
        multi_m32_mode="none",
        k16_blockdiag_rows=(),
        fp4_unpack_mode="via_fp8",
        kc_step=256,
        k256_strip_mode="adaptive",
        k256_dual_mode="hybrid",
        async_gather=False,
        fused_norm_scatter=True,
        rs_wire_bf16=True,
        pair_rs=use_paired_ep,
        pair_ag_relay=use_paired_ep,
        route_chunk_rows=32,
    )
    return _apply_v2_env_overrides(base, batch_size, ep, scale_block)


def _pad_rows_to(arr: jax.Array, rows: int) -> jax.Array:
    if arr.shape[0] == rows:
        return arr
    return jnp.concatenate(
        [arr, jnp.zeros((rows - arr.shape[0], arr.shape[1]), arr.dtype)], axis=0
    )


def _pick_k256_nc(m_rows: int, max_n: int, strip_mode: str) -> int:
    if strip_mode == "fixed256":
        return min(256, max_n)
    if strip_mode == "fixed512":
        return min(512, max_n)
    if strip_mode == "fixed1024":
        return min(1024, max_n)
    if strip_mode == "fixed2048":
        return min(2048, max_n)
    if strip_mode == "wide":
        nc = 2048 if m_rows <= 16 else (1024 if m_rows <= 64 else 512)
        return min(nc, max_n)
    nc = 1024 if m_rows <= 16 else 512
    return min(nc, max_n)


def _pick_k16_nc(m_rows: int, max_n: int, base_nc: int, use_dual: bool) -> int:
    if use_dual:
        # Keep 2 * [m_rows, nc] f32 accumulators <= 32 VREGs -> 0 spills!
        cap = 512 if m_rows <= 32 else (256 if m_rows <= 64 else 128)
        return min(base_nc, cap, max_n)
    # Single projection (vstage): [m_rows, nc] f32 <= 32 VREGs -> 512 at m<=64, 256 at m=128.
    cap = 512 if m_rows <= 64 else 256
    return min(base_nc, cap, max_n)


def _pick_bd_nc(bd_rows: int, max_n: int, narrow: bool = False) -> int:
    if narrow:
        if bd_rows <= 2:
            return min(512, max_n)
        if bd_rows <= 4:
            return min(256, max_n)
        return min(128, max_n)
    if bd_rows == 1:
        return min(1024, max_n)
    if bd_rows <= 4:
        return min(512, max_n)
    if bd_rows <= 8:
        return min(256, max_n)
    return min(128, max_n)


def _unpack_fp4_to_bf16(words: jax.Array, mode: str) -> jax.Array:
    """Unpack packed ``uint32`` NVFP4 words ``[K//8, N]`` to ``[K, N]`` ``bfloat16``."""
    if mode == "lut_u4":
        u16 = pltpu.bitcast(words, jnp.uint4).astype(jnp.uint16)
        sign = (u16 & jnp.uint16(8)) << jnp.uint16(12)
        mag = u16 & jnp.uint16(7)
        eff = jnp.where(mag == jnp.uint16(1), jnp.uint16(0), mag)
        bf_bits = jnp.where(
            mag == jnp.uint16(0),
            sign,
            sign | (jnp.uint16(0x3F00) + (eff << jnp.uint16(6))),
        )
        return pltpu.bitcast(bf_bits, jnp.bfloat16)
    f4 = pltpu.bitcast(words, jnp.float4_e2m1fn)
    if mode == "via_fp8":
        return f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
    return f4.astype(jnp.bfloat16)


def _load_scale_tile(
    s_ref, g0: int, gq: int, n0: jax.Array | int, nc: int, out_dtype
) -> jax.Array:
    """Load ``[gq, nc]`` scales from ``s_ref``, handling ``float8_e4m3fn`` 32-row tile alignment."""
    if s_ref.dtype != jnp.float8_e4m3fn or (gq % 32 == 0 and g0 % 32 == 0):
        s = s_ref[pl.ds(g0, gq), pl.ds(n0, nc)]
        return s if s.dtype == out_dtype else s.astype(out_dtype)
    base32 = (g0 // 32) * 32
    off = g0 - base32
    blk = s_ref[pl.ds(base32, 32), pl.ds(n0, nc)].astype(out_dtype)
    return blk[off : off + gq, :]


def _dequant_k16_tile(
    w_ref,
    s_ref,
    c0: int,
    kc: int,
    n0: jax.Array | int,
    nc: int,
    unpack_mode: str,
    kperm_order: bool = False,
    *,
    is_w3: bool = False,
) -> jax.Array:
    """Unpack & scale one ``[kc, nc]`` K16 weight block into ``bfloat16``."""
    gpc = kc // _FP4_GROUP
    if is_w3:
        w_f4 = unpack_w3_u32_chunk_pallas(w_ref, c0, n0, nc)
        w_bf = (
            w_f4.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
            if unpack_mode == "via_fp8"
            else w_f4.astype(jnp.bfloat16)
        )
    else:
        words = w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)]
        w_bf = _unpack_fp4_to_bf16(words, unpack_mode)
    s_bf = _load_scale_tile(s_ref, c0 // _FP4_GROUP, gpc, n0, nc, jnp.bfloat16)
    if kperm_order and kc == 256:
        # In kperm(256) order, row r*16 + g belongs to group g: broadcast-free VREG mul!
        return (w_bf.reshape(_FP4_GROUP, gpc, nc) * s_bf[None, :, :]).reshape(
            kc, nc
        )
    return w_bf * jnp.repeat(s_bf, _FP4_GROUP, axis=0)


def _prequant_lhs_k256(x_src_ref, lhs_fp8_ref, lhs_sc_ref, m_rows: int) -> None:
    """Online-quantize ``x_src_ref[0:m_rows, 4096]`` per K256 block into FP8 + scale."""
    kc = 256
    read_m = 16 if m_rows == 8 else m_rows
    fp8_m = max(32, read_m)
    for b_id in range(_H // kc):
        c0 = b_id * kc
        xf = x_src_ref[pl.ds(0, read_m), pl.ds(c0, kc)].astype(jnp.float32)
        amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
        sc = jnp.maximum(amax, 1e-12) * (1.0 / 448.0)
        xq = (xf / sc).astype(jnp.float8_e4m3fn)
        lhs_fp8_ref[pl.ds(0, fp8_m), pl.ds(c0, kc)] = _pad_rows_to(xq, fp8_m)
        lhs_sc_ref[pl.ds(0, read_m), pl.ds(b_id * 128, 128)] = jnp.broadcast_to(
            sc, (read_m, 128)
        )


def _fp4_gmm_chunked_v2(
    x_2d_ref,
    w_ref,
    s_ref,
    m_rows: int,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    block256: bool,
    scale_block: int,
    *,
    kc_step: int = 256,
    unpack_mode: str = "via_fp8",
    kperm_order: bool = False,
    lhs_sc_ref=None,
    gs_scale: jax.Array | None = None,
    hoist_scales: bool = False,
    is_w3: bool = False,
    x_col_base: int = 0,
    b_offset: int = 0,
) -> jax.Array:
    """Single-projection pipelined FP4/W3 matmul over ``kc_step`` chunks."""
    eff_kc = 256 if (block256 or kperm_order or is_w3) else kc_step
    read_m = 16 if m_rows == 8 else m_rows
    use_w4a8 = block256 and (lhs_sc_ref is not None)
    if block256 and not use_w4a8:
        n_k_blocks = k_dim // scale_block
        sub_nc = min(nc, 512)
        n_sub = nc // sub_nc
        can_hoist = hoist_scales and n_k_blocks >= 8 and n_k_blocks % 8 == 0
        s_subs = [
            _load_scale_tile(
                s_ref, 0, n_k_blocks, n0 + si * sub_nc, sub_nc, jnp.float32
            )
            if can_hoist
            else None
            for si in range(n_sub)
        ]
        accs: list[jax.Array | None] = [None] * n_sub
        chunks_per_sb = scale_block // eff_kc
        for sb_id in range(n_k_blocks):
            c_base = sb_id * scale_block
            blk_accs: list[jax.Array | None] = [None] * n_sub
            for ci in range(chunks_per_sb):
                c0 = c_base + ci * eff_kc
                x = x_2d_ref[pl.ds(0, read_m), pl.ds(x_col_base + c0, eff_kc)]
                if read_m != m_rows:
                    x = x[:m_rows, :]
                for si in range(n_sub):
                    col0 = n0 + si * sub_nc
                    if is_w3:
                        w_fp8 = unpack_w3_u32_chunk_pallas(
                            w_ref, c0, col0, sub_nc
                        ).astype(jnp.float8_e4m3fn)
                    else:
                        w_fp8 = pltpu.bitcast(
                            w_ref[
                                pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK),
                                pl.ds(col0, sub_nc),
                            ],
                            jnp.float4_e2m1fn,
                        ).astype(jnp.float8_e4m3fn)
                    part = jnp.dot(
                        x, w_fp8, preferred_element_type=jnp.float32
                    )
                    blk_accs[si] = (
                        part if blk_accs[si] is None else blk_accs[si] + part
                    )
            for si in range(n_sub):
                col0 = n0 + si * sub_nc
                s = (
                    s_subs[si][sb_id : sb_id + 1, :]
                    if s_subs[si] is not None
                    else _load_scale_tile(
                        s_ref, sb_id, 1, col0, sub_nc, jnp.float32
                    )
                )
                scaled = blk_accs[si] * s
                accs[si] = scaled if accs[si] is None else accs[si] + scaled
        acc = accs[0] if n_sub == 1 else jnp.concatenate(accs, axis=1)
        if gs_scale is not None:
            acc = acc * gs_scale
        return acc
    x_read_m = max(32, read_m) if use_w4a8 else read_m
    acc = None
    s_blk32 = None
    n_k_blocks = k_dim // scale_block if block256 else 0
    s_hoisted = (
        _load_scale_tile(s_ref, 0, n_k_blocks, n0, nc, jnp.float32)
        if (block256 and hoist_scales and n_k_blocks >= 8 and n_k_blocks % 8 == 0)
        else None
    )
    for b_id, c0 in enumerate(range(0, k_dim, eff_kc)):
        x = x_2d_ref[pl.ds(0, x_read_m), pl.ds(x_col_base + c0, eff_kc)]
        if x_read_m != m_rows:
            x = x[:m_rows, :]
        if block256:
            if is_w3:
                w_fp8 = unpack_w3_u32_chunk_pallas(w_ref, c0, n0, nc).astype(
                    jnp.float8_e4m3fn
                )
            else:
                w_fp8 = pltpu.bitcast(
                    w_ref[pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0, nc)],
                    jnp.float4_e2m1fn,
                ).astype(jnp.float8_e4m3fn)
            res = jnp.dot(x, w_fp8, preferred_element_type=jnp.float32)
            sb_id = c0 // scale_block
            s = (
                s_hoisted[sb_id : sb_id + 1, :]
                if s_hoisted is not None
                else _load_scale_tile(
                    s_ref, sb_id, 1, n0, nc, jnp.float32
                )
            )
            if use_w4a8:
                x_sc = lhs_sc_ref[
                    pl.ds(0, read_m), pl.ds((b_offset + b_id) * 128, 128)
                ]
                if m_rows == 8:
                    x_sc = x_sc[:8, :]
                res = (res * x_sc[:, 0:1]) * s
            else:
                res = res * s
        else:
            if s_ref.dtype == jnp.float8_e4m3fn and eff_kc < 512 and not is_w3:
                g0 = c0 // _FP4_GROUP
                gq = eff_kc // _FP4_GROUP
                if g0 % 32 == 0 or s_blk32 is None:
                    s_blk32 = s_ref[pl.ds((g0 // 32) * 32, 32), pl.ds(n0, nc)].astype(
                        jnp.bfloat16
                    )
                s_bf = s_blk32[g0 % 32 : (g0 % 32) + gq, :]
                words = w_ref[
                    pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0, nc)
                ]
                w_u = _unpack_fp4_to_bf16(words, unpack_mode)
                if kperm_order and eff_kc == 256:
                    w_bf = (
                        w_u.reshape(_FP4_GROUP, gq, nc) * s_bf[None, :, :]
                    ).reshape(eff_kc, nc)
                else:
                    w_bf = w_u * jnp.repeat(s_bf, _FP4_GROUP, axis=0)
            else:
                w_bf = _dequant_k16_tile(
                    w_ref,
                    s_ref,
                    c0,
                    eff_kc,
                    n0,
                    nc,
                    unpack_mode,
                    kperm_order=kperm_order,
                    is_w3=is_w3,
                )
            res = jnp.dot(x, w_bf, preferred_element_type=jnp.float32)
        acc = res if acc is None else acc + res
    if gs_scale is not None:
        acc = acc * gs_scale
    return acc


def _fp4_dual_gmm_chunked_v2(
    x_2d_ref,
    w_ref,
    s_ref,
    m_rows: int,
    k_dim: int,
    n0_g: jax.Array | int,
    n0_u: jax.Array | int,
    nc: int,
    block256: bool,
    scale_block: int,
    *,
    kc_step: int = 256,
    unpack_mode: str = "via_fp8",
    kperm_order: bool = False,
    lhs_sc_ref=None,
    gs_g: jax.Array | None = None,
    gs_u: jax.Array | None = None,
    hoist_scales: bool = False,
    is_w3: bool = False,
    x_col_base: int = 0,
    b_offset: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Interleaved gate + up matmul pushing ``wg`` to MXU BEFORE unpacking ``wu``!

    Supports W4A16, W4A8 (``lhs_sc_ref``), Cold-Expert W3A16/W3A8 (``is_w3``),
    hoisted K256 scales, and K-half ``x_col_base`` offsets.
    """
    eff_kc = 256 if (block256 or kperm_order or is_w3) else kc_step
    read_m = 16 if m_rows == 8 else m_rows
    use_w4a8 = block256 and (lhs_sc_ref is not None)
    if block256 and not use_w4a8:
        n_k_blocks = k_dim // scale_block
        sub_nc = min(nc, 512)
        n_sub = nc // sub_nc
        can_hoist = hoist_scales and n_k_blocks >= 8 and n_k_blocks % 8 == 0
        sg_subs = [
            _load_scale_tile(
                s_ref, 0, n_k_blocks, n0_g + si * sub_nc, sub_nc, jnp.float32
            )
            if can_hoist
            else None
            for si in range(n_sub)
        ]
        su_subs = [
            _load_scale_tile(
                s_ref, 0, n_k_blocks, n0_u + si * sub_nc, sub_nc, jnp.float32
            )
            if can_hoist
            else None
            for si in range(n_sub)
        ]
        accs_g: list[jax.Array | None] = [None] * n_sub
        accs_u: list[jax.Array | None] = [None] * n_sub
        chunks_per_sb = scale_block // eff_kc
        for sb_id in range(n_k_blocks):
            c_base = sb_id * scale_block
            bg_list: list[jax.Array | None] = [None] * n_sub
            bu_list: list[jax.Array | None] = [None] * n_sub
            for ci in range(chunks_per_sb):
                c0 = c_base + ci * eff_kc
                x = x_2d_ref[pl.ds(0, read_m), pl.ds(x_col_base + c0, eff_kc)]
                if read_m != m_rows:
                    x = x[:m_rows, :]
                for si in range(n_sub):
                    cg = n0_g + si * sub_nc
                    cu = n0_u + si * sub_nc
                    if is_w3:
                        wg_fp8 = unpack_w3_u32_chunk_pallas(
                            w_ref, c0, cg, sub_nc
                        ).astype(jnp.float8_e4m3fn)
                    else:
                        wg_fp8 = pltpu.bitcast(
                            w_ref[
                                pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK),
                                pl.ds(cg, sub_nc),
                            ],
                            jnp.float4_e2m1fn,
                        ).astype(jnp.float8_e4m3fn)
                    rg = jnp.dot(
                        x, wg_fp8, preferred_element_type=jnp.float32
                    )
                    bg_list[si] = rg if bg_list[si] is None else bg_list[si] + rg
                    if is_w3:
                        wu_fp8 = unpack_w3_u32_chunk_pallas(
                            w_ref, c0, cu, sub_nc
                        ).astype(jnp.float8_e4m3fn)
                    else:
                        wu_fp8 = pltpu.bitcast(
                            w_ref[
                                pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK),
                                pl.ds(cu, sub_nc),
                            ],
                            jnp.float4_e2m1fn,
                        ).astype(jnp.float8_e4m3fn)
                    ru = jnp.dot(
                        x, wu_fp8, preferred_element_type=jnp.float32
                    )
                    bu_list[si] = ru if bu_list[si] is None else bu_list[si] + ru
            for si in range(n_sub):
                cg = n0_g + si * sub_nc
                cu = n0_u + si * sub_nc
                sg = (
                    sg_subs[si][sb_id : sb_id + 1, :]
                    if sg_subs[si] is not None
                    else _load_scale_tile(
                        s_ref, sb_id, 1, cg, sub_nc, jnp.float32
                    )
                )
                su = (
                    su_subs[si][sb_id : sb_id + 1, :]
                    if su_subs[si] is not None
                    else _load_scale_tile(
                        s_ref, sb_id, 1, cu, sub_nc, jnp.float32
                    )
                )
                gs_v = bg_list[si] * sg
                us_v = bu_list[si] * su
                accs_g[si] = gs_v if accs_g[si] is None else accs_g[si] + gs_v
                accs_u[si] = us_v if accs_u[si] is None else accs_u[si] + us_v
        acc_g = accs_g[0] if n_sub == 1 else jnp.concatenate(accs_g, axis=1)
        acc_u = accs_u[0] if n_sub == 1 else jnp.concatenate(accs_u, axis=1)
        if gs_g is not None:
            acc_g = acc_g * gs_g
        if gs_u is not None:
            acc_u = acc_u * gs_u
        return acc_g, acc_u
    x_read_m = max(32, read_m) if use_w4a8 else read_m
    acc_g = None
    acc_u = None
    sg_blk32 = None
    su_blk32 = None
    n_k_blocks = k_dim // scale_block if block256 else 0
    can_hoist = block256 and hoist_scales and n_k_blocks >= 8 and n_k_blocks % 8 == 0
    sg_hoisted = (
        _load_scale_tile(s_ref, 0, n_k_blocks, n0_g, nc, jnp.float32)
        if can_hoist
        else None
    )
    su_hoisted = (
        _load_scale_tile(s_ref, 0, n_k_blocks, n0_u, nc, jnp.float32)
        if can_hoist
        else None
    )
    for b_id, c0 in enumerate(range(0, k_dim, eff_kc)):
        x = x_2d_ref[pl.ds(0, x_read_m), pl.ds(x_col_base + c0, eff_kc)]
        if x_read_m != m_rows:
            x = x[:m_rows, :]
        if block256:
            x_sc_col = None
            if use_w4a8:
                x_sc = lhs_sc_ref[
                    pl.ds(0, read_m), pl.ds((b_offset + b_id) * 128, 128)
                ]
                if m_rows == 8:
                    x_sc = x_sc[:8, :]
                x_sc_col = x_sc[:, 0:1]

            if is_w3:
                wg_fp8 = unpack_w3_u32_chunk_pallas(w_ref, c0, n0_g, nc).astype(
                    jnp.float8_e4m3fn
                )
            else:
                wg_fp8 = pltpu.bitcast(
                    w_ref[pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0_g, nc)],
                    jnp.float4_e2m1fn,
                ).astype(jnp.float8_e4m3fn)
            rg = jnp.dot(x, wg_fp8, preferred_element_type=jnp.float32)
            sb_id = c0 // scale_block
            sg = (
                sg_hoisted[sb_id : sb_id + 1, :]
                if sg_hoisted is not None
                else _load_scale_tile(
                    s_ref, sb_id, 1, n0_g, nc, jnp.float32
                )
            )
            if x_sc_col is not None:
                rg = rg * x_sc_col
            acc_g = rg * sg if acc_g is None else acc_g + rg * sg

            if is_w3:
                wu_fp8 = unpack_w3_u32_chunk_pallas(w_ref, c0, n0_u, nc).astype(
                    jnp.float8_e4m3fn
                )
            else:
                wu_fp8 = pltpu.bitcast(
                    w_ref[pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0_u, nc)],
                    jnp.float4_e2m1fn,
                ).astype(jnp.float8_e4m3fn)
            ru = jnp.dot(x, wu_fp8, preferred_element_type=jnp.float32)
            su = (
                su_hoisted[sb_id : sb_id + 1, :]
                if su_hoisted is not None
                else _load_scale_tile(
                    s_ref, sb_id, 1, n0_u, nc, jnp.float32
                )
            )
            if x_sc_col is not None:
                ru = ru * x_sc_col
            acc_u = ru * su if acc_u is None else acc_u + ru * su
        else:
            g0 = c0 // _FP4_GROUP
            gq = eff_kc // _FP4_GROUP
            if s_ref.dtype == jnp.float8_e4m3fn and eff_kc < 512 and not is_w3:
                if g0 % 32 == 0 or sg_blk32 is None:
                    sg_blk32 = s_ref[
                        pl.ds((g0 // 32) * 32, 32), pl.ds(n0_g, nc)
                    ].astype(jnp.bfloat16)
                    su_blk32 = s_ref[
                        pl.ds((g0 // 32) * 32, 32), pl.ds(n0_u, nc)
                    ].astype(jnp.bfloat16)
                sg_bf = sg_blk32[g0 % 32 : (g0 % 32) + gq, :]
                su_bf = su_blk32[g0 % 32 : (g0 % 32) + gq, :]
                wg_words = w_ref[
                    pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0_g, nc)
                ]
                wg_u = _unpack_fp4_to_bf16(wg_words, unpack_mode)
                wg_bf = (
                    (wg_u.reshape(_FP4_GROUP, gq, nc) * sg_bf[None, :, :]).reshape(
                        eff_kc, nc
                    )
                    if (kperm_order and eff_kc == 256)
                    else wg_u * jnp.repeat(sg_bf, _FP4_GROUP, axis=0)
                )
                rg = jnp.dot(x, wg_bf, preferred_element_type=jnp.float32)
                acc_g = rg if acc_g is None else acc_g + rg

                wu_words = w_ref[
                    pl.ds(c0 // _FP4_PACK, eff_kc // _FP4_PACK), pl.ds(n0_u, nc)
                ]
                wu_u = _unpack_fp4_to_bf16(wu_words, unpack_mode)
                wu_bf = (
                    (wu_u.reshape(_FP4_GROUP, gq, nc) * su_bf[None, :, :]).reshape(
                        eff_kc, nc
                    )
                    if (kperm_order and eff_kc == 256)
                    else wu_u * jnp.repeat(su_bf, _FP4_GROUP, axis=0)
                )
                ru = jnp.dot(x, wu_bf, preferred_element_type=jnp.float32)
                acc_u = ru if acc_u is None else acc_u + ru
            else:
                # 1. Unpack wg -> immediately push rg to MXU!
                wg_bf = _dequant_k16_tile(
                    w_ref,
                    s_ref,
                    c0,
                    eff_kc,
                    n0_g,
                    nc,
                    unpack_mode,
                    kperm_order=kperm_order,
                    is_w3=is_w3,
                )
                rg = jnp.dot(x, wg_bf, preferred_element_type=jnp.float32)
                acc_g = rg if acc_g is None else acc_g + rg

                # 2. Unpack wu while MXU executes rg -> push ru to MXU!
                wu_bf = _dequant_k16_tile(
                    w_ref,
                    s_ref,
                    c0,
                    eff_kc,
                    n0_u,
                    nc,
                    unpack_mode,
                    kperm_order=kperm_order,
                    is_w3=is_w3,
                )
                ru = jnp.dot(x, wu_bf, preferred_element_type=jnp.float32)
                acc_u = ru if acc_u is None else acc_u + ru
    if gs_g is not None:
        acc_g = acc_g * gs_g
    if gs_u is not None:
        acc_u = acc_u * gs_u
    return acc_g, acc_u


def _dequant_strip_to_vmem_k16(
    w_ref,
    s_ref,
    w_deq_buf,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    unpack_mode: str,
    kperm_order: bool = False,
) -> None:
    """Unpack an entire ``[k_dim, nc]`` K16 strip ONCE into ``w_deq_buf`` in VMEM."""
    kc = 256
    gpc = kc // _FP4_GROUP
    s_blk32 = None
    for c0 in range(0, k_dim, kc):
        g0 = c0 // _FP4_GROUP
        if s_ref.dtype == jnp.float8_e4m3fn:
            if g0 % 32 == 0 or s_blk32 is None:
                s_blk32 = s_ref[pl.ds((g0 // 32) * 32, 32), pl.ds(n0, nc)].astype(
                    jnp.bfloat16
                )
            s_bf = s_blk32[g0 % 32 : (g0 % 32) + gpc, :]
        else:
            s_bf = s_ref[pl.ds(g0, gpc), pl.ds(n0, nc)].astype(jnp.bfloat16)
        words = w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)]
        w_u = _unpack_fp4_to_bf16(words, unpack_mode)
        if kperm_order:
            w_bf = (w_u.reshape(_FP4_GROUP, gpc, nc) * s_bf[None, :, :]).reshape(
                kc, nc
            )
        else:
            w_bf = w_u * jnp.repeat(s_bf, _FP4_GROUP, axis=0)
        w_deq_buf[pl.ds(c0, kc), pl.ds(0, nc)] = w_bf


def _unpack_strip_to_vmem_k256(
    w_ref,
    w_deq_buf,
    k_dim: int,
    n0: jax.Array | int,
    nc: int,
    *,
    is_w3: bool = False,
) -> None:
    """Unpack an entire ``[k_dim, nc]`` K256 strip ONCE into ``float8_e4m3fn`` VMEM."""
    kc = 256
    for c0 in range(0, k_dim, kc):
        if is_w3:
            w_fp8 = unpack_w3_u32_chunk_pallas(w_ref, c0, n0, nc).astype(
                jnp.float8_e4m3fn
            )
        else:
            w_fp8 = pltpu.bitcast(
                w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
        w_deq_buf[pl.ds(c0, kc), pl.ds(0, nc)] = w_fp8


def _dot_from_vmem_deq(
    x_2d_ref,
    w_deq_buf,
    r0: jax.Array | int,
    m_rows: int,
    k_dim: int,
    nc: int,
    gs_scale: jax.Array | None = None,
    *,
    s_k256_ref=None,
    n0_s: jax.Array | int = 0,
    hoist_scales: bool = False,
    scale_block: int = 256,
) -> jax.Array:
    """Compute ``x[r0:r0+m_rows, 0:k_dim] @ w_deq_buf[0:k_dim, 0:nc]`` with zero VPU unpack."""
    kc = 256
    acc = None
    n_sb = k_dim // scale_block
    s_all = (
        _load_scale_tile(s_k256_ref, 0, n_sb, n0_s, nc, jnp.float32)
        if (s_k256_ref is not None and hoist_scales and n_sb >= 8 and n_sb % 8 == 0)
        else None
    )
    for b_id, c0 in enumerate(range(0, k_dim, kc)):
        x = x_2d_ref[pl.ds(r0, m_rows), pl.ds(c0, kc)]
        w_t = w_deq_buf[pl.ds(c0, kc), pl.ds(0, nc)]
        res = jnp.dot(x, w_t, preferred_element_type=jnp.float32)
        if s_k256_ref is not None:
            sb_id = c0 // scale_block
            s = (
                s_all[sb_id : sb_id + 1, :]
                if s_all is not None
                else _load_scale_tile(s_k256_ref, sb_id, 1, n0_s, nc, jnp.float32)
            )
            res = res * s
        acc = res if acc is None else acc + res
    if gs_scale is not None:
        acc = acc * gs_scale
    return acc


def _build_masked_x_from_3d(
    lhs_3d_ref, masked_x_ref, bd_rows: int, kperm_order: bool = False
) -> None:
    """Build block-diagonal ``[bd_rows * 16, 4096]`` bf16 once per tile."""
    kc = 256
    gpc = kc // _FP4_GROUP
    col_i = lax.broadcasted_iota(jnp.int32, (gpc, kc), 1)
    grp_i = (col_i % _FP4_GROUP) if kperm_order else (col_i // _FP4_GROUP)
    diag_16 = lax.broadcasted_iota(jnp.int32, (gpc, kc), 0) == grp_i
    zeros_16 = jnp.zeros((gpc, kc), dtype=jnp.bfloat16)
    for a in range(bd_rows):
        row_2d = lhs_3d_ref[pl.ds(a, 1), :, :].reshape(1, _H)
        dst_rows = pl.ds(a * gpc, gpc)
        for b_id in range(_H // kc):
            c0 = b_id * kc
            bcast = jnp.broadcast_to(row_2d[:, c0 : c0 + kc], (gpc, kc))
            masked_x_ref[dst_rows, pl.ds(c0, kc)] = jnp.where(
                diag_16, bcast, zeros_16
            )


def _build_masked_x_from_2d(
    src_2d_ref, masked_x_ref, bd_rows: int, kperm_order: bool = False
) -> None:
    """Build block-diagonal ``[bd_rows * 16, 4096]`` bf16 from 2D ``act_2d_buf``."""
    kc = 256
    gpc = kc // _FP4_GROUP
    col_i = lax.broadcasted_iota(jnp.int32, (gpc, kc), 1)
    grp_i = (col_i % _FP4_GROUP) if kperm_order else (col_i // _FP4_GROUP)
    diag_16 = lax.broadcasted_iota(jnp.int32, (gpc, kc), 0) == grp_i
    zeros_16 = jnp.zeros((gpc, kc), dtype=jnp.bfloat16)
    for s in range(4):
        c_base = s * 1024
        strip = src_2d_ref[pl.ds(0, 16), pl.ds(c_base, 1024)]
        for a in range(bd_rows):
            dst_rows = pl.ds(a * gpc, gpc)
            for sub in range(4):
                off = sub * kc
                bcast = jnp.broadcast_to(
                    strip[a : a + 1, off : off + kc], (gpc, kc)
                )
                masked_x_ref[dst_rows, pl.ds(c_base + off, kc)] = jnp.where(
                    diag_16, bcast, zeros_16
                )


def _fp4_blockdiag_hoisted_strip_v2(
    masked_x_ref,
    w_ref,
    s_ref,
    bd_rows: int,
    n0: jax.Array | int,
    nc: int,
    gs_scale: jax.Array | None = None,
) -> jax.Array:
    """Block-diagonal K16 matmul over ``[n0 : n0+nc]`` using hoisted ``masked_x``."""
    kc = 256
    gpc = kc // _FP4_GROUP
    rows = bd_rows * gpc
    acc = None
    s_blk32 = None
    for b_id in range(_H // kc):
        c0 = b_id * kc
        g0 = c0 // _FP4_GROUP
        mx = masked_x_ref[pl.ds(0, rows), pl.ds(c0, kc)]
        w = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        res = jnp.dot(mx, w, preferred_element_type=jnp.float32)
        if s_ref.dtype == jnp.float8_e4m3fn:
            if g0 % 32 == 0 or s_blk32 is None:
                s_blk32 = s_ref[pl.ds((g0 // 32) * 32, 32), pl.ds(n0, nc)].astype(
                    jnp.float32
                )
            s = s_blk32[g0 % 32 : (g0 % 32) + gpc, :]
        else:
            s = s_ref[pl.ds(g0, gpc), pl.ds(n0, nc)].astype(jnp.float32)
        part = (
            (res.reshape(bd_rows, 2, 8, nc) * s.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )
        acc = part if acc is None else acc + part
    out = jnp.concatenate(
        [
            jnp.sum(acc[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )
    return out * gs_scale if gs_scale is not None else out


def _fp4_blockdiag_hoisted_dual_strip_v2(
    masked_x_ref,
    w_ref,
    s_ref,
    bd_rows: int,
    n0_g: jax.Array | int,
    n0_u: jax.Array | int,
    nc: int,
    gs_g: jax.Array | None = None,
    gs_u: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Dual gate+up block-diagonal K16 strip sharing one ``masked_x`` load per K256."""
    kc = 256
    gpc = kc // _FP4_GROUP
    rows = bd_rows * gpc
    acc_g = None
    acc_u = None
    sg_blk32 = None
    su_blk32 = None
    for b_id in range(_H // kc):
        c0 = b_id * kc
        g0 = c0 // _FP4_GROUP
        mx = masked_x_ref[pl.ds(0, rows), pl.ds(c0, kc)]
        if s_ref.dtype == jnp.float8_e4m3fn:
            if g0 % 32 == 0 or sg_blk32 is None:
                sg_blk32 = s_ref[
                    pl.ds((g0 // 32) * 32, 32), pl.ds(n0_g, nc)
                ].astype(jnp.float32)
                su_blk32 = s_ref[
                    pl.ds((g0 // 32) * 32, 32), pl.ds(n0_u, nc)
                ].astype(jnp.float32)
            sg = sg_blk32[g0 % 32 : (g0 % 32) + gpc, :]
            su = su_blk32[g0 % 32 : (g0 % 32) + gpc, :]
        else:
            sg = s_ref[pl.ds(g0, gpc), pl.ds(n0_g, nc)].astype(jnp.float32)
            su = s_ref[pl.ds(g0, gpc), pl.ds(n0_u, nc)].astype(jnp.float32)

        wg = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0_g, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        rg = jnp.dot(mx, wg, preferred_element_type=jnp.float32)
        pg = (
            (rg.reshape(bd_rows, 2, 8, nc) * sg.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )

        wu = pltpu.bitcast(
            w_ref[pl.ds(c0 // _FP4_PACK, kc // _FP4_PACK), pl.ds(n0_u, nc)],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        ru = jnp.dot(mx, wu, preferred_element_type=jnp.float32)
        pu = (
            (ru.reshape(bd_rows, 2, 8, nc) * su.reshape(1, 2, 8, nc))
            .sum(axis=1)
            .reshape(bd_rows * 8, nc)
        )
        acc_g = pg if acc_g is None else acc_g + pg
        acc_u = pu if acc_u is None else acc_u + pu

    out_g = jnp.concatenate(
        [
            jnp.sum(acc_g[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )
    out_u = jnp.concatenate(
        [
            jnp.sum(acc_u[a * 8 : (a + 1) * 8], axis=0, keepdims=True)
            for a in range(bd_rows)
        ],
        axis=0,
    )
    if gs_g is not None:
        out_g = out_g * gs_g
    if gs_u is not None:
        out_u = out_u * gs_u
    return out_g, out_u


def _build_flat_tile_plan_smem(
    num_exec: jax.Array,
    tile_m: int,
    r_scratch: RoutingScratch,
    tile_k_smem,
    tile_base_smem,
    tile_rows_smem,
    tile_flags_smem,
) -> jax.Array:
    """Flatten ``(expert k, row_tile)`` into a 1D SMEM table for async gather pipelining."""

    def _per_expert(k, t_cnt):
        row_start = r_scratch.exec_start[k]
        row_end = r_scratch.exec_start[k + 1]
        r_e = row_end - row_start
        n_rt = (r_e + tile_m - 1) // tile_m

        def _per_tile(ti, tc):
            off = ti * tile_m
            cur = jnp.minimum(tile_m, r_e - off)
            f_first = jnp.where(ti == 0, jnp.int32(1), jnp.int32(0))
            f_last = jnp.where(ti == (n_rt - 1), jnp.int32(2), jnp.int32(0))
            tile_k_smem[tc] = k
            tile_base_smem[tc] = row_start + off
            tile_rows_smem[tc] = cur
            tile_flags_smem[tc] = f_first | f_last
            return tc + 1

        return lax.fori_loop(0, n_rt, _per_tile, t_cnt)

    return lax.fori_loop(0, num_exec, _per_expert, jnp.int32(0))


def _advance_unit(e_idx: jax.Array, part: jax.Array, by: int) -> tuple[jax.Array, jax.Array]:
    """Advance ``(e_idx, part)`` by ``by`` units (``1 <= by <= 5``) with zero division."""
    p1 = part + by
    w1 = p1 >= 3
    e1 = jnp.where(w1, e_idx + 1, e_idx)
    p2 = jnp.where(w1, p1 - 3, p1)
    w2 = p2 >= 3
    return jnp.where(w2, e1 + 1, e1), jnp.where(w2, p2 - 3, p2)


def _ring_slot(u: jax.Array, n_slots: int) -> jax.Array:
    if n_slots == 4:
        return u & jnp.int32(3)
    return lax.rem(u, jnp.int32(n_slots))


def _sparse_decode_v2_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    scale_block: int,
    num_replicas: int,
    has_gs: bool,
    has_w3: bool,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    cfg: SparseDecodeV2Config,
):
    has_replicas = num_replicas > 0
    is_k256 = scale_block >= 256
    is_dense = cfg.row_mode == "dense"
    is_mxu_sparse = cfg.row_mode == "mxu_sparse"
    is_dense_route = is_dense or is_mxu_sparse
    is_khalf = cfg.ring_mode.endswith("_khalf") and (not is_dense_route)
    n_slots = (
        6
        if cfg.ring_mode.startswith("ring6")
        else (4 if cfg.ring_mode.startswith("ring4") else 3)
    )
    n_kh_w1 = 4 if cfg.ring_mode == "ring4_khalf" else 2
    n_kh_w2 = 2 if cfg.ring_mode == "ring4_khalf" else 1
    dual_units = cfg.ring_mode.endswith("_dual")
    contig_mode = cfg.contig_parts_mode
    use_bd = (
        (not is_dense)
        and (not is_mxu_sparse)
        and (not is_khalf)
        and (not is_k256)
        and (not has_w3)
        and bool(cfg.k16_blockdiag_rows)
    )
    use_vmem_deq = (
        (not is_dense)
        and (not is_khalf)
        and (cfg.multi_m32_mode == "vmem_deq")
        and (cfg.tile_m > 32)
    )
    use_loop32 = (
        (not is_dense)
        and (not is_khalf)
        and (cfg.multi_m32_mode == "loop32")
        and (cfg.tile_m > 32)
    )
    tile_m = cfg.tile_m
    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep

    it = iter(all_refs)
    rank_ref = next(it)
    replica_meta_ref = next(it) if has_replicas else None
    w3_mask_ref = next(it) if has_w3 else None
    logits_ref = next(it)
    bias_ref = next(it)
    x_in_ref = next(it)
    w1_ref = next(it)
    s1_ref = next(it)
    w2_ref = next(it)
    s2_ref = next(it)
    norm_ref = next(it)
    if has_gs:
        w1_gs_ref = next(it)
        w2_gs_ref = next(it)
    else:
        w1_gs_ref = w2_gs_ref = None
    if has_replicas:
        rep_w1_ref = next(it)
        rep_s1_ref = next(it)
        rep_w2_ref = next(it)
        rep_s2_ref = next(it)
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None
    output_ref = next(it)

    if is_dense_route:
        d_scratch = DenseRoutingScratch(
            *(next(it) for _ in range(len(DenseRoutingScratch._fields)))
        )
        r_scratch = None
        tile_k_smem = tile_base_smem = tile_rows_smem = tile_flags_smem = None
    else:
        r_scratch = RoutingScratch(
            *(next(it) for _ in range(len(RoutingScratch._fields)))
        )
        d_scratch = None
        tile_k_smem = next(it)
        tile_base_smem = next(it)
        tile_rows_smem = next(it)
        tile_flags_smem = next(it)

    if has_replicas:
        owner_load_smem = next(it)
        moved_flag_smem = next(it)
        local_suppress_smem = next(it)
        holder_slot_expert_smem = next(it)
        holder_cand_smem = (
            next(it)
            if (not is_dense_route and not cfg.static_holder_mask)
            else None
        )
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = holder_cand_smem = None

    if is_khalf:
        w1_kh_ring = next(it)
        s1_kh_ring = next(it)
        w1_kh_sem = next(it)
        w2_kh_ring = next(it)
        s2_kh_ring = next(it)
        w2_kh_sem = next(it)
        gu_kh_buf = next(it)
        w_ring = s_ring = ring_sem = None
    else:
        w_ring = next(it)
        s_ring = next(it)
        ring_sem = next(it)
        w1_kh_ring = s1_kh_ring = w1_kh_sem = None
        w2_kh_ring = s2_kh_ring = w2_kh_sem = gu_kh_buf = None
    misc_sem = next(it)
    use_lhs_fp8 = is_k256 and cfg.lhs_fp8

    if is_dense:
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        sq_dense_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it) if not return_sharded else None
        x_clean_2d = x_3d_buf = gather_3d_buf = gather_sem = io_3d_buf = None
        lhs_2d_buf = masked_x_buf = w_deq_buf = None
    elif is_mxu_sparse:
        x_clean_2d = next(it)
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        w_deq_buf = next(it) if use_vmem_deq else None
        gain_f32_buf = next(it)
        accum_2d = next(it)
        accum_3d = next(it)
        fan_3d_buf = next(it) if not return_sharded else None
        x_3d_buf = gather_3d_buf = gather_sem = io_3d_buf = None
        masked_x_buf = sq_dense_buf = None
    else:
        x_clean_2d = None
        x_3d_buf = next(it)
        if cfg.async_gather:
            gather_3d_buf = next(it)
            gather_sem = next(it)
            io_3d_buf = None if cfg.fused_norm_scatter else next(it)
        else:
            io_3d_buf = next(it)
            gather_3d_buf = gather_sem = None
        lhs_2d_buf = next(it)
        act_2d_buf = next(it)
        y_stage_buf = next(it)
        lhs_fp8_buf = next(it) if use_lhs_fp8 else None
        lhs_sc_buf = next(it) if use_lhs_fp8 else None
        masked_x_buf = next(it) if use_bd else None
        w_deq_buf = next(it) if use_vmem_deq else None
        gain_f32_buf = next(it)
        accum_3d = next(it)
        accum_2d = sq_dense_buf = None
        fan_3d_buf = x_3d_buf if not return_sharded else None

    if cfg.pair_rs and ep in (8, 16):
        rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
        rs_pair_recv = next(it)
        rs_chip_send = next(it)
        rs_chip_recv = next(it)
        rs_pair_ssem = next(it)
        rs_pair_rsem = next(it)
        rs_chip_ssem = next(it)
        rs_chip_rsem = next(it)
        rs_recv_3d = rs_send_sem = rs_recv_sem = None
    else:
        rs_send_bf16 = next(it) if cfg.rs_wire_bf16 else None
        rs_recv_3d = next(it)
        rs_send_sem = next(it)
        rs_recv_sem = next(it)
        rs_pair_recv = rs_chip_send = rs_chip_recv = None
        rs_pair_ssem = rs_pair_rsem = rs_chip_ssem = rs_chip_rsem = None

    owner_3d = next(it)
    if not return_sharded:
        if cfg.pair_ag_relay and ep in (8, 16):
            ag_ssem = next(it)
            ag_psem = next(it)
            fan_send_sem = fan_recv_sem = None
        else:
            fan_send_sem = next(it)
            fan_recv_sem = next(it)
            ag_ssem = ag_psem = None
    else:
        fan_send_sem = fan_recv_sem = ag_ssem = ag_psem = None

    # 1. Signal entry barrier immediately.
    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    w3_mask_val = w3_mask_ref[0] if has_w3 else jnp.int32(0)
    w1_u32 = w1_ref.bitcast(jnp.uint32)
    w2_u32 = w2_ref.bitcast(jnp.uint32)
    if has_replicas:
        rep_w1_u32 = rep_w1_ref.bitcast(jnp.uint32)
        rep_w2_u32 = rep_w2_ref.bitcast(jnp.uint32)

    def _slot_is_w3(src_kind, w_slot):
        if not has_w3:
            return None
        return (src_kind == 0) & (
            (lax.shift_right_logical(w3_mask_val, w_slot) & jnp.int32(1)) != 0
        )

    # --- Homogeneous 8-MiB / 6-MiB (W3) circular & K-half weight ring DMA helpers ---
    half_i = _I // 2
    nqb = _H // scale_block

    if is_khalf:
        half_nqb = nqb // 2

        def _issue_kh_w1_bank(w1_b, s1_b, kh_slot, w_slot, kh_part):
            k_off = pl.multiple_of(kh_part * 256, 256)
            s_off = pl.multiple_of(kh_part * half_nqb, half_nqb)
            pltpu.make_async_copy(
                w1_b.at[w_slot, pl.ds(k_off, 256), :],
                w1_kh_ring.at[kh_slot],
                w1_kh_sem.at[kh_slot],
            ).start()
            pltpu.make_async_copy(
                s1_b.at[w_slot, pl.ds(s_off, half_nqb), :],
                s1_kh_ring.at[kh_slot],
                w1_kh_sem.at[kh_slot],
            ).start()

        def _issue_kh_w2_bank(w2_b, s2_b, w2_slot, w_slot):
            pltpu.make_async_copy(
                w2_b.at[w_slot], w2_kh_ring.at[w2_slot], w2_kh_sem.at[w2_slot]
            ).start()
            pltpu.make_async_copy(
                s2_b.at[w_slot], s2_kh_ring.at[w2_slot], w2_kh_sem.at[w2_slot]
            ).start()

        def start_kh_w1(kh_slot, src_kind, w_slot, kh_part: int):
            if has_replicas:
                @pl.when(src_kind == 0)
                def _h_w1():
                    _issue_kh_w1_bank(w1_u32, s1_ref, kh_slot, w_slot, kh_part)

                @pl.when(src_kind == 1)
                def _r_w1():
                    _issue_kh_w1_bank(
                        rep_w1_u32, rep_s1_ref, kh_slot, w_slot, kh_part
                    )
            else:
                _issue_kh_w1_bank(w1_u32, s1_ref, kh_slot, w_slot, kh_part)

        def start_kh_w2(w2_slot, src_kind, w_slot):
            if has_replicas:
                @pl.when(src_kind == 0)
                def _h_w2():
                    _issue_kh_w2_bank(w2_u32, s2_ref, w2_slot, w_slot)

                @pl.when(src_kind == 1)
                def _r_w2():
                    _issue_kh_w2_bank(rep_w2_u32, rep_s2_ref, w2_slot, w_slot)
            else:
                _issue_kh_w2_bank(w2_u32, s2_ref, w2_slot, w_slot)

        def wait_kh_w1(kh_slot):
            pltpu.make_async_copy(
                w1_kh_ring.at[kh_slot],
                w1_kh_ring.at[kh_slot],
                w1_kh_sem.at[kh_slot],
            ).wait()
            pltpu.make_async_copy(
                s1_kh_ring.at[kh_slot],
                s1_kh_ring.at[kh_slot],
                w1_kh_sem.at[kh_slot],
            ).wait()

        def wait_kh_w2(w2_slot):
            pltpu.make_async_copy(
                w2_kh_ring.at[w2_slot],
                w2_kh_ring.at[w2_slot],
                w2_kh_sem.at[w2_slot],
            ).wait()
            pltpu.make_async_copy(
                s2_kh_ring.at[w2_slot],
                s2_kh_ring.at[w2_slot],
                w2_kh_sem.at[w2_slot],
            ).wait()

    else:

        def _issue_unit_typed(w1_b, s1_b, w2_b, s2_b, r_slot, w_slot, part, use_w3_rows: bool):
            w_rows = 384 if use_w3_rows else (_H // _FP4_PACK)
            if contig_mode in (2, 3):
                # Contiguous 8-MiB / 6-MiB part-major layout [2*E, 512, 4096]!
                @pl.when(part < 2)
                def _pm01():
                    u_idx = w_slot * jnp.int32(2) + part
                    if use_w3_rows:
                        pltpu.make_async_copy(
                            w1_b.at[u_idx, 0:w_rows, :],
                            w_ring.at[r_slot, 0:w_rows, :],
                            ring_sem.at[r_slot],
                        ).start()
                    else:
                        pltpu.make_async_copy(
                            w1_b.at[u_idx],
                            w_ring.at[r_slot],
                            ring_sem.at[r_slot],
                        ).start()
                    pltpu.make_async_copy(
                        s1_b.at[u_idx], s_ring.at[r_slot], ring_sem.at[r_slot]
                    ).start()

                @pl.when(part == 2)
                def _pm2():
                    if use_w3_rows:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot, 0:w_rows, :],
                            w_ring.at[r_slot, 0:w_rows, :],
                            ring_sem.at[r_slot],
                        ).start()
                    else:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
                        ).start()
                    pltpu.make_async_copy(
                        s2_b.at[w_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
                    ).start()

            elif dual_units:
                @pl.when(part < 2)
                def _u01():
                    g_col = pl.multiple_of(part * half_i, half_i)
                    u_col = _I + g_col
                    pltpu.make_async_copy(
                        w1_b.at[w_slot, 0:w_rows, pl.ds(g_col, half_i)],
                        w_ring.at[r_slot, 0:w_rows, 0:half_i],
                        ring_sem.at[r_slot],
                    ).start()
                    pltpu.make_async_copy(
                        w1_b.at[w_slot, 0:w_rows, pl.ds(u_col, half_i)],
                        w_ring.at[r_slot, 0:w_rows, half_i:_I],
                        ring_sem.at[r_slot],
                    ).start()
                    pltpu.make_async_copy(
                        s1_b.at[w_slot, :, pl.ds(g_col, half_i)],
                        s_ring.at[r_slot, :, 0:half_i],
                        ring_sem.at[r_slot],
                    ).start()
                    pltpu.make_async_copy(
                        s1_b.at[w_slot, :, pl.ds(u_col, half_i)],
                        s_ring.at[r_slot, :, half_i:_I],
                        ring_sem.at[r_slot],
                    ).start()

                @pl.when(part == 2)
                def _u2():
                    if use_w3_rows:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot, 0:w_rows, :],
                            w_ring.at[r_slot, 0:w_rows, :],
                            ring_sem.at[r_slot],
                        ).start()
                    else:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
                        ).start()
                    pltpu.make_async_copy(
                        s2_b.at[w_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
                    ).start()
            else:
                @pl.when(part < 2)
                def _p01():
                    col = pl.multiple_of(part * _I, _I)
                    if use_w3_rows:
                        pltpu.make_async_copy(
                            w1_b.at[w_slot, 0:w_rows, pl.ds(col, _I)],
                            w_ring.at[r_slot, 0:w_rows, :],
                            ring_sem.at[r_slot],
                        ).start()
                    else:
                        pltpu.make_async_copy(
                            w1_b.at[w_slot, :, pl.ds(col, _I)],
                            w_ring.at[r_slot],
                            ring_sem.at[r_slot],
                        ).start()
                    pltpu.make_async_copy(
                        s1_b.at[w_slot, :, pl.ds(col, _I)],
                        s_ring.at[r_slot],
                        ring_sem.at[r_slot],
                    ).start()

                @pl.when(part == 2)
                def _p2():
                    if use_w3_rows:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot, 0:w_rows, :],
                            w_ring.at[r_slot, 0:w_rows, :],
                            ring_sem.at[r_slot],
                        ).start()
                    else:
                        pltpu.make_async_copy(
                            w2_b.at[w_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
                        ).start()
                    pltpu.make_async_copy(
                        s2_b.at[w_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
                    ).start()

        def _issue_home_unit(r_slot, w_slot, part):
            if has_w3:
                w3_cond = _slot_is_w3(jnp.int32(0), w_slot)

                @pl.when(w3_cond)
                def _h_w3():
                    _issue_unit_typed(
                        w1_u32, s1_ref, w2_u32, s2_ref, r_slot, w_slot, part, True
                    )

                @pl.when(~w3_cond)
                def _h_w4():
                    _issue_unit_typed(
                        w1_u32, s1_ref, w2_u32, s2_ref, r_slot, w_slot, part, False
                    )
            else:
                _issue_unit_typed(
                    w1_u32, s1_ref, w2_u32, s2_ref, r_slot, w_slot, part, False
                )

        def start_ring_unit(r_slot, src_kind, w_slot, part):
            if has_replicas:
                @pl.when(src_kind == 0)
                def _home():
                    _issue_home_unit(r_slot, w_slot, part)

                @pl.when(src_kind == 1)
                def _rep():
                    _issue_unit_typed(
                        rep_w1_u32,
                        rep_s1_ref,
                        rep_w2_u32,
                        rep_s2_ref,
                        r_slot,
                        w_slot,
                        part,
                        False,
                    )
            else:
                _issue_home_unit(r_slot, w_slot, part)

        def wait_ring_slot(r_slot, is_w3_slot=None):
            if has_w3 and is_w3_slot is not None:
                @pl.when(is_w3_slot)
                def _w_w3():
                    pltpu.make_async_copy(
                        w_ring.at[r_slot, 0:384, :],
                        w_ring.at[r_slot, 0:384, :],
                        ring_sem.at[r_slot],
                    ).wait()

                @pl.when(~is_w3_slot)
                def _w_w4():
                    pltpu.make_async_copy(
                        w_ring.at[r_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
                    ).wait()
            else:
                pltpu.make_async_copy(
                    w_ring.at[r_slot], w_ring.at[r_slot], ring_sem.at[r_slot]
                ).wait()
            pltpu.make_async_copy(
                s_ring.at[r_slot], s_ring.at[r_slot], ring_sem.at[r_slot]
            ).wait()

    # 1B. Optional speculative first-expert HBM->VMEM prime BEFORE VPU routing!
    spec_p = min(cfg.spec_prime_parts, 3, n_slots - 1) if not is_khalf else min(cfg.spec_prime_parts, 3)
    do_spec = spec_p > 0
    if do_spec:
        if has_replicas and cfg.spec_skip_replicated:
            spec_slot = select_non_replicated_home_slot(
                me,
                e_local,
                replica_meta_ref,
                preferred_slot=0,
                num_replicas=num_replicas,
            )
        else:
            spec_slot = jnp.int32(0)
        spec_enabled = (spec_slot >= 0) & (spec_slot < e_local)
        safe_spec_slot = jnp.where(spec_enabled, spec_slot, jnp.int32(0))

        @pl.when(spec_enabled)
        def _issue_spec_prime_v2():
            if is_khalf:
                start_kh_w1(jnp.int32(0), jnp.int32(0), safe_spec_slot, 0)
                if spec_p >= 2:
                    start_kh_w1(jnp.int32(1), jnp.int32(0), safe_spec_slot, 1)
                if spec_p >= 3:
                    start_kh_w2(jnp.int32(0), jnp.int32(0), safe_spec_slot)
            else:
                for sp_i in range(spec_p):
                    start_ring_unit(
                        jnp.int32(sp_i),
                        jnp.int32(0),
                        safe_spec_slot,
                        jnp.int32(sp_i),
                    )
    else:
        spec_slot = None
        spec_enabled = jnp.bool_(False)
        safe_spec_slot = jnp.int32(0)

    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    # 2. In-kernel routing (tie-exact OR HBM-breaker Min-P / Mass / Free-an-Expert).
    if is_dense_route:
        route_dense_and_counts_v2_in_kernel(
            logits_ref,
            bias_ref,
            d_scratch,
            misc_sem.at[0],
            cfg.routing_prune,
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
        )
        counts_smem = d_scratch.counts_smem
    else:
        if has_replicas:
            if cfg.static_holder_mask:
                cand_eids = extract_my_holder_candidates_tuple(
                    me, num_replicas, replica_meta_ref
                )
            else:
                extract_my_holder_candidates_smem(
                    me, num_replicas, replica_meta_ref, holder_cand_smem
                )
                cand_eids = tuple(
                    holder_cand_smem[s] for s in range(num_replicas)
                )
        else:
            cand_eids = ()
        route_topk_and_counts_v2_in_kernel(
            logits_ref,
            bias_ref,
            r_scratch,
            misc_sem.at[0],
            cfg.routing_prune,
            my_owner=me,
            e_per_owner=e_local,
            holder_cand_eids=cand_eids,
            hoist_holder_vecs=cfg.static_holder_mask,
            precompute_owner_loads=(has_replicas and cfg.static_holder_mask),
            replica_cost_mode=replica_cost_mode,
            replica_tile_m=tile_m,
            topk=_TOPK,
            route_eps=1e-15,
            route_scale=1.0,
            route_chunk_rows=cfg.route_chunk_rows,
        )
        counts_smem = r_scratch.counts_smem

    # 3. 32-step SMEM execution plan.
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
            vpu_precomputed_loads=(not is_dense_route and cfg.static_holder_mask),
        )
        if is_dense_route:
            num_exec = build_dense_exec_plan_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                d_scratch,
                spec_slot=spec_slot,
            )
        else:
            num_exec = build_exec_plan_from_masks_smem(
                me,
                e_local,
                num_replicas,
                local_suppress_smem,
                holder_slot_expert_smem,
                r_scratch,
                spec_slot=spec_slot,
            )
    else:
        if is_dense_route:
            num_exec = build_dense_exec_plan_smem(
                me, e_local, 0, None, None, d_scratch, spec_slot=spec_slot
            )
        else:
            num_exec = build_default_exec_plan_smem(
                me, e_local, r_scratch, spec_slot=spec_slot
            )

    exec_eid_smem = (
        d_scratch.exec_expert_id if is_dense_route else r_scratch.exec_expert_id
    )
    exec_src_smem = (
        d_scratch.exec_weight_src if is_dense_route else r_scratch.exec_weight_src
    )
    exec_slot_smem = (
        d_scratch.exec_weight_slot if is_dense_route else r_scratch.exec_weight_slot
    )

    # 4. Prime circular / K-half ring BEFORE prologue!
    n_units_total = num_exec * 3
    if is_khalf:
        if do_spec:
            spec_hit_kh = (
                spec_enabled
                & (num_exec > 0)
                & (exec_src_smem[0] == 0)
                & (exec_slot_smem[0] == spec_slot)
            )

            @pl.when(spec_enabled & (~spec_hit_kh))
            def _drain_spec_kh():
                wait_kh_w1(jnp.int32(0))
                if spec_p >= 2:
                    wait_kh_w1(jnp.int32(1))
                if spec_p >= 3:
                    wait_kh_w2(jnp.int32(0))

            @pl.when(spec_hit_kh)
            def _prime_hit_kh():
                s0_k = exec_src_smem[0]
                sl0_k = exec_slot_smem[0]
                if spec_p < 2:
                    start_kh_w1(jnp.int32(1), s0_k, sl0_k, 1)
                if spec_p < 3:
                    start_kh_w2(jnp.int32(0), s0_k, sl0_k)

            @pl.when((~spec_hit_kh) & (num_exec > 0))
            def _prime_miss_kh():
                s0_k = exec_src_smem[0]
                sl0_k = exec_slot_smem[0]
                start_kh_w1(jnp.int32(0), s0_k, sl0_k, 0)
                start_kh_w1(jnp.int32(1), s0_k, sl0_k, 1)
                start_kh_w2(jnp.int32(0), s0_k, sl0_k)
        else:
            @pl.when(num_exec > 0)
            def _prime_kh0():
                s0_k = exec_src_smem[0]
                sl0_k = exec_slot_smem[0]
                start_kh_w1(jnp.int32(0), s0_k, sl0_k, 0)
                start_kh_w1(jnp.int32(1), s0_k, sl0_k, 1)
                start_kh_w2(jnp.int32(0), s0_k, sl0_k)
    elif do_spec:
        spec_hit = (
            spec_enabled
            & (num_exec > 0)
            & (exec_src_smem[0] == 0)
            & (exec_slot_smem[0] == spec_slot)
        )

        @pl.when(spec_enabled & (~spec_hit))
        def _drain_spec_v2():
            sp_w3 = _slot_is_w3(jnp.int32(0), safe_spec_slot)
            for sp_i in range(spec_p):
                wait_ring_slot(jnp.int32(sp_i), sp_w3)

        @pl.when(spec_hit)
        def _prime_rest_hit():
            for ahead in range(spec_p, n_slots - 1):
                @pl.when(ahead < n_units_total)
                def _ph(ahead=ahead):
                    e_init = ahead // 3
                    p_init = ahead % 3
                    start_ring_unit(
                        jnp.int32(ahead),
                        exec_src_smem[e_init],
                        exec_slot_smem[e_init],
                        jnp.int32(p_init),
                    )

        @pl.when(~spec_hit)
        def _prime_all_miss():
            for ahead in range(n_slots - 1):
                @pl.when(ahead < n_units_total)
                def _pm(ahead=ahead):
                    e_init = ahead // 3
                    p_init = ahead % 3
                    start_ring_unit(
                        jnp.int32(ahead),
                        exec_src_smem[e_init],
                        exec_slot_smem[e_init],
                        jnp.int32(p_init),
                    )
    else:
        for ahead in range(n_slots - 1):
            @pl.when(ahead < n_units_total)
            def _prime_unit(ahead=ahead):
                e_init = ahead // 3
                p_init = ahead % 3
                start_ring_unit(
                    jnp.int32(ahead),
                    exec_src_smem[e_init],
                    exec_slot_smem[e_init],
                    jnp.int32(p_init),
                )

    # 5. Overlapped prologue while initial 8-MiB units stream from HBM.
    if is_dense:
        for s_col in range(0, _H, 1024):
            accum_2d[:, pl.ds(s_col, 1024)] = jnp.zeros(
                (batch_size, 1024), dtype=jnp.float32
            )
        sq_dense_buf[...] = jnp.zeros_like(sq_dense_buf)
        if use_lhs_fp8:
            _prequant_lhs_k256(x_in_ref, lhs_fp8_buf, lhs_sc_buf, batch_size)
    elif is_mxu_sparse:
        # Zero-SALU prologue: zero accum_2d + sanitize NaN-padded dead tail rows of x!
        row_live = jnp.any(d_scratch.dense_w_vmem[...] > 0.0, axis=1, keepdims=True)
        b_step_p = min(batch_size, 64)
        for r0 in range(0, batch_size, b_step_p):
            live_c = row_live[r0 : r0 + b_step_p, :]
            for s_col in range(0, _H, 1024):
                col_s = pl.ds(s_col, 1024)
                accum_2d[pl.ds(r0, b_step_p), col_s] = jnp.zeros(
                    (b_step_p, 1024), dtype=jnp.float32
                )
                x_sub = x_in_ref[pl.ds(r0, b_step_p), col_s]
                x_clean_2d[pl.ds(r0, b_step_p), col_s] = jnp.where(
                    live_c, x_sub, jnp.bfloat16(0.0)
                )
    else:
        zero_rc = min(batch_size, 8)
        zeros_acc = jnp.zeros((zero_rc, 32, 128), dtype=jnp.float32)

        def _zero_acc_chunk(ci, _):
            accum_3d[pl.ds(ci * zero_rc, zero_rc), :, :] = zeros_acc
            return _

        lax.fori_loop(0, batch_size // zero_rc, _zero_acc_chunk, jnp.int32(0))

        if io_3d_buf is not None:
            zeros_io = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)

            def _zero_io_chunk(ci, _):
                io_3d_buf[pl.ds(ci * 8, 8), :, :] = zeros_io
                return _

            lax.fori_loop(0, tile_m // 8, _zero_io_chunk, jnp.int32(0))

        if gather_3d_buf is not None:
            zeros_g = jnp.zeros((8, 32, 128), dtype=jnp.bfloat16)

            def _zero_g_chunk(ci, _):
                gather_3d_buf[0, pl.ds(ci * 8, 8), :, :] = zeros_g
                gather_3d_buf[1, pl.ds(ci * 8, 8), :, :] = zeros_g
                return _

            lax.fori_loop(0, tile_m // 8, _zero_g_chunk, jnp.int32(0))

        x_rc = 16

        def _reshape_x_chunk(ci, _):
            r0 = pl.multiple_of(ci * x_rc, x_rc)
            for h_idx in range(2):
                x_3d_buf[pl.ds(r0, x_rc), pl.ds(h_idx * 16, 16), :] = x_in_ref[
                    pl.ds(r0, x_rc), pl.ds(h_idx * 2048, 2048)
                ].reshape(x_rc, 16, 128)
            return _

        lax.fori_loop(0, batch_size // x_rc, _reshape_x_chunk, jnp.int32(0))

        pack_active_tokens_smem(
            batch_size, r_scratch, has_replicas=has_replicas
        )
        num_tiles = _build_flat_tile_plan_smem(
            num_exec,
            tile_m,
            r_scratch,
            tile_k_smem,
            tile_base_smem,
            tile_rows_smem,
            tile_flags_smem,
        )

    # --- Per-expert optional global scales (for lossless e4m3 K16 scales) ---
    def _get_gs(w_slot, is_w2: bool):
        if not has_gs:
            return None, None
        if is_w2:
            val = w2_gs_ref[0, w_slot].astype(jnp.float32)
            return val, None
        g_val = w1_gs_ref[0, w_slot].astype(jnp.float32)
        u_val = w1_gs_ref[1, w_slot].astype(jnp.float32)
        return g_val, u_val

    # --- Compute strip helpers ---
    def _run_proj_unit(
        x_src_ref,
        w_slot_ref,
        s_slot_ref,
        m_rows: int,
        is_up: bool,
        gs_val: jax.Array | None = None,
        act_buf=None,
        is_w3_flag=None,
    ):
        """Run full-width Gate (part 0) or Up (part 1) in ``ring*_proj`` mode."""
        act_dst = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, _I, cfg.k256_strip_mode)
            if is_k256
            else _pick_k16_nc(m_rows, _I, cfg.k16_strip_nc, use_dual=False)
        )
        write_m = 16 if m_rows == 8 else m_rows
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None

        def _impl(use_w3_mode: bool):
            def _strip(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(n0, nc)
                res = _fp4_gmm_chunked_v2(
                    lhs_eff,
                    w_slot_ref,
                    s_slot_ref,
                    m_rows,
                    _H,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    kc_step=cfg.kc_step,
                    unpack_mode=cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                    lhs_sc_ref=sc_eff,
                    gs_scale=gs_val,
                    hoist_scales=cfg.hoist_scales,
                    is_w3=use_w3_mode,
                )
                if not is_up:
                    sg_bf = jax.nn.silu(res).astype(jnp.bfloat16)
                    act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                        sg_bf, write_m
                    )
                else:
                    sg_read = act_dst[pl.ds(0, write_m), col_out]
                    if m_rows == 8:
                        sg_read = sg_read[:8, :]
                    h = (sg_read.astype(jnp.float32) * res).astype(jnp.bfloat16)
                    act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
                return carry

            n_strips = _I // nc
            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)
            return jnp.int32(0)

        if has_w3 and is_w3_flag is not None:
            lax.cond(is_w3_flag, lambda: _impl(True), lambda: _impl(False))
        else:
            _impl(False)

    def _run_dual_half_unit(
        x_src_ref,
        w_slot_ref,
        s_slot_ref,
        col_offset: int,
        m_rows: int,
        gs_g: jax.Array | None = None,
        gs_u: jax.Array | None = None,
        act_buf=None,
        is_w3_flag=None,
    ):
        """Run one 2048-col ``[Gate_half | Up_half]`` unit in ``ring*_dual`` mode."""
        act_dst = act_2d_buf if act_buf is None else act_buf
        width = _I // 2
        if is_k256:
            nc = _pick_k256_nc(m_rows, width, cfg.k256_strip_mode)
            dual_fits = (m_rows * nc) // 1024 <= 16
            use_dual = dual_fits and (
                cfg.k256_dual_mode == "dual"
                or (cfg.k256_dual_mode == "hybrid" and m_rows <= 16)
            )
        else:
            want_dual = cfg.k16_dual_mode == "dual" or (
                cfg.k16_dual_mode == "hybrid" and m_rows <= 32
            )
            nc = _pick_k16_nc(m_rows, width, cfg.k16_strip_nc, use_dual=want_dual)
            use_dual = want_dual and ((m_rows * nc) // 1024 <= 16)

        write_m = 16 if m_rows == 8 else m_rows
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None

        def _impl(use_w3_mode: bool):
            def _strip(j, carry):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(col_offset + n0, nc)
                if use_dual:
                    g, u = _fp4_dual_gmm_chunked_v2(
                        lhs_eff,
                        w_slot_ref,
                        s_slot_ref,
                        m_rows,
                        _H,
                        n0,
                        width + n0,
                        nc,
                        is_k256,
                        scale_block,
                        kc_step=cfg.kc_step,
                        unpack_mode=cfg.fp4_unpack_mode,
                        kperm_order=cfg.kperm_k16,
                        lhs_sc_ref=sc_eff,
                        gs_g=gs_g,
                        gs_u=gs_u,
                        hoist_scales=cfg.hoist_scales,
                        is_w3=use_w3_mode,
                    )
                    h = (
                        jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32)
                        * u
                    ).astype(jnp.bfloat16)
                    act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
                else:
                    g = _fp4_gmm_chunked_v2(
                        lhs_eff,
                        w_slot_ref,
                        s_slot_ref,
                        m_rows,
                        _H,
                        n0,
                        nc,
                        is_k256,
                        scale_block,
                        kc_step=cfg.kc_step,
                        unpack_mode=cfg.fp4_unpack_mode,
                        kperm_order=cfg.kperm_k16,
                        lhs_sc_ref=sc_eff,
                        gs_scale=gs_g,
                        hoist_scales=cfg.hoist_scales,
                        is_w3=use_w3_mode,
                    )
                    sg_bf = jax.nn.silu(g).astype(jnp.bfloat16)
                    act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(
                        sg_bf, write_m
                    )
                    u = _fp4_gmm_chunked_v2(
                        lhs_eff,
                        w_slot_ref,
                        s_slot_ref,
                        m_rows,
                        _H,
                        width + n0,
                        nc,
                        is_k256,
                        scale_block,
                        kc_step=cfg.kc_step,
                        unpack_mode=cfg.fp4_unpack_mode,
                        kperm_order=cfg.kperm_k16,
                        lhs_sc_ref=sc_eff,
                        gs_scale=gs_u,
                        hoist_scales=cfg.hoist_scales,
                        is_w3=use_w3_mode,
                    )
                    sg_read = act_dst[pl.ds(0, write_m), col_out]
                    if m_rows == 8:
                        sg_read = sg_read[:8, :]
                    h = (sg_read.astype(jnp.float32) * u).astype(jnp.bfloat16)
                    act_dst[pl.ds(0, write_m), col_out] = _pad_rows_to(h, write_m)
                return carry

            n_strips = width // nc
            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)
            return jnp.int32(0)

        if has_w3 and is_w3_flag is not None:
            lax.cond(is_w3_flag, lambda: _impl(True), lambda: _impl(False))
        else:
            _impl(False)

    def _run_khalf_unit(
        x_src_ref,
        w_kh_ref,
        s_kh_ref,
        kh_part: int,
        m_rows: int,
        gs_g: jax.Array | None = None,
        gs_u: jax.Array | None = None,
    ):
        """Run K-half ``[256, 8192] uint32`` unstrided 8-MiB unit (``K=0..2048`` or ``2048..4096``)."""
        k_half = _H // 2
        x_off = kh_part * k_half
        b_off = kh_part * (k_half // 256)
        if is_k256:
            nc = _pick_k256_nc(m_rows, _I, cfg.k256_strip_mode)
            dual_fits = (m_rows * nc) // 1024 <= 16
            use_dual = dual_fits and (
                cfg.k256_dual_mode == "dual"
                or (cfg.k256_dual_mode == "hybrid" and m_rows <= 16)
            )
        else:
            want_dual = cfg.k16_dual_mode == "dual" or (
                cfg.k16_dual_mode == "hybrid" and m_rows <= 32
            )
            nc = _pick_k16_nc(m_rows, _I, cfg.k16_strip_nc, use_dual=want_dual)
            use_dual = want_dual and ((m_rows * nc) // 1024 <= 16)

        write_m = 16 if m_rows == 8 else m_rows
        gu_rows = max(16, m_rows)
        lhs_eff = lhs_fp8_buf if use_lhs_fp8 else x_src_ref
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None

        def _strip_kh(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_g = pl.ds(n0, nc)
            col_u = pl.ds(_I + n0, nc)
            if use_dual:
                g, u = _fp4_dual_gmm_chunked_v2(
                    lhs_eff,
                    w_kh_ref,
                    s_kh_ref,
                    m_rows,
                    k_half,
                    n0,
                    _I + n0,
                    nc,
                    is_k256,
                    scale_block,
                    kc_step=cfg.kc_step,
                    unpack_mode=cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                    lhs_sc_ref=sc_eff,
                    gs_g=gs_g,
                    gs_u=gs_u,
                    hoist_scales=cfg.hoist_scales,
                    x_col_base=x_off,
                    b_offset=b_off,
                )
            else:
                g = _fp4_gmm_chunked_v2(
                    lhs_eff,
                    w_kh_ref,
                    s_kh_ref,
                    m_rows,
                    k_half,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    kc_step=cfg.kc_step,
                    unpack_mode=cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                    lhs_sc_ref=sc_eff,
                    gs_scale=gs_g,
                    hoist_scales=cfg.hoist_scales,
                    x_col_base=x_off,
                    b_offset=b_off,
                )
                u = _fp4_gmm_chunked_v2(
                    lhs_eff,
                    w_kh_ref,
                    s_kh_ref,
                    m_rows,
                    k_half,
                    _I + n0,
                    nc,
                    is_k256,
                    scale_block,
                    kc_step=cfg.kc_step,
                    unpack_mode=cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                    lhs_sc_ref=sc_eff,
                    gs_scale=gs_u,
                    hoist_scales=cfg.hoist_scales,
                    x_col_base=x_off,
                    b_offset=b_off,
                )
            if kh_part == 0:
                gu_kh_buf[pl.ds(0, gu_rows), col_g] = _pad_rows_to(g, gu_rows)
                gu_kh_buf[pl.ds(0, gu_rows), col_u] = _pad_rows_to(u, gu_rows)
            else:
                g0 = gu_kh_buf[pl.ds(0, gu_rows), col_g][:m_rows, :]
                u0 = gu_kh_buf[pl.ds(0, gu_rows), col_u][:m_rows, :]
                g_tot = g0 + g
                u_tot = u0 + u
                h = (
                    jax.nn.silu(g_tot).astype(jnp.bfloat16).astype(jnp.float32)
                    * u_tot
                ).astype(jnp.bfloat16)
                act_2d_buf[pl.ds(0, write_m), col_g] = _pad_rows_to(h, write_m)
            return carry

        n_strips = _I // nc
        if n_strips == 1:
            _strip_kh(0, 0)
        elif n_strips == 2:
            _strip_kh(0, 0)
            _strip_kh(1, 0)
        else:
            lax.fori_loop(0, n_strips, _strip_kh, 0)

    def _run_down_unit(
        w_slot_ref,
        s_slot_ref,
        m_rows: int,
        sq_sum: jax.Array,
        gs_w2: jax.Array | None = None,
        act_buf=None,
        y_row_base: int | jax.Array = 0,
        is_w3_flag=None,
    ) -> jax.Array:
        """Run full 4096-col Down projection ``W2`` from ``w_slot_ref``."""
        act_src = act_2d_buf if act_buf is None else act_buf
        nc = (
            _pick_k256_nc(m_rows, _H, cfg.k256_strip_mode)
            if is_k256
            else _pick_k16_nc(m_rows, _H, cfg.k16_strip_nc, use_dual=False)
        )
        write_m = (
            m_rows if cfg.norm_f32_stage else (16 if m_rows == 8 else m_rows)
        )
        act_eff = lhs_fp8_buf if use_lhs_fp8 else act_src
        sc_eff = lhs_sc_buf if use_lhs_fp8 else None
        if use_lhs_fp8:
            _prequant_lhs_k256(act_src, lhs_fp8_buf, lhs_sc_buf, m_rows)

        def _impl(use_w3_mode: bool) -> jax.Array:
            def _strip(j, sq_acc):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(n0, nc)
                d = _fp4_gmm_chunked_v2(
                    act_eff,
                    w_slot_ref,
                    s_slot_ref,
                    m_rows,
                    _I,
                    n0,
                    nc,
                    is_k256,
                    scale_block,
                    kc_step=cfg.kc_step,
                    unpack_mode=cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                    lhs_sc_ref=sc_eff,
                    gs_scale=gs_w2,
                    hoist_scales=cfg.hoist_scales,
                    is_w3=use_w3_mode,
                ).astype(jnp.bfloat16)
                g_col = gain_f32_buf[0:1, col_out]
                if cfg.norm_f32_stage:
                    u = d.astype(jnp.float32) * g_col
                    sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                    y_stage_buf[pl.ds(y_row_base, write_m), col_out] = u
                else:
                    u_bf = d * g_col.astype(jnp.bfloat16)
                    u_f32 = u_bf.astype(jnp.float32)
                    sq_acc = sq_acc + jnp.sum(u_f32 * u_f32, axis=1, keepdims=True)
                    y_stage_buf[pl.ds(y_row_base, write_m), col_out] = _pad_rows_to(
                        u_bf, write_m
                    )
                return sq_acc

            n_strips = _H // nc
            if n_strips == 1:
                return _strip(0, sq_sum)
            if n_strips == 2:
                s0 = _strip(0, sq_sum)
                return _strip(1, s0)
            return lax.fori_loop(0, n_strips, _strip, sq_sum)

        if has_w3 and is_w3_flag is not None:
            return lax.cond(is_w3_flag, lambda: _impl(True), lambda: _impl(False))
        return _impl(False)

    # --- VMEM-dequantized multi-M32 reuse helpers (K16 & K256 cur_rows > 32, 1x unpack!) ---
    def _run_dual_half_vmem_deq(
        w_slot_ref,
        s_slot_ref,
        col_offset: int,
        n_m32: jax.Array,
        gs_g: jax.Array | None = None,
        gs_u: jax.Array | None = None,
        is_w3: bool = False,
    ):
        width = _I // 2
        nc = 1024 if is_k256 else min(cfg.k16_strip_nc, 512, width)

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(col_offset + n0, nc)
            if is_k256:
                _unpack_strip_to_vmem_k256(
                    w_slot_ref, w_deq_buf, _H, n0, nc, is_w3=is_w3
                )
            else:
                _dequant_strip_to_vmem_k16(
                    w_slot_ref,
                    s_slot_ref,
                    w_deq_buf,
                    _H,
                    n0,
                    nc,
                    cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                )

            def _gate_m(mi, _):
                r0 = pl.multiple_of(mi * 32, 32)
                g = _dot_from_vmem_deq(
                    lhs_2d_buf,
                    w_deq_buf,
                    r0,
                    32,
                    _H,
                    nc,
                    gs_g,
                    s_k256_ref=s_slot_ref if is_k256 else None,
                    n0_s=n0,
                    hoist_scales=cfg.hoist_scales,
                    scale_block=scale_block,
                )
                act_2d_buf[pl.ds(r0, 32), col_out] = jax.nn.silu(g).astype(
                    jnp.bfloat16
                )
                return _

            lax.fori_loop(0, n_m32, _gate_m, jnp.int32(0))

            if is_k256:
                _unpack_strip_to_vmem_k256(
                    w_slot_ref, w_deq_buf, _H, width + n0, nc, is_w3=is_w3
                )
            else:
                _dequant_strip_to_vmem_k16(
                    w_slot_ref,
                    s_slot_ref,
                    w_deq_buf,
                    _H,
                    width + n0,
                    nc,
                    cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                )

            def _up_m(mi, _):
                r0 = pl.multiple_of(mi * 32, 32)
                u = _dot_from_vmem_deq(
                    lhs_2d_buf,
                    w_deq_buf,
                    r0,
                    32,
                    _H,
                    nc,
                    gs_u,
                    s_k256_ref=s_slot_ref if is_k256 else None,
                    n0_s=width + n0,
                    hoist_scales=cfg.hoist_scales,
                    scale_block=scale_block,
                )
                sg = act_2d_buf[pl.ds(r0, 32), col_out].astype(jnp.float32)
                act_2d_buf[pl.ds(r0, 32), col_out] = (sg * u).astype(jnp.bfloat16)
                return _

            lax.fori_loop(0, n_m32, _up_m, jnp.int32(0))
            return carry

        lax.fori_loop(0, width // nc, _strip, 0)

    def _run_proj_vmem_deq(
        w_slot_ref,
        s_slot_ref,
        n_m32: jax.Array,
        is_up: bool,
        gs_val: jax.Array | None = None,
        is_w3: bool = False,
    ):
        nc = 1024 if is_k256 else min(cfg.k16_strip_nc, 512)

        def _strip(j, carry):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            if is_k256:
                _unpack_strip_to_vmem_k256(
                    w_slot_ref, w_deq_buf, _H, n0, nc, is_w3=is_w3
                )
            else:
                _dequant_strip_to_vmem_k16(
                    w_slot_ref,
                    s_slot_ref,
                    w_deq_buf,
                    _H,
                    n0,
                    nc,
                    cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                )

            def _proj_m(mi, _):
                r0 = pl.multiple_of(mi * 32, 32)
                res = _dot_from_vmem_deq(
                    lhs_2d_buf,
                    w_deq_buf,
                    r0,
                    32,
                    _H,
                    nc,
                    gs_val,
                    s_k256_ref=s_slot_ref if is_k256 else None,
                    n0_s=n0,
                    hoist_scales=cfg.hoist_scales,
                    scale_block=scale_block,
                )
                if not is_up:
                    act_2d_buf[pl.ds(r0, 32), col_out] = jax.nn.silu(res).astype(
                        jnp.bfloat16
                    )
                else:
                    sg = act_2d_buf[pl.ds(r0, 32), col_out].astype(jnp.float32)
                    act_2d_buf[pl.ds(r0, 32), col_out] = (sg * res).astype(
                        jnp.bfloat16
                    )
                return _

            lax.fori_loop(0, n_m32, _proj_m, jnp.int32(0))
            return carry

        lax.fori_loop(0, _I // nc, _strip, 0)

    def _run_down_vmem_deq(
        w_slot_ref,
        s_slot_ref,
        n_m32: jax.Array,
        gs_w2: jax.Array | None = None,
        is_w3: bool = False,
    ) -> jax.Array:
        """Unpack each W2 strip ONCE to VMEM and accumulate ``sq_sum [tile_m, 1]``."""
        nc = 1024 if is_k256 else min(cfg.k16_strip_nc, 512)
        sq_init = jnp.zeros((tile_m, 1), dtype=jnp.float32)
        row_iota = lax.broadcasted_iota(jnp.int32, (tile_m, 1), 0)

        def _strip(j, sq_acc):
            n0 = pl.multiple_of(j * nc, nc)
            col_out = pl.ds(n0, nc)
            if is_k256:
                _unpack_strip_to_vmem_k256(
                    w_slot_ref, w_deq_buf, _I, n0, nc, is_w3=is_w3
                )
            else:
                _dequant_strip_to_vmem_k16(
                    w_slot_ref,
                    s_slot_ref,
                    w_deq_buf,
                    _I,
                    n0,
                    nc,
                    cfg.fp4_unpack_mode,
                    kperm_order=cfg.kperm_k16,
                )
            g_col = gain_f32_buf[0:1, col_out]

            def _down_m(mi, sq_in):
                r0 = pl.multiple_of(mi * 32, 32)
                d = _dot_from_vmem_deq(
                    act_2d_buf,
                    w_deq_buf,
                    r0,
                    32,
                    _I,
                    nc,
                    gs_w2,
                    s_k256_ref=s_slot_ref if is_k256 else None,
                    n0_s=n0,
                    hoist_scales=cfg.hoist_scales,
                    scale_block=scale_block,
                ).astype(jnp.bfloat16)
                u = d.astype(jnp.float32) * g_col
                y_stage_buf[pl.ds(r0, 32), col_out] = u
                part_sq = jnp.sum(u * u, axis=1, keepdims=True)
                part_pad = jnp.broadcast_to(
                    part_sq, (tile_m // 32, 32, 1)
                ).reshape(tile_m, 1)
                in_slice = (row_iota >= r0) & (row_iota < r0 + 32)
                return sq_in + jnp.where(in_slice, part_pad, 0.0)

            return lax.fori_loop(0, n_m32, _down_m, sq_acc)

        return lax.fori_loop(0, _H // nc, _strip, sq_init)

    # =========================================================================
    # 6A. DENSE-ROWS 1D FLATTENED CIRCULAR UNIT LOOP (exact moe_lowtok pattern!)
    # =========================================================================
    if is_dense:
        lane_128 = lax.broadcasted_iota(jnp.int32, (batch_size, 128), 1)

        def _dense_unit_body(u, carry):
            e_idx, part = carry
            r_slot = _ring_slot(u, n_slots)

            # 1. Prefetch unit u + (n_slots - 1) AT TOP OF LOOP BEFORE wait/compute!
            pf_u = u + (n_slots - 1)

            @pl.when(pf_u < n_units_total)
            def _pf():
                pf_e, pf_p = _advance_unit(e_idx, part, n_slots - 1)
                pf_rslot = _ring_slot(pf_u, n_slots)
                start_ring_unit(
                    pf_rslot,
                    exec_src_smem[pf_e],
                    exec_slot_smem[pf_e],
                    pf_p,
                )

            w_src_cur = exec_src_smem[e_idx]
            w_slot_cur = exec_slot_smem[e_idx]
            cur_is_w3 = _slot_is_w3(w_src_cur, w_slot_cur)

            # 2. Wait for current unit's circular slot.
            wait_ring_slot(r_slot, cur_is_w3)

            gs_g, gs_u = _get_gs(w_slot_cur, False)
            gs_w2, _ = _get_gs(w_slot_cur, True)

            if dual_units:
                @pl.when(part == 0)
                def _d_u0():
                    _run_dual_half_unit(
                        x_in_ref,
                        w_ring.at[r_slot],
                        s_ring.at[r_slot],
                        0,
                        batch_size,
                        gs_g,
                        gs_u,
                        is_w3_flag=cur_is_w3,
                    )

                @pl.when(part == 1)
                def _d_u1():
                    _run_dual_half_unit(
                        x_in_ref,
                        w_ring.at[r_slot],
                        s_ring.at[r_slot],
                        _I // 2,
                        batch_size,
                        gs_g,
                        gs_u,
                        is_w3_flag=cur_is_w3,
                    )
            else:
                @pl.when(part == 0)
                def _p_u0():
                    _run_proj_unit(
                        x_in_ref,
                        w_ring.at[r_slot],
                        s_ring.at[r_slot],
                        batch_size,
                        False,
                        gs_g,
                        is_w3_flag=cur_is_w3,
                    )

                @pl.when(part == 1)
                def _p_u1():
                    _run_proj_unit(
                        x_in_ref,
                        w_ring.at[r_slot],
                        s_ring.at[r_slot],
                        batch_size,
                        True,
                        gs_u,
                        is_w3_flag=cur_is_w3,
                    )

            @pl.when(part == 2)
            def _down_and_acc():
                sq_sum = _run_down_unit(
                    w_ring.at[r_slot],
                    s_ring.at[r_slot],
                    batch_size,
                    jnp.zeros((batch_size, 1), dtype=jnp.float32),
                    gs_w2,
                    is_w3_flag=cur_is_w3,
                )
                g_e = exec_eid_smem[e_idx]
                blk = pl.multiple_of((g_e >> 7) << 7, 128)
                w_slice = d_scratch.dense_w_vmem[:, pl.ds(blk, 128)]
                w_col = jnp.sum(
                    jnp.where((lane_128 + blk) == g_e, w_slice, 0.0),
                    axis=1,
                    keepdims=True,
                )
                inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
                rc = min(batch_size, 8 if cfg.norm_f32_stage else 16)
                for r0 in range(0, batch_size, rc):
                    inv_r = inv_rms[r0 : r0 + rc, :]
                    w_r = w_col[r0 : r0 + rc, :]
                    for s_col in range(0, _H, 1024):
                        col_s = pl.ds(s_col, 1024)
                        u_c = y_stage_buf[pl.ds(r0, rc), col_s]
                        if cfg.norm_f32_stage:
                            v_bf = (u_c * inv_r).astype(jnp.bfloat16)
                        else:
                            v_bf = u_c * inv_r.astype(jnp.bfloat16)
                        contrib = jnp.where(
                            w_r > 0.0, v_bf.astype(jnp.float32) * w_r, 0.0
                        )
                        accum_2d[pl.ds(r0, rc), col_s] = (
                            accum_2d[pl.ds(r0, rc), col_s] + contrib
                        )

            return _advance_unit(e_idx, part, 1)

        lax.fori_loop(
            0, n_units_total, _dense_unit_body, (jnp.int32(0), jnp.int32(0))
        )

        conv_rc = min(batch_size, 8)
        for r0 in range(0, batch_size, conv_rc):
            for h_idx in range(2):
                accum_3d[pl.ds(r0, conv_rc), pl.ds(h_idx * 16, 16), :] = (
                    accum_2d[
                        pl.ds(r0, conv_rc), pl.ds(h_idx * 2048, 2048)
                    ].reshape(conv_rc, 16, 128)
                )

    # =========================================================================
    # 6A-2. ZERO-SALU MXU PERMUTATION-GATHER & SCATTER SPARSE LOOP (#439/#440)
    # =========================================================================
    elif is_mxu_sparse:
        lane_128 = lax.broadcasted_iota(jnp.int32, (batch_size, 128), 1)
        tri_b = (
            lax.broadcasted_iota(jnp.int32, (batch_size, batch_size), 1)
            < lax.broadcasted_iota(jnp.int32, (batch_size, batch_size), 0)
        ).astype(jnp.float32)
        tm_iota = lax.broadcasted_iota(jnp.int32, (batch_size, tile_m), 1)
        b_step = min(batch_size, 64)
        m_buckets_mxu = cfg.m_buckets if cfg.m_buckets else (tile_m,)

        def _mxu_expert_body(k_exp, _):
            g_e = exec_eid_smem[k_exp]
            r_e = counts_smem[0, g_e]
            n_rt = (r_e + tile_m - 1) // tile_m

            has_k1 = (k_exp + 1) < num_exec
            k1_clamped = jnp.where(has_k1, k_exp + 1, jnp.int32(0))
            k1_src = exec_src_smem[k1_clamped]
            k1_slot = exec_slot_smem[k1_clamped]

            has_k2 = (k_exp + 2) < num_exec
            k2_clamped = jnp.where(has_k2, k_exp + 2, jnp.int32(0))
            k2_src = exec_src_smem[k2_clamped]
            k2_slot = exec_slot_smem[k2_clamped]

            u_base = k_exp * 3
            s0 = _ring_slot(u_base, n_slots)
            s1 = _ring_slot(u_base + 1, n_slots)
            s2 = _ring_slot(u_base + 2, n_slots)

            # Extract w_col [B, 1] and compute within-expert rank pos_col [B, 1] via 1 MXU op!
            blk = pl.multiple_of((g_e >> 7) << 7, 128)
            w_slice = d_scratch.dense_w_vmem[:, pl.ds(blk, 128)]
            w_col = jnp.sum(
                jnp.where((lane_128 + blk) == g_e, w_slice, 0.0),
                axis=1,
                keepdims=True,
            )
            hit_col = w_col > 0.0
            pos_col = jnp.dot(
                tri_b,
                hit_col.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)
            # 2-limb exact FP32->BF16 split for bit-accurate router weight scaling on MXU!
            w_hi = w_col.astype(jnp.bfloat16)
            w_lo = (w_col - w_hi.astype(jnp.float32)).astype(jnp.bfloat16)

            def _mxu_tile_body(ti, _c):
                off = ti * tile_m
                cur_rows = jnp.minimum(tile_m, r_e - off)
                is_first = ti == 0
                is_last = ti == (n_rt - 1)
                do_pf_k1 = jnp.logical_and(is_last, has_k1)
                do_pf_k2 = jnp.logical_and(is_last, has_k2)

                @pl.when(
                    jnp.logical_and(
                        jnp.logical_and(k_exp == 0, is_first),
                        (n_slots - 1) < n_units_total,
                    )
                )
                def _pf_init_mxu():
                    if n_slots == 3:
                        start_ring_unit(
                            jnp.int32(2),
                            exec_src_smem[0],
                            exec_slot_smem[0],
                            jnp.int32(2),
                        )
                    elif n_slots == 4:
                        start_ring_unit(
                            jnp.int32(3), k1_src, k1_slot, jnp.int32(0)
                        )
                    else:
                        start_ring_unit(
                            jnp.int32(5), k1_src, k1_slot, jnp.int32(2)
                        )

                pos_rel = pos_col - off
                g_bool = (
                    hit_col
                    & (pos_rel >= 0)
                    & (pos_rel < tile_m)
                    & (pos_rel == tm_iota)
                )  # [B, tile_m]
                g_bf = g_bool.astype(jnp.bfloat16)
                g_bf_t = jnp.transpose(g_bf)  # [tile_m, B]

                # 1. MXU Permutation Gather -> lhs_2d_buf [tile_m, 4096] (~0.05 us!)
                for c0 in range(0, _H, 1024):
                    col_s = pl.ds(c0, 1024)
                    acc_g = None
                    for b0 in range(0, batch_size, b_step):
                        part_g = jnp.dot(
                            g_bf_t[:, b0 : b0 + b_step],
                            x_clean_2d[pl.ds(b0, b_step), col_s],
                            preferred_element_type=jnp.float32,
                        )
                        acc_g = part_g if acc_g is None else acc_g + part_g
                    lhs_2d_buf[:, col_s] = acc_g.astype(jnp.bfloat16)

                w_src_cur = exec_src_smem[k_exp]
                w_slot_cur = exec_slot_smem[k_exp]
                cur_is_w3 = _slot_is_w3(w_src_cur, w_slot_cur)
                gs_g, gs_u = _get_gs(w_slot_cur, False)
                gs_w2, _unused_gs2 = _get_gs(w_slot_cur, True)

                def _call_mxu_u01(p_idx: int, r_sl: jax.Array):
                    def _do_m(mb: int):
                        if use_lhs_fp8 and p_idx == 0:
                            _prequant_lhs_k256(
                                lhs_2d_buf, lhs_fp8_buf, lhs_sc_buf, mb
                            )
                        if dual_units:
                            _run_dual_half_unit(
                                lhs_2d_buf,
                                w_ring.at[r_sl],
                                s_ring.at[r_sl],
                                p_idx * (_I // 2),
                                mb,
                                gs_g,
                                gs_u,
                                is_w3_flag=cur_is_w3,
                            )
                        else:
                            _run_proj_unit(
                                lhs_2d_buf,
                                w_ring.at[r_sl],
                                s_ring.at[r_sl],
                                mb,
                                p_idx == 1,
                                gs_u if p_idx == 1 else gs_g,
                                is_w3_flag=cur_is_w3,
                            )

                    if len(m_buckets_mxu) > 1:
                        bi = jnp.int32(0)
                        for _b in m_buckets_mxu[:-1]:
                            bi = bi + (cur_rows > _b).astype(jnp.int32)
                        lax.switch(
                            bi,
                            [
                                (lambda b=b: (_do_m(b), jnp.int32(0))[1])
                                for b in m_buckets_mxu
                            ],
                        )
                    else:
                        _do_m(m_buckets_mxu[0])

                @pl.when(is_first)
                def _mw0():
                    wait_ring_slot(s0, cur_is_w3)

                _call_mxu_u01(0, s0)

                if n_slots == 6:
                    @pl.when(do_pf_k2)
                    def _mpf0_6():
                        start_ring_unit(s0, k2_src, k2_slot, jnp.int32(0))
                else:
                    @pl.when(do_pf_k1)
                    def _mpf0():
                        start_ring_unit(
                            s0, k1_src, k1_slot, jnp.int32(0 if n_slots == 3 else 1)
                        )

                @pl.when(is_first)
                def _mw1():
                    wait_ring_slot(s1, cur_is_w3)

                _call_mxu_u01(1, s1)

                if n_slots == 6:
                    @pl.when(do_pf_k2)
                    def _mpf1_6():
                        start_ring_unit(s1, k2_src, k2_slot, jnp.int32(1))
                else:
                    @pl.when(do_pf_k1)
                    def _mpf1():
                        start_ring_unit(
                            s1, k1_src, k1_slot, jnp.int32(1 if n_slots == 3 else 2)
                        )

                @pl.when(is_first)
                def _mw2():
                    wait_ring_slot(s2, cur_is_w3)

                if len(m_buckets_mxu) > 1:
                    bi2 = jnp.int32(0)
                    for _b in m_buckets_mxu[:-1]:
                        bi2 = bi2 + (cur_rows > _b).astype(jnp.int32)
                    sq_tile = lax.switch(
                        bi2,
                        [
                            (
                                lambda b=b: _pad_rows_to(
                                    _run_down_unit(
                                        w_ring.at[s2],
                                        s_ring.at[s2],
                                        b,
                                        jnp.zeros((b, 1), dtype=jnp.float32),
                                        gs_w2,
                                        is_w3_flag=cur_is_w3,
                                    ),
                                    y_stage_buf.shape[0],
                                )
                            )
                            for b in m_buckets_mxu
                        ],
                    )
                else:
                    mb0 = m_buckets_mxu[0]
                    sq_tile = _pad_rows_to(
                        _run_down_unit(
                            w_ring.at[s2],
                            s_ring.at[s2],
                            mb0,
                            jnp.zeros((mb0, 1), dtype=jnp.float32),
                            gs_w2,
                            is_w3_flag=cur_is_w3,
                        ),
                        y_stage_buf.shape[0],
                    )

                if n_slots == 3:
                    @pl.when(do_pf_k1)
                    def _mpf2_3():
                        start_ring_unit(s2, k1_src, k1_slot, jnp.int32(2))
                elif n_slots == 4:
                    @pl.when(do_pf_k2)
                    def _mpf2_4():
                        start_ring_unit(s2, k2_src, k2_slot, jnp.int32(0))
                else:
                    @pl.when(do_pf_k2)
                    def _mpf2_6():
                        start_ring_unit(s2, k2_src, k2_slot, jnp.int32(2))

                # 2. MXU Permutation Scatter-Accumulate -> accum_2d [B, 4096] (~0.08 us!)
                inv_rms_t = lax.rsqrt(sq_tile[:tile_m, :] * (1.0 / _H) + 1e-8)
                valid_m_mask = (
                    lax.broadcasted_iota(jnp.int32, (tile_m, 1024), 0) < cur_rows
                )
                s_hi = g_bf * w_hi  # [B, tile_m]
                s_lo = g_bf * w_lo  # [B, tile_m]
                for c0 in range(0, _H, 1024):
                    col_s = pl.ds(c0, 1024)
                    u_c = y_stage_buf[pl.ds(0, tile_m), col_s]
                    v_bf = (
                        (u_c * inv_rms_t).astype(jnp.bfloat16)
                        if cfg.norm_f32_stage
                        else (u_c * inv_rms_t.astype(jnp.bfloat16))
                    )
                    # Mask inactive rows >= cur_rows so uninitialized VMEM NaNs never hit 0 * NaN = NaN!
                    v_bf = jnp.where(valid_m_mask, v_bf, jnp.bfloat16(0.0))
                    for b0 in range(0, batch_size, b_step):
                        row_b = pl.ds(b0, b_step)
                        c_hi = jnp.dot(
                            s_hi[b0 : b0 + b_step, :],
                            v_bf,
                            preferred_element_type=jnp.float32,
                        )
                        c_lo = jnp.dot(
                            s_lo[b0 : b0 + b_step, :],
                            v_bf,
                            preferred_element_type=jnp.float32,
                        )
                        accum_2d[row_b, col_s] = (
                            accum_2d[row_b, col_s] + c_hi + c_lo
                        )
                return _c

            lax.fori_loop(0, n_rt, _mxu_tile_body, jnp.int32(0))
            return _

        lax.fori_loop(0, num_exec, _mxu_expert_body, jnp.int32(0))

        conv_rc = min(batch_size, 8)
        for r0 in range(0, batch_size, conv_rc):
            for h_idx in range(2):
                accum_3d[pl.ds(r0, conv_rc), pl.ds(h_idx * 16, 16), :] = (
                    accum_2d[
                        pl.ds(r0, conv_rc), pl.ds(h_idx * 2048, 2048)
                    ].reshape(conv_rc, 16, 128)
                )

    # =========================================================================
    # 6B. SPARSE FLATTENED TILE LOOP + CIRCULAR RING + ASYNC GATHER + FUSED NORM-SCATTER
    # =========================================================================
    else:
        def _start_async_gather(buf_idx, base_pos, nrows):
            def _one(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                pltpu.make_async_copy(
                    x_3d_buf.at[pl.ds(tok, 1), :, :],
                    gather_3d_buf.at[buf_idx, pl.ds(i, 1), :, :],
                    gather_sem.at[buf_idx],
                ).start()
                return _

            lax.fori_loop(0, nrows, _one, jnp.int32(0))

        def _wait_async_gather(buf_idx, nrows):
            tmpl = pltpu.make_async_copy(
                gather_3d_buf.at[buf_idx, pl.ds(0, 1), :, :],
                gather_3d_buf.at[buf_idx, pl.ds(0, 1), :, :],
                gather_sem.at[buf_idx],
            )

            def _one(_, c):
                tmpl.wait()
                return c

            lax.fori_loop(0, nrows, _one, jnp.int32(0))

        def _sync_gather_to_3d(dst_3d_ref, base_pos, cur_rows):
            n8 = cur_rows >> 3

            def _g8(q, _):
                i0 = pl.multiple_of(q << 3, 8)
                p0 = base_pos + i0
                ts = [r_scratch.packed_tok_ids[p0 + s] for s in range(8)]
                rows = [x_3d_buf[pl.ds(t, 1), :, :] for t in ts]
                dst_3d_ref[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
                return _

            lax.fori_loop(0, n8, _g8, jnp.int32(0))

            def _gtail(i, _):
                tok = r_scratch.packed_tok_ids[base_pos + i]
                dst_3d_ref[pl.ds(i, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
                return _

            lax.fori_loop(n8 << 3, cur_rows, _gtail, jnp.int32(0))

        def _copy_3d_to_lhs_2d(src_3d_ref, r_base, m_rows: int):
            copy_m = 16 if m_rows == 8 else m_rows
            for r0 in range(0, copy_m, 16):
                rr = r_base + r0
                for h_idx in range(2):
                    lhs_2d_buf[pl.ds(rr, 16), pl.ds(h_idx * 2048, 2048)] = (
                        src_3d_ref[
                            pl.ds(rr, 16), pl.ds(h_idx * 16, 16), :
                        ].reshape(16, 2048)
                    )

        def _fused_norm_and_scatter(base_pos, cur_rows, sq_sum_full: jax.Array):
            """Single-pass ``y_stage_buf`` -> ``inv_rms`` BF16 -> ``accum_3d`` scatter (zero VREG dynamic_slice)."""
            del sq_sum_full
            n8 = (cur_rows + 7) >> 3

            def _norm_scatter_8(q, carry_ns):
                i0 = pl.multiple_of(q << 3, 8)
                u0 = y_stage_buf[pl.ds(i0, 8), pl.ds(0, 2048)]
                u1 = y_stage_buf[pl.ds(i0, 8), pl.ds(2048, 2048)]
                u0_f = u0.astype(jnp.float32)
                u1_f = u1.astype(jnp.float32)
                sq8 = jnp.sum(u0_f * u0_f, axis=1, keepdims=True) + jnp.sum(
                    u1_f * u1_f, axis=1, keepdims=True
                )
                inv_8 = lax.rsqrt(sq8 * (1.0 / _H) + 1e-8)
                if cfg.norm_f32_stage:
                    y0 = (u0_f * inv_8).astype(jnp.bfloat16).reshape(8, 16, 128)
                    y1 = (u1_f * inv_8).astype(jnp.bfloat16).reshape(8, 16, 128)
                else:
                    inv8_bf = inv_8.astype(jnp.bfloat16)
                    y0 = (u0 * inv8_bf).reshape(8, 16, 128)
                    y1 = (u1 * inv8_bf).reshape(8, 16, 128)
                y8_3d = jnp.concatenate([y0, y1], axis=1)

                for s in range(8):
                    idx_s = i0 + s

                    @pl.when(idx_s < cur_rows)
                    def _do_add(s=s, idx_s=idx_s):
                        tok = r_scratch.packed_tok_ids[base_pos + idx_s]
                        w = r_scratch.packed_tok_w[base_pos + idx_s]
                        ys = y8_3d[s : s + 1, :, :].astype(jnp.float32) * w
                        accum_3d[pl.ds(tok, 1), :, :] = (
                            accum_3d[pl.ds(tok, 1), :, :] + ys
                        )

                return carry_ns

            lax.fori_loop(0, n8, _norm_scatter_8, jnp.int32(0))

        def _legacy_norm_then_scatter(base_pos, cur_rows, m_rows_pad: int, sq_sum: jax.Array):
            inv_rms = lax.rsqrt(sq_sum * (1.0 / _H) + 1e-8)
            for r0 in range(0, m_rows_pad, 8):
                inv_r = inv_rms[r0 : r0 + 8, :]
                for h_idx in range(2):
                    u_c = y_stage_buf[pl.ds(r0, 8), pl.ds(h_idx * 2048, 2048)]
                    v_bf = (
                        (u_c * inv_r).astype(jnp.bfloat16)
                        if cfg.norm_f32_stage
                        else (u_c * inv_r.astype(jnp.bfloat16))
                    )
                    io_3d_buf[pl.ds(r0, 8), pl.ds(h_idx * 16, 16), :] = (
                        v_bf.reshape(8, 16, 128)
                    )
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

        # --- Block-diagonal K16 unit helpers ---
        def _bd_unit01(
            w_slot_ref,
            s_slot_ref,
            part_idx: int,
            bd_rows: int,
            gs_g: jax.Array | None,
            gs_u: jax.Array | None,
        ):
            if dual_units:
                width = _I // 2
                col_offset = part_idx * width
                nc = _pick_bd_nc(bd_rows, width, cfg.k16_bd_narrow_nc)
                use_dual_bd = (bd_rows * nc) // 128 <= 4

                def _strip(j, carry):
                    n0 = pl.multiple_of(j * nc, nc)
                    col_out = pl.ds(col_offset + n0, nc)
                    if use_dual_bd:
                        g, u = _fp4_blockdiag_hoisted_dual_strip_v2(
                            masked_x_buf,
                            w_slot_ref,
                            s_slot_ref,
                            bd_rows,
                            n0,
                            width + n0,
                            nc,
                            gs_g,
                            gs_u,
                        )
                        h = (
                            jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32)
                            * u
                        ).astype(jnp.bfloat16)
                        act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(h, 16)
                    else:
                        g = _fp4_blockdiag_hoisted_strip_v2(
                            masked_x_buf, w_slot_ref, s_slot_ref, bd_rows, n0, nc, gs_g
                        )
                        act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                            jax.nn.silu(g).astype(jnp.bfloat16), 16
                        )
                        u = _fp4_blockdiag_hoisted_strip_v2(
                            masked_x_buf,
                            w_slot_ref,
                            s_slot_ref,
                            bd_rows,
                            width + n0,
                            nc,
                            gs_u,
                        )
                        sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                            jnp.float32
                        )
                        act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                            (sg * u).astype(jnp.bfloat16), 16
                        )
                    return carry

                n_strips = width // nc
            else:
                nc = _pick_bd_nc(bd_rows, _I, cfg.k16_bd_narrow_nc)
                is_up = part_idx == 1
                gs_eff = gs_u if is_up else gs_g

                def _strip(j, carry):
                    n0 = pl.multiple_of(j * nc, nc)
                    col_out = pl.ds(n0, nc)
                    res = _fp4_blockdiag_hoisted_strip_v2(
                        masked_x_buf, w_slot_ref, s_slot_ref, bd_rows, n0, nc, gs_eff
                    )
                    if not is_up:
                        act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                            jax.nn.silu(res).astype(jnp.bfloat16), 16
                        )
                    else:
                        sg = act_2d_buf[pl.ds(0, 16), col_out][:bd_rows, :].astype(
                            jnp.float32
                        )
                        act_2d_buf[pl.ds(0, 16), col_out] = _pad_rows_to(
                            (sg * res).astype(jnp.bfloat16), 16
                        )
                    return carry

                n_strips = _I // nc

            if n_strips == 1:
                _strip(0, 0)
            elif n_strips == 2:
                _strip(0, 0)
                _strip(1, 0)
            else:
                lax.fori_loop(0, n_strips, _strip, 0)

        def _bd_unit2(
            w_slot_ref,
            s_slot_ref,
            bd_rows: int,
            gs_w2: jax.Array | None,
        ) -> jax.Array:
            nc = _pick_bd_nc(bd_rows, _H, cfg.k16_bd_narrow_nc)
            pad_f32 = 16 if bd_rows > 8 else 8
            sq_init = jnp.zeros((bd_rows, 1), dtype=jnp.float32)

            def _strip(j, sq_acc):
                n0 = pl.multiple_of(j * nc, nc)
                col_out = pl.ds(n0, nc)
                d = _fp4_blockdiag_hoisted_strip_v2(
                    masked_x_buf, w_slot_ref, s_slot_ref, bd_rows, n0, nc, gs_w2
                ).astype(jnp.bfloat16)
                g_col = gain_f32_buf[0:1, col_out]
                if cfg.norm_f32_stage:
                    u = d.astype(jnp.float32) * g_col
                    sq_acc = sq_acc + jnp.sum(u * u, axis=1, keepdims=True)
                    y_stage_buf[pl.ds(0, pad_f32), col_out] = _pad_rows_to(
                        u, pad_f32
                    )
                else:
                    u_bf = d * g_col.astype(jnp.bfloat16)
                    u_f32 = u_bf.astype(jnp.float32)
                    sq_acc = sq_acc + jnp.sum(
                        u_f32 * u_f32, axis=1, keepdims=True
                    )
                    y_stage_buf[pl.ds(0, 16), col_out] = _pad_rows_to(u_bf, 16)
                return sq_acc

            n_strips = _H // nc
            if n_strips == 1:
                sq_bd = _strip(0, sq_init)
            elif n_strips == 2:
                sq_bd = _strip(1, _strip(0, sq_init))
            else:
                sq_bd = lax.fori_loop(0, n_strips, _strip, sq_init)
            return _pad_rows_to(sq_bd, y_stage_buf.shape[0])

        bd_rows_tuple = cfg.k16_blockdiag_rows if use_bd else ()
        m_buckets = (
            tuple(b for b in cfg.m_buckets if b > bd_rows_tuple[-1])
            if bd_rows_tuple
            else cfg.m_buckets
        )
        if not m_buckets:
            m_buckets = (tile_m,)

        # Prime first tile's async gather right after SMEM plan!
        if cfg.async_gather:
            @pl.when(num_tiles > 0)
            def _prime_gather_0():
                _start_async_gather(0, tile_base_smem[0], tile_rows_smem[0])

        def _dispatch_khalf01(
            kh_part: int,
            kh_slot: jax.Array,
            cur_rows: jax.Array,
            gs_g: jax.Array | None,
            gs_u: jax.Array | None,
        ):
            w_kh = w1_kh_ring.at[kh_slot]
            s_kh = s1_kh_ring.at[kh_slot]

            def _do_kh(m_bkt: int):
                if use_lhs_fp8 and kh_part == 0:
                    _prequant_lhs_k256(
                        lhs_2d_buf, lhs_fp8_buf, lhs_sc_buf, m_bkt
                    )
                _run_khalf_unit(
                    lhs_2d_buf, w_kh, s_kh, kh_part, m_bkt, gs_g, gs_u
                )

            if len(m_buckets) > 1:
                b_idx = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                lax.switch(
                    b_idx,
                    [
                        (lambda b=b: (_do_kh(b), jnp.int32(0))[1])
                        for b in m_buckets
                    ],
                )
            else:
                _do_kh(m_buckets[0])

        def _dispatch_khalf2(
            w2_slot: jax.Array,
            cur_rows: jax.Array,
            gs_w2: jax.Array | None,
        ) -> jax.Array:
            w_s = w2_kh_ring.at[w2_slot]
            s_s = s2_kh_ring.at[w2_slot]
            sq_rows = y_stage_buf.shape[0]

            def _do_kh2(m_bkt: int) -> jax.Array:
                sq_m = _run_down_unit(
                    w_s,
                    s_s,
                    m_bkt,
                    jnp.zeros((m_bkt, 1), dtype=jnp.float32),
                    gs_w2,
                )
                return _pad_rows_to(sq_m, sq_rows)

            if len(m_buckets) > 1:
                b_idx = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                return lax.switch(
                    b_idx, [(lambda b=b: _do_kh2(b)) for b in m_buckets]
                )
            return _do_kh2(m_buckets[0])

        def _dispatch_unit01(
            part_idx: int,
            r_slot: jax.Array,
            cur_rows: jax.Array,
            n_m32: jax.Array,
            gs_g: jax.Array | None,
            gs_u: jax.Array | None,
            cur_is_w3=None,
        ):
            w_s = w_ring.at[r_slot]
            s_s = s_ring.at[r_slot]

            def _std_u01(m_bkt: int):
                if (use_vmem_deq or use_loop32) and m_bkt == 32 and tile_m > 32:
                    if use_vmem_deq:
                        # Fast path when cur_rows <= 32: VREG dual dot (zero VMEM write);
                        # when cur_rows > 32: unpack ONCE to VMEM & reuse across all slices!
                        @pl.when(n_m32 == 1)
                        def _one32():
                            # The unit helpers read ``lhs_fp8_buf`` under W4A8, so it
                            # must be re-quantized for this tile (unit 1 reuses unit 0's).
                            if use_lhs_fp8 and part_idx == 0:
                                _prequant_lhs_k256(
                                    lhs_2d_buf, lhs_fp8_buf, lhs_sc_buf, 32
                                )
                            if dual_units:
                                _run_dual_half_unit(
                                    lhs_2d_buf,
                                    w_s,
                                    s_s,
                                    part_idx * (_I // 2),
                                    32,
                                    gs_g,
                                    gs_u,
                                    is_w3_flag=cur_is_w3,
                                )
                            else:
                                _run_proj_unit(
                                    lhs_2d_buf,
                                    w_s,
                                    s_s,
                                    32,
                                    part_idx == 1,
                                    gs_u if part_idx == 1 else gs_g,
                                    is_w3_flag=cur_is_w3,
                                )

                        @pl.when(n_m32 > 1)
                        def _many32():
                            def _go(use_w3_mode: bool):
                                if dual_units:
                                    _run_dual_half_vmem_deq(
                                        w_s,
                                        s_s,
                                        part_idx * (_I // 2),
                                        n_m32,
                                        gs_g,
                                        gs_u,
                                        is_w3=use_w3_mode,
                                    )
                                else:
                                    _run_proj_vmem_deq(
                                        w_s,
                                        s_s,
                                        n_m32,
                                        part_idx == 1,
                                        gs_u if part_idx == 1 else gs_g,
                                        is_w3=use_w3_mode,
                                    )
                                return jnp.int32(0)

                            if has_w3 and cur_is_w3 is not None:
                                lax.cond(
                                    cur_is_w3, lambda: _go(True), lambda: _go(False)
                                )
                            else:
                                _go(False)
                        return

                    # loop32 baseline
                    def _l32_step(mi, _):
                        r0 = pl.multiple_of(mi * 32, 32)
                        # Each 32-row slice overwrites ``lhs_fp8_buf``, so both
                        # units must re-quantize their own slice under W4A8.
                        if use_lhs_fp8:
                            _prequant_lhs_k256(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                lhs_fp8_buf,
                                lhs_sc_buf,
                                32,
                            )
                        if dual_units:
                            _run_dual_half_unit(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w_s,
                                s_s,
                                part_idx * (_I // 2),
                                32,
                                gs_g,
                                gs_u,
                                act_2d_buf.at[pl.ds(r0, 32)],
                                is_w3_flag=cur_is_w3,
                            )
                        else:
                            _run_proj_unit(
                                lhs_2d_buf.at[pl.ds(r0, 32)],
                                w_s,
                                s_s,
                                32,
                                part_idx == 1,
                                gs_u if part_idx == 1 else gs_g,
                                act_2d_buf.at[pl.ds(r0, 32)],
                                is_w3_flag=cur_is_w3,
                            )
                        return _

                    lax.fori_loop(0, n_m32, _l32_step, jnp.int32(0))
                    return

                if use_lhs_fp8 and part_idx == 0:
                    _prequant_lhs_k256(
                        lhs_2d_buf, lhs_fp8_buf, lhs_sc_buf, m_bkt
                    )
                if dual_units:
                    _run_dual_half_unit(
                        lhs_2d_buf,
                        w_s,
                        s_s,
                        part_idx * (_I // 2),
                        m_bkt,
                        gs_g,
                        gs_u,
                        is_w3_flag=cur_is_w3,
                    )
                else:
                    _run_proj_unit(
                        lhs_2d_buf,
                        w_s,
                        s_s,
                        m_bkt,
                        part_idx == 1,
                        gs_u if part_idx == 1 else gs_g,
                        is_w3_flag=cur_is_w3,
                    )

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
                    (
                        lambda r=r: (
                            _bd_unit01(w_s, s_s, part_idx, r, gs_g, gs_u),
                            jnp.int32(0),
                        )[1]
                    )
                    for r in bd_rows_tuple
                ] + [
                    (lambda b=b: (_std_u01(b), jnp.int32(0))[1])
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
                        (lambda b=b: (_std_u01(b), jnp.int32(0))[1])
                        for b in m_buckets
                    ],
                )
            else:
                _std_u01(m_buckets[0])

        def _dispatch_unit2(
            r_slot: jax.Array,
            cur_rows: jax.Array,
            n_m32: jax.Array,
            gs_w2: jax.Array | None,
            cur_is_w3=None,
        ) -> jax.Array:
            w_s = w_ring.at[r_slot]
            s_s = s_ring.at[r_slot]
            sq_rows = y_stage_buf.shape[0]

            def _std_u2(m_bkt: int) -> jax.Array:
                if (use_vmem_deq or use_loop32) and m_bkt == 32 and tile_m > 32:
                    if use_vmem_deq:

                        def _down_many32() -> jax.Array:
                            def _go(use_w3_mode: bool) -> jax.Array:
                                return _pad_rows_to(
                                    _run_down_vmem_deq(
                                        w_s, s_s, n_m32, gs_w2, is_w3=use_w3_mode
                                    ),
                                    sq_rows,
                                )

                            if has_w3 and cur_is_w3 is not None:
                                return lax.cond(
                                    cur_is_w3, lambda: _go(True), lambda: _go(False)
                                )
                            return _go(False)

                        return lax.cond(
                            n_m32 == 1,
                            lambda: _pad_rows_to(
                                _run_down_unit(
                                    w_s,
                                    s_s,
                                    32,
                                    jnp.zeros((32, 1), dtype=jnp.float32),
                                    gs_w2,
                                    is_w3_flag=cur_is_w3,
                                ),
                                sq_rows,
                            ),
                            lambda: _down_many32(),
                        )

                    row_iota = lax.broadcasted_iota(jnp.int32, (sq_rows, 1), 0)

                    def _l32_d(mi, sq_acc):
                        r0 = pl.multiple_of(mi * 32, 32)
                        sq_part = _run_down_unit(
                            w_s,
                            s_s,
                            32,
                            jnp.zeros((32, 1), dtype=jnp.float32),
                            gs_w2,
                            act_2d_buf.at[pl.ds(r0, 32)],
                            y_row_base=r0,
                            is_w3_flag=cur_is_w3,
                        )
                        part_pad = jnp.broadcast_to(
                            sq_part, (sq_rows // 32, 32, 1)
                        ).reshape(sq_rows, 1)
                        in_s = (row_iota >= r0) & (row_iota < r0 + 32)
                        return sq_acc + jnp.where(in_s, part_pad, 0.0)

                    return lax.fori_loop(
                        0,
                        n_m32,
                        _l32_d,
                        jnp.zeros((sq_rows, 1), dtype=jnp.float32),
                    )

                sq_m = _run_down_unit(
                    w_s,
                    s_s,
                    m_bkt,
                    jnp.zeros((m_bkt, 1), dtype=jnp.float32),
                    gs_w2,
                    is_w3_flag=cur_is_w3,
                )
                return _pad_rows_to(sq_m, sq_rows)

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
                    (lambda r=r: _bd_unit2(w_s, s_s, r, gs_w2))
                    for r in bd_rows_tuple
                ] + [(lambda b=b: _std_u2(b)) for b in m_buckets]
                return lax.switch(idx, branches)
            if len(m_buckets) > 1:
                b_idx = jnp.int32(0)
                for _b in m_buckets[:-1]:
                    b_idx = b_idx + (cur_rows > _b).astype(jnp.int32)
                return lax.switch(
                    b_idx, [(lambda b=b: _std_u2(b)) for b in m_buckets]
                )
            return _std_u2(m_buckets[0])

        def _flat_tile_body(t, carry_tile):
            k_exp = tile_k_smem[t]
            base_pos = tile_base_smem[t]
            cur_rows = tile_rows_smem[t]
            flags = tile_flags_smem[t]
            is_first_tile = (flags & 1) != 0
            is_last_tile = (flags & 2) != 0

            has_k1 = (k_exp + 1) < num_exec
            k1_clamped = jnp.where(has_k1, k_exp + 1, jnp.int32(0))
            k1_src = exec_src_smem[k1_clamped]
            k1_slot = exec_slot_smem[k1_clamped]

            has_k2 = (k_exp + 2) < num_exec
            k2_clamped = jnp.where(has_k2, k_exp + 2, jnp.int32(0))
            k2_src = exec_src_smem[k2_clamped]
            k2_slot = exec_slot_smem[k2_clamped]

            if is_khalf:
                if n_kh_w1 == 4:
                    @pl.when(jnp.logical_and(t == 0, num_exec >= 2))
                    def _pf_kh_e1():
                        start_kh_w1(jnp.int32(2), k1_src, k1_slot, 0)
                        start_kh_w1(jnp.int32(3), k1_src, k1_slot, 1)
                        start_kh_w2(jnp.int32(1), k1_src, k1_slot)
            else:
                u_base = k_exp * 3
                s0 = _ring_slot(u_base, n_slots)
                s1 = _ring_slot(u_base + 1, n_slots)
                s2 = _ring_slot(u_base + 2, n_slots)

                # Top-of-tile=0 prefetch for the (n_slots - 1)-th ahead unit (strict FIFO!):
                @pl.when(jnp.logical_and(t == 0, (n_slots - 1) < n_units_total))
                def _pf_initial_tail():
                    if n_slots == 3:
                        start_ring_unit(
                            jnp.int32(2),
                            exec_src_smem[0],
                            exec_slot_smem[0],
                            jnp.int32(2),
                        )
                    elif n_slots == 4:
                        start_ring_unit(
                            jnp.int32(3), k1_src, k1_slot, jnp.int32(0)
                        )
                    else:
                        start_ring_unit(
                            jnp.int32(5), k1_src, k1_slot, jnp.int32(2)
                        )

            # Gather current tile's rows into 2D / blockdiag buffer.
            if cfg.async_gather:
                g_cur = t & jnp.int32(1)
                _wait_async_gather(g_cur, cur_rows)
                cur_3d_src = gather_3d_buf.at[g_cur]
            else:
                _sync_gather_to_3d(io_3d_buf, base_pos, cur_rows)
                cur_3d_src = io_3d_buf

            n_m32 = jnp.maximum(jnp.int32(1), (cur_rows + 31) >> 5)
            if use_bd:
                max_bd = bd_rows_tuple[-1]

                @pl.when(cur_rows <= max_bd)
                def _prep_bd_x():
                    bd_i = jnp.int32(0)
                    for _r in bd_rows_tuple[:-1]:
                        bd_i = bd_i + (cur_rows > _r).astype(jnp.int32)
                    lax.switch(
                        bd_i,
                        [
                            (
                                lambda r=r: (
                                    _build_masked_x_from_3d(
                                        cur_3d_src,
                                        masked_x_buf,
                                        r,
                                        kperm_order=cfg.kperm_k16,
                                    ),
                                    jnp.int32(0),
                                )[1]
                            )
                            for r in bd_rows_tuple
                        ],
                    )

                @pl.when(cur_rows > max_bd)
                def _prep_std_x():
                    n_c16 = (cur_rows + 15) >> 4
                    lax.fori_loop(
                        0,
                        n_c16,
                        lambda ci, _: (
                            _copy_3d_to_lhs_2d(
                                cur_3d_src, pl.multiple_of(ci * 16, 16), 16
                            ),
                            _,
                        )[1],
                        jnp.int32(0),
                    )
            else:
                n_c16 = (cur_rows + 15) >> 4
                lax.fori_loop(
                    0,
                    n_c16,
                    lambda ci, _: (
                        _copy_3d_to_lhs_2d(
                            cur_3d_src, pl.multiple_of(ci * 16, 16), 16
                        ),
                        _,
                    )[1],
                    jnp.int32(0),
                )

            # Immediately launch NEXT tile's async gather on DMA engine while W1+W2 run!
            if cfg.async_gather:
                @pl.when((t + 1) < num_tiles)
                def _pf_next_gather():
                    g_nxt = (t + 1) & jnp.int32(1)
                    _start_async_gather(
                        g_nxt, tile_base_smem[t + 1], tile_rows_smem[t + 1]
                    )

            w_src_cur = exec_src_smem[k_exp]
            w_slot_cur = exec_slot_smem[k_exp]
            cur_is_w3 = _slot_is_w3(w_src_cur, w_slot_cur)
            gs_g, gs_u = _get_gs(w_slot_cur, False)
            gs_w2, _unused_gs2 = _get_gs(w_slot_cur, True)
            do_pf_k1 = jnp.logical_and(is_last_tile, has_k1)
            do_pf_k2 = jnp.logical_and(is_last_tile, has_k2)

            if is_khalf:
                kh0_s = (k_exp & jnp.int32(1)) * jnp.int32(2) if n_kh_w1 == 4 else jnp.int32(0)
                kh1_s = kh0_s + jnp.int32(1) if n_kh_w1 == 4 else jnp.int32(1)
                w2_s = (k_exp & jnp.int32(1)) if n_kh_w2 == 2 else jnp.int32(0)

                @pl.when(is_first_tile)
                def _w_kh0():
                    wait_kh_w1(kh0_s)

                _dispatch_khalf01(0, kh0_s, cur_rows, gs_g, gs_u)

                if n_kh_w1 == 4:
                    @pl.when(do_pf_k2)
                    def _pf_kh0_4():
                        start_kh_w1(kh0_s, k2_src, k2_slot, 0)
                else:
                    @pl.when(do_pf_k1)
                    def _pf_kh0_2():
                        start_kh_w1(kh0_s, k1_src, k1_slot, 0)

                @pl.when(is_first_tile)
                def _w_kh1():
                    wait_kh_w1(kh1_s)

                _dispatch_khalf01(1, kh1_s, cur_rows, gs_g, gs_u)

                if n_kh_w1 == 4:
                    @pl.when(do_pf_k2)
                    def _pf_kh1_4():
                        start_kh_w1(kh1_s, k2_src, k2_slot, 1)
                else:
                    @pl.when(do_pf_k1)
                    def _pf_kh1_2():
                        start_kh_w1(kh1_s, k1_src, k1_slot, 1)

                @pl.when(is_first_tile)
                def _w_kh2():
                    wait_kh_w2(w2_s)

                sq_sum_tile = _dispatch_khalf2(w2_s, cur_rows, gs_w2)

                if n_kh_w2 == 2:
                    @pl.when(do_pf_k2)
                    def _pf_khw2_2():
                        start_kh_w2(w2_s, k2_src, k2_slot)
                else:
                    @pl.when(do_pf_k1)
                    def _pf_khw2_1():
                        start_kh_w2(w2_s, k1_src, k1_slot)
            else:
                # --- UNIT 0 (OUTSIDE lax.switch: wait -> compute -> immediate ring refill!) ---
                @pl.when(is_first_tile)
                def _w_u0():
                    wait_ring_slot(s0, cur_is_w3)

                _dispatch_unit01(
                    0, s0, cur_rows, n_m32, gs_g, gs_u, cur_is_w3=cur_is_w3
                )

                if n_slots == 6:
                    @pl.when(do_pf_k2)
                    def _pf_after_u0_6():
                        start_ring_unit(s0, k2_src, k2_slot, jnp.int32(0))
                else:
                    @pl.when(do_pf_k1)
                    def _pf_after_u0():
                        if n_slots == 3:
                            start_ring_unit(s0, k1_src, k1_slot, jnp.int32(0))
                        else:
                            start_ring_unit(s0, k1_src, k1_slot, jnp.int32(1))

                # --- UNIT 1 (OUTSIDE lax.switch: wait -> compute -> immediate ring refill!) ---
                @pl.when(is_first_tile)
                def _w_u1():
                    wait_ring_slot(s1, cur_is_w3)

                _dispatch_unit01(
                    1, s1, cur_rows, n_m32, gs_g, gs_u, cur_is_w3=cur_is_w3
                )

                if n_slots == 6:
                    @pl.when(do_pf_k2)
                    def _pf_after_u1_6():
                        start_ring_unit(s1, k2_src, k2_slot, jnp.int32(1))
                else:
                    @pl.when(do_pf_k1)
                    def _pf_after_u1():
                        if n_slots == 3:
                            start_ring_unit(s1, k1_src, k1_slot, jnp.int32(1))
                        else:
                            start_ring_unit(s1, k1_src, k1_slot, jnp.int32(2))

                if use_bd:
                    max_bd = bd_rows_tuple[-1]

                    @pl.when(cur_rows <= max_bd)
                    def _prep_bd_act():
                        bd_i = jnp.int32(0)
                        for _r in bd_rows_tuple[:-1]:
                            bd_i = bd_i + (cur_rows > _r).astype(jnp.int32)
                        lax.switch(
                            bd_i,
                            [
                                (
                                    lambda r=r: (
                                        _build_masked_x_from_2d(
                                            act_2d_buf,
                                            masked_x_buf,
                                            r,
                                            kperm_order=cfg.kperm_k16,
                                        ),
                                        jnp.int32(0),
                                    )[1]
                                )
                                for r in bd_rows_tuple
                            ],
                        )

                # --- UNIT 2 / W2 (OUTSIDE lax.switch: wait -> compute -> immediate ring refill!) ---
                @pl.when(is_first_tile)
                def _w_u2():
                    wait_ring_slot(s2, cur_is_w3)

                sq_sum_tile = _dispatch_unit2(
                    s2, cur_rows, n_m32, gs_w2, cur_is_w3=cur_is_w3
                )

                if n_slots == 3:
                    @pl.when(do_pf_k1)
                    def _pf_after_u2_3():
                        start_ring_unit(s2, k1_src, k1_slot, jnp.int32(2))
                elif n_slots == 4:
                    @pl.when(do_pf_k2)
                    def _pf_after_u2_4():
                        start_ring_unit(s2, k2_src, k2_slot, jnp.int32(0))
                else:
                    @pl.when(do_pf_k2)
                    def _pf_after_u2_6():
                        start_ring_unit(s2, k2_src, k2_slot, jnp.int32(2))

            # --- Fused single-pass RMSNorm + router-weighted scatter-accumulate ---
            if cfg.fused_norm_scatter:
                _fused_norm_and_scatter(base_pos, cur_rows, sq_sum_tile)
            else:
                _legacy_norm_then_scatter(
                    base_pos, cur_rows, y_stage_buf.shape[0], sq_sum_tile
                )
            return carry_tile

        lax.fori_loop(0, num_tiles, _flat_tile_body, jnp.int32(0))

    # =========================================================================
    # 7. REDUCE-SCATTER (2-Stage Intra-Chip Paired Coll OR Flat EP-1 Direct)
    # =========================================================================
    pl.semaphore_wait(barrier_sem, ep)

    if cfg.pair_rs and ep in (8, 16):
        # Ported from muse_megakernel/decode/spep/gather.py (_u_reduce_pair):
        # Halves off-chip ICI Reduce-Scatter traffic via 733 GB/s intra-chip pair sum!
        num_chips = ep // 2
        if cfg.rs_wire_bf16:
            cast_rc = min(batch_size, 8)

            def _cast_rs_p(ci, _):
                r0 = ci * cast_rc
                rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                    pl.ds(r0, cast_rc), :, :
                ].astype(jnp.bfloat16)
                return _

            lax.fori_loop(0, batch_size // cast_rc, _cast_rs_p, jnp.int32(0))
            pair_src = rs_send_bf16
        else:
            pair_src = accum_3d

        core = me & jnp.int32(1)
        chip = me >> 1
        partner = jnp.bitwise_xor(me, jnp.int32(1))

        # Stage 1: Exchange partner's blocks across 733 GB/s intra-chip link.
        p1_ops = []
        for c in range(num_chips):
            dest_for_partner = jnp.int32(2 * c) + (jnp.int32(1) - core)
            op_p = pltpu.make_async_remote_copy(
                pair_src.at[
                    pl.ds(dest_for_partner * rows_per_rank, rows_per_rank), :, :
                ],
                rs_pair_recv.at[c],
                rs_pair_ssem.at[c],
                rs_pair_rsem.at[c],
                device_id={ep_axis_name: partner},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_p.start()
            p1_ops.append(op_p)

        # Stage 2: As each partner block arrives, sum in FP32 and send ONLY (num_chips - 1) blocks off-chip!
        chip_ops = []
        for c in range(num_chips):
            pltpu.make_async_copy(
                rs_pair_recv.at[c], rs_pair_recv.at[c], rs_pair_rsem.at[c]
            ).wait()
            target_rank = jnp.int32(2 * c) + core

            @pl.when(c != chip)
            def _sum_and_send_chip(c=c, target_rank=target_rank):
                r_off = target_rank * rows_per_rank
                pair_f32 = accum_3d[
                    pl.ds(r_off, rows_per_rank), :, :
                ] + rs_pair_recv[c].astype(jnp.float32)
                rs_chip_send[c] = pair_f32.astype(rs_chip_send.dtype)

            op_c = pltpu.make_async_remote_copy(
                rs_chip_send.at[c],
                rs_chip_recv.at[chip],
                rs_chip_ssem.at[c],
                rs_chip_rsem,
                device_id={ep_axis_name: target_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            chip_ops.append((c, op_c))

            @pl.when(c != chip)
            def _start_chip_rs(_op=op_c):
                _op.start()

        # Wait for the (num_chips - 1) remote same-parity chips' summed blocks.
        pltpu.make_async_copy(
            rs_chip_recv.at[pl.ds(0, num_chips - 1)],
            rs_chip_recv.at[pl.ds(0, num_chips - 1)],
            rs_chip_rsem,
        ).wait()

        red_rc = min(rows_per_rank, 4)

        def _reduce_owner_paired(ci, _):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for c in range(num_chips):
                part_c = jnp.where(
                    c == chip,
                    rs_pair_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                    rs_chip_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                )
                acc = acc + part_c
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return _

        lax.fori_loop(
            0, rows_per_rank // red_rc, _reduce_owner_paired, jnp.int32(0)
        )

        def _finish_rs_sends():
            for op_p in p1_ops:
                op_p.wait_send()
            for c, op_c in chip_ops:
                @pl.when(c != chip)
                def _wc(_op=op_c):
                    _op.wait_send()
    else:
        if cfg.rs_wire_bf16:
            cast_rc = min(batch_size, 8)

            def _cast_rs_chunk(ci, _):
                r0 = ci * cast_rc
                rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                    pl.ds(r0, cast_rc), :, :
                ].astype(jnp.bfloat16)
                return _

            lax.fori_loop(
                0, batch_size // cast_rc, _cast_rs_chunk, jnp.int32(0)
            )
            rs_src_ref = rs_send_bf16
        else:
            rs_src_ref = accum_3d

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

        lax.fori_loop(
            0, rows_per_rank // red_rc, _reduce_owner_chunk, jnp.int32(0)
        )

        def _finish_rs_sends():
            for d, copy_op in rs_copies:
                @pl.when(me != d)
                def _wait_rs(_op=copy_op):
                    _op.wait_send()

    # =========================================================================
    # 8. ALL-GATHER FAN-OUT (Per-Block Relayed Coll.pair_all_gather OR Flat)
    # =========================================================================
    if return_sharded:
        if rows_per_rank >= 16:
            for r0 in range(0, rows_per_rank, 16):
                output_ref[pl.ds(r0, 16), :] = owner_3d[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            output_ref[...] = owner_3d[...].reshape(rows_per_rank, _H)
    elif cfg.pair_ag_relay and ep in (8, 16):
        # Ported from muse_megakernel/common/coll.py (pair_all_gather_start):
        # Relays remote chips over 733 GB/s partner bridge immediately upon arrival!
        num_chips = ep // 2
        core = me & jnp.int32(1)
        chip = me >> 1
        partner = jnp.bitwise_xor(me, jnp.int32(1))
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]

        def _ag_send_slot(slot_rank, peer_rank, ssem_idx: int):
            r_off = slot_rank * rows_per_rank
            op = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                ag_ssem.at[ssem_idx],
                ag_psem.at[slot_rank],
                device_id={ep_axis_name: peer_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            return op

        def _ag_wait_arrived(slot_rank):
            pltpu.make_async_copy(
                owner_3d, owner_3d, ag_psem.at[slot_rank]
            ).wait()

        far_ranks = [
            jnp.int32(2) * jnp.bitwise_xor(chip, jnp.int32(d)) + core
            for d in range(1, num_chips)
        ]
        ag_ops = [_ag_send_slot(me, partner, 0)]
        for idx_f, fk in enumerate(far_ranks):
            ag_ops.append(_ag_send_slot(me, fk, 1 + idx_f))

        # Relay blocks immediately on arrival!
        for idx_f, fk in enumerate(far_ranks):
            _ag_wait_arrived(fk)
            ag_ops.append(_ag_send_slot(fk, partner, num_chips + idx_f))

        _ag_wait_arrived(partner)
        for fk in far_ranks:
            _ag_wait_arrived(jnp.bitwise_xor(fk, jnp.int32(1)))

        if batch_size >= 16:
            for r0 in range(0, batch_size, 16):
                output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            output_ref[...] = fan_3d_buf[...].reshape(batch_size, _H)

        for op_a in ag_ops:
            op_a.wait_send()
    else:
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]
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
            def _start_fan(_op=fan_op):
                _op.start()

        recv_rows = (ep - 1) * rows_per_rank
        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_3d_buf.at[pl.ds(0, recv_rows), :, :],
            fan_recv_sem,
        ).wait()

        for r0 in range(0, batch_size, 16):
            output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                pl.ds(r0, 16), :, :
            ].reshape(16, _H)

        for dest, fan_op in fan_copies:
            @pl.when(me != dest)
            def _wait_fan(_op=fan_op):
                _op.wait_send()

    _finish_rs_sends()


def fused_decode_sparse_v2_from_logits(
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
    w3_mask: jax.Array | None = None,
    *,
    w1_gs: jax.Array | None = None,
    w2_gs: jax.Array | None = None,
    ep: int = 8,
    return_sharded: bool = False,
    replica_cost_mode: str = "active",
    ep_axis_name: str = "model",
    config: SparseDecodeV2Config | None = None,
) -> jax.Array:
    """Circular-ring + pipelined W4A16/W4A8/W3 + paired-Coll Fuse-All Decode MoE (v2)."""
    if ep not in (8, 16):
        raise ValueError(f"Sparse v2 supports ep in (8, 16), got {ep}")
    batch_size = x.shape[0]
    if batch_size not in (16, 32, 64, 128, 256, 384):
        raise ValueError(
            f"Sparse v2 requires B in (16, 32, 64, 128, 256, 384), got {batch_size}"
        )
    if logits.shape != (batch_size, _E_TOTAL):
        raise ValueError(
            f"Expected logits shape ({batch_size}, {_E_TOTAL}), got {logits.shape}"
        )
    e_local = _E_TOTAL // ep
    w_groups = w2.shape[0]
    is_part_major = w1.shape == (2 * w_groups, _H, _I)
    if (
        not (e_local <= w_groups <= e_local + 8)
        or (w1.shape != (w_groups, _H, _N1) and not is_part_major)
        or w2.shape != (w_groups, _I, _H)
    ):
        raise ValueError(
            f"Unexpected FP4 weight shapes: w1={w1.shape}, w2={w2.shape} for ep={ep}"
        )
    if w1.dtype != jnp.float4_e2m1fn or w2.dtype != jnp.float4_e2m1fn:
        raise ValueError(
            f"Expected float4_e2m1fn weights, got w1={w1.dtype}, w2={w2.dtype}"
        )
    scale_block = _H // s1.shape[1]
    if scale_block not in (16, 256, 512):
        raise ValueError(
            f"Expected K16 or K256 scales, got s1.shape={s1.shape}"
        )
    exp_s1_shape = (
        (2 * w_groups, _H // scale_block, _I)
        if is_part_major
        else (w_groups, _H // scale_block, _N1)
    )
    if s1.shape != exp_s1_shape or s2.shape != (
        w_groups,
        _I // scale_block,
        _H,
    ):
        raise ValueError(
            f"Mismatched scale shapes: s1={s1.shape}, s2={s2.shape} for block={scale_block}"
        )
    if s1.dtype == jnp.float8_e4m3fn and scale_block >= 256:
        raise ValueError(
            "float8_e4m3fn scales require >= 32 scale rows (K16 scale_block=16)"
        )

    has_gs = (w1_gs is not None) or (w2_gs is not None)
    if has_gs and (w1_gs is None or w2_gs is None):
        raise ValueError("w1_gs and w2_gs must be provided together")
    has_w3 = w3_mask is not None

    rep_args = (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
    has_replicas = any(a is not None for a in rep_args)
    if has_replicas:
        if any(a is None for a in rep_args):
            raise ValueError(
                "All of rep_w1, rep_s1, rep_w2, rep_s2, replica_meta must be provided together"
            )
        num_replicas = int(rep_w2.shape[0])
    else:
        num_replicas = 0

    if config is None:
        config = default_sparse_decode_v2_config(batch_size, ep, scale_block)
    if is_part_major and config.contig_parts_mode not in (2, 3):
        config = dataclasses.replace(
            config,
            contig_parts_mode=3 if config.ring_mode.endswith("_dual") else 2,
        )
    if config.ring_mode.startswith("ring6") and (
        scale_block == 16 or batch_size >= 256 or config.tile_m > 64
    ):
        config = dataclasses.replace(
            config, ring_mode="ring4" + config.ring_mode[5:]
        )
    elif config.ring_mode == "ring4_khalf" and (
        scale_block == 16 or batch_size >= 256
    ):
        config = dataclasses.replace(config, ring_mode="ring3_khalf")
    if has_w3 and config.ring_mode.endswith("_khalf"):
        # The K-half ring has no Cold-W3 row packing support.
        config = dataclasses.replace(config, ring_mode="ring4_dual")
    if has_w3 and scale_block == 16 and config.multi_m32_mode == "vmem_deq":
        # The K16 strip-dequant helper cannot decode Cold-W3 slots.
        config = dataclasses.replace(config, multi_m32_mode="loop32")

    bias_2d = bias.reshape(1, _E_TOTAL)
    norm_2d = norm.reshape(1, _H)
    x_2d = x.reshape(batch_size, _H) if x.ndim == 3 else x

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    s_spec = pl.BlockSpec(memory_space=smem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    is_dense = config.row_mode == "dense"
    is_mxu_sparse = config.row_mode == "mxu_sparse"
    is_dense_route = is_dense or is_mxu_sparse
    is_khalf = config.ring_mode.endswith("_khalf") and (not is_dense_route)
    tile_m = config.tile_m
    max_exec = e_local + num_replicas
    max_tiles = max_exec + (batch_size * _TOPK) // tile_m + 4

    scratch_shapes: list = []
    if is_dense_route:
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
        scratch_shapes.extend(
            [
                smem((max_tiles,), jnp.int32),  # tile_k_smem
                smem((max_tiles,), jnp.int32),  # tile_base_smem
                smem((max_tiles,), jnp.int32),  # tile_rows_smem
                smem((max_tiles,), jnp.int32),  # tile_flags_smem
            ]
        )

    if has_replicas:
        moved_smem_len = (
            8 if (not is_dense_route and config.static_holder_mask) else _E_TOTAL
        )
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((moved_smem_len,), jnp.int32),
                smem((e_local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
        if not is_dense_route and not config.static_holder_mask:
            scratch_shapes.append(smem((num_replicas,), jnp.int32))

    n_slots = (
        6
        if config.ring_mode.startswith("ring6")
        else (4 if config.ring_mode.startswith("ring4") else 3)
    )
    nqb = _H // scale_block
    if is_khalf:
        n_kh_w1 = 4 if config.ring_mode == "ring4_khalf" else 2
        n_kh_w2 = 2 if config.ring_mode == "ring4_khalf" else 1
        scratch_shapes.extend(
            [
                vmem((n_kh_w1, (_H // 2) // _FP4_PACK, _N1), jnp.uint32),
                vmem((n_kh_w1, nqb // 2, _N1), s1.dtype),
                sem((n_kh_w1,)),
                vmem((n_kh_w2, _I // _FP4_PACK, _H), jnp.uint32),
                vmem((n_kh_w2, nqb, _H), s2.dtype),
                sem((n_kh_w2,)),
                vmem((max(16, tile_m), _N1), jnp.float32),  # gu_kh_buf
                sem((1,)),  # misc_sem
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((n_slots, _H // _FP4_PACK, _I), jnp.uint32),
                vmem((n_slots, nqb, _I), s1.dtype),
                sem((n_slots,)),
                sem((1,)),
            ]
        )

    use_bd = (
        (not is_dense)
        and (not is_mxu_sparse)
        and (not is_khalf)
        and (scale_block == 16)
        and (not has_w3)
        and bool(config.k16_blockdiag_rows)
    )
    use_vmem_deq = (
        (not is_dense)
        and (not is_khalf)
        and (config.multi_m32_mode == "vmem_deq")
        and (tile_m > 32)
    )
    deq_nc = 1024 if scale_block >= 256 else min(config.k16_strip_nc, 512)
    deq_dtype = jnp.float8_e4m3fn if scale_block >= 256 else jnp.bfloat16
    use_lhs_fp8 = (scale_block >= 256) and config.lhs_fp8
    y_dtype = jnp.float32 if config.norm_f32_stage else jnp.bfloat16
    y_stage_rows = max(16, tile_m)

    if is_dense:
        scratch_shapes.extend(
            [
                vmem((batch_size, _I), jnp.bfloat16),  # act_2d_buf
                vmem((batch_size, _H), y_dtype),  # y_stage_buf
                vmem((batch_size, 128), jnp.float32),  # sq_dense_buf
            ]
        )
        if use_lhs_fp8:
            scratch_shapes.extend(
                [
                    vmem((max(32, batch_size), _H), jnp.float8_e4m3fn),
                    vmem((batch_size, (_H // 256) * 128), jnp.float32),
                ]
            )
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),  # gain_f32_buf
                vmem((batch_size, _H), jnp.float32),  # accum_2d
                vmem((batch_size, 32, 128), jnp.float32),  # accum_3d
            ]
        )
        if not return_sharded:
            scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    elif is_mxu_sparse:
        scratch_shapes.extend(
            [
                vmem((batch_size, _H), jnp.bfloat16),  # x_clean_2d
                vmem((tile_m, _H), jnp.bfloat16),  # lhs_2d_buf
                vmem((tile_m, _I), jnp.bfloat16),  # act_2d_buf
                vmem((y_stage_rows, _H), y_dtype),  # y_stage_buf
            ]
        )
        if use_lhs_fp8:
            scratch_shapes.extend(
                [
                    vmem((max(32, tile_m), _H), jnp.float8_e4m3fn),
                    vmem((tile_m, (_H // 256) * 128), jnp.float32),
                ]
            )
        if use_vmem_deq:
            scratch_shapes.append(vmem((_H, deq_nc), deq_dtype))
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),  # gain_f32_buf
                vmem((batch_size, _H), jnp.float32),  # accum_2d
                vmem((batch_size, 32, 128), jnp.float32),  # accum_3d
            ]
        )
        if not return_sharded:
            scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    else:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))  # x_3d_buf
        if config.async_gather:
            scratch_shapes.extend(
                [
                    vmem((2, tile_m, 32, 128), jnp.bfloat16),  # gather_3d_buf
                    sem((2,)),  # gather_sem
                ]
            )
            if not config.fused_norm_scatter:
                scratch_shapes.append(vmem((tile_m, 32, 128), jnp.bfloat16))
        else:
            scratch_shapes.append(vmem((tile_m, 32, 128), jnp.bfloat16))  # io_3d_buf

        scratch_shapes.extend(
            [
                vmem((tile_m, _H), jnp.bfloat16),  # lhs_2d_buf
                vmem((tile_m, _I), jnp.bfloat16),  # act_2d_buf
                vmem((y_stage_rows, _H), y_dtype),  # y_stage_buf
            ]
        )
        if use_lhs_fp8:
            scratch_shapes.extend(
                [
                    vmem((max(32, tile_m), _H), jnp.float8_e4m3fn),
                    vmem((tile_m, (_H // 256) * 128), jnp.float32),
                ]
            )
        if use_bd:
            max_bd = max(config.k16_blockdiag_rows)
            scratch_shapes.append(vmem((max_bd * 16, _H), jnp.bfloat16))
        if use_vmem_deq:
            scratch_shapes.append(vmem((_H, deq_nc), deq_dtype))
        scratch_shapes.extend(
            [
                vmem((1, _H), jnp.float32),  # gain_f32_buf
                vmem((batch_size, 32, 128), jnp.float32),  # accum_3d
            ]
        )

    rows_per_rank = batch_size // ep
    rs_wire_dtype = jnp.bfloat16 if config.rs_wire_bf16 else jnp.float32
    if config.rs_wire_bf16:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))

    if config.pair_rs and ep in (8, 16):
        num_chips_rs = ep // 2
        scratch_shapes.extend(
            [
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),  # rs_pair_recv
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),  # rs_chip_send
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),  # rs_chip_recv
                sem((num_chips_rs,)),  # rs_pair_ssem
                sem((num_chips_rs,)),  # rs_pair_rsem
                sem((num_chips_rs,)),  # rs_chip_ssem
                sem,  # rs_chip_rsem
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((ep - 1, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((ep,)),
                sem,
            ]
        )
    scratch_shapes.append(vmem((rows_per_rank, 32, 128), jnp.bfloat16))  # owner_3d

    if not return_sharded:
        if config.pair_ag_relay and ep in (8, 16):
            num_chips_ag = ep // 2
            scratch_shapes.extend([sem((2 * num_chips_ag - 1,)), sem((ep,))])  # ag_ssem, ag_psem
        else:
            scratch_shapes.extend([sem((ep,)), sem])

    in_specs: list = [
        v_spec,  # logits
        v_spec,  # bias
        v_spec,  # x
        h_spec,  # w1
        h_spec,  # s1
        h_spec,  # w2
        h_spec,  # s2
        v_spec,  # norm
    ]
    extra_args: list = []
    if has_gs:
        # w1_gs: [2, max_exec] f32 in SMEM; w2_gs: [1, max_exec] f32 in SMEM
        w1_gs_2d = w1_gs.reshape(2, -1).astype(jnp.float32)
        w2_gs_2d = w2_gs.reshape(1, -1).astype(jnp.float32)
        in_specs.extend([s_spec, s_spec])
        extra_args.extend([w1_gs_2d, w2_gs_2d])
    prefix_list: list = [rank]
    if has_replicas:
        in_specs.extend([h_spec, h_spec, h_spec, h_spec])
        extra_args.extend([rep_w1, rep_s1, rep_w2, rep_s2])
        prefix_list.append(replica_meta)
    if has_w3:
        prefix_list.append(w3_mask.reshape((1,)).astype(jnp.int32))
    num_scalar_prefetch = len(prefix_list)
    prefix_args = tuple(prefix_list)

    call_args = (
        *prefix_args,
        logits,
        bias_2d,
        x_2d,
        w1,
        s1,
        w2,
        s2,
        norm_2d,
        *extra_args,
    )

    out_rows = rows_per_rank if return_sharded else batch_size
    out_shape = jax.ShapeDtypeStruct((out_rows, _H), jnp.bfloat16)

    kernel_fn = functools.partial(
        _sparse_decode_v2_kernel,
        ep=ep,
        batch_size=batch_size,
        scale_block=scale_block,
        num_replicas=num_replicas,
        has_gs=has_gs,
        has_w3=has_w3,
        return_sharded=return_sharded,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        cfg=config,
    )
    rep_tag = f"_r{num_replicas}" if has_replicas else ""
    w3_tag = "_w3" if has_w3 else ""
    out_tag = "_sh" if return_sharded else "_rep"
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=(
            f"muse_sparse_v2_ep{ep}_b{batch_size}_k{scale_block}"
            f"_{config.row_mode}_{config.ring_mode}_tm{tile_m}{rep_tag}{w3_tag}{out_tag}"
        ),
    )(*call_args)


# =============================================================================
# Mission 3 (#452 Item 2): Circular-Ring Sparse Decode for FP8 Layers (L0 & L61)
# =============================================================================
#
# In Muse Spark (62 layers), L0 and L61 are channelwise FP8 MoE layers (48 MiB
# per expert: W1 [4096, 8192] fp8 = 32 MiB, W2 [4096, 4096] fp8 = 16 MiB) which
# XPROF_E2E_BREAKDOWN.md identified as Rank #1 and Rank #2 HLO ops in steady
# decode (314.6 us & 247.2 us = 561.8 us/step via gmm_v2_fused_rs + external
# TopK/norm/AG).
#
# ``fused_decode_sparse_fp8_from_logits`` fuses in-kernel TopK routing (with
# R>=0 redundant expert dispatch), continuous unstrided 8-MiB circular double
# rings for W1 (4 x [1024, 8192] fp8) and W2 (2 x [2048, 4096] fp8), direct
# zero-unpack TPU v7x MXU bf16 x fp8 matmuls, single-pass post_expert_norm
# scatter, 2-stage paired Coll Reduce-Scatter, and relayed Coll All-Gather.


def _sparse_decode_fp8_v2_kernel(
    *all_refs,
    ep: int,
    batch_size: int,
    num_replicas: int,
    return_sharded: bool,
    replica_cost_mode: str,
    ep_axis_name: str,
    pair_rs: bool,
    pair_ag_relay: bool,
    rs_wire_bf16: bool,
    norm_f32_stage: bool,
):
    has_replicas = num_replicas > 0
    e_local = _E_TOTAL // ep
    rows_per_rank = batch_size // ep
    m_step = 16 if batch_size <= 16 else 32
    max_exec = e_local + (num_replicas if has_replicas else 0)
    r_pad = max(8, ((num_replicas + 7) // 8) * 8) if has_replicas else 8

    it = iter(all_refs)
    rank_ref = next(it)
    replica_meta_ref = next(it) if has_replicas else None

    logits_ref = next(it)  # [B, 256] f32 VMEM
    bias_ref = next(it)  # [1, 256] f32 VMEM
    x_ref = next(it)  # [B, 4096] bf16 VMEM
    w1_ref = next(it)  # [32, 4096, 8192] fp8 HBM
    s1_ref = next(it)  # [32, 8192] f32 VMEM
    w2_ref = next(it)  # [32, 4096, 4096] fp8 HBM
    s2_ref = next(it)  # [32, 4096] f32 VMEM
    norm_ref = next(it)  # [1, 4096] bf16/f32 VMEM
    if has_replicas:
        rep_w1_ref = next(it)  # [R, 4096, 8192] fp8 HBM
        rep_s1_ref = next(it)  # [r_pad, 8192] f32 VMEM
        rep_w2_ref = next(it)  # [R, 4096, 4096] fp8 HBM
        rep_s2_ref = next(it)  # [r_pad, 4096] f32 VMEM
    else:
        rep_w1_ref = rep_s1_ref = rep_w2_ref = rep_s2_ref = None
    output_ref = next(it)

    r_scratch = RoutingScratch(
        *(next(it) for _ in range(len(RoutingScratch._fields)))
    )
    if has_replicas:
        owner_load_smem = next(it)
        moved_flag_smem = next(it)
        local_suppress_smem = next(it)
        holder_slot_expert_smem = next(it)
        holder_cand_smem = next(it)
    else:
        owner_load_smem = moved_flag_smem = None
        local_suppress_smem = holder_slot_expert_smem = holder_cand_smem = None

    w1_ring = next(it)  # [2, 1024, 8192] fp8 VMEM (16 MiB)
    w2_ring = next(it)  # [2, 2048, 4096] fp8 VMEM (16 MiB)
    w1_sem = next(it)  # DMA sem (2,)
    w2_sem = next(it)  # DMA sem (2,)
    misc_sem = next(it)  # DMA sem (1,)

    x_3d_buf = next(it)  # [B, 32, 128] bf16
    lhs_exp_3d = next(it)  # [B, 32, 128] bf16
    lhs_m_2d = next(it)  # [m_step, 4096] bf16
    gu_acc_2d = next(it)  # [B, 8192] f32 (also reused for W2 [B, 4096] f32!)
    act_exp_2d = next(it)  # [B, 4096] bf16
    gain_f32_buf = next(it)  # [1, 4096] f32
    accum_3d = next(it)  # [B, 32, 128] f32
    rs_send_bf16 = next(it) if rs_wire_bf16 else None

    if pair_rs and ep in (8, 16):
        rs_pair_recv = next(it)
        rs_chip_send = next(it)
        rs_chip_recv = next(it)
        rs_pair_ssem = next(it)
        rs_pair_rsem = next(it)
        rs_chip_ssem = next(it)
        rs_chip_rsem = next(it)
        rs_recv_3d = rs_send_sem = rs_recv_sem = None
    else:
        rs_recv_3d = next(it)
        rs_send_sem = next(it)
        rs_recv_sem = next(it)
        rs_pair_recv = rs_chip_send = rs_chip_recv = None
        rs_pair_ssem = rs_pair_rsem = rs_chip_ssem = rs_chip_rsem = None

    owner_3d = next(it)
    fan_3d_buf = x_3d_buf

    if not return_sharded:
        if pair_ag_relay and ep in (8, 16):
            ag_ssem = next(it)
            ag_psem = next(it)
            fan_send_sem = fan_recv_sem = None
        else:
            fan_send_sem = next(it)
            fan_recv_sem = next(it)
            ag_ssem = ag_psem = None
    else:
        ag_ssem = ag_psem = fan_send_sem = fan_recv_sem = None

    # 1. Entry barrier across EP ranks.
    barrier_sem = pltpu.get_barrier_semaphore()
    for dest in range(ep):
        pl.semaphore_signal(
            barrier_sem,
            device_id={ep_axis_name: jnp.int32(dest)},
            device_id_type=pl.DeviceIdType.MESH,
        )

    me = rank_ref[0]
    gain_f32_buf[...] = norm_ref[...].reshape(1, _H).astype(jnp.float32) + 1.0

    # 2. Fused TopK routing + optional R>=1 replica planner + SMEM token pack.
    if has_replicas:
        extract_my_holder_candidates_smem(
            me, num_replicas, replica_meta_ref, holder_cand_smem
        )
        cand_eids = tuple(holder_cand_smem[s] for s in range(num_replicas))
    else:
        cand_eids = ()

    route_topk_and_counts_in_kernel(
        logits_ref,
        bias_ref,
        r_scratch,
        misc_sem.at[0],
        my_owner=me,
        e_per_owner=e_local,
        holder_cand_eids=cand_eids,
        topk=_TOPK,
        route_eps=1e-15,
        route_scale=1.0,
        route_chunk_rows=min(batch_size, 32),
    )
    if has_replicas:
        run_replica_planner_smem(
            me,
            e_local,
            ep,
            num_replicas,
            r_scratch.counts_smem,
            replica_meta_ref,
            local_suppress_smem,
            holder_slot_expert_smem,
            owner_load_smem,
            cost_mode=replica_cost_mode,
            tile_m=32,
            moved_flag_smem=moved_flag_smem,
        )
        num_exec = build_exec_plan_from_masks_smem(
            me,
            e_local,
            num_replicas,
            local_suppress_smem,
            holder_slot_expert_smem,
            r_scratch,
        )
    else:
        num_exec = build_default_exec_plan_smem(me, e_local, r_scratch)

    exec_src_smem = r_scratch.exec_weight_src
    exec_slot_smem = r_scratch.exec_weight_slot
    pack_active_tokens_smem(batch_size, r_scratch, has_replicas=has_replicas)

    # 3. Unstrided 8-MiB circular double-ring DMA helpers for W1 and W2.
    def _start_w1_q(r_slot, src_bank, w_slot, q_idx):
        if has_replicas:
            @pl.when(src_bank == 0)
            def _loc_w1():
                pltpu.make_async_copy(
                    w1_ref.at[w_slot, pl.ds(q_idx * 1024, 1024), :],
                    w1_ring.at[r_slot],
                    w1_sem.at[r_slot],
                ).start()

            @pl.when(src_bank != 0)
            def _rep_w1():
                pltpu.make_async_copy(
                    rep_w1_ref.at[w_slot, pl.ds(q_idx * 1024, 1024), :],
                    w1_ring.at[r_slot],
                    w1_sem.at[r_slot],
                ).start()
        else:
            pltpu.make_async_copy(
                w1_ref.at[w_slot, pl.ds(q_idx * 1024, 1024), :],
                w1_ring.at[r_slot],
                w1_sem.at[r_slot],
            ).start()

    def _wait_w1_q(r_slot):
        pltpu.make_async_copy(
            w1_ring.at[r_slot], w1_ring.at[r_slot], w1_sem.at[r_slot]
        ).wait()

    def _start_w2_h(r_slot, src_bank, w_slot, h_idx):
        if has_replicas:
            @pl.when(src_bank == 0)
            def _loc_w2():
                pltpu.make_async_copy(
                    w2_ref.at[w_slot, pl.ds(h_idx * 2048, 2048), :],
                    w2_ring.at[r_slot],
                    w2_sem.at[r_slot],
                ).start()

            @pl.when(src_bank != 0)
            def _rep_w2():
                pltpu.make_async_copy(
                    rep_w2_ref.at[w_slot, pl.ds(h_idx * 2048, 2048), :],
                    w2_ring.at[r_slot],
                    w2_sem.at[r_slot],
                ).start()
        else:
            pltpu.make_async_copy(
                w2_ref.at[w_slot, pl.ds(h_idx * 2048, 2048), :],
                w2_ring.at[r_slot],
                w2_sem.at[r_slot],
            ).start()

    def _wait_w2_h(r_slot):
        pltpu.make_async_copy(
            w2_ring.at[r_slot], w2_ring.at[r_slot], w2_sem.at[r_slot]
        ).wait()

    # 4. Prologue: launch expert 0's W1 q=0,1 (16 MiB) + W2 h=0,1 (16 MiB) while initializing VMEM!
    @pl.when(num_exec > 0)
    def _prime_fp8_rings():
        s0_b = exec_src_smem[0]
        sl0 = exec_slot_smem[0]
        _start_w1_q(0, s0_b, sl0, jnp.int32(0))
        _start_w1_q(1, s0_b, sl0, jnp.int32(1))
        _start_w2_h(0, s0_b, sl0, jnp.int32(0))
        _start_w2_h(1, s0_b, sl0, jnp.int32(1))

    zero_rc = min(batch_size, 16)

    def _zero_acc_fp8(ci, carry_z):
        r0 = ci * zero_rc
        accum_3d[pl.ds(r0, zero_rc), :, :] = jnp.zeros(
            (zero_rc, 32, 128), dtype=jnp.float32
        )
        for h_i in range(2):
            x_3d_buf[pl.ds(r0, zero_rc), pl.ds(h_i * 16, 16), :] = x_ref[
                pl.ds(r0, zero_rc), pl.ds(h_i * 2048, 2048)
            ].reshape(zero_rc, 16, 128)
        return carry_z

    lax.fori_loop(0, batch_size // zero_rc, _zero_acc_fp8, jnp.int32(0))

    e_iota_loc = lax.broadcasted_iota(jnp.int32, (e_local, 1), 0)
    e_iota_rep = (
        lax.broadcasted_iota(jnp.int32, (r_pad, 1), 0) if has_replicas else None
    )

    def _load_s1_strip(src_bank, w_slot, col_off: int, nc: int) -> jax.Array:
        loc_blk = s1_ref[pl.ds(0, e_local), pl.ds(col_off, nc)]
        loc_row = jnp.sum(
            jnp.where(e_iota_loc == w_slot, loc_blk, 0.0), axis=0, keepdims=True
        )
        if not has_replicas:
            return loc_row
        rep_blk = rep_s1_ref[pl.ds(0, r_pad), pl.ds(col_off, nc)]
        rep_row = jnp.sum(
            jnp.where(e_iota_rep == w_slot, rep_blk, 0.0), axis=0, keepdims=True
        )
        return jnp.where(src_bank == 0, loc_row, rep_row)

    def _load_s2_strip(src_bank, w_slot, col_off: int, nc: int) -> jax.Array:
        loc_blk = s2_ref[pl.ds(0, e_local), pl.ds(col_off, nc)]
        loc_row = jnp.sum(
            jnp.where(e_iota_loc == w_slot, loc_blk, 0.0), axis=0, keepdims=True
        )
        if not has_replicas:
            return loc_row
        rep_blk = rep_s2_ref[pl.ds(0, r_pad), pl.ds(col_off, nc)]
        rep_row = jnp.sum(
            jnp.where(e_iota_rep == w_slot, rep_blk, 0.0), axis=0, keepdims=True
        )
        return jnp.where(src_bank == 0, loc_row, rep_row)

    # 5. Main pipelined FP8 expert loop (1.0x HBM weight read per active expert at any r_e!).
    def _fp8_expert_body(k_exp, carry_exp):
        base_pos = r_scratch.exec_start[k_exp]
        cnt_e = r_scratch.exec_start[k_exp + 1] - base_pos
        src_cur = exec_src_smem[k_exp]
        slot_cur = exec_slot_smem[k_exp]

        has_nxt = (k_exp + 1) < num_exec
        nxt_idx = jnp.where(has_nxt, k_exp + 1, jnp.int32(0))
        nxt_src = exec_src_smem[nxt_idx]
        nxt_slot = exec_slot_smem[nxt_idx]

        n_mslices = jnp.maximum(jnp.int32(1), (cnt_e + (m_step - 1)) // m_step)

        # Gather cnt_e active rows into lhs_exp_3d[0:cnt_e] while W1/W2 DMAs stream!
        n8_g = cnt_e >> 3

        def _g8_fp8(q_g, c_g):
            i0 = pl.multiple_of(q_g << 3, 8)
            p0 = base_pos + i0
            rows = [
                x_3d_buf[pl.ds(r_scratch.packed_tok_ids[p0 + s_i], 1), :, :]
                for s_i in range(8)
            ]
            lhs_exp_3d[pl.ds(i0, 8), :, :] = jnp.concatenate(rows, axis=0)
            return c_g

        lax.fori_loop(0, n8_g, _g8_fp8, jnp.int32(0))

        def _gtail_fp8(i_g, c_g):
            tok = r_scratch.packed_tok_ids[base_pos + i_g]
            lhs_exp_3d[pl.ds(i_g, 1), :, :] = x_3d_buf[pl.ds(tok, 1), :, :]
            return c_g

        lax.fori_loop(n8_g << 3, cnt_e, _gtail_fp8, jnp.int32(0))

        def _load_lhs_slice(r0):
            for sub in range(0, m_step, 16):
                rr = r0 + sub
                for h_i in range(2):
                    lhs_m_2d[pl.ds(sub, 16), pl.ds(h_i * 2048, 2048)] = (
                        lhs_exp_3d[pl.ds(rr, 16), pl.ds(h_i * 16, 16), :].reshape(
                            16, 2048
                        )
                    )

        def _accum_w1_quarter(r_slot: int, k_off: int, is_first_q: bool):
            def _m_body(mi, c_m):
                r0 = pl.multiple_of(mi * m_step, m_step)
                _load_lhs_slice(r0)
                x_q = lhs_m_2d[:, k_off : k_off + 1024]
                for c0 in range(0, _N1, 2048):
                    col_s = pl.ds(c0, 2048)
                    w_q = w1_ring[r_slot, :, col_s]
                    part = jnp.dot(x_q, w_q, preferred_element_type=jnp.float32)
                    if is_first_q:
                        gu_acc_2d[pl.ds(r0, m_step), col_s] = part
                    else:
                        gu_acc_2d[pl.ds(r0, m_step), col_s] = (
                            gu_acc_2d[pl.ds(r0, m_step), col_s] + part
                        )
                return c_m

            lax.fori_loop(0, n_mslices, _m_body, jnp.int32(0))

        # --- W1 q=0 (K=0..1024, slot 0) -> immediately launch q=2 into slot 0! ---
        _wait_w1_q(0)
        _accum_w1_quarter(0, 0, True)
        _start_w1_q(0, src_cur, slot_cur, jnp.int32(2))

        # --- W1 q=1 (K=1024..2048, slot 1) -> immediately launch q=3 into slot 1! ---
        _wait_w1_q(1)
        _accum_w1_quarter(1, 1024, False)
        _start_w1_q(1, src_cur, slot_cur, jnp.int32(3))

        # --- W1 q=2 (K=2048..3072, slot 0) -> immediately prefetch next expert's q=0! ---
        _wait_w1_q(0)
        _accum_w1_quarter(0, 2048, False)

        @pl.when(has_nxt)
        def _pf_nxt_w1_0():
            _start_w1_q(0, nxt_src, nxt_slot, jnp.int32(0))

        # --- W1 q=3 (K=3072..4096, slot 1) + channelwise scale + SwiGLU -> prefetch next q=1! ---
        _wait_w1_q(1)

        def _finish_w1_m(mi, c_m):
            r0 = pl.multiple_of(mi * m_step, m_step)
            _load_lhs_slice(r0)
            x_q3 = lhs_m_2d[:, 3072:4096]
            for n0 in range(0, _I, 1024):
                s1_g = _load_s1_strip(src_cur, slot_cur, n0, 1024)
                s1_u = _load_s1_strip(src_cur, slot_cur, _I + n0, 1024)
                wg3 = w1_ring[1, :, pl.ds(n0, 1024)]
                g = (
                    gu_acc_2d[pl.ds(r0, m_step), pl.ds(n0, 1024)]
                    + jnp.dot(x_q3, wg3, preferred_element_type=jnp.float32)
                ) * s1_g
                wu3 = w1_ring[1, :, pl.ds(_I + n0, 1024)]
                u = (
                    gu_acc_2d[pl.ds(r0, m_step), pl.ds(_I + n0, 1024)]
                    + jnp.dot(x_q3, wu3, preferred_element_type=jnp.float32)
                ) * s1_u
                h_act = (
                    jax.nn.silu(g).astype(jnp.bfloat16).astype(jnp.float32) * u
                ).astype(jnp.bfloat16)
                act_exp_2d[pl.ds(r0, m_step), pl.ds(n0, 1024)] = h_act
            return c_m

        lax.fori_loop(0, n_mslices, _finish_w1_m, jnp.int32(0))

        @pl.when(has_nxt)
        def _pf_nxt_w1_1():
            _start_w1_q(1, nxt_src, nxt_slot, jnp.int32(1))

        # --- W2 h=0 (K=0..2048, slot 0; reuse gu_acc_2d[:, 0:4096]!) -> prefetch next W2 h=0! ---
        _wait_w2_h(0)

        def _w2_h0_m(mi, c_m):
            r0 = pl.multiple_of(mi * m_step, m_step)
            a0 = act_exp_2d[pl.ds(r0, m_step), pl.ds(0, 2048)]
            for c0 in range(0, _H, 2048):
                col_s = pl.ds(c0, 2048)
                w2_0 = w2_ring[0, :, col_s]
                gu_acc_2d[pl.ds(r0, m_step), col_s] = jnp.dot(
                    a0, w2_0, preferred_element_type=jnp.float32
                )
            return c_m

        lax.fori_loop(0, n_mslices, _w2_h0_m, jnp.int32(0))

        @pl.when(has_nxt)
        def _pf_nxt_w2_0():
            _start_w2_h(0, nxt_src, nxt_slot, jnp.int32(0))

        # --- W2 h=1 (K=2048..4096, slot 1) + scale + gain -> prefetch next W2 h=1! ---
        _wait_w2_h(1)

        def _w2_h1_m(mi, c_m):
            r0 = pl.multiple_of(mi * m_step, m_step)
            a1 = act_exp_2d[pl.ds(r0, m_step), pl.ds(2048, 2048)]
            for c0 in range(0, _H, 2048):
                col_s = pl.ds(c0, 2048)
                s2_c = _load_s2_strip(src_cur, slot_cur, c0, 2048)
                g_col = gain_f32_buf[0:1, col_s]
                w2_1 = w2_ring[1, :, col_s]
                d = (
                    gu_acc_2d[pl.ds(r0, m_step), col_s]
                    + jnp.dot(a1, w2_1, preferred_element_type=jnp.float32)
                ) * s2_c
                d_bf = d.astype(jnp.bfloat16)
                if norm_f32_stage:
                    gu_acc_2d[pl.ds(r0, m_step), col_s] = (
                        d_bf.astype(jnp.float32) * g_col
                    )
                else:
                    gu_acc_2d[pl.ds(r0, m_step), col_s] = (
                        d_bf * g_col.astype(jnp.bfloat16)
                    ).astype(jnp.float32)
            return c_m

        lax.fori_loop(0, n_mslices, _w2_h1_m, jnp.int32(0))

        @pl.when(has_nxt)
        def _pf_nxt_w2_1():
            _start_w2_h(1, nxt_src, nxt_slot, jnp.int32(1))

        # --- Fused post_expert_norm RMSNorm + weighted FP32 scatter into accum_3d ---
        n8_s = (cnt_e + 7) >> 3

        def _scatter8_fp8(q_s, c_s):
            i0 = pl.multiple_of(q_s << 3, 8)
            u0_f = gu_acc_2d[pl.ds(i0, 8), pl.ds(0, 2048)]
            u1_f = gu_acc_2d[pl.ds(i0, 8), pl.ds(2048, 2048)]
            sq8 = jnp.sum(u0_f * u0_f, axis=1, keepdims=True) + jnp.sum(
                u1_f * u1_f, axis=1, keepdims=True
            )
            inv8 = lax.rsqrt(sq8 * (1.0 / _H) + 1e-8)
            if norm_f32_stage:
                y0 = (u0_f * inv8).astype(jnp.bfloat16).reshape(8, 16, 128)
                y1 = (u1_f * inv8).astype(jnp.bfloat16).reshape(8, 16, 128)
            else:
                inv8_bf = inv8.astype(jnp.bfloat16)
                y0 = (
                    (u0_f.astype(jnp.bfloat16) * inv8_bf)
                    .reshape(8, 16, 128)
                )
                y1 = (
                    (u1_f.astype(jnp.bfloat16) * inv8_bf)
                    .reshape(8, 16, 128)
                )
            y8 = jnp.concatenate([y0, y1], axis=1)
            for s_i in range(8):
                idx_s = i0 + s_i

                @pl.when(idx_s < cnt_e)
                def _do_sc(s_i=s_i, idx_s=idx_s):
                    tok = r_scratch.packed_tok_ids[base_pos + idx_s]
                    wt = r_scratch.packed_tok_w[base_pos + idx_s]
                    accum_3d[pl.ds(tok, 1), :, :] = (
                        accum_3d[pl.ds(tok, 1), :, :]
                        + y8[s_i : s_i + 1, :, :].astype(jnp.float32) * wt
                    )

            return c_s

        lax.fori_loop(0, n8_s, _scatter8_fp8, jnp.int32(0))
        return carry_exp

    lax.fori_loop(0, num_exec, _fp8_expert_body, jnp.int32(0))

    # 6. Reduce-Scatter + optional All-Gather (identical to _sparse_decode_v2_kernel).
    pl.semaphore_wait(barrier_sem, ep)

    if pair_rs and ep in (8, 16):
        num_chips = ep // 2
        if rs_wire_bf16:
            cast_rc = min(batch_size, 8)

            def _cast_fp8_p(ci, c_p):
                r0 = ci * cast_rc
                rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                    pl.ds(r0, cast_rc), :, :
                ].astype(jnp.bfloat16)
                return c_p

            lax.fori_loop(0, batch_size // cast_rc, _cast_fp8_p, jnp.int32(0))
            pair_src = rs_send_bf16
        else:
            pair_src = accum_3d

        core = me & jnp.int32(1)
        chip = me >> 1
        partner = jnp.bitwise_xor(me, jnp.int32(1))

        p1_ops = []
        for c in range(num_chips):
            dest_for_partner = jnp.int32(2 * c) + (jnp.int32(1) - core)
            op_p = pltpu.make_async_remote_copy(
                pair_src.at[
                    pl.ds(dest_for_partner * rows_per_rank, rows_per_rank), :, :
                ],
                rs_pair_recv.at[c],
                rs_pair_ssem.at[c],
                rs_pair_rsem.at[c],
                device_id={ep_axis_name: partner},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op_p.start()
            p1_ops.append(op_p)

        chip_ops = []
        for c in range(num_chips):
            pltpu.make_async_copy(
                rs_pair_recv.at[c], rs_pair_recv.at[c], rs_pair_rsem.at[c]
            ).wait()
            target_rank = jnp.int32(2 * c) + core

            @pl.when(c != chip)
            def _sum_chip_fp8(c=c, target_rank=target_rank):
                r_off = target_rank * rows_per_rank
                pair_f32 = accum_3d[
                    pl.ds(r_off, rows_per_rank), :, :
                ] + rs_pair_recv[c].astype(jnp.float32)
                rs_chip_send[c] = pair_f32.astype(rs_chip_send.dtype)

            op_c = pltpu.make_async_remote_copy(
                rs_chip_send.at[c],
                rs_chip_recv.at[chip],
                rs_chip_ssem.at[c],
                rs_chip_rsem,
                device_id={ep_axis_name: target_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            chip_ops.append((c, op_c))

            @pl.when(c != chip)
            def _st_chip_fp8(_op=op_c):
                _op.start()

        pltpu.make_async_copy(
            rs_chip_recv.at[pl.ds(0, num_chips - 1)],
            rs_chip_recv.at[pl.ds(0, num_chips - 1)],
            rs_chip_rsem,
        ).wait()

        red_rc = min(rows_per_rank, 4)

        def _red_own_fp8(ci, c_r):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for c in range(num_chips):
                part_c = jnp.where(
                    c == chip,
                    rs_pair_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                    rs_chip_recv[c, pl.ds(r0, red_rc), :, :].astype(jnp.float32),
                )
                acc = acc + part_c
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return c_r

        lax.fori_loop(0, rows_per_rank // red_rc, _red_own_fp8, jnp.int32(0))

        def _finish_fp8_rs():
            for op_p in p1_ops:
                op_p.wait_send()
            for c, op_c in chip_ops:
                @pl.when(c != chip)
                def _wc(_op=op_c):
                    _op.wait_send()
    else:
        if rs_wire_bf16:
            cast_rc = min(batch_size, 8)

            def _cast_fp8_f(ci, c_f):
                r0 = ci * cast_rc
                rs_send_bf16[pl.ds(r0, cast_rc), :, :] = accum_3d[
                    pl.ds(r0, cast_rc), :, :
                ].astype(jnp.bfloat16)
                return c_f

            lax.fori_loop(0, batch_size // cast_rc, _cast_fp8_f, jnp.int32(0))
            rs_src_ref = rs_send_bf16
        else:
            rs_src_ref = accum_3d

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
            def _st_rs_f(_op=copy_op):
                _op.start()

        pltpu.make_async_copy(rs_recv_3d, rs_recv_3d, rs_recv_sem).wait()
        red_rc = min(rows_per_rank, 4)

        def _red_f_fp8(ci, c_r):
            r0 = ci * red_rc
            acc = accum_3d[pl.ds(me * rows_per_rank + r0, red_rc), :, :]
            for src in range(ep - 1):
                acc = acc + rs_recv_3d[src, pl.ds(r0, red_rc), :, :].astype(
                    jnp.float32
                )
            owner_3d[pl.ds(r0, red_rc), :, :] = acc.astype(jnp.bfloat16)
            return c_r

        lax.fori_loop(0, rows_per_rank // red_rc, _red_f_fp8, jnp.int32(0))

        def _finish_fp8_rs():
            for d, copy_op in rs_copies:
                @pl.when(me != d)
                def _w_rs_f(_op=copy_op):
                    _op.wait_send()

    if return_sharded:
        if rows_per_rank >= 16:
            for r0 in range(0, rows_per_rank, 16):
                output_ref[pl.ds(r0, 16), :] = owner_3d[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            output_ref[...] = owner_3d[...].reshape(rows_per_rank, _H)
    elif pair_ag_relay and ep in (8, 16):
        num_chips = ep // 2
        core = me & jnp.int32(1)
        chip = me >> 1
        partner = jnp.bitwise_xor(me, jnp.int32(1))
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]

        def _ag_send_fp8(slot_rank, peer_rank, ssem_idx: int):
            r_off = slot_rank * rows_per_rank
            op = pltpu.make_async_remote_copy(
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                fan_3d_buf.at[pl.ds(r_off, rows_per_rank), :, :],
                ag_ssem.at[ssem_idx],
                ag_psem.at[slot_rank],
                device_id={ep_axis_name: peer_rank},
                device_id_type=pl.DeviceIdType.MESH,
            )
            op.start()
            return op

        def _ag_wait_fp8(slot_rank):
            pltpu.make_async_copy(
                owner_3d, owner_3d, ag_psem.at[slot_rank]
            ).wait()

        far_ranks = [
            jnp.int32(2) * jnp.bitwise_xor(chip, jnp.int32(d)) + core
            for d in range(1, num_chips)
        ]
        ag_ops = [_ag_send_fp8(me, partner, 0)]
        for idx_f, fk in enumerate(far_ranks):
            ag_ops.append(_ag_send_fp8(me, fk, 1 + idx_f))
        for idx_f, fk in enumerate(far_ranks):
            _ag_wait_fp8(fk)
            ag_ops.append(_ag_send_fp8(fk, partner, num_chips + idx_f))
        _ag_wait_fp8(partner)
        for fk in far_ranks:
            _ag_wait_fp8(jnp.bitwise_xor(fk, jnp.int32(1)))

        if batch_size >= 16:
            for r0 in range(0, batch_size, 16):
                output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            output_ref[...] = fan_3d_buf[...].reshape(batch_size, _H)
        for op_a in ag_ops:
            op_a.wait_send()
    else:
        my_row_off = me * rows_per_rank
        fan_3d_buf[pl.ds(my_row_off, rows_per_rank), :, :] = owner_3d[...]
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
            def _st_fan_f(_op=fan_op):
                _op.start()

        pltpu.make_async_copy(
            fan_3d_buf.at[pl.ds(0, (ep - 1) * rows_per_rank), :, :],
            fan_3d_buf.at[pl.ds(0, (ep - 1) * rows_per_rank), :, :],
            fan_recv_sem,
        ).wait()
        if batch_size >= 16:
            for r0 in range(0, batch_size, 16):
                output_ref[pl.ds(r0, 16), :] = fan_3d_buf[
                    pl.ds(r0, 16), :, :
                ].reshape(16, _H)
        else:
            output_ref[...] = fan_3d_buf[...].reshape(batch_size, _H)
        for dest, fan_op in fan_copies:
            @pl.when(me != dest)
            def _w_fan_f(_op=fan_op):
                _op.wait_send()

    _finish_fp8_rs()


def fused_decode_sparse_fp8_from_logits(
    rank: jax.Array,
    logits: jax.Array,
    correction_bias: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    post_norm_weight: jax.Array,
    rep_w1: jax.Array | None = None,
    rep_s1: jax.Array | None = None,
    rep_w2: jax.Array | None = None,
    rep_s2: jax.Array | None = None,
    replica_meta: jax.Array | None = None,
    *,
    ep: int = 8,
    return_sharded: bool = False,
    replica_cost_mode: str = "tiles",
    ep_axis_name: str = "model",
    pair_rs: bool = True,
    pair_ag_relay: bool = True,
    rs_wire_bf16: bool | None = None,
    norm_f32_stage: bool = False,
) -> jax.Array:
    """Single-kernel FP8 Sparse Fuse-All Decode MoE for Muse Spark L0 & L61."""
    batch_size = logits.shape[0]
    if batch_size % ep != 0 or batch_size < 16 or batch_size > 384:
        raise ValueError(f"Unsupported batch_size={batch_size} for ep={ep}")
    if rs_wire_bf16 is None:
        rs_wire_bf16 = batch_size >= 128

    e_local = _E_TOTAL // ep
    w_groups = w1.shape[0]
    if w1.dtype != jnp.float8_e4m3fn or w2.dtype != jnp.float8_e4m3fn:
        raise ValueError(f"Expected float8_e4m3fn weights, got {w1.dtype}, {w2.dtype}")
    if (
        not (e_local <= w_groups <= e_local + 8)
        or w1.shape != (w_groups, _H, _N1)
        or w2.shape != (w_groups, _I, _H)
    ):
        raise ValueError(f"Unexpected FP8 weight shapes: {w1.shape}, {w2.shape}")

    def _norm_fp8_scale(sc: jax.Array, e_cnt: int, out_n: int) -> jax.Array:
        s_f = sc.astype(jnp.float32).reshape(e_cnt, -1, out_n)
        return s_f[:e_local, 0, :]

    s1_2d = _norm_fp8_scale(s1, w_groups, _N1)
    s2_2d = _norm_fp8_scale(s2, w_groups, _H)

    has_replicas = replica_meta is not None
    num_replicas = int(rep_w1.shape[0]) if has_replicas else 0
    if has_replicas:
        r_pad = max(8, ((num_replicas + 7) // 8) * 8)
        rs1_raw = _norm_fp8_scale(rep_s1, num_replicas, _N1)
        rs2_raw = _norm_fp8_scale(rep_s2, num_replicas, _H)
        rep_s1_pad = (
            jnp.pad(rs1_raw, ((0, r_pad - num_replicas), (0, 0)))
            if r_pad > num_replicas
            else rs1_raw
        )
        rep_s2_pad = (
            jnp.pad(rs2_raw, ((0, r_pad - num_replicas), (0, 0)))
            if r_pad > num_replicas
            else rs2_raw
        )
    else:
        rep_s1_pad = rep_s2_pad = None

    bias_2d = correction_bias.reshape(1, _E_TOTAL).astype(jnp.float32)
    norm_2d = post_norm_weight.reshape(1, _H)
    x_2d = x.astype(jnp.bfloat16)

    vmem = pltpu.VMEM
    smem = pltpu.SMEM
    sem = pltpu.SemaphoreType.DMA
    v_spec = pl.BlockSpec(memory_space=vmem)
    h_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_exec = e_local + num_replicas
    m_step = 16 if batch_size <= 16 else 32
    rows_per_rank = batch_size // ep
    rs_wire_dtype = jnp.bfloat16 if rs_wire_bf16 else jnp.float32

    scratch_shapes: list = [
        *routing_scratch_shapes(
            batch_size, max_exec, topk=_TOPK, num_experts=_E_TOTAL
        ),
    ]
    if has_replicas:
        scratch_shapes.extend(
            [
                smem((ep,), jnp.int32),
                smem((_E_TOTAL,), jnp.int32),
                smem((e_local,), jnp.int32),
                smem((num_replicas,), jnp.int32),
                smem((num_replicas,), jnp.int32),
            ]
        )
    scratch_shapes.extend(
        [
            vmem((2, 1024, _N1), jnp.float8_e4m3fn),  # w1_ring (16 MiB)
            vmem((2, 2048, _H), jnp.float8_e4m3fn),  # w2_ring (16 MiB)
            sem((2,)),
            sem((2,)),
            sem((1,)),  # misc_sem
            vmem((batch_size, 32, 128), jnp.bfloat16),  # x_3d_buf
            vmem((batch_size, 32, 128), jnp.bfloat16),  # lhs_exp_3d
            vmem((m_step, _H), jnp.bfloat16),  # lhs_m_2d
            vmem((batch_size, _N1), jnp.float32),  # gu_acc_2d
            vmem((batch_size, _I), jnp.bfloat16),  # act_exp_2d
            vmem((1, _H), jnp.float32),  # gain_f32_buf
            vmem((batch_size, 32, 128), jnp.float32),  # accum_3d
        ]
    )
    if rs_wire_bf16:
        scratch_shapes.append(vmem((batch_size, 32, 128), jnp.bfloat16))
    if pair_rs and ep in (8, 16):
        num_chips_rs = ep // 2
        scratch_shapes.extend(
            [
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                vmem((num_chips_rs, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem((num_chips_rs,)),
                sem,
            ]
        )
    else:
        scratch_shapes.extend(
            [
                vmem((ep - 1, rows_per_rank, 32, 128), rs_wire_dtype),
                sem((ep,)),
                sem,
            ]
        )
    scratch_shapes.append(vmem((rows_per_rank, 32, 128), jnp.bfloat16))
    if not return_sharded:
        if pair_ag_relay and ep in (8, 16):
            num_chips_ag = ep // 2
            scratch_shapes.extend([sem((2 * num_chips_ag - 1,)), sem((ep,))])
        else:
            scratch_shapes.extend([sem((ep,)), sem])

    in_specs: list = [
        v_spec,  # logits
        v_spec,  # bias
        v_spec,  # x
        h_spec,  # w1
        v_spec,  # s1_2d (768 KiB total in VMEM!)
        h_spec,  # w2
        v_spec,  # s2_2d
        v_spec,  # norm
    ]
    extra_args: list = []
    if has_replicas:
        in_specs.extend([h_spec, v_spec, h_spec, v_spec])
        extra_args.extend([rep_w1, rep_s1_pad, rep_w2, rep_s2_pad])
        num_scalar_prefetch = 2
        prefix_args = (rank, replica_meta)
    else:
        num_scalar_prefetch = 1
        prefix_args = (rank,)

    call_args = (
        *prefix_args,
        logits,
        bias_2d,
        x_2d,
        w1,
        s1_2d,
        w2,
        s2_2d,
        norm_2d,
        *extra_args,
    )
    out_rows = rows_per_rank if return_sharded else batch_size
    out_shape = jax.ShapeDtypeStruct((out_rows, _H), jnp.bfloat16)

    kernel_fn = functools.partial(
        _sparse_decode_fp8_v2_kernel,
        ep=ep,
        batch_size=batch_size,
        num_replicas=num_replicas,
        return_sharded=return_sharded,
        replica_cost_mode=replica_cost_mode,
        ep_axis_name=ep_axis_name,
        pair_rs=pair_rs,
        pair_ag_relay=pair_ag_relay,
        rs_wire_bf16=rs_wire_bf16,
        norm_f32_stage=norm_f32_stage,
    )
    rep_tag = f"_r{num_replicas}" if has_replicas else ""
    out_tag = "_sh" if return_sharded else "_rep"
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            grid=(1,),
            in_specs=tuple(in_specs),
            out_specs=v_spec,
            scratch_shapes=tuple(scratch_shapes),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=64 * 1024**2,
            collective_id=0,
        ),
        name=f"muse_sparse_fp8_v2_ep{ep}_b{batch_size}_ms{m_step}{rep_tag}{out_tag}",
    )(*call_args)
