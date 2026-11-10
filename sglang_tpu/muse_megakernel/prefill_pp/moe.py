# SPDX-License-Identifier: Apache-2.0
"""The PP prefill stage's experts: sglang_tpu's fused MoE (sort, SparseCore gather, gmm_v2 with
fused SiLU, gmm_v2, un-permute gather), called as the native Muse layer calls it (no combine)."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.prefill_pp.config import (
    BF16,
    PPConfig,
    fp4_qblock,
    fp4_qblock_a8,
)

ACTIVATION = "silu"
E2M1_MAX = 6.0
E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_GROUP = 16  # K rows per served NVFP4 scale


def _requant_one(packed, scales, block: int):
    """One expert: [K / 2, N] u8 e2m1 pairs (low nibble row 2b) + [K / 32, N] i32 bf16 scale
    pairs (low half block 2j) -> codes and [K / (2 block), N] i32 scale pairs of block rows."""
    k2, n = packed.shape
    k = 2 * k2
    codes = (
        jnp.stack([packed & 0xF, packed >> 4], axis=1).reshape(k, n).astype(jnp.int32)
    )
    mags = jnp.asarray(E2M1_MAGS, jnp.float32)
    v = jnp.where(codes >= 8, -1.0, 1.0) * mags[codes & 7]
    u = jax.lax.bitcast_convert_type(scales, jnp.uint32)
    halves = jnp.stack([u & 0xFFFF, u >> 16], axis=1).astype(jnp.uint16)
    s16 = jax.lax.bitcast_convert_type(halves, BF16).astype(jnp.float32).reshape(-1, n)
    w = (v * jnp.repeat(s16, FP4_GROUP, axis=0)).reshape(k // block, block, n)
    sb = (jnp.max(jnp.abs(w), axis=1) / E2M1_MAX).astype(BF16)  # [K / block, N]
    sbf = sb.astype(jnp.float32)
    q = jnp.clip(w / jnp.where(sbf > 0, sbf, 1.0)[:, None, :], -E2M1_MAX, E2M1_MAX)
    qa = jnp.abs(
        q.astype(jnp.float4_e2m1fn).astype(jnp.float32)
    )  # round to nearest even
    idx = jnp.argmax(qa[..., None] == mags, axis=-1).astype(jnp.uint8)
    qc = (idx | jnp.where(q < 0, 8, 0).astype(jnp.uint8)).reshape(k2, 2, n)
    out = qc[:, 0] | (qc[:, 1] << 4)
    sh = (
        jax.lax.bitcast_convert_type(sb, jnp.uint16)
        .astype(jnp.uint32)
        .reshape(-1, 2, n)
    )
    pairs = jax.lax.bitcast_convert_type(sh[:, 0] | (sh[:, 1] << 16), jnp.int32)
    return out, pairs


def requant_fp4_blocks(packed, scales, block: int):
    """Served NVFP4 experts [E, K / 2, N] u8 + [E, K / 32, N] i32 -> the same codes layout
    requantized to one bf16 scale per block K rows (amax / 6, e2m1 re-rounded) and the scales
    as [E, K / (2 block), N] i32 pairs, the MUSE_PP_QBLOCK layout. Lossy."""
    return jax.lax.map(lambda a: _requant_one(a[0], a[1], block), (packed, scales))


def experts(c: PPConfig, xe, ids, tw, ex):
    """[T, top_k, He] bf16 slot outputs of xe [T, He] bf16 routed by ids / tw [T, >= top_k]
    through ex = (w13, w13_scale, w2, w2_scale): the served call (under MUSE_PP_QBLOCK, its
    routing around the block-scaled grouped matmuls)."""
    if fp4_qblock():
        rows, group_sizes, order, tw_k = _permute(c, xe, ids, tw)
        return _unpermute(c, _gmm(c, rows, group_sizes, ex), order, tw_k)
    # fused_moe_op loads torch_tpu at import; deviceless compiles of the rest go without it
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    w13, s13, w2, s2 = ex
    return _fused_moe_jax(
        xe, w13, w2, tw[:, : c.top_k].astype(BF16), ids[:, : c.top_k], s13, s2, None, None,
        c.top_k, ACTIVATION, no_combine=True, use_sc_gather=1,
    )  # fmt: skip


def _permute(c: PPConfig, xe, ids, tw):
    """_fused_moe_jax's routing for these shapes (aligned rows, all experts local, SparseCore
    gather): (expert-major rows, group sizes, sort order, slot weights)."""
    from sglang_tpu.moe.pallas.fused_moe_op import (
        _drop_invalid_slots,
        _padded_token_count,
    )
    from sglang_tpu.moe.pallas.gmm_v2 import lhs_sublane_tiling
    from sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2 import ragged_gather_v2

    T, K, E = xe.shape[0], c.top_k, c.n_experts
    if _padded_token_count(T, K, lhs_sublane_tiling(xe.dtype)) != T:
        raise ValueError(f"{T} x {K} routed rows do not tile gmm_v2's LHS")
    xe, ids, tw = _drop_invalid_slots(
        xe, ids[:, :K], tw[:, :K].astype(BF16), E, zero_dropped_tokens=False
    )
    flat = ids.reshape(-1).astype(jnp.int32)
    order = jnp.argsort(flat)
    n = flat.shape[0]
    rows = ragged_gather_v2(
        xe,
        (order // K).astype(jnp.int32),
        jnp.array([0], jnp.int32),
        jnp.array([n], jnp.int32),
    )
    group_sizes = jax.nn.one_hot(flat, E, dtype=jnp.int32).sum(axis=0)
    return rows, group_sizes, order, tw


def _gmm(c: PPConfig, rows, group_sizes, ex):
    """_fused_moe_jax's two grouped matmuls (fused SiLU in the first), its tiles and options."""
    from sglang_tpu.moe.pallas.fused_moe_op import (
        _dividing_tile_k,
        _fuse_act_name,
        _gmm2_tile_source,
        _lookup_tile_info,
        _tuned_or_calculated,
        _vmem_safe_packed_tile_n,
    )
    from sglang_tpu.moe.pallas.gmm_v2 import calculate_tiling, gmm_v2

    w1, s1, w2, s2 = ex
    E = w1.shape[0]
    w1_packed, w2_packed = w1.dtype == jnp.uint8, w2.dtype == jnp.uint8
    if s1.shape[-1] != w1.shape[-1]:
        raise ValueError("compact block scales run _fused_moe_jax's unfused GMM1")
    qb = fp4_qblock() if w1_packed else 0
    if qb and not w2_packed:
        raise ValueError("MUSE_PP_QBLOCK needs both NVFP4 projections")
    tile1 = (
        _lookup_tile_info(rows, w1, s1, E, packed_fp4=True, gmm_v2_only=True)
        if w1_packed
        else None
    )
    tile1 = _dividing_tile_k(
        _vmem_safe_packed_tile_n(
            tile1
            or (
                calculate_tiling if w1_packed else _tuned_or_calculated(rows, w1, s1, E)
            )
        )
    )
    common = dict(
        group_sizes=group_sizes, rhs_bias=None, rhs_channel_scale=None, group_offset=None,
        zero_initialize=True, preferred_element_type=BF16,
        maybe_quantize_lhs=fp4_qblock_a8() if qb else True,
        rhs_quant_dtype=None, dequantize_rhs=False,
    )  # fmt: skip
    h = gmm_v2(
        lhs=rows,
        rhs=w1,
        rhs_scale=s1,
        tile_info=tile1,
        fuse_act=_fuse_act_name(ACTIVATION),
        packed_fp4=w1_packed,
        fp4_qblock=qb or None,
        **common,
    )
    tile2 = _dividing_tile_k(
        _vmem_safe_packed_tile_n(_gmm2_tile_source(h, w2, s2, E, w2_packed=w2_packed))
    )
    y = gmm_v2(
        lhs=h, rhs=w2, rhs_scale=s2, tile_info=tile2, packed_fp4=w2_packed,
        fp4_qblock=qb or None, **common,
    )  # fmt: skip
    return y[:, : c.expert_hidden]


def _unpermute(c: PPConfig, y, order, tw):
    """_fused_moe_jax's no-combine un-permute: token-major [T, top_k, He], dropped slots 0."""
    from sglang_tpu.moe.pallas.sparse_core.ragged_gather_v2 import ragged_gather_v2

    T, K = tw.shape
    n = order.shape[0]
    back = ragged_gather_v2(
        y, jnp.argsort(order).astype(jnp.int32), jnp.array([0], jnp.int32),
        jnp.array([n], jnp.int32),
    ).reshape(T, K, c.expert_hidden)  # fmt: skip
    return jnp.where(
        (tw != 0)[:, :, None], back.astype(BF16), jnp.zeros_like(back, BF16)
    )


def slot_experts(c: PPConfig, slot: int, is_fp8, xe, ids, tw, ex_fp4, ex_fp8):
    """One slot's experts. A slot that holds a stage's fp8 layer on some stage picks the set at
    run time; the SparseCore gathers cannot sit in a conditional, so there the served
    permute / un-permute run outside it and only the grouped matmuls inside. Elsewhere the call
    is the served one."""
    if slot not in c.fp8_slots:
        return experts(c, xe, ids, tw, ex_fp4)
    rows, group_sizes, order, tw8 = _permute(c, xe, ids, tw)
    y = jax.lax.cond(
        is_fp8,
        lambda: _gmm(c, rows, group_sizes, ex_fp8),
        lambda: _gmm(c, rows, group_sizes, ex_fp4),
    )
    return _unpermute(c, y, order, tw8)


def no_experts(c: PPConfig):
    """MUSE_PP_SKIP=experts: zeros (timing ablation only)."""
    return jnp.zeros((c.rows, c.top_k, c.expert_hidden), BF16)
