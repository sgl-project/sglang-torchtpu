# SPDX-License-Identifier: Apache-2.0

"""Fused MoE local per-chip execution using the ICI reduce-scatter kernel (gmm_fused_rs).

Routing order and group sizes are computed once before the per-chip kernel.
Receive accounting stays with the kernel so it cannot diverge from issued DMAs.
"""

import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    _select_fused_rs_block_sizes,
    compute_num_gm,
)
from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup,
)
from sglang.srt.layers.moe.pallas.gmm_v2 import recover_quant_block_size

EXPERT = "model"

# ============================================================================
# SMEM safety guard
# ============================================================================

# ``gmm_v2_fused_rs`` SMEM budget (v7x): ~1.0 MB. Per-row scalar-prefetch
# arrays (``lhs_indices``, ``output_indices``, ``topk_weights``, etc.) scale
# with ``size_m`` as ``4 * size_m`` bytes each, so at ``m = 262144`` the
# kernel overflows SMEM by ~40 KB. We guard at ``m >= 262144`` (empirical
# OOM threshold); callers hitting this must reduce ``max-num-batched-tokens``,
# switch to ``gmm_ep_dma`` / ``gmm_ep_original``, or shard more (tp>1).
_FUSED_RS_MAX_SAFE_SIZE_M = 262144


def _assert_fused_rs_smem_safe(size_m: int) -> None:
    """Assert ``size_m`` fits the v7x SMEM budget for ``gmm_v2_fused_rs``."""
    assert size_m < _FUSED_RS_MAX_SAFE_SIZE_M, (
        f"gmm_v2_fused_rs SMEM OOM at size_m={size_m} (>= "
        f"{_FUSED_RS_MAX_SAFE_SIZE_M}). Per-row scalar-prefetch arrays "
        f"would exceed the 1 MB SMEM budget."
    )


# ============================================================================
# Per-chip MoE body (nodedup kernel variant)
# ============================================================================


def moe_gmm_local_rs_nodedup(
    hidden_states_local: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    w1_global_scale: jax.Array | None = None,
    w2_global_scale: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    topk_weights: jax.Array | None = None,
    output_indices: jax.Array | None = None,
    group_sizes: jax.Array | None = None,
    local_start: jax.Array | None = None,
    post_expert_norm_weight_input: jax.Array | None = None,
    *,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name: str | None = EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = False,
    fp8_direct_write: bool = False,
    num_tokens: int | None = None,
    max_num_gm: int | jax.Array | None = None,
) -> jax.Array:
    """Per-chip local MoE execution using Pallas GMM with fused ICI reduce-scatter.

    Routing order, group sizes, and the local start are precomputed by the JAX
    wrapper. The fused kernel derives its matching receive total itself.

    Returns:
        jax.Array: Local token hidden states of shape (chunk_size, hidden_size).
    """
    if num_tokens is None:
        num_tokens = hidden_states_local.shape[0]
    hidden_size = w2.shape[-1]
    chunk_size = (num_tokens + ep_size - 1) // ep_size
    num_local_experts = w1.shape[0]

    go_flat = (
        group_offset.reshape(-1)
        if group_offset is not None
        else jnp.array([0, 0], dtype=jnp.int32)
    )
    go_val = go_flat[0]

    assert topk_weights is not None, "topk_weights must be provided."
    assert (
        output_indices is not None
    ), "output_indices must be precomputed and provided."
    assert group_sizes is not None, "group_sizes must be precomputed and provided."
    assert local_start is not None, "local_start must be precomputed and provided."

    lhs_indices = (output_indices // topk).astype(jnp.int32)
    topk_slot_indices = (output_indices % topk).astype(jnp.int32)

    size_m = lhs_indices.shape[0]
    _assert_fused_rs_smem_safe(size_m)

    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=w1.shape[1],
        size_n1=w1.shape[2],
        size_k2=w2.shape[1],
        size_n2=w2.shape[2],
        size_group=num_local_experts,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=hidden_states_local.dtype,
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        is_quantized=w1_scale is not None,
        quant_block_size=(
            recover_quant_block_size(
                w1.shape[1],
                (
                    w1_scale.shape[1]
                    if (w1_scale.ndim > 1 and w1_scale.shape[1] > 1)
                    else 1
                ),
            )
            if w1_scale is not None
            else None
        ),
        act_fn=activation,
        fp8_direct_write=fp8_direct_write,
    )
    tile_m = block_sizes.tile_m
    size_lhs_sublane = min(
        pltpu.get_tpu_info().get_sublane_tiling(hidden_states_local.dtype),
        tile_m,
        size_m,
    )
    # Dynamic max_num_gm computation across all EP shards
    if max_num_gm is None:
        global_starts = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(group_sizes)[:-1].astype(jnp.int32),
            ]
        )
        shard_num_gms = []
        for i in range(ep_size):
            s_go = i * num_local_experts
            s_gs = jax.lax.dynamic_slice(group_sizes, (s_go,), (num_local_experts,))
            shard_num_gms.append(
                compute_num_gm(
                    jax.lax.dynamic_slice(global_starts, (s_go,), (num_local_experts,)),
                    s_gs,
                    tile_m,
                    size_lhs_sublane,
                )
            )
        max_num_gm = jnp.max(jnp.stack(shard_num_gms))

    out_buf = gmm_v2_fused_rs_nodedup(
        hidden_states_local,
        w1,
        w2,
        group_sizes,
        lhs_indices,
        output_indices,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        act_fn=activation,
        output_size=num_tokens,
        group_offset=group_offset,
        group_start=jnp.asarray(local_start).reshape(-1),
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        caller_tile_m=tile_m,
        top_k=topk,
        fp8_direct_write=fp8_direct_write,
    )

    out_3d = out_buf.reshape(chunk_size, topk, hidden_size)
    post_expert_norm_weight = post_expert_norm_weight_input if has_post_norm else None
    if post_expert_norm_weight is not None:
        norm_size = post_expert_norm_weight.shape[0]  # actual hidden_size (unpadded)
        pnw_raw = post_expert_norm_weight.astype(jnp.float32) + 1.0
        # Handle padded hidden: out_3d may have padded_hidden_size > norm_size
        if hidden_size > norm_size:
            col_idx = jnp.arange(hidden_size, dtype=jnp.int32)
            # Zero out padded columns so they don't affect variance
            valid_mask = (col_idx < norm_size)[None, None, :]
            out_f32 = out_3d.astype(jnp.float32) * valid_mask
            # Pad PNW to match hidden_size
            pnw = jnp.concatenate(
                [pnw_raw, jnp.zeros(hidden_size - norm_size, dtype=jnp.float32)]
            )
        else:
            out_f32 = out_3d.astype(jnp.float32)
            pnw = pnw_raw
        out_f32 = out_f32 * pnw[None, None, :]
        # Variance computed only over valid (unpadded) elements
        var = jnp.sum(out_f32**2, axis=-1, keepdims=True) / norm_size
        out_3d = (out_f32 * jax.lax.rsqrt(var + 1e-8)).astype(out_3d.dtype)

    # Post-kernel: apply topk weights and reduce over top_k.
    output_size = max(chunk_size, 1)
    logical_rank = go_val // num_local_experts
    my_token_start = logical_rank * output_size
    max_start = jnp.maximum(topk_weights.shape[0] - output_size, 0)
    clamped_token_start = jnp.minimum(my_token_start, max_start)
    my_weights = (
        topk_weights
        if sp_enabled
        # SP off: topk_weights is replicated; slice this chip's [chunk_size]
        # shard to match out_3d (SP on already passes the local shard).
        else jax.lax.dynamic_slice_in_dim(
            topk_weights, clamped_token_start, output_size, axis=0
        )
    )

    # Return the per-chip token shard (reduce-scattered output).
    return jnp.sum(out_3d * my_weights[:, :, None], axis=1).astype(
        hidden_states_local.dtype
    )


_MOE_GMM_LOCAL_RS = moe_gmm_local_rs_nodedup


__all__ = [
    "moe_gmm_local_rs_nodedup",
    "_MOE_GMM_LOCAL_RS",
    "_FUSED_RS_MAX_SAFE_SIZE_M",
    "_assert_fused_rs_smem_safe",
]
