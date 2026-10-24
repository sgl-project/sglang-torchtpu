# SPDX-License-Identifier: Apache-2.0

"""Fused MoE entry point using the ICI reduce-scatter kernel (gmm_fused_rs).

This path fuses ``gather + GMM1 + act + GMM2 + ICI reduce-scatter`` into a
single Pallas kernel call (``gmm_v2_fused_rs``), eliminating the need for
an explicit ``psum`` all-reduce after GMM2. The per-chip shard_map body
computes routing indices (argsort + group_sizes) INSIDE the shard_map so
XLA can fuse them with the kernel pipeline.

Entry point: ``fused_moe_func_rs`` — EP-only, with optional
``post_expert_norm_weight`` (input-scaled RMS norm) applied post-kernel before
the top-k weighted reduction.

When SP is disabled the kernel returns the per-chip reduce-scattered token
shard (sharded over the expert axis); the model (forward_tpu_moe) gathers it
back to the full replicated batch. The kernel can additionally quantize the
in-kernel ICI reduce-scatter payload to ``float8_e4m3fn`` on device, reducing
ICI traffic without a host roundtrip. Per-expert post-normalization is applied
before the post-kernel top-k weighted reduction.
"""

import functools

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    _select_fused_rs_block_sizes,
)
from sglang_tpu.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup,
)
from sglang_tpu.moe.pallas.fused_rs.post_combine import (
    fused_post_combine,
    pair_post_combine,
    supports_fused_post_combine,
)
from sglang_tpu.moe.pallas.fused_rs.utils import (
    combine_partition_axes,
    get_mesh_shape_product,
    get_moe_expert_axis,
)
from sglang_tpu.moe.pallas.gmm_v2 import recover_quant_block_size

EXPERT = "model"
MLP_DATA = "data"

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
        f"would exceed the 1 MB SMEM budget. Reduce "
        f"max-num-batched-tokens × top_k, switch to ep_backend "
        f"'gmm_ep_dma'/'gmm_ep_original', or increase tp so m per chip "
        f"is smaller."
    )


# NOTE: the SP-off post-kernel gather of the reduce-scattered token shard back to
# the full replicated batch (previously `_all_gather_token_hidden` here) now lives
# in the model-side TPU MoE gather path.
# The kernel returns the EP-token-sharded output (see expert_parallel_gmm_rs).


# ============================================================================
# Routing helper (inline in shard_map)
# ============================================================================


def _compute_rs_routing(topk_indices, *, num_experts, topk):
    """Inline routing for the RS path. Computes:
      - lhs_indices: int32[size_m] — gather indices (token ID per row)
      - group_sizes: int32[num_experts] — rows per expert
      - output_indices: int32[size_m] — = lhs_indices for EP
      - topk_slot_indices: int32[size_m] — top-k slot (0..topk-1) per row

    Two TPU-specific perf optimizations vs the textbook formulation:

    1. ``lhs_indices`` and ``topk_slot_indices`` computed via integer arithmetic
       (``// topk``, ``% topk``) instead of gathers — avoids gather_custom_fusion.

    2. ``group_sizes`` uses one-hot + sum instead of ``jnp.bincount`` — avoids
       contended scatter_add. ~7.7x faster at 65K rows.

    Note: dtype must remain int32 — Pallas/Mosaic requires 32-bit signless
    integers for scalar prefetch refs (``tpu.memref_slice`` constraint).
    """
    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    group_sizes = jnp.sum(
        (topk_indices_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
        axis=0,
    )
    lhs_indices = topk_argsort_indices // topk
    topk_slot_indices = topk_argsort_indices % topk
    output_indices = lhs_indices
    return lhs_indices, group_sizes, output_indices, topk_slot_indices


def _compute_expert_tp_routing(
    topk_indices: jax.Array, *, num_experts: int, topk: int, ep_size: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Expert-TP routing over virtual experts ``shard * E + expert``.

    Every (token, slot) is routed to all ``ep_size`` slices of its expert, as
    kernel slot ``slot * ep_size + shard``. Sorting by virtual expert is the
    real expert-sorted order repeated once per shard, so derive it from the
    real routing instead of sorting and bincounting ``ep_size`` times as many
    rows over ``ep_size`` times as many experts.
    """
    lhs_real, group_sizes_real, _, slot_real = _compute_rs_routing(
        topk_indices, num_experts=num_experts, topk=topk
    )
    real_order = lhs_real * topk + slot_real  # flat (token, slot), expert-sorted
    shards = jnp.arange(ep_size, dtype=jnp.int32)
    virtual_order = (real_order[None, :] * ep_size + shards[:, None]).reshape(-1)
    kernel_topk = topk * ep_size
    lhs_indices = virtual_order // kernel_topk
    topk_slot_indices = virtual_order % kernel_topk
    group_sizes = jnp.tile(group_sizes_real, ep_size)
    return lhs_indices, group_sizes, lhs_indices, topk_slot_indices


def _pair_post_combine(
    recv: jax.Array,  # [blocks * chunk * topk, ...] receive blocks, rows in owned-row rank order
    topk_indices: jax.Array,  # [num_tokens, topk], replicated
    topk_weights: jax.Array,  # [chunk, topk], this rank's tokens
    post_expert_norm_weight: jax.Array | None,
    *,
    my_id: jax.Array,
    chunk_size: int,
    topk: int,
    hidden_size: int,
) -> jax.Array:
    """[chunk, hidden]: sum the source blocks, post-expert norm, top-k weighted sum.

    Rank r is the r-th row owned by this rank in the kernel's row order (expert-sorted,
    then token-sorted); the weighted sum maps ranks back to tokens with a 0/1 matmul.
    """
    rows = chunk_size * topk
    flat_sorted = jnp.argsort(topk_indices.reshape(-1))
    owned = (flat_sorted // topk) // chunk_size == my_id
    flat = flat_sorted[jnp.nonzero(owned, size=rows)[0]]
    token = flat // topk - my_id * chunk_size
    weight = topk_weights.astype(jnp.float32)[token, flat % topk]
    out_dtype = jnp.result_type(recv.dtype, topk_weights.dtype)
    if post_expert_norm_weight is not None and recv.ndim == 3:
        weighted = pair_post_combine(
            recv, post_expert_norm_weight, weight, rows=rows, eps=1e-8
        ).reshape(rows, hidden_size)
    else:
        y = jnp.sum(recv.reshape(-1, rows, hidden_size).astype(jnp.float32), axis=0)
        y = y.astype(recv.dtype).astype(jnp.float32)
        if post_expert_norm_weight is not None:
            norm_size = post_expert_norm_weight.shape[0]
            y = y * (post_expert_norm_weight.astype(jnp.float32) + 1.0)[None, :]
            y = y * jax.lax.rsqrt(jnp.sum(y * y, axis=-1, keepdims=True) / norm_size + 1e-8)
        weighted = (y.astype(recv.dtype).astype(jnp.float32) * weight[:, None]).astype(recv.dtype)
    scatter = (token[None, :] == jnp.arange(chunk_size, dtype=jnp.int32)[:, None]).astype(weighted.dtype)
    return jnp.dot(scatter, weighted, preferred_element_type=jnp.float32).astype(out_dtype)


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
    w1_global_scale: jax.Array | None,
    w2_global_scale: jax.Array | None,
    group_offset: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    post_expert_norm_weight_input: jax.Array | None,
    *,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name=EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = True,
    fp8_direct_write: bool = False,
    expert_tp: bool = False,
) -> jax.Array:
    """Per-chip MoE using the nodedup gmm_fused_rs kernel.

    Sends every row to its (local_row, topk_slot) slot via ICI direct-write,
    then applies topk weights and reduces over the top_k dim post-kernel.
    """
    num_tokens = hidden_states_local.shape[0]
    hidden_size = w2.shape[-1]
    chunk_size = num_tokens // ep_size
    num_experts = w1.shape[0] * ep_size  # global num_experts
    num_local_experts = w1.shape[0]
    # Experts are assigned in contiguous, equal-sized blocks by logical rank.
    # Derive the rank from the offset already needed for routing instead of
    # carrying a second scalar through every compiled MoE call.
    my_id = group_offset[0] // num_local_experts

    # Compiler workaround for the non-SP fused remote-DMA graph: materialize
    # IDs and weights from logical rank 0 in one BF16 psum and feed both
    # slices forward. Altering this dataflow can halt the compiled graph. This
    # should not be interpreted as evidence that routing IDs differ across
    # ranks; the ablations establish a lowering dependency, not its root cause.
    # Backend selection limits this path to 256 experts, so IDs 0..255 are
    # represented exactly in BF16.
    # TODO: Remove this block once the mixed communicating-Pallas core halt is
    # fixed at its root cause:
    # https://github.com/sgl-project/sglang-torchtpu/issues/308
    # https://github.com/google-pytorch/torch_tpu/issues/2794
    if not sp_enabled:
        route_source = my_id == 0
        routing = jnp.stack(
            (topk_indices.astype(jnp.bfloat16), topk_weights.astype(jnp.bfloat16)),
            axis=-1,
        )
        routing = jax.lax.psum(
            jnp.where(route_source, routing, jnp.zeros_like(routing)),
            axis_name=ep_axis_name,
        )
        topk_indices = routing[..., 0].astype(jnp.int32)
        topk_weights = routing[..., 1].astype(topk_weights.dtype)

    # Expert TP: every rank holds a 1/ep_size slice of every expert's
    # intermediate dim, addressed as virtual experts
    # [rank * E, (rank + 1) * E). Route each (token, slot) to all ep_size
    # slices; each partial output lands in its own slot and the partials are
    # summed after the kernel, so the kernel's direct writes never collide and
    # every rank does 1/ep_size of the work of each routed expert.
    kernel_topk = topk * ep_size if expert_tp else topk

    # Routing — inlined here so bincount lives in the SAME shard_map as the
    # kernel call. Mirrors v1 where bincount is co-located with the heavy
    # ``hidden_states[token_indices_sorted]`` gather, letting XLA fuse them.
    if expert_tp:
        lhs_indices, group_sizes, output_indices, topk_slot_indices = (
            _compute_expert_tp_routing(
                topk_indices, num_experts=num_local_experts, topk=topk, ep_size=ep_size
            )
        )
    else:
        lhs_indices, group_sizes, output_indices, topk_slot_indices = (
            _compute_rs_routing(topk_indices, num_experts=num_experts, topk=topk)
        )
    size_m = lhs_indices.shape[0]
    local_rows = expert_tp and tpu_envs.SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS.get()
    tiled_output = (
        expert_tp
        and not fp8_direct_write
        and tpu_envs.SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE.get()
    )
    _assert_fused_rs_smem_safe(size_m // ep_size if local_rows else size_m)

    # Compute local row range [local_start, local_end) without per-row source
    # materialization. Avoids the heavy ``jnp.repeat -> scatter_add``.
    go_val = group_offset[0]
    expert_idx = jnp.arange(num_experts, dtype=jnp.int32)
    local_start = jnp.sum(jnp.where(expert_idx < go_val, group_sizes, 0))
    # Static loop bound — kernel pads with ``is_active`` gating.
    #
    # ``tile_m`` MUST match the value the kernel picks internally (via
    # ``_select_fused_rs_block_sizes``). Hardcoding e.g. 128 here while
    # the kernel autotunes to 64 causes ``max_num_gm`` to under-count
    # relative to ``local_num_gm`` computed inside ``fill_metadata``,
    # leaving the kernel's final prefetch gather DMA unawaited and
    # leaking ``gather_sem`` on kernel exit.
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
        # Use the same power-of-two recovery the in-kernel path uses
        # (gmm_v2_fused_rs via recover_quant_block_size). A raw floor
        # ``w1.shape[1] // w1_scale.shape[1]`` underestimates the block size
        # when K is not a multiple of it (e.g. K=2304, 5 scale blocks -> 460
        # instead of 512), which makes this host-side tile lookup key differ
        # from the kernel's (rhs_bs-460 vs rhs_bs-512). That misses the tuned
        # entry here and can pick a different tile_m than the kernel -> the
        # max_num_gm undercount / gather_sem leak described above.
        quant_block_size=(
            recover_quant_block_size(w1.shape[1], w1_scale.shape[1])
            if w1_scale is not None
            else None
        ),
        act_fn=activation,
        fp8_direct_write=fp8_direct_write,
    )
    tile_m = block_sizes.tile_m
    # Every device must execute the same padded GM loop. Use a static upper
    # bound based on the number of experts owned by one device: in the worst
    # case all routed rows land on that device. This is much tighter than the
    # old global-expert bound while keeping the communication schedule wholly
    # independent of runtime routing and physical rank order.
    #
    # Expert TP routes each (token, slot) to exactly one slice per device, so a
    # device holds exactly size_m / ep_size rows, over at most that many of its
    # (many more) virtual experts.
    if expert_tp:
        max_rows = size_m // ep_size
        max_groups = min(num_local_experts, max_rows)
    else:
        max_rows, max_groups = size_m, num_local_experts
    max_num_gm = jnp.array(
        max_groups + (max_rows + tile_m - 1) // tile_m - 1,
        dtype=jnp.int32,
    )

    # Below 4096 tokens per call the extra sibling stage costs more than it saves (bench).
    pair_exchange = (
        expert_tp
        and num_tokens >= 4096
        and local_rows
        and ep_size % 2 == 0
        and ep_size >= 4
        and not fp8_direct_write
        and tpu_envs.SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE.get()
    )
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
        group_start=jnp.array([local_start], dtype=jnp.int32),
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        caller_tile_m=tile_m,
        top_k=kernel_topk,
        fp8_direct_write=fp8_direct_write,
        expert_tp=expert_tp,
        local_rows=local_rows,
        tiled_output=tiled_output,
        row_loop_unroll=tpu_envs.SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL.get(),
        skip_out_zero=expert_tp
        and tpu_envs.SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO.get(),
        staging_slots=tpu_envs.SGLANG_TPU_FUSED_RS_STAGING_SLOTS.get(),
        pair_exchange=pair_exchange,
    )

    # Post-kernel: apply topk weights and reduce over top_k. Stay in bf16
    # (matches v1 — no fp32 cast needed for topk=8).
    # SP off: topk_weights is replicated; slice this chip's [chunk_size] shard
    # to match out_3d (SP on already passes the local shard).
    local_topk_weights = (
        topk_weights
        if sp_enabled
        else jax.lax.dynamic_slice_in_dim(
            topk_weights, my_id * chunk_size, chunk_size, axis=0
        )
    )
    post_expert_norm_weight = post_expert_norm_weight_input if has_post_norm else None
    if pair_exchange:
        return _pair_post_combine(
            out_buf,
            topk_indices,
            local_topk_weights,
            post_expert_norm_weight,
            my_id=my_id,
            chunk_size=chunk_size,
            topk=topk,
            hidden_size=hidden_size,
        )
    if (
        expert_tp
        and post_expert_norm_weight is not None
        and tpu_envs.SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE.get()
        and supports_fused_post_combine(
            out_buf, chunk_size, post_expert_norm_weight.shape[0]
        )
    ):
        return fused_post_combine(
            out_buf,
            post_expert_norm_weight,
            local_topk_weights,
            chunk_size=chunk_size,
            topk=topk,
            ep_size=ep_size,
            eps=1e-8,
        )
    if expert_tp and out_buf.ndim == 3:
        # Split rows only: the ep sum reads the kernel's tiled layout in place.
        out_3d = (
            out_buf.reshape(chunk_size, topk, ep_size, *out_buf.shape[1:])
            .astype(jnp.float32)
            .sum(axis=2)
            .astype(out_buf.dtype)
            .reshape(chunk_size, topk, hidden_size)
        )
    elif expert_tp:
        out_3d = (
            out_buf.reshape(chunk_size, topk, ep_size, hidden_size)
            .astype(jnp.float32)
            .sum(axis=2)
            .astype(out_buf.dtype)
        )
    else:
        out_3d = out_buf.reshape(chunk_size, topk, hidden_size)
    post_expert_norm_weight = post_expert_norm_weight_input if has_post_norm else None
    if post_expert_norm_weight is not None:
        norm_size = post_expert_norm_weight.shape[0]  # actual hidden_size (unpadded)
        pnw_raw = post_expert_norm_weight.astype(jnp.float32) + 1.0
        # Handle padded hidden: out_3d may have padded_hidden_size > norm_size
        if hidden_size > norm_size:
            # Zero out padded columns so they don't affect variance
            col_idx = jnp.arange(hidden_size, dtype=jnp.int32)
            valid_mask = (col_idx < norm_size)[None, None, :]  # (1, 1, hidden_size)
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

    token_hidden = jnp.sum(out_3d * local_topk_weights[:, :, None], axis=1)

    # Return the per-chip token shard (reduce-scattered output). The SP exit is
    # owned by the model: SP on keeps it token-sharded; SP off gathers it back to
    # replicated in the model-side TPU MoE path, optionally as an FP8
    # payload. See expert_parallel_gmm_rs's out_spec.
    return token_hidden


_MOE_GMM_LOCAL_RS = moe_gmm_local_rs_nodedup


# ============================================================================
# shard_map-wrapped EP driver
# ============================================================================


def expert_parallel_gmm_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    group_offset_input: jax.Array | None = None,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    post_expert_norm_weight: jax.Array | None = None,
    w1_global_scale: jax.Array | None = None,
    w2_global_scale: jax.Array | None = None,
    fp8_direct_write: bool = False,
    expert_tp: bool = False,
) -> jax.Array:
    """EP MoE with ICI reduce-scatter fused in kernel.

    Routing (argsort + bincount) is computed INSIDE the per-chip shard_map
    so XLA can fuse it with the kernel pipeline (matches v1's pattern).
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    ep_p_spec = P(expert_axis)
    data_p_spec = P(MLP_DATA)
    # SGLang enters this kernel with replicated tokens and gathers the returned
    # expert-axis token shard in the Torch wrapper.
    sp_enabled = False
    hidden_in_spec = data_p_spec if sp_enabled else P()
    moe_out_spec = (
        P(combine_partition_axes(MLP_DATA, expert_axis)) if sp_enabled else P()
    )
    fp8_direct_write = (
        (not sp_enabled)
        and w1_scale is not None
        and w2_scale is not None
        and fp8_direct_write
    )
    # When SP off, replicate the routing tensors too (consistent with the
    # replicated hidden in/out) so XLA needs no per-tensor reshard; the body
    # slices its local output chunk of topk_weights.
    topk_w_spec = (
        P(combine_partition_axes(MLP_DATA, expert_axis), None) if sp_enabled else P()
    )
    topk_i_spec = P(MLP_DATA, None) if sp_enabled else P()
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = (
        group_offset_input
        if group_offset_input is not None
        else jnp.arange(0, num_experts, num_experts_per_shard)
    )

    # Weight tensors and group_offset are already partitioned per-rank on the PyTorch side.
    # Specify P() so shard_map does not attempt to partition local expert tensors across the mesh.
    weight_p_spec = P()

    w1_scale_spec = None if w1_scale is None else weight_p_spec
    w1_bias_spec = None if w1_bias is None else weight_p_spec
    w2_scale_spec = None if w2_scale is None else weight_p_spec
    w2_bias_spec = None if w2_bias is None else weight_p_spec
    w1_gs_spec = None if w1_global_scale is None else weight_p_spec
    w2_gs_spec = None if w2_global_scale is None else weight_p_spec

    _has_pn_rs = post_expert_norm_weight is not None
    _pnw_rs = (
        post_expert_norm_weight
        if post_expert_norm_weight is not None
        else jnp.zeros((1,), jnp.bfloat16)
    )
    result = jax.shard_map(
        functools.partial(
            _MOE_GMM_LOCAL_RS,
            activation=activation,
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=expert_axis,
            has_post_norm=_has_pn_rs,
            sp_enabled=sp_enabled,
            fp8_direct_write=fp8_direct_write,
            expert_tp=expert_tp,
        ),
        mesh=mesh,
        in_specs=(
            hidden_in_spec,  # hidden_states (replicated when SP off)
            weight_p_spec,  # w1 (local per-rank experts)
            w1_scale_spec,
            w1_bias_spec,
            weight_p_spec,  # w2 (local per-rank experts)
            w2_scale_spec,
            w2_bias_spec,
            w1_gs_spec,  # w1_global_scale
            w2_gs_spec,  # w2_global_scale
            weight_p_spec,  # group_offset (local per-rank tensor)
            topk_w_spec,  # topk_weights (SP-sharded local chunk; replicated when SP off)
            topk_i_spec,  # topk_indices (replicated when SP off)
            P(),  # post_expert_norm_weight (replicated)
        ),
        out_specs=moe_out_spec,
        # SP on: token sharded over data plus the runtime expert axes;
        # SP off: replicated.
        check_vma=False,
    )(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        w1_global_scale,
        w2_global_scale,
        group_offset,
        topk_weights,
        topk_indices,
        _pnw_rs,
    )

    return result


# ============================================================================
# Public jitted entry point
# ============================================================================


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "activation",
        "scoring_fn",
        "fp8_direct_write",
        "expert_tp",
    ),
    compiler_options={
        # "xla_enable_transpose_trace": True,
    },
)
def fused_moe_func_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array | None,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    post_expert_norm_weight: jax.Array | None = None,
    topk_weights: jax.Array | None = None,
    topk_indices: jax.Array | None = None,
    group_offset_input: jax.Array | None = None,
    fp8_direct_write: bool = False,
    expert_tp: bool = False,
    w1_global_scale: jax.Array | None = None,
    w2_global_scale: jax.Array | None = None,
) -> jax.Array:
    """EP MoE with ICI reduce-scatter fused in kernel (gmm_fused_rs).

    Flow:
      1. Routing: use pre-computed ``topk_weights``/``topk_indices`` if the
         caller supplied them (framework-side ``custom_routing_function``
         that already applied any model-specific routing adjustments).
         Otherwise fall back to
         kernel-internal top-k from ``gating_output``.
      2. Token processing: derive lhs_indices, group_sizes, output_indices,
         topk_slot_indices in expert-sorted order.
      3. gmm_fused_rs kernel: gather → GMM1 → act → GMM2 → ICI reduce-scatter.
         hidden_states is replicated on all chips (P("data") with data=1).
      4. Post-kernel weighted reduction over top_k dimension.

    Communication pattern: ICI reduce-scatter fused in kernel.
    This replaces the standard EP pattern of psum (after kernel).
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0
    del gating_output, global_num_experts, renormalize, scoring_fn, dtype
    assert topk_weights is not None and topk_indices is not None, (
        "fused_moe_func_rs requires pre-computed topk weights and indices"
    )
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(MLP_DATA, None))
    )
    topk_indices = jax.lax.with_sharding_constraint(
        topk_indices, NamedSharding(mesh, P(MLP_DATA, None))
    )

    # Pad hidden_states to match w1's K dimension if needed.
    if padded_hidden_size != hidden_size:
        hidden_states = jnp.pad(
            hidden_states, ((0, 0), (0, padded_hidden_size - hidden_size))
        )

    # Routing (argsort + bincount) is now done INSIDE the per-chip shard_map
    # below — no separate ``_shard_map_process_tokens_for_rs`` call here.
    result = expert_parallel_gmm_rs(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        topk_weights,
        topk_indices,
        group_offset_input,
        activation=activation,
        topk=topk,
        mesh=mesh,
        post_expert_norm_weight=post_expert_norm_weight,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        fp8_direct_write=fp8_direct_write,
        expert_tp=expert_tp,
    )

    return result[:num_tokens, :hidden_size]


__all__ = [
    "fused_moe_func_rs",
    "expert_parallel_gmm_rs",
    "moe_gmm_local_rs_nodedup",
    "_compute_rs_routing",
    "_FUSED_RS_MAX_SAFE_SIZE_M",
    "_assert_fused_rs_smem_safe",
]
