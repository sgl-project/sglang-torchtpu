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
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    compute_num_gm,
    _select_fused_rs_block_sizes,
)
from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
    gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup,
)
from sglang.srt.layers.moe.pallas.fused_rs.utils import (
    combine_partition_axes,
    get_mesh_shape_product,
    get_moe_expert_axis,
)
from sglang.srt.layers.moe.pallas.gmm_v2 import recover_quant_block_size

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


def _compute_rs_routing(topk_indices, *, num_experts, topk, ep_size, my_id):
    """Inline routing for the RS path with full VPU vectorized precomputations.
    Computes:
      - lhs_indices: int32[size_m] — gather indices (token ID per row)
      - group_sizes: int32[num_experts] — rows per expert
      - topk_slot_indices: int32[size_m] — topk slot (0..topk-1) per row
      - peer_recv_counts: int32[ep_size] — incoming row counts per peer
    """
    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    sorted_experts = topk_indices_flat[topk_argsort_indices]
    num_experts_per_shard = num_experts // ep_size
    source_peers = (sorted_experts // num_experts_per_shard).astype(jnp.int32)

    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    group_sizes = jnp.sum(
        (topk_indices_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
        axis=0,
    )
    lhs_indices = (topk_argsort_indices // topk).astype(jnp.int32)
    topk_slot_indices = (topk_argsort_indices % topk).astype(jnp.int32)

    num_tokens = topk_indices.shape[0]
    chunk_size = (num_tokens + ep_size - 1) // ep_size
    output_tokens_per_chip = max(chunk_size, 1)
    dest_chips = (lhs_indices // output_tokens_per_chip).astype(jnp.int32)

    is_to_me = (dest_chips == my_id) & (source_peers != my_id)
    filtered_peers = jnp.where(is_to_me, source_peers, ep_size)
    peer_recv_counts = jnp.bincount(filtered_peers, length=ep_size + 1)[:ep_size].astype(jnp.int32)

    return lhs_indices, group_sizes, topk_slot_indices, peer_recv_counts


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
    topk_indices: jax.Array | None = None,
    post_expert_norm_weight_input: jax.Array | None = None,
    *,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name=EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = True,
    fp8_direct_write: bool = False,
    w1_gs_gate: jax.Array | None = None,
    w1_gs_up: jax.Array | None = None,
    w2_gs: jax.Array | None = None,
) -> jax.Array:
    """Per-chip MoE using the nodedup gmm_fused_rs kernel."""
    if w1_global_scale is None and w1_gs_gate is not None:
        w1_global_scale = w1_gs_gate
    if w2_global_scale is None and w2_gs is not None:
        w2_global_scale = w2_gs
    num_tokens = hidden_states_local.shape[0]
    hidden_size = w2.shape[-1]
    chunk_size = (num_tokens + ep_size - 1) // ep_size
    num_experts = w1.shape[0] * ep_size  # global num_experts
    num_local_experts = w1.shape[0]
    go_flat = group_offset.reshape(-1)
    go_val = go_flat[0]
    my_id = go_flat[1] if go_flat.shape[0] > 1 else (go_val // num_local_experts)

    # Routing with vectorized VPU precomputations:
    lhs_indices, group_sizes, topk_slot_indices, peer_recv_counts = _compute_rs_routing(
        topk_indices, num_experts=num_experts, topk=topk, ep_size=ep_size, my_id=my_id
    )
    output_indices = (lhs_indices * topk + topk_slot_indices).astype(jnp.int32)
    size_m = lhs_indices.shape[0]
    _assert_fused_rs_smem_safe(size_m)

    # Compute local row range [local_start, local_end) without per-row source
    # materialization. Avoids the heavy ``jnp.repeat -> scatter_add``.
    expert_idx = jnp.arange(num_experts, dtype=jnp.int32)
    local_start = jnp.sum(jnp.where(expert_idx < go_val, group_sizes, 0))
    local_end = local_start + jnp.sum(
        jnp.where(
            (expert_idx >= go_val) & (expert_idx < go_val + num_local_experts),
            group_sizes,
            0,
        )
    )
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
    size_lhs_sublane = min(
        pltpu.get_tpu_info().get_sublane_tiling(hidden_states_local.dtype),
        tile_m,
        size_m,
    )
    local_group_sizes = jax.lax.dynamic_slice(group_sizes, (go_val,), (num_local_experts,))
    local_num_gm = compute_num_gm(
        local_group_sizes, tile_m, size_lhs_sublane, start_offset=local_start
    )
    max_num_gm = jax.lax.pmax(local_num_gm, axis_name=ep_axis_name)

    out_buf = gmm_v2_fused_rs_nodedup(
        hidden_states_local,
        w1,
        w2,
        group_sizes,
        lhs_indices,
        output_indices,  # <--- MUST BE output_indices!
        peer_recv_counts=peer_recv_counts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        activation=activation,
        output_size=num_tokens,
        group_offset=group_offset,
        group_start=jnp.array([local_start], dtype=jnp.int32),
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        top_k=topk,
        fp8_direct_write=fp8_direct_write,
    )

    # out_buf has arrived directly on this chip via in-kernel ICI direct-write!
    # Shape: (output_size, top_k, hidden_size) where output_size = chunk_size.
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

    # Post-kernel: apply topk weights and reduce over top_k. Stay in bf16
    # (matches v1 — no fp32 cast needed for topk=8).
    # SP off: topk_weights is replicated; slice this chip's [chunk_size] shard
    # to match out_3d (SP on already passes the local shard).
    output_size = max(chunk_size, 1)
    my_token_start = my_id * output_size
    my_weights = (
        topk_weights
        if sp_enabled
        else jax.lax.dynamic_slice_in_dim(
            topk_weights, my_token_start, output_size, axis=0
        )
    )

    # Return the per-chip token shard (reduce-scattered output). The SP exit is
    # owned by the model: SP on keeps it token-sharded; SP off gathers it back to
    # replicated in the model-side TPU MoE path, optionally as an FP8
    # payload. See expert_parallel_gmm_rs's out_spec.
    token_hidden = jnp.sum(out_3d * my_weights[:, :, None].astype(out_3d.dtype), axis=1)
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
) -> jax.Array:
    """EP MoE with ICI reduce-scatter fused in kernel.

    Routing (argsort + bincount) is computed INSIDE the per-chip shard_map
    so XLA can fuse it with the kernel pipeline (matches v1's pattern).
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    ep_p_spec = P(expert_axis)
    ep_2d_spec = P(expert_axis, None)
    data_p_spec = P(MLP_DATA)
    # SGLang enters this kernel with replicated tokens and gathers the returned
    # expert-axis token shard in the Torch wrapper.
    sp_enabled = False
    hidden_in_spec = data_p_spec if sp_enabled else P()
    moe_out_spec = (
        P(combine_partition_axes(MLP_DATA, expert_axis), None)
        if sp_enabled
        else P(expert_axis, None)
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
    if not sp_enabled:
        topk_weights = jax.lax.with_sharding_constraint(
            topk_weights, NamedSharding(mesh, P())
        )
        topk_indices = jax.lax.with_sharding_constraint(
            topk_indices, NamedSharding(mesh, P())
        )
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    if group_offset_input is None:
        group_offset = jnp.stack([
            jnp.arange(0, num_experts, num_experts_per_shard, dtype=jnp.int32),
            jnp.arange(ep_size, dtype=jnp.int32),
        ], axis=-1)  # shape (ep_size, 2)
    else:
        # Ensure 2D (ep_size, 2)
        if group_offset_input.ndim == 1:
            group_offset = jnp.stack([
                group_offset_input,
                jnp.arange(ep_size, dtype=jnp.int32),
            ], axis=-1)
        else:
            group_offset = group_offset_input

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec
    w1_gs_spec = None if w1_global_scale is None else ep_p_spec
    w2_gs_spec = None if w2_global_scale is None else ep_p_spec

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
        ),
        mesh=mesh,
        in_specs=(
            hidden_in_spec,  # hidden_states (replicated when SP off)
            ep_p_spec,  # w1
            w1_scale_spec,
            w1_bias_spec,
            ep_p_spec,  # w2
            w2_scale_spec,
            w2_bias_spec,
            w1_gs_spec,  # w1_global_scale
            w2_gs_spec,  # w2_global_scale
            ep_2d_spec,  # group_offset
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
    assert (
        topk_weights is not None and topk_indices is not None
    ), "fused_moe_func_rs requires pre-computed topk weights and indices"
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P())
    )
    topk_indices = jax.lax.with_sharding_constraint(
        topk_indices, NamedSharding(mesh, P())
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
        fp8_direct_write=fp8_direct_write,
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
