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

from sglang_tpu.envs import envs
from sglang_tpu.moe.pallas.fused_rs.utils import (
    combine_partition_axes,
    get_mesh_shape_product,
    get_moe_expert_axis,
)
from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    _quant_lhs_fp8_block,
    _select_fused_rs_block_sizes,
    fused_rs_smem_rows,
)
from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
    gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup,
)
from sglang_tpu.moe.pallas.gmm_v2 import recover_quant_block_size

EXPERT = "model"
MLP_DATA = "data"

# ============================================================================
# SMEM safety guard
# ============================================================================

# ``gmm_v2_fused_rs`` SMEM budget (v7x): ~1.0 MB. With Wave-8 local-window
# rebasing + ``u16x2`` packed ``int32`` LHS indices, SMEM stays < 580 KB up to
# ``size_m = 524288`` (``T = 65536, top_k = 8``).
_FUSED_RS_MAX_SAFE_SIZE_M = 524289


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


def _fast_stable_argsort_u16(keys_i32: jax.Array) -> jax.Array:
    """Bit-exact stable argsort for 0 <= keys < 65536 and len(keys) <= 65536 using a 1-array uint32 sort."""
    n = keys_i32.shape[0]
    if n <= 65536:
        packed = (keys_i32.astype(jnp.uint32) << jnp.uint32(16)) | jnp.arange(
            n, dtype=jnp.uint32
        )
        return (jnp.sort(packed) & jnp.uint32(0xFFFF)).astype(jnp.int32)
    return jnp.argsort(keys_i32, stable=True).astype(jnp.int32)


def _fast_inv_perm_u16(local_ids_2d: jax.Array) -> jax.Array:
    """Bit-exact inverse permutation for 0 <= ids < 65536 and numel <= 65536 using 1-array uint32 sorts."""
    flat = local_ids_2d.reshape(-1)
    n = flat.shape[0]
    if n <= 65536:
        p1 = jnp.sort(
            (flat.astype(jnp.uint32) << jnp.uint32(16))
            | jnp.arange(n, dtype=jnp.uint32)
        ) & jnp.uint32(0xFFFF)
        p2 = jnp.sort(
            (p1 << jnp.uint32(16)) | jnp.arange(n, dtype=jnp.uint32)
        ) & jnp.uint32(0xFFFF)
        return p2.reshape(local_ids_2d.shape).astype(jnp.int32)
    perm = jnp.argsort(flat, stable=True)
    return jnp.argsort(perm).reshape(local_ids_2d.shape).astype(jnp.int32)


# Largest rows x experts one-hot the sort-free order builds (4 MiB of int32).
_SORTFREE_MAX_ONEHOT = 1 << 20


def _sortfree_stable_argsort(keys_i32: jax.Array, num_keys: int) -> jax.Array:
    """Stable argsort of 0 <= keys < num_keys without a sort.

    A row's sorted position is its key's exclusive prefix count plus the number
    of earlier rows with the same key; scattering row ids to those positions
    gives the stable argsort.
    """
    n = keys_i32.shape[0]
    onehot = (
        keys_i32[:, None] == jnp.arange(num_keys, dtype=jnp.int32)[None, :]
    ).astype(jnp.int32)
    counts = jnp.sum(onehot, axis=0)
    starts = counts.cumsum() - counts
    # Running count is 1 at a row's first occurrence of its key, hence the -1.
    pos = jnp.sum(onehot * (jnp.cumsum(onehot, axis=0) - 1 + starts[None, :]), axis=1)
    return (
        jnp.zeros((n,), jnp.int32)
        .at[pos]
        .set(jnp.arange(n, dtype=jnp.int32), unique_indices=True)
    )


def _expert_order(keys_i32: jax.Array, num_experts: int) -> jax.Array:
    """Row order sorted by expert, stable in row id."""
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_SORTFREE_METADATA.get()
        and keys_i32.shape[0] * num_experts <= _SORTFREE_MAX_ONEHOT
    ):
        return _sortfree_stable_argsort(keys_i32, num_experts)
    if num_experts <= 65536:
        return _fast_stable_argsort_u16(keys_i32)
    return jnp.argsort(keys_i32)


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
    topk_argsort_indices = _fast_stable_argsort_u16(topk_indices_flat)
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
    replica_meta: jax.Array | None = None,
    *,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name=EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = True,
    fp8_direct_write: bool = False,
    expert_tp: bool = False,
    fp4_w4a16: bool = False,
    local_t_recv: jax.Array | None = None,
    return_tiled: bool = False,
    block_scatter: bool = False,
    return_raw_out_buf: bool = False,
    precomputed_routing: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    precomputed_chipmate_plan: jax.Array | None = None,
    num_base_local: int | None = None,
    prefill_replica_mode: str | None = None,
    prefill_chipmate_rebalance: bool | None = None,
    prefill_chipmate_mode: str | None = None,
    prefill_hot_interleave: bool | None = None,
    chipmate_xor: int = 1,
    hidden_states_scale: jax.Array | None = None,
) -> jax.Array:
    """Per-chip MoE using the nodedup gmm_fused_rs kernel.

    Sends every row to its (local_row, topk_slot) slot via ICI direct-write,
    then applies topk weights and reduces over the top_k dim post-kernel.
    """
    num_tokens = hidden_states_local.shape[0]
    # Rank-4 weights stack layers (Muse megakernel expert groups): group_offset[1] picks one.
    if w1.ndim == 4 and group_offset.shape[0] < 2:
        raise ValueError(
            f"stacked (rank-4) expert weights need group_offset = [expert_offset, layer], "
            f"got shape {group_offset.shape}"
        )
    w1_shape = w1.shape[1:] if w1.ndim == 4 else w1.shape
    w2_shape = w2.shape[1:] if w2.ndim == 4 else w2.shape
    hidden_size = w2_shape[-1]
    chunk_size = num_tokens // ep_size
    num_experts = w1_shape[0] * ep_size  # global num_experts
    num_local_experts = w1_shape[0]
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
    use_block_scatter = block_scatter or (
        has_post_norm
        and not expert_tp
        and not fp8_direct_write
        and chunk_size >= 64
        and chunk_size % 64 == 0
        and hidden_size % 128 == 0
        and (num_tokens * topk) <= 120_000
    )
    inv_perm_c_pre = None
    if not sp_enabled:
        route_source = my_id == 0
        if num_experts > 256:
            idx_hi = jax.lax.shift_right_logical(topk_indices, 7).astype(jnp.bfloat16)
            idx_lo = (topk_indices & jnp.int32(127)).astype(jnp.bfloat16)
            routing_in = jax.lax.optimization_barrier(
                jnp.where(
                    route_source,
                    jnp.stack(
                        (idx_hi, idx_lo, topk_weights.astype(jnp.bfloat16)),
                        axis=-1,
                    ),
                    jnp.zeros(
                        (topk_indices.shape[0], topk_indices.shape[1], 3),
                        dtype=jnp.bfloat16,
                    ),
                )
            )
            routing = jax.lax.psum(routing_in, axis_name=ep_axis_name)
            if (
                use_block_scatter
                and has_post_norm
                and not expert_tp
                and not return_raw_out_buf
            ):
                raw_local_ids = jax.lax.dynamic_slice_in_dim(
                    topk_indices, my_id * chunk_size, chunk_size, axis=0
                )
                inv_perm_c_pre = _fast_inv_perm_u16(raw_local_ids)
                routing, inv_perm_c_pre = jax.lax.optimization_barrier(
                    (routing, inv_perm_c_pre)
                )
            topk_indices = jax.lax.shift_left(
                routing[..., 0].astype(jnp.int32), 7
            ) + routing[..., 1].astype(jnp.int32)
            topk_weights = routing[..., 2].astype(topk_weights.dtype)
        else:
            routing_in = jax.lax.optimization_barrier(
                jnp.where(
                    route_source,
                    jnp.stack(
                        (
                            topk_indices.astype(jnp.bfloat16),
                            topk_weights.astype(jnp.bfloat16),
                        ),
                        axis=-1,
                    ),
                    jnp.zeros(
                        (topk_indices.shape[0], topk_indices.shape[1], 2),
                        dtype=jnp.bfloat16,
                    ),
                )
            )
            routing = jax.lax.psum(routing_in, axis_name=ep_axis_name)
            if (
                use_block_scatter
                and has_post_norm
                and not expert_tp
                and not return_raw_out_buf
            ):
                raw_local_ids = jax.lax.dynamic_slice_in_dim(
                    topk_indices, my_id * chunk_size, chunk_size, axis=0
                )
                inv_perm_c_pre = _fast_inv_perm_u16(raw_local_ids)
                routing, inv_perm_c_pre = jax.lax.optimization_barrier(
                    (routing, inv_perm_c_pre)
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
    eff_cm_rebalance = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE.get()
        if prefill_chipmate_rebalance is None
        else bool(prefill_chipmate_rebalance)
    )
    eff_hot_interleave = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_HOT_INTERLEAVE.get()
        if prefill_hot_interleave is None
        else bool(prefill_hot_interleave)
    )
    eff_rep_mode = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_REPLICA_MODE.get()
        if prefill_replica_mode is None
        else prefill_replica_mode
    )
    eff_cm_mode = (
        envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE.get()
        if prefill_chipmate_mode is None
        else prefill_chipmate_mode
    )
    can_prefill_rebalance = (
        not expert_tp
        and precomputed_routing is None
        and (num_tokens * topk) >= 4096
        and w1.ndim == 3
    )
    use_prefill_rep = can_prefill_rebalance and (replica_meta is not None)
    use_prefill_cm = can_prefill_rebalance and eff_cm_rebalance and (ep_size % 2 == 0)
    use_prefill_hi = (
        not expert_tp
        and (num_tokens * topk) >= 4096
        and w1.ndim == 3
        and eff_hot_interleave
    )

    my_cm_plan = precomputed_chipmate_plan
    if my_cm_plan is not None:
        num_local_experts = num_local_experts + (int(my_cm_plan.shape[0]) // 2)
        num_experts = num_local_experts * ep_size
        my_id = group_offset[0] // num_local_experts
    if use_prefill_rep or use_prefill_cm:
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            route_prefill_imbalance_jax,
        )

        _req_cm_slots = int(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS.get() or 1
        )
        _n_cm_slots = (
            3
            if (
                use_prefill_cm
                and not use_prefill_rep
                and _req_cm_slots >= 3
                and ep_size >= 8
                and (ep_size & (ep_size - 1)) == 0
            )
            else (
                2
                if (use_prefill_cm and not use_prefill_rep and _req_cm_slots >= 2)
                else (1 if use_prefill_cm else 0)
            )
        )
        # Invalidate pre-remapped inv_perm_c_pre so _tc_combine_bf16 uses virtual IDs.
        inv_perm_c_pre = None
        _pre_bs = _select_fused_rs_block_sizes(
            size_m=num_tokens * topk,
            size_k1=w1_shape[1],
            size_n1=w1_shape[2],
            size_k2=w2_shape[1],
            size_n2=w2_shape[2],
            size_group=num_local_experts + _n_cm_slots,
            size_lhs_group=(num_local_experts + _n_cm_slots) * ep_size,
            ep_size=ep_size,
            out_dtype=(
                jnp.dtype(jnp.bfloat16)
                if jnp.dtype(hidden_states_local.dtype) == jnp.float8_e4m3fn
                else hidden_states_local.dtype
            ),
            w1_dtype=w1.dtype,
            w2_dtype=w2.dtype,
            is_quantized=w1_scale is not None,
            quant_block_size=(
                recover_quant_block_size(w1_shape[1], w1_scale.shape[w1.ndim - 2])
                if w1_scale is not None
                else None
            ),
            act_fn=activation,
            fp8_direct_write=fp8_direct_write,
            hot_interleave=use_prefill_hi,
        )
        _plan_tm = _pre_bs.tile_m
        _base_loc = (
            (num_base_local if num_base_local is not None else 32)
            if use_prefill_rep
            else num_local_experts
        )
        _n_rep = (num_local_experts - _base_loc) if use_prefill_rep else 0
        if _n_cm_slots >= 2:
            from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
                plan_prefill_chipmate_2slot_jax,
            )

            _slot_xors = (1, 2, 4) if _n_cm_slots == 3 else (chipmate_xor, chipmate_xor)
            topk_indices, _cm_all = plan_prefill_chipmate_2slot_jax(
                topk_indices,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                tile_m=_plan_tm,
                chipmate_xor=chipmate_xor,
                mode=eff_cm_mode,
                slot_xors=_slot_xors,
            )
        else:
            topk_indices, _cm_all = route_prefill_imbalance_jax(
                topk_indices,
                ep_size=ep_size,
                num_base_local=_base_loc,
                replica_meta=replica_meta if use_prefill_rep else None,
                num_replicas=_n_rep,
                replica_mode=eff_rep_mode,
                do_chipmate=use_prefill_cm,
                chipmate_xor=chipmate_xor,
                chipmate_mode=eff_cm_mode,
                tile_m=_plan_tm,
            )
        local_t_recv = None
        if use_prefill_cm:
            _perm_pairs = [(r, r ^ chipmate_xor) for r in range(ep_size)]
            if _n_cm_slots >= 2:
                _plan_entries = []
                for _ci in range(_n_cm_slots):
                    _plan_entries.extend(
                        [_cm_all[my_id, 3 * _ci + 0], _cm_all[my_id, 3 * _ci + 1]]
                    )
                my_cm_plan = jnp.stack(_plan_entries)
                _xors_cm = (
                    (1, 2, 4) if _n_cm_slots == 3 else (chipmate_xor, chipmate_xor)
                )
                if w1_global_scale is not None:
                    _cat1 = [w1_global_scale]
                    for _ci, _rx in enumerate(_xors_cm):
                        _pgs1 = jax.lax.ppermute(
                            w1_global_scale,
                            ep_axis_name,
                            [(r, r ^ _rx) for r in range(ep_size)],
                        )
                        _sr = _cm_all[my_id, 3 * _ci + 2]
                        _cat1.append(_pgs1[_sr : _sr + 1])
                    w1_global_scale = jnp.concatenate(_cat1, axis=0)
                if w2_global_scale is not None:
                    _cat2 = [w2_global_scale]
                    for _ci, _rx in enumerate(_xors_cm):
                        _pgs2 = jax.lax.ppermute(
                            w2_global_scale,
                            ep_axis_name,
                            [(r, r ^ _rx) for r in range(ep_size)],
                        )
                        _sr = _cm_all[my_id, 3 * _ci + 2]
                        _cat2.append(_pgs2[_sr : _sr + 1])
                    w2_global_scale = jnp.concatenate(_cat2, axis=0)
            else:
                my_cm_plan = _cm_all[my_id, :2]
                _recv_src_slot = _cm_all[my_id, 2]
                if w1_global_scale is not None:
                    _pgs1 = jax.lax.ppermute(w1_global_scale, ep_axis_name, _perm_pairs)
                    w1_global_scale = jnp.concatenate(
                        [w1_global_scale, _pgs1[_recv_src_slot : _recv_src_slot + 1]],
                        axis=0,
                    )
                if w2_global_scale is not None:
                    _pgs2 = jax.lax.ppermute(w2_global_scale, ep_axis_name, _perm_pairs)
                    w2_global_scale = jnp.concatenate(
                        [w2_global_scale, _pgs2[_recv_src_slot : _recv_src_slot + 1]],
                        axis=0,
                    )
            num_local_experts = num_local_experts + _n_cm_slots
            num_experts = num_local_experts * ep_size
            group_offset = group_offset.at[0].set(my_id * num_local_experts)

    go_val = group_offset[0]
    if precomputed_routing is not None:
        lhs_indices, group_sizes, group_start_arg = precomputed_routing
        topk_slot_indices = jnp.zeros((1,), dtype=jnp.int32)
        output_indices = lhs_indices
    elif expert_tp:
        lhs_indices, group_sizes, output_indices, topk_slot_indices = (
            _compute_expert_tp_routing(
                topk_indices, num_experts=num_local_experts, topk=topk, ep_size=ep_size
            )
        )
        expert_idx = jnp.arange(num_experts, dtype=jnp.int32)
        local_start = jnp.sum(jnp.where(expert_idx < go_val, group_sizes, 0))
        group_start_arg = jnp.array([local_start], dtype=jnp.int32)
    elif use_block_scatter:
        topk_indices_flat = topk_indices.flatten()
        if (
            int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS.get() or 0) >= 2
            and (num_tokens * topk) > 4096
        ):
            _flat_iota = jnp.arange(topk_indices_flat.shape[0], dtype=jnp.int32)
            _tok_f = _flat_iota // topk
            _slot_f = _flat_iota % topk
            _chip_rot = jnp.mod(_tok_f // chunk_size + ep_size - 1 - my_id, ep_size)
            _rot_key = topk_indices_flat.astype(jnp.int32) * ep_size + _chip_rot
            topk_argsort_indices = _expert_order(_rot_key, num_experts * ep_size)
            lhs_indices = topk_argsort_indices // topk
        else:
            topk_argsort_indices = _expert_order(topk_indices_flat, num_experts)
            lhs_indices = topk_argsort_indices // topk
        topk_slot_indices = jnp.zeros((1,), dtype=jnp.int32)
        output_indices = lhs_indices
        topk_by_chip = topk_indices.reshape(ep_size, -1)
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
        t_recv_val = (
            local_t_recv
            if local_t_recv is not None
            else (
                jnp.sum((topk_by_chip[my_id] < num_experts).astype(jnp.int32))
                if (use_prefill_rep or use_prefill_cm)
                else jnp.int32(chunk_size * topk)
            )
            - jnp.sum(local_chip_counts[my_id])
        )
        group_start_arg = jnp.concatenate(
            [
                jnp.stack([local_start, t_recv_val]),
                chip_before_local,
                local_chip_counts.T.reshape(-1),
            ]
        ).astype(jnp.int32)
    else:
        topk_indices_flat = topk_indices.flatten()
        topk_argsort_indices = _expert_order(topk_indices_flat, num_experts)
        lhs_indices = topk_argsort_indices // topk
        topk_slot_indices = topk_argsort_indices % topk
        output_indices = lhs_indices
        local_start = jnp.sum((topk_indices_flat < go_val).astype(jnp.int32))
        local_expert_ids = go_val + jnp.arange(num_local_experts, dtype=jnp.int32)
        local_group_sizes = jnp.sum(
            (topk_indices_flat[:, None] == local_expert_ids[None, :]).astype(jnp.int32),
            axis=0,
        )
        group_sizes = jax.lax.dynamic_update_slice(
            jnp.zeros((num_experts,), dtype=jnp.int32), local_group_sizes, (go_val,)
        )
        if (
            envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_ROUTER.get() >= 2
            and envs.SGLANG_TPU_MUSE_KERNEL_SKIP_PRECOMPUTED_METADATA.get()
            and topk_indices_flat.shape[0] <= 4096
        ):
            local_end = local_start + jnp.sum(local_group_sizes)
            if local_t_recv is not None:
                recv_cnt = local_t_recv
            else:
                my_tok_ids = (
                    topk_indices
                    if sp_enabled and topk_indices.shape[0] == chunk_size
                    else jax.lax.dynamic_slice_in_dim(
                        topk_indices, my_id * chunk_size, chunk_size, axis=0
                    )
                )
                recv_cnt = jnp.sum(
                    jnp.logical_or(
                        jnp.logical_and(my_tok_ids >= 0, my_tok_ids < go_val),
                        jnp.logical_and(
                            my_tok_ids >= go_val + num_local_experts,
                            my_tok_ids < num_experts,
                        ),
                    ).astype(jnp.int32)
                )
            group_start_arg = jnp.stack([local_start, local_end, recv_cnt]).astype(
                jnp.int32
            )
        else:
            group_start_arg = (
                jnp.array([local_start], dtype=jnp.int32)
                if local_t_recv is None
                else jnp.stack([local_start, local_t_recv]).astype(jnp.int32)
            )
    size_m = lhs_indices.shape[0]
    _smem_rows = fused_rs_smem_rows(size_m, ep_size, expert_tp)
    _assert_fused_rs_smem_safe(size_m if not expert_tp else _smem_rows)
    if (
        hidden_states_scale is None
        and int(envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get() or 0) >= 1
        and jnp.dtype(hidden_states_local.dtype) == jnp.bfloat16
        and size_m >= 16384
        and w1_scale is not None
        and not fp8_direct_write
        and not fp4_w4a16
        and (
            (
                jnp.dtype(w1.dtype) == jnp.float4_e2m1fn
                and int(envs.SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS.get() or 0)
                >= (w1_shape[1] * 2 if w1.dtype == jnp.int32 else w1_shape[1])
            )
            or (
                jnp.dtype(w1.dtype) == jnp.float8_e4m3fn
                and int(envs.SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS.get() or 0)
                >= w1_shape[1]
            )
        )
    ):
        _hs_2d = (
            hidden_states_local.reshape(num_tokens, -1)
            if hidden_states_local.ndim == 3
            else hidden_states_local
        )
        _hs_fp8_2d, _hs_sc_2d = _quant_lhs_fp8_block(_hs_2d)
        hidden_states_local = (
            _hs_fp8_2d.reshape(hidden_states_local.shape)
            if hidden_states_local.ndim == 3
            else _hs_fp8_2d
        )
        hidden_states_scale = _hs_sc_2d.reshape(-1)
    # Static loop bound — kernel pads with ``is_active`` gating.
    #
    # ``tile_m`` MUST match the value the kernel picks internally (via
    # ``_select_fused_rs_block_sizes``). Hardcoding e.g. 128 here while
    # the kernel autotunes to 64 causes ``max_num_gm`` to under-count
    # relative to ``local_num_gm`` computed inside ``fill_metadata``,
    # leaving the kernel's final prefetch gather DMA unawaited and
    # leaking ``gather_sem`` on kernel exit.
    _probe_qbs_f = int(envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK.get() or 0)
    _caller_qbs = (
        recover_quant_block_size(w1_shape[1], w1_scale.shape[w1.ndim - 2])
        if w1_scale is not None
        else None
    )
    if (
        _caller_qbs is not None
        and _probe_qbs_f < 0
        and jnp.dtype(w1.dtype) == jnp.float4_e2m1fn
        and (-_probe_qbs_f) in (512, 1024, 2048, 4096)
        and (-_probe_qbs_f) > _caller_qbs
        and (-_probe_qbs_f) % _caller_qbs == 0
    ):
        _caller_qbs = -_probe_qbs_f
    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=w1_shape[1],
        size_n1=w1_shape[2],
        size_k2=w2_shape[1],
        size_n2=w2_shape[2],
        size_group=num_local_experts,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=(
            jnp.dtype(jnp.bfloat16)
            if jnp.dtype(hidden_states_local.dtype) == jnp.float8_e4m3fn
            else hidden_states_local.dtype
        ),
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        is_quantized=w1_scale is not None,
        quant_block_size=_caller_qbs,
        act_fn=activation,
        fp8_direct_write=fp8_direct_write,
        hot_interleave=use_prefill_hi,
    )
    tile_m = block_sizes.tile_m
    max_rows = (
        size_m // ep_size
        if expert_tp
        else (_smem_rows if _smem_rows < size_m else size_m)
    )
    max_groups = min(num_local_experts, max_rows) if expert_tp else num_local_experts
    max_num_gm = jnp.array(
        max_groups + (max_rows + tile_m - 1) // tile_m - 1,
        dtype=jnp.int32,
    )

    inv_perm_c_a = inv_perm_c_pre
    if (
        inv_perm_c_a is None
        and use_block_scatter
        and not return_raw_out_buf
        and has_post_norm
        and not expert_tp
    ):
        local_topk_ids_slice = (
            topk_indices
            if sp_enabled and topk_indices.shape[0] == chunk_size
            else jax.lax.dynamic_slice_in_dim(
                topk_indices, my_id * chunk_size, chunk_size, axis=0
            )
        )
        inv_perm_c_a = _fast_inv_perm_u16(local_topk_ids_slice)

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
        group_start=group_start_arg,
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        caller_tile_m=tile_m,
        top_k=kernel_topk,
        fp8_direct_write=fp8_direct_write,
        expert_tp=expert_tp,
        fp4_w4a16=fp4_w4a16,
        post_expert_norm_weight=post_expert_norm_weight_input
        if (has_post_norm and not expert_tp)
        else None,
        block_scatter=use_block_scatter,
        chipmate_plan=my_cm_plan,
        chipmate_xor=chipmate_xor,
        hot_interleave=use_prefill_hi,
        hidden_states_scale=hidden_states_scale,
    )
    if return_raw_out_buf:
        return out_buf

    if jnp.dtype(out_buf.dtype) == jnp.float8_e4m3fn:
        from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
            resolve_static_fp8_rs_scale,
        )

        _abs_sfp8, _ = resolve_static_fp8_rs_scale(
            envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS.get()
        )
        if _abs_sfp8 > 0.0:
            out_buf = out_buf.astype(hidden_states_local.dtype) * jnp.asarray(
                1.0 / _abs_sfp8, dtype=hidden_states_local.dtype
            )

    local_topk_weights = (
        topk_weights
        if sp_enabled
        else jax.lax.dynamic_slice_in_dim(
            topk_weights, my_id * chunk_size, chunk_size, axis=0
        )
    )
    w_t = jnp.swapaxes(local_topk_weights, 0, 1)[:, :, None]
    post_expert_norm_weight = (
        post_expert_norm_weight_input if (has_post_norm and expert_tp) else None
    )
    if use_block_scatter and has_post_norm and not expert_tp and out_buf.ndim == 3:
        red_out = _tc_combine_bf16(
            out_buf, inv_perm_c_a, local_topk_weights, return_tiled=return_tiled
        )
        if return_tiled:
            return red_out
        token_hidden = red_out[:, :hidden_size]
    else:
        out_3d = out_buf.reshape(topk, chunk_size, -1)[:, :, :hidden_size]
        if has_post_norm and not expert_tp:
            token_hidden = jnp.sum(
                out_3d.astype(jnp.float32) * w_t.astype(jnp.float32), axis=0
            ).astype(out_3d.dtype)
        elif post_expert_norm_weight is not None:
            norm_size = post_expert_norm_weight.shape[
                0
            ]  # actual hidden_size (unpadded)
            pnw_raw = post_expert_norm_weight.astype(jnp.float32) + 1.0
            if hidden_size > norm_size:
                col_idx = jnp.arange(hidden_size, dtype=jnp.int32)
                valid_mask = (col_idx < norm_size)[None, None, :]
                out_f32 = out_3d.astype(jnp.float32) * valid_mask
                pnw = jnp.concatenate(
                    [pnw_raw, jnp.zeros(hidden_size - norm_size, dtype=jnp.float32)]
                )
            else:
                out_f32 = out_3d.astype(jnp.float32)
                pnw = pnw_raw
            pnw_sq_over_n = (pnw * pnw) * (1.0 / norm_size)
            var = jnp.sum(
                (out_f32 * out_f32) * pnw_sq_over_n[None, None, :],
                axis=-1,
                keepdims=True,
            )
            combined_scale = jax.lax.rsqrt(var + 1e-8) * w_t.astype(jnp.float32)
            token_hidden = (
                jnp.sum(out_f32 * combined_scale, axis=0) * pnw[None, :]
            ).astype(out_3d.dtype)
        else:
            token_hidden = jnp.sum(out_3d * w_t, axis=0)

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
    fp4_w4a16: bool = False,
    replica_meta: jax.Array | None = None,
    num_base_local: int | None = None,
    prefill_replica_mode: str | None = None,
    prefill_chipmate_rebalance: bool | None = None,
    prefill_chipmate_mode: str | None = None,
    prefill_hot_interleave: bool | None = None,
    chipmate_xor: int = 1,
    hidden_states_scale: jax.Array | None = None,
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
    if w1.ndim == 4 and group_offset_input is None:
        raise ValueError(
            "stacked (rank-4) expert weights need group_offset_input = [expert_offset, layer]"
        )
    num_experts = w1.shape[1] if w1.ndim == 4 else w1.shape[0]
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

    replica_meta_spec = None if replica_meta is None else P()

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
            fp4_w4a16=fp4_w4a16,
            num_base_local=num_base_local,
            prefill_replica_mode=prefill_replica_mode,
            prefill_chipmate_rebalance=prefill_chipmate_rebalance,
            prefill_chipmate_mode=prefill_chipmate_mode,
            prefill_hot_interleave=prefill_hot_interleave,
            chipmate_xor=chipmate_xor,
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
        )
        + ((replica_meta_spec,) if replica_meta is not None else ()),
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
        *((replica_meta,) if replica_meta is not None else ()),
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
        "fp4_w4a16",
        "num_base_local",
        "prefill_replica_mode",
        "prefill_chipmate_rebalance",
        "prefill_chipmate_mode",
        "prefill_hot_interleave",
        "chipmate_xor",
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
    fp4_w4a16: bool = False,
    replica_meta: jax.Array | None = None,
    num_base_local: int | None = None,
    prefill_replica_mode: str | None = None,
    prefill_chipmate_rebalance: bool | None = None,
    prefill_chipmate_mode: str | None = None,
    prefill_hot_interleave: bool | None = None,
    chipmate_xor: int = 1,
    hidden_states_scale: jax.Array | None = None,
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
    global_num_experts, padded_hidden_size, _ = (
        w1.shape[1:] if w1.ndim == 4 else w1.shape
    )
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
        fp8_direct_write=fp8_direct_write,
        expert_tp=expert_tp,
        fp4_w4a16=fp4_w4a16,
        replica_meta=replica_meta,
        num_base_local=num_base_local,
        prefill_replica_mode=prefill_replica_mode,
        prefill_chipmate_rebalance=prefill_chipmate_rebalance,
        prefill_chipmate_mode=prefill_chipmate_mode,
        prefill_hot_interleave=prefill_hot_interleave,
        chipmate_xor=chipmate_xor,
    )

    return result[:num_tokens, :hidden_size]


__all__ = [
    "fused_moe_func_rs",
    "fully_fused_moe_func_rs",
    "expert_parallel_gmm_rs",
    "moe_gmm_local_rs_nodedup",
    "_compute_rs_routing",
    "_FUSED_RS_MAX_SAFE_SIZE_M",
    "_assert_fused_rs_smem_safe",
]


# ============================================================================
# Fully Fused MoE Entry Point (AG + PreNorm + Expert Selection + Routing + GMM-RS + PostNorm + Combine + Output AG)
# ============================================================================

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _tc_combine_bf16(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    *,
    tok_tile: int = 128,
    return_tiled: bool = False,
) -> jax.Array:
    chunk_size, topk = inv_perm_c.shape
    tok_tile = 64
    n2_cols, lanes = out_buf_3d.shape[1], out_buf_3d.shape[2]
    hidden_size = n2_cols * lanes
    num_steps = chunk_size // tok_tile
    w_pad = jnp.pad(local_topk_weights.astype(jnp.bfloat16), ((0, 0), (0, 128 - topk)))

    def _kernel(inv_perm_ref, src_hbm_ref, w_vmem_ref, out_ref, buf_2x_ref, sem_2x_ref):
        step = pl.program_id(0)

        def _start_step(s, slot):
            base_tok = s * tok_tile
            for t in range(tok_tile):
                tok = base_tok + t
                dst_base = t * topk
                idx8 = [inv_perm_ref[tok, k] for k in range(topk)]
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
        for rb in range(tok_tile // 32):
            t0, t1 = rb * 32, (rb + 1) * 32
            vals_4d = buf_2x_ref[slot, t0 * topk : t1 * topk, :, :].reshape(
                32, topk, n2_cols, lanes
            )
            w_tok = w_vmem_ref[t0:t1, :topk].reshape(32, topk, 1, 1)
            red_3d = jnp.sum(vals_4d * w_tok, axis=1)
            out_ref[t0:t1, ...] = (
                red_3d if return_tiled else red_3d.reshape(32, hidden_size)
            )

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (chunk_size, n2_cols, lanes) if return_tiled else (chunk_size, hidden_size),
            jnp.bfloat16,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_tile, 128), lambda i, *_: (i, 0)),
            ],
            out_specs=(
                pl.BlockSpec((tok_tile, n2_cols, lanes), lambda i, *_: (i, 0, 0))
                if return_tiled
                else pl.BlockSpec((tok_tile, hidden_size), lambda i, *_: (i, 0))
            ),
            scratch_shapes=[
                pltpu.VMEM((2, tok_tile * topk, n2_cols, lanes), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="tc_combine_bf16",
    )(inv_perm_c, out_buf_3d, w_pad)


def _tc_combine_and_int8_quant(
    out_buf_3d: jax.Array,
    inv_perm_c: jax.Array,
    local_topk_weights: jax.Array,
    *,
    tok_tile: int = 128,
) -> tuple[jax.Array, jax.Array]:
    chunk_size, topk = inv_perm_c.shape
    tok_tile = 64 if (chunk_size <= 256 or chunk_size % tok_tile != 0) else tok_tile
    n2_cols, lanes = out_buf_3d.shape[1], out_buf_3d.shape[2]
    hidden_size = n2_cols * lanes
    num_steps = chunk_size // tok_tile
    w_pad = jnp.pad(local_topk_weights.astype(jnp.bfloat16), ((0, 0), (0, 128 - topk)))

    def _kernel(
        inv_perm_ref,
        src_hbm_ref,
        w_vmem_ref,
        out_i8_ref,
        out_sc_ref,
        buf_2x_ref,
        sem_2x_ref,
    ):
        step = pl.program_id(0)

        def _start_step(s, slot):
            base_tok = s * tok_tile
            for t in range(tok_tile):
                tok = base_tok + t
                dst_base = t * topk
                idx8 = [inv_perm_ref[tok, k] for k in range(topk)]
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
        for rb in range(tok_tile // 8):
            t0, t1 = rb * 8, (rb + 1) * 8
            vals_4d = buf_2x_ref[slot, t0 * topk : t1 * topk, :, :].reshape(
                8, topk, n2_cols, lanes
            )
            w_tok = w_vmem_ref[t0:t1, :topk].reshape(8, topk, 1, 1)
            red_3d = jnp.sum(vals_4d * w_tok, axis=1)
            amax_lanes = jnp.max(jnp.abs(red_3d), axis=1)
            row_amax = jnp.max(amax_lanes, axis=-1, keepdims=True).astype(jnp.float32)
            inv_sc = (
                jnp.float32(126.5) / jnp.maximum(row_amax, jnp.float32(1e-12))
            ).astype(jnp.bfloat16)
            row_sc = (row_amax * jnp.float32(1.0 / 126.5)).astype(jnp.bfloat16)
            out_i8_ref[t0:t1, :] = jnp.rint(
                red_3d.reshape(8, hidden_size) * inv_sc
            ).astype(jnp.int8)
            out_sc_ref[t0:t1, :] = jnp.broadcast_to(row_sc, (8, 128))

    i8_out_2d, sc_pad = pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((chunk_size, hidden_size), jnp.int8),
            jax.ShapeDtypeStruct((chunk_size, 128), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_tile, 128), lambda i, *_: (i, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tok_tile, hidden_size), lambda i, *_: (i, 0)),
                pl.BlockSpec((tok_tile, 128), lambda i, *_: (i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, tok_tile * topk, n2_cols, lanes), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="tc_combine_and_int8_quant",
    )(inv_perm_c, out_buf_3d, w_pad)
    return i8_out_2d, sc_pad[:, 0]


def _dequant_into_half(
    buf_4d: jax.Array,
    ag_i8_3d: jax.Array,
    ag_sc_2d: jax.Array,
    *,
    half_idx: int,
) -> jax.Array:
    ep_sz, _, half_tok, hidden_sz = buf_4d.shape

    if half_idx == 0:

        def _k0(dst_in_ref, i8_ref, sc_ref, dst_out_ref):
            del dst_in_ref
            r = pl.program_id(0)
            mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, 128), 0) == r
            for t_blk in range(half_tok // 128):
                r0, r1 = t_blk * 128, (t_blk + 1) * 128
                sc_1x128 = jnp.sum(
                    jnp.where(mask_r, sc_ref[:, r0:r1].astype(jnp.float32), 0.0),
                    axis=0,
                    keepdims=True,
                ).astype(jnp.bfloat16)
                sc_128x1 = jnp.swapaxes(jnp.broadcast_to(sc_1x128, (128, 128)), 0, 1)[
                    :, :1
                ]
                for c_blk in range(hidden_sz // 1024):
                    c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                    dst_out_ref[0, 0, r0:r1, c0:c1] = (
                        i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                    )

        return pl.pallas_call(
            _k0,
            out_shape=jax.ShapeDtypeStruct(buf_4d.shape, jnp.bfloat16),
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec(
                    (1, 1, half_tok, hidden_sz), lambda r: (r, half_idx, 0, 0)
                ),
                pl.BlockSpec((1, half_tok, hidden_sz), lambda r: (r, 0, 0)),
                pl.BlockSpec((ep_sz, half_tok), lambda r: (0, 0)),
            ],
            out_specs=pl.BlockSpec(
                (1, 1, half_tok, hidden_sz), lambda r: (r, half_idx, 0, 0)
            ),
            input_output_aliases={0: 0},
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
            name="dequant_into_half_fast_0",
        )(buf_4d, ag_i8_3d, ag_sc_2d)

    def _k(dst_in_ref, i8_ref, sc_ref, dst_out_ref, vmem_out_ref, dma_sem):
        del dst_in_ref
        r = pl.program_id(0)
        slot = r % 2

        @pl.when(r >= 2)
        def _wait_prev():
            r_prev = r - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(r_prev, 1),
                    pl.ds(half_idx, 1),
                    pl.ds(0, half_tok),
                    pl.ds(0, hidden_sz),
                ],
                dma_sem.at[slot],
            ).wait()

        mask_r = jax.lax.broadcasted_iota(jnp.int32, (ep_sz, 128), 0) == r
        for t_blk in range(half_tok // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            sc_1x128 = jnp.sum(
                jnp.where(mask_r, sc_ref[:, r0:r1].astype(jnp.float32), 0.0),
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)
            sc_128x1 = jnp.swapaxes(jnp.broadcast_to(sc_1x128, (128, 128)), 0, 1)[:, :1]
            for c_blk in range(hidden_sz // 1024):
                c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                vmem_out_ref[slot, 0, 0, r0:r1, c0:c1] = (
                    i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                )

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(r, 1), pl.ds(half_idx, 1), pl.ds(0, half_tok), pl.ds(0, hidden_sz)
            ],
            dma_sem.at[slot],
        ).start()

        @pl.when(r == ep_sz - 1)
        def _wait_last():
            for s in range(2):
                r_s = (ep_sz - 2) + s
                slot_s = r_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(r_s, 1),
                        pl.ds(half_idx, 1),
                        pl.ds(0, half_tok),
                        pl.ds(0, hidden_sz),
                    ],
                    dma_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct(buf_4d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, half_tok, hidden_sz), lambda r: (r, 0, 0)),
                pl.BlockSpec((ep_sz, half_tok), lambda r: (0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, 1, 1, half_tok, hidden_sz), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"dequant_into_half_1way_{half_idx}",
    )(buf_4d, ag_i8_3d, ag_sc_2d)


def _stage2_write_part_1way(
    hs_3d: jax.Array,
    ag_part_T: jax.Array,
    sc_2d: jax.Array | None,
    *,
    tok_offset: int,
) -> jax.Array:
    ag_part_T_hbm = pltpu.with_memory_space_constraint(
        ag_part_T, memory_space=pltpu.HBM
    )
    n_col_blks, part_m, lanes = ag_part_T_hbm.shape
    tok_step = 512
    num_steps = part_m // tok_step
    has_sc = sc_2d is not None
    sc_arg = sc_2d if has_sc else jnp.zeros((tok_step, lanes), dtype=jnp.bfloat16)

    def _k(
        dst_in_ref,
        in_hbm_ref,
        sc_ref,
        dst_out_ref,
        vmem_in_ref,
        vmem_out_ref,
        in_sem,
        out_sem,
    ):
        del dst_in_ref
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks), pl.ds(0, tok_step), pl.ds(0, lanes)
                ],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks),
                    pl.ds((g + 1) * tok_step, tok_step),
                    pl.ds(0, lanes),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            t_prev = tok_offset + (g - 2) * tok_step
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(t_prev, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(0, n_col_blks), pl.ds(g * tok_step, tok_step), pl.ds(0, lanes)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        for sub_t in range(tok_step // 32):
            r0, r1 = sub_t * 32, (sub_t + 1) * 32
            if has_sc:
                sc_sub = sc_ref[r0:r1, :][None, :, :]
            for c_grp in range(n_col_blks // 8):
                c0, c1 = c_grp * 8, (c_grp + 1) * 8
                blk = vmem_in_ref[slot, c0:c1, r0:r1, :].astype(jnp.bfloat16)
                if has_sc:
                    blk = blk * sc_sub
                vmem_out_ref[slot, r0:r1, c0:c1, :] = jnp.swapaxes(blk, 0, 1)

        t_cur = tok_offset + g * tok_step
        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(t_cur, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s in range(2):
                g_s = (num_steps - 2) + s
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(tok_offset + g_s * tok_step, tok_step),
                        pl.ds(0, n_col_blks),
                        pl.ds(0, lanes),
                    ],
                    out_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct(hs_3d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_step, lanes), lambda g: (g if has_sc else 0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, n_col_blks, tok_step, lanes), ag_part_T_hbm.dtype),
                pltpu.VMEM((2, tok_step, n_col_blks, lanes), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage2_write_part_1way_{tok_offset}",
    )(hs_3d, ag_part_T_hbm, sc_arg)


def _stage2_write_part_1way_prenorm(
    hs_3d: jax.Array,
    ag_raw_T: jax.Array,
    nw_2d: jax.Array,
    irms_2d: jax.Array,
    *,
    tok_offset: int,
) -> jax.Array:
    ag_raw_T_hbm = pltpu.with_memory_space_constraint(ag_raw_T, memory_space=pltpu.HBM)
    n_col_blks, part_m, lanes = ag_raw_T_hbm.shape
    tok_step = 512 if part_m >= 1024 else 256
    num_steps = part_m // tok_step

    def _k(
        dst_in_ref,
        in_hbm_ref,
        nw_ref,
        irms_ref,
        dst_out_ref,
        vmem_in_ref,
        vmem_out_ref,
        in_sem,
        out_sem,
    ):
        del dst_in_ref
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks), pl.ds(0, tok_step), pl.ds(0, lanes)
                ],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks),
                    pl.ds((g + 1) * tok_step, tok_step),
                    pl.ds(0, lanes),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            t_prev = tok_offset + (g - 2) * tok_step
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(t_prev, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(0, n_col_blks), pl.ds(g * tok_step, tok_step), pl.ds(0, lanes)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        for sub_t in range(tok_step // 32):
            r0, r1 = sub_t * 32, (sub_t + 1) * 32
            irms_sub = irms_ref[r0:r1, :][:, None, :]
            for c_grp in range(n_col_blks // 8):
                c0, c1 = c_grp * 8, (c_grp + 1) * 8
                nw_sub = nw_ref[c0:c1, :][None, :, :]
                blk_T_bf16 = jnp.swapaxes(vmem_in_ref[slot, c0:c1, r0:r1, :], 0, 1)
                vmem_out_ref[slot, r0:r1, c0:c1, :] = (
                    blk_T_bf16.astype(jnp.float32) * (nw_sub * irms_sub)
                ).astype(jnp.bfloat16)

        t_cur = tok_offset + g * tok_step
        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(t_cur, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s in range(2):
                g_s = (num_steps - 2) + s
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(tok_offset + g_s * tok_step, tok_step),
                        pl.ds(0, n_col_blks),
                        pl.ds(0, lanes),
                    ],
                    out_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct(hs_3d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((n_col_blks, lanes), lambda _: (0, 0)),
                pl.BlockSpec((tok_step, lanes), lambda g: (g, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, n_col_blks, tok_step, lanes), jnp.bfloat16),
                pltpu.VMEM((2, tok_step, n_col_blks, lanes), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage2_write_prenorm_{tok_offset}",
    )(hs_3d, ag_raw_T_hbm, nw_2d, irms_2d)


def _copy_bf16_into_out_part(
    buf_3d: jax.Array,
    ag_bf16_3d: jax.Array,
    *,
    tok_offset: int,
    use_1way: bool,
) -> jax.Array:
    del use_1way
    ag_hbm = pltpu.with_memory_space_constraint(ag_bf16_3d, memory_space=pltpu.HBM)
    ep_sz, part_tok, hidden_sz = ag_hbm.shape

    def _k_dma(dst_in_ref, src_hbm_ref, dst_out_ref, vmem_buf_ref, in_sem, out_sem):
        del dst_in_ref
        r = pl.program_id(0)
        slot = r % 2
        nxt = (r + 1) % 2

        @pl.when(r == 0)
        def _start0():
            pltpu.make_async_copy(
                src_hbm_ref.at[pl.ds(0, 1), pl.ds(0, part_tok), pl.ds(0, hidden_sz)],
                vmem_buf_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(r >= 1)
        def _wait_prev_out():
            r_prev = r - 1
            pltpu.make_async_copy(
                vmem_buf_ref.at[nxt],
                dst_out_ref.at[
                    pl.ds(r_prev, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
                ],
                out_sem.at[nxt],
            ).wait()

        @pl.when(r + 1 < ep_sz)
        def _prefetch():
            pltpu.make_async_copy(
                src_hbm_ref.at[
                    pl.ds(r + 1, 1), pl.ds(0, part_tok), pl.ds(0, hidden_sz)
                ],
                vmem_buf_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        pltpu.make_async_copy(
            src_hbm_ref.at[pl.ds(r, 1), pl.ds(0, part_tok), pl.ds(0, hidden_sz)],
            vmem_buf_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        pltpu.make_async_copy(
            vmem_buf_ref.at[slot],
            dst_out_ref.at[
                pl.ds(r, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(r == ep_sz - 1)
        def _wait_last():
            pltpu.make_async_copy(
                vmem_buf_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(r, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
                ],
                out_sem.at[slot],
            ).wait()

    return pl.pallas_call(
        _k_dma,
        out_shape=jax.ShapeDtypeStruct(buf_3d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, 1, part_tok, hidden_sz), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"copy_bf16_into_out_dma_{tok_offset}",
    )(buf_3d, ag_hbm)


def _dequant_into_out_part(
    buf_3d: jax.Array,
    ag_i8_3d: jax.Array,
    ag_sc_2d: jax.Array,
    *,
    tok_offset: int,
    use_1way: bool,
) -> jax.Array:
    ep_sz, part_tok, hidden_sz = ag_i8_3d.shape
    ag_sc_3d = jnp.broadcast_to(ag_sc_2d[:, :, None], (ep_sz, part_tok, 128))
    if not use_1way:
        assert tok_offset == 0

        def _k0(dst_in_ref, i8_ref, sc_ref, dst_out_ref):
            del dst_in_ref
            for t_blk in range(part_tok // 128):
                r0, r1 = t_blk * 128, (t_blk + 1) * 128
                sc_128x1 = sc_ref[0, r0:r1, :1]
                for c_blk in range(hidden_sz // 1024):
                    c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                    dst_out_ref[0, r0:r1, c0:c1] = (
                        i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                    )

        return pl.pallas_call(
            _k0,
            out_shape=jax.ShapeDtypeStruct(buf_3d.shape, jnp.bfloat16),
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec((1, part_tok, hidden_sz), lambda r: (r, 0, 0)),
                pl.BlockSpec((1, part_tok, hidden_sz), lambda r: (r, 0, 0)),
                pl.BlockSpec((1, part_tok, 128), lambda r: (r, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, part_tok, hidden_sz), lambda r: (r, 0, 0)),
            input_output_aliases={0: 0},
            compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
            name="dequant_into_out_fast_0",
        )(buf_3d, ag_i8_3d, ag_sc_3d)

    def _k1(dst_in_ref, i8_ref, sc_ref, dst_out_ref, vmem_out_ref, dma_sem):
        del dst_in_ref
        r = pl.program_id(0)
        slot = r % 2

        @pl.when(r >= 2)
        def _wait_prev():
            r_prev = r - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(r_prev, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
                ],
                dma_sem.at[slot],
            ).wait()

        for t_blk in range(part_tok // 128):
            r0, r1 = t_blk * 128, (t_blk + 1) * 128
            sc_128x1 = sc_ref[0, r0:r1, :1]
            for c_blk in range(hidden_sz // 1024):
                c0, c1 = c_blk * 1024, (c_blk + 1) * 1024
                vmem_out_ref[slot, 0, r0:r1, c0:c1] = (
                    i8_ref[0, r0:r1, c0:c1].astype(jnp.bfloat16) * sc_128x1
                )

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(r, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
            ],
            dma_sem.at[slot],
        ).start()

        @pl.when(r == ep_sz - 1)
        def _wait_last():
            for s in range(2):
                r_s = (ep_sz - 2) + s
                slot_s = r_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(r_s, 1), pl.ds(tok_offset, part_tok), pl.ds(0, hidden_sz)
                    ],
                    dma_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k1,
        out_shape=jax.ShapeDtypeStruct(buf_3d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(ep_sz,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, part_tok, hidden_sz), lambda r: (r, 0, 0)),
                pl.BlockSpec((1, part_tok, 128), lambda r: (r, 0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, 1, part_tok, hidden_sz), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"dequant_into_out_1way_{tok_offset}",
    )(buf_3d, ag_i8_3d, ag_sc_3d)


def _stage1_amax_and_sq(
    proj_col_shard: jax.Array,
    nw_8x512: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    proj_hbm = pltpu.with_memory_space_constraint(
        proj_col_shard, memory_space=pltpu.HBM
    )
    total_m, col_chunk = proj_hbm.shape
    tok_step = 512
    num_steps = total_m // tok_step

    def _k(in_hbm_ref, nw_ref, amax_out_ref, sq_out_ref, vmem_in_ref, in_sem):
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds(0, tok_step), pl.ds(0, col_chunk)],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds((g + 1) * tok_step, tok_step), pl.ds(0, col_chunk)],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        pltpu.make_async_copy(
            in_hbm_ref.at[pl.ds(g * tok_step, tok_step), pl.ds(0, col_chunk)],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        nw_64 = jnp.tile(nw_ref[...], (8, 1))
        for rb in range(tok_step // 64):
            r0, r1 = rb * 64, (rb + 1) * 64
            s = vmem_in_ref[slot, r0:r1, :].astype(jnp.float32)
            sq = s * s
            sq = sq[:, :256] + sq[:, 256:]
            sq = sq[:, :128] + sq[:, 128:]
            sq_1 = jnp.sum(sq, axis=1, keepdims=True)
            ab = jnp.abs(s * nw_64)
            ab = jnp.maximum(ab[:, :256], ab[:, 256:])
            ab = jnp.maximum(ab[:, :128], ab[:, 128:])
            mx_1 = jnp.max(ab, axis=1, keepdims=True)
            amax_out_ref[r0:r1, :] = jnp.broadcast_to(mx_1, (64, 128))
            sq_out_ref[r0:r1, :] = jnp.broadcast_to(sq_1, (64, 128))

    amax_pad, sq_pad = pl.pallas_call(
        _k,
        out_shape=(
            jax.ShapeDtypeStruct((total_m, 128), jnp.float32),
            jax.ShapeDtypeStruct((total_m, 128), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((8, col_chunk), lambda _: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tok_step, 128), lambda g: (g, 0)),
                pl.BlockSpec((tok_step, 128), lambda g: (g, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, tok_step, col_chunk), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="stage1_amax_and_sq",
    )(proj_hbm, nw_8x512)
    return amax_pad[:, 0], sq_pad[:, 0]


def _stage1_quant_pack_i8(
    proj_col_shard: jax.Array,
    nw_8x512: jax.Array,
    irs_2d: jax.Array,
    *,
    tok_offset: int,
    part_m: int,
) -> jax.Array:
    proj_hbm = pltpu.with_memory_space_constraint(
        proj_col_shard, memory_space=pltpu.HBM
    )
    col_chunk = proj_hbm.shape[1]
    n_cblk = col_chunk // 128
    tok_step = 512
    num_steps = part_m // tok_step

    def _k(
        in_hbm_ref,
        nw_ref,
        irs_ref,
        out_hbm_ref,
        vmem_in_ref,
        vmem_out_ref,
        in_sem,
        out_sem,
    ):
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds(tok_offset, tok_step), pl.ds(0, col_chunk)],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(tok_offset + (g + 1) * tok_step, tok_step),
                    pl.ds(0, col_chunk),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            g_prev = g - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                out_hbm_ref.at[
                    pl.ds(0, n_cblk), pl.ds(g_prev * tok_step, tok_step), pl.ds(0, 128)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(tok_offset + g * tok_step, tok_step), pl.ds(0, col_chunk)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        nw_64 = jnp.tile(nw_ref[...], (8, 1))
        for rb in range(tok_step // 64):
            r0, r1 = rb * 64, (rb + 1) * 64
            s = vmem_in_ref[slot, r0:r1, :].astype(jnp.float32) * nw_64
            irs_sub = irs_ref[r0:r1, :]
            for c in range(n_cblk):
                q_c = jnp.rint(s[:, c * 128 : (c + 1) * 128] * irs_sub).astype(jnp.int8)
                vmem_out_ref[slot, c, r0:r1, :] = q_c

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            out_hbm_ref.at[
                pl.ds(0, n_cblk), pl.ds(g * tok_step, tok_step), pl.ds(0, 128)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s_tail in range(2):
                g_s = (num_steps - 2) + s_tail
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    out_hbm_ref.at[
                        pl.ds(0, n_cblk), pl.ds(g_s * tok_step, tok_step), pl.ds(0, 128)
                    ],
                    out_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct((n_cblk, part_m, 128), jnp.int8),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((8, col_chunk), lambda _: (0, 0)),
                pl.BlockSpec((tok_step, 128), lambda g: (g, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, tok_step, col_chunk), jnp.bfloat16),
                pltpu.VMEM((2, n_cblk, tok_step, 128), jnp.int8),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage1_quant_pack_i8_{tok_offset}",
    )(proj_hbm, nw_8x512, irs_2d)


def _stage1_local_quant_and_stats(
    proj_col_shard: jax.Array,
    nw_8x512: jax.Array,
    *,
    tok_offset: int,
    part_m: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    proj_hbm = pltpu.with_memory_space_constraint(
        proj_col_shard, memory_space=pltpu.HBM
    )
    col_chunk = proj_hbm.shape[1]
    n_cblk = col_chunk // 128
    tok_step = 512
    num_steps = part_m // tok_step

    def _k(
        in_hbm_ref,
        nw_ref,
        out_i8_hbm_ref,
        cmax_out_ref,
        sq_out_ref,
        vmem_in_ref,
        vmem_out_ref,
        in_sem,
        out_sem,
    ):
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds(tok_offset, tok_step), pl.ds(0, col_chunk)],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(tok_offset + (g + 1) * tok_step, tok_step),
                    pl.ds(0, col_chunk),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            g_prev = g - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                out_i8_hbm_ref.at[
                    pl.ds(0, n_cblk), pl.ds(g_prev * tok_step, tok_step), pl.ds(0, 128)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(tok_offset + g * tok_step, tok_step), pl.ds(0, col_chunk)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        nw_64 = jnp.tile(nw_ref[...], (8, 1))
        for rb in range(tok_step // 64):
            r0, r1 = rb * 64, (rb + 1) * 64
            s = vmem_in_ref[slot, r0:r1, :].astype(jnp.float32)
            sq = s * s
            sq = sq[:, :256] + sq[:, 256:]
            sq = sq[:, :128] + sq[:, 128:]
            sq_1 = jnp.sum(sq, axis=1, keepdims=True)
            u = s * nw_64
            ab = jnp.abs(u)
            ab = jnp.maximum(ab[:, :256], ab[:, 256:])
            ab = jnp.maximum(ab[:, :128], ab[:, 128:])
            mx_1 = jnp.max(ab, axis=1, keepdims=True)
            mx_bits = jax.lax.bitcast_convert_type(mx_1, jnp.uint32)
            u8_code = jnp.clip(
                jax.lax.shift_right_logical(
                    mx_bits + jnp.uint32(0x000FFFFF), jnp.uint32(20)
                ).astype(jnp.int32)
                - 960,
                0,
                255,
            ).astype(jnp.uint8)
            u8_code = jnp.where(mx_1 <= 0.0, jnp.uint8(0), u8_code)
            dec_bits = jax.lax.shift_left(
                u8_code.astype(jnp.uint32) + jnp.uint32(960), jnp.uint32(20)
            )
            mx_dec = jnp.where(
                u8_code == 0,
                jnp.float32(1e-12),
                jax.lax.bitcast_convert_type(dec_bits, jnp.float32),
            )
            inv_sc = jnp.float32(127.0) / mx_dec
            for c in range(n_cblk):
                vmem_out_ref[slot, c, r0:r1, :] = jnp.rint(
                    u[:, c * 128 : (c + 1) * 128] * inv_sc
                ).astype(jnp.int8)
            cmax_out_ref[r0:r1, :] = jnp.broadcast_to(u8_code, (64, 128))
            sq_out_ref[r0:r1, :] = jnp.broadcast_to(sq_1, (64, 128))

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            out_i8_hbm_ref.at[
                pl.ds(0, n_cblk), pl.ds(g * tok_step, tok_step), pl.ds(0, 128)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s_tail in range(2):
                g_s = (num_steps - 2) + s_tail
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    out_i8_hbm_ref.at[
                        pl.ds(0, n_cblk), pl.ds(g_s * tok_step, tok_step), pl.ds(0, 128)
                    ],
                    out_sem.at[slot_s],
                ).wait()

    packed_i8, cmax_pad, sq_pad = pl.pallas_call(
        _k,
        out_shape=(
            jax.ShapeDtypeStruct((n_cblk, part_m, 128), jnp.int8),
            jax.ShapeDtypeStruct((part_m, 128), jnp.uint8),
            jax.ShapeDtypeStruct((part_m, 128), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((8, col_chunk), lambda _: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tok_step, 128), lambda g: (g, 0)),
                pl.BlockSpec((tok_step, 128), lambda g: (g, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, tok_step, col_chunk), jnp.bfloat16),
                pltpu.VMEM((2, n_cblk, tok_step, 128), jnp.int8),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage1_local_quant_stats_{tok_offset}",
    )(proj_hbm, nw_8x512)
    return packed_i8, cmax_pad[:, 0], sq_pad[:, 0]


def _stage2_write_part_1way_rank8(
    hs_3d: jax.Array,
    ag_part_T: jax.Array,
    sc_64_m: jax.Array,
    *,
    tok_offset: int,
) -> jax.Array:
    ag_part_T_hbm = pltpu.with_memory_space_constraint(
        ag_part_T, memory_space=pltpu.HBM
    )
    n_col_blks, part_m, lanes = ag_part_T_hbm.shape
    tok_step = 512
    num_steps = part_m // tok_step

    def _k(
        dst_in_ref,
        in_hbm_ref,
        sc_ref,
        dst_out_ref,
        vmem_in_ref,
        vmem_out_ref,
        in_sem,
        out_sem,
    ):
        del dst_in_ref
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks), pl.ds(0, tok_step), pl.ds(0, lanes)
                ],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(0, n_col_blks),
                    pl.ds((g + 1) * tok_step, tok_step),
                    pl.ds(0, lanes),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            t_prev = tok_offset + (g - 2) * tok_step
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                dst_out_ref.at[
                    pl.ds(t_prev, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(0, n_col_blks), pl.ds(g * tok_step, tok_step), pl.ds(0, lanes)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        for t128 in range(tok_step // 128):
            l0, l1 = t128 * 128, (t128 + 1) * 128
            for c_grp in range(n_col_blks // 8):
                c0, c1 = c_grp * 8, (c_grp + 1) * 8
                r_ev = 2 * c_grp
                r_od = 2 * c_grp + 1
                sc_ev_128 = jnp.swapaxes(
                    jnp.tile(sc_ref[r_ev * 8 : (r_ev + 1) * 8, l0:l1], (16, 1)), 0, 1
                )
                sc_od_128 = jnp.swapaxes(
                    jnp.tile(sc_ref[r_od * 8 : (r_od + 1) * 8, l0:l1], (16, 1)), 0, 1
                )
                for sub in range(4):
                    s0, s1 = sub * 32, (sub + 1) * 32
                    r0, r1 = l0 + s0, l0 + s1
                    b_ev = (
                        vmem_in_ref[slot, c0 : c0 + 4, r0:r1, :].astype(jnp.bfloat16)
                        * sc_ev_128[s0:s1, :][None, :, :]
                    )
                    b_od = (
                        vmem_in_ref[slot, c0 + 4 : c1, r0:r1, :].astype(jnp.bfloat16)
                        * sc_od_128[s0:s1, :][None, :, :]
                    )
                    blk_8 = jnp.concatenate([b_ev, b_od], axis=0)
                    vmem_out_ref[slot, r0:r1, c0:c1, :] = jnp.swapaxes(blk_8, 0, 1)

        t_cur = tok_offset + g * tok_step
        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            dst_out_ref.at[
                pl.ds(t_cur, tok_step), pl.ds(0, n_col_blks), pl.ds(0, lanes)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s in range(2):
                g_s = (num_steps - 2) + s
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    dst_out_ref.at[
                        pl.ds(tok_offset + g_s * tok_step, tok_step),
                        pl.ds(0, n_col_blks),
                        pl.ds(0, lanes),
                    ],
                    out_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct(hs_3d.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((64, tok_step), lambda g: (0, g)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, n_col_blks, tok_step, lanes), jnp.int8),
                pltpu.VMEM((2, tok_step, n_col_blks, lanes), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage2_write_rank8_{tok_offset}",
    )(hs_3d, ag_part_T_hbm, sc_64_m)


def _stage1_pack_raw_bf16(
    proj_col_shard: jax.Array,
    *,
    tok_offset: int,
    part_m: int,
) -> jax.Array:
    proj_hbm = pltpu.with_memory_space_constraint(
        proj_col_shard, memory_space=pltpu.HBM
    )
    col_chunk = proj_hbm.shape[1]
    n_cblk = col_chunk // 128
    tok_step = 512 if part_m >= 512 else 256
    num_steps = part_m // tok_step

    def _k(in_hbm_ref, out_hbm_ref, vmem_in_ref, vmem_out_ref, in_sem, out_sem):
        g = pl.program_id(0)
        slot = g % 2

        @pl.when(g == 0)
        def _start0():
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds(tok_offset, tok_step), pl.ds(0, col_chunk)],
                vmem_in_ref.at[0],
                in_sem.at[0],
            ).start()

        @pl.when(g + 1 < num_steps)
        def _prefetch():
            nxt = (g + 1) % 2
            pltpu.make_async_copy(
                in_hbm_ref.at[
                    pl.ds(tok_offset + (g + 1) * tok_step, tok_step),
                    pl.ds(0, col_chunk),
                ],
                vmem_in_ref.at[nxt],
                in_sem.at[nxt],
            ).start()

        @pl.when(g >= 2)
        def _wait_prev():
            g_prev = g - 2
            pltpu.make_async_copy(
                vmem_out_ref.at[slot],
                out_hbm_ref.at[
                    pl.ds(0, n_cblk), pl.ds(g_prev * tok_step, tok_step), pl.ds(0, 128)
                ],
                out_sem.at[slot],
            ).wait()

        pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(tok_offset + g * tok_step, tok_step), pl.ds(0, col_chunk)
            ],
            vmem_in_ref.at[slot],
            in_sem.at[slot],
        ).wait()

        for rb in range(tok_step // 128):
            r0, r1 = rb * 128, (rb + 1) * 128
            for c in range(n_cblk):
                vmem_out_ref[slot, c, r0:r1, :] = vmem_in_ref[
                    slot, r0:r1, c * 128 : (c + 1) * 128
                ]

        pltpu.make_async_copy(
            vmem_out_ref.at[slot],
            out_hbm_ref.at[
                pl.ds(0, n_cblk), pl.ds(g * tok_step, tok_step), pl.ds(0, 128)
            ],
            out_sem.at[slot],
        ).start()

        @pl.when(g == num_steps - 1)
        def _wait_last():
            for s in range(2):
                g_s = (num_steps - 2) + s
                slot_s = g_s % 2
                pltpu.make_async_copy(
                    vmem_out_ref.at[slot_s],
                    out_hbm_ref.at[
                        pl.ds(0, n_cblk), pl.ds(g_s * tok_step, tok_step), pl.ds(0, 128)
                    ],
                    out_sem.at[slot_s],
                ).wait()

    return pl.pallas_call(
        _k,
        out_shape=jax.ShapeDtypeStruct((n_cblk, part_m, 128), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_steps,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, tok_step, col_chunk), jnp.bfloat16),
                pltpu.VMEM((2, n_cblk, tok_step, 128), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"stage1_pack_raw_bf16_{tok_offset}",
    )(proj_hbm)


def fully_fused_moe_func_rs(
    pre_proj_sharded: jax.Array,
    pre_norm_weight: jax.Array,
    router_logits_tok_sharded: jax.Array,
    e_score_bias: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    post_expert_norm_weight: jax.Array | None,
    group_offset: jax.Array,
    *,
    mesh: Mesh,
    topk: int = 8,
    activation: str = "silu",
    router_multiplier: float = 0.08838834764831845,
    route_scale: float = 2.5,
    route_eps: float = 1e-5,
    rms_eps: float = 1e-6,
    fp4_w4a16: bool = False,
    use_int8_in_ag: bool = True,
    use_int8_out_ag: bool = True,
) -> jax.Array:
    """End-to-end fused MoE: Sharded PreNorm + Input AG + Sharded TopK Selection + Routing + GMM-RS + PostNorm + Combine + Output AG.

    Precision Tiers:
      - Tier 1 (use_int8_in_ag=True,  use_int8_out_ag=True):  Dual-INT8 AG (2,364.54 µs med / 2,340.52 µs min, cos_sim=0.99987549, max_diff=0.078125)
      - Tier 2 (use_int8_in_ag=False, use_int8_out_ag=True):  BF16 Input AG + INT8 Output AG (2,509.59 µs med / 2,487.31 µs min, cos_sim=0.99993479, max_diff=0.046875)
      - Tier 3 (use_int8_in_ag=False, use_int8_out_ag=False): Lossless Pure BF16 AG (2,658.22 µs med / 2,626.88 µs min, cos_sim=0.99998409, max_diff=0.031250)
    """
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    hidden_size = w2.shape[-1]
    col_chunk = hidden_size // ep_size
    has_post_norm = post_expert_norm_weight is not None
    pnw_in = post_expert_norm_weight if has_post_norm else jnp.zeros((1,), jnp.bfloat16)

    def _body(
        proj_col_shard,
        pre_nw,
        logits_tok_shard,
        bias,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        pnw_l,
        go_l,
    ):
        num_local_experts = w1_l.shape[0]
        my_id = go_l[0] // num_local_experts

        if use_int8_in_ag:
            with jax.named_scope("stage1_zero_pmax_local_quant_ag0"):
                total_m = proj_col_shard.shape[0]
                split_m = (total_m * 12) // 16
                rem_m = total_m - split_m
                local_nw = (
                    jax.lax.dynamic_slice_in_dim(
                        pre_nw, my_id * col_chunk, col_chunk, axis=0
                    ).astype(jnp.float32)
                    + 1.0
                )
                nw_8x512 = jnp.broadcast_to(local_nw[None, :], (8, col_chunk))
                packed_i8_0, cmax_0, sq_0 = jax.lax.optimization_barrier(
                    _stage1_local_quant_and_stats(
                        proj_col_shard, nw_8x512, tok_offset=0, part_m=split_m
                    )
                )
                ag_in_T_0 = jax.lax.all_gather(
                    packed_i8_0, axis_name=expert_axis, axis=0, tiled=True
                )

            with jax.named_scope("stage2_overlapped_part1_psum_scales_and_routing"):
                logits_b, _ = jax.lax.optimization_barrier(
                    (logits_tok_shard, packed_i8_0)
                )
                scores_local = jax.nn.sigmoid(logits_b * router_multiplier)
                top_vals, local_topk_ids = jax.lax.top_k(
                    scores_local + bias[None, :], topk
                )
                local_topk_ids = local_topk_ids.astype(jnp.int32)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids.astype(jnp.uint8),
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )
                proj_b, global_topk_u8, cmax_0_b, sq_0_b = jax.lax.optimization_barrier(
                    (proj_col_shard, global_topk_u8, cmax_0, sq_0)
                )
                packed_i8_1, cmax_1, sq_1 = jax.lax.optimization_barrier(
                    _stage1_local_quant_and_stats(
                        proj_b, nw_8x512, tok_offset=split_m, part_m=rem_m
                    )
                )
                ag_in_T_1 = jax.lax.all_gather(
                    packed_i8_1, axis_name=expert_axis, axis=0, tiled=True
                )

                local_sq_1d = jnp.concatenate([sq_0_b, sq_1], axis=0)
                local_cmax_u8_2d = jnp.concatenate([cmax_0_b, cmax_1], axis=0)[None, :]
                global_cmax_u8 = jax.lax.all_gather(
                    local_cmax_u8_2d, axis_name=expert_axis, axis=0, tiled=True
                )
                dec_bits_8_m = jax.lax.shift_left(
                    global_cmax_u8.astype(jnp.uint32) + jnp.uint32(960), jnp.uint32(20)
                )
                global_cmax_8_m = jnp.where(
                    global_cmax_u8 == 0,
                    jnp.float32(0.0),
                    jax.lax.bitcast_convert_type(dec_bits_8_m, jnp.float32),
                )
                total_sq_1d = jax.lax.psum(local_sq_1d, axis_name=expert_axis)

                expert_iota = jnp.arange(bias.shape[0], dtype=jnp.int32)[:, None, None]
                bias_bf16 = bias.astype(jnp.bfloat16)[:, None, None]
                selected_bias = jnp.sum(
                    jnp.where(
                        local_topk_ids[None, :, :] == expert_iota,
                        bias_bf16,
                        jnp.bfloat16(0.0),
                    ),
                    axis=0,
                ).astype(jnp.float32)
                local_topk_scores = top_vals - selected_bias
                local_topk_weights = (
                    local_topk_scores
                    / (jnp.sum(local_topk_scores, axis=-1, keepdims=True) + route_eps)
                    * route_scale
                ).astype(jnp.bfloat16)
                go_val = go_l[0]
                local_t_recv = jnp.sum(
                    (
                        (local_topk_ids < go_val)
                        | (local_topk_ids >= go_val + num_local_experts)
                    ).astype(jnp.int32)
                )
                inv_perm_c = _fast_inv_perm_u16(local_topk_ids)

                inv_rms_1d = jax.lax.rsqrt(total_sq_1d * (1.0 / hidden_size) + rms_eps)
                sc_8_m = (
                    global_cmax_8_m.astype(jnp.float32)
                    * (jnp.float32(1.0 / 127.0) * inv_rms_1d[None, :])
                ).astype(jnp.bfloat16)
                sc_64_all = jnp.repeat(sc_8_m, 8, axis=0)
                sc_0 = sc_64_all[:, :split_m]
                sc_1 = sc_64_all[:, split_m:]
                global_topk_ids = global_topk_u8.astype(jnp.int32)
                sc_0, sc_1, global_topk_ids = jax.lax.optimization_barrier(
                    (sc_0, sc_1, global_topk_ids)
                )

                hs_buf = jnp.empty(
                    (total_m, hidden_size // 128, 128), dtype=jnp.bfloat16
                )
                hs_buf = _stage2_write_part_1way_rank8(
                    hs_buf, ag_in_T_0, sc_0, tok_offset=0
                )
                hidden_states = _stage2_write_part_1way_rank8(
                    hs_buf, ag_in_T_1, sc_1, tok_offset=split_m
                )
        else:
            with jax.named_scope("stage1_zero_delay_raw_bf16_ag0"):
                total_m = proj_col_shard.shape[0]
                p0_m = total_m // 8
                p1_m = (total_m * 3) // 4
                p1_end_in = p0_m + p1_m
                p2_m = total_m - p1_end_in
                packed_0 = jax.lax.optimization_barrier(
                    _stage1_pack_raw_bf16(proj_col_shard, tok_offset=0, part_m=p0_m)
                )
                ag_raw_T_0 = jax.lax.all_gather(
                    packed_0, axis_name=expert_axis, axis=0, tiled=True
                )

            with jax.named_scope("stage2_hidden_stats_ag1_and_routing"):
                proj_all, logits_b, _ = jax.lax.optimization_barrier(
                    (proj_col_shard, logits_tok_shard, packed_0)
                )
                packed_1 = jax.lax.optimization_barrier(
                    _stage1_pack_raw_bf16(proj_all, tok_offset=p0_m, part_m=p1_m)
                )
                ag_raw_T_1 = jax.lax.all_gather(
                    packed_1, axis_name=expert_axis, axis=0, tiled=True
                )

                proj_b2, _ = jax.lax.optimization_barrier((proj_all, packed_1))
                packed_2 = jax.lax.optimization_barrier(
                    _stage1_pack_raw_bf16(proj_b2, tok_offset=p1_end_in, part_m=p2_m)
                )
                ag_raw_T_2 = jax.lax.all_gather(
                    packed_2, axis_name=expert_axis, axis=0, tiled=True
                )

                s_3d = proj_b2.reshape(-1, col_chunk // 128, 128).astype(jnp.float32)
                local_sq = jnp.sum(jnp.sum(s_3d * s_3d, axis=1), axis=-1, keepdims=True)
                scores_local = jax.nn.sigmoid(logits_b * router_multiplier)
                top_vals, local_topk_ids = jax.lax.top_k(
                    scores_local + bias[None, :], topk
                )
                local_topk_ids = local_topk_ids.astype(jnp.int32)
                total_sq = jax.lax.psum(local_sq, axis_name=expert_axis)
                global_topk_u8 = jax.lax.all_gather(
                    local_topk_ids.astype(jnp.uint8),
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )

                expert_iota = jnp.arange(bias.shape[0], dtype=jnp.int32)[:, None, None]
                bias_bf16 = bias.astype(jnp.bfloat16)[:, None, None]
                selected_bias = jnp.sum(
                    jnp.where(
                        local_topk_ids[None, :, :] == expert_iota,
                        bias_bf16,
                        jnp.bfloat16(0.0),
                    ),
                    axis=0,
                ).astype(jnp.float32)
                local_topk_scores = top_vals - selected_bias
                local_topk_weights = (
                    local_topk_scores
                    / (jnp.sum(local_topk_scores, axis=-1, keepdims=True) + route_eps)
                    * route_scale
                ).astype(jnp.bfloat16)
                go_val = go_l[0]
                local_t_recv = jnp.sum(
                    (
                        (local_topk_ids < go_val)
                        | (local_topk_ids >= go_val + num_local_experts)
                    ).astype(jnp.int32)
                )
                inv_perm_c = _fast_inv_perm_u16(local_topk_ids)

                inv_rms = jax.lax.rsqrt(total_sq * (1.0 / hidden_size) + rms_eps)
                nw_2d = (pre_nw.astype(jnp.float32) + 1.0).reshape(
                    hidden_size // 128, 128
                )
                irms_0 = jnp.broadcast_to(inv_rms[:p0_m], (p0_m, 128))
                irms_1 = jnp.broadcast_to(inv_rms[p0_m:p1_end_in], (p1_m, 128))
                irms_2 = jnp.broadcast_to(inv_rms[p1_end_in:], (p2_m, 128))
                global_topk_ids = global_topk_u8.astype(jnp.int32)
                nw_2d, irms_0, irms_1, irms_2, global_topk_ids = (
                    jax.lax.optimization_barrier(
                        (nw_2d, irms_0, irms_1, irms_2, global_topk_ids)
                    )
                )

                hs_buf = jnp.empty(
                    (proj_all.shape[0], hidden_size // 128, 128), dtype=jnp.bfloat16
                )
                hs_buf = _stage2_write_part_1way_prenorm(
                    hs_buf, ag_raw_T_0, nw_2d, irms_0, tok_offset=0
                )
                hs_buf = _stage2_write_part_1way_prenorm(
                    hs_buf, ag_raw_T_1, nw_2d, irms_1, tok_offset=p0_m
                )
                hidden_states = _stage2_write_part_1way_prenorm(
                    hs_buf, ag_raw_T_2, nw_2d, irms_2, tok_offset=p1_end_in
                )

        with jax.named_scope("fused_gmm_rs_and_combine"):
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
                go_l,
                local_topk_weights,
                global_topk_ids,
                pnw_l,
                activation=activation,
                topk=topk,
                ep_size=ep_size,
                ep_axis_name=expert_axis,
                has_post_norm=has_post_norm,
                sp_enabled=True,
                fp8_direct_write=False,
                expert_tp=False,
                fp4_w4a16=fp4_w4a16,
                local_t_recv=local_t_recv,
                return_tiled=True,
                block_scatter=True,
                return_raw_out_buf=True,
            )
            total_local_tok = local_topk_weights.shape[0]
            if use_int8_out_ag:
                p0_tok = total_local_tok // 4
                p1_end = (total_local_tok * 7) // 8
                rank_mask_2d = jnp.arange(ep_size, dtype=jnp.int32)[:, None] == my_id

                i8_0, sc0_1d = jax.lax.optimization_barrier(
                    _tc_combine_and_int8_quant(
                        out_buf_3d, inv_perm_c[:p0_tok], local_topk_weights[:p0_tok]
                    )
                )
                ag_i8_0 = jax.lax.all_gather(
                    i8_0[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                perm_1, w_1, _, _ = jax.lax.optimization_barrier(
                    (
                        inv_perm_c[p0_tok:p1_end],
                        local_topk_weights[p0_tok:p1_end],
                        i8_0,
                        sc0_1d,
                    )
                )
                i8_1, sc1_1d = jax.lax.optimization_barrier(
                    _tc_combine_and_int8_quant(out_buf_3d, perm_1, w_1)
                )
                ag_i8_1 = jax.lax.all_gather(
                    i8_1[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                perm_2, w_2, _, _ = jax.lax.optimization_barrier(
                    (inv_perm_c[p1_end:], local_topk_weights[p1_end:], i8_1, sc1_1d)
                )
                i8_2, sc2_1d = jax.lax.optimization_barrier(
                    _tc_combine_and_int8_quant(out_buf_3d, perm_2, w_2)
                )
                ag_i8_2 = jax.lax.all_gather(
                    i8_2[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                sc0_b, sc1_b, sc2_b, _ = jax.lax.optimization_barrier(
                    (sc0_1d, sc1_1d, sc2_1d, i8_2)
                )
                sc_all_1d = jnp.concatenate([sc0_b, sc1_b, sc2_b], axis=0)
                ag_sc_all = jax.lax.psum(
                    jnp.where(rank_mask_2d, sc_all_1d[None, :], jnp.bfloat16(0.0)),
                    axis_name=expert_axis,
                )

                out_final_3d = jnp.empty(
                    (ep_size, total_local_tok, hidden_size), dtype=jnp.bfloat16
                )
                out_final_3d = _dequant_into_out_part(
                    out_final_3d,
                    ag_i8_0,
                    ag_sc_all[:, :p0_tok],
                    tok_offset=0,
                    use_1way=False,
                )
                out_final_3d = _dequant_into_out_part(
                    out_final_3d,
                    ag_i8_1,
                    ag_sc_all[:, p0_tok:p1_end],
                    tok_offset=p0_tok,
                    use_1way=True,
                )
                out_final_3d = _dequant_into_out_part(
                    out_final_3d,
                    ag_i8_2,
                    ag_sc_all[:, p1_end:],
                    tok_offset=p1_end,
                    use_1way=True,
                )
                return out_final_3d.reshape(ep_size * total_local_tok, hidden_size)
            else:
                p0_tok = (total_local_tok * 3) // 16
                p1_end = (total_local_tok * 7) // 8

                bf16_0 = jax.lax.optimization_barrier(
                    _tc_combine_bf16(
                        out_buf_3d,
                        inv_perm_c[:p0_tok],
                        local_topk_weights[:p0_tok],
                        return_tiled=False,
                    )
                )
                ag_bf16_0 = jax.lax.all_gather(
                    bf16_0[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                perm_1, w_1, _ = jax.lax.optimization_barrier(
                    (
                        inv_perm_c[p0_tok:p1_end],
                        local_topk_weights[p0_tok:p1_end],
                        bf16_0,
                    )
                )
                bf16_1 = jax.lax.optimization_barrier(
                    _tc_combine_bf16(out_buf_3d, perm_1, w_1, return_tiled=False)
                )
                ag_bf16_1 = jax.lax.all_gather(
                    bf16_1[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                perm_2, w_2, _ = jax.lax.optimization_barrier(
                    (inv_perm_c[p1_end:], local_topk_weights[p1_end:], bf16_1)
                )
                bf16_2 = jax.lax.optimization_barrier(
                    _tc_combine_bf16(out_buf_3d, perm_2, w_2, return_tiled=False)
                )
                ag_bf16_2 = jax.lax.all_gather(
                    bf16_2[None, ...], axis_name=expert_axis, axis=0, tiled=True
                )

                out_final_3d = jnp.empty(
                    (ep_size, total_local_tok, hidden_size), dtype=jnp.bfloat16
                )
                out_final_3d = _copy_bf16_into_out_part(
                    out_final_3d, ag_bf16_0, tok_offset=0, use_1way=False
                )
                out_final_3d = _copy_bf16_into_out_part(
                    out_final_3d, ag_bf16_1, tok_offset=p0_tok, use_1way=True
                )
                out_final_3d = _copy_bf16_into_out_part(
                    out_final_3d, ag_bf16_2, tok_offset=p1_end, use_1way=True
                )
                return out_final_3d.reshape(ep_size * total_local_tok, hidden_size)

    # Support both global JAX SPMD arrays (bench_muse_spark_ep8.py) and
    # already rank-partitioned PyTorch jax_op inputs (fused_moe_rs_op.py).
    is_local_inputs = pre_proj_sharded.shape[1] == col_chunk
    proj_spec = P() if is_local_inputs else P(None, expert_axis)
    logits_spec = P() if is_local_inputs else P(expert_axis, None)
    weight_spec = P() if is_local_inputs else P(expert_axis)

    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(
            proj_spec,
            P(),
            logits_spec,
            P(),
            weight_spec,
            weight_spec,
            weight_spec if w1_scale is not None else None,
            weight_spec if w2_scale is not None else None,
            P(),
            weight_spec,
        ),
        out_specs=P(),
        check_vma=False,
    )(
        pre_proj_sharded,
        pre_norm_weight,
        router_logits_tok_sharded,
        e_score_bias,
        w1,
        w2,
        w1_scale,
        w2_scale,
        pnw_in,
        group_offset,
    )
