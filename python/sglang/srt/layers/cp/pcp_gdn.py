# SPDX-License-Identifier: Apache-2.0
"""Drive the fused context-parallel Gated-DeltaNet kernel over a PCP mesh.

A linear-attention layer cannot run on a token shard: its recurrence
integrates along the token axis, so a rank holding every eighth chunk of the
sequence would produce a state that was never conditioned on the tokens in
between. Context parallelism therefore turns into *head* parallelism here --
each rank exchanges its token shard for a head shard, holding all tokens for
``1/pcp_size`` of the key and value heads, and runs the ordinary scan.

The exchange is driven from inside the Pallas pipeline rather than as separate
collectives, so the remote DMA for one stage overlaps the recurrence compute
of the previous one.

Weights arrive replicated and are sliced to this rank's head shard here;
``b``/``a`` are small enough to exchange outside the kernel.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sglang.srt.layers.attention.pallas.gdn import pcp_wrapper, pool_state


def reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jax.Array, split_sizes: list[int], n_shards: int, dim: int
):
    """
    Reorder a replicated concatenated tensor such that when sharded on multiple chips, each shard is a concatenation of the shards of the individual tensors.
    For example, let the concatenated_tensor be:
        AAAAAAAAAAAABBBBBBBBCCCC
            12 As     8 Bs  4 Cs
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is:
        AAABBCAAABBCAAABBCAAABBC
    In other words, it reorders the input tensor into 4 segments, with each segment corresponding to a shard and being AAABBC.
    Args:
        concatenated_tensor: the tensor, concatenated on the dimension specified by `dim`.
        split_sizes: each individual tensor's size on the dimension specified by `dim`.
        n_shards: num of shards.
        dim: the dimension on which the concatenated_tensor is concatenated.
    """
    # Split the concatenated tensor into individual tensors.
    if dim < 0:
        dim += concatenated_tensor.ndim
    split_tensors = []
    start_offset = 0
    old_shape = concatenated_tensor.shape
    # New shape ensures each split_tensor[i] maps to a tensor in ith shards
    new_shape = old_shape[:dim] + (n_shards, -1) + old_shape[dim + 1 :]
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(
            concatenated_tensor, start_offset, start_offset + split_size, axis=dim
        )
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += split_size
    # While maintaining 0th dim as a shard dim, we concatenate along 1th dim to
    # to create concatenated tnensor where 0th dim maps to shard dim.
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def _exchange_pcp_token_shards_for_head_shards(
    tensor: jnp.ndarray,
    pcp_axis: str,
    pcp_size: int,
    *,
    head_axis: int = -1,
) -> jnp.ndarray:
    """Exchange PCP token shards for PCP head shards.

    Callers hand this op rank-local token shards. The op-local PCP mesh then
    treats the group as extra head parallelism for the recurrent scan: each
    rank comes out holding every token for only its own head slice.
    """
    if head_axis < 0:
        head_axis += tensor.ndim
    assert tensor.shape[head_axis] % pcp_size == 0
    return jax.lax.all_to_all(
        tensor, axis_name=pcp_axis, split_axis=head_axis, concat_axis=0, tiled=True
    )


def _select_replicated_shard_for_pcp_rank(
    tensor: jnp.ndarray,
    pcp_axis: str,
    pcp_size: int,
    *,
    axis: int,
) -> jnp.ndarray:
    """Select this PCP rank shard without emitting axis_index/partitionId."""
    if axis < 0:
        axis += tensor.ndim
    assert tensor.shape[axis] % pcp_size == 0
    shard_size = tensor.shape[axis] // pcp_size
    # Every PCP rank starts with the same replicated full tensor. all_to_all
    # routes split ``rank`` to PCP rank ``rank``; because all inputs are equal,
    # each rank receives pcp_size copies of its own shard. Taking the first
    # shard with a constant slice avoids emitting a partitionId/axis_index op.
    exchanged = jax.lax.all_to_all(
        tensor, axis_name=pcp_axis, split_axis=axis, concat_axis=axis, tiled=True
    )
    return jax.lax.dynamic_slice_in_dim(exchanged, 0, shard_size, axis=axis)


def _derive_pcp_rank_major_reorder_indices(
    query_start_loc: jnp.ndarray,
    token_owner_starts: jnp.ndarray,
    *,
    pcp_size: int,
    interleave_size: int,
    local_padded_num_tokens: int,
) -> jnp.ndarray:
    """Rebuild packed-rank-major -> request-major token indices.

    ``token_owner_starts`` is the coordinate used only to assign query rows to
    PCP ranks.  It is deliberately independent from each request's absolute
    sequence position, which remains an input to the GDN state/conv path.
    """
    query_start_loc = jnp.asarray(query_start_loc, dtype=jnp.int32)
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be one-dimensional.")
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    token_owner_starts = jnp.asarray(token_owner_starts, dtype=jnp.int32)
    if q_lens.shape != token_owner_starts.shape:
        raise ValueError("q_lens and token_owner_starts must have equal shape.")
    req_count = q_lens.shape[0]
    token_owner_ends = token_owner_starts + q_lens

    rank_ids = jnp.arange(pcp_size, dtype=jnp.int32).reshape(pcp_size, 1)
    rank_counts = _pcp_rank_token_count_before(
        token_owner_ends.reshape(1, req_count),
        rank_ids,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ) - _pcp_rank_token_count_before(
        token_owner_starts.reshape(1, req_count),
        rank_ids,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    rank_req_ends = jnp.cumsum(rank_counts, axis=1)
    rank_req_starts = rank_req_ends - rank_counts

    full_padded_num_tokens = local_padded_num_tokens * pcp_size
    original_indices = jnp.arange(full_padded_num_tokens, dtype=jnp.int32)
    total_valid_tokens = query_start_loc[-1].astype(jnp.int32)
    original_valid = original_indices < total_valid_tokens
    request_ids = jnp.sum(
        original_indices[:, None] >= query_start_loc[1:].astype(jnp.int32)[None, :],
        axis=1,
    ).astype(jnp.int32)
    max_request_id = req_count - 1
    safe_request_ids = jnp.minimum(request_ids, max_request_id)
    original_offsets = original_indices - query_start_loc[safe_request_ids].astype(
        jnp.int32
    )
    original_owner_positions = token_owner_starts[safe_request_ids] + original_offsets

    cycle = pcp_size * interleave_size
    cycle_offsets = (
        original_owner_positions - (original_owner_positions // cycle) * cycle
    )
    original_ranks = (cycle_offsets // interleave_size).astype(jnp.int32)
    in_req_rank_offsets = _pcp_rank_token_count_before(
        original_owner_positions,
        original_ranks,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ) - _pcp_rank_token_count_before(
        token_owner_starts[safe_request_ids],
        original_ranks,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    rank_offsets = rank_req_starts[original_ranks, safe_request_ids]
    packed_destinations = (
        original_ranks * local_padded_num_tokens + rank_offsets + in_req_rank_offsets
    )

    sentinel = jnp.full_like(packed_destinations, full_padded_num_tokens)
    packed_destinations = jnp.where(original_valid, packed_destinations, sentinel)
    reorder_with_sentinel = jnp.zeros((full_padded_num_tokens + 1,), dtype=jnp.int32)
    reorder_values = jnp.where(
        original_valid, original_indices, jnp.zeros_like(original_indices)
    )
    reorder_with_sentinel = reorder_with_sentinel.at[packed_destinations].set(
        reorder_values
    )
    valid_with_sentinel = jnp.zeros((full_padded_num_tokens + 1,), dtype=jnp.bool_)
    valid_with_sentinel = valid_with_sentinel.at[packed_destinations].set(
        original_valid
    )
    reorder_indices = reorder_with_sentinel[:-1]
    valid_mask = valid_with_sentinel[:-1]
    return jnp.where(valid_mask, reorder_indices, -1).astype(jnp.int32)


def _pcp_rank_token_count_before(
    x: jnp.ndarray, ranks: jnp.ndarray, *, pcp_size: int, interleave_size: int
) -> jnp.ndarray:
    cycle = pcp_size * interleave_size
    full_cycles = x // cycle
    cycle_offsets = x - full_cycles * cycle
    rank_starts = ranks * interleave_size
    in_cycle = jnp.clip(cycle_offsets - rank_starts, min=0, max=interleave_size)
    return full_cycles * interleave_size + in_cycle


def run_fused_pcp_gdn_prefill(
    j_hidden_states: jnp.ndarray,
    j_qkvz_weight: jnp.ndarray,
    j_qkvz_weight_scale: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    pcp_size: int,
    interleave_size: int,
    mesh: jax.sharding.Mesh,
    qk_pair_layout: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fused QKVZ projection, PCP exchange, and GDN over the unified pool."""
    pcp_axis = "pcp"
    if pcp_axis not in mesh.axis_names:
        raise NotImplementedError("GDN pooled PCP prefill requires a pcp mesh axis.")
    if mesh.shape[pcp_axis] != pcp_size:
        raise ValueError(
            f"pcp_size={pcp_size} does not match mesh axis size "
            f"{mesh.shape[pcp_axis]}."
        )
    if n_kq % pcp_size != 0:
        raise ValueError(f"n_kq={n_kq} must be divisible by pcp_size={pcp_size}.")
    if n_v % pcp_size != 0:
        raise ValueError(f"n_v={n_v} must be divisible by pcp_size={pcp_size}.")

    local_n_kq = n_kq // pcp_size
    local_n_v = n_v // pcp_size
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    local_conv_dim = 2 * local_n_kq * d_k + local_n_v * d_v

    token_spec = P(pcp_axis, None)
    replicated_spec = P()
    pool_spec = P(pcp_axis)
    in_specs = (
        token_spec,  # hidden states
        replicated_spec,  # QKVZ weight
        replicated_spec,  # QKVZ scale
        token_spec,  # b
        token_spec,  # a
        pool_spec,  # rank-local unified pool
        replicated_spec,  # conv weight
        replicated_spec if j_conv_bias is not None else None,
        replicated_spec,  # A_log
        replicated_spec,  # dt_bias
        replicated_spec,  # query_start_loc
        replicated_spec,  # state_indices
        replicated_spec,  # distribution
        replicated_spec,  # seq_lens
    )
    output_spec = P(pcp_axis, None, None)
    out_specs = (pool_spec, output_spec, output_spec)

    def _pooled_pcp_prefill_fn(
        local_hidden_states,
        qkvz_weight_,
        qkvz_weight_scale_,
        local_b,
        local_a,
        pool_,
        conv_weight_,
        conv_bias_,
        A_log_,
        dt_bias_,
        query_start_loc_,
        state_indices_,
        distribution_,
        seq_lens_,
    ):
        # BA is small enough to exchange outside the fused kernel. Restore it
        # to request-major order before the GDN schedule consumes it.
        local_ba = jnp.stack((local_b, local_a), axis=-1).reshape(local_b.shape[0], -1)
        packed_ba_shard = _exchange_pcp_token_shards_for_head_shards(
            local_ba, pcp_axis, pcp_size
        )
        # PCP ownership follows the batch-flat request-major coordinate.
        # State semantics remain request-absolute through ``seq_lens_`` below.
        token_owner_starts = query_start_loc_[:-1]
        full_reorder = _derive_pcp_rank_major_reorder_indices(
            query_start_loc_,
            token_owner_starts,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_num_tokens=local_hidden_states.shape[0],
        )
        valid_mask = full_reorder >= 0
        scatter_indices = jnp.where(valid_mask, full_reorder, full_reorder.size)
        ba_shard = (
            jnp.zeros_like(packed_ba_shard)
            .at[scatter_indices]
            .set(packed_ba_shard, mode="drop")
        )
        ba_shard = ba_shard.reshape(ba_shard.shape[0], local_n_v, 2)
        b_shard = ba_shard[:, :, 0]
        a_shard = ba_shard[:, :, 1]

        conv_weight_interleaved = reorder_concatenated_tensor_for_sharding(
            conv_weight_,
            [key_dim, key_dim, value_dim],
            pcp_size,
            0,
        )
        weight_shard = _select_replicated_shard_for_pcp_rank(
            conv_weight_interleaved, pcp_axis, pcp_size, axis=0
        )
        if conv_bias_ is None:
            bias_shard = None
        else:
            conv_bias_interleaved = reorder_concatenated_tensor_for_sharding(
                conv_bias_,
                [key_dim, key_dim, value_dim],
                pcp_size,
                0,
            )
            bias_shard = _select_replicated_shard_for_pcp_rank(
                conv_bias_interleaved, pcp_axis, pcp_size, axis=0
            )
        A_shard = _select_replicated_shard_for_pcp_rank(
            A_log_, pcp_axis, pcp_size, axis=0
        )
        dt_shard = _select_replicated_shard_for_pcp_rank(
            dt_bias_, pcp_axis, pcp_size, axis=0
        )

        if weight_shard.shape[0] != local_conv_dim:
            raise ValueError(
                "GDN PCP conv weights must be PCP-local after "
                f"selection: expected {local_conv_dim}, got "
                f"{weight_shard.shape[0]}."
            )
        state_plan = pool_state.build_pool_state_plan(
            pool_,
            conv_dim=local_conv_dim,
            n_v=local_n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=pool_block_tokens,
            qk_pair_layout=qk_pair_layout,
        )

        return pcp_wrapper.fused_qkvz_projection_pcp_gdn(
            local_hidden_states,
            qkvz_weight_,
            qkvz_weight_scale_,
            b_shard,
            a_shard,
            pool_,
            weight_shard,
            bias_shard,
            A_shard,
            dt_shard,
            query_start_loc_,
            state_indices_,
            distribution_,
            seq_lens_,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pcp_size=pcp_size,
            comm_chunk_size=interleave_size,
            mesh_axis_names=tuple(mesh.axis_names),
            pcp_axis_name=pcp_axis,
            state_plan=state_plan,
        )

    mapped_fn = jax.shard_map(
        _pooled_pcp_prefill_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return mapped_fn(
        j_hidden_states,
        j_qkvz_weight,
        j_qkvz_weight_scale,
        j_b,
        j_a,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )
