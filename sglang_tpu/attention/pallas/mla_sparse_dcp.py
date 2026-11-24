# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention with the cache write fused in, over a DCP-sharded cache.

The latent cache is page-striped over the DCP group with
``interleave_size == page_size``: global position ``p`` lives on rank
``(p // page_size) % dcp`` at local position
``(p // (page_size * dcp)) * page_size + p % page_size``. Every rank's page
table is the same (one physical page per ``page_size * dcp`` positions), so a
local position resolves through it exactly as an unsharded position would.
"""

import jax
import jax.numpy as jnp

from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils
from sglang_tpu.attention.pallas.mla_sparse.kernel import (
    DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    TENSORCORE_CACHE_LAYOUT,
    sparse_ragged_paged_attention,
)
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    NOPE_DIM,
    ROPE_DIM,
    _split_specs,
)


def compact_left(values: jax.Array, keep: jax.Array) -> jax.Array:
    """Stable compaction along axis 1: kept entries first, in order, then -1.

    Each kept entry moves left by the number of dropped entries before it.
    Applying the shift one bit at a time from the lowest never lands two kept
    entries on one slot (the shifts never decrease along the row), so every
    stage is a static slice and a select instead of a dynamic scatter or sort.
    """
    values = jnp.where(keep, values, -1)
    shift = jnp.cumsum(~keep, axis=1, dtype=jnp.int32)
    width = keep.shape[1]
    bit = 1
    while bit < width:
        moves = keep & ((shift & bit) != 0)

        def left(x, fill):
            pad = jnp.full(x.shape[:1] + (bit,) + x.shape[2:], fill, x.dtype)
            return jnp.concatenate([x[:, bit:], pad], axis=1)

        arrive = left(moves, False)
        stay = keep & ~moves
        values = jnp.where(
            arrive,
            left(values, -1),
            jnp.where(stay, values, -1),
        )
        shift = jnp.where(arrive, left(shift, 0), shift)
        keep = arrive | stay
        bit *= 2
    return values


def localize_topk(
    topk_indices: jax.Array,  # i32[T, K], global positions, -1 padded at the tail
    dcp_rank: jax.Array,  # i32 scalar
    *,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """This rank's share of every row, in local positions."""
    if interleave_size & (interleave_size - 1) == 0:
        interleave_shift = interleave_size.bit_length() - 1
        chunk = topk_indices >> interleave_shift
        offset = topk_indices & (interleave_size - 1)
    else:
        chunk = topk_indices // interleave_size
        offset = topk_indices % interleave_size

    if dcp_size & (dcp_size - 1) == 0:
        dcp_shift = dcp_size.bit_length() - 1
        owned = (topk_indices >= 0) & ((chunk & (dcp_size - 1)) == dcp_rank)
        local_chunk = chunk >> dcp_shift
    else:
        owned = (topk_indices >= 0) & (chunk % dcp_size == dcp_rank)
        local_chunk = chunk // dcp_size

    local = local_chunk * interleave_size + offset
    packed = compact_left(local, owned)
    return packed, jnp.any(owned, axis=1)


def sparse_mla_attention_dcp(
    ql_nope: jax.Array,  # bf16[max_num_tokens, dcp * local heads, 512]
    q_pe: jax.Array,  # bf16[max_num_tokens, dcp * local heads, 64]
    new_kv_c: jax.Array,  # fp8[max_num_tokens, 512]
    new_k_pe: jax.Array,  # fp8[max_num_tokens, 64]
    cache_nope: jax.Array,  # u8[num_pages, page_size, 4, 128], this rank's stripe
    cache_rope: jax.Array,  # u8[num_pages, page_size // 4, 4, 128], or a placeholder
    kv_lens: jax.Array,  # i32[max_num_seqs]; global, counts this step's tokens
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq], stripe pages
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    topk_indices: jax.Array,  # i32[max_num_tokens, K], global positions
    dcp_rank: jax.Array,  # i32[1]
    *,
    sm_scale: float,
    k_scale: float,
    dcp_size: int,
    gather_chunk_tokens: int = DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    zero_rope: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Store the step's owned tokens, then attend this rank's share of the top-k.

    Each rank contains same new latents but only writes the positions it owns.

    Returns ``(output, lse, cache_nope, cache_rope)``: the partial output
    ``[max_num_tokens, heads, 512]``, its natural-log LSE ``f32[max_num_tokens,
    heads]`` (``-inf`` on rows this rank owns none of), and the two buffers.
    """
    if new_kv_c.shape[-1] != NOPE_DIM or new_k_pe.shape[-1] != ROPE_DIM:
        raise ValueError(
            f"sparse MLA needs a {NOPE_DIM}-wide latent and a {ROPE_DIM}-wide rope, "
            f"got {new_kv_c.shape[-1]} and {new_k_pe.shape[-1]}."
        )
    num_pages, page_size = cache_nope.shape[:2]
    rank = dcp_rank.reshape(())
    nope_spec, rope_spec = _split_specs(num_pages, page_size)
    active_cu_q_lens = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])

    cache_nope, cache_rope = kv_cache_utils.update_sparse_mla_kv_cache_dcp(
        cache_nope,
        cache_rope,
        new_kv_c,
        new_k_pe,
        kv_lens,
        page_indices,
        active_cu_q_lens,
        rank,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        dcp_size=dcp_size,
        interleave_size=page_size,
        write_rope=not zero_rope,
    )

    local, owns = localize_topk(
        topk_indices, rank, dcp_size=dcp_size, interleave_size=page_size
    )

    local = local.at[:, 0].set(jnp.where(owns, local[:, 0], 0))
    out, lse = sparse_ragged_paged_attention(
        jnp.concatenate([ql_nope, q_pe], axis=-1),
        cache_nope,
        cache_rope,
        local,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        cache_layout=TENSORCORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=min(
            gather_chunk_tokens, max(64, ql_nope.shape[0])
        ),
        gather_rope=not zero_rope,
        return_lse=True,
    )
    lse = jnp.where(owns[:, None], lse, -jnp.inf)
    return out[..., :NOPE_DIM], lse, cache_nope, cache_rope
