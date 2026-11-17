"""Paged-cache Pallas scorers for DSA decode and extend query blocks."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def _cache_keys(keys, dtype):
    """Read the existing 256-byte Torch index record after its bounded page DMA.

    Byte 0:128 is E4M3; byte 128 is UE8M0. No whole-cache conversion or
    additional HBM key tensor is introduced. The source page prefetch remains.
    """
    if keys.dtype != jnp.uint8:
        return keys.astype(dtype)
    fp8 = lax.bitcast_convert_type(keys[..., :128], jnp.float8_e4m3fn)
    scale = lax.bitcast_convert_type(keys[..., 128].astype(jnp.int32) << 23, jnp.float32)
    return (fp8.astype(jnp.float32) * scale[..., None]).astype(dtype)


_NEG_INF = float("-inf")
_DEFAULT_BLOCK_K = 22528
_DEFAULT_EXTEND_BLOCK_K = 2048
_DECODE_SEQS_PER_GROUP = 2

# Keep only kernel-level source attribution; inner instruction details come from
# XProf custom-call/LLO profiling. These scopes add no timers or barriers.


def _resolve_block_k(max_kv: int, page_size: int, requested_block_k: int) -> int:
    """Choose the largest TPU-aligned divisor at or below the requested tile."""
    candidate = min(max_kv, requested_block_k)
    candidate -= candidate % 128
    while candidate >= 128:
        if max_kv % candidate == 0 and candidate % page_size == 0:
            return candidate
        candidate -= 128
    raise ValueError(
        f"cannot tile max_kv={max_kv} with page_size={page_size} into 128-aligned key blocks"
    )


def _weighted_relu_scores(
    query: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    *,
    batched_keys: bool,
) -> jax.Array:
    """Apply the common DSA score formula for one resident key tile.

    ``query`` is either ``[H, D]`` or ``[Bq, H, D]``. Shared keys have shape
    ``[Bk, D]``; batched keys have shape ``[Bq, Bk, D]``. Keeping this
    mathematical core shared lets decode and extend specialize only their page
    DMA and launch schedules.
    """

    if query.ndim == 2:
        if batched_keys or keys.ndim != 2 or weights.ndim != 1:
            raise ValueError(
                "a rank-2 query requires rank-2 shared keys and rank-1 weights, "
                f"got query={query.shape}, keys={keys.shape}, weights={weights.shape}"
            )
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return lax.dot_general(
            weights.astype(jnp.float32),
            jnp.maximum(similarities, jnp.float32(0.0)),
            dimension_numbers=(((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    if query.ndim != 3 or weights.ndim != 2:
        raise ValueError(
            "a query block requires query [Bq, H, D] and weights [Bq, H], "
            f"got query={query.shape}, weights={weights.shape}"
        )
    if batched_keys:
        if keys.ndim != 3 or keys.shape[0] != query.shape[0]:
            raise ValueError(
                "batched keys must have shape [Bq, Bk, D], "
                f"got query={query.shape}, keys={keys.shape}"
            )
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
    else:
        if keys.ndim != 2:
            raise ValueError(f"shared keys must have shape [Bk, D], got {keys.shape}")
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((2,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    return lax.dot_general(
        weights.astype(jnp.float32),
        jnp.maximum(similarities, jnp.float32(0.0)),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _weighted_relu_scores_packed_sequences(
    query: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Score every sequence/query row in one batched MXU operation.

    ``query`` is ``[S, R, H, D]``, ``weights`` is ``[S, R, H]``, and each
    sequence has its own resident key tile ``[S, Bk, D]``.  The explicit
    sequence batch axis lets Mosaic lower all ``S * R`` query rows together
    without slicing the tiled query dimension at sequence boundaries.
    """

    if query.ndim != 4 or weights.shape != query.shape[:3]:
        raise ValueError(
            "packed queries require query [S, R, H, D] and weights [S, R, H], "
            f"got query={query.shape}, weights={weights.shape}"
        )
    if keys.ndim != 3 or keys.shape[0] != query.shape[0] or keys.shape[2] != query.shape[3]:
        raise ValueError(
            "packed keys must have shape [S, Bk, D] matching the queries, "
            f"got query={query.shape}, keys={keys.shape}"
        )
    similarities = lax.dot_general(
        query,
        keys,
        dimension_numbers=(((3,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    num_sequences, rows_per_sequence, num_heads = weights.shape
    packed_rows = num_sequences * rows_per_sequence
    packed_weights = weights.reshape(packed_rows, num_heads)
    packed_similarities = similarities.reshape(packed_rows, num_heads, similarities.shape[-1])
    scores = lax.dot_general(
        packed_weights.astype(jnp.float32),
        jnp.maximum(packed_similarities, jnp.float32(0.0)),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    return scores.reshape(num_sequences, rows_per_sequence, similarities.shape[-1])


@jax.named_scope("DecodeSingleScore")
def _paged_decode_score_kernel(
    seq_lens_ref,  # i32[S], SMEM
    page_indices_ref,  # i32[S * pages_per_seq], SMEM
    cu_kv_lens_ref,  # i32[S + 1], SMEM
    q_vmem_ref,  # f[1, H, D], VMEM
    weights_vmem_ref,  # f[1, 1, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[1, 1, max_kv], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one sequence while double-buffering paged key DMA."""
    seq_id = pl.program_id(0)
    block_k = pages_per_block * page_size if coalesce_page_dma else keys_x2_vmem_ref.shape[1]
    page_table_start = cu_kv_lens_ref[seq_id] // page_size
    kv_len = seq_lens_ref[seq_id]
    active_blocks = jnp.clip(pl.cdiv(kv_len, block_k), 0, num_k_blocks)

    def fetch_key_block(block_id, buffer, *, wait: bool):
        dst = keys_x2_vmem_ref.at[buffer]
        sem = key_dma_sems.at[buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        first_page = page_table_start + block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=4)

    # Prologue: seed the first key buffer. Query and head weights are already
    # resident in per-program VMEM through their BlockSpecs.
    @pl.when(active_blocks > 0)
    def start_fetch():
        fetch_key_block(0, 0, wait=False)

    def score_block(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer

        fetch_key_block(block_id, buffer, wait=True)

        @pl.when(block_id + 1 < active_blocks)
        def prefetch_next_keys():
            fetch_key_block(block_id + 1, next_buffer, wait=False)

        # Mosaic's matmul lowering does not reliably support mixed FP32/BF16
        # operands.  The production query is FP32 after the Hadamard transform,
        # while the paged index-key cache is BF16, so make the promotion that
        # jnp.einsum applies in the reference path explicit here.
        if first_dot_bf16:
            # Performance ablation: keep both MXU operands in BF16 while
            # accumulating into FP32. This avoids promoting the complete
            # [block_k, D] key tile to FP32, but may change candidates near the
            # top-k boundary when the production query arrives in FP32.
            query = q_vmem_ref[0].astype(jnp.bfloat16)
            keys = keys_x2_vmem_ref[buffer]
            if coalesce_page_dma:
                keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
            keys = _cache_keys(keys, jnp.bfloat16)
        else:
            query = q_vmem_ref[0]
            keys = keys_x2_vmem_ref[buffer]
            if coalesce_page_dma:
                keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
            keys = _cache_keys(keys, query.dtype)
        # Match the two reference einsums exactly. A plain FP32 multiply plus
        # reduce for the head mixture uses a different TPU numerical path and
        # can perturb candidates around the top-k boundary.
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[0, 0].astype(jnp.float32),
            keys,
            batched_keys=False,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        scores_out_vmem_ref[0, 0, pl.ds(block_id * block_k, block_k)] = jnp.where(
            positions < kv_len,
            scores,
            _NEG_INF,
        )
        return None

    lax.fori_loop(0, active_blocks, score_block, None, unroll=False)

    def clear_tail(block_id, _):
        scores_out_vmem_ref[0, 0, pl.ds(block_id * block_k, block_k)] = jnp.full(
            (block_k,), _NEG_INF, jnp.float32
        )

    lax.fori_loop(active_blocks, num_k_blocks, clear_tail, None, unroll=False)


@jax.named_scope("DecodePairScore")
def _paged_decode_score_two_seq_kernel(
    seq_lens_ref,  # i32[S], SMEM
    page_indices_ref,  # i32[S * pages_per_seq], SMEM
    cu_kv_lens_ref,  # i32[S + 1], SMEM
    q_vmem_ref,  # f[2, H, D], VMEM
    weights_vmem_ref,  # f[2, 1, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[2, 1, max_kv], VMEM
    keys_vmem_ref,  # flat or page-shaped two-sequence ping-pong buffers
    key_dma_sems,  # DMA semaphore [2 sequences, 2 buffers]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one pair of sequences with a shared prefetch schedule."""
    seq_base = pl.program_id(0) * _DECODE_SEQS_PER_GROUP
    block_k = pages_per_block * page_size if coalesce_page_dma else keys_vmem_ref.shape[2]

    max_length = jnp.maximum(seq_lens_ref[seq_base], seq_lens_ref[seq_base + 1])
    active_blocks = jnp.clip(pl.cdiv(max_length, block_k), 0, num_k_blocks)

    def fetch_key_block(seq_id, block_id, buffer, *, wait: bool):
        dst = keys_vmem_ref.at[seq_id, buffer]
        sem = key_dma_sems.at[seq_id, buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        global_seq_id = seq_base + seq_id
        page_table_start = cu_kv_lens_ref[global_seq_id] // page_size
        first_page = page_table_start + block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=4)

    def score_key_block_pair(block_id, buffer):
        if first_dot_bf16:
            query = q_vmem_ref[...].astype(jnp.bfloat16)
            keys = keys_vmem_ref[:, buffer]
            if coalesce_page_dma:
                keys = keys.reshape(2, block_k, keys_vmem_ref.shape[-1])
            keys = _cache_keys(keys, jnp.bfloat16)
        else:
            query = q_vmem_ref[...]
            keys = keys_vmem_ref[:, buffer]
            if coalesce_page_dma:
                keys = keys.reshape(2, block_k, keys_vmem_ref.shape[-1])
            keys = _cache_keys(keys, query.dtype)
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[:, 0].astype(jnp.float32),
            keys,
            batched_keys=True,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        for seq_id in range(2):
            scores_out_vmem_ref[
                seq_id,
                0,
                pl.ds(block_id * block_k, block_k),
            ] = jnp.where(
                positions < seq_lens_ref[seq_base + seq_id],
                scores[seq_id],
                _NEG_INF,
            )

    # Seed both current buffers. Both page streams can make progress before
    # either sequence reaches its first wait.
    @pl.when(active_blocks > 0)
    def start_fetch_pair():
        for seq_id in range(2):
            fetch_key_block(seq_id, 0, 0, wait=False)

    def score_block_pair(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer

        # Make both key blocks resident, then present both sequences to Mosaic
        # as one pair of batched matrix products. This gives the compiler an
        # explicit batch axis instead of two sequential H=32 dot operations.
        for seq_id in range(2):
            fetch_key_block(seq_id, block_id, buffer, wait=True)

            @pl.when(block_id + 1 < active_blocks)
            def prefetch_next_keys(seq_id=seq_id):
                fetch_key_block(seq_id, block_id + 1, next_buffer, wait=False)

        score_key_block_pair(block_id, buffer)
        return None

    lax.fori_loop(0, active_blocks, score_block_pair, None, unroll=False)

    def clear_tail(block_id, _):
        for seq_id in range(2):
            scores_out_vmem_ref[seq_id, 0, pl.ds(block_id * block_k, block_k)] = jnp.full(
                (block_k,), _NEG_INF, jnp.float32
            )

    lax.fori_loop(active_blocks, num_k_blocks, clear_tail, None, unroll=False)


@jax.named_scope("ExtendBlockScore")
def _paged_extend_score_block_kernel(
    valid_lengths_ref,  # i32[Bq], SMEM
    page_indices_ref,  # i32[pages_per_seq], SMEM
    q_vmem_ref,  # f[Bq, H, D], VMEM
    weights_vmem_ref,  # f[Bq, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[Bq, max_kv], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one extend query block against one shared paged key stream."""

    block_k = pages_per_block * page_size if coalesce_page_dma else keys_x2_vmem_ref.shape[1]

    active_blocks = num_k_blocks
    if q_vmem_ref.shape[0] == 32:
        max_length = jnp.int32(0)
        for row in range(32):
            max_length = jnp.maximum(max_length, valid_lengths_ref[row])
        active_blocks = jnp.clip(pl.cdiv(max_length, block_k), 0, num_k_blocks)

    def fetch_key_block(block_id, buffer, *, wait: bool):
        dst = keys_x2_vmem_ref.at[buffer]
        sem = key_dma_sems.at[buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        first_page = block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(
                0,
                pages_per_block,
                fetch_page,
                None,
                unroll=4 if q_vmem_ref.shape[0] == 32 else False,
            )

    @pl.when(active_blocks > 0)
    def start_fetch():
        fetch_key_block(0, 0, wait=False)

    def score_block(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer
        fetch_key_block(block_id, buffer, wait=True)

        @pl.when(block_id + 1 < active_blocks)
        def prefetch_next_keys():
            fetch_key_block(block_id + 1, next_buffer, wait=False)

        keys = keys_x2_vmem_ref[buffer]
        if coalesce_page_dma:
            keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
        if first_dot_bf16:
            query = q_vmem_ref[...].astype(jnp.bfloat16)
            keys = _cache_keys(keys, jnp.bfloat16)
        else:
            query = q_vmem_ref[...]
            keys = _cache_keys(keys, query.dtype)
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[...],
            keys,
            batched_keys=False,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        # Mosaic SMEM supports scalar loads only. Keep score construction
        # batched, then use static row stores so each causal length is loaded
        # as one scalar instead of materializing valid_lengths[Bq] in VMEM.
        for query_id in range(q_vmem_ref.shape[0]):
            valid = positions < valid_lengths_ref[query_id]
            scores_out_vmem_ref[query_id, pl.ds(block_id * block_k, block_k)] = jnp.where(
                valid,
                scores[query_id],
                _NEG_INF,
            )
        return None

    lax.fori_loop(0, active_blocks, score_block, None, unroll=False)
    if q_vmem_ref.shape[0] == 32:

        def clear_tail(block_id, _):
            for row in range(32):
                scores_out_vmem_ref[row, pl.ds(block_id * block_k, block_k)] = jnp.full(
                    (block_k,), _NEG_INF, jnp.float32
                )

        lax.fori_loop(active_blocks, num_k_blocks, clear_tail, None, unroll=False)


@jax.named_scope("ExtendPackedScore")
def _paged_extend_score_uniform_packed_kernel(
    valid_lengths_ref,  # i32[S * R], SMEM
    page_indices_ref,  # i32[N_pages], SMEM
    cu_kv_lens_ref,  # i32[S + 1], SMEM
    q_vmem_ref,  # f[S * R, H, D], VMEM
    weights_vmem_ref,  # f[S * R, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[S * R, max_kv], VMEM
    keys_vmem_ref,  # f[S, 2, block_k, D] or page-shaped equivalent
    key_dma_sems,
    *,
    num_sequences: int,
    rows_per_sequence: int,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
    dma_unroll: int,
    num_buffers: int,
):
    """Score all uniform query chains as one batched query block."""

    block_k = pages_per_block * page_size if coalesce_page_dma else keys_vmem_ref.shape[2]

    max_length = jnp.int32(0)
    for row in range(num_sequences * rows_per_sequence):
        max_length = jnp.maximum(max_length, valid_lengths_ref[row])
    active_blocks = jnp.clip(pl.cdiv(max_length, block_k), 0, num_k_blocks)

    def fetch_key_block(seq_id, block_id, buffer, *, wait: bool):
        dst = keys_vmem_ref.at[seq_id, buffer]
        sem = key_dma_sems.at[seq_id, buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        page_start = cu_kv_lens_ref[seq_id] // page_size
        first_page = page_start + block_id * pages_per_block
        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
            return

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(cache_hbm_ref.at[physical_page], page_dst, sem).start()
            return None

        lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=dma_unroll)

    for initial_block in range(num_buffers - 1):

        @pl.when(initial_block < active_blocks)
        def start_fetch(initial_block=initial_block):
            for seq_id in range(num_sequences):
                fetch_key_block(seq_id, initial_block, initial_block, wait=False)

    def score_block(block_id, _):
        buffer = block_id % num_buffers
        prefetch_block = block_id + num_buffers - 1
        next_buffer = prefetch_block % num_buffers
        for seq_id in range(num_sequences):
            fetch_key_block(seq_id, block_id, buffer, wait=True)

            @pl.when(prefetch_block < active_blocks)
            def prefetch_next_keys(seq_id=seq_id):
                fetch_key_block(seq_id, prefetch_block, next_buffer, wait=False)

        query = q_vmem_ref[...].reshape(
            num_sequences, rows_per_sequence, q_vmem_ref.shape[1], q_vmem_ref.shape[2]
        )
        weights = weights_vmem_ref[...].reshape(
            num_sequences, rows_per_sequence, weights_vmem_ref.shape[1]
        )
        keys = keys_vmem_ref[:, buffer]
        if coalesce_page_dma:
            keys = keys.reshape(num_sequences, block_k, keys_vmem_ref.shape[-1])
        if first_dot_bf16:
            query = query.astype(jnp.bfloat16)
            keys = _cache_keys(keys, jnp.bfloat16)
        else:
            keys = _cache_keys(keys, query.dtype)
        scores = _weighted_relu_scores_packed_sequences(query, weights, keys)
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        # Prefetched scalar inputs live in SMEM, where Mosaic only permits
        # scalar loads. Materialize the small (at most 32-row) mask from
        # statically indexed scalars while keeping score computation packed.
        packed_valid_lengths = jnp.stack(
            [valid_lengths_ref[row] for row in range(num_sequences * rows_per_sequence)]
        )
        packed_scores = scores.reshape(num_sequences * rows_per_sequence, block_k)
        packed_scores = jnp.where(
            positions[None, :] < packed_valid_lengths[:, None],
            packed_scores,
            _NEG_INF,
        )
        scores_out_vmem_ref[:, pl.ds(block_id * block_k, block_k)] = packed_scores
        return None

    lax.fori_loop(0, active_blocks, score_block, None, unroll=False)

    def clear_tail(block_id, _):
        scores_out_vmem_ref[:, pl.ds(block_id * block_k, block_k)] = jnp.full(
            (num_sequences * rows_per_sequence, block_k), _NEG_INF, jnp.float32
        )

    lax.fori_loop(active_blocks, num_k_blocks, clear_tail, None, unroll=False)


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def _paged_extend_score_block_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    valid_lengths: jax.Array,
    page_indices: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score a fixed query block directly from one sequence's page table."""

    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [Bq, H, D], got {q_idx.shape}")
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if index_key_cache.ndim != 3:
        raise ValueError(
            f"index_key_cache must have shape [P, page_size, D], got {index_key_cache.shape}"
        )
    num_queries, _, head_dim = q_idx.shape
    if valid_lengths.shape != (num_queries,):
        raise ValueError(
            f"valid_lengths must have shape {(num_queries,)}, got {valid_lengths.shape}"
        )
    if page_indices.shape != (pages_per_seq,):
        raise ValueError(
            f"page_indices must have shape {(pages_per_seq,)}, got {page_indices.shape}"
        )
    if index_key_cache.shape[-1] != (256 if index_key_cache.dtype == jnp.uint8 else head_dim):
        raise ValueError(
            f"cache head dim {index_key_cache.shape[-1]} does not match q head dim {head_dim}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")
    for name, value in (("valid_lengths", valid_lengths), ("page_indices", page_indices)):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128 or block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k

    key_shape = (
        (2, pages_per_block, page_size, index_key_cache.shape[-1]) if coalesce_page_dma else (2, block_k, index_key_cache.shape[-1])
    )
    scores = pl.pallas_call(
        functools.partial(
            _paged_extend_score_block_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        ),
        out_shape=jax.ShapeDtypeStruct((num_queries, max_kv), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(q_idx.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(idx_weights.shape, lambda *_: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (num_queries, max_kv),
                lambda *_: (0, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM(key_shape, index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            # Long-context score tiles need more than the default 32 MiB
            # scoped reservation. The indexer caps FP32 score tiles at 32 MiB.
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
        name=(
            "dsa_paged_extend_score_block_contiguous"
            if coalesce_page_dma
            else "dsa_paged_extend_score_block_paged"
        ),
    )(
        valid_lengths,
        page_indices,
        q_idx,
        idx_weights,
        index_key_cache,
    )
    return scores


def _paged_extend_score_uniform_packed_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    valid_lengths: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    num_sequences: int,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
    dma_unroll: int = 4,
    num_buffers: int = 2,
) -> jax.Array:
    """Score equal-length packed query chains as one small query block."""

    if dma_unroll not in (1, 2, 4, 8) or num_buffers not in (2, 3):
        raise ValueError("packed score requires dma_unroll in (1,2,4,8) and num_buffers in (2,3)")
    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [T, H, D], got {q_idx.shape}")
    num_queries, _, head_dim = q_idx.shape
    if num_sequences < 1 or num_queries % num_sequences:
        raise ValueError(
            f"num_queries={num_queries} must be divisible by num_sequences={num_sequences}"
        )
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if valid_lengths.shape != (num_queries,):
        raise ValueError(
            f"valid_lengths must have shape {(num_queries,)}, got {valid_lengths.shape}"
        )
    if cu_kv_lens.shape != (num_sequences + 1,):
        raise ValueError(
            f"cu_kv_lens must have shape {(num_sequences + 1,)}, got {cu_kv_lens.shape}"
        )
    if page_indices.ndim != 1:
        raise ValueError(f"page_indices must be rank 1, got {page_indices.shape}")
    if index_key_cache.ndim != 3 or index_key_cache.shape[-1] != (256 if index_key_cache.dtype == jnp.uint8 else head_dim):
        raise ValueError(
            "index_key_cache must have shape [P, page_size, D] matching q_idx, "
            f"got {index_key_cache.shape} and {q_idx.shape}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")
    for name, value in (
        ("valid_lengths", valid_lengths),
        ("page_indices", page_indices),
        ("cu_kv_lens", cu_kv_lens),
    ):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128 or block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k
    rows_per_sequence = num_queries // num_sequences
    key_shape = (
        (num_sequences, num_buffers, pages_per_block, page_size, index_key_cache.shape[-1])
        if coalesce_page_dma
        else (num_sequences, num_buffers, block_k, index_key_cache.shape[-1])
    )

    scores = pl.pallas_call(
        functools.partial(
            _paged_extend_score_uniform_packed_kernel,
            num_sequences=num_sequences,
            rows_per_sequence=rows_per_sequence,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
            dma_unroll=dma_unroll,
            num_buffers=num_buffers,
        ),
        out_shape=jax.ShapeDtypeStruct((num_queries, max_kv), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(q_idx.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(idx_weights.shape, lambda *_: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((num_queries, max_kv), lambda *_: (0, 0)),
            scratch_shapes=[
                pltpu.VMEM(key_shape, index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((num_sequences, num_buffers)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
        name="dsa_paged_extend_score_uniform_packed",
    )(
        valid_lengths,
        page_indices,
        cu_kv_lens,
        q_idx,
        idx_weights,
        index_key_cache,
    )
    return scores


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_sequences",
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
        "dma_unroll",
        "num_buffers",
    ),
)
def paged_extend_score_uniform_packed_pallas(
    q_idx,
    idx_weights,
    index_key_cache,
    valid_lengths,
    page_indices,
    cu_kv_lens,
    *,
    num_sequences: int,
    pages_per_seq: int,
    block_k: int | None = None,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
    dma_unroll: int = 4,
    num_buffers: int = 2,
):
    """Score all packed rows together, coalescing only verified contiguous pages."""
    # Measured on v7x for two six-token MTP chains near a 128K context.
    # Keep the prior schedule for other shapes, precision modes, and short contexts.
    use_tuned_shape = (
        block_k is None
        and not interpret
        and not first_dot_bf16
        and dma_unroll == 4
        and num_buffers == 2
        and num_sequences == 2
        and q_idx.shape == (12, 32, 128)
        and q_idx.dtype == jnp.float32
        and idx_weights.dtype == jnp.float32
        and index_key_cache.dtype == jnp.bfloat16
        and index_key_cache.shape[1] == 64
        and pages_per_seq == 2112
        and jax.default_backend() == "tpu"
        and jax.devices()[0].device_kind in ("TPU7x", "TPU v7", "TPU v7x")
    )
    if block_k is None:
        block_k = _DEFAULT_EXTEND_BLOCK_K
    args = (q_idx, idx_weights, index_key_cache, valid_lengths, page_indices, cu_kv_lens)
    kwargs = dict(
        num_sequences=num_sequences,
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        interpret=interpret,
        dma_unroll=dma_unroll,
        num_buffers=num_buffers,
    )

    def score_for_layout(contiguous_page_table):
        def score_with_block(tile):
            return _paged_extend_score_uniform_packed_pallas_impl(
                *args, coalesce_page_dma=contiguous_page_table, **{**kwargs, "block_k": tile}
            )

        if use_tuned_shape:
            return lax.cond(
                jnp.max(valid_lengths) >= 131072,
                lambda: score_with_block(11264),
                lambda: score_with_block(block_k),
            )
        return score_with_block(block_k)

    if not coalesce_page_dma:
        return score_for_layout(False)
    page_size = index_key_cache.shape[1]
    starts = cu_kv_lens[:num_sequences] // page_size
    offsets = jnp.arange(pages_per_seq, dtype=jnp.int32)
    # Check each sequence independently: adjacent sequences need not be adjacent
    # in physical memory. A failed check retains the exact paged DMA path.
    slots = starts[:, None] + offsets[None, :]
    pages = page_indices[jnp.clip(slots, 0, page_indices.shape[0] - 1)]
    in_bounds = (starts >= 0) & (starts + pages_per_seq <= page_indices.shape[0])
    contiguous = jnp.all(in_bounds) & jnp.all(pages == pages[:, :1] + offsets[None, :])
    contiguous &= jnp.all((pages[:, :1] >= 0) & (pages[:, -1:] < index_key_cache.shape[0]))
    return lax.cond(
        contiguous,
        lambda: score_for_layout(True),
        lambda: score_for_layout(False),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def paged_extend_score_block_pallas(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    valid_lengths: jax.Array,
    page_indices: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score one extend query block against a shared paged key stream.

    This is the general page-aware score schedule used by extend. Decode keeps
    its persistent two-sequence schedule because its query rows use different
    page tables; both schedules call the same weighted-ReLU score core.
    """

    kwargs = dict(
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        interpret=interpret,
    )
    if not coalesce_page_dma:
        return _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=False,
            **kwargs,
        )

    expected_pages = page_indices[:1] + jnp.arange(pages_per_seq, dtype=jnp.int32)
    page_table_in_bounds = jnp.all((page_indices >= 0) & (page_indices < index_key_cache.shape[0]))
    contiguous_page_table = page_table_in_bounds & jnp.all(page_indices == expected_pages)
    return jax.lax.cond(
        contiguous_page_table,
        lambda: _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=True,
            **kwargs,
        ),
        lambda: _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=False,
            **kwargs,
        ),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "persistent_two_seq",
        "coalesce_page_dma",
        "interpret",
    ),
)
def _paged_decode_scores_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_BLOCK_K,
    first_dot_bf16: bool = False,
    persistent_two_seq: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Compute decode index scores directly from a paged key cache.

    ``coalesce_page_dma=True`` is an internal fast-path assertion: callers
    must verify that every active sequence page table is contiguous.

    Query row ``i`` belongs to sequence ``i``. The kernel never constructs a
    contiguous ``[T, max_kv, D]`` or ``[max_kv, D]`` gathered-key array; it
    follows ``page_indices`` while double-buffering page-aligned key blocks in
    VMEM and writes only the FP32 ``[T, max_kv]`` score matrix to HBM. The
    default block is tuned for the 135168-position v7x bucket and is reduced to
    the largest compatible divisor for other score shapes.
    """
    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [T, H, D], got {q_idx.shape}")
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if index_key_cache.ndim != 3:
        raise ValueError(
            f"index_key_cache must have shape [P, page_size, D], got {index_key_cache.shape}"
        )
    num_queries, _, head_dim = q_idx.shape
    if seq_lens.shape != (num_queries,):
        raise ValueError(
            "decode scorer requires one query per sequence, "
            f"got q_idx.shape[0]={num_queries} and seq_lens.shape={seq_lens.shape}"
        )
    if index_key_cache.shape[-1] != (256 if index_key_cache.dtype == jnp.uint8 else head_dim):
        raise ValueError(
            f"cache head dim {index_key_cache.shape[-1]} does not match q head dim {head_dim}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128:
        raise ValueError(f"block_k must be at least 128, got {block_k}")
    if block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    for name, value in (
        ("seq_lens", seq_lens),
        ("page_indices", page_indices),
        ("cu_kv_lens", cu_kv_lens),
        ("distribution", distribution),
    ):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k
    active_num_seqs = jnp.clip(distribution[2], 0, num_queries)
    scalar_prefetches = (seq_lens, page_indices, cu_kv_lens)
    idx_weights_3d = idx_weights[:, None, :]
    use_persistent_two_seq = (
        persistent_two_seq
        and num_queries >= _DECODE_SEQS_PER_GROUP
        and num_queries % _DECODE_SEQS_PER_GROUP == 0
    )
    if use_persistent_two_seq:
        kernel = functools.partial(
            _paged_decode_score_two_seq_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        )
        grid = (num_queries // _DECODE_SEQS_PER_GROUP,)
        q_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, *q_idx.shape[1:]),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        weights_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, 1, idx_weights.shape[1]),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        score_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, 1, max_kv),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        persistent_key_shape = (
            (
                _DECODE_SEQS_PER_GROUP,
                2,
                pages_per_block,
                page_size,
                index_key_cache.shape[-1],
            )
            if coalesce_page_dma
            else (_DECODE_SEQS_PER_GROUP, 2, block_k, index_key_cache.shape[-1])
        )
        scratch_shapes = [
            pltpu.VMEM(persistent_key_shape, index_key_cache.dtype),
            pltpu.SemaphoreType.DMA((_DECODE_SEQS_PER_GROUP, 2)),
        ]
    else:
        kernel = functools.partial(
            _paged_decode_score_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        )
        grid = (active_num_seqs,)
        q_spec = pl.BlockSpec(
            (1, *q_idx.shape[1:]),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        weights_spec = pl.BlockSpec(
            (1, 1, idx_weights.shape[1]),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        score_spec = pl.BlockSpec(
            (1, 1, max_kv),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        key_shape = (
            (2, pages_per_block, page_size, index_key_cache.shape[-1])
            if coalesce_page_dma
            else (2, block_k, index_key_cache.shape[-1])
        )
        scratch_shapes = [
            pltpu.VMEM(key_shape, index_key_cache.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ]

    scores = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_queries, 1, max_kv), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=grid,
            in_specs=[
                q_spec,
                weights_spec,
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=score_spec,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
            # Long DCP score rows exceed the default 32 MiB scoped reservation.
            vmem_limit_bytes=63 * 1024 * 1024,
        ),
        interpret=interpret,
        name=(
            "dsa_paged_decode_score_opt_contiguous"
            if coalesce_page_dma
            else "dsa_paged_decode_score_opt_paged"
        ),
    )(
        *scalar_prefetches,
        q_idx,
        idx_weights_3d,
        index_key_cache,
    )
    active_rows = jnp.arange(num_queries, dtype=jnp.int32) < active_num_seqs
    return jnp.where(active_rows[:, None], scores[:, 0, :], _NEG_INF)


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "persistent_two_seq",
        "coalesce_page_dma",
        "interpret",
    ),
)
def paged_decode_scores_pallas(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_BLOCK_K,
    first_dot_bf16: bool = False,
    persistent_two_seq: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score decode rows directly from paged keys.

    When requested, block DMA is used only after verifying complete active
    sequence page tables; fragmented or malformed tables take the exact
    per-page fallback.
    """

    kwargs = dict(
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        interpret=interpret,
    )
    if not coalesce_page_dma:
        return _paged_decode_scores_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_kv_lens,
            distribution,
            coalesce_page_dma=False,
            **kwargs,
        )

    num_queries = q_idx.shape[0]
    page_size = index_key_cache.shape[1]
    active_num_seqs = jnp.clip(distribution[2], 0, num_queries)
    page_table_starts = cu_kv_lens[:num_queries] // page_size

    def read_sequence_pages(page_table_start):
        return jax.lax.dynamic_slice_in_dim(
            page_indices,
            page_table_start,
            pages_per_seq,
        )

    sequence_pages = jax.vmap(read_sequence_pages)(page_table_starts)
    expected_pages = (
        sequence_pages[:, :1]
        + jnp.arange(
            pages_per_seq,
            dtype=jnp.int32,
        )[None, :]
    )
    page_table_in_bounds = (page_table_starts >= 0) & (
        page_table_starts + pages_per_seq <= page_indices.shape[0]
    )
    contiguous_rows = page_table_in_bounds & jnp.all(
        sequence_pages == expected_pages,
        axis=1,
    )
    active_rows = jnp.arange(num_queries, dtype=jnp.int32) < active_num_seqs
    all_active_rows_contiguous = jnp.all(~active_rows | contiguous_rows)

    def score(*, contiguous_page_table: bool):
        return _paged_decode_scores_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_kv_lens,
            distribution,
            coalesce_page_dma=contiguous_page_table,
            **kwargs,
        )

    return jax.lax.cond(
        all_active_rows_contiguous,
        lambda: score(contiguous_page_table=True),
        lambda: score(contiguous_page_table=False),
    )
