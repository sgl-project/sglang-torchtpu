"""DSA indexer scoring, masking, selection, and software pipelining."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from glm52_fp8.configs.dsa_config import pipeline_short_indexer_requests
from glm52_fp8.kernels.dsa.paged_score import (
    paged_decode_scores_pallas,
    paged_extend_score_block_pallas,
    paged_extend_score_uniform_packed_pallas,
)
from glm52_fp8.kernels.dsa.physical_mapping import map_logical_topk
from glm52_fp8.kernels.dsa.topk import (
    _compact_topk_indices,
    _compact_valid_topk_indices,
    _supports_radix_topk_v2,
    select_indexer_radix_topk_indices,
    select_indexer_topk,
)
from glm52_fp8.kernels.radix_topk.v2.tuned_configs import get_score_width_config

_NEG_INF = float("-inf")
# Keep 32 as the supported Pallas query-block ceiling and explicit benchmark
# default. Serving chooses its Extend block from the global device count.
_INDEXER_QUERY_BLOCK_SIZE = 32
_SMALL_DEVICE_EXTEND_QUERY_BLOCK_SIZE = 32
_LARGE_DEVICE_EXTEND_QUERY_BLOCK_SIZE = 24
_LARGE_EXTEND_DEVICE_COUNT_THRESHOLD = 32
# Reserve half of v7x's 64 MiB VMEM for score tiles, leaving room for key
# buffers, query operands, and compiler spills.
_MAX_EXTEND_SCORE_ELEMENTS = 32 * 262144


def _score_length_buckets(capacity: int, k: int, page_size: int) -> tuple[int, ...]:
    if capacity < k or capacity % page_size:
        raise ValueError("Indexer capacity must be page-aligned and at least k")
    widths = []
    width = 32768
    while width < capacity:
        if width >= k and width % page_size == 0:
            widths.append(width)
        width *= 2
    return (*widths, capacity)


def _bounded_extend_query_block_size(requested: int, max_kv: int) -> int:
    limit = max(1, _MAX_EXTEND_SCORE_ELEMENTS // max_kv)
    # TPU score rows are tiled in eights; don't exceed the budget after padding.
    if limit >= 8:
        limit = limit // 8 * 8
    return min(requested, limit)


def _default_extend_query_block_size() -> int:
    """Choose the measured Extend query block for the global TPU slice."""

    if jax.device_count() >= _LARGE_EXTEND_DEVICE_COUNT_THRESHOLD:
        return _LARGE_DEVICE_EXTEND_QUERY_BLOCK_SIZE
    return _SMALL_DEVICE_EXTEND_QUERY_BLOCK_SIZE


def _mask_and_compact_topk_indices(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Mask invalid candidates and compact them without sorting valid scores."""

    return _compact_topk_indices(values > _NEG_INF, indices)


def _compute_score_tile(
    q_tile: jax.Array,
    weights_tile: jax.Array,
    sequence_keys: jax.Array,
) -> jax.Array:
    """Compute DSA index scores for one query tile and one sequence."""

    num_queries, num_heads, _ = q_tile.shape
    max_kv = sequence_keys.shape[0]
    if num_queries * num_heads * max_kv <= 1 << 28:
        similarities = jnp.einsum(
            "thd,kd->thk",
            q_tile,
            sequence_keys,
            preferred_element_type=jnp.float32,
        )
        return jnp.einsum("th,thk->tk", weights_tile, jax.nn.relu(similarities))

    def accumulate_head(head_id, scores):
        q_head = jax.lax.dynamic_index_in_dim(q_tile, head_id, axis=1, keepdims=False)
        weight_head = jax.lax.dynamic_index_in_dim(weights_tile, head_id, axis=1, keepdims=True)
        similarities = jnp.einsum(
            "td,kd->tk",
            q_head,
            sequence_keys,
            preferred_element_type=jnp.float32,
        )
        return scores + jax.nn.relu(similarities) * weight_head

    return jax.lax.fori_loop(
        0,
        num_heads,
        accumulate_head,
        jnp.zeros((num_queries, max_kv), jnp.float32),
    )


def _select_topk_indices(
    scores: jax.Array,
    valid_lengths: jax.Array,
    *,
    k: int,
    topk_impl: str,
    page_table: jax.Array | None = None,
    page_size: int = 64,
    radix_v2_config=None,
) -> jax.Array:
    if valid_lengths.shape != scores.shape[:1]:
        raise ValueError(
            f"valid_lengths must have shape {scores.shape[:1]}, got {valid_lengths.shape}"
        )
    if topk_impl == "radix":
        indices = select_indexer_radix_topk_indices(
            scores,
            k=k,
            valid_lengths=valid_lengths,
            page_table=page_table,
            page_size=page_size,
            v2_config=radix_v2_config,
        )
        if page_table is not None:
            return indices
        return _compact_valid_topk_indices(indices, valid_lengths)

    values, indices = select_indexer_topk(
        scores,
        k=k,
        implementation=topk_impl,
    )
    indices = _mask_and_compact_topk_indices(values, indices)
    if page_table is not None:
        return _map_logical_topk_to_physical_slots(indices, page_table, valid_lengths, page_size)
    return indices


def _map_logical_topk_to_physical_slots(logical_topk, seq_pages, valid_lengths, page_size):
    """Normalize shared/per-row page tables for the independent mapper."""
    if seq_pages.ndim == 1:
        tables = seq_pages[None, :]
        rows = jnp.zeros((logical_topk.shape[0],), jnp.int32)
    elif seq_pages.ndim == 2 and seq_pages.shape[0] == logical_topk.shape[0]:
        tables = seq_pages
        rows = jnp.arange(logical_topk.shape[0], dtype=jnp.int32)
    else:
        raise ValueError("page tables must be shared or one per query row")
    return map_logical_topk(logical_topk, tables, rows, valid_lengths, page_size=page_size)


def _map_packed_logical_topk_to_physical_slots(
    logical_topk: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    row_sequences: jax.Array,
    valid_lengths: jax.Array,
    *,
    pages_per_seq: int,
    page_size: int,
) -> jax.Array:
    """Map packed selections using the scorer's existing row metadata."""
    num_seqs = cu_kv_lens.shape[0] - 1
    ptr = cu_kv_lens[:num_seqs, None] // page_size + jnp.arange(pages_per_seq)[None, :]
    table = page_indices[jnp.clip(ptr, 0, page_indices.shape[0] - 1)]
    table = jnp.where((ptr >= 0) & (ptr < page_indices.shape[0]), table, -1)
    return map_logical_topk(
        logical_topk,
        table,
        row_sequences,
        valid_lengths,
        page_size=page_size,
    )


def _pipeline_score_and_topk_stages(num_tiles, prepare_score, score, select, write, out):
    """Pair tiles and gate score-buffer reuse on TopK completion before writeback."""

    def nonempty(output):
        first = score(prepare_score(jnp.int32(0)))

        def advance(tile, carry):
            current, output = carry
            following = score(prepare_score(tile + 1))
            output = write(tile, select(tile, current), output)
            return following, output

        def body(pair, carry):
            tile = 2 * pair
            current, output = carry
            following = score(prepare_score(tile + 1))
            selected = select(tile, current)
            # Both outputs are consumed through the barrier. Input preparation
            # and the other TopK can run independently; output writeback is
            # not a prerequisite for computing the replacement score.
            prepared, selected = jax.lax.optimization_barrier((prepare_score(tile + 2), selected))
            replacement = score(prepared)
            following_selected = select(tile + 1, following)
            output = write(tile, selected, output)
            output = write(tile + 1, following_selected, output)
            return replacement, output

        pairs = (num_tiles - 1) // 2
        carry = jax.lax.fori_loop(0, pairs, body, (first, output))
        carry = jax.lax.cond(
            2 * pairs < num_tiles - 1,
            lambda state: advance(2 * pairs, state),
            lambda state: state,
            carry,
        )
        last, output = carry
        return write(num_tiles - 1, select(num_tiles - 1, last), output)

    with jax.named_scope("ScoreTopKPipeline"):
        return jax.lax.cond(num_tiles > 0, nonempty, lambda output: output, out)


def _compute_decode_scores_and_select_topk_indices(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    topk_impl: str,
    output_physical_slots: bool,
) -> jax.Array:
    """Score one query per sequence and select all decode rows in one top-k call.

    Decode requires ``T == num_seqs`` and query row ``i`` to belong to sequence
    ``i``. On TPU, a Pallas scorer follows the page table directly and only
    materializes bounded key tiles in VMEM. A two-row decode uses one persistent
    program that prefetches both page streams and scores them through batched
    matrix products. The resulting ``[T, max_kv]`` FP32 score matrix is
    submitted to top-k as one batch so two-row decode can use both SparseCores
    in the radix implementation. Non-TPU backends keep the JAX gather
    implementation as a correctness fallback.

    Args:
      q_idx:         f[T, H, D]  indexer query heads
      idx_weights:   f[T, H]     per-head mixing weights
      index_key_cache: f[P, page_size, D]  paged indexer keys
      seq_lens:     i32[S]      kv length per sequence
      page_indices: i32[N_pages]  packed; seq i's pages start at cu_kv_lens[i]//page_size
      cu_kv_lens:   i32[S+1]    cumsum of page-aligned kv lens (page_indices stride)
      distribution: i32[3]      (decode_end, prefill_end, num_seqs)
      k:            top-k budget
      pages_per_seq: static, maximum pages materialized per sequence
      topk_impl: selection backend: approximate XLA, exact XLA, or exact
        SparseCore radix selection (``approx``, ``exact_lax``, or ``radix``).

    Returns:
      i32[T, k]  top-k kv positions per query token; -1 for padding.
    """
    T, _, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]
    if num_seqs != T:
        raise ValueError(
            "decode indexer requires one query row per sequence, "
            f"got q_idx.shape[0]={T} and num_seqs={num_seqs}"
        )

    active_num_seqs = jnp.clip(distribution[2], 0, num_seqs)
    idx_weights_f32 = idx_weights.astype(jnp.float32)
    kv_pos = jnp.arange(max_kv, dtype=jnp.int32)

    with jax.named_scope("Decode"), jax.named_scope("PagedScore"):
        pallas_compatible = (
            D % 128 == 0
            and max_kv % 128 == 0
            and page_size <= 2048
            and q_idx.dtype in (jnp.bfloat16, jnp.float32)
            and idx_weights.dtype in (jnp.bfloat16, jnp.float32)
            and index_key_cache.dtype in (jnp.bfloat16, jnp.float32, jnp.uint8)
        )
        if jax.default_backend() == "tpu" and pallas_compatible:
            batched_scores = paged_decode_scores_pallas(
                q_idx,
                idx_weights,
                index_key_cache,
                seq_lens,
                page_indices,
                cu_kv_lens,
                distribution,
                pages_per_seq=pages_per_seq,
                persistent_two_seq=T >= 2 and T % 2 == 0,
                coalesce_page_dma=True,
            )
        else:
            batched_scores = jnp.full((T, max_kv), _NEG_INF, dtype=jnp.float32)

            def score_sequence(seq_id, scores):
                kv_len = seq_lens[seq_id]
                seq_pages = jax.lax.dynamic_slice_in_dim(
                    page_indices,
                    cu_kv_lens[seq_id] // page_size,
                    pages_per_seq,
                )
                sequence_keys = index_key_cache[seq_pages].reshape(max_kv, D)
                q_i = jax.lax.dynamic_slice_in_dim(q_idx, seq_id, 1, axis=0)
                weights_i = jax.lax.dynamic_slice_in_dim(
                    idx_weights_f32,
                    seq_id,
                    1,
                    axis=0,
                )
                row_scores = _compute_score_tile(q_i, weights_i, sequence_keys)
                row_scores = jnp.where(
                    kv_pos[None, :] < kv_len,
                    row_scores,
                    _NEG_INF,
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    scores,
                    row_scores,
                    seq_id,
                    axis=0,
                )

            batched_scores = jax.lax.fori_loop(
                0,
                active_num_seqs,
                score_sequence,
                batched_scores,
            )

    with jax.named_scope("Decode"):
        valid_lengths = jnp.where(
            jnp.arange(T, dtype=jnp.int32) < active_num_seqs,
            jnp.clip(seq_lens, 0, max_kv),
            0,
        )
        if output_physical_slots:
            page_offsets = jnp.arange(pages_per_seq, dtype=jnp.int32)
            page_starts = cu_kv_lens[:-1] // page_size
            page_ptrs = page_starts[:, None] + page_offsets[None, :]
            page_ptr_valid = (page_ptrs >= 0) & (page_ptrs < page_indices.shape[0])
            safe_page_ptrs = jnp.clip(page_ptrs, 0, page_indices.shape[0] - 1)
            seq_pages = page_indices[safe_page_ptrs]
            seq_pages = jnp.where(page_ptr_valid, seq_pages, -1)
        selection_scope = "RadixTopK" if topk_impl == "radix" else f"TopK_{topk_impl}"
        with jax.named_scope(selection_scope):
            selected_topk = _select_topk_indices(
                batched_scores,
                valid_lengths,
                k=k,
                topk_impl=topk_impl,
            )
        if not output_physical_slots:
            return selected_topk
        # Resolve only the selected [T, K] positions through per-row page tables.
        with jax.named_scope("LogicalToPhysicalSlots"):
            return _map_logical_topk_to_physical_slots(
                selected_topk,
                seq_pages,
                valid_lengths,
                page_size,
            )


def _compute_extend_scores_and_select_topk_indices_general(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    topk_impl: str,
    output_physical_slots: bool,
    score_query_block_size: int | None = None,
    max_queries_per_seq: int | None = None,
) -> jax.Array:
    """Score packed extend queries through a bounded score/top-k pipeline.

    TPU query blocks follow the sequence page table directly and double-buffer
    bounded key tiles in VMEM. Other backends gather each sequence's keys once
    as a correctness fallback. Fixed-size score tiles bound peak memory and
    allow scoring block ``n + 1`` to overlap top-k selection for block ``n``.
    """
    T, _, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]
    active_num_seqs = jnp.clip(distribution[2], 0, num_seqs)
    if score_query_block_size is None:
        score_query_block_size = _default_extend_query_block_size()
    if score_query_block_size < 1:
        raise ValueError(f"score_query_block_size must be positive, got {score_query_block_size}")

    idx_weights_f32 = idx_weights.astype(jnp.float32)
    kv_pos = jnp.arange(max_kv, dtype=jnp.int32)

    # Keep each loop-carried score tile small enough to avoid materializing the
    # OOM-sized [T, H, max_kv] tensor. Padding the query axis prevents
    # dynamic_slice from clamping the final ragged block back into valid rows.
    query_block_size = _bounded_extend_query_block_size(min(score_query_block_size, T), max_kv)
    use_paged_score = (
        jax.default_backend() == "tpu"
        and D % 128 == 0
        and max_kv % 128 == 0
        and page_size <= 2048
        and query_block_size <= _INDEXER_QUERY_BLOCK_SIZE
        and q_idx.dtype in (jnp.bfloat16, jnp.float32)
        and idx_weights.dtype in (jnp.bfloat16, jnp.float32)
        and index_key_cache.dtype in (jnp.bfloat16, jnp.float32, jnp.uint8)
    )
    score_tile_scope = "PagedScoreTile" if use_paged_score else "ScoreTile"
    pipeline_scope = "ScoreTopKMappingPipeline" if output_physical_slots else "ScoreTopKPipeline"
    q_idx_padded = jnp.pad(q_idx, ((0, query_block_size), (0, 0), (0, 0)))
    idx_weights_padded = jnp.pad(idx_weights_f32, ((0, query_block_size), (0, 0)))
    out_padded = jnp.full((T + query_block_size, k), -1, dtype=jnp.int32)

    def body(seq_id, packed_out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        q_len = jnp.maximum(q_end - q_start, 0)
        kv_len = seq_lens[seq_id]
        with jax.named_scope("Extend"), jax.named_scope("IndexKeyPages"):
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            if not use_paged_score:
                seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)

        num_query_blocks = (q_len + query_block_size - 1) // query_block_size

        def block_inputs(block_id):
            block_start = q_start + block_id * query_block_size
            q_block = jax.lax.dynamic_slice_in_dim(
                q_idx_padded, block_start, query_block_size, axis=0
            )
            weights_block = jax.lax.dynamic_slice_in_dim(
                idx_weights_padded, block_start, query_block_size, axis=0
            )
            local_q_pos = block_id * query_block_size + jnp.arange(
                query_block_size, dtype=jnp.int32
            )
            query_valid = local_q_pos < q_len
            valid_lengths = jnp.where(
                query_valid,
                jnp.clip(kv_len - q_len + local_q_pos + 1, 0, max_kv),
                0,
            )
            return q_block, weights_block, valid_lengths

        def run_pipeline(radix_v2_config=None):
            def score_prepared(prepared):
                with (
                    jax.named_scope("Extend"),
                    jax.named_scope(pipeline_scope),
                    jax.named_scope(score_tile_scope),
                ):
                    q_block, weights_block, valid_lengths = prepared
                    if use_paged_score:
                        scores_block = paged_extend_score_block_pallas(
                            q_block,
                            weights_block,
                            index_key_cache,
                            valid_lengths,
                            seq_pages,
                            pages_per_seq=pages_per_seq,
                            block_k=4096 if query_block_size == 32 else 2048,
                            coalesce_page_dma=True,
                        )
                    else:
                        scores_block = _compute_score_tile(
                            q_block,
                            weights_block,
                            seq_k_idx,
                        )
                        mask = kv_pos[None, :] < valid_lengths[:, None]
                        scores_block = jnp.where(mask, scores_block, _NEG_INF)
                    return scores_block

            def select_block(block_id, scores_block):
                # The adapter owns candidate selection. Radix candidates remain
                # unordered; invalid/padded entries are compacted behind them.
                _, _, valid_lengths = block_inputs(block_id)
                selection_scope = "RadixTopK" if topk_impl == "radix" else f"TopK_{topk_impl}"
                with (
                    jax.named_scope("Extend"),
                    jax.named_scope(pipeline_scope),
                    jax.named_scope(selection_scope),
                ):
                    return _select_topk_indices(
                        scores_block,
                        valid_lengths,
                        k=k,
                        topk_impl=topk_impl,
                        page_table=seq_pages if output_physical_slots else None,
                        page_size=page_size,
                        radix_v2_config=radix_v2_config,
                    )

            def write_block(block_id, block_indices, block_out):
                block_start = q_start + block_id * query_block_size
                with (
                    jax.named_scope("Extend"),
                    jax.named_scope(pipeline_scope),
                    jax.named_scope("TopKOutputUpdate"),
                ):
                    return jax.lax.dynamic_update_slice_in_dim(
                        block_out,
                        block_indices,
                        block_start,
                        axis=0,
                    )

            with jax.named_scope("Extend"):
                return _pipeline_score_and_topk_stages(
                    num_query_blocks,
                    block_inputs,
                    score_prepared,
                    select_block,
                    write_block,
                    packed_out,
                )

        if topk_impl == "radix":
            # Bind one config per outer score-width bucket, avoiding a second
            # switch that clones the entire pipeline. Keep the async TopK
            # calls beside scoring; per-row live lengths still control masking.
            config = (
                get_score_width_config(max_kv, k)
                if _supports_radix_topk_v2(query_block_size, k, max_kv)
                else None
            )
            return run_pipeline(config)
        return run_pipeline()

    # Keep query/score builders inline in each pipeline. Sharing them changed
    # the native Q6 executable schedule and regressed measured TPU time.
    if pipeline_short_indexer_requests(
        query_block_size=query_block_size,
        max_queries_per_seq=max_queries_per_seq,
        max_kv=max_kv,
        physical=output_physical_slots,
    ):
        query_counts = jnp.maximum(jnp.diff(cu_q_lens), 0)
        request_order = jnp.arange(num_seqs, dtype=jnp.int32)
        live = (request_order < active_num_seqs) & (query_counts > 0)

        def pipeline_requests(output):
            # Unique keys retain request order without nonzero's prefix scan
            # and scatter. Only the first sum(live) entries are consumed.
            sequence_ids = jnp.argsort(
                jnp.where(live, request_order, request_order + num_seqs), stable=False
            )

            def prepare_request(tile):
                seq_id = sequence_ids[tile]
                q_start = cu_q_lens[seq_id]
                q_len = query_counts[seq_id]
                query = jax.lax.dynamic_slice_in_dim(q_idx_padded, q_start, query_block_size)
                weights = jax.lax.dynamic_slice_in_dim(
                    idx_weights_padded, q_start, query_block_size
                )
                positions = jnp.arange(query_block_size, dtype=jnp.int32)
                lengths = jnp.where(
                    positions < q_len,
                    jnp.clip(seq_lens[seq_id] - q_len + positions + 1, 0, max_kv),
                    0,
                )
                pages = jax.lax.dynamic_slice_in_dim(
                    page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
                )
                return query, weights, lengths, pages

            def score_request(prepared):
                query, weights, lengths, pages = prepared
                # Preserve the original Q6 MXU and radix layouts. Combining
                # requests into Q12 changes real-model token sequences.
                if use_paged_score:
                    return paged_extend_score_block_pallas(
                        query,
                        weights,
                        index_key_cache,
                        lengths,
                        pages,
                        pages_per_seq=pages_per_seq,
                        block_k=2048,
                        coalesce_page_dma=True,
                    )
                keys = index_key_cache[pages].reshape(max_kv, D)
                scores = _compute_score_tile(query, weights, keys)
                return jnp.where(kv_pos[None, :] < lengths[:, None], scores, _NEG_INF)

            def select_request(tile, scores):
                _, _, lengths, _ = prepare_request(tile)
                return _select_topk_indices(
                    scores,
                    lengths,
                    k=k,
                    topk_impl=topk_impl,
                    page_size=page_size,
                )

            def write_request(tile, indices, output):
                # Ragged tails may overlap the following packed request; the
                # pipeline writes in the original request order.
                return jax.lax.dynamic_update_slice_in_dim(
                    output, indices, cu_q_lens[sequence_ids[tile]], axis=0
                )

            with jax.named_scope("ShortRequestPipeline"):
                return _pipeline_score_and_topk_stages(
                    jnp.sum(live, dtype=jnp.int32),
                    prepare_request,
                    score_request,
                    select_request,
                    write_request,
                    output,
                )

        out_padded = pipeline_requests(out_padded)
    else:
        out_padded = jax.lax.fori_loop(0, active_num_seqs, body, out_padded)
    return out_padded[:T]


def can_pack_uniform_queries(
    q_idx, idx_weights, index_key_cache, *, num_sequences, pages_per_seq, topk_impl
):
    """Whether the native scorer can share one wave across fixed-width chains."""
    num_queries, _, head_dim = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    return (
        jax.default_backend() == "tpu"
        and 1 < num_sequences <= num_queries <= _INDEXER_QUERY_BLOCK_SIZE
        and num_queries % num_sequences == 0
        and num_queries * max_kv <= _MAX_EXTEND_SCORE_ELEMENTS
        and head_dim % 128 == 0
        and max_kv % 128 == 0
        and page_size <= 2048
        and topk_impl == "radix"
        and q_idx.dtype in (jnp.bfloat16, jnp.float32)
        and idx_weights.dtype in (jnp.bfloat16, jnp.float32)
        and index_key_cache.dtype in (jnp.bfloat16, jnp.float32, jnp.uint8)
    )


def _compute_extend_scores_and_select_topk_indices(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    topk_impl: str,
    output_physical_slots: bool,
    score_query_block_size: int | None = None,
    max_queries_per_seq: int | None = None,
) -> jax.Array:
    """Prefer one packed score/Top-K wave for uniform small query chains.

    ``score_query_block_size`` controls only the per-sequence fallback; it does
    not limit a packed wave that combines queries from multiple requests.
    """

    num_queries = q_idx.shape[0]
    num_sequences = seq_lens.shape[0]
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    can_pack_uniform_chains = can_pack_uniform_queries(
        q_idx,
        idx_weights,
        index_key_cache,
        num_sequences=num_sequences,
        pages_per_seq=pages_per_seq,
        topk_impl=topk_impl,
    )
    if not can_pack_uniform_chains:
        return _compute_extend_scores_and_select_topk_indices_general(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl=topk_impl,
            output_physical_slots=output_physical_slots,
            score_query_block_size=score_query_block_size,
            max_queries_per_seq=max_queries_per_seq,
        )

    rows_per_sequence = num_queries // num_sequences
    expected_cu_q_lens = jnp.arange(num_sequences + 1, dtype=jnp.int32) * rows_per_sequence
    active_num_sequences = jnp.clip(distribution[2], 0, num_sequences)
    use_packed = (active_num_sequences == num_sequences) & jnp.all(cu_q_lens == expected_cu_q_lens)

    def packed_uniform(_):
        row_ids = jnp.arange(num_queries, dtype=jnp.int32)
        row_sequence_ids = row_ids // rows_per_sequence
        local_query_positions = row_ids % rows_per_sequence
        row_kv_lens = seq_lens[row_sequence_ids]
        valid_lengths = jnp.clip(
            row_kv_lens - rows_per_sequence + local_query_positions + 1,
            0,
            max_kv,
        )
        with (
            jax.named_scope("Extend"),
            jax.named_scope("UniformPackedSmallQuery"),
            jax.named_scope("PagedScore"),
        ):
            scores = paged_extend_score_uniform_packed_pallas(
                q_idx,
                idx_weights.astype(jnp.float32),
                index_key_cache,
                valid_lengths,
                page_indices,
                cu_kv_lens,
                num_sequences=num_sequences,
                pages_per_seq=pages_per_seq,
                coalesce_page_dma=True,
            )
        with (
            jax.named_scope("Extend"),
            jax.named_scope("UniformPackedSmallQuery"),
            jax.named_scope("RadixTopK"),
        ):
            selected_topk = _select_topk_indices(
                scores,
                valid_lengths,
                k=k,
                topk_impl=topk_impl,
            )
        if not output_physical_slots:
            return selected_topk
        with (
            jax.named_scope("Extend"),
            jax.named_scope("UniformPackedSmallQuery"),
            jax.named_scope("LogicalToPhysicalSlots"),
        ):
            return _map_packed_logical_topk_to_physical_slots(
                selected_topk,
                page_indices,
                cu_kv_lens,
                row_sequence_ids,
                valid_lengths,
                pages_per_seq=pages_per_seq,
                page_size=page_size,
            )

    def general(_):
        return _compute_extend_scores_and_select_topk_indices_general(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl=topk_impl,
            output_physical_slots=output_physical_slots,
            score_query_block_size=score_query_block_size,
            max_queries_per_seq=max_queries_per_seq,
        )

    return jax.lax.cond(use_packed, packed_uniform, general, operand=None)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "pages_per_seq",
        "one_token_per_seq",
        "topk_impl",
        "output_physical_slots",
        "score_query_block_size",
        "max_queries_per_seq",
    ),
)
def compute_scores_and_select_topk_indices(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
    topk_impl: str,
    output_physical_slots: bool = False,
    score_query_block_size: int | None = None,
    max_queries_per_seq: int | None = None,
) -> jax.Array:
    """Score indexer queries and select top-k token positions.

    ``one_token_per_seq=True`` selects the decode implementation: it scores
    each sequence without materializing ``[T, max_kv, D]`` keys, then submits
    the resulting ``[T, max_kv]`` score matrix to one batched top-k call.

    Otherwise the extend implementation gathers each sequence's keys once and
    streams its query blocks through the score/top-k ping-pong pipeline. Its
    default query block is 24 on global slices with at least 32 devices and 32
    on smaller slices; an explicit ``score_query_block_size`` overrides it.

    ``max_queries_per_seq`` is a caller-proven static upper bound on every
    difference in ``cu_q_lens``. It enables request overlap without changing
    query blocks or score-width buckets. Generic callers must leave it unset
    unless their input construction guarantees the bound.

    By default the result contains sequence-local logical positions. When
    ``output_physical_slots=True``, selection also resolves those positions to
    flattened KV-cache slots through the independent mapping stage.
    """
    if max_queries_per_seq is not None and max_queries_per_seq < 1:
        raise ValueError("max_queries_per_seq must be positive when provided")

    # Select using the live device lengths: fused MTP updates them inside its
    # decode loop, so a host-side cached bound can become stale. Every branch
    # has a static score width and the same [T, K] output. Page-table offsets
    # remain the original ragged cu_kv_lens offsets, independent of this width.
    page_size = index_key_cache.shape[1]
    capacity = pages_per_seq * page_size
    # Geometric capacities include 1M and its page-aligned output headroom.
    # Cap the last bucket at the actual page
    # table capacity: widening a ragged slice would read another sequence's
    # pages (or clamp its start). Short prefixes retain exact length masking.
    bucket_widths = _score_length_buckets(capacity, k, page_size)

    def run_bucket(bucket_pages):
        if one_token_per_seq:
            return _compute_decode_scores_and_select_topk_indices(
                q_idx,
                idx_weights,
                index_key_cache,
                seq_lens,
                page_indices,
                cu_kv_lens,
                distribution,
                k=k,
                pages_per_seq=bucket_pages,
                topk_impl=topk_impl,
                output_physical_slots=output_physical_slots,
            )

        return _compute_extend_scores_and_select_topk_indices(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=bucket_pages,
            topk_impl=topk_impl,
            output_physical_slots=output_physical_slots,
            score_query_block_size=score_query_block_size,
            max_queries_per_seq=max_queries_per_seq,
        )

    if len(bucket_widths) == 1:
        return run_bucket(pages_per_seq)
    longest = jnp.max(seq_lens)
    bucket_id = jnp.searchsorted(jnp.array(bucket_widths), longest, side="left")
    branches = tuple(
        lambda _, width=width: run_bucket(width // page_size) for width in bucket_widths
    )
    with jax.named_scope("IndexerScoreLengthBucket"):
        return jax.lax.switch(bucket_id, branches, operand=None)
