# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Sources (epic/dsv41):
#   src/vllm_torchtpu/kernels/deepseek_v41/indexer.py (score/top-k pipelines)
#   src/vllm_torchtpu/models/vllm/deepseek_v41/v2/indexer_select.py
#   src/vllm_torchtpu/models/vllm/deepseek_v41/v2/indexer_fp8.py (JAX parts)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score/top-k pipelines, row grouping and selection over visible scores."""

import functools
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu


def pipeline_score_topk_unrolled(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
) -> Any:
    """Leave independent score/selection tiles available to the scheduler.

    Use for tuned small-query batches; score lifetimes are not bounded here.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    for tile in range(num_tiles):
        index = jnp.int32(tile)
        scores = score(prepare(index), None)
        selected = select(index, scores, None)
        output = write(index, selected, output)
    return output


def pipeline_score_topk_bounded(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
) -> Any:
    """Bound score lifetime and pair next-tile scoring with current selection.

    Callbacks accept an optional scheduling group after their data arguments.
    Each pair gets a program-unique group so the compiler keeps TensorCore
    scoring between SparseCore selection start and completion.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    if num_tiles > 16:
        return pipeline_score_topk(
            num_tiles,
            prepare,
            lambda prepared: score(prepared, None),
            lambda tile, scores: select(tile, scores, None),
            write,
            output,
        )

    # Share V4's process-wide counter so the two pipelines never reuse an ID.
    # Imported here: the dsv4 package pulls in its attention backend.
    from sglang_tpu.dsv4.streamindex_topk import config

    group_base = (
        config.reserve_scheduling_group_ids(num_tiles - 1) if num_tiles > 1 else 0
    )
    # Gate current scores and next preparation on the previous selection
    # so unrolling does not extend every score buffer's lifetime.
    with jax.named_scope("DSV41ScoreTopKBoundedPipeline"):
        current = score(prepare(jnp.int32(0)), None)
        previous = None
        for tile in range(num_tiles - 1):
            prepared = prepare(jnp.int32(tile + 1))
            if previous is not None:
                previous, current, prepared = lax.optimization_barrier(
                    (previous, current, prepared)
                )
                output = write(jnp.int32(tile - 1), previous, output)
            group = group_base + tile
            following = score(prepared, group)
            previous = select(jnp.int32(tile), current, group)
            current = following
        if previous is not None:
            previous, current = lax.optimization_barrier((previous, current))
            output = write(jnp.int32(num_tiles - 2), previous, output)
        last = jnp.int32(num_tiles - 1)
        return write(last, select(last, current, None), output)


def pipeline_score_topk(num_tiles, prepare, score, select, write, output):
    """Order TensorCore scoring and SparseCore selection across two query tiles.

    The barrier gates score-buffer reuse on selection completion, before
    output scatter. Physical execution overlap depends on the compiler.
    """
    first = score(prepare(jnp.int32(0)))

    def advance(tile, state):
        current, result = state
        following = score(prepare(tile + 1))
        return following, write(tile, select(tile, current), result)

    def body(pair, state):
        tile = 2 * pair
        current, result = state
        following = score(prepare(tile + 1))
        selected = select(tile, current)
        prepared, selected = lax.optimization_barrier((prepare(tile + 2), selected))
        replacement = score(prepared)
        following_selected = select(tile + 1, following)
        result = write(tile, selected, result)
        result = write(tile + 1, following_selected, result)
        return replacement, result

    pairs = (num_tiles - 1) // 2
    with jax.named_scope("DSV41ScoreTopKPipeline"):
        state = lax.fori_loop(0, pairs, body, (first, output))
        if 2 * pairs < num_tiles - 1:
            state = advance(jnp.int32(2 * pairs), state)
        last, result = state
        return write(
            jnp.int32(num_tiles - 1), select(jnp.int32(num_tiles - 1), last), result
        )


def _compact_selected(records: jax.Array) -> jax.Array:
    """Sort valid (>= 0) indices ascending and move -1 to the tail."""
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


def _prefix_count(mask: jax.Array, block: int = 128) -> jax.Array:
    """Inclusive ``cumsum(mask, -1)`` as exact blockwise 0/1 matmuls."""
    rows, width = mask.shape
    blocks = -(-width // block)
    x = jnp.pad(mask, ((0, 0), (0, blocks * block - width))).astype(jnp.bfloat16)
    x = x.reshape(rows, blocks, block)
    upper = jnp.triu(jnp.ones((block, block), jnp.bfloat16))
    inner = jnp.einsum("nbi,ij->nbj", x, upper, preferred_element_type=jnp.float32)
    totals = inner[..., -1].astype(jnp.bfloat16)  # at most `block`: exact
    strict = jnp.triu(jnp.ones((blocks, blocks), jnp.bfloat16), 1)
    before = jnp.einsum("nb,bc->nc", totals, strict, preferred_element_type=jnp.float32)
    return (inner + before[..., None]).reshape(rows, -1)[:, :width].astype(jnp.int32)


def _threshold_top_k(scores: jax.Array, k: int) -> jax.Array:
    """Ascending indices of ``jax.lax.top_k(scores, k)``, no sort.

    Finds the k-th largest value by bisecting an order-preserving integer key,
    breaks ties at it by lowest index (as top_k does), then compacts.
    """
    rows = scores.shape[0]
    # No -0.0 -> +0.0 folding: top_k ranks +0.0 above -0.0 (TPU and CPU), as
    # does the order-preserving key below.
    bits = jax.lax.bitcast_convert_type(scores, jnp.int32)
    key = jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits) ^ jnp.int32(-(2**31))
    key = jax.lax.bitcast_convert_type(key, jnp.uint32)

    def bisect(bit, threshold):
        candidate = threshold | (jnp.uint32(1) << (31 - bit).astype(jnp.uint32))
        enough = (key >= candidate[:, None]).sum(-1) >= k
        return jnp.where(enough, candidate, threshold)

    threshold = jax.lax.fori_loop(0, 32, bisect, jnp.zeros((rows,), jnp.uint32))
    above = key > threshold[:, None]
    tied = key == threshold[:, None]
    remaining = k - above.sum(-1)
    chosen = above | (tied & (_prefix_count(tied) <= remaining[:, None]))
    ranks = _prefix_count(chosen)
    targets = jnp.broadcast_to(jnp.arange(1, k + 1, dtype=jnp.int32), (rows, k))
    return jax.vmap(functools.partial(jnp.searchsorted, side="left"))(
        ranks, targets
    ).astype(jnp.int32)


def _sparsecore_top_k(
    scores: jax.Array,
    k: int,
    lengths: jax.Array,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """V4's SparseCore top-k (`sglang_tpu/dsv4/sparsecore_topk.py`).

    Same set as ``jax.lax.top_k`` over the finite scores (ties at the k-th
    value by lowest index), ascending (score order for ``k > 4096``), ``-1``
    suffix-padded; ``-inf`` and columns ``>= lengths`` are never selected.
    """
    from sglang_tpu.dsv4.sparsecore_topk import LANES, sparsecore_topk

    if not (0 < k <= 4096 and k % LANES == 0):
        masked = jnp.where(
            jnp.arange(scores.shape[1]) < lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(masked, k)
        return jnp.where(values > -jnp.inf, ids, -1)
    # The SGLang ops run unpartitioned on each rank, so no shard_map is needed
    # where the source wraps this call for its SPMD model mesh.
    return sparsecore_topk(
        scores, k=k, row_lengths=lengths, scheduling_group_id=scheduling_group_id
    )


def _visible_top_k(
    scores: jax.Array,
    k: int,
    bound: jax.Array,
    lengths: jax.Array | None = None,
    unordered: bool = False,
) -> jax.Array:
    """Ascending ``jax.lax.top_k(scores, k)`` indices when columns ``>= bound``
    are ``-inf``.

    Selects only within a static power-of-two bucket of columns covering
    ``bound`` (the full width otherwise). Exact: buckets hold at least ``k``
    columns, so ``-inf`` ties still resolve to the lowest indices. Large
    selections use `_threshold_top_k`; small ones (decode) sort.

    ``unordered`` callers (which re-sort and drop ``-1``) accept the finite
    top-k set in any order, ``-1`` padded instead of ``-inf`` columns; large
    selections then run on SparseCore over the full width, bounded by
    ``lengths`` instead of buckets (no SparseCore op inside `lax.switch`).
    Small decode query batches stay unpadded to avoid extra subcore waves.
    """
    width = scores.shape[1]
    if (
        unordered
        and lengths is not None
        and width >= 2048
        and width % 16 == 0
        and k % 16 == 0
    ):
        return _sparsecore_top_k(scores, k, jnp.minimum(lengths, width))

    def select(columns, scores):
        scores = scores[:, :columns]
        if scores.shape[0] >= 16 and columns >= 2048:
            return _threshold_top_k(scores, k)
        return jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)

    buckets = []
    bucket = max(k, 1024)
    while bucket < width:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(width)
    if len(buckets) == 1:
        return select(width, scores)
    branch = jnp.searchsorted(jnp.asarray(buckets[:-1]), bound, side="left")
    return jax.lax.switch(
        branch, [functools.partial(select, b) for b in buckets], scores
    )


def indexer_query_rows(tokens: int, query_tile: int) -> int:
    """Derive query chunks from query tiles and the SparseCore wave size."""
    if jax.default_backend() != "tpu":
        return tokens
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        return tokens
    wave = math.lcm(query_tile, sc.num_cores * sc.num_subcores)
    chunk = math.gcd(tokens, wave)
    return chunk if chunk % query_tile == 0 else tokens


def _query_start_loc(requests: jax.Array, num_requests: int) -> jax.Array:
    """[num_requests + 1] row boundaries; padding (-1) rows join the last request."""
    last = num_requests - 1
    owners = jnp.where(requests >= 0, jnp.minimum(requests, last), last)
    bounds = jnp.arange(num_requests + 1, dtype=owners.dtype)
    return jnp.sum(owners[:, None] < bounds, axis=0, dtype=jnp.int32)
