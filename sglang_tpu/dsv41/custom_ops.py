# SPDX-License-Identifier: Apache-2.0
"""SGLang boundaries for the selected EPIC CSA2 and streaming index kernels."""

from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.dsv4.rope import cos_sin_rows
from sglang_tpu.kernels.deepseek_v4.o_projection import wo_a_projection
from sglang_tpu.kernels.deepseek_v41.csa2.cache import (
    prepare_main_records,
    prepare_swa_metadata,
    swa_word_pages,
)
from sglang_tpu.kernels.deepseek_v41.csa2.csa2 import (
    csa2,
    csa2_gather_chunk_size,
    csa2_pipeline,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import PackedFP8Layout
from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
    prepare_fp8_candidates,
    select_fp8_indexer_jax,
    select_fp8_prepared,
)

# Candidate scoring gathers whole blocks; the epic route supports these sizes.
PREPARED_BLOCK_SIZES = (8, 16, 32, 64, 128)


def uses_prepared_candidates(candidate_count, block_size, score_width):
    """Static epic routing: rescore gathered candidates only when they filter.

    When the candidates cover the whole context, publication returns its input
    unchanged and filtering is an identity, so consumers score densely.
    """
    return (
        block_size in PREPARED_BLOCK_SIZES
        and 0 < candidate_count * block_size < score_width
    )


def _layout_args(layout):
    return (layout.pages, layout.page_stride, layout.page_offset, layout.page_size)


def inverse_rope(cos_sin, positions):
    """EPIC v2/csa2_adapter.py inverse table; CSA2 performs the rotation once."""
    table = cos_sin_rows(cos_sin, positions).astype(jnp.float32)
    cos = jnp.concatenate(
        (jnp.ones((positions.shape[0], 64)), jnp.repeat(table[:, :32], 2, axis=1)), 1
    )
    sin = jnp.repeat(table[:, 32:], 2, axis=1)
    sin *= jnp.where(jnp.arange(64) % 2 == 0, 1.0, -1.0)
    sin = jnp.concatenate((jnp.zeros((positions.shape[0], 64)), sin), 1)
    return jnp.stack((cos, sin), 1)


def index_geometry(words, pages, page_records, storage_records):
    """Expand scheduler pages to the allocator's physical FP8 page geometry."""
    if page_records % storage_records:
        raise ValueError("Scheduler index pages must contain whole storage pages")
    parts = page_records // storage_records
    physical = jnp.where(
        pages[..., None] >= 0,
        pages[..., None] * parts + jnp.arange(parts, dtype=jnp.int32),
        -1,
    ).reshape(pages.shape[0], -1)
    rows = (33 * storage_records + 127) // 128
    return physical, PackedFP8Layout(words.shape[0] // rows, page_size=storage_records)


def select(
    q: jax.Array,
    weights: jax.Array,
    words: jax.Array,
    pages: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    page_records: int,
    storage_records: int,
    topk: int,
    block_size: int,
    publish: bool,
    filter_candidates: bool,
    layer_id: int,
) -> tuple[jax.Array, jax.Array]:
    """Dense paged scoring: sources, unfiltered layers and covering consumers.

    Selections are unordered with -1 at the tail, the valid prefix that
    prepare_main_records requires (the epic FP8 path's precompacted records).
    """
    del layer_id
    physical, layout = index_geometry(words, pages, page_records, storage_records)
    live = (requests >= 0) & (requests < pages.shape[0])
    batch_max = jnp.max(jnp.where(live, lengths, 0), initial=0).reshape(1)
    return select_fp8_indexer_jax(
        q,
        (words,),
        physical,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        _layout_args(layout),
        topk,
        block_size,
        publish,
        filter_candidates,
        True,
        pages.shape[1] * page_records,
    )


def prepare_index(
    words: jax.Array,
    pages: jax.Array,
    requests: jax.Array,
    candidates: jax.Array,
    page_records: int,
    storage_records: int,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gather each query pair's candidate union once per K source and publication.

    Requires each request's query rows to be contiguous, as both forward modes
    build them; invalid rows (-1) may follow.
    """
    physical, layout = index_geometry(words, pages, page_records, storage_records)
    return prepare_fp8_candidates(
        (words,),
        physical,
        candidates,
        requests,
        _layout_args(layout),
        block_size,
        pages.shape[1] * page_records,
    )


def select_index_prepared(
    q: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    keys: jax.Array,
    block_ids: jax.Array,
    offsets: jax.Array,
    valid: jax.Array,
    spans: jax.Array,
    score_width: int,
    topk: int,
    block_size: int,
    layer_id: int,
) -> jax.Array:
    # Keep operands flat: cpu_export_jax_op does not map nested torch inputs.
    del layer_id
    return select_fp8_prepared(
        q,
        weights,
        lengths,
        (keys, block_ids, offsets, valid, spans),
        topk,
        block_size,
        True,
        score_width,
    )


def metadata(
    pages: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    valid: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    meta, _, records = prepare_swa_metadata(
        pages,
        starts,
        positions,
        jnp.zeros(pages.shape[0], jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
    )
    return meta, jnp.where(valid, records, -1)


def gather_attend(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    selected: jax.Array,
    pages: jax.Array,
    swa: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    cos_sin: jax.Array,
    meta: jax.Array,
    page_size: int,
    ratio: int,
    scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    records, counts = prepare_main_records(
        selected, pages, starts, page_size=page_size // ratio
    )
    out, gathered, gathered_scales = csa2_pipeline(
        q,
        words,
        scales,
        records,
        swa_word_pages(swa, page_size),
        starts,
        positions,
        sink,
        inverse_rope(cos_sin, positions),
        meta,
        None,
        counts,
        chunk_size=csa2_gather_chunk_size(q.shape[0]),
        bq=6 if q.shape[0] <= 384 else 16,
        scale=scale,
        compress_ratio=ratio,
    )
    return out, gathered, gathered_scales, counts


def attend(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    swa: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    cos_sin: jax.Array,
    meta: jax.Array,
    page_size: int,
    ratio: int,
    scale: float,
) -> jax.Array:
    return csa2(
        q,
        words,
        scales,
        swa_word_pages(swa, page_size),
        starts,
        positions,
        sink,
        inverse_rope(cos_sin, positions),
        swa_metadata=meta,
        main_lengths=counts,
        compress_ratio=ratio,
        scale=scale,
        bq=6 if q.shape[0] <= 384 else 16,
    )


def attend_swa(
    q: jax.Array,
    swa: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    cos_sin: jax.Array,
    meta: jax.Array,
    page_size: int,
    scale: float,
) -> jax.Array:
    # Pure SWA uses the same CSA2 normalization/inverse-RoPE path with zero
    # main lengths. Bound its unused main operands to one query chunk.
    outputs = []
    for offset in range(0, q.shape[0], 128):
        stop = min(offset + 128, q.shape[0])
        count = stop - offset
        outputs.append(
            csa2(
                q[offset:stop],
                jnp.zeros((count, 512, 128), jnp.int32),
                jnp.zeros((count, 512), jnp.float32),
                swa_word_pages(swa, page_size),
                jnp.clip(starts - offset, 0, count),
                positions[offset:stop],
                sink,
                inverse_rope(cos_sin, positions[offset:stop]),
                swa_metadata=meta,
                main_lengths=jnp.zeros(count, jnp.int32),
                compress_ratio=1,
                scale=scale,
                bq=6,
            )
        )
    return jnp.concatenate(outputs)


def project(x: jax.Array, weight: jax.Array, scales: jax.Array) -> jax.Array:
    return wo_a_projection(x, weight, scales.reshape(-1), None)


class AttentionOps(NamedTuple):
    select: object
    prepare_index: object
    select_prepared: object
    metadata: object
    gather: object
    attend: object
    swa: object
    project: object


@cache
def register_ops():
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def fake_metadata(pages, starts, positions, valid, page_size):
        columns = (positions.shape[0] + 128 + page_size - 1) // page_size + 1
        return pages.new_empty((pages.shape[0], columns + 2)), positions.new_empty(
            positions.shape
        )

    def fake_gather(q, words, scales, *_):
        return (
            torch.empty_like(q),
            words.new_empty((q.shape[0], 512, 128)),
            scales.new_empty((q.shape[0], 512)),
            words.new_empty(q.shape[:1]),
        )

    def fake_select(
        q,
        weights,
        words,
        pages,
        requests,
        lengths,
        candidates,
        page_records,
        storage_records,
        topk,
        block_size,
        publish,
        filter_candidates,
        layer_id,
    ):
        width = min(topk, pages.shape[1] * page_records)
        return lengths.new_empty((q.shape[0], width)), torch.empty_like(candidates)

    def fake_prepare(
        words, pages, requests, candidates, page_records, storage_records, block_size
    ):
        # Epic indexer_fp8.fake_prepare: one group per query bounds the pairing.
        tokens, count = candidates.shape
        key_pages = (2 * count * block_size + 127) // 128
        return (
            words.new_empty((tokens * key_pages * 33, 128)),
            candidates.new_empty(candidates.shape),
            requests.new_empty((tokens + 1,)),
            requests.new_empty((tokens,), dtype=torch.bool),
            requests.new_empty((tokens, 2)),
        )

    def fake_prepared(
        q,
        weights,
        lengths,
        keys,
        block_ids,
        offsets,
        valid,
        spans,
        score_width,
        topk,
        block_size,
        layer_id,
    ):
        return lengths.new_empty((q.shape[0], min(topk, score_width)))

    definitions = (
        ("select", select, fake_select),
        ("prepare_index", prepare_index, fake_prepare),
        ("select_prepared", select_index_prepared, fake_prepared),
        ("metadata", metadata, fake_metadata),
        ("gather", gather_attend, fake_gather),
        ("attend", attend, lambda q, *_: torch.empty_like(q)),
        ("swa", attend_swa, lambda q, *_: torch.empty_like(q)),
        ("project", project, lambda x, w, _: x.new_empty((x.shape[0], w.shape[1]))),
    )
    ops = []
    for name, function, fake in definitions:
        op = cpu_export_jax_op("sglang_tpu_pallas::dsv41_" + name, function)
        op.register_fake(fake)
        ops.append(op)
    return AttentionOps(*ops)


class CacheOps(NamedTuple):
    compressed: object
    index: object
    swa: object


@cache
def register_cache_ops():
    import jax

    from sglang_tpu.kernels.deepseek_v41.csa2.cache import (
        publish_csa2_kv,
        scatter_index_keys,
        write_swa_word_pool,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def compressed(
        words: jax.Array,
        scales: jax.Array,
        records: jax.Array,
        values: jax.Array,
        after: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        words, scales, _ = jax.lax.optimization_barrier((words, scales, after))
        return publish_csa2_kv(words, scales, records, values)

    def index(
        words: jax.Array,
        records: jax.Array,
        values: jax.Array,
        after: jax.Array,
        page_records: int,
    ) -> jax.Array:
        words, _ = jax.lax.optimization_barrier((words, after))
        return scatter_index_keys(words, records, values, page_records=page_records)

    def swa(
        words: jax.Array,
        records: jax.Array,
        values: jax.Array,
        page_size: int,
    ) -> jax.Array:
        return write_swa_word_pool(words, records, values, page_size=page_size)

    compressed_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_compressed",
        compressed,
        donate_argnums=(0, 1),
    )
    compressed_op.register_fake(
        lambda words, scales, *_: (torch.empty_like(words), torch.empty_like(scales))
    )
    index_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_index", index, donate_argnums=(0,)
    )
    index_op.register_fake(lambda words, *_: torch.empty_like(words))
    swa_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_publish_swa", swa, donate_argnums=(0,)
    )
    swa_op.register_fake(lambda words, *_: torch.empty_like(words))
    return CacheOps(compressed_op, index_op, swa_op)


def publish_source(ops, pool, layer_id, records, latent, key, after):
    """Preserve pool tensor identity and serialize reuse after its last gather."""
    words = pool.low_ratio_cache(layer_id)
    scales = pool.low_ratio_scales(layer_id)
    new_words, new_scales = ops.compressed(words, scales, records, latent, after)
    words.copy_(new_words)
    scales.copy_(new_scales)
    index = pool.low_ratio_cache(layer_id, index=True)
    index.copy_(
        ops.index(index, records, key, after, pool.index_page_records(layer_id))
    )
