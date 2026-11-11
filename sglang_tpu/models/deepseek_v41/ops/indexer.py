# SPDX-License-Identifier: Apache-2.0
"""Torch ops over the DeepSeek-V4.1 packed FP8 indexer (ratio 1 and 2 layers).

Single bank, one device per process (no mesh). Each op is a
``cpu_export_jax_op`` over the shared kernels in
``sglang_tpu.kernels.deepseek_v41.index_selection``. ``write_index_k`` donates
the cache and returns its new value; the caller does ``index_k.copy_(result)``.

Layout (``memory_pool.DSV41TPUTokenToKVPool.get_index_k``): INT32
``[P * 33, 128]``, ``PackedFP8Layout(P, 33, 0, 128)``. Record ``full_loc // r``
lives on indexer page ``record // 128``; the per-request indexer page table is
``expand_index_page_table`` of the full page table for ratio 1 (full page p ->
indexer pages 2p, 2p + 1) and the full page table itself for ratio 2.

Numerics follow the vllm-torchtpu epic (``indexer_fp8._quantize_query``,
``_produce_index_k``): Q and K are RoPE'd and rounded to BF16, then quantized
per 128-vector to E4M3 with a power-of-two scale. K scales live in the cache;
the Q scale is folded into the FP32 head weights once, inside ``select``.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

from threading import Lock

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
    write_fp8_records,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
    select_fp8_indexer_jax,
)

INDEX_TOPK = 512
CANDIDATE_BLOCK_SIZE = 8

_OP_PREFIX = "sglang_tpu_pallas::dsv41_indexer_"
_OPS: dict = {}
_OPS_LOCK = Lock()


# =============================================================================
# JAX bridges (traced by cpu_export_jax_op; non-array arguments are static)
# =============================================================================


def _write_index_k(
    index_k: jax.Array,
    records: jax.Array,
    keys: jax.Array,
    pages: int,
    page_stride: int,
    page_offset: int,
    page_size: int,
) -> jax.Array:
    layout = PackedFP8Layout(pages, page_stride, page_offset, page_size)
    return write_fp8_records((index_k,), records, keys.astype(jnp.bfloat16), layout)[0]


def _compact(ids: jax.Array) -> jax.Array:
    """Valid (>= 0) ids ascending, then -1."""
    limit = jnp.iinfo(jnp.int32).max
    ids = jnp.sort(jnp.where(ids >= 0, ids, limit), axis=-1)
    return jnp.where(ids == limit, -1, ids)


def _visible_ids(lengths: jax.Array, count: int) -> jax.Array:
    ids = jnp.arange(count, dtype=jnp.int32)[None, :]
    return jnp.where(ids < lengths[:, None], ids, -1)


def _select(
    q: jax.Array,
    weights: jax.Array,
    index_k: jax.Array,
    index_pages: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    pages: int,
    page_stride: int,
    page_offset: int,
    page_size: int,
    topk: int,
    block_size: int,
    publish: bool,
    consume: bool,
) -> tuple[jax.Array, jax.Array]:
    layout_args = (pages, page_stride, page_offset, page_size)
    tokens, count = candidates.shape
    width = index_pages.shape[1] * page_size
    k = min(topk, width)
    if tokens == 0:
        return jnp.empty((0, k), jnp.int32), candidates
    live = (requests >= 0) & (requests < index_pages.shape[0])
    lengths = jnp.where(live, jnp.clip(lengths, 0, width), 0)
    batch_max = lengths.max().reshape(1)
    # The candidate buffer covers every block: publication and filtering are
    # identities and the buffer passes through, as in the epic.
    filters = (publish or consume) and count * block_size < width

    def every_visible_id():
        published = candidates
        if publish and filters:
            published = _visible_ids(-(-lengths // block_size), count)
        return _visible_ids(lengths, k), published

    def scored():
        q8, q_scales, _ = quantize_index_rows(q)
        selected, published = select_fp8_indexer_jax(
            q8,
            (index_k,),
            index_pages,
            weights.astype(jnp.float32) * q_scales,
            lengths,
            candidates,
            requests,
            batch_max,
            layout_args,
            topk,
            block_size,
            publish,
            consume,
            True,
            width,
        )
        return _compact(selected[:, :k]), published

    # A source whose buffer cannot hold every block of a top-k-long context
    # must rank its blocks even when the selection itself is short.
    if publish and filters and count * block_size < topk:
        return scored()
    # Batch-wide short-context rule of the epic (`_all_indices_fit` statically,
    # `batch_max <= topk` at run time): every visible id, no scoring.
    return jax.lax.cond(batch_max[0] <= topk, every_visible_id, scored)


# =============================================================================
# Fakes and registration
# =============================================================================


def _fake_write_index_k(index_k, records, keys, *_):
    return torch.empty_like(index_k)


def _fake_select(
    q, weights, index_k, index_pages, requests, lengths, candidates, *static_args
):
    page_size, topk = static_args[3], static_args[4]
    width = index_pages.shape[1] * page_size
    return (
        q.new_empty((q.shape[0], min(topk, width)), dtype=torch.int32),
        torch.empty_like(candidates),
    )


# name -> (JAX function, fake, donated argument positions)
_DEFINITIONS = {
    "write_index_k": (_write_index_k, _fake_write_index_k, (0,)),
    "select": (_select, _fake_select, ()),
}


def _op(name: str):
    op = _OPS.get(name)
    if op is not None:
        return op
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OPS_LOCK:
        op = _OPS.get(name)
        if op is None:
            function, fake, donated = _DEFINITIONS[name]
            op = cpu_export_jax_op(_OP_PREFIX + name, function, donate_argnums=donated)
            op.register_fake(fake)
            _OPS[name] = op
    return op


def register_indexer_ops() -> None:
    """Register every op; call before torch.compile traces the model."""
    for name in _DEFINITIONS:
        _op(name)


# =============================================================================
# Public interface
# =============================================================================


def _i32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(torch.int32)


def head_weights(
    weights_proj_out: torch.Tensor, *, n_heads: int, head_dim: int
) -> torch.Tensor:
    """FP32 ``weights_proj(x) * head_dim**-0.5 * n_heads**-0.5`` [T, H].

    FP32 and two roundings as in the epic (``attention._prepare_tpu``); core
    ``DeepseekV41Indexer.head_weights`` scales once, in BF16.
    """
    return weights_proj_out.float() * head_dim**-0.5 * n_heads**-0.5


def write_index_k(
    index_k: torch.Tensor,
    records: torch.Tensor,
    keys: torch.Tensor,
    layout: tuple[int, int, int, int],
) -> torch.Tensor:
    """Quantize normed, RoPE'd BF16 ``keys`` [N, 128] and write them at ``records``.

    ``records`` [N] i32 is ``full_loc // r`` per compressed key; -1 skips, and
    valid records must be unique. ``layout`` is ``pool.index_k_layout(layer)``.
    Returns the new ``index_k``; the caller does ``index_k.copy_(...)``.
    """
    return _op("write_index_k")(
        index_k, _i32(records), keys.reshape(-1, PAGE_RECORDS), *layout
    )


def select(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    index_pages: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    layout: tuple[int, int, int, int],
    *,
    topk: int = INDEX_TOPK,
    block_size: int = CANDIDATE_BLOCK_SIZE,
    publish: bool = False,
    consume: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``topk`` logical compressed ids per query from the packed FP8 cache.

    ``q`` [T, 32, 128] BF16, RoPE'd, not quantized (the op quantizes it).
    ``weights`` [T, 32] FP32 from `head_weights`, without any Q scale.
    ``index_pages`` [B, C'] i32 indexer page table; the score width is
    ``C' * 128`` compressed positions. ``requests`` [T] i32 row into
    ``index_pages``, -1 on padding rows, each request's rows contiguous.
    ``lengths`` [T] i32 visible compressed keys, ``(pos + 1) // r``.
    ``candidates`` [T, K] i32 candidate block ids (``K`` may be 0 off the
    candidate scheme); ``publish`` for the candidate-source layer,
    ``consume`` for later index layers.

    Returns ``selected`` [T, min(topk, C' * 128)] i32 ids in ``[0, length)``,
    ascending, -1 padded; and ``blocks`` [T, K] i32, the published block ids
    (unordered, -1 padded) under ``publish``, else ``candidates`` unchanged.
    When every live length is at most ``topk`` the selection is every visible
    id and ``blocks`` every visible block, with no scoring.
    """
    assert not (publish and consume), "a candidate source does not filter"
    return _op("select")(
        q,
        weights,
        index_k,
        _i32(index_pages),
        _i32(requests),
        _i32(lengths).reshape(-1),
        _i32(candidates),
        *layout,
        topk,
        block_size,
        publish,
        consume,
    )


def index_k_rows(pages: int) -> int:
    """INT32 rows of a ``pages``-page cache."""
    return pages * PAGE_ROWS
