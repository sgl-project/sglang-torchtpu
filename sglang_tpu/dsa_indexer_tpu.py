# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer on TPU.

``Indexer.forward_tpu`` asks the attention backend to run the indexer, and the
backend calls ``indexer_forward`` in this module. It has two parts:

* In torch: the indexer's linear layers and the rotary embedding.
* In one custom op per layer: quantize the queries and keys, store the keys in
  the layer's index-key cache, and pick the top-k keys for each query. The JAX
  code for that op is in ``sglang_tpu.attention.pallas.dsa_index_topk``.

The attention backend owns the cache and the per-forward metadata and passes
them to the op. This module keeps no state except the registered op.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import TYPE_CHECKING

import torch

from sglang_tpu.pallas_export import (
    argnums,
    tpu_info_for_cpu_export,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

_PALLAS_DSA_INDEX_TOPK_OP = None
_PALLAS_DSA_KPOOL_TOPK_OP = None
_PALLAS_DSA_KPOOL_TOPK_SPLIT_OP = None
_PALLAS_DSA_KPOOL_EXPAND_OP = None


def build_dsa_index_topk_op():
    """Register the indexer step as a torch custom op, once per process."""
    global _PALLAS_DSA_INDEX_TOPK_OP
    if _PALLAS_DSA_INDEX_TOPK_OP is not None:
        return _PALLAS_DSA_INDEX_TOPK_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_index_topk import dsa_indexer_step

    op_name = "sglang_tpu_pallas::dsa_index_topk"

    # custom_jax_kernel reads this function's signature: every tensor has to
    # be a positional parameter annotated ``jax.Array``. The JAX step has
    # keyword-only arguments, so it cannot be passed in directly. This wrapper
    # adapts one to the other.
    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        head_weights: jax.Array,
        cache_k: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        weight_scale: float,
        topk: int,
        decode_query_size: int,
        segments: str,
        key_store: str,
    ) -> tuple[jax.Array, jax.Array]:
        return dsa_indexer_step(
            query,
            key,
            head_weights,
            cache_k,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            weight_scale=weight_scale,
            topk=topk,
            decode_query_size=decode_query_size,
            segments=segments,
            key_store=key_store,
        )

    (cache_argnum,) = argnums(_wrapped, "cache_k")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped,
            "weight_scale",
            "topk",
            "decode_query_size",
            "segments",
            "key_store",
        ),
        # The step updates the cache in place: its second output is the same
        # buffer as the cache_k input.
        donate_argnums=(cache_argnum,),
        input_output_aliases={cache_argnum: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("cache_k",),
        device_types=["tpu"],
    )
    def custom_op(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        weight_scale: float,
        topk: int,
        decode_query_size: int = 1,
        segments: str = "all",
        key_store: str = "xla",
    ) -> torch.Tensor:
        # The first call with a new shape compiles the kernel. In this process
        # JAX has no TPU device of its own (torch_tpu owns the device), so
        # Pallas cannot find out which TPU it is compiling for. This context
        # gives it the answer.
        with tpu_info_for_cpu_export():
            topk_indices, _updated_cache = jax_kernel(
                query,
                key,
                head_weights,
                cache_k,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                weight_scale,
                topk,
                decode_query_size,
                segments,
                key_store,
            )
        return topk_indices

    # Shape-only version for torch.compile tracing: one row of topk per token.
    def _fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        weight_scale: float,
        topk: int,
        decode_query_size: int = 1,
        segments: str = "all",
        key_store: str = "xla",
    ):
        return torch.empty(
            (query.shape[0], topk), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_INDEX_TOPK_OP = custom_op
    return custom_op


_PALLAS_DSA_INDEX_TOPK_SPLIT_OP = None


def build_dsa_index_topk_split_op():
    """Register the token-split indexer step as a torch custom op, once.

    DSA indexer token split: same as ``dsa_index_topk`` (it stores
    the keys of every token), but it scores and selects only the query rows
    ``[row_start, row_start + row_count)`` and returns ``i32[row_count,
    topk]``. ``row_start`` is a device tensor (this rank's TP rank times
    ``row_count``), so every rank runs one identical compiled program. A
    separate op, so the default path's op and its compiled programs do not
    change.
    """
    global _PALLAS_DSA_INDEX_TOPK_SPLIT_OP
    if _PALLAS_DSA_INDEX_TOPK_SPLIT_OP is not None:
        return _PALLAS_DSA_INDEX_TOPK_SPLIT_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_index_topk import dsa_indexer_step

    op_name = "sglang_tpu_pallas::dsa_index_topk_split"

    # Positional jax.Array tensors, as in build_dsa_index_topk_op.
    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        head_weights: jax.Array,
        cache_k: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        row_start: jax.Array,
        weight_scale: float,
        topk: int,
        decode_query_size: int,
        segments: str,
        key_store: str,
        row_count: int,
    ) -> tuple[jax.Array, jax.Array]:
        return dsa_indexer_step(
            query,
            key,
            head_weights,
            cache_k,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            row_start,
            weight_scale=weight_scale,
            topk=topk,
            decode_query_size=decode_query_size,
            segments=segments,
            key_store=key_store,
            row_count=row_count,
        )

    (cache_argnum,) = argnums(_wrapped, "cache_k")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped,
            "weight_scale",
            "topk",
            "decode_query_size",
            "segments",
            "key_store",
            "row_count",
        ),
        donate_argnums=(cache_argnum,),
        input_output_aliases={cache_argnum: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("cache_k",),
        device_types=["tpu"],
    )
    def custom_op(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        row_start: torch.Tensor,
        weight_scale: float,
        topk: int,
        decode_query_size: int,
        segments: str,
        key_store: str,
        row_count: int,
    ) -> torch.Tensor:
        # See build_dsa_index_topk_op: JAX needs the TPU info to compile here.
        with tpu_info_for_cpu_export():
            topk_indices, _updated_cache = jax_kernel(
                query,
                key,
                head_weights,
                cache_k,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                row_start,
                weight_scale,
                topk,
                decode_query_size,
                segments,
                key_store,
                row_count,
            )
        return topk_indices

    def _fake_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        cache_k: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        row_start: torch.Tensor,
        weight_scale: float,
        topk: int,
        decode_query_size: int,
        segments: str,
        key_store: str,
        row_count: int,
    ):
        return torch.empty((row_count, topk), dtype=torch.int32, device=query.device)

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_INDEX_TOPK_SPLIT_OP = custom_op
    return custom_op


def indexer_forward(
    indexer: "Indexer",
    *,
    backend: "TpuPallasSparseMlaAttnBackend",
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    layer_id: int,
) -> torch.Tensor:
    """Return the top-k key positions for each query token.

    The result is ``i32[num_tokens, index_topk]``; see
    ``TpuPallasSparseMlaAttnBackend.index_topk`` for the row format.

    The keys are stored without the Hadamard rotation. The fused CUDA writer,
    which is what this checkpoint family uses on GPU, skips it as well.
    """
    query, _ = indexer.wq_b(q_lora)
    query = query.view(-1, indexer.n_heads, indexer.head_dim)
    key = _project_index_key(indexer, x)

    # Only the first rope_dim dimensions of each vector are rotated; a NoPE
    # indexer (Kimi-K3.1) has no rotary embedding at all. index_key mirrors
    # this for the key alone; keep the two in step.
    if not indexer.skip_rope:
        rope_dim = indexer.rope_head_dim
        q_rope, k_rope = indexer.rotary_emb(
            positions, query[..., :rope_dim], key[..., :rope_dim]
        )
        # TODO(sglang-torchtpu#1072): write the rotated part back in place
        # instead of building new tensors with torch.cat.
        query = torch.cat([q_rope.to(query.dtype), query[..., rope_dim:]], dim=-1)
        key = torch.cat([k_rope.to(key.dtype), key[..., rope_dim:]], dim=-1)

    head_weights, _ = indexer.weights_proj(x)
    weight_scale = indexer.softmax_scale * indexer.n_heads**-0.5
    if indexer.bf16_head_gate:
        # The gate is rounded to bf16 after the n_heads**-0.5 factor.
        head_weights = (head_weights * indexer.n_heads**-0.5).to(torch.bfloat16)
        weight_scale = indexer.softmax_scale
    return backend.index_topk(
        forward_batch=forward_batch,
        layer_id=layer_id,
        query=query,
        key=key,
        head_weights=head_weights,
        weight_scale=weight_scale,
        key_store=key_store_mode(indexer, x.shape[0]),
    )


def _project_index_key(indexer: "Indexer", x: torch.Tensor) -> torch.Tensor:
    """The indexer's key before rope, ``[num_tokens, index_head_dim]``."""
    key, _ = indexer.wk(x)
    return indexer.k_norm(key)


def index_key(
    indexer: "Indexer", *, x: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """The key ``indexer_forward`` stores for each token, without its query side.

    For a writer that fills the index-key cache without scoring: same
    projection, same rotary module and the same rotated columns, so the key
    matches the one ``indexer_forward`` hands ``index_topk``.
    """
    key = _project_index_key(indexer, x)
    if indexer.skip_rope:
        return key
    rope_dim = indexer.rope_head_dim
    k_rope = key[..., :rope_dim]
    # The rotary module ropes a query and a key in one call. A copy stands in
    # for the query: aliasing the key would rotate it twice if rope is in place.
    _, k_rope = indexer.rotary_emb(positions, k_rope.clone(), k_rope)
    return torch.cat([k_rope.to(key.dtype), key[..., rope_dim:]], dim=-1)


# Batches up to this many tokens store their index keys with the Pallas
# writer, which walks the tokens one by one (about 1 us each); larger batches
# (prefill) keep the XLA whole-row scatter.
_PALLAS_KEY_STORE_MAX_TOKENS = 16


def key_store_mode(indexer: "Indexer", num_tokens: int) -> str:
    """The ``key_store`` of ``dsa_indexer_step`` for this layer and batch."""
    from sglang_tpu.envs import envs as tpu_envs

    if (
        num_tokens > _PALLAS_KEY_STORE_MAX_TOKENS
        or indexer.head_dim % 128
        or not tpu_envs.SGLANG_TPU_DSA_PALLAS_KEY_STORE.get()
    ):
        return "xla"
    return "pallas"


def pooled_index_key_and_gate(
    indexer: "IndexerKPool", x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pooled indexer's raw record inputs: the key (no rope) and the
    compression gate, each ``[num_tokens, index_head_dim]``.

    ``kpool_indexer_forward`` stores them; any other writer of the pooled
    caches must store the same.
    """
    gate = torch.nn.functional.linear(x, indexer.index_kpool_compress_gate)
    return _project_index_key(indexer, x), gate


def build_dsa_kpool_topk_op():
    """Register the pooled (kpool) indexer step as a torch custom op, once per process."""
    global _PALLAS_DSA_KPOOL_TOPK_OP
    if _PALLAS_DSA_KPOOL_TOPK_OP is not None:
        return _PALLAS_DSA_KPOOL_TOPK_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_kpool_topk import kpool_indexer_step

    op_name = "sglang_tpu_pallas::dsa_kpool_topk"

    # Same adapter as build_dsa_index_topk_op: positional jax.Array tensors.
    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        gate: jax.Array,
        head_weights: jax.Array,
        ape: jax.Array,
        raw_cache: jax.Array,
        pooled_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        raw_page_base: jax.Array,
        weight_scale: float,
        topk: int,
        kpool: int,
        out_width: int,
        decode_query_size: int,
        max_query_length: int,
        raw_ring_pages_cnt: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return kpool_indexer_step(
            query,
            key,
            gate,
            head_weights,
            ape,
            raw_cache,
            pooled_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            weight_scale=weight_scale,
            topk=topk,
            kpool=kpool,
            out_width=out_width,
            decode_query_size=decode_query_size,
            max_query_length=max_query_length or None,
            raw_page_base=raw_page_base,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
        )

    raw_argnum, pooled_argnum = argnums(_wrapped, "raw_cache", "pooled_cache")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped,
            "weight_scale",
            "topk",
            "kpool",
            "out_width",
            "decode_query_size",
            "max_query_length",
            "raw_ring_pages_cnt",
        ),
        # Both caches are updated in place: outputs 1 and 2 alias them.
        donate_argnums=(raw_argnum, pooled_argnum),
        input_output_aliases={raw_argnum: 1, pooled_argnum: 2},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("raw_cache", "pooled_cache"),
        device_types=["tpu"],
    )
    def custom_op(
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        head_weights: torch.Tensor,
        ape: torch.Tensor,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        raw_page_base: torch.Tensor,
        weight_scale: float,
        topk: int,
        kpool: int,
        out_width: int,
        decode_query_size: int = 1,
        max_query_length: int = 0,
        raw_ring_pages_cnt: int = 0,
    ) -> torch.Tensor:
        # See build_dsa_index_topk_op: JAX needs the TPU info to compile here.
        with tpu_info_for_cpu_export():
            topk_indices, _raw, _pooled = jax_kernel(
                query,
                key,
                gate,
                head_weights,
                ape,
                raw_cache,
                pooled_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                raw_page_base,
                weight_scale,
                topk,
                kpool,
                out_width,
                decode_query_size,
                max_query_length,
                raw_ring_pages_cnt,
            )
        return topk_indices

    def _fake_impl(query: torch.Tensor, *args, **kwargs):
        topk, kpool, out_width = args[12:15]
        width = max(out_width, topk + kpool - 1)
        return torch.empty(
            (query.shape[0], width), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_KPOOL_TOPK_OP = custom_op
    return custom_op


def build_dsa_kpool_topk_split_op():
    """Register the token-split pooled indexer step as a torch custom op, once.

    DSA indexer token split for the pooled indexer: same as
    ``dsa_kpool_topk`` (it stores every token's raw key and closes every pool),
    but it scores only the query rows ``[row_start, row_start + row_count)``
    and returns their picked pools, ``i32[row_count, topk // kpool]``. The
    backend puts the ranks' slices together, then ``dsa_kpool_expand`` turns
    the pools into token rows. A separate op, so the default path's op and its
    compiled programs do not change.
    """
    global _PALLAS_DSA_KPOOL_TOPK_SPLIT_OP
    if _PALLAS_DSA_KPOOL_TOPK_SPLIT_OP is not None:
        return _PALLAS_DSA_KPOOL_TOPK_SPLIT_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_kpool_topk import kpool_indexer_step

    op_name = "sglang_tpu_pallas::dsa_kpool_topk_split"

    # Positional jax.Array tensors, as in build_dsa_index_topk_op.
    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        gate: jax.Array,
        head_weights: jax.Array,
        ape: jax.Array,
        raw_cache: jax.Array,
        pooled_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        row_start: jax.Array,
        raw_page_base: jax.Array,
        weight_scale: float,
        topk: int,
        kpool: int,
        max_query_length: int,
        row_count: int,
        raw_ring_pages_cnt: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return kpool_indexer_step(
            query,
            key,
            gate,
            head_weights,
            ape,
            raw_cache,
            pooled_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            weight_scale=weight_scale,
            topk=topk,
            kpool=kpool,
            max_query_length=max_query_length or None,
            row_start=row_start,
            row_count=row_count,
            raw_page_base=raw_page_base,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
        )

    raw_argnum, pooled_argnum = argnums(_wrapped, "raw_cache", "pooled_cache")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped,
            "weight_scale",
            "topk",
            "kpool",
            "max_query_length",
            "row_count",
            "raw_ring_pages_cnt",
        ),
        donate_argnums=(raw_argnum, pooled_argnum),
        input_output_aliases={raw_argnum: 1, pooled_argnum: 2},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("raw_cache", "pooled_cache"),
        device_types=["tpu"],
    )
    def custom_op(
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        head_weights: torch.Tensor,
        ape: torch.Tensor,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        row_start: torch.Tensor,
        raw_page_base: torch.Tensor,
        weight_scale: float,
        topk: int,
        kpool: int,
        max_query_length: int,
        row_count: int,
        raw_ring_pages_cnt: int = 0,
    ) -> torch.Tensor:
        # See build_dsa_index_topk_op: JAX needs the TPU info to compile here.
        with tpu_info_for_cpu_export():
            picked, _raw, _pooled = jax_kernel(
                query,
                key,
                gate,
                head_weights,
                ape,
                raw_cache,
                pooled_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                row_start,
                raw_page_base,
                weight_scale,
                topk,
                kpool,
                max_query_length,
                row_count,
                raw_ring_pages_cnt,
            )
        return picked

    def _fake_impl(query: torch.Tensor, *args, **kwargs):
        topk, kpool, _max_query_length, row_count = args[13:17]
        return torch.empty(
            (row_count, topk // kpool), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_KPOOL_TOPK_SPLIT_OP = custom_op
    return custom_op


def build_dsa_kpool_expand_op():
    """Register ``kpool_expand_rows`` (picked pools -> token rows) as a torch
    custom op, once: the last step of the pooled indexer's token split.
    ``row_start`` (i32[1]) is the batch row of the first picked row: 0 for
    every rank's slices put together, the rank's window start under the
    attention token split."""
    global _PALLAS_DSA_KPOOL_EXPAND_OP
    if _PALLAS_DSA_KPOOL_EXPAND_OP is not None:
        return _PALLAS_DSA_KPOOL_EXPAND_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_kpool_topk import kpool_expand_rows

    op_name = "sglang_tpu_pallas::dsa_kpool_expand"

    def _wrapped(
        picked: jax.Array,
        kv_lens: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        row_start: jax.Array,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> jax.Array:
        return kpool_expand_rows(
            picked,
            kv_lens,
            cu_q_lens,
            distribution,
            row_start,
            topk=topk,
            kpool=kpool,
            out_width=out_width,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "topk", "kpool", "out_width"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=(), device_types=["tpu"])
    def custom_op(
        picked: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        row_start: torch.Tensor,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return jax_kernel(
                picked,
                kv_lens,
                cu_q_lens,
                distribution,
                row_start,
                topk,
                kpool,
                out_width,
            )

    def _fake_impl(
        picked: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        row_start: torch.Tensor,
        topk: int,
        kpool: int,
        out_width: int,
    ):
        width = max(out_width, topk + kpool - 1)
        return torch.empty(
            (picked.shape[0], width), dtype=torch.int32, device=picked.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_KPOOL_EXPAND_OP = custom_op
    return custom_op


# (shards, role) -> op: the split pooled indexer's two halves (kpool_indexer_local /
# kpool_merge_rows); the backend all-gathers the local scores between them.
_KPOOL_SHARD_OPS: dict = {}


def build_dsa_kpool_shard_ops(shards: int, role: int):
    """Register the split pooled indexer as two torch custom ops, once per process:
    ``local`` stores this rank's pools and returns its local top pools and their
    score bits; ``merge`` turns every rank's scores into this rank's token rows.
    ``role`` is static: each rank compiles its own programs."""
    key = (shards, role)
    if key in _KPOOL_SHARD_OPS:
        return _KPOOL_SHARD_OPS[key]

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_kpool_topk import (
        kpool_indexer_local,
        kpool_merge_rows,
    )

    local_name = f"sglang_tpu_pallas::dsa_kpool_topk_s{shards}r{role}"
    merge_name = f"sglang_tpu_pallas::dsa_kpool_merge_s{shards}r{role}"

    def _local(
        query: jax.Array,
        key: jax.Array,
        gate: jax.Array,
        head_weights: jax.Array,
        ape: jax.Array,
        raw_cache: jax.Array,
        pooled_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        raw_page_base: jax.Array,
        weight_scale: float,
        topk: int,
        kpool: int,
        decode_query_size: int,
        max_query_length: int,
        raw_ring_pages_cnt: int,
    ):
        return kpool_indexer_local(
            query,
            key,
            gate,
            head_weights,
            ape,
            raw_cache,
            pooled_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            weight_scale=weight_scale,
            topk=topk,
            kpool=kpool,
            shards=shards,
            role=role,
            decode_query_size=decode_query_size,
            max_query_length=max_query_length or None,
            raw_page_base=raw_page_base,
            raw_ring_pages_cnt=raw_ring_pages_cnt,
        )

    raw_argnum, pooled_argnum = argnums(_local, "raw_cache", "pooled_cache")
    local_kernel = _torch_pallas.custom_jax_kernel(
        _local,
        name=f"jax_{local_name}",
        static_argnums=argnums(
            _local,
            "weight_scale",
            "topk",
            "kpool",
            "decode_query_size",
            "max_query_length",
            "raw_ring_pages_cnt",
        ),
        # Both caches are updated in place: outputs 2 and 3 alias them.
        donate_argnums=(raw_argnum, pooled_argnum),
        input_output_aliases={raw_argnum: 2, pooled_argnum: 3},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        local_name, mutates_args=("raw_cache", "pooled_cache"), device_types=["tpu"]
    )
    def local_op(
        query: torch.Tensor,
        key: torch.Tensor,
        gate: torch.Tensor,
        head_weights: torch.Tensor,
        ape: torch.Tensor,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        raw_page_base: torch.Tensor,
        weight_scale: float,
        topk: int,
        kpool: int,
        decode_query_size: int,
        max_query_length: int,
        raw_ring_pages_cnt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with tpu_info_for_cpu_export():
            picked, scores, _raw, _pooled = local_kernel(
                query,
                key,
                gate,
                head_weights,
                ape,
                raw_cache,
                pooled_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                raw_page_base,
                weight_scale,
                topk,
                kpool,
                decode_query_size,
                max_query_length,
                raw_ring_pages_cnt,
            )
        return picked, scores

    def _local_fake(query: torch.Tensor, *args, **kwargs):
        topk, kpool = args[12:14]
        shape = (query.shape[0], topk // kpool)
        return (
            torch.empty(shape, dtype=torch.int32, device=query.device),
            torch.empty(shape, dtype=torch.int32, device=query.device),
        )

    local_op.register_fake(_local_fake)

    def _merge(
        all_scores: jax.Array,
        picked: jax.Array,
        kv_lens: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> jax.Array:
        return kpool_merge_rows(
            all_scores,
            picked,
            kv_lens,
            cu_q_lens,
            distribution,
            shards=shards,
            role=role,
            own_pools=own_pools,
            topk=topk,
            kpool=kpool,
            out_width=out_width,
        )

    merge_kernel = _torch_pallas.custom_jax_kernel(
        _merge,
        name=f"jax_{merge_name}",
        static_argnums=argnums(_merge, "own_pools", "topk", "kpool", "out_width"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(merge_name, mutates_args=(), device_types=["tpu"])
    def merge_op(
        all_scores: torch.Tensor,
        picked: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return merge_kernel(
                all_scores,
                picked,
                kv_lens,
                cu_q_lens,
                distribution,
                own_pools,
                topk,
                kpool,
                out_width,
            )

    def _merge_fake(
        all_scores: torch.Tensor,
        picked: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ):
        width = max(out_width, topk + kpool - 1)
        return torch.empty(
            (picked.shape[0], width), dtype=torch.int32, device=picked.device
        )

    merge_op.register_fake(_merge_fake)

    _KPOOL_SHARD_OPS[key] = (local_op, merge_op)
    return local_op, merge_op


# (shards, role) -> (cuts op, rows op): the split pooled indexer's prefill
# exchange (kpool_slice_cuts / kpool_cut_rows); the backend runs the all-to-alls.
_KPOOL_ROW_SPLIT_OPS: dict = {}


def build_dsa_kpool_row_split_ops(shards: int, role: int):
    """Register the split pooled indexer's prefill exchange as two torch custom
    ops, once per process: ``cuts`` turns every rank's local scores of this
    rank's row slice (``i32[shards, rows * n_pick]``) into each rank's cuts
    (``i32[shards, rows * 2]``); ``rows`` turns this rank's local pools and the
    cuts it received into its token rows, as the merge op would."""
    key = (shards, role)
    if key in _KPOOL_ROW_SPLIT_OPS:
        return _KPOOL_ROW_SPLIT_OPS[key]

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_kpool_topk import (
        kpool_cut_rows,
        kpool_slice_cuts,
    )

    cuts_name = f"sglang_tpu_pallas::dsa_kpool_cuts_s{shards}"
    rows_name = f"sglang_tpu_pallas::dsa_kpool_cut_rows_s{shards}r{role}"

    def _cuts(slice_scores: jax.Array, n_pick: int) -> jax.Array:
        cuts = kpool_slice_cuts(slice_scores.reshape(shards, -1, n_pick))
        return cuts.reshape(shards, -1)

    cuts_kernel = _torch_pallas.custom_jax_kernel(
        _cuts,
        name=f"jax_{cuts_name}",
        static_argnums=argnums(_cuts, "n_pick"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(cuts_name, mutates_args=(), device_types=["tpu"])
    def cuts_op(slice_scores: torch.Tensor, n_pick: int) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return cuts_kernel(slice_scores, n_pick)

    def _cuts_fake(slice_scores: torch.Tensor, n_pick: int):
        rows = slice_scores.shape[1] // n_pick
        return slice_scores.new_empty((shards, rows * 2))

    cuts_op.register_fake(_cuts_fake)

    def _rows(
        picked: jax.Array,
        scores: jax.Array,
        cuts: jax.Array,
        kv_lens: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> jax.Array:
        return kpool_cut_rows(
            picked,
            scores,
            cuts.reshape(-1, 2),
            kv_lens,
            cu_q_lens,
            distribution,
            shards=shards,
            role=role,
            own_pools=own_pools,
            topk=topk,
            kpool=kpool,
            out_width=out_width,
        )

    rows_kernel = _torch_pallas.custom_jax_kernel(
        _rows,
        name=f"jax_{rows_name}",
        static_argnums=argnums(_rows, "own_pools", "topk", "kpool", "out_width"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(rows_name, mutates_args=(), device_types=["tpu"])
    def rows_op(
        picked: torch.Tensor,
        scores: torch.Tensor,
        cuts: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ) -> torch.Tensor:
        with tpu_info_for_cpu_export():
            return rows_kernel(
                picked,
                scores,
                cuts,
                kv_lens,
                cu_q_lens,
                distribution,
                own_pools,
                topk,
                kpool,
                out_width,
            )

    def _rows_fake(
        picked: torch.Tensor,
        scores: torch.Tensor,
        cuts: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        own_pools: int,
        topk: int,
        kpool: int,
        out_width: int,
    ):
        width = max(out_width, topk + kpool - 1)
        return picked.new_empty((picked.shape[0], width))

    rows_op.register_fake(_rows_fake)

    _KPOOL_ROW_SPLIT_OPS[key] = (cuts_op, rows_op)
    return cuts_op, rows_op


def kpool_indexer_forward(
    indexer: "IndexerKPool",
    *,
    backend: "TpuPallasSparseMlaAttnBackend",
    x: torch.Tensor,
    q_lora: torch.Tensor,
    forward_batch: "ForwardBatch",
    layer_id: int,
) -> torch.Tensor:
    """Top-k token positions for each query of a pooled (kpool) indexer.

    GLM-5.3-Flash's indexer has no rope. The op Hadamard-rotates the query and
    each pooled key, as upstream's kpool path does.
    """
    query, _ = indexer.wq_b(q_lora)
    query = query.view(-1, indexer.n_heads, indexer.head_dim)
    key, gate = pooled_index_key_and_gate(indexer, x)
    head_weights, _ = indexer.weights_proj(x.float())
    return backend.index_topk_kpool(
        forward_batch=forward_batch,
        layer_id=layer_id,
        query=query,
        key=key,
        gate=gate,
        head_weights=head_weights,
        ape=indexer.index_kpool_compress_ape,
        weight_scale=indexer.softmax_scale * indexer.n_heads**-0.5,
        kpool=indexer.index_kpool,
    )
