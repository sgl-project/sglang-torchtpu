# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer on TPU.

``Indexer.forward_tpu`` asks the attention backend to run the indexer, and the
backend calls ``indexer_forward`` in this module. It has two parts:

* In torch: the indexer's linear layers and the rotary embedding.
* In one custom op per layer: quantize the queries and keys, store the keys in
  the layer's index-key cache, and pick the top-k keys for each query. The JAX
  code for that op is in ``sglang_tpu.kernels.ops.attention.dsa_index_topk``.

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


def build_dsa_index_topk_op():
    """Register the indexer step as a torch custom op, once per process."""
    global _PALLAS_DSA_INDEX_TOPK_OP
    if _PALLAS_DSA_INDEX_TOPK_OP is not None:
        return _PALLAS_DSA_INDEX_TOPK_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.kernels.ops.attention.dsa_index_topk import dsa_indexer_step

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

    SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT: same as ``dsa_index_topk`` (it stores
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

    from sglang_tpu.kernels.ops.attention.dsa_index_topk import dsa_indexer_step

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
    key, _ = indexer.wk(x)
    key = indexer.k_norm(key)

    # Only the first rope_dim dimensions of each vector are rotated; a NoPE
    # indexer (Kimi-K3.1) has no rotary embedding at all.
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
        key_store=_key_store_mode(indexer, x.shape[0]),
    )


# Batches up to this many tokens store their index keys with the Pallas
# writer, which walks the tokens one by one (about 1 us each); larger batches
# (prefill) keep the XLA whole-row scatter.
_PALLAS_KEY_STORE_MAX_TOKENS = 16


def _key_store_mode(indexer: "Indexer", num_tokens: int) -> str:
    """The ``key_store`` of ``dsa_indexer_step`` for this layer and batch."""
    from sglang_tpu.envs import envs as tpu_envs

    if (
        num_tokens > _PALLAS_KEY_STORE_MAX_TOKENS
        or indexer.head_dim % 128
        or not tpu_envs.SGLANG_TPU_DSA_PALLAS_KEY_STORE.get()
    ):
        return "xla"
    return "pallas"


def build_dsa_kpool_topk_op():
    """Register the pooled (kpool) indexer step as a torch custom op, once per process."""
    global _PALLAS_DSA_KPOOL_TOPK_OP
    if _PALLAS_DSA_KPOOL_TOPK_OP is not None:
        return _PALLAS_DSA_KPOOL_TOPK_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.kernels.ops.attention.dsa_kpool_topk import kpool_indexer_step

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
        weight_scale: float,
        topk: int,
        kpool: int,
        out_width: int,
        decode_query_size: int,
        max_query_length: int,
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
        weight_scale: float,
        topk: int,
        kpool: int,
        out_width: int,
        decode_query_size: int = 1,
        max_query_length: int = 0,
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
                weight_scale,
                topk,
                kpool,
                out_width,
                decode_query_size,
                max_query_length,
            )
        return topk_indices

    def _fake_impl(query: torch.Tensor, *args, **kwargs):
        topk, kpool, out_width = args[11:14]
        width = max(out_width, topk + kpool - 1)
        return torch.empty(
            (query.shape[0], width), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_KPOOL_TOPK_OP = custom_op
    return custom_op


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
    key, _ = indexer.wk(x)
    key = indexer.k_norm(key)
    gate = torch.nn.functional.linear(x, indexer.index_kpool_compress_gate)
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
