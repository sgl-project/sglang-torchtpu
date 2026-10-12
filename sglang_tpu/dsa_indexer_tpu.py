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
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

_PALLAS_DSA_INDEX_TOPK_OP = None


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
        )

    (cache_argnum,) = argnums(_wrapped, "cache_k")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "weight_scale", "topk"),
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
    ):
        return torch.empty(
            (query.shape[0], topk), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_INDEX_TOPK_OP = custom_op
    return custom_op


def indexer_forward(
    indexer: "Indexer",
    *,
    backend: "TpuPallasSparseMlaAttnBackend",
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
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

    # Only the first rope_dim dimensions of each vector are rotated.
    rope_dim = indexer.rope_head_dim
    q_rope, k_rope = indexer.rotary_emb(
        positions, query[..., :rope_dim], key[..., :rope_dim]
    )
    # TODO(sglang-torchtpu#1072): write the rotated part back in place instead
    # of building new tensors with torch.cat.
    query = torch.cat([q_rope.to(query.dtype), query[..., rope_dim:]], dim=-1)
    key = torch.cat([k_rope.to(key.dtype), key[..., rope_dim:]], dim=-1)

    head_weights, _ = indexer.weights_proj(x)
    return backend.index_topk(
        layer_id=layer_id,
        query=query,
        key=key,
        head_weights=head_weights,
        weight_scale=indexer.softmax_scale * indexer.n_heads**-0.5,
    )
