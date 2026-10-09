# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer on TPU: its forward, and the custom op behind it.

``Indexer.forward_tpu`` lands here. The projections and the rotary embedding
stay ordinary torch; everything after them is one custom op per layer
(``sglang.srt.layers.attention.pallas.dsa_index_topk``): quantise the queries
and keys, append the keys to the layer's index-key cache, select top-k.

The attention backend supplies the metadata and the cache, so this module owns
no state beyond the registered op.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from sglang.srt.model_executor.forward_context import get_attn_backend

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

_PALLAS_DSA_INDEX_TOPK_OP = None


def build_dsa_index_topk_op():
    """Lazily register the indexer's device step as a torch custom op."""
    global _PALLAS_DSA_INDEX_TOPK_OP
    if _PALLAS_DSA_INDEX_TOPK_OP is not None:
        return _PALLAS_DSA_INDEX_TOPK_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.dsa_index_topk import dsa_indexer_step

    op_name = "sglang_tpu_pallas::dsa_index_topk"

    def _wrapped(
        query: jax.Array,
        key: jax.Array,
        head_weights: jax.Array,
        cache_k: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        weight_scale: float = 1.0,
        topk: int = 0,
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

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(8, 9),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
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
        weight_scale: float = 1.0,
        topk: int = 0,
    ) -> torch.Tensor:
        # JAX is CPU-only here, so Mosaic needs the TPU generation patched in.
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

    def _fake_impl(query: torch.Tensor, *args, **kwargs):
        topk = kwargs["topk"] if "topk" in kwargs else args[8]
        return torch.empty(
            (query.shape[0], topk), dtype=torch.int32, device=query.device
        )

    custom_op.register_fake(_fake_impl)

    _PALLAS_DSA_INDEX_TOPK_OP = custom_op
    return custom_op


def indexer_forward(
    indexer: "Indexer",
    *,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Top-k key positions for each query token, ``i32[num_tokens, index_topk]``.

    No Hadamard rotation, as in the fused CUDA writer this checkpoint family
    takes on GPU.
    """
    query, _ = indexer.wq_b(q_lora)
    query = query.view(-1, indexer.n_heads, indexer.head_dim)
    key, _ = indexer.wk(x)
    key = indexer.k_norm(key)

    rope_dim = indexer.rope_head_dim
    q_rope, k_rope = indexer.rotary_emb(
        positions, query[..., :rope_dim], key[..., :rope_dim]
    )
    query = torch.cat([q_rope.to(query.dtype), query[..., rope_dim:]], dim=-1)
    key = torch.cat([k_rope.to(key.dtype), key[..., rope_dim:]], dim=-1)

    head_weights, _ = indexer.weights_proj(x)
    return get_attn_backend().index_topk(
        layer_id=layer_id,
        query=query,
        key=key,
        head_weights=head_weights,
        weight_scale=indexer.softmax_scale * indexer.n_heads**-0.5,
    )
