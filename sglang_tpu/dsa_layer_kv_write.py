# SPDX-License-Identifier: Apache-2.0
"""Fill a DSA layer's KV caches without running the layer.

A sparse MLA (DSA) layer stores two things per token on each forward pass: its
MLA latent, which the attention op writes before it attends, and its index
key, which the indexer op writes before it scores (for the pooled indexer, a
raw key | gate record, and the pools the step closes). A pass whose outputs
are not needed for some rows can still have to leave those rows' cache entries
behind for later queries; ``write_dsa_layer_kv`` stores exactly those entries,
in one op, without the query side, the attention, the scoring or the MLP.

Every value comes from the code the full layer runs: core's latent projection
and norms, the plugin's index-key projections, the backend's store form and
the full steps' writers (``attention.pallas.dsa_layer_kv_store``).
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang_tpu.dsa_indexer_tpu import (
    index_key,
    key_store_mode,
    pooled_index_key_and_gate,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend


def mla_cache_latents(
    attn: "DeepseekV2AttentionMLA",
    *,
    qkv_latent: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The latent ``forward_absorb_prepare`` hands the backend, without its query.

    ``qkv_latent`` is the attention input's fused projection,
    ``[T, q_lora_rank + kv_lora_rank + qk_rope_head_dim]``, as
    ``get_attn_tp_context().fetch_qkv_latent()`` returns it. Returns
    ``(k_nope [T, 1, kv_lora_rank], k_pe [T, 1, qk_rope_head_dim])``: the
    normed latent and its rotated rope part, the ``k`` / ``k_rope`` the
    backend's ``_forward_mla`` receives. The query columns are dropped, not
    recomputed, so the projection is the full step's GEMM.

    Mirrors ``DeepseekV2AttentionMLA.forward_absorb_prepare`` on the TPU path.
    Out of contract (the caller must keep the full layer): no ``q_lora_rank``,
    context parallelism (core gathers or materializes the latent), and model
    classes with their own ``forward_absorb_prepare``.
    """
    _, latent = qkv_latent.split(
        [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1
    )
    k_nope = attn.kv_a_layernorm(latent[..., : attn.kv_lora_rank]).unsqueeze(1)
    k_pe = latent[..., attn.kv_lora_rank :].unsqueeze(1)
    if attn.rotary_emb is not None:
        # The rotary module ropes a query and a key in one call. A copy stands
        # in for the query: aliasing the key would rotate it twice if rope is
        # in place. No cast back: core passes rope's output dtype on as is.
        _, k_pe = attn.rotary_emb(positions, k_pe.clone(), k_pe)
    return k_nope, k_pe


def dsa_layer_kv_unsupported_reason(
    *, attn: "DeepseekV2AttentionMLA", backend: object
) -> Optional[str]:
    """Why ``write_dsa_layer_kv`` cannot fill this layer's caches, or None.

    Each reason names a path on which the full layer stores something other
    than what this module computes, so the caller must keep the full layer.
    """
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

    if not isinstance(attn, DeepseekV2AttentionMLA):
        return f"the layer's attention is {type(attn).__name__}, not DeepseekV2AttentionMLA"
    # mla_cache_latents mirrors core's k path, not an override of it.
    if (
        type(attn).forward_absorb_prepare
        is not DeepseekV2AttentionMLA.forward_absorb_prepare
    ):
        return f"{type(attn).__name__} overrides forward_absorb_prepare"
    if not isinstance(backend, TpuPallasSparseMlaAttnBackend):
        name = type(backend).__name__
        return f"the attention backend is {name}, not pallas_tpu_mla_sparse"
    if not attn.use_dsa:
        return "the layer is not a DSA layer"
    if attn.q_lora_rank is None:
        return "the layer has no q_lora_rank, so no fused qkv_a projection"
    if attn.skip_topk and not attn.is_nextn:
        return "the layer reuses another layer's top-k and has no index-key cache"
    # Fail-safes: the platform allows neither with this backend today, and
    # either would need a sharded or materialized latent store.
    if get_parallel().dcp_enabled:
        return "decode context parallelism is on"
    if get_parallel().attn_cp_size > 1:
        return "prefill context parallelism is on"
    return None


def write_dsa_layer_kv(
    attn: "DeepseekV2AttentionMLA",
    *,
    backend: "TpuPallasSparseMlaAttnBackend",
    forward_batch: "ForwardBatch",
    qkv_latent: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Store this forward pass's latents and index keys of one DSA layer.

    ``qkv_latent`` is as in ``mla_cache_latents``; ``hidden_states`` is the
    attention input the indexer reads (``x`` of ``indexer_forward``). The
    backend's metadata must be initialized for ``forward_batch``. The caches
    end up as a full forward of the layer that runs its own indexer leaves
    them (a prompt pass; not a draft step that reuses carried top-k, which
    stores no index keys). The caller checks ``dsa_layer_kv_unsupported_reason``
    once, at build time.
    """
    k, k_rope = mla_cache_latents(attn, qkv_latent=qkv_latent, positions=positions)
    indexer = attn.indexer
    if isinstance(indexer, IndexerKPool):
        keys, gates = pooled_index_key_and_gate(indexer, hidden_states)
        backend.store_layer_kv_pooled(
            forward_batch=forward_batch,
            layer=attn.attn_mqa,
            index_layer_id=attn.layer_id,
            k=k,
            k_rope=k_rope,
            keys=keys,
            gates=gates,
            ape=indexer.index_kpool_compress_ape,
        )
        return
    backend.store_layer_kv(
        forward_batch=forward_batch,
        layer=attn.attn_mqa,
        index_layer_id=attn.layer_id,
        k=k,
        k_rope=k_rope,
        index_keys=index_key(indexer, x=hidden_states, positions=positions),
        key_store=key_store_mode(indexer, hidden_states.shape[0]),
    )
