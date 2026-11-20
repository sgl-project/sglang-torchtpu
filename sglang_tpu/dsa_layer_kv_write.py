# SPDX-License-Identifier: Apache-2.0
"""Inputs for filling a DSA layer's latent cache without running the layer.

A sparse MLA (DSA) layer stores its MLA latent inside the attention op. A pass
whose outputs are not needed for some rows can still have to leave those rows'
cache entries behind for later queries; ``mla_cache_latents`` computes the
latent such a pass must store, from core's own projection and norms.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


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
