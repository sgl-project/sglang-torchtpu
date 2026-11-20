# SPDX-License-Identifier: Apache-2.0
"""Serve MLA DFlash drafts (``dflash_config.attention_mode == "mla"``) as MHA.

select_dflash_mla_attention
  Why: for an MLA draft (Kimi-K3.1 DFlash) core ``DFlashDraftModel`` builds its
       absorbed ``DFlashMLAAttention`` on the MLA pool. The TPU draft pool is
       the fused per-layer MHA pool, so the plugin serves the MLA draft as MHA
       over the expanded latent (``sglang_tpu.models.dflash_mla``).
  How: AROUND DFlashDecoderLayer.__init__: for an MLA draft config, replace
       the ``attention_cls`` argument core passes (core ``DFlashMLAAttention``
       or ``DFlashAttention``) with the TPU ``DFlashMLAAttention``.
  Upstream PR or issue: not filed yet (seam: pick the draft attention class
       from ``dflash_config.attention_mode``; GPU sglang has its own absorbed
       ``DFlashMLAAttention`` on the MLA pool).
  Remove when: core selects the draft attention class through a seam the
       platform can answer.

dflash_mla_head_dims
  Why: core derives an MLA draft's shapes as MLA (attention_arch MLA, latent
       head dims). The draft pool, its byte budget and the RPA backend need MHA
       with the expanded dims (qk 192, v 128).
  How: AFTER ModelConfig._derive_model_shapes: for a DFlash MLA draft, set
       attention_arch to MHA and head_dim / v_head_dim (and the SWA copies) to
       the expanded dims, and
       refuse a config whose KV head count differs from its attention heads
       (DFlashMLAAttention serves one KV head per attention head).
  Upstream PR or issue: same seam as above.
  Remove when: as above.
"""

from __future__ import annotations

from typing import Any


def select_dflash_mla_attention(original, layer: Any, config, *args, **kwargs):
    """AROUND sglang.srt.models.dflash.DFlashDecoderLayer.__init__."""
    from sglang.srt.models.dflash import DFlashAttention
    from sglang.srt.models.dflash import DFlashMLAAttention as CoreDFlashMLAAttention
    from sglang_tpu.models.dflash_mla import DFlashMLAAttention, is_dflash_mla_draft

    if not is_dflash_mla_draft(config):
        return original(layer, config, *args, **kwargs)
    # Core passes attention_cls by keyword; the signature allows it positionally
    # after layer_id. Subclasses with their own attention class (Laguna) keep it.
    if "attention_cls" in kwargs:
        if kwargs["attention_cls"] in (DFlashAttention, CoreDFlashMLAAttention):
            kwargs["attention_cls"] = DFlashMLAAttention
    elif len(args) >= 2 and args[1] in (DFlashAttention, CoreDFlashMLAAttention):
        args = (args[0], DFlashMLAAttention, *args[2:])
    return original(layer, config, *args, **kwargs)


def dflash_mla_head_dims(result, model_config: Any, *args, **kwargs):
    """AFTER sglang.srt.configs.model_config.ModelConfig._derive_model_shapes."""
    from sglang_tpu.models.dflash_mla import is_dflash_mla_draft, mla_draft_head_dims

    archs = model_config.hf_config.architectures or []
    if not any(a.startswith("DFlash") for a in archs):
        return result
    if not is_dflash_mla_draft(model_config.hf_config):
        return result
    # DFlashMLAAttention runs every head as its own KV head, while the draft KV
    # pool is sized from the config's KV head count: the two must agree.
    num_heads = int(model_config.hf_text_config.num_attention_heads)
    num_kv_heads = int(model_config.get_total_num_kv_heads())
    if num_kv_heads != num_heads:
        raise ValueError(
            "A DFLASH MLA draft runs every attention head as its own KV head, so "
            "its config needs num_key_value_heads == num_attention_heads; got "
            f"{num_kv_heads} KV heads for {num_heads} attention heads."
        )
    from sglang.srt.configs.model_config import AttentionArch

    model_config.attention_arch = AttentionArch.MHA
    head_dim, v_head_dim = mla_draft_head_dims(model_config.hf_text_config)
    model_config.head_dim = model_config.swa_head_dim = head_dim
    model_config.v_head_dim = model_config.swa_v_head_dim = v_head_dim
    return result
