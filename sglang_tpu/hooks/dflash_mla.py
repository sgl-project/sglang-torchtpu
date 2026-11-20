# SPDX-License-Identifier: Apache-2.0
"""Serve MLA DFlash drafts (``dflash_config.attention_mode == "mla"``) as MHA.

select_dflash_mla_attention
  Why: core ``DFlashDecoderLayer`` always builds the GQA ``DFlashAttention``;
       an MLA draft (Kimi-K3.1 DFlash) has DeepSeek MLA weights. The TPU draft
       pool is the fused per-layer MHA pool, so the plugin serves the MLA draft
       as MHA over the expanded latent (``sglang_tpu.models.dflash_mla``).
  How: AROUND DFlashDecoderLayer.__init__: for an MLA draft config, set the
       instance's ``attention_cls`` to ``DFlashMLAAttention`` before the
       original constructor reads it.
  Upstream PR or issue: not filed yet (seam: pick the draft attention class
       from ``dflash_config.attention_mode``; GPU sglang has its own absorbed
       ``DFlashMLAAttention`` on the MLA pool).
  Remove when: core selects the draft attention class through a seam the
       platform can answer.

dflash_mla_head_dims
  Why: an MLA draft config's ``head_dim`` is the rope dim (64). The draft pool,
       its byte budget and the RPA backend need the expanded MHA dims
       (qk 192, v 128).
  How: AFTER ModelConfig._derive_model_shapes: for a DFlash MLA draft, set
       head_dim / v_head_dim (and the SWA copies) to the expanded dims, and
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
    from sglang_tpu.models.dflash_mla import DFlashMLAAttention, is_dflash_mla_draft

    # Subclasses with their own attention class (Laguna) keep it.
    if type(layer).attention_cls is DFlashAttention and is_dflash_mla_draft(config):
        # nn.Module.__init__ has not run yet; a plain instance attribute shadows the class one.
        object.__setattr__(layer, "attention_cls", DFlashMLAAttention)
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
    head_dim, v_head_dim = mla_draft_head_dims(model_config.hf_text_config)
    model_config.head_dim = model_config.swa_head_dim = head_dim
    model_config.v_head_dim = model_config.swa_v_head_dim = v_head_dim
    return result
