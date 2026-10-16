# SPDX-License-Identifier: Apache-2.0
"""Model-config and server-arg adjustments for GLM-5.3-Flash on TPU.

Applied from TpuSRTPlatform.apply_server_args_defaults through
--json-model-override-args, so core's ModelConfig needs no TPU special case:

- The checkpoint is NoPE (qk_rope_head_dim = 0), and the TPU MLA kernels need a
  rope segment. The override sets a zero segment of ROPE_PAD lanes, which the
  model fills with zeros (q_pe . k_pe = 0), so the KV pool is sized and laid out
  for what the kernels read.
- SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT clears index_topk, so the model serves
  as plain MLA (pallas_tpu_mla) instead of DeepSeek Sparse Attention.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

ARCH = "Glm5NextForConditionalGeneration"
# Rope lanes of the zero segment; 64 is the TPU MLA kernels' tested DeepSeek shape.
ROPE_PAD = 64


def _arch(hf_config) -> Optional[str]:
    archs = getattr(hf_config, "architectures", None)
    return archs[0] if archs else None


def model_overrides(hf_config, dense_max_context: Optional[int]) -> dict[str, Any]:
    """--json-model-override-args entries for GLM-5.3-Flash on TPU, or {} for
    any other model. ``dense_max_context`` is the dense opt-in (None = DSA)."""
    if _arch(hf_config) != ARCH:
        return {}
    text = {"qk_rope_head_dim": ROPE_PAD}
    if dense_max_context is not None:
        # index_topk is what marks a DSA model (is_deepseek_dsa); it is read
        # from both the top-level and the text config.
        text["index_topk"] = None
    top = dict(text)
    top["text_config"] = text
    return top


def _merge(user: dict, ours: dict) -> dict:
    """Ours on top of the user's overrides, merging the nested text_config."""
    merged = {**user, **ours}
    if isinstance(user.get("text_config"), dict) and "text_config" in ours:
        merged["text_config"] = {**user["text_config"], **ours["text_config"]}
    return merged


def apply_glm5_next_server_args(server_args) -> None:
    """Declare the overrides (and the dense attention backend) for GLM-5.3-Flash."""
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.utils.hf_transformers.config import get_config

    cfg = resolving_view(server_args)
    try:
        hf_config = get_config(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.revision,
        )
    except Exception:
        # A model that cannot be loaded here fails later with its real error.
        return
    dense_max_context = tpu_envs.SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT.get()
    ours = model_overrides(hf_config, dense_max_context)
    if not ours:
        return
    index_topk = getattr(hf_config.get_text_config(), "index_topk", None)
    if dense_max_context is not None and index_topk and dense_max_context > index_topk:
        logger.warning(
            "GLM-5.3-Flash attends densely up to %d tokens, past index_topk=%d; "
            "outputs beyond index_topk do not match DSA",
            dense_max_context,
            index_topk,
        )
    if (
        dense_max_context is not None
        and cfg.attention_backend == "pallas_tpu_mla_sparse"
    ):
        raise ValueError(
            "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT serves GLM-5.3-Flash densely on "
            "--attention-backend pallas_tpu_mla, not pallas_tpu_mla_sparse."
        )
    user = json.loads(cfg.json_model_override_args or "{}")
    resolved = {"json_model_override_args": json.dumps(_merge(user, ours))}
    if dense_max_context is not None and cfg.attention_backend is None:
        resolved["attention_backend"] = "pallas_tpu_mla"
    declare_resolution(server_args, "tpu.apply_glm5_next_server_args", **resolved)
