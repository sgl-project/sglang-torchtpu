# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash model-config overrides on TPU, declared as
--json-model-override-args from TpuSRTPlatform.apply_server_args_defaults:

- qk_rope_head_dim: a zero rope segment for the NoPE checkpoint, which the TPU
  MLA kernels require; the KV pool is then sized for it.
- index_topk cleared under the dense opt-in, so the model serves as plain MLA.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

ARCH = "Glm5NextForConditionalGeneration"
# The TPU MLA kernels' tested rope width (DeepSeek).
ROPE_PAD = 64


def _arch(hf_config) -> Optional[str]:
    archs = getattr(hf_config, "architectures", None)
    return archs[0] if archs else None


def model_overrides(hf_config, dense_max_context: Optional[int]) -> dict[str, Any]:
    """Overrides for GLM-5.3-Flash ({} for other models); dense_max_context=None is DSA."""
    if _arch(hf_config) != ARCH:
        return {}
    text = {"qk_rope_head_dim": ROPE_PAD}
    if dense_max_context is not None:
        # is_deepseek_dsa reads index_topk from both config levels.
        text["index_topk"] = None
    top = dict(text)
    top["text_config"] = text
    return top


def _merge(user: dict, ours: dict) -> dict:
    """Ours over the user's, merging text_config."""
    merged = {**user, **ours}
    if isinstance(user.get("text_config"), dict) and "text_config" in ours:
        merged["text_config"] = {**user["text_config"], **ours["text_config"]}
    return merged


def apply_glm5_next_server_args(server_args) -> None:
    """Declare the overrides, and pallas_tpu_mla when dense."""
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
