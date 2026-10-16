# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash server args on TPU, from TpuSRTPlatform.apply_server_args_defaults.

The shared model runs mHC on its torch path, since the TileLang kernels are
CUDA-only. SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT opts into dense attention:
--json-model-override-args clears index_topk, so the model builds no indexer and
serves plain MLA on pallas_tpu_mla, matching DSA only up to index_topk tokens.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

ARCH = "Glm5NextForConditionalGeneration"


def _arch(hf_config) -> Optional[str]:
    archs = getattr(hf_config, "architectures", None)
    return archs[0] if archs else None


def dense_overrides(hf_config, dense_max_context: Optional[int]) -> dict[str, Any]:
    """Overrides for the dense opt-in ({} when off or for other models)."""
    if dense_max_context is None or _arch(hf_config) != ARCH:
        return {}
    # is_deepseek_dsa reads index_topk from both config levels.
    return {"index_topk": None, "text_config": {"index_topk": None}}


def _merge(user: dict, ours: dict) -> dict:
    """Ours over the user's, merging text_config."""
    merged = {**user, **ours}
    if isinstance(user.get("text_config"), dict) and "text_config" in ours:
        merged["text_config"] = {**user["text_config"], **ours["text_config"]}
    return merged


def _validate(cfg, text_config) -> None:
    if not cfg.disable_radix_cache:
        raise ValueError(
            "GLM-5.3-Flash on TPU currently requires --disable-radix-cache"
        )
    dp_attention = cfg.attn_dp_size > 1 or cfg.enable_dp_attention
    if cfg.speculative_algorithm or dp_attention or cfg.pp_size != 1:
        raise ValueError(
            "GLM-5.3-Flash on TPU supports TP/EP serving without speculative "
            "decoding, DP attention or PP"
        )
    if getattr(text_config, "index_kpool", 1) > 1 and not getattr(
        text_config, "index_kpool_always_select_tail", False
    ):
        raise ValueError(
            "The TPU pooled indexer always selects the tail pool; the checkpoint "
            "must set index_kpool_always_select_tail"
        )


def _use_torch_mhc() -> None:
    # The TileLang mHC kernels, and the cross-layer post+pre fusion on top of
    # them, are CUDA-only.
    from sglang.srt.environ import envs

    envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
    envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
    envs.SGLANG_OPT_FUSE_MHC_POST_PRE.set(False)


def apply_glm5_next_server_args(server_args) -> None:
    """Check GLM-5.3-Flash's TPU limits, pick the torch mHC path, declare dense."""
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
    if _arch(hf_config) != ARCH:
        return
    _validate(cfg, hf_config.get_text_config())
    _use_torch_mhc()

    dense_max_context = tpu_envs.SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT.get()
    ours = dense_overrides(hf_config, dense_max_context)
    if not ours:
        return
    if cfg.attention_backend == "pallas_tpu_mla_sparse":
        raise ValueError(
            "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT serves GLM-5.3-Flash densely on "
            "--attention-backend pallas_tpu_mla, not pallas_tpu_mla_sparse."
        )
    if cfg.context_length is not None and cfg.context_length > dense_max_context:
        raise ValueError(
            f"Dense GLM-5.3-Flash requires --context-length <= {dense_max_context}"
        )
    index_topk = getattr(hf_config.get_text_config(), "index_topk", None)
    if index_topk and dense_max_context > index_topk:
        logger.warning(
            "GLM-5.3-Flash attends densely up to %d tokens, past index_topk=%d; "
            "outputs beyond index_topk do not match DSA",
            dense_max_context,
            index_topk,
        )
    user = json.loads(cfg.json_model_override_args or "{}")
    resolved = {"json_model_override_args": json.dumps(_merge(user, ours))}
    if cfg.attention_backend is None:
        resolved["attention_backend"] = "pallas_tpu_mla"
    declare_resolution(server_args, "tpu.apply_glm5_next_server_args", **resolved)
