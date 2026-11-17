# SPDX-License-Identifier: Apache-2.0
"""GLM-specific TPU hooks; loader hook debt is documented in loader.py."""

from sglang.srt.plugins.hook_registry import HookType


def _select_glm_nextn_loader(result, load_config, model_config=None):
    # Keep loader/model imports out of plugin startup and the default-off path.
    from sglang_tpu.envs import envs

    if not envs.SGLANG_ENABLE_MTP_SHARD_FILTER.get():
        return result
    from sglang_tpu.glm.loader import select_glm_nextn_loader

    return select_glm_nextn_loader(result, load_config, model_config)


def _resolve_nextn_quant_config(result, model, config, quant_config):
    from sglang_tpu.glm.quantization import resolve_nextn_quant_config

    return resolve_nextn_quant_config(result, model, config, quant_config)


def hook_specs():
    return [
        (
            "sglang.srt.model_loader.loader.get_model_loader",
            _select_glm_nextn_loader,
            HookType.AFTER,
        ),
        (
            "sglang.srt.models.glm4_moe.GlmMoeDsaForCausalLMNextN._resolve_nextn_quant_config",
            _resolve_nextn_quant_config,
            HookType.AFTER,
        ),
    ]
