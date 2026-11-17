# SPDX-License-Identifier: Apache-2.0
"""GLM NextN quantization naming adaptations."""

import copy

from sglang.srt.layers.quantization.fp8 import Fp8Config


def _remap_glm_nextn_ignored_layers(config: Fp8Config, hf_config) -> None:
    """Use the model's mapping for checkpoint-named GLM NextN exclusions."""
    if "GlmMoeDsaForCausalLMNextN" not in (
        getattr(hf_config, "architectures", None) or []
    ):
        return
    from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

    num_layers = hf_config.num_hidden_layers
    layer_id = 0 if num_layers == 1 else num_layers
    prefix = f"model.layers.{layer_id}"
    mapped = []
    for name in config.ignored_layers:
        normalized = name if name.startswith("model.") else f"model.{name}"
        if normalized == prefix or normalized.startswith(prefix + "."):
            mapped.append(
                GlmMoeDsaForCausalLMNextN._map_mtp_ckpt_name(normalized, prefix)
            )
    config.ignored_layers = list(dict.fromkeys([*config.ignored_layers, *mapped]))


def resolve_nextn_quant_config(result, model, config, quant_config):
    """Adapt a private draft config without changing target/lm_head exclusions.

    Why: GLM NextN checkpoint names differ from its decoder module names.
    How: AFTER the model-owned quant-config resolver, remap channelwise FP8
         exclusions before constructing draft layers.
    Upstream PR or issue: not filed (plugin channelwise checkpoint format).
    Remove when: the model resolver supports FP8 checkpoint exclusion mapping.
    """
    from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN
    from sglang_tpu.fp8 import TpuChannelwiseFp8Config

    if type(model) is not GlmMoeDsaForCausalLMNextN or not isinstance(
        result, TpuChannelwiseFp8Config
    ):
        return result
    result = copy.copy(result)
    _remap_glm_nextn_ignored_layers(result, config)
    return result
