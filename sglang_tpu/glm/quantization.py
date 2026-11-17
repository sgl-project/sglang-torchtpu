# SPDX-License-Identifier: Apache-2.0
"""GLM NextN quantization naming adaptations."""

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
