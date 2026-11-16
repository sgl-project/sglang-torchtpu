# SPDX-License-Identifier: Apache-2.0
"""TPU FP8 checkpoint factory; existing block-FP8 loading stays upstream."""

from __future__ import annotations

from typing import Any

from sglang.srt.layers.quantization.fp8 import Fp8Config


class TpuFp8Config(Fp8Config):
    """Select the offline channelwise ABI only through its explicit layout key.

    Both block-FP8 runtime choices and metadata-free BF16 online quantization
    retain the upstream config. The existing --tpu-fp8-runtime flag selects
    block-to-channel conversion independently of the checkpoint factory.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config:
        layout = config.get("tpu_checkpoint_layout")
        if layout is None:
            return Fp8Config.from_config(config)
        from sglang_tpu.glm52_channelwise_fp8 import (
            CHECKPOINT_LAYOUT,
            Glm52ChannelwiseFp8Config,
        )

        if layout != CHECKPOINT_LAYOUT:
            raise ValueError(f"Unknown TPU FP8 checkpoint layout: {layout!r}")
        if config.get("is_checkpoint_fp8_serialized") is False:
            raise ValueError(
                "Channelwise checkpoint layout requires serialized FP8 weights"
            )
        return Glm52ChannelwiseFp8Config.from_fp8_config(config)


def _remap_draft_ignored_layers(config: Fp8Config, hf_config) -> None:
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
