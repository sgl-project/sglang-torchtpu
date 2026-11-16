# SPDX-License-Identifier: Apache-2.0
"""Explicit TPU FP8 loading policies; kernel selection stays independent."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter

from sglang.srt.layers.moe.fused_moe_triton import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped


class TpuFp8Config(Fp8Config):
    """Factory selected by the platform for both serialized and online FP8.

    The metadata-free ``TpuFp8Config()`` entry retains the upstream online
    behavior. A channelwise online policy must explicitly declare that the
    source checkpoint is unquantized; ``activation_scheme`` is independent.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config:
        layout = config.get("tpu_checkpoint_layout")
        policy = config.get("tpu_moe_weight_quantization")
        if layout is None and policy is None:
            # Only a TPU-specific key opts in. Preserve upstream handling of
            # all other metadata, including keys it does not currently parse.
            return Fp8Config.from_config(config)

        if layout is not None:
            from sglang_tpu.glm52_channelwise_fp8 import (
                CHECKPOINT_LAYOUT,
                Glm52ChannelwiseFp8Config,
            )

            if layout != CHECKPOINT_LAYOUT:
                raise ValueError(f"Unknown TPU FP8 checkpoint layout: {layout!r}")
            if (
                policy is not None
                or config.get("is_checkpoint_fp8_serialized") is False
            ):
                raise ValueError(
                    "Serialized channelwise checkpoint layout cannot select "
                    "an online weight quantization policy"
                )
            return Glm52ChannelwiseFp8Config.from_fp8_config(config)

        if policy != "channelwise":
            raise ValueError(f"Unknown TPU MoE weight quantization policy: {policy!r}")
        if (
            config.get("is_checkpoint_fp8_serialized") is not False
            or config.get("quant_method") != "fp8"
            or config.get("activation_scheme") != "dynamic"
            or config.get("weight_block_size") is not None
            or config.get("scale_fmt") is not None
            or config.get("fmt", "e4m3") != "e4m3"
        ):
            raise ValueError(
                "Online channelwise MoE requires is_checkpoint_fp8_serialized=false, "
                "quant_method=fp8, dynamic activations, fmt=e4m3 and no block/scale format"
            )

        # Reuse upstream parsing, including ignored-layer normalization, without
        # making from_config's quant_method='fp8' infer serialized FP8 weights.
        metadata = dict(config, quant_method="unquantized")
        result = Fp8Config.from_config.__func__(TpuChannelwiseFp8Config, metadata)
        result.tpu_moe_weight_quantization = policy
        _remap_draft_ignored_layers(result, config.get("hf_config"))
        return result


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


class TpuChannelwiseFp8Config(Fp8Config):
    """Shared MoE execution contract for online and serialized channel scales."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, FusedMoE) and not is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return TpuChannelwiseFp8MoEMethod(self)
        return super().get_quant_method(layer, prefix)


class TpuChannelwiseFp8MoEMethod(Fp8MoEMethod):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )
        if not self.quant_config.is_checkpoint_fp8_serialized:
            return
        for stem in ("w13", "w2"):
            name = f"{stem}_weight_scale"
            old = getattr(layer, name)
            weight = getattr(layer, f"{stem}_weight")
            scale = Parameter(
                torch.empty(
                    *weight.shape[:2], dtype=torch.float32, device=weight.device
                ),
                requires_grad=False,
            )
            scale.__dict__.update(old.__dict__)
            scale.quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
            delattr(layer, name)
            layer.register_parameter(f"{name}_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang_tpu.fp8_moe import (
            prepare_tpu_fp8_moe,
            quantize_tpu_moe_weights_channelwise,
        )

        if self.quant_config.is_checkpoint_fp8_serialized:
            for stem in ("w13", "w2"):
                scale = getattr(layer, f"{stem}_weight_scale_inv")
                expected = tuple(getattr(layer, f"{stem}_weight").shape[:2])
                if tuple(scale.shape) != expected:
                    raise ValueError(
                        f"Channelwise expert scales must have shape {expected}"
                    )
                layer.register_parameter(f"{stem}_weight_scale", scale)
        else:
            quantize_tpu_moe_weights_channelwise(layer)

        self.handles_moe_reduction = prepare_tpu_fp8_moe(
            layer,
            block_quant=False,
            weight_block_size=None,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
