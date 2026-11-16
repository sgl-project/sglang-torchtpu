# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 offline channelwise FP8 checkpoint loading.

The existing block-FP8 paths keep their checkpoint metadata unchanged:

* Existing defaults: ``--quantization fp8``.
* Explicit conversion of the SAME block-FP8 checkpoint:
  ``--quantization fp8 --tpu-fp8-runtime channelwise``.
  The current runtime default is already channelwise, so these are equivalent.

The offline channelwise checkpoint instead uses ``--quantization fp8`` with
``--load-format safetensors`` and this explicit HF model override:

.. code-block:: json

    {"quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": null,
        "tpu_checkpoint_layout": "glm52_channelwise_fp8"
    }}

Pass the JSON with ``--json-model-override-args``. It replaces the whole quant
config: preserve the checkpoint's ignored_layers/modules_to_not_convert.
Null weight_block_size alone never selects this ABI. The checkpoint's
weight_scale_inv is a MULTIPLICATIVE channel dequantization coefficient,
not a reciprocal or a block scale. Already quantized bytes are loaded directly.

Kernel/backend selection and dynamic activation quantization remain independent
of checkpoint source. Native NEXTN uses the same checkpoint; a 5-step chain uses
``--speculative-algorithm NEXTN --speculative-num-steps 5`` plus
``--speculative-eagle-topk 1 --speculative-num-draft-tokens 6``. DP attention
also requires ``--enable-dp-lm-head``. Full-model MTP serving needs separate
E2E validation; component loading tests do not establish it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

CHECKPOINT_LAYOUT = "glm52_channelwise_fp8"


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


class Glm52ChannelwiseFp8Config(Fp8Config):
    """Loader-only specialization; existing TPU channelwise kernels execute."""

    @classmethod
    def from_fp8_config(cls, config: dict[str, Any]) -> Glm52ChannelwiseFp8Config:
        hf_config = config.get("hf_config")
        architectures = getattr(hf_config, "architectures", None) or []
        if not any(
            arch in ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN")
            for arch in architectures
        ):
            raise ValueError(
                f"{CHECKPOINT_LAYOUT} requires GlmMoeDsaForCausalLM or its NextN draft; "
                f"got {architectures!r}"
            )
        if (
            config.get("quant_method") != "fp8"
            or config.get("activation_scheme") != "dynamic"
            or config.get("fmt") != "e4m3"
            or config.get("weight_block_size") is not None
            or config.get("scale_fmt") is not None
        ):
            raise ValueError(
                f"{CHECKPOINT_LAYOUT} requires serialized fp8, dynamic activations, "
                "fmt=e4m3, null weight_block_size and no scale_fmt"
            )
        from sglang.srt.runtime_context import get_exec, get_model

        if get_model().load_format not in ("auto", "pt", "safetensors"):
            raise ValueError(
                f"{CHECKPOINT_LAYOUT} requires the default checkpoint loader "
                "(auto, pt or safetensors); npcache cannot serialize FP8 dtypes"
            )
        if get_exec().offload.cpu_offload_gb:
            raise ValueError(f"{CHECKPOINT_LAYOUT} does not support CPU offloading")
        # Reuse upstream parsing/normalization without changing the input metadata.
        result = Fp8Config.from_config.__func__(cls, config)
        result.tpu_checkpoint_layout = CHECKPOINT_LAYOUT
        _remap_draft_ignored_layers(result, hf_config)
        return result

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            return Glm52ChannelwiseFp8LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return Glm52ChannelwiseFp8MoEMethod(self)
        # Preserve KV-cache quantization and other upstream consumers.
        return super().get_quant_method(layer, prefix)


class Glm52ChannelwiseFp8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Glm52ChannelwiseFp8Config):
        self.quant_config = quant_config
        self.block_quant = False
        self.weight_block_size = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):
        del input_size, output_size, skip_block_quant_check
        n = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = n
        layer.orig_dtype = params_dtype
        loader = extra_weight_attrs.get("weight_loader")
        # Channel scales have no input_dim: RowParallelLinear must replicate
        # them, not shard the output-channel vector alongside the weight's K axis.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    n, input_size_per_partition, dtype=torch.float8_e4m3fn
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            ),
        )
        scale = ChannelQuantScaleParameter(
            data=torch.empty(n, dtype=torch.float32),
            output_dim=0,
            weight_loader=loader,
        )
        metadata = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        set_weight_attrs(layer.weight, metadata)
        set_weight_attrs(scale, metadata)
        layer.register_parameter("weight_scale_inv", scale)
        # The model's early kv_b reader runs before quant-method postprocessing.
        # Do not register a second parameter name: named_parameters must expose
        # the checkpoint's canonical weight_scale_inv key, exactly once.
        object.__setattr__(layer, "weight_scale", scale)
        layer.register_parameter("input_scale", None)
        layer.weight_block_size = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight_scale is not layer.weight_scale_inv:
            raise ValueError(
                "GLM52 channelwise FP8 requires direct target-device construction "
                "with the default loader; moving/replacing parameters before "
                "postprocessing invalidates the early kv_b scale alias"
            )
        if layer.weight_scale_inv.shape != (layer.weight.shape[0],):
            raise ValueError("GLM52 dense scales must have shape [output_channels]")
        layer.weight = Parameter(
            layer.weight.data.t().contiguous(), requires_grad=False
        )
        object.__setattr__(layer, "weight_scale", layer.weight_scale_inv)

    def apply(self, layer, x, bias=None):
        from sglang_tpu.fp8_linear import apply_tpu_fp8_linear

        return apply_tpu_fp8_linear(x, layer.weight, layer.weight_scale_inv, bias)


class Glm52ChannelwiseFp8MoEMethod(Fp8MoEMethod):
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
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )
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
        )

        for stem in ("w13", "w2"):
            scale = getattr(layer, f"{stem}_weight_scale_inv")
            expected = tuple(getattr(layer, f"{stem}_weight").shape[:2])
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"Channelwise expert scales must have shape {expected}"
                )
            layer.register_parameter(f"{stem}_weight_scale", scale)

        self.handles_moe_reduction = prepare_tpu_fp8_moe(
            layer,
            block_quant=False,
            weight_block_size=None,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
