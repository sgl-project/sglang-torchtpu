# SPDX-License-Identifier: Apache-2.0
"""Private, explicitly opted-in GLM52 channelwise FP8 checkpoint ABI.

Set ``quantization_config.tpu_checkpoint_layout`` to
``glm52_channelwise_fp8`` (including through the HF config override). Null
``weight_block_size`` alone is deliberately NOT an opt-in: ordinary FP8 and
Flash checkpoints keep the upstream config and methods unchanged.

The checkpoint calls its multiplicative dequant scales ``weight_scale_inv``;
these are rank-one output-channel [N] scales, not block scales or reciprocal
scales (the paired weights, not the scales, have rank two).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs
from sglang_tpu.fp8_config import (
    TpuChannelwiseFp8Config,
    _remap_draft_ignored_layers,
)

CHECKPOINT_LAYOUT = "glm52_channelwise_fp8"


class Glm52ChannelwiseFp8Config(TpuChannelwiseFp8Config):
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
        if is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            return Glm52ChannelwiseFp8LinearMethod(self)
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
