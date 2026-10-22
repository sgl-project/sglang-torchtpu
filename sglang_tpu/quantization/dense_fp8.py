# SPDX-License-Identifier: Apache-2.0
"""TPU dense FP8 method and compressed-tensor scheme replacements."""

import logging

import torch
from torch.nn import Parameter

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.plugins.oot_classes import register_oot_replacement
from sglang.srt.runtime_context import get_exec
from sglang_tpu.fp8_linear import (
    apply_tpu_fp8_linear,
    compute_tpu_fp8_linear_scale_shape,
    prepare_tpu_fp8_linear_weights,
)

logger = logging.getLogger(__name__)


def _requantize_tpu_block_fp8_to_channelwise(layer: torch.nn.Module) -> None:
    assert layer.weight_block_size is not None
    weight = layer.weight.data
    weight_dequant = block_quant_dequant(
        weight,
        layer.weight_scale.data,
        layer.weight_block_size,
        torch.float32,
    )
    dtype_max = torch.finfo(weight.dtype).max
    weight_scale = torch.clamp(
        weight_dequant.abs().amax(dim=1, keepdim=True) / dtype_max,
        min=1e-12,
    )
    weight = torch.clamp(
        weight_dequant / weight_scale,
        min=-dtype_max,
        max=dtype_max,
    ).to(weight.dtype)

    layer.weight = Parameter(weight.t().contiguous(), requires_grad=False)
    layer.weight_scale = Parameter(weight_scale, requires_grad=False)
    layer.weight_block_size = None


class TpuFp8LinearMethod(Fp8LinearMethod):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        skip_block_quant_check=False,
        **extra_weight_attrs,
    ):
        return super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            skip_block_quant_check=skip_block_quant_check or not self.use_mxfp8,
            **extra_weight_attrs,
        )

    @staticmethod
    def _block_scale_shape(layer, *, use_mxfp8, **shape):
        if use_mxfp8:
            return Fp8LinearMethod._block_scale_shape(layer, use_mxfp8=True, **shape)
        tp_rank, _ = get_group_rank_size(layer.tp_group)
        return compute_tpu_fp8_linear_scale_shape(layer, tp_rank=tp_rank, **shape)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        if self.block_quant and not self.use_mxfp8:
            prepare_tpu_fp8_linear_weights(
                layer, weight_block_size=self.quant_config.weight_block_size
            )

    def apply(self, layer, x, bias=None):
        if not self.use_mxfp8 and (
            not self.block_quant or layer.weight_block_size is None
        ):
            return apply_tpu_fp8_linear(
                x=x, weight=layer.weight, weight_scale=layer.weight_scale, bias=bias
            )
        return super().apply(layer, x, bias)


class TpuCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):
    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        if (
            layer.weight_block_size is not None
            and get_exec().kernel.tpu_fp8_runtime == "channelwise"
        ):
            logger.warning_once(
                "TPU dense FP8 linear is converting block-FP8 checkpoint "
                "weights to channelwise runtime scales by default for "
                "performance. Use --tpu-fp8-runtime=channelwise to make "
                "this explicit, or --tpu-fp8-runtime=blockwise to preserve "
                "blockwise runtime scales."
            )
            _requantize_tpu_block_fp8_to_channelwise(layer)

    def apply_weights(self, layer, x, bias=None):
        if isinstance(x, tuple) or layer.weight_block_size is not None:
            return super().apply_weights(layer, x, bias)
        return apply_tpu_fp8_linear(
            x=x, weight=layer.weight, weight_scale=layer.weight_scale, bias=bias
        )


def register_dense_fp8_methods():
    register_oot_replacement(Fp8LinearMethod, TpuFp8LinearMethod)
    register_oot_replacement(CompressedTensorsW8A8Fp8, TpuCompressedTensorsW8A8Fp8)
