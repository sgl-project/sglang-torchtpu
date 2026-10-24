# SPDX-License-Identifier: Apache-2.0
"""ModelOpt NVFP4 checkpoints on TPU.

``TpuSRTPlatform.get_quantization_config("modelopt_fp4")`` returns
:class:`TpuModelOptFp4Config`. Scheme selection is core's TPU ModelOpt FP4
schema (``TpuModelOptCompatibleFp4Config``): only routed experts may be
quantized; every dense layer must be on the checkpoint's ``ignore`` list and
stays in its checkpoint dtype. Experts run through ``sglang_tpu.fp4_moe``:
packed E2M1 weights stay packed in HBM with the per-tensor ``weight_scale_2``
folded into the bf16 per-block scales, executed W4A16 (group 16) on the Pallas
fused reduce-scatter kernel. The static activation ``input_scale`` is unused:
activations stay bf16 instead of emulating the GPU's FP4 activations.

:class:`TpuModelOptFp4MoEMethod` keeps that path and fixes its load contract:

* ``FusedMoE``'s weight loader reads ``use_flashinfer_trtllm_weight_layout``
  from every ModelOpt NVFP4 method, but the core TPU method skips the parent
  ``__init__`` that sets it.
* The parent registers the (unused) activation ``input_scale`` for every
  global expert. Those parameters turn off EP-local expert reads, so each EP
  rank would read the whole checkpoint instead of its own experts. TPU keeps
  them per local expert, like the weights.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.modelopt_quant import (
    TpuModelOptCompatibleFp4Config,
    TpuModelOptNvFp4FusedMoEMethod,
    _make_per_tensor_scale_parameter,
)

_INPUT_SCALE_PARAMS = ("w13_input_scale", "w2_input_scale")


class TpuModelOptFp4MoEMethod(TpuModelOptNvFp4FusedMoEMethod):
    """TPU NVFP4 experts with expert-local loading."""

    def __init__(self, quant_config: TpuModelOptCompatibleFp4Config):
        super().__init__(quant_config)
        self.use_flashinfer_trtllm_weight_layout = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        for name in _INPUT_SCALE_PARAMS:
            param = getattr(layer, name, None)
            if param is None or not getattr(
                param, "_sglang_require_global_experts", False
            ):
                continue
            # ``num_experts`` is the local expert count; FusedMoE maps each
            # checkpoint expert id to its local slot and skips other ranks'.
            layer.register_parameter(
                name,
                _make_per_tensor_scale_parameter(
                    (num_experts, *param.shape[1:]),
                    weight_loader=param.weight_loader,
                    fill_value=1.0,
                ),
            )


class TpuModelOptFp4Config(TpuModelOptCompatibleFp4Config):
    """``modelopt_fp4`` on TPU: core's MoE-only schema with the TPU MoE method."""

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        method = super().get_quant_method(layer, prefix)
        if type(method) is TpuModelOptNvFp4FusedMoEMethod:
            return TpuModelOptFp4MoEMethod(self)
        return method
