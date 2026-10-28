# SPDX-License-Identifier: Apache-2.0
"""Inkling2 checkpoint precision on TPU.

Only routed experts are quantized. NVFP4 experts keep their packed E2M1 weights
and block-16 FP8 scales and run W4A16 through the Pallas fused-RS kernel; the
per-expert ``amax / outer_scale_denominator`` is ModelOpt's ``weight_scale_2``.
MXFP8 experts are dequantized to BF16 at load and requantized to channelwise
FP8 (W8A8 on TPU). Everything else is BF16 or FP32.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.tml.layers.quantization.config import Inkling2QuantConfig


def expert_layer_prefix(prefix: str) -> str:
    """``model.llm.layers.{i}.mlp.experts.`` checkpoint prefix of a FusedMoE module prefix."""
    marker = "layers."
    start = prefix.index(marker)
    layer = prefix[start + len(marker) :].split(".", 1)[0]
    return f"model.llm.layers.{layer}.mlp.experts."


class TpuInkling2QuantConfig(Inkling2QuantConfig):
    """Inkling2 precision groups mapped onto the TPU MoE quant methods."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.layers.quantization.modelopt_quant import (
            TpuModelOptCompatibleFp4Config,
        )

        self._fp4_config = TpuModelOptCompatibleFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[],
        )
        self._online_fp8_config = Fp8Config(is_checkpoint_fp8_serialized=False)

    def expert_precision(self, prefix: str) -> str:
        """'nvfp4', 'mxfp8' or 'bf16' for the experts of the layer at ``prefix``."""
        w13 = expert_layer_prefix(prefix) + "w13_weight"
        if w13 in self.nvfp4_weights:
            return "nvfp4"
        if w13 in self.mxfp8_weights:
            return "mxfp8"
        return "bf16"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            TpuModelOptNvFp4FusedMoEMethod,
        )

        if not isinstance(layer, FusedMoE):
            return None
        precision = self.expert_precision(prefix)
        if precision == "nvfp4":
            return TpuModelOptNvFp4FusedMoEMethod(self._fp4_config)
        # MXFP8 experts arrive dequantized to BF16 and are quantized at load,
        # as are BF16 experts: one FP8 MoE kernel for every non-FP4 layer.
        return self._online_fp8_config.get_quant_method(layer, prefix)
