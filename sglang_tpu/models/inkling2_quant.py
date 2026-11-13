# SPDX-License-Identifier: Apache-2.0
"""Inkling2 checkpoint precision on TPU.

Only routed experts are quantized. NVFP4 and MXFP8 experts are dequantized on the
host at load and stored as block FP8, which the TPU FP8 MoE path requantizes to
channelwise FP8 and runs W8A8 through gmm_v2. With SGLANG_TPU_INKLING2_NVFP4_EXPERTS
the NVFP4 experts instead keep their packed E2M1 weights and block-16 FP8 scales and
run W4A16 through the Pallas fused-RS kernel; the per-expert
``amax / outer_scale_denominator`` is ModelOpt's ``weight_scale_2``.
Everything else is BF16 or FP32.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.tml.layers.quantization.config import Inkling2QuantConfig
from sglang_tpu.envs import envs as tpu_envs

# (N, K) block of the FP8 expert scales the loader writes.
FP8_BLOCK = (128, 128)


def expert_layer_prefix(prefix: str) -> str:
    """``model.llm.layers.{i}.mlp.experts.`` checkpoint prefix of a FusedMoE module prefix."""
    marker = "layers."
    start = prefix.index(marker)
    layer = prefix[start + len(marker) :].split(".", 1)[0]
    return f"model.llm.layers.{layer}.mlp.experts."


class TpuInkling2QuantConfig(Inkling2QuantConfig):
    """Inkling2 precision groups mapped onto the TPU FP8 MoE quant methods."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        self._online_fp8_config = Fp8Config(is_checkpoint_fp8_serialized=False)
        # Quantized experts are stored as block FP8 from the start: BF16 staging of every layer
        # exceeds HBM (~163 GB/rank). The loader quantizes each expert tensor on the host.
        self._block_fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=list(FP8_BLOCK)
        )
        # Set by inkling2_convert_fp8: the NVFP4/MXFP8 experts were already stored as block FP8.
        self.experts_prequantized = (
            config.get("tpu_expert_format", {}).get("format") == "fp8_e4m3_block"
        )
        self.nvfp4_packed = tpu_envs.SGLANG_TPU_INKLING2_NVFP4_EXPERTS.get()
        if self.nvfp4_packed:
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptCompatibleFp4Config,
            )

            self._fp4_config = TpuModelOptCompatibleFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16, exclude_modules=[]
            )

    def expert_precision(self, prefix: str) -> str:
        """'nvfp4', 'nvfp4_packed', 'mxfp8', 'fp8_block' (pre-converted) or 'bf16' for the experts at ``prefix``."""
        w13 = expert_layer_prefix(prefix) + "w13_weight"
        if w13 in self.nvfp4_weights or w13 in self.mxfp8_weights:
            if self.experts_prequantized:
                return "fp8_block"
            if w13 in self.nvfp4_weights:
                return "nvfp4_packed" if self.nvfp4_packed else "nvfp4"
            return "mxfp8"
        return "bf16"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if not isinstance(layer, FusedMoE):
            return None
        precision = self.expert_precision(prefix)
        if precision == "nvfp4_packed":
            from sglang.srt.layers.quantization.modelopt_quant import (
                TpuModelOptNvFp4FusedMoEMethod,
            )

            return TpuModelOptNvFp4FusedMoEMethod(self._fp4_config)
        if precision in ("nvfp4", "mxfp8", "fp8_block"):
            return self._block_fp8_config.get_quant_method(layer, prefix)
        return self._online_fp8_config.get_quant_method(layer, prefix)
