# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors FP8 W8A8 on TPU.

Serves llm-compressor ``FP8_DYNAMIC`` checkpoints: float8_e4m3 weights with a
static per-output-channel scale (``weight_scale`` ``[N, 1]``) and dynamic
per-token FP8 activations. Dense layers run ``apply_tpu_fp8_linear``; fused
MoE experts run the Pallas gmm_v2 bridge (``apply_tpu_fp8_moe``) or, behind
``--moe-runner-backend pallas_tpu_fused_rs``, the fused reduce-scatter kernel.

Core reaches this module through ``TpuSRTPlatform.get_quantization_config``,
which hands back :class:`TpuCompressedTensorsConfig` for ``compressed-tensors``
(the same factory that serves ``TpuModelOptCompatibleFp4Config``). The config
subclass removes the CUDA compute-capability probe from scheme selection and
swaps in the TPU schemes below; every other scheme decision stays in core.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.moe import MoeRunnerConfig, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.fp8_linear import apply_tpu_fp8_linear
from sglang_tpu.fp8_moe import apply_tpu_fp8_moe, prepare_tpu_fp8_moe_weights

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)


def to_tpu_channelwise_fp8_linear(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint ``[N, K]`` fp8 + ``[N, 1]`` scale -> TPU ``[K, N]`` fp8 + fp32 ``[N]``.

    Both results own their storage: a transposed or reshaped view would leave
    the parameter declaring one shape over another shape's device buffer, which
    torch_tpu reconciles with an as_strided program on every use.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D FP8 weight, got {tuple(weight.shape)}")
    out_size = weight.shape[0]
    if weight_scale.numel() != out_size:
        raise ValueError(
            "Channelwise FP8 weight_scale must hold one value per output channel; "
            f"got {tuple(weight_scale.shape)} for weight {tuple(weight.shape)}"
        )
    weight_kn = weight.t().contiguous()
    scale_n = weight_scale.reshape(out_size).to(torch.float32).clone()
    return weight_kn, scale_n


class TpuCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):
    """Channelwise-weight / dynamic-token FP8 dense linear on TPU.

    Tensor and block strategies keep the parent's handling, which already has
    its TPU branches (block -> channelwise requantization, block dispatch).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.strategy != QuantizationStrategy.CHANNEL:
            super().process_weights_after_loading(layer)
            return
        weight, weight_scale = to_tpu_channelwise_fp8_linear(
            layer.weight.data, layer.weight_scale.data
        )
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        # The TPU kernel always quantizes activations per token at runtime; a
        # static checkpoint input_scale has no consumer.
        layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.strategy != QuantizationStrategy.CHANNEL or isinstance(x, tuple):
            return super().apply_weights(layer, x, bias=bias)
        return apply_tpu_fp8_linear(
            x=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            bias=bias,
        )


class TpuCompressedTensorsW8A8Fp8MoE(CompressedTensorsW8A8Fp8MoE):
    """Channelwise FP8 MoE experts on the TPU Pallas fused-MoE backend.

    Weight creation and checkpoint loading are inherited, so the parameters are
    exactly what the compressed-tensors loader fills: ``w13_weight [E, 2I, H]``
    / ``w2_weight [E, H, I]`` fp8 and ``w13_weight_scale [E, 2I, 1]`` /
    ``w2_weight_scale [E, H, 1]`` fp32. Loading-time conversion then hands them
    to ``prepare_tpu_fp8_moe_weights``, the same formatter the ``fp8`` method
    uses for channelwise runtime scales.
    """

    def __init__(self, weight_quant, input_quant):
        super().__init__(weight_quant, input_quant)
        if not (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
            and self.input_quant.dynamic
        ):
            raise NotImplementedError(
                "TPU compressed-tensors FP8 MoE supports channelwise weights with "
                "dynamic per-token activations only; got "
                f"weights={self.weight_quant.strategy} "
                f"activations={self.input_quant.strategy} "
                f"dynamic={self.input_quant.dynamic}"
            )
        self.handles_moe_reduction = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=False,
            weight_block_size=None,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
        # Explicit opt-in only, as for the fp8 method (TODO(#1065) in fp8.py).
        if get_moe_runner_backend().is_pallas_tpu_fused_rs():
            from sglang_tpu.fp8_moe_fused_rs import prepare_fp8_fused_rs

            self.handles_moe_reduction = prepare_fp8_fused_rs(layer)
            if not self.handles_moe_reduction:
                reason = getattr(
                    layer,
                    "_tpu_fused_rs_unsupported_reason",
                    "unknown incompatibility",
                )
                raise ValueError(
                    "--moe-runner-backend pallas_tpu_fused_rs is not supported by "
                    f"this model or parallel layout: {reason}"
                )
            if get_parallel().tp_rank == 0:
                logger.info_once(
                    "Using TPU Pallas FP8 fused-RS MoE backend (compressed-tensors)"
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        # No MoeRunner backend: apply_weights dispatches directly to the
        # Pallas fused grouped-GEMM bridge.
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        cfg = self.moe_runner_config
        swiglu_limit = getattr(cfg, "swiglu_limit", None)
        if self.handles_moe_reduction:
            from sglang_tpu.fp8_moe_fused_rs import apply_fp8_fused_rs

            return apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation=cfg.activation,
                swiglu_limit=swiglu_limit,
            )
        return apply_tpu_fp8_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=cfg.activation,
            block_quant=False,
            apply_router_weight_on_input=cfg.apply_router_weight_on_input,
            no_combine=cfg.no_combine,
            swiglu_limit=swiglu_limit,
        )


class TpuCompressedTensorsFusedMoEMethod(CompressedTensorsFusedMoEMethod):
    """Publishes the scheme's ``handles_moe_reduction`` to ``FusedMoE``.

    ``FusedMoE`` reads the flag from its quant method, but compressed-tensors
    keeps the backend decision on the per-layer scheme.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        self.handles_moe_reduction = bool(
            getattr(layer.scheme, "handles_moe_reduction", False)
        )


class TpuCompressedTensorsConfig(CompressedTensorsConfig):
    """compressed-tensors with TPU scheme selection.

    TPU has no CUDA compute capability, so the capability probe is answered
    here; the schemes that can actually run are vetted by class instead.
    """

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        return True

    def _get_scheme_from_parts(self, weight_quant, input_quant, format=None):
        scheme = super()._get_scheme_from_parts(
            weight_quant=weight_quant, input_quant=input_quant, format=format
        )
        if type(scheme) is CompressedTensorsW8A8Fp8:
            return TpuCompressedTensorsW8A8Fp8(
                weight_quant=scheme.weight_quant,
                is_static_input_scheme=scheme.is_static_input_scheme,
            )
        return scheme

    def get_linear_scheme(
        self,
        layer: torch.nn.Module,
        layer_name: Optional[str] = None,
        matched_target: Optional[str] = None,
    ) -> Optional[CompressedTensorsLinearScheme]:
        scheme = super().get_linear_scheme(
            layer=layer, layer_name=layer_name, matched_target=matched_target
        )
        if scheme is not None and not isinstance(scheme, CompressedTensorsW8A8Fp8):
            raise NotImplementedError(
                f"{type(scheme).__name__} ({layer_name}) has no TPU kernel; TPU "
                "compressed-tensors linear layers support FP8 W8A8 only."
            )
        return scheme

    def get_moe_scheme(
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ) -> Optional[CompressedTensorsMoEScheme]:
        scheme = super().get_moe_scheme(layer=layer, layer_name=layer_name)
        if type(scheme) is CompressedTensorsW8A8Fp8MoE:
            logger.info_once(
                "Using TpuCompressedTensorsW8A8Fp8MoE (Pallas gmm_v2 FP8 W8A8)"
            )
            return TpuCompressedTensorsW8A8Fp8MoE(
                scheme.weight_quant, scheme.input_quant
            )
        return scheme

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        method = super().get_quant_method(layer, prefix)
        if type(method) is CompressedTensorsFusedMoEMethod:
            return TpuCompressedTensorsFusedMoEMethod(self)
        return method
