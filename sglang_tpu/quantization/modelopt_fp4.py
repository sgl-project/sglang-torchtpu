# SPDX-License-Identifier: Apache-2.0
"""TPU ModelOpt-compatible packed-FP4 configs and MoE method.

Selected by ``TpuSRTPlatform.get_quantization_config`` for ``modelopt_fp4`` and
``modelopt_mixed``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.layers.moe import MoeRunnerConfig, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8KVCacheMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)


class TpuModelOptCompatibleFp4Config(ModelOptFp4Config):
    """TPU block-16/32/64/128/256/512 packed-FP4 checkpoint schema.

    Reuses ModelOpt tensor names and packed E2M1 weights, but is not standard
    NVIDIA NVFP4: block scales may use fp32/bf16/fp8, the per-tensor global
    scale may be absent, only MoE experts are quantized, and execution is W4A8
    (block 256/512) or W4A16 (blocks 16/32/64/128) through Pallas fused-RS rather than
    native Blackwell FP4.
    """

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping or {}
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            raise NotImplementedError(
                "TPU ModelOpt-compatible FP4 supports only MoE experts; "
                f"dense layer {prefix!r} must be excluded from quantization."
            )
        if self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            if self.is_layer_excluded(prefix):
                return None
            return TpuModelOptNvFp4FusedMoEMethod(self)
        return None


class TpuModelOptMixedPrecisionConfig(ModelOptMixedPrecisionConfig):
    """Fail-closed boundary for unsupported TPU ModelOpt mixed checkpoints."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TpuModelOptMixedPrecisionConfig:
        raise NotImplementedError(
            "ModelOpt MIXED_PRECISION checkpoints are not supported on TPU. "
            "TPU packed-FP4 support requires the MoE-only block-128/256/512 "
            "modelopt_fp4 checkpoint schema."
        )


class TpuModelOptNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """TPU adapter for ModelOpt NVFP4 MoE checkpoints."""

    SUPPORTED_GROUP_SIZES = (16, 32, 64, 128, 256, 512)

    def __init__(self, quant_config: TpuModelOptCompatibleFp4Config):
        # Do not run the GPU backend/capability validation in the parent
        # constructor. TPU validates its checkpoint and runner contracts below.
        if quant_config.group_size not in self.SUPPORTED_GROUP_SIZES:
            raise NotImplementedError(
                "TPU NVFP4 MoE requires group_size in "
                f"{self.SUPPORTED_GROUP_SIZES}, "
                f"got group_size={quant_config.group_size}."
            )
        if not isinstance(quant_config.global_scale, bool):
            raise ValueError(
                "TPU ModelOpt-compatible FP4 requires global_scale to be a bool, "
                f"got {quant_config.global_scale!r}."
            )
        scale_dtypes = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
        }
        scale_dtype_name = quant_config.scale_dtype
        if scale_dtype_name is not None:
            if not isinstance(scale_dtype_name, str):
                raise ValueError(
                    "TPU ModelOpt-compatible FP4 requires scale_dtype to be a "
                    f"string, got {scale_dtype_name!r}."
                )
            scale_dtype_name = scale_dtype_name.strip().lower()
            if scale_dtype_name not in scale_dtypes:
                raise ValueError(
                    "Unsupported TPU ModelOpt-compatible FP4 checkpoint "
                    f"scale_dtype={quant_config.scale_dtype!r}."
                )
        self._checkpoint_scale_dtype = (
            scale_dtypes[scale_dtype_name]
            if scale_dtype_name is not None
            else (torch.float8_e4m3fn if quant_config.global_scale else torch.float32)
        )
        self._prepared_scale_dtype = (
            scale_dtypes[scale_dtype_name]
            if scale_dtype_name is not None
            else (torch.bfloat16 if quant_config.global_scale else torch.float32)
        )
        self.quant_config = quant_config
        self.enable_flashinfer_trtllm_moe = False

    def _weight_scale_dtype(self) -> torch.dtype:
        return self._checkpoint_scale_dtype

    def _uses_swizzled_blockscales(self) -> bool:
        return False

    def _uses_global_scale_parameters(self) -> bool:
        return self.quant_config.global_scale

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        return super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def _resolve_fused_rs(self, layer: torch.nn.Module) -> None:
        """Resolve the TPU MoE backend after the model publishes its contract."""
        from sglang_tpu.fp4_moe import (
            unmet_tpu_nvfp4_fused_rs_requirements,
        )
        from sglang_tpu.fused_rs_sequence_parallel import (
            has_tpu_fused_rs_post_expert_norm,
        )

        has_post_expert_norm = has_tpu_fused_rs_post_expert_norm(layer)
        if has_post_expert_norm:
            layer.allow_backend_reduction = True

        moe_runner_backend = get_moe_runner_backend()
        if not (
            moe_runner_backend.is_auto()
            or moe_runner_backend.is_pallas_tpu()
            or moe_runner_backend.is_pallas_tpu_fused_rs()
        ):
            raise ValueError(
                "TPU NVFP4 MoE supports only --moe-runner-backend auto, "
                "pallas_tpu, or pallas_tpu_fused_rs"
            )

        if moe_runner_backend.is_pallas_tpu():
            self.handles_moe_reduction = False
            return

        unmet_requirements = unmet_tpu_nvfp4_fused_rs_requirements(
            layer,
            group_size=self.quant_config.group_size,
            no_combine=self.moe_runner_config.no_combine,
        )
        self.handles_moe_reduction = not unmet_requirements
        if self.handles_moe_reduction:
            logger.info("Using TPU Pallas NVFP4 fused-RS MoE backend")
        elif (
            self.quant_config.group_size != 512
            or moe_runner_backend.is_pallas_tpu_fused_rs()
        ):
            block_size_requirement = (
                f"TPU NVFP4 group_size={self.quant_config.group_size} requires "
                "the fused-RS backend. "
                if self.quant_config.group_size != 512
                else ""
            )
            raise ValueError(
                block_size_requirement
                + "--moe-runner-backend pallas_tpu_fused_rs cannot honor this "
                "TPU NVFP4 layer. Unmet requirements: " + ", ".join(unmet_requirements)
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        # Resolve the path before formatting because fused-RS uses a different
        # packed-weight layout from the stock Pallas fallback.
        self._resolve_fused_rs(layer)

        prepare_tpu_nvfp4_moe_weights(
            layer,
            group_size=self.quant_config.group_size,
            scale_dtype=self._prepared_scale_dtype,
            use_fused_rs=self.handles_moe_reduction,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if not (
            moe_runner_backend.is_auto()
            or moe_runner_backend.is_pallas_tpu()
            or moe_runner_backend.is_pallas_tpu_fused_rs()
        ):
            raise ValueError(
                "TPU NVFP4 MoE supports only --moe-runner-backend auto, "
                "pallas_tpu, or pallas_tpu_fused_rs"
            )
        # Compatibility is resolved after model setup in _resolve_fused_rs.

    def apply(
        self,
        layer: FusedMoE,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang_tpu.fp4_moe import apply_tpu_nvfp4_moe

        return apply_tpu_nvfp4_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=self.moe_runner_config.activation,
            apply_router_weight_on_input=(
                self.moe_runner_config.apply_router_weight_on_input
            ),
            no_combine=self.moe_runner_config.no_combine,
            use_fused_rs=self.handles_moe_reduction,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
