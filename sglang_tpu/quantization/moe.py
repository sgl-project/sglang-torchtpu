# SPDX-License-Identifier: Apache-2.0
"""TPU MoE methods selected before shared checkpoint loading and model tracing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod, fp8_dtype
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.plugins.oot_classes import register_oot_replacement
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import round_up

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)


def _quantize_fp8_weight(
    weight: torch.Tensor, scale: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8 quantization fallback for TPU weight post-processing."""
    fp8_max = torch.finfo(fp8_dtype).max
    device = weight.device
    weight = weight.detach().cpu()
    if scale is None:
        scale = weight.abs().max().to(torch.float32).clamp(min=1e-12) / fp8_max
    else:
        scale = scale.detach().cpu().to(torch.float32)
    qweight = (weight / scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return qweight.to(device), scale.to(device)


class TpuUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def create_moe_runner(self, layer, moe_runner_config):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Pad gate/up separately before transposing (experts, n, k) to (experts, k, n).
        inter = layer.w2_weight.shape[-1]
        pad = (-inter) % 128
        if pad and getattr(layer, "w13_weight_bias", None) is None:
            from sglang_tpu.moe.mxfp4_tpu_util import (
                _pad_gate_up_halves_axis1,
                _pad_last_axis,
            )

            w13 = _pad_gate_up_halves_axis1(layer.w13_weight.data, inter, pad)
            copy_or_rebind_param(layer, "w13_weight", w13)
            w2 = _pad_last_axis(layer.w2_weight.data, pad)
            copy_or_rebind_param(layer, "w2_weight", w2)
        for name in ("w13_weight", "w2_weight"):
            old = getattr(layer, name).data
            new = old.transpose(1, 2).contiguous()
            copy_or_rebind_param(layer, name, new)
            # copy_or_rebind_param copies in place when the shape is
            # unchanged, so a shape-preserving (square) transpose leaves the
            # param pointing at `old`'s storage. Only free `old` when the
            # param was actually rebound to a new storage; otherwise we would
            # zero out the live weight (e.g. w2 when intermediate == hidden).
            if (
                getattr(layer, name).data.untyped_storage().data_ptr()
                != old.untyped_storage().data_ptr()
            ):
                old.untyped_storage().resize_(0)

    def _torch_compile_forward(self, num_tokens):
        return None

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        *,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

        moe_runner_config = self.moe_runner_config
        if moe_runner_config.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input is not supported on TPU yet."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        if moe_runner_config.num_fused_shared_experts:
            raise NotImplementedError(
                "Fused shared experts are not supported on the TPU MoE path."
            )
        no_combine = moe_runner_config.no_combine
        kernel_topk_weights = topk_weights.to(x.dtype)
        # Hand the router's global expert IDs to the kernel unchanged, together
        # with the window this rank owns, [expert_group_offset,
        # expert_group_offset + num_local_experts). gmm_v2 only touches the rows
        # of expert groups inside that window, so non-local experts cost nothing
        # and no remapping to local IDs is needed. Under EP1 the offset is 0 and
        # the window spans every expert, so the window check is a no-op. The
        # kernel drops IDs outside the global range (TopK's -1 padded tokens).
        global_num_experts = layer.num_local_experts * get_parallel().moe_ep_size
        expert_group_offset = layer.num_local_experts * get_parallel().moe_ep_rank

        # Weights are already in gmm_v2's (num_experts, k, n) layout — the swap
        # from SGLang's (num_experts, n, k) happens once in
        # process_weights_after_loading above. Per-expert biases are optional:
        # present for GPT-OSS (MXFP4 dequantized to bf16, which reuses this
        # forward) and absent for plain-bf16 models like Qwen3, so read them via
        # getattr (mirrors the XPU path).
        output = fused_moe_call(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=getattr(layer, "w13_weight_bias", None),
            w2_bias=getattr(layer, "w2_weight_bias", None),
            topk_weights=kernel_topk_weights,
            topk_ids=topk_ids,
            topk=int(topk_ids.shape[-1]),
            activation=moe_runner_config.activation,
            no_combine=no_combine,
            global_num_experts=global_num_experts,
            expert_group_offset=expert_group_offset,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_clamp_limit=moe_runner_config.gemm1_clamp_limit,
        )
        if no_combine:
            # The EP kernel masks by ownership only; zero-weight slots stay zero
            # for the caller's per-expert normalization, as without EP.
            output = torch.where(kernel_topk_weights.unsqueeze(-1) != 0, output, 0)
        return StandardCombineInput(hidden_states=output)


class TpuFp8MoEMethod(Fp8MoEMethod):
    keep_fp4_weights_packed = True

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        self.moe_runner_config = moe_runner_config

    def _quantize_weight(self, weight, scale=None):
        return _quantize_fp8_weight(weight, scale)

    def _prepare_weights_for_backend(self, layer: torch.nn.Module) -> None:
        from sglang_tpu.fp8_moe import (
            prepare_tpu_fp8_moe_weights,
        )

        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=self.block_quant,
            weight_block_size=self.quant_config.weight_block_size,
            use_mxfp8=self.use_mxfp8,
            is_fp4_expert=self.is_fp4_expert,
        )
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto() or moe_runner_backend.is_pallas_tpu_fused_rs():
            from sglang_tpu.fp8_moe_fused_rs import (
                prepare_fp8_fused_rs,
            )

            self.handles_moe_reduction = prepare_fp8_fused_rs(layer)
            if self.handles_moe_reduction and get_parallel().tp_rank == 0:
                logger.info_once("Using TPU Pallas FP8 fused-RS MoE backend")
            if (
                moe_runner_backend.is_pallas_tpu_fused_rs()
                and not self.handles_moe_reduction
            ):
                reason = getattr(
                    layer,
                    "_tpu_fused_rs_unsupported_reason",
                    "unknown incompatibility",
                )
                raise ValueError(
                    "--moe-runner-backend pallas_tpu_fused_rs is not supported by "
                    f"this model or parallel layout: {reason}"
                )

    def apply(self, layer, dispatch_output):
        moe_runner_config = self.moe_runner_config
        if self.handles_moe_reduction:
            from sglang_tpu.fp8_moe_fused_rs import (
                apply_fp8_fused_rs,
            )

            return apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation=moe_runner_config.activation,
                swiglu_limit=moe_runner_config.swiglu_limit,
            )
        from sglang_tpu.fp8_moe import apply_tpu_fp8_moe

        return apply_tpu_fp8_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=moe_runner_config.activation,
            block_quant=self.block_quant,
            apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
            no_combine=moe_runner_config.no_combine,
            swiglu_limit=moe_runner_config.swiglu_limit,
        )


class TpuMxfp4MoEMethod(Mxfp4MoEMethod):
    def create_moe_runner(self, layer, moe_runner_config):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        self.moe_runner_config = moe_runner_config

    def _get_intermediate_size_after_pad(self, size: int) -> int:
        return round_up(size, 32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang_tpu.moe.mxfp4_tpu_util import (
            prepare_tpu_mxfp4_moe_weights,
        )

        prepare_tpu_mxfp4_moe_weights(layer, with_bias=self.with_bias)

    def apply(self, layer, dispatch_output):
        # Scales are None on the BF16 path, packed-FP4 block scales otherwise.
        return TpuUnquantizedFusedMoEMethod.forward_tpu(
            self,
            layer,
            dispatch_output,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )


class TpuCompressedTensorsWNA16MoE(CompressedTensorsWNA16MoE):
    """TPU W4A16 MoE via the Pallas fused grouped-GEMM (gmm_v2) backend.

    Inherits pack-quantized weight creation (packed int32 + group scales)
    from CompressedTensorsWNA16MoE; replaces the Marlin repack with the TPU
    relayout (nibble sign flip + scale reshape) and dispatches apply
    straight to the Pallas bridge — same idiom as the fp8 TPU MoE path.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        prepare_tpu_w4a16_moe_weights(layer, group_size=self.group_size)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        # No MoeRunner backend: apply_weights dispatches directly to the
        # Pallas fused grouped-GEMM bridge.
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        cfg = self.moe_runner_config
        from sglang_tpu.w4a16_moe import apply_tpu_w4a16_moe

        return apply_tpu_w4a16_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=cfg.activation,
            apply_router_weight_on_input=cfg.apply_router_weight_on_input,
            no_combine=cfg.no_combine,
        )


class TpuCompressedTensorsConfig(CompressedTensorsConfig):
    def _is_tpu_w4a16_moe(self, weight_quant: QuantizationArgs) -> bool:
        """Symmetric 4-bit int group-quantized: the only WNA16 shape the TPU
        Pallas gmm_v2 path implements. It decodes signed-int4 nibbles, reads no
        zero points and drops g_idx, so float4, asymmetric or act-order
        checkpoints would silently decode wrong.
        """
        return (
            weight_quant.num_bits == 4
            and weight_quant.type == QuantizationType.INT
            and weight_quant.symmetric
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and not weight_quant.actorder
        )

    def _get_wna16_moe_scheme(self, weight_quant, input_quant):
        if not self._is_tpu_w4a16_moe(weight_quant):
            raise ValueError(
                "TPU W4A16 MoE supports only symmetric 4-bit "
                "group-quantized checkpoints without act-order; got "
                f"num_bits={weight_quant.num_bits} "
                f"symmetric={weight_quant.symmetric} "
                f"strategy={weight_quant.strategy} "
                f"actorder={weight_quant.actorder}"
            )
        logger.info_once("Using TpuCompressedTensorsWNA16MoE (Pallas gmm_v2 W4A16)")
        return CompressedTensorsWNA16MoE(self, weight_quant=weight_quant)


def register_moe_quantization_methods():
    register_oot_replacement(UnquantizedFusedMoEMethod, TpuUnquantizedFusedMoEMethod)
    register_oot_replacement(Fp8MoEMethod, TpuFp8MoEMethod)
    register_oot_replacement(Mxfp4MoEMethod, TpuMxfp4MoEMethod)
    register_oot_replacement(CompressedTensorsWNA16MoE, TpuCompressedTensorsWNA16MoE)
