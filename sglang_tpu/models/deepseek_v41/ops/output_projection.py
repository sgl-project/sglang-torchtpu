# SPDX-License-Identifier: Apache-2.0
"""Torch bridge and startup weights for the DSV4.1 output projection."""

import logging
from threading import Lock

import torch

from sglang_tpu.models.deepseek_v41.kernels import output_projection as kernel

_OP = None
_OP_LOCK = Lock()
logger = logging.getLogger(__name__)


def register_output_projection_op():
    """Register once before the first trace."""
    global _OP
    with _OP_LOCK:
        if _OP is None:
            from sglang_tpu.pallas_jax_op import cpu_export_jax_op

            _OP = cpu_export_jax_op(
                "sglang_tpu_pallas::dsv41_fused_output_projection",
                kernel.fused_output_projection,
            )
    return _OP


def _wo_b_has_channelwise_fp8(wo_b: torch.nn.Module) -> bool:
    scale = getattr(wo_b, "weight_scale", None)
    weight = getattr(wo_b, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight.shape == kernel.WO_B_SHAPE
        and scale is not None
        and scale.dtype == torch.float32
        and scale.shape in ((kernel.WO_B_SHAPE[1],), (kernel.WO_B_SHAPE[1], 1))
        and wo_b.bias is None
        and getattr(wo_b, "input_scale", None) is None
        and getattr(wo_b, "weight_block_size", None) is None
    )


def prepare_output_projection_weights(model: torch.nn.Module) -> None:
    """Bind the derived wo_a weights on the owning modules before KV sizing."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, MQALayer
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang_tpu.fp8_linear import quantize_channelwise_fp8_weight

    config = getattr(model, "config", None)
    if (
        type(model) is not DeepseekV4ForCausalLM
        or getattr(config, "model_type", None) != "deepseek_v41"
        or get_parallel().attn_tp_size != 1
        or get_exec().kernel.tpu_fp8_runtime != "channelwise"
        or kernel.get_tpu_version() != 7
    ):
        return

    # These copies are lifted as graph inputs. Direct weight updates would
    # bypass packing, and an existing compiled graph can retain the copies.
    if not getattr(model, "_derived_weight_cache_error", None):
        model._derived_weight_cache_error = (
            "Online weight updates require restarting with "
            "SGLANG_TPU_ENABLE_DSV41_FUSED_OUTPUT_PROJECTION=0."
        )
    packed_layers = 0
    for layer in model.modules():
        if not isinstance(layer, MQALayer) or not _wo_b_has_channelwise_fp8(layer.wo_b):
            continue
        wo_a = layer.wo_a
        weight = getattr(wo_a, "weight", None)
        if (
            weight is None
            or weight.device.type != "tpu"
            or weight.dtype != torch.bfloat16
            or weight.shape != kernel.WO_A_SHAPE[::-1]
            or hasattr(wo_a, "_dsv41_output_projection_weight")
        ):
            continue

        # The block-scale checkpoint has already been dequantized by the model.
        # Reuse the channelwise packer and keep the BF16 weights for fallback.
        weight_fp8, scale = quantize_channelwise_fp8_weight(
            weight.detach(), torch.float8_e4m3fn
        )
        wo_a.register_buffer(
            "_dsv41_output_projection_weight", weight_fp8, persistent=False
        )
        wo_a.register_buffer(
            "_dsv41_output_projection_scale", scale.squeeze(-1), persistent=False
        )
        # TorchTPU can retain the FP32 packing intermediates until a sync.
        torch.accelerator.synchronize()
        packed_layers += 1
    if packed_layers:
        register_output_projection_op()
    logger.info("DSV4.1 fused output projection packed %d layer(s)", packed_layers)


def project_output(layer, output: torch.Tensor) -> torch.Tensor | None:
    """Run fused wo_a and wo_b on a compatible, inverse-rotated CSA2 output."""
    if (
        _OP is None
        or layer.attn_tp_size != 1
        or layer.wo_b.use_decode_attn_tp
        or output.ndim != 3
        or output.shape[1:] != kernel.ACTIVATION_SHAPE
        or output.dtype != torch.bfloat16
        or not kernel.supports_tokens(output.shape[0])
    ):
        return None

    weight = getattr(layer.wo_a, "_dsv41_output_projection_weight", None)
    if weight is None or not _wo_b_has_channelwise_fp8(layer.wo_b):
        return None
    return _OP(
        output,
        weight,
        layer.wo_a._dsv41_output_projection_scale,
        layer.wo_b.weight,
        layer.wo_b.weight_scale.reshape(-1),
    )
