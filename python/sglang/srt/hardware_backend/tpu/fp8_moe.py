"""TPU FP8 MoE helpers.

TPU MoE follows the same backend-dispatch shape as upstream SGLang MoE:
weights are formatted once after loading, then runtime calls a fused grouped
GEMM backend. Today that backend is the Pallas gmm_v2 bridge. Once torch_tpu
registers scaled grouped matmul ops, channelwise/blockwise paths can move to
native torch here without touching the generic FP8 quantization method.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils import get_bool_env_var


def _free_replaced_storage(old: torch.Tensor, new: torch.Tensor) -> None:
    if old.untyped_storage().data_ptr() != new.untyped_storage().data_ptr():
        old.untyped_storage().resize_(0)


def _replace_tpu_param_from_cpu(param: Parameter, new_cpu: torch.Tensor) -> None:
    device = param.data.device
    dtype = param.data.dtype
    old = param.data
    param.data = torch.empty(0, dtype=dtype, device=device)
    _free_replaced_storage(old, param.data)
    param.data = new_cpu.to(device=device, dtype=dtype)


def _to_tpu_fp8_moe_scale_layout(
    scale: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """Convert per-expert FP8 MoE scales to gmm_v2's RHS scale layout."""
    scale = scale.to(torch.bfloat16)
    if scale.dim() == 1:
        scale = scale[:, None].expand(-1, out_size)
    elif scale.dim() == 2:
        if scale.shape[1] == 1:
            scale = scale.expand(-1, out_size)
        elif scale.shape[1] == 2 and out_size % 2 == 0:
            scale = scale.repeat_interleave(out_size // 2, dim=1)
        elif scale.shape[1] != out_size:
            raise ValueError(
                "Unsupported TPU FP8 MoE scale shape: "
                f"{tuple(scale.shape)} for {out_size=}"
            )
    elif scale.dim() == 4:
        if scale.shape[1:] != (1, 1, out_size):
            raise ValueError(
                "Unsupported TPU FP8 MoE scale shape: "
                f"{tuple(scale.shape)} for {out_size=}"
            )
        return scale.contiguous()
    else:
        raise ValueError(
            "Unsupported TPU FP8 MoE scale rank: "
            f"{scale.dim()} for shape {tuple(scale.shape)}"
        )
    return scale[:, None, None, :].contiguous()


def _to_tpu_fp8_moe_block_scale_layout(
    scale: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """Convert block FP8 MoE scales to gmm_v2's RHS scale layout."""
    scale_tile_n = 512
    scale_block_n = 128
    scale_tile_stride = 128
    scale = scale.to(torch.bfloat16)
    if scale.dim() == 4:
        if scale.shape[2] != 1:
            raise ValueError(
                "Unsupported TPU FP8 MoE block scale shape: "
                f"{tuple(scale.shape)} for {out_size=}"
            )
        if scale.shape[-1] == out_size:
            return scale.contiguous()
        scale = scale.squeeze(2)
    if scale.dim() != 3:
        raise ValueError(
            "Unsupported TPU FP8 MoE block scale rank: "
            f"{scale.dim()} for shape {tuple(scale.shape)}"
        )

    scale = scale.transpose(1, 2).contiguous()
    if out_size % scale.shape[2] != 0:
        raise ValueError(
            "TPU FP8 MoE block scale output blocks do not cover output channels: "
            f"{tuple(scale.shape)} for {out_size=}"
        )
    block_n = out_size // scale.shape[2]
    use_compact_scales = (
        get_bool_env_var("SGLANG_TPU_FP8_MOE_COMPACT_SCALES")
        or (
            get_bool_env_var("SGLANG_TPU_FP8_MOE_COMPACT_W1_SCALES")
            and out_size == 5120
        )
        or (
            get_bool_env_var("SGLANG_TPU_FP8_MOE_COMPACT_W2_SCALES")
            and out_size == 6144
        )
    )
    if not use_compact_scales:
        return scale.repeat_interleave(block_n, dim=2)[:, :, None, :].contiguous()
    if block_n > scale_block_n:
        if block_n % scale_block_n != 0:
            raise ValueError(
                "Unsupported TPU FP8 MoE block scale output block size: "
                f"{block_n=} for {out_size=} and scale shape {tuple(scale.shape)}"
            )
        scale = scale.repeat_interleave(block_n // scale_block_n, dim=2)
        block_n = scale_block_n
    if block_n == scale_block_n and out_size > scale_tile_n:
        if out_size % scale_tile_n != 0:
            raise ValueError(
                "TPU FP8 MoE compact scale layout requires output size divisible by "
                f"{scale_tile_n}, got {out_size=}"
            )
        scale_blocks_per_tile = scale_tile_n // scale_block_n
        num_tiles = out_size // scale_tile_n
        expected_blocks = num_tiles * scale_blocks_per_tile
        if scale.shape[2] != expected_blocks:
            raise ValueError(
                "TPU FP8 MoE compact scale block count mismatch: "
                f"{tuple(scale.shape)} for {out_size=}, expected {expected_blocks}"
            )
        tiled_scale = torch.zeros(
            scale.shape[0],
            scale.shape[1],
            num_tiles * scale_tile_stride,
            dtype=scale.dtype,
            device=scale.device,
        )
        for tile_id in range(num_tiles):
            src_start = tile_id * scale_blocks_per_tile
            dst_start = tile_id * scale_tile_stride
            tiled_scale[:, :, dst_start : dst_start + scale_blocks_per_tile].copy_(
                scale[:, :, src_start : src_start + scale_blocks_per_tile]
            )
        scale = tiled_scale
    return scale[:, :, None, :].contiguous()


def prepare_tpu_fp8_moe_weights(
    layer: torch.nn.Module,
    *,
    block_quant: bool,
    use_mxfp8: bool,
    is_fp4_expert: bool,
) -> None:
    """Format FP8 MoE weights/scales for the TPU Pallas gmm_v2 backend."""
    if use_mxfp8 or is_fp4_expert:
        raise NotImplementedError(
            "TPU FP8 MoE currently supports FP8 and block-FP8 experts only."
        )

    # gmm_v2 consumes RHS weights as [expert, K, N]. SGLang loads MoE weights
    # as [expert, N, K], so make the TPU runtime layout explicit at load time.
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        new_cpu = param.data.detach().cpu().transpose(1, 2).contiguous()
        _replace_tpu_param_from_cpu(param, new_cpu)
        del new_cpu

    if block_quant:
        old_w13_scale = layer.w13_weight_scale_inv.data
        old_w2_scale = layer.w2_weight_scale_inv.data
        w13_scale = _to_tpu_fp8_moe_block_scale_layout(
            old_w13_scale,
            layer.w13_weight.shape[-1],
        )
        w2_scale = _to_tpu_fp8_moe_block_scale_layout(
            old_w2_scale,
            layer.w2_weight.shape[-1],
        )
        copy_or_rebind_param(layer, "w13_weight_scale_inv", w13_scale)
        copy_or_rebind_param(layer, "w2_weight_scale_inv", w2_scale)
        _free_replaced_storage(old_w13_scale, layer.w13_weight_scale_inv.data)
        _free_replaced_storage(old_w2_scale, layer.w2_weight_scale_inv.data)
    else:
        old_w13_scale = layer.w13_weight_scale.data
        old_w2_scale = layer.w2_weight_scale.data
        w13_scale = _to_tpu_fp8_moe_scale_layout(
            old_w13_scale, layer.w13_weight.shape[-1]
        )
        w2_scale = _to_tpu_fp8_moe_scale_layout(old_w2_scale, layer.w2_weight.shape[-1])
        copy_or_rebind_param(layer, "w13_weight_scale", w13_scale)
        copy_or_rebind_param(layer, "w2_weight_scale", w2_scale)
        _free_replaced_storage(old_w13_scale, layer.w13_weight_scale.data)
        _free_replaced_storage(old_w2_scale, layer.w2_weight_scale.data)


def apply_tpu_fp8_moe(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    block_quant: bool,
    apply_router_weight_on_input: bool,
):
    """Run TPU FP8 MoE through the Pallas fused grouped-GEMM backend."""
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights, topk_ids = mask_tpu_moe_non_local_experts(topk_weights, topk_ids)

    w1_scale = layer.w13_weight_scale_inv if block_quant else layer.w13_weight_scale
    w2_scale = layer.w2_weight_scale_inv if block_quant else layer.w2_weight_scale

    # TODO(torch_tpu): try native torch._scaled_grouped_mm for channelwise
    # scales and torch._scaled_grouped_mm_v2 for block scales once torch_tpu
    # registers those ops. Until then, keep MoE on the same backend-dispatched
    # fused grouped-GEMM model used by upstream CUDA paths and torchtpu-vllm.
    output = fused_moe_call(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    return StandardCombineInput(hidden_states=output)
