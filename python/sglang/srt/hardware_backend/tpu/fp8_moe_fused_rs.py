# SPDX-License-Identifier: Apache-2.0

"""FP8 fused reduce-scatter MoE path for TPU.

Routes the FP8 MoE through the in-tree SGLang-owned fused-RS kernel
(`fused_moe_rs_gather`) instead of `fused_moe_call` + a separate all-reduce.
Load-time `prepare_fp8_fused_rs` validates the supported runtime layout and
caches physical-device ordering. Larger prefills retain the stock FP8 MoE path.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

# TODO: Enable larger M, including prefill, once these issues are resolved:
# https://github.com/sgl-project/sglang-torchtpu/issues/308
# https://github.com/google-pytorch/torch_tpu/issues/2794
_FUSED_RS_MAX_SIZE_M = 1024


def supports_fp8_fused_rs(
    *,
    ep_size: int,
    tp_size: int,
    pp_size: int,
    world_size: int,
    reduction_compatible: bool,
    attn_dp_size: int,
    block_quantized: bool,
    gated: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool,
    has_fused_shared_experts: bool,
    standard_dispatcher: bool,
    enable_dp_attention: bool,
    enable_fused_moe_sum_all_reduce: bool,
) -> bool:
    """Return whether the runtime layout matches the validated fused-RS path."""
    return (
        fp8_fused_rs_unsupported_reason(
            ep_size=ep_size,
            tp_size=tp_size,
            pp_size=pp_size,
            world_size=world_size,
            reduction_compatible=reduction_compatible,
            attn_dp_size=attn_dp_size,
            block_quantized=block_quantized,
            gated=gated,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            no_combine=no_combine,
            has_fused_shared_experts=has_fused_shared_experts,
            standard_dispatcher=standard_dispatcher,
            enable_dp_attention=enable_dp_attention,
            enable_fused_moe_sum_all_reduce=enable_fused_moe_sum_all_reduce,
        )
        is None
    )


def fp8_fused_rs_unsupported_reason(
    *,
    ep_size: int,
    tp_size: int,
    pp_size: int,
    world_size: int,
    reduction_compatible: bool,
    attn_dp_size: int,
    block_quantized: bool,
    gated: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool,
    has_fused_shared_experts: bool,
    standard_dispatcher: bool,
    enable_dp_attention: bool,
    enable_fused_moe_sum_all_reduce: bool,
) -> str | None:
    """Return the first reason the fused-RS backend cannot be selected."""
    if ep_size <= 1:
        return f"expert parallel size must be greater than 1, got {ep_size}"
    if pp_size != 1:
        return f"pipeline parallelism is not supported, got PP={pp_size}"
    if ep_size != tp_size or ep_size != world_size:
        return (
            "expert parallel, tensor parallel, and world sizes must match, got "
            f"EP={ep_size}, TP={tp_size}, world={world_size}"
        )
    if not reduction_compatible:
        return "the model does not allow backend-owned MoE reduction"
    if attn_dp_size != 1 or enable_dp_attention:
        return "DP attention and hybrid DP/TP attention are not supported"
    if not gated:
        return "the MoE activation must be gated"
    if activation != "silu":
        return f"the MoE activation must be silu, got {activation}"
    if apply_router_weight_on_input:
        return "applying router weights on the expert input is not supported"
    if no_combine:
        return "no_combine output is not supported"
    if has_fused_shared_experts:
        return "fused shared experts are not supported"
    if not standard_dispatcher:
        return "a custom MoE all-to-all dispatcher is not supported"
    if enable_fused_moe_sum_all_reduce:
        return "the external fused MoE sum all-reduce must be disabled"
    return None


def should_use_fp8_fused_rs(size_m: int) -> bool:
    """Use fused-RS only for validated routed-assignment counts."""
    return size_m <= _FUSED_RS_MAX_SIZE_M


def prepare_fp8_fused_rs(layer: torch.nn.Module) -> bool:
    """Eager, load-time prep for the FP8 fused-RS path. Returns True if enabled."""
    return _prepare_fp8_fused_rs(layer, allow_no_combine=False)


def _prepare_fp8_fused_rs(layer: torch.nn.Module, *, allow_no_combine: bool) -> bool:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec, get_parallel

    runner_config = layer.moe_runner_config
    parallel = get_parallel()
    ep_size = parallel.moe_ep_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    unsupported_reason = fp8_fused_rs_unsupported_reason(
        ep_size=ep_size,
        tp_size=get_tp_group().world_size,
        pp_size=parallel.pp_size,
        world_size=world_size,
        reduction_compatible=layer.allow_backend_reduction,
        attn_dp_size=parallel.attn_dp_size,
        block_quantized=getattr(layer, "weight_block_size", None) is not None,
        gated=runner_config.is_gated,
        activation=runner_config.activation,
        apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
        no_combine=runner_config.no_combine and not allow_no_combine,
        has_fused_shared_experts=bool(layer.num_fused_shared_experts),
        standard_dispatcher=get_moe_a2a_backend().is_none(),
        enable_dp_attention=parallel.enable_dp_attention,
        enable_fused_moe_sum_all_reduce=(
            get_exec().moe.enable_fused_moe_sum_all_reduce
        ),
    )
    if unsupported_reason is not None:
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = unsupported_reason
        return False
    layer.__dict__.pop("_tpu_fused_rs_unsupported_reason", None)

    # prepare_tpu_fp8_moe_weights stores weights as (E, K, N) and scales as
    # (E, 1, 1, N) channelwise or (E, K // block, 1, N) blockwise.
    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import prepare_fused_moe_rs

    prepare_fused_moe_rs(layer)
    return True


def apply_fp8_fused_rs(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
) -> StandardCombineInput:
    return _apply_fp8_fused_rs(
        layer=layer,
        dispatch_output=dispatch_output,
        activation=activation,
        post_expert_norm_weight=None,
        finalize_fallback=None,
    )


def _apply_fp8_fused_rs(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    post_expert_norm_weight: torch.Tensor | None = None,
    finalize_fallback: Callable | None = None,
) -> StandardCombineInput:
    x = dispatch_output.hidden_states
    has_post_expert_norm = post_expert_norm_weight is not None
    if has_post_expert_norm != (finalize_fallback is not None):
        raise ValueError(
            "post_expert_norm_weight and finalize_fallback must be provided together"
        )
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk = int(topk_ids.shape[-1])
    block_quantized = getattr(layer, "weight_block_size", None) is not None
    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
        fused_moe_rs_gather,
        fused_moe_rs_padded_tokens,
    )

    ep_size = int(layer._tpu_fused_rs_physical_order.shape[0])
    padded_tokens = fused_moe_rs_padded_tokens(x.shape[0], topk, ep_size)
    size_m = padded_tokens * topk
    if not should_use_fp8_fused_rs(size_m):
        from sglang.srt.distributed import moe_expert_parallel_all_reduce
        from sglang.srt.hardware_backend.tpu.fp8_moe import apply_tpu_fp8_moe

        stock_output = apply_tpu_fp8_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=activation,
            block_quant=block_quantized,
            apply_router_weight_on_input=(
                layer.moe_runner_config.apply_router_weight_on_input
            ),
            no_combine=layer.moe_runner_config.no_combine,
        )
        if finalize_fallback is not None:
            stock_output = finalize_fallback(stock_output, dispatch_output)
        # The model-level EP reduction is disabled whenever fused-RS is
        # selected, so large-prefill fallback must perform it here.
        return StandardCombineInput(
            hidden_states=moe_expert_parallel_all_reduce(stock_output.hidden_states)
        )

    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    valid = topk_ids >= 0
    topk_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    topk_weights = topk_weights * valid.to(topk_weights.dtype)

    norm_weight = (
        post_expert_norm_weight
        if post_expert_norm_weight is not None
        else x.new_zeros((1,))
    )
    w1_scale = layer.w13_weight_scale_inv if block_quantized else layer.w13_weight_scale
    w2_scale = layer.w2_weight_scale_inv if block_quantized else layer.w2_weight_scale

    result = fused_moe_rs_gather(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=norm_weight,
        group_offset=layer._tpu_fused_rs_group_offset,
        topk=topk,
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        fp8_direct_write=True,
        physical_order_to_tp_rank=layer._tpu_fused_rs_physical_order,
    )
    return StandardCombineInput(hidden_states=result[: x.shape[0]])
