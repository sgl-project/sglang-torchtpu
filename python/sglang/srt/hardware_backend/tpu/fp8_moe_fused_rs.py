# SPDX-License-Identifier: Apache-2.0

"""FP8 fused reduce-scatter MoE path for TPU.

Routes the FP8 MoE through the in-tree SGLang-owned fused-RS kernel
(`fused_moe_rs_gather`) instead of `fused_moe_call` + a separate all-reduce.
Load-time `prepare_fp8_fused_rs` validates the supported runtime layout and
caches physical-device ordering. Large prefills use periodic compiler
partitions to keep routing and fused-RS calls within the validated graph size.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

_FUSED_RS_INLINE_MAX_SIZE_M = 2048
_FUSED_RS_MAX_SIZE_M = 32768
# The TPU compiler miscompiles 12 large fused-RS routing/kernel regions in one
# executable and trips a generated SparseCore descriptor-queue bounds check.
# Isolating every 11th layer keeps each ordinary partition below that boundary.
_FUSED_RS_PARTITION_INTERVAL = 11


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
    if block_quantized:
        return "block-quantized FP8 weights are not supported"
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


def should_partition_fp8_fused_rs(size_m: int, layer_id: int) -> bool:
    """Return whether this layer is a large-prefill compiler boundary."""
    return (
        size_m > _FUSED_RS_INLINE_MAX_SIZE_M
        and (layer_id + 1) % _FUSED_RS_PARTITION_INTERVAL == 0
    )


def prepare_fp8_fused_rs(layer: torch.nn.Module) -> bool:
    """Eager, load-time prep for the FP8 fused-RS path. Returns True if enabled."""
    return _prepare_fp8_fused_rs(layer, allow_no_combine=False)


def _prepare_fp8_fused_rs(layer: torch.nn.Module, *, allow_no_combine: bool) -> bool:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_world_size,
    )

    ep_size = get_moe_expert_parallel_world_size()
    runner_config = layer.moe_runner_config

    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_parallel, get_server_args

    parallel = get_parallel()
    server_args = get_server_args()
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
        enable_dp_attention=server_args.enable_dp_attention,
        enable_fused_moe_sum_all_reduce=(server_args.enable_fused_moe_sum_all_reduce),
    )
    if unsupported_reason is not None:
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = unsupported_reason
        return False
    layer.__dict__.pop("_tpu_fused_rs_unsupported_reason", None)

    # prepare_tpu_fp8_moe_weights already stores weights as (E, K, N) and
    # channelwise scales as (E, 1, 1, N), exactly as fused-RS expects.
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
    size_m = x.shape[0] * topk_ids.shape[-1]
    if not should_use_fp8_fused_rs(size_m):
        from sglang.srt.distributed import moe_expert_parallel_all_reduce
        from sglang.srt.hardware_backend.tpu.fp8_moe import apply_tpu_fp8_moe

        stock_output = apply_tpu_fp8_moe(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=activation,
            block_quant=False,
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

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import fused_moe_rs_gather

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

    large_prefill = size_m > _FUSED_RS_INLINE_MAX_SIZE_M
    result = fused_moe_rs_gather(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=norm_weight,
        group_offset=layer._tpu_fused_rs_group_offset,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        # Large prefills use the validated BF16-payload specialization. The
        # FP8 row-scale staging path is retained for decode and small prefills.
        fp8_direct_write=not large_prefill,
        partition_boundary=should_partition_fp8_fused_rs(size_m, int(layer.layer_id)),
        physical_order_to_tp_rank=layer._tpu_fused_rs_physical_order,
    )
    return StandardCombineInput(hidden_states=result[: x.shape[0]])
