# SPDX-License-Identifier: Apache-2.0

"""FP8 fused reduce-scatter MoE path for TPU.

Routes the FP8 MoE through the in-tree SGLang-owned fused-RS kernel
(`fused_moe_rs_gather`) instead of `fused_moe_call` + a separate all-reduce.
Load-time `prepare_fp8_fused_rs` validates the supported runtime layout and
caches physical-device ordering. The compiled forward uses fused-RS for decode
and small prefills, with the stock FP8 MoE path for larger prefills.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

# Keep larger prefills on the stock path; the fused kernel is device-validated
# through the 256-token Qwen3 specialization.
_FUSED_RS_MAX_TOKENS = 256
_POST_EXPERT_NORM_CAPABILITY = "supports_tpu_fused_rs_post_expert_norm"
_POST_EXPERT_NORM_WEIGHT_GETTER = "tpu_fused_rs_post_expert_norm_weight"
_POST_EXPERT_NORM_FALLBACK = "finalize_tpu_fused_rs_fallback"


def _has_post_expert_norm_contract(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, _POST_EXPERT_NORM_CAPABILITY, False))


def _prepare_post_expert_norm_contract(layer: torch.nn.Module) -> None:
    """Validate and cache the model-owned post-expert norm contract."""
    has_post_expert_norm = _has_post_expert_norm_contract(layer)
    layer.__dict__["_tpu_fused_rs_has_post_expert_norm"] = has_post_expert_norm
    if not has_post_expert_norm:
        return

    weight_getter = getattr(layer, _POST_EXPERT_NORM_WEIGHT_GETTER, None)
    fallback = getattr(layer, _POST_EXPERT_NORM_FALLBACK, None)
    if not callable(weight_getter) or not callable(fallback):
        raise TypeError(
            "TPU fused-RS post-expert norm requires model-provided weight and "
            "fallback-finalization hooks"
        )

    # The model hook owns semantic validation (norm type, gain convention, and
    # epsilon). Calling it after weight loading also proves the weight exists.
    weight = weight_getter()
    if not isinstance(weight, torch.Tensor):
        raise TypeError("TPU fused-RS post-expert norm weight must be a tensor")
    if weight.ndim != 1 or weight.numel() != layer.hidden_size:
        raise RuntimeError(
            "TPU fused-RS post-expert norm weight must have shape "
            f"({layer.hidden_size},), got {tuple(weight.shape)}"
        )
    if weight.device != layer.w2_weight.device:
        raise RuntimeError(
            "TPU fused-RS post-expert norm weight must be on the MoE device"
        )
    # Bypass nn.Module registration: this is an alias of the model-owned
    # parameter, not a second checkpoint parameter.
    layer.__dict__["_tpu_fused_rs_post_expert_norm_weight"] = weight


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
    has_compatible_post_expert_norm: bool,
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
            has_compatible_post_expert_norm=has_compatible_post_expert_norm,
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
    has_compatible_post_expert_norm: bool,
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
    if no_combine != has_compatible_post_expert_norm:
        return "no_combine must match the model's post-expert norm contract"
    if has_fused_shared_experts:
        return "fused shared experts are not supported"
    if not standard_dispatcher:
        return "a custom MoE all-to-all dispatcher is not supported"
    if enable_fused_moe_sum_all_reduce:
        return "the external fused MoE sum all-reduce must be disabled"
    return None


def should_use_fp8_fused_rs(num_tokens: int) -> bool:
    """Use fused-RS only for device-validated decode and small-prefill shapes."""
    return num_tokens <= _FUSED_RS_MAX_TOKENS


def prepare_fp8_fused_rs(layer: torch.nn.Module) -> bool:
    """Eager, load-time prep for the FP8 fused-RS path. Returns True if enabled."""
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
    has_post_expert_norm = _has_post_expert_norm_contract(layer)
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
        no_combine=runner_config.no_combine,
        has_compatible_post_expert_norm=has_post_expert_norm,
        has_fused_shared_experts=bool(layer.num_fused_shared_experts),
        standard_dispatcher=get_moe_a2a_backend().is_none(),
        enable_dp_attention=server_args.enable_dp_attention,
        enable_fused_moe_sum_all_reduce=(server_args.enable_fused_moe_sum_all_reduce),
    )
    if unsupported_reason is not None:
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = unsupported_reason
        return False
    layer.__dict__.pop("_tpu_fused_rs_unsupported_reason", None)

    _prepare_post_expert_norm_contract(layer)

    # prepare_tpu_fp8_moe_weights already stores weights as (E, K, N) and
    # channelwise scales as (E, 1, 1, N), exactly as fused-RS expects.
    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import prepare_fused_moe_rs

    # fused_moe_rs_gather uses the standard get_tp_group().all_gather, which is
    # already in split_compiler._COLLECTIVE_OPS -- no register_collective_op
    # (that #2429 API is only needed for custom collectives like inplace_all_reduce).
    prepare_fused_moe_rs(layer)
    return True


def apply_fp8_fused_rs(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
) -> StandardCombineInput:
    x = dispatch_output.hidden_states
    has_post_expert_norm = layer._tpu_fused_rs_has_post_expert_norm
    if not should_use_fp8_fused_rs(x.shape[0]):
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
        if has_post_expert_norm:
            stock_output = getattr(layer, _POST_EXPERT_NORM_FALLBACK)(
                stock_output, dispatch_output
            )
        # The model-level EP reduction is disabled whenever fused-RS is
        # selected, so large-prefill fallback must perform it here.
        return StandardCombineInput(
            hidden_states=moe_expert_parallel_all_reduce(stock_output.hidden_states)
        )

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import fused_moe_rs_gather

    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    valid = topk_ids >= 0
    topk_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    topk_weights = topk_weights * valid.to(topk_weights.dtype)

    post_expert_norm_weight = (
        layer._tpu_fused_rs_post_expert_norm_weight
        if has_post_expert_norm
        else x.new_zeros((1,))
    )

    result = fused_moe_rs_gather(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=post_expert_norm_weight,
        group_offset=layer._tpu_fused_rs_group_offset,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        fp8_direct_write=True,
        collective_id=layer._tpu_fused_rs_collective_id,
        physical_order_to_tp_rank=layer._tpu_fused_rs_physical_order,
    )
    return StandardCombineInput(hidden_states=result[: x.shape[0]])
