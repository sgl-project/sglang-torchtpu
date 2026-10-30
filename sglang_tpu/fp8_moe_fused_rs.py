# SPDX-License-Identifier: Apache-2.0

"""FP8 fused reduce-scatter MoE path for TPU.

Routes the FP8 MoE through the in-tree SGLang-owned fused-RS kernel
(`fused_moe_rs_gather`) instead of `fused_moe_call` + a separate all-reduce.
Load-time `prepare_fp8_fused_rs` validates the supported runtime layout and
caches physical-device ordering.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang_tpu.fused_rs_sequence_parallel import (
    get_tpu_fused_rs_post_expert_norm_weight,
    has_tpu_fused_rs_post_expert_norm,
    prepare_fused_moe_sequence_parallel_input,
    prepare_tpu_fused_rs_post_expert_norm,
)


def supports_fp8_fused_rs(
    *,
    num_experts: int,
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
            num_experts=num_experts,
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
    num_experts: int,
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
    expert_tp: bool = False,
    gate_up_width: int | None = None,
) -> str | None:
    """Return the first reason the fused-RS backend cannot be selected.

    ``expert_tp`` (EP=1, every expert's intermediate dim sharded over TP) runs
    with the full TP width as the kernel's reduction axis; ``gate_up_width`` is
    then the per-rank width of the fused gate/up projection.
    """
    if expert_tp:
        # The kernel pads the gate/up width only at its tail, which would split
        # gate from up at the wrong column.
        if gate_up_width is None or gate_up_width % 256:
            return (
                "expert TP needs a per-rank gate/up width that is a multiple of "
                f"256, got {gate_up_width}"
            )
        ep_size = tp_size
    if ep_size <= 1:
        return f"expert parallel size must be greater than 1, got {ep_size}"
    if num_experts > 256:
        return (
            "BF16 routing materialization supports at most 256 experts, got "
            f"{num_experts}"
        )
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
    if activation not in ("silu", "silu_and_mul_clamp"):
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


def prepare_fp8_fused_rs(layer: torch.nn.Module) -> bool:
    """Eager, load-time prep for the FP8 fused-RS path. Returns True if enabled."""
    has_post_expert_norm = has_tpu_fused_rs_post_expert_norm(layer)
    prepare_tpu_fused_rs_post_expert_norm(layer)
    return _prepare_fp8_fused_rs(
        layer,
        allow_no_combine=has_post_expert_norm,
    )


def _prepare_fp8_fused_rs(layer: torch.nn.Module, *, allow_no_combine: bool) -> bool:
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang_tpu.moe.expert_tp import is_fused_moe_rs_expert_tp

    w13_weight = getattr(layer, "w13_weight", None)
    if w13_weight is not None and w13_weight.dtype == torch.uint8:
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = (
            "packed-FP4 uint8 MoE weights use the gmm_v2 backend"
        )
        return False

    runner_config = layer.moe_runner_config
    parallel = get_parallel()
    ep_size = parallel.moe_ep_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expert_tp = is_fused_moe_rs_expert_tp(
        ep_size, parallel.moe_tp_size, parallel.tp_group.world_size
    )
    unsupported_reason = fp8_fused_rs_unsupported_reason(
        num_experts=layer.num_experts,
        ep_size=ep_size,
        tp_size=get_parallel().tp_group.world_size,
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
        enable_dp_attention=parallel.attn_dp_enabled,
        enable_fused_moe_sum_all_reduce=(
            get_exec().moe.enable_fused_moe_sum_all_reduce
        ),
        expert_tp=expert_tp,
        gate_up_width=2 * layer.intermediate_size_per_partition,
    )
    if unsupported_reason is not None:
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = unsupported_reason
        return False
    layer.__dict__.pop("_tpu_fused_rs_unsupported_reason", None)

    # prepare_tpu_fp8_moe_weights stores weights as (E, K, N) and scales as
    # (E, 1, 1, N) channelwise or (E, K // block, 1, N) blockwise.
    from sglang_tpu.moe.pallas.fused_moe_rs_op import prepare_fused_moe_rs

    prepare_fused_moe_rs(layer)
    if expert_tp:
        _replicate_channelwise_fused_rs_scales(layer)
    return True


def _replicate_channelwise_fused_rs_scales(layer: torch.nn.Module) -> None:
    """Store channelwise fused-RS FP8 scales as (E, 2, N) instead of (E, 1, 1, N).

    torch-tpu stores a size-1 K-block dim major-most ({2,0,1:T(8,128)}), but the
    kernel operand needs it tiled ({2,1,0:T(2,128)}), so XLA relayouts both scale
    tensors before every call.
    Replicating the channel scale into two K blocks gives a real second-minor dim
    whose default layout already matches. The kernel then scales two K/2 partial
    matmuls; K/2 stays >= the 256-deep MXU, so the scale is still applied after
    the matmul.
    """
    from sglang.srt.layers.utils import copy_or_rebind_param
    from sglang_tpu.fp8_moe import _own_storage

    if getattr(layer, "weight_block_size", None) is not None:
        return
    names = ("w13", "w2")
    # The kernel needs both scales in one layout: replicate both or neither.
    for name in names:
        scale = getattr(layer, f"{name}_weight_scale").data
        k = getattr(layer, f"{name}_weight").shape[1]
        if scale.dim() != 4 or scale.shape[1:3] != (1, 1) or k % 2 or k // 2 < 256:
            return
    for name in names:
        scale = getattr(layer, f"{name}_weight_scale").data
        replicated = scale.squeeze(2).expand(-1, 2, -1).contiguous()
        copy_or_rebind_param(layer, f"{name}_weight_scale", _own_storage(replicated))


def apply_fp8_fused_rs(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    swiglu_limit: float | None = None,
) -> StandardCombineInput:
    return _apply_fp8_fused_rs(
        layer=layer,
        dispatch_output=dispatch_output,
        activation=activation,
        swiglu_limit=swiglu_limit,
        post_expert_norm_weight=get_tpu_fused_rs_post_expert_norm_weight(layer),
    )


def _apply_fp8_fused_rs(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    swiglu_limit: float | None = None,
    post_expert_norm_weight: torch.Tensor | None = None,
) -> StandardCombineInput:
    routed_scaling_factor = layer.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        # The fused-RS reduction stack is not validated with an rsf != 1
        # model; refuse rather than serving the routed channels at the
        # wrong magnitude.
        raise NotImplementedError(
            "routed_scaling_factor is not yet applied on the TPU FP8 "
            "fused-RS path; use the BF16 checkpoint or the standard FP8 "
            "path first."
        )
    if swiglu_limit is not None:
        if activation != "silu":
            raise ValueError("TPU clamped MoE requires silu activation")
        activation = f"silu_and_mul_clamp:{float(swiglu_limit)}"
    prepared_input = prepare_fused_moe_sequence_parallel_input(layer, dispatch_output)
    dispatch_output = prepared_input.dispatch_output
    keep_output_shard = prepared_input.plan.keep_output_shard
    x = dispatch_output.hidden_states
    has_post_expert_norm = post_expert_norm_weight is not None
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk = int(topk_ids.shape[-1])
    block_quantized = getattr(layer, "weight_block_size", None) is not None
    expert_tp = layer.__dict__.get("_tpu_fused_rs_expert_tp", False)
    from sglang_tpu.moe.pallas.fused_moe_rs_op import fused_moe_rs_gather

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
        # Keep the fused-RS payload in BF16 for both the legacy all-gather and
        # sequence-parallel local-shard paths. A direct FP8 write would add row
        # quantization and scale DMAs before a BF16 consumer.
        fp8_direct_write=False,
        # Expert TP runs with moe_ep_size == 1 and shards over the MoE TP ranks.
        world_size=layer.moe_tp_size if expert_tp else layer.moe_ep_size,
        keep_token_shard=keep_output_shard,
        expert_tp=expert_tp,
        oneshot_rank=getattr(layer, "_tpu_oneshot_rank", None),
    )
    return StandardCombineInput(
        hidden_states=result[: prepared_input.output_token_count]
    )
