# SPDX-License-Identifier: Apache-2.0

"""Model-facing contracts for TPU fused-RS sequence parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from typing import NamedTuple

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_parallel

__all__ = [
    "DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN",
    "FusedMoeSequenceParallelPlan",
    "PreparedFusedMoeSequenceParallelInput",
    "fused_moe_entry_group",
    "get_tpu_fused_rs_post_expert_norm_weight",
    "get_fused_moe_sequence_parallel_plan",
    "has_tpu_fused_rs_post_expert_norm",
    "prepare_fused_moe_sequence_parallel_input",
    "prepare_tpu_fused_rs_post_expert_norm",
    "use_fused_moe_sequence_parallel_plan",
    "uses_tpu_fused_rs_dp_attention_layout",
]

_POST_EXPERT_NORM_CAPABILITY = "supports_tpu_fused_rs_post_expert_norm"
_POST_EXPERT_NORM_WEIGHT_GETTER = "tpu_fused_rs_post_expert_norm_weight"
_CACHED_POST_EXPERT_NORM_CAPABILITY = "_tpu_fused_rs_has_post_expert_norm"
_CACHED_POST_EXPERT_NORM_WEIGHT = "_tpu_fused_rs_post_expert_norm_weight"
_DP_ATTENTION_LAYOUT_CAPABILITY = "tpu_fused_rs_dp_attention_layout"


class FusedMoeSequenceParallelPlan(NamedTuple):
    """Layout presented by a model to one fused-MoE forward."""

    gather_hidden_states: bool
    gather_topk: bool
    keep_output_shard: bool


class PreparedFusedMoeSequenceParallelInput(NamedTuple):
    dispatch_output: StandardDispatchOutput
    plan: FusedMoeSequenceParallelPlan
    output_token_count: int


DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN = FusedMoeSequenceParallelPlan(
    False, False, False
)


def has_tpu_fused_rs_post_expert_norm(layer: torch.nn.Module) -> bool:
    """Whether fused-RS may own this layer's norm, combine, and reduction."""
    return bool(getattr(layer, _POST_EXPERT_NORM_CAPABILITY, False))


def uses_tpu_fused_rs_dp_attention_layout(layer: torch.nn.Module) -> bool:
    """Whether the model feeds DP-attention rows in the gathered fused-RS layout.

    Under DP attention each group holds different rows, so the in-kernel
    reduce-scatter is correct only when every attention-TP rank passes its own
    share of its group's rows and the entry is gathered over the whole TP group.
    """
    return bool(getattr(layer, _DP_ATTENTION_LAYOUT_CAPABILITY, False))


def fused_moe_entry_group():
    """Return the group the fused-RS expert entry is gathered over."""
    parallel = get_parallel()
    # TP ranks are attention-DP-major, so a TP-group gather is group-major and
    # output block r holds exactly the rows TP rank r passed.
    if parallel.attn_dp_size > 1:
        return parallel.tp_group
    return parallel.attn_tp_group


def _all_gather_topk_packed(group, topk_weights, topk_ids):
    # One fp32 all-gather instead of two; expert IDs are exact in fp32.
    top_k = topk_weights.shape[-1]
    packed = torch.cat((topk_weights.float(), topk_ids.float()), dim=-1)
    packed = group.all_gather(packed, dim=0)
    return (
        packed[:, :top_k].to(topk_weights.dtype).contiguous(),
        packed[:, top_k:].to(topk_ids.dtype).contiguous(),
    )


def prepare_tpu_fused_rs_post_expert_norm(layer: torch.nn.Module) -> None:
    """Validate and cache the model-provided post-expert norm contract."""
    has_post_expert_norm = has_tpu_fused_rs_post_expert_norm(layer)
    layer.__dict__[_CACHED_POST_EXPERT_NORM_CAPABILITY] = has_post_expert_norm
    if not has_post_expert_norm:
        return

    weight_getter = getattr(layer, _POST_EXPERT_NORM_WEIGHT_GETTER, None)
    if not callable(weight_getter):
        raise TypeError(
            "TPU fused-RS post-expert norm requires a model-provided weight hook"
        )

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

    layer.allow_backend_reduction = True
    layer.register_buffer(
        _CACHED_POST_EXPERT_NORM_WEIGHT,
        weight,
        persistent=False,
    )


def get_tpu_fused_rs_post_expert_norm_weight(
    layer: torch.nn.Module,
) -> torch.Tensor | None:
    """Return the prepared norm weight, or ``None`` when the model opted out."""
    if not getattr(layer, _CACHED_POST_EXPERT_NORM_CAPABILITY, False):
        return None
    weight = getattr(layer, _CACHED_POST_EXPERT_NORM_WEIGHT, None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("TPU fused-RS post-expert norm was enabled but not prepared")
    return weight


def get_fused_moe_sequence_parallel_plan() -> FusedMoeSequenceParallelPlan:
    """Return the model-published plan for the active forward."""
    from sglang.srt.runtime_context import get_forward

    plan = get_forward().tpu_fused_rs_sequence_parallel_plan
    if plan is None:
        return DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN
    if not isinstance(plan, FusedMoeSequenceParallelPlan):
        raise TypeError(
            "the scoped TPU fused-RS sequence-parallel plan must be a "
            "FusedMoeSequenceParallelPlan"
        )
    return plan


@contextmanager
def use_fused_moe_sequence_parallel_plan(plan: FusedMoeSequenceParallelPlan):
    """Publish one model-owned layout plan for the duration of a forward."""
    if not isinstance(plan, FusedMoeSequenceParallelPlan):
        raise TypeError("plan must be a FusedMoeSequenceParallelPlan")

    from sglang.srt.runtime_context import get_forward

    with get_forward().scoped(tpu_fused_rs_sequence_parallel_plan=plan):
        yield


def prepare_fused_moe_sequence_parallel_input(
    layer, dispatch_output
) -> PreparedFusedMoeSequenceParallelInput:
    """Apply the declared expert-entry gathers and derive output row count."""
    plan = get_fused_moe_sequence_parallel_plan()
    if plan == DISABLED_FUSED_MOE_SEQUENCE_PARALLEL_PLAN:
        return PreparedFusedMoeSequenceParallelInput(
            dispatch_output=dispatch_output,
            plan=plan,
            output_token_count=dispatch_output.hidden_states.shape[0],
        )

    if not isinstance(dispatch_output, StandardDispatchOutput):
        raise TypeError(
            "TPU fused-RS sequence parallelism requires standard dispatch output"
        )
    if (
        dispatch_output.hidden_states_scale is not None
        or dispatch_output.hidden_states_pre_quant is not None
        or dispatch_output.hidden_states.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires BF16 expert input"
        )

    topk_output = dispatch_output.topk_output
    if not isinstance(topk_output, StandardTopKOutput):
        raise TypeError(
            "TPU fused-RS sequence parallelism requires standard top-k output"
        )
    if dispatch_output.hidden_states.ndim != 2:
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires rank-2 hidden states"
        )
    if topk_output.topk_weights.ndim != 2 or topk_output.topk_ids.ndim != 2:
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires rank-2 top-k metadata"
        )
    if topk_output.topk_weights.shape != topk_output.topk_ids.shape:
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires top-k weights and IDs "
            "with the same shape"
        )

    parallel = get_parallel()
    dp_attention = parallel.attn_dp_size > 1
    if dp_attention and not uses_tpu_fused_rs_dp_attention_layout(layer):
        raise RuntimeError(
            "TPU fused-RS sequence parallelism under DP attention requires the "
            "model's gathered DP-attention layout"
        )
    group = fused_moe_entry_group()
    tp_group = getattr(parallel, "tp_group", None)
    fused_rs_world_size = layer.moe_ep_size
    if group.world_size != fused_rs_world_size:
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires matching attention-TP "
            f"and fused-RS groups, got {group.world_size} and {fused_rs_world_size}"
        )
    attn_ranks = getattr(group, "ranks", None)
    tp_ranks = getattr(tp_group, "ranks", None)
    if (
        attn_ranks is not None
        and tp_ranks is not None
        and tuple(attn_ranks) != tuple(tp_ranks)
    ):
        raise RuntimeError(
            "TPU fused-RS sequence parallelism requires attention-TP and TP "
            f"groups in the same rank order, got {attn_ranks} and {tp_ranks}"
        )

    hidden_token_count = dispatch_output.hidden_states.shape[0]
    topk_token_count = topk_output.topk_weights.shape[0]
    global_hidden_token_count = hidden_token_count * (
        fused_rs_world_size if plan.gather_hidden_states else 1
    )
    global_topk_token_count = topk_token_count * (
        fused_rs_world_size if plan.gather_topk else 1
    )
    if global_hidden_token_count != global_topk_token_count:
        raise RuntimeError(
            "TPU fused-RS sequence-parallel plan declares inconsistent input "
            f"layouts: hidden states produce {global_hidden_token_count} global "
            f"rows but top-k metadata produces {global_topk_token_count}"
        )
    if plan.keep_output_shard and global_hidden_token_count % fused_rs_world_size:
        raise RuntimeError(
            "TPU fused-RS sequence-parallel output rows must be divisible by "
            f"TP={fused_rs_world_size}, got {global_hidden_token_count}"
        )

    hidden_states = dispatch_output.hidden_states
    if plan.gather_hidden_states:
        hidden_states = group.all_gather(hidden_states, dim=0).contiguous()

    if plan.gather_topk and dp_attention:
        topk_weights, topk_ids = _all_gather_topk_packed(
            group, topk_output.topk_weights, topk_output.topk_ids
        )
    elif plan.gather_topk:
        topk_weights = group.all_gather(topk_output.topk_weights, dim=0).contiguous()
        topk_ids = group.all_gather(topk_output.topk_ids, dim=0).contiguous()
    else:
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

    if hidden_states.shape[0] != global_hidden_token_count:
        raise RuntimeError(
            "TPU fused-RS hidden-state all-gather returned an unexpected shape: "
            f"got {tuple(hidden_states.shape)}, expected first dimension "
            f"{global_hidden_token_count}"
        )
    expected_topk_shape = (global_topk_token_count, topk_output.topk_weights.shape[-1])
    if (
        topk_weights.shape != expected_topk_shape
        or topk_ids.shape != expected_topk_shape
    ):
        raise RuntimeError(
            "TPU fused-RS top-k all-gather returned unexpected shapes: "
            f"weights {tuple(topk_weights.shape)}, IDs {tuple(topk_ids.shape)}, "
            f"expected {expected_topk_shape}"
        )

    transformed = dispatch_output._replace(
        hidden_states=hidden_states,
        topk_output=topk_output._replace(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=None,
        ),
    )
    output_token_count = global_hidden_token_count
    if plan.keep_output_shard:
        output_token_count //= fused_rs_world_size
    return PreparedFusedMoeSequenceParallelInput(
        dispatch_output=transformed,
        plan=plan,
        output_token_count=output_token_count,
    )
