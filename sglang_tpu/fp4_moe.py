# SPDX-License-Identifier: Apache-2.0
"""TPU NVFP4 (``modelopt_fp4``) MoE helpers.

Serves block-16/32/64/128/256/512 NVFP4 MoE weights via the packed-FP4 path of
Pallas ``gmm_v2`` kernels. Checkpoint global scales are folded into the per-block
scales. No native fp4 MXU on TPU. Block-512 and block-256 weights use W4A8;
blocks 16/32/64/128 use W4A16. Blocks below 512 require the fused-RS kernel because
the stock Pallas packed-FP4 path supports block-512 only. The static
``input_scale`` is unused.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _pad_gate_up_halves_axis1,
    _pad_last_axis,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_moe import _clear_replaced_param, _own_storage
from sglang_tpu.fused_rs_sequence_parallel import (
    get_tpu_fused_rs_post_expert_norm_weight,
    has_tpu_fused_rs_post_expert_norm,
    prepare_fused_moe_sequence_parallel_input,
    prepare_tpu_fused_rs_post_expert_norm,
)
from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

# Block-512 matches the original FP8 activation quantization block. The fused-RS
# kernel additionally supports block-256 with W4A8 and blocks 16-128 with W4A16.
# Keep the singular constant as the default/alignment contract for compatibility.
TPU_NVFP4_GROUP_SIZE = 512
TPU_NVFP4_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128, 256, TPU_NVFP4_GROUP_SIZE)
PACKING = 2  # two FP4 codes per uint8 byte

# Keep a bounded fallback above the largest configured serving bucket. The 8K
# bucket uses one fused-RS invocation; smaller 1K invocations reload expert
# weights and repeat kernel setup inside the same compiled graph. Calls above
# 8K remain deterministically chunked.
_FUSED_RS_MAX_TOKENS_PER_CALL = 8192


def _transpose_packed_fp4(matrix: torch.Tensor) -> torch.Tensor:
    """Transpose ``(E, M, K/2)`` packed FP4 into ``(E, K, M/2)`` bytes."""
    if matrix.dtype != torch.uint8 or matrix.ndim != 3:
        raise ValueError("packed FP4 matrix must be a rank-3 uint8 tensor")
    if matrix.shape[1] % PACKING:
        raise ValueError("packed FP4 transpose requires an even row dimension")

    even_rows = matrix[:, 0::2, :]
    odd_rows = matrix[:, 1::2, :]
    even_columns = (even_rows & 0x0F) | ((odd_rows & 0x0F) << 4)
    odd_columns = ((even_rows >> 4) & 0x0F) | (odd_rows & 0xF0)
    return (
        torch.stack((even_columns, odd_columns), dim=3)
        .permute(0, 2, 3, 1)
        .reshape(
            matrix.shape[0],
            matrix.shape[2] * PACKING,
            matrix.shape[1] // PACKING,
        )
        .contiguous()
    )


def _materialize_packed_fp4(packed: torch.Tensor) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError(
            "SGLang fused-RS requires TorchTPU native torch.float4_e2m1fn_x2 support"
        )
    output = torch.empty_like(packed, dtype=fp4_dtype)
    output.copy_(packed.view(fp4_dtype))
    return output


# NVFP4 create_weights params the TPU gmm_v2 path doesn't consume after global
# scales are folded into the block scales (input scales and CUDA layouts unused).
_UNUSED_NVFP4_MOE_PARAMS = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "w13_blockscale_mma",
    "w2_blockscale_mma",
    "g1_alphas",
    "g2_alphas",
    "w13_input_scale_quant",
    "w2_input_scale_quant",
)


def _to_gmm_v2_scale_layout(
    scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``(E, OUT, num_blocks_K)`` -> gmm_v2 ``(E, num_blocks_K, 1, OUT)``.

    The stored scale ``dtype`` defaults to fp32. Passing ``torch.float8_e4m3fn``
    (or ``torch.bfloat16``) re-quantizes the scale to a narrower type,
    cutting the per-block scale stream 4x (fp8) / 2x (bf16) — gmm_v2 sizes its
    scale traffic from the tensor's own dtype, so no kernel change is needed
    beyond honoring ``scale.dtype``.

    Materialize the declared rank-4 layout in its own buffer. Adding a size-1
    dimension to a contiguous rank-3 tensor produces a contiguous view, so a
    later ``contiguous()`` call alone cannot guarantee independent storage.
    """
    return _own_storage(scale.to(dtype).unsqueeze(-1).permute(0, 2, 3, 1).contiguous())


def _expert_global_scales(
    w13_global_scale: torch.Tensor | None,
    w2_global_scale: torch.Tensor | None,
    *,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert globals as f32 ``(E, 2)`` gate/up and ``(E,)``, ones where absent.

    Built by computing ops: a view of a parameter bound into a compiled graph is
    rebuilt from its base on every call (torch-tpu 0.2).
    """
    ones = torch.ones(num_experts, 1, dtype=torch.float32, device=device)
    if w13_global_scale is None:
        w13 = torch.cat([ones, ones], dim=1)
    else:
        g = w13_global_scale.float().reshape(num_experts, -1)
        w13 = torch.cat([g[:, :1], g[:, -1:]], dim=1)
    if w2_global_scale is None:
        w2 = ones.reshape(num_experts) * 1.0
    else:
        w2 = w2_global_scale.float().reshape(num_experts) * 1.0
    return w13, w2


def _fold_w13_global_scale(
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Fold per-expert gate/up global scales into W13 block scales."""
    scale = block_scale.float()
    global_scale = global_scale.float()
    if global_scale.ndim == 2 and global_scale.shape[1] > 1:
        intermediate = scale.shape[1] // 2
        scale[:, :intermediate] *= global_scale[:, 0:1, None]
        scale[:, intermediate:] *= global_scale[:, 1:2, None]
    else:
        scale *= global_scale.reshape(global_scale.shape[0], -1)[:, 0, None, None]
    return scale


def unmet_tpu_nvfp4_fused_rs_requirements(
    layer: torch.nn.Module,
    *,
    group_size: int,
    no_combine: bool,
) -> list[str]:
    """Return the fused-RS contract requirements this layer does not meet."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang_tpu.moe.expert_tp import is_fused_moe_rs_expert_tp

    parallel = get_parallel()
    runner_config = layer.moe_runner_config
    requirements = (
        (
            f"group_size in {TPU_NVFP4_SUPPORTED_GROUP_SIZES}",
            group_size in TPU_NVFP4_SUPPORTED_GROUP_SIZES,
        ),
        (
            "moe_ep_size == tp_size == world_size > 1, or expert TP "
            "(moe_ep_size == 1, moe_tp_size == tp_size == world_size > 1)",
            parallel.tp_group.world_size == parallel.world_group.world_size
            and (
                layer.moe_ep_size == parallel.tp_group.world_size > 1
                or is_fused_moe_rs_expert_tp(
                    layer.moe_ep_size,
                    getattr(parallel, "moe_tp_size", 1),
                    parallel.tp_group.world_size,
                )
            ),
        ),
        ("layer allows backend reduction", layer.allow_backend_reduction),
        ("attn_dp_size == 1", parallel.attn_dp_size == 1),
        ("gated MoE", runner_config.is_gated),
        (
            "silu activation",
            runner_config.activation in ("silu", "silu_and_mul_clamp"),
        ),
        (
            "router weight not applied on input",
            not runner_config.apply_router_weight_on_input,
        ),
        # Same contract as the shared FP8 fused-RS path: the standard path is
        # already combined, while a model that declares a compatible
        # post-expert norm asks for per-expert output so the norm can run
        # before the weighted combination.
        (
            "no_combine matches the model's post-expert-norm contract",
            no_combine == has_tpu_fused_rs_post_expert_norm(layer),
        ),
        ("no fused shared experts", not layer.num_fused_shared_experts),
        ("standard token dispatcher", get_moe_a2a_backend().is_none()),
        ("dp attention disabled", not parallel.attn_dp_enabled),
        (
            "fused MoE sum all-reduce disabled",
            not get_exec().moe.enable_fused_moe_sum_all_reduce,
        ),
    )
    return [name for name, met in requirements if not met]


def can_use_tpu_nvfp4_fused_rs(
    layer: torch.nn.Module,
    *,
    group_size: int,
    no_combine: bool,
) -> bool:
    """Return whether fused-RS can honor this layer's MoE contract."""
    return not unmet_tpu_nvfp4_fused_rs_requirements(
        layer, group_size=group_size, no_combine=no_combine
    )


def prepare_tpu_nvfp4_moe_weights(
    layer: torch.nn.Module,
    *,
    group_size: int = TPU_NVFP4_GROUP_SIZE,
    align_to: int = TPU_NVFP4_GROUP_SIZE,
    scale_dtype: torch.dtype = torch.float32,
    use_fused_rs: bool = False,
) -> None:
    """Format NVFP4 MoE weights/scales for the TPU gmm_v2 backend.

    Keeps weights packed (uint8), stores the checkpoint's effective per-block
    scale as ``scale_dtype``, transposes to gmm_v2 layout, and pads the
    intermediate to ``align_to``.

    ``group_size`` is the per-block scale granularity along the contracting dim.
    TPU serving folds any checkpoint global scale into each per-block scale.

    Layout in (checkpoint) -> out (local gmm_v2), with ``G = group_size``:
      * ``w13_weight``       (E, 2I, H/2) u8  -> (E, H/2, 2*I_pad) u8
      * ``w13_weight_scale`` (E, 2I, H/G) f8  -> (E, H/G, 1, 2*I_pad) scale_dtype
      * ``w2_weight``        (E, H, I/2) u8   -> (E, I_pad/2, H) u8
      * ``w2_weight_scale``  (E, H, I/G) f8   -> (E, I_pad/G, 1, H) scale_dtype

    The fused-RS kernel takes the scales without the unit dim, (E, K/G, N).
    """
    if group_size not in TPU_NVFP4_SUPPORTED_GROUP_SIZES:
        raise NotImplementedError(
            "TPU NVFP4 MoE requires group_size in "
            f"{TPU_NVFP4_SUPPORTED_GROUP_SIZES}, "
            f"got {group_size}."
        )
    if group_size != TPU_NVFP4_GROUP_SIZE and not use_fused_rs:
        raise NotImplementedError(
            f"TPU NVFP4 MoE group_size={group_size} requires the fused-RS "
            "backend; the stock Pallas packed-FP4 path supports only "
            f"group_size={TPU_NVFP4_GROUP_SIZE}."
        )

    if not layer.moe_runner_config.is_gated:
        # GMM1 always splits its output in half for the gated SwiGLU, so a
        # non-gated expert would be mis-activated. Reject rather than miscompute.
        raise NotImplementedError("TPU NVFP4 MoE only supports gated (SwiGLU) experts.")

    w13 = layer.w13_weight.data  # (E, 2I, H/2) uint8
    w2 = layer.w2_weight.data  # (E, H, I/2) uint8
    _E, two_I, _hidden_packed = w13.shape
    intermediate = two_I // 2

    w13_s = layer.w13_weight_scale.data.float()
    w2_s = layer.w2_weight_scale.data.float()
    w13_global_scale = getattr(layer, "w13_weight_scale_2", None)
    w2_global_scale = getattr(layer, "w2_weight_scale_2", None)
    w13_gs = w2_gs = None
    if use_fused_rs and tpu_envs.SGLANG_TPU_NVFP4_E4M3_SCALES.get():
        # The block scales stay e4m3 as stored; the kernels apply the globals.
        w13_gs, w2_gs = _expert_global_scales(
            None if w13_global_scale is None else w13_global_scale.data,
            None if w2_global_scale is None else w2_global_scale.data,
            num_experts=_E,
            device=w13.device,
        )
        scale_dtype = torch.float8_e4m3fn
    else:
        if w13_global_scale is not None:
            w13_s = _fold_w13_global_scale(w13_s, w13_global_scale.data)
        if w2_global_scale is not None:
            w2_s *= w2_global_scale.data.float()[:, None, None]

    # Pad intermediate to align_to; pad_n must stay a multiple of group_size and
    # PACKING to keep scale blocks and packed bytes whole (enforced below).
    pad_n = (align_to - intermediate % align_to) % align_to
    if pad_n % group_size != 0 or pad_n % PACKING != 0:
        raise ValueError(
            f"align_to={align_to} incompatible with group_size={group_size}: "
            f"pad_n={pad_n} must be a multiple of both {group_size} and {PACKING}."
        )

    # W13: pad gate/up halves independently, then transpose to gmm_v2 (E, K, N).
    w13_w = _pad_gate_up_halves_axis1(w13, intermediate, pad_n)
    w13_s = _pad_gate_up_halves_axis1(w13_s, intermediate, pad_n)
    w13_scale = _to_gmm_v2_scale_layout(w13_s, scale_dtype)  # (E, H/G, 1, 2*I_pad)

    # W2: pad packed-K and scale blocks to match, then transpose so N = hidden.
    pad_k_packed = pad_n // PACKING
    pad_k_blocks = pad_n // group_size
    w2_w = _pad_last_axis(w2, pad_k_packed)
    w2_s = _pad_last_axis(w2_s, pad_k_blocks)  # (E, H, I_pad/G)
    w2_scale = _to_gmm_v2_scale_layout(w2_s, scale_dtype)  # (E, I_pad/G, 1, H)

    if use_fused_rs:
        from sglang_tpu.moe.pallas.fused_moe_rs_op import (
            prepare_fused_moe_rs,
        )

        prepare_tpu_fused_rs_post_expert_norm(layer)
        prepare_fused_moe_rs(layer)
        w13_w = _materialize_packed_fp4(_transpose_packed_fp4(w13_w))
        w2_w = _materialize_packed_fp4(_transpose_packed_fp4(w2_w))
        # Fused-RS reads [E, K_blocks, N]. The unit dim of the gmm_v2 layout is
        # stored major-most on TPU while the kernel needs it tiled, so XLA would
        # relayout both scale tensors before every call. Transpose directly: a
        # clone of a squeezed view is rebuilt from its base on every compiled
        # call (torch-tpu 0.2).
        w13_scale = w13_s.to(scale_dtype).permute(0, 2, 1).contiguous()
        w2_scale = w2_s.to(scale_dtype).permute(0, 2, 1).contiguous()
    else:
        w13_w = w13_w.permute(0, 2, 1).contiguous()
        w2_w = w2_w.permute(0, 2, 1).contiguous()

    copy_or_rebind_param(layer, "w13_weight", w13_w)
    copy_or_rebind_param(layer, "w2_weight", w2_w)
    copy_or_rebind_param(layer, "w13_weight_scale", w13_scale)
    copy_or_rebind_param(layer, "w2_weight_scale", w2_scale)
    layer.register_buffer("_tpu_w13_global_scale", w13_gs, persistent=False)
    layer.register_buffer("_tpu_w2_global_scale", w2_gs, persistent=False)

    for name in _UNUSED_NVFP4_MOE_PARAMS:
        _clear_replaced_param(layer, name)


def apply_tpu_nvfp4_moe(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool = False,
    use_fused_rs: bool = False,
    swiglu_limit: float | None = None,
):
    """Run TPU NVFP4 MoE through the Pallas grouped-GEMM backend.

    With ``no_combine=True`` returns raw per-slot outputs ``[num_tokens, topk,
    hidden]`` (non-local EP slots zeroed) for a caller doing its own post-expert
    norm + combine.
    """
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    routed_scaling_factor = layer.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        # Neither the fused-RS nor the gmm_v2 combine applies it, and the
        # reduction-layer interactions here are not validated with an
        # rsf != 1 model. Refuse rather than serving the routed channels at
        # the wrong magnitude.
        raise NotImplementedError(
            "routed_scaling_factor is not yet applied on the TPU NVFP4 path; "
            "use the BF16 checkpoint or validate against a concrete model "
            "first."
        )

    if swiglu_limit is not None:
        if activation != "silu":
            raise ValueError("TPU clamped MoE requires silu activation")
        activation = f"silu_and_mul_clamp:{float(swiglu_limit)}"

    if use_fused_rs:
        from sglang_tpu.moe.pallas.fused_moe_rs_op import (
            fused_moe_rs_gather,
        )

        prepared_input = prepare_fused_moe_sequence_parallel_input(
            layer, dispatch_output
        )
        dispatch_output = prepared_input.dispatch_output
        keep_output_shard = prepared_input.plan.keep_output_shard
        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        valid = topk_ids >= 0
        topk_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
        topk_weights = topk_weights * valid.to(topk_weights.dtype)

        post_expert_norm_weight = get_tpu_fused_rs_post_expert_norm_weight(layer)
        has_post_expert_norm = post_expert_norm_weight is not None
        if post_expert_norm_weight is None:
            post_expert_norm_weight = hidden_states.new_zeros((1,))

        expert_tp = layer.__dict__.get("_tpu_fused_rs_expert_tp", False)
        # Expert TP runs with moe_ep_size == 1 and shards over the MoE TP ranks.
        tp_world_size = layer.moe_tp_size if expert_tp else layer.moe_ep_size
        if keep_output_shard:
            expected_global_rows = prepared_input.output_token_count * tp_world_size
            if hidden_states.shape[0] != expected_global_rows:
                raise RuntimeError(
                    "TPU fused-RS sequence-parallel input has an unexpected "
                    f"token count: got {hidden_states.shape[0]}, expected "
                    f"{expected_global_rows}"
                )
            partition_count = tp_world_size
            rows_per_partition = prepared_input.output_token_count
        else:
            partition_count = 1
            rows_per_partition = hidden_states.shape[0]

        rows_per_call = _FUSED_RS_MAX_TOKENS_PER_CALL // partition_count
        if rows_per_call == 0:
            raise RuntimeError(
                "TPU fused-RS world size exceeds the NVFP4 token limit: "
                f"TP={tp_world_size}, limit={_FUSED_RS_MAX_TOKENS_PER_CALL}"
            )

        hidden_by_partition = hidden_states.reshape(
            partition_count, rows_per_partition, hidden_states.shape[-1]
        )
        weights_by_partition = topk_weights.reshape(
            partition_count, rows_per_partition, topk_weights.shape[-1]
        )
        ids_by_partition = topk_ids.reshape(
            partition_count, rows_per_partition, topk_ids.shape[-1]
        )

        outputs = []
        for token_start in range(0, rows_per_partition, rows_per_call):
            token_end = min(token_start + rows_per_call, rows_per_partition)
            output_rows = token_end - token_start
            chunk_hidden = hidden_by_partition[:, token_start:token_end].flatten(0, 1)
            chunk_weights = weights_by_partition[:, token_start:token_end].flatten(0, 1)
            chunk_ids = ids_by_partition[:, token_start:token_end].flatten(0, 1)

            output = fused_moe_rs_gather(
                hidden_states=chunk_hidden,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                topk_weights=chunk_weights,
                topk_ids=chunk_ids,
                post_expert_norm_weight=post_expert_norm_weight,
                group_offset=layer._tpu_fused_rs_group_offset,
                topk=int(topk_ids.shape[-1]),
                activation=activation,
                has_post_expert_norm=has_post_expert_norm,
                # Blocks 16-128 use W4A16; block-256/512 already quantize the
                # GMM activation. A direct FP8 write would add another loss.
                fp8_direct_write=False,
                world_size=tp_world_size,
                keep_token_shard=keep_output_shard,
                expert_tp=expert_tp,
                oneshot_rank=getattr(layer, "_tpu_oneshot_rank", None),
                w1_global_scale=layer._tpu_w13_global_scale,
                w2_global_scale=layer._tpu_w2_global_scale,
            )
            outputs.append(output[:output_rows])
        return StandardCombineInput(hidden_states=torch.cat(outputs, dim=0))

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)

    topk_weights, topk_ids = mask_tpu_moe_non_local_experts(
        topk_weights,
        topk_ids,
        layer.num_local_experts,
        layer.moe_ep_rank,
    )

    output = fused_moe_call(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        no_combine=no_combine,
    )
    return StandardCombineInput(hidden_states=output)
