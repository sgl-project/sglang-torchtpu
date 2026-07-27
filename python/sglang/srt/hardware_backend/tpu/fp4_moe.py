# SPDX-License-Identifier: Apache-2.0
"""TPU NVFP4 (``modelopt_fp4``) MoE helpers.

Serves block-512 NVFP4 MoE weights via the packed-FP4 path of the Pallas
``gmm_v2`` kernel. Checkpoint global scales are folded into the per-block scales.
No native fp4 MXU on TPU. For block-512 weights, gmm_v2 online-quantizes
activations to FP8 and uses the W4A8 path. Smaller quantization blocks are not
supported because they regress performance. The static ``input_scale`` is unused.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe import _clear_replaced_param
from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _pad_gate_up_halves_axis1,
    _pad_last_axis,
)
from sglang.srt.layers.utils import copy_or_rebind_param

# Block-512 matches the FP8 activation quantization block, enabling a 512-deep
# W4A8 MXU contraction. Block-16 fragments the contraction or requires bf16
# dequantization and measured slower than bf16, so it is intentionally rejected.
TPU_NVFP4_GROUP_SIZE = 512
PACKING = 2  # two FP4 codes per uint8 byte

# Models with post-expert normalization opt in and provide the semantic hooks;
# the NVFP4 adapter only validates and caches that downstream-owned contract.
_POST_EXPERT_NORM_CAPABILITY = "supports_tpu_fused_rs_post_expert_norm"
_POST_EXPERT_NORM_WEIGHT_GETTER = "tpu_fused_rs_post_expert_norm_weight"
_POST_EXPERT_NORM_FALLBACK = "finalize_tpu_fused_rs_fallback"

# Load-bearing, not a tuning knob: a single fused-RS invocation over a full
# 4096-token prefill reproducibly triggered RuntimeUnexpectedCoreHalt during the
# EXTEND4096 precompile on tpu7x. Chunking the call at 1024 tokens avoids it, so
# do not raise this without re-validating that precompile shape on device.
_FUSED_RS_MAX_TOKENS_PER_CALL = 1024


def has_tpu_fused_rs_post_expert_norm(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, _POST_EXPERT_NORM_CAPABILITY, False))


def prepare_tpu_fused_rs_post_expert_norm(layer: torch.nn.Module) -> None:
    """Validate and cache the downstream-owned post-expert norm contract."""
    has_post_expert_norm = has_tpu_fused_rs_post_expert_norm(layer)
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
    # This is an alias of the model-owned parameter, not a second checkpoint
    # parameter, so bypass nn.Module registration.
    layer.__dict__["_tpu_fused_rs_post_expert_norm_weight"] = weight


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
    """
    return scale.permute(0, 2, 1).contiguous().unsqueeze(2).to(dtype)


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


def can_use_tpu_nvfp4_fused_rs(
    layer: torch.nn.Module,
    *,
    group_size: int,
    no_combine: bool,
) -> bool:
    """Return whether fused-RS can honor this layer's MoE contract."""
    from sglang.srt.distributed import get_tp_group, get_world_group
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_parallel, get_server_args

    parallel = get_parallel()
    server_args = get_server_args()
    runner_config = layer.moe_runner_config
    return (
        group_size == TPU_NVFP4_GROUP_SIZE
        and layer.moe_ep_size > 1
        and layer.moe_ep_size
        == get_tp_group().world_size
        == get_world_group().world_size
        and layer.allow_backend_reduction
        and parallel.attn_dp_size == 1
        and runner_config.is_gated
        and runner_config.activation == "silu"
        and not runner_config.apply_router_weight_on_input
        # Same contract as the shared FP8 fused-RS path: the standard path is
        # already combined, while a model that declares a compatible
        # post-expert norm asks for per-expert output so the norm can run
        # before the weighted combination.
        and no_combine == has_tpu_fused_rs_post_expert_norm(layer)
        and not layer.num_fused_shared_experts
        and get_moe_a2a_backend().is_none()
        and not server_args.enable_dp_attention
        and not server_args.enable_fused_moe_sum_all_reduce
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
    """
    if group_size != TPU_NVFP4_GROUP_SIZE:
        raise NotImplementedError(
            f"TPU NVFP4 MoE requires group_size={TPU_NVFP4_GROUP_SIZE}, "
            f"got {group_size}."
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
        from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
            prepare_fused_moe_rs,
        )

        prepare_tpu_fused_rs_post_expert_norm(layer)
        prepare_fused_moe_rs(layer)
        w13_w = _materialize_packed_fp4(_transpose_packed_fp4(w13_w))
        w2_w = _materialize_packed_fp4(_transpose_packed_fp4(w2_w))
    else:
        w13_w = w13_w.permute(0, 2, 1).contiguous()
        w2_w = w2_w.permute(0, 2, 1).contiguous()

    copy_or_rebind_param(layer, "w13_weight", w13_w)
    copy_or_rebind_param(layer, "w2_weight", w2_w)
    copy_or_rebind_param(layer, "w13_weight_scale", w13_scale)
    copy_or_rebind_param(layer, "w2_weight_scale", w2_scale)

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

    if use_fused_rs:
        from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
            fused_moe_rs_gather,
        )

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        valid = topk_ids >= 0
        topk_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
        topk_weights = topk_weights * valid.to(topk_weights.dtype)

        # Cached by prepare_tpu_fused_rs_post_expert_norm at load time; the
        # can_use_tpu_nvfp4_fused_rs gate guarantees no_combine agrees with it.
        has_post_expert_norm = layer._tpu_fused_rs_has_post_expert_norm
        post_expert_norm_weight = (
            layer._tpu_fused_rs_post_expert_norm_weight
            if has_post_expert_norm
            else hidden_states.new_zeros((1,))
        )
        outputs = []
        for token_start in range(
            0, hidden_states.shape[0], _FUSED_RS_MAX_TOKENS_PER_CALL
        ):
            token_end = min(
                token_start + _FUSED_RS_MAX_TOKENS_PER_CALL,
                hidden_states.shape[0],
            )
            output = fused_moe_rs_gather(
                hidden_states=hidden_states[token_start:token_end],
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                topk_weights=topk_weights[token_start:token_end],
                topk_ids=topk_ids[token_start:token_end],
                post_expert_norm_weight=post_expert_norm_weight,
                group_offset=layer._tpu_fused_rs_group_offset,
                topk=int(topk_ids.shape[-1]),
                activation=activation,
                has_post_expert_norm=has_post_expert_norm,
                # NVFP4 weights already force the W4A8 path, so GMM2 consumes an
                # fp8-quantized activation regardless. Writing the fused-RS
                # payload as fp8 as well would re-quantize an already-lossy
                # intermediate for no bandwidth win, so keep the bf16 payload.
                fp8_direct_write=False,
                collective_id=layer._tpu_fused_rs_collective_id,
                physical_order_to_tp_rank=layer._tpu_fused_rs_physical_order,
            )
            outputs.append(output[: token_end - token_start])
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
