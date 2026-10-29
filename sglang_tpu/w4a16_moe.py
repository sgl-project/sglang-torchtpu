# SPDX-License-Identifier: Apache-2.0
"""TPU int4 MoE helpers for compressed-tensors pack-quantized checkpoints (Kimi-K2.x).

Weights are formatted once after loading and served through ``fused_moe_call``
with ``w_quant_bits=4``. Checkpoint layout (group strategy, symmetric):
  * ``w13_weight_packed`` [E, K//8, 2N] int32, ``w2_weight_packed`` [E, N//8, K]:
    8 K-values per word, LSB-first nibbles, unsigned offset-binary (q + 8).
  * ``w13_weight_scale`` [E, K//group, 2N], ``w2_weight_scale`` [E, N//group, K] bf16.

``pltpu.bitcast(..., int4)`` reads nibbles as two's complement, so each nibble's
top bit is flipped once at load ((q+8) ^ 0x8 == int4(q)); the words then go to
gmm_v2 as-is. Scales are stored in the kernel's ``[E, blocks, 1, out]`` layout,
f32 when ``SGLANG_TPU_INT4_MOE_FP32_SCALES`` is set, else bf16.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.fp8_moe import _own_storage


def _free_replaced_storage(old: torch.Tensor, new: torch.Tensor) -> None:
    if old.untyped_storage().data_ptr() != new.untyped_storage().data_ptr():
        old.untyped_storage().resize_(0)


def _drop_param(layer: torch.nn.Module, name: str) -> None:
    """Release a load-time-only parameter (g_idx bookkeeping etc.)."""
    param = getattr(layer, name, None)
    if isinstance(param, Parameter):
        old = param.data
        param.data = torch.empty(0, dtype=old.dtype, device=old.device)
        _free_replaced_storage(old, param.data)
    setattr(layer, name, None)


def _unpack_int4_offset_binary(packed: torch.Tensor) -> torch.Tensor:
    """[.., K//8, N] int32 of LSB-first offset-binary nibbles -> [.., K, N] int8."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    nibbles = (packed.unsqueeze(-2) >> shifts.view(1, -1, 1)) & 0xF
    values = (nibbles - 8).to(torch.int8)
    return values.reshape(*packed.shape[:-2], packed.shape[-2] * 8, packed.shape[-1])


def prepare_tpu_vmem_fp8_scales(layer: torch.nn.Module, group_size: int) -> None:
    """Keep INT4 words; prepare normalized group and FP8 channel scales once."""
    if getattr(layer, "is_tpu_vmem_fp8", False):
        return
    if group_size != 32:
        raise ValueError("vmem-fp8 requires INT4 group size 32")
    for packed_name, scale_name, channel_name in (
        ("w13_weight_packed", "w13_weight_scale", "w13_vmem_channel_scale"),
        ("w2_weight_packed", "w2_weight_scale", "w2_vmem_channel_scale"),
    ):
        packed = getattr(layer, packed_name).data
        old_scale = getattr(layer, scale_name).data
        normalized = torch.empty(old_scale.shape, dtype=torch.float32, device="cpu")
        channels = torch.empty(
            (packed.shape[0], 1, packed.shape[-1]), dtype=torch.float32
        )
        for e in range(packed.shape[0]):
            scale = old_scale[e].detach().cpu().to(torch.bfloat16)
            q = _unpack_int4_offset_binary(packed[e].detach().cpu())
            # Column max = max over groups of max|q| * |scale|; no dequantized copy.
            group_max = q.reshape(-1, group_size, q.shape[-1]).abs().amax(dim=1)
            group_max = group_max.to(torch.bfloat16) * scale.abs()
            c = (group_max.float().amax(dim=0, keepdim=True) / 448.0).clamp_min(1e-12)
            normalized[e] = scale.float() / c
            channels[e] = c
        new = normalized.to(old_scale.device)
        copy_or_rebind_param(layer, scale_name, new)
        setattr(
            layer,
            channel_name,
            Parameter(channels.to(old_scale.device), requires_grad=False),
        )
        _free_replaced_storage(old_scale, getattr(layer, scale_name).data)
    layer.is_tpu_vmem_fp8 = True


def prepare_tpu_w4a16_moe_weights(layer: torch.nn.Module, group_size: int) -> None:
    """One-time post-load formatting for the Pallas W4A16 MoE path.

    Keeps the packed int32 weights exactly as loaded (their [E, K//8, N]
    orientation is already gmm_v2's (g, k, n) convention) and converts the
    group scales to the kernel's [E, num_blocks, 1, out] layout, f32 when
    SGLANG_TPU_INT4_MOE_FP32_SCALES=1, else bf16.
    """
    if getattr(layer, "is_tpu_w4a16_converted", False):
        return

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.tpu_moe_utils import (
        INT4_MOE_MODE_VMEM_FP8,
        resolve_int4_moe_mode,
    )

    mode = resolve_int4_moe_mode()
    if mode == INT4_MOE_MODE_VMEM_FP8:
        prepare_tpu_vmem_fp8_scales(layer, group_size)
    # vmem-fp8 normalizes the scales to f32 and the kernel requires them so.
    scale_dtype = (
        torch.float32
        if envs.SGLANG_TPU_INT4_MOE_FP32_SCALES.get() or mode == INT4_MOE_MODE_VMEM_FP8
        else torch.bfloat16
    )

    # Offset-binary -> two's complement nibbles for pltpu's signed int4
    # unpack: flip each nibble's top bit ((q+8) ^ 0x8 == int4(q)).
    INT4_SIGN_XOR = -2004318072  # 0x88888888 as int32
    for packed_name in ("w13_weight_packed", "w2_weight_packed"):
        packed = getattr(layer, packed_name).data
        try:
            packed.bitwise_xor_(INT4_SIGN_XOR)
        except (RuntimeError, NotImplementedError):
            # torch_tpu may lack an in-place int32 xor; fall back to a
            # chunked CPU round-trip to bound host memory.
            for e in range(packed.shape[0]):
                cpu_chunk = packed[e].cpu()
                cpu_chunk.bitwise_xor_(INT4_SIGN_XOR)
                packed[e].copy_(cpu_chunk)

    for scale_name, packed_name in (
        ("w13_weight_scale", "w13_weight_packed"),
        ("w2_weight_scale", "w2_weight_packed"),
    ):
        old_scale = getattr(layer, scale_name).data
        packed = getattr(layer, packed_name).data
        num_blocks = old_scale.shape[1]
        # [E, num_blocks, out] -> [E, num_blocks, 1, out]; f32 by default so
        # gmm_v2 does not re-cast the scales every step.
        new_scale = _own_storage(old_scale.unsqueeze(2).contiguous().to(scale_dtype))
        assert packed.shape[1] * 8 == num_blocks * group_size, (
            f"{packed_name} packed K {packed.shape[1]}*8 inconsistent with "
            f"{scale_name} blocks {num_blocks}*group_size {group_size}"
        )
        copy_or_rebind_param(layer, scale_name, new_scale)
        _free_replaced_storage(old_scale, getattr(layer, scale_name).data)

    # Load-time-only bookkeeping params (no actorder on this path).
    for name in (
        "w13_weight_g_idx",
        "w2_weight_g_idx",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
        "w13_weight_shape",
        "w2_weight_shape",
    ):
        _drop_param(layer, name)

    layer.is_tpu_w4a16_converted = True


def apply_tpu_w4a16_moe(
    *,
    layer: torch.nn.Module,
    dispatch_output,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool = False,
):
    """Run TPU W4A16 MoE through the Pallas fused grouped-GEMM backend."""
    # Lazy: the Pallas bridge imports torch_tpu, which the CPU test runner lacks.
    from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    routed_scaling_factor = layer.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        # Shares the fused_moe_call combine with the hardware-validated
        # bf16/fp8 runners, but no rsf != 1 model has run on this stack;
        # refuse rather than promise an unverified magnitude.
        raise NotImplementedError(
            "routed_scaling_factor is not yet applied on the TPU w4a16 path; "
            "validate against a concrete rsf != 1 model first."
        )

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    # Padded rows carry expert id -1, which the kernel drops.
    topk_ids = topk_ids.to(torch.int32)
    if layer.num_fused_shared_experts:
        raise NotImplementedError(
            "Fused shared experts are not supported on the TPU MoE path."
        )

    # Global expert ids + this rank's group window (as the fp8 path does), so
    # the permute and combine only touch this rank's routed rows.
    global_num_experts = 0
    expert_group_offset = 0
    ep_size = get_parallel().moe_ep_size
    if ep_size > 1:
        global_num_experts = layer.num_local_experts * ep_size
        expert_group_offset = get_parallel().moe_ep_rank * layer.num_local_experts

    output = fused_moe_call(
        hidden_states=x,
        w1=layer.w13_weight_packed,
        w2=layer.w2_weight_packed,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=int(topk_ids.shape[-1]),
        activation=activation,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        no_combine=no_combine,
        global_num_experts=global_num_experts,
        expert_group_offset=expert_group_offset,
        w_quant_bits=4,
        w1_channel_scale=getattr(layer, "w13_vmem_channel_scale", None),
        w2_channel_scale=getattr(layer, "w2_vmem_channel_scale", None),
    )
    return StandardCombineInput(hidden_states=output)
