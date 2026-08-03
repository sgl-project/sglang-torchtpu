# SPDX-License-Identifier: Apache-2.0
"""TPU W4A16 (int4 weight-only) MoE helpers for compressed-tensors
pack-quantized checkpoints (e.g. Kimi-K2.5).

Follows the same shape as fp8_moe.py: weights are formatted once after
loading, then runtime dispatches to the Pallas fused grouped-GEMM bridge
(``fused_moe_call`` with ``w_quant_bits=4``).

Checkpoint format (CompressedTensorsWNA16MoE.create_weights, group strategy,
symmetric, no actorder):
  * ``w13_weight_packed``  [E, K//8, 2N] int32 — 8 consecutive K-values per
    word, LSB-first nibbles, stored as UNSIGNED offset-binary (q + 8).
  * ``w2_weight_packed``   [E, N//8, K]  int32
  * ``w13_weight_scale``   [E, K//group_size, 2N] bf16
  * ``w2_weight_scale``    [E, N//group_size, K]  bf16

``pltpu.bitcast(..., int4)`` unpacks 8 consecutive K-values per word but
reads them as SIGNED two's complement, so each nibble's top bit is flipped
once at load time (XOR 0x88888888): (q+8) ^ 0x8 is exactly the two's
complement encoding of q. Same conversion torchtpu-vllm uses for Kimi-K2.6
INT4 (commit 62b04dea). After that, the packed int32 words are handed to
gmm_v2 as-is — its uint32 bitcast is a same-width reinterpret. Scales keep
their checkpoint dtype (the kernel casts per-call) and only gain the
``[E, num_blocks, 1, out]`` layout.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_moe_expert_parallel_rank
from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.tpu_moe_utils import mask_tpu_moe_non_local_experts
from sglang.srt.layers.utils import copy_or_rebind_param


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


def prepare_tpu_w4a16_moe_weights(layer: torch.nn.Module, group_size: int) -> None:
    """One-time post-load formatting for the Pallas W4A16 MoE path.

    Keeps the packed int32 weights exactly as loaded (their [E, K//8, N]
    orientation is already gmm_v2's (g, k, n) convention) and converts the
    group scales to the kernel's [E, num_blocks, 1, out] fp32 layout.
    """
    if getattr(layer, "is_tpu_w4a16_converted", False):
        return

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
        # [E, num_blocks, out] -> [E, num_blocks, 1, out]; keep the
        # checkpoint dtype — gmm_v2 casts scales to f32 per call.
        new_scale = old_scale.unsqueeze(2).contiguous()
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
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported on TPU yet."
        )

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    if layer.num_fused_shared_experts:
        raise NotImplementedError(
            "Fused shared experts are not supported on the TPU MoE path."
        )
    topk_weights, topk_ids = mask_tpu_moe_non_local_experts(
        topk_weights,
        topk_ids,
        layer.num_local_experts,
        get_moe_expert_parallel_rank(),
    )

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
        w_quant_bits=4,
    )
    return StandardCombineInput(hidden_states=output)
