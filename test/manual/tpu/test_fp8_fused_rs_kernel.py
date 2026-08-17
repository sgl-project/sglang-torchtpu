# SPDX-License-Identifier: Apache-2.0
"""Numerical TP8/EP8 test for the compiled FP8 fused-RS Pallas kernel."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

import torch
import torch.distributed as dist
from torch_tpu._internal.distributed import tpu_distributed


def _quantize_channelwise(weight: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.abs().amax(dim=1).clamp_min(1e-6) / fp8_max
    quantized = (weight / scale[:, None, :]).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.bfloat16)


def _quantize_blockwise(weight: torch.Tensor, block_size: int = 512):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    num_experts, k_size, n_size = weight.shape
    assert k_size % block_size == 0
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        num_experts,
        k_size // block_size,
        1,
        n_size,
        dtype=torch.bfloat16,
    )
    dequantized = torch.empty_like(weight)
    for k_start in range(0, k_size, block_size):
        k_end = k_start + block_size
        block = weight[:, k_start:k_end, :]
        block_scale = block.abs().amax(dim=1).clamp_min(1e-6) / fp8_max
        block_quantized = (block / block_scale[:, None, :]).clamp(-fp8_max, fp8_max)
        block_quantized = block_quantized.to(torch.float8_e4m3fn)
        quantized[:, k_start:k_end, :] = block_quantized
        stored_scale = block_scale.to(torch.bfloat16)
        scale[:, k_start // block_size, 0, :] = stored_scale
        dequantized[:, k_start:k_end, :] = (
            block_quantized.float() * stored_scale.float()[:, None, :]
        )
    return quantized, scale, dequantized


def _reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    post_expert_norm_weight=None,
):
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token in range(hidden_states.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = hidden_states[token].float() @ w1[expert]
            gate, up = gate_up.chunk(2)
            expert_output = (torch.nn.functional.silu(gate) * up) @ w2[expert]
            if post_expert_norm_weight is not None:
                expert_output = expert_output * (1 + post_expert_norm_weight.float())
                variance = expert_output.square().mean()
                expert_output = expert_output * torch.rsqrt(variance + 1e-8)
            output[token] += topk_weights[token, slot].float() * expert_output
    return output.to(torch.bfloat16)


def main():
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8

    torch.manual_seed(7)
    test_post_expert_norm = (
        os.environ.get("SGLANG_TPU_FUSED_RS_TEST_POST_EXPERT_NORM", "0") == "1"
    )
    test_blockwise = os.environ.get("SGLANG_TPU_FUSED_RS_TEST_BLOCKWISE", "0") == "1"
    test_divergent_weights = (
        os.environ.get("SGLANG_TPU_FUSED_RS_TEST_DIVERGENT_WEIGHTS", "0") == "1"
    )
    test_divergent_ids = (
        os.environ.get("SGLANG_TPU_FUSED_RS_TEST_DIVERGENT_IDS", "0") == "1"
    )
    fp8_direct_write = (
        os.environ.get("SGLANG_TPU_FUSED_RS_TEST_FP8_DIRECT_WRITE", "0") == "1"
    )
    tokens = int(os.environ.get("SGLANG_TPU_FUSED_RS_TEST_TOKENS", "64"))
    topk = int(os.environ.get("SGLANG_TPU_FUSED_RS_TEST_TOPK", "2"))
    hidden_size = intermediate_size = 1024 if test_blockwise else 256
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16) * 0.1
    w1_float = (
        torch.randn(world_size, hidden_size, 2 * intermediate_size, dtype=torch.float32)
        * 0.02
    )
    w2_float = (
        torch.randn(world_size, intermediate_size, hidden_size, dtype=torch.float32)
        * 0.02
    )
    if test_blockwise:
        # Separate adjacent K-block magnitudes so layout mistakes are visible.
        w1_float[:, :512, :] *= 0.03125
        w2_float[:, 512:1024, :] *= 0.03125
        w1_quantized, w1_scale, w1_dequantized = _quantize_blockwise(w1_float)
        w2_quantized, w2_scale, w2_dequantized = _quantize_blockwise(w2_float)
    else:
        w1_quantized, w1_scale = _quantize_channelwise(w1_float)
        w2_quantized, w2_scale = _quantize_channelwise(w2_float)
        w1_dequantized = w1_quantized.float() * w1_scale.float()[:, None, :]
        w2_dequantized = w2_quantized.float() * w2_scale.float()[:, None, :]

    token_ids = torch.arange(tokens, dtype=torch.int32)[:, None]
    slots = torch.arange(topk, dtype=torch.int32)[None, :]
    topk_ids = (token_ids + slots) % world_size
    topk_weights = torch.full((tokens, topk), 1 / topk, dtype=torch.bfloat16)
    post_expert_norm_weight = torch.linspace(
        -0.2, 0.2, hidden_size, dtype=torch.bfloat16
    )
    expected = _reference(
        hidden_states,
        w1_dequantized,
        w2_dequantized,
        topk_weights,
        topk_ids,
        post_expert_norm_weight if test_post_expert_norm else None,
    )

    device = torch.device("tpu")
    physical_id = tpu_distributed.global_device_id()
    local_expert = rank
    kernel_topk_ids = topk_ids
    if test_divergent_ids and physical_id != 0:
        # A synthetic mismatch ensures that the synchronized ID slice remains
        # in the compiled dataflow. It does not imply that production routing
        # IDs have been observed to diverge.
        kernel_topk_ids = (topk_ids + physical_id) % world_size
    kernel_topk_weights = topk_weights
    if test_divergent_weights and physical_id != 0:
        # The compiled fused-RS path must consume physical device 0's routing
        # weights. Keep IDs identical because production ablations do not show
        # that routing IDs diverge or need to be canonicalized.
        kernel_topk_weights = topk_weights * (physical_id + 1)

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
        _build_fused_moe_rs_op,
        fused_moe_rs_call,
        fused_moe_rs_padded_tokens,
    )

    _build_fused_moe_rs_op()

    def kernel(x, w1, w2, s1, s2, weights, ids, group_offset, norm_weight):
        valid = ids >= 0
        ids = torch.where(valid, ids, torch.zeros_like(ids))
        weights = weights * valid.to(weights.dtype)
        return fused_moe_rs_call(
            hidden_states=x,
            w1=w1,
            w2=w2,
            w1_scale=s1,
            w2_scale=s2,
            topk_weights=weights,
            topk_ids=ids,
            post_expert_norm_weight=norm_weight,
            group_offset=group_offset,
            topk=topk,
            activation="silu",
            has_post_expert_norm=test_post_expert_norm,
            fp8_direct_write=fp8_direct_write,
        )

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    w1_scale_local = w1_scale[local_expert : local_expert + 1]
    w2_scale_local = w2_scale[local_expert : local_expert + 1]
    if not test_blockwise:
        w1_scale_local = w1_scale_local[:, None, None, :]
        w2_scale_local = w2_scale_local[:, None, None, :]
    kernel_args = (
        hidden_states.to(device),
        w1_quantized[local_expert : local_expert + 1].to(device),
        w2_quantized[local_expert : local_expert + 1].to(device),
        w1_scale_local.to(device),
        w2_scale_local.to(device),
        kernel_topk_weights.to(device),
        kernel_topk_ids.to(device),
        torch.tensor([local_expert, physical_id], dtype=torch.int32, device=device),
    )
    actual = compiled(
        *kernel_args,
        (
            post_expert_norm_weight.to(device)
            if test_post_expert_norm
            else torch.zeros(1, dtype=torch.bfloat16, device=device)
        ),
    ).cpu()

    padded_tokens = fused_moe_rs_padded_tokens(tokens, topk, world_size)
    if padded_tokens != tokens:
        expected = torch.nn.functional.pad(expected, (0, 0, 0, padded_tokens - tokens))
    shard_size = padded_tokens // world_size
    shard = slice(physical_id * shard_size, (physical_id + 1) * shard_size)
    torch.testing.assert_close(
        actual,
        expected[shard],
        rtol=0.10 if test_post_expert_norm else 0.08,
        atol=0.15 if test_post_expert_norm else 0.03,
    )
    print(
        f"FP8_FUSED_RS_KERNEL_PASS rank={rank} physical_id={physical_id} "
        f"shape={tuple(actual.shape)} post_norm={test_post_expert_norm} "
        f"blockwise={test_blockwise} divergent_ids={test_divergent_ids} "
        f"divergent_weights={test_divergent_weights} "
        f"fp8_direct_write={fp8_direct_write}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
