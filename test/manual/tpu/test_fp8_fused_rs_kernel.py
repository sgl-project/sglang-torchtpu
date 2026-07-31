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
    tokens = int(os.environ.get("SGLANG_TPU_FUSED_RS_TEST_TOKENS", "64"))
    topk = int(os.environ.get("SGLANG_TPU_FUSED_RS_TEST_TOPK", "2"))
    hidden_size, intermediate_size = 256, 256
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16) * 0.1
    w1_float = (
        torch.randn(world_size, hidden_size, 2 * intermediate_size, dtype=torch.float32)
        * 0.02
    )
    w2_float = (
        torch.randn(world_size, intermediate_size, hidden_size, dtype=torch.float32)
        * 0.02
    )
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

    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
        _build_fused_moe_rs_op,
        fused_moe_rs_call,
    )

    _build_fused_moe_rs_op()

    def kernel(x, w1, w2, s1, s2, weights, ids, group_offset, norm_weight):
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
            fp8_direct_write=True,
        )

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    kernel_args = (
        hidden_states.to(device),
        w1_quantized[local_expert : local_expert + 1].to(device),
        w2_quantized[local_expert : local_expert + 1].to(device),
        w1_scale[local_expert : local_expert + 1, None, None, :].to(device),
        w2_scale[local_expert : local_expert + 1, None, None, :].to(device),
        topk_weights.to(device),
        topk_ids.to(device),
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

    shard_size = tokens // world_size
    shard = slice(physical_id * shard_size, (physical_id + 1) * shard_size)
    torch.testing.assert_close(
        actual,
        expected[shard],
        rtol=0.10 if test_post_expert_norm else 0.08,
        atol=0.15 if test_post_expert_norm else 0.03,
    )
    print(
        f"FP8_FUSED_RS_KERNEL_PASS rank={rank} physical_id={physical_id} "
        f"shape={tuple(actual.shape)} post_norm={test_post_expert_norm}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
