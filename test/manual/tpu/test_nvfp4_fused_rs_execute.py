# SPDX-License-Identifier: Apache-2.0
"""TP16/EP16 execution gate for the production NVFP4 fused-RS geometry."""

import gc
import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

_REQUIRED_LIBTPU_FLAGS = (
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
    "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
    "--xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false",
    "--xla_tpu_scoped_vmem_limit_kib=24576",
)
_existing_libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    (_existing_libtpu_args, *_REQUIRED_LIBTPU_FLAGS)
).strip()
os.environ.setdefault("SGLANG_TPU_FP4_FUSED_RS", "1")

import torch
import torch.distributed as dist
from torch_tpu._internal.distributed import tpu_distributed


def _make_local_nvfp4_layer(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(1000 + rank)
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (local_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (local_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            generator=generator,
        ),
        requires_grad=False,
    )
    scale = torch.tensor(0.01, dtype=torch.float8_e4m3fn)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (local_experts, 2 * intermediate_size, hidden_size // 512),
            scale,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (local_experts, hidden_size, intermediate_size // 512),
            scale,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.ones(local_experts, 2), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.ones(local_experts), requires_grad=False
    )
    return layer


def _make_inputs(
    tokens: int,
    hidden_size: int,
    global_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2000 + tokens)
    hidden_states = (torch.randn(tokens, hidden_size, generator=generator) * 0.01).to(
        torch.bfloat16
    )
    token_ids = torch.arange(tokens, dtype=torch.int32)[:, None]
    slots = torch.arange(topk, dtype=torch.int32)[None, :]
    topk_ids = (token_ids * topk + slots * 31) % global_experts
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, generator=generator), dim=-1
    ).to(torch.bfloat16)
    return hidden_states, topk_ids, topk_weights


def main() -> None:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 16:
        raise RuntimeError(f"This execution gate requires TP16, got {world_size}")

    from sglang.srt.hardware_backend.tpu.fp4_moe import (
        prepare_tpu_nvfp4_moe_weights,
    )
    from sglang.srt.layers.moe.pallas import fused_moe_rs_op

    global_experts = 256
    local_experts = global_experts // world_size
    hidden_size = 4096
    intermediate_size = 4096
    topk = 8
    device = torch.device("tpu")

    layer = _make_local_nvfp4_layer(
        local_experts, hidden_size, intermediate_size, rank
    ).to(device)
    original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
    fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
    try:
        prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)
    finally:
        fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
    gc.collect()

    physical_id = tpu_distributed.global_device_id()
    group_offset = torch.tensor(
        [rank * local_experts, physical_id], dtype=torch.int32, device=device
    )
    no_post_norm = torch.zeros(1, dtype=torch.bfloat16, device=device)

    def kernel(hidden_states, topk_weights, topk_ids):
        return fused_moe_rs_op.fused_moe_rs_call(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            post_expert_norm_weight=no_post_norm,
            group_offset=group_offset,
            topk=topk,
            activation="silu",
            has_post_expert_norm=False,
            fp8_direct_write=False,
        )

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    for tokens in (1, 16, 64, 256):
        hidden_states, topk_ids, topk_weights = _make_inputs(
            tokens, hidden_size, global_experts, topk
        )
        previous = None
        for iteration in range(2):
            actual = compiled(
                hidden_states.to(device),
                topk_weights.to(device),
                topk_ids.to(device),
            ).cpu()
            if not torch.isfinite(actual).all().item():
                raise AssertionError(f"non-finite output for tokens={tokens}")
            checksum = actual.float().abs().sum().item()
            if checksum == 0:
                raise AssertionError(f"zero output for tokens={tokens}")
            if previous is not None:
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)
            previous = actual
            dist.barrier()
            print(
                "NVFP4_FUSED_RS_EXECUTE_PASS "
                f"rank={rank} physical_id={physical_id} tokens={tokens} "
                f"iteration={iteration} shape={tuple(actual.shape)} "
                f"checksum={checksum:.6f}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
