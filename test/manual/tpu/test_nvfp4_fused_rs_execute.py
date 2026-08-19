# SPDX-License-Identifier: Apache-2.0
"""TP16/EP16 execution and routing-sync gate for production NVFP4 fused-RS."""

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
    ).to(torch.float32)
    if torch.equal(topk_weights, topk_weights.to(torch.bfloat16).to(torch.float32)):
        raise AssertionError("router-weight fixture must exercise non-BF16 FP32 bits")
    return hidden_states, topk_ids, topk_weights


def _physical_order_to_rank(
    physical_id: int, world_size: int, device: torch.device
) -> list[int]:
    local_id = torch.tensor([physical_id], dtype=torch.int32, device=device)
    ids_by_rank = [torch.empty_like(local_id) for _ in range(world_size)]
    dist.all_gather(ids_by_rank, local_id)
    physical_ids = [int(value.cpu().item()) for value in ids_by_rank]
    if sorted(physical_ids) != list(range(world_size)):
        raise AssertionError(f"unexpected physical device IDs: {physical_ids}")
    return [physical_ids.index(pid) for pid in range(world_size)]


def _gather_logical_rows(
    local_shard: torch.Tensor,
    physical_order_to_rank: list[int],
    tokens: int,
) -> torch.Tensor:
    shards_by_rank = [torch.empty_like(local_shard) for _ in physical_order_to_rank]
    dist.all_gather(shards_by_rank, local_shard)
    physical_order = [shards_by_rank[rank] for rank in physical_order_to_rank]
    return torch.cat(physical_order, dim=0)[:tokens].cpu()


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
    physical_order_to_rank = _physical_order_to_rank(physical_id, world_size, device)
    group_offset = torch.tensor(
        [rank * local_experts, physical_id], dtype=torch.int32, device=device
    )
    post_norm = torch.linspace(
        -0.25, 0.25, hidden_size, dtype=torch.bfloat16, device=device
    )

    def kernel(hidden_states, topk_weights, topk_ids):
        return fused_moe_rs_op.fused_moe_rs_call(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            post_expert_norm_weight=post_norm,
            group_offset=group_offset,
            topk=topk,
            activation="silu",
            has_post_expert_norm=True,
            fp8_direct_write=False,
            postprocess_accum_dtype="float32",
        )

    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    for tokens in (1, 16, 64, 256, 1024):
        hidden_states, topk_ids, topk_weights = _make_inputs(
            tokens, hidden_size, global_experts, topk
        )

        def run_case(label: str, weights: torch.Tensor) -> torch.Tensor:
            local_shard = compiled(
                hidden_states.to(device),
                weights.to(device),
                topk_ids.to(device),
            )
            logical_rows = _gather_logical_rows(
                local_shard,
                physical_order_to_rank,
                tokens,
            )
            dist.barrier()
            print(
                "NVFP4_FUSED_RS_CASE_PASS "
                f"rank={rank} physical_id={physical_id} tokens={tokens} "
                f"case={label} local_shape={tuple(local_shard.shape)} "
                f"logical_shape={tuple(logical_rows.shape)}",
                flush=True,
            )
            return logical_rows

        reference = run_case("identical", topk_weights)
        repeated = run_case("identical_repeat", topk_weights)

        # Only non-source physical devices receive deliberately different FP32
        # weights. The compiled kernel must still consume physical device 0's
        # exact values, including its sub-BF16 residual.
        divergent_weights = topk_weights
        if physical_id != 0:
            divergent_weights = torch.flip(topk_weights, dims=(1,)) * (physical_id + 1)
        divergent = run_case("divergent_non_source", divergent_weights)

        # This arm uses the same shape/dtype and compiled function, but removes
        # only the FP32 bits below the BF16 grid. The production 1024-token
        # chunk must change after the final BF16 output round, otherwise its
        # gate is not sensitive to the numerics this patch claims to preserve.
        rounded_weights = topk_weights.to(torch.bfloat16).to(torch.float32)
        bf16_grid = run_case("source_bf16_grid", rounded_weights)

        if not torch.isfinite(reference).all().item():
            raise AssertionError(f"non-finite logical output for tokens={tokens}")
        checksum = reference.float().abs().sum().item()
        if checksum == 0:
            raise AssertionError(f"zero logical output for tokens={tokens}")
        torch.testing.assert_close(reference, repeated, rtol=0, atol=0)
        torch.testing.assert_close(reference, divergent, rtol=0, atol=0)
        low_bit_changed = not torch.equal(reference, bf16_grid)
        if tokens == 1024 and not low_bit_changed:
            raise AssertionError(
                "FP32 router weights and their BF16-grid projection produced "
                "identical outputs for the production 1024-token chunk"
            )
        print(
            "NVFP4_FUSED_RS_EXECUTE_PASS "
            f"rank={rank} physical_id={physical_id} tokens={tokens} "
            f"checksum={checksum:.6f} low_bit_changed={low_bit_changed}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
