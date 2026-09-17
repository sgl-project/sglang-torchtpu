# SPDX-License-Identifier: Apache-2.0
"""Numerical and determinism gate for the Helion NVFP4 fused-RS kernel.

Run this file with one Torch process per TPU device. The default dimensions are
the Avocado Tier5 TP16/EP16 MoE dimensions; ``--world-size`` may be lowered for
faster development checks while preserving 16 local experts per rank.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HELION_TPU_COMPILE_CAPTURE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

import torch
import torch.distributed as dist

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2,
    dtype=torch.float32,
)
_E2M1_VALUES[8:] *= -1


def _dequantize_checkpoint_rows(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )
    scales = block_scale.float().repeat_interleave(group_size, dim=-1)
    return (_E2M1_VALUES[codes.long()] * scales).to(torch.bfloat16)


def _make_inputs(
    *,
    tokens: int,
    hidden: int,
    topk: int,
    global_experts: int,
    skewed_routing: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260829 + tokens)
    hidden_states = (torch.randn(tokens, hidden, generator=generator) * 0.02).to(
        torch.bfloat16
    )
    if skewed_routing:
        topk_ids = torch.zeros(tokens, topk, dtype=torch.int32)
    else:
        token_ids = torch.arange(tokens, dtype=torch.int32)[:, None]
        slots = torch.arange(topk, dtype=torch.int32)[None, :]
        topk_ids = (token_ids * topk + slots * 31) % global_experts
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, generator=generator), dim=-1
    ).to(torch.bfloat16)
    post_norm_weight = torch.linspace(-0.2, 0.2, hidden, dtype=torch.bfloat16)
    return hidden_states, topk_ids, topk_weights, post_norm_weight


def _make_local_weights(
    *,
    rank: int,
    local_experts: int,
    hidden: int,
    intermediate: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 + rank)
    w1 = torch.randint(
        0,
        256,
        (local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (local_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    w1_scale = (
        torch.rand(
            local_experts,
            2 * intermediate,
            hidden // group_size,
            generator=generator,
        )
        * 0.015
        + 0.005
    )
    w2_scale = (
        torch.rand(
            local_experts,
            hidden,
            intermediate // group_size,
            generator=generator,
        )
        * 0.015
        + 0.005
    )
    return w1, w2, w1_scale, w2_scale


def _local_reference(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    post_norm_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    expert_offset: int,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for local_expert in range(w1.shape[0]):
        routes = (topk_ids == expert_offset + local_expert).nonzero()
        if routes.numel() == 0:
            continue
        token_indices = routes[:, 0]
        slots = routes[:, 1]
        dense_w1 = _dequantize_checkpoint_rows(
            w1[local_expert], w1_scale[local_expert], group_size
        ).transpose(0, 1)
        dense_w2 = _dequantize_checkpoint_rows(
            w2[local_expert], w2_scale[local_expert], group_size
        ).transpose(0, 1)
        gate, up = (hidden_states[token_indices] @ dense_w1).float().chunk(2, dim=-1)
        projected = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16) @ dense_w2
        scaled = projected.float() * (post_norm_weight.float() + 1)
        normalized = scaled * torch.rsqrt(
            scaled.square().mean(dim=-1, keepdim=True) + 1e-8
        )
        output.index_add_(
            0,
            token_indices,
            normalized * topk_weights[token_indices, slots, None].float(),
        )
    return output


def _run_case(
    *,
    tokens: int,
    rank: int,
    world_size: int,
    local_experts: int,
    hidden: int,
    intermediate: int,
    topk: int,
    group_size: int,
    skewed_routing: bool,
    compare_pallas: bool,
) -> None:
    from sglang.srt.hardware_backend.tpu.fp4_moe import (
        _materialize_packed_fp4,
        _to_gmm_v2_scale_layout,
        _transpose_packed_fp4,
    )
    from sglang.srt.layers.moe.helion.fused_moe_rs import (
        _padded_token_count,
        _route_block_size,
        helion_fused_moe_rs_call,
        tile_weights_for_helion_fused_rs,
    )

    global_experts = world_size * local_experts
    hidden_states, topk_ids, topk_weights, post_norm_weight = _make_inputs(
        tokens=tokens,
        hidden=hidden,
        topk=topk,
        global_experts=global_experts,
        skewed_routing=skewed_routing,
    )
    w1, w2, w1_scale, w2_scale = _make_local_weights(
        rank=rank,
        local_experts=local_experts,
        hidden=hidden,
        intermediate=intermediate,
        group_size=group_size,
    )
    local_reference = _local_reference(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        post_norm_weight=post_norm_weight,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        group_size=group_size,
        expert_offset=rank * local_experts,
    )
    if rank == 0:
        print(
            f"NVFP4_HELION_FUSED_RS_REFERENCE_READY tokens={tokens} "
            f"hidden={hidden} intermediate={intermediate}",
            flush=True,
        )

    device = torch.device("tpu")
    w1_kernel, w2_kernel, w1_scale_kernel, w2_scale_kernel = (
        tile_weights_for_helion_fused_rs(
            _transpose_packed_fp4(w1.to(device)),
            _transpose_packed_fp4(w2.to(device)),
            _to_gmm_v2_scale_layout(w1_scale).to(device),
            _to_gmm_v2_scale_layout(w2_scale).to(device),
        )
    )
    w1_kernel = _materialize_packed_fp4(w1_kernel)
    w2_kernel = _materialize_packed_fp4(w2_kernel)
    from torch_tpu._internal.distributed import tpu_distributed

    physical_rank = tpu_distributed.global_device_id()
    physical_rank_device = torch.tensor(
        [physical_rank], dtype=torch.int32, device=device
    )
    physical_ids_by_tp_rank = [
        torch.empty_like(physical_rank_device) for _ in range(world_size)
    ]
    dist.all_gather(physical_ids_by_tp_rank, physical_rank_device)
    physical_ids_by_tp_rank = [
        int(value.cpu().item()) for value in physical_ids_by_tp_rank
    ]
    if sorted(physical_ids_by_tp_rank) != list(range(world_size)):
        raise RuntimeError(
            f"Unexpected TPU physical device IDs: {physical_ids_by_tp_rank}"
        )
    tp_rank_by_physical_id = tuple(
        physical_ids_by_tp_rank.index(physical_id) for physical_id in range(world_size)
    )
    expert_offsets = torch.tensor(
        [ep_rank * local_experts for ep_rank in tp_rank_by_physical_id],
        dtype=torch.int32,
        device=device,
    )
    tp_rank_to_physical = torch.tensor(
        physical_ids_by_tp_rank, dtype=torch.int32, device=device
    )

    def kernel(
        x,
        route_weights,
        route_ids,
        kernel_w1,
        kernel_w2,
        kernel_w1_scale,
        kernel_w2_scale,
        kernel_post_norm_weight,
        kernel_expert_offsets,
        kernel_tp_rank_to_physical,
        kernel_physical_rank,
    ):
        return helion_fused_moe_rs_call(
            hidden_states=x,
            w1=kernel_w1,
            w2=kernel_w2,
            w1_scale=kernel_w1_scale,
            w2_scale=kernel_w2_scale,
            topk_weights=route_weights,
            topk_ids=route_ids,
            post_expert_norm_weight=kernel_post_norm_weight,
            expert_offsets=kernel_expert_offsets,
            tp_rank_to_physical=kernel_tp_rank_to_physical,
            physical_rank=kernel_physical_rank,
            world_size=world_size,
            process_group_name=dist.group.WORLD.group_name,
        )

    w1_kernel_device = w1_kernel.to(device)
    w2_kernel_device = w2_kernel.to(device)
    w1_scale_kernel_device = w1_scale_kernel.to(device)
    w2_scale_kernel_device = w2_scale_kernel.to(device)
    post_norm_weight_device = post_norm_weight.to(device)
    compiled = (
        kernel
        if os.getenv("HELION_FUSED_MOE_EAGER_DEBUG") == "1"
        else torch.compile(kernel, backend="tpu", fullgraph=True, dynamic=False)
    )
    call_args = (
        hidden_states.to(device),
        topk_weights.to(device),
        topk_ids.to(device),
        w1_kernel_device,
        w2_kernel_device,
        w1_scale_kernel_device,
        w2_scale_kernel_device,
        post_norm_weight_device,
        expert_offsets,
        tp_rank_to_physical,
        physical_rank_device,
    )
    if rank == 0:
        print("NVFP4_HELION_FUSED_RS_FIRST_CALL", flush=True)
    actual = compiled(*call_args)
    actual_cpu = actual.cpu()
    if rank == 0:
        print("NVFP4_HELION_FUSED_RS_FIRST_CALL_DONE", flush=True)
    repeated = compiled(*call_args)
    repeated_cpu = repeated.cpu()
    if rank == 0:
        print("NVFP4_HELION_FUSED_RS_REPEAT_CALL_DONE", flush=True)
    if not torch.isfinite(actual_cpu).all().item():
        raise AssertionError(f"Helion produced non-finite values for {tokens} tokens")
    if not torch.isfinite(repeated_cpu).all().item():
        raise AssertionError(
            f"Repeated Helion call produced non-finite values for {tokens} tokens"
        )
    torch.testing.assert_close(actual_cpu, repeated_cpu, rtol=0, atol=0)

    route_block_size = _route_block_size(
        tokens,
        topk,
        world_size,
        local_experts,
    )
    padded_tokens = _padded_token_count(
        tokens,
        topk,
        world_size,
        route_block_size,
    )
    expected = torch.zeros(padded_tokens, hidden, dtype=torch.float32, device=device)
    expected[:tokens] = local_reference.to(device)
    dist.all_reduce(expected)
    shard_tokens = padded_tokens // world_size
    expected_shard = expected[rank * shard_tokens : (rank + 1) * shard_tokens].cpu()

    if compare_pallas:
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            _materialize_packed_fp4,
            _to_gmm_v2_scale_layout,
            _transpose_packed_fp4,
        )
        from sglang.srt.layers.moe.pallas import fused_moe_rs_op

        pallas_w1 = _materialize_packed_fp4(_transpose_packed_fp4(w1.to(device)))
        pallas_w2 = _materialize_packed_fp4(_transpose_packed_fp4(w2.to(device)))
        pallas_w1_scale = _to_gmm_v2_scale_layout(w1_scale).to(device)
        pallas_w2_scale = _to_gmm_v2_scale_layout(w2_scale).to(device)
        group_offset = torch.tensor(
            [rank * local_experts, rank],
            dtype=torch.int32,
            device=device,
        )
        fused_moe_rs_op._build_fused_moe_rs_op()

        def pallas_kernel(
            x,
            route_weights,
            route_ids,
            kernel_w1,
            kernel_w2,
            kernel_w1_scale,
            kernel_w2_scale,
            kernel_post_norm_weight,
            kernel_group_offset,
            kernel_tp_rank_to_physical,
        ):
            return fused_moe_rs_op.fused_moe_rs_call(
                hidden_states=x,
                w1=kernel_w1,
                w2=kernel_w2,
                w1_scale=kernel_w1_scale,
                w2_scale=kernel_w2_scale,
                topk_weights=route_weights,
                topk_ids=route_ids,
                post_expert_norm_weight=kernel_post_norm_weight,
                group_offset=kernel_group_offset,
                topk=topk,
                activation="silu",
                has_post_expert_norm=True,
                fp8_direct_write=False,
                tp_rank_to_device_id=kernel_tp_rank_to_physical,
            )

        compiled_pallas = torch.compile(
            pallas_kernel, backend="tpu", fullgraph=True, dynamic=False
        )
        pallas = compiled_pallas(
            hidden_states.to(device),
            topk_weights.to(device),
            topk_ids.to(device),
            pallas_w1,
            pallas_w2,
            pallas_w1_scale,
            pallas_w2_scale,
            post_norm_weight_device,
            group_offset,
            tp_rank_to_physical,
        )
        helion_shards = [torch.empty_like(actual) for _ in range(world_size)]
        pallas_shards = [torch.empty_like(pallas) for _ in range(world_size)]
        dist.all_gather(helion_shards, actual)
        dist.all_gather(pallas_shards, pallas)
        if rank == 0:
            helion_full = torch.cat(helion_shards, dim=0).cpu().float()
            # Both kernels map logical TP destinations to physical device IDs
            # before sending, so rank order is logical token-shard order.
            pallas_full = torch.cat(pallas_shards, dim=0).cpu().float()
            expected_full = expected.cpu()
            pallas_difference = (helion_full - pallas_full).abs()
            pallas_relative_mean = pallas_difference.mean() / pallas_full.abs().mean()
            pallas_cosine = torch.nn.functional.cosine_similarity(
                helion_full.flatten(), pallas_full.flatten(), dim=0
            )
            print(
                "NVFP4_HELION_VS_PALLAS "
                f"relative_mean={pallas_relative_mean.item():.6f} "
                f"cosine={pallas_cosine.item():.6f} "
                f"max_abs={pallas_difference.max().item():.6f}",
                flush=True,
            )
            for name, candidate in (
                ("HELION", helion_full),
                ("PALLAS", pallas_full),
            ):
                reference_difference = (candidate - expected_full).abs()
                reference_relative_mean = (
                    reference_difference.mean() / expected_full.abs().mean()
                )
                reference_cosine = torch.nn.functional.cosine_similarity(
                    candidate.flatten(), expected_full.flatten(), dim=0
                )
                print(
                    f"NVFP4_{name}_VS_REFERENCE "
                    f"relative_mean={reference_relative_mean.item():.6f} "
                    f"cosine={reference_cosine.item():.6f} "
                    f"max_abs={reference_difference.max().item():.6f}",
                    flush=True,
                )

    actual_float = actual_cpu.float()
    difference = (actual_float - expected_shard).abs()
    if not torch.count_nonzero(expected_shard):
        torch.testing.assert_close(actual_float, expected_shard, rtol=0, atol=0)
        print(
            "NVFP4_HELION_FUSED_RS_PASS "
            f"rank={rank} tokens={tokens} skewed={skewed_routing} "
            "empty_output_shard=True",
            flush=True,
        )
        return
    relative_mean = difference.mean() / expected_shard.abs().mean().clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_shard.flatten(), dim=0
    )
    if relative_mean.item() >= 0.08 or cosine.item() <= 0.995:
        raise AssertionError(
            f"tokens={tokens}: relative_mean={relative_mean.item():.6f}, "
            f"cosine={cosine.item():.6f}, max_abs={difference.max().item():.6f}"
        )
    print(
        "NVFP4_HELION_FUSED_RS_PASS "
        f"rank={rank} tokens={tokens} skewed={skewed_routing} "
        f"relative_mean={relative_mean.item():.6f} "
        f"cosine={cosine.item():.6f} max_abs={difference.max().item():.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=(1, 16, 64, 256))
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--skewed-routing", action="store_true")
    parser.add_argument("--compare-pallas", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.world_size:
        raise RuntimeError(f"Expected {args.world_size} TPU ranks, got {world_size}")

    for tokens in args.tokens:
        _run_case(
            tokens=tokens,
            rank=rank,
            world_size=world_size,
            local_experts=args.local_experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
            topk=args.topk,
            group_size=256,
            skewed_routing=args.skewed_routing,
            compare_pallas=args.compare_pallas,
        )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
