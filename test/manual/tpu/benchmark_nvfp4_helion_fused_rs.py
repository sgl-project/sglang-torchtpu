# SPDX-License-Identifier: Apache-2.0
"""Compare the direct Helion and production Pallas NVFP4 fused-RS paths.

Run this file with one Torch process per TPU device. It measures the complete
token-sharded fused-RS operator, including routing preparation, expert math,
post-expert RMSNorm, top-k reduction, and inter-chip communication.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("HELION_TPU_COMPILE_CAPTURE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('WORLD_SIZE', '1')}",
)

import torch
import torch.distributed as dist
from test_nvfp4_helion_fused_rs_execute import _make_inputs, _make_local_weights

_MAX_RELATIVE_MEAN = 0.08
_MIN_COSINE = 0.995


def _physical_rank_metadata(
    world_size: int, local_experts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torch_tpu._internal.distributed import tpu_distributed

    physical_rank = tpu_distributed.global_device_id()
    physical_rank_tensor = torch.tensor(
        [physical_rank], dtype=torch.int32, device=device
    )
    gathered_physical_ids = [
        torch.empty_like(physical_rank_tensor) for _ in range(world_size)
    ]
    dist.all_gather(gathered_physical_ids, physical_rank_tensor)
    physical_ids_by_tp_rank = [
        int(value.cpu().item()) for value in gathered_physical_ids
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
    return physical_rank_tensor, expert_offsets, tp_rank_to_physical


def _median_ms(
    fn,
    args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.tpu.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args)
        torch.tpu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _paired_median_ms(
    first_fn,
    first_args: tuple[torch.Tensor, ...],
    second_fn,
    second_args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Time two distributed paths in alternating order to limit drift bias."""
    for iteration in range(warmup):
        calls = (
            ((first_fn, first_args), (second_fn, second_args))
            if iteration % 2 == 0
            else ((second_fn, second_args), (first_fn, first_args))
        )
        for fn, fn_args in calls:
            fn(*fn_args)
            torch.tpu.synchronize()
    dist.barrier()

    first_samples = []
    second_samples = []
    for iteration in range(iterations):
        calls = (
            (
                (first_fn, first_args, first_samples),
                (second_fn, second_args, second_samples),
            )
            if iteration % 2 == 0
            else (
                (second_fn, second_args, second_samples),
                (first_fn, first_args, first_samples),
            )
        )
        for fn, fn_args, samples in calls:
            start = time.perf_counter()
            fn(*fn_args)
            torch.tpu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(first_samples), statistics.median(second_samples)


def _run_shape(args: argparse.Namespace, rank: int, world_size: int) -> None:
    from sglang.srt.hardware_backend.tpu.fp4_moe import (
        _materialize_packed_fp4,
        _to_gmm_v2_scale_layout,
        _transpose_packed_fp4,
    )
    from sglang.srt.layers.moe.helion.fused_moe_rs import (
        _prepare_sorted_routes,
        helion_fused_moe_rs_call,
        tile_weights_for_helion_fused_rs,
    )
    from sglang.srt.layers.moe.pallas import fused_moe_rs_op

    device = torch.device("tpu")
    global_experts = world_size * args.local_experts
    hidden_states, topk_ids, topk_weights, post_norm_weight = _make_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        global_experts=global_experts,
        skewed_routing=args.skewed_routing,
    )
    w1, w2, w1_scale, w2_scale = _make_local_weights(
        rank=rank,
        local_experts=args.local_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        group_size=args.group_size,
    )
    if rank == 0:
        print(f"tokens={args.tokens} stage=host_inputs_ready", flush=True)

    pallas_w1 = _materialize_packed_fp4(_transpose_packed_fp4(w1.to(device)))
    pallas_w2 = _materialize_packed_fp4(_transpose_packed_fp4(w2.to(device)))
    pallas_w1_scale = _to_gmm_v2_scale_layout(w1_scale).to(device)
    pallas_w2_scale = _to_gmm_v2_scale_layout(w2_scale).to(device)
    helion_w1, helion_w2, helion_w1_scale, helion_w2_scale = (
        tile_weights_for_helion_fused_rs(
            _transpose_packed_fp4(w1.to(device)),
            _transpose_packed_fp4(w2.to(device)),
            pallas_w1_scale,
            pallas_w2_scale,
        )
    )
    helion_w1 = _materialize_packed_fp4(helion_w1)
    helion_w2 = _materialize_packed_fp4(helion_w2)

    hidden_states = hidden_states.to(device)
    topk_ids = topk_ids.to(device)
    topk_weights = topk_weights.to(device)
    post_norm_weight = post_norm_weight.to(device)
    physical_rank, expert_offsets, tp_rank_to_physical = _physical_rank_metadata(
        world_size, args.local_experts, device
    )
    group_offset = torch.tensor(
        [rank * args.local_experts, rank],
        dtype=torch.int32,
        device=device,
    )
    fused_moe_rs_op._build_fused_moe_rs_op()
    if rank == 0:
        print(f"tokens={args.tokens} stage=device_inputs_ready", flush=True)

    def helion_step(
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

    def pallas_step(
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
            topk=args.topk,
            activation="silu",
            has_post_expert_norm=True,
            fp8_direct_write=False,
            tp_rank_to_device_id=kernel_tp_rank_to_physical,
        )

    helion_compiled = torch.compile(
        helion_step, backend="tpu", fullgraph=True, dynamic=False
    )
    pallas_compiled = torch.compile(
        pallas_step, backend="tpu", fullgraph=True, dynamic=False
    )
    helion_args = (
        hidden_states,
        topk_weights,
        topk_ids,
        helion_w1,
        helion_w2,
        helion_w1_scale,
        helion_w2_scale,
        post_norm_weight,
        expert_offsets,
        tp_rank_to_physical,
        physical_rank,
    )
    pallas_args = (
        hidden_states,
        topk_weights,
        topk_ids,
        pallas_w1,
        pallas_w2,
        pallas_w1_scale,
        pallas_w2_scale,
        post_norm_weight,
        group_offset,
        tp_rank_to_physical,
    )
    if rank == 0:
        print(f"tokens={args.tokens} stage=compile_helion", flush=True)
    helion_output = helion_compiled(*helion_args)
    helion_repeated = helion_compiled(*helion_args)
    if rank == 0:
        print(f"tokens={args.tokens} stage=compile_pallas", flush=True)
    pallas_output = pallas_compiled(*pallas_args)
    torch.tpu.synchronize()
    if not torch.isfinite(helion_output).all().item():
        raise AssertionError(
            f"Helion produced non-finite values for {args.tokens} tokens"
        )
    if not torch.isfinite(helion_repeated).all().item():
        raise AssertionError(
            f"Repeated Helion call produced non-finite values for {args.tokens} tokens"
        )
    torch.testing.assert_close(
        helion_output.cpu(),
        helion_repeated.cpu(),
        rtol=0,
        atol=0,
    )

    argsort_ms = None
    route_ms = None
    if args.diagnostics:

        def argsort_only(route_ids):
            return torch.argsort(route_ids.reshape(-1), stable=True).to(torch.int32)

        argsort_compiled = torch.compile(
            argsort_only, backend="tpu", fullgraph=True, dynamic=False
        )
        argsort_args = (topk_ids,)
        argsort_compiled(*argsort_args)
        torch.tpu.synchronize()
        argsort_ms = _median_ms(
            argsort_compiled,
            argsort_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        def route_only(route_ids, route_weights):
            return _prepare_sorted_routes(
                route_ids,
                route_weights,
                global_experts=world_size * args.local_experts,
            )

        route_compiled = torch.compile(
            route_only, backend="tpu", fullgraph=True, dynamic=False
        )
        route_args = (topk_ids, topk_weights)
        route_compiled(*route_args)
        torch.tpu.synchronize()
        route_ms = _median_ms(
            route_compiled,
            route_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    helion_shards = [torch.empty_like(helion_output) for _ in range(world_size)]
    pallas_shards = [torch.empty_like(pallas_output) for _ in range(world_size)]
    dist.all_gather(helion_shards, helion_output)
    dist.all_gather(pallas_shards, pallas_output)
    helion_full = torch.cat(helion_shards, dim=0).cpu().float()
    # Both kernels map logical TP destinations to physical device IDs before
    # sending, so distributed rank order is already logical token-shard order.
    pallas_full = torch.cat(pallas_shards, dim=0).cpu().float()
    difference = (helion_full - pallas_full).abs()
    relative_mean = difference.mean() / pallas_full.abs().mean().clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        helion_full.flatten(), pallas_full.flatten(), dim=0
    )
    if relative_mean.item() >= _MAX_RELATIVE_MEAN or cosine.item() <= _MIN_COSINE:
        raise AssertionError(
            f"tokens={args.tokens}: Helion/Pallas relative_mean="
            f"{relative_mean.item():.6f}, cosine={cosine.item():.6f}"
        )
    if rank == 0:
        print(
            "correctness "
            f"relative_mean={relative_mean.item():.6f} "
            f"cosine={cosine.item():.6f} "
            f"max_abs={difference.max().item():.6f} "
            "deterministic=true",
            flush=True,
        )

    if args.paired_timing:
        helion_ms, pallas_ms = _paired_median_ms(
            helion_compiled,
            helion_args,
            pallas_compiled,
            pallas_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    else:
        helion_ms = _median_ms(
            helion_compiled,
            helion_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        pallas_ms = _median_ms(
            pallas_compiled,
            pallas_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    if rank == 0:
        diagnostic_text = (
            f" argsort_ms={argsort_ms:.4f} route_ms={route_ms:.4f}"
            if route_ms is not None and argsort_ms is not None
            else ""
        )
        print(
            f"tokens={args.tokens} helion_ms={helion_ms:.4f} "
            f"pallas_ms={pallas_ms:.4f} "
            f"speedup={pallas_ms / helion_ms:.4f}x{diagnostic_text}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skewed-routing", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--paired-timing", action="store_true")
    args = parser.parse_args()

    if int(os.environ.get("RANK", "0")) == 0:
        print("stage=initialize_process_group", flush=True)
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print("stage=process_group_ready", flush=True)
    if world_size != args.world_size:
        raise RuntimeError(f"Expected {args.world_size} TPU ranks, got {world_size}")
    token_counts = args.tokens
    for tokens in token_counts:
        shape_args = argparse.Namespace(**vars(args))
        shape_args.tokens = tokens
        _run_shape(shape_args, rank, world_size)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
