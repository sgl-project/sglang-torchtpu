#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Canonical Distributed Verification of torch.compile(..., backend='tpu') for MoE Fused Reduce-Scatter
using dedicated per-bucket torch.compile graphs with multiprocessing.Barrier synchronization."""

import gc
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_tpu._internal.distributed.launchers import singlehost_wrapper


def worker_fn(rank: int, world_size: int, master_port: int, barrier: mp.Barrier) -> None:
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.suppress_errors = False

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["JAX_PLATFORMS"] = "cpu"
    existing_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
    flags_to_add = []
    if "--xla_tpu_enable_diagnostic_coordinator" not in existing_libtpu:
        flags_to_add.append("--xla_tpu_enable_diagnostic_coordinator=false")
    if "--xla_tpu_debug_sflag_wait_timeout_ms" not in existing_libtpu:
        flags_to_add.append("--xla_tpu_debug_sflag_wait_timeout_ms=10000")
    if "--xla_tpu_debug_sflag_wait_shalt_on_detection" not in existing_libtpu:
        flags_to_add.append("--xla_tpu_debug_sflag_wait_shalt_on_detection=true")
    if flags_to_add:
        os.environ["LIBTPU_INIT_ARGS"] = f"{existing_libtpu} {' '.join(flags_to_add)}".strip()

    # Initialize distributed process group
    try:
        dist.init_process_group(
            backend="tpu_dist",
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        print(f"[Rank {rank}] dist.init_process_group failed: {e}", flush=True)
        sys.exit(1)

    from sglang.srt.hardware_backend.tpu import pallas_export
    from sglang.srt.layers.moe.pallas import fused_moe_rs_op

    try:
        pallas_export.configure_jax_export_env(world_size)
    except Exception:
        pass

    fused_moe_rs_op._build_fused_moe_rs_op()

    # Parameters matching Qwen3-Coder-480B
    H = 6144
    I = 2560
    E_local = 20
    num_global_experts = E_local * world_size
    topk = 8
    device = "tpu"
    dtype = torch.bfloat16

    if rank == 0:
        print(f"Cluster / Model Config: H={H}, I={I}, E_local={E_local}, topk={topk}, world_size={world_size}, device={device}", flush=True)

    # Weight tensors
    w1_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else dtype
    w2_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else dtype
    w1 = torch.zeros(E_local, H, 2 * I, dtype=w1_dtype, device=device)
    w2 = torch.zeros(E_local, I, H, dtype=w2_dtype, device=device)
    w1_scale = torch.ones(E_local, 1, 1, 2 * I, dtype=torch.float32, device=device)
    w2_scale = torch.ones(E_local, 1, 1, H, dtype=torch.float32, device=device)
    post_expert_norm = torch.ones(1, H, dtype=dtype, device=device)
    global_device_id, physical_order, physical_device_ids = fused_moe_rs_op._get_physical_rank_metadata()
    group_offset = torch.tensor(
        [[rank * E_local, global_device_id, *physical_device_ids]],
        dtype=torch.int32,
        device=device,
    )

    def make_forward_fn(bucket_size):
        def forward_fn_bucket(hs, tw, ti):
            return fused_moe_rs_op.fused_moe_rs_call(
                hidden_states=hs,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                topk_weights=tw,
                topk_ids=ti,
                post_expert_norm_weight=post_expert_norm,
                group_offset=group_offset,
                topk=topk,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
            )
        forward_fn_bucket.__name__ = f"forward_fn_bucket_{bucket_size}"
        return forward_fn_bucket

    t0 = time.time()
    token_buckets = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, 5120]
    if rank == 0:
        print(f"Pre-compiling dedicated graphs across buckets: {token_buckets}", flush=True)

    compiled_buckets = {}
    for bucket in token_buckets:
        if rank == 0:
            print(f"Compiling dedicated graph for bucket T={bucket}...", flush=True)
        compiled_buckets[bucket] = torch.compile(make_forward_fn(bucket), backend="tpu", dynamic=False)

    for T in token_buckets:
        torch.manual_seed(42 + T)
        # Multiprocessing barrier before each bucket iteration
        barrier.wait()

        if hasattr(torch, "tpu"):
            torch.tpu.synchronize()
            if hasattr(torch.tpu, "empty_cache"):
                torch.tpu.empty_cache()
        gc.collect()

        if rank == 0:
            print(f"--> Executing dedicated graph for bucket T={T}...", flush=True)
        hs = torch.zeros(T, H, dtype=dtype, device=device)
        # CRITICAL: Identical ti across all ranks ensures symmetric ICI DMA signals and semaphore waits
        ti = (torch.arange(T * topk, device=device, dtype=torch.int32) % num_global_experts).reshape(T, topk)
        tw = torch.ones(T, topk, dtype=torch.float32, device=device) / topk

        try:
            out = compiled_buckets[T](hs, tw, ti)
            if hasattr(torch, "tpu"):
                torch.tpu.synchronize()
                if hasattr(torch.tpu, "empty_cache"):
                    torch.tpu.empty_cache()
            gc.collect()
            expected_shape = ((T + world_size - 1) // world_size, H)
            assert out.shape == expected_shape, (
                f"[Rank {rank}] Shape mismatch: got {out.shape}, expected {expected_shape}"
            )
            if rank == 0:
                print(f"    Bucket T={T} PASSED: out shape = {out.shape}", flush=True)
        except Exception as e:
            print(f"[Rank {rank}] [FAIL] Bucket T={T} execution failed with error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Multiprocessing barrier after each bucket iteration
        barrier.wait()

    if rank == 0:
        print(f"=== ALL TOKEN BUCKETS PASSED IN {time.time() - t0:.2f}s WITH 0 ERRORS! ===", flush=True)
        print("SUCCESS", flush=True)


def main():
    print("=== STARTING DISTRIBUTED TORCH.COMPILE TPU MOE TEST (WITH MP.BARRIER) ===", flush=True)
    world_size = 8
    singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
    
    import portpicker
    master_port = portpicker.pick_unused_port()

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(world_size)

    mp.spawn(
        worker_fn,
        args=(world_size, master_port, barrier),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
