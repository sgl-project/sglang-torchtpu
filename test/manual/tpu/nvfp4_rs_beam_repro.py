# SPDX-License-Identifier: Apache-2.0
"""Beam-shaped NVFP4 fused-RS hang repro: TP8/EP8, H=5120, I=2560, g=16.

Mirrors test_nvfp4_fused_rs_execute.py but with the Beam-1.0-501B-A23B-NVFP4
MoE geometry that deadlocks warmup at extend tokens=256 (size_m=2048).

Usage: torchrun --nproc_per_node=8 nvfp4_rs_beam_repro.py [tokens ...]
Prints RS_PASS lines; a missing line means that token count hung.
"""

import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.distributed as dist

from test_nvfp4_fused_rs_execute import (
    _make_inputs,
    _make_local_nvfp4_layer,
)


def main() -> None:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"expected TP8/EP8, got {world_size}")

    from sglang_tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights
    from sglang_tpu.moe.pallas import fused_moe_rs_op
    from test_nvfp4_fused_rs_execute import _make_fused_rs_kernel

    global_experts = 256
    local_experts = global_experts // world_size
    hidden_size = 5120
    intermediate_size = 2560
    topk = 8
    block_size = 16
    device = torch.device("tpu")

    group_offset = torch.tensor(
        [rank * local_experts], dtype=torch.int32, device=device
    )
    no_post_norm = torch.zeros(1, dtype=torch.bfloat16, device=device)

    layer = _make_local_nvfp4_layer(
        local_experts, hidden_size, intermediate_size, rank, block_size
    ).to(device)
    original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
    fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
    try:
        prepare_tpu_nvfp4_moe_weights(layer, group_size=block_size, use_fused_rs=True)
    finally:
        fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
    gc.collect()

    kernel = _make_fused_rs_kernel(
        fused_moe_rs_op.fused_moe_rs_call,
        layer,
        no_post_norm,
        group_offset,
        topk,
    )
    token_counts = [int(t) for t in sys.argv[1:]] or [64, 128, 256, 512]
    for tokens in token_counts:
        hidden_states, topk_ids, topk_weights = _make_inputs(
            tokens, hidden_size, global_experts, topk
        )
        if os.environ.get("REPRO_SKEW"):
            # Server warmup feeds identical rows through the router: every
            # token lands on the same topk experts (ties resolve to 0..7).
            slots = torch.arange(topk, dtype=torch.int32)
            topk_ids = slots[None, :].expand(tokens, topk).contiguous()
        padded_tokens = fused_moe_rs_op.fused_moe_rs_padded_tokens(
            tokens, topk, world_size
        )
        if padded_tokens != tokens:
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, padded_tokens - tokens)
            )
            topk_ids = torch.nn.functional.pad(
                topk_ids, (0, 0, 0, padded_tokens - tokens)
            )
            topk_weights = torch.nn.functional.pad(
                topk_weights, (0, 0, 0, padded_tokens - tokens)
            )
        iters = int(os.environ.get("REPRO_ITERS", "1"))
        h_dev = hidden_states.to(device)
        w_dev = topk_weights.to(device)
        i_dev = topk_ids.to(device)
        out = None
        for it in range(iters):
            out = kernel(h_dev, w_dev, i_dev)
            if iters > 1 and it % 20 == 0:
                print(f"RS_ITER rank={rank} tokens={tokens} it={it}", flush=True)
        out = out.cpu()
        assert torch.isfinite(out).all().item(), f"non-finite at tokens={tokens}"
        dist.barrier()
        print(
            f"RS_PASS rank={rank} tokens={tokens} padded={padded_tokens} "
            f"shape={tuple(out.shape)} checksum={out.float().abs().sum().item():.4f}",
            flush=True,
        )
        gc.collect()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
