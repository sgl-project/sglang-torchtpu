# SPDX-License-Identifier: Apache-2.0
"""TP16/EP16 execution gate for supported NVFP4 fused-RS block sizes.

For block sizes 128, 256, and 512, run the packed-FP4 fused-RS kernel and
compare its token shard with a distributed sum of the stock BF16 MoE kernel
using weights dequantized from the exact same FP4 codes and block scales.
"""

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

SUPPORTED_BLOCK_SIZES = (128, 256, 512)
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2,
    dtype=torch.float32,
)
E2M1_VALUES[8:] *= -1


def _make_local_nvfp4_layer(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    block_size: int,
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
            (local_experts, 2 * intermediate_size, hidden_size // block_size),
            scale,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (local_experts, hidden_size, intermediate_size // block_size),
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


def _dequantize_rows_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize checkpoint-layout packed E2M1 rows to BF16."""
    codes = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    if codes.shape[-1] != block_scale.shape[-1] * block_size:
        raise ValueError(
            "Packed FP4 width and block-scale count disagree: "
            f"width={codes.shape[-1]}, scales={block_scale.shape[-1]}, "
            f"block_size={block_size}"
        )
    values = E2M1_VALUES[codes.long()]
    scales = block_scale.float().repeat_interleave(block_size, dim=-1)
    return (values * scales).to(torch.bfloat16)


def _make_bf16_reference_weights(
    layer: torch.nn.Module,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gmm_v2-layout BF16 weights dequantized from ``layer``."""
    w1 = _dequantize_rows_nvfp4(
        layer.w13_weight.detach(),
        layer.w13_weight_scale.detach(),
        block_size,
    )
    w2 = _dequantize_rows_nvfp4(
        layer.w2_weight.detach(),
        layer.w2_weight_scale.detach(),
        block_size,
    )
    return w1.permute(0, 2, 1).contiguous(), w2.permute(0, 2, 1).contiguous()


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


def _make_fused_rs_kernel(
    fused_moe_rs_call,
    layer: torch.nn.Module,
    no_post_norm: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
):
    w1 = layer.w13_weight
    w2 = layer.w2_weight
    w1_scale = layer.w13_weight_scale
    w2_scale = layer.w2_weight_scale

    def kernel(hidden_states, topk_weights, topk_ids):
        return fused_moe_rs_call(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            post_expert_norm_weight=no_post_norm,
            group_offset=group_offset,
            topk=topk,
            activation="silu",
            has_post_expert_norm=False,
            fp8_direct_write=False,
        )

    return kernel


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
    from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

    global_experts = 256
    local_experts = global_experts // world_size
    hidden_size = 4096
    intermediate_size = 4096
    topk = 8
    device = torch.device("tpu")

    physical_id = tpu_distributed.global_device_id()
    group_offset = torch.tensor(
        [rank * local_experts, physical_id], dtype=torch.int32, device=device
    )
    no_post_norm = torch.zeros(1, dtype=torch.bfloat16, device=device)

    for block_size in SUPPORTED_BLOCK_SIZES:
        layer = _make_local_nvfp4_layer(
            local_experts,
            hidden_size,
            intermediate_size,
            rank,
            block_size,
        )
        reference_w1, reference_w2 = _make_bf16_reference_weights(layer, block_size)
        reference_w1 = reference_w1.to(device)
        reference_w2 = reference_w2.to(device)
        layer = layer.to(device)

        original_prepare = fused_moe_rs_op.prepare_fused_moe_rs
        fused_moe_rs_op.prepare_fused_moe_rs = lambda _: None
        try:
            prepare_tpu_nvfp4_moe_weights(
                layer,
                group_size=block_size,
                use_fused_rs=True,
            )
        finally:
            fused_moe_rs_op.prepare_fused_moe_rs = original_prepare
        gc.collect()

        compiled = torch.compile(
            _make_fused_rs_kernel(
                fused_moe_rs_op.fused_moe_rs_call,
                layer,
                no_post_norm,
                group_offset,
                topk,
            ),
            fullgraph=True,
            dynamic=False,
        )
        for tokens in (1, 16, 64, 256):
            hidden_states, topk_ids, topk_weights = _make_inputs(
                tokens, hidden_size, global_experts, topk
            )
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

            hidden_device = hidden_states.to(device)
            topk_weights_device = topk_weights.to(device)
            topk_ids_device = topk_ids.to(device)
            reference = fused_moe_call(
                hidden_states=hidden_device,
                w1=reference_w1,
                w2=reference_w2,
                topk_weights=topk_weights_device,
                topk_ids=topk_ids_device,
                topk=topk,
                activation="silu",
                global_num_experts=global_experts,
                expert_group_offset=rank * local_experts,
            )
            dist.all_reduce(reference, op=dist.ReduceOp.SUM)
            shard_tokens = padded_tokens // world_size
            expected = reference[
                physical_id * shard_tokens : (physical_id + 1) * shard_tokens
            ].cpu()

            previous = None
            for iteration in range(2):
                actual = compiled(
                    hidden_device[:tokens],
                    topk_weights_device[:tokens],
                    topk_ids_device[:tokens],
                ).cpu()
                if not torch.isfinite(actual).all().item():
                    raise AssertionError(
                        f"non-finite output for block={block_size}, tokens={tokens}"
                    )
                if actual.shape != expected.shape:
                    raise AssertionError(
                        f"output shape mismatch for block={block_size}, tokens={tokens}: "
                        f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
                    )
                if previous is not None:
                    torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                previous = actual

                actual_float = actual.float()
                expected_float = expected.float()
                denom = expected_float.abs().mean().item()
                if denom == 0:
                    torch.testing.assert_close(
                        actual_float, expected_float, rtol=0, atol=0
                    )
                    rel_mean = 0.0
                    cos = 1.0
                else:
                    rel_mean = (
                        actual_float - expected_float
                    ).abs().mean().item() / denom
                    cos = torch.nn.functional.cosine_similarity(
                        actual_float.flatten(), expected_float.flatten(), dim=0
                    ).item()
                    if rel_mean >= 0.08:
                        raise AssertionError(
                            f"block={block_size}, tokens={tokens}: "
                            f"rel_mean={rel_mean} >= 0.08"
                        )
                    if cos <= 0.995:
                        raise AssertionError(
                            f"block={block_size}, tokens={tokens}: cos={cos} <= 0.995"
                        )

                checksum = actual_float.abs().sum().item()
                dist.barrier()
                print(
                    "NVFP4_FUSED_RS_EXECUTE_PASS "
                    f"rank={rank} physical_id={physical_id} block={block_size} "
                    f"tokens={tokens} iteration={iteration} "
                    f"shape={tuple(actual.shape)} checksum={checksum:.6f} "
                    f"rel_mean={rel_mean:.6f} cos={cos:.6f}",
                    flush=True,
                )

        del compiled, layer, reference_w1, reference_w2
        gc.collect()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
