# SPDX-License-Identifier: Apache-2.0
"""PyTorch custom op wrapper for the native Pallas fused MoE + reduce-scatter kernel."""

import functools
import math
import os

import torch

_FUSED_MOE_RS_OP_NAME = "sglang_tpu_pallas::fused_moe_rs"
_FUSED_MOE_RS_OP = None
_FUSED_RS_ROUTED_ROW_ALIGNMENT = 16


def fused_moe_rs_padded_tokens(num_tokens: int, topk: int, ep_size: int) -> int:
    """Pad tokens for both EP sharding and the fused kernel's routed rows."""
    routed_row_token_alignment = _FUSED_RS_ROUTED_ROW_ALIGNMENT // math.gcd(
        _FUSED_RS_ROUTED_ROW_ALIGNMENT, topk
    )
    token_alignment = math.lcm(ep_size, routed_row_token_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare logical expert ownership before Dynamo captures the MoE layer."""
    from sglang.srt.runtime_context import get_parallel

    moe_ep_rank = get_parallel().moe_ep_rank
    group_offset = torch.tensor(
        [
            moe_ep_rank * layer.num_local_experts,
            moe_ep_rank,
        ],
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    layer.register_buffer("_tpu_fused_rs_group_offset", group_offset, persistent=False)

    _build_fused_moe_rs_op()


@functools.cache
def _get_mesh():
    from torch_tpu._internal.pallas import get_pallas_mesh

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return get_pallas_mesh(("data", "model"), (1, world_size))


def _build_fused_moe_rs_op():
    global _FUSED_MOE_RS_OP
    if _FUSED_MOE_RS_OP is not None:
        return _FUSED_MOE_RS_OP

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from sglang.srt.hardware_backend.tpu.patches import get_tpu_jax_op

    jax_op = get_tpu_jax_op()

    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import fused_moe_func_rs

    mesh = _get_mesh()

    def _wrapped(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        post_expert_norm_weight: jax.Array,
        group_offset: jax.Array,
        topk: int,
        activation: str,
        has_post_expert_norm: bool,
        fp8_direct_write: bool,
    ) -> jax.Array:
        num_tokens = hidden_states.shape[0]
        ep_size = mesh.shape["model"]
        padded_tokens = fused_moe_rs_padded_tokens(num_tokens, topk, ep_size)
        if padded_tokens != num_tokens:
            token_padding = padded_tokens - num_tokens
            hidden_states = jnp.pad(hidden_states, ((0, token_padding), (0, 0)))
            topk_weights = jnp.pad(topk_weights, ((0, token_padding), (0, 0)))
            topk_ids = jnp.pad(topk_ids, ((0, token_padding), (0, 0)))

        # Run the experts and fuse their model-axis reduce-scatter in one kernel.
        # MoE layers are data-dependent and cannot overlap. Reuse the default
        # collective ID so the exported kernel can be compiled once per shape.
        result = fused_moe_func_rs(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=None,
            w2_bias=None,
            gating_output=None,
            topk=topk,
            renormalize=False,
            mesh=mesh,
            activation=activation,
            scoring_fn="softmax",
            topk_weights=topk_weights,
            topk_indices=topk_ids,
            group_offset_input=group_offset,
            post_expert_norm_weight=(
                post_expert_norm_weight if has_post_expert_norm else None
            ),
            fp8_direct_write=fp8_direct_write,
        )
        if not has_post_expert_norm:
            result = result + post_expert_norm_weight[0].astype(result.dtype) * 0
        return result

    # Forward the mesh's device assignment while preserving the Torch-local
    # shapes: the kernel already consumes local expert weights and returns a
    # local token shard.
    custom_op = jax_op(
        _FUSED_MOE_RS_OP_NAME,
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P(),) * 9 + (None,) * 4,
        output_partition_specs=(P(),),
    )

    def _fake_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm_weight: torch.Tensor,
        group_offset: torch.Tensor,
        topk: int,
        activation: str,
        has_post_expert_norm: bool,
        fp8_direct_write: bool,
    ) -> torch.Tensor:
        ep_size = mesh.shape["model"]
        padded_tokens = fused_moe_rs_padded_tokens(
            hidden_states.shape[0], topk, ep_size
        )
        shard_size = padded_tokens // ep_size
        return hidden_states.new_empty((shard_size, hidden_states.shape[1]))

    custom_op.register_fake(_fake_impl)
    _FUSED_MOE_RS_OP = custom_op
    return _FUSED_MOE_RS_OP


def fused_moe_rs_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool,
) -> torch.Tensor:
    return _build_fused_moe_rs_op()(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        post_expert_norm_weight,
        group_offset,
        topk,
        activation,
        has_post_expert_norm,
        fp8_direct_write,
    )


def fused_moe_rs_gather(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    group_offset: torch.Tensor,
    topk: int,
    activation: str,
    has_post_expert_norm: bool,
    fp8_direct_write: bool,
    world_size: int,
    keep_token_shard: bool = False,
) -> torch.Tensor:
    import torch.distributed as dist

    if keep_token_shard and hidden_states.shape[0] % world_size:
        raise RuntimeError(
            "TPU fused-RS sharded output requires input rows divisible by "
            f"TP={world_size}, got {hidden_states.shape[0]}"
        )

    input_tokens = hidden_states.shape[0]
    padded_tokens = fused_moe_rs_padded_tokens(input_tokens, topk, world_size)
    logical_shard_rows = input_tokens // world_size if keep_token_shard else None
    padded_shard_rows = padded_tokens // world_size

    if keep_token_shard and padded_tokens != input_tokens:
        # SP inputs are logical-rank-major. Pad each rank's block independently
        # so padding cannot move a real row into another rank's output shard.
        def _pad_rank_blocks(value: torch.Tensor) -> torch.Tensor:
            rank_major = value.reshape(world_size, logical_shard_rows, *value.shape[1:])
            padding = value.new_zeros(
                (
                    world_size,
                    padded_shard_rows - logical_shard_rows,
                    *value.shape[1:],
                )
            )
            return torch.cat((rank_major, padding), dim=1).reshape(
                padded_tokens, *value.shape[1:]
            )

        hidden_states = _pad_rank_blocks(hidden_states)
        topk_weights = _pad_rank_blocks(topk_weights)
        topk_ids = _pad_rank_blocks(topk_ids)

    token_shard = fused_moe_rs_call(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=post_expert_norm_weight,
        group_offset=group_offset,
        topk=topk,
        activation=activation,
        has_post_expert_norm=has_post_expert_norm,
        fp8_direct_write=fp8_direct_write,
    )
    if token_shard.ndim != 2 or token_shard.shape != (
        padded_shard_rows,
        hidden_states.shape[-1],
    ):
        raise RuntimeError(
            "TPU fused-RS kernel returned an unexpected shard shape: got "
            f"{tuple(token_shard.shape)}, expected "
            f"({padded_shard_rows}, {hidden_states.shape[-1]})"
        )
    if keep_token_shard:
        return token_shard[:logical_shard_rows].contiguous()

    try:
        from sglang.srt.runtime_context import get_parallel

        tp_group = get_parallel().tp_group
        if tp_group is not None:
            return tp_group.all_gather(token_shard, dim=0)
    except (AssertionError, RuntimeError):
        pass

    if dist.is_initialized():
        world_size = dist.get_world_size()
        output_shards = [torch.empty_like(token_shard) for _ in range(world_size)]
        dist.all_gather(output_shards, token_shard)
        return torch.cat(output_shards, dim=0)
    return token_shard
