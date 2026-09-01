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


def _get_global_device_id() -> int:
    """Return the physical TPU device ID, defaulting to 0 if unavailable."""
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        return int(tpu_distributed.global_device_id())
    except Exception:
        return 0


@functools.cache
def _get_tp_rank_to_device_id_mapping() -> tuple[int, tuple[int, ...]]:
    """Return (global_device_id, tp_rank_to_device_id) for the active TP communication group."""
    import torch.distributed as dist

    global_device_id = _get_global_device_id()
    if not dist.is_initialized():
        return global_device_id, (global_device_id,)

    try:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        group, world_size = tp_group.cpu_group, tp_group.world_size
    except (AssertionError, RuntimeError):
        # Fallback for standalone unit tests running raw torch.distributed without SGLang parallel_state
        group, world_size = None, dist.get_world_size()

    physical_ids = [None] * world_size
    dist.all_gather_object(physical_ids, global_device_id, group=group)
    if sorted(physical_ids) != list(range(world_size)):
        raise RuntimeError(f"Unexpected TPU physical device IDs: {physical_ids}")

    return global_device_id, tuple(int(x) for x in physical_ids)


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare physical-rank metadata before Dynamo captures the MoE layer."""
    from sglang.srt.runtime_context import get_parallel

    moe_ep_rank = get_parallel().moe_ep_rank
    _, tp_rank_to_device_id = _get_tp_rank_to_device_id_mapping()
    group_offset = torch.tensor(
        [
            moe_ep_rank * layer.num_local_experts,
            moe_ep_rank,
        ],
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    layer.register_buffer("_tpu_fused_rs_group_offset", group_offset, persistent=False)

    tp_rank_to_device_id_tensor = torch.tensor(
        tp_rank_to_device_id,
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    layer.register_buffer(
        "_tpu_fused_rs_tp_rank_to_device_id",
        tp_rank_to_device_id_tensor,
        persistent=False,
    )
    _build_fused_moe_rs_op()


@functools.cache
def _get_mesh():
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    export_devices = np.asarray(get_torchtpu_export_devices(world_size)).reshape(1, -1)
    return Mesh(export_devices, ("data", "model"))


def _build_fused_moe_rs_op():
    global _FUSED_MOE_RS_OP
    if _FUSED_MOE_RS_OP is not None:
        return _FUSED_MOE_RS_OP

    import jax
    import jax.numpy as jnp
    from torch_tpu._internal.pallas import jax_op

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
        tp_rank_to_device_id: jax.Array,
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
            tp_rank_to_device_id=tp_rank_to_device_id,
        )
        if not has_post_expert_norm:
            result = result + post_expert_norm_weight[0].astype(result.dtype) * 0
        return result

    # Bridge the JAX kernel into Torch as clean non-sharded custom op.
    custom_op = jax_op(_FUSED_MOE_RS_OP_NAME, _wrapped)

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
        tp_rank_to_device_id: torch.Tensor,
    ) -> torch.Tensor:
        ep_size = int(tp_rank_to_device_id.shape[0])
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
    tp_rank_to_device_id: torch.Tensor,
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
        tp_rank_to_device_id,
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
    tp_rank_to_device_id: torch.Tensor,
    keep_token_shard: bool = False,
) -> torch.Tensor:
    import torch.distributed as dist

    world_size = int(tp_rank_to_device_id.numel())
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
        tp_rank_to_device_id=tp_rank_to_device_id,
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
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
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
