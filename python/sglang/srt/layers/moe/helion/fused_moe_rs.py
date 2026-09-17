# SPDX-License-Identifier: Apache-2.0
"""Direct Helion TPU implementation of NVFP4 MoE with reduce-scatter."""

from __future__ import annotations

import math
import os

import torch

# A source Helion kernel must be opaque to the outer SGLang torch.compile graph.
# Set this before importing Helion so its Pallas launcher registers the custom op.
os.environ.setdefault("HELION_TPU_COMPILE_CAPTURE", "1")

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from torch_tpu._internal.distributed.spmd_util import spmd_safe  # noqa: E402

_MIN_ROUTE_BLOCK_SIZE = 8
_MAX_ROUTE_BLOCK_SIZE = 32
_NVFP4_GROUP_SIZE = 256
_W1_OUTPUT_TILE_SIZE = 2048
_W2_OUTPUT_TILE_SIZE = 1024
_E4M3_MAX = 448.0


def _route_block_size(
    num_tokens: int,
    topk: int,
    world_size: int,
    local_experts: int,
) -> int:
    """Choose a larger MXU batch when each local expert receives more rows."""
    average_local_routes = math.ceil(num_tokens * topk / (world_size * local_experts))
    block_size = _MIN_ROUTE_BLOCK_SIZE
    while block_size < min(average_local_routes, _MAX_ROUTE_BLOCK_SIZE):
        block_size *= 2
    return block_size


def _padded_token_count(
    num_tokens: int,
    topk: int,
    ep_size: int,
    route_block_size: int,
) -> int:
    routed_row_alignment = route_block_size // math.gcd(route_block_size, topk)
    token_alignment = math.lcm(ep_size, routed_row_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


def _route_block_count(
    num_tokens: int,
    topk: int,
    local_experts: int,
    route_block_size: int,
) -> int:
    assignments = num_tokens * topk
    return local_experts + (assignments + route_block_size - 1) // route_block_size - 1


def tile_weights_for_helion_fused_rs(
    w1_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arrange packed weights as fixed aligned panels for the device loops.

    W1 places one 256-column gate tile beside its matching up tile, so one MXU
    operation produces both SwiGLU inputs. Each W2 contracting panel is split
    into 1,024-column output tiles to keep the down-projection accumulator
    small.
    """
    experts, hidden, packed_two_intermediate = w1_packed.shape
    packed_intermediate = packed_two_intermediate // 2
    packed_tile_columns = _W1_OUTPUT_TILE_SIZE // 2
    if packed_intermediate % packed_tile_columns:
        raise ValueError(
            "Helion fused-RS W1 requires the intermediate dimension to be a "
            f"multiple of {_W1_OUTPUT_TILE_SIZE}"
        )
    w1_output_tile_count = packed_intermediate // packed_tile_columns
    groups_per_w1_tile = _W1_OUTPUT_TILE_SIZE // _NVFP4_GROUP_SIZE
    w1_scale_group_width = (
        hidden // _NVFP4_GROUP_SIZE * 2 * _W1_OUTPUT_TILE_SIZE // groups_per_w1_tile
    )
    w1_packed = (
        w1_packed.reshape(
            experts,
            hidden,
            2,
            w1_output_tile_count,
            packed_tile_columns,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(experts, hidden, w1_output_tile_count, 2 * packed_tile_columns)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(
            experts,
            w1_output_tile_count,
            hidden // _NVFP4_GROUP_SIZE,
            _NVFP4_GROUP_SIZE,
            2 * packed_tile_columns,
        )
    )
    w1_scale = (
        w1_scale.reshape(
            experts,
            hidden // _NVFP4_GROUP_SIZE,
            2,
            w1_output_tile_count,
            _W1_OUTPUT_TILE_SIZE,
        )
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(
            experts,
            w1_output_tile_count,
            groups_per_w1_tile,
            w1_scale_group_width,
        )
    )
    if hidden % _W2_OUTPUT_TILE_SIZE:
        raise ValueError(
            "Helion fused-RS W2 requires the hidden dimension to be a "
            f"multiple of {_W2_OUTPUT_TILE_SIZE}"
        )
    w2_output_tile_count = hidden // _W2_OUTPUT_TILE_SIZE
    w2_packed = (
        w2_packed.reshape(
            experts,
            w1_output_tile_count,
            groups_per_w1_tile,
            _NVFP4_GROUP_SIZE,
            w2_output_tile_count,
            _W2_OUTPUT_TILE_SIZE // 2,
        )
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
    )
    w2_scale = (
        w2_scale.squeeze(2)
        .reshape(
            experts,
            w1_output_tile_count,
            groups_per_w1_tile,
            w2_output_tile_count,
            _W2_OUTPUT_TILE_SIZE,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return w1_packed, w2_packed, w1_scale, w2_scale


def prepare_helion_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Cache process-group metadata before Dynamo captures the MoE layer."""
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group, get_world_group
    from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
        _get_physical_rank_metadata,
    )

    tp_group = get_tp_group()
    world_group = get_world_group()
    if tp_group.world_size != world_group.world_size:
        raise RuntimeError(
            "Helion fused-RS requires the TP group to contain the full world"
        )
    if layer.moe_ep_size != tp_group.world_size:
        raise RuntimeError("Helion fused-RS requires EP, TP, and world to match")
    if layer.moe_ep_rank != tp_group.rank_in_group:
        raise RuntimeError("Helion fused-RS requires matching EP and TP rank order")

    physical_rank, tp_rank_by_physical_id = _get_physical_rank_metadata()
    physical_ids_by_tp_rank = [0] * tp_group.world_size
    for physical_id, tp_rank in enumerate(tp_rank_by_physical_id):
        physical_ids_by_tp_rank[tp_rank] = physical_id
    expert_offsets = (
        torch.tensor(
            tp_rank_by_physical_id,
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        * layer.num_local_experts
    )
    tp_rank_to_physical = torch.tensor(
        physical_ids_by_tp_rank,
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    physical_rank_tensor = torch.tensor(
        [physical_rank],
        dtype=torch.int32,
        device=layer.w13_weight.device,
    )
    layer.register_buffer(
        "_tpu_helion_fused_rs_expert_offsets", expert_offsets, persistent=False
    )
    layer.register_buffer(
        "_tpu_helion_fused_rs_tp_rank_to_physical",
        tp_rank_to_physical,
        persistent=False,
    )
    layer.register_buffer(
        "_tpu_helion_fused_rs_physical_rank",
        physical_rank_tensor,
        persistent=False,
    )
    assert dist.group.WORLD is not None
    layer._tpu_helion_fused_rs_process_group_name = dist.group.WORLD.group_name


def _quantize_e4m3_block(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one contracting-dimension block for the TPU MXU."""
    values_f32 = values.to(torch.float32)
    scale = torch.amax(torch.abs(values_f32), dim=-1, keepdim=True) / _E4M3_MAX
    inverse_scale = torch.where(scale == 0, 0, 1 / scale)
    return (values_f32 * inverse_scale).to(torch.float8_e4m3fn), scale


def _prepare_sorted_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    global_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routed assignments and count rows for each global expert."""
    flat_ids = topk_ids.reshape(-1)
    sorted_assignments = torch.argsort(flat_ids, stable=True).to(torch.int32)
    sorted_route_weights = topk_weights.reshape(-1)[sorted_assignments.to(torch.int64)]
    expert_ids = torch.arange(
        global_experts,
        device=topk_ids.device,
        dtype=torch.int32,
    )
    group_sizes = (flat_ids[:, None] == expert_ids[None, :]).sum(
        dim=0,
        dtype=torch.int32,
    )
    return sorted_assignments, sorted_route_weights, group_sizes


def _build_local_route_schedule(
    group_sizes: torch.Tensor,
    *,
    expert_offsets: torch.Tensor,
    physical_rank: torch.Tensor,
    local_experts: int,
    route_blocks: int,
    route_block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Describe this rank's compact, expert-grouped route blocks."""
    loaded_group_sizes = hl.load(group_sizes, [slice(None)])
    global_expert_ids = torch.arange(
        group_sizes.size(0),
        device=group_sizes.device,
        dtype=torch.int32,
    )
    global_group_starts = hl.cumsum(loaded_group_sizes, dim=-1) - loaded_group_sizes
    rank = hl.load(physical_rank, [0])
    expert_offset = hl.load(expert_offsets, [rank])
    local_expert_ids = torch.arange(
        local_experts,
        device=group_sizes.device,
        dtype=torch.int32,
    )
    owned_global_experts = expert_offset + local_expert_ids
    local_membership = (global_expert_ids[:, None] == owned_global_experts[None, :]).to(
        torch.int32
    )
    local_counts = (loaded_group_sizes[:, None] * local_membership).sum(
        dim=0,
        dtype=torch.int32,
    )
    group_starts = (global_group_starts[:, None] * local_membership).sum(
        dim=0,
        dtype=torch.int32,
    )
    blocks_by_expert = (local_counts + route_block_size - 1) // route_block_size
    block_ends = hl.cumsum(blocks_by_expert, dim=-1)
    block_starts = block_ends - blocks_by_expert
    total_blocks = blocks_by_expert.sum(dim=0, dtype=torch.int32)

    block_ids = torch.arange(
        route_blocks,
        device=group_sizes.device,
        dtype=torch.int32,
    )
    source_experts = (block_ids[:, None] >= block_ends[None, :]).sum(
        dim=-1, dtype=torch.int32
    )
    source_experts = source_experts.clamp_max(local_experts - 1)
    source_expert_mask = (local_expert_ids[:, None] == source_experts[None, :]).to(
        torch.int32
    )
    first_expert_blocks = (block_starts[:, None] * source_expert_mask).sum(
        dim=0, dtype=torch.int32
    )
    route_offsets = (block_ids - first_expert_blocks) * route_block_size
    source_group_sizes = (local_counts[:, None] * source_expert_mask).sum(
        dim=0, dtype=torch.int32
    )
    source_group_starts = (group_starts[:, None] * source_expert_mask).sum(
        dim=0,
        dtype=torch.int32,
    )
    return (
        source_experts,
        source_group_starts,
        source_group_sizes,
        route_offsets,
        total_blocks,
    )


@helion.kernel(
    backend="pallas",
    static_shapes=True,
    distributed=True,
    pallas_topk_recall_target=1.0,
    config=helion.Config(
        pallas_loop_type="fori_loop",
        pallas_pre_broadcast=False,
        pallas_indirect_access_mode="dma",
        # Double-buffer the W1, W1-scale, W2, and W2-scale streams so the next
        # panel can arrive while the MXU consumes the current one.
        pallas_load_buffer_count=[1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
    ),
)
def _fused_nvfp4_moe_reduce_scatter(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_assignments: torch.Tensor,
    sorted_route_weights: torch.Tensor,
    group_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    tp_rank_to_physical: torch.Tensor,
    physical_rank: torch.Tensor,
    post_norm_weight: torch.Tensor,
    world_size: int,
    route_blocks: int,
    route_block_size: int,
    topk: int,
    output_tokens: int,
    hidden: int,
    intermediate: int,
    group_size: int,
    w1_output_tile_size: int,
    process_group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """Project routed rows and push each result to its token-shard owner."""
    world_size = hl.specialize(world_size)
    route_blocks = hl.specialize(route_blocks)
    route_block_size = hl.specialize(route_block_size)
    topk = hl.specialize(topk)
    output_tokens = hl.specialize(output_tokens)
    hidden = hl.specialize(hidden)
    intermediate = hl.specialize(intermediate)
    group_size = hl.specialize(group_size)
    w1_output_tile_size = hl.specialize(w1_output_tile_size)
    expert_output = torch.empty(
        (route_block_size, 1, hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output_tiles = intermediate // w1_output_tile_size
    w2_output_tiles = hidden // _W2_OUTPUT_TILE_SIZE
    hidden_groups = hidden // group_size
    groups_per_output_tile = w1_output_tile_size // group_size
    received = torch.empty(
        (output_tokens * topk, 1, hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for _program in hl.grid(1):
        (
            block_experts,
            source_group_starts,
            source_group_sizes,
            route_offsets,
            active_route_blocks,
        ) = _build_local_route_schedule(
            group_sizes,
            expert_offsets=expert_offsets,
            physical_rank=physical_rank,
            local_experts=w1.size(0),
            route_blocks=route_blocks,
            route_block_size=route_block_size,
        )
        ordered_assignments = hl.load(sorted_assignments, [slice(None)])
        ordered_route_weights = hl.load(
            sorted_route_weights,
            [slice(None)],
        ).to(torch.float32)
        route_block_ids = torch.arange(
            route_blocks,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        sorted_position_ids = torch.arange(
            sorted_assignments.size(0),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        scheduled_route_blocks = active_route_blocks.clamp_min(1)
        block_receive_counts = (route_block_ids == scheduled_route_blocks - 1).to(
            torch.int32
        ) * (output_tokens * topk)
        for route_block in hl.tile(scheduled_route_blocks, block_size=1):
            lanes = hl.arange(route_block_size)
            selected_block = (route_block_ids == route_block.begin).to(torch.int32)
            expert = (block_experts * selected_block).sum(dim=0, dtype=torch.int32)
            group_start = (source_group_starts * selected_block).sum(
                dim=0, dtype=torch.int32
            )
            expert_route_count = (source_group_sizes * selected_block).sum(
                dim=0, dtype=torch.int32
            )
            route_offset = (route_offsets * selected_block).sum(
                dim=0, dtype=torch.int32
            )
            route_positions = route_offset + lanes
            valid = (route_positions < expert_route_count).to(torch.int32)
            sorted_positions = (group_start + route_positions).clamp(
                0, sorted_assignments.size(0) - 1
            )
            position_mask = (
                sorted_position_ids[None, :] == sorted_positions[:, None]
            ).to(torch.int32)
            assignments = (position_mask * ordered_assignments[None, :]).sum(
                dim=-1,
                dtype=torch.int32,
            )
            route_weights = (
                position_mask.to(torch.float32) * ordered_route_weights[None, :]
            ).sum(dim=-1) * valid.to(torch.float32)
            valid_count = valid.sum(dim=0, dtype=torch.int32)
            receive_count = (block_receive_counts * selected_block).sum(
                dim=0, dtype=torch.int32
            )
            token_indices = assignments // topk
            destinations = hl.load(
                tp_rank_to_physical,
                [token_indices // output_tokens],
            )
            expert_output[:, 0, :] = hl.zeros(
                [route_block_size, hidden], dtype=expert_output.dtype
            )
            routed_hidden = hl.load(
                hidden_states,
                [token_indices, slice(None), slice(None)],
            )
            routed_hidden, hidden_scale = _quantize_e4m3_block(routed_hidden)
            routed_hidden = routed_hidden.permute(1, 0, 2)
            hidden_scale = hidden_scale.permute(1, 0, 2)
            accumulator = hl.zeros(
                [w2_output_tiles, route_block_size, _W2_OUTPUT_TILE_SIZE],
                dtype=torch.float32,
            )
            for output_tile in hl.tile(output_tiles, block_size=1):
                w1_panel = w1[expert, output_tile.begin, :, :, :]
                projection_scale = w1_scale[expert, output_tile.begin, :, :].reshape(
                    hidden_groups, 2, w1_output_tile_size
                )
                projection = hl.zeros(
                    [route_block_size, 2, w1_output_tile_size],
                    dtype=torch.float32,
                )
                for hidden_group in hl.static_range(hidden_groups):
                    partial_projection = hl.dot(
                        routed_hidden[hidden_group, :, :],
                        w1_panel[hidden_group, :, :],
                        out_dtype=torch.float32,
                    ).reshape(route_block_size, 2, w1_output_tile_size)
                    projection = projection + (
                        partial_projection
                        * hidden_scale[hidden_group, :, None, :]
                        * projection_scale[hidden_group, None, :, :]
                    )

                gate = projection[:, 0, :]
                up = projection[:, 1, :]
                activation = (
                    (gate * torch.sigmoid(gate) * up)
                    .to(hidden_states.dtype)
                    .reshape(
                        route_block_size,
                        groups_per_output_tile,
                        group_size,
                    )
                    .permute(1, 0, 2)
                )
                activation, activation_scale = _quantize_e4m3_block(activation)
                w2_panel = w2[expert, output_tile.begin, :, :, :, :]
                scale = w2_scale[expert, output_tile.begin, :, :, :]
                for intermediate_group in hl.static_range(groups_per_output_tile):
                    activation_group = activation[
                        intermediate_group, None, :, :
                    ].expand(w2_output_tiles, route_block_size, group_size)
                    partial_output = hl.dot(
                        activation_group,
                        w2_panel[:, intermediate_group, :, :],
                        out_dtype=torch.float32,
                    )
                    accumulator = accumulator + (
                        partial_output
                        * activation_scale[intermediate_group, None, :, :]
                        * scale[:, intermediate_group, None, :]
                    )

            projected = (
                accumulator.to(hidden_states.dtype)
                .permute(1, 0, 2)
                .reshape(route_block_size, hidden)
                .to(torch.float32)
            )
            scaled = projected * (post_norm_weight[None, :].to(torch.float32) + 1)
            inverse_rms = torch.rsqrt(scaled.square().mean(dim=-1) + 1e-8)
            normalized = scaled * inverse_rms[:, None]
            expert_output[:, 0, :] = (
                normalized * route_weights[:, None].to(torch.float32)
            ).to(expert_output.dtype)

            for lane in hl.tile(route_block_size, block_size=1):
                assignment = (
                    assignments * (lanes == lane.begin).to(assignments.dtype)
                ).sum(dim=0, dtype=torch.int32)
                destination = (
                    destinations * (lanes == lane.begin).to(destinations.dtype)
                ).sum(dim=0, dtype=torch.int32)
                copy = hl.make_async_remote_copy(
                    expert_output,
                    [lane.begin],
                    destination,
                    dst=received,
                    dst_index=[assignment % (output_tokens * topk)],
                )
                if lane.begin < valid_count:
                    copy.start()
                if lane.begin == route_block_size - 1:
                    for _ in hl.tile(valid_count, block_size=1):
                        copy.wait_send()
                    for _ in hl.tile(receive_count, block_size=1):
                        copy.wait_recv()

    return (
        received.reshape(output_tokens, topk, hidden)
        .to(torch.float32)
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def helion_fused_moe_rs_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    tp_rank_to_physical: torch.Tensor,
    physical_rank: torch.Tensor,
    world_size: int,
    process_group_name: str,
) -> torch.Tensor:
    """Run the Helion kernel and return this rank's token shard."""
    tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    local_experts = w1.shape[0]
    intermediate = w1.shape[1] * _W1_OUTPUT_TILE_SIZE
    expected_w1_scale_shape = (
        local_experts,
        intermediate // _W1_OUTPUT_TILE_SIZE,
        8,
        ((hidden // _NVFP4_GROUP_SIZE) * 2 * _W1_OUTPUT_TILE_SIZE) // 8,
    )
    expected_w2_scale_shape = (
        local_experts,
        intermediate // _W1_OUTPUT_TILE_SIZE,
        hidden // _W2_OUTPUT_TILE_SIZE,
        _W1_OUTPUT_TILE_SIZE // _NVFP4_GROUP_SIZE,
        _W2_OUTPUT_TILE_SIZE,
    )
    if (
        tuple(w1_scale.shape) != expected_w1_scale_shape
        or tuple(w2_scale.shape) != expected_w2_scale_shape
    ):
        raise ValueError(
            "Helion fused-RS scale shapes do not match NVFP4 group size "
            f"{_NVFP4_GROUP_SIZE}: got {tuple(w1_scale.shape)} and "
            f"{tuple(w2_scale.shape)}"
        )
    route_block_size = _route_block_size(tokens, topk, world_size, local_experts)
    padded_tokens = _padded_token_count(
        tokens,
        topk,
        world_size,
        route_block_size,
    )
    if padded_tokens != tokens:
        padding = padded_tokens - tokens
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
        topk_weights = torch.nn.functional.pad(topk_weights, (0, 0, 0, padding))
        topk_ids = torch.nn.functional.pad(topk_ids, (0, 0, 0, padding))

    route_blocks = _route_block_count(
        padded_tokens,
        topk,
        local_experts,
        route_block_size,
    )
    sorted_assignments, sorted_route_weights, group_sizes = _prepare_sorted_routes(
        topk_ids,
        topk_weights,
        global_experts=world_size * local_experts,
    )
    output = _fused_nvfp4_moe_reduce_scatter(
        hidden_states.reshape(
            padded_tokens,
            hidden // _NVFP4_GROUP_SIZE,
            _NVFP4_GROUP_SIZE,
        ),
        w1,
        w2,
        w1_scale,
        w2_scale,
        sorted_assignments,
        sorted_route_weights,
        group_sizes,
        expert_offsets,
        tp_rank_to_physical,
        physical_rank,
        post_expert_norm_weight,
        world_size,
        route_blocks,
        route_block_size,
        topk,
        padded_tokens // world_size,
        hidden,
        intermediate,
        _NVFP4_GROUP_SIZE,
        _W1_OUTPUT_TILE_SIZE,
        process_group_name,
    )
    return output


@spmd_safe
def helion_fused_moe_rs_gather(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    post_expert_norm_weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    tp_rank_to_physical: torch.Tensor,
    physical_rank: torch.Tensor,
    world_size: int,
    process_group_name: str,
    keep_token_shard: bool = False,
) -> torch.Tensor:
    """Run fused-RS and return either a token shard or replicated rows."""
    from sglang.srt.distributed import get_tp_group

    tokens = hidden_states.shape[0]
    if keep_token_shard and tokens % world_size:
        raise RuntimeError(
            "Helion fused-RS sharded output requires input rows divisible by "
            f"TP={world_size}, got {tokens}"
        )

    topk = topk_ids.shape[1]
    route_block_size = _route_block_size(tokens, topk, world_size, w1.shape[0])
    padded_tokens = _padded_token_count(
        tokens,
        topk,
        world_size,
        route_block_size,
    )
    logical_shard_rows = tokens // world_size if keep_token_shard else None
    padded_shard_rows = padded_tokens // world_size

    if keep_token_shard and padded_tokens != tokens:
        # Inputs are logical-rank-major. Pad every rank's block independently
        # so a real row cannot move into another rank's output shard.
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

    token_shard = helion_fused_moe_rs_call(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        post_expert_norm_weight=post_expert_norm_weight,
        expert_offsets=expert_offsets,
        tp_rank_to_physical=tp_rank_to_physical,
        physical_rank=physical_rank,
        world_size=world_size,
        process_group_name=process_group_name,
    )
    if token_shard.ndim != 2 or token_shard.shape != (
        padded_shard_rows,
        hidden_states.shape[-1],
    ):
        raise RuntimeError(
            "Helion fused-RS kernel returned an unexpected shard shape: got "
            f"{tuple(token_shard.shape)}, expected "
            f"({padded_shard_rows}, {hidden_states.shape[-1]})"
        )
    if keep_token_shard:
        return token_shard[:logical_shard_rows].contiguous()

    output = get_tp_group().all_gather(token_shard, dim=0)
    return output[:tokens]
