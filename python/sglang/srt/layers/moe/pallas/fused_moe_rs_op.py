# SPDX-License-Identifier: Apache-2.0
"""TPU MoE with fused ICI reduce-scatter."""

import functools
import math
import os

import torch
from torch_tpu._internal.distributed.spmd_util import spmd_safe

_FUSED_MOE_RS_OP = None
_FUSED_MOE_RS_OP_NAME = "sglang_tpu_pallas::fused_moe_rs"
_FUSED_RS_ROUTED_ROW_ALIGNMENT = 16


def fused_moe_rs_padded_tokens(num_tokens: int, topk: int, ep_size: int) -> int:
    """Pad tokens for both EP sharding and the fused kernel's routed rows."""
    routed_row_token_alignment = _FUSED_RS_ROUTED_ROW_ALIGNMENT // math.gcd(
        _FUSED_RS_ROUTED_ROW_ALIGNMENT, topk
    )
    token_alignment = math.lcm(ep_size, routed_row_token_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


@functools.cache
def _get_physical_rank_metadata() -> tuple[int, tuple[int, ...]]:
    from torch_tpu._internal.distributed import tpu_distributed

    from sglang.srt.distributed import get_tp_group

    try:
        global_device_id = tpu_distributed.global_device_id()
    except Exception:
        global_device_id = 0

    tp_group = get_tp_group()
    world_size = tp_group.world_size if tp_group is not None else 1
    if world_size == 1:
        return global_device_id, (0,)

    try:
        import torch.distributed as dist

        if dist.is_initialized() and getattr(tp_group, "cpu_group", None) is not None:
            physical_ids = [None] * world_size
            dist.all_gather_object(
                physical_ids, global_device_id, group=tp_group.cpu_group
            )
            if sorted(physical_ids) == list(range(world_size)):
                physical_order = tuple(
                    physical_ids.index(pid) for pid in range(world_size)
                )
                return global_device_id, physical_order
    except Exception:
        pass

    physical_order = tuple(range(world_size))
    return global_device_id, physical_order


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare physical-rank metadata before Dynamo captures the MoE layer."""
    from sglang.srt.runtime_context import get_parallel

    global_device_id, physical_order = _get_physical_rank_metadata()
    device = layer.w13_weight.device

    physical_order_tensor = torch.tensor(
        physical_order,
        dtype=torch.int64,
        device=device,
    )
    group_offset = torch.tensor(
        [
            get_parallel().moe_ep_rank * layer.num_local_experts,
            global_device_id,
        ],
        dtype=torch.int32,
        device=device,
    )
    layer.register_buffer(
        "_tpu_fused_rs_physical_order", physical_order_tensor, persistent=False
    )
    layer.register_buffer("_tpu_fused_rs_group_offset", group_offset, persistent=False)
    _build_fused_moe_rs_op()


@functools.cache
def _get_mesh():
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_jax_export_devices,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    export_devices = np.asarray(get_jax_export_devices(world_size)).reshape(1, -1)
    return Mesh(export_devices, ("data", "model"))


def _build_fused_moe_rs_op():
    global _FUSED_MOE_RS_OP
    if _FUSED_MOE_RS_OP is not None:
        return _FUSED_MOE_RS_OP

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op
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

    # Bridge the JAX kernel into Torch with model-sharded weights and output.
    _FUSED_MOE_RS_OP = cpu_export_jax_op(
        _FUSED_MOE_RS_OP_NAME,
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(),
            P("model"),
            P("model"),
            P("model"),
            P("model"),
            P(),
            P(),
            P(),
            P("model"),
            None,
            None,
            None,
            None,
        ),
        output_partition_specs=(P("model", None),),
    )
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


@spmd_safe
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
    physical_order_to_tp_rank: torch.Tensor,
) -> torch.Tensor:
    from sglang.srt.distributed import get_tp_group

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
    
    shard_tokens = token_shard.shape[0]
    hidden_size = token_shard.shape[1]
    shard_size_in_bytes = token_shard.numel() * token_shard.element_size()
    
    max_chunk_size = 256 * 1024  # 256 KiB per rank (2.0 MiB total per gathered chunk, under 2.5 MiB Spmem limit)
    num_chunks = 1
    if shard_size_in_bytes > max_chunk_size:
        raw_chunks = (shard_size_in_bytes + max_chunk_size - 1) // max_chunk_size
        while num_chunks < raw_chunks:
            num_chunks *= 2

    tp_group = get_tp_group()
    tp_size = physical_order_to_tp_rank.shape[0]
    
    if num_chunks > 1:
        chunks = token_shard.chunk(num_chunks, dim=0)
        gathered_chunks = []
        for chunk in chunks:
            gathered_chunk = tp_group.all_gather(chunk, dim=0)
            gathered_chunk = gathered_chunk.reshape(tp_size, chunk.shape[0], -1)[physical_order_to_tp_rank].contiguous()
            gathered_chunks.append(gathered_chunk)
        output = torch.cat(gathered_chunks, dim=1).reshape(-1, hidden_size)
    else:
        output = tp_group.all_gather(token_shard, dim=0)
        output = output.reshape(tp_size, shard_tokens, -1)[physical_order_to_tp_rank].contiguous().reshape(-1, hidden_size)
        
    return output

