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
_POSTPROCESS_ACCUM_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _validate_postprocess_contract(
    topk_weights: torch.Tensor, postprocess_accum_dtype: str
) -> None:
    if postprocess_accum_dtype not in _POSTPROCESS_ACCUM_DTYPES:
        supported = ", ".join(sorted(_POSTPROCESS_ACCUM_DTYPES))
        raise ValueError(
            f"Unsupported postprocess_accum_dtype={postprocess_accum_dtype!r}; "
            f"expected one of: {supported}"
        )
    if topk_weights.dtype not in _POSTPROCESS_ACCUM_DTYPES.values():
        raise TypeError(
            "Fused-RS router weights must be bfloat16 or float32, got "
            f"{topk_weights.dtype}"
        )
    expected_dtype = _POSTPROCESS_ACCUM_DTYPES[postprocess_accum_dtype]
    if topk_weights.dtype != expected_dtype:
        raise TypeError(
            f"postprocess_accum_dtype={postprocess_accum_dtype!r} requires "
            f"router weights with dtype {expected_dtype}, got "
            f"{topk_weights.dtype}"
        )


def fused_moe_rs_padded_tokens(num_tokens: int, topk: int, ep_size: int) -> int:
    """Pad tokens for both EP sharding and the fused kernel's routed rows."""
    routed_row_token_alignment = _FUSED_RS_ROUTED_ROW_ALIGNMENT // math.gcd(
        _FUSED_RS_ROUTED_ROW_ALIGNMENT, topk
    )
    token_alignment = math.lcm(ep_size, routed_row_token_alignment)
    return ((num_tokens + token_alignment - 1) // token_alignment) * token_alignment


@functools.cache
def _get_physical_rank_metadata() -> tuple[int, tuple[int, ...]]:
    import torch.distributed as dist
    from torch_tpu._internal.distributed import tpu_distributed

    from sglang.srt.distributed import get_tp_group

    global_device_id = tpu_distributed.global_device_id()
    tp_group = get_tp_group()
    physical_ids = [None] * tp_group.world_size
    dist.all_gather_object(physical_ids, global_device_id, group=tp_group.cpu_group)
    if sorted(physical_ids) != list(range(tp_group.world_size)):
        raise RuntimeError(f"Unexpected TPU physical device IDs: {physical_ids}")
    physical_order = tuple(
        physical_ids.index(pid) for pid in range(tp_group.world_size)
    )
    return global_device_id, physical_order


def prepare_fused_moe_rs(layer: torch.nn.Module) -> None:
    """Prepare physical-rank metadata before Dynamo captures the MoE layer."""
    from sglang.srt.runtime_context import get_parallel

    global_device_id, physical_order = _get_physical_rank_metadata()

    physical_order_tensor = torch.tensor(
        physical_order,
        dtype=torch.int64,
        device=layer.w13_weight.device,
    )
    group_offset = torch.tensor(
        [
            get_parallel().moe_ep_rank * layer.num_local_experts,
            global_device_id,
        ],
        dtype=torch.int32,
        device=layer.w13_weight.device,
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
        postprocess_accum_dtype: str,
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
            postprocess_accum_dtype=postprocess_accum_dtype,
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
    postprocess_accum_dtype: str = "bfloat16",
) -> torch.Tensor:
    _validate_postprocess_contract(topk_weights, postprocess_accum_dtype)
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
        postprocess_accum_dtype,
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
    postprocess_accum_dtype: str = "bfloat16",
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
        postprocess_accum_dtype=postprocess_accum_dtype,
    )
    output = get_tp_group().all_gather(token_shard, dim=0)
    shard_tokens = token_shard.shape[0]
    return output.reshape(
        physical_order_to_tp_rank.shape[0], shard_tokens, output.shape[-1]
    )[physical_order_to_tp_rank].reshape(-1, output.shape[-1])
