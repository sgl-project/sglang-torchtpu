# SPDX-License-Identifier: Apache-2.0
"""TorchTPU bridge for the selected-route GLM A2A fused kernel.

The FFN boundary has already gathered DP rows. Split that replica into disjoint
mesh-index slices, run the kernel's dispatch/combine, then recover the replica.
Expert ids remain logical EP ids at the public boundary. A rank operand maps
those ids to the physical Pallas mesh; pod ranks never enter that mapping.
"""

import logging
import os
from functools import cache

import torch

logger = logging.getLogger(__name__)

_SOURCE_DEBUG_ENVS = (
    "SGLJAX_SKIP_PREQUANT",
    "FUSED_MOE_V2_PAD_TOPK_TO_128",
    "FUSED_MOE_V2_ROUTE_SMEM_TOPK_ONLY",
)


def validate_source_environment():
    """Reject source-only tuning/debug settings, including a truthy '0'."""
    inherited = [name for name in _SOURCE_DEBUG_ENVS if name in os.environ]
    if inherited:
        raise ValueError(
            "A2A fused MoE V2 requires the safe source defaults; unset "
            + ", ".join(inherited)
            + ". Prequantization is required, TopK padding is 128, "
            "and the route-SMEM-only debug path is disabled."
        )


def pack_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """[E,H,2I] -> [2,E,H,I], once at load; each Pallas weight is a bitcast."""
    e, h, two_i = weight.shape
    return weight.reshape(e, h, 2, two_i // 2).permute(2, 0, 1, 3).contiguous()


def pack_gate_up_scale(scale: torch.Tensor) -> torch.Tensor:
    e, _, _, two_i = scale.shape
    return scale.reshape(e, 2, two_i // 2).permute(1, 0, 2).contiguous()


@cache
def get_op(ep_size: int, has_shared: bool):
    validate_source_environment()
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.kernel import a2a_fused_moe_v2
    from sglang_tpu.moe.pallas.a2a_fused_moe_v2.tuned_block_configs import (
        get_tuned_a2a_fused_moe_v2_block_config,
    )
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("data", "tensor"), (1, ep_size))
    ep = ("data", "tensor")
    sharded = P(ep)
    packed = P(None, ep)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(sharded,) * 4,
        out_specs=(sharded,) * 4,
        check_vma=False,
    )
    def split(x, ids, weights, logical_rank):
        # Every shard receives the same FFN replica. Physical mesh index owns
        # one contiguous slice, independently of the process/EP rank ordering.
        mesh_ranks = lax.all_gather(logical_rank, ep, tiled=True)
        # TorchTPU partitions this export again after embedding it. A host-HLO
        # partition-id is ambiguous there, so derive mesh position from the
        # explicit rank operand, just as the existing fused EP adapter does.
        me = jnp.argmax(mesh_ranks == logical_rank[0]).astype(jnp.int32)
        local_rows = (x.shape[0] + ep_size - 1) // ep_size
        padding = local_rows * ep_size - x.shape[0]
        x = jnp.pad(x, ((0, padding), (0, 0)))
        ids = jnp.pad(ids, ((0, padding), (0, 0)), constant_values=-1)
        weights = jnp.pad(weights, ((0, padding), (0, 0)))
        x = lax.dynamic_slice_in_dim(x, me * local_rows, local_rows, axis=0)
        ids = lax.dynamic_slice_in_dim(ids, me * local_rows, local_rows, axis=0)
        weights = lax.dynamic_slice_in_dim(weights, me * local_rows, local_rows, axis=0)
        # Position j in mesh_ranks owns logical experts [rank[j]*Elocal, ...).
        return x, ids, weights, mesh_ranks

    @jax.shard_map(
        mesh=mesh, in_specs=(packed, packed), out_specs=(sharded,) * 4, check_vma=False
    )
    def unpack(w13, s13):
        return w13[0], w13[1], s13[0, :, None, None, :], s13[1, :, None, None, :]

    @jax.shard_map(mesh=mesh, in_specs=sharded, out_specs=sharded, check_vma=False)
    def gather(y):
        return lax.all_gather(y, ep, axis=0, tiled=True)

    def run(
        x: jax.Array,
        ids: jax.Array,
        weights: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        rank: jax.Array,
        sw13: jax.Array,
        sw2: jax.Array,
        ss13: jax.Array,
        ss2: jax.Array,
    ) -> jax.Array:
        original_rows = x.shape[0] // ep_size
        # This runs while exporting a shape's actual custom-call graph, not
        # once per decode step; it gives E2E an operator/shape provenance marker.
        logger.info(
            "A2A_FUSED_MOE_V2_EXPORT op=a2a_fused_moe_v2_ep%d_shared%d rows=%d hidden=%d",
            ep_size,
            int(has_shared),
            original_rows,
            x.shape[-1],
        )
        x, ids, weights, mesh_ranks = split(x, ids, weights, rank)
        w1, w3, s1, s3 = unpack(w13, s13)
        # Build-time shape arithmetic; expert ownership never reads a device
        # scalar back to the host.
        elocal = w2.shape[0] // ep_size

        @jax.shard_map(
            mesh=mesh, in_specs=(sharded, sharded), out_specs=sharded, check_vma=False
        )
        def remap(i, r):
            inverse = jnp.argsort(r)
            safe = jnp.maximum(i, 0)
            mapped = inverse[safe // elocal] * elocal + safe % elocal
            return jnp.where(i >= 0, mapped, -1).astype(jnp.int32)

        ids = remap(ids, mesh_ranks)
        shared = {}
        if has_shared:
            shared = dict(
                w1_shared=sw13[0],
                w3_shared=sw13[1],
                w2_shared=sw2,
                w1_shared_scale=ss13[0, None, :],
                w3_shared_scale=ss13[1, None, :],
                w2_shared_scale=ss2.reshape(1, -1),
            )
        config = get_tuned_a2a_fused_moe_v2_block_config(
            num_tokens=x.shape[0],
            num_experts=w2.shape[0],
            top_k=ids.shape[1],
            hidden_size=x.shape[-1],
            intermediate_size=w2.shape[1],
            dtype=x.dtype,
            weight_dtype=w2.dtype,
            ep_size=ep_size,
            use_shared_expert=has_shared,
            use_grouped_topk=False,
            enable_act_quant=True,
            quant_mode="per_channel",
        )
        y = a2a_fused_moe_v2(
            mesh,
            x,
            w1,
            w2,
            w3,
            weights,
            ids,
            ids.shape[1],
            w1_scale=s1.astype(jnp.float32),
            w2_scale=s2.astype(jnp.float32),
            w3_scale=s3.astype(jnp.float32),
            direct_scaled_dot=True,
            enable_act_quant=True,
            block_config=config,
            **shared,
        )
        y = gather(y)

        @jax.shard_map(mesh=mesh, in_specs=sharded, out_specs=sharded, check_vma=False)
        def trim(a):
            return a[:original_rows]

        return trim(y)

    specs = (sharded,) * 3 + (packed, sharded, packed, sharded, sharded) + (P(),) * 4
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::a2a_fused_moe_v2_ep{ep_size}_shared{int(has_shared)}",
        run,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(sharded,),
    )
