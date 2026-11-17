# SPDX-License-Identifier: Apache-2.0
"""TorchTPU selected-route bridge to the existing fused EP MoE v2 layer.

The FFN caller supplies replicated DP rows and logical expert IDs. Each mesh
member handles a disjoint token slice, and the layer restores the completed
replica. The explicit EP-rank operand preserves expert ownership when the
physical mesh order differs from the process-group rank order.
"""

import logging
from functools import cache

logger = logging.getLogger(__name__)


@cache
def get_selected_op(ep_size: int, capacity: int = 128):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.moe.pallas.fused_moe_v2.host import (
        U32_SUBLANE_TILE,
        pair_block_hist,
        routing_block,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.kernel import rowquant_fp8
    from sglang_tpu.moe.pallas.fused_moe_v2.layer import (
        _exchange_layout,
        _pack_routing_exchange,
        fused_ep_moe_v2,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.token_parallel import TokenReplicaLayout
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    axis = "d"
    mesh = get_pallas_mesh((axis,), (ep_size,))
    sharded = P(axis)
    groups = (tuple(range(ep_size)),)

    def run(
        x: jax.Array,
        ids: jax.Array,
        coefficients: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        logical_rank: jax.Array,
    ) -> jax.Array:
        experts = w2.shape[0]
        topk = ids.shape[1]
        layout = TokenReplicaLayout.create(
            ep=ep_size,
            input_rows=x.shape[0] // ep_size,
            groups=groups,
            row_alignment=U32_SUBLANE_TILE,
        )
        rows = layout.local_rows
        block = routing_block(rows, topk)
        payload_layout = _exchange_layout(rows, topk, experts, True)
        logger.info(
            "GLM_FUSED_MOE_V2_EXPORT ep=%d rows=%d hidden=%d topk=%d capacity=%d",
            ep_size,
            layout.input_rows,
            x.shape[-1],
            topk,
            capacity,
        )

        @jax.shard_map(
            mesh=mesh,
            in_specs=(sharded,) * 4,
            out_specs=(sharded,) * 4,
            check_vma=False,
        )
        def prepare(x_local, ids_local, coefficients_local, rank_local):
            ranks = lax.all_gather(rank_local, axis, tiled=True)
            me = jnp.argmax(ranks == rank_local[0]).astype(jnp.int32)
            local_x = layout.split(x_local, me)
            local_ids = layout.split(ids_local, me, fill_value=-1)
            local_coefficients = layout.split(coefficients_local, me)
            valid = (local_ids >= 0) & (local_ids < experts)
            safe_ids = jnp.clip(local_ids, 0, experts - 1)
            inverse = jnp.argsort(ranks)
            experts_per_rank = experts // ep_size
            mapped = (
                inverse[safe_ids // experts_per_rank] * experts_per_rank
                + safe_ids % experts_per_rank
            )
            mapped = jnp.where(valid, mapped, experts).astype(jnp.int32)
            local_coefficients = jnp.where(valid, local_coefficients, 0)
            quantized, scales = rowquant_fp8(local_x.astype(jnp.bfloat16))
            quantized = quantized.reshape(rows, x.shape[-1] // 128, 128)
            blocks = mapped.reshape(rows * topk // block, block)
            histogram = pair_block_hist(blocks, experts)
            payload = _pack_routing_exchange(
                blocks.reshape(-1),
                lax.bitcast_convert_type(scales[:, 0].astype(jnp.float32), jnp.int32),
                histogram.sum(axis=0),
                payload_layout,
            )
            gathered = lax.all_gather(quantized, axis, axis=0, tiled=True)
            return gathered, local_coefficients, payload[None, :], me.reshape(1, 1)

        gathered, local_coefficients, payload, physical_rank = prepare(
            x, ids, coefficients, logical_rank
        )
        return fused_ep_moe_v2(
            x,
            w13,
            w2,
            s13.reshape(experts, -1).astype(jnp.float32),
            s2.reshape(experts, -1).astype(jnp.float32),
            jnp.zeros((x.shape[0], experts), jnp.float32),
            topk=topk,
            renormalize=True,
            mesh=mesh,
            capacity=capacity,
            rank=physical_rank,
            token_replica_groups=groups,
            stage="compute",
            routed=(gathered, local_coefficients, payload),
        )

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::selected_fused_ep_moe_v2_ep{ep_size}_c{capacity}",
        run,
        mesh=mesh,
        input_partition_specs=(sharded,) * 8,
        output_partition_specs=(sharded,),
    )
