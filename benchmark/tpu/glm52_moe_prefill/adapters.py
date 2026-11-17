"""Benchmark adapters using the three existing MoE implementations."""

from functools import cache


@cache
def baseline_op():
    import jax
    import jax.numpy as jnp
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def run(x: jax.Array, ids: jax.Array, coefficients: jax.Array,
            w13: jax.Array, w2: jax.Array, s13: jax.Array, s2: jax.Array,
            logical_rank: jax.Array) -> jax.Array:
        return _fused_moe_jax(
            x, w13, w2, coefficients.astype(x.dtype), ids, s13, s2, None, None,
            topk=8, activation_name="silu", global_num_experts=256,
            use_sc_gather=1, expert_group_offset_tensor=logical_rank * w2.shape[0],
        )

    return cpu_export_jax_op("sglang_tpu_pallas::glm_prefill_gmm_baseline", run)


@cache
def existing_v2_selected_op(ep):
    """Build the existing route payload from caller-selected IDs and weights.

    The original stage='compute' entry owns routing-table construction,
    expert FFN, remote transport, weighted combine, and replica restoration.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op
    from sglang_tpu.moe.pallas.fused_moe_v2.host import (
        U32_SUBLANE_TILE, pair_block_hist, routing_block,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.kernel import rowquant_fp8
    from sglang_tpu.moe.pallas.fused_moe_v2.layer import (
        _exchange_layout, _pack_routing_exchange, fused_ep_moe_v2,
    )
    from sglang_tpu.moe.pallas.fused_moe_v2.token_parallel import TokenReplicaLayout

    axis = "d"
    mesh = get_pallas_mesh((axis,), (ep,))
    sharded = P(axis)
    groups = (tuple(range(ep)),)

    def run(x: jax.Array, ids: jax.Array, coefficients: jax.Array,
            w13: jax.Array, w2: jax.Array, s13: jax.Array, s2: jax.Array,
            logical_rank: jax.Array) -> jax.Array:
        layout = TokenReplicaLayout.create(
            ep=ep, input_rows=x.shape[0] // ep, groups=groups,
            row_alignment=U32_SUBLANE_TILE,
        )
        rows = layout.local_rows
        block = routing_block(rows, 8)
        payload_layout = _exchange_layout(rows, 8, 256, True)

        @jax.shard_map(mesh=mesh, in_specs=(sharded,) * 4,
                       out_specs=(sharded,) * 4, check_vma=False)
        def prepare(x_local, ids_local, coefficients_local, rank_local):
            ranks = lax.all_gather(rank_local, axis, tiled=True)
            me = jnp.argmax(ranks == rank_local[0]).astype(jnp.int32)
            local_x = layout.split(x_local, me)
            local_ids = layout.split(ids_local, me, fill_value=-1)
            local_coefficients = layout.split(coefficients_local, me)
            inverse = jnp.argsort(ranks)
            experts_per_rank = 256 // ep
            safe = jnp.maximum(local_ids, 0)
            mapped = inverse[safe // experts_per_rank] * experts_per_rank + safe % experts_per_rank
            mapped = jnp.where(local_ids >= 0, mapped, 256).astype(jnp.int32)
            quantized, scales = rowquant_fp8(local_x.astype(jnp.bfloat16))
            quantized = quantized.reshape(rows, x.shape[-1] // 128, 128)
            blocks = mapped.reshape(rows * 8 // block, block)
            histogram = pair_block_hist(blocks, 256)
            payload = _pack_routing_exchange(
                blocks.reshape(-1),
                lax.bitcast_convert_type(scales[:, 0].astype(jnp.float32), jnp.int32),
                histogram.sum(axis=0), payload_layout,
            )
            gathered = lax.all_gather(quantized, axis, axis=0, tiled=True)
            return gathered, local_coefficients, payload[None, :], me.reshape(1, 1)

        gathered, local_coefficients, payload, physical_rank = prepare(
            x, ids, coefficients, logical_rank,
        )
        # The compute stage consumes route's prepared data, including row scales.
        return fused_ep_moe_v2(
            x, w13, w2, s13.reshape(s13.shape[0], -1).astype(jnp.float32),
            s2.reshape(s2.shape[0], -1).astype(jnp.float32),
            jnp.zeros((x.shape[0], 256), jnp.float32),
            topk=8, renormalize=True, mesh=mesh, capacity=128,
            rank=physical_rank, token_replica_groups=groups,
            stage="compute", routed=(gathered, local_coefficients, payload),
        )

    return cpu_export_jax_op(
        "sglang_tpu_pallas::glm_prefill_existing_v2_selected", run,
        mesh=mesh, input_partition_specs=(sharded,) * 8,
        output_partition_specs=(sharded,),
    )
