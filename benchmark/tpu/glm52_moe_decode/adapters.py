"""Benchmark adapters using the three existing MoE implementations."""

from functools import cache


@cache
def baseline_op():
    import jax
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

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
        return _fused_moe_jax(
            x,
            w13,
            w2,
            coefficients.astype(x.dtype),
            ids,
            s13,
            s2,
            None,
            None,
            topk=8,
            activation_name="silu",
            global_num_experts=256,
            use_sc_gather=1,
            expert_group_offset_tensor=logical_rank * w2.shape[0],
        )

    return cpu_export_jax_op("sglang_tpu_pallas::glm_decode_gmm_baseline", run)
