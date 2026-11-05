# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-torchtpu (Apache-2.0), commit 0aff0785fea6:
# layers/adapter/custom_ops/mamba_state_copy_op.py and kernels/pool_adapters.py.
"""Bucketed, donating copies of native TPU recurrent-state slots."""

import functools

import torch


def copy_bucket_size(count: int) -> int:
    size = 8
    while size < count:
        size *= 4
    return size


@functools.cache
def compiled_state_copy():
    # Keep TPU/JAX imports out of CPU pool-layout tests.
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def copy_slots(
        pool: jax.Array, src: jax.Array, dst: jax.Array, count: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # Native COW allocates distinct destinations disjoint from all sources.
        # Transfer directly between HBM refs: a whole recurrent slot can exceed
        # VMEM. Padded pairs must not even read/write the reserved null slot.
        def kernel(src_ref, dst_ref, count_ref, pool_in, pool_out, sem):
            i = pl.program_id(0)

            @pl.when(i < count_ref[0])
            def copy_valid_slot():
                copy = pltpu.make_async_copy(
                    pool_in.at[pl.ds(src_ref[i], 1)],
                    pool_out.at[pl.ds(dst_ref[i], 1)],
                    sem,
                )
                copy.start()
                copy.wait()

        with tpu_info_for_cpu_export():
            result = pl.pallas_call(
                kernel,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=3,
                    grid=(src.shape[0],),
                    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    scratch_shapes=[pltpu.SemaphoreType.DMA],
                ),
                out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
                input_output_aliases={3: 0},
            )(src, dst, count, pool)
        return result, src[0]

    op = pallas.jax_op("sglang::tpu_mamba_state_copy", copy_slots, donate_argnums=(0,))

    @op.register_fake
    def fake(pool, src, dst, count):
        return torch.empty_like(pool), src.new_empty(())

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def run(pools, src, dst, count):
        # A single dispatch covers every layer and both state kinds. Passing
        # owning tensors lets XLA alias the donated outputs without pool copies.
        for pool in pools:
            result, marker = op(pool, src, dst, count)
            pool.copy_(result)
        return marker

    return run
