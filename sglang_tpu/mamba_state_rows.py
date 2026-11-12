# SPDX-License-Identifier: Apache-2.0
"""In-place row writes into a TPU recurrent-state pool tensor.

An XLA scatter into a ``[slots, K-1, 1, dim]`` pool relayouts the whole pool
to its preferred tiling and back on every call, so its cost grows with the
slot count. This op DMAs each live row into the donated pool instead: the pool
keeps its storage layout and only the written rows move.
"""

import functools

import torch


@functools.cache
def state_row_write_op():
    """``op(pool, rows, live, new) -> pool``: ``pool[rows[i]] = new[i]`` where ``live[i]``.

    ``rows`` and ``live`` are int32 ``[n]``, ``new`` is ``[n, *pool.shape[1:]]`` in
    ``pool.dtype``. Rows must be distinct among live entries.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def write_rows(
        pool: jax.Array, rows: jax.Array, live: jax.Array, new: jax.Array
    ) -> jax.Array:
        num_rows = new.shape[0]

        def kernel(rows_ref, live_ref, new_ref, pool_in, pool_out, sems):
            del pool_in  # aliased to pool_out

            def copy(i):
                return pltpu.make_async_copy(
                    new_ref.at[pl.ds(i, 1)],
                    pool_out.at[pl.ds(rows_ref[i], 1)],
                    sems.at[i],
                )

            # Issue every row's DMA before waiting on any.
            for i in range(num_rows):

                @pl.when(live_ref[i] != 0)
                def _start():
                    copy(i).start()

            for i in range(num_rows):

                @pl.when(live_ref[i] != 0)
                def _wait():
                    copy(i).wait()

        with tpu_info_for_cpu_export():
            return pl.pallas_call(
                kernel,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    grid=(1,),
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    scratch_shapes=[pltpu.SemaphoreType.DMA((num_rows,))],
                ),
                out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
                input_output_aliases={3: 0},
            )(rows, live, new, pool)

    op = pallas.jax_op("sglang::tpu_state_row_write", write_rows, donate_argnums=(0,))

    @op.register_fake
    def fake(pool, rows, live, new):
        return torch.empty_like(pool)

    return op


@functools.cache
def state_row_read_op():
    """``op(pool, rows) -> (pool, pool[rows])`` for int32 ``rows`` in ``[0, slots)``.

    The pool passes through aliased so XLA keeps it in place in HBM rather than
    prefetching the whole tensor for a few rows.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def read_rows(pool: jax.Array, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
        num_rows = rows.shape[0]

        def kernel(rows_ref, pool_ref, pool_out, out_ref, sems):
            del pool_out  # aliased to pool_ref, unchanged
            copies = [
                pltpu.make_async_copy(
                    pool_ref.at[pl.ds(rows_ref[i], 1)],
                    out_ref.at[pl.ds(i, 1)],
                    sems.at[i],
                )
                for i in range(num_rows)
            ]
            for copy in copies:
                copy.start()
            for copy in copies:
                copy.wait()

        with tpu_info_for_cpu_export():
            return pl.pallas_call(
                kernel,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    grid=(1,),
                    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                    out_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    scratch_shapes=[pltpu.SemaphoreType.DMA((num_rows,))],
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(pool.shape, pool.dtype),
                    jax.ShapeDtypeStruct((num_rows, *pool.shape[1:]), pool.dtype),
                ],
                input_output_aliases={1: 0},
            )(rows, pool)

    op = pallas.jax_op("sglang::tpu_state_row_read", read_rows, donate_argnums=(0,))

    @op.register_fake
    def fake(pool, rows):
        return torch.empty_like(pool), pool.new_empty((rows.shape[0], *pool.shape[1:]))

    return op
