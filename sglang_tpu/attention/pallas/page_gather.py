# SPDX-License-Identifier: Apache-2.0
"""Copy selected pages of a paged cache, one DMA per page.

An XLA gather along the page axis of a large tiled cache (an ``index_select`` in
a compiled graph) can lay the whole cache out again for the gather: a temporary
the size of the cache. This kernel leaves the cache in HBM and copies each
requested page straight into the output.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def gather_pages(cache: jax.Array, page_ids: jax.Array) -> jax.Array:
    """``cache[page_ids]`` for ``cache[num_pages, ...]`` and ``i32[G]`` ids."""
    num_out = page_ids.shape[0]

    def kernel(ids_ref, cache_hbm, out_hbm, sem):
        def start(i, carry):
            pltpu.make_async_copy(cache_hbm.at[ids_ref[i]], out_hbm.at[i], sem).start()
            return carry

        def wait(i, carry):
            # Every copy is one page, so each wait retires one of them.
            pltpu.make_async_copy(cache_hbm.at[0], out_hbm.at[0], sem).wait()
            return carry

        lax.fori_loop(0, num_out, start, 0)
        lax.fori_loop(0, num_out, wait, 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())],
        ),
        out_shape=jax.ShapeDtypeStruct((num_out, *cache.shape[1:]), cache.dtype),
        name="gather_pages",
    )(
        # A DMA from an out-of-range page faults the chip, where an XLA gather
        # clamps; warmup's dummy page tables are not guaranteed in range.
        jnp.clip(page_ids.astype(jnp.int32), 0, cache.shape[0] - 1),
        cache,
    )
