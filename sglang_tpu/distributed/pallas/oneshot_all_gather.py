# SPDX-License-Identifier: Apache-2.0
"""One-shot all-gather for small, latency-bound payloads on TPU.

Every device DMA-writes its whole shard straight into every peer's output
(one hop, all peers at once) instead of XLA's multi-step collective, whose
fixed latency dominates even for a single token row.
"""

import functools


def _oneshot_all_gather_kernel(
    rank_ref,
    x_ref,
    out_ref,
    send_semaphores,
    receive_semaphores,
    *,
    world_size: int,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    mesh_rank = lax.axis_index("model").astype(jnp.int32)
    # Entry barrier: no peer writes into this device's output before it has entered
    # the kernel, so XLA's buffer reuse around the call cannot race the DMAs.
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            device_id=(jnp.mod(mesh_rank + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)
    # Output slot follows the TP rank, which need not match the mesh order.
    slot = rank_ref[0].astype(jnp.int32)
    copies = []
    for offset in range(world_size):
        # Symmetric pairing: the copy this device receives from
        # (mesh_rank - offset) uses the same semaphore index `offset` as the
        # copy it sends to (mesh_rank + offset).
        peer = jnp.mod(mesh_rank + offset, world_size)
        copies.append(
            pltpu.make_async_remote_copy(
                x_ref,
                out_ref.at[slot],
                send_semaphores.at[offset],
                receive_semaphores.at[offset],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        )
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.wait()


def oneshot_all_gather(x, rank_input, *, world_size: int, collective_id: int = 5):
    """Gather ``x`` [m, n] from every TP rank into [world_size, m, n] (by TP rank)."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, cols = x.shape
    kernel = functools.partial(_oneshot_all_gather_kernel, world_size=world_size)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((world_size, rows, cols), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((world_size,)),
                pltpu.SemaphoreType.DMA((world_size,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"oneshot_all_gather_tp{world_size}",
    )(rank_input, x)
