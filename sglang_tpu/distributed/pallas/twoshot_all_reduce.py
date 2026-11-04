# SPDX-License-Identifier: Apache-2.0
"""Two-shot all-reduce for small, latency-bound payloads on TPU.

A column reduce-scatter then an all-gather, both as direct remote DMAs between
VMEM buffers; faster than XLA's all-reduce for decode-sized payloads.
"""

import functools


def _twoshot_all_reduce_kernel(
    x_ref,
    out_ref,
    scatter_ref,
    own_ref,
    scatter_send,
    scatter_receive,
    gather_send,
    gather_receive,
    *,
    world_size: int,
    chunk: int,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    # Peers write into this kernel's VMEM, so every device must be resident before any send.
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(jnp.mod(me + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)

    def columns(rank):
        return pl.ds(pl.multiple_of(rank * chunk, 128), chunk)

    def exchange(source, slot_of_offset, send, receive):
        # Symmetric pairing: what this device receives from (me - offset) lands in the slot
        # it sends to (me + offset) under, with the same semaphore index.
        copies = []
        for offset in range(1, world_size):
            peer = jnp.mod(me + offset, world_size)
            copy = pltpu.make_async_remote_copy(
                source(peer),
                slot_of_offset(offset),
                send.at[offset - 1],
                receive.at[offset - 1],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    # Reduce-scatter: column chunk `peer` of this partial goes to `peer`, summed there in f32.
    exchange(
        lambda peer: x_ref.at[:, columns(peer)],
        lambda offset: scatter_ref.at[offset - 1],
        scatter_send,
        scatter_receive,
    )
    total = x_ref[:, columns(me)].astype(jnp.float32)
    for slot in range(world_size - 1):
        total = total + scatter_ref[slot].astype(jnp.float32)
    own_ref[...] = total.astype(own_ref.dtype)
    out_ref[:, columns(me)] = own_ref[...]
    # All-gather: this device's reduced chunk goes to the same columns of every peer's output.
    exchange(
        lambda peer: own_ref,
        lambda offset: out_ref.at[:, columns(me)],
        gather_send,
        gather_receive,
    )


def twoshot_all_reduce(x, *, world_size: int, collective_id: int = 7):
    """Sum ``x`` [m, n] over the ``model`` mesh axis, accumulating in f32.

    Every device gets the same rounded total. ``n`` must be a multiple of
    ``128 * world_size``.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, cols = x.shape
    chunk = cols // world_size
    kernel = functools.partial(
        _twoshot_all_reduce_kernel, world_size=world_size, chunk=chunk
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((world_size - 1, rows, chunk), x.dtype),
            pltpu.VMEM((rows, chunk), x.dtype),
            pltpu.SemaphoreType.DMA((world_size - 1,)),
            pltpu.SemaphoreType.DMA((world_size - 1,)),
            pltpu.SemaphoreType.DMA((world_size - 1,)),
            pltpu.SemaphoreType.DMA((world_size - 1,)),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"twoshot_all_reduce_tp{world_size}",
    )(x)
