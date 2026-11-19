# SPDX-License-Identifier: Apache-2.0
"""One-shot all-to-all on TPU: block ``j`` of every device goes to device ``j``.

Each device DMAs block ``j`` of its HBM input into slot ``my mesh rank`` of
device ``j``'s HBM output, all peers at once: ``oneshot_all_gather`` with a
different block per peer, HBM on both ends, for multi-MB payloads and without
XLA collectives (see kimi_k3/logits_gather.py). Peers are addressed by mesh
position; the caller makes that the rank it exchanges over.
"""

import functools

# Pallas barrier id; see the list in k3_megakernel/config.py.
ALL_TO_ALL_COLLECTIVE_ID = 3


def _oneshot_all_to_all_kernel(
    x_ref,  # [world_size, rows, 128] in HBM: block j goes to device j
    out_ref,  # [world_size, rows, 128] in HBM: block i came from device i
    send_semaphores,
    receive_semaphores,
    *,
    world_size: int,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    # Entry barrier: no peer writes into this device's output before it has
    # entered the kernel, so XLA's buffer reuse around the call cannot race
    # the DMAs.
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            device_id=(jnp.mod(me + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)
    copies = []
    for offset in range(world_size):
        # The copy this device receives from (me - offset) uses the same
        # semaphore index as the copy it sends to (me + offset).
        peer = jnp.mod(me + offset, world_size)
        copies.append(
            pltpu.make_async_remote_copy(
                x_ref.at[peer],
                out_ref.at[me],
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


def oneshot_all_to_all(
    x, *, world_size: int, collective_id: int = ALL_TO_ALL_COLLECTIVE_ID
):
    """All-to-all of ``x`` [world_size, rows, 128] over the ``model`` mesh axis.

    Returns the same shape: block ``i`` is block ``my mesh rank`` of device
    ``i``'s input. ``rows`` must fill whole tiles of ``x``'s dtype.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if x.ndim != 3 or x.shape[0] != world_size or x.shape[2] != 128:
        raise ValueError(f"expected [{world_size}, rows, 128], got {x.shape}")
    kernel = functools.partial(_oneshot_all_to_all_kernel, world_size=world_size)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((world_size,)),
                pltpu.SemaphoreType.DMA((world_size,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"oneshot_all_to_all_tp{world_size}",
    )(x)
