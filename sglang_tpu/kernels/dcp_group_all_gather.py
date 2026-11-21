# SPDX-License-Identifier: Apache-2.0
"""All-gather inside each decode-context-parallel group on TPU.

Mesh positions ``[g * group_size, (g + 1) * group_size)`` of the ``model``
axis form DCP group ``g``. Each device DMA-writes its HBM input into slot
``dcp rank`` of every group peer's HBM output: one hop, no XLA collective, and
only the group's own data on the wire. It replaces the zero-padded all-reduce
gather, which sends ``group_size`` times the payload. The slot follows the
DCP rank the caller passes, which need not match the position in the mesh.

Follows ``oneshot_all_gather`` (same barrier and semaphore pairing), with HBM
on both ends for multi-MB payloads.
"""

import functools

# Pallas barrier id; ids in use are listed next to each kernel's COLLECTIVE_ID.
DCP_GROUP_ALL_GATHER_COLLECTIVE_ID = 7

# Rows of one remote copy (1 MiB of bytes). Measured on tpu7x in a TP8 server:
# copies of up to 17536 rows arrive intact, copies of 32768 rows and more do not.
CHUNK_ROWS = 8192


def _dcp_group_all_gather_kernel(
    rank_ref,  # i32[1]: this device's DCP rank, the output slot it fills
    x_ref,  # [rows, 128] in HBM
    out_ref,  # [group_size, rows, 128] in HBM: slot r came from DCP rank r
    send_semaphores,
    receive_semaphores,
    *,
    group_size: int,
    rows: int,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    member = jnp.mod(me, group_size)
    base = me - member

    def peer(offset: int):
        return base + jnp.mod(member + offset, group_size)

    # Entry barrier: no peer writes into this device's output before it has
    # entered the kernel, so XLA's buffer reuse around the call cannot race
    # the DMAs.
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, group_size):
        pl.semaphore_signal(
            barrier,
            device_id=(peer(offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, group_size - 1)
    slot = rank_ref[0].astype(jnp.int32)
    # One wave per chunk: every peer's copy of the chunk starts, then all are
    # awaited, so the per-peer semaphores are free again for the next wave.
    for start in range(0, rows, CHUNK_ROWS):
        size = min(CHUNK_ROWS, rows - start)
        copies = []
        for offset in range(group_size):
            # The copy this device receives from member (member - offset) uses the
            # same semaphore index as the copy it sends to member (member + offset).
            copies.append(
                pltpu.make_async_remote_copy(
                    x_ref.at[pl.ds(start, size)],
                    out_ref.at[slot, pl.ds(start, size)],
                    send_semaphores.at[offset],
                    receive_semaphores.at[offset],
                    device_id=(peer(offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()


def dcp_group_all_gather(
    x,
    rank_input,
    *,
    group_size: int,
    collective_id: int = DCP_GROUP_ALL_GATHER_COLLECTIVE_ID,
):
    """``x`` [rows, 128] of every member of this device's DCP group as
    [group_size, rows, 128], by DCP rank (``rank_input``: i32[1])."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if x.ndim != 2 or x.shape[1] != 128:
        raise ValueError(f"expected [rows, 128], got {x.shape}")
    kernel = functools.partial(
        _dcp_group_all_gather_kernel, group_size=group_size, rows=x.shape[0]
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((group_size, *x.shape), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((group_size,)),
                pltpu.SemaphoreType.DMA((group_size,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"dcp_group_all_gather_g{group_size}",
    )(rank_input, x)
