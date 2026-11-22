# SPDX-License-Identifier: Apache-2.0
"""Two-shot all-reduce for small, latency-bound payloads on TPU.

A column reduce-scatter then an all-gather, both as direct remote DMAs between
VMEM buffers; faster than XLA's all-reduce for decode-sized payloads.
"""

import functools

# Largest per-rank payload worth sending to this kernel instead of XLA's
# all-reduce. Measured on tpu7x 2x2x1 TP8, bf16, with the serving LIBTPU flags
# (SparseCore all-reduce offload off): two-shot 17-25 us vs XLA 19-28 us from
# 64 to 512 KiB, break-even near 768 KiB, XLA faster from 1 MiB. The cost tracks
# bytes, not columns ([64, 4096] and [32, 8192] time the same).
TWOSHOT_ALL_REDUCE_MAX_BYTES = 512 * 1024


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


def _bridge_twoshot_all_reduce_kernel(
    x_ref,
    out_ref,
    mate_recv_ref,
    half_ref,
    scatter_ref,
    own_ref,
    s1a_send,
    s1a_recv,
    s1b_send,
    s1b_recv,
    s2a_send,
    s2a_recv,
    s2b_send,
    s2b_recv,
    *,
    world_size: int,
    half: int,
    chunk: int,
) -> None:
    """2-stage chipmate-bridge all-reduce for TP8 on TPU7x (XOR=1 intra-chip bridge).

    Cuts off-chip ICI bytes by 4.5x vs flat two-shot (2 * 3/8 = 0.75x payload vs
    2 * 7/8 = 1.75x payload) by routing half-tensor exchanges over the 733 GB/s
    XOR=1 intra-chip die-to-die link and only 1/8-tensor slices across the 4 chips.
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index("model").astype(jnp.int32)
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(jnp.mod(me + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)

    mate = me ^ jnp.int32(1)
    side = me & jnp.int32(1)
    other_side = jnp.int32(1) - side
    chip = me >> jnp.int32(1)

    def half_cols(s):
        return pl.ds(pl.multiple_of(s * half, 128), half)

    def quarter_cols(c):
        return pl.ds(pl.multiple_of(c * chunk, 128), chunk)

    # Phase 1a: XOR=1 intra-chip reduce-scatter (each die gets its 4096-col half).
    c1a = pltpu.make_async_remote_copy(
        x_ref.at[:, half_cols(other_side)],
        mate_recv_ref,
        s1a_send.at[0],
        s1a_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c1a.start()
    c1a.wait()
    half_ref[...] = (
        x_ref[:, half_cols(side)].astype(jnp.float32)
        + mate_recv_ref[...].astype(jnp.float32)
    ).astype(half_ref.dtype)

    # Phase 1b: 4-chip reduce-scatter across peers me ^ (d << 1) for d in (1, 2, 3).
    c1b = []
    for d in range(1, 4):
        peer = me ^ jnp.int32(d << 1)
        peer_chip = chip ^ jnp.int32(d)
        cp = pltpu.make_async_remote_copy(
            half_ref.at[:, quarter_cols(peer_chip)],
            scatter_ref.at[d - 1],
            s1b_send.at[d - 1],
            s1b_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        cp.start()
        c1b.append(cp)
    for cp in c1b:
        cp.wait()

    total = half_ref[:, quarter_cols(chip)].astype(jnp.float32)
    for slot in range(3):
        total = total + scatter_ref[slot].astype(jnp.float32)
    own_ref[...] = total.astype(own_ref.dtype)
    half_ref[:, quarter_cols(chip)] = own_ref[...]

    # Phase 2a: 4-chip all-gather across peers me ^ (d << 1) for d in (1, 2, 3).
    c2a = []
    for d in range(1, 4):
        peer = me ^ jnp.int32(d << 1)
        cp = pltpu.make_async_remote_copy(
            own_ref,
            half_ref.at[:, quarter_cols(chip)],
            s2a_send.at[d - 1],
            s2a_recv.at[d - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        cp.start()
        c2a.append(cp)
    for cp in c2a:
        cp.wait()

    # Phase 2b: XOR=1 intra-chip all-gather to assemble full [rows, cols] in VMEM.
    c2b = pltpu.make_async_remote_copy(
        half_ref,
        out_ref.at[:, half_cols(side)],
        s2b_send.at[0],
        s2b_recv.at[0],
        device_id=(mate,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    c2b.start()
    out_ref[:, half_cols(side)] = half_ref[...]
    c2b.wait()


def bridge_twoshot_all_reduce(x, *, world_size: int = 8, collective_id: int = 8):
    """2-stage chipmate-bridge all-reduce for TP8 ``[m, n]`` on TPU7x."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if world_size != 8:
        return twoshot_all_reduce(x, world_size=world_size, collective_id=collective_id)
    rows, cols = x.shape
    half = cols // 2
    chunk = cols // world_size
    kernel = functools.partial(
        _bridge_twoshot_all_reduce_kernel,
        world_size=world_size,
        half=half,
        chunk=chunk,
    )
    compiler_kw = {"collective_id": collective_id}
    if rows * cols * x.dtype.itemsize > 4 * 1024 * 1024:
        compiler_kw["vmem_limit_bytes"] = 32 * 1024 * 1024
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((rows, half), x.dtype),
            pltpu.VMEM((rows, half), x.dtype),
            pltpu.VMEM((3, rows, chunk), x.dtype),
            pltpu.VMEM((rows, chunk), x.dtype),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA((3,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        compiler_params=pltpu.CompilerParams(**compiler_kw),
        name=f"bridge_twoshot_all_reduce_tp{world_size}",
    )(x)
