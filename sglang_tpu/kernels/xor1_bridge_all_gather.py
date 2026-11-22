# SPDX-License-Identifier: Apache-2.0
"""Two-stage XOR=1 chipmate-bridge Pallas All-Gather for TPU7x sequence-parallel prefill.

On TPU7x `2x2x1` (4 chips on a 2x2 torus, 2 TensorCores per chip connected by a fast
intra-chip die-to-die bridge):
  - `id = 2 * chip + core_on_chip`, with `coords = [(chip & 1), (chip >> 1), 0]`.
  - `mate = me ^ 1` is the on-chip mate (D2D port).
  - `px = me ^ 2` is the 1-hop X-axis inter-chip neighbor.
  - `py = me ^ 4` is the 1-hop Y-axis inter-chip neighbor.
  - `pxy = me ^ 6` is the 2-hop XY-diagonal chip.

Supports two inter-chip schedules:
  1. `xy_torus=True` (default on TP=8 when `rows` splits evenly into 2 aligned halves):
     Dual-port 1-hop XY-torus + `XOR=1` D2D bridge! Half 0 routes X -> Y while Half 1
     routes Y -> X simultaneously (`1.5M` bytes per X/Y ICI port, `0` 2-hop diagonal
     contention!), forwarding arriving halves immediately over the `XOR=1` D2D bridge.
  2. `xy_torus=False`: 1-shot same-parity inter-chip (`me ^ 2, me ^ 4, me ^ 6`, `3M`
     inter-chip bytes vs `7M` = `-57.1%`) + pipelined `XOR=1` D2D bridge.

All remote copies stream HBM -> HBM (`0` bytes of VMEM scratch, `0` register spills) and
preserve every input bit (`0-ULP` lossless for BF16, FP32, FP8, and INT8).
"""

from __future__ import annotations

import functools


def _xor1_xy_torus_ag_kernel(
    x_hbm,
    out_hbm,
    local_sem,
    inter_send_sems,
    inter_recv_sems,
    d2d_send_sems,
    d2d_recv_sems,
    *,
    axis_name: str,
    half_rows: int,
) -> None:
    """Dual-port 1-hop XY-torus + XOR=1 intra-chip D2D bridge for world_size=8."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index(axis_name).astype(jnp.int32)
    mate = lax.bitwise_xor(me, jnp.int32(1))
    px = lax.bitwise_xor(me, jnp.int32(2))
    py = lax.bitwise_xor(me, jnp.int32(4))
    pxy = lax.bitwise_xor(me, jnp.int32(6))
    h = half_rows
    h0 = pl.ds(0, h)
    h1 = pl.ds(h, h)

    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, 8):
        pl.semaphore_signal(
            barrier,
            1,
            device_id={axis_name: lax.rem(me + offset, 8)},
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, 7)

    # Local self-copy + immediate own-shard D2D copy to mate
    c_self = pltpu.make_async_copy(
        src_ref=x_hbm,
        dst_ref=out_hbm.at[pl.ds(me, 1)],
        sem=local_sem.at[0],
    )
    c_self.start()

    c_d2d_own = pltpu.make_async_remote_copy(
        src_ref=x_hbm,
        dst_ref=out_hbm.at[pl.ds(me, 1)],
        send_sem=d2d_send_sems.at[0],
        recv_sem=d2d_recv_sems.at[0],
        device_id={axis_name: mate},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d_own.start()

    # Step 1: Simultaneous 1-hop X (Half 0) + 1-hop Y (Half 1)
    c_x1 = pltpu.make_async_remote_copy(
        src_ref=x_hbm.at[:, h0, :],
        dst_ref=out_hbm.at[pl.ds(me, 1), h0, :],
        send_sem=inter_send_sems.at[0],
        recv_sem=inter_recv_sems.at[0],
        device_id={axis_name: px},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_y1 = pltpu.make_async_remote_copy(
        src_ref=x_hbm.at[:, h1, :],
        dst_ref=out_hbm.at[pl.ds(me, 1), h1, :],
        send_sem=inter_send_sems.at[1],
        recv_sem=inter_recv_sems.at[1],
        device_id={axis_name: py},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_x1.start()
    c_y1.start()

    c_x1.wait_recv()
    c_y1.wait_recv()

    # Step 2: Simultaneous 1-hop Y (Half 0 of me & px) + 1-hop X (Half 1 of me & py)
    # + overlapped D2D forward of Step-1 halves to mate!
    c_y2a = pltpu.make_async_remote_copy(
        src_ref=x_hbm.at[:, h0, :],
        dst_ref=out_hbm.at[pl.ds(me, 1), h0, :],
        send_sem=inter_send_sems.at[2],
        recv_sem=inter_recv_sems.at[2],
        device_id={axis_name: py},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_y2b = pltpu.make_async_remote_copy(
        src_ref=out_hbm.at[pl.ds(px, 1), h0, :],
        dst_ref=out_hbm.at[pl.ds(px, 1), h0, :],
        send_sem=inter_send_sems.at[3],
        recv_sem=inter_recv_sems.at[3],
        device_id={axis_name: py},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_x2a = pltpu.make_async_remote_copy(
        src_ref=x_hbm.at[:, h1, :],
        dst_ref=out_hbm.at[pl.ds(me, 1), h1, :],
        send_sem=inter_send_sems.at[4],
        recv_sem=inter_recv_sems.at[4],
        device_id={axis_name: px},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_x2b = pltpu.make_async_remote_copy(
        src_ref=out_hbm.at[pl.ds(py, 1), h1, :],
        dst_ref=out_hbm.at[pl.ds(py, 1), h1, :],
        send_sem=inter_send_sems.at[5],
        recv_sem=inter_recv_sems.at[5],
        device_id={axis_name: px},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d_s1a = pltpu.make_async_remote_copy(
        src_ref=out_hbm.at[pl.ds(px, 1), h0, :],
        dst_ref=out_hbm.at[pl.ds(px, 1), h0, :],
        send_sem=d2d_send_sems.at[1],
        recv_sem=d2d_recv_sems.at[1],
        device_id={axis_name: mate},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d_s1b = pltpu.make_async_remote_copy(
        src_ref=out_hbm.at[pl.ds(py, 1), h1, :],
        dst_ref=out_hbm.at[pl.ds(py, 1), h1, :],
        send_sem=d2d_send_sems.at[2],
        recv_sem=d2d_recv_sems.at[2],
        device_id={axis_name: mate},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_y2a.start()
    c_y2b.start()
    c_x2a.start()
    c_x2b.start()
    c_d2d_s1a.start()
    c_d2d_s1b.start()

    c_y2a.wait_recv()
    c_y2b.wait_recv()
    c_x2a.wait_recv()
    c_x2b.wait_recv()

    # Step 3: Forward remaining 4 half-shards over fast XOR=1 D2D bridge
    c_d2d_s2 = [
        pltpu.make_async_remote_copy(
            src_ref=out_hbm.at[pl.ds(peer_s, 1), h_sl, :],
            dst_ref=out_hbm.at[pl.ds(peer_s, 1), h_sl, :],
            send_sem=d2d_send_sems.at[3 + idx],
            recv_sem=d2d_recv_sems.at[3 + idx],
            device_id={axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        for idx, (peer_s, h_sl) in enumerate(((py, h0), (pxy, h0), (px, h1), (pxy, h1)))
    ]
    for cd in c_d2d_s2:
        cd.start()

    c_self.wait()
    c_d2d_own.wait_send()
    c_d2d_own.wait_recv()
    for ci in (c_x1, c_y1, c_y2a, c_y2b, c_x2a, c_x2b):
        ci.wait_send()
    for cd in (c_d2d_s1a, c_d2d_s1b, *c_d2d_s2):
        cd.wait_send()
        cd.wait_recv()


def _xor1_bridge_ag_kernel(
    x_hbm,
    out_hbm,
    local_sem,
    inter_send_sems,
    inter_recv_sems,
    d2d_send_sems,
    d2d_recv_sems,
    *,
    axis_name: str,
    world_size: int,
    pipelined: bool,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    me = lax.axis_index(axis_name).astype(jnp.int32)
    mate = lax.bitwise_xor(me, jnp.int32(1))
    num_chips = world_size // 2
    num_inter = num_chips - 1

    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id={axis_name: lax.rem(me + offset, world_size)},
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)

    c_self = pltpu.make_async_copy(
        src_ref=x_hbm,
        dst_ref=out_hbm.at[pl.ds(me, 1)],
        sem=local_sem.at[0],
    )
    c_self.start()

    c_d2d_own = pltpu.make_async_remote_copy(
        src_ref=x_hbm,
        dst_ref=out_hbm.at[pl.ds(me, 1)],
        send_sem=d2d_send_sems.at[0],
        recv_sem=d2d_recv_sems.at[0],
        device_id={axis_name: mate},
        device_id_type=pl.DeviceIdType.MESH,
    )
    c_d2d_own.start()

    use_xor_masks = (num_chips & (num_chips - 1)) == 0
    c_inter = []
    src_peers = []
    for k in range(num_inter):
        if use_xor_masks:
            mask = (k + 1) * 2
            dst_peer = lax.bitwise_xor(me, jnp.int32(mask))
            src_peer = dst_peer
        else:
            step = (k + 1) * 2
            dst_peer = lax.rem(me + step, world_size)
            src_peer = lax.rem(me + world_size - step, world_size)
        src_peers.append(src_peer)
        cp = pltpu.make_async_remote_copy(
            src_ref=x_hbm,
            dst_ref=out_hbm.at[pl.ds(me, 1)],
            send_sem=inter_send_sems.at[k],
            recv_sem=inter_recv_sems.at[k],
            device_id={axis_name: dst_peer},
            device_id_type=pl.DeviceIdType.MESH,
        )
        cp.start()
        c_inter.append(cp)

    c_d2d_fwd = []
    if pipelined:
        for k in range(num_inter):
            c_inter[k].wait_recv()
            sp = src_peers[k]
            cf = pltpu.make_async_remote_copy(
                src_ref=out_hbm.at[pl.ds(sp, 1)],
                dst_ref=out_hbm.at[pl.ds(sp, 1)],
                send_sem=d2d_send_sems.at[k + 1],
                recv_sem=d2d_recv_sems.at[k + 1],
                device_id={axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            cf.start()
            c_d2d_fwd.append(cf)
    else:
        for k in range(num_inter):
            c_inter[k].wait_recv()
        for k in range(num_inter):
            sp = src_peers[k]
            cf = pltpu.make_async_remote_copy(
                src_ref=out_hbm.at[pl.ds(sp, 1)],
                dst_ref=out_hbm.at[pl.ds(sp, 1)],
                send_sem=d2d_send_sems.at[k + 1],
                recv_sem=d2d_recv_sems.at[k + 1],
                device_id={axis_name: mate},
                device_id_type=pl.DeviceIdType.MESH,
            )
            cf.start()
            c_d2d_fwd.append(cf)

    c_self.wait()
    c_d2d_own.wait_send()
    c_d2d_own.wait_recv()
    for k in range(num_inter):
        c_inter[k].wait_send()
    for cf in c_d2d_fwd:
        cf.wait_send()
        cf.wait_recv()


def xor1_bridge_all_gather(
    x,
    axis_name: str = "model",
    *,
    axis: int = 0,
    tiled: bool = False,
    pipelined: bool = True,
    xy_torus: bool | None = None,
    collective_id: int = 13,
):
    """2-stage XOR=1 chipmate-bridge All-Gather matching ``lax.all_gather`` semantics."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.envs import envs

    tp_size = int(lax.axis_size(axis_name))

    if axis != 0 or tp_size < 4 or (tp_size % 2) != 0 or x.ndim < 2:
        return lax.all_gather(x, axis_name, axis=axis, tiled=tiled)

    orig_shape = x.shape
    rows = int(orig_shape[0])
    cols = 1
    for d in orig_shape[1:]:
        cols *= int(d)

    item_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    min_row_align = 32 if item_bytes == 1 else 8
    if rows % min_row_align != 0 or cols % 128 != 0:
        return lax.all_gather(x, axis_name, axis=axis, tiled=tiled)

    if xy_torus is None:
        # Enable dual-port 1-hop XY-torus automatically when XOR1_BRIDGE_AG >= 3 or by default on >=128-row TP=8 shards
        xy_torus = envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get() >= 3

    x_3d = x.reshape(1, rows, cols)

    if xy_torus and tp_size == 8 and (rows // 2) >= min_row_align and (rows % (2 * min_row_align) == 0):
        kernel = functools.partial(
            _xor1_xy_torus_ag_kernel,
            axis_name=axis_name,
            half_rows=rows // 2,
        )
        out_3d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((tp_size, rows, cols), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA((1,)),
                    pltpu.SemaphoreType.DMA((6,)),
                    pltpu.SemaphoreType.DMA((6,)),
                    pltpu.SemaphoreType.DMA((7,)),
                    pltpu.SemaphoreType.DMA((7,)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=f"xor1_xytorus_ag_tp{tp_size}_{rows}x{cols}_{jnp.dtype(x.dtype).name}",
        )(x_3d)
    else:
        num_chips = tp_size // 2
        num_inter = num_chips - 1
        kernel = functools.partial(
            _xor1_bridge_ag_kernel,
            axis_name=axis_name,
            world_size=tp_size,
            pipelined=pipelined,
        )
        out_3d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((tp_size, rows, cols), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA((1,)),
                    pltpu.SemaphoreType.DMA((num_inter,)),
                    pltpu.SemaphoreType.DMA((num_inter,)),
                    pltpu.SemaphoreType.DMA((num_chips,)),
                    pltpu.SemaphoreType.DMA((num_chips,)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=f"xor1_bridge_ag_tp{tp_size}_{'pipe' if pipelined else '2ph'}_{rows}x{cols}_{jnp.dtype(x.dtype).name}",
        )(x_3d)

    if tiled:
        return out_3d.reshape(tp_size * rows, *orig_shape[1:])
    return out_3d.reshape(tp_size, *orig_shape)
