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
    for ph in emit_xor1_ag_phases(
        x_hbm,
        out_hbm,
        local_sem,
        inter_send_sems,
        inter_recv_sems,
        d2d_send_sems,
        d2d_recv_sems,
        axis_name=axis_name,
        tp_size=8,
        kind="xy",
        part_rows=(half_rows, half_rows),
    ):
        ph()


def _xyz_row_splits(rows: int, min_row_align: int) -> tuple[int, int, int]:
    """Balanced 3-way split of ``rows`` into ``min_row_align``-aligned positive chunks."""
    n_blocks = rows // min_row_align
    b0 = (n_blocks + 2) // 3
    b1 = (n_blocks + 1) // 3
    b2 = n_blocks // 3
    return b0 * min_row_align, b1 * min_row_align, b2 * min_row_align


def xor1_ag_schedule_spec(
    tp_size: int, rows: int, item_bytes: int, *, xy_torus: bool = True
) -> tuple[str, tuple[int, ...], tuple[int, int, int, int, int]]:
    """Return ``(schedule_kind, part_rows, (local, inter_s, inter_r, d2d_s, d2d_r))``."""
    min_row_align = 32 if item_bytes == 1 else 8
    if (
        xy_torus
        and tp_size == 8
        and (rows // 2) >= min_row_align
        and (rows % (2 * min_row_align) == 0)
    ):
        return "xy", (rows // 2, rows // 2), (1, 6, 6, 7, 7)
    if xy_torus and tp_size == 16 and rows >= 3 * min_row_align:
        return "xyz", _xyz_row_splits(rows, min_row_align), (1, 21, 21, 22, 22)
    num_chips = tp_size // 2
    num_inter = num_chips - 1
    return "flat", (rows,), (1, num_inter, num_inter, num_chips, num_chips)


def emit_xor1_ag_phases(
    x_hbm,
    out_hbm,
    local_sem,
    inter_send_sems,
    inter_recv_sems,
    d2d_send_sems,
    d2d_recv_sems,
    *,
    axis_name: str,
    tp_size: int,
    kind: str,
    part_rows: tuple[int, ...],
):
    """Return zero-arg phase callables so standalone or fused Pallas kernels share one schedule.

    Each returned callable lazily creates its descriptors when invoked so phases can be placed
    in separate ``@pl.when`` blocks across grid steps (overlapped with compute).
    """
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def _me_and_mate():
        me = lax.axis_index(axis_name).astype(jnp.int32)
        mate = lax.bitwise_xor(me, jnp.int32(1))
        return me, mate

    def _copy_self_and_own(me, mate):
        b = pltpu.get_barrier_semaphore()
        for off in range(1, tp_size):
            pl.semaphore_signal(
                b,
                1,
                device_id={axis_name: lax.rem(me + off, tp_size)},
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(b, tp_size - 1)
        pltpu.make_async_copy(
            src_ref=x_hbm,
            dst_ref=out_hbm.at[pl.ds(me, 1)],
            sem=local_sem.at[0],
        ).start()
        pltpu.make_async_remote_copy(
            src_ref=x_hbm,
            dst_ref=out_hbm.at[pl.ds(me, 1)],
            send_sem=d2d_send_sems.at[0],
            recv_sem=d2d_recv_sems.at[0],
            device_id={axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

    def _wait_self_and_own(me, mate):
        pltpu.make_async_copy(
            src_ref=x_hbm,
            dst_ref=out_hbm.at[pl.ds(me, 1)],
            sem=local_sem.at[0],
        ).wait()
        c_own = pltpu.make_async_remote_copy(
            src_ref=x_hbm,
            dst_ref=out_hbm.at[pl.ds(me, 1)],
            send_sem=d2d_send_sems.at[0],
            recv_sem=d2d_recv_sems.at[0],
            device_id={axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )
        c_own.wait_send()
        c_own.wait_recv()

    def _slice_inter(sem_idx: int, me, owner_rank, h_sl, dst_peer):
        src = (
            x_hbm.at[:, h_sl, :]
            if owner_rank is None
            else out_hbm.at[pl.ds(owner_rank, 1), h_sl, :]
        )
        dst_slot = me if owner_rank is None else owner_rank
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=out_hbm.at[pl.ds(dst_slot, 1), h_sl, :],
            send_sem=inter_send_sems.at[sem_idx],
            recv_sem=inter_recv_sems.at[sem_idx],
            device_id={axis_name: dst_peer},
            device_id_type=pl.DeviceIdType.MESH,
        )

    def _slice_d2d(sem_idx: int, mate, owner_rank, h_sl):
        return pltpu.make_async_remote_copy(
            src_ref=out_hbm.at[pl.ds(owner_rank, 1), h_sl, :],
            dst_ref=out_hbm.at[pl.ds(owner_rank, 1), h_sl, :],
            send_sem=d2d_send_sems.at[sem_idx],
            recv_sem=d2d_recv_sems.at[sem_idx],
            device_id={axis_name: mate},
            device_id_type=pl.DeviceIdType.MESH,
        )

    if kind == "xy":
        h = part_rows[0]
        h0, h1 = pl.ds(0, h), pl.ds(h, h)

        def _xy_peers():
            me, mate = _me_and_mate()
            return (
                me,
                mate,
                lax.bitwise_xor(me, jnp.int32(2)),
                lax.bitwise_xor(me, jnp.int32(4)),
                lax.bitwise_xor(me, jnp.int32(6)),
            )

        def _xy_s1_ops(me, px, py):
            return (
                _slice_inter(0, me, None, h0, px),
                _slice_inter(1, me, None, h1, py),
            )

        def _xy_s2_inter(me, px, py):
            return (
                _slice_inter(2, me, None, h0, py),
                _slice_inter(3, me, px, h0, py),
                _slice_inter(4, me, None, h1, px),
                _slice_inter(5, me, py, h1, px),
            )

        def _xy_s1_d2d(mate, px, py):
            return (
                _slice_d2d(1, mate, px, h0),
                _slice_d2d(2, mate, py, h1),
            )

        def _xy_s3_ops(mate, px, py, pxy):
            return tuple(
                _slice_d2d(3 + idx, mate, peer_s, h_sl)
                for idx, (peer_s, h_sl) in enumerate(
                    ((py, h0), (pxy, h0), (px, h1), (pxy, h1))
                )
            )

        def _xy_p1():
            me, mate, px, py, _ = _xy_peers()
            _copy_self_and_own(me, mate)
            for c in _xy_s1_ops(me, px, py):
                c.start()

        def _xy_p2():
            me, mate, px, py, _ = _xy_peers()
            for c in _xy_s1_ops(me, px, py):
                c.wait_recv()
            for c in (*_xy_s2_inter(me, px, py), *_xy_s1_d2d(mate, px, py)):
                c.start()

        def _xy_p3():
            me, mate, px, py, pxy = _xy_peers()
            for c in _xy_s2_inter(me, px, py):
                c.wait_recv()
            for cd in _xy_s3_ops(mate, px, py, pxy):
                cd.start()

        def _xy_drain():
            me, mate, px, py, pxy = _xy_peers()
            _wait_self_and_own(me, mate)
            for c in (*_xy_s1_ops(me, px, py), *_xy_s2_inter(me, px, py)):
                c.wait_send()
            for cd in (*_xy_s1_d2d(mate, px, py), *_xy_s3_ops(mate, px, py, pxy)):
                cd.wait_send()
                cd.wait_recv()

        return (_xy_p1, _xy_p2, _xy_p3, _xy_drain)

    if kind == "xyz":
        r0, r1, r2 = part_rows
        parts = (pl.ds(0, r0), pl.ds(r0, r1), pl.ds(r0 + r1, r2))

        def _xyz_peers():
            me, mate = _me_and_mate()
            return (
                me,
                mate,
                lax.bitwise_xor(me, jnp.int32(2)),
                lax.bitwise_xor(me, jnp.int32(4)),
                lax.bitwise_xor(me, jnp.int32(8)),
                lax.bitwise_xor(me, jnp.int32(6)),
                lax.bitwise_xor(me, jnp.int32(10)),
                lax.bitwise_xor(me, jnp.int32(12)),
                lax.bitwise_xor(me, jnp.int32(14)),
            )

        def _xyz_s1_ops(me, px, py, pz):
            return (
                _slice_inter(0, me, None, parts[0], px),
                _slice_inter(1, me, None, parts[1], py),
                _slice_inter(2, me, None, parts[2], pz),
            )

        def _xyz_s2_inter(me, px, py, pz):
            inter = []
            sem_i = 3
            for h_sl, dst_peer, owners in (
                (parts[0], py, (None, px)),
                (parts[1], pz, (None, py)),
                (parts[2], px, (None, pz)),
            ):
                for owner in owners:
                    inter.append(_slice_inter(sem_i, me, owner, h_sl, dst_peer))
                    sem_i += 1
            return tuple(inter)

        def _xyz_s1_d2d(mate, px, py, pz):
            return tuple(
                _slice_d2d(1 + idx, mate, peer, parts[idx])
                for idx, peer in enumerate((px, py, pz))
            )

        def _xyz_s3_inter(me, px, py, pz, pxy, pxz, pyz):
            inter = []
            sem_i = 9
            for h_sl, dst_peer, owners in (
                (parts[0], pz, (None, px, py, pxy)),
                (parts[1], px, (None, py, pz, pyz)),
                (parts[2], py, (None, pz, px, pxz)),
            ):
                for owner in owners:
                    inter.append(_slice_inter(sem_i, me, owner, h_sl, dst_peer))
                    sem_i += 1
            return tuple(inter)

        def _xyz_s2_d2d(mate, px, py, pz, pxy, pxz, pyz):
            return tuple(
                _slice_d2d(4 + idx, mate, peer, h_sl)
                for idx, (peer, h_sl) in enumerate(
                    (
                        (py, parts[0]),
                        (pxy, parts[0]),
                        (pz, parts[1]),
                        (pyz, parts[1]),
                        (px, parts[2]),
                        (pxz, parts[2]),
                    )
                )
            )

        def _xyz_s4_ops(mate, px, py, pz, pxy, pxz, pyz, pxyz):
            return tuple(
                _slice_d2d(10 + idx, mate, peer, h_sl)
                for idx, (peer, h_sl) in enumerate(
                    (
                        (pz, parts[0]),
                        (pxz, parts[0]),
                        (pyz, parts[0]),
                        (pxyz, parts[0]),
                        (px, parts[1]),
                        (pxy, parts[1]),
                        (pxz, parts[1]),
                        (pxyz, parts[1]),
                        (py, parts[2]),
                        (pyz, parts[2]),
                        (pxy, parts[2]),
                        (pxyz, parts[2]),
                    )
                )
            )

        def _xyz_p1():
            me, mate, px, py, pz, *_ = _xyz_peers()
            _copy_self_and_own(me, mate)
            for c in _xyz_s1_ops(me, px, py, pz):
                c.start()

        def _xyz_p2():
            me, mate, px, py, pz, *_ = _xyz_peers()
            for c in _xyz_s1_ops(me, px, py, pz):
                c.wait_recv()
            for c in (*_xyz_s2_inter(me, px, py, pz), *_xyz_s1_d2d(mate, px, py, pz)):
                c.start()

        def _xyz_p3():
            me, mate, px, py, pz, pxy, pxz, pyz, _ = _xyz_peers()
            for c in _xyz_s2_inter(me, px, py, pz):
                c.wait_recv()
            for c in (
                *_xyz_s3_inter(me, px, py, pz, pxy, pxz, pyz),
                *_xyz_s2_d2d(mate, px, py, pz, pxy, pxz, pyz),
            ):
                c.start()

        def _xyz_p4():
            me, mate, px, py, pz, pxy, pxz, pyz, pxyz = _xyz_peers()
            for c in _xyz_s3_inter(me, px, py, pz, pxy, pxz, pyz):
                c.wait_recv()
            for cd in _xyz_s4_ops(mate, px, py, pz, pxy, pxz, pyz, pxyz):
                cd.start()

        def _xyz_drain():
            me, mate, px, py, pz, pxy, pxz, pyz, pxyz = _xyz_peers()
            _wait_self_and_own(me, mate)
            for c in (
                *_xyz_s1_ops(me, px, py, pz),
                *_xyz_s2_inter(me, px, py, pz),
                *_xyz_s3_inter(me, px, py, pz, pxy, pxz, pyz),
            ):
                c.wait_send()
            for cd in (
                *_xyz_s1_d2d(mate, px, py, pz),
                *_xyz_s2_d2d(mate, px, py, pz, pxy, pxz, pyz),
                *_xyz_s4_ops(mate, px, py, pz, pxy, pxz, pyz, pxyz),
            ):
                cd.wait_send()
                cd.wait_recv()

        return (_xyz_p1, _xyz_p2, _xyz_p3, _xyz_p4, _xyz_drain)

    num_chips = tp_size // 2
    num_inter = num_chips - 1
    use_xor_masks = (num_chips & (num_chips - 1)) == 0

    def _flat_inter(me):
        inter = []
        for k in range(num_inter):
            if use_xor_masks:
                dst_peer = lax.bitwise_xor(me, jnp.int32((k + 1) * 2))
            else:
                step = (k + 1) * 2
                dst_peer = lax.rem(me + step, tp_size)
            inter.append(
                pltpu.make_async_remote_copy(
                    src_ref=x_hbm,
                    dst_ref=out_hbm.at[pl.ds(me, 1)],
                    send_sem=inter_send_sems.at[k],
                    recv_sem=inter_recv_sems.at[k],
                    device_id={axis_name: dst_peer},
                    device_id_type=pl.DeviceIdType.MESH,
                )
            )
        return tuple(inter)

    def _flat_d2d(me, mate):
        d2d = []
        for k in range(num_inter):
            if use_xor_masks:
                src_peer = lax.bitwise_xor(me, jnp.int32((k + 1) * 2))
            else:
                step = (k + 1) * 2
                src_peer = lax.rem(me + tp_size - step, tp_size)
            d2d.append(
                pltpu.make_async_remote_copy(
                    src_ref=out_hbm.at[pl.ds(src_peer, 1)],
                    dst_ref=out_hbm.at[pl.ds(src_peer, 1)],
                    send_sem=d2d_send_sems.at[k + 1],
                    recv_sem=d2d_recv_sems.at[k + 1],
                    device_id={axis_name: mate},
                    device_id_type=pl.DeviceIdType.MESH,
                )
            )
        return tuple(d2d)

    def _flat_p1():
        me, mate = _me_and_mate()
        _copy_self_and_own(me, mate)
        for c in _flat_inter(me):
            c.start()

    def _flat_p2():
        me, mate = _me_and_mate()
        for c in _flat_inter(me):
            c.wait_recv()
        for cd in _flat_d2d(me, mate):
            cd.start()

    def _flat_drain():
        me, mate = _me_and_mate()
        _wait_self_and_own(me, mate)
        for c in _flat_inter(me):
            c.wait_send()
        for cd in _flat_d2d(me, mate):
            cd.wait_send()
            cd.wait_recv()

    return (_flat_p1, _flat_p2, _flat_drain)


def _xor1_xyz_torus_ag_kernel(
    x_hbm,
    out_hbm,
    local_sem,
    inter_send_sems,
    inter_recv_sems,
    d2d_send_sems,
    d2d_recv_sems,
    *,
    axis_name: str,
    r0: int,
    r1: int,
    r2: int,
) -> None:
    """Tri-port 1-hop XYZ-torus + XOR=1 intra-chip D2D bridge for world_size=16 (2x2x2)."""
    for ph in emit_xor1_ag_phases(
        x_hbm,
        out_hbm,
        local_sem,
        inter_send_sems,
        inter_recv_sems,
        d2d_send_sems,
        d2d_recv_sems,
        axis_name=axis_name,
        tp_size=16,
        kind="xyz",
        part_rows=(r0, r1, r2),
    ):
        ph()


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
    elif xy_torus and tp_size == 16 and rows >= 3 * min_row_align:
        r0, r1, r2 = _xyz_row_splits(rows, min_row_align)
        kernel = functools.partial(
            _xor1_xyz_torus_ag_kernel,
            axis_name=axis_name,
            r0=r0,
            r1=r1,
            r2=r2,
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
                    pltpu.SemaphoreType.DMA((21,)),
                    pltpu.SemaphoreType.DMA((21,)),
                    pltpu.SemaphoreType.DMA((22,)),
                    pltpu.SemaphoreType.DMA((22,)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                disable_bounds_checks=True,
            ),
            name=f"xor1_xyztorus_ag_tp{tp_size}_{rows}x{cols}_{jnp.dtype(x.dtype).name}",
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
