# SPDX-License-Identifier: Apache-2.0
"""Tiered one-shot collectives on remote DMA, called from inside a kernel.

Every level is a one-shot: each rank pushes its block to all peers of the
level, then waits for the same number of bytes from them. A reduction runs the
in-host level (lanes) first and the cross-host level (hosts) second, so a
TP32 all-reduce costs two dependent round trips instead of one 31-peer fan-out
(flat fan-outs are egress-bound: ~44 GB/s per core).

Sums are taken in f32 in a fixed peer order, so every rank ends with
bit-identical values. The transport dtype is f32 (exact) unless the caller
opts into bf16 partial sums.

Buffer reuse: a round uses receive slot ``round % 2``. A peer can be at most
one round ahead -- finishing round k+1 needs this rank's round-k+1 block,
which is only sent after round k has been consumed -- so two slots suffice.
The round counter lives in SMEM and is reset by ``init`` at kernel entry;
every rank runs the same sequence of collectives.
"""

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import Topology

LANE = 128
_MESH = pl.DeviceIdType.MESH
_LOCAL, _HOST = 0, 1


def entry_barrier(me, topo: Topology) -> None:
    """Every peer's kernel (and so its VMEM scratch) is live before any remote
    write lands. Needs ``CompilerParams(collective_id=...)`` on the call."""
    if topo.world == 1:
        return
    barrier = pltpu.get_barrier_semaphore()
    for off in topo.all_offsets:
        pltpu.semaphore_signal(barrier, 1, device_id=(me ^ off,), device_id_type=_MESH)
    pltpu.semaphore_wait(barrier, topo.world - 1)


def _coords(me, topo: Topology, stride: int = 1):
    """(lane, host) of global rank ``me`` inside ``topo``. A collective may
    run over a sub-topology of the world, e.g. ``Topology(1, 2)`` for the
    chip pair or ``Topology(1, lanes)`` for one host; peers are always
    ``me ^ (stride * offset)``, so ``stride=2`` spans the ranks of one chip-pair
    parity."""
    q = me // stride
    return q % topo.lanes, (q // topo.lanes) % topo.hosts


def _min_rows(dtype) -> int:
    return 8 * (4 // jnp.dtype(dtype).itemsize)


class AllReduceSpec(msgspec.Struct, frozen=True):
    """All-reduce of a ``[rows, 128]`` f32 block (a flat view of the vector). A bf16
    transport pads the block to whole 16-row tiles. ``host_transport`` (default:
    ``transport``) is the wire dtype of the cross-host level.

    ``flat``: one one-shot round over every rank of ``topo`` (``world - 1`` sends) instead of
    the in-host round followed by the cross-host round. The sum still runs lanes-then-hosts,
    so with an f32 wire the bits equal the tiered result; a bf16 wire rounds each partial
    once (the tiered bf16 path also rounds the host partial sums)."""

    topo: Topology
    rows: int
    transport: object = jnp.float32
    stride: int = 1  # peers are me ^ (stride * offset); see _coords
    host_transport: object = None
    flat: bool = False

    def __post_init__(self):
        if self.rows % 8:
            raise ValueError(f"rows={self.rows} must be a multiple of 8")

    @property
    def wire_host(self):
        return self.transport if self.host_transport is None else self.host_transport

    def _wire(self, dtype) -> int:
        m = _min_rows(dtype)
        return -(-self.rows // m) * m

    @property
    def wire_rows(self) -> int:
        return self._wire(self.transport)

    @property
    def host_rows(self) -> int:
        return self._wire(self.wire_host)

    def scratch_shapes(self) -> list:
        t, r, th, hr = self.transport, self.wire_rows, self.wire_host, self.host_rows
        if self.flat:
            m = _min_rows(t)
            return [
                pltpu.VMEM((r, LANE), t),  # own block in transport dtype
                pltpu.VMEM(
                    (2, self.topo.hosts, self.topo.lanes, r, LANE), t
                ),  # receive slots, rank order
                pltpu.VMEM((1, 1, m, LANE), t),  # unused
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SMEM((1,), jnp.int32),  # round counter
                pltpu.VMEM((m, LANE), t),  # unused
            ]
        return [
            pltpu.VMEM((r, LANE), t),  # own block in transport dtype
            pltpu.VMEM((2, self.topo.lanes, r, LANE), t),  # in-host receive slots
            pltpu.VMEM((2, self.topo.hosts, hr, LANE), th),  # cross-host receive slots
            pltpu.SemaphoreType.DMA((2, 2)),  # send [level, slot]
            pltpu.SemaphoreType.DMA((2, 2)),  # recv [level, slot]
            pltpu.SMEM((1,), jnp.int32),  # round counter
            pltpu.VMEM((hr, LANE), th),  # host partial sum in the cross-host wire dtype
        ]


# Issue order of a level's remote copies: farthest peers (largest XOR offset: other hosts) first.
# Set by ``StackKernel(far_first=...)``; issue order only.
FAR_FIRST = [False]


def _remote_copies(me, src, dst, send, recv, offsets):
    if FAR_FIRST[0]:
        offsets = sorted(offsets, reverse=True)
    return [
        pltpu.make_async_remote_copy(
            src, dst, send, recv, device_id=(me ^ off,), device_id_type=_MESH
        )
        for off in offsets
    ]


# One wait per level and direction (on a view covering all peers' bytes) instead of one
# wait per copy; flipped only by the collectives benchmark.
AGGREGATE_WAITS = [True]


def _wait(copies, send=None, recv=None, views=()) -> None:
    """``copies``: the level's remote copies, or a function building them. Waits until they are sent and the same number of bytes has arrived. With
    ``AGGREGATE_WAITS``, ``views`` (refs whose sizes add up to the copies' total bytes) carry
    one byte-count wait per semaphore: DMA semaphores count bytes, so the peers' copies
    need not be waited on one by one."""
    if AGGREGATE_WAITS[0] and views:
        if isinstance(copies, list):  # already started: mark them consumed
            for cp in copies:
                cp._used = True
        for sem in (send, recv):
            for v in views:
                pltpu.make_async_copy(v, v, sem).wait()
        return
    copies = copies() if callable(copies) else copies
    for cp in copies:
        cp.wait_send()
    for cp in copies:
        cp.wait_recv()


def _peer_views(buf, n: int) -> list:
    """Views of ``buf`` (``[k, ...]`` blocks) covering ``n`` blocks."""
    return [buf.at[pl.ds(0, n)]] if n else []


class AllReduce:
    """Binds an ``AllReduceSpec`` to its scratch refs inside the kernel body.

    ``start`` sends the in-host level and returns at once so the caller can
    overlap work; ``finish`` completes both levels. Calls on one instance must
    alternate start/finish (``__call__`` does both).
    """

    def __init__(self, spec: AllReduceSpec, me, *scratch):
        self.spec, self.me = spec, me
        (
            self.stage,
            self.local_buf,
            self.host_buf,
            self.send,
            self.recv,
            self.counter,
            self.host_stage,
        ) = scratch

    def init(self) -> None:
        self.counter[0] = 0

    def _level_copies(self, level, buf, index, offsets, slot):
        offsets = tuple(self.spec.stride * off for off in offsets)
        src = self.stage if level == _LOCAL else self.host_stage
        return _remote_copies(
            self.me,
            src,
            buf.at[slot, index],
            self.send.at[level, slot],
            self.recv.at[level, slot],
            offsets,
        )

    @staticmethod
    def _sum(buf, slot, n):
        acc = buf[slot, 0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + buf[slot, j].astype(jnp.float32)
        return acc

    def start(self, x: jax.Array) -> None:
        """``x``: this rank's ``[rows, 128]`` f32 partial."""
        topo, t = self.spec.topo, self.spec.transport
        rnd = self.counter[0]
        self.counter[0] = rnd + 1
        slot = rnd % 2
        lane, host = _coords(self.me, topo, self.spec.stride)
        pad = self.spec.wire_rows - self.spec.rows
        if pad:
            x = jnp.concatenate([x, jnp.zeros((pad, LANE), x.dtype)], axis=0)
        self.stage[...] = x.astype(t)
        if self.spec.flat:
            for cp in self._flat_copies(slot):
                cp.start()
            self.local_buf[slot, host, lane] = self.stage[...]
        elif topo.lanes > 1:
            for cp in self._level_copies(
                _LOCAL, self.local_buf, lane, topo.local_offsets, slot
            ):
                cp.start()
            self.local_buf[slot, lane] = self.stage[...]

    def _flat_copies(self, slot):
        lane, host = _coords(self.me, self.spec.topo, self.spec.stride)
        offsets = tuple(self.spec.stride * off for off in self.spec.topo.all_offsets)
        return _remote_copies(
            self.me,
            self.stage,
            self.local_buf.at[slot, host, lane],
            self.send.at[_LOCAL, slot],
            self.recv.at[_LOCAL, slot],
            offsets,
        )

    def _finish_flat(self, slot):
        topo, buf = self.spec.topo, self.local_buf
        views = _peer_views(buf.at[slot], topo.hosts - 1) + _peer_views(
            buf.at[slot, topo.hosts - 1], topo.lanes - 1
        )
        _wait(
            lambda: self._flat_copies(slot),
            self.send.at[_LOCAL, slot],
            self.recv.at[_LOCAL, slot],
            views,
        )
        total = None
        for h in range(topo.hosts):
            acc = buf[slot, h, 0].astype(jnp.float32)
            for j in range(1, topo.lanes):
                acc = acc + buf[slot, h, j].astype(jnp.float32)
            total = acc if total is None else total + acc
        return total[: self.spec.rows]

    def finish(self) -> jax.Array:
        """Returns the f32 sum (identical on every rank)."""
        topo, t = self.spec.topo, self.spec.transport
        slot = (self.counter[0] - 1) % 2
        lane, host = _coords(self.me, topo, self.spec.stride)
        if self.spec.flat:
            return self._finish_flat(slot)
        if topo.lanes > 1:
            _wait(
                lambda: self._level_copies(
                    _LOCAL, self.local_buf, lane, topo.local_offsets, slot
                ),
                self.send.at[_LOCAL, slot],
                self.recv.at[_LOCAL, slot],
                _peer_views(self.local_buf.at[slot], topo.lanes - 1),
            )
            x = self._sum(self.local_buf, slot, topo.lanes)
        else:
            x = self.stage[...].astype(jnp.float32)
        if topo.hosts > 1:
            hr = self.spec.host_rows
            if hr > x.shape[0]:
                x = jnp.concatenate(
                    [x, jnp.zeros((hr - x.shape[0], LANE), x.dtype)], axis=0
                )
            self.host_stage[...] = x[:hr].astype(self.spec.wire_host)
            copies = self._level_copies(
                _HOST, self.host_buf, host, topo.host_offsets, slot
            )
            for cp in copies:
                cp.start()
            self.host_buf[slot, host] = self.host_stage[...]
            _wait(
                copies,
                self.send.at[_HOST, slot],
                self.recv.at[_HOST, slot],
                _peer_views(self.host_buf.at[slot], topo.hosts - 1),
            )
            x = self._sum(self.host_buf, slot, topo.hosts)
        return x[: self.spec.rows]

    def __call__(self, x: jax.Array) -> jax.Array:
        self.start(x)
        return self.finish()


class AllGatherSpec(msgspec.Struct, frozen=True):
    """All-gather of each rank's ``[rows, 128]`` f32 block over ``topo``;
    the result is ``[hosts, lanes, rows, 128]`` in rank order.

    ``transport``: wire dtype; a bf16 wire is exact only for bf16-valued blocks (the
    caller's contract). ``flat``: one one-shot round over the whole world instead of
    the in-host round followed by a host-level forward of the gathered host block
    (one dependent round trip instead of two at ``hosts > 1``; ``world - 1`` sends).

    Row mode (``rows == 1`` or ``by_cols``; flat or one host): each rank's ``[rows, cols]`` block lands
    in its ``cols``-lane slice of one ``[rows, world * cols]`` block, so a message is the block itself
    (256 B for one 128-wide bf16 row) instead of whole flat ``[8 | 16, 128]`` tiles, and the result
    needs no relayout."""

    topo: Topology
    rows: int
    transport: object = jnp.float32
    flat: bool = False
    cols: int = LANE
    stride: int = 1  # peers are me ^ (stride * offset); see _coords
    by_cols: bool = False  # row mode for rows > 1: [rows, cols] blocks side by side -> [rows, world * cols]

    def __post_init__(self):
        if self.row_mode:
            if self.cols % LANE or not (self.flat or self.topo.hosts == 1):
                raise ValueError(
                    f"row mode needs cols % 128 == 0 (cols={self.cols}) and flat or one host"
                )
        elif self.rows % 8 or self.cols != LANE:
            raise ValueError(
                f"rows={self.rows} must be a multiple of 8 (cols={self.cols} must be 128)"
            )

    @property
    def row_mode(self) -> bool:
        return self.rows == 1 or self.by_cols

    @property
    def wire_rows(self) -> int:
        m = _min_rows(self.transport)
        return -(-self.rows // m) * m

    def scratch_shapes(self) -> list:
        t, r, topo = self.transport, self.wire_rows, self.topo
        sems = [
            pltpu.SemaphoreType.DMA(
                (2, 2)
            ),  # send [level, slot] (row mode: [own copy | peers, slot])
            pltpu.SemaphoreType.DMA((2, 2)),  # recv [level, slot]
            pltpu.SMEM((1,), jnp.int32),  # round counter
        ]
        if self.row_mode:
            r = self.rows
            return [
                pltpu.VMEM((r, self.cols), t),  # own block, wire dtype
                pltpu.VMEM((2, r, topo.world * self.cols), t),
            ] + sems  # gathered slots
        local = (
            (1, 1, 8 * 4 // jnp.dtype(t).itemsize, LANE)
            if self.flat
            else (2, topo.lanes, r, LANE)
        )
        return [
            pltpu.VMEM(local, t),  # in-host receive slots (unused when flat)
            pltpu.VMEM(
                (2, topo.hosts, topo.lanes, r, LANE), t
            ),  # cross-host (flat: all) receive slots
        ] + sems


class AllGather:
    """Binds an ``AllGatherSpec`` to its scratch refs. Tiered: the host level forwards
    the whole gathered host block, so it is two dependent round trips; flat: one."""

    def __init__(self, spec: AllGatherSpec, me, *scratch):
        self.spec, self.me = spec, me
        self.local_buf, self.host_buf, self.send, self.recv, self.counter = scratch

    def init(self) -> None:
        self.counter[0] = 0

    def _offs(self, offsets):
        return tuple(self.spec.stride * off for off in offsets)

    def _local_copies(self, slot, lane):
        src = self.local_buf.at[slot, lane]
        return _remote_copies(
            self.me,
            src,
            src,
            self.send.at[_LOCAL, slot],
            self.recv.at[_LOCAL, slot],
            self._offs(self.spec.topo.local_offsets),
        )

    def _flat_copies(self, slot):
        lane, host = _coords(self.me, self.spec.topo, self.spec.stride)
        src = self.host_buf.at[slot, host, lane]
        return _remote_copies(
            self.me,
            src,
            src,
            self.send.at[_HOST, slot],
            self.recv.at[_HOST, slot],
            self._offs(self.spec.topo.all_offsets),
        )

    def _row_slice(self, slot):
        lane, host = _coords(self.me, self.spec.topo, self.spec.stride)
        c = self.spec.cols
        start = pl.multiple_of((host * self.spec.topo.lanes + lane) * c, LANE)
        return self.host_buf.at[slot, :, pl.ds(start, c)]

    def _row_own(self, slot):
        """Local copy of the stage row into this rank's own slice."""
        return pltpu.make_async_copy(
            self.local_buf, self._row_slice(slot), self.send.at[_LOCAL, slot]
        )

    def _row_remote(self, slot):
        """The stage row into this rank's slice on every peer."""
        return _remote_copies(
            self.me,
            self.local_buf,
            self._row_slice(slot),
            self.send.at[_HOST, slot],
            self.recv.at[_HOST, slot],
            self._offs(self.spec.topo.all_offsets),
        )

    def _out(self, blocks):
        return blocks[..., : self.spec.rows, :].astype(jnp.float32)

    def start(self, x: jax.Array) -> None:
        """``x``: this rank's ``[rows, 128]`` (row mode ``[1, cols]``) f32 block."""
        rnd = self.counter[0]
        self.counter[0] = rnd + 1
        slot = rnd % 2
        lane, host = _coords(self.me, self.spec.topo, self.spec.stride)
        if self.spec.row_mode:
            self.local_buf[...] = x.astype(self.spec.transport)
            for cp in self._row_remote(slot):
                cp.start()
            self._row_own(slot).start()
            return
        pad = self.spec.wire_rows - self.spec.rows
        if pad:
            x = jnp.concatenate([x, jnp.zeros((pad, LANE), x.dtype)], axis=0)
        x = x.astype(self.spec.transport)
        if self.spec.flat:
            self.host_buf[slot, host, lane] = x
            for cp in self._flat_copies(slot):
                cp.start()
            return
        self.local_buf[slot, lane] = x
        if self.spec.topo.lanes > 1:
            for cp in self._local_copies(slot, lane):
                cp.start()

    def finish(self) -> jax.Array:
        """Returns ``[hosts, lanes, rows, 128]`` f32 in rank order (row mode: ``[1, world * cols]``)."""
        topo = self.spec.topo
        slot = (self.counter[0] - 1) % 2
        lane, host = _coords(self.me, topo, self.spec.stride)
        if self.spec.row_mode:
            self._row_own(slot).wait()
            n = (topo.world - 1) * self.spec.cols
            views = [self.host_buf.at[slot, :, pl.ds(0, n)]] if n else []
            _wait(
                lambda: self._row_remote(slot),
                self.send.at[_HOST, slot],
                self.recv.at[_HOST, slot],
                views,
            )
            return self.host_buf[slot].astype(jnp.float32)
        if self.spec.flat:
            views = _peer_views(self.host_buf.at[slot], topo.hosts - 1)
            views += _peer_views(self.host_buf.at[slot, topo.hosts - 1], topo.lanes - 1)
            _wait(
                lambda: self._flat_copies(slot),
                self.send.at[_HOST, slot],
                self.recv.at[_HOST, slot],
                views,
            )
            return self._out(self.host_buf[slot])
        if topo.lanes > 1:
            _wait(
                lambda: self._local_copies(slot, lane),
                self.send.at[_LOCAL, slot],
                self.recv.at[_LOCAL, slot],
                _peer_views(self.local_buf.at[slot], topo.lanes - 1),
            )
        if topo.hosts == 1:
            return self._out(self.local_buf[slot][None])
        self.host_buf[slot, host] = self.local_buf[slot]
        src = self.host_buf.at[slot, host]
        copies = _remote_copies(
            self.me,
            src,
            src,
            self.send.at[_HOST, slot],
            self.recv.at[_HOST, slot],
            self._offs(topo.host_offsets),
        )
        for cp in copies:
            cp.start()
        _wait(
            copies,
            self.send.at[_HOST, slot],
            self.recv.at[_HOST, slot],
            _peer_views(self.host_buf.at[slot], topo.hosts - 1),
        )
        return self._out(self.host_buf[slot])

    def __call__(self, x: jax.Array) -> jax.Array:
        self.start(x)
        return self.finish()


class PairExchangeSpec(msgspec.Struct, frozen=True):
    """Swap of a ``[rows, 128]`` block with the other core of the chip (``me ^ 1``).

    One receive slot suffices when every exchange round is followed by another
    exchange (or reduction) that needs this rank's next contribution before the
    peer can send again; callers keep that ordering (see ``PairProjection``).
    """

    rows: int
    transport: object = jnp.bfloat16

    @property
    def wire_rows(self) -> int:
        m = _min_rows(self.transport)
        return -(-self.rows // m) * m

    def scratch_shapes(self) -> list:
        return [
            pltpu.VMEM((2, self.wire_rows, LANE), self.transport),  # [own, peer's]
            pltpu.SemaphoreType.DMA((2,)),  # send, recv
        ]


class PairExchange:
    def __init__(self, spec: PairExchangeSpec, me, buf, sems):
        self.spec, self.me, self.buf, self.sems = spec, me, buf, sems

    def init(self) -> None:
        pass

    def _copy(self):
        return pltpu.make_async_remote_copy(
            self.buf.at[0],
            self.buf.at[1],
            self.sems.at[0],
            self.sems.at[1],
            device_id=(self.me ^ 1,),
            device_id_type=_MESH,
        )

    def start(self, x: jax.Array) -> None:
        """``x``: this rank's ``[rows, 128]`` f32 block (exact in the transport dtype)."""
        pad = self.spec.wire_rows - self.spec.rows
        if pad:
            x = jnp.concatenate([x, jnp.zeros((pad, LANE), x.dtype)], axis=0)
        self.buf[0] = x.astype(self.spec.transport)
        self._copy().start()

    def finish(self) -> jax.Array:
        """Returns the peer's ``[rows, 128]`` block as f32."""
        cp = self._copy()
        cp.wait_send()
        cp.wait_recv()
        return self.buf[1, : self.spec.rows].astype(jnp.float32)


class ReducePart(msgspec.Struct, frozen=True):
    """One block of a ``ReduceSpec``. ``rows``: rows of this rank's result. ``scatter``:
    the in-host level is a reduce-scatter -- the input is ``[lanes, rows, 128]`` (chunk
    ``j`` belongs to lane ``j``) and each lane keeps the sum of its own chunk; otherwise
    the input and the result are ``[rows, 128]`` (an all-reduce)."""

    rows: int
    scatter: bool = False


class ReduceSpec(msgspec.Struct, frozen=True):
    """Tiered reduction of several independent f32 blocks whose cross-host levels
    share one round.

    Every part's in-host level starts on its own (``start(i, x)``), as early as its
    partial exists; ``finish`` completes the in-host levels and sends all host
    partial sums in one cross-host one-shot. Per element the sums run in the same
    fixed order as ``AllReduce`` (lanes, then hosts), so f32 results are the same
    bits as one ``AllReduce`` per part. ``host_transport`` (default: ``transport``)
    is the wire dtype of the cross-host level."""

    topo: Topology
    parts: tuple[ReducePart, ...]
    transport: object = jnp.float32
    host_transport: object = None
    stride: int = 1  # peers are me ^ (stride * offset); see _coords

    def __post_init__(self):
        for p in self.parts:
            if p.rows % 8:
                raise ValueError(f"rows={p.rows} must be a multiple of 8")

    @property
    def wire_host(self):
        return self.transport if self.host_transport is None else self.host_transport

    def wire_rows(self, i: int, dtype=None) -> int:
        m = _min_rows(self.transport if dtype is None else dtype)
        return -(-self.parts[i].rows // m) * m

    def host_offsets(self) -> list[int]:
        """Row offset of each part in the cross-host block (plus the total)."""
        offs = [0]
        for i in range(len(self.parts)):
            offs.append(offs[-1] + self.wire_rows(i, self.wire_host))
        return offs

    def scratch_shapes(self) -> list:
        t, topo, n = self.transport, self.topo, len(self.parts)
        shapes = []
        for i, p in enumerate(self.parts):
            r = self.wire_rows(i)
            shapes += [
                pltpu.VMEM(
                    (topo.lanes if p.scatter else 1, r, LANE), t
                ),  # own block(s), wire dtype
                pltpu.VMEM((2, topo.lanes, r, LANE), t),  # in-host receive slots
            ]
        hr = self.host_offsets()[-1]
        shapes += [
            pltpu.VMEM((hr, LANE), self.wire_host),  # host partial sums, wire dtype
            pltpu.VMEM(
                (2, topo.hosts, hr, LANE), self.wire_host
            ),  # cross-host receive slots
            pltpu.SemaphoreType.DMA((n + 1, 2)),  # send [part | host level, slot]
            pltpu.SemaphoreType.DMA((n + 1, 2)),  # recv [part | host level, slot]
            pltpu.SMEM((1,), jnp.int32),  # round counter
        ]
        return shapes


class Reduce:
    """Binds a ``ReduceSpec`` to its scratch refs. Each round starts every part once
    (any order) and then calls ``finish``; a round uses receive slot ``round % 2`` (see
    the module docstring: finishing round k+1 needs every peer's round-k+1 blocks)."""

    def __init__(self, spec: ReduceSpec, me, *scratch):
        self.spec, self.me = spec, me
        n = len(spec.parts)
        self.stage = scratch[0 : 2 * n : 2]
        self.local_buf = scratch[1 : 2 * n : 2]
        self.host_stage, self.host_buf, self.send, self.recv, self.counter = scratch[
            2 * n :
        ]

    def init(self) -> None:
        self.counter[0] = 0

    def _coords(self):
        return _coords(self.me, self.spec.topo, self.spec.stride)

    def _local_copies(self, i, slot):
        """Part ``i``'s in-host sends: peer ``me ^ off`` gets block ``lane ^ off`` (scatter)
        or the whole block, into its slot row ``lane``."""
        lane, _ = self._coords()
        scatter = self.spec.parts[i].scatter
        copies = []
        for off in self.spec.topo.local_offsets:
            src = self.stage[i].at[(lane ^ off) if scatter else 0]
            copies.append(
                pltpu.make_async_remote_copy(
                    src,
                    self.local_buf[i].at[slot, lane],
                    self.send.at[i, slot],
                    self.recv.at[i, slot],
                    device_id=(self.me ^ (self.spec.stride * off),),
                    device_id_type=_MESH,
                )
            )
        return copies

    def start(self, i: int, x: jax.Array) -> None:
        """``x``: part ``i``'s f32 partial, ``[lanes, rows, 128]`` (scatter) or ``[rows, 128]``."""
        spec, topo = self.spec, self.spec.topo
        part, t = spec.parts[i], spec.transport
        slot = self.counter[0] % 2
        lane, _ = self._coords()
        x = x if part.scatter else x[None]
        pad = spec.wire_rows(i) - part.rows
        if pad:
            x = jnp.concatenate(
                [x, jnp.zeros((x.shape[0], pad, LANE), x.dtype)], axis=1
            )
        self.stage[i][...] = x.astype(t)
        if topo.lanes > 1:
            for cp in self._local_copies(i, slot):
                cp.start()
            self.local_buf[i][slot, lane] = self.stage[i][lane if part.scatter else 0]

    def _local_sum(self, i, slot):
        topo, part = self.spec.topo, self.spec.parts[i]
        if topo.lanes == 1:
            return self.stage[i][0].astype(jnp.float32)
        _wait(
            lambda: self._local_copies(i, slot),
            self.send.at[i, slot],
            self.recv.at[i, slot],
            _peer_views(self.local_buf[i].at[slot], topo.lanes - 1),
        )
        return AllReduce._sum(self.local_buf[i], slot, topo.lanes)

    def finish(self) -> list[jax.Array]:
        """Returns each part's f32 result (identical on every rank of a scatter lane / of the world)."""
        spec, topo = self.spec, self.spec.topo
        rnd = self.counter[0]
        slot = rnd % 2
        sums = [self._local_sum(i, slot) for i in range(len(spec.parts))]
        if topo.hosts > 1:
            _, host = self._coords()
            offs, th = spec.host_offsets(), spec.wire_host
            for i, s in enumerate(sums):
                rows = offs[i + 1] - offs[i]
                if rows > s.shape[0]:
                    s = jnp.concatenate(
                        [s, jnp.zeros((rows - s.shape[0], LANE), s.dtype)], axis=0
                    )
                self.host_stage[offs[i] : offs[i + 1]] = s[:rows].astype(th)
            n = len(spec.parts)
            copies = _remote_copies(
                self.me,
                self.host_stage,
                self.host_buf.at[slot, host],
                self.send.at[n, slot],
                self.recv.at[n, slot],
                tuple(spec.stride * o for o in topo.host_offsets),
            )
            for cp in copies:
                cp.start()
            self.host_buf[slot, host] = self.host_stage[...]
            _wait(
                copies,
                self.send.at[n, slot],
                self.recv.at[n, slot],
                _peer_views(self.host_buf.at[slot], topo.hosts - 1),
            )
            total = AllReduce._sum(self.host_buf, slot, topo.hosts)
            sums = [total[offs[i] : offs[i + 1]] for i in range(n)]
        self.counter[0] = rnd + 1
        return [s[: p.rows] for s, p in zip(sums, spec.parts)]


class RowScatterSpec(msgspec.Struct, frozen=True):
    """All-reduce of an ``[R, H]`` f32 partial (flat ``[R * H / 128, 128]``) whose result is rounded
    to bf16, scattered by row: lane j sums its ``R / lanes`` rows (in-host reduce-scatter, then the
    cross-host round on those rows), and the lanes all-gather the bf16 rows. Per element the sums
    run lanes-then-hosts like ``AllReduce``: the same bits, at ``1 / lanes`` of the reduce bytes."""

    topo: Topology
    rows: int
    hidden: int
    transport: object = jnp.float32
    host_transport: object = None
    stride: int = 1  # over one chip-pair parity (pair_oproj): 2
    # stride 2: the bf16 rows are gathered from all lanes of the host (both parities), so every
    # rank ends with the full [R, 2 * hidden] block and the chip pair's halves swap goes away.
    full_host: bool = False

    def __post_init__(self):
        if self.full_host and self.stride != 2:
            raise ValueError("full_host gathers both parities: stride must be 2")
        if self.rows % self.topo.lanes or self.hidden % 128:
            raise ValueError(
                f"rows={self.rows} must split over {self.topo.lanes} lanes"
            )

    @property
    def per_lane(self) -> int:
        return self.rows // self.topo.lanes

    @property
    def chunk_rows(self) -> int:
        return self.per_lane * self.hidden // 128

    @property
    def reduce(self) -> "ReduceSpec":
        return ReduceSpec(
            self.topo,
            (ReducePart(-(-self.chunk_rows // 8) * 8, scatter=True),),
            self.transport,
            self.host_transport,
            self.stride,
        )

    @property
    def gather(self) -> "AllGatherSpec":
        if self.full_host:
            return AllGatherSpec(
                Topology(1, 2 * self.topo.lanes),
                1,
                jnp.bfloat16,
                flat=True,
                cols=self.per_lane * self.hidden,
            )
        return AllGatherSpec(
            Topology(1, self.topo.lanes),
            1,
            jnp.bfloat16,
            flat=True,
            cols=self.per_lane * self.hidden,
            stride=self.stride,
        )

    def scratch_shapes(self) -> list:
        return self.reduce.scratch_shapes() + self.gather.scratch_shapes()


class RowScatterAllReduce:
    """``RowScatterSpec`` bound to its scratch; ``start`` / ``finish`` / ``__call__`` like ``AllReduce``
    (``finish`` returns the flat f32 block holding bf16 values)."""

    def __init__(self, spec: RowScatterSpec, me, *scratch):
        n = len(spec.reduce.scratch_shapes())
        self.spec = spec
        self.red = Reduce(spec.reduce, me, *scratch[:n])
        self.gat = AllGather(spec.gather, me, *scratch[n:])

    def init(self) -> None:
        self.red.init()
        self.gat.init()

    def start(self, x: jax.Array) -> None:
        sp = self.spec
        c, wire = sp.chunk_rows, sp.reduce.parts[0].rows
        chunks = [x[j * c : (j + 1) * c] for j in range(sp.topo.lanes)]
        if wire > c:
            chunks = [
                jnp.concatenate([ch, jnp.zeros((wire - c, LANE), x.dtype)], axis=0)
                for ch in chunks
            ]
        self.red.start(0, jnp.stack(chunks))

    def finish(self) -> jax.Array:
        sp = self.spec
        own = self.red.finish()[0][: sp.chunk_rows]  # this lane's rows, flat
        row = (
            own.reshape(1, sp.chunk_rows * LANE)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        # [1, R * H]; full_host: [1, 2 * R * H] in host-lane order
        every = self.gat(row)
        if not sp.full_host:
            return every.reshape(sp.rows * sp.hidden // LANE, LANE)
        # Host lane 2j + p holds parity p's half of rows [j * per_lane, (j + 1) * per_lane).
        blocks = every.reshape(2 * sp.rows, sp.hidden)
        out = []
        for r in range(sp.rows):
            j, i = divmod(r, sp.per_lane)
            b = [(2 * j + p) * sp.per_lane + i for p in range(2)]
            out.append(jnp.concatenate([blocks[k : k + 1] for k in b], axis=1))
        return jnp.concatenate(out, axis=0).reshape(
            2 * sp.rows * sp.hidden // LANE, LANE
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        self.start(x)
        return self.finish()
