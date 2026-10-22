# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the megakernel stages: flat block packing for the
collectives, bf16 rounding boundaries and HBM-streamed matmuls."""

import contextlib

import jax
import jax.numpy as jnp
import msgspec
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.k3_megakernel.config import Relax, Topology
from sglang_tpu.k3_megakernel.kernel.collectives import (
    AllReduce,
    AllReduceSpec,
    PairExchange,
    PairExchangeSpec,
    RowScatterAllReduce,
    RowScatterSpec,
)

DN = (((1,), (0,)), ((), ()))

# Set while a kernel body is traced with phase markers on (benchmark-only); Mosaic lowers
# named scopes to in-kernel trace_start / trace_stop events.
_TRACE_PHASES = [False]


def phase(name: str):
    return jax.named_scope(name) if _TRACE_PHASES[0] else contextlib.nullcontext()


class trace_phases:
    """Context manager enabling ``phase`` markers while a kernel is traced."""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.prev = _TRACE_PHASES[0]
        _TRACE_PHASES[0] = self.on

    def __exit__(self, *exc):
        _TRACE_PHASES[0] = self.prev


def tree_when(idx, n: int, fn) -> None:
    """``fn(i)`` for the static ``i`` equal to the traced ``idx`` (0 <= idx < n), through a binary
    tree of ``pl.when`` (about 2 log2(n) scalar tests instead of n; one branch per per-layer pool)."""

    def rec(lo, hi):
        if hi - lo == 1:
            fn(lo)
            return
        mid = (lo + hi) // 2
        pl.when(idx < mid)(lambda: rec(lo, mid))
        pl.when(idx >= mid)(lambda: rec(mid, hi))

    rec(0, n)


def flat_rows(rows: int, cols: int) -> int:
    n = rows * cols // 128
    return -(-n // 8) * 8


def flat(x: jax.Array) -> jax.Array:
    """[R, C] f32 -> [ceil8(R*C/128), 128] (the collectives' block layout)."""
    rows, cols = x.shape
    f = x.reshape(rows * cols // 128, 128)
    pad = flat_rows(rows, cols) - f.shape[0]
    if pad:
        f = jnp.concatenate([f, jnp.zeros((pad, 128), f.dtype)], axis=0)
    return f


def unflat(f: jax.Array, rows: int, cols: int) -> jax.Array:
    return f[: rows * cols // 128].reshape(rows, cols)


def bf16(x: jax.Array) -> jax.Array:
    """Round f32 to bf16 and back (a bf16 tensor boundary in the reference)."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


class Streamed:
    """``x @ W`` with ``W`` streamed from HBM in K-chunks through a ring of VMEM slots.

    ``buf`` is ``[slots, >= chunk, N]``; a shared buffer may be larger than this use's
    chunk. Chunk ``i`` lands in slot ``i % slots``; up to ``slots - 1`` chunks are in
    flight ahead of the one being consumed. ``prefetched`` chunks (at most ``slots``)
    may have been started earlier, e.g. by a cross-layer prefetch on an equivalent
    descriptor.

    An e4m3 ``W`` comes with ``scale``: ``(hbm [1, N] f32, vmem [1, N] f32, DMA sem [1])``,
    the per-output-column scale applied to the f32 result (``Relax.dense_fp8``). Its copy
    starts together with chunk 0 (so a cross-layer prefetch of chunk 0 also covers it).
    """

    def __init__(
        self,
        w_hbm,
        buf,
        sems,
        chunk: int | None = None,
        prefetched: int = 0,
        name: str = "w",
        scale=None,
    ):
        self.w, self.buf, self.sems, self.name, self.scale = (
            w_hbm,
            buf,
            sems,
            name,
            scale,
        )
        self.slots = buf.shape[0]
        self.chunk = buf.shape[1] if chunk is None else chunk
        self.n = w_hbm.shape[0] // self.chunk
        self.pre = min(prefetched, self.n, self.slots)
        assert self.n * self.chunk == w_hbm.shape[0], (w_hbm.shape, self.chunk)
        assert (scale is not None) == (buf.dtype == jnp.float8_e4m3fn), (
            buf.dtype,
            scale is None,
        )

    def _slot(self, i):
        if self.chunk == self.buf.shape[1]:
            return self.buf.at[i % self.slots]
        return self.buf.at[i % self.slots, pl.ds(0, self.chunk)]

    def _copy(self, i):
        return pltpu.make_async_copy(
            self.w.at[pl.ds(i * self.chunk, self.chunk)],
            self._slot(i),
            self.sems.at[i % self.slots],
        )

    def _scale_copy(self):
        hbm, vmem, sem = self.scale
        return pltpu.make_async_copy(hbm, vmem, sem.at[0])

    def _start(self, i):
        self._copy(i).start()
        if i == 0 and self.scale is not None:
            self._scale_copy().start()

    def prefetch(self, n: int = 1) -> None:
        """Ensure the first ``n`` chunks (clamped to the ring) are started."""
        n = min(n, self.n, self.slots)
        for i in range(self.pre, n):
            self._start(i)
        self.pre = max(self.pre, n)

    def wait_prefetched(self) -> None:
        """Consume the ``prefetched`` chunks without computing (keeps the semaphores balanced)."""
        for i in range(self.pre):
            self._copy(i).wait()
        if self.pre and self.scale is not None:
            self._scale_copy().wait()

    def matmul(self, x, *, prefetched) -> jax.Array:
        """``x``: ``[R, K]`` or a function ``(lo, hi) -> x[:, lo:hi]`` called once per chunk in
        K order, so late parts of ``x`` may still be in flight when the stream starts.
        ``prefetched``: chunks already started (bool or count; the larger of it and the
        constructor's count is used)."""
        pre = max(int(prefetched), self.pre)
        depth = min(self.slots - 1, self.n)
        for i in range(pre, depth):
            self._start(i)
        pre = max(pre, depth)
        acc = None
        for i in range(self.n):
            j = i + self.slots - 1
            if pre <= j < self.n:
                self._copy(j).start()
            with phase("wait_" + self.name):
                self._copy(i).wait()
            lo, hi = i * self.chunk, (i + 1) * self.chunk
            lhs = x(lo, hi) if callable(x) else x[:, lo:hi]
            part = jax.lax.dot_general(
                lhs, self._slot(i)[...], DN, preferred_element_type=jnp.float32
            )
            acc = part if acc is None else acc + part
        if self.scale is not None:
            self._scale_copy().wait()
            acc = acc * self.scale[1][...]
        return acc


def stream_chunk(k: int, n: int, itemsize: int = 2, budget: int = 4 << 20) -> int:
    """Largest K-chunk (a multiple of 128 dividing K) whose slot fits ``budget`` bytes;
    a K with no such divisor (e.g. 192 shared-expert channels at TP32) streams whole."""
    fits = [
        c for c in range(128, k + 1, 128) if k % c == 0 and c * n * itemsize <= budget
    ]
    if fits:
        return max(fits)
    divisors = [c for c in range(128, k + 1, 128) if k % c == 0]
    return min(divisors) if divisors else k


def stream_scratch(k: int, n: int, dtype=jnp.bfloat16) -> list:
    """Scratch of one stand-alone ``Streamed`` use (stage kernels); see ``bind_stream``."""
    c = stream_chunk(k, n, jnp.dtype(dtype).itemsize)
    shapes = [pltpu.VMEM((2, c, n), dtype), pltpu.SemaphoreType.DMA((2,))]
    if dtype == jnp.float8_e4m3fn:
        shapes += [pltpu.VMEM((1, n), jnp.float32), pltpu.SemaphoreType.DMA((1,))]
    return shapes


def bind_stream(w_hbm, refs, scale_hbm=None) -> Streamed:
    """``refs``: the ``stream_scratch`` refs; ``scale_hbm`` is required for e4m3 weights."""
    scale = None if scale_hbm is None else (scale_hbm, refs[2], refs[3])
    return Streamed(w_hbm, refs[0], refs[1], scale=scale)


def build_sharded(
    kernel, *, mesh, in_spaces, out_shapes, out_spaces, scratch, collective_id
):
    """jit(shard_map(pallas_call)) over ``mesh`` axis "x" for a per-rank kernel.

    Every input and output carries a leading rank axis (dropped inside), so
    tests can pass rank-stacked numpy layouts straight through.
    """
    from jax.sharding import PartitionSpec as P

    spaces = {"vmem": pltpu.VMEM, "hbm": pltpu.HBM, "smem": pltpu.SMEM}
    call = pl.pallas_call(
        kernel,
        out_shape=tuple(out_shapes),
        in_specs=[pl.BlockSpec(memory_space=spaces[s]) for s in in_spaces],
        out_specs=tuple(pl.BlockSpec(memory_space=spaces[s]) for s in out_spaces),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024 * 1024
        ),
    )

    def local(*args):
        outs = call(*[a[0] for a in args])
        return tuple(o[None] for o in outs)

    spec = P("x")
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec,) * len(in_spaces),
            out_specs=(spec,) * len(out_shapes),
            check_vma=False,
        )
    )


class StreamPool:
    """Named double buffers for streamed matmuls, shared by uses that never overlap.

    ``uses`` lists ``(key, K, N)``; every key gets one ``[slots, chunk_max, N]``
    buffer and each use streams in its own chunk (a multiple of 128 dividing
    its K) that fits ``budget`` bytes per slot. Uses under one key must not be
    in flight at the same time (prefetch included). An e4m3 pool also holds one
    ``[1, N]`` column-scale buffer per key.
    """

    def __init__(
        self,
        uses,
        *,
        budget: int = 2 << 20,
        dtype=jnp.bfloat16,
        slots: dict | None = None,
        budgets: dict | None = None,
        external=(),
        chunks: dict | None = None,
    ):
        """``slots`` / ``budgets`` override the ring depth (default 2) / per-slot bytes per key;
        ``chunks`` fixes a key's K-chunk rows (the f32 accumulation order follows the chunking);
        ``external`` keys get their buffer from ``bind`` (e.g. an arena view) and only their
        semaphores from ``scratch_shapes``."""
        self.chunks = dict(chunks or {})
        self.dtype, self.budget = dtype, budget
        self.scaled = dtype == jnp.float8_e4m3fn
        self.slots, self.budgets = dict(slots or {}), budgets or {}
        self.external = frozenset(external)
        self.shape: dict[str, tuple[int, int]] = {}
        self.max_chunks: dict[str, int] = {}  # a ring deeper than this never fills
        for key, k, n in uses:
            rows, n0 = self.shape.get(key, (0, n))
            assert n0 == n, f"stream key {key} used with N={n0} and N={n}"
            c = self.chunk(k, n, key)
            self.shape[key] = (max(rows, c), n)
            self.max_chunks[key] = max(self.max_chunks.get(key, 0), k // c)
        self.order = sorted(self.shape)

    def chunk(self, k: int, n: int, key: str) -> int:
        if key in self.chunks:
            return self.chunks[key]
        return stream_chunk(
            k, n, jnp.dtype(self.dtype).itemsize, self.budgets.get(key, self.budget)
        )

    def buf_shape(self, key: str) -> tuple[int, int, int]:
        rows, n = self.shape[key]
        if key not in self.slots:
            return (2, rows, n)
        return (min(self.slots[key], max(1, self.max_chunks[key])), rows, n)

    def scratch_shapes(self) -> list:
        shapes = []
        for key in self.order:
            slots = self.buf_shape(key)[0]
            if key not in self.external:
                shapes.append(pltpu.VMEM(self.buf_shape(key), self.dtype))
            shapes.append(pltpu.SemaphoreType.DMA((slots,)))
            if self.scaled:
                shapes += [
                    pltpu.VMEM((1, self.shape[key][1]), jnp.float32),
                    pltpu.SemaphoreType.DMA((1,)),
                ]
        return shapes

    def bind(self, refs, external: dict | None = None):
        """``refs``: the scratch refs in ``scratch_shapes`` order; ``external``: key -> buffer."""
        it = iter(refs)
        self.bufs = {}
        for key in self.order:
            buf = external[key] if key in self.external else next(it)
            self.bufs[key] = (buf, next(it)) + (
                (next(it), next(it)) if self.scaled else ()
            )
        return self

    def stream(
        self,
        key: str,
        w_ref,
        prefetched: int = 0,
        name: str | None = None,
        scale_ref=None,
    ) -> "Streamed":
        """``scale_ref``: the use's ``[1, N]`` f32 column scales in HBM (e4m3 pools only)."""
        k, n = w_ref.shape[-2], w_ref.shape[-1]
        bufs = self.bufs[key]
        scale = (scale_ref, bufs[2], bufs[3]) if self.scaled else None
        return Streamed(
            w_ref,
            bufs[0],
            bufs[1],
            self.chunk(k, n, key),
            prefetched,
            name or key,
            scale,
        )


class OutProjSpec(msgspec.Struct, frozen=True):
    """Row-parallel output projection ``sum_ranks(x_rank @ W_rank)`` of one decode row,
    ``x_rank`` ``[1, k]`` (this rank's input channels) -> ``[1, hidden]`` bf16-valued.

    Default: ``W_rank`` ``[k, hidden]``, partials all-reduced over the world.
    ``Relax.pair_oproj``: the two cores of a chip swap their features; each projects
    its half of the hidden columns (half = ``rank & 1``) for both cores' channels with
    ``W_rank`` ``[own k; peer k] x [hidden / 2]``, the half partials are all-reduced over
    the ranks of the same parity, and the cores swap the rounded halves. Same weight bytes and
    MXU work; the in-host fan-out halves and the payload of every cross-chip round halves.

    The all-reduce is one flat one-shot over its ranks when they are at most ``FLAT_RANKS``
    (or with ``Relax.oproj_flat``), else in-host then cross-host rounds; with an f32 wire both
    give the same bits (``AllReduceSpec.flat``).
    """

    FLAT_RANKS = 16  # one-shot fan-out limit: a 32-rank flat f32 round is egress-bound (31 x 28 KB)

    topo: Topology
    k: int
    hidden: int
    relax: Relax = Relax()
    rows: int = 1  # decode rows (speculative verify: 8)
    # pair, rows > 1: the row-scattered reduce gathers both parities' rows (RowScatterSpec.full_host),
    # so the halves swap goes away (same bits).
    rows_full_gather: bool = False

    @property
    def pair(self) -> bool:
        return self.relax.pair_oproj

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (2 * self.k, self.hidden // 2) if self.pair else (self.k, self.hidden)

    @property
    def full_spec(self) -> AllReduceSpec:
        """The world-wide all-reduce of a ``[1, hidden]`` partial under this spec's relaxations."""
        rx = self.relax
        return AllReduceSpec(
            self.topo,
            flat_rows(self.rows, self.hidden),
            rx.transport,
            host_transport=rx.host_transport,
            flat=self._flat(self.topo.world),
        )

    def _flat(self, ranks: int) -> bool:
        return self.relax.oproj_flat or ranks <= self.FLAT_RANKS

    def _specs(self):
        t, th = self.relax.transport, self.relax.host_transport
        flat = self._flat(self.topo.world // 2)
        if not self.pair:
            return [self.full_spec]
        assert self.topo.lanes >= 2 and self.hidden % 256 == 0
        half_topo = Topology(self.topo.hosts, self.topo.lanes // 2)
        r, half = self.rows, self.hidden // 2
        if (
            r > 1
        ):  # verify rows: the half partials reduced scattered by row over the parity's lanes
            ar = RowScatterSpec(
                half_topo, r, half, t, th, stride=2, full_host=self.rows_full_gather
            )
        else:
            ar = AllReduceSpec(
                half_topo, flat_rows(1, half), t, stride=2, host_transport=th, flat=flat
            )
        return [
            PairExchangeSpec(flat_rows(r, self.k)),
            ar,
            PairExchangeSpec(flat_rows(r, half)),
        ]

    def scratch_shapes(self) -> list:
        return [x for spec in self._specs() for x in spec.scratch_shapes()]

    def bind(self, me, refs):
        it = iter(refs)
        parts = []
        for spec in self._specs():
            cls = {AllReduceSpec: AllReduce, RowScatterSpec: RowScatterAllReduce}.get(
                type(spec), PairExchange
            )
            parts.append(cls(spec, me, *[next(it) for _ in spec.scratch_shapes()]))
        if not self.pair:
            return AllReduceProjection(self, parts[0])
        return PairProjection(self, me, *parts)


def _call(fn) -> None:
    if fn is not None:
        fn()


class AllReduceProjection:
    def __init__(self, spec: OutProjSpec, ar: AllReduce):
        self.spec, self.ar = spec, ar
        # Called once the reduction's first sends are queued (e.g. to start unrelated DMAs).
        self.before_reduce = None
        self.skip_reduce = False

    def init(self) -> None:
        self.ar.init()

    def __call__(
        self, x: jax.Array, stream: Streamed, *, prefetched: bool = False
    ) -> jax.Array:
        part = stream.matmul(x, prefetched=prefetched)
        if self.skip_reduce:  # benchmark-only: no collective, the local partial
            _call(self.before_reduce)
            return bf16(part)
        self.ar.start(flat(part))
        _call(self.before_reduce)
        return bf16(unflat(self.ar.finish(), self.spec.rows, self.spec.hidden))


class PairProjection:
    def __init__(
        self,
        spec: OutProjSpec,
        me,
        feat: PairExchange,
        ar: AllReduce,
        halves: PairExchange,
    ):
        self.spec, self.me, self.feat, self.ar, self.halves = spec, me, feat, ar, halves
        self.before_reduce = None

    def init(self) -> None:
        for c in (self.feat, self.ar, self.halves):
            c.init()

    def __call__(
        self, x: jax.Array, stream: Streamed, *, prefetched: bool = False
    ) -> jax.Array:
        k, half = self.spec.k, self.spec.hidden // 2
        self.feat.start(flat(x.astype(jnp.float32)))
        both = []

        def lhs(lo, hi):
            if hi <= k:
                return x[:, lo:hi]
            if (
                not both
            ):  # first chunk reaching the peer's rows: its features must have landed
                peer = unflat(self.feat.finish(), x.shape[0], k).astype(x.dtype)
                both.append(jnp.concatenate([x, peer], axis=1))
            return both[0][:, lo:hi]

        part = stream.matmul(lhs, prefetched=prefetched)
        self.ar.start(flat(part))
        _call(self.before_reduce)
        rows = x.shape[0]
        if rows > 1 and self.spec.rows_full_gather:
            return unflat(self.ar.finish(), rows, 2 * half)  # bf16-valued, both halves
        mine = bf16(unflat(self.ar.finish(), rows, half))
        self.halves.start(flat(mine))
        theirs = unflat(self.halves.finish(), rows, half)
        first = self.me % 2 == 0
        return jnp.concatenate(
            [jnp.where(first, mine, theirs), jnp.where(first, theirs, mine)], axis=1
        )
