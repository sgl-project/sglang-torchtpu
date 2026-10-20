"""K3 megakernel tiered all-reduce / fused reduction on one host's 8 TensorCores.

One host's cores are split into fake hosts, (hosts, lanes) in {(1,8), (2,4),
(4,2)}, so both levels run. Rounds feed back into each other to exercise the
two-slot buffer reuse; the f32 transport must match a sequential f32 sum in
the kernel's fixed peer order bit for bit.
"""

import os
import time
import unittest

import numpy as np

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = []

_ROUNDS = 6


def _build(topo, rows, transport, rounds, flat=False, stride=1, host_transport=None):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.config import Topology
    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllReduce,
        AllReduceSpec,
        entry_barrier,
    )

    spec = AllReduceSpec(
        topo, rows, transport, stride=stride, flat=flat, host_transport=host_transport
    )
    world = Topology(1, topo.world * stride)

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index("x")
        ar = AllReduce(spec, me, *scratch)
        ar.init()
        entry_barrier(me, world)

        def body(i, v):
            # Round i reduces (previous sum / world + input * (i + 1)).
            return ar(v * (1.0 / topo.world) + x_ref[...] * (i + 1).astype(jnp.float32))

        o_ref[...] = jax.lax.fori_loop(
            0, rounds, body, jnp.zeros((rows, 128), jnp.float32)
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[: world.world]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


def _build_reduce(topo, parts, transport, host_transport, rounds, mode="fused"):
    """Rounds of a two-part reduction (routed-like all-reduce + shared-like
    reduce-scatter), each round fed by the previous results. ``mode="fused"`` runs
    one ``Reduce``; ``"split"`` runs the MoE's former pair of ``AllReduce`` (the
    scatter part as a full all-reduce of ``lanes * rows`` rows, own chunk kept)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllReduce,
        AllReduceSpec,
        Reduce,
        ReducePart,
        ReduceSpec,
        entry_barrier,
    )

    (ra, rs), lanes = parts, topo.lanes
    if mode == "fused":
        specs = [
            ReduceSpec(
                topo,
                (ReducePart(ra), ReducePart(rs, scatter=True)),
                transport,
                host_transport,
            )
        ]
    else:
        specs = [
            AllReduceSpec(topo, ra, transport, host_transport=host_transport),
            AllReduceSpec(topo, lanes * rs, transport, host_transport=host_transport),
        ]
    scratch = [x for sp in specs for x in sp.scratch_shapes()]

    def kernel(xa_ref, xs_ref, oa_ref, os_ref, *refs):
        me = jax.lax.axis_index("x")
        lane = me % lanes
        it = iter(refs)
        if mode == "fused":
            colls = [
                Reduce(specs[0], me, *[next(it) for _ in specs[0].scratch_shapes()])
            ]
        else:
            colls = [
                AllReduce(sp, me, *[next(it) for _ in sp.scratch_shapes()])
                for sp in specs
            ]
        for c in colls:
            c.init()
        entry_barrier(me, topo)

        def body(i, carry):
            va, vs = carry
            k = (i + 1).astype(jnp.float32)
            xa = va * (1.0 / topo.world) + xa_ref[...] * k
            xs = vs[None] * (1.0 / topo.world) + xs_ref[...] * k  # [lanes, rs, 128]
            if mode == "fused":
                colls[0].start(1, xs)
                colls[0].start(0, xa)
                return tuple(colls[0].finish())
            colls[1].start(xs.reshape(lanes * rs, 128))
            a = colls[0](xa)
            full = colls[1].finish().reshape(lanes, rs, 128)
            own = full[0]
            for j in range(1, lanes):
                own = jnp.where(lane == j, full[j], own)
            return a, own

        zero = (jnp.zeros((ra, 128), jnp.float32), jnp.zeros((rs, 128), jnp.float32))
        oa_ref[...], os_ref[...] = jax.lax.fori_loop(0, rounds, body, zero)

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((ra, 128), jnp.float32),
            jax.ShapeDtypeStruct((rs, 128), jnp.float32),
        ),
        in_specs=[vm, vm],
        out_specs=(vm, vm),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[: topo.world]), ("x",))

    def local(xa, xs):
        a, b = call(xa[0], xs[0])
        return a[None], b[None]

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("x"), P("x")),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
    )


def _reference_reduce(xa, xs, topo, rounds):
    """Per rank (all-reduce result, own-lane chunk) of ``_build_reduce`` with f32 sums."""
    world, lanes = topo.world, topo.lanes
    inv = np.float32(1.0 / world)

    def tiered(parts):
        host_sums = []
        for h in range(topo.hosts):
            acc = parts[h * lanes].astype(np.float32)
            for j in range(1, lanes):
                acc = acc + parts[h * lanes + j]
            host_sums.append(acc)
        acc = host_sums[0]
        for h in range(1, topo.hosts):
            acc = acc + host_sums[h]
        return acc

    va = np.zeros_like(xa[0])
    vs = [np.zeros_like(xs[0][0]) for _ in range(lanes)]  # per lane
    for i in range(rounds):
        k = np.float32(i + 1)
        a = tiered([np.float32(va * inv) + xa[r] * k for r in range(world)])
        ins = [np.float32(vs[r % lanes][None] * inv) + xs[r] * k for r in range(world)]
        vs = [tiered([ins[r][j] for r in range(world)]) for j in range(lanes)]
        va = a
    return va, vs


def _build_gather(topo, rows, transport, flat, rounds):
    """Rounds of an all-gather; round i gathers (previous gathered block of rank
    ``me ^ 1`` + this rank's input), bf16-rounded so a bf16 wire is exact."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllGather,
        AllGatherSpec,
        entry_barrier,
    )

    spec = AllGatherSpec(topo, rows, transport, flat)

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index("x")
        ag = AllGather(spec, me, *scratch)
        ag.init()
        entry_barrier(me, topo)
        peer = me ^ 1

        def body(i, v):
            got = ag((v + x_ref[...]).astype(jnp.bfloat16).astype(jnp.float32))
            flat_got = got.reshape(topo.world, rows, 128)
            return jnp.sum(
                jnp.where(
                    jax.lax.broadcasted_iota(jnp.int32, (topo.world, 1, 1), 0) == peer,
                    flat_got,
                    0.0,
                ),
                axis=0,
            )

        o_ref[...] = jax.lax.fori_loop(
            0, rounds, body, jnp.zeros((rows, 128), jnp.float32)
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[: topo.world]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


def _build_row_gather(topo, cols, transport, rounds):
    """Row-mode gather rounds: round i gathers bf16(previous row of rank ``me ^ 1`` + input);
    the output is this rank's last gathered ``[1, world * cols]`` row."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllGather,
        AllGatherSpec,
        entry_barrier,
    )

    spec = AllGatherSpec(topo, 1, transport, flat=True, cols=cols)
    world = topo.world

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index("x")
        ag = AllGather(spec, me, *scratch)
        ag.init()
        entry_barrier(me, topo)
        peer = me ^ 1

        def body(i, carry):
            v, _ = carry
            got = ag((v + x_ref[...]).astype(jnp.bfloat16).astype(jnp.float32))
            nxt = got[:, 0:cols]
            for r in range(1, world):
                nxt = jnp.where(peer == r, got[:, r * cols : (r + 1) * cols], nxt)
            return nxt, got

        _, o_ref[...] = jax.lax.fori_loop(
            0,
            rounds,
            body,
            (
                jnp.zeros((1, cols), jnp.float32),
                jnp.zeros((1, world * cols), jnp.float32),
            ),
        )

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, world * cols), jnp.float32),
        in_specs=[vm],
        out_specs=vm,
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[:world]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


def _build_stacklike(topo, rows, transport, host_transport, stride, layers):
    """The stack's use of one all-reduce: once before a ``pl.loop`` over layers, then once per
    layer inside a ``lax.cond`` whose two branches both call it (KDA / MLA)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.config import Topology
    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllReduce,
        AllReduceSpec,
        entry_barrier,
    )

    spec = AllReduceSpec(
        topo, rows, transport, stride=stride, host_transport=host_transport
    )
    world = Topology(1, topo.world * stride)

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index("x")
        ar = AllReduce(spec, me, *scratch)
        ar.init()
        entry_barrier(me, world)
        o_ref[...] = ar(x_ref[...]).astype(jnp.bfloat16).astype(jnp.float32)

        @pl.loop(0, layers)
        def _(l):
            v = o_ref[...] * (1.0 / topo.world)
            o_ref[...] = jax.lax.cond(
                l % 2 == 0,
                lambda: ar(v + x_ref[...]).astype(jnp.bfloat16).astype(jnp.float32),
                lambda: ar(v - x_ref[...]).astype(jnp.bfloat16).astype(jnp.float32),
            )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[: world.world]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


def _build_rowscatter(topo, rows, hidden, transport, rounds, mode, stride=1):
    """Rounds of a bf16-rounded all-reduce of ``[rows, hidden]`` (flat): ``"scatter"`` =
    ``RowScatterAllReduce``, ``"ar"`` = ``AllReduce`` then bf16 rounding."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.k3_megakernel.config import Topology
    from sglang_tpu.k3_megakernel.kernel.collectives import (
        AllReduce,
        AllReduceSpec,
        RowScatterAllReduce,
        RowScatterSpec,
        entry_barrier,
    )

    n = rows * hidden // 128
    if mode == "scatter":
        spec, cls = (
            RowScatterSpec(topo, rows, hidden, transport, stride=stride),
            RowScatterAllReduce,
        )
    else:
        spec, cls = AllReduceSpec(topo, n, transport, stride=stride), AllReduce

    def kernel(x_ref, o_ref, *scratch):
        me = jax.lax.axis_index("x")
        coll = cls(spec, me, *scratch)
        coll.init()
        entry_barrier(me, Topology(1, topo.world * stride))

        def body(i, v):
            y = coll(v * (1.0 / topo.world) + x_ref[...] * (i + 1).astype(jnp.float32))
            return y.astype(jnp.bfloat16).astype(jnp.float32)

        o_ref[...] = jax.lax.fori_loop(
            0, rounds, body, jnp.zeros((n, 128), jnp.float32)
        )

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n, 128), jnp.float32),
        in_specs=[vm],
        out_specs=vm,
        scratch_shapes=spec.scratch_shapes(),
        compiler_params=pltpu.CompilerParams(collective_id=7),
    )
    mesh = Mesh(np.array(_DEVICES[: topo.world * stride]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


def _reference_gather(xs, rounds):
    world = len(xs)
    v = [np.zeros_like(xs[0]) for _ in range(world)]
    for _ in range(rounds):
        sent = [
            np.asarray(
                jnp.asarray(v[r] + xs[r]).astype(jnp.bfloat16).astype(jnp.float32)
            )
            for r in range(world)
        ]
        v = [sent[r ^ 1] for r in range(world)]
    return v


def _reference(xs, topo, rounds):
    """Sequential f32 sums in the kernel's order: lanes, then hosts."""
    world = topo.world
    v = np.zeros_like(xs[0])
    for i in range(rounds):
        parts = [
            np.float32(v * np.float32(1.0 / world)) + xs[r] * np.float32(i + 1)
            for r in range(world)
        ]
        host_sums = []
        for h in range(topo.hosts):
            acc = parts[h * topo.lanes].astype(np.float32)
            for j in range(1, topo.lanes):
                acc = acc + parts[h * topo.lanes + j]
            host_sums.append(acc)
        acc = host_sums[0]
        for h in range(1, topo.hosts):
            acc = acc + host_sums[h]
        v = acc
    return v


@unittest.skipUnless(len(_DEVICES) >= 8, "requires 8 TPU cores (one tpu7x host)")
class TestTieredAllReduce(CustomTestCase):
    def _topologies(self):
        from sglang_tpu.k3_megakernel.config import Topology

        return [Topology(1, 8), Topology(2, 4), Topology(4, 2)]

    def test_f32_exact(self):
        for topo in self._topologies():
            rows = 56  # one 7168-wide f32 row
            rng = np.random.default_rng(topo.hosts)
            xs = rng.standard_normal((topo.world, rows, 128)).astype(np.float32)
            fn = _build(topo, rows, jnp.float32, _ROUNDS)
            got = np.asarray(fn(jnp.asarray(xs.reshape(-1, 128)))).reshape(
                topo.world, rows, 128
            )
            want = _reference(list(xs), topo, _ROUNDS)
            for r in range(topo.world):
                np.testing.assert_array_equal(got[r], want, err_msg=f"{topo} rank {r}")

    def test_bf16_transport(self):
        for topo in self._topologies():
            rows = 64
            rng = np.random.default_rng(10 + topo.hosts)
            xs = rng.standard_normal((topo.world, rows, 128)).astype(np.float32)
            fn = _build(topo, rows, jnp.bfloat16, _ROUNDS)
            got = np.asarray(fn(jnp.asarray(xs.reshape(-1, 128)))).reshape(
                topo.world, rows, 128
            )
            want = _reference(list(xs), topo, _ROUNDS)
            err = np.abs(got - want[None]).max() / np.abs(want).max()
            self.assertLess(err, 2e-2, f"{topo}")
            for r in range(1, topo.world):
                np.testing.assert_array_equal(
                    got[r], got[0], err_msg=f"{topo} rank {r}"
                )

    def _run_reduce(self, topo, t, th, mode):
        ra, rs = 32, 8
        rng = np.random.default_rng(20 + topo.hosts)
        xa = rng.standard_normal((topo.world, ra, 128)).astype(np.float32)
        xs = rng.standard_normal((topo.world, topo.lanes, rs, 128)).astype(np.float32)
        fn = _build_reduce(topo, (ra, rs), t, th, _ROUNDS, mode)
        ga, gs = fn(jnp.asarray(xa), jnp.asarray(xs))
        ga = np.asarray(ga).reshape(topo.world, ra, 128)
        gs = np.asarray(gs).reshape(topo.world, rs, 128)
        return ga, gs, _reference_reduce(list(xa), list(xs), topo, _ROUNDS)

    def test_reduce_f32_exact(self):
        for topo in self._topologies():
            ga, gs, (wa, ws) = self._run_reduce(topo, jnp.float32, None, "fused")
            for r in range(topo.world):
                np.testing.assert_array_equal(
                    ga[r], wa, err_msg=f"{topo} rank {r} all-reduce part"
                )
                np.testing.assert_array_equal(
                    gs[r], ws[r % topo.lanes], err_msg=f"{topo} rank {r} scatter part"
                )

    def test_reduce_matches_split_bf16(self):
        """bf16 wire (both levels, or the host level only): the fused reduction gives the
        bits of the two separate all-reduces it replaces."""
        for topo in self._topologies():
            for t, th in ((jnp.bfloat16, None), (jnp.float32, jnp.bfloat16)):
                fa, fs, (wa, _) = self._run_reduce(topo, t, th, "fused")
                sa, ss, _ = self._run_reduce(topo, t, th, "split")
                np.testing.assert_array_equal(fa, sa, err_msg=f"{topo} {t} {th}")
                np.testing.assert_array_equal(fs, ss, err_msg=f"{topo} {t} {th}")
                err = np.abs(fa - wa[None]).max() / np.abs(wa).max()
                self.assertLess(err, 2e-2, f"{topo} {t} {th}")

    def test_gather(self):
        for topo in self._topologies():
            rows = 8
            rng = np.random.default_rng(30 + topo.hosts)
            xs = rng.standard_normal((topo.world, rows, 128)).astype(np.float32)
            want = _reference_gather(list(xs), _ROUNDS)
            for t in (jnp.float32, jnp.bfloat16):
                for flat in (False, True):
                    fn = _build_gather(topo, rows, t, flat, _ROUNDS)
                    got = np.asarray(fn(jnp.asarray(xs.reshape(-1, 128)))).reshape(
                        topo.world, rows, 128
                    )
                    for r in range(topo.world):
                        np.testing.assert_array_equal(
                            got[r], want[r], err_msg=f"{topo} {t} flat={flat} rank {r}"
                        )

    def _flat_cases(self):
        from sglang_tpu.k3_megakernel.config import Topology

        return [(t, 1) for t in self._topologies()] + [
            (Topology(2, 2), 2),
            (Topology(1, 4), 2),
        ]

    def test_bf16_wires_tiered(self):
        """bf16 on both levels or the host level only, odd tile counts and one chip-pair parity:
        within bf16 error of the f32 result."""
        for topo, stride in self._flat_cases():
            for rows in (56, 32):
                rng = np.random.default_rng(70 + rows + topo.hosts + stride)
                x = jnp.asarray(rng.standard_normal((8 * rows, 128)).astype(np.float32))
                ref = np.asarray(
                    _build(topo, rows, jnp.float32, _ROUNDS, stride=stride)(x)
                )
                for t, th in ((jnp.bfloat16, None), (jnp.float32, jnp.bfloat16)):
                    got = np.asarray(
                        _build(
                            topo, rows, t, _ROUNDS, stride=stride, host_transport=th
                        )(x)
                    )
                    err = np.abs(got - ref).max() / np.abs(ref).max()
                    self.assertLess(
                        err, 2e-2, f"{topo} stride={stride} rows={rows} {t}/{th}: {err}"
                    )

    def test_stacklike_bf16(self):
        for topo, stride in self._flat_cases():
            for rows in (56, 32):
                rng = np.random.default_rng(80 + rows + topo.hosts + stride)
                x = jnp.asarray(rng.standard_normal((8 * rows, 128)).astype(np.float32))
                ref = np.asarray(
                    _build_stacklike(topo, rows, jnp.float32, None, stride, 13)(x)
                )
                for t, th in ((jnp.bfloat16, None), (jnp.float32, jnp.bfloat16)):
                    got = np.asarray(_build_stacklike(topo, rows, t, th, stride, 13)(x))
                    err = np.abs(got - ref).max() / np.abs(ref).max()
                    self.assertLess(
                        err, 5e-2, f"{topo} stride={stride} rows={rows} {t}/{th}: {err}"
                    )

    def test_flat_allreduce(self):
        """Flat one-shot == tiered bit for bit with an f32 wire (also over one chip-pair parity)."""
        for topo, stride in self._flat_cases():
            rows = 32
            rng = np.random.default_rng(60 + topo.hosts + stride)
            x = jnp.asarray(rng.standard_normal((8 * rows, 128)).astype(np.float32))
            a = np.asarray(
                _build(topo, rows, jnp.float32, _ROUNDS, flat=True, stride=stride)(x)
            )
            b = np.asarray(
                _build(topo, rows, jnp.float32, _ROUNDS, flat=False, stride=stride)(x)
            )
            np.testing.assert_array_equal(a, b, err_msg=f"{topo} stride={stride}")
            c = np.asarray(
                _build(topo, rows, jnp.bfloat16, _ROUNDS, flat=True, stride=stride)(x)
            )
            self.assertLess(
                np.abs(c - a).max() / np.abs(a).max(), 2e-2, f"{topo} stride={stride}"
            )
            c = c.reshape(8, rows, 128)
            for r in range(8):
                np.testing.assert_array_equal(
                    c[r], c[r % stride], err_msg=f"{topo} stride={stride} rank {r}"
                )

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench_flat_allreduce(self):
        """Attention-shaped: full 7168 (56 rows) and pair-oproj half 3584 (32 rows) over one parity."""
        for topo, stride in self._flat_cases():
            for rows in (56, 32) if stride == 1 else (32,):
                x = jnp.ones((8 * rows, 128), jnp.float32)
                for t in (jnp.float32, jnp.bfloat16):
                    for flat in (False, True):
                        per = self._time(
                            lambda n: _build(
                                topo, rows, t, n, flat=flat, stride=stride
                            ),
                            (x,),
                        )
                        print(
                            f"BENCH allreduce {topo} stride={stride} rows={rows} {jnp.dtype(t).name} "
                            f"flat={flat}: {per:.2f} us",
                            flush=True,
                        )

    def test_rowscatter_matches_allreduce(self):
        for topo, stride in self._flat_cases():
            hidden = 7168 // stride
            for t in (jnp.float32, jnp.bfloat16):
                rng = np.random.default_rng(90 + topo.hosts)
                x = jnp.asarray(
                    rng.standard_normal((8 * 8 * hidden // 128, 128)).astype(np.float32)
                )
                a = np.asarray(
                    _build_rowscatter(topo, 8, hidden, t, _ROUNDS, "scatter", stride)(x)
                )
                b = np.asarray(
                    _build_rowscatter(topo, 8, hidden, t, _ROUNDS, "ar", stride)(x)
                )
                np.testing.assert_array_equal(
                    a, b, err_msg=f"{topo} stride={stride} {t}"
                )

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench_rowscatter(self):
        """8-row o_proj reduce: full-width AllReduce vs row-scattered."""
        for topo in self._topologies():
            x = jnp.ones((topo.world * 8 * 56, 128), jnp.float32)
            for t in (jnp.float32, jnp.bfloat16):
                for mode in ("ar", "scatter"):
                    per = self._time(
                        lambda n: _build_rowscatter(topo, 8, 7168, t, n, mode), (x,)
                    )
                    print(
                        f"BENCH rows8 {topo} {mode} {jnp.dtype(t).name}: {per:.2f} us",
                        flush=True,
                    )

    def test_row_gather(self):
        for topo in self._topologies():
            for cols in (128, 512):
                rng = np.random.default_rng(50 + cols + topo.hosts)
                xs = rng.standard_normal((topo.world, 1, cols)).astype(np.float32)
                v = [np.zeros((1, cols), np.float32) for _ in range(topo.world)]
                for _ in range(_ROUNDS):
                    sent = [
                        np.asarray(
                            jnp.asarray(v[r] + xs[r])
                            .astype(jnp.bfloat16)
                            .astype(jnp.float32)
                        )
                        for r in range(topo.world)
                    ]
                    v = [sent[r ^ 1] for r in range(topo.world)]
                want = np.concatenate(sent, axis=1)
                for t in (jnp.float32, jnp.bfloat16):
                    got = np.asarray(
                        _build_row_gather(topo, cols, t, _ROUNDS)(
                            jnp.asarray(xs.reshape(-1, cols))
                        )
                    )
                    for r in range(topo.world):
                        np.testing.assert_array_equal(
                            got[r : r + 1],
                            want,
                            err_msg=f"{topo} cols={cols} {t} rank {r}",
                        )

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench_gather(self):
        for topo in self._topologies():
            for rows in (8, 16):
                x = jnp.ones((topo.world * rows, 128), jnp.float32)
                for t in (jnp.float32, jnp.bfloat16):
                    for flat in (False, True):
                        per = self._time(
                            lambda n: _build_gather(topo, rows, t, flat, n), (x,)
                        )
                        print(
                            f"BENCH gather {topo} rows={rows} {jnp.dtype(t).name} flat={flat}: {per:.2f} us",
                            flush=True,
                        )
            for cols in (128, 512, 896):
                x = jnp.ones((topo.world, cols), jnp.float32)
                for t in (jnp.float32, jnp.bfloat16):
                    per = self._time(
                        lambda n: _build_row_gather(topo, cols, t, n), (x,)
                    )
                    print(
                        f"BENCH gather {topo} row cols={cols} {jnp.dtype(t).name} flat=True: {per:.2f} us",
                        flush=True,
                    )

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench_waits(self):
        """Per-copy vs aggregated (one per level) DMA waits."""
        from sglang_tpu.k3_megakernel.kernel import collectives

        try:
            for topo in self._topologies():
                rs = -(-(56 // topo.lanes) // 8) * 8
                xa = jnp.ones((topo.world, 32, 128), jnp.float32)
                xs = jnp.ones((topo.world, topo.lanes, rs, 128), jnp.float32)
                x56 = jnp.ones((topo.world * 56, 128), jnp.float32)
                x8 = jnp.ones((topo.world * 8, 128), jnp.float32)
                for agg in (False, True):
                    collectives.AGGREGATE_WAITS[0] = agg
                    ar = self._time(lambda n: _build(topo, 56, jnp.float32, n), (x56,))
                    red = self._time(
                        lambda n: _build_reduce(
                            topo, (32, rs), jnp.float32, None, n, "fused"
                        ),
                        (xa, xs),
                    )
                    g = self._time(
                        lambda n: _build_gather(topo, 8, jnp.float32, False, n), (x8,)
                    )
                    gf = self._time(
                        lambda n: _build_gather(topo, 8, jnp.float32, True, n), (x8,)
                    )
                    print(
                        f"BENCH waits {topo} aggregate={agg}: allreduce56 {ar:.2f} fused {red:.2f} "
                        f"gather {g:.2f} flat-gather {gf:.2f} us",
                        flush=True,
                    )
        finally:
            collectives.AGGREGATE_WAITS[0] = True

    @staticmethod
    def _time(build, args):
        times = {}
        for n in (1, 401):
            fn = build(n)
            jax.block_until_ready(fn(*args))
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*args))
                ts.append(time.perf_counter() - t0)
            times[n] = float(np.median(ts))
        return (times[401] - times[1]) / 400 * 1e6

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench_reduce(self):
        """MoE-shaped round: routed latent (28 rows) + shared hidden (56 rows, lane chunk 7)."""
        for topo in self._topologies():
            rs = -(-(56 // topo.lanes) // 8) * 8
            xa = jnp.ones((topo.world, 32, 128), jnp.float32)
            xs = jnp.ones((topo.world, topo.lanes, rs, 128), jnp.float32)
            for t, th in (
                (jnp.float32, None),
                (jnp.float32, jnp.bfloat16),
                (jnp.bfloat16, None),
            ):
                for mode in ("split", "fused"):
                    per = self._time(
                        lambda n: _build_reduce(topo, (32, rs), t, th, n, mode),
                        (xa, xs),
                    )
                    print(
                        f"BENCH reduce {topo} {mode} wire={jnp.dtype(t).name}/"
                        f"{jnp.dtype(th or t).name}: {per:.2f} us/round",
                        flush=True,
                    )

    @unittest.skipUnless(os.environ.get("K3MK_BENCH"), "timing only on request")
    def test_bench(self):
        for topo in self._topologies():
            for rows, t in ((56, jnp.float32), (64, jnp.bfloat16), (32, jnp.bfloat16)):
                x = jnp.ones((topo.world * rows, 128), jnp.float32)
                times = {}
                for n in (1, 401):
                    fn = _build(topo, rows, t, n)
                    fn(x).block_until_ready()
                    ts = []
                    for _ in range(10):
                        t0 = time.perf_counter()
                        fn(x).block_until_ready()
                        ts.append(time.perf_counter() - t0)
                    times[n] = float(np.median(ts))
                per = (times[401] - times[1]) / 400 * 1e6
                print(
                    f"BENCH {topo} rows={rows} {jnp.dtype(t).name}: {per:.2f} us/all-reduce",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
