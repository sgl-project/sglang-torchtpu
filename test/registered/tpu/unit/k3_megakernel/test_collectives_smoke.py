# SPDX-License-Identifier: Apache-2.0
"""K3 megakernel in-kernel collectives smoke (CI): the remote-DMA all-reduce on one host's 8
TensorCores, split into fake hosts so both levels run. The f32 tiered all-reduce must equal a
sequential f32 sum in the kernel's peer order bit for bit, and the flat one-shot must equal
the tiered one. The full collectives suite is test/manual/k3_megakernel/test_collectives.py."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=4)

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

    from sglang_tpu.k3_megakernel.config import K3MK_COLLECTIVE_ID, Topology
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
        compiler_params=pltpu.CompilerParams(collective_id=K3MK_COLLECTIVE_ID),
    )
    mesh = Mesh(np.array(_DEVICES[: world.world]), ("x",))
    return jax.jit(
        jax.shard_map(
            call, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_vma=False
        )
    )


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
class TestCollectivesSmoke(CustomTestCase):
    def _topologies(self):
        from sglang_tpu.k3_megakernel.config import Topology

        return [Topology(1, 8), Topology(2, 4)]

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

    def test_flat_allreduce(self):
        for topo in self._topologies():
            rows = 32
            rng = np.random.default_rng(60 + topo.hosts)
            x = jnp.asarray(rng.standard_normal((8 * rows, 128)).astype(np.float32))
            a = np.asarray(_build(topo, rows, jnp.float32, _ROUNDS, flat=True)(x))
            b = np.asarray(_build(topo, rows, jnp.float32, _ROUNDS, flat=False)(x))
            np.testing.assert_array_equal(a, b, err_msg=f"{topo}")


if __name__ == "__main__":
    unittest.main()
