"""Two-level one-shot Pallas all-reduce vs a plain sum on one host's devices.

The kernel's host split is a parameter, so splitting one host's 8 TensorCores
into 2 or 4 "hosts" exercises both the host-local and the cross-host phase.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=4)

import functools
import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = []


@unittest.skipUnless(len(_DEVICES) >= 4, "requires >= 4 TPU devices")
class TestTwoLevelAllReduce(CustomTestCase):
    def _run(self, widths, local, seed):
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.twolevel_allreduce import twolevel_all_reduce_local

        world = len(_DEVICES)
        mesh = Mesh(np.array(_DEVICES), ("model",))
        rng = np.random.default_rng(seed)
        # Each rank's [1, H] input is one row of a [world, H] array.
        xs = [
            jnp.asarray(rng.standard_normal((world, h)), jnp.bfloat16) for h in widths
        ]
        n = len(xs)
        fn = jax.jit(
            jax.shard_map(
                functools.partial(twolevel_all_reduce_local, world=world, local=local),
                mesh=mesh,
                in_specs=(P("model"),) * n,
                out_specs=P("model") if n == 1 else (P("model"),) * n,
                check_vma=False,
            )
        )
        outs = fn(*xs)
        outs = outs if n > 1 else (outs,)
        for x, out in zip(xs, outs):
            want = np.asarray(x, np.float32).sum(axis=0)
            got = np.asarray(out, np.float32)
            self.assertEqual(got.shape, (world, x.shape[1]))
            # Every rank holds the same sum; the cross-host partial is bf16.
            for r in range(world):
                err = np.abs(got[r] - want).max() / np.abs(want).max()
                self.assertLess(err, 2e-2, f"local={local} rank={r} H={x.shape[1]}")
            np.testing.assert_array_equal(got, np.broadcast_to(got[0], got.shape))

    def test_single_tensor(self):
        world = len(_DEVICES)
        # local=1: one rank per "host", so only the cross-host phase runs.
        for local in sorted({world, world // 2, 2, 1}):
            self._run((7168,), local, seed=local)

    def test_packed_pair(self):
        # Kimi-K3 TP32: routed latent (3584) + shared-expert output (7168).
        world = len(_DEVICES)
        for local in sorted({world, world // 2}):
            self._run((3584, 7168), local, seed=100 + local)

    def test_alternating_single_and_pair_calls(self):
        """n=1 and n=2 exchanges share collective id 1, so the barrier carries
        over between consecutive calls, inside one program and across two.
        Round i reduces (previous result / world + input), which sums to i * the
        plain sum on every rank."""
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.twolevel_allreduce import twolevel_all_reduce_local

        world = len(_DEVICES)
        local = world // 2
        rounds = 3
        mesh = Mesh(np.array(_DEVICES), ("model",))
        ar = functools.partial(twolevel_all_reduce_local, world=world, local=local)
        rng = np.random.default_rng(7)
        x, a, b = (
            jnp.asarray(rng.standard_normal((world, h)), jnp.bfloat16)
            for h in (7168, 3584, 7168)
        )

        def body(x, a, b):
            y, p, q = jnp.zeros_like(x), jnp.zeros_like(a), jnp.zeros_like(b)
            for _ in range(rounds):
                y = ar(y / world + x)
                p, q = ar(p / world + a, q / world + b)
            return y, p, q

        fn = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("model"),) * 3,
                out_specs=(P("model"),) * 3,
                check_vma=False,
            )
        )
        for call in range(2):
            for inp, out in zip((x, a, b), fn(x, a, b)):
                want = rounds * np.asarray(inp, np.float32).sum(axis=0)
                got = np.asarray(out, np.float32)
                err = np.abs(got - want[None]).max() / np.abs(want).max()
                self.assertLess(err, 3e-2, f"call={call} H={inp.shape[1]}")
                np.testing.assert_array_equal(got, np.broadcast_to(got[0], got.shape))


if __name__ == "__main__":
    unittest.main()
