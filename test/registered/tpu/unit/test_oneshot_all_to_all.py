"""One-shot Pallas all-to-all vs NumPy on one host's TensorCores.

Block j of every device must land on device j, in device order, for the
payloads of the DSA attention token split (Kimi-K3.1 at TP32: a rank's queries
are 128 rows x 3 heads x 576 bf16 = 1728 [*, 128] rows per peer block, its
outputs 1536), and two exchanges in one program (the queries out and the
outputs back) must not interfere.
"""

from sglang.test.ci.ci_register import register_tpu_ci

# Behind the default-off SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT, and a
# four-chip test holds the whole host: nightly, outside the stage-a budget.
register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=4,
    labels=["singlehost", "infra", "collectives"],
)

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


def _exchanged(x, world):
    """Device i's expected output: block i of every device's input."""
    rows = x.shape[1]
    blocks = np.asarray(x).reshape(world, world, rows, 128)
    return blocks.transpose(1, 0, 2, 3).reshape(world * world, rows, 128)


@unittest.skipUnless(len(_DEVICES) >= 2, "requires >= 2 TPU devices")
class TestOneshotAllToAll(CustomTestCase):
    def setUp(self):
        from jax.sharding import Mesh

        self.world = len(_DEVICES)
        self.mesh = Mesh(np.array(_DEVICES), ("model",))

    def _sharded(self, fn, num_out=1):
        from jax.sharding import PartitionSpec as P

        out = P("model") if num_out == 1 else (P("model"),) * num_out
        return jax.jit(
            jax.shard_map(
                fn, mesh=self.mesh, in_specs=P("model"), out_specs=out, check_vma=False
            )
        )

    def _input(self, rows, dtype, seed):
        rng = np.random.default_rng(seed)
        shape = (self.world * self.world, rows, 128)
        if dtype == jnp.int32:
            return jnp.asarray(rng.integers(-(2**31), 2**31 - 1, shape), dtype)
        return jnp.asarray(rng.standard_normal(shape), dtype)

    def test_exchange(self):
        from sglang_tpu.kernels.oneshot_all_to_all import oneshot_all_to_all

        fn = self._sharded(functools.partial(oneshot_all_to_all, world_size=self.world))
        for rows, dtype in (
            (32, jnp.int32),
            (1728, jnp.bfloat16),
            (1536, jnp.bfloat16),
        ):
            with self.subTest(rows=rows, dtype=jnp.dtype(dtype).name):
                x = self._input(rows, dtype, seed=rows)
                out = np.asarray(jax.block_until_ready(fn(x)))
                np.testing.assert_array_equal(out, _exchanged(x, self.world))

    def test_two_exchanges_in_one_program(self):
        from sglang_tpu.kernels.oneshot_all_to_all import oneshot_all_to_all

        a2a = functools.partial(oneshot_all_to_all, world_size=self.world)

        def there_and_back(x):
            # As in a DSA layer: 1728-row blocks out, 1536-row blocks back (two
            # kernels, one collective id). The second reads the first's output,
            # so it returns each device's own blocks, doubled (exact in bf16).
            y = a2a(x)
            return y, a2a(y[:, :1536] * 2)

        x = self._input(1728, jnp.bfloat16, seed=7)
        y, z = jax.block_until_ready(self._sharded(there_and_back, num_out=2)(x))
        np.testing.assert_array_equal(np.asarray(y), _exchanged(x, self.world))
        np.testing.assert_array_equal(np.asarray(z), np.asarray(x[:, :1536] * 2))


if __name__ == "__main__":
    unittest.main()
