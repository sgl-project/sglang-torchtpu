"""Row-streaming hierarchical reduce-scatter vs a plain sum on one tpu7x 2x2x1 host.

kernels/hier_all_reduce.py's reduce_scatter_stream_local is what
SGLANG_TPU_SP_HIER_REDUCE_SCATTER runs for sequence-parallel stage outputs:
device i must receive row block i of the sum of every device's [T, H] input.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=4)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover - non-TPU host
    _DEVICES = []


@unittest.skipUnless(len(_DEVICES) == 8, "requires one 8-device tpu7x host")
class TestSpHierReduceScatterKernel(CustomTestCase):
    def _run(self, rows, hidden, seed):
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.kernels.hier_all_reduce import reduce_scatter_stream_local

        mesh = Mesh(np.array(_DEVICES), ("model",))
        rng = np.random.default_rng(seed)
        # Device d's [rows, hidden] input is block d of an [8 * rows, hidden] array.
        x = rng.standard_normal((8, rows, hidden)).astype(np.float32)
        x_bf16 = jnp.asarray(x.reshape(8 * rows, hidden), jnp.bfloat16)
        fn = jax.jit(
            jax.shard_map(
                lambda local: reduce_scatter_stream_local(local, axis_name="model"),
                mesh=mesh,
                in_specs=P("model", None),
                out_specs=P("model", None),
                check_vma=False,
            )
        )
        got = np.asarray(fn(x_bf16), np.float32).reshape(8, rows // 8, hidden)
        want = (
            np.asarray(x_bf16, np.float32)
            .reshape(8, rows, hidden)
            .sum(axis=0)
            .reshape(8, rows // 8, hidden)
        )
        np.testing.assert_allclose(got, want, atol=0.1, rtol=0.02)

    def test_sp_prefill_buckets(self):
        for rows, hidden in ((1024, 256), (2048, 4096), (8192, 4096), (1024, 6144)):
            with self.subTest(rows=rows, hidden=hidden):
                self._run(rows, hidden, seed=rows + hidden)


if __name__ == "__main__":
    unittest.main()
