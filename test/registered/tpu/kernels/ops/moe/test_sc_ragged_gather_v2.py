import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=150, suite="stage-a-test-tpu-unit", chips=1)

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# Platform discovery above enforces CPU-only JAX for SGLang, but does not import
# JAX. Select the physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"
# Ported from tpu-inference (https://github.com/vllm-project/tpu-inference)
#   tests/kernels/ragged_gather_v2_test.py @ 4367dcff1 (2026-07-19).
# Local adaptations: import paths. Added the row-count-not-a-multiple-of-
# packing test. Dropped the benchmark, which asserted nothing.

import unittest

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax._src import test_util as jtu

from sglang_tpu.compile_cache import initialize_jax_compile_cache
from sglang_tpu.kernels.ops.moe.sparse_core.ragged_gather_v2 import ragged_gather_v2

initialize_jax_compile_cache()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GatherTest(jtu.JaxTestCase):
    @parameterized.product(
        in_out_size=[(512, 400), (512, 1024)],
        start_end=[(3, 338), (10, 422)],
        hidden_size=[128, 512, 8192],
        dtype=[jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
    )
    def test_sc_gather(self, in_out_size, hidden_size, start_end, dtype):
        in_size, out_size = in_out_size
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

        start_arr = jnp.array([start], jnp.int32)
        end_arr = jnp.array([end], jnp.int32)

        actual = ragged_gather_v2(x, indices, start_arr, end_arr)
        actual.block_until_ready()

        # Correctness check.
        actual = actual[start:end]
        desired = x[indices][start:end]

        self.assertArraysEqual(actual, desired)

    @parameterized.product(
        in_size=[1, 3, 21, 255],
        hidden_size=[3584, 7168],
        dtype=[jnp.int4, jnp.int8, jnp.bfloat16],
    )
    def test_sc_gather_rows_not_a_multiple_of_packing(
        self, in_size, hidden_size, dtype
    ):
        # Top-16 fused-MoE permute: m tokens -> 16 * m sorted rows, m odd.
        key = jax.random.key(1)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.float32).astype(dtype)
        out_size = 16 * in_size
        indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

        actual = ragged_gather_v2(
            x, indices, jnp.array([0], jnp.int32), jnp.array([out_size], jnp.int32)
        )

        self.assertArraysEqual(actual, x[indices])


if __name__ == "__main__":
    unittest.main(testLoader=jtu.JaxTestLoader())
