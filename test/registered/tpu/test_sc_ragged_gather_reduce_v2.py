
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")
# Ported from tpu-inference (https://github.com/vllm-project/tpu-inference)
#   tests/kernels/ragged_gather_reduce_v2_test.py @ 4367dcff1 (2026-07-19).
# Local adaptations: import paths. Dropped ragged_scatter perf-comparison
# test.

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_reduce_v2 import (
    ragged_gather_reduce as ragged_gather_reduce_v2,
)

jax.config.parse_flags_with_absl()


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o, start_end=se, hidden_size=h, dtype=d, reduce_group_size=rg)
        for o, se, h, d, rg in itertools.chain(
            itertools.product(
                [400, 840],
                [(3, 338), (10, 255)],
                [128, 512, 8192],
                [jnp.bfloat16, jnp.float32],
                [8, 5],
            ),
            itertools.product(
                [16384],
                [(99, 1120)],
                [7168],
                [jnp.bfloat16],
                [8],
            ),
            itertools.product(
                [16384],
                [(300, 2358)],
                [6144],
                [jnp.bfloat16],
                [8],
            ),
            itertools.product(
                [20480],
                [(300, 2850)],
                [4096],
                [jnp.bfloat16],
                [10],
            ),
        )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce(
        self, out_size, hidden_size, start_end, dtype, reduce_group_size
    ):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size,), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )
        # Correctness check.
        desired = reference_ragged_gather_reduce(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )
        try:
            actual = ragged_gather_reduce_v2(
                x, indices, topk_weights, valid_rows_mask, reduce_group_size
            )
            np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)
        except AssertionError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            print(
                "Skipping ragged_gather_reduce_v2 correctness check due to "
                f"error: {e}"
            )

    # The first perf test case approximates the DeepSeekV3, 2k-batch-size, EP=16.
    # The second case approximates the Qwen3-Coder-480B, 2k-batch-size, EP=8.


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
