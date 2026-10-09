import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# Platform discovery above enforces CPU-only JAX for SGLang, but does not import
# JAX. Select the physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_reduce_v3 import (
    ragged_gather_reduce as ragged_gather_reduce_v3,
)
from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_reduce_v3 import (
    token_block_alignment,
)


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


def _runs_on_sparse_core(x: jax.Array) -> bool:
    # The wrapper's SparseCore conditions; anything else takes its TensorCore
    # fallback.
    info = pltpu.get_tpu_info()
    return (
        info.sparse_core is not None
        and x.dtype == jnp.bfloat16
        and x.shape[-1] % info.num_lanes == 0
        and x.size * 4 >= info.vmem_capacity_bytes * 0.6
    )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedGatherReduceV3Test(jtu.JaxTestCase):
    # Source rows are ``tokens * reduce_group_size``; the kernel packs source
    # indices into 20 bits and pads destination tokens to one 64-token block per
    # SparseCore row partition.
    _test_cases = [
        dict(out_size=o, start_end=se, hidden_size=h, dtype=d, reduce_group_size=rg)
        for o, se, h, d, rg in itertools.chain(
            # Small inputs take the wrapper's TensorCore fallback.
            itertools.product(
                [400, 840],
                [(3, 338), (10, 255)],
                [128, 512],
                [jnp.bfloat16, jnp.float32],
                [8, 5],
            ),
            # Qwen3.5 prefill: 2K / 8K tokens, top-k 10, hidden 4096.
            itertools.product(
                [20480, 81920],
                [(0, 1 << 30), (300, 2850)],
                [4096],
                [jnp.bfloat16],
                [10],
            ),
            # Other lane-aligned widths and route counts.
            itertools.product(
                [16384],
                [(0, 1 << 30), (99, 1120)],
                [2048, 5120, 7168],
                [jnp.bfloat16],
                [8, 2],
            ),
            # Odd source-row count pads one row before the kernel.
            itertools.product(
                [20485],
                [(0, 1 << 30)],
                [4096],
                [jnp.bfloat16],
                [5],
            ),
        )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce_v3(
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
        valid_rows_mask = jnp.logical_and(
            jnp.array([start], jnp.int32) <= indices,
            indices < jnp.array([end], jnp.int32),
        )
        desired = reference_ragged_gather_reduce(
            x, indices, topk_weights, valid_rows_mask, reduce_group_size
        )
        actual = jax.block_until_ready(
            ragged_gather_reduce_v3(
                x, indices, topk_weights, valid_rows_mask, reduce_group_size
            )
        )
        self.assertEqual(actual.shape, (out_size // reduce_group_size, hidden_size))
        # The kernel emits BF16 tiles; the fallback keeps the input dtype.
        expected_dtype = jnp.bfloat16 if _runs_on_sparse_core(x) else dtype
        self.assertEqual(actual.dtype, expected_dtype)
        self.assertTrue(bool(jnp.all(jnp.isfinite(actual.astype(jnp.float32)))))
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

    def test_prefill_shapes_run_on_sparse_core(self):
        """The Qwen3.5 prefill shapes must reach the kernel, not the fallback."""
        info = pltpu.get_tpu_info()
        for tokens in (2048, 8192, 16384):
            x = jax.ShapeDtypeStruct((tokens * 10, 4096), jnp.bfloat16)
            self.assertTrue(_runs_on_sparse_core(x), tokens)
            alignment = token_block_alignment(4096, info)
            self.assertEqual(tokens % alignment, 0, (tokens, alignment))

    def test_rejects_too_many_source_rows(self):
        x = jax.ShapeDtypeStruct(((1 << 20) + 2, 4096), jnp.bfloat16)
        rows = (1 << 20) + 2
        with self.assertRaisesRegex(ValueError, "source rows"):
            jax.eval_shape(
                lambda x: ragged_gather_reduce_v3(
                    x,
                    jnp.arange(rows, dtype=jnp.int32),
                    jnp.ones((rows,), jnp.bfloat16),
                    jnp.ones((rows,), jnp.bool_),
                    2,
                ),
                x,
            )


if __name__ == "__main__":
    unittest.main(testLoader=jtu.JaxTestLoader())
