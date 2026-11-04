import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

# This is a standalone raw-JAX kernel test, not a TorchTPU runtime process.
# Platform discovery above enforces CPU-only JAX for SGLang, but does not import
# JAX. Select the physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"
# Ported from tpu-inference (https://github.com/vllm-project/tpu-inference)
#   tests/kernels/ragged_gather_reduce_v2_test.py @ 4367dcff1 (2026-07-19).
# Local adaptations: import paths. Dropped ragged_scatter perf-comparison
# test. Added BlockBoundaryTest (groups straddling 64-row blocks).

import itertools
import unittest
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.compilation.compile_cache import initialize_jax_compile_cache
from sglang_tpu.layers.moe.pallas.sparse_core.ragged_gather_reduce_v2 import (
    ragged_gather_reduce as ragged_gather_reduce_v2,
)

initialize_jax_compile_cache()


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
                f"Skipping ragged_gather_reduce_v2 correctness check due to error: {e}"
            )

    # The first perf test case approximates the DeepSeekV3, 2k-batch-size, EP=16.
    # The second case approximates the Qwen3-Coder-480B, 2k-batch-size, EP=8.


def _num_tokens(topk: int, hidden_size: int, min_tokens: int = 2048) -> int:
    """Token count above the kernel's TensorCore-fallback size for bf16 x."""
    vmem_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    return max(min_tokens, pl.cdiv(int(vmem_bytes * 0.6) + 1, 4 * topk * hidden_size))


def _masked_inputs(
    num_tokens: int,
    topk: int,
    hidden_size: int,
    seed: int,
    valid_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Routes are valid per ``valid_fn``; rows of x no valid route reads are NaN."""
    num_rows = num_tokens * topk
    key_x, key_perm, key_w, key_valid = jax.random.split(jax.random.key(seed), 4)
    indices = jax.random.permutation(key_perm, num_rows).astype(jnp.int32)
    valid = valid_fn(indices, key_valid)
    read = jnp.zeros((num_rows,), jnp.bool_).at[indices].set(valid)
    x = jax.random.normal(key_x, (num_rows, hidden_size), jnp.bfloat16)
    x = jnp.where(read[:, None], x, jnp.nan)
    topk_weights = jax.random.uniform(key_w, (num_rows,), jnp.bfloat16)
    return x, indices, topk_weights, valid


def _half_window(num_rows: int) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return lambda indices, _: (indices >= num_rows // 4) & (indices < 3 * num_rows // 4)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class BlockBoundaryTest(jtu.JaxTestCase):
    """Groups whose valid rows cross the kernel's row blocks, run repeatedly."""

    def setUp(self):
        super().setUp()
        if pltpu.get_tpu_info().sparse_core is None:
            self.skipTest("SparseCore is not available")

    def _assert_matches_reference(self, x, indices, topk_weights, valid, topk):
        desired = reference_ragged_gather_reduce(
            x, indices, topk_weights, valid, topk
        ).astype(jnp.float32)
        for rep in range(4):
            actual = ragged_gather_reduce_v2(x, indices, topk_weights, valid, topk)
            np.testing.assert_allclose(
                actual.astype(jnp.float32),
                desired,
                atol=2e-2,
                rtol=2e-2,
                err_msg=f"rep {rep}",
            )

    @parameterized.named_parameters(
        [
            (f"topk{topk}_hidden{hidden_size}", topk, hidden_size)
            for topk in (8, 12, 16)
            for hidden_size in (1024, 3584, 7168)
        ]
    )
    def test_groups_straddling_blocks(self, topk, hidden_size):
        num_tokens = _num_tokens(topk, hidden_size)
        x, indices, topk_weights, valid = _masked_inputs(
            num_tokens, topk, hidden_size, topk, _half_window(num_tokens * topk)
        )
        self._assert_matches_reference(x, indices, topk_weights, valid, topk)

    def test_group_spanning_several_blocks(self):
        topk, hidden_size = 160, 1024
        x, indices, topk_weights, valid = _masked_inputs(
            _num_tokens(topk, hidden_size, min_tokens=128),
            topk,
            hidden_size,
            0,
            lambda indices, key: jax.random.bernoulli(key, 0.8, indices.shape),
        )
        self._assert_matches_reference(x, indices, topk_weights, valid, topk)

    def test_all_rows_valid(self):
        # Full row partitions: the last block ends at the partition end.
        topk, hidden_size = 12, 1024
        x, indices, topk_weights, valid = _masked_inputs(
            _num_tokens(topk, hidden_size),
            topk,
            hidden_size,
            1,
            lambda indices, _: jnp.ones(indices.shape, jnp.bool_),
        )
        self._assert_matches_reference(x, indices, topk_weights, valid, topk)

    def test_row_partitions_without_valid_rows(self):
        # Only the first quarter of the rows can be valid.
        topk, hidden_size = 16, 3584
        num_tokens = _num_tokens(topk, hidden_size)
        num_rows = num_tokens * topk
        in_window = _half_window(num_rows)
        x, indices, topk_weights, valid = _masked_inputs(
            num_tokens,
            topk,
            hidden_size,
            2,
            lambda indices, key: (
                in_window(indices, key) & (jnp.arange(num_rows) < num_rows // 4)
            ),
        )
        self._assert_matches_reference(x, indices, topk_weights, valid, topk)

    def test_no_valid_rows(self):
        topk, hidden_size = 8, 1024
        x, indices, topk_weights, valid = _masked_inputs(
            _num_tokens(topk, hidden_size),
            topk,
            hidden_size,
            3,
            lambda indices, _: jnp.zeros(indices.shape, jnp.bool_),
        )
        self._assert_matches_reference(x, indices, topk_weights, valid, topk)


if __name__ == "__main__":
    unittest.main(testLoader=jtu.JaxTestLoader())
