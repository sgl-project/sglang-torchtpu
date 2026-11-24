# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 indexer page-table masking."""

import importlib.util
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

_HAS_JAX = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="requires JAX")

if _HAS_JAX:
    import jax.numpy as jnp

    from sglang_tpu.models.deepseek_v41.ops import indexer


def test_mask_pages_past_rows_invalidates_trailing_and_idle_columns():
    index_pages = jnp.array(
        [
            [6, 7, 10, 11, 0, 1, 0, 1],
            [4, 5, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    requests = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
    # Row 0 max visible length = 300 -> ceil(300/128) = 3 pages (cols 0..2 kept)
    # Row 1 max visible length = 128 -> ceil(128/128) = 1 page  (col 0 kept)
    # Row 2 has no live queries -> all -1
    lengths = jnp.array([250, 300, 128, 999], dtype=jnp.int32)
    masked = np.asarray(
        indexer._mask_pages_past_rows(index_pages, requests, lengths, 128)
    )
    np.testing.assert_array_equal(
        masked,
        np.array(
            [
                [6, 7, 10, -1, -1, -1, -1, -1],
                [4, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        ),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
