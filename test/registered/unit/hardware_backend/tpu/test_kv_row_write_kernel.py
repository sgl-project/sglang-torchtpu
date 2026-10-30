# SPDX-License-Identifier: Apache-2.0
"""The row-run write kernel lands every run in place on a TPU (JAX only)."""

import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sglang.test.ci.ci_register import register_tpu_ci  # noqa: E402
from sglang_tpu.kv_row_write import (  # noqa: E402
    kv_write_segments,
    write_rows,
)

register_tpu_ci(est_time=3, suite="stage-a-test-tpu-unit", chips=1)

pytestmark = pytest.mark.skipif(
    not any(d.platform == "tpu" for d in jax.devices()), reason="needs a TPU"
)


# 16-bit pools pack 2 values per sublane, 8-bit pools 4 (the fp8 byte view).
@pytest.mark.parametrize("dtype,packing", [(np.float32, 2), (np.uint8, 4)])
def test_runs_land_in_place(dtype, packing):
    rng = np.random.default_rng(0)
    num_blocks, page, packed, hd = 16, 16, 2, 128
    kv = np.zeros((num_blocks, page, packed, packing, hd), dtype=dtype)
    rows = (rng.integers(1, 200, size=(60, packed, packing, hd))).astype(dtype)
    # Three runs of different lengths on distinct pages, padding rows between.
    slots = np.concatenate(
        [np.arange(16, 36), np.zeros(4), np.arange(70, 76), np.arange(100, 130)]
    )
    valid = np.ones(60, dtype=bool)
    valid[20:24] = False
    segments, count = kv_write_segments(slots, valid, max_segments=64)
    out = write_rows(
        jax.numpy.asarray([count], dtype=jax.numpy.int32),
        jax.numpy.asarray(segments.numpy()),
        jax.numpy.asarray(rows),
        jax.numpy.asarray(kv),
    )
    expected = kv.reshape(-1, packed, packing, hd).copy()
    expected[slots[valid].astype(np.int64)] = rows[valid]
    np.testing.assert_array_equal(np.asarray(out).reshape(expected.shape), expected)


def test_long_runs_are_staged_in_pieces():
    rng = np.random.default_rng(1)
    num_blocks, page, packed, packing, hd = 64, 128, 2, 2, 128
    kv = np.zeros((num_blocks, page, packed, packing, hd), dtype=np.float32)
    rows = rng.integers(1, 200, size=(3000, packed, packing, hd)).astype(np.float32)
    # One 2500-row run (three staging pieces) and one 500-row run.
    slots = np.concatenate([np.arange(300, 2800), np.arange(5000, 5500)])
    valid = np.ones(3000, dtype=bool)
    segments, count = kv_write_segments(slots, valid, max_segments=64)
    assert count == 4
    out = write_rows(
        jax.numpy.asarray([count], dtype=jax.numpy.int32),
        jax.numpy.asarray(segments.numpy()),
        jax.numpy.asarray(rows),
        jax.numpy.asarray(kv),
    )
    expected = kv.reshape(-1, packed, packing, hd).copy()
    expected[slots] = rows
    np.testing.assert_array_equal(np.asarray(out).reshape(expected.shape), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
