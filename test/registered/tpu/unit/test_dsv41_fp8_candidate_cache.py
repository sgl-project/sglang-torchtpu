# SPDX-License-Identifier: Apache-2.0
"""Candidate metadata and packed-page gather regressions (no checkpoint)."""

import os

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=300,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.kernels.index_selection.block_max import representative_block_max
from sglang_tpu.kernels.index_selection.fp8_candidate_cache import (
    canonical_candidate_blocks,
    gather_candidate_keys,
)
from sglang_tpu.kernels.index_selection.packed_fp8_cache import PackedFP8Layout


@pytest.mark.parametrize("page_size", [4, 8, 64, 128, 256])
@pytest.mark.parametrize("width", [127, 129, 1048576])
def test_candidate_metadata(page_size, width):
    pages = (width + page_size - 1) // page_size
    table = np.arange(3 * pages, dtype=np.int32).reshape(3, pages) % pages
    table[0, 0] = -1
    table[2, -1] = pages  # allocated-owner bound, not just table bound
    candidates = np.random.default_rng(3).integers(
        -2, (width + 7) // 8 + 2, (3, 2048), dtype=np.int32
    )
    offsets = jnp.array([0, 3, 3, 9], jnp.int32)
    actual, physical = jax.jit(
        lambda c, t: canonical_candidate_blocks(
            c, t, offsets, block=8, width=width, cache_pages=pages, page_size=page_size
        )
    )(jnp.asarray(candidates), jnp.asarray(table))
    expected = np.full((3, 2048), -1, np.int32)
    for row in range(3):
        good = []
        for b in sorted(set(candidates[row].tolist())):
            if row == 1 or not 0 <= b < (width + 7) // 8:
                continue
            if all(
                0 <= table[row, k // page_size] < pages
                for k in range(b * 8, min(b * 8 + 8, width), min(8, page_size))
            ):
                good.append(b)
        expected[row, : len(good)] = good
    np.testing.assert_array_equal(actual, expected)
    parts = max(1, 8 // page_size)
    expected_physical = np.full((3, 2048 * parts), -1, np.int32)
    fragment = min(8, page_size)
    for row in range(3):
        for column, b in enumerate(expected[row]):
            if b < 0:
                continue
            for part in range(parts):
                token = b * 8 + part * fragment
                page_column = token // page_size
                if page_column < pages and 0 <= table[row, page_column] < pages:
                    expected_physical[row, column * parts + part] = (
                        table[row, page_column] * (page_size // fragment)
                        + token % page_size // fragment
                    )
    np.testing.assert_array_equal(physical, expected_physical)


@pytest.mark.parametrize("width", [256, 4352, 1048576])
def test_block_max(width):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore execution requires TPU")
    scores = np.random.default_rng(17).normal(size=(4, width)).astype(np.float32)
    rows = np.array([3, 0, 1, -1, 4], np.int32)
    actual = jax.jit(lambda s, r: representative_block_max(s, r, block=8))(
        jnp.asarray(scores), jnp.asarray(rows)
    )
    expected = scores[[3, 0, 1, 0, 0]].reshape(5, -1, 8).max(-1)
    expected[3:] = -np.inf
    np.testing.assert_array_equal(actual, expected)


def test_block_max_empty():
    out = representative_block_max(
        jnp.zeros((0, 256), jnp.float32), jnp.array([0, -1], jnp.int32), block=8
    )
    np.testing.assert_array_equal(out, np.full((2, 32), -np.inf, np.float32))
    assert representative_block_max(
        jnp.zeros((3, 0), jnp.float32), jnp.array([0], jnp.int32), block=8
    ).shape == (1, 0)


@pytest.mark.parametrize("banks_count", [1, 2])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("fragments", [16, 2048, 131072])
def test_candidate_gather(banks_count, page_size, fragments):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore execution requires TPU")
    # Nonzero page offset/stride proves that physical addressing does not
    # accidentally depend on tightly packed source pages.
    # A 64-key page occupies 16.5 owner rows, rounded up to 17; the
    # final half-row is padding, not part of the next physical page.
    page_rows = PackedFP8Layout(8, page_size=page_size).page_rows
    layout = PackedFP8Layout(
        8, page_stride=page_rows + 1, page_offset=1, page_size=page_size
    )
    owner_rows = 8 * layout.page_stride
    raw = np.arange(owner_rows * 128, dtype=np.int32).reshape(owner_rows, 128) + 19
    banks = tuple(jnp.asarray(b) for b in np.split(raw, banks_count))
    ids = np.random.default_rng(19).integers(
        -1, layout.pages * (page_size // 8) + 1, (3, fragments), dtype=np.int32
    )
    ids[:, ::7] = -1
    offsets = jnp.array([0, 3, 3, 3], jnp.int32)  # two trailing empty groups
    out = jax.jit(
        lambda b, p: gather_candidate_keys(b, p, offsets, block=8, layout=layout)
    )(banks, jnp.asarray(ids))
    expected = np.zeros((3, fragments // 16, 33, 128), np.int32)
    for row in range(1):
        for i, physical in enumerate(ids[row]):
            if not 0 <= physical < layout.pages * (page_size // 8):
                continue
            page, fragment = divmod(int(physical), page_size // 8)
            base = layout.page_offset + page * layout.page_stride
            for field in range(33):
                start = base * 128 + field * page_size + fragment * 8
                expected[row, i // 16, field, i % 16 * 8 : (i % 16 + 1) * 8] = (
                    raw.reshape(-1)[start : start + 8]
                )
    expected = expected.reshape(-1, 128)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)
