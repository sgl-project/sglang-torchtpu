# SPDX-License-Identifier: Apache-2.0
"""Candidate pairing metadata and packed-page gather regressions (no checkpoint).

The metadata contract is checked on whichever backend runs the test: the host
reference partition on CPU, the SparseCore kernels on TPU.
"""

import os
import sys

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

from sglang_tpu.kernels.index_selection.fp8.cache import PackedFP8Layout
from sglang_tpu.kernels.index_selection.fp8.candidates import (
    gather_candidate_keys,
    prepare_candidate_metadata,
)


def _expected_group_starts(owners, groups):
    """Even-sized requests pair adjacent queries; odd and invalid rows stay single."""
    starts, row = [], 0
    while row < len(owners):
        end = row + 1
        while end < len(owners) and owners[end] == owners[row]:
            end += 1
        size = end - row
        paired = 0 <= owners[row] < groups and size % 2 == 0
        starts.extend(range(row, end, 2 if paired else 1))
        row = end
    return starts


# Per-request query counts: decode, uniform and mixed speculative widths,
# empty requests, and prefill-sized requests on both sides of the 4096-query
# switch between compare_all and nonzero offsets.
@pytest.mark.parametrize(
    "counts",
    [
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (3, 3, 3, 3),
        (6, 6, 6, 6),
        (1, 2, 3, 6),
        (5, 0, 6, 1),
        (64, 0, 63, 1),
        (4096, 0, 4097, 0),
    ],
)
def test_candidate_metadata_pairs_queries_per_request(counts):
    width, block, count, page_size = 4096, 8, 128, 128
    groups = len(counts)
    owners = np.concatenate(
        [np.repeat(np.arange(groups, dtype=np.int32), counts), [-1, -1, -1]]
    ).astype(np.int32)
    tokens = owners.size
    per_request = width // page_size
    layout = PackedFP8Layout(groups * per_request, page_size=page_size)
    table = np.random.default_rng(5).permutation(layout.pages)
    table = table.reshape(groups, per_request).astype(np.int32)
    table[1, 3] = -1  # Blocks on a missing page are dropped from every union.
    blocks = width // block
    # Pair members overlap in [64, 128): A-only, shared and B-only all occur.
    candidates = np.stack(
        [np.arange(count) + (0 if row % 2 == 0 else 64) for row in range(tokens)]
    ).astype(np.int32)
    candidates[:, -1] = blocks + 5  # Out-of-range IDs are ignored.

    metadata = jax.jit(
        lambda p, c, r: prepare_candidate_metadata(p, c, r, layout, block, width)
    )(jnp.asarray(table), jnp.asarray(candidates), jnp.asarray(owners))
    offsets = np.asarray(metadata.offsets)
    ids = np.asarray(metadata.candidate_block_ids)
    spans = np.asarray(metadata.block_spans)

    starts = _expected_group_starts(owners, groups)
    expected_offsets = np.full(tokens + 1, tokens, np.int32)
    expected_offsets[: len(starts)] = starts
    np.testing.assert_array_equal(offsets, expected_offsets)
    np.testing.assert_array_equal(np.asarray(metadata.valid_queries), owners >= 0)

    for group in range(len(starts)):
        members = range(offsets[group], offsets[group + 1])
        for member, row in enumerate(members):
            owner = owners[row]
            got = ids[row][ids[row] >= 0]
            if owner < 0:
                assert got.size == 0, row
                continue
            pages = table[owner]
            wanted = {
                b
                for b in candidates[row]
                if 0 <= b < blocks and pages[b * block // page_size] >= 0
            }
            assert set(got.tolist()) == wanted, (row, member)
            assert np.unique(got).size == got.size
            # Each query's window is two ascending segments split at spans[1].
            split = spans[row, 1]
            assert np.all(np.diff(got[:split]) > 0) and np.all(np.diff(got[split:]) > 0)
        if len(members) == 2:
            # Both members address the same shared union slots.
            first, second = (ids[row][ids[row] >= 0] for row in members)
            a_only, shared = spans[members[0], 1], spans[members[1], 1]
            assert spans[members[1], 0] == a_only
            np.testing.assert_array_equal(first[a_only:], second[:shared])


@pytest.mark.parametrize("banks_count", [1, 2])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("fragments", [16, 2048, 131072])
def test_candidate_gather(banks_count, page_size, fragments):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore execution requires TPU")
    # Nonzero page offset/stride proves that physical addressing does not
    # accidentally depend on tightly packed source pages. A 64-key page
    # occupies 16.5 owner rows, rounded up to 17.
    page_rows = PackedFP8Layout(8, page_size=page_size).page_rows
    layout = PackedFP8Layout(
        8, page_stride=page_rows + 1, page_offset=1, page_size=page_size
    )
    owner_rows = 8 * layout.page_stride + 1
    raw = np.arange(owner_rows * 128, dtype=np.int32).reshape(owner_rows, 128) + 19
    split = 1 + 4 * layout.page_stride  # Bank boundary between whole pages.
    banks = (
        (jnp.asarray(raw),)
        if banks_count == 1
        else (jnp.asarray(raw[:split]), jnp.asarray(raw[split:]))
    )
    ids = np.random.default_rng(19).integers(
        -1, layout.pages * (page_size // 8), (3, fragments), dtype=np.int32
    )
    ids[:, ::7] = -1
    offsets = jnp.array([0, 3, 3, 3], jnp.int32)  # Two trailing empty groups.
    out = jax.jit(
        lambda b, p: gather_candidate_keys(b, p, offsets, block=8, layout=layout)
    )(banks, jnp.asarray(ids))
    assert out.shape == (3 * fragments // 16 * 33, 128)
    out = np.asarray(out).reshape(3, fragments // 16, 33, 128)
    # Only valid fragments of live groups are defined; skipped tiles and
    # ignored fragments are left uninitialized by the gather.
    for i, physical in enumerate(ids[0]):
        if physical < 0:
            continue
        page, fragment = divmod(int(physical), page_size // 8)
        base = layout.page_offset + page * layout.page_stride
        for field in range(33):
            start = base * 128 + field * page_size + fragment * 8
            np.testing.assert_array_equal(
                out[0, i // 16, field, i % 16 * 8 : (i % 16 + 1) * 8],
                raw.reshape(-1)[start : start + 8],
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
