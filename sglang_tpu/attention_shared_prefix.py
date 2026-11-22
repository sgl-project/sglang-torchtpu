# SPDX-License-Identifier: Apache-2.0
"""Host-side full-attention page-sharing proof for a single decode forward."""

import numpy as np


def compute_shared_mismatch(
    pages: np.ndarray, live: int, group_size: int = 16
) -> np.ndarray:
    """Return per-group first mismatch page index across ``pages[:live]``."""
    rows, width = pages.shape
    groups = (live + group_size - 1) // group_size
    physical = pages[: groups * group_size].reshape(groups, group_size, width)
    mismatches = physical[:, 1:] != physical[:, :1]
    if live % group_size:
        mismatches[-1, live % group_size - 1 :] = False
    different = np.any(mismatches, axis=1)
    return np.where(different.any(axis=1), different.argmax(axis=1), width)


def shared_decode_distribution(
    pages,
    lengths,
    live,
    page_size,
    *,
    group_size: int = 16,
    block_pages: int | None = None,
    cached_mismatch: np.ndarray | None = None,
):
    """Return [live, live, live, tiles-as-page-counts...] or None for stock."""
    rows, width = pages.shape
    if (
        live < 8
        or live > rows
        or group_size < 8
        or rows % group_size
        or width * page_size <= 16384
        or page_size not in (128, 256, 512)
    ):
        return None
    if block_pages is None:
        block_pages = max(1, 2048 // page_size)
    # Compare every physical page ID, stopping at the first difference even if
    # later IDs match again. Ignore padded rows in the final group.
    groups = (live + group_size - 1) // group_size
    group_sizes = np.full(groups, group_size)
    if live % group_size:
        group_sizes[-1] = live % group_size
    mismatch = (
        cached_mismatch
        if cached_mismatch is not None
        else compute_shared_mismatch(pages, live, group_size)
    )
    cached = np.maximum(np.asarray(lengths[:live], dtype=np.int64) - 1, 0)
    full = np.minimum.reduceat(cached, np.arange(0, live, group_size)) // page_size
    shared = np.minimum(full, mismatch) // block_pages * block_pages
    # Keep small tails private and count savings using only the requests
    # that would have fetched each page.
    shared[group_sizes < 8] = 0
    if np.dot(shared, group_sizes - 1) * 2 * page_size <= cached.sum():
        return None
    plan = np.zeros(3 + rows // group_size, dtype=np.int32)
    plan[:3] = live
    plan[3 : 3 + groups] = shared
    return plan
