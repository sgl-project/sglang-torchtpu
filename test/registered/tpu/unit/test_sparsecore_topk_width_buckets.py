# SPDX-License-Identifier: Apache-2.0
"""The SparseCore top-k width switch must never pick a branch narrower than the longest row.

Runs on CPU with the kernel replaced by a reference of its contract: the exact
top-k of each row's first ``min(row_length, width)`` columns, ascending, lowest
index on ties, ``-1`` suffix. The kernel's own width independence is checked on
TPU in test_dsv4_sparsecore_topk.py.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "cpu"

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.dsv4 import sparsecore_topk as sc
from sglang_tpu.envs import envs as tpu_envs

# Indexer rows of a 4096-token chunk on 8 shards, pools of a 1M context, 2048 tokens.
B, N, K = 512, 263168, 512
LADDER = [4096 * i for i in range(1, 9)] + [49152, 65536, 98304, 131072, 196608]
SEED = 20261120
INT_MIN = np.iinfo(np.int32).min


def _ref_kernel(scores, k, row_lengths, *, return_scores, **_):
    words = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    b, n = words.shape
    key = words ^ ((words >> 31) & sc.MONO_MASK)
    lengths = jnp.minimum(row_lengths, n)
    live = (jnp.arange(n)[None, :] < lengths[:, None]) & (key > sc.KEY_NEG_INF)
    vals, idx = jax.lax.top_k(jnp.where(live, key, INT_MIN), min(k, n))
    idx = jnp.sort(jnp.where(vals > INT_MIN, idx, n), axis=1)
    idx = jnp.pad(
        jnp.where(idx < n, idx, -1), ((0, 0), (0, k - idx.shape[1])), constant_values=-1
    )
    if return_scores:
        return idx, jnp.zeros_like(idx)
    return idx


def _width_kernel(scores, k, row_lengths, **_):
    return jnp.full((scores.shape[0], k), scores.shape[1], jnp.int32)


def _topk(monkeypatch, kernel, scores, row_lengths, *, long_rows=True, **kwargs):
    monkeypatch.setattr(sc, "_sc_topk_direct", kernel)
    with tpu_envs.SGLANG_TPU_SCTOPK_LONG_ROWS.override(long_rows):
        return np.asarray(
            sc.sparsecore_topk.__wrapped__(
                jnp.asarray(scores), K, jnp.asarray(row_lengths, jnp.int32), **kwargs
            )
        )


def _edges():
    return sorted({0, 1, N} | {w + d for w in LADDER + [32768] for d in (-1, 0, 1)})


def test_ladder_at_the_indexer_shape():
    buckets = sc._width_buckets(B, N, K)
    assert [w for w, _ in buckets] == LADDER
    for w, p in buckets:
        # Whole slices of whole lanes: the kernel pads nothing.
        assert w % (p * sc.LANES) == 0 and w // p <= sc.MAX_SLICE_WORDS, (w, p)
    assert [p for w, p in buckets if w > sc.MAX_SLICE_WORDS] == [2, 2, 4, 4, 8]
    assert sc._pick_partition(B, N, K) == 16


@pytest.mark.parametrize(
    "b,n,k",
    [(16, 263168, 512), (1, 263168, 512), (64, 1048576, 2048), (16, 133120, 2048)],
)
def test_ladder_is_ascending_and_below_the_full_row(b, n, k):
    widths = [w for w, _ in sc._width_buckets(b, n, k)]
    assert widths == sorted(set(widths)) and widths[-1] + sc.MAX_SLICE_WORDS <= n
    for w, p in sc._width_buckets(b, n, k):
        assert w >= k and w % (p * sc.LANES) == 0 and 1 <= p <= 16, (w, p)


def test_off_keeps_only_the_short_rows():
    with tpu_envs.SGLANG_TPU_SCTOPK_LONG_ROWS.override(False):
        assert [w for w, _ in sc._width_buckets(B, N, K)] == LADDER[:8]


@pytest.mark.parametrize("max_len", _edges())
def test_branch_is_the_narrowest_bucket_holding_the_longest_row(monkeypatch, max_len):
    lengths = np.minimum(np.arange(B, dtype=np.int32), max_len)
    lengths[-1] = max_len
    scores = np.zeros((B, N), np.float32)
    want = min([w for w in LADDER + [N] if w >= max_len])
    got = _topk(monkeypatch, _width_kernel, scores, lengths)
    assert (got == want).all(), (max_len, got[0, 0], want)


@pytest.mark.parametrize("max_len", [32769, 65537, 196609])
def test_window_takes_the_whole_batch_branch(monkeypatch, max_len):
    """A token-split window passes the whole batch's longest row."""
    lengths = np.full(B, 100, np.int32)
    scores = np.zeros((B, N), np.float32)
    want = min([w for w in LADDER + [N] if w >= max_len])
    got = _topk(
        monkeypatch, _width_kernel, scores, lengths, branch_max_len=jnp.int32(max_len)
    )
    assert (got == want).all(), (max_len, got[0, 0], want)


def _scores(kind, b, rng):
    if kind == "random":
        return rng.standard_normal((b, N), dtype=np.float32)
    if kind == "heavy_ties":
        return rng.integers(0, 17, size=(b, N), dtype=np.int8).astype(np.float32)
    if kind == "neg_inf":
        s = rng.standard_normal((b, N), dtype=np.float32)
        s[rng.random((b, N)) < 0.7] = -np.inf
        return s
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["random", "heavy_ties", "neg_inf"])
def test_bucketed_selection_equals_the_full_width_scan(monkeypatch, kind):
    """Every row length at and around each edge, with large garbage past it."""
    b = 64
    rng = np.random.default_rng(SEED)
    edges = _edges()
    scores = _scores(kind, b, rng)
    for start in range(0, len(edges), b // 2):
        lengths = np.array(
            [
                edges[min(start + i // 2, len(edges) - 1)] - (i % 2) * 3
                for i in range(b)
            ],
            np.int32,
        ).clip(0, N)
        cols = np.arange(N)[None, :]
        garbage = np.where(cols >= lengths[:, None], np.float32(1e30), scores)
        bucketed = _topk(monkeypatch, _ref_kernel, garbage, lengths)
        full = _topk(monkeypatch, _ref_kernel, garbage, lengths, long_rows=False)
        np.testing.assert_array_equal(bucketed, full)
        assert (bucketed < np.maximum(lengths, 1)[:, None]).all()
