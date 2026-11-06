# SPDX-License-Identifier: Apache-2.0
"""Actual TPU selection regressions: no mocked scorer, gather, or selector.

Zero Q produces exactly tied zero scores over valid keys, which checks the
smaller-logical-ID tie contract of ordered selection without floating-point
tolerances. Random Q checks candidate scoring against an independent dense
reference across decode, speculative and prefill query counts.
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

from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    quantize_index_rows,
    read_fp8_records,
    write_fp8_records,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
    _select_candidate_tokens,
    prepare_fp8_candidates,
    select_fp8_indexer_jax,
    select_fp8_prepared,
)


def _layout_args(layout):
    return (layout.pages, layout.page_stride, layout.page_offset, layout.page_size)


def _zero_banks(layout):
    return (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)


def _select(q, w, banks, pages, lengths, candidates, requests, layout, **kwargs):
    """Run the source/unfiltered or in-op filtered route of the model op."""
    options = dict(
        topk=512,
        block_size=8,
        candidate_write=False,
        candidate_filter=False,
        unordered=True,
        score_width=pages.shape[1] * layout.page_size,
    )
    options.update(kwargs)
    live = (requests >= 0) & (requests < pages.shape[0])
    batch_max = jnp.max(jnp.where(live, lengths, 0), initial=0).reshape(1)
    return jax.jit(
        lambda q, w: select_fp8_indexer_jax(
            q,
            banks,
            pages,
            w,
            lengths,
            candidates,
            requests,
            batch_max,
            _layout_args(layout),
            options["topk"],
            options["block_size"],
            options["candidate_write"],
            options["candidate_filter"],
            options["unordered"],
            options["score_width"],
        )
    )(q, w)


def _assert_tied_selection(selected, lengths, live, k, ordered):
    """Zero Q ties every visible key: ordered picks the lowest IDs, unordered any."""
    for row, (length, valid) in enumerate(zip(lengths, live)):
        n = min(int(length), k) if valid else 0
        ids = selected[row]
        if ordered:
            np.testing.assert_array_equal(ids[:n], np.arange(n))
            assert np.all(ids[n:] == -1), row
            continue
        chosen = ids[ids >= 0]
        assert chosen.size == n and np.unique(chosen).size == n, row
        assert np.all(chosen < length), row


def _sorted_ids(ids):
    """Valid IDs ascending with -1 at the tail, for unordered comparisons."""
    limit = np.iinfo(np.int32).max
    ids = np.sort(np.where(ids >= 0, ids, limit), axis=-1)
    return np.where(ids == limit, -1, ids)


@pytest.mark.parametrize(
    "tokens", [1, 7, 48, 128, 193, 256, 512, 1024, 2048, 4096, 8192]
)
@pytest.mark.parametrize("unordered", [False, True])
def test_dense_selection_and_publication_real_tpu(tokens, unordered):
    assert jax.default_backend() == "tpu", "This test validates actual TPU kernels"
    width, groups, block, k, count = 2048, 4, 8, 512, 128
    layout = PackedFP8Layout(groups * width // 128)
    # A page permutation makes physical/logical ID confusion visible.
    pages = jnp.asarray(
        np.random.default_rng(19).permutation(layout.pages).reshape(groups, -1),
        jnp.int32,
    )
    # Unsorted owners, an invalid row and an out-of-range owner.
    owners = np.resize(np.array([2, 0, 2, -1, 0, 1, 7], np.int32), tokens)
    lengths = np.resize(
        np.array([1801, 1001, 1937, 1800, 1503, 0, 2048], np.int32), tokens
    )
    q = jnp.zeros((tokens, 32, 128), jnp.float8_e4m3fn)
    weights = jnp.ones((tokens, 32), jnp.float32)
    selected, published = map(
        np.asarray,
        _select(
            q,
            weights,
            _zero_banks(layout),
            pages,
            jnp.asarray(lengths),
            jnp.full((tokens, count), -1, jnp.int32),
            jnp.asarray(owners),
            layout,
            candidate_write=True,
            unordered=unordered,
        ),
    )
    live = (owners >= 0) & (owners < groups)
    _assert_tied_selection(selected, lengths, live, k, ordered=not unordered)
    for row, (valid, length) in enumerate(zip(live, lengths)):
        chosen = published[row][published[row] >= 0]
        if not valid or length == 0:
            assert chosen.size == 0, row
            continue
        blocks = (length + block - 1) // block
        # The newest visible block is always published; the rest tie.
        assert blocks - 1 in chosen, row
        assert chosen.size == min(count, blocks), row
        assert np.unique(chosen).size == chosen.size and chosen.max() < blocks, row
        if not unordered:
            expected = sorted(set(range(min(count - 1, blocks))) | {blocks - 1})
            np.testing.assert_array_equal(np.sort(chosen), expected)


# Longest context inside one 64K key tile narrows the score width; a longer
# one keeps the full width. Both take the chunked prefill pipeline.
@pytest.mark.parametrize("longest", [60000, 131072])
def test_chunked_prefill_selection_real_tpu(longest):
    assert jax.default_backend() == "tpu"
    width, k = 131072, 512
    counts = np.array([1000, 1000, 48])
    layout = PackedFP8Layout(len(counts) * width // 128)
    pages = jnp.arange(layout.pages, dtype=jnp.int32).reshape(len(counts), -1)
    owners = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    lengths = np.concatenate(
        [
            longest - 1500 + np.arange(1000),
            longest - 999 + np.arange(1000),
            1 + np.arange(48),
        ]
    ).astype(np.int32)
    tokens = int(counts.sum())
    selected = np.asarray(
        _select(
            jnp.zeros((tokens, 32, 128), jnp.float8_e4m3fn),
            jnp.ones((tokens, 32), jnp.float32),
            _zero_banks(layout),
            pages,
            jnp.asarray(lengths),
            jnp.full((tokens, 128), -1, jnp.int32),
            jnp.asarray(owners),
            layout,
        )[0]
    )
    _assert_tied_selection(selected, lengths, np.ones(tokens, bool), k, ordered=False)


def _candidate_fixture(counts, short, seed=371):
    rng = np.random.default_rng(seed)
    width, page_size, block, count, padding = 4096, 128, 8, 128, 3
    groups = len(counts)
    layout = PackedFP8Layout(groups * width // page_size, page_size=page_size)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    records = jnp.arange(keys.shape[0], dtype=jnp.int32)
    banks = jax.jit(lambda b: write_fp8_records(b, records, keys, layout))(
        _zero_banks(layout)
    )
    pages = rng.permutation(layout.pages).reshape(groups, -1).astype(np.int32)
    # Contiguous request rows with causal lengths, then invalid padding rows.
    owners = np.concatenate(
        [np.repeat(np.arange(groups), counts), np.full(padding, -1)]
    ).astype(np.int32)
    lengths = []
    for size in counts:
        top = rng.integers(64, 512) if short else rng.integers(width // 2, width)
        lengths.append(np.maximum(top - size + 1 + np.arange(size), 1))
    lengths = np.concatenate(lengths + [np.full(padding, width)]).astype(np.int32)
    tokens = owners.size
    candidates = np.full((tokens, count), -1, np.int32)
    for row in range(tokens):
        chosen = rng.choice(width // block, count - 8, replace=False)
        candidates[row, : chosen.size] = np.sort(chosen)
    raw_q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.bfloat16)
    q, scales, _ = quantize_index_rows(raw_q)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32) * scales
    return dict(
        layout=layout,
        banks=banks,
        pages=pages,
        owners=owners,
        lengths=lengths,
        candidates=candidates,
        q=q,
        weights=weights,
        width=width,
        block=block,
    )


def _reference_scores(f):
    """Dense FP32 scores from decoded keys, -inf outside the visible context."""
    layout, width = f["layout"], f["width"]
    position = np.arange(width)
    scores = np.full((f["owners"].size, width), -np.inf, np.float32)
    for group in range(f["pages"].shape[0]):
        rows = np.flatnonzero(f["owners"] == group)
        records = (
            f["pages"][group, position // layout.page_size] * layout.page_size
            + position % layout.page_size
        )
        keys = read_fp8_records(f["banks"], jnp.asarray(records), layout)
        logits = jnp.einsum(
            "qhd,kd->qhk",
            f["q"][rows].astype(jnp.float32),
            keys.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        dense = np.asarray(
            (jnp.maximum(logits, 0) * f["weights"][rows][..., None]).sum(1)
        )
        visible = position[None] < f["lengths"][rows][:, None]
        scores[rows] = np.where(visible, dense, -np.inf)
    return scores


def _assert_top_scores(selected, scores, allowed, k, ordered):
    for row in range(selected.shape[0]):
        ids = selected[row][selected[row] >= 0]
        live = np.flatnonzero(allowed[row])
        assert np.unique(ids).size == ids.size, row
        assert np.all(allowed[row, ids]), row
        assert ids.size == min(k, live.size), (row, ids.size, live.size)
        if ordered:
            np.testing.assert_array_equal(selected[row, : ids.size], np.sort(ids))
        if ids.size < live.size:
            rest = np.setdiff1d(live, ids)
            tolerance = 1e-2 * np.abs(scores[row, live]).max()
            assert scores[row, ids].min() >= scores[row, rest].max() - tolerance, row


# Per-request query counts: decode, uniform and mixed speculative widths
# (odd counts stay unpaired), and prefill-sized requests.
@pytest.mark.parametrize(
    "counts",
    [
        [1] * 8,
        [2] * 4,
        [3] * 4,
        [4] * 3,
        [6] * 2,
        [1, 3, 2, 6, 5],
        [257, 128],
    ],
)
@pytest.mark.parametrize("route", ["prepared", "in_op"])
@pytest.mark.parametrize("unordered", [False, True])
def test_candidate_scoring_matches_dense_reference_real_tpu(counts, route, unordered):
    assert jax.default_backend() == "tpu"
    f = _candidate_fixture(counts, short=False)
    k, block, width = 512, f["block"], f["width"]
    pages, owners, lengths, candidates = map(
        jnp.asarray, (f["pages"], f["owners"], f["lengths"], f["candidates"])
    )
    if route == "prepared":
        prepared = jax.jit(
            lambda: prepare_fp8_candidates(
                f["banks"],
                pages,
                candidates,
                owners,
                _layout_args(f["layout"]),
                block,
                width,
            )
        )()
        select = jax.jit(
            lambda q, w: select_fp8_prepared(
                q, w, lengths, prepared, k, block, unordered, width
            )
        )
        selected = np.asarray(select(f["q"], f["weights"]))
        # A prepared union is reused by later layers with other queries.
        # Unordered selection fixes the set, not its order.
        reused = np.asarray(select(f["q"], f["weights"] * 2))
        if unordered:
            reused, expected = _sorted_ids(reused), _sorted_ids(selected)
        else:
            expected = selected
        np.testing.assert_array_equal(reused, expected)
    else:
        selected = np.asarray(
            _select(
                f["q"],
                f["weights"],
                f["banks"],
                pages,
                lengths,
                candidates,
                owners,
                f["layout"],
                candidate_filter=True,
                unordered=unordered,
            )[0]
        )
    scores = _reference_scores(f)
    position = np.arange(width)
    member = np.stack(
        [np.isin(position // block, row[row >= 0]) for row in f["candidates"]]
    )
    allowed = (
        (f["owners"] >= 0)[:, None] & (position[None] < f["lengths"][:, None]) & member
    )
    assert selected.shape == (f["owners"].size, k)
    _assert_top_scores(selected, scores, allowed, k, ordered=not unordered)


# A batch whose longest context fits top-k bypasses candidate filtering.
@pytest.mark.parametrize("route", ["prepared", "in_op"])
def test_short_batch_bypasses_candidates_real_tpu(route):
    assert jax.default_backend() == "tpu"
    f = _candidate_fixture([1, 3, 2], short=True)
    k, block, width = 512, f["block"], f["width"]
    pages, owners, lengths, candidates = map(
        jnp.asarray, (f["pages"], f["owners"], f["lengths"], f["candidates"])
    )
    # Padding rows carry a full-width length; only valid rows set batch_max.
    if route == "prepared":
        prepared = jax.jit(
            lambda: prepare_fp8_candidates(
                f["banks"],
                pages,
                candidates,
                owners,
                _layout_args(f["layout"]),
                block,
                width,
            )
        )()
        selected = jax.jit(
            lambda: select_fp8_prepared(
                f["q"], f["weights"], lengths, prepared, k, block, True, width
            )
        )()
    else:
        selected = _select(
            f["q"],
            f["weights"],
            f["banks"],
            pages,
            lengths,
            candidates,
            owners,
            f["layout"],
            candidate_filter=True,
        )[0]
    expected = np.full((f["owners"].size, k), -1, np.int32)
    for row, (owner, length) in enumerate(zip(f["owners"], f["lengths"])):
        if owner >= 0:
            expected[row, :length] = np.arange(length)
    np.testing.assert_array_equal(_sorted_ids(np.asarray(selected)), expected)


def test_empty_query_selection_real_tpu():
    assert jax.default_backend() == "tpu"
    layout = PackedFP8Layout(1)
    rows = jnp.zeros((0,), jnp.int32)
    candidates = jnp.zeros((0, 128), jnp.int32)
    selected, published = _select(
        jnp.zeros((0, 32, 128), jnp.float8_e4m3fn),
        jnp.zeros((0, 32), jnp.float32),
        _zero_banks(layout),
        jnp.zeros((1, 1), jnp.int32),
        rows,
        candidates,
        rows,
        layout,
        candidate_write=True,
    )
    assert selected.shape == (0, 128)
    assert published.shape == (0, 128)


def test_million_key_topk_maps_only_winners_real_tpu():
    """The full mapping is too large for SparseCore VMEM; map only winners."""
    assert jax.default_backend() == "tpu"
    width, block, k = 1048576, 8, 512
    lengths = jnp.array([width - 3], jnp.int32)
    # The gathered scorer writes -inf outside each query window, and the
    # lax.top_k fallback relies on that rather than on valid_lengths.
    scores = jnp.where(
        jnp.arange(width) < lengths[0], jnp.arange(width, dtype=jnp.float32), -jnp.inf
    )[None, :]
    # Strictly increasing, noncontiguous logical blocks distinguish identity
    # from the actual compact->logical mapping, including at the final page.
    mapping = (jnp.arange(width // block, dtype=jnp.int32) * 2 + 5)[None, :]
    selected = jax.jit(
        lambda: _select_candidate_tokens(
            scores,
            mapping,
            k=k,
            block_size=block,
            unordered=True,
            valid_lengths=lengths,
        )
    )()
    slots = np.arange(width - 3 - k, width - 3, dtype=np.int32)
    expected = ((slots // block * 2 + 5) * block + slots % block)[None, :]
    np.testing.assert_array_equal(_sorted_ids(np.asarray(selected)), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
