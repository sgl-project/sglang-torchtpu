# SPDX-License-Identifier: Apache-2.0
"""Sparse FP8 index scoring: candidate gather and candidate scores vs a dense reference."""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=500,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.kernels.index_selection.fp8.cache import (
    quantize_index_rows,
    write_index_keys,
)
from sglang_tpu.kernels.index_selection.fp8.pages import cache_shape
from sglang_tpu.models.deepseek_v41.kernels.index_candidates.candidates import (
    gather_candidate_keys,
)
from sglang_tpu.models.deepseek_v41.kernels.index_candidates.selection import (
    prepare_candidate_keys,
    select_from_prepared_candidates,
)


def _sorted_ids(ids):
    """Valid IDs ascending with -1 at the tail, for unordered comparisons."""
    limit = np.iinfo(np.int32).max
    ids = np.sort(np.where(ids >= 0, ids, limit), axis=-1)
    return np.where(ids == limit, -1, ids)


def _candidate_fixture(counts, short, seed=371):
    rng = np.random.default_rng(seed)
    width, page_size, block, count, padding = 4096, 128, 8, 128, 3
    groups = len(counts)
    num_pages = groups * width // page_size
    keys = jnp.asarray(rng.normal(size=(num_pages * page_size, 128)), jnp.bfloat16)
    records = jnp.arange(keys.shape[0], dtype=jnp.int32)
    cache = jax.jit(lambda c: write_index_keys(c, records, keys))(
        jnp.zeros(cache_shape(num_pages, page_size, keys.shape[-1]), jnp.uint8)
    )
    pages = rng.permutation(num_pages).reshape(groups, -1).astype(np.int32)
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
        page_size=page_size,
        cache=cache,
        pages=pages,
        owners=owners,
        lengths=lengths,
        candidates=candidates,
        q=q,
        weights=weights,
        width=width,
        block=block,
    )


def _decode_keys(cache, records):
    page_size = cache.shape[-1]
    raw = cache[records // page_size, :, :, records % page_size]
    keys = raw[:, :-1].reshape(records.size, -1).view(jnp.float8_e4m3fn)
    scales = jax.lax.bitcast_convert_type(raw[:, -1], jnp.float32)
    return (keys.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)


def _reference_scores(f):
    """Dense FP32 scores from decoded keys, -inf outside the visible context."""
    page_size, width = f["page_size"], f["width"]
    position = np.arange(width)
    scores = np.full((f["owners"].size, width), -np.inf, np.float32)
    for group in range(f["pages"].shape[0]):
        rows = np.flatnonzero(f["owners"] == group)
        records = (
            f["pages"][group, position // page_size] * page_size + position % page_size
        )
        keys = _decode_keys(f["cache"], jnp.asarray(records))
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
@pytest.mark.parametrize("unordered", [False, True])
def test_candidate_scoring_matches_dense_reference_real_tpu(counts, unordered):
    assert jax.default_backend() == "tpu"
    f = _candidate_fixture(counts, short=False)
    k, block, width = 512, f["block"], f["width"]
    pages, owners, lengths, candidates = map(
        jnp.asarray, (f["pages"], f["owners"], f["lengths"], f["candidates"])
    )
    prepared = jax.jit(
        lambda: prepare_candidate_keys(
            f["cache"],
            pages,
            candidates,
            owners,
            block,
            width,
        )
    )()
    select = jax.jit(
        lambda q, w: select_from_prepared_candidates(
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


@pytest.mark.parametrize("page_size,head_dim", [(128, 128), (256, 128), (256, 64)])
def test_write_records_round_trip(page_size, head_dim):
    """Each written key decodes from its own slot; every other slot keeps its bytes."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 writes require TPU")
    # The scoring tests read keys back through the layout they were written
    # with, so only this case catches a key written to the wrong slot.
    rng = np.random.default_rng(29)
    pages = 6
    slots = pages * page_size
    # More keys than one 128-record write batch, shuffled across pages.
    records = rng.permutation(slots)[:300].astype(np.int32)
    records[::17] = -1
    records[5] = slots  # Past the cache: skipped like -1.
    keys = jnp.asarray(rng.normal(size=(records.size, head_dim)), jnp.bfloat16)
    prior = rng.integers(
        0, 256, cache_shape(pages, page_size, head_dim), dtype=np.uint8
    )
    cache = jax.jit(write_index_keys)(jnp.asarray(prior), jnp.asarray(records), keys)
    live = (records >= 0) & (records < slots)
    values, scales, _ = quantize_index_rows(keys)
    expected = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    decoded = _decode_keys(cache, jnp.asarray(records[live]))
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(expected)[live])
    untouched = np.ones(slots, bool)
    untouched[records[live]] = False
    untouched = untouched.reshape(pages, page_size)
    np.testing.assert_array_equal(
        np.asarray(cache).transpose(0, 3, 1, 2)[untouched],
        prior.transpose(0, 3, 1, 2)[untouched],
    )


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("fragments", [16, 2048, 131072])
def test_candidate_gather(page_size, fragments):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore execution requires TPU")
    # Random bytes expose any byte- or word-order slip in the int32 cache view.
    pages, per_page = 8, page_size // 8
    raw = np.random.default_rng(23).integers(
        0, 256, cache_shape(pages, page_size, 128), dtype=np.uint8
    )
    # Word [p, f, t] is bytes [p, f, :, t], little-endian.
    words = np.ascontiguousarray(raw.transpose(0, 1, 3, 2)).view(np.int32)[..., 0]
    ids = np.random.default_rng(19).integers(
        -1, pages * per_page, (3, fragments), dtype=np.int32
    )
    ids[:, ::7] = -1
    offsets = jnp.array([0, 3, 3, 3], jnp.int32)  # Two trailing empty groups.
    out = jax.jit(lambda b, p: gather_candidate_keys(b, p, offsets, block=8))(
        jnp.asarray(raw), jnp.asarray(ids)
    )
    assert out.shape == (3 * fragments // 16 * raw.shape[1], 128)
    out = np.asarray(out).reshape(3, fragments // 16, raw.shape[1], 128)
    # Only valid fragments of live groups are defined; skipped tiles and
    # ignored fragments are left uninitialized by the gather.
    for i, physical in enumerate(ids[0]):
        if physical < 0:
            continue
        page, fragment = divmod(int(physical), per_page)
        np.testing.assert_array_equal(
            out[0, i // 16, :, i % 16 * 8 : (i % 16 + 1) * 8],
            words[page, :, fragment * 8 : (fragment + 1) * 8],
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
