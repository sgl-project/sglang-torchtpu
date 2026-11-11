# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 indexer ops on TPU against a float reference.

Keys go through the real writer onto sglang's page geometry (record
``full_loc // r``, indexer page = full page for ratio 2, full page p -> 2p,
2p + 1 for ratio 1). The reference scores dequantized Q and K with
``relu(q . k) * w`` summed over heads; selections must be the reference top-k
up to FP8/BF16 score ties, as logical compressed ids, valid prefix then -1.
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
)
from sglang_tpu.models.deepseek_v41.ops import indexer

FULL_PAGE, INDEX_PAGE, TOPK, BLOCK = 256, 128, 512, 8


def _fixture(ratio, counts, ends, seed=11, width=2048):
    """Requests with ``counts[i]`` query rows ending at compressed length
    ``ends[i]``, then two padding rows; keys written at their pool records."""
    rng = np.random.default_rng(seed)
    full_per_request = width * ratio // FULL_PAGE
    full_pages = len(counts) * full_per_request + 1  # page 0 is the null page
    pages = full_pages * FULL_PAGE // ratio // INDEX_PAGE
    layout = (pages, 33, 0, INDEX_PAGE)
    tables = rng.permutation(np.arange(1, full_pages)).reshape(len(counts), -1)
    j = np.arange(width)
    index_pages, records = [], []
    for table in tables:
        full_loc = table[j * ratio // FULL_PAGE] * FULL_PAGE + j * ratio % FULL_PAGE
        records.append(full_loc // ratio)
        index_pages.append(
            np.stack([2 * table, 2 * table + 1], -1).reshape(-1)
            if ratio == 1
            else table
        )
    index_pages = np.stack(index_pages).astype(np.int32)
    records = np.concatenate(records).astype(np.int32)
    keys = jnp.asarray(rng.normal(size=(records.size, 128)), jnp.bfloat16)
    index_k = jax.jit(
        lambda k: indexer._write_index_k(k, jnp.asarray(records), keys, *layout)
    )(jnp.zeros((pages * 33, 128), jnp.int32))

    requests, lengths = [], []
    for request, (count, end) in enumerate(zip(counts, ends)):
        # Causal rows: positions end*r - count .. end*r - 1.
        positions = end * ratio - count + np.arange(count)
        requests.append(np.full(count, request))
        lengths.append((positions + 1) // ratio)
    requests = np.concatenate(requests + [np.full(2, -1)]).astype(np.int32)
    lengths = np.concatenate(lengths + [np.full(2, width)]).astype(np.int32)
    tokens = requests.size
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(tokens, 32)) * 0.05, jnp.float32)
    return dict(
        layout=layout,
        index_k=index_k,
        index_pages=index_pages,
        requests=requests,
        lengths=lengths,
        q=q,
        weights=weights,
        width=width,
    )


def _reference_scores(f):
    """FP32 scores from dequantized Q and K, -inf past each row's length."""
    q8, q_scales, _ = quantize_index_rows(f["q"])
    q = np.asarray(q8.astype(jnp.float32)) * np.asarray(q_scales)[..., None]
    width = f["width"]
    j = np.arange(width)
    scores = np.full((f["requests"].size, width), -np.inf, np.float32)
    for row, (request, length) in enumerate(zip(f["requests"], f["lengths"])):
        if request < 0:
            continue
        logical = f["index_pages"][request][j // INDEX_PAGE] * INDEX_PAGE + j % 128
        keys = np.asarray(
            read_fp8_records(
                (f["index_k"],), jnp.asarray(logical), PackedFP8Layout(*f["layout"])
            ),
            np.float32,
        )
        dense = (np.maximum(keys @ q[row].T, 0) * np.asarray(f["weights"][row])).sum(-1)
        scores[row] = np.where(j < length, dense, -np.inf)
    return scores


def _select(f, candidates, **kwargs):
    return map(
        np.asarray,
        jax.jit(
            lambda: indexer._select(
                f["q"],
                f["weights"],
                f["index_k"],
                jnp.asarray(f["index_pages"]),
                jnp.asarray(f["requests"]),
                jnp.asarray(f["lengths"]),
                jnp.asarray(candidates),
                *f["layout"],
                TOPK,
                BLOCK,
                kwargs.get("publish", False),
                kwargs.get("consume", False),
            )
        )(),
    )


def _assert_top(selected, scores, allowed, k=TOPK):
    """Valid prefix then -1; ids allowed and unique; top-k up to score ties."""
    for row in range(selected.shape[0]):
        valid = selected[row] >= 0
        n = int(valid.sum())
        assert valid[:n].all() and not valid[n:].any(), row
        ids = selected[row, :n]
        live = np.flatnonzero(allowed[row])
        assert np.unique(ids).size == n and allowed[row, ids].all(), row
        assert n == min(k, live.size), (row, n, live.size)
        if n < live.size:
            rest = np.setdiff1d(live, ids)
            tolerance = 1e-2 * np.abs(scores[row, live]).max()
            assert scores[row, ids].min() >= scores[row, rest].max() - tolerance, row


def _visible(f):
    j = np.arange(f["width"])
    return (f["requests"] >= 0)[:, None] & (j[None] < f["lengths"][:, None])


# Ragged prefill (one request crosses top-k mid-chunk), mixed decode/verify,
# and a decode batch at the full score width.
SCENARIOS = {
    "prefill": ([300, 40], [900, 2048]),
    "decode": ([1, 1, 3, 1], [513, 2048, 1500, 700]),
}


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_dense_selection_matches_reference(ratio, scenario):
    assert jax.default_backend() == "tpu"
    f = _fixture(ratio, *SCENARIOS[scenario])
    tokens = f["requests"].size
    candidates = np.zeros((tokens, 0), np.int32)
    selected, blocks = _select(f, candidates)
    assert selected.shape == (tokens, TOPK) and blocks.shape == (tokens, 0)
    _assert_top(selected, _reference_scores(f), _visible(f))


@pytest.mark.parametrize("ratio", [1, 2])
def test_candidate_publish_then_consume(ratio):
    """The source publishes its best blocks (the newest always); a consumer
    selects the reference top-k restricted to those blocks."""
    assert jax.default_backend() == "tpu"
    f = _fixture(ratio, *SCENARIOS["prefill"])
    tokens, count = f["requests"].size, 128  # 1024 candidate tokens < width
    scores = _reference_scores(f)
    visible = _visible(f)
    selected, published = _select(
        f, np.full((tokens, count), -1, np.int32), publish=True
    )
    # The source's own selection is not filtered.
    _assert_top(selected, scores, visible)
    blocks = -(-f["lengths"] // BLOCK)
    padded = np.pad(scores, ((0, 0), (0, -f["width"] % BLOCK)), constant_values=-np.inf)
    block_max = padded.reshape(tokens, -1, BLOCK).max(-1)
    for row in range(tokens):
        chosen = published[row][published[row] >= 0]
        if f["requests"][row] < 0:
            assert chosen.size == 0, row
            continue
        assert blocks[row] - 1 in chosen, row
        assert chosen.size == min(count, blocks[row]), row
        assert np.unique(chosen).size == chosen.size and chosen.max() < blocks[row]
        rest = np.setdiff1d(np.arange(blocks[row] - 1), chosen)
        if rest.size:
            others = chosen[chosen != blocks[row] - 1]
            tolerance = 1e-2 * np.abs(block_max[row, : blocks[row]]).max()
            assert block_max[row, others].min() >= block_max[row, rest].max() - (
                tolerance
            ), row

    selected, passed = _select(f, published, consume=True)
    np.testing.assert_array_equal(passed, published)
    member = np.stack(
        [np.isin(np.arange(f["width"]) // BLOCK, p[p >= 0]) for p in published]
    )
    _assert_top(selected, scores, visible & member)


@pytest.mark.parametrize("ratio", [1, 2])
def test_wide_candidate_buffer_is_identity(ratio):
    """Flash's 2048 candidate blocks cover the whole 2048-wide context: the
    buffer passes through untouched and selection is unfiltered."""
    assert jax.default_backend() == "tpu"
    f = _fixture(ratio, *SCENARIOS["decode"])
    tokens = f["requests"].size
    buffer = np.random.default_rng(3).integers(-1, 256, (tokens, 2048), np.int32)
    for flags in ({"publish": True}, {"consume": True}):
        selected, passed = _select(f, buffer, **flags)
        np.testing.assert_array_equal(passed, buffer)
        _assert_top(selected, _reference_scores(f), _visible(f))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("flags", [{}, {"publish": True}, {"consume": True}])
def test_short_context_selects_every_visible_id(ratio, flags):
    """Every live length <= top-k: ids 0..length-1 exactly, even for a consumer
    whose candidates exclude them; a source publishes every visible block."""
    assert jax.default_backend() == "tpu"
    f = _fixture(ratio, [3, 1, 2], [512, 37, 300])
    tokens = f["requests"].size
    candidates = np.full((tokens, 128), -1, np.int32)
    candidates[:, 0] = 5
    selected, published = _select(f, candidates, **flags)
    expected = np.full((tokens, TOPK), -1, np.int32)
    for row, (request, length) in enumerate(zip(f["requests"], f["lengths"])):
        if request >= 0:
            expected[row, :length] = np.arange(length)
    np.testing.assert_array_equal(selected, expected)
    if flags.get("publish"):
        blocks = -(-f["lengths"] // BLOCK)
        for row in range(tokens):
            n = blocks[row] if f["requests"][row] >= 0 else 0
            np.testing.assert_array_equal(published[row, :n], np.arange(n))
            assert (published[row, n:] == -1).all(), row
    else:
        np.testing.assert_array_equal(published, candidates)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
