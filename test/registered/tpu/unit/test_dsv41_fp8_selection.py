# SPDX-License-Identifier: Apache-2.0
"""Actual TPU selection regressions: no mocked scorer, gather, or selector.

Zero Q produces exactly tied zero scores over valid keys. This deliberately
checks streaming_topk's smaller-logical-ID tie contract without depending on
floating-point reduction tolerances or dense/compact query grouping.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=600,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.kernels.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    write_fp8_records,
)
from sglang_tpu.kernels.index_selection.selection import (
    prepare_selection,
    select_dense,
    select_prepared,
)


@pytest.mark.parametrize(
    "tokens", [1, 7, 48, 128, 193, 256, 512, 1024, 2048, 4096, 8192]
)
@pytest.mark.parametrize("short", [False, True])
def test_exact_per_query_candidates_real_tpu(tokens, short):
    assert jax.default_backend() == "tpu", "This test validates actual TPU kernels"
    width, groups, block, k, count = 2048, 4, 8, 512, 128
    layout = PackedFP8Layout(groups * width // 128)
    # Nontrivial page permutation makes accidental physical/logical ID confusion
    # visible even though score ties remain exact.
    pages = jnp.asarray(
        np.random.default_rng(19).permutation(layout.pages).reshape(groups, -1),
        jnp.int32,
    )
    banks = (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)
    keys = jnp.ones((layout.pages * 128, 128), jnp.bfloat16)
    banks = jax.jit(
        lambda b: write_fp8_records(
            b, jnp.arange(keys.shape[0], dtype=jnp.int32), keys, layout
        )
    )(banks)
    owners = np.resize(np.array([2, 0, 2, -1, 0, 1, 7], np.int32), tokens)
    lengths = np.resize(
        np.array(
            [129, 511, 256, 199, 97, 0, 512]
            if short
            else [1801, 1001, 1937, 1800, 1503, 0, 2048],
            np.int32,
        ),
        tokens,
    )
    # Two queries of owner 2 have intentionally disjoint publications. Reusing
    # the first query's candidate set for that request is observably incorrect.
    candidates = np.full((tokens, count), -1, np.int32)
    for row in range(tokens):
        first = 0 if (row // 2) % 2 == 0 else 128
        candidates[row] = np.arange(first, first + count)
    q = jnp.zeros((tokens, 32, 128), jnp.float8_e4m3fn)
    weights = jnp.ones((tokens, 32), jnp.float32)
    requests, visible, proposed = map(jnp.asarray, (owners, lengths, candidates))

    dense = jax.jit(
        lambda q, w: select_dense(
            q,
            w,
            banks,
            pages,
            requests,
            visible,
            proposed,
            layout=layout,
            score_width=width,
            topk=k,
            block_size=block,
            publish=True,
        )
    )
    dense_ids, published = map(np.asarray, dense(q, weights))
    expected_dense = np.full((tokens, k), -1, np.int32)
    expected_publication = np.full((tokens, count), -1, np.int32)
    for row, (owner, length) in enumerate(zip(owners, lengths)):
        if not 0 <= owner < groups or length == 0:
            continue
        n = min(length, k)
        expected_dense[row, :n] = np.arange(n)
        blocks = (length + block - 1) // block
        chosen = sorted(set(range(min(count - 1, blocks))) | {blocks - 1})
        expected_publication[row, : len(chosen)] = chosen
    np.testing.assert_array_equal(dense_ids, expected_dense)
    np.testing.assert_array_equal(published, expected_publication)

    prepare = jax.jit(
        lambda: prepare_selection(
            banks,
            pages,
            requests,
            visible,
            proposed,
            layout=layout,
            score_width=width,
            block_size=block,
            topk=k,
            max_queries_per_request=max(
                np.count_nonzero(owners == g) for g in range(groups)
            ),
        )
    )
    prepared = prepare()
    selected = jax.jit(
        lambda q, w: select_prepared(
            q,
            w,
            visible,
            prepared,
            score_width=width,
            topk=k,
            block_size=block,
        )
    )
    result = np.asarray(selected(q, weights))
    expected = np.full((tokens, k), -1, np.int32)
    for row, (owner, length) in enumerate(zip(owners, lengths)):
        if not 0 <= owner < groups:
            continue
        allowed = np.arange(length)
        if not short:
            allowed = allowed[np.isin(allowed // block, candidates[row])]
        chosen = allowed[:k]
        expected[row, : len(chosen)] = chosen
    np.testing.assert_array_equal(result, expected)
    # Reuse the already gathered union with a second indexer query tensor.
    np.testing.assert_array_equal(np.asarray(selected(q, weights * 2)), expected)


def test_empty_query_selection_real_tpu():
    assert jax.default_backend() == "tpu"
    q = jnp.zeros((0, 32, 128), jnp.float8_e4m3fn)
    w = jnp.zeros((0, 32), jnp.float32)
    rows = jnp.zeros((0,), jnp.int32)
    candidates = jnp.zeros((0, 128), jnp.int32)
    banks = (jnp.zeros((33, 128), jnp.int32),)
    pages = jnp.zeros((1, 1), jnp.int32)
    layout = PackedFP8Layout(1)
    ids, published = select_dense(
        q,
        w,
        banks,
        pages,
        rows,
        rows,
        candidates,
        layout=layout,
        score_width=128,
        publish=True,
    )
    prepared = prepare_selection(
        banks, pages, rows, rows, candidates, layout=layout, score_width=128
    )
    assert ids.shape == (0, 512)
    assert published.shape == (0, 128)
    assert select_prepared(q, w, rows, prepared, score_width=128).shape == (0, 512)


def test_random_prepared_selection_matches_masked_dense_scores_real_tpu():
    """Independent dense-cache scoring oracle, not a second prepared execution."""
    from sglang_tpu.kernels.index_selection.fp8_paged_mqa_logits import (
        fp8_paged_mqa_logits,
    )
    from sglang_tpu.kernels.index_selection.packed_fp8_cache import quantize_index_rows
    from sglang_tpu.kernels.index_selection.streaming_topk import streaming_topk

    assert jax.default_backend() == "tpu"
    rng = np.random.default_rng(371)
    tokens, groups, width, block, k = 7, 2, 4096, 8, 512
    layout = PackedFP8Layout(groups * width // 128)
    keys = jnp.asarray(rng.normal(size=(groups * width, 128)), jnp.bfloat16)
    banks = (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)
    banks = jax.jit(
        lambda b: write_fp8_records(
            b, jnp.arange(keys.shape[0], dtype=jnp.int32), keys, layout
        )
    )(banks)
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(groups, -1), jnp.int32)
    raw_q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.bfloat16)
    q, scales, _ = quantize_index_rows(raw_q)
    weights = jnp.asarray(rng.normal(size=(tokens, 32)), jnp.float32) * scales
    owners = np.array([1, 0, 1, 0, 1, 0, 1], np.int32)
    lengths = jnp.array([4001, 2703, 3309, 4096, 3001, 3503, 2307], jnp.int32)
    candidates = jnp.asarray(
        np.stack(
            [
                np.sort(rng.choice(width // block, 128, replace=False))
                for _ in range(tokens)
            ]
        ),
        jnp.int32,
    )
    requests = jnp.asarray(owners)
    prepared = jax.jit(
        lambda: prepare_selection(
            banks,
            pages,
            requests,
            lengths,
            candidates,
            layout=layout,
            score_width=width,
            block_size=block,
            topk=k,
            max_queries_per_request=4,
        )
    )()
    actual = np.asarray(
        jax.jit(
            lambda: select_prepared(
                q,
                weights,
                lengths,
                prepared,
                score_width=width,
                block_size=block,
                topk=k,
            )
        )()
    )

    # Both calls see the same stable request order, request boundaries, and
    # six-query tile bound. Only K placement differs; FP8 dot/head rounding
    # therefore has the same arithmetic grouping in dense and compact scoring.
    order = jnp.asarray(np.argsort(owners, kind="stable"))
    inverse = jnp.argsort(order)
    offsets = jnp.array([0, 3, 7], jnp.int32)

    def oracle():
        scores = fp8_paged_mqa_logits(
            q[order],
            banks,
            weights[order],
            lengths[order],
            pages,
            query_start_loc=offsets,
            max_model_len=width,
            q_scale=jnp.ones_like(weights),
            query_size=6,
            cache_layout=layout,
        )
        mask = jnp.zeros((tokens, width // block), jnp.bool_)
        mask = mask.at[jnp.arange(tokens)[:, None], candidates[order]].set(True)
        scores = jnp.where(jnp.repeat(mask, block, axis=1), scores, -jnp.inf)
        return streaming_topk(
            scores,
            valid_lengths=lengths[order],
            k=k,
            exclude_neg_inf=True,
            reconstruct_threshold=2048,
        )[inverse]

    expected = np.asarray(jax.jit(oracle)())
    # Exact ascending logical ID contract, including boundary ties; no overlap
    # threshold that could hide wrong candidate membership or cache addresses.
    np.testing.assert_array_equal(actual, expected)


def test_million_key_topk_maps_only_winners_real_tpu():
    """Full mapping is too large for SC VMEM; only winners should be gathered."""
    from sglang_tpu.kernels.index_selection.selection import _topk

    assert jax.default_backend() == "tpu"
    width, block, k = 1048576, 8, 512
    scores = jnp.arange(width, dtype=jnp.float32)[None, :]
    # Strictly increasing, noncontiguous logical blocks distinguish identity
    # from the actual compact->logical mapping, including at the final page.
    mapping = (jnp.arange(width // block, dtype=jnp.int32) * 2 + 5)[None, :]
    lengths = jnp.array([width - 3], jnp.int32)
    selected = np.asarray(jax.jit(lambda: _topk(scores, lengths, k, mapping))())
    slots = np.arange(width - 3 - k, width - 3, dtype=np.int32)
    expected = ((slots // block * 2 + 5) * block + slots % block)[None, :]
    np.testing.assert_array_equal(selected, expected)
