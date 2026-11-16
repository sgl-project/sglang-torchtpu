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

from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    quantize_index_rows,
    read_fp8_records,
    write_fp8_records,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    gather_candidate_keys,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.selection import (
    prepare_fp8_candidates,
    select_fp8_prepared,
)


def _layout_args(layout):
    return (layout.pages, layout.page_stride, layout.page_offset, layout.page_size)


def _zero_banks(layout):
    return (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)


def _sorted_ids(ids):
    """Valid IDs ascending with -1 at the tail, for unordered comparisons."""
    limit = np.iinfo(np.int32).max
    ids = np.sort(np.where(ids >= 0, ids, limit), axis=-1)
    return np.where(ids == limit, -1, ids)


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
@pytest.mark.parametrize("unordered", [False, True])
def test_candidate_scoring_matches_dense_reference_real_tpu(counts, unordered):
    assert jax.default_backend() == "tpu"
    f = _candidate_fixture(counts, short=False)
    k, block, width = 512, f["block"], f["width"]
    pages, owners, lengths, candidates = map(
        jnp.asarray, (f["pages"], f["owners"], f["lengths"], f["candidates"])
    )
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
