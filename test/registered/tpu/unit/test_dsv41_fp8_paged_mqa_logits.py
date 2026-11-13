# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/deepseek_v41/test_fp8_paged_mqa_logits.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP8 MQA logits across page sizes, padded widths and ragged lengths."""

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
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
)
from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)


def _pack(keys, layout):
    """Feature-major pages: field f of record r at row-word f * page_size + r."""
    values, scales, _ = quantize_index_rows(keys)
    bits = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
    bits = bits.astype(np.uint32).reshape(-1, 32, 4)
    words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
    fields = np.concatenate(
        (words.view(np.int32), np.asarray(scales).view(np.int32)[:, None]), axis=1
    )
    size, records = layout.page_size, np.arange(len(keys))
    words_at = (records // size * layout.page_stride * 128)[:, None]
    words_at = (
        words_at + np.arange(PAGE_ROWS)[None, :] * size + (records % size)[:, None]
    )
    pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
    pool[words_at] = fields
    dequantized = values.astype(jnp.float32) * scales[:, None]
    return pool.reshape(-1, 128), dequantized


@pytest.mark.parametrize(
    "page_size,max_model_len",
    [(128, 8192), (128, 50000), (128, 98816), (256, 8192), (256, 98816)],
)
def test_paged_mqa_logits_matches_reference(page_size, max_model_len):
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 scoring requires TPU")
    rng = np.random.default_rng(7)
    counts = np.array([6, 5, 6])  # A ragged request leaves a partial query tile.
    # Near-full, empty-to-short, and mid-width histories; one invalid page.
    bases = np.array([max_model_len - 5, 0, max_model_len // 2 - 3])
    lengths = np.concatenate([b + np.arange(c) for b, c in zip(bases, counts)])
    owners = np.repeat(np.arange(len(counts)), counts)
    per_request = -(-max_model_len // page_size)
    layout = PackedFP8Layout(len(counts) * per_request, page_size=page_size)
    table = rng.permutation(layout.pages).reshape(len(counts), per_request)
    table[2, 1] = -1
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    pool, dequantized = _pack(keys, layout)
    split = int(layout.pages * 0.6) * layout.page_stride
    tokens = len(lengths)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)

    result = fp8_paged_mqa_logits(
        q,
        (jnp.asarray(pool[:split]), jnp.asarray(pool[split:])),
        weights,
        jnp.asarray(lengths, jnp.int32),
        jnp.asarray(table, jnp.int32),
        query_start_loc=jnp.asarray(
            np.concatenate(([0], np.cumsum(counts))), jnp.int32
        ),
        max_model_len=max_model_len,
        q_scale=jnp.ones_like(weights),
        query_size=6,
        cache_layout=layout,
    )
    scores = np.asarray(result)

    position = np.arange(max_model_len)
    for request in range(len(counts)):
        rows = owners == request
        pages = table[request][position // page_size]
        logical = dequantized[np.maximum(pages, 0) * page_size + position % page_size]
        logits = jnp.einsum(
            "qhd,kd->qhk",
            q[rows].astype(jnp.float32),
            logical,
            precision=jax.lax.Precision.HIGHEST,
        )
        expected = np.asarray(
            (jnp.maximum(logits, 0) * weights[rows][..., None]).sum(1)
        )
        visible = (position[None] < lengths[rows][:, None]) & (pages >= 0)[None]
        actual = scores[rows]
        np.testing.assert_array_equal(np.isneginf(actual), ~visible)
        np.testing.assert_allclose(
            actual[visible], expected[visible], rtol=2e-2, atol=2e-2 * expected.max()
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
