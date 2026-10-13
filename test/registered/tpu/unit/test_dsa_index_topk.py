# SPDX-License-Identifier: Apache-2.0
"""Checks the DSA indexer step on TPU against a NumPy reference.

``dsa_indexer_step`` is what one indexer layer runs on each forward pass:
quantize the new queries and keys, store the keys in the index-key cache, and
pick the top-k keys for each query. Each test below guards one way this can go
wrong without raising an error:

* The quantizer. A key's scale is stored as one byte. If the scale used to
  quantize is not exactly the power of two that byte stands for, every later
  read of the key uses a different scale.
* The cache write. Each key must go to the slot the page table gives its
  token, and nothing else may change. Padded batch rows must not be stored.
* The selection. Keys must be ranked by the indexer's score, and a query may
  only select keys at or before its own position.

The kernel computes scores with bf16 precision, so its picks can differ from
the reference near the cut-off. The tests therefore compare scores: no picked
key may score more than one rounding step below the k-th best score.

The metadata has the same layout as ``TpuPallasRpaV3AttnBackend._build_metadata``
produces, with the MLA backend's ``distribution``. Sizes are GLM-5.2's: 32
index heads, 128-wide keys, page size 128.

Usage:
    python test/registered/tpu/unit/test_dsa_index_topk.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax.numpy as jnp
import ml_dtypes
import msgspec
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.dsa_index_topk import (
    dsa_indexer_step,
    index_key_cache_shape,
)
from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

INDEX_HEAD_DIM = 128
_PAGE_SIZE = 128
_NUM_HEADS = 32
_TOPK = 256
_WEIGHT_SCALE = float(INDEX_HEAD_DIM**-0.5 * _NUM_HEADS**-0.5)
_FP8 = ml_dtypes.float8_e4m3fn
_FP8_MAX = 448.0
# Measured on v7x: a picked key scores at most 0.4% below the k-th best score,
# which is one bf16 rounding step. For comparison, a bug that drops the query
# scale from the weights gives picks 60% below.
_SCORE_SLACK = 0.02


def _quantize(x):
    """NumPy reference: FP8 values, a power-of-two scale, and the scale byte."""
    abs_max = np.abs(x).max(axis=-1, keepdims=True).astype(np.float64)
    # The scale is the smallest power of two that is >= abs_max / 448.
    mantissa, exponent = np.frexp(abs_max / _FP8_MAX)
    exponent = np.where(mantissa == 0.5, exponent - 1, exponent).clip(-126, 127)
    scale = np.exp2(exponent).astype(np.float32)
    values = (x.astype(np.float32) / scale).astype(_FP8)
    return values, scale, (exponent + 127).astype(np.uint8)


class _Step(msgspec.Struct, frozen=True):
    """Inputs for one step, the cache it must leave, and reference scores."""

    num_active_tokens: int
    query: np.ndarray  # [token_bucket, heads, 128] f32
    key: np.ndarray  # [token_bucket, 128] f32
    head_weights: np.ndarray  # [token_bucket, heads] f32
    cache: np.ndarray  # [num_pages, page // 4, 4, 256] u8, prefixes stored
    expected_cache: np.ndarray  # same, with the active new keys stored
    kv_lens: np.ndarray
    page_indices: np.ndarray
    cu_q_lens: np.ndarray
    distribution: np.ndarray
    # One array per real query token: the reference score of every key the
    # token may select, in position order.
    scores: list


def _build_step(*, lens, token_bucket, is_decode, padded_rows=0, seed=0):
    """Build a step. ``lens`` has one ``(q_len, kv_len)`` per real request;
    ``kv_len`` already counts the ``q_len`` new tokens."""
    rng = np.random.default_rng(seed)
    num_active = len(lens)
    pages_needed = [-(-kv_len // _PAGE_SIZE) for _, kv_len in lens]
    pages_per_seq = max(pages_needed) + 1
    # The allocator never hands out page 0. A few extra pages stay unused.
    num_pages = 1 + sum(pages_needed) + 3
    page_pool = 1 + rng.permutation(num_pages - 1)

    query = rng.normal(size=(token_bucket, _NUM_HEADS, INDEX_HEAD_DIM))
    key = rng.normal(size=(token_bucket, INDEX_HEAD_DIM))
    # Give each query head and each key a different magnitude. Then a bug that
    # ignores the query scale or the key scale changes the ranking.
    query *= np.exp2(rng.integers(-3, 4, size=(token_bucket, _NUM_HEADS, 1)))
    key *= np.exp2(rng.integers(-3, 4, size=(token_bucket, 1)))
    head_weights = rng.uniform(0.25, 1.75, size=(token_bucket, _NUM_HEADS))
    query, key, head_weights = (
        a.astype(np.float32) for a in (query, key, head_weights)
    )

    q_fp8, q_scale, _ = _quantize(query)
    k_fp8, k_scale, k_byte = _quantize(key)
    weights = head_weights * q_scale[..., 0] * np.float32(_WEIGHT_SCALE)

    shape = index_key_cache_shape(num_pages, _PAGE_SIZE, INDEX_HEAD_DIM)
    records = np.zeros((num_pages * _PAGE_SIZE, shape[-1]), np.uint8)
    expected = None
    page_table = np.zeros((num_active + padded_rows, pages_per_seq), np.int32)
    scores, used, token = [], 0, 0
    stored_new = []
    for seq, (q_len, kv_len) in enumerate(lens):
        pages = page_pool[used : used + pages_needed[seq]]
        used += pages_needed[seq]
        page_table[seq, : len(pages)] = pages
        positions = np.arange(kv_len)
        rows = pages[positions // _PAGE_SIZE] * _PAGE_SIZE + positions % _PAGE_SIZE

        prefix_len = kv_len - q_len
        prefix = rng.normal(size=(prefix_len, INDEX_HEAD_DIM)).astype(np.float32)
        p_fp8, p_scale, p_byte = _quantize(prefix)
        records[rows[:prefix_len], :INDEX_HEAD_DIM] = p_fp8.view(np.uint8)
        records[rows[:prefix_len], INDEX_HEAD_DIM] = p_byte[:, 0]
        stored_new.append((rows[prefix_len:], slice(token, token + q_len)))

        new = slice(token, token + q_len)
        dequantized = np.concatenate(
            [
                p_fp8.astype(np.float32) * p_scale,
                k_fp8[new].astype(np.float32) * k_scale[new],
            ]
        )
        for j in range(q_len):
            visible = dequantized[: prefix_len + j + 1]
            dots = q_fp8[token + j].astype(np.float32) @ visible.T
            scores.append((np.maximum(dots, 0) * weights[token + j][:, None]).sum(0))
        token += q_len

    expected = records.copy()
    for rows, new in stored_new:
        expected[rows, :INDEX_HEAD_DIM] = k_fp8[new].view(np.uint8)
        expected[rows, INDEX_HEAD_DIM] = k_byte[new, 0]

    # A padded row has one query token and reuses request 0's page table row.
    page_table[num_active:] = page_table[0]
    q_lens = [q_len for q_len, _ in lens] + [1] * padded_rows
    kv_lens = [kv_len for _, kv_len in lens] + [1] * padded_rows
    return _Step(
        num_active_tokens=token,
        query=query,
        key=key,
        head_weights=head_weights,
        cache=records.reshape(shape),
        expected_cache=expected.reshape(shape),
        kv_lens=np.array(kv_lens, np.int32),
        page_indices=page_table.reshape(-1),
        cu_q_lens=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        distribution=np.array(
            [num_active] * 3 if is_decode else [0, 0, num_active], np.int32
        ),
        scores=scores,
    )


def _run(step):
    rows, cache = dsa_indexer_step(
        jnp.asarray(step.query),
        jnp.asarray(step.key),
        jnp.asarray(step.head_weights),
        jnp.asarray(step.cache),
        jnp.asarray(step.kv_lens),
        jnp.asarray(step.page_indices),
        jnp.asarray(step.cu_q_lens),
        jnp.asarray(step.distribution),
        weight_scale=_WEIGHT_SCALE,
        topk=_TOPK,
    )
    return np.asarray(rows), np.asarray(cache)


class TestDsaIndexTopk(CustomTestCase):
    def _assert_selection(self, step, rows):
        self.assertEqual(rows.shape, (step.query.shape[0], _TOPK))
        self.assertEqual(rows.dtype, np.int32)
        for token, scores in enumerate(step.scores):
            num_causal = len(scores)
            picks = rows[token][rows[token] >= 0]
            num_expected = min(_TOPK, num_causal)
            # The picks come first, are all different, and are all visible.
            np.testing.assert_array_equal(rows[token][num_expected:], -1)
            self.assertEqual(len(set(picks.tolist())), num_expected, f"token {token}")
            self.assertLess(int(picks.max()), num_causal, f"token {token}")
            if num_causal <= _TOPK:
                continue
            kth_best = np.sort(scores)[::-1][_TOPK - 1]
            self.assertGreaterEqual(
                float(scores[picks].min()),
                float(kth_best) * (1 - _SCORE_SLACK),
                f"token {token} selected a key below the top-k cut",
            )

    def test_quantizer_scale_is_the_power_of_two_its_byte_names(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2048, INDEX_HEAD_DIM))
        x = (x * np.exp2(rng.integers(-20, 20, size=(2048, 1)))).astype(np.float32)
        # Edge cases: all zeros, a value exactly at the top of the FP8 range,
        # and the float32 values just above and just below it.
        x[0] = 0.0
        x[1], x[2], x[3] = 0.0, 0.0, 0.0
        x[1, 5] = _FP8_MAX * 4
        x[2, 7] = np.nextafter(np.float32(_FP8_MAX * 4), np.float32(np.inf))
        x[3, 9] = np.nextafter(np.float32(_FP8_MAX * 4), np.float32(0))

        values, scale, byte = (
            np.asarray(a) for a in quantize_fp8_ue8m0(jnp.asarray(x))
        )
        expected_values, expected_scale, expected_byte = _quantize(x)

        np.testing.assert_array_equal(byte, expected_byte)
        np.testing.assert_array_equal(scale, expected_scale)
        np.testing.assert_array_equal(
            values.view(np.uint8), expected_values.view(np.uint8)
        )
        self.assertEqual([int(b) - 127 for b in byte[1:4, 0]], [2, 3, 2])

    def test_extend_stores_keys_and_selects_by_score(self):
        # Three requests: one longer than top-k, one shorter, one with no
        # cached keys yet.
        step = _build_step(
            lens=[(48, 700), (48, 120), (32, 32)], token_bucket=128, is_decode=False
        )
        rows, cache = _run(step)

        np.testing.assert_array_equal(cache, step.expected_cache)
        self._assert_selection(step, rows)

    def test_decode_skips_padded_rows(self):
        # Two real requests in a batch of four rows. The two padded rows point
        # at request 0's pages, so storing their keys would overwrite request
        # 0's first key.
        step = _build_step(
            lens=[(1, 900), (1, 37)], token_bucket=4, is_decode=True, padded_rows=2
        )
        rows, cache = _run(step)

        np.testing.assert_array_equal(cache, step.expected_cache)
        self._assert_selection(step, rows)


if __name__ == "__main__":
    unittest.main()
