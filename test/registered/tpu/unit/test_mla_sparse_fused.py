# SPDX-License-Identifier: Apache-2.0
"""Compares the sparse MLA step with the dense MLA v2 kernel on TPU.

``pallas_tpu_mla_sparse`` uses the same forward code, metadata and fused FP8
pool as ``pallas_tpu_mla``. Only the kernel behind the op is different. So the
tests give both kernels the same step and check that:

* ``sparse_mla_attention_fused`` returns the same output as
  ``mla_ragged_paged_attention``, and
* both leave the same tokens in the cache.

The outputs can only match when the top-k rows contain every key a token may
attend to. That is what the indexer returns while a sequence is no longer
than top-k, so the tests build such rows.

Each test also compares the sparse step's cache, byte for byte, with a cache
computed in NumPy. This catches writes to slots that must not change. Batches
are padded to a fixed size, and a padded decode row still carries one query
token, which must not be stored.

The metadata has the same layout as ``TpuPallasRpaV3AttnBackend._build_metadata``
produces, with the MLA backend's ``distribution``. Sizes are GLM-5.2's at TP8:
8 query heads per rank, 512 latent + 64 rope dimensions, page size 128.

Usage:
    python test/registered/tpu/unit/test_mla_sparse_fused.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import msgspec
import numpy as np

from sglang.srt.layers.attention.pallas.mla_ragged_paged_attention_v2 import (
    mla_ragged_paged_attention,
)
from sglang.srt.layers.attention.pallas.mla_sparse_fused import (
    NOPE_DIM,
    ROPE_DIM,
    sparse_mla_attention_fused,
)
from sglang.test.test_utils import CustomTestCase

_PAGE_SIZE = 128
_NUM_HEADS = 8
_TOKEN_BYTES = 640  # 512 nope + rope padded to 128
_TOPK = 256
# GLM-5.2: (qk_nope_head_dim 192 + qk_rope_head_dim 64) ** -0.5.
_SM_SCALE = float((192 + ROPE_DIM) ** -0.5)
_FP8 = ml_dtypes.float8_e4m3fn
# What TpuPallasMlaAttnBackend passes in production.
_DECODE_BATCH_SIZE = 4

# Measured on v7x: the two kernels differ by at most 0.016, one bf16 step at
# this magnitude (mean 0.0009), while a top-k row off by one position in either
# direction differs by more than 3 (mean above 0.03).
_MAX_ABS_DIFF = 0.05
_MEAN_ABS_DIFF = 0.003


class _Step(msgspec.Struct, frozen=True):
    """One padded serving step on the fused cache, and the cache it must leave."""

    num_active_tokens: int
    k_scale: float
    ql_nope: np.ndarray  # [token_bucket, heads, 512] bf16
    q_pe: np.ndarray  # [token_bucket, heads, 64] bf16
    new_kv_c: np.ndarray  # [token_bucket, 512] fp8
    new_k_pe: np.ndarray  # [token_bucket, 64] fp8
    kv_lens: np.ndarray  # [seq_bucket] i32
    page_indices: np.ndarray  # [seq_bucket * pages_per_seq] i32
    cu_q_lens: np.ndarray  # [seq_bucket + 1] i32
    cu_kv_lens: np.ndarray  # [seq_bucket + 1] i32, page-table offsets
    distribution: np.ndarray  # [3] i32
    cache_before: np.ndarray  # fp8[pages, page_size // 4, 4, 640]
    cache_after: np.ndarray  # same, with the active tokens written
    live_rows: np.ndarray  # token slots of the active sequences, prefix included


def _cdiv(a, b):
    return (a + b - 1) // b


def _random_fp8(rng, shape, k_scale):
    values = rng.normal(size=shape).astype(np.float32)
    return (values / k_scale).astype(_FP8)


def _token_records(rng, num_tokens, k_scale):
    """Fused records: 512 nope bytes, 64 rope bytes, 64 zero bytes."""
    records = np.zeros((num_tokens, _TOKEN_BYTES), np.uint8)
    nope = _random_fp8(rng, (num_tokens, NOPE_DIM), k_scale)
    rope = _random_fp8(rng, (num_tokens, ROPE_DIM), k_scale)
    records[:, :NOPE_DIM] = nope.view(np.uint8)
    records[:, NOPE_DIM : NOPE_DIM + ROPE_DIM] = rope.view(np.uint8)
    return records, nope, rope


def _build_step(
    lens, *, padded_rows, token_bucket, is_decode, k_scale=1.0, seed=0
) -> _Step:
    """``lens`` is ``[(q_len, kv_len)]`` per active sequence, ``kv_len`` counting
    the ``q_len`` new tokens. ``padded_rows`` more rows follow, each with one
    query token, a stale ``kv_len`` and the first active sequence's page row:
    the worst a compiled batch can hand over."""
    rng = np.random.default_rng(seed)
    num_active = len(lens)
    seq_bucket = num_active + padded_rows
    needed = [_cdiv(kv_len, _PAGE_SIZE) for _, kv_len in lens]
    pages_per_seq = max(needed) + 1
    # Page 0 is the pool's padded-token slot and is never handed out.
    total_pages = 1 + sum(needed) + 8
    page_pool = 1 + rng.permutation(total_pages - 1)
    page_table = np.zeros((seq_bucket, pages_per_seq), np.int32)

    # Stale but valid FP8 everywhere, as pages freed by earlier requests hold.
    before, _, _ = _token_records(rng, total_pages * _PAGE_SIZE, k_scale)
    after = before.copy()

    q_lens = [q_len for q_len, _ in lens] + [1] * padded_rows
    kv_lens = [kv_len for _, kv_len in lens] + [77] * padded_rows
    ql_nope = rng.normal(size=(token_bucket, _NUM_HEADS, NOPE_DIM))
    q_pe = rng.normal(size=(token_bucket, _NUM_HEADS, ROPE_DIM))
    new_kv_c = _random_fp8(rng, (token_bucket, NOPE_DIM), k_scale)
    new_k_pe = _random_fp8(rng, (token_bucket, ROPE_DIM), k_scale)

    pages_used = 0
    token = 0
    live_rows = []
    for i, (q_len, kv_len) in enumerate(lens):
        pages = page_pool[pages_used : pages_used + needed[i]]
        pages_used += needed[i]
        page_table[i, : needed[i]] = pages
        positions = np.arange(kv_len)
        rows = pages[positions // _PAGE_SIZE] * _PAGE_SIZE + positions % _PAGE_SIZE
        live_rows.append(rows)

        prefix = kv_len - q_len
        records, _, _ = _token_records(rng, prefix, k_scale)
        before[rows[:prefix]] = records
        after[rows[:prefix]] = records
        step = slice(token, token + q_len)
        after[rows[prefix:], :NOPE_DIM] = new_kv_c[step].view(np.uint8)
        after[rows[prefix:], NOPE_DIM : NOPE_DIM + ROPE_DIM] = new_k_pe[step].view(
            np.uint8
        )
        after[rows[prefix:], NOPE_DIM + ROPE_DIM :] = 0
        token += q_len
    page_table[num_active:] = page_table[0]

    cache_shape = (total_pages, _PAGE_SIZE // 4, 4, _TOKEN_BYTES)
    return _Step(
        num_active_tokens=token,
        k_scale=k_scale,
        ql_nope=ql_nope.astype(ml_dtypes.bfloat16),
        q_pe=q_pe.astype(ml_dtypes.bfloat16),
        new_kv_c=new_kv_c,
        new_k_pe=new_k_pe,
        kv_lens=np.array(kv_lens, np.int32),
        page_indices=page_table.reshape(-1),
        cu_q_lens=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        cu_kv_lens=(np.arange(seq_bucket + 1) * pages_per_seq * _PAGE_SIZE).astype(
            np.int32
        ),
        # TpuPallasMlaAttnBackend: decode is [N, N, N], extend is [0, 0, N].
        distribution=np.array(
            [num_active] * 3 if is_decode else [0, 0, num_active], np.int32
        ),
        cache_before=before.view(_FP8).reshape(cache_shape),
        cache_after=after.view(_FP8).reshape(cache_shape),
        live_rows=np.concatenate(live_rows),
    )


def _causal_rows(step):
    """Top-k rows that select every key each token may attend to.

    The tokens of padded batch rows get rows too, because the real indexer
    produces them. Those rows point at slots of a real request (the padded row
    reuses its page table row), and they must have no effect.
    """
    rows = np.full((step.ql_nope.shape[0], _TOPK), -1, np.int32)
    for seq, kv_len in enumerate(step.kv_lens):
        start, end = step.cu_q_lens[seq], step.cu_q_lens[seq + 1]
        for token in range(start, end):
            num_causal = kv_len - (end - token) + 1
            rows[token, :num_causal] = np.arange(num_causal)
    return rows


def _stored_tokens(cache, rows):
    """The nope and rope bytes of ``rows``, without the padding either kernel
    is free to leave as it likes."""
    records = np.asarray(cache).view(np.uint8).reshape(-1, _TOKEN_BYTES)
    return records[rows, : NOPE_DIM + ROPE_DIM]


@unittest.skipUnless(
    any(d.platform == "tpu" for d in jax.devices()), "needs a TPU device"
)
class TestSparseMlaFused(CustomTestCase):
    def _run(self, lens, **step_kwargs):
        step = _build_step(lens, **step_kwargs)
        new_tokens = [
            jnp.asarray(step.ql_nope),
            jnp.asarray(step.q_pe),
            jnp.asarray(step.new_kv_c),
            jnp.asarray(step.new_k_pe),
        ]
        metadata = [
            jnp.asarray(step.kv_lens),
            jnp.asarray(step.page_indices),
            jnp.asarray(step.cu_q_lens),
        ]
        distribution = jnp.asarray(step.distribution)
        # The dense kernel donates its cache, so each call gets its own copy.
        dense_out, dense_cache = mla_ragged_paged_attention(
            *new_tokens,
            jnp.asarray(step.cache_before),
            *metadata,
            jnp.asarray(step.cu_kv_lens),
            distribution,
            sm_scale=_SM_SCALE,
            k_scale=step.k_scale,
            v_scale=step.k_scale,
            decode_batch_size=_DECODE_BATCH_SIZE,
            # By default it casts probabilities to the cache dtype, FP8 here,
            # which is an approximation of its own and four times this gap.
            p_same_dtype_as_v=False,
        )
        sparse_out, sparse_cache = sparse_mla_attention_fused(
            *new_tokens,
            jnp.asarray(step.cache_before),
            *metadata,
            distribution,
            jnp.asarray(_causal_rows(step)),
            sm_scale=_SM_SCALE,
            k_scale=step.k_scale,
        )
        dense_out, dense_cache, sparse_out, sparse_cache = jax.block_until_ready(
            (dense_out, dense_cache, sparse_out, sparse_cache)
        )

        # Nothing outside the active tokens moves, padded rows included.
        np.testing.assert_array_equal(
            np.asarray(sparse_cache).view(np.uint8),
            step.cache_after.view(np.uint8),
        )
        # Both kernels leave the same tokens behind for the next step to read.
        np.testing.assert_array_equal(
            _stored_tokens(sparse_cache, step.live_rows),
            _stored_tokens(dense_cache, step.live_rows),
        )
        self.assertEqual(sparse_out.shape, dense_out.shape)
        active = slice(0, step.num_active_tokens)
        diff = np.abs(
            np.asarray(sparse_out[active].astype(jnp.float32))
            - np.asarray(dense_out[active].astype(jnp.float32))
        )
        self.assertLess(diff.max(), _MAX_ABS_DIFF)
        self.assertLess(diff.mean(), _MEAN_ABS_DIFF)

    def test_extend_matches_dense_kernel(self):
        # Fresh prefill, a chunk continuing a prefix across a page boundary, and
        # a short extend, padded to the 128-token compile bucket.
        self._run(
            [(37, 37), (64, 200), (5, 133)],
            padded_rows=0,
            token_bucket=128,
            is_decode=False,
        )

    def test_decode_matches_dense_kernel_and_skips_padded_rows(self):
        # First token, one past a page boundary, and kv_len == topk; then two
        # padded rows whose page table aliases the first sequence.
        self._run(
            [(1, 1), (1, 129), (1, _TOPK)],
            padded_rows=2,
            token_bucket=5,
            is_decode=True,
            seed=1,
        )

    def test_extend_with_k_scale(self):
        self._run(
            [(64, 64), (64, 190)],
            padded_rows=0,
            token_bucket=128,
            is_decode=False,
            k_scale=0.5,
            seed=2,
        )


if __name__ == "__main__":
    unittest.main()
