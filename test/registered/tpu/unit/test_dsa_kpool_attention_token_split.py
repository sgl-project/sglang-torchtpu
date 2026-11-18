# SPDX-License-Identifier: Apache-2.0
"""DSA attention token split with the pooled (kpool) indexer, against the unsplit path.

One device simulates TP8 with GLM-5.3-Flash's DSA shapes (64 heads, no rope,
index kpool 4, top-k 2048, 2176-wide rows, gather chunk 256). For a prefill
chunk that scores against an earlier chunk's pools, every rank r:

* expands its window of picked pools (``kpool_expand_rows`` at its
  ``row_start``) into exactly rows ``[r * R, (r + 1) * R)`` of the unsplit
  indexer's rows;
* for a layer whose rows leave it (MTP top-k sharing), puts every rank's pools
  together into the unsplit rows and cuts its window from them, with the same
  result;
* attends its window with every head (``sparse_mla_prefill_window``) to the
  unsplit per-query op's output for those tokens and its heads, bit for bit,
  storing the same latent caches.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

# Behind the default-off SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT: nightly,
# outside the stage-a budget.
register_tpu_ci(
    est_time=120,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "glm-5.3-flash", "tp"],
)

# Standalone raw-JAX kernel test: select the physical backend before the first
# JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_kpool_topk as kp
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    QBLOCK_GATHER_CHUNK_TOKENS,
    sparse_mla_attention_fused,
    sparse_mla_prefill_window,
)

_WORLD = 8  # attention-TP ranks
_HEADS = 8  # attention heads per rank (64 at TP8)
_INDEX_HEADS, _D = 32, 128  # indexer heads and head dim
_KPOOL, _TOPK, _WIDTH = 4, 2048, 2176
_PAGE = 128
_NUM_SEQS = 4  # two requests and two padding slots
_PAGES_PER_SEQ = 32
_SM_SCALE = 256**-0.5
_K_SCALE = 0.75


def _split_rows(num_tokens):
    """Rows per rank as the backend splits them: ceil(T / ranks), rounded to 8."""
    rows = -(-num_tokens // _WORLD)
    return -(-rows // 8) * 8


class TestKpoolAttentionTokenSplit(CustomTestCase):
    def _indexer(self, rng, new, num_tokens):
        """Two prefill chunks of ``new`` tokens per request through the pooled
        indexer; returns the second chunk's inputs, metadata and unsplit rows."""

        def f(*shape, dtype=jnp.bfloat16):
            return jnp.asarray(rng.standard_normal(shape), jnp.float32).astype(dtype)

        pages = np.zeros((_NUM_SEQS, _PAGES_PER_SEQ), np.int32)
        pages[: len(new)] = 1 + np.arange(len(new) * _PAGES_PER_SEQ).reshape(
            len(new), _PAGES_PER_SEQ
        )
        num_pages = len(new) * _PAGES_PER_SEQ + 1
        raw = jnp.zeros((num_pages, _PAGE, 2 * _D), jnp.bfloat16)
        pooled = jnp.zeros(
            (num_pages, _PAGE // 16, 4, kp.pooled_record_bytes(_D)), jnp.uint8
        )
        cu = np.zeros(_NUM_SEQS + 1, np.int32)
        cu[1 : len(new) + 1] = np.cumsum(new)
        cu[len(new) + 1 :] = cu[len(new)]
        meta = None

        def step(args, **kw):
            return jax.jit(
                lambda *a: kp.kpool_indexer_step(
                    *a,
                    weight_scale=1.0,
                    topk=_TOPK,
                    kpool=_KPOOL,
                    out_width=_WIDTH,
                    **kw,
                )
            )(*args)

        for chunk in range(2):
            kv_lens = np.zeros(_NUM_SEQS, np.int32)
            kv_lens[: len(new)] = [(chunk + 1) * n for n in new]
            meta = (
                jnp.asarray(kv_lens),
                jnp.asarray(pages.reshape(-1)),
                jnp.asarray(cu),
                jnp.asarray([0, 0, len(new)], jnp.int32),
            )
            args = [
                f(num_tokens, _INDEX_HEADS, _D),
                f(num_tokens, _D),
                f(num_tokens, _D),
                f(num_tokens, _INDEX_HEADS, dtype=jnp.float32),
                f(_KPOOL, _D, dtype=jnp.float32),
                raw,
                pooled,
                *meta,
            ]
            full, raw_next, pooled_next = step(args)
            if chunk == 0:
                raw, pooled = raw_next, pooled_next
        return args, meta, np.asarray(full), num_pages, step, (raw_next, pooled_next)

    def _check(self, new, num_tokens, seed):
        rng = np.random.default_rng(seed)
        args, meta, full, num_pages, step, caches = self._indexer(rng, new, num_tokens)
        kv_lens, page_indices, cu_q_lens, distribution = meta
        rows = _split_rows(num_tokens)
        padded = _WORLD * rows
        full_padded = np.pad(
            full, ((0, padded - num_tokens), (0, 0)), constant_values=-1
        )
        real = int(sum(new))
        # Every row of a real token past the first pools selects history.
        self.assertTrue((full[:real, 0] >= 0).all())
        self.assertTrue((full[real:] == -1).all())

        # Indexer: each rank's window, local and from the whole rows.
        window_rows, picked = [], []
        for r in range(_WORLD):
            row_start = jnp.asarray([r * rows], jnp.int32)
            local, raw_r, pooled_r = step(args, row_start=row_start, row_count=rows)
            # Every rank stores every token's raw keys and closes every pool.
            np.testing.assert_array_equal(np.asarray(raw_r), np.asarray(caches[0]))
            np.testing.assert_array_equal(np.asarray(pooled_r), np.asarray(caches[1]))
            picked.append(np.asarray(local))
            win = np.asarray(
                kp.kpool_expand_rows(
                    local,
                    kv_lens,
                    cu_q_lens,
                    distribution,
                    row_start,
                    topk=_TOPK,
                    kpool=_KPOOL,
                    out_width=_WIDTH,
                )
            )
            with self.subTest(T=num_tokens, rank=r, rows="window"):
                np.testing.assert_array_equal(
                    win, full_padded[r * rows : (r + 1) * rows]
                )
            window_rows.append(win)
        whole = np.asarray(
            kp.kpool_expand_rows(
                jnp.asarray(np.concatenate(picked)[:num_tokens]),
                kv_lens,
                cu_q_lens,
                distribution,
                topk=_TOPK,
                kpool=_KPOOL,
                out_width=_WIDTH,
            )
        )
        # The layer whose rows leave it (MTP top-k sharing): whole rows equal
        # the unsplit rows, so its window cut equals the window rows above.
        np.testing.assert_array_equal(whole, full)

        # Attention: unsplit per-query op per rank vs every rank's window.
        def fp8(*shape):
            return jnp.asarray(rng.normal(size=shape).astype(ml_dtypes.float8_e4m3fn))

        ql_nope = jnp.asarray(
            rng.normal(size=(_WORLD, num_tokens, _HEADS, 512)), jnp.bfloat16
        )
        q_pe = jnp.zeros((num_tokens, _HEADS, 64), jnp.bfloat16)
        store = (
            fp8(num_tokens, 512),
            jnp.zeros((num_tokens, 64), ml_dtypes.float8_e4m3fn),
            # Finite FP8 bytes (below 0x70) in every slot, as freed pages hold.
            jnp.asarray(rng.integers(0, 0x70, (num_pages, _PAGE, 4, 128)), jnp.uint8),
            jnp.zeros((num_pages, _PAGE // 4, 4, 128), jnp.uint8),
        )
        opts = dict(
            sm_scale=_SM_SCALE,
            k_scale=_K_SCALE,
            gather_chunk_tokens=QBLOCK_GATHER_CHUNK_TOKENS,
            zero_rope=True,
        )
        unsplit, unsplit_caches = [], None
        for r in range(_WORLD):
            out, nope, rope = sparse_mla_attention_fused(
                ql_nope[r], q_pe, *store, *meta, jnp.asarray(full), **opts
            )
            unsplit.append(np.asarray(out[:real].astype(jnp.float32)))
            unsplit_caches = unsplit_caches or (np.asarray(nope), np.asarray(rope))
        pad = ((0, 0), (0, padded - num_tokens), (0, 0), (0, 0))
        ql_all = jnp.pad(ql_nope, pad)
        split = []
        for w in range(_WORLD):
            win = slice(w * rows, (w + 1) * rows)
            ql_w = ql_all[:, win].transpose(1, 0, 2, 3).reshape(rows, -1, 512)
            out, nope, rope = sparse_mla_prefill_window(
                ql_w,
                jnp.zeros((rows, _WORLD * _HEADS, 64), jnp.bfloat16),
                *store,
                kv_lens,
                page_indices,
                cu_q_lens,
                jnp.clip(cu_q_lens - w * rows, 0, rows),
                distribution,
                jnp.asarray(window_rows[w]),
                **opts,
            )
            np.testing.assert_array_equal(np.asarray(nope), unsplit_caches[0])
            np.testing.assert_array_equal(np.asarray(rope), unsplit_caches[1])
            split.append(
                np.asarray(out.astype(jnp.float32)).reshape(rows, _WORLD, _HEADS, 512)
            )
        split = np.concatenate(split)[:real]
        for r in range(_WORLD):
            with self.subTest(T=num_tokens, rank=r, out="attention"):
                self.assertTrue(np.isfinite(unsplit[r]).all())
                np.testing.assert_array_equal(split[:, r], unsplit[r])

    def test_windows(self):
        # Two requests whose second chunk scores against the first's pools
        # (750 pools > 512 picks for the first), padding slots and tokens;
        # T=2000 leaves the last window past T.
        self._check((1500, 400), 2048, seed=0)
        self._check((1500, 400), 2000, seed=1)


if __name__ == "__main__":
    unittest.main()
