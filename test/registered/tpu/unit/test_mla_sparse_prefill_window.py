# SPDX-License-Identifier: Apache-2.0
"""Windowed sparse MLA prefill (DSA attention token split) vs the per-query op.

For every simulated rank r and window w, ``sparse_mla_prefill_window``'s output
for (token in w, head of r) must equal ``sparse_mla_attention_fused``'s (rank
r, every token) bit for bit, with the same caches. Kimi-K3.1 TP32 shapes: 32
ranks x 3 heads, 4096-token bucket, top-k 2048, prefixes longer than it,
requests across windows, padded tokens, a last window past T.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

# Behind the default-off SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT: nightly,
# outside the stage-a budget.
register_tpu_ci(
    est_time=150,
    suite="nightly-tpu-4-v7",
    nightly=True,
    chips=1,
    labels=["singlehost", "kimi-k3"],
)

# Standalone raw-JAX kernel test: select the physical backend before the first
# JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.mla_sparse_fused import (
    sparse_mla_attention_fused,
    sparse_mla_prefill_window,
)

_WORLD = 32  # simulated attention-TP ranks
_HEADS = 3  # heads per rank (Kimi-K3.1: 96 heads at TP32)
_TOPK = 2048
_PAGE = 128
_SM_SCALE = 192**-0.5
_K_SCALE = 0.75


def _case(requests, num_tokens, seed):
    """One prefill step; ``requests`` = [(prefix, new tokens)]."""
    rng = np.random.default_rng(seed)
    num_seqs = 8
    pages_per_seq = max(-(-(p + q) // _PAGE) for p, q in requests) + 1
    num_pages = 1 + sum(-(-(p + q) // _PAGE) for p, q in requests) + 8
    cu = np.zeros(num_seqs + 1, np.int32)
    for s, (_, q) in enumerate(requests):
        cu[s + 1] = cu[s] + q
    cu[len(requests) + 1 :] = cu[len(requests)]
    assert cu[-1] <= num_tokens
    kv_lens = np.zeros(num_seqs, np.int32)
    pages = np.zeros((num_seqs, pages_per_seq), np.int32)
    pool, used = 1 + rng.permutation(num_pages - 1), 0
    rows = np.full((num_tokens, _TOPK), -1, np.int32)
    token = 0
    for s, (p, q) in enumerate(requests):
        kv_lens[s] = p + q
        n = -(-(p + q) // _PAGE)
        pages[s, :n] = pool[used : used + n]
        used += n
        for i in range(q):
            m = min(_TOPK, p + i + 1)
            rows[token, :m] = rng.choice(p + i + 1, size=m, replace=False)
            token += 1

    def fp8(*shape):
        return jnp.asarray(rng.normal(size=shape).astype(ml_dtypes.float8_e4m3fn))

    def finite_bytes(*shape):
        # Every slot holds finite FP8 (bytes below 0x70), as freed pages do.
        return jnp.asarray(rng.integers(0, 0x70, shape), jnp.uint8)

    def bf16(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

    return dict(
        ql_nope=bf16(_WORLD, num_tokens, _HEADS, 512),
        q_pe=bf16(_WORLD, num_tokens, _HEADS, 64),
        store=(
            fp8(num_tokens, 512),
            fp8(num_tokens, 64),
            finite_bytes(num_pages, _PAGE, 4, 128),
            finite_bytes(num_pages, _PAGE // 4, 4, 128),
        ),
        metadata=(
            jnp.asarray(kv_lens),
            jnp.asarray(pages.reshape(-1)),
            jnp.asarray(cu),
            jnp.asarray([0, 0, len(requests)], jnp.int32),
        ),
        rows=rows,
        real=int(cu[len(requests)]),
    )


class TestPrefillWindow(CustomTestCase):
    def _check(self, requests, num_tokens, seed):
        c = _case(requests, num_tokens, seed)
        # Rows per rank as the backend splits them: ceil(T / ranks), rounded to 8.
        window = -(-num_tokens // _WORLD)
        window = -(-window // 8) * 8
        padded = _WORLD * window
        real = c["real"]
        full, full_caches = [], None
        for r in range(_WORLD):  # each rank today: every token, its heads
            out, nope, rope = sparse_mla_attention_fused(
                c["ql_nope"][r],
                c["q_pe"][r],
                *c["store"],
                *c["metadata"],
                jnp.asarray(c["rows"]),
                sm_scale=_SM_SCALE,
                k_scale=_K_SCALE,
            )
            full.append(np.asarray(out[:real].astype(jnp.float32)))
            full_caches = full_caches or (np.asarray(nope), np.asarray(rope))
        kv_lens, page_indices, cu_q_lens, distribution = c["metadata"]
        pad = ((0, 0), (0, padded - num_tokens), (0, 0), (0, 0))
        ql_nope, q_pe = jnp.pad(c["ql_nope"], pad), jnp.pad(c["q_pe"], pad)
        rows = np.pad(c["rows"], ((0, padded - num_tokens), (0, 0)), constant_values=-1)
        split = []
        for w in range(_WORLD):  # the token split: one window, every head
            win = slice(w * window, (w + 1) * window)

            def every_head(x):
                return (
                    x[:, win].transpose(1, 0, 2, 3).reshape(window, _WORLD * _HEADS, -1)
                )

            out, nope, rope = sparse_mla_prefill_window(
                every_head(ql_nope),
                every_head(q_pe),
                *c["store"],
                kv_lens,
                page_indices,
                cu_q_lens,
                jnp.clip(cu_q_lens - w * window, 0, window),
                distribution,
                jnp.asarray(rows[win]),
                sm_scale=_SM_SCALE,
                k_scale=_K_SCALE,
            )
            np.testing.assert_array_equal(np.asarray(nope), full_caches[0])
            np.testing.assert_array_equal(np.asarray(rope), full_caches[1])
            out = np.asarray(out.astype(jnp.float32))
            split.append(out.reshape(window, _WORLD, _HEADS, 512))
        split = np.concatenate(split)[:real]
        for r in range(_WORLD):
            with self.subTest(T=num_tokens, rank=r):
                self.assertTrue(np.isfinite(full[r]).all())
                np.testing.assert_array_equal(split[:, r], full[r])

    def test_windows(self):
        # 3900 real tokens: a 5000-token prefix whose 2000 new tokens span 16
        # windows, a fresh request, a third with a prefix; T=4096 leaves
        # padded tokens in the last windows, T=4000 runs the last past T.
        requests = [(5000, 2000), (0, 1500), (3000, 400)]
        self._check(requests, 4096, seed=0)
        self._check(requests, 4000, seed=1)


if __name__ == "__main__":
    unittest.main()
