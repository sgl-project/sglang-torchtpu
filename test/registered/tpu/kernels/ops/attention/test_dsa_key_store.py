# SPDX-License-Identifier: Apache-2.0
"""Checks the Pallas DSA index-key writer against the XLA path, byte for byte.

``store_keys_pallas`` (decode-sized batches) must leave the index-key cache
exactly as ``quantize_fp8_ue8m0`` + ``_store_keys`` do: same records at the
same (page, row, byte), nothing else changed, padded rows not stored. Batches
cover one token per request, several tokens of a request sharing a cache row
(speculative verify / short extends), and padded requests.

Usage:
    python test/registered/tpu/kernels/ops/attention/test_dsa_key_store.py
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before JAX loads.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kernels.ops.attention.dsa_index_topk import (
    _store_keys,
    index_key_cache_shape,
)
from sglang_tpu.kernels.ops.attention.dsa_key_store import store_keys_pallas
from sglang_tpu.kernels.ops.attention.util import quantize_fp8_ue8m0

_PAGE = 128
_HEAD_DIM = 128
_NUM_PAGES = 512


def _xla_store(cache, key, kv, pg, cu, dist):
    k_fp8, _, scale_byte = quantize_fp8_ue8m0(key)
    return _store_keys(cache, k_fp8, scale_byte, kv, pg, cu, dist)


def _pallas_store(cache, key, kv, pg, cu, dist):
    return store_keys_pallas(cache, key, kv, pg, cu, dist)


def _batch(rng, num_tokens, q_lens, num_real):
    num_seqs = len(q_lens)
    pages_per_seq = min(72, _NUM_PAGES // num_seqs)
    q = np.array(q_lens, np.int32)
    prefix = rng.integers(0, pages_per_seq * _PAGE - q.max() - 1, num_seqs)
    kv = (prefix + q).astype(np.int32)
    cu = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    pages = rng.choice(_NUM_PAGES, num_seqs * pages_per_seq, replace=False)
    key = rng.standard_normal((num_tokens, _HEAD_DIM)) * rng.uniform(
        0.01, 30, (num_tokens, 1)
    )
    key[rng.integers(0, num_tokens)] = 0  # an all-zero key gets scale 1
    dist = np.array([0, num_real, num_real], np.int32)
    return (
        jnp.asarray(key, jnp.bfloat16),
        jnp.asarray(kv),
        jnp.asarray(pages.astype(np.int32)),
        jnp.asarray(cu),
        jnp.asarray(dist),
    )


class TestDsaKeyStore(CustomTestCase):
    def test_matches_xla_store(self):
        rng = np.random.default_rng(0)
        shape = index_key_cache_shape(_NUM_PAGES, _PAGE, _HEAD_DIM)
        xla = jax.jit(_xla_store)
        pallas = jax.jit(_pallas_store)
        cases = [
            (1, [1], 1),
            (4, [1, 1, 1, 1], 4),
            (4, [1, 1, 1, 1], 2),  # two padded requests
            (8, [2, 2, 2, 2], 4),  # tokens of a request share a row
            (16, [4, 4, 4, 4], 3),
            (8, [3, 5], 2),
            (16, [16], 1),
            (16, [1] * 16, 16),
            (16, [7, 9], 2),
            # A second random draw of the shared-row case.
            (8, [2, 2, 2, 2], 4),
        ]
        for num_tokens, q_lens, num_real in cases:
            with self.subTest(tokens=num_tokens, q_lens=q_lens, real=num_real):
                args = _batch(rng, num_tokens, q_lens, num_real)
                base = jnp.asarray(rng.integers(0, 256, shape, dtype=np.uint8))
                want = np.asarray(xla(base, *args))
                got = np.asarray(pallas(base, *args))
                np.testing.assert_array_equal(got, want)
                stored = (want != np.asarray(base)).any(-1).sum()
                self.assertEqual(stored, sum(q_lens[:num_real]))


if __name__ == "__main__":
    unittest.main()
