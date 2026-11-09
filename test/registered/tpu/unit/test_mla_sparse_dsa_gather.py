# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/test_dsa_gather.py
"""Byte-exact tests for the SparseCore dsa_gather kernel.

The gather is content-agnostic byte transport, so the host-side oracle is
direct indexing into the source byte arrays and every comparison is exact.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=25, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test, not a TorchTPU runtime process: select the
# physical backend before the first JAX import below.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from sglang_tpu.attention.pallas.mla_sparse import dsa_gather, kv_cache_utils

LKV_DIM = 512
PAGE_SIZE = 32
# GLM-5.3-Flash's top-k row width (topk_row_width): index_topk (2048) selected
# tokens plus up to index_kpool - 1 (3) tokens of the open pool, 2051, padded
# with -1 to a multiple of 128, the TPU vector lane count (the sparse MLA kernels
# lay a top-k row along lanes). A decode chunk of t tokens gathers t times this.
GLM53_FLASH_TOPK_ROW_WIDTH = -(-(2048 + 4 - 1) // 128) * 128  # 2176


class CsaGatherTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("dsa_gather requires a TPU (SparseCore).")
        self.rng = np.random.default_rng(42)

    def _make_caches(self, pages):
        """Random caches plus the plain arrays the oracle indexes into."""
        num_tokens = pages * PAGE_SIZE
        nope_bytes = self.rng.integers(0, 256, (num_tokens, LKV_DIM), np.uint8)
        rope_bytes = self.rng.integers(
            0, 256, (num_tokens, kv_cache_utils.TILE_LANE_BYTES), np.uint8
        )

        nope_cache = jnp.asarray(nope_bytes).reshape(
            pages, PAGE_SIZE, kv_cache_utils.WORD_BYTES, kv_cache_utils.TILE_LANE_BYTES
        )
        rope_cache = jnp.asarray(rope_bytes).reshape(
            pages,
            PAGE_SIZE // kv_cache_utils.WORD_BYTES,
            kv_cache_utils.WORD_BYTES,
            kv_cache_utils.TILE_LANE_BYTES,
        )
        return nope_cache, rope_cache, nope_bytes, rope_bytes

    def _check(
        self, nope_cache, rope_cache, nope_bytes, rope_bytes, indices, num_valid=None
    ):
        nope_out, rope_out = dsa_gather.dsa_gather(
            nope_cache,
            rope_cache,
            jnp.asarray(indices),
            num_valid_indices=None if num_valid is None else jnp.int32(num_valid),
        )
        n = len(indices)
        # Always exactly N rows, whether the kernel wrote them in place or
        # sliced block-padded outputs.
        self.assertEqual(nope_out.shape[0], n)
        self.assertEqual(rope_out.shape[0], n)
        # Rows past num_valid_indices are not written (callers mask them).
        live = n if num_valid is None else num_valid
        np.testing.assert_array_equal(
            np.asarray(nope_out).reshape(n, -1)[:live], nope_bytes[indices[:live]]
        )
        np.testing.assert_array_equal(
            np.asarray(rope_out)[:live], rope_bytes[indices[:live]]
        )

    @parameterized.named_parameters(
        # Counts that are not a multiple of a pipeline step (64 rows) take the
        # block-padded path and its slice; sampling with replacement exercises
        # duplicate indices.
        dict(testcase_name="random_with_duplicates", pages=8, n=1000),
        dict(testcase_name="small_odd", pages=2, n=17),
        dict(testcase_name="large", pages=16, n=8192),
    )
    def test_gather_matches_direct_indexing(self, pages, n):
        nope_cache, rope_cache, nope_bytes, rope_bytes = self._make_caches(pages)
        indices = self.rng.integers(0, pages * PAGE_SIZE, n).astype(np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_bytes, indices)

    @parameterized.named_parameters(
        # N a multiple of a pipeline step (64 rows) but not of the block: the
        # kernel writes exactly N rows and stops mid-block. These are sparse
        # MLA decode chunks of 1, 16 and 64 tokens.
        dict(testcase_name="decode_1_token", n=GLM53_FLASH_TOPK_ROW_WIDTH),
        dict(testcase_name="decode_16_tokens", n=16 * GLM53_FLASH_TOPK_ROW_WIDTH),
        dict(testcase_name="decode_64_tokens", n=64 * GLM53_FLASH_TOPK_ROW_WIDTH),
        # One pipeline step (4 streams x 16 lanes = 64 rows) past a full tpu7x
        # block (2 cores x 16 subcores x 16 lanes x 32 subchunks = 16384 rows).
        dict(testcase_name="one_step_past_block", n=16384 + 64),
    )
    def test_partial_last_block(self, n):
        pages = 16
        nope_cache, rope_cache, nope_bytes, rope_bytes = self._make_caches(pages)
        indices = self.rng.integers(0, pages * PAGE_SIZE, n).astype(np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_bytes, indices)

    def test_partial_last_block_with_fewer_live_rows(self):
        # A padded decode batch: only the first 5 of 16 tokens' rows are live,
        # so most subcores skip; the live rows still land exactly.
        pages = 16
        nope_cache, rope_cache, nope_bytes, rope_bytes = self._make_caches(pages)
        n = 16 * GLM53_FLASH_TOPK_ROW_WIDTH
        indices = self.rng.integers(0, pages * PAGE_SIZE, n).astype(np.int32)
        self._check(
            nope_cache,
            rope_cache,
            nope_bytes,
            rope_bytes,
            indices,
            num_valid=5 * GLM53_FLASH_TOPK_ROW_WIDTH,
        )

    def test_all_byte_lanes(self):
        """Every position of every page: covers all four rope byte lanes and
        every tile row, in permuted (non-contiguous) order."""
        pages = 4
        nope_cache, rope_cache, nope_bytes, rope_bytes = self._make_caches(pages)
        indices = self.rng.permutation(pages * PAGE_SIZE).astype(np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_bytes, indices)

    def test_single_repeated_index(self):
        """Maximal duplication: the gather-hotspot access pattern."""
        pages = 4
        nope_cache, rope_cache, nope_bytes, rope_bytes = self._make_caches(pages)
        indices = np.full(512, 77, np.int32)
        self._check(nope_cache, rope_cache, nope_bytes, rope_bytes, indices)


if __name__ == "__main__":
    unittest.main()
