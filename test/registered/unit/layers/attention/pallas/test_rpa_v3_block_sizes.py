# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RPA v3 kernel's default block sizes on tpu7x.

Pure arithmetic: ``get_tpu_version`` reads ``jax.devices()[0]`` and is patched
out, so the file runs on a CPU-only host as long as JAX imports.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

import jax.numpy as jnp

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as rpa

MIB = 1024 * 1024
TPU7X_VMEM = 64 * MIB


def _block_sizes(*, num_q_heads, num_kv_heads, head_dim, kv_dtype, vmem_limit):
    """Mixed-case blocks for a 16k-token, 32k-context prefill at page 128."""
    with mock.patch.object(rpa, "get_tpu_version", return_value=7):
        return rpa.get_default_block_sizes(
            jnp.bfloat16,
            kv_dtype,
            num_q_heads,
            num_kv_heads,
            head_dim,
            128,
            16384,
            64,
            258,
            case=rpa.RpaCase.MIXED,
            vmem_limit_bytes=vmem_limit,
        )


class TestTpu7xPrefillBlocks(CustomTestCase):
    def test_qwen3_30b_prefill_gets_128_query_rows_per_block(self):
        """32 query heads over 4 KV heads, head_dim 128, fp8 KV, the whole
        64 MiB of VMEM: the query block settles at 128 rows with 2048-row KV
        blocks. Measured on tpu7x: 128 rows run a 16k-context stage forward
        28% faster than the 32 rows a 32 MiB limit with a 4x intermediate
        estimate produced, and 256 rows run no faster than 128."""
        blocks = _block_sizes(
            num_q_heads=32,
            num_kv_heads=4,
            head_dim=128,
            kv_dtype=jnp.float8_e4m3fn,
            vmem_limit=TPU7X_VMEM,
        )
        self.assertEqual(
            (blocks["bq_sz"], blocks["bkv_sz"], blocks["bkv_csz"]), (128, 2048, 512)
        )

    def test_a_256_row_query_block_exceeds_the_estimate_budget(self):
        """The same shape halves exactly once: 256 rows are over the 60%
        budget of the whole VMEM, 128 rows are within it."""
        est = rpa.get_vmem_estimate_bytes(
            4, 8, 128, 256, 2048, jnp.bfloat16, jnp.float8_e4m3fn, bkv_csz=512
        )
        self.assertGreater(est, int(TPU7X_VMEM * 0.60))
        est_128 = rpa.get_vmem_estimate_bytes(
            4, 8, 128, 128, 2048, jnp.bfloat16, jnp.float8_e4m3fn, bkv_csz=512
        )
        self.assertLessEqual(est_128, int(TPU7X_VMEM * 0.60))

    def test_twice_the_kv_heads_in_bf16_halve_the_query_block(self):
        """gpt-oss-120b's 64 query heads over 8 KV heads keep twice as many
        heads' intermediates live at once, and bf16 KV blocks take twice the
        VMEM of fp8 ones, so the query block settles at 64 rows. Measured on
        tpu7x: 64 rows run 8 to 13% faster than 32, 128 no faster than 64."""
        blocks = _block_sizes(
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=64,
            kv_dtype=jnp.bfloat16,
            vmem_limit=TPU7X_VMEM,
        )
        self.assertEqual((blocks["bq_sz"], blocks["bkv_sz"]), (64, 2048))


def _decode_block_sizes(*, num_q_heads, num_kv_heads, head_dim, kv_dtype, vmem_limit):
    """Decode-case blocks for a 32k-context batch at page 128."""
    with mock.patch.object(rpa, "get_tpu_version", return_value=7):
        return rpa.get_default_block_sizes(
            jnp.bfloat16,
            kv_dtype,
            num_q_heads,
            num_kv_heads,
            head_dim,
            128,
            256,
            256,
            258,
            case=rpa.RpaCase.DECODE,
            vmem_limit_bytes=vmem_limit,
        )


class TestTpu7xDecodeBlocks(CustomTestCase):
    def test_qwen3_30b_tp8_decode_keeps_8192_kv_rows_per_block(self):
        """One KV head of 4 query heads per core with fp8 KV and the whole
        64 MiB as the limit: the decode KV block stays at 8,192 rows, where it
        settled with half the capacity as the limit. Measured on tpu7x at 4K
        context with 32 concurrent requests: 16,384 rows run 14% slower per
        token."""
        blocks = _decode_block_sizes(
            num_q_heads=4,
            num_kv_heads=1,
            head_dim=128,
            kv_dtype=jnp.float8_e4m3fn,
            vmem_limit=TPU7X_VMEM,
        )
        self.assertEqual((blocks["bq_sz"], blocks["bkv_sz"]), (1, 8192))


if __name__ == "__main__":
    unittest.main()
