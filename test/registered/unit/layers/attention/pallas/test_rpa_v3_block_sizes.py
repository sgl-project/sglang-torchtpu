# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RPA v3 kernel's default block sizes on tpu7x.

Pure arithmetic: ``get_tpu_version`` reads ``jax.devices()[0]`` and is patched
out, so the file runs on a CPU-only host as long as JAX imports.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

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


class TestTpu7xVerifyBlocks(CustomTestCase):
    def test_verify_window_over_budget_shrinks_kv_instead_of_spinning(self):
        """A decode-shaped verify window pins the query tile at its row count,
        so an estimate still over budget must keep halving the KV tile; the
        loop once retried the pinned query tile forever. GLM-5.3-Flash's MHA
        full attention per TP8 rank (8 heads of 256) verifying 8 tokens."""
        with mock.patch.object(rpa, "get_tpu_version", return_value=7):
            blocks = rpa.get_default_block_sizes(
                jnp.bfloat16,
                jnp.bfloat16,
                8,
                8,
                256,
                128,
                8,
                1,
                82,
                case=rpa.RpaCase.DECODE,
                vmem_limit_bytes=TPU7X_VMEM,
                decode_query_size=8,
            )
        self.assertEqual(blocks["bq_sz"], 8)
        est = rpa.get_vmem_estimate_bytes(
            8,
            1,
            256,
            8,
            blocks["bkv_sz"],
            jnp.bfloat16,
            jnp.bfloat16,
            bkv_csz=blocks["bkv_csz"],
        )
        self.assertTrue(
            est <= int(TPU7X_VMEM * 0.15) or blocks["bkv_sz"] == 128, blocks
        )


def _fp8_single_kv_decode_blocks(
    *,
    page_size,
    pages_per_seq,
    decode_query_size,
    sliding_window=None,
    num_q_heads=6,
    max_num_seqs=64,
):
    """Decode-case blocks for 1 fp8 KV head with the whole tpu7x VMEM as limit."""
    with mock.patch.object(rpa, "get_tpu_version", return_value=7):
        blocks = rpa.get_default_block_sizes(
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            num_q_heads,
            1,
            128,
            page_size,
            512,
            max_num_seqs,
            pages_per_seq,
            case=rpa.RpaCase.DECODE,
            vmem_limit_bytes=TPU7X_VMEM,
            sliding_window=sliding_window,
            decode_query_size=decode_query_size,
        )
    return tuple(blocks[k] for k in ("bq_sz", "bkv_sz", "bq_csz", "bkv_csz"))


class TestTpu7xFp8SingleKvDecodeBlocks(CustomTestCase):
    def test_8_row_verify_full_attention_takes_8192_kv_rows(self):
        """6 query heads over 1 fp8 KV head verifying 8 rows: 8192-row DMA
        blocks in 4096-row compute tiles at pages 128 and 512, 8k to 1M
        context. Measured on tpu7x (rel bias on, bs 1-64): 3-12% faster than
        the VMEM-shrunk (4096, 4096) at 8k, 16-19% at 120k and 1M."""
        for page_size in (128, 512):
            for max_kv in (8192, 131072, 1048576):
                with self.subTest(page_size=page_size, max_kv=max_kv):
                    self.assertEqual(
                        _fp8_single_kv_decode_blocks(
                            page_size=page_size,
                            pages_per_seq=max_kv // page_size,
                            decode_query_size=8,
                        ),
                        (8, 8192, 8, 4096),
                    )

    def test_1_row_window_layer_takes_a_2_window_block(self):
        """1-row decode of a 1024-token window over 16+ sequences: a 2048-row
        DMA block in 1024-row compute tiles. Measured on tpu7x at 8k and 120k
        context, bs 16-64: 15-40% faster than one window-sized block (bs 1-8
        mixed, so small batches keep it)."""
        for page_size in (128, 512):
            with self.subTest(page_size=page_size):
                self.assertEqual(
                    _fp8_single_kv_decode_blocks(
                        page_size=page_size,
                        pages_per_seq=1048576 // page_size,
                        decode_query_size=1,
                        sliding_window=1024,
                    ),
                    (1, 2048, 1, 1024),
                )

    def test_unmeasured_shapes_keep_the_heuristic(self):
        """1-row full attention, 8-row window layers, 1-row window layers
        under 16 sequences, a context below one 8192-row block and a 4-head
        group keep the general heuristic."""
        cases = [
            (
                dict(page_size=512, pages_per_seq=2048, decode_query_size=1),
                (1, 8192, 1, 8192),
            ),
            (
                dict(
                    page_size=512,
                    pages_per_seq=2048,
                    decode_query_size=8,
                    sliding_window=1024,
                ),
                (8, 1024, 8, 1024),
            ),
            (
                dict(
                    page_size=512,
                    pages_per_seq=2048,
                    decode_query_size=1,
                    sliding_window=1024,
                    max_num_seqs=8,
                ),
                (1, 1024, 1, 1024),
            ),
            (
                dict(page_size=128, pages_per_seq=32, decode_query_size=8),
                (8, 4096, 8, 4096),
            ),
            (
                dict(
                    page_size=512,
                    pages_per_seq=2048,
                    decode_query_size=8,
                    num_q_heads=4,
                ),
                None,
            ),
        ]
        for kw, want in cases:
            with self.subTest(**kw):
                got = _fp8_single_kv_decode_blocks(**kw)
                if want is None:
                    self.assertNotEqual(got[1:4:2], (8192, 4096))
                else:
                    self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
