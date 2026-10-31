# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU Pallas batched-RPA attention backend (CPU-runnable)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import types
import unittest
from typing import Optional
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_batched import (
    PallasForwardMetadata,
    TpuPallasRpaBatchedAttnBackend,
)
from sglang_tpu.host_metadata_cache import HostMetadataCache


class _BackendUnderTest(TpuPallasRpaBatchedAttnBackend):
    """Backend instance that skips the JAX-requiring kernel-op registration."""

    def __init__(
        self,
        page_size: int,
        req_to_token: torch.Tensor,
        speculative_step_id: int = 0,
    ) -> None:
        AttentionBackend.__init__(self)
        self.device = req_to_token.device
        self.page_size = page_size
        self.speculative_step_id = speculative_step_id
        self.max_context_len = req_to_token.shape[1]
        self.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        self.forward_metadata = None
        self._op = None
        self._distribution_cache = {}
        self.decode_query_size = 1
        self._host_metadata_cache = HostMetadataCache()
        self._host_page_cols = None
        self._is_hybrid_swa = False
        self.token_to_kv_pool = None
        # No decode context parallelism: one page covers page_size tokens.
        self._dcp = 1
        self._dcp_rank = 0
        self._page_stride = page_size


def _make_forward_batch(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    forward_mode: ForwardMode,
    extend_seq_lens: Optional[torch.Tensor] = None,
    spec_info=None,
    seq_lens_cpu: Optional[torch.Tensor] = None,
    req_pool_indices_cpu: Optional[torch.Tensor] = None,
):
    if req_pool_indices_cpu is None:
        req_pool_indices_cpu = req_pool_indices.clone()

    return types.SimpleNamespace(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices_cpu,
        forward_mode=forward_mode,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=None,
        spec_info=spec_info,
        seq_lens_cpu=seq_lens_cpu,
    )


def _build_req_to_token(
    num_reqs: int, max_context_len: int, base_slot: int = 0
) -> torch.Tensor:
    """req_to_token[r, s] = base_slot + r * max_context_len + s (deterministic)."""
    return (
        base_slot
        + torch.arange(num_reqs).unsqueeze(1) * max_context_len
        + torch.arange(max_context_len).unsqueeze(0)
    ).to(torch.int32)


def _build_meta(backend, fb) -> PallasForwardMetadata:
    """Mirror ``init_forward_metadata`` without the KV-pool commit hook."""
    non_empty_bs = backend._non_empty_batch_size(fb)
    return backend._build_metadata(fb, non_empty_bs)


class TestBatchedRpaMetadataDecode(CustomTestCase):
    """Decode-mode metadata: q_len == 1 per sequence, distribution = [n, n, n]."""

    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 64

    def setUp(self):
        self.bs = 3
        self.req_to_token = _build_req_to_token(
            num_reqs=8, max_context_len=self.MAX_CONTEXT_LEN
        )
        self.backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        self.req_pool_indices = torch.tensor([5, 2, 7], dtype=torch.int64)
        self.seq_lens = torch.tensor([10, 24, 33], dtype=torch.int32)
        self.fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
        )

    def _build(self) -> PallasForwardMetadata:
        return _build_meta(self.backend, self.fb)

    def test_returns_pallas_forward_metadata(self):
        self.assertIsInstance(self._build(), PallasForwardMetadata)

    def test_kv_lens_matches_seq_lens(self):
        meta = self._build()
        self.assertTrue(torch.equal(meta.kv_lens, self.seq_lens.to(torch.int32)))
        self.assertEqual(meta.kv_lens.dtype, torch.int32)

    def test_cu_q_lens_decode_is_arange(self):
        meta = self._build()
        expected = torch.arange(self.bs + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected))
        self.assertEqual(meta.cu_q_lens.dtype, torch.int32)

    def test_cu_kv_lens_is_rectangular(self):
        meta = self._build()
        stride = self.PAGES_PER_SEQ * self.PAGE_SIZE
        expected = (torch.arange(self.bs + 1) * stride).to(torch.int32)
        self.assertTrue(torch.equal(meta.cu_kv_lens, expected))
        self.assertEqual(meta.cu_kv_lens.dtype, torch.int32)

    def test_page_indices_layout(self):
        meta = self._build()
        self.assertEqual(
            tuple(meta.page_indices.shape), (self.bs * self.PAGES_PER_SEQ,)
        )
        self.assertEqual(meta.page_indices.dtype, torch.int32)
        for i in range(self.bs):
            pool_idx = int(self.req_pool_indices[i])
            for j in range(self.PAGES_PER_SEQ):
                expected = (
                    int(self.req_to_token[pool_idx, j * self.PAGE_SIZE])
                    // self.PAGE_SIZE
                )
                got = int(meta.page_indices[i * self.PAGES_PER_SEQ + j])
                self.assertEqual(got, expected, f"seq {i} page {j} mismatch")

    def test_distribution_decode_all_active(self):
        meta = self._build()
        expected = torch.full((3,), self.bs, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.distribution, expected))
        self.assertEqual(meta.distribution.dtype, torch.int32)

    def test_no_swa_page_indices(self):
        self.assertIsNone(self._build().swa_page_indices)


class TestBatchedRpaMetadataExtend(CustomTestCase):
    """Extend-mode metadata: cu_q_lens from extend_seq_lens, distribution [0,0,n]."""

    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 64

    def setUp(self):
        self.bs = 3
        self.req_to_token = _build_req_to_token(
            num_reqs=8, max_context_len=self.MAX_CONTEXT_LEN
        )
        self.backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        self.req_pool_indices = torch.tensor([0, 3, 6], dtype=torch.int64)
        self.seq_lens = torch.tensor([12, 40, 20], dtype=torch.int32)
        self.extend_seq_lens = torch.tensor([12, 8, 20], dtype=torch.int32)
        self.fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=self.extend_seq_lens,
        )

    def _build(self) -> PallasForwardMetadata:
        return _build_meta(self.backend, self.fb)

    def test_cu_q_lens_is_cumsum_of_extend_seq_lens(self):
        meta = self._build()
        expected = torch.zeros(self.bs + 1, dtype=torch.int32)
        expected[1:] = torch.cumsum(self.extend_seq_lens, dim=0, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected))

    def test_distribution_extend_is_mixed_only(self):
        # Batched-specific: extend batches are encoded MIXED [0, 0, n], NOT
        # v3's [0, n, n]. The batched kernel never runs the pure-prefill range.
        meta = self._build()
        expected = torch.tensor([0, 0, self.bs], dtype=torch.int32)
        self.assertTrue(
            torch.equal(meta.distribution, expected),
            f"expected MIXED [0,0,{self.bs}], got {meta.distribution.tolist()}",
        )

    def test_kv_lens_is_full_seq_lens(self):
        meta = self._build()
        self.assertTrue(torch.equal(meta.kv_lens, self.seq_lens.to(torch.int32)))


class TestBatchedRpaDistributionEncoding(CustomTestCase):
    """Directly exercise the batched distribution encoder."""

    def _backend(self):
        return _BackendUnderTest(
            16, _build_req_to_token(num_reqs=4, max_context_len=64)
        )

    def test_decode_repeats_active_count(self):
        dist = self._backend()._make_rpa_distribution_cpu(
            is_decode=True, non_empty_bs=5
        )
        self.assertTrue(torch.equal(dist, torch.tensor([5, 5, 5], dtype=torch.int32)))

    def test_extend_is_mixed_only(self):
        dist = self._backend()._make_rpa_distribution_cpu(
            is_decode=False, non_empty_bs=5
        )
        self.assertTrue(torch.equal(dist, torch.tensor([0, 0, 5], dtype=torch.int32)))
        self.assertEqual(dist.dtype, torch.int32)


class TestBatchedRpaNonEmptyBatchSize(CustomTestCase):
    """`_non_empty_batch_size` clips DP-sync tail padding; interior zeros raise."""

    def _backend(self):
        return _BackendUnderTest(
            16, _build_req_to_token(num_reqs=8, max_context_len=64)
        )

    def test_no_padding_returns_full_count(self):
        fb = _make_forward_batch(
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        self.assertEqual(self._backend()._non_empty_batch_size(fb), 3)

    def test_trailing_zero_rows_are_clipped(self):
        fb = _make_forward_batch(
            seq_lens=torch.tensor([10, 20, 0, 0], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        self.assertEqual(self._backend()._non_empty_batch_size(fb), 2)

    def test_all_zero_rows_is_zero(self):
        fb = _make_forward_batch(
            seq_lens=torch.zeros(4, dtype=torch.int32),
            req_pool_indices=torch.arange(4, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        self.assertEqual(self._backend()._non_empty_batch_size(fb), 0)

    def test_interior_zero_row_raises(self):
        fb = _make_forward_batch(
            seq_lens=torch.tensor([10, 0, 30], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        with self.assertRaises(ValueError):
            self._backend()._non_empty_batch_size(fb)


class TestBatchedRpaKernelArguments(CustomTestCase):
    def test_attention_sink_reaches_the_kernel(self):
        from sglang_tpu import attention_pallas_rpa_batched

        kernel = Mock(return_value="out")
        sinks = object()
        with patch.object(
            attention_pallas_rpa_batched, "kv_layout", return_value="layout"
        ):
            attention_pallas_rpa_batched._invoke_batched_kernel(
                kernel,
                queries="q",
                keys="k",
                values="v",
                kv_cache_fused="cache",
                kv_lens="kv_lens",
                page_indices="pages",
                cu_q_lens="cu_q_lens",
                cu_kv_lens="cu_kv_lens",
                distribution="distribution",
                attention_sink=sinks,
                sm_scale=0.125,
                sliding_window=128,
                soft_cap=None,
                k_scale=None,
                v_scale=None,
            )
        self.assertIs(kernel.call_args.kwargs["attention_sink"], sinks)


class TestMixedVerifyTailsLast(CustomTestCase):
    """SGLANG_TPU_MIXED_VERIFY_TAILS_LAST: the tails stay last and no row moves."""

    def test_distribution_marks_the_trailing_tails(self):
        from sglang_tpu.attention_pallas_rpa_base import (
            _mixed_verify_tails_last_distribution,
        )

        d = _mixed_verify_tails_last_distribution(
            query_lens=(20, 4, 8, 8, 0), live=4, width=8
        )
        self.assertEqual(d.tolist(), [0, 2, 4])

    def _run(self, distribution, cu_q_lens, num_rows):
        import os

        import jax.numpy as jnp

        from sglang_tpu import attention_pallas_rpa_batched

        calls = []

        def kernel(q, k, v, cache, kv_lens, pages, cu, dist, **kwargs):
            calls.append((dist.tolist(), kwargs["prefill_block_sizes"] is not None))
            return jnp.full(q.shape, float(len(calls))), cache

        with (
            patch.object(
                attention_pallas_rpa_batched, "kv_layout", return_value="layout"
            ),
            patch.dict(os.environ, {"SGLANG_TPU_MIXED_VERIFY_TAILS_LAST": "1"}),
        ):
            out, _ = attention_pallas_rpa_batched._invoke_batched_kernel(
                kernel,
                queries=jnp.zeros((num_rows, 2, 4)),
                keys=None,
                values=None,
                kv_cache_fused="cache",
                kv_lens=None,
                page_indices=None,
                cu_q_lens=jnp.array(cu_q_lens, dtype=jnp.int32),
                cu_kv_lens=None,
                distribution=jnp.array(distribution, dtype=jnp.int32),
                attention_sink=None,
                sm_scale=1.0,
                sliding_window=None,
                soft_cap=None,
                k_scale=None,
                v_scale=None,
                decode_query_size=8,
            )
        return calls, out[:, 0, 0].tolist()

    def test_tails_run_in_the_verify_pass_and_prefill_in_the_rest(self):
        # Two prefill sequences (20 and 4 rows), then two 8-row tails, then 2 padding rows.
        calls, rows = self._run([0, 2, 4], [0, 20, 24, 32, 40], num_rows=42)
        self.assertEqual(calls, [([0, 2, 4], True), ([0, 0, 2], False)])
        self.assertEqual(rows[24:40], [1.0] * 16)
        self.assertEqual(rows[:24] + rows[40:], [2.0] * 26)

    def test_decode_ordered_verify_keeps_the_original_split(self):
        calls, rows = self._run([4, 4, 4], [0, 8, 16, 24, 32], num_rows=34)
        self.assertEqual(calls, [([0, 0, 4], True), ([0, 4, 4], False)])
        self.assertEqual(rows[:32], [1.0] * 32)


if __name__ == "__main__":
    unittest.main()
