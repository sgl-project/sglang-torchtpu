"""Unit tests for the TPU Pallas RPA v3 backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    _zero_inactive_rpa_rows,
)
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    _SCOPED_SMEM_BYTES,
    _get_pages_per_seq,
    _page_table_smem_bytes,
)


class TestPageTableSmemBudget(unittest.TestCase):
    def test_1m_context_default_batch_cap_exceeds_smem(self):
        # 1,052,672 tokens at page 64 -> 16,448 pages/seq; the default compile
        # cap of 64 sequences needs 4.2 MB, far above the 1 MiB scoped SMEM.
        pages = _get_pages_per_seq(1052672, 64)
        self.assertEqual(pages, 16448)
        self.assertGreater(_page_table_smem_bytes(64, pages), _SCOPED_SMEM_BYTES)
        self.assertLessEqual(_page_table_smem_bytes(8, pages), _SCOPED_SMEM_BYTES)

    def test_entries_padded_to_128(self):
        self.assertEqual(_page_table_smem_bytes(1, 1), 128 * 4)
        self.assertEqual(_page_table_smem_bytes(2, 100), 256 * 4)


class TestAttentionPallasRpaV3(unittest.TestCase):
    CONTEXT_LEN = 131072
    POOL_WIDTH = 262144
    PAGE_SIZE = 64
    DRAFT_TOKENS = 8

    def _make_backend(
        self,
        speculative_step_id=0,
        pool_width=None,
        req_to_token_device="cpu",
    ):
        pool_width = self.POOL_WIDTH if pool_width is None else pool_width
        override = get_context().override_server_args(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=self.PAGE_SIZE,
            model_config=SimpleNamespace(context_len=self.CONTEXT_LEN),
            server_args=server_args,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(
                    1,
                    pool_width,
                    dtype=torch.int32,
                    device=req_to_token_device,
                )
            ),
            token_to_kv_pool=SimpleNamespace(),
            is_hybrid_swa=False,
            kv_cache_dtype=torch.bfloat16,
        )
        with patch(
            "sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base."
            "_build_pallas_kernel_op",
            return_value=None,
        ):
            return TpuPallasRpaV3AttnBackend(
                model_runner, speculative_step_id=speculative_step_id
            )

    def _build_metadata(self, backend, forward_mode, spec_info, seq_len=None):
        seq_len = self.CONTEXT_LEN - 2 if seq_len is None else seq_len
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            forward_mode=forward_mode,
            extend_seq_lens=None,
            spec_info=spec_info,
        )
        return backend._build_metadata(forward_batch, non_empty_bs=1)

    def test_pages_per_seq_uses_context_plus_spec_headroom(self):
        backend = self._make_backend()
        # extra = max(4 + draft, 2*max(steps*topk, draft) + page - 1) = 79
        self.assertEqual(backend.max_context_len, self.CONTEXT_LEN + 79)
        self.assertEqual(
            (backend.max_context_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE,
            2050,
        )
        self.assertLess(backend.max_context_len, self.POOL_WIDTH)

    def test_rejects_context_capacity_larger_than_request_table(self):
        with self.assertRaisesRegex(
            ValueError, "exceeds the request-to-token table width"
        ):
            self._make_backend(pool_width=self.CONTEXT_LEN)

    def test_target_verify_bump_fits_rectangular_page_stride(self):
        backend = self._make_backend()
        metadata = self._build_metadata(
            backend,
            ForwardMode.TARGET_VERIFY,
            SimpleNamespace(num_tokens_per_req=self.DRAFT_TOKENS),
        )
        capacity = metadata.cu_kv_lens[1].item()
        self.assertEqual(metadata.kv_lens.item(), self.CONTEXT_LEN + 6)
        self.assertLessEqual(metadata.kv_lens.item(), capacity)
        self.assertEqual(metadata.page_indices.numel(), 2050)

    def test_target_verify_rejects_bump_above_page_stride(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(req_to_token_device=req_to_token_device)
                with self.assertRaisesRegex(
                    ValueError, "exceeds the per-request page-table capacity"
                ):
                    self._build_metadata(
                        backend,
                        ForwardMode.TARGET_VERIFY,
                        SimpleNamespace(num_tokens_per_req=self.DRAFT_TOKENS),
                        seq_len=131194,
                    )

    def test_multistep_decode_bump_fits_rectangular_page_stride(self):
        backend = self._make_backend(speculative_step_id=self.DRAFT_TOKENS - 1)
        metadata = self._build_metadata(
            backend,
            ForwardMode.DECODE,
            SimpleNamespace(),
        )
        capacity = metadata.cu_kv_lens[1].item()
        self.assertEqual(metadata.kv_lens.item(), self.CONTEXT_LEN + 6)
        self.assertLessEqual(metadata.kv_lens.item(), capacity)

    def test_multistep_decode_rejects_bump_above_page_stride(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(
                    speculative_step_id=self.DRAFT_TOKENS - 1,
                    req_to_token_device=req_to_token_device,
                )
                with self.assertRaisesRegex(
                    ValueError, "exceeds the per-request page-table capacity"
                ):
                    self._build_metadata(
                        backend,
                        ForwardMode.DECODE,
                        SimpleNamespace(),
                        seq_len=131194,
                    )

    def test_seq_lens_cpu_list_and_empty_inputs(self):
        backend = self._make_backend()

        # 1. seq_lens_cpu as a Python list
        fb_list = SimpleNamespace(
            seq_lens=torch.tensor([100], dtype=torch.int32),
            seq_lens_cpu=[100],
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_list = backend._build_metadata(fb_list, non_empty_bs=1)
        self.assertIsNotNone(meta_list)

        # 2. seq_lens_cpu as empty Tensor (e.g. idle rank forced into decode for DP collectives)
        fb_empty_tensor = SimpleNamespace(
            seq_lens=torch.empty(0, dtype=torch.int32),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.empty(0, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_empty_tensor = backend._build_metadata(fb_empty_tensor, non_empty_bs=0)
        self.assertIsNotNone(meta_empty_tensor)

        # 3. seq_lens_cpu as an empty list
        fb_empty_list = SimpleNamespace(
            seq_lens=torch.empty(0, dtype=torch.int32),
            seq_lens_cpu=[],
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.empty(0, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        meta_empty_list = backend._build_metadata(fb_empty_list, non_empty_bs=0)
        self.assertIsNotNone(meta_empty_list)


class TestInactiveRpaRows(unittest.TestCase):
    def test_zeros_only_inactive_nan_rows(self):
        output = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        live_rows = output[:2].clone()
        output[2:] = float("nan")

        output = _zero_inactive_rpa_rows(output, torch.tensor(2))

        self.assertTrue(torch.equal(output[:2], live_rows))
        self.assertTrue(torch.equal(output[2:], torch.zeros(2, 8)))

    def test_preserves_non_finite_live_values(self):
        output = torch.ones(3, 4)
        output[0, 0] = float("nan")
        output[2] = float("nan")

        output = _zero_inactive_rpa_rows(output, torch.tensor(2))

        self.assertTrue(torch.isnan(output[0, 0]))
        self.assertTrue(torch.equal(output[2], torch.zeros(4)))


if __name__ == "__main__":
    unittest.main()
