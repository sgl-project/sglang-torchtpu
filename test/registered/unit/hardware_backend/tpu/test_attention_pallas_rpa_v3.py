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
    TpuPallasRpaV3MultiStepDraftBackend,
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
            extend_seq_lens_cpu=None,
            spec_info=spec_info,
        )
        return backend._build_metadata(forward_batch, non_empty_bs=1)

    def _build_padded_decode_metadata(self, backend, active_bs):
        physical_bs = 128
        seq_lens = torch.full((physical_bs,), 100, dtype=torch.int32)
        req_pool_indices = torch.zeros(physical_bs, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices.clone(),
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            spec_info=None,
        )
        return backend._build_metadata(forward_batch, non_empty_bs=active_bs)

    def test_pages_per_seq_uses_context_plus_spec_headroom(self):
        backend = self._make_backend()
        # extra = max(4 + draft, 2*max(steps*topk, draft) + page - 1) = 79
        self.assertEqual(backend.max_context_len, self.CONTEXT_LEN + 79)
        self.assertEqual(
            (backend.max_context_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE,
            2050,
        )
        self.assertLess(backend.max_context_len, self.POOL_WIDTH)

    def test_fixed_offsets_do_not_cache_request_state(self):
        """A reused layout must still observe a new KV length and page owner."""
        backend = self._make_backend()
        first = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=63)
        backend.req_to_token_pool.req_to_token[0, 0] = 2 * self.PAGE_SIZE
        second = self._build_metadata(backend, ForwardMode.DECODE, None, seq_len=65)
        self.assertIs(first.cu_q_lens, second.cu_q_lens)
        self.assertIs(first.cu_kv_lens, second.cu_kv_lens)
        self.assertIs(first.active_num_tokens, second.active_num_tokens)
        self.assertEqual(first.kv_lens.tolist(), [63])
        self.assertEqual(second.kv_lens.tolist(), [65])
        self.assertEqual(first.page_indices[0].item(), 0)
        self.assertEqual(second.page_indices[0].item(), 2)

    def test_draft_steps_share_pages_only_within_one_forward(self):
        """Sharing pages across steps must preserve old outputs after row/SWA reuse."""
        first, second = (self._make_backend(speculative_step_id=i) for i in range(2))
        second.req_to_token_pool = first.req_to_token_pool
        pool = SimpleNamespace(
            tpu_commit_pending_loads=lambda: None,
            full_to_swa_index_mapping=torch.arange(1024, dtype=torch.int32),
        )
        for backend in (first, second):
            backend._is_hybrid_swa = True
            backend.token_to_kv_pool = pool
        draft = object.__new__(TpuPallasRpaV3MultiStepDraftBackend)
        draft.attn_backends = [first, second]
        table = first.req_to_token_pool.req_to_token
        table[0, 0] = self.PAGE_SIZE
        batch = SimpleNamespace(
            seq_lens=torch.tensor([63, 65, 0, 0]),
            seq_lens_cpu=torch.tensor([63, 65, 0, 0]),
            req_pool_indices_cpu=torch.zeros(4, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            spec_info=SimpleNamespace(),
            _tpu_real_b=2,
        )
        draft.init_forward_metadata(batch)
        old = first.forward_metadata
        self.assertIs(old.page_indices, second.forward_metadata.page_indices)
        self.assertIs(old.swa_page_indices, second.forward_metadata.swa_page_indices)
        self.assertEqual(old.kv_lens.tolist(), [64, 66, 1, 1])
        self.assertEqual(second.forward_metadata.kv_lens.tolist(), [65, 67, 2, 2])

        table[0, 0] = 3 * self.PAGE_SIZE
        pool.full_to_swa_index_mapping[3 * self.PAGE_SIZE] = 7 * self.PAGE_SIZE
        batch.seq_lens = batch.seq_lens_cpu = torch.tensor([90, 0])
        batch.req_pool_indices_cpu = torch.zeros(2, dtype=torch.int64)
        batch._tpu_real_b = 1
        draft.init_forward_metadata(batch)
        new = first.forward_metadata
        self.assertIs(new.page_indices, second.forward_metadata.page_indices)
        self.assertIsNot(new.page_indices, old.page_indices)
        self.assertEqual(old.page_indices[0].item(), 1)
        self.assertEqual(old.swa_page_indices[0].item(), 1)
        self.assertEqual(new.page_indices[0].item(), 3)
        self.assertEqual(new.swa_page_indices[0].item(), 7)
        self.assertEqual(new.cu_q_lens.tolist(), [0, 1, 2])
        self.assertEqual(new.active_num_tokens.item(), 1)

    def test_shared_draft_pages_still_validate_later_step_capacity(self):
        """A valid first step does not license a later step beyond the page stride."""
        first, second = (self._make_backend(speculative_step_id=i) for i in range(2))
        second.req_to_token_pool = first.req_to_token_pool
        for backend in (first, second):
            backend.token_to_kv_pool = SimpleNamespace(
                tpu_commit_pending_loads=lambda: None
            )
        draft = object.__new__(TpuPallasRpaV3MultiStepDraftBackend)
        draft.attn_backends = [first, second]
        capacity = (
            _get_pages_per_seq(first.max_context_len, self.PAGE_SIZE) * self.PAGE_SIZE
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([capacity - 1]),
            seq_lens_cpu=torch.tensor([capacity - 1]),
            req_pool_indices_cpu=torch.zeros(1, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            spec_info=SimpleNamespace(),
        )
        with self.assertRaisesRegex(
            ValueError, "exceeds the per-request page-table capacity"
        ):
            draft.init_forward_metadata(batch)

    def test_fused_draft_defers_lengths_but_preserves_capacity_validation(self):
        backend = self._make_backend(speculative_step_id=1)
        backend.token_to_kv_pool = SimpleNamespace(
            tpu_commit_pending_loads=lambda: None
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([63, 0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([63, 0]),
            req_pool_indices_cpu=torch.zeros(2, dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            spec_info=SimpleNamespace(),
        )
        backend.init_forward_metadata(batch, fused_draft=True)
        self.assertIs(backend.forward_metadata.kv_lens, batch.seq_lens)
        backend.init_forward_metadata(batch)
        self.assertEqual(backend.forward_metadata.kv_lens.dtype, torch.int32)
        self.assertEqual(backend.forward_metadata.kv_lens.tolist(), [65, 2])
        capacity = (
            _get_pages_per_seq(backend.max_context_len, self.PAGE_SIZE) * self.PAGE_SIZE
        )
        batch.seq_lens_cpu = torch.tensor([capacity - 1, 0])
        with self.assertRaisesRegex(
            ValueError, "exceeds the per-request page-table capacity"
        ):
            backend.init_forward_metadata(batch, fused_draft=True)

    def test_fixed_metadata_tracks_physical_and_live_widths(self):
        """Bucket reuse must not bake the preceding batch's live rows or width."""
        backend = self._make_backend()
        for bs, live, width in ((4, 4, 1), (4, 2, 1), (4, 2, 4), (8, 4, 4), (4, 0, 1)):
            with self.subTest(bs=bs, live=live, width=width):
                q, kv, active = backend._get_fixed_query_metadata(
                    bs=bs,
                    query_width=width,
                    non_empty_bs=live,
                    pages_per_seq=3,
                    page_size=64,
                    device="cpu",
                )
                self.assertEqual(q.tolist(), [i * width for i in range(bs + 1)])
                self.assertEqual(kv.tolist(), [i * 192 for i in range(bs + 1)])
                self.assertEqual(active.shape, ())
                self.assertEqual(active.item(), live * width)
                self.assertNotEqual(
                    active.untyped_storage().data_ptr(), q.untyped_storage().data_ptr()
                )

    def test_fixed_offsets_reused_across_live_batch_sizes(self):
        """Changing only the live prefix must not duplicate bucket offsets."""
        backend = self._make_backend()
        first = backend._get_fixed_query_metadata(
            bs=256,
            query_width=4,
            non_empty_bs=0,
            pages_per_seq=3,
            page_size=64,
            device="cpu",
        )
        for live in range(257):
            q, kv, active = backend._get_fixed_query_metadata(
                bs=256,
                query_width=4,
                non_empty_bs=live,
                pages_per_seq=3,
                page_size=64,
                device="cpu",
            )
            self.assertIs(q, first[0])
            self.assertIs(kv, first[1])
            self.assertEqual(active.item(), 4 * live)
        self.assertEqual(first[2].item(), 0)

    def test_active_scalar_reused_across_query_layouts(self):
        """Equal active-token counts can share storage across different layouts."""
        backend = self._make_backend()
        decode = backend._get_fixed_query_metadata(
            bs=256,
            query_width=1,
            non_empty_bs=256,
            pages_per_seq=3,
            page_size=64,
            device="cpu",
        )
        verify = backend._get_fixed_query_metadata(
            bs=128,
            query_width=4,
            non_empty_bs=64,
            pages_per_seq=3,
            page_size=64,
            device="cpu",
        )
        self.assertIs(decode[2], verify[2])
        self.assertEqual(verify[2].item(), 256)
        self.assertEqual(decode[0][-1].item(), 256)
        self.assertEqual(verify[0][-1].item(), 512)

    def test_extend_offsets_remain_request_dependent(self):
        """Same-bucket prefills with different query lengths cannot share offsets."""
        backend = self._make_backend()
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20]),
            seq_lens_cpu=torch.tensor([10, 20]),
            req_pool_indices_cpu=torch.tensor([0, 0]),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([2, 3]),
            extend_seq_lens_cpu=None,
            spec_info=None,
        )
        first = backend._build_metadata(batch, non_empty_bs=2)
        batch.extend_seq_lens = torch.tensor([3, 1])
        second = backend._build_metadata(batch, non_empty_bs=2)
        self.assertEqual(first.cu_q_lens.tolist(), [0, 2, 5])
        self.assertEqual(second.cu_q_lens.tolist(), [0, 3, 4])
        self.assertEqual(first.active_num_tokens.item(), 5)
        self.assertEqual(second.active_num_tokens.item(), 4)

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

    def test_active_num_tokens_owns_scalar_storage_for_padded_decode(self):
        for req_to_token_device in ("cpu", "meta"):
            with self.subTest(req_to_token_device=req_to_token_device):
                backend = self._make_backend(req_to_token_device=req_to_token_device)
                for active_bs in (128, 125):
                    with self.subTest(active_bs=active_bs):
                        metadata = self._build_padded_decode_metadata(
                            backend, active_bs
                        )

                        self.assertEqual(metadata.cu_q_lens.shape, (129,))
                        self.assertEqual(metadata.active_num_tokens.shape, ())
                        self.assertEqual(metadata.active_num_tokens.item(), active_bs)
                        self.assertNotEqual(
                            metadata.active_num_tokens.untyped_storage().data_ptr(),
                            metadata.cu_q_lens.untyped_storage().data_ptr(),
                        )

                        metadata.cu_q_lens[active_bs] = -1
                        self.assertEqual(metadata.active_num_tokens.item(), active_bs)


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
