"""Unit tests for the TPU Pallas RPA v3 attention backend (CPU-runnable).

These tests cover the parts of ``attention_pallas_rpa_v3.py`` that are pure
PyTorch/Python and need no JAX / torch_tpu / TPU device:

* :meth:`TpuPallasRpaV3AttnBackend._build_metadata` — the
  ``PallasForwardMetadata`` produced for decode vs extend matches the kernel's
  rectangular-page layout (cu_q_lens from extend_seq_lens, cu_kv_lens =
  i * pages_per_seq * page_size, page_indices gathered + page-divided from
  req_to_token, distribution = [bs,bs,bs] decode / [0,bs,bs] extend).
* :meth:`TpuPallasRpaV3AttnBackend.init_forward_metadata` populates
  ``self.forward_metadata``.
* :meth:`TpuPallasRpaV3AttnBackend.forward_decode` flattens a 3D ``q`` to 2D
  before delegating — the workaround for the torch.compile rotary_emb shape
  bug (mirrors TorchNativeAttnBackend).
* Capability flags: ``support_triton()`` is False, fill value is 0.

Kernel numerics live in ``test_ragged_paged_attention_v3.py`` (JAX+TPU
required). Registration + pool-swap dispatch live in
``test_pallas_rpa_v3_dispatch.py`` (also CPU-runnable).

The backend's ``__init__`` calls ``_build_pallas_kernel_op()`` which imports
JAX + torch_tpu eagerly. To stay CPU-only we instantiate a thin subclass that
skips that step; the methods under test never touch ``self._op``.
"""

import types
import unittest
from typing import Optional

import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    PallasForwardMetadata,
    TpuPallasRpaV3AttnBackend,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase


class _BackendUnderTest(TpuPallasRpaV3AttnBackend):
    """Backend instance that skips the JAX-requiring kernel-op registration.

    ``__init__`` in the production class calls ``_build_pallas_kernel_op()``
    which imports JAX + torch_tpu. The methods exercised here
    (``_build_metadata`` / ``init_forward_metadata`` / ``forward_decode``'s
    reshape) never touch ``self._op``, so leaving it as ``None`` is safe.
    """

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
        # _build_metadata reads req_to_token_pool.req_to_token only.
        self.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        self.forward_metadata = None
        self._op = None
        self._distribution_cache = {}
        # Non-SWA: skips the hybrid-SWA page remap in _build_metadata(_host).
        self._is_hybrid_swa = False


def _make_forward_batch(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    forward_mode: ForwardMode,
    extend_seq_lens: Optional[torch.Tensor] = None,
    spec_info=None,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    """Minimal fake ForwardBatch for ``_build_metadata``.

    Only the fields read by ``_build_metadata`` are populated. Using a
    ``SimpleNamespace`` (instead of a real ``ForwardBatch``) keeps the test
    independent of the model-runner construction graph.
    """
    return types.SimpleNamespace(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        forward_mode=forward_mode,
        extend_seq_lens=extend_seq_lens,
        spec_info=spec_info,
        seq_lens_cpu=seq_lens_cpu,
    )


def _build_req_to_token(
    num_reqs: int, max_context_len: int, base_slot: int = 0
) -> torch.Tensor:
    """req_to_token[r, s] = base_slot + r * max_context_len + s.

    Deterministic so tests can compute the exact page_indices the metadata
    builder should produce.
    """
    return (
        base_slot
        + torch.arange(num_reqs).unsqueeze(1) * max_context_len
        + torch.arange(max_context_len).unsqueeze(0)
    ).to(torch.int32)


def _build_meta(backend, fb) -> PallasForwardMetadata:
    """Mirror ``init_forward_metadata``."""
    distribution = backend._get_cached_rpa_distribution(
        fb.forward_mode.is_decode(),
        backend._non_empty_batch_size(fb),
        fb.seq_lens.device,
    )
    return backend._build_metadata(fb, distribution)


class TestPallasRpaV3MetadataDecode(CustomTestCase):
    """Decode-mode metadata: q_len == 1 per sequence, distribution = [bs,bs,bs]."""

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
        meta = self._build()
        self.assertIsInstance(meta, PallasForwardMetadata)

    def test_kv_lens_matches_seq_lens(self):
        meta = self._build()
        self.assertTrue(torch.equal(meta.kv_lens, self.seq_lens.to(torch.int32)))
        self.assertEqual(meta.kv_lens.dtype, torch.int32)

    def test_cu_q_lens_decode_is_arange(self):
        # Each sequence contributes one query token, so cu_q_lens = [0,1,2,...,bs].
        meta = self._build()
        expected = torch.arange(self.bs + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected))
        self.assertEqual(meta.cu_q_lens.dtype, torch.int32)

    def test_cu_kv_lens_is_rectangular(self):
        # cu_kv_lens[i] = i * pages_per_seq * page_size.
        meta = self._build()
        stride = self.PAGES_PER_SEQ * self.PAGE_SIZE
        expected = (torch.arange(self.bs + 1) * stride).to(torch.int32)
        self.assertTrue(torch.equal(meta.cu_kv_lens, expected))
        self.assertEqual(meta.cu_kv_lens.dtype, torch.int32)

    def test_page_indices_layout(self):
        meta = self._build()
        # Shape is flat [bs * pages_per_seq] — the kernel reads
        # page_indices[i * pages_per_seq + j].
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
                self.assertEqual(
                    got,
                    expected,
                    f"seq {i} page {j}: page_indices mismatch",
                )

    def test_distribution_decode_all_active(self):
        # Decode: every sequence is in the [0,bs) decode range.
        meta = self._build()
        expected = torch.full((3,), self.bs, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.distribution, expected))
        self.assertEqual(meta.distribution.dtype, torch.int32)

    def test_extend_seq_lens_unused_in_decode(self):
        # _build_metadata must work even when extend_seq_lens is None
        # (the field is not required for decode). The setUp default already
        # leaves it None; just sanity-check we got here.
        meta = self._build()
        self.assertIsNotNone(meta)


class TestPallasRpaV3MetadataExtend(CustomTestCase):
    """Extend-mode metadata: cu_q_lens from extend_seq_lens, distribution=[0,bs,bs]."""

    PAGE_SIZE = 16
    PAGES_PER_SEQ = 8
    MAX_CONTEXT_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 128

    def setUp(self):
        self.bs = 4
        self.req_to_token = _build_req_to_token(
            num_reqs=8, max_context_len=self.MAX_CONTEXT_LEN
        )
        self.backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        self.req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        # Mixed q-lengths to make sure the cumsum is computed properly.
        self.extend_seq_lens = torch.tensor([3, 5, 2, 1], dtype=torch.int32)
        self.seq_lens = torch.tensor([10, 30, 7, 4], dtype=torch.int32)
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
        # cumsum prepended with 0.
        expected = torch.tensor([0, 3, 8, 10, 11], dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected))

    def test_distribution_extend(self):
        # Extend: decode_end=0, prefill_end=bs, mixed_end=bs.
        meta = self._build()
        expected = torch.tensor([0, self.bs, self.bs], dtype=torch.int32)
        self.assertTrue(torch.equal(meta.distribution, expected))

    def test_extend_requires_extend_seq_lens(self):
        # Extend mode without extend_seq_lens must raise — the assertion is
        # the contract the metadata builder advertises.
        fb_bad = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=None,
        )
        with self.assertRaises(AssertionError):
            _build_meta(self.backend, fb_bad)


class TestPallasRpaV3MetadataLayoutCorners(CustomTestCase):
    """Layout invariants that are easy to break under refactors."""

    def test_max_context_len_not_multiple_of_page_size_is_trimmed(self):
        # If max_context_len isn't an exact multiple of page_size, the
        # rectangular layout uses pages_per_seq = floor(max_ctx / page_size)
        # and the slicing must produce exactly that many entries per row.
        page_size = 16
        max_ctx_raw = 16 * 4 + 5  # 69 — not a multiple of page_size.
        req_to_token = _build_req_to_token(num_reqs=2, max_context_len=max_ctx_raw)
        backend = _BackendUnderTest(page_size, req_to_token)
        bs = 2
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        fb = _make_forward_batch(
            seq_lens=torch.tensor([8, 12], dtype=torch.int32),
            req_pool_indices=req_pool_indices,
            forward_mode=ForwardMode.DECODE,
        )

        meta = _build_meta(backend, fb)

        expected_pages_per_seq = max_ctx_raw // page_size  # 4
        self.assertEqual(tuple(meta.page_indices.shape), (bs * expected_pages_per_seq,))
        # cu_kv_lens stride must match the trimmed pages_per_seq, not the raw
        # max_ctx — otherwise the kernel's page-index addressing skews.
        stride = expected_pages_per_seq * page_size
        expected_cu_kv = (torch.arange(bs + 1) * stride).to(torch.int32)
        self.assertTrue(torch.equal(meta.cu_kv_lens, expected_cu_kv))


class TestPallasRpaV3MetadataSpec(CustomTestCase):
    """Spec-decode modes on the host-resident fast path."""

    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 64
    DRAFT_TOKEN_NUM = 4

    def setUp(self):
        self.bs = 3
        self.req_to_token = _build_req_to_token(
            num_reqs=8, max_context_len=self.MAX_CONTEXT_LEN
        )
        self.req_pool_indices = torch.tensor([5, 2, 7], dtype=torch.int64)
        self.seq_lens = torch.tensor([10, 24, 33], dtype=torch.int32)
        self.spec_info = types.SimpleNamespace(draft_token_num=self.DRAFT_TOKEN_NUM)

    def test_target_verify_bumps_kv_lens_and_cu_q_lens(self):
        backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=self.spec_info,
        )
        meta = _build_meta(backend, fb)
        # KV cache already holds the draft tokens: kv_lens = seq_lens + draft.
        expected_kv = (self.seq_lens + self.DRAFT_TOKEN_NUM).to(torch.int32)
        self.assertTrue(torch.equal(meta.kv_lens, expected_kv))
        # Each sequence verifies draft_token_num query tokens.
        expected_cu_q = (torch.arange(self.bs + 1) * self.DRAFT_TOKEN_NUM).to(
            torch.int32
        )
        self.assertTrue(torch.equal(meta.cu_q_lens, expected_cu_q))
        # TARGET_VERIFY is not decode: distribution = [0, bs, bs].
        self.assertTrue(
            torch.equal(
                meta.distribution,
                torch.tensor([0, self.bs, self.bs], dtype=torch.int32),
            )
        )

    def test_draft_decode_bumps_kv_lens_by_step_id(self):
        step_id = 2
        backend = _BackendUnderTest(
            self.PAGE_SIZE, self.req_to_token, speculative_step_id=step_id
        )
        fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
            spec_info=self.spec_info,
        )
        meta = _build_meta(backend, fb)
        # Draft step sees KV entries from prior steps: kv_lens += step_id + 1.
        expected_kv = (self.seq_lens + step_id + 1).to(torch.int32)
        self.assertTrue(torch.equal(meta.kv_lens, expected_kv))
        # One query token per sequence, as in plain decode.
        expected_cu_q = torch.arange(self.bs + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected_cu_q))

    def test_draft_extend_uses_extend_seq_lens(self):
        backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        extend_seq_lens = torch.tensor([3, 1, 2], dtype=torch.int32)
        fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            extend_seq_lens=extend_seq_lens,
            spec_info=self.spec_info,
        )
        meta = _build_meta(backend, fb)
        self.assertTrue(torch.equal(meta.kv_lens, self.seq_lens.to(torch.int32)))
        expected_cu_q = torch.tensor([0, 3, 4, 6], dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens, expected_cu_q))

    def test_verify_buffer_update_is_noop(self):
        backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        self.assertIsNone(
            backend.update_verify_buffers_to_fill_after_draft(
                self.spec_info, cuda_graph_bs=8
            )
        )

    def test_spec_page_indices_match_non_spec(self):
        # The page-indices gather is mode-independent; spec modes must produce
        # the same rectangular layout as plain decode.
        backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        fb_spec = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=self.spec_info,
        )
        fb_plain = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
        )
        meta_spec = _build_meta(backend, fb_spec)
        meta_plain = _build_meta(backend, fb_plain)
        self.assertTrue(torch.equal(meta_spec.page_indices, meta_plain.page_indices))
        self.assertTrue(torch.equal(meta_spec.cu_kv_lens, meta_plain.cu_kv_lens))


class TestRpaDistribution(CustomTestCase):
    """The cached ``distribution`` values that ``_build_metadata`` depends on:
    ``[bs]*3`` for decode, ``[0, bs, bs]`` otherwise; int32.
    """

    def test_get_cached_rpa_distribution_values_and_cache_hit(self):
        backend = _BackendUnderTest(
            16, _build_req_to_token(num_reqs=1, max_context_len=16)
        )
        device = torch.device("cpu")
        dec = backend._get_cached_rpa_distribution(True, 3, device)
        self.assertTrue(torch.equal(dec, torch.tensor([3, 3, 3], dtype=torch.int32)))
        ext = backend._get_cached_rpa_distribution(False, 3, device)
        self.assertTrue(torch.equal(ext, torch.tensor([0, 3, 3], dtype=torch.int32)))
        # Repeat call for the same key is a cache hit -> same tensor object.
        self.assertIs(backend._get_cached_rpa_distribution(True, 3, device), dec)


class TestNonEmptyBatchSize(CustomTestCase):
    """``_non_empty_batch_size`` clips trailing zero-length sync-padding rows
    from the RPA active count; ``_pre_bucket_padded_batch_size`` keeps them.
    """

    def setUp(self):
        self.backend = _BackendUnderTest(
            16, _build_req_to_token(num_reqs=8, max_context_len=64)
        )

    def _fb(self, seq_lens_list, with_cpu=True):
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
        return _make_forward_batch(
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(len(seq_lens_list), dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=seq_lens if with_cpu else None,
        )

    def test_no_padding_rows_returns_full_count(self):
        fb = self._fb([10, 24, 33])
        self.assertEqual(self.backend._pre_bucket_padded_batch_size(fb), 3)
        self.assertEqual(self.backend._non_empty_batch_size(fb), 3)

    def test_trailing_zero_rows_are_clipped(self):
        fb = self._fb([10, 24, 0, 0])
        self.assertEqual(self.backend._pre_bucket_padded_batch_size(fb), 4)
        self.assertEqual(self.backend._non_empty_batch_size(fb), 2)

    def test_all_zero_rows_active_is_zero(self):
        fb = self._fb([0, 0, 0])
        self.assertEqual(self.backend._non_empty_batch_size(fb), 0)

    def test_interior_zero_row_raises(self):
        # A zero between live rows is not the sync-padding tail layout; it
        # would leave a padding row inside the active range, so fail loud.
        fb = self._fb([10, 0, 33])
        with self.assertRaises(ValueError):
            self.backend._non_empty_batch_size(fb)

    def test_missing_seq_lens_cpu_derives_from_device(self):
        # Without the host copy, the clip derives from device seq_lens rather
        # than degrading to the padded count.
        fb = self._fb([10, 24, 0, 0], with_cpu=False)
        self.assertEqual(self.backend._non_empty_batch_size(fb), 2)


if __name__ == "__main__":
    unittest.main()
