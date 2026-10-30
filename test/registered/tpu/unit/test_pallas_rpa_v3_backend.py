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
* The shared RPA forward adapter converts SGLang's left-window span to the
  kernel's total-token count.
* Capability flags: ``support_triton()`` is False, fill value is 0.

Kernel numerics live in ``test_ragged_paged_attention_v3.py`` (JAX+TPU
required). Registration + pool-swap dispatch live in
``test_pallas_rpa_v3_dispatch.py`` (also CPU-runnable).

The backend's ``__init__`` calls ``_build_pallas_kernel_op()`` which imports
JAX + torch_tpu eagerly. To stay CPU-only we instantiate a thin subclass that
skips that step; the methods under test never touch ``self._op``.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import types
import unittest
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import (
    _resolve_decode_query_size,
)
from sglang_tpu.attention_pallas_rpa_v3 import (
    PallasForwardMetadata,
    TpuPallasRpaV3AttnBackend,
    _invoke_v3_kernel,
)
from sglang_tpu.envs import envs
from sglang_tpu.host_metadata_cache import HostMetadataCache


class _BackendUnderTest(TpuPallasRpaV3AttnBackend):
    """Backend instance that skips the JAX-requiring kernel-op registration.

    ``__init__`` in the production class calls ``_build_pallas_kernel_op()``
    which imports JAX + torch_tpu. Tests that exercise ``_forward`` replace
    ``self._op`` with a CPU-only recording function.
    """

    def __init__(
        self,
        page_size: int,
        req_to_token: torch.Tensor,
        speculative_step_id: int = 0,
        decode_query_size: int = 1,
    ) -> None:
        AttentionBackend.__init__(self)
        self.device = req_to_token.device
        self.page_size = page_size
        self.speculative_step_id = speculative_step_id
        self.decode_query_size = decode_query_size
        self.max_context_len = req_to_token.shape[1]
        # _build_metadata reads req_to_token_pool.req_to_token only.
        self.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        self.forward_metadata = None
        self._op = None
        self._distribution_cache = {}
        self._host_metadata_cache = HostMetadataCache()
        self._host_page_cols = None
        # Non-SWA: skips the hybrid-SWA page remap in _build_metadata(_host).
        self._is_hybrid_swa = False
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
    extend_seq_lens_cpu: Optional[list] = None,
):
    """Minimal fake ForwardBatch for ``_build_metadata``.

    Only the fields read by ``_build_metadata`` are populated. Using a
    ``SimpleNamespace`` (instead of a real ``ForwardBatch``) keeps the test
    independent of the model-runner construction graph.
    """
    if req_pool_indices_cpu is None:
        req_pool_indices_cpu = req_pool_indices.clone()

    return types.SimpleNamespace(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices_cpu,
        forward_mode=forward_mode,
        extend_seq_lens=extend_seq_lens,
        spec_info=spec_info,
        seq_lens_cpu=seq_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
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
    non_empty_bs = backend._non_empty_batch_size(fb)
    return backend._build_metadata(fb, non_empty_bs)


class TestPallasRpaSlidingWindowAdapter(CustomTestCase):
    """The shared adapter translates SGLang window spans for RPA kernels."""

    def _forward_with_window(self, sliding_window_size, **layer_attrs):
        backend = _BackendUnderTest(
            page_size=16,
            req_to_token=_build_req_to_token(num_reqs=1, max_context_len=16),
        )
        backend.kv_cache_quant_dtype = None
        backend.forward_metadata = PallasForwardMetadata(
            kv_lens=torch.tensor([1], dtype=torch.int32),
            cu_q_lens=torch.tensor([0, 1], dtype=torch.int32),
            cu_kv_lens=torch.tensor([0, 16], dtype=torch.int32),
            page_indices=torch.tensor([0], dtype=torch.int32),
            distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
            active_num_tokens=torch.tensor(1, dtype=torch.int32),
        )
        # ``decode_query_size`` defaults to 1 on the metadata, as for decode.
        backend.token_to_kv_pool = types.SimpleNamespace(
            get_fused_kv_buffer=lambda _layer_id: torch.empty(0)
        )

        captured = {}

        def recording_op(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sinks,
            scaling,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            decode_query_size=1,
            use_causal_mask=True,
            layer_idx=None,
        ):
            captured["sliding_window"] = sliding_window
            captured["use_causal_mask"] = use_causal_mask
            return torch.zeros_like(q)

        backend._op = recording_op
        layer = types.SimpleNamespace(
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=4,
            v_head_dim=4,
            layer_id=0,
            sliding_window_size=sliding_window_size,
            logit_cap=0,
            scaling=1.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        vars(layer).update(layer_attrs)
        forward_batch = types.SimpleNamespace(
            spec_info=None,
            capture_hidden_mode=None,
        )

        backend._forward(
            q=torch.zeros(1, 8),
            k=torch.zeros(1, 4),
            v=torch.zeros(1, 4),
            layer=layer,
            forward_batch=forward_batch,
        )
        self._captured = captured
        return captured["sliding_window"]

    def test_positive_window_is_converted_to_total_token_count(self):
        self.assertEqual(self._forward_with_window(7), 8)

    def test_disabled_window_is_passed_as_none(self):
        for sliding_window_size in (None, 0, -1):
            with self.subTest(sliding_window_size=sliding_window_size):
                self.assertIsNone(self._forward_with_window(sliding_window_size))

    def test_attention_type_selects_causal_mask(self):
        for attrs, use_causal_mask in (
            ({}, True),
            ({"attn_type": AttentionType.ENCODER_ONLY}, False),
            ({"is_cross_attention": True}, False),
        ):
            with self.subTest(attrs=attrs):
                self._forward_with_window(7, **attrs)
                self.assertIs(self._captured["use_causal_mask"], use_causal_mask)


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

    def test_host_metadata_uses_req_pool_indices_cpu(self):
        # The host-resident RPA metadata path must use req_pool_indices_cpu, not
        # req_pool_indices, which is TPU-resident in production. Give them
        # different values so this CPU-only test can identify which tensor was read.
        host_indices = torch.tensor([5, 2, 7], dtype=torch.int64)
        device_indices = torch.tensor([0, 0, 0], dtype=torch.int64)
        fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=device_indices,
            req_pool_indices_cpu=host_indices,
            forward_mode=ForwardMode.DECODE,
        )

        meta = _build_meta(self.backend, fb)

        for i, pool_idx in enumerate(host_indices.tolist()):
            expected = int(self.req_to_token[pool_idx, 0]) // self.PAGE_SIZE
            self.assertEqual(int(meta.page_indices[i * self.PAGES_PER_SEQ]), expected)

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


class TestPallasRpaHostLensReuse(CustomTestCase):
    def setUp(self):
        self.enterContext(envs.SGLANG_TPU_ENABLE_HOST_QUERY_LENS.override(True))

    def test_missing_or_wrong_size_host_mirror_falls_back_to_current_lengths(self):
        backend = _BackendUnderTest(16, _build_req_to_token(3, 128))
        fb = _make_forward_batch(
            seq_lens=torch.tensor([20, 30]),
            req_pool_indices=torch.arange(2),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        )
        for mirror in (None, [99], [99, 99, 99]):
            with self.subTest(mirror=mirror):
                fb.extend_seq_lens_cpu = mirror
                meta = _build_meta(backend, fb)
                self.assertEqual(meta.cu_q_lens.tolist(), [0, 3, 5])
                self.assertEqual(meta.active_num_tokens.item(), 5)

    def test_variable_cache_is_bounded_and_does_not_mutate_old_buffers(self):
        backend = _BackendUnderTest(16, _build_req_to_token(3, 128))
        original = backend._get_query_metadata(
            bs=2, kv_stride=128, non_empty_bs=2, device="cpu", query_lens=(3, 2)
        )
        changed = backend._get_query_metadata(
            bs=2, kv_stride=128, non_empty_bs=2, device="cpu", query_lens=(1, 4)
        )
        self.assertEqual(original[0].tolist(), [0, 3, 5])
        self.assertEqual(changed[0].tolist(), [0, 1, 5])
        self.assertIs(
            backend._host_metadata_cache._by_value["cu_q_lens"][2], changed[0]
        )
        again = backend._get_query_metadata(
            bs=2, kv_stride=128, non_empty_bs=2, device="cpu", query_lens=(3, 2)
        )
        self.assertIsNot(original[0], again[0])
        self.assertEqual(original[0].tolist(), again[0].tolist())

    def test_fixed_layout_survives_alternating_decode_verify_and_extend(self):
        backend = _BackendUnderTest(16, _build_req_to_token(3, 128))
        fb = _make_forward_batch(
            seq_lens=torch.tensor([20, 30]),
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=types.SimpleNamespace(spec_input_type=None, num_tokens_per_req=4),
        )
        verify = _build_meta(backend, fb)
        fb.forward_mode = ForwardMode.DECODE
        decode = _build_meta(backend, fb)
        self.assertEqual(decode.cu_q_lens.tolist(), [0, 1, 2])
        fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        fb.extend_seq_lens = torch.empty(2, dtype=torch.int32, device="meta")
        fb.extend_seq_lens_cpu = [3, 2]
        extend = _build_meta(backend, fb)
        self.assertEqual(extend.cu_q_lens.tolist(), [0, 3, 5])
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        again = _build_meta(backend, fb)
        self.assertIs(verify.cu_q_lens, again.cu_q_lens)
        self.assertIs(verify.cu_kv_lens, again.cu_kv_lens)
        self.assertIs(verify.active_num_tokens, again.active_num_tokens)

    def test_draft_host_lengths_avoid_device_read_and_preserve_padding(self):
        backend = _BackendUnderTest(16, _build_req_to_token(3, 128))
        fb = _make_forward_batch(
            seq_lens=torch.tensor([20, 30, 0]),
            req_pool_indices=torch.arange(3),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            # A meta tensor cannot be copied to CPU: metadata must use the
            # padded host mirror supplied by the draft compile policy.
            extend_seq_lens=torch.empty(3, dtype=torch.int32, device="meta"),
        )
        fb.extend_seq_lens_cpu = [4, 4, 0]
        first = _build_meta(backend, fb)
        second = _build_meta(backend, fb)
        self.assertTrue(torch.equal(first.cu_q_lens, torch.tensor([0, 4, 8, 8])))
        self.assertEqual(first.active_num_tokens.item(), 8)
        self.assertIs(first.cu_q_lens, second.cu_q_lens)
        self.assertIs(first.cu_kv_lens, second.cu_kv_lens)
        self.assertIs(first.active_num_tokens, second.active_num_tokens)
        self.assertEqual(first.active_num_tokens.untyped_storage().nbytes(), 4)

        # Identical physical width can have fewer active rows on the next
        # cycle. Neither the old scalar nor its backing storage may change.
        fb.seq_lens = torch.tensor([20, 0, 0])
        third = _build_meta(backend, fb)
        self.assertEqual(third.active_num_tokens.item(), 4)
        self.assertEqual(first.active_num_tokens.item(), 8)

        fb.extend_seq_lens_cpu = torch.tensor([2, 0, 0])
        fourth = _build_meta(backend, fb)
        self.assertTrue(torch.equal(fourth.cu_q_lens, torch.tensor([0, 2, 2, 2])))
        self.assertEqual(fourth.active_num_tokens.item(), 2)

    def test_reused_lens_do_not_cache_request_pages_or_kv_lengths(self):
        table = _build_req_to_token(3, 128)
        backend = _BackendUnderTest(16, table)
        fb = _make_forward_batch(
            seq_lens=torch.tensor([20, 30]),
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=types.SimpleNamespace(spec_input_type=None, num_tokens_per_req=4),
        )
        first = _build_meta(backend, fb)
        fb.seq_lens = torch.tensor([23, 33])
        fb.req_pool_indices_cpu = torch.tensor([2, 0])
        table[2, 0] = 1024
        second = _build_meta(backend, fb)
        self.assertIs(first.cu_q_lens, second.cu_q_lens)
        self.assertEqual(second.page_indices[0].item(), 64)
        self.assertEqual(first.page_indices[0].item(), 0)
        self.assertTrue(torch.equal(second.kv_lens, torch.tensor([27, 37])))

    def test_layout_changes_and_empty_batch(self):
        backend = _BackendUnderTest(16, _build_req_to_token(3, 128))
        for lengths, stride, active in [
            ((1, 1), 128, 2),
            ((4, 4), 128, 2),
            ((4, 4, 0), 128, 2),
            ((4, 4, 0), 256, 0),
            ((), 256, 0),
        ]:
            with self.subTest(lengths=lengths, stride=stride, active=active):
                q, kv, count = backend._get_query_metadata(
                    bs=len(lengths),
                    kv_stride=stride,
                    non_empty_bs=active,
                    device=torch.device("cpu"),
                    query_lens=lengths,
                )
                self.assertEqual(
                    q.tolist(), [sum(lengths[:i]) for i in range(len(lengths) + 1)]
                )
                self.assertEqual(
                    kv.tolist(), [i * stride for i in range(len(lengths) + 1)]
                )
                self.assertEqual(count.item(), sum(lengths[:active]))
                self.assertEqual(count.untyped_storage().nbytes(), 4)


class TestPallasRpaV3MetadataLayoutCorners(CustomTestCase):
    """Layout invariants that are easy to break under refactors."""

    def test_max_context_len_not_multiple_of_page_size_is_rounded_up(self):
        # If max_context_len isn't an exact multiple of page_size, the
        # rectangular layout reserves the final partial page so its capacity
        # still covers max_context_len.
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

        expected_pages_per_seq = (max_ctx_raw + page_size - 1) // page_size  # 5
        self.assertEqual(tuple(meta.page_indices.shape), (bs * expected_pages_per_seq,))
        # cu_kv_lens stride must match the rounded-up pages_per_seq, not the
        # raw max_ctx — otherwise the kernel's page-index addressing skews.
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
        self.spec_info = types.SimpleNamespace(
            spec_input_type=None,
            draft_token_num=self.DRAFT_TOKEN_NUM,
            num_tokens_per_req=self.DRAFT_TOKEN_NUM,
        )

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
        # Single-token decode (the default width): TARGET_VERIFY takes the
        # extend range, distribution = [0, bs, bs].
        self.assertTrue(
            torch.equal(
                meta.distribution,
                torch.tensor([0, self.bs, self.bs], dtype=torch.int32),
            )
        )

    def test_target_verify_uses_algorithm_specific_width(self):
        backend = _BackendUnderTest(self.PAGE_SIZE, self.req_to_token)
        verify_width = self.DRAFT_TOKEN_NUM + 1
        fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=types.SimpleNamespace(
                draft_token_num=self.DRAFT_TOKEN_NUM,
                num_tokens_per_req=verify_width,
            ),
        )

        meta = _build_meta(backend, fb)

        self.assertTrue(
            torch.equal(meta.kv_lens, (self.seq_lens + verify_width).to(torch.int32))
        )
        self.assertTrue(
            torch.equal(
                meta.cu_q_lens,
                (torch.arange(self.bs + 1) * verify_width).to(torch.int32),
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


class TestFixedWidthAsDecode(CustomTestCase):
    """Fixed-width spec phases take the kernel's DECODE range.

    The kernel's decode loop bounds are static, so the width forwarded to the
    op must match the width this backend was built for. Target verification
    of that width and a decode-phase draft extension whose every row has that
    width qualify; every other forward stays on the dynamic routes with 1.
    """

    PAGE_SIZE = 16
    WIDTH = 4

    def _backend(self, decode_query_size):
        backend = _BackendUnderTest(
            page_size=self.PAGE_SIZE,
            req_to_token=_build_req_to_token(num_reqs=8, max_context_len=64),
            decode_query_size=decode_query_size,
        )
        backend.kv_cache_quant_dtype = None
        backend.token_to_kv_pool = types.SimpleNamespace(
            get_fused_kv_buffer=lambda _layer_id: torch.empty(0)
        )
        return backend

    def _forward_batch(self, mode, num_tokens_per_req, extend_lens=None):
        bs = 3
        extend = (
            None
            if extend_lens is None
            else torch.tensor(extend_lens, dtype=torch.int32)
        )
        fb = _make_forward_batch(
            seq_lens=torch.tensor([10, 24, 33], dtype=torch.int32),
            req_pool_indices=torch.tensor([5, 2, 7], dtype=torch.int64),
            forward_mode=mode,
            extend_seq_lens=extend,
            spec_info=types.SimpleNamespace(
                num_tokens_per_req=num_tokens_per_req,
                spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND,
            ),
            extend_seq_lens_cpu=extend_lens,
        )
        fb.capture_hidden_mode = None
        return fb, bs

    def _forward_width(self, backend, mode, num_tokens_per_req, extend_lens=None):
        """The ``decode_query_size`` the op receives for one forward."""
        forward_batch, bs = self._forward_batch(mode, num_tokens_per_req, extend_lens)
        backend.forward_metadata = _build_meta(backend, forward_batch)
        captured = {}

        def recording_op(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sinks,
            scaling,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
            decode_query_size=1,
            use_causal_mask=True,
            layer_idx=None,
        ):
            captured["width"] = decode_query_size

            def kernel(*_args, **kwargs):
                captured["kernel_kwargs"] = kwargs

            _invoke_v3_kernel(
                kernel,
                queries=q,
                keys=k,
                values=v,
                kv_cache_fused=kv_cache,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                cu_kv_lens=cu_kv_lens,
                distribution=distribution,
                attention_sink=sinks,
                sm_scale=scaling,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                k_scale=k_scale,
                v_scale=v_scale,
                decode_query_size=decode_query_size,
                use_causal_mask=use_causal_mask,
            )
            return torch.zeros_like(q)

        backend._op = recording_op
        layer = types.SimpleNamespace(
            tp_q_head_num=2,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=4,
            v_head_dim=4,
            layer_id=0,
            sliding_window_size=None,
            logit_cap=0,
            scaling=1.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        num_tokens = int(backend.forward_metadata.cu_q_lens[-1])
        backend._forward(
            q=torch.zeros(num_tokens, 8),
            k=torch.zeros(num_tokens, 4),
            v=torch.zeros(num_tokens, 4),
            layer=layer,
            forward_batch=forward_batch,
        )
        # The v3 invoker forwards the width as the kernel's static argument.
        self.assertEqual(
            captured["kernel_kwargs"]["decode_query_size"], captured["width"]
        )
        self.assertEqual(captured["kernel_kwargs"]["causal"], 1)
        return captured["width"]

    def test_verify_of_the_built_width_is_decode(self):
        backend = self._backend(self.WIDTH)
        self.assertEqual(
            self._forward_width(backend, ForwardMode.TARGET_VERIFY, self.WIDTH),
            self.WIDTH,
        )

    def test_other_modes_keep_single_token_decode(self):
        # A width on spec_info is also present during draft decode and the
        # ragged prompt-phase extension. Passing it there would impose static
        # loop bounds on rows of other lengths.
        backend = self._backend(self.WIDTH)
        for mode, lens in (
            (ForwardMode.DECODE, None),
            (ForwardMode.EXTEND, [4, 4, 4]),
            (ForwardMode.DRAFT_EXTEND_V2, [3, 4, 4]),
            (ForwardMode.DRAFT_EXTEND_V2, [5, 5, 5]),
        ):
            with self.subTest(mode=mode, lens=lens):
                self.assertEqual(
                    self._forward_width(backend, mode, self.WIDTH, lens), 1
                )

    def test_uniform_draft_extension_is_decode(self):
        backend = self._backend(self.WIDTH)
        lens = [self.WIDTH] * 3
        self.assertEqual(
            self._forward_width(backend, ForwardMode.DRAFT_EXTEND_V2, self.WIDTH, lens),
            self.WIDTH,
        )
        fb, bs = self._forward_batch(ForwardMode.DRAFT_EXTEND_V2, self.WIDTH, lens)
        meta = _build_meta(backend, fb)
        self.assertEqual(meta.distribution.tolist(), [bs, bs, bs])
        # The draft's KV already holds the window; query rows are the window.
        self.assertEqual(meta.cu_q_lens.tolist(), [0, 4, 8, 12])
        fb, bs = self._forward_batch(ForwardMode.DRAFT_EXTEND_V2, self.WIDTH, [3, 4, 4])
        self.assertEqual(_build_meta(backend, fb).distribution.tolist(), [0, bs, bs])

    def test_verify_of_another_width_stays_dynamic(self):
        backend = self._backend(self.WIDTH)
        self.assertEqual(
            self._forward_width(backend, ForwardMode.TARGET_VERIFY, self.WIDTH + 1),
            1,
        )

    def _verify_distribution(self, backend, num_tokens_per_req):
        bs = 3
        fb = _make_forward_batch(
            seq_lens=torch.tensor([10, 24, 33], dtype=torch.int32),
            req_pool_indices=torch.tensor([5, 2, 7], dtype=torch.int64),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=types.SimpleNamespace(num_tokens_per_req=num_tokens_per_req),
        )
        return _build_meta(backend, fb).distribution.tolist(), bs

    def test_verify_distribution_is_decode_only_for_the_built_width(self):
        backend = self._backend(self.WIDTH)
        dist, bs = self._verify_distribution(backend, self.WIDTH)
        self.assertEqual(dist, [bs, bs, bs])
        dist, bs = self._verify_distribution(backend, self.WIDTH + 1)
        self.assertEqual(dist, [0, bs, bs])
        dist, bs = self._verify_distribution(self._backend(1), self.WIDTH)
        self.assertEqual(dist, [0, bs, bs])

    def test_resolved_width_is_the_speculative_draft_token_count(self):
        spec = types.SimpleNamespace(speculative_num_draft_tokens=self.WIDTH)
        with patch(
            "sglang_tpu.attention_pallas_rpa_base.get_spec",
            return_value=spec,
        ):
            target = dict(supports_decode_query_size=True, spec_enabled=True)
            self.assertEqual(_resolve_decode_query_size(**target), self.WIDTH)
            for override in (
                {"supports_decode_query_size": False},
                {"spec_enabled": False},
            ):
                with self.subTest(override=override):
                    self.assertEqual(
                        _resolve_decode_query_size(**{**target, **override}), 1
                    )


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
