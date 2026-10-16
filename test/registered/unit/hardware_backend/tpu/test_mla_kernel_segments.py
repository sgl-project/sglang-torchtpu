"""MLA v2 ``segments`` vs the ``distribution`` the Pallas MLA backend builds."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
CONTEXT_LEN = 4096
DRAFT_TOKENS = 4
T, H, R, D = 3, 4, 512, 64


def _batch(mode, seq_lens, extend_lens=None, spec_info=None):
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    bs = seq_lens.shape[0]
    return SimpleNamespace(
        forward_mode=mode,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        req_pool_indices_cpu=torch.arange(bs, dtype=torch.int64),
        extend_seq_lens=(
            None
            if extend_lens is None
            else torch.tensor(extend_lens, dtype=torch.int32)
        ),
        extend_seq_lens_cpu=extend_lens,
        spec_info=spec_info,
    )


class TestMlaKernelSegments(CustomTestCase):
    def _backend(self, spec: bool) -> TpuPallasMlaAttnBackend:
        spec_args = (
            dict(
                speculative_num_draft_tokens=DRAFT_TOKENS,
                speculative_algorithm="EAGLE3",
                speculative_num_steps=None,
                speculative_eagle_topk=1,
            )
            if spec
            else {}
        )
        override = get_context().override_server_args(page_size=PAGE_SIZE, **spec_args)
        override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=PAGE_SIZE,
            model_config=SimpleNamespace(context_len=CONTEXT_LEN),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(4, 2 * CONTEXT_LEN, dtype=torch.int32)
            ),
            token_to_kv_pool=SimpleNamespace(get_fused_kv_buffer=lambda _: "kv"),
            is_hybrid_swa=False,
            kv_cache_dtype=torch.bfloat16,
            is_draft_worker=False,
            spec_algorithm=(
                SpeculativeAlgorithm.EAGLE3 if spec else SpeculativeAlgorithm.NONE
            ),
        )
        op = MagicMock(return_value=torch.zeros(T, H, R))
        with patch(
            "sglang_tpu.attention_pallas_mla._build_pallas_mla_kernel_op",
            return_value=op,
        ):
            backend = TpuPallasMlaAttnBackend(model_runner)
        self.assertEqual(backend.decode_query_size, DRAFT_TOKENS if spec else 1)
        return backend

    def _metadata(self, backend, batch):
        return backend._build_metadata(batch, backend._non_empty_batch_size(batch))

    def _assert_covered(self, metadata, expected):
        """Every range ``distribution`` populates is a pass ``segments`` runs."""
        self.assertEqual(metadata.segments, expected)
        decode_end, prefill_end, mixed_end = metadata.distribution.tolist()
        # The kernel's prefill-only pass is dead; MLA never populates it.
        self.assertEqual(decode_end, prefill_end)
        if expected == "decode":
            self.assertEqual(prefill_end, mixed_end)
        elif expected == "mixed":
            self.assertEqual(decode_end, 0)

    def test_segments_match_built_distribution(self):
        backend = self._backend(spec=False)
        cases = {
            "decode": (_batch(ForwardMode.DECODE, [10, 20, 30]), "decode"),
            "prefill": (_batch(ForwardMode.EXTEND, [16], [16]), "mixed"),
            "mixed": (_batch(ForwardMode.MIXED, [40, 21, 31], [8, 1, 1]), "mixed"),
            "extend_with_prefix": (
                _batch(ForwardMode.EXTEND, [50, 70], [5, 6]),
                "mixed",
            ),
            # DP idle runs as DECODE over zero-length rows: no active sequence.
            "dp_idle": (_batch(ForwardMode.DECODE, [0, 0]), "decode"),
        }
        for name, (batch, expected) in cases.items():
            with self.subTest(case=name):
                self._assert_covered(self._metadata(backend, batch), expected)

    def test_spec_server_keeps_every_pass_outside_decode(self):
        backend = self._backend(spec=True)
        verify = SimpleNamespace(num_tokens_per_req=DRAFT_TOKENS)
        cases = {
            "draft_decode": (
                _batch(ForwardMode.DECODE, [10, 20], spec_info=SimpleNamespace()),
                "decode",
            ),
            "target_verify": (
                _batch(ForwardMode.TARGET_VERIFY, [10, 20], spec_info=verify),
                "all",
            ),
            "prefill": (_batch(ForwardMode.EXTEND, [16, 9], [16, 9]), "all"),
        }
        for name, (batch, expected) in cases.items():
            with self.subTest(case=name):
                self._assert_covered(self._metadata(backend, batch), expected)

    def test_target_verify_runs_on_the_decode_passes(self):
        backend = self._backend(spec=True)
        verify = SimpleNamespace(num_tokens_per_req=DRAFT_TOKENS)
        metadata = self._metadata(
            backend, _batch(ForwardMode.TARGET_VERIFY, [10, 20], spec_info=verify)
        )
        self.assertEqual(metadata.decode_query_size, DRAFT_TOKENS)
        self.assertEqual(metadata.distribution.tolist(), [2, 2, 2])
        # A verify width other than the backend's stays on MIXED.
        other = SimpleNamespace(num_tokens_per_req=DRAFT_TOKENS - 1)
        metadata = self._metadata(
            backend, _batch(ForwardMode.TARGET_VERIFY, [10, 20], spec_info=other)
        )
        self.assertEqual(metadata.decode_query_size, 1)
        self.assertEqual(metadata.distribution.tolist(), [0, 0, 2])

    def test_forward_passes_metadata_segments_to_op(self):
        backend = self._backend(spec=False)
        layer = SimpleNamespace(
            v_head_dim=R,
            qk_head_dim=R + D,
            tp_q_head_num=H,
            scaling=0.125,
            layer_id=0,
            k_scale_float=None,
        )
        for batch in (
            _batch(ForwardMode.DECODE, [10, 20, 30]),
            _batch(ForwardMode.EXTEND, [12], [3]),
        ):
            backend.forward_metadata = self._metadata(backend, batch)
            backend._forward_mla(
                torch.randn(T, H, R, dtype=torch.bfloat16),
                torch.randn(T, 1, R, dtype=torch.bfloat16),
                layer,
                batch,
                q_rope=torch.randn(T, H, D, dtype=torch.bfloat16),
                k_rope=torch.randn(T, 1, D, dtype=torch.bfloat16),
            )
            kwargs = backend._op.call_args.kwargs
            self.assertEqual(kwargs["segments"], backend.forward_metadata.segments)
            self.assertEqual(
                kwargs["decode_query_size"], backend.forward_metadata.decode_query_size
            )


if __name__ == "__main__":
    unittest.main()
