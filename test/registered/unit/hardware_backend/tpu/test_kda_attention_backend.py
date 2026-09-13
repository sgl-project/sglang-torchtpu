# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ragged KDA metadata and the RadixLinearAttention boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang_tpu.attention_kda import TpuKDAAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKDAAttentionBackend(unittest.TestCase):
    def setUp(self):
        self.cache = SimpleNamespace(
            conv=[torch.zeros(4, 3, 24)], temporal=torch.zeros(4, 2, 4, 4)
        )
        self.pool = SimpleNamespace(
            get_mamba_indices=lambda indices: torch.tensor([0, 2, 1])[indices],
            translate_mamba_indices=lambda indices: indices,
            mamba2_layer_cache=Mock(return_value=self.cache),
        )
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            is_draft_worker=False,
            req_to_token_pool=self.pool,
            token_to_kv_pool=None,
            server_args=SimpleNamespace(enable_unified_memory=False),
        )
        self.op = Mock()
        with patch(
            "sglang_tpu.attention_kda.build_kda_op",
            return_value=self.op,
        ) as kda_builder, patch(
            "sglang_tpu.attention_gdn_pallas.build_gdn_pallas_op"
        ) as gdn_builder, patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.get_spec",
            return_value=SimpleNamespace(speculative_eagle_topk=None),
        ):
            self.backend = TpuKDAAttnBackend(runner)
        kda_builder.assert_called_once_with()
        gdn_builder.assert_not_called()
        self.hybrid_backend = HybridLinearAttnBackend(
            SimpleNamespace(
                token_to_kv_pool=None,
                req_to_token_pool=self.pool,
                needs_cpu_seq_lens=False,
            ),
            self.backend,
            full_attn_layers=[],
        )

    def batch(self, mode):
        return SimpleNamespace(
            forward_mode=mode,
            batch_size=3,
            _original_batch_size=2,
            _tpu_real_b=2,
            req_pool_indices=torch.tensor([1, 2, 0]),
            seq_lens=torch.tensor([8, 3, 0], dtype=torch.int64),
            extend_seq_lens=torch.tensor([1, 3, 0], dtype=torch.int32),
            # An unpadded start tensor must not truncate the padded metadata.
            extend_start_loc=torch.tensor([0, 1]),
            mamba_track_indices=None,
            mamba_track_mask=None,
        )

    def test_ragged_extend_offsets_and_null_slot(self):
        batch = self.batch(ForwardMode.EXTEND)
        self.backend.init_forward_metadata(batch)
        self.assertEqual(self.backend._query_start_loc.tolist(), [0, 1, 4, 4])
        self.assertEqual(self.backend._state_indices.tolist(), [2, 1, 0])
        self.assertEqual(self.backend._seq_lens.tolist(), [8, 3, 0])
        for tensor in (
            self.backend._query_start_loc,
            self.backend._state_indices,
            self.backend._seq_lens,
        ):
            self.assertEqual(tensor.dtype, torch.int32)
        self.assertEqual(self.backend._distribution.tolist(), [0, 0, 2])

    def test_decode_metadata_refreshes_after_extend(self):
        self.backend.init_forward_metadata(self.batch(ForwardMode.EXTEND))
        batch = self.batch(ForwardMode.DECODE)
        batch.seq_lens = torch.tensor([9, 4, 0])
        self.backend.init_forward_metadata(batch)
        self.assertEqual(self.backend._query_start_loc.tolist(), [0, 1, 2, 3])
        self.assertEqual(self.backend._state_indices.tolist(), [2, 1, 0])
        self.assertEqual(self.backend._seq_lens.tolist(), [9, 4, 0])
        self.assertEqual(self.backend._distribution.tolist(), [2, 2, 2])

    def test_radix_dispatch_preserves_token_axis_and_owned_state(self):
        layer = RadixLinearAttention(
            layer_id=7,
            num_q_heads=2,
            num_k_heads=2,
            num_v_heads=2,
            head_q_dim=4,
            head_k_dim=4,
            head_v_dim=4,
            conv_weights=torch.ones(24, 4),
            A_log=torch.zeros(2),
            dt_bias=torch.zeros(8),
            lower_bound=-5.0,
        )
        for mode, tokens in ((ForwardMode.EXTEND, 6), (ForwardMode.DECODE, 3)):
            with self.subTest(mode=mode):
                batch = self.batch(mode)
                self.backend.init_forward_metadata(batch)
                qkv, a, b = (
                    torch.ones(tokens, 24),
                    torch.ones(1, tokens, 8),
                    torch.ones(1, tokens, 2),
                )
                expected = torch.arange(tokens * 8).reshape(tokens, 2, 4).float()
                self.op.return_value = expected
                with patch(
                    "sglang.srt.layers.radix_linear_attention._is_tpu", True
                ), patch(
                    "sglang.srt.layers.radix_linear_attention.get_attn_backend",
                    return_value=self.hybrid_backend,
                ):
                    output = layer(batch, qkv, a, b)
                self.assertIs(output, expected)
                args = self.op.call_args.args
                for actual, original in zip(
                    args[:5], (qkv, a, b, self.cache.conv[0], self.cache.temporal)
                ):
                    self.assertIs(actual, original)
                self.assertEqual(
                    args[8].tolist(), self.backend._query_start_loc.tolist()
                )
                self.assertEqual(args[9].tolist(), [2, 1, 0])
                self.assertEqual(args[-3:], (2, 4, -5.0))
                self.pool.mamba2_layer_cache.assert_called_with(7)


if __name__ == "__main__":
    unittest.main()
