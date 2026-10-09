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
"""Unit tests for the DeepSeek-V4 TPU compressor and indexer Pallas custom ops.

Covers:
  * ``get_or_build_cos_sin_cache``, ``ensure_tpu_compressor_caches``, and
    ``ensure_tpu_indexer_caches`` weight/cache preparation helpers (run on CPU
    and TPU);
  * ``_register_csa_compressor_op`` + ``_register_indexer_compressor_op`` +
    ``_register_streamindex_topk_op`` (C4 compressor + TopK indexer pipeline);
  * ``_register_hca_compressor_op`` (C128 HCA compressor).

The Pallas kernel execution tests require a free TPU device and are skipped
when ``torch_tpu`` is unavailable.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import importlib.util
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.tpu.dsv4.dsv4_compressor import (
    _register_csa_compressor_op,
    _register_hca_compressor_op,
    _register_indexer_compressor_op,
    ensure_tpu_compressor_caches,
)
from sglang.srt.hardware_backend.tpu.dsv4.dsv4_indexer import (
    _register_streamindex_topk_op,
    ensure_tpu_indexer_caches,
)
from sglang.srt.hardware_backend.tpu.dsv4.rope import get_or_build_cos_sin_cache
from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


class TestDSV4TPUHelpers(CustomTestCase):
    """Pure tensor/cache preparation helpers — run on CPU or TPU."""

    def test_cos_sin_and_compressor_indexer_cache_preparation(self):
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(
            dsv4_attn_backend="tpu_dsv4", page_size=256
        )
        override.install()
        self.addCleanup(override.restore)

        device = torch.device("cpu")
        freqs_cis = torch.polar(
            torch.ones((256, 32), dtype=torch.float32, device=device),
            torch.zeros((256, 32), dtype=torch.float32, device=device),
        )
        cos_sin = get_or_build_cos_sin_cache(freqs_cis)
        self.assertEqual(tuple(cos_sin.shape), (256, 64))
        self.assertEqual(cos_sin.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(cos_sin[:, :32], torch.ones_like(cos_sin[:, :32]))
        )
        self.assertTrue(
            torch.allclose(cos_sin[:, 32:], torch.zeros_like(cos_sin[:, 32:]))
        )

        compressor = SimpleNamespace(
            is_in_indexer=False,
            wkv_gate=SimpleNamespace(
                weight=SimpleNamespace(
                    data=torch.randn(2048, 512, dtype=torch.bfloat16, device=device)
                )
            ),
            ape=SimpleNamespace(
                data=torch.randn(4, 512, dtype=torch.bfloat16, device=device)
            ),
            norm=SimpleNamespace(
                weight=SimpleNamespace(
                    data=torch.ones(448, dtype=torch.bfloat16, device=device)
                ),
                variance_epsilon=1e-6,
            ),
            freqs_cis=freqs_cis,
        )
        ensure_tpu_compressor_caches(compressor)
        self.assertTrue(compressor._tpu_caches_ready)
        self.assertEqual(tuple(compressor._tpu_wkv_wgate.shape), (512, 2048))
        self.assertEqual(compressor._tpu_ape.dtype, torch.float32)
        self.assertEqual(compressor._tpu_norm_weight.dtype, torch.float32)
        self.assertEqual(tuple(compressor._tpu_cos_sin_cache.shape), (256, 64))

        indexer = SimpleNamespace(
            use_fp4_indexer=False,
            freqs_cis=freqs_cis,
        )
        ensure_tpu_indexer_caches(indexer)
        self.assertTrue(indexer._tpu_caches_ready)
        self.assertEqual(tuple(indexer._tpu_cos_sin_cache.shape), (256, 64))

        fp4_indexer = SimpleNamespace(
            use_fp4_indexer=True,
            freqs_cis=freqs_cis,
        )
        with self.assertRaises(NotImplementedError):
            ensure_tpu_indexer_caches(fp4_indexer)


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestDSV4TPUPallasKernels(CustomTestCase):
    """Hardware Pallas custom-op tests on TPU."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        cls.device = torch.device("tpu")

    def _make_cos_sin(self, max_pos: int = 256) -> torch.Tensor:
        freqs_cis = torch.polar(
            torch.ones((max_pos, 32), dtype=torch.float32),
            torch.zeros((max_pos, 32), dtype=torch.float32),
        ).to(self.device)
        return get_or_build_cos_sin_cache(freqs_cis)

    def test_compressor_and_indexer_ops(self):
        num_tokens = 16
        head_dim = 512
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        cos_sin = self._make_cos_sin(256)

        hidden_states = torch.randn(
            num_tokens, head_dim, dtype=torch.bfloat16, device=self.device
        )
        idx_k = torch.randn(num_tokens, 128, dtype=torch.bfloat16, device=self.device)
        cu_q_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)
        kv_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=self.device)
        distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device=self.device)
        state_bt_flat = torch.tensor(
            [1, 0, 0, 0, 0, 0], dtype=torch.int32, device=self.device
        )

        csa_nope = torch.zeros((4, 64, 4, 128), dtype=torch.uint8, device=self.device)
        csa_rope = torch.zeros((4, 16, 4, 128), dtype=torch.uint8, device=self.device)
        idx_k_buf = torch.zeros((4, 16, 4, 256), dtype=torch.uint8, device=self.device)
        hca_kv = torch.zeros((4, 4, 4, 128), dtype=torch.uint8, device=self.device)
        c4_state = torch.zeros((16, 64, 4, 128), dtype=torch.uint8, device=self.device)
        c4_idx_state = torch.zeros(
            (16, 8, 4, 256), dtype=torch.uint8, device=self.device
        )
        c128_state = torch.zeros(
            (32, 64, 4, 128), dtype=torch.uint8, device=self.device
        )

        csa_comp_op = _register_csa_compressor_op()
        csa_comp_op(
            hidden_states,
            torch.randn(512, 2048, dtype=torch.bfloat16, device=self.device) * 0.02,
            torch.zeros(4, 1024, dtype=torch.float32, device=self.device),
            torch.ones(512, dtype=torch.float32, device=self.device),
            cos_sin,
            positions,
            state_bt_flat,
            cu_q_lens,
            page_indices,
            distribution,
            csa_nope,
            csa_rope,
            c4_state,
            4,
            512,
            4,
            True,
            1e-6,
            64,
        )

        idx_comp_op = _register_indexer_compressor_op()
        idx_comp_op(
            idx_k,
            torch.randn(128, 512, dtype=torch.bfloat16, device=self.device) * 0.02,
            torch.zeros(4, 256, dtype=torch.float32, device=self.device),
            torch.ones(128, dtype=torch.float32, device=self.device),
            cos_sin,
            positions,
            state_bt_flat,
            cu_q_lens,
            page_indices,
            distribution,
            idx_k_buf,
            c4_idx_state,
            4,
            128,
            4,
            True,
            1e-6,
            128,
        )

        topk_op = _register_streamindex_topk_op()
        q_idx = torch.randn(
            num_tokens, 64, 128, dtype=torch.bfloat16, device=self.device
        )
        idx_weights = torch.ones(
            num_tokens, 64, dtype=torch.bfloat16, device=self.device
        )
        topk_indices = topk_op(
            q_idx,
            positions,
            cos_sin,
            idx_weights,
            idx_k_buf,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            512,
            4,
            1.0 / math.sqrt(128),
            64,
        )
        self.assertEqual(tuple(topk_indices.shape), (num_tokens, 512))

        hca_comp_op = _register_hca_compressor_op()
        hca_comp_op(
            hidden_states,
            torch.randn(512, 1024, dtype=torch.bfloat16, device=self.device) * 0.02,
            torch.zeros(128, 512, dtype=torch.float32, device=self.device),
            torch.ones(512, dtype=torch.float32, device=self.device),
            cos_sin,
            positions,
            state_bt_flat,
            cu_q_lens,
            page_indices,
            distribution,
            hca_kv,
            c128_state,
            8,
            512,
            128,
            False,
            1e-6,
            0,
        )


if __name__ == "__main__":
    unittest.main()
