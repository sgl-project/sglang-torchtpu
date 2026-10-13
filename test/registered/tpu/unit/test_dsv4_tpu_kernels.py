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
"""Unit tests for the DeepSeek-V4 TPU Pallas custom ops and cache-prep helpers.

Covers:
  * ``tpu_dsv4`` registration in ``ATTENTION_BACKENDS``;
  * ``get_or_build_cos_sin_cache``, ``ensure_tpu_compressor_caches``, and
    ``ensure_tpu_indexer_caches`` weight/cache preparation helpers (run on CPU
    and TPU);
  * ``_register_rope_op`` and ``_register_qnorm_rope_op`` Pallas custom ops;
  * ``_register_attention_swa_op`` sliding-window MLA Pallas custom op;
  * ``_register_csa_compressor_op`` + ``_register_indexer_compressor_op`` +
    ``_register_streamindex_topk_op`` + ``_register_attention_csa_op`` (CSA
    pipeline);
  * ``_register_hca_compressor_op`` + ``_register_attention_hca_op`` (HCA
    pipeline).

The Pallas kernel execution tests require a free TPU device and are skipped
when ``torch_tpu`` is unavailable.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv4.dsv4_compressor import (
    _register_csa_compressor_op,
    _register_hca_compressor_op,
    _register_indexer_compressor_op,
    ensure_tpu_compressor_caches,
)
from sglang_tpu.dsv4.dsv4_indexer import (
    _register_streamindex_topk_op,
    ensure_tpu_indexer_caches,
)
from sglang_tpu.dsv4.rope import get_or_build_cos_sin_cache


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


class TestDSV4TPUHelpers(CustomTestCase):
    """Pure tensor/registry helpers — run on CPU or TPU."""

    def test_dsv4_registered_in_attention_backends(self):
        import sglang.srt.layers.attention.attention_registry as reg

        self.assertIn("dsv4", reg.ATTENTION_BACKENDS)

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

    def test_rope_and_qnorm_rope_ops(self):
        from sglang_tpu.dsv4.dsv4_attention_backend import (
            _register_qnorm_rope_op,
            _register_rope_op,
        )

        num_tokens = 16
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        cos_sin = self._make_cos_sin(256)

        rope_op = _register_rope_op()
        k_pe = torch.randn(num_tokens, 1, 512, dtype=torch.bfloat16, device=self.device)
        k_pe_out = rope_op(k_pe, positions, cos_sin, 0)
        self.assertEqual(tuple(k_pe_out.shape), (num_tokens, 1, 512))
        self.assertFalse(torch.isnan(k_pe_out).any().item())

        qnorm_rope_op = _register_qnorm_rope_op()
        q_raw = torch.randn(
            num_tokens, 16, 512, dtype=torch.bfloat16, device=self.device
        )
        q_out = qnorm_rope_op(q_raw, positions, cos_sin, 1e-6)
        self.assertEqual(tuple(q_out.shape), (num_tokens, 16, 512))
        self.assertFalse(torch.isnan(q_out).any().item())

    def test_swa_attention_op(self):
        from sglang_tpu.dsv4.dsv4_attention_backend import (
            _register_attention_swa_op,
        )

        num_tokens = 16
        num_q_heads = 16
        head_dim = 512
        q = torch.randn(
            num_tokens, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device
        )
        k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=self.device)
        sw_cache = torch.zeros((4, 512, 4, 128), dtype=torch.uint8, device=self.device)
        kv_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=self.device)
        cu_q_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)
        distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device=self.device)
        sinks = torch.zeros(num_q_heads, dtype=torch.float32, device=self.device)

        swa_op = _register_attention_swa_op()
        out = swa_op(
            q,
            k,
            sw_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            1.0 / math.sqrt(512),
            512,
            256,
        )
        self.assertEqual(tuple(out.shape), (num_tokens, num_q_heads, head_dim))
        self.assertFalse(torch.isnan(out).any().item())

    def test_csa_and_hca_pipeline_ops(self):
        from sglang_tpu.dsv4.dsv4_attention_backend import (
            _register_attention_csa_op,
            _register_attention_hca_op,
        )

        num_tokens = 16
        num_q_heads = 16
        head_dim = 512
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        cos_sin = self._make_cos_sin(256)

        hidden_states = torch.randn(
            num_tokens, head_dim, dtype=torch.bfloat16, device=self.device
        )
        idx_k = torch.randn(num_tokens, 128, dtype=torch.bfloat16, device=self.device)
        cu_q_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)
        kv_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        c128_kv_lens = torch.tensor([0], dtype=torch.int32, device=self.device)
        c128_kv_lens_to_attend = torch.zeros(
            num_tokens, dtype=torch.int32, device=self.device
        )
        page_indices = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=self.device)
        distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device=self.device)
        state_bt_flat = torch.tensor(
            [1, 0, 0, 0, 0, 0], dtype=torch.int32, device=self.device
        )

        sw_cache = torch.zeros((4, 512, 4, 128), dtype=torch.uint8, device=self.device)
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

        q = torch.randn(
            num_tokens, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device
        )
        sinks = torch.zeros(num_q_heads, dtype=torch.float32, device=self.device)
        csa_attn_op = _register_attention_csa_op()
        out_csa = csa_attn_op(
            q,
            hidden_states,
            sw_cache,
            kv_lens,
            page_indices,
            csa_nope,
            csa_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            1.0 / math.sqrt(512),
            512,
            256,
        )
        self.assertEqual(tuple(out_csa.shape), (num_tokens, num_q_heads, head_dim))
        self.assertFalse(torch.isnan(out_csa).any().item())

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

        hca_attn_op = _register_attention_hca_op()
        out_hca = hca_attn_op(
            q,
            hidden_states,
            sw_cache,
            kv_lens,
            page_indices,
            hca_kv,
            c128_kv_lens,
            c128_kv_lens_to_attend,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            1.0 / math.sqrt(512),
            512,
            256,
        )
        self.assertEqual(tuple(out_hca.shape), (num_tokens, num_q_heads, head_dim))
        self.assertFalse(torch.isnan(out_hca).any().item())

        # DeepSeek-V4-Pro CSA & Indexer (`index_topk = 1024, sliding_window = 128`):
        topk_indices_pro = topk_op(
            q_idx,
            positions,
            cos_sin,
            idx_weights,
            idx_k_buf,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            1024,
            4,
            1.0 / math.sqrt(128),
            64,
        )
        self.assertEqual(tuple(topk_indices_pro.shape), (num_tokens, 1024))
        out_csa_pro = csa_attn_op(
            q,
            hidden_states,
            sw_cache,
            kv_lens,
            page_indices,
            csa_nope,
            csa_rope,
            topk_indices_pro,
            page_indices,
            cu_q_lens,
            distribution,
            sinks,
            1.0 / math.sqrt(512),
            128,
            256,
        )
        self.assertEqual(tuple(out_csa_pro.shape), (num_tokens, num_q_heads, head_dim))
        self.assertFalse(torch.isnan(out_csa_pro).any().item())

    def test_dsv4_mxfp4_w4a8_moe_equivalence(self):
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn.parameter import Parameter

        from sglang.srt.layers.quantization.fp8 import cast_e2m1fn_to_e4m3fn
        from sglang.srt.runtime_context import get_context, get_parallel
        from sglang.test.test_utils import enter_scope
        from sglang_tpu.fp8_moe import (
            _dequantize_dsv4_mxfp4_block32,
            prepare_tpu_fp8_moe_weights,
        )
        from sglang_tpu.moe.pallas.fused_moe_op import fused_moe_call

        enter_scope(self, get_parallel().override(tp_size=1))
        override = get_context().override_server_args(
            pp_size=1, tpu_fp8_runtime="channelwise"
        )
        override.install()
        self.addCleanup(override.restore)

        num_tokens = 16
        num_experts = 4
        topk = 2
        hidden_size = 1024

        g = torch.Generator().manual_seed(2026)
        x_cpu = torch.randn(num_tokens, hidden_size, generator=g, dtype=torch.bfloat16)
        topk_ids_cpu = (
            torch.arange(num_tokens * topk, dtype=torch.int32).reshape(num_tokens, topk)
            % num_experts
        )
        topk_weights_cpu = torch.softmax(
            torch.randn(num_tokens, topk, generator=g, dtype=torch.float32), dim=-1
        ).to(torch.bfloat16)

        x = x_cpu.to(self.device)
        topk_ids = topk_ids_cpu.to(self.device)
        topk_weights = topk_weights_cpu.to(self.device)

        for inter_size in (512, 384):
            with self.subTest(inter_size=inter_size):
                w13_u8 = torch.randint(
                    0,
                    256,
                    (num_experts, 2 * inter_size, hidden_size // 2),
                    dtype=torch.uint8,
                    generator=g,
                )
                w2_u8 = torch.randint(
                    0,
                    256,
                    (num_experts, hidden_size, inter_size // 2),
                    dtype=torch.uint8,
                    generator=g,
                )
                w13_scale_fp32 = torch.full(
                    (num_experts, 2 * inter_size, hidden_size // 32),
                    2.0**-5,
                    dtype=torch.float32,
                )
                w2_scale_fp32 = torch.full(
                    (num_experts, hidden_size, inter_size // 32),
                    2.0**-5,
                    dtype=torch.float32,
                )

                # 1. Exact PyTorch BF16 reference on CPU
                w13_ref = _dequantize_dsv4_mxfp4_block32(w13_u8, w13_scale_fp32).to(
                    torch.bfloat16
                )
                w2_ref = _dequantize_dsv4_mxfp4_block32(w2_u8, w2_scale_fp32).to(
                    torch.bfloat16
                )
                ref_out = torch.zeros_like(x_cpu, dtype=torch.float32)
                for t in range(num_tokens):
                    for k_idx in range(topk):
                        e = int(topk_ids_cpu[t, k_idx].item())
                        weight = float(topk_weights_cpu[t, k_idx].item())
                        gate_up = F.linear(x_cpu[t : t + 1], w13_ref[e]).float()
                        gate = torch.clamp(gate_up[:, :inter_size], max=10.0)
                        up = torch.clamp(gate_up[:, inter_size:], min=-10.0, max=10.0)
                        act = (F.silu(gate) * up).to(torch.bfloat16)
                        down = F.linear(act, w2_ref[e]).float()
                        ref_out[t] += weight * down.squeeze(0)
                ref_out = ref_out.to(torch.bfloat16)

                # 2. Phase 2 Native Packed-FP4 (W4A8) on TPU
                layer_w4a8 = nn.Module()
                layer_w4a8.w13_weight = Parameter(
                    w13_u8.view(torch.int8).to(self.device), requires_grad=False
                )
                layer_w4a8.w2_weight = Parameter(
                    w2_u8.view(torch.int8).to(self.device), requires_grad=False
                )
                layer_w4a8.w13_weight_scale_inv = Parameter(
                    w13_scale_fp32.to(self.device), requires_grad=False
                )
                layer_w4a8.w2_weight_scale_inv = Parameter(
                    w2_scale_fp32.to(self.device), requires_grad=False
                )
                prepare_tpu_fp8_moe_weights(
                    layer_w4a8,
                    block_quant=True,
                    weight_block_size=[128, 128],
                    use_mxfp8=False,
                    is_fp4_expert=True,
                )
                out_w4a8 = fused_moe_call(
                    hidden_states=x,
                    w1=layer_w4a8.w13_weight,
                    w2=layer_w4a8.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    topk=topk,
                    activation="silu_and_mul_clamp",
                    w1_scale=layer_w4a8.w13_weight_scale,
                    w2_scale=layer_w4a8.w2_weight_scale,
                ).cpu()

                self.assertEqual(tuple(out_w4a8.shape), (num_tokens, hidden_size))
                self.assertFalse(torch.isnan(out_w4a8).any().item())
                self.assertFalse(torch.isinf(out_w4a8).any().item())

                cos_vs_ref = F.cosine_similarity(
                    out_w4a8.float().flatten().unsqueeze(0),
                    ref_out.float().flatten().unsqueeze(0),
                ).item()
                self.assertGreater(cos_vs_ref, 0.99)

                # 3. Compare against Phase 1 W8A8 dequant-to-FP8 path on aligned size
                if inter_size == 512:
                    layer_w8a8 = nn.Module()
                    w13_fp8_list, s13_fp8_list = [], []
                    w2_fp8_list, s2_fp8_list = [], []
                    for e in range(num_experts):
                        w13_e, s13_e = cast_e2m1fn_to_e4m3fn(
                            w13_u8[e].view(torch.int8), w13_scale_fp32[e]
                        )
                        w2_e, s2_e = cast_e2m1fn_to_e4m3fn(
                            w2_u8[e].view(torch.int8), w2_scale_fp32[e]
                        )
                        w13_fp8_list.append(w13_e)
                        s13_fp8_list.append(s13_e)
                        w2_fp8_list.append(w2_e)
                        s2_fp8_list.append(s2_e)
                    layer_w8a8.w13_weight = Parameter(
                        torch.stack(w13_fp8_list).to(self.device), requires_grad=False
                    )
                    layer_w8a8.w13_weight_scale_inv = Parameter(
                        torch.stack(s13_fp8_list).float().to(self.device),
                        requires_grad=False,
                    )
                    layer_w8a8.w2_weight = Parameter(
                        torch.stack(w2_fp8_list).to(self.device), requires_grad=False
                    )
                    layer_w8a8.w2_weight_scale_inv = Parameter(
                        torch.stack(s2_fp8_list).float().to(self.device),
                        requires_grad=False,
                    )
                    prepare_tpu_fp8_moe_weights(
                        layer_w8a8,
                        block_quant=True,
                        weight_block_size=[128, 128],
                        use_mxfp8=False,
                        is_fp4_expert=False,
                    )
                    out_w8a8 = fused_moe_call(
                        hidden_states=x,
                        w1=layer_w8a8.w13_weight,
                        w2=layer_w8a8.w2_weight,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        topk=topk,
                        activation="silu_and_mul_clamp",
                        w1_scale=layer_w8a8.w13_weight_scale,
                        w2_scale=layer_w8a8.w2_weight_scale,
                    ).cpu()
                    cos_vs_w8a8 = F.cosine_similarity(
                        out_w4a8.float().flatten().unsqueeze(0),
                        out_w8a8.float().flatten().unsqueeze(0),
                    ).item()
                    self.assertGreater(cos_vs_w8a8, 0.99)


if __name__ == "__main__":
    unittest.main()
