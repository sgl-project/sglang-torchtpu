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
"""CPU unit tests for DeepSeek-V4 MXFP4 -> TPU block-512 packed-FP4 (W4A8) MoE weights."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.fp8 import cast_e2m1fn_to_e4m3fn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu.fp8_moe import (
    _FP4_E2M1_LUT,
    _FP4_E2M1_THRESHOLDS,
    _TPU_FP4_BLOCK_SIZE,
    _dequantize_dsv4_mxfp4_block32,
    prepare_tpu_fp8_moe_weights,
)
from sglang_tpu.fp8_moe_fused_rs import _prepare_fp8_fused_rs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _unpack_gmm_v2_fp4_to_nk(
    weight_kn_u8: torch.Tensor,
    scale_gmm: torch.Tensor,
) -> torch.Tensor:
    """Unpack gmm_v2 ``(E, K/2, N)`` uint8 + ``(E, K/512, 1, N)`` scale back to ``(E, N, K)`` fp32."""
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=weight_kn_u8.device)
    w_nk_u8 = weight_kn_u8.permute(0, 2, 1).contiguous()
    low = (w_nk_u8 & 0x0F).to(torch.long)
    high = ((w_nk_u8 >> 4) & 0x0F).to(torch.long)
    w_fp32 = torch.stack((lut[low], lut[high]), dim=-1).flatten(-2)
    scale_nk = scale_gmm.squeeze(2).permute(0, 2, 1).contiguous()
    return (
        w_fp32.unflatten(-1, (-1, _TPU_FP4_BLOCK_SIZE)) * scale_nk.unsqueeze(-1)
    ).flatten(-2)


def _make_mock_mxfp4_layer(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    scale_mode: str = "uniform",
    seed: int = 42,
) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    w13_u8 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        generator=g,
    )
    w2_u8 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        generator=g,
    )
    if scale_mode == "uniform":
        w13_exp = torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            125,
            dtype=torch.uint8,
        )
        w2_exp = torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            125,
            dtype=torch.uint8,
        )
    elif scale_mode == "varying":
        # Smoothly varying exponents across 32-element sub-blocks within each 512-block
        w13_base = torch.randint(
            123,
            126,
            (num_experts, 2 * intermediate_size, hidden_size // 512, 1),
            dtype=torch.int32,
            generator=g,
        )
        w13_delta = torch.randint(
            0,
            2,
            (num_experts, 2 * intermediate_size, hidden_size // 512, 16),
            dtype=torch.int32,
            generator=g,
        )
        w13_exp = (w13_base + w13_delta).flatten(-2).to(torch.uint8)

        w2_blocks_512 = max(1, intermediate_size // 512)
        sub_blocks = (intermediate_size // 32) // w2_blocks_512
        w2_base = torch.randint(
            123,
            126,
            (num_experts, hidden_size, w2_blocks_512, 1),
            dtype=torch.int32,
            generator=g,
        )
        w2_delta = torch.randint(
            0,
            2,
            (num_experts, hidden_size, w2_blocks_512, sub_blocks),
            dtype=torch.int32,
            generator=g,
        )
        w2_exp = (w2_base + w2_delta).flatten(-2).to(torch.uint8)
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")

    w13_scale_fp32 = torch.exp2(w13_exp.to(torch.int32).to(torch.float32) - 127.0)
    w2_scale_fp32 = torch.exp2(w2_exp.to(torch.int32).to(torch.float32) - 127.0)

    layer = nn.Module()
    layer.w13_weight = Parameter(w13_u8.view(torch.int8), requires_grad=False)
    layer.w2_weight = Parameter(w2_u8.view(torch.int8), requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(w13_scale_fp32, requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(w2_scale_fp32, requires_grad=False)
    layer.w13_exp_u8 = w13_exp
    layer.w2_exp_u8 = w2_exp
    return layer


class TestDsv4Mxfp4MoeWeights(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_parallel

        enter_scope(self, get_parallel().override(tp_size=1))

    def test_e2m1_threshold_quantization_matches_lut_argmin(self):
        lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32)
        g = torch.Generator().manual_seed(123)
        random_vals = (torch.rand(50000, generator=g, dtype=torch.float32) * 14.0) - 7.0
        special_vals = torch.tensor(
            [
                0.0,
                -0.0,
                *_FP4_E2M1_LUT,
                *_FP4_E2M1_THRESHOLDS,
                *[-t for t in _FP4_E2M1_THRESHOLDS],
            ],
            dtype=torch.float32,
        )
        x = torch.cat([random_vals, special_vals]).clamp(-6.0, 6.0)

        # Reference LUT argmin
        ref_idx = torch.abs(x.unsqueeze(-1) - lut).argmin(dim=-1).to(torch.uint8)

        # Threshold implementation used in _quantize_tpu_fp4_block512_kmajor
        abs_x = x.abs()
        mag_idx = torch.zeros_like(abs_x, dtype=torch.uint8)
        for threshold in _FP4_E2M1_THRESHOLDS:
            mag_idx.add_((abs_x > threshold).to(torch.uint8))
        sign_bit = ((x < 0) & (mag_idx > 0)).to(torch.uint8) << 3
        got_idx = mag_idx | sign_bit

        self.assertTrue(torch.equal(got_idx, ref_idx))

    def test_aligned_flash_shapes_dtypes_and_buffer_ownership(self):
        num_experts, hidden_size, inter_size = 4, 1024, 512
        layer = _make_mock_mxfp4_layer(num_experts, hidden_size, inter_size)

        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=True,
            weight_block_size=[128, 128],
            use_mxfp8=False,
            is_fp4_expert=True,
        )

        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(
            tuple(layer.w13_weight.shape),
            (num_experts, hidden_size // 2, 2 * inter_size),
        )
        self.assertEqual(
            tuple(layer.w2_weight.shape),
            (num_experts, inter_size // 2, hidden_size),
        )

        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape),
            (num_experts, hidden_size // 512, 1, 2 * inter_size),
        )
        self.assertEqual(
            tuple(layer.w2_weight_scale.shape),
            (num_experts, inter_size // 512, 1, hidden_size),
        )

        # Scales must own their 4D storage (no view aliasing)
        self.assertIsNone(layer.w13_weight_scale.data._base)
        self.assertIsNone(layer.w2_weight_scale.data._base)
        self.assertTrue(layer.w13_weight.data.is_contiguous())
        self.assertTrue(layer.w2_weight.data.is_contiguous())

        # Old block-quant scale_inv params must be cleared and weight_block_size unset
        self.assertIsNone(getattr(layer, "w13_weight_scale_inv", None))
        self.assertIsNone(getattr(layer, "w2_weight_scale_inv", None))
        self.assertIsNone(layer.weight_block_size)

        # Fused-RS must decline packed-FP4 uint8 weights and leave layer on gmm_v2
        self.assertFalse(_prepare_fp8_fused_rs(layer, allow_no_combine=False))

    def test_unaligned_pro_intermediate_symmetric_padding(self):
        # DeepSeek-V4 Pro at TP=8 has intermediate_size_per_partition = 3072 // 8 = 384,
        # which pads symmetrically to 512.
        num_experts, hidden_size, inter_size = 2, 1024, 384
        inter_padded = 512
        layer = _make_mock_mxfp4_layer(num_experts, hidden_size, inter_size)

        ref_w13_fp32 = _dequantize_dsv4_mxfp4_block32(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        ref_w2_fp32 = _dequantize_dsv4_mxfp4_block32(
            layer.w2_weight.data, layer.w2_weight_scale_inv.data
        )

        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=True,
            weight_block_size=[128, 128],
            use_mxfp8=False,
            is_fp4_expert=True,
        )

        self.assertEqual(
            tuple(layer.w13_weight.shape),
            (num_experts, hidden_size // 2, 2 * inter_padded),
        )
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape),
            (num_experts, hidden_size // 512, 1, 2 * inter_padded),
        )
        self.assertEqual(
            tuple(layer.w2_weight.shape),
            (num_experts, inter_padded // 2, hidden_size),
        )
        self.assertEqual(
            tuple(layer.w2_weight_scale.shape),
            (num_experts, inter_padded // 512, 1, hidden_size),
        )

        # Gate padding [384:512] and up padding [512+384:1024] must be strictly zero
        self.assertTrue(torch.all(layer.w13_weight[:, :, inter_size:inter_padded] == 0))
        self.assertTrue(
            torch.all(
                layer.w13_weight[:, :, inter_padded + inter_size : 2 * inter_padded]
                == 0
            )
        )
        self.assertTrue(
            torch.all(layer.w13_weight_scale[:, :, :, inter_size:inter_padded] == 0.0)
        )
        self.assertTrue(
            torch.all(
                layer.w13_weight_scale[
                    :, :, :, inter_padded + inter_size : 2 * inter_padded
                ]
                == 0.0
            )
        )

        # Unpack and verify active gate [0:384] and up [512:512+384] slices match reference
        w13_unpacked = _unpack_gmm_v2_fp4_to_nk(
            layer.w13_weight.data, layer.w13_weight_scale.data
        )
        torch.testing.assert_close(
            w13_unpacked[:, :inter_size, :], ref_w13_fp32[:, :inter_size, :]
        )
        torch.testing.assert_close(
            w13_unpacked[:, inter_padded : inter_padded + inter_size, :],
            ref_w13_fp32[:, inter_size:, :],
        )

        # W2 padded K region [384 // 2 : 512 // 2] must be strictly zero
        self.assertTrue(
            torch.all(layer.w2_weight[:, inter_size // 2 : inter_padded // 2, :] == 0)
        )
        w2_unpacked = _unpack_gmm_v2_fp4_to_nk(
            layer.w2_weight.data, layer.w2_weight_scale.data
        )
        torch.testing.assert_close(w2_unpacked[:, :, :inter_size], ref_w2_fp32)
        self.assertTrue(torch.all(w2_unpacked[:, :, inter_size:] == 0.0))

    def test_roundtrip_numerical_fidelity_and_uint8_vs_fp32_scale(self):
        num_experts, hidden_size, inter_size = 4, 1024, 512
        layer_fp32_scale = _make_mock_mxfp4_layer(
            num_experts, hidden_size, inter_size, scale_mode="varying", seed=99
        )
        layer_u8_scale = _make_mock_mxfp4_layer(
            num_experts, hidden_size, inter_size, scale_mode="varying", seed=99
        )
        layer_u8_scale.w13_weight_scale_inv = Parameter(
            layer_u8_scale.w13_exp_u8, requires_grad=False
        )
        layer_u8_scale.w2_weight_scale_inv = Parameter(
            layer_u8_scale.w2_exp_u8, requires_grad=False
        )

        ref_w13 = _dequantize_dsv4_mxfp4_block32(
            layer_fp32_scale.w13_weight.data,
            layer_fp32_scale.w13_weight_scale_inv.data,
        )
        ref_w2 = _dequantize_dsv4_mxfp4_block32(
            layer_fp32_scale.w2_weight.data,
            layer_fp32_scale.w2_weight_scale_inv.data,
        )

        for layer in (layer_fp32_scale, layer_u8_scale):
            prepare_tpu_fp8_moe_weights(
                layer,
                block_quant=True,
                weight_block_size=[128, 128],
                use_mxfp8=False,
                is_fp4_expert=True,
            )

        # uint8 e8m0 scale input and float32 scale input must produce identical results
        self.assertTrue(
            torch.equal(layer_fp32_scale.w13_weight, layer_u8_scale.w13_weight)
        )
        self.assertTrue(
            torch.equal(layer_fp32_scale.w2_weight, layer_u8_scale.w2_weight)
        )
        self.assertTrue(
            torch.equal(
                layer_fp32_scale.w13_weight_scale, layer_u8_scale.w13_weight_scale
            )
        )
        self.assertTrue(
            torch.equal(
                layer_fp32_scale.w2_weight_scale, layer_u8_scale.w2_weight_scale
            )
        )

        recon_w13 = _unpack_gmm_v2_fp4_to_nk(
            layer_fp32_scale.w13_weight.data, layer_fp32_scale.w13_weight_scale.data
        )
        recon_w2 = _unpack_gmm_v2_fp4_to_nk(
            layer_fp32_scale.w2_weight.data, layer_fp32_scale.w2_weight_scale.data
        )
        cos_w13 = F.cosine_similarity(
            recon_w13.flatten().unsqueeze(0), ref_w13.flatten().unsqueeze(0)
        ).item()
        cos_w2 = F.cosine_similarity(
            recon_w2.flatten().unsqueeze(0), ref_w2.flatten().unsqueeze(0)
        ).item()
        self.assertGreater(cos_w13, 0.99)
        self.assertGreater(cos_w2, 0.99)

    def test_hbm_footprint_halved_vs_fp8_dequant_and_storage_freed(self):
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(
            pp_size=1, tpu_fp8_runtime="blockwise"
        )
        override.install()
        self.addCleanup(override.restore)

        num_experts, hidden_size, inter_size = 4, 1024, 512
        layer_w4a8 = _make_mock_mxfp4_layer(num_experts, hidden_size, inter_size)
        layer_w8a8 = _make_mock_mxfp4_layer(num_experts, hidden_size, inter_size)

        old_w13_storage = layer_w4a8.w13_weight.data.untyped_storage()
        old_w2_storage = layer_w4a8.w2_weight.data.untyped_storage()
        old_w13_s_storage = layer_w4a8.w13_weight_scale_inv.data.untyped_storage()
        old_w2_s_storage = layer_w4a8.w2_weight_scale_inv.data.untyped_storage()

        # Native packed-FP4 (W4A8) path
        prepare_tpu_fp8_moe_weights(
            layer_w4a8,
            block_quant=True,
            weight_block_size=[128, 128],
            use_mxfp8=False,
            is_fp4_expert=True,
        )

        # Old storages must be resized to 0 bytes
        self.assertEqual(old_w13_storage.nbytes(), 0)
        self.assertEqual(old_w2_storage.nbytes(), 0)
        self.assertEqual(old_w13_s_storage.nbytes(), 0)
        self.assertEqual(old_w2_s_storage.nbytes(), 0)

        # Phase 1 dequant-to-FP8 (W8A8) path
        for weight_param, scale_param in (
            (layer_w8a8.w13_weight, layer_w8a8.w13_weight_scale_inv),
            (layer_w8a8.w2_weight, layer_w8a8.w2_weight_scale_inv),
        ):
            new_w, new_s = [], []
            for e in range(num_experts):
                w, s = cast_e2m1fn_to_e4m3fn(weight_param.data[e], scale_param.data[e])
                new_w.append(w)
                new_s.append(s)
            weight_param.data = torch.stack(new_w)
            scale_param.data = torch.stack(new_s).float()
        prepare_tpu_fp8_moe_weights(
            layer_w8a8,
            block_quant=True,
            weight_block_size=[128, 128],
            use_mxfp8=False,
            is_fp4_expert=False,
        )

        w4_weight_bytes = layer_w4a8.w13_weight.nbytes + layer_w4a8.w2_weight.nbytes
        w8_weight_bytes = layer_w8a8.w13_weight.nbytes + layer_w8a8.w2_weight.nbytes
        self.assertEqual(w4_weight_bytes * 2, w8_weight_bytes)

        w4_total_bytes = (
            w4_weight_bytes
            + layer_w4a8.w13_weight_scale.nbytes
            + layer_w4a8.w2_weight_scale.nbytes
        )
        w8_total_bytes = (
            w8_weight_bytes
            + layer_w8a8.w13_weight_scale_inv.nbytes
            + layer_w8a8.w2_weight_scale_inv.nbytes
        )
        self.assertLess(w4_total_bytes, w8_total_bytes * 0.52)

    def test_deepseek_v4_pro_exact_expert_dimensions_zero_padding(self):
        """Verify DeepSeek-V4-Pro routed expert dimensions (`H=7168, I=3072`).

        In DeepSeek-V4-Pro (`384` routed experts, `24` experts/rank at `EP=16`),
        `hidden_size=7168` (`14 * 512`) and `moe_intermediate_size=3072`
        (`6 * 512`) are exact multiples of 512, so native packed-FP4 (`W4A8`)
        requires zero padding (`pad_h=0, pad_i=0`).
        """
        num_experts, hidden_size, inter_size = 1, 7168, 3072
        layer = _make_mock_mxfp4_layer(num_experts, hidden_size, inter_size)

        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=True,
            weight_block_size=[128, 128],
            use_mxfp8=False,
            is_fp4_expert=True,
        )

        # W13: K = 7168 -> K_packed = 3584, N = 2 * 3072 = 6144, num_blocks = 7168 // 512 = 14
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.w13_weight.shape), (1, 3584, 6144))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (1, 14, 1, 6144))

        # W2: K = 3072 -> K_packed = 1536, N = 7168, num_blocks = 3072 // 512 = 6
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.w2_weight.shape), (1, 1536, 7168))
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (1, 6, 1, 7168))

    def test_deepseek_v4_pro_shared_experts_subblock_tp16_sharding(self):
        """Verify DeepSeek-V4-Pro shared_experts sub-block TP=16 sharding (`3072 // 16 = 192`)."""
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.fp8_linear import (
            compute_tpu_fp8_linear_scale_shape,
            prepare_tpu_fp8_linear_weights,
        )

        override = get_context().override_server_args(tpu_fp8_runtime="blockwise")
        override.install()
        self.addCleanup(override.restore)

        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def _make_block_fp8(shape):
            n, k = shape
            w = torch.randn(shape, dtype=torch.float32) * 0.1
            blocked = w.reshape(n // 128, 128, k // 128, 128)
            s = blocked.abs().amax(dim=(-3, -1)).clamp(min=1e-12) / fp8_max
            wq = (
                (blocked / s[:, None, :, None])
                .clamp(min=-fp8_max, max=fp8_max)
                .reshape(shape)
                .to(torch.float8_e4m3fn)
            )
            return wq, s, block_quant_dequant(wq, s, [128, 128], torch.float32)

        gate_w, gate_s, gate_deq = _make_block_fp8((384, 256))
        up_w, up_s, up_deq = _make_block_fp8((384, 256))
        down_w, down_s, down_deq = _make_block_fp8((256, 384))

        for tp_rank in (0, 1):
            start, end = tp_rank * 192, (tp_rank + 1) * 192
            s_start, s_end = start // 128, (end + 127) // 128

            gate_up_layer = torch.nn.Module()
            gate_up_layer.tp_rank = tp_rank
            gate_up_layer.logical_widths = [192, 192]
            scale_rows, scale_cols = compute_tpu_fp8_linear_scale_shape(
                gate_up_layer,
                input_size=256,
                input_size_per_partition=256,
                output_size=768,
                output_size_per_partition=384,
                output_partition_sizes=[192, 192],
                block_n=128,
                block_k=128,
            )
            self.assertEqual((scale_rows, scale_cols), (2 * (s_end - s_start), 2))
            gate_up_layer.weight = torch.nn.Parameter(
                torch.cat([gate_w[start:end], up_w[start:end]], dim=0),
                requires_grad=False,
            )
            gate_up_layer.weight_scale_inv = torch.nn.Parameter(
                torch.cat([gate_s[s_start:s_end], up_s[s_start:s_end]], dim=0),
                requires_grad=False,
            )
            gate_up_layer.input_scale = None
            prepare_tpu_fp8_linear_weights(gate_up_layer, weight_block_size=[128, 128])
            recovered_gu = (
                gate_up_layer.weight.float() * gate_up_layer.weight_scale.reshape(1, -1)
            ).t()
            expected_gu = torch.cat([gate_deq[start:end], up_deq[start:end]], dim=0)
            torch.testing.assert_close(recovered_gu, expected_gu, rtol=0.1, atol=0.1)

            down_layer = torch.nn.Module()
            down_layer.tp_rank = tp_rank
            down_layer.logical_widths = [256]
            scale_rows, scale_cols = compute_tpu_fp8_linear_scale_shape(
                down_layer,
                input_size=384,
                input_size_per_partition=192,
                output_size=256,
                output_size_per_partition=256,
                output_partition_sizes=[256],
                block_n=128,
                block_k=128,
            )
            self.assertEqual((scale_rows, scale_cols), (2, s_end - s_start))
            down_layer.weight = torch.nn.Parameter(
                down_w[:, start:end].contiguous(), requires_grad=False
            )
            down_layer.weight_scale_inv = torch.nn.Parameter(
                down_s[:, s_start:s_end].contiguous(), requires_grad=False
            )
            down_layer.input_scale = None
            prepare_tpu_fp8_linear_weights(down_layer, weight_block_size=[128, 128])
            recovered_down = (
                down_layer.weight.float() * down_layer.weight_scale.reshape(1, -1)
            ).t()
            expected_down = down_deq[:, start:end]
            torch.testing.assert_close(
                recovered_down, expected_down, rtol=0.1, atol=0.1
            )


if __name__ == "__main__":
    unittest.main()
