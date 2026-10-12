"""Bounded FP8 expert conversion and runtime transposition of a pipeline stage."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import expert_loading, fp8_moe

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertChunks(CustomTestCase):
    def test_transpose_preserves_expert_order_with_an_uneven_final_chunk(self):
        weight = torch.arange(5 * 4 * 6).reshape(5, 4, 6)
        param = torch.nn.Parameter(weight.clone(), requires_grad=False)
        with patch.object(expert_loading, "_TRANSPOSE_CHUNK_ELEMENTS", 2 * 4 * 6):
            expert_loading.transpose_expert_chunks(param)
        torch.testing.assert_close(param.data, weight.transpose(1, 2).contiguous())
        self.assertTrue(param.data.is_contiguous())

    def test_old_weight_is_freed_before_the_chunks_are_joined(self):
        """Joining the converted chunks must not also hold the old weight."""
        param = torch.nn.Parameter(
            torch.arange(5 * 4 * 6).reshape(5, 4, 6), requires_grad=False
        )
        old = param.data
        freed_at_join = []
        cat = torch.cat

        def recording_cat(tensors, dim=0):
            freed_at_join.append(old.untyped_storage().nbytes() == 0)
            return cat(tensors, dim=dim)

        with (
            patch.object(expert_loading, "_TRANSPOSE_CHUNK_ELEMENTS", 2 * 4 * 6),
            patch.object(torch, "cat", recording_cat),
        ):
            expert_loading.transpose_expert_chunks(param)
        self.assertEqual(freed_at_join, [True])

    def test_requantized_chunks_match_whole_experts_in_both_layouts(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                layer = torch.nn.Module()
                expected = {}
                for name in ("w13_weight", "w2_weight"):
                    weight = torch.arange(5 * 4 * 6).reshape(5, 4, 6).float() / 32
                    weight = weight.to(torch.float8_e4m3fn)
                    scale = torch.arange(5 * 2 * 3).reshape(5, 2, 3).float() / 16 + 1
                    layer.register_parameter(
                        name, torch.nn.Parameter(weight.clone(), requires_grad=False)
                    )
                    layer.register_parameter(
                        f"{name}_scale_inv",
                        torch.nn.Parameter(scale.clone(), requires_grad=False),
                    )
                    dequantized = fp8_moe.block_quant_dequant(
                        weight, scale, (2, 2), torch.float32
                    )
                    q, s = fp8_moe._quantize_moe_weight_channelwise(
                        dequantized, weight.dtype
                    )
                    expected[name] = q.transpose(1, 2).contiguous() if transpose else q
                    expected[f"{name}_scale"] = s
                with (
                    patch.object(expert_loading, "_REQUANT_CHUNK_BYTES", 2 * 4 * 6 * 4),
                    patch.object(
                        fp8_moe.current_platform,
                        "get_available_memory",
                        return_value=(1 << 30, 1 << 30),
                    ),
                ):
                    expert_loading.requantize_stage_experts_to_channelwise(
                        layer, (2, 2), transpose=transpose
                    )
                for name, value in expected.items():
                    torch.testing.assert_close(
                        getattr(layer, name).float(), value.float()
                    )
                self.assertIsNone(layer.weight_block_size)
                self.assertIsNone(layer.w13_weight_scale_inv)
                self.assertIsNone(layer.w2_weight_scale_inv)


if __name__ == "__main__":
    unittest.main()
