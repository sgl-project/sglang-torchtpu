"""Online RL weight reload tests for TPU FP8 dense linear and MoE layers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.parameter import BlockQuantScaleParameter, ModelWeightParameter
from sglang_tpu.fp8_linear import (
    prepare_tpu_fp8_linear_weights,
    prepare_tpu_online_fp8_linear_weights,
    record_tpu_fp8_linear_reload_state,
    restore_tpu_fp8_linear_weights_before_loading,
)
from sglang_tpu.fp8_moe import (
    prepare_tpu_online_fp8_moe_weights,
    record_tpu_fp8_moe_reload_state,
    restore_tpu_fp8_moe_weights_before_loading,
)

_FP8 = torch.float8_e4m3fn


def _loader_fn(param: torch.Tensor, loaded: torch.Tensor) -> None:
    param.data.copy_(loaded)


class TestTpuFp8Reload(unittest.TestCase):
    def test_online_fp8_linear_preserves_parameter_identity_and_weight_loader_across_reloads(
        self,
    ):
        layer = nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.weight = ModelWeightParameter(
            data=torch.randn(32, 64, dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=_loader_fn,
        )
        record_tpu_fp8_linear_reload_state(layer)
        orig_weight_param = layer.weight

        prepare_tpu_online_fp8_linear_weights(layer, dtype=_FP8)
        orig_scale_param = layer.weight_scale
        self.assertIs(layer.weight, orig_weight_param)
        self.assertIs(layer.weight.weight_loader, _loader_fn)
        self.assertEqual(tuple(layer.weight.shape), (64, 32))

        restore_tpu_fp8_linear_weights_before_loading(layer)
        self.assertIs(layer.weight, orig_weight_param)
        self.assertEqual(tuple(layer.weight.shape), (32, 64))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

        fresh = torch.randn(32, 64, dtype=torch.bfloat16) * 3.0
        layer.weight.weight_loader(layer.weight, fresh)
        prepare_tpu_online_fp8_linear_weights(layer, dtype=_FP8)

        self.assertIs(layer.weight, orig_weight_param)
        self.assertIs(layer.weight_scale, orig_scale_param)
        self.assertEqual(tuple(layer.weight.shape), (64, 32))
        dequant = layer.weight.t().float() * layer.weight_scale
        rel_err = (dequant - fresh.float()).abs().max() / fresh.float().abs().max()
        self.assertLess(rel_err.item(), 0.08)

    def test_block_fp8_linear_restores_scale_inv_for_channelwise_reload(self):
        layer = nn.Module()
        layer.logical_widths = [128]
        layer.weight = ModelWeightParameter(
            data=torch.ones(128, 128, dtype=_FP8),
            input_dim=1,
            output_dim=0,
            weight_loader=_loader_fn,
        )
        scale_inv = BlockQuantScaleParameter(
            data=torch.full((1, 1), 0.5, dtype=torch.float32),
            input_dim=1,
            output_dim=0,
            weight_loader=_loader_fn,
        )
        scale_inv.format_ue8m0 = False
        layer.register_parameter("weight_scale_inv", scale_inv)
        record_tpu_fp8_linear_reload_state(layer)

        with patch(
            "sglang_tpu.fp8_linear._get_tpu_fp8_runtime", return_value="channelwise"
        ):
            prepare_tpu_fp8_linear_weights(layer, weight_block_size=[128, 128])
            self.assertIsNone(layer.weight_scale_inv)
            self.assertEqual(tuple(layer.weight.shape), (128, 128))

            restore_tpu_fp8_linear_weights_before_loading(layer)
            self.assertIsNotNone(layer.weight_scale_inv)
            self.assertIs(layer.weight_scale_inv.weight_loader, _loader_fn)

            layer.weight.weight_loader(
                layer.weight, torch.full((128, 128), 2.0, dtype=_FP8)
            )
            layer.weight_scale_inv.weight_loader(
                layer.weight_scale_inv, torch.full((1, 1), 4.0, dtype=torch.float32)
            )
            prepare_tpu_fp8_linear_weights(layer, weight_block_size=[128, 128])

        dequant = layer.weight.t().float() * layer.weight_scale
        self.assertTrue(torch.allclose(dequant, torch.full((128, 128), 8.0), atol=0.5))

    def test_online_fp8_moe_channelwise_reload_preserves_parameters_and_scales(self):
        layer = nn.Module()
        w13 = nn.Parameter(
            torch.randn(4, 32, 64, dtype=torch.bfloat16), requires_grad=False
        )
        w2 = nn.Parameter(
            torch.randn(4, 64, 16, dtype=torch.bfloat16), requires_grad=False
        )
        w13.weight_loader = _loader_fn
        w2.weight_loader = _loader_fn
        layer.register_parameter("w13_weight", w13)
        layer.register_parameter("w2_weight", w2)
        layer.register_parameter(
            "w13_weight_scale", nn.Parameter(torch.ones(4, 2), requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight_scale", nn.Parameter(torch.ones(4), requires_grad=False)
        )
        record_tpu_fp8_moe_reload_state(layer)

        fake_parallel = SimpleNamespace(pp_size=1)
        with (
            patch("sglang_tpu.fp8_moe.get_parallel", return_value=fake_parallel),
            patch(
                "sglang_tpu.fp8_moe._get_tpu_fp8_runtime", return_value="channelwise"
            ),
        ):
            prepare_tpu_online_fp8_moe_weights(layer, dtype=_FP8)
            w13_scale_param = layer.w13_weight_scale
            w2_scale_param = layer.w2_weight_scale
            self.assertEqual(tuple(layer.w13_weight.shape), (4, 64, 32))
            self.assertEqual(tuple(layer.w13_weight_scale.shape), (4, 1, 1, 32))

            restore_tpu_fp8_moe_weights_before_loading(layer)
            self.assertIs(layer.w13_weight, w13)
            self.assertIs(layer.w2_weight, w2)
            self.assertIs(layer.w13_weight.weight_loader, _loader_fn)
            self.assertEqual(tuple(layer.w13_weight.shape), (4, 32, 64))
            self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)

            fresh_w13 = torch.randn(4, 32, 64, dtype=torch.bfloat16) * 2.0
            fresh_w2 = torch.randn(4, 64, 16, dtype=torch.bfloat16) * 2.0
            layer.w13_weight.weight_loader(layer.w13_weight, fresh_w13)
            layer.w2_weight.weight_loader(layer.w2_weight, fresh_w2)
            prepare_tpu_online_fp8_moe_weights(layer, dtype=_FP8)

        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w13_weight_scale, w13_scale_param)
        self.assertIs(layer.w2_weight_scale, w2_scale_param)
        dequant_w13 = (
            layer.w13_weight.transpose(1, 2).float()
            * layer.w13_weight_scale[:, 0, 0, :, None].float()
        )
        rel_err = (
            dequant_w13 - fresh_w13.float()
        ).abs().max() / fresh_w13.float().abs().max()
        self.assertLess(rel_err.item(), 0.08)


if __name__ == "__main__":
    unittest.main()
