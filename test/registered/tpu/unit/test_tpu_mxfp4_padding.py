"""Lossless gate/up and contraction padding for GPT-OSS BF16 TPU experts."""

import types
import unittest

import torch

from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    prepare_tpu_mxfp4_moe_weights,
    process_mxfp4_moe_weights_bf16,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


class TestMxfp4Padding(unittest.TestCase):
    def test_gate_bias_has_final_shape_storage(self):
        """Static expert biases must not retain a view requiring each-call materialization."""
        bias = torch.arange(256, dtype=torch.float32).reshape(1, 256)
        _, prepared_bias, _, _ = process_mxfp4_moe_weights_bf16(
            w13_packed=torch.zeros((1, 256, 16), dtype=torch.uint8),
            w13_scale=torch.full((1, 256, 1), 127, dtype=torch.uint8),
            w13_bias=bias,
            w2_packed=torch.zeros((1, 32, 64), dtype=torch.uint8),
            w2_scale=torch.full((1, 32, 4), 127, dtype=torch.uint8),
            w2_bias=torch.zeros((1, 32)),
            w13_interleaved=False,
        )
        torch.testing.assert_close(prepared_bias, bias.unsqueeze(1), rtol=0, atol=0)
        self.assertIsNone(prepared_bias._base)

    def test_live_weights_and_biases_survive_symmetric_padding(self):
        """Moving the gate/up midpoint must not move or alter live channels."""
        torch.manual_seed(7)
        for intermediate, interleaved in ((2880, True), (2880, False), (2560, True)):
            with self.subTest(intermediate=intermediate, interleaved=interleaved):
                hidden = 64
                layer = torch.nn.Module()
                layer.moe_runner_config = types.SimpleNamespace(
                    activation="swigluoai" if interleaved else "silu"
                )
                shapes = {
                    "w13_weight": (1, 2 * intermediate, hidden // 2),
                    "w2_weight": (1, hidden, intermediate // 2),
                }
                raw = {}
                for name, shape in shapes.items():
                    raw[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
                    setattr(
                        layer, name, torch.nn.Parameter(raw[name], requires_grad=False)
                    )
                    scale_shape = (*shape[:-1], shape[-1] // 16)
                    setattr(
                        layer,
                        name + "_scale",
                        torch.nn.Parameter(
                            torch.full(scale_shape, 127, dtype=torch.uint8),
                            requires_grad=False,
                        ),
                    )
                    bias = torch.randn(shape[:2]).bfloat16()
                    raw[name + "_bias"] = bias
                    setattr(
                        layer,
                        name + "_bias",
                        torch.nn.Parameter(bias, requires_grad=False),
                    )

                # Unit E8M0 scales: decode the packed FP4 codes independently.
                lut = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
                )
                decoded = {}
                for name in shapes:
                    packed = raw[name].long()
                    decoded[name] = torch.stack(
                        [lut[packed & 15], lut[packed >> 4]], dim=-1
                    ).flatten(-2)
                gate = slice(0, None, 2) if interleaved else slice(0, intermediate)
                up = slice(1, None, 2) if interleaved else slice(intermediate, None)
                prepare_tpu_mxfp4_moe_weights(layer)
                padded = 3072 if intermediate == 2880 else intermediate
                self.assertEqual(tuple(layer.w13_weight.shape), (1, 128, 2 * padded))
                self.assertEqual(tuple(layer.w2_weight.shape), (1, padded, 128))
                for source, offset in ((gate, 0), (up, padded)):
                    torch.testing.assert_close(
                        layer.w13_weight[
                            :, :hidden, offset : offset + intermediate
                        ].float(),
                        decoded["w13_weight"][:, source].transpose(1, 2),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        layer.w13_weight_bias[:, 0, offset : offset + intermediate],
                        raw["w13_weight_bias"][:, source].float(),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            layer.w13_weight[
                                :, :, offset + intermediate : offset + padded
                            ]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            layer.w13_weight_bias[
                                :, :, offset + intermediate : offset + padded
                            ]
                        ).item(),
                        0,
                    )
                torch.testing.assert_close(
                    layer.w2_weight[:, :intermediate, :hidden].float(),
                    decoded["w2_weight"].transpose(1, 2),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    layer.w2_weight_bias[:, 0, :hidden],
                    raw["w2_weight_bias"].float(),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(
                    torch.count_nonzero(layer.w2_weight[:, intermediate:]).item(), 0
                )
                self.assertEqual(
                    torch.count_nonzero(layer.w13_weight[:, hidden:]).item(), 0
                )
                self.assertEqual(
                    torch.count_nonzero(layer.w2_weight[:, :, hidden:]).item(), 0
                )


if __name__ == "__main__":
    unittest.main()
