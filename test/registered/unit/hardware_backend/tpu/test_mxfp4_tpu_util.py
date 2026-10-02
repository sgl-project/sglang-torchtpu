"""Unit tests for the TPU MXFP4 MoE weight preparation (CPU only)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _EXPERTS_PER_CHUNK,
    prepare_tpu_mxfp4_moe_weights,
    process_mxfp4_moe_weights_bf16,
)
from sglang.test.test_utils import CustomTestCase

HIDDEN, INTER = 64, 96


def _fake_layer(
    num_experts: int,
    seed: int = 0,
    activation: str = "swigluoai",
    hidden: int = HIDDEN,
) -> types.SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    u8 = lambda *shape: torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    # e8m0 scales near 1.0 keep the dequantized values finite.
    e8 = lambda *shape: torch.randint(120, 134, shape, dtype=torch.uint8, generator=g)
    layer = types.SimpleNamespace()
    layer.w13_weight = Parameter(
        u8(num_experts, 2 * INTER, hidden // 2), requires_grad=False
    )
    layer.w13_weight_scale = Parameter(
        e8(num_experts, 2 * INTER, hidden // 32), requires_grad=False
    )
    layer.w13_weight_bias = Parameter(
        torch.randn(num_experts, 2 * INTER, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        u8(num_experts, hidden, INTER // 2), requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        e8(num_experts, hidden, INTER // 32), requires_grad=False
    )
    layer.w2_weight_bias = Parameter(
        torch.randn(num_experts, hidden, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.moe_runner_config = types.SimpleNamespace(activation=activation)
    return layer


class TestChunkedExpertPreparation(CustomTestCase):
    def test_experts_prepared_in_groups_match_the_whole_tensor(self):
        """The experts are dequantized a group at a time; with a group count
        that does not divide the expert count, the assembled weights and
        biases must equal the single-tensor result expert for expert."""
        num_experts = 2 * _EXPERTS_PER_CHUNK + 8
        layer = _fake_layer(num_experts)
        whole = process_mxfp4_moe_weights_bf16(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            layer.w13_weight_bias.data,
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
            layer.w2_weight_bias.data,
            w13_interleaved=True,
        )
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        got = (
            layer.w13_weight,
            layer.w13_weight_bias,
            layer.w2_weight,
            layer.w2_weight_bias,
        )
        for name, a, b in zip(("w13", "w13_bias", "w2", "w2_bias"), whole, got):
            self.assertEqual(tuple(a.shape), tuple(b.shape), name)
            self.assertEqual(a.dtype, b.dtype, name)
            self.assertTrue(torch.equal(a, b), name)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)


class TestPackedPreparation(CustomTestCase):
    def test_non_interleaved_aligned_keeps_packed_fp4(self):
        """Kimi-K3-style checkpoints keep packed FP4 in gmm_v2's (E, K, N) layout."""
        layer = _fake_layer(num_experts=4, activation="situ", hidden=128)
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=False)
        n_pad = 128  # INTER=96 pads each gate/up half to 128
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.w13_weight.shape), (4, 128 // 2, 2 * n_pad))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (4, 128 // 32, 1, 2 * n_pad)
        )
        self.assertEqual(tuple(layer.w2_weight.shape), (4, n_pad // 2, 128))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (4, n_pad // 32, 1, 128))
        self.assertIsNone(layer.w13_weight_bias)
        self.assertIsNone(layer.w2_weight_bias)

    def test_misaligned_hidden_falls_back_to_bf16(self):
        """The packed path cannot pad K, so a misaligned hidden size dequantizes."""
        layer = _fake_layer(num_experts=4, activation="situ")
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)


if __name__ == "__main__":
    unittest.main()
