"""Online fp8 quantization of dense linear weights on TPU (#1767)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization import fp8
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import per_output_channel_quant_fp8
from sglang.test.test_utils import CustomTestCase

_FP8 = torch.float8_e4m3fn


def _online_fp8_method() -> Fp8LinearMethod:
    """An Fp8LinearMethod for an unquantized checkpoint, as `Fp8Config()` builds."""
    method = object.__new__(Fp8LinearMethod)
    method.block_quant = False
    method.use_mxfp8 = False
    method.is_checkpoint_fp8_serialized = False
    method.cutlass_fp8_supported = False
    method.use_marlin = False
    return method


def _quantize_on_tpu(weight: torch.Tensor) -> nn.Module:
    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    with patch.object(fp8, "_is_tpu", True):
        _online_fp8_method().process_weights_after_loading(layer)
    return layer


def _replica_qkv_shards() -> tuple[torch.Tensor, torch.Tensor]:
    """Two TP ranks' fused q|k|v shards that share one replicated kv head.

    Each rank holds different q rows. Rank 0's q rows have the larger amax,
    so a shard-wide scale would differ between the replicas.
    """
    generator = torch.Generator().manual_seed(0)
    kv = torch.randn(2 * 16, 32, generator=generator)
    q_rank0 = torch.randn(32, 32, generator=generator) * 8.0
    q_rank1 = torch.randn(32, 32, generator=generator)
    return torch.cat([q_rank0, kv]), torch.cat([q_rank1, kv])


class TestTpuOnlineFp8Linear(CustomTestCase):
    def test_replicated_kv_rows_are_byte_identical_across_ranks(self):
        shard0, shard1 = _replica_qkv_shards()

        layer0 = _quantize_on_tpu(shard0)
        layer1 = _quantize_on_tpu(shard1)

        # Post-load layout is [K, N]; kv columns follow the 32 q columns.
        kv0 = layer0.weight[:, 32:].contiguous().view(torch.uint8)
        kv1 = layer1.weight[:, 32:].contiguous().view(torch.uint8)
        self.assertTrue(torch.equal(kv0, kv1))
        self.assertTrue(torch.equal(layer0.weight_scale[32:], layer1.weight_scale[32:]))

    def test_per_tensor_scale_would_split_the_replicas(self):
        # Guards the premise of the test above: without per-row scales the
        # same kv rows get different fp8 bytes on the two ranks.
        shard0, shard1 = _replica_qkv_shards()

        q0, _ = fp8.input_to_float8(shard0, dtype=_FP8)
        q1, _ = fp8.input_to_float8(shard1, dtype=_FP8)

        self.assertFalse(
            torch.equal(q0[32:].view(torch.uint8), q1[32:].view(torch.uint8))
        )

    def test_layer_keeps_post_load_layout_and_row_scales(self):
        weight = torch.tensor([[1.0, -2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])

        layer = _quantize_on_tpu(weight)

        self.assertEqual(layer.weight.dtype, _FP8)
        self.assertEqual(tuple(layer.weight.shape), (4, 2))
        self.assertEqual(tuple(layer.weight_scale.shape), (2, 1))
        self.assertIsNone(layer.input_scale)
        fp8_max = torch.finfo(_FP8).max
        expected_scales = torch.tensor([[2.0 / fp8_max], [1e-12]])
        self.assertTrue(torch.equal(layer.weight_scale, expected_scales))
        # An all-zero row keeps the finite floor scale and quantizes to zeros.
        self.assertEqual(layer.weight[:, 1].float().tolist(), [0.0] * 4)
        self.assertEqual(
            layer.weight[:, 0].float().tolist(), [224.0, -448.0, 112.0, 0.0]
        )

    def test_channelwise_round_trip_is_within_fp8_precision(self):
        weight = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))

        qweight, scale = per_output_channel_quant_fp8(weight, dtype=_FP8)

        dequant = qweight.float() * scale
        relative_error = (dequant - weight).abs().max() / weight.abs().max()
        self.assertLess(relative_error.item(), 0.07)


if __name__ == "__main__":
    unittest.main()
