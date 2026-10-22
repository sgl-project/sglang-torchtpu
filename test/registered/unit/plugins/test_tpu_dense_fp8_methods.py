# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for plugin-owned dense FP8 methods and checkpoint layouts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.plugins.oot_classes import clear_oot_replacements
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.quantization import dense_fp8
from sglang_tpu.quantization.dense_fp8 import (
    TpuCompressedTensorsW8A8Fp8,
    TpuFp8LinearMethod,
    register_dense_fp8_methods,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config():
    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


class TestDenseFp8Methods(unittest.TestCase):
    def setUp(self):
        self.addCleanup(clear_oot_replacements)
        clear_oot_replacements()

    def test_direct_construction_selects_only_registered_classes(self):
        config = _config()
        self.assertIs(type(Fp8LinearMethod(config)), Fp8LinearMethod)
        register_dense_fp8_methods()
        self.assertIs(type(Fp8LinearMethod(config)), TpuFp8LinearMethod)
        scheme = CompressedTensorsW8A8Fp8(
            QuantizationArgs(strategy=QuantizationStrategy.CHANNEL), False
        )
        self.assertIs(type(scheme), TpuCompressedTensorsW8A8Fp8)

    def test_partition_scale_geometry_and_loader_metadata_are_preserved(self):
        method = TpuFp8LinearMethod(_config())
        layer = torch.nn.Module()
        layer.tp_group = SimpleNamespace(rank_in_group=1, world_size=2)
        loader = lambda *args: None
        method.create_weights(
            layer, 80, [80, 48], 160, 256, torch.bfloat16, weight_loader=loader
        )
        # Both output shards overlap two global scale blocks at this TP rank;
        # the input shard also overlaps two blocks starting at column 80.
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (3, 2))
        self.assertEqual(layer.tpu_fp8_input_offset, 80)
        self.assertEqual(layer.tpu_fp8_output_tp_rank, 1)
        self.assertIs(layer.weight.weight_loader, loader)
        self.assertIs(layer.weight_scale_inv.weight_loader, loader)
        self.assertEqual(layer.logical_widths, [80, 48])

    def test_generic_scale_helper_keeps_ceil_division(self):
        layer = torch.nn.Module()
        Fp8LinearMethod.create_fp8_weight_(
            layer=layer,
            block_quant=True,
            quant_config=_config(),
            use_mxfp8=False,
            output_size_per_partition=128,
            input_size_per_partition=80,
            output_partition_sizes=[80, 48],
            input_size=160,
            output_size=256,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args: None,
            is_checkpoint_fp8_serialized=True,
            skip_block_quant_check=True,
        )
        self.assertEqual(tuple(layer.weight_scale_inv.shape), (1, 1))
        self.assertFalse(hasattr(layer, "tpu_fp8_input_offset"))

    def test_block_conversion_preserves_channel_values(self):
        scheme = TpuCompressedTensorsW8A8Fp8(
            QuantizationArgs(
                strategy=QuantizationStrategy.BLOCK, block_structure=[2, 2]
            ),
            False,
        )
        layer = torch.nn.Module()
        layer.tp_group = None
        scheme.create_weights(layer, 4, [4], 4, 4, torch.bfloat16, lambda *args: None)
        layer.weight.data.fill_(1.0)
        layer.weight_scale.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        runtime = SimpleNamespace(kernel=SimpleNamespace(tpu_fp8_runtime="channelwise"))
        with patch.object(dense_fp8, "get_exec", return_value=runtime):
            scheme.process_weights_after_loading(layer)
        restored = layer.weight.float().t() * layer.weight_scale.float()
        expected = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )
        torch.testing.assert_close(restored, expected, atol=0.15, rtol=0.05)
        self.assertIsNone(layer.weight_block_size)
        self.assertIsNone(layer.input_scale)


if __name__ == "__main__":
    unittest.main()
