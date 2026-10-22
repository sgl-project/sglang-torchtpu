# SPDX-License-Identifier: Apache-2.0
"""In-tree defaults of the MoE method hooks a platform plugin may override."""

import unittest

from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.plugins.oot_classes import clear_oot_replacements, oot_base_class
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMoeMethodSeams(CustomTestCase):
    def setUp(self):
        clear_oot_replacements()
        self.addCleanup(clear_oot_replacements)

    def test_wna16_scheme_selection_keeps_the_in_tree_scheme(self):
        config = CompressedTensorsConfig({}, [], "pack-quantized", {}, [])
        weight_quant = QuantizationArgs(
            num_bits=4, strategy=QuantizationStrategy.GROUP, group_size=128
        )
        scheme = config._get_wna16_moe_scheme(weight_quant, None)
        self.assertIs(type(scheme), CompressedTensorsWNA16MoE)
        # The loader's transposed-weight check names the in-tree base class.
        self.assertEqual(
            oot_base_class(type(scheme)).__name__, "CompressedTensorsWNA16MoE"
        )

    def test_mxfp4_intermediate_padding_defaults_to_the_shard_size(self):
        method = object.__new__(Mxfp4MoEMethod)
        self.assertEqual(method._get_intermediate_size_after_pad(720), 720)


if __name__ == "__main__":
    unittest.main()
