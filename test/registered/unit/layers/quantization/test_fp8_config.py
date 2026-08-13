# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import (
    Fp8Config,
    _should_prepare_tpu_fp8_fused_rs,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFp8Config(unittest.TestCase):
    def test_block_fp8_fused_rs_requires_explicit_backend(self):
        auto = MoeRunnerBackend("auto")
        explicit = MoeRunnerBackend("pallas_tpu_fused_rs")

        self.assertFalse(_should_prepare_tpu_fp8_fused_rs(auto, block_quant=True))
        self.assertTrue(_should_prepare_tpu_fp8_fused_rs(auto, block_quant=False))
        self.assertTrue(_should_prepare_tpu_fp8_fused_rs(explicit, block_quant=True))

    def test_exclude_modules_uses_standard_ignored_layer_matching(self):
        config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [1, 512],
                "exclude_modules": ["layers.0.", "self_attn", "router"],
            }
        )

        for prefix in (
            "model.layers.0.mlp.experts",
            "model.layers.4.self_attn.qkv_proj",
            "model.layers.4.mlp.router",
        ):
            with self.subTest(prefix=prefix):
                self.assertTrue(is_layer_skipped(prefix, config.ignored_layers))

        self.assertFalse(
            is_layer_skipped("model.layers.4.mlp.experts", config.ignored_layers)
        )


if __name__ == "__main__":
    unittest.main()
