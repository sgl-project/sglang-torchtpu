"""Kimi-K3.1 image-input server recipe (CPU only, no server)."""

import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.recipe.configs.kimi_k3_1.server import (  # noqa: E402
    CONFIG as TEXT_CONFIG,
)
from benchmark.tpu.recipe.configs.kimi_k3_1.server_vl import CONFIG  # noqa: E402


class TestKimiK31ServerVl(unittest.TestCase):
    def test_sizing_is_consistent(self):
        args = CONFIG["server_args"]
        self.assertEqual(args["max_running_requests"], 32)
        self.assertEqual(args["torch_compile_max_bs"], args["max_running_requests"])
        self.assertEqual(args["tpu_compile_bs"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(args["tpu_compile_tokens"], [args["chunked_prefill_size"]])
        self.assertEqual(args["chunked_prefill_size"], args["max_prefill_tokens"])
        self.assertTrue(args["skip_server_warmup"])
        self.assertNotIn("speculative_algorithm", args)

    def test_only_the_vl_settings_differ_from_the_text_recipe(self):
        vl_args = {
            "max_running_requests",
            "torch_compile_max_bs",
            "tpu_compile_bs",
        }
        args, text_args = CONFIG["server_args"], TEXT_CONFIG["server_args"]
        self.assertEqual(set(args), set(text_args))
        for name, value in text_args.items():
            if name not in vl_args:
                self.assertEqual(args[name], value, name)

        env, text_env = CONFIG["runtime_env"], TEXT_CONFIG["runtime_env"]
        self.assertEqual(env["SGLANG_TPU_VL_TEXT_ONLY_EXTEND"], "0")
        self.assertEqual(env["SGLANG_TPU_ENABLE_VISION_SPLASH_ATTENTION"], "1")
        for name, value in text_env.items():
            if name != "SGLANG_TPU_VL_TEXT_ONLY_EXTEND":
                self.assertEqual(env[name], value, name)
        self.assertEqual(
            set(env) - set(text_env), {"SGLANG_TPU_ENABLE_VISION_SPLASH_ATTENTION"}
        )

    def test_vit_ladder_is_the_splash_default(self):
        # Unset, the splash default reaches 66564 patches (Kimi-K3's processor
        # admits 65536); the flash-only ladder would leave larger images eager.
        env = CONFIG["runtime_env"]
        self.assertNotIn("SGLANG_TPU_VIT_PRECOMPILE_SIZES", env)
        self.assertNotIn("SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES", env)

    def test_text_recipe_is_unchanged(self):
        args = TEXT_CONFIG["server_args"]
        self.assertEqual(args["max_running_requests"], 64)
        self.assertEqual(
            TEXT_CONFIG["runtime_env"]["SGLANG_TPU_VL_TEXT_ONLY_EXTEND"], "1"
        )


if __name__ == "__main__":
    unittest.main()
