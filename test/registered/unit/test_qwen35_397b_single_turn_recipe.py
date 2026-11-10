"""Unit tests for the Qwen3.5-397B-A17B-FP8 single-turn 8k/1k TPU recipe."""

import argparse
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.launch_server import load_recipe, plan_launch  # noqa: E402
from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG  # noqa: E402
from benchmark.tpu.recipe.configs.qwen3_5_397b import single_turn_8k1k  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402


class TestQwen35SingleTurn8k1kRecipe(unittest.TestCase):
    RECIPE = "recipe/configs/qwen3_5_397b/single_turn_8k1k.py"

    def test_low_and_high_concurrency_hybrid_parallelism(self):
        base_before = deepcopy(_FP8_BASE_CONFIG)

        c4 = load_recipe(self.RECIPE, concurrency=4)
        self.assertEqual(c4["server_args"]["tp_size"], 8)
        self.assertEqual(c4["server_args"]["ep_size"], 8)
        self.assertNotIn("attn_dp_size", c4["server_args"])
        self.assertEqual(c4["server_args"]["max_running_requests"], 8)
        self.assertEqual(c4["server_args"]["max_mamba_cache_size"], 16)
        self.assertEqual(c4["server_args"]["prefill_max_requests"], 3)
        self.assertEqual(c4["server_args"]["scheduler_recv_interval"], 2)
        self.assertEqual(
            c4["server_args"]["tpu_compile_tokens"], [4096, 8192, 10240, 16384, 20480]
        )
        self.assertEqual(c4["runtime_env"]["SGLANG_TPU_ENABLE_TWOSHOT_ALLREDUCE"], "1")
        self.assertNotIn("SGLANG_TPU_ENABLE_RDMA_ALL_GATHER", c4["runtime_env"])
        self.assertNotIn("SGLANG_TPU_PREFILL_DRAIN_INTERVAL", c4["runtime_env"])

        c16 = load_recipe(self.RECIPE, concurrency=16)
        self.assertEqual(c16["server_args"]["max_running_requests"], 24)
        self.assertEqual(c16["server_args"]["tpu_compile_bs"], [1, 2, 4, 8, 16, 24])
        self.assertTrue(c16["server_args"]["language_model_only"])

        c64 = load_recipe(self.RECIPE, concurrency=64)
        self.assertEqual(c64["server_args"]["attn_dp_size"], 8)
        self.assertTrue(c64["server_args"]["enable_dp_lm_head"])
        self.assertEqual(c64["server_args"]["max_running_requests"], 256)
        self.assertEqual(c64["server_args"]["torch_compile_max_bs"], 32)
        self.assertEqual(c64["server_args"]["max_mamba_cache_size"], 320)
        self.assertEqual(c64["server_args"]["max_total_tokens"], 786432)
        self.assertEqual(c64["server_args"]["tpu_compile_tokens"], [2048, 4096])
        self.assertEqual(c64["runtime_env"]["SGLANG_TPU_PREFILL_DRAIN_INTERVAL"], "256")
        self.assertEqual(c64["runtime_env"]["SGLANG_TPU_ENABLE_RDMA_ALL_GATHER"], "1")
        self.assertNotIn("SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS", c64["runtime_env"])
        self.assertNotIn("SGLANG_TPU_ENABLE_TWOSHOT_ALLREDUCE", c64["runtime_env"])

        c128 = load_recipe(self.RECIPE, concurrency=128)
        self.assertEqual(c128["server_args"]["max_running_requests"], 384)
        self.assertEqual(c128["server_args"]["torch_compile_max_bs"], 48)
        self.assertEqual(c128["server_args"]["max_mamba_cache_size"], 448)
        self.assertEqual(c128["server_args"]["max_total_tokens"], 851968)
        self.assertEqual(
            c128["runtime_env"]["SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS"], "384"
        )

        c256 = load_recipe(self.RECIPE, concurrency=256)
        self.assertEqual(c256["server_args"]["max_running_requests"], 512)
        self.assertEqual(c256["server_args"]["torch_compile_max_bs"], 64)
        self.assertEqual(c256["server_args"]["max_mamba_cache_size"], 576)
        self.assertEqual(c256["server_args"]["max_total_tokens"], 983040)
        self.assertEqual(
            c256["runtime_env"]["SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS"], "512"
        )

        self.assertEqual(_FP8_BASE_CONFIG, base_before)

    def test_dp_attention_override_and_model_path_resolution(self):
        forced_dp = single_turn_8k1k.get_config(concurrency=16, dp_attention=True)
        self.assertEqual(forced_dp["server_args"]["attn_dp_size"], 8)

        forced_tp = single_turn_8k1k.get_config(concurrency=256, dp_attention=False)
        self.assertNotIn("attn_dp_size", forced_tp["server_args"])
        self.assertEqual(forced_tp["server_args"]["max_running_requests"], 256)
        self.assertEqual(forced_tp["server_args"]["max_total_tokens"], 2490368)

        with (
            patch.dict(
                os.environ,
                {"QWEN35_397B_WEIGHTS_PATH": "/custom/qwen35-fp8"},
                clear=False,
            ),
            patch("os.path.exists", side_effect=lambda p: p == "/custom/qwen35-fp8"),
        ):
            self.assertEqual(
                single_turn_8k1k.resolve_model_path(), "/custom/qwen35-fp8"
            )

        with patch("os.path.exists", return_value=False):
            self.assertEqual(
                single_turn_8k1k.resolve_model_path(),
                "/weights/Qwen3.5-397B-A17B-FP8",
            )

    def test_plan_launch_parses_cleanly_as_server_args(self):
        for conc in (16, 256):
            with self.subTest(concurrency=conc):
                plan = plan_launch(
                    load_recipe(self.RECIPE, concurrency=conc), inherited={}
                )
                argv = plan.command[plan.command.index("sglang.launch_server") + 1 :]
                parser = argparse.ArgumentParser()
                ServerArgs.add_cli_args(parser)
                parsed = parser.parse_args(argv)
                self.assertTrue(parsed.language_model_only)
                self.assertEqual(parsed.page_size, 256)


if __name__ == "__main__":
    unittest.main()
