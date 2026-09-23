"""Unit tests for Python recipe dictionaries and CLI flag conversion."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.recipe.configs import accuracy, config_eval
from benchmark.tpu.recipe.configs.base import CONFIG as BASE_CONFIG
from benchmark.tpu.recipe.configs.qwen3_0_6b import CONFIG


class TestTpuConfigEval(unittest.TestCase):
    def test_example_config(self):
        self.assertEqual(CONFIG["server_args"]["context_length"], 128)
        self.assertEqual(CONFIG["server_args"]["max_running_requests"], 1)
        self.assertEqual(CONFIG["runtime_env"]["TPU_VISIBLE_CHIPS"], "0")
        self.assertNotIn("model_path", BASE_CONFIG["server_args"])
        self.assertEqual(BASE_CONFIG["runtime_env"], {"SGLANG_PLATFORM": "tpu"})

    def test_config_overrides_do_not_modify_shared_defaults(self):
        original = deepcopy(CONFIG)
        original_base = deepcopy(BASE_CONFIG)
        config = deepcopy(CONFIG)
        config["server_args"].update(port=30001, context_length=256)
        config["server_args"]["tpu_compile_bs"].append(4)
        config["runtime_env"]["TPU_VISIBLE_CHIPS"] = "1"
        args = config_eval.server_args_to_cli(config["server_args"])
        self.assertEqual(args[args.index("--port") + 1], "30001")
        self.assertEqual(args[args.index("--context-length") + 1], "256")
        start = args.index("--tpu-compile-bs")
        self.assertEqual(args[start : start + 3], ["--tpu-compile-bs", "1", "4"])
        self.assertEqual(CONFIG, original)
        self.assertEqual(BASE_CONFIG, original_base)

    def test_cli_value_types(self):
        settings = {
            "model_path": "a b; $(not-a-command)",
            "port": 30000,
            "enable_torch_compile": True,
            "disable_radix_cache": False,
            "quantization": None,
            "cuda_graph_bs": [1, 4, 8],
            "json_model_override_args": {"rope_scaling": {"factor": 2}},
        }
        original = deepcopy(settings)
        args = config_eval.server_args_to_cli(settings)
        self.assertEqual(
            args[:-2],
            [
                "--model-path",
                "a b; $(not-a-command)",
                "--port",
                "30000",
                "--enable-torch-compile",
                "--cuda-graph-bs",
                "1",
                "4",
                "8",
            ],
        )
        flag, value = args[-2:]
        self.assertEqual(flag, "--json-model-override-args")
        self.assertEqual(json.loads(value), settings["json_model_override_args"])
        self.assertEqual(settings, original)

    def test_flags_are_visible_to_existing_ci_argument_checks(self):
        args = config_eval.server_args_to_cli(
            {"quantization": "modelopt_fp8", "lora_paths": {"adapter": "/tmp/adapter"}}
        )
        self.assertEqual(args[:2], ["--quantization", "modelopt_fp8"])
        self.assertEqual(args[2], "--lora-paths")
        self.assertEqual(json.loads(args[3]), {"adapter": "/tmp/adapter"})


class TestAccuracyRecipe(unittest.TestCase):
    def test_model_override_preserves_ci_host_discovery(self):
        inherited = {
            "MODEL": "/local weights/R1",
            "TPU_WORKER_ID": "1",
            "TPU_WORKER_HOSTNAMES": "rank0,rank1",
            "DIST_INIT_ADDR": "rank0:10011",
            "MAX_RUNNING": "1",
        }
        original = inherited.copy()
        env = accuracy.recipe_env("deepseek_r1_0528_fp8kv_mh", inherited)
        for key in ("MODEL", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "DIST_INIT_ADDR"):
            self.assertEqual(env[key], inherited[key])
        self.assertEqual(env["MAX_RUNNING"], "128")
        self.assertEqual(env["DP_SIZE"], "1")
        self.assertEqual(env["SGLANG_TPU_FP8_MOE_FP32_SCALES"], "1")
        self.assertEqual(inherited, original)

    def test_empty_model_uses_recipe_default(self):
        self.assertEqual(
            accuracy.recipe_env("gpt_oss_20b_accuracy", {"MODEL": ""})["MODEL"],
            "openai/gpt-oss-20b",
        )

    def test_recipe_results_do_not_mutate_shared_defaults(self):
        env = accuracy.recipe_env("deepseek_v3_1_fp8kv_mh", {})
        env["MODEL"] = "modified"
        for recipe, expected in (
            ("deepseek_v3_1_fp8kv_mh", "/weights/DeepSeek-V3.1"),
            ("deepseek_r1_0528_fp8kv_mh", "/weights/DeepSeek-R1-0528"),
        ):
            self.assertEqual(accuracy.recipe_env(recipe, {})["MODEL"], expected)

    def test_rejects_recipe_paths(self):
        for name in ("../base", "base.py", "a/b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                accuracy.recipe_env(name, {})

    def test_executes_existing_launcher(self):
        with (
            patch.object(sys, "argv", ["accuracy", "--config", "kimi_k26_mmmu"]),
            patch.object(accuracy.os, "execvpe") as execute,
        ):
            accuracy.main()
        executable, args, env = execute.call_args.args
        self.assertEqual(executable, "bash")
        self.assertEqual(args, ["bash", str(ROOT / "benchmark/tpu/launch_server.sh")])
        self.assertEqual(env["QUANTIZATION"], "")
        self.assertEqual(
            (env["TP_SIZE"], env["EP_SIZE"], env["DP_SIZE"]), ("16", "16", "1")
        )


if __name__ == "__main__":
    unittest.main()
