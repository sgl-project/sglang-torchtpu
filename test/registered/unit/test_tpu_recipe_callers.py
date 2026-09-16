"""CPU checks for recipe callers; server processes and health checks are mocked."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tpu import eval_support, model_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.tpu.recipe.bench_matrix.bench_leg import (
    selected_recipe,
    validate_workload,
    workload_settings,
)
from benchmark.tpu.recipe.bench_matrix.serve_group import server_command
from benchmark.tpu.recipe.configs.kimi_k26_mh import CONFIG as KIMI_CONFIG
from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.config import BENCHMARK

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRecipeCallers(unittest.TestCase):
    def test_python_config_uses_existing_popen_and_reuses_server(self):
        config = {
            "server_args": {
                "model_path": "model with spaces",
                "device": "tpu",
                "host": "0.0.0.0",
                "port": 30000,
                "tpu_compile_bs": [1, 4],
                "enable_torch_compile": True,
            },
            "runtime_env": {"SGLANG_PLATFORM": "tpu"},
        }
        original = deepcopy(config)
        process = SimpleNamespace(pid=123, poll=lambda: None)
        launcher = model_config._build_launcher(config)
        with (
            patch.object(
                eval_support, "popen_launch_server", return_value=process
            ) as popen,
            patch.object(eval_support, "dump_launch"),
            patch.object(eval_support, "wait_for_health"),
            patch.object(eval_support, "kill_process_tree") as kill,
        ):
            try:
                for _ in range(2):
                    with launcher("http://127.0.0.1:31234") as actual:
                        self.assertIs(actual, process)
                popen.assert_called_once_with(
                    "model with spaces",
                    "http://127.0.0.1:31234",
                    timeout=1800.0,
                    other_args=[
                        "--device",
                        "tpu",
                        "--tpu-compile-bs",
                        "1",
                        "4",
                        "--enable-torch-compile",
                    ],
                    env={"SGLANG_PLATFORM": "tpu"},
                    device="tpu",
                )
                self.assertEqual(config, original)
            finally:
                launcher.shutdown()
            kill.assert_called_once_with(123, wait_timeout=60)

    def test_legacy_shell_launch_still_works(self):
        process = SimpleNamespace(pid=124, poll=lambda: None)
        launcher = eval_support.RecipeLauncher(__file__)
        with (
            patch.object(
                eval_support.subprocess, "Popen", return_value=process
            ) as popen,
            patch.object(eval_support, "wait_for_health"),
            patch.object(eval_support, "kill_process_tree"),
        ):
            try:
                with launcher("http://127.0.0.1:31235"):
                    pass
                self.assertEqual(
                    popen.call_args.args[0], ["bash", str(Path(__file__).absolute())]
                )
            finally:
                launcher.shutdown()

    def test_multihost_rank_is_assigned_by_caller(self):
        original = deepcopy(KIMI_CONFIG)
        command, env = server_command(
            KIMI_CONFIG,
            {
                "TPU_WORKER_ID": "1",
                "TPU_WORKER_HOSTNAMES": "leader,worker",
                "TPU_VISIBLE_CHIPS": "0",
            },
        )
        self.assertEqual(command[command.index("--node-rank") + 1], "1")
        self.assertEqual(command[command.index("--dist-init-addr") + 1], "leader:10011")
        self.assertNotIn("TPU_VISIBLE_CHIPS", env)
        self.assertEqual(KIMI_CONFIG, original)

    def test_single_host_group_strips_slice_topology(self):
        command, env = server_command(
            KIMI_CONFIG,
            {
                "TPU_WORKER_HOSTNAMES": "a,b,c,d",
                "TPU_WORKER_ID": "2",
                "MEGASCALE_COORDINATOR_ADDRESS": "a",
            },
            single_host=True,
        )
        self.assertEqual(command[command.index("--nnodes") + 1], "1")
        self.assertNotIn("TPU_WORKER_HOSTNAMES", env)
        self.assertNotIn("MEGASCALE_COORDINATOR_ADDRESS", env)
        self.assertNotIn("--node-rank", command)

    def test_long_workload_must_fit_model_context(self):
        module = selected_recipe(
            {"BENCH_MODEL": "gpt_oss_120b", "BENCH_RECIPE": "1host_pc"}
        )
        settings = workload_settings(module.BENCHMARK, {"PROFILE": "ci"})
        validate_workload(settings, 131072)
        with self.assertRaises(ValueError):
            validate_workload(settings, 32768)

    def test_workload_profiles_preserve_grid_and_deadlines(self):
        for model in ("gpt_oss_120b", "qwen3_coder_480b"):
            module = selected_recipe({"BENCH_MODEL": model, "BENCH_RECIPE": "1host_pc"})
            for workload, longs in (("agentic", [1, 2, 4]), ("8k1k", [0])):
                settings = workload_settings(
                    module.BENCHMARK, {"WORKLOAD": workload, "PROFILE": "ci"}
                )
                self.assertEqual(settings["sweep_bs"], [1, 16, 32])
                self.assertEqual(settings["sweep_long"], longs)
                self.assertEqual(settings["max_leg_seconds"], 900)
        nightly = workload_settings(BENCHMARK, {"PROFILE": "ci"}, nightly=True)
        self.assertEqual(nightly["sweep_bs"], [1, 16])
        self.assertEqual(nightly["target_context"], 30720)
        self.assertEqual(nightly["long_target_context"], 65536)
        self.assertEqual(BENCHMARK["profile"], "full")


if __name__ == "__main__":
    unittest.main()
