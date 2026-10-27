"""Verify Qwen3-VL-30B-A3B single-host and P/D recipes produce SPMD-safe warmup graphs."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.testing.spmd_safe import run_recipes, verify_startup

register_tpu_ci(
    est_time=600,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=[
        "singlehost",
        "qwen3-vl-30b-a3b",
        "pd",
        "smoke",
        "multimodal",
        "compile-tracer",
    ],
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SINGLEHOST_RECIPE = REPO / "benchmark/tpu/recipe/qwen3_vl_30b_a3b/server.sh"
PD_CONFIG_MODULE = "recipe/configs/qwen3_vl_30b_a3b/pd_singlehost.py"
WORLD_SIZE = 8
VERIFY_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "600"))


def _resolve_model_path() -> str:
    return (
        os.environ.get("QWEN3_VL_30B_MODEL")
        or os.environ.get("MODEL")
        or "/weights/Qwen3-VL-30B-A3B-Instruct"
    )


def _recipe_command_and_env(
    recipe: str, model_path: str
) -> tuple[list[str], dict[str, str]]:
    base_env = {
        "MODEL": model_path,
        "QWEN3_VL_30B_MODEL": model_path,
        "ENABLE_COMPILE_CACHE": "0",
        "ENABLE_PROFILER_ENDPOINT": "0",
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
    }
    if recipe == "singlehost":
        return (
            ["bash", str(SINGLEHOST_RECIPE)],
            {**base_env, "HOST": "127.0.0.1", "PORT": "30000"},
        )
    role_config = {
        "pd_prefill": f"{PD_CONFIG_MODULE}:PREFILL_CONFIG",
        "pd_decode": f"{PD_CONFIG_MODULE}:DECODE_CONFIG",
    }[recipe]
    return (
        [
            sys.executable,
            "-m",
            "benchmark.tpu.launch_server",
            "--config",
            role_config,
            "--runtime-envs",
            json.dumps({"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}),
        ],
        base_env,
    )


class TestQwen3Vl30bA3bSpmdSafe(CustomTestCase):
    def test_singlehost_and_pd_recipes_are_spmd_safe(self) -> None:
        model_path = _resolve_model_path()
        recipes = ("singlehost", "pd_prefill", "pd_decode")
        with tempfile.TemporaryDirectory(prefix="qwen3-vl-30b-spmd-") as tmp:
            output = Path(tmp)

            def launch_recipe(recipe: str, directory: Path) -> None:
                command, env = _recipe_command_and_env(recipe, model_path)
                verify_startup(
                    command,
                    world_size=WORLD_SIZE,
                    local_world_size=WORLD_SIZE,
                    timeout=VERIFY_TIMEOUT,
                    dump_dir=directory,
                    env=env,
                )

            status = run_recipes(recipes, launch_recipe, output=output)
            self.assertEqual(status, 0)

            for index, recipe in enumerate(recipes):
                result_files = list(
                    (output / f"{index}-{recipe}").glob("spmd-*/result.json")
                )
                self.assertEqual(
                    len(result_files), 1, f"missing result.json for {recipe}"
                )
                result = json.loads(result_files[0].read_text())
                self.assertTrue(
                    result["passed"], f"{recipe} failed: {result['errors']}"
                )
                self.assertEqual(result["ranks"], list(range(WORLD_SIZE)))
                keys = result["keys"]
                self.assertTrue(
                    any(k.startswith("target/visual.forward:") for k in keys),
                    f"{recipe} did not verify Qwen3VLVisionTpuCompilePolicy: {keys}",
                )
                self.assertTrue(
                    any(k.startswith("target/model.forward:") for k in keys),
                    f"{recipe} did not verify Qwen3VLDecoderTpuCompilePolicy: {keys}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
