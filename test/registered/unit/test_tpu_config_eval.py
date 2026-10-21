"""Unit tests for Python recipe dictionaries and CLI flag conversion."""

import importlib
import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.launch_server import load_recipe, plan_launch
from benchmark.tpu.recipe.configs import base, config_eval
from benchmark.tpu.recipe.configs.base import CONFIG as BASE_CONFIG
from benchmark.tpu.recipe.configs.qwen3_0_6b import CONFIG


class TestTpuConfigEval(unittest.TestCase):
    def test_example_config(self):
        self.assertEqual(CONFIG["server_args"]["context_length"], 128)
        self.assertEqual(CONFIG["server_args"]["max_running_requests"], 1)
        self.assertEqual(CONFIG["runtime_env"]["TPU_VISIBLE_CHIPS"], "0")
        self.assertNotIn("model_path", BASE_CONFIG["server_args"])
        self.assertEqual(list(BASE_CONFIG["runtime_env"]), ["SGLANG_PLATFORM"])

    def test_inherited_platform_reaches_recipe_servers(self):
        """A mode-B run (SGLANG_PLATFORM=tpu_oot) must launch mode-B servers;
        the recipe pinned "tpu", which silently ran them in mode A."""
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_PLATFORM"}
        try:
            for inherited, expected in ((None, "tpu"), ("tpu_oot", "tpu_oot")):
                case = (
                    env if inherited is None else {**env, "SGLANG_PLATFORM": inherited}
                )
                with (
                    self.subTest(inherited=inherited),
                    patch.dict(os.environ, case, clear=True),
                ):
                    runtime_env = importlib.reload(base).CONFIG["runtime_env"]
                    self.assertEqual(runtime_env["SGLANG_PLATFORM"], expected)
        finally:
            importlib.reload(base)

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


class TestAccuracyRecipes(unittest.TestCase):
    MULTIHOST = (
        "recipe/configs/deepseek_r1_0528/accuracy.py",
        "recipe/configs/deepseek_v3_1/accuracy.py",
        "recipe/configs/kimi_k26/mmmu.py",
    )
    SINGLE_HOST = (
        "recipe/configs/qwen3_5_397b/server.py",
        "recipe/configs/gpt_oss_20b/accuracy.py",
        "recipe/configs/gemma4_31b/accuracy.py",
        "recipe/configs/gemma4_26b_a4b/accuracy.py",
        "recipe/configs/gpt_oss_120b/accuracy_aime.py",
        "recipe/configs/gpt_oss_120b/accuracy_gpqa.py",
        "recipe/configs/gpt_oss_120b/accuracy_gsm8k.py",
        "recipe/configs/gpt_oss_120b/accuracy_mmlupro.py",
    )

    def test_multihost_rank_resolves_from_jobset_topology(self):
        topology = {"TPU_WORKER_ID": "1", "TPU_WORKER_HOSTNAMES": "rank0,rank1"}
        for recipe in self.MULTIHOST:
            with self.subTest(recipe=recipe):
                plan = plan_launch(load_recipe(recipe), inherited=topology)
                args = plan.config.server_args
                self.assertEqual((args["nnodes"], args["node_rank"]), (2, 1))
                self.assertEqual(args["dist_init_addr"], "rank0:10011")
                self.assertEqual((args["tp_size"], args["ep_size"]), (16, 16))
                # DP attention is all-or-nothing: --dp-size without
                # --enable-dp-attention silently serves one attention group.
                dp_size = args.get("dp_size", 1)
                if dp_size > 1:
                    self.assertTrue(args.get("enable_dp_attention"), recipe)
                    self.assertEqual(args["tp_size"] % dp_size, 0, recipe)
                    # attention TP > 1 needs the LM head split the same way.
                    self.assertTrue(args.get("enable_dp_lm_head"), recipe)
                    self.assertIn("--dp-size", plan.command)
                else:
                    self.assertNotIn("--dp-size", plan.command)
                    self.assertNotIn("--enable-dp-attention", plan.command)

    def test_derived_recipe_does_not_mutate_its_base(self):
        from benchmark.tpu.recipe.configs.deepseek_r1_0528.accuracy import CONFIG as R1
        from benchmark.tpu.recipe.configs.deepseek_v3_1.accuracy import CONFIG as V31

        self.assertEqual(R1["server_args"]["model_path"], "/weights/DeepSeek-R1-0528")
        self.assertEqual(R1["server_args"]["reasoning_parser"], "deepseek-r1")
        self.assertEqual(V31["server_args"]["reasoning_parser"], "deepseek-v3")
        self.assertEqual(
            {
                k: v
                for k, v in V31["server_args"].items()
                if k not in ("model_path", "reasoning_parser")
            },
            {
                k: v
                for k, v in R1["server_args"].items()
                if k not in ("model_path", "reasoning_parser")
            },
        )

    def test_recipes_include_shell_runtime_defaults(self):
        for recipe in self.MULTIHOST + self.SINGLE_HOST:
            with self.subTest(recipe=recipe):
                plan = plan_launch(load_recipe(recipe), inherited={})
                self.assertEqual(plan.env["TORCH_LOGS"], "+recompiles")
                self.assertEqual(plan.env["SGLANG_TPU_PRELOAD_TELEMETRY_STUB"], "1")
                self.assertIn("--enable-torch-compile", plan.command)

    def test_every_recipe_still_parses_as_server_args(self):
        """Every flag a recipe emits must exist on ServerArgs at this revision.

        An upstream rename leaves the recipe emitting a flag the server no
        longer takes, and the job dies in bring-up on the slice: the sync that
        split --cuda-graph-bs into -decode/-prefill made the bare flag an
        ambiguous prefix, which argparse rejects.
        """
        import argparse

        from sglang.srt.server_args import ServerArgs

        topology = {"TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "h0,h1"}
        for recipe in self.MULTIHOST + self.SINGLE_HOST:
            with self.subTest(recipe=recipe):
                plan = plan_launch(load_recipe(recipe), inherited=topology)
                argv = plan.command[plan.command.index("sglang.launch_server") + 1 :]
                parser = argparse.ArgumentParser()
                ServerArgs.add_cli_args(parser)
                try:
                    parser.parse_args(argv)
                except SystemExit as exit_code:
                    self.fail(f"{recipe}: server rejects its own flags ({exit_code})")

    def test_every_multihost_accuracy_test_launches_one_of_these_recipes(self):
        """The JobSet runs MULTIHOST_LAUNCH["launch_script"] on every host.

        A .py value is handed to the Python launcher by jobset-tpu.yaml; a typo
        would only surface as a JobSet that never serves.
        """
        import ast

        declared = set()
        for path in sorted(
            (ROOT / "test/registered/tpu/nightly-accuracy").glob("test_*.py")
        ):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "MULTIHOST_LAUNCH" for t in node.targets
                ):
                    script = ast.literal_eval(node.value)["launch_script"]
                    self.assertTrue(
                        (ROOT / "benchmark/tpu" / script).is_file(), f"{path}: {script}"
                    )
                    declared.add(script)
        self.assertEqual(declared, set(self.MULTIHOST))


class TestWorkloadProfiles(unittest.TestCase):
    """A profile sizes a workload; it must not restate how a model is served."""

    def test_profiles_set_only_sizing_keys(self):
        """A stray key here would silently re-serve the model.

        apply_profile() feeds the cookbook generator, so that would publish a
        different parallelism or backend than the gates validated.
        """
        from benchmark.tpu.recipe.configs.profiles import PROFILE_KEYS, PROFILES

        for name, profile in PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(set(profile["server_args"]) - set(PROFILE_KEYS), set())

    def test_apply_profile_does_not_modify_its_inputs(self):
        from benchmark.tpu.recipe.configs.profiles import PROFILES, apply_profile

        config = deepcopy(CONFIG)
        original, profile = deepcopy(config), PROFILES["chat_1k1k"]
        original_profile = deepcopy(profile)
        sized = apply_profile(config, profile)
        self.assertEqual(config, original)
        self.assertEqual(profile, original_profile)
        self.assertEqual(sized["server_args"]["max_running_requests"], 64)
        # The chunk follows the profile's prefill bucket, not the base's.
        self.assertEqual(
            sized["server_args"]["chunked_prefill_size"], profile["prefill_tokens"]
        )


class TestRenderedCookbookCommands(unittest.TestCase):
    """Every generated cookbook command must be one the server accepts.

    The page tells a reader to paste it on a clean host, so a flag the server
    has since renamed is a bring-up failure nobody sees in CI.
    """

    PAGES = (
        "docs/cookbook/autoregressive/OpenAI/GPT-OSS.mdx",
        "docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx",
    )

    def test_generated_blocks_parse_as_server_args(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        sys.path.insert(0, str(ROOT / "scripts/ci/tpu"))
        import render_recipe

        for page in self.PAGES:
            text = (ROOT / page).read_text()
            markers = list(render_recipe.MARKER.finditer(text))
            self.assertTrue(markers, page)
            for match in markers:
                attrs = render_recipe.parse_attrs(match.group("attrs"))
                with self.subTest(page=page, **attrs):
                    block = render_recipe.render(attrs["config"], attrs["profile"])
                    argv = []
                    for line in block.splitlines():
                        line = line.strip().removesuffix(" \\")
                        if line.startswith("--"):
                            argv.extend(line.split())
                    parser = argparse.ArgumentParser()
                    ServerArgs.add_cli_args(parser)
                    try:
                        parser.parse_args(argv)
                    except SystemExit as exit_code:
                        self.fail(
                            f"{page}: server rejects its own command ({exit_code})"
                        )


if __name__ == "__main__":
    unittest.main()
