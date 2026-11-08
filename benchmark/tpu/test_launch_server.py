"""CPU-only regression checks for launcher topology and runtime defaults."""

import unittest
import unittest.mock

from benchmark.tpu.launch_server import (
    LaunchConfig,
    load_recipe,
    plan_launch,
    resolve_server,
)


class TestRuntimeEnvironment(unittest.TestCase):
    def test_nightly_checks_final_extra_args_before_enabling_prefetch(self):
        for extra_args, enabled in (
            (["--base-gpu-id", "4"], False),
            (["--use-ray"], False),
            (["--weight-loader-disable-mmap"], False),
            (["--weight-loader-drop-cache-after-load"], True),
        ):
            with self.subTest(extra_args=extra_args):
                plan = plan_launch(
                    {
                        "server_args": {"model_path": "checkpoint", "tp_size": 8},
                        "extra_args": extra_args,
                    },
                    {"CI_WEIGHT_LOADER_PREFETCH": "1"},
                )
                self.assertEqual(
                    "--weight-loader-prefetch-checkpoints" in plan.command, enabled
                )
                if enabled:
                    self.assertNotIn(
                        "--weight-loader-drop-cache-after-load", plan.command
                    )
                    self.assertFalse(
                        plan.config.server_args["weight_loader_drop_cache_after_load"]
                    )
                else:
                    self.assertNotIn("LOCAL_SIZE", plan.env)

    def test_recipe_mmap_opt_out_keeps_the_launch_unchanged(self):
        plan = plan_launch(
            {
                "server_args": {
                    "model_path": "checkpoint",
                    "tp_size": 8,
                    "weight_loader_disable_mmap": True,
                }
            },
            {"CI_WEIGHT_LOADER_PREFETCH": "1"},
        )
        self.assertNotIn("--weight-loader-prefetch-checkpoints", plan.command)
        self.assertIn("--weight-loader-disable-mmap", plan.command)
        self.assertNotIn("LOCAL_SIZE", plan.env)

    def config(self, **runtime_envs):
        return LaunchConfig(
            {
                "model_path": "openai/gpt-oss-120b",
                "tp_size": 8,
                "enable_torch_compile": True,
            },
            runtime_envs,
        )

    def test_single_host_preserves_hardware_metadata_not_cross_host_rendezvous(self):
        inherited = {
            "TPU_TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "v7x",
            "TPU_WORKER_HOSTNAMES": "prefill,decode",
            "TPU_WORKER_ID": "1",
            "TPU_PROCESS_ADDRESSES": "prefill:8470,decode:8470",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": "prefill:8070,decode:8070",
            "TORCH_TPU_TOPOLOGY": "4,2,1,2",
            "MEGASCALE_COORDINATOR_ADDRESS": "prefill:8080",
            "TPU_VISIBLE_CHIPS": "0",
        }
        resolved, env = resolve_server(self.config(), inherited, single_host=True)
        self.assertEqual(resolved.server_args["nnodes"], 1)
        for name in ("TPU_TOPOLOGY", "TPU_ACCELERATOR_TYPE"):
            self.assertEqual(env[name], inherited[name])
        for name in inherited.keys() - {"TPU_TOPOLOGY", "TPU_ACCELERATOR_TYPE"}:
            self.assertNotIn(name, env)
        self.assertIn("TPU_WORKER_HOSTNAMES", inherited)

    def test_multihost_preserves_topology(self):
        config = self.config()
        config.server_args.update(nnodes=2, node_rank=1)
        inherited = {"TPU_TOPOLOGY": "4x2x1", "TPU_WORKER_HOSTNAMES": "host0,host1"}
        resolved, env = resolve_server(config, inherited)
        self.assertEqual(env["TPU_TOPOLOGY"], "4x2x1")
        self.assertEqual(env["TPU_WORKER_HOSTNAMES"], "host0,host1")
        self.assertEqual(resolved.server_args["dist_init_addr"], "host0:10011")

    def test_ci_prefetch_reaches_the_server_command_and_metadata(self):
        plan = plan_launch(
            {"server_args": self.config().server_args},
            inherited={"CI_WEIGHT_LOADER_PREFETCH": "1"},
        )
        self.assertIn("--weight-loader-prefetch-checkpoints", plan.command)
        self.assertEqual(plan.env["LOCAL_SIZE"], "8")
        self.assertTrue(plan.config.server_args["weight_loader_prefetch_checkpoints"])

    def test_cache_defaults_follow_resolved_model_and_home(self):
        _, env = resolve_server(
            self.config(), {"MODEL": "Qwen/Model-FP8", "HOME": "/tmp/launcher-home"}
        )
        self.assertEqual(
            env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"], "torchtpu_qwen_model_fp8"
        )
        self.assertEqual(
            env["TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"],
            "/tmp/launcher-home/.cache/torch_tpu_compile_cache",
        )
        self.assertEqual(
            env["TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK"], "1"
        )

    def test_cache_overrides_and_explicit_removal(self):
        tier2 = "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"
        tier3 = "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
        _, env = resolve_server(
            self.config(**{tier2: None, tier3: "/tmp/explicit-cache"}),
            {tier2: "inherited"},
        )
        self.assertNotIn(tier2, env)
        self.assertEqual(env[tier3], "/tmp/explicit-cache")
        _, env = resolve_server(self.config(), {tier2: "retained"})
        self.assertEqual(env[tier2], "retained")
        _, env = resolve_server(self.config(), {"ENABLE_COMPILE_CACHE": "0"})
        self.assertNotIn(tier2, env)
        self.assertNotIn(tier3, env)

    def test_cache_off_disables_torch_tpu_implicit_tier2(self):
        # With no tier-2 name torch_tpu falls back to an implicit "default"
        # tier-2 cache whenever WORLD_SIZE > 1; off must say "disabled".
        sentinel = "TORCH_TPU_TIER2_COMPILATION_CACHE"
        _, env = resolve_server(self.config(), {"ENABLE_COMPILE_CACHE": "0"})
        self.assertEqual(env[sentinel], "disabled")
        _, env = resolve_server(
            self.config(), {"ENABLE_COMPILE_CACHE": "0", sentinel: "explicit"}
        )
        self.assertEqual(env[sentinel], "explicit")
        _, env = resolve_server(
            self.config(**{sentinel: None}), {"ENABLE_COMPILE_CACHE": "0"}
        )
        self.assertNotIn(sentinel, env)
        _, env = resolve_server(self.config(), {})
        self.assertNotIn(sentinel, env)

    def test_migrated_recipes_include_shell_runtime_defaults(self):
        for recipe in (
            "gpt_oss_120b/singlehost_spec.py",
            "qwen3_coder_30b_a3b_fp8.py",
            "qwen3_5_397b/server.py",
        ):
            with self.subTest(recipe=recipe):
                recipe_config = load_recipe("recipe/configs/" + recipe)
                _, env = resolve_server(LaunchConfig.from_dict(recipe_config), {})
                for name, expected in {
                    "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
                    "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
                    "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
                    "TORCH_LOGS": "+recompiles",
                    "HF_HUB_OFFLINE": "0",
                    "TRANSFORMERS_OFFLINE": "0",
                }.items():
                    self.assertEqual(env[name], expected)
                self.assertIn("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", env)
                _, overridden = resolve_server(
                    LaunchConfig.from_dict(
                        recipe_config,
                        {"HF_HUB_OFFLINE": "1", "TORCH_LOGS": None},
                    ),
                    {},
                )
                self.assertEqual(overridden["HF_HUB_OFFLINE"], "1")
                self.assertNotIn("TORCH_LOGS", overridden)

    def test_concurrency_selects_get_config(self):
        recipe = "recipe/configs/muse_spark/agentic.py"
        config = load_recipe(recipe, 24)
        self.assertEqual(config["server_args"]["max_running_requests"], 6)
        self.assertEqual(config["runtime_env"]["SGLANG_TPU_MUSE_MK_MAX_ROWS"], "48")
        for name, concurrency in (
            (recipe, 2),
            (recipe, None),
            ("recipe/configs/muse_spark/server.py", 16),
        ):
            with self.subTest(name=name, concurrency=concurrency):
                with self.assertRaises(ValueError):
                    load_recipe(name, concurrency)

    def test_agentx_pd_modes(self):
        """A PP prefill must run without the megakernel (the model rejects it under PP)."""
        from benchmark.tpu.recipe.configs.muse_spark.agentic import get_config

        prefill, decode = (get_config(32, mode=m) for m in ("prefill", "decode"))
        self.assertEqual(prefill["server_args"]["pp_size"], 8)
        self.assertEqual(prefill["server_args"]["tp_size"], 1)
        self.assertEqual(decode["server_args"]["max_running_requests"], 32)
        for config in (prefill, decode):
            # A rank != 0 in a one-host group makes launch_server start nothing.
            self.assertEqual(config["runtime_env"]["SGLANG_LAUNCHER_RANK"], "0")
        self.assertEqual(prefill["runtime_env"]["SGLANG_TPU_MUSE_MEGAKERNEL"], "0")
        self.assertEqual(decode["runtime_env"]["SGLANG_TPU_MUSE_MEGAKERNEL"], "1")
        self.assertEqual(decode["runtime_env"]["SGLANG_TPU_MUSE_MK_MAX_ROWS"], "256")
        for config in (prefill, decode):
            self.assertEqual(config["server_args"]["kv_cache_dtype"], "fp8_e4m3")
            env = config["runtime_env"]
            self.assertEqual(env["SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER"], "1")
            self.assertNotIn("SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY", env)
        self.assertEqual(prefill["runtime_env"]["SGLANG_TPU_MUSE_PREFILL_SP"], "0")
        self.assertIsNone(prefill["runtime_env"]["SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS"])
        self.assertEqual(
            prefill["runtime_env"]["SGLANG_TPU_FORCE_RPA_PREFILL_BLOCKS"],
            "256:32:512:2",
        )
        self.assertEqual(prefill["server_args"]["max_total_tokens"], 1048576)
        self.assertEqual(prefill["server_args"]["mem_fraction_static"], 0.88)
        self.assertEqual(decode["server_args"]["mem_fraction_static"], 0.95)
        self.assertEqual(decode["runtime_env"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"], "1")
        # HiCache on a decode needs its radix tree; without it the decode refuses to start.
        self.assertTrue(
            decode["server_args"]["disaggregation_decode_enable_radix_cache"]
        )
        for config, mode in ((prefill, "prefill"), (decode, "decode")):
            self.assertEqual(config["server_args"]["disaggregation_mode"], mode)
        self.assertNotIn("disaggregation_mode", get_config(24)["server_args"])
        for concurrency, mode in ((24, "router"), (24, "decode")):
            with self.assertRaises(ValueError):
                get_config(concurrency, mode=mode)

    def test_agentx_pd_ships_sal_draft_kv(self):
        """Both P/D ends run the SEQ_ALONG_LANE draft whose KV Raiden ships."""
        from benchmark.tpu.recipe.configs.muse_spark.agentic import get_config, main

        for mode in ("prefill", "decode"):
            config = get_config(32, mode=mode)
            self.assertEqual(
                config["server_args"]["speculative_draft_attention_backend"],
                "pallas_tpu_rpa_batched",
            )
            self.assertEqual(
                config["runtime_env"]["SGLANG_TPU_PD_DECODE_DRAFT_PREFILL"], "0"
            )
            # The draft pool follows the draft backend: SEQ_ALONG_LANE needs this flag.
            self.assertEqual(
                config["runtime_env"]["SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"], "1"
            )
        with unittest.mock.patch("builtins.print") as printed:
            self.assertEqual(main(["decode", "32", "--dry-run"]), 0)
        self.assertIn(
            '"SGLANG_TPU_PD_DECODE_DRAFT_PREFILL": "0"', printed.call_args[0][0]
        )


if __name__ == "__main__":
    unittest.main()
