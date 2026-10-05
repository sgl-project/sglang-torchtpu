"""CPU-only regression checks for launcher topology and runtime defaults."""

import unittest

from benchmark.tpu.launch_server import (
    LaunchConfig,
    load_recipe,
    resolve_server,
)


class TestRuntimeEnvironment(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
