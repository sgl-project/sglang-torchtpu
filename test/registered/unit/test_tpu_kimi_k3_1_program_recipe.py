"""Kimi-K3.1 colocated program recipe: its defaults, the environment variables
that change them, and the worker's simulated-acceptance warning (CPU only, no
server)."""

import importlib
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# In import order: the K3.1 recipe sizes and extends the Kimi-K3 one.
MODULES = (
    "benchmark.tpu.recipe.configs.kimi_k3.colocated_program",
    "benchmark.tpu.recipe.configs.kimi_k3_1._program",
    "benchmark.tpu.recipe.configs.kimi_k3_1.colocated_program",
)
FP8_WEIGHTS = ("decode_router_fp8", "kda_fp8_projections")
PREVIEW_CURVE = "7:3.69,3:2.88,1:1.82"
RELEASED_CURVE = "7:4.16,3:3.08,1:1.86"
# K3_PROGRAM_K31_DRAFT unset.
DEFAULT_DRAFT = "released"


def resolve(**env):
    """The K3.1 colocated recipe's CONFIG with exactly ``env`` for its
    K3_PROGRAM variables (the recipes read them at import)."""
    base = {k: v for k, v in os.environ.items() if "K3_PROGRAM" not in k}
    with patch.dict(os.environ, {**base, **env}, clear=True):
        for name in MODULES:
            module = importlib.reload(importlib.import_module(name))
        return deepcopy(module.CONFIG)


def kernel_items(config):
    text = config["runtime_env"]["SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG"]
    return [item for item in text.split(";") if item]


def kernel_settings(config):
    """The kernel settings, the last of a name winning, as
    sglang_program.install_kernel_config reads them."""
    return dict(item.split("=", 1) for item in kernel_items(config))


class TestKimiK31ColocatedProgramRecipe(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        with patch.dict(os.environ):
            for name in MODULES:
                importlib.reload(importlib.import_module(name))

    def program(self):
        return importlib.import_module(MODULES[1])

    def test_defaults_serve_concurrency_one(self):
        config = resolve()
        args, env = config["server_args"], config["runtime_env"]
        self.assertEqual(args["max_running_requests"], 4)
        self.assertEqual(args["max_mamba_cache_size"], 24)
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS"], "8")
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS"], "150")
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING"], "1")
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS"], "1")
        self.assertIn("--xla_tpu_dvfs_p_state=7", env["LIBTPU_INIT_ARGS"].split())
        # K3_PROGRAM_CONCURRENCY sizes the engine as the Kimi-K3 recipe does.
        config = resolve(K3_PROGRAM_CONCURRENCY="4")
        k3 = importlib.import_module(MODULES[0]).CONFIG
        for key in ("max_running_requests", "max_mamba_cache_size"):
            self.assertEqual(config["server_args"][key], k3["server_args"][key])
        self.assertEqual(
            config["runtime_env"]["SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS"],
            k3["runtime_env"]["SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS"],
        )

    def test_agentx_server_settings(self):
        config = resolve()
        args = config["server_args"]
        self.assertEqual(args["model_path"], "moonshotai/Kimi-K3.1-Pro")
        # The draft id carries its pinned revision.
        self.assertNotIn("speculative_draft_model_revision", args)
        self.assertEqual(
            config["runtime_env"]["SGLANG_TPU_K3_PROGRAM_DRAFT"], self.program().DRAFT
        )
        self.assertTrue(args["enable_metrics"])
        self.assertNotIn("reasoning_parser", args)
        self.assertNotIn("tool_call_parser", args)

    def test_kernel_settings(self):
        from sglang_tpu.k3_programs.kimi_k3.config import KernelConfig

        program = self.program()
        config = resolve()
        settings = kernel_settings(config)
        expected = dict(item.split("=", 1) for item in program.DECODE_SETTINGS)
        expected.update(
            prefill_expert_parallel="8",
            decode_expert_parallel="8",
            read_streams="16",
            dsa_select_rounds="5",
        )
        self.assertEqual(settings, expected)
        self.assertEqual(len(kernel_items(config)), len(settings))
        self.assertLessEqual(set(settings), set(KernelConfig.__dataclass_fields__))
        # The FP8 weights are opt-in, and K3_PROGRAM_KERNEL_EXTRA comes last.
        for name in FP8_WEIGHTS:
            self.assertNotIn(name, settings)
        extra = "decode_router_fp8=true;kda_fp8_projections=true;dsa_compact_tokens=0"
        config = resolve(K3_PROGRAM_KERNEL_EXTRA=extra)
        self.assertEqual(kernel_items(config)[-3:], extra.split(";"))
        settings = kernel_settings(config)
        self.assertEqual([settings[name] for name in FP8_WEIGHTS], ["true", "true"])
        self.assertEqual(settings["dsa_compact_tokens"], "0")

    def test_simulated_acceptance_is_opt_in(self):
        env = resolve()["runtime_env"]
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS"], "")
        env = resolve(K3_PROGRAM_SIMULATE_ACC_LENGTHS=PREVIEW_CURVE)["runtime_env"]
        self.assertEqual(
            env["SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS"], PREVIEW_CURVE
        )

    def test_draft_selection(self):
        program = self.program()
        self.assertEqual(
            program.DRAFTS,
            {
                "released": (
                    "Inferact/Kimi-K3.1-Pro-DSpark"
                    "@e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f",
                    RELEASED_CURVE,
                ),
                "preview": (
                    "Inferact/Kimi-K3.1-Pro-DSpark-preview"
                    "@6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94",
                    PREVIEW_CURVE,
                ),
            },
        )
        env = resolve()["runtime_env"]
        self.assertEqual(
            env["SGLANG_TPU_K3_PROGRAM_DRAFT"], program.DRAFTS[DEFAULT_DRAFT][0]
        )
        for choice, (draft, curve) in program.DRAFTS.items():
            env = resolve(K3_PROGRAM_K31_DRAFT=choice)["runtime_env"]
            self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_DRAFT"], draft)
            self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS"], "")
            # "curve": the selected draft's InferenceX curve.
            env = resolve(
                K3_PROGRAM_K31_DRAFT=choice, K3_PROGRAM_SIMULATE_ACC_LENGTHS="curve"
            )["runtime_env"]
            self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS"], curve)
        with self.assertRaisesRegex(ValueError, "K3_PROGRAM_K31_DRAFT"):
            resolve(K3_PROGRAM_K31_DRAFT="other")

    def test_environment_overrides(self):
        config = resolve(
            K3_PROGRAM_PREFILL_WAIT="0",
            K3_PROGRAM_LIBTPU_EXTRA="",
            K3_PROGRAM_EXPERT_PARALLEL="16",
            SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS="0",
        )
        env = config["runtime_env"]
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS"], "0")
        self.assertNotIn("xla_tpu_dvfs_p_state", env["LIBTPU_INIT_ARGS"])
        self.assertEqual(env["SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS"], "0")
        settings = kernel_settings(config)
        self.assertEqual(settings["prefill_expert_parallel"], "16")
        self.assertEqual(settings["decode_expert_parallel"], "16")


class TestSimulatedAcceptanceWarning(unittest.TestCase):
    def config(self, role, lengths):
        from sglang_tpu.k3_programs import sglang_config

        return sglang_config.engine_config(
            role=role,
            max_running_requests=4,
            context_length=1 << 20,
            chunked_prefill_size=16384,
            kv_cache_dtype="fp8_e4m3",
            speculative_tokens=7 if role == "decode" else 0,
            schedule=((1, 1, 7), (2, 16, 3), (17, 64, 0)),
            simulate_acc_lengths=lengths,
        )

    def test_names_each_width_rates(self):
        from sglang_tpu.k3_programs.sglang_config import simulated_acceptance_warning

        lengths = {7: 3.69, 3: 2.88, 1: 1.82}
        warning = simulated_acceptance_warning(self.config("decode", lengths))
        self.assertIn("SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS", warning)
        self.assertIn(
            "7 drafts: mean length 3.69, rates [1.0, 1.0, 0.69, 0.0, 0.0, 0.0, 0.0]",
            warning,
        )
        self.assertIn("3 drafts: mean length 2.88, rates [1.0, 0.88, 0.0]", warning)
        self.assertIn("not the model's", warning)
        # Verifying for real, or a role that does not verify: no warning.
        self.assertIsNone(simulated_acceptance_warning(self.config("decode", None)))
        self.assertIsNone(simulated_acceptance_warning(self.config("prefill", lengths)))


if __name__ == "__main__":
    unittest.main()
