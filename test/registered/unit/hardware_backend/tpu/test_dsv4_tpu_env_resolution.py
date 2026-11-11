# SPDX-License-Identifier: Apache-2.0
"""R5 regression: callable HIP env defaults must resolve before DP MoE tracing.

Uses a CPU-only resolution/compile fixture rather than the pool tests' device
allocations. Model metadata is the only external boundary mocked.
"""

import importlib.util
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.arg_groups import model_override_base
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Avoid unrelated plugin package activation (including MLX imports on macOS).
_spec = importlib.util.spec_from_file_location(
    "_dsv4_env_hooks", Path(__file__).resolve().parents[5] / "sglang_tpu/hooks/dsv4.py"
)
_hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hooks)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4TPUEnvResolution(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(torch._dynamo.reset)
        # Restore every environment flag the actual resolution hook may modify.
        for name in (
            "SGLANG_OPT_FP8_WO_A_GEMM",
            "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
            "SGLANG_OPT_USE_TILELANG_MHC_PRE",
            "SGLANG_OPT_USE_TILELANG_MHC_POST",
            "SGLANG_OPT_FUSE_MHC_POST_PRE",
            "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
            "SGLANG_OPT_USE_FUSED_HASH_TOPK",
            "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",
            "SGLANG_DP_USE_REDUCE_SCATTER",
        ):
            field = getattr(envs, name)
            self.stack.enter_context(field.override(field.get()))

    def resolve(self, device="tpu", architecture="DeepseekV4ForCausalLM"):
        args = ServerArgs(model_path="dummy", device=device)
        model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))
        with patch.object(model_override_base, "model_config_of", return_value=model):
            _hooks._tpu_deepseek_v4_env_adjustments(args, lambda sa: None)

    def test_unset_callable_default_resolves_before_fullgraph(self):
        field = envs.SGLANG_DP_USE_REDUCE_SCATTER
        self.assertTrue(callable(field.default))
        field.clear()
        ordinary_default = field.get()
        self.resolve()

        def dp_branch(x):
            return x + 1 if envs.SGLANG_DP_USE_REDUCE_SCATTER.get() else x - 1

        # No test-side set/get cache: the actual model-resolution hook must make
        # the real descriptor safe for fullgraph, as the production DP MoE reads it.
        compiled = torch.compile(dp_branch, backend="eager", fullgraph=True)
        x = torch.arange(4, dtype=torch.float32)
        torch.testing.assert_close(compiled(x), x + (1 if ordinary_default else -1))
        self.assertTrue(field.is_set())
        self.assertIs(field.get(), ordinary_default)

    def test_explicit_choices_are_preserved(self):
        field = envs.SGLANG_DP_USE_REDUCE_SCATTER
        for value in (True, False, None):
            with self.subTest(value=value), field.override(value):
                self.resolve()
                self.assertIs(field.get(), value)

    def test_non_tpu_or_non_v4_does_not_materialize_default(self):
        field = envs.SGLANG_DP_USE_REDUCE_SCATTER
        for device, architecture in (
            ("cpu", "DeepseekV4ForCausalLM"),
            ("tpu", "LlamaForCausalLM"),
        ):
            with self.subTest(device=device, architecture=architecture):
                field.clear()
                ordinary_default = field.get()
                self.resolve(device, architecture)
                self.assertFalse(field.is_set())
                self.assertIs(field.get(), ordinary_default)


if __name__ == "__main__":
    unittest.main()
