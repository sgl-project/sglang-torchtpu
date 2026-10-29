"""TPU-only backend initialization through the real scheduler plugin hook."""

import unittest
from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.runtime_context import get_context, get_flags, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import hooks

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuSchedulerHook(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in (
            "initialize_fp8_gemm_config",
            "initialize_fp4_gemm_config",
            "initialize_bf16_gemm_config",
        ):
            self.stack.enter_context(patch.object(scheduler_module, name))
        self.stack.enter_context(
            patch.object(HookRegistry, "_hooks", defaultdict(list))
        )
        self.stack.enter_context(patch.object(HookRegistry, "_patched", set()))
        self.stack.enter_context(patch.object(hooks, "_registered", False))
        original = scheduler_module.Scheduler.init_moe_gemm_config
        self.stack.enter_context(
            patch.object(scheduler_module.Scheduler, "init_moe_gemm_config", original)
        )

    def initialize(self, *, tpu, moe=False):
        platform = SimpleNamespace(is_tpu=lambda: tpu)
        with patch("sglang.srt.platforms.current_platform", platform):
            hooks.register()
            hooks.register()  # Both TPU distribution entry points may load.
        HookRegistry.apply_hooks()
        config = SimpleNamespace(num_experts_per_tok=2) if moe else SimpleNamespace()
        scheduler = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config))
        scheduler_module.Scheduler.init_moe_gemm_config(scheduler)
        self.assertFalse(scheduler.require_mlp_sync)

    def test_dense_tpu_flags_are_initialized_before_tracing(self):
        self.initialize(tpu=True)
        self.assertIsNotNone(get_flags().moe.speculative_runner_backend)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        def forward(x):
            if (
                moe_utils.get_moe_runner_backend().is_auto()
                and moe_utils.get_moe_a2a_backend().is_none()
            ):
                return x + 1
            return x - 1

        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.ones(2)), torch.full((2,), 2.0))

    def test_moe_model_keeps_base_initialization(self):
        with patch.object(moe_utils, "initialize_moe_config") as extra_init:
            self.initialize(tpu=True, moe=True)
        extra_init.assert_not_called()
        self.assertIsNotNone(get_flags().moe.runner_backend)

    def test_non_tpu_dense_initialization_is_unchanged(self):
        self.initialize(tpu=False)
        self.assertIsNone(get_flags().moe.runner_backend)
        self.assertTrue(moe_utils.get_moe_a2a_backend().is_none())


if __name__ == "__main__":
    unittest.main()
