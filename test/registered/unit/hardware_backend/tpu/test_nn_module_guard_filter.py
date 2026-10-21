"""SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER: when the decoder compile drops
Dynamo guards on nn.Module state (CPU-runnable)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import decoder_compile_policy as dcp
from sglang_tpu.envs import envs as tpu_envs


def _server_args(**overrides):
    args = dict(
        enable_lora=None,
        lora_paths=None,
        enable_eplb=False,
        elastic_ep_backend=None,
        unsafe_save_torch_compile_bundle=None,
        unsafe_load_torch_compile_bundle=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class TestGuardFilterOptions(CustomTestCase):
    def test_off_by_default(self):
        self.assertIsNone(dcp._guard_filter_options(_server_args()))

    def test_on_skips_nn_module_guards(self):
        with tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.override(True):
            options = dcp._guard_filter_options(_server_args())
        self.assertEqual(
            options,
            {"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
        )

    def test_runtime_module_state_features_keep_all_guards(self):
        for overrides in (
            {"enable_lora": True},
            {"lora_paths": ["adapter"]},
            {"enable_eplb": True},
            {"elastic_ep_backend": "mooncake"},
        ):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                with tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.override(True):
                    self.assertIsNone(
                        dcp._guard_filter_options(_server_args(**overrides))
                    )

    def test_lora_explicitly_off_with_paths_keeps_filter(self):
        # enable_lora=False wins over leftover lora_paths, as in server_args.
        with tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.override(True):
            options = dcp._guard_filter_options(
                _server_args(enable_lora=False, lora_paths=["adapter"])
            )
        self.assertIsNotNone(options)


class TestDecoderCompileUsesFilter(CustomTestCase):
    def _compile(self, server_args):
        policy = SimpleNamespace(
            model_runner=SimpleNamespace(server_args=server_args),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        with (
            patch.object(dcp.torch, "compile") as compile_mock,
            patch.object(dcp.torch_compile_bundle, "Compiler") as bundle_mock,
        ):
            dcp.DecoderTpuCompilePolicy._compile(policy, lambda x: x)
        return compile_mock, bundle_mock

    def test_options_only_when_enabled(self):
        compile_mock, _ = self._compile(_server_args())
        self.assertNotIn("options", compile_mock.call_args.kwargs)
        with tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.override(True):
            compile_mock, _ = self._compile(_server_args())
        self.assertEqual(
            compile_mock.call_args.kwargs["options"],
            {"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
        )

    def test_bundle_keeps_its_own_filter(self):
        with tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.override(True):
            compile_mock, bundle_mock = self._compile(
                _server_args(unsafe_load_torch_compile_bundle="/tmp/bundle")
            )
        compile_mock.assert_not_called()
        self.assertNotIn("options", bundle_mock.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
