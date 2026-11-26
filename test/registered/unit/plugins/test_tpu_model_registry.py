# SPDX-License-Identifier: Apache-2.0
"""The TPU plugin's model classes win over upstream classes of the same architecture.

``sglang.srt.models`` may gain a model whose architecture ``sglang_tpu.models``
also implements (``MuseSparkForConditionalGeneration``); a TPU worker must still
build the plugin's class, under ``SGLANG_PLATFORM=tpu`` and ``tpu_oot`` alike.
Every other architecture resolves exactly as it does without the plugin.

Runs on CPU: ``init_backend`` runs with its TorchTPU setup mocked out, against a
fresh registry so the process-wide one is left alone.
"""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.models.registry import _ModelRegistry, import_model_classes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.platform import TpuOOTPlatform  # noqa: E402
from sglang_tpu.tpu_platform import TpuSRTPlatform  # noqa: E402

DEFAULT_PACKAGE = "sglang.srt.models"
PLUGIN_PACKAGE = "sglang_tpu.models"
MUSE_SPARK = "MuseSparkForConditionalGeneration"
PLATFORMS = (TpuSRTPlatform, TpuOOTPlatform)

# Set in the scheduler worker; TpuOOTPlatform defers init_backend until then.
_WORKER_MARKER = "SGLANG_TPU_CACHE_DP_SCOPE"


class _ExternalModel(nn.Module):
    pass


def _init_backend(platform_cls, registry: _ModelRegistry) -> None:
    """Run ``platform_cls().init_backend()`` with ``registry`` as the model registry."""
    with (
        patch.dict(os.environ, {"WORLD_SIZE": "1", _WORKER_MARKER: "test"}),
        patch.dict(sys.modules, {"torch_tpu": ModuleType("torch_tpu")}),
        patch("sglang_tpu.patches._prepare_tpu_env"),
        patch("sglang_tpu.compile_cache.add_tpu_kernel_inputs_to_compile_cache_key"),
        patch("sglang_tpu.patches.apply_tpu_patches"),
        patch("sglang_tpu.tpu_platform.rank_slice._load"),
        patch("sglang_tpu.tpu_platform.traced_rank._load"),
        patch("sglang_tpu.tpu_platform._register_attention_backends"),
        patch("sglang.srt.models.registry.ModelRegistry", registry),
    ):
        platform_cls().init_backend()


class TestTpuModelRegistration(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_models = dict(import_model_classes(DEFAULT_PACKAGE))
        cls.plugin_models = dict(import_model_classes(PLUGIN_PACKAGE, strict=True))
        cls.collisions = sorted(cls.default_models.keys() & cls.plugin_models.keys())

    def _tpu_registry(self, platform_cls) -> _ModelRegistry:
        # The registry registry.py builds at import without an external package.
        registry = _ModelRegistry(models=dict(self.default_models))
        _init_backend(platform_cls, registry)
        return registry

    def test_first_lookup_of_a_colliding_arch_returns_the_plugin_class(self):
        """A TPU worker builds the plugin's class, not upstream's CUDA one."""
        for platform_cls in PLATFORMS:
            for arch in sorted({MUSE_SPARK, *self.collisions}):
                with self.subTest(platform=platform_cls.__name__, arch=arch):
                    registry = self._tpu_registry(platform_cls)
                    model_cls, resolved = registry.resolve_model_cls([arch])
                    self.assertIs(model_cls, self.plugin_models[arch])
                    self.assertEqual(resolved, arch)

    def test_every_arch_resolves_to_the_plugin_or_else_the_upstream_class(self):
        expected = self.default_models | self.plugin_models
        for platform_cls in PLATFORMS:
            registry = self._tpu_registry(platform_cls)
            for arch in sorted(expected):
                with self.subTest(platform=platform_cls.__name__, arch=arch):
                    self.assertIs(registry.resolve_model_cls(arch)[0], expected[arch])

    def test_unknown_arch_falls_back_to_transformers(self):
        """Loading the full catalog must not reject the plugin package."""
        for platform_cls in PLATFORMS:
            with self.subTest(platform=platform_cls.__name__):
                registry = self._tpu_registry(platform_cls)
                # get_model_architecture's order: support check, then resolve.
                self.assertFalse(registry.is_arch_supported("NotAModelForCausalLM"))
                model_cls, arch = registry.resolve_model_cls(["NotAModelForCausalLM"])
                self.assertEqual(arch, "TransformersForCausalLM")
                self.assertIs(model_cls, self.default_models[arch])

    def test_external_model_package_keeps_its_fast_path(self):
        # The registry registry.py builds with SGLANG_EXTERNAL_MODEL_PACKAGE set.
        for platform_cls in PLATFORMS:
            with self.subTest(platform=platform_cls.__name__):
                registry = _ModelRegistry(
                    models={"ExternalArch": _ExternalModel},
                    _default_models_loaded=False,
                )
                _init_backend(platform_cls, registry)
                self.assertIs(
                    registry.resolve_model_cls("ExternalArch")[0], _ExternalModel
                )
                # Neither the upstream nor the plugin catalog was loaded for it.
                self.assertFalse(registry._default_models_loaded)
                self.assertEqual(set(registry.models), {"ExternalArch"})
                self.assertIs(
                    registry.resolve_model_cls(MUSE_SPARK)[0],
                    self.plugin_models[MUSE_SPARK],
                )


if __name__ == "__main__":
    unittest.main()
