"""The plugin attaches TPU compile policies to core model classes it lists."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.compilation.support_torch_compile import (
    _COMPILE_POLICY_ATTR,
    CompilePolicy,
    get_torch_compile_policy,
    support_torch_compile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import core_compile_annotations

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

QWEN3_VL_MOE = "sglang.srt.models.qwen3_vl_moe"


class FakeQwen3VLDecoderTpuCompilePolicy(CompilePolicy):
    pass


class OtherPolicy(CompilePolicy):
    pass


def _fake_policy_module():
    # The real policy module imports TorchTPU; the annotator only needs the name.
    module = types.ModuleType("sglang_tpu.compilation.decoder_compile_policy")
    module.Qwen3VLDecoderTpuCompilePolicy = FakeQwen3VLDecoderTpuCompilePolicy
    return {module.__name__: module}


def _drop_policy(cls):
    if _COMPILE_POLICY_ATTR in vars(cls):
        delattr(cls, _COMPILE_POLICY_ATTR)


class TestCoreCompileAnnotations(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.qwen3_vl_moe = importlib.import_module(QWEN3_VL_MOE)

    def setUp(self):
        # Annotations are class attributes: undo them so other tests in this
        # process see the core classes as core defines them.
        for cls in (
            self.qwen3_vl_moe.Qwen3MoeLLMModel,
            self.qwen3_vl_moe.Qwen3MoeModel,
        ):
            self.assertNotIn(_COMPILE_POLICY_ATTR, vars(cls))
            self.addCleanup(_drop_policy, cls)
        modules_patch = patch.dict(sys.modules, _fake_policy_module())
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def test_qwen3_vl_moe_decoder_gets_the_vl_decoder_policy(self):
        core_compile_annotations.annotate_core_compile_boundaries()

        self.assertIs(
            get_torch_compile_policy(self.qwen3_vl_moe.Qwen3MoeLLMModel),
            FakeQwen3VLDecoderTpuCompilePolicy,
        )
        # The text-only Qwen3-MoE decoder stays unannotated, so it keeps
        # TpuGraphRunner's whole-model fallback.
        self.assertIsNone(get_torch_compile_policy(self.qwen3_vl_moe.Qwen3MoeModel))

    def test_unloaded_model_module_is_not_imported(self):
        with patch.dict(sys.modules):
            sys.modules.pop(QWEN3_VL_MOE)
            core_compile_annotations.annotate_core_compile_boundaries()
            self.assertNotIn(QWEN3_VL_MOE, sys.modules)
        self.assertIsNone(get_torch_compile_policy(self.qwen3_vl_moe.Qwen3MoeLLMModel))

    def test_a_policy_core_already_declares_wins(self):
        support_torch_compile(self.qwen3_vl_moe.Qwen3MoeLLMModel, policy=OtherPolicy)

        core_compile_annotations.annotate_core_compile_boundaries()

        self.assertIs(
            get_torch_compile_policy(self.qwen3_vl_moe.Qwen3MoeLLMModel), OtherPolicy
        )

    def test_every_listed_boundary_names_a_real_class(self):
        for boundary in core_compile_annotations.CORE_COMPILE_BOUNDARIES:
            module = importlib.import_module(boundary.module)
            self.assertTrue(
                isinstance(getattr(module, boundary.class_name, None), type),
                f"{boundary.module}.{boundary.class_name} is not a class",
            )


if __name__ == "__main__":
    unittest.main()
