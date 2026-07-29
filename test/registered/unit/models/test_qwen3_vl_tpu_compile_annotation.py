import importlib
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    get_torch_compile_policy,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


class FakeTpuPlatform:
    device_type = "tpu"

    def is_cpu(self):
        return False

    def is_npu(self):
        return False

    def is_out_of_tree(self):
        return False

    def is_tpu(self):
        return True

    def supports_torch_compile(self):
        return True


class FakeViTTpuCompileRunner:
    pass


class FakeVisionTpuCompilePolicy(CompilePolicy):
    platform = "tpu"
    backend = "tpu"


class FakeQwen3VLDecoderTpuCompilePolicy(CompilePolicy):
    platform = "tpu"


def _stub_tpu_modules():
    vit_module = types.ModuleType(
        "sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner"
    )
    vit_module.ViTTpuCompileRunner = FakeViTTpuCompileRunner
    vit_module.VisionTpuCompilePolicy = FakeVisionTpuCompilePolicy

    decoder_policy_module = types.ModuleType(
        "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy"
    )
    decoder_policy_module.DecoderTpuCompilePolicy = FakeQwen3VLDecoderTpuCompilePolicy
    decoder_policy_module.Qwen3VLDecoderTpuCompilePolicy = (
        FakeQwen3VLDecoderTpuCompilePolicy
    )

    return {
        vit_module.__name__: vit_module,
        decoder_policy_module.__name__: decoder_policy_module,
    }


class TestQwen3VLTpuCompileAnnotation(CustomTestCase):
    def test_qwen3_vl_decoder_policy_is_attached_to_class_forward(self):
        import sglang.srt.platforms as platforms
        import sglang.srt.utils.common as common

        module_name = "sglang.srt.models.qwen3_vl"
        old_module = sys.modules.pop(module_name, None)
        common.is_cpu.cache_clear()
        common.is_npu.cache_clear()
        common.is_tpu.cache_clear()

        try:
            with (
                patch.object(platforms, "_current_platform", FakeTpuPlatform()),
                patch.dict(sys.modules, _stub_tpu_modules()),
            ):
                qwen3_vl = importlib.import_module(module_name)

                policy = get_torch_compile_policy(qwen3_vl.Qwen3LLMModel)
                self.assertIs(policy, FakeQwen3VLDecoderTpuCompilePolicy)
                self.assertEqual(policy.platform, "tpu")

                vision_policy = get_torch_compile_policy(qwen3_vl.Qwen3VLMoeVisionModel)
                self.assertIs(vision_policy, FakeVisionTpuCompilePolicy)
                self.assertEqual(vision_policy.platform, "tpu")
                self.assertIs(
                    qwen3_vl.graph_runners_dict["tpu"], FakeViTTpuCompileRunner
                )
        finally:
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module
            common.is_cpu.cache_clear()
            common.is_npu.cache_clear()
            common.is_tpu.cache_clear()

    def test_tpu_platform_selects_tpu_vision_backend(self):
        import sglang.srt.layers.attention.vision as vision

        attention = object.__new__(vision.VisionAttention)
        with (
            patch.object(
                vision,
                "get_global_server_args",
                return_value=types.SimpleNamespace(mm_attention_backend=None),
            ),
            patch.object(vision, "is_cuda", return_value=False),
            patch.multiple(
                vision,
                _is_tpu=True,
                _is_musa=False,
                _is_hip=False,
                _is_cpu=False,
                _is_xpu=False,
            ),
        ):
            self.assertEqual(attention._determine_attention_backend(None), "tpu")
            self.assertEqual(attention._determine_attention_backend("sdpa"), "sdpa")


if __name__ == "__main__":
    unittest.main()
