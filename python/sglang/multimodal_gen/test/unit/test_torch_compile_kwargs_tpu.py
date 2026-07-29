import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class _FakeTpuBackend:
    def __init__(self, debug=False, dynamism=False, enable_serialization=True):
        self.debug = debug
        self.dynamism = dynamism
        self.enable_serialization = enable_serialization


class TestBuildTorchCompileKwargsTpu(unittest.TestCase):
    def test_tpu_branch_disables_serialization_and_dynamic_shapes(self):
        from sglang.multimodal_gen.runtime.utils import torch_compile

        fake_compile_module = types.ModuleType("torch_tpu._internal.compile")
        fake_compile_module.TpuBackend = _FakeTpuBackend
        platform = MagicMock()
        platform.is_npu.return_value = False
        platform.is_tpu.return_value = True
        with (
            patch.object(torch_compile, "current_platform", platform),
            patch.dict(
                sys.modules, {"torch_tpu._internal.compile": fake_compile_module}
            ),
        ):
            kwargs = torch_compile.build_torch_compile_kwargs(mode="max-autotune")

        backend = kwargs["backend"]
        self.assertIsInstance(backend, _FakeTpuBackend)
        self.assertFalse(backend.enable_serialization)
        self.assertFalse(backend.dynamism)
        self.assertFalse(kwargs["dynamic"])
        # inductor modes must not reach the tpu backend
        self.assertNotIn("mode", kwargs)


if __name__ == "__main__":
    unittest.main()
