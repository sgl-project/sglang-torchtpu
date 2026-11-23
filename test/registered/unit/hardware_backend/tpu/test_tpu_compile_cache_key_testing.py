"""CPU tests: sglang_tpu/testing stays out of the TPU compile cache key."""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope
from sglang_tpu import compile_cache
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(compile_cache.__file__).resolve().parent
_SRT_ROOT = Path(sglang.__file__).resolve().parent / "srt"


class TestUnhashedTestingTree(CustomTestCase):
    def setUp(self):
        self.root = Path(enter_scope(self, tempfile.TemporaryDirectory()))
        self.write("sglang_tpu/kv_page_write.py", "out = pages\n")
        self.write("sglang_tpu/testing/evals/model_config.py", "GATES = []\n")
        self.write(
            "sglang/srt/layers/moe/tpu_moe_utils.py",
            "MODE = envs.SGLANG_TPU_INT4_MOE_MODE.get()\n",
        )
        enter_scope(
            self, patch.object(compile_cache, "_tpu_pci_device_ids", return_value=[])
        )
        enter_scope(self, tpu_envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.override(False))
        self.base = self.digest()

    def write(self, relative_path, text):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def digest(self):
        return compile_cache.tpu_kernel_inputs_digest(
            package_root=self.root / "sglang_tpu", srt_root=self.root / "sglang/srt"
        )

    def test_testing_edits_leave_the_key_and_package_edits_change_it(self):
        self.write("sglang_tpu/testing/evals/model_config.py", "GATES = [1]\n")
        self.write("sglang_tpu/testing/new_harness.py", "x = 1\n")
        self.assertEqual(self.digest(), self.base)
        # A knob read only by a test harness compiles nothing, so it is not keyed.
        self.write(
            "sglang_tpu/testing/utils/probe.py",
            "FP32 = tpu_envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.get()\n",
        )
        with tpu_envs.SGLANG_TPU_FP8_MOE_FP32_SCALES.override(True):
            self.assertEqual(self.digest(), self.base)

        self.write("sglang_tpu/kv_page_write.py", "out = pages  # edited\n")
        self.assertNotEqual(self.digest(), self.base)

    def test_unhashed_dirs_are_off_the_serving_path(self):
        """Excluding a directory is safe only while no serving module imports it
        and it builds no kernels; either would reuse stale executables."""
        unhashed = tuple(
            f"sglang_tpu.{name}" for name in sorted(compile_cache.UNHASHED_PACKAGE_DIRS)
        )
        serving = [
            path
            for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
            if path.relative_to(_PACKAGE_ROOT).parts[0]
            not in compile_cache.UNHASHED_PACKAGE_DIRS
        ]
        serving += sorted(_SRT_ROOT.rglob("*.py"))
        importers = [
            str(path)
            for path in serving
            if any(module.startswith(unhashed) for module in _imported_modules(path))
        ]
        self.assertEqual(importers, [], "serving code imports an unhashed directory")

        kernel_users = [
            str(path)
            for name in compile_cache.UNHASHED_PACKAGE_DIRS
            for path in sorted((_PACKAGE_ROOT / name).rglob("*.py"))
            if any(
                module.split(".")[0] == "jax"
                or module.startswith("torch_tpu._internal.pallas")
                for module in _imported_modules(path)
            )
        ]
        self.assertEqual(kernel_users, [], "a kernel in an unhashed directory")


def _imported_modules(path):
    """Absolute names of the modules a file imports, or names in string literals."""
    source = path.read_bytes()
    if not any(
        word in source for word in (b"sglang_tpu", b"from .", b"jax", b"pallas")
    ):
        return []
    modules = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_module(path, node.module, node.level)
            modules += [base, *(f"{base}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # importlib.import_module("sglang_tpu.testing...") and similar.
            if node.value.startswith("sglang_tpu."):
                modules.append(node.value)
    return modules


def _absolute_module(path, module, level):
    if not level or not path.is_relative_to(_PACKAGE_ROOT):
        return module or ""
    package = path.relative_to(_PACKAGE_ROOT.parent).parts[:-1]
    parts = [*package[: len(package) - level + 1], *([module] if module else [])]
    return ".".join(parts)


if __name__ == "__main__":
    unittest.main()
