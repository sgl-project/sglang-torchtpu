# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU Pallas jax_op bridge."""

import importlib.util
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.tpu.patches import get_tpu_jax_op
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestTpuJaxOpBridge(CustomTestCase):
    """Test TPU Pallas jax_op compatibility bridge."""

    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu")
        or (
            importlib.util.find_spec("torch")
            and hasattr(sys.modules.get("torch"), "tpu")
        ),
        "requires TorchTPU",
    )
    def test_get_tpu_jax_op_runtime(self):
        """Verify get_tpu_jax_op returns a callable in current environment."""
        op = get_tpu_jax_op()
        self.assertTrue(callable(op))

    def test_get_tpu_jax_op_prefers_new_api(self):
        """Verify get_tpu_jax_op prioritizes torch.tpu.pallas.jax_op."""
        mock_new_op = Mock(name="torch_tpu_pallas_jax_op")
        fake_module = ModuleType("torch.tpu.pallas")
        fake_module.jax_op = mock_new_op

        with patch.dict(sys.modules, {"torch.tpu.pallas": fake_module}):
            resolved_op = get_tpu_jax_op()
            self.assertIs(resolved_op, mock_new_op)

    def test_get_tpu_jax_op_legacy_fallback(self):
        """Verify get_tpu_jax_op falls back to torch_tpu._internal.pallas.jax_op when new API is missing."""
        mock_legacy_op = Mock(name="legacy_pallas_jax_op")
        fake_legacy_module = ModuleType("torch_tpu._internal.pallas")
        fake_legacy_module.jax_op = mock_legacy_op

        modules_to_patch = {
            "torch.tpu.pallas": None,
            "torch_tpu._internal.pallas": fake_legacy_module,
        }
        with patch.dict(sys.modules, modules_to_patch):
            resolved_op = get_tpu_jax_op()
            self.assertIs(resolved_op, mock_legacy_op)

    def test_get_tpu_jax_op_attribute_error_fallback(self):
        """Verify get_tpu_jax_op falls back when torch.tpu.pallas lacks jax_op."""
        mock_legacy_op = Mock(name="legacy_pallas_jax_op")
        fake_legacy_module = ModuleType("torch_tpu._internal.pallas")
        fake_legacy_module.jax_op = mock_legacy_op

        empty_new_module = ModuleType("torch.tpu.pallas")

        modules_to_patch = {
            "torch.tpu.pallas": empty_new_module,
            "torch_tpu._internal.pallas": fake_legacy_module,
        }
        with patch.dict(sys.modules, modules_to_patch):
            resolved_op = get_tpu_jax_op()
            self.assertIs(resolved_op, mock_legacy_op)


if __name__ == "__main__":
    unittest.main()
