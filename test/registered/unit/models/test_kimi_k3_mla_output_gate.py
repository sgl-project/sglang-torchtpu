"""MLA output-gate kernel selection."""

import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.mla_output_gate as mla_output_gate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaOutputGateCoverage(CustomTestCase):
    def test_tpu_uses_torch_fallback(self):
        """TPU must not reach the JIT CUDA kernel."""
        x = torch.zeros(4, 16, dtype=torch.bfloat16)
        with patch.object(mla_output_gate, "_is_tpu", True):
            self.assertFalse(mla_output_gate.covered(x, x.clone()))


if __name__ == "__main__":
    unittest.main()
