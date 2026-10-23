# SPDX-License-Identifier: Apache-2.0
"""srt's torch_tpu ATen fallbacks match the CPU kernels."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.env_override import apply_tpu_env_overrides
from sglang_tpu.patches import _register_aten_fallbacks


@unittest.skipUnless(hasattr(torch, "tpu"), "requires torch_tpu")
class TestAtenFallbacksTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        apply_tpu_env_overrides()
        _register_aten_fallbacks()

    def test_logaddexp(self):
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(2, 8, 1024, generator=gen) * 30
        b = torch.randn(2, 8, 1024, generator=gen) * 30
        a[0, 0, :2] = float("-inf")
        b[0, 0, :4] = float("-inf")
        a[0, 1, :2] = float("inf")
        b[0, 1, 1:3] = float("inf")
        got = torch.logaddexp(a.to("tpu"), b.to("tpu")).cpu()
        torch.testing.assert_close(got, torch.logaddexp(a, b), rtol=1e-5, atol=1e-5)

    def test_logaddexp_broadcasts(self):
        a = torch.randn(4, 1, 256)
        b = torch.randn(1, 16, 256)
        got = torch.logaddexp(a.to("tpu"), b.to("tpu")).cpu()
        torch.testing.assert_close(got, torch.logaddexp(a, b), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
