# SPDX-License-Identifier: Apache-2.0
"""srt's torch_tpu ATen fallbacks match the CPU kernels."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch
import torch.nn.functional as F

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

    def test_antialiased_bilinear_resize(self):
        # The op behind torchvision's Resize(antialias=True) (Hunyuan3D-2's
        # conditioner): the CPU fallback, or torch_tpu's kernel where it has one,
        # which need not round like the CPU kernel.
        image = torch.rand(1, 3, 512, 512)
        for dtype in (torch.float32, torch.float16):
            x = image.to(dtype)
            kwargs = dict(size=(518, 518), mode="bilinear", antialias=True)
            got = F.interpolate(x.to("tpu"), **kwargs).cpu()
            want = F.interpolate(x.float(), **kwargs).to(dtype)
            torch.testing.assert_close(got, want, rtol=0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
