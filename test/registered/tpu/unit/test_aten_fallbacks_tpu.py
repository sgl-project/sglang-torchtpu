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

    def test_copysign_fullgraph_signed_zeros_and_nan(self):
        from sglang_tpu.patches import _register_tpu_compile_backend

        _register_tpu_compile_backend()

        def forward(a, b):
            result = torch.copysign(a, b)
            out = torch.empty_like(result)
            torch.copysign(a, b, out=out)
            return result, out

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        1.0,
                        -1.0,
                        float("inf"),
                        -float("inf"),
                        float("nan"),
                        -float("nan"),
                    ],
                    dtype=dtype,
                ).reshape(2, 4)
                b = torch.tensor([-0.0, 0.0, -float("nan"), float("nan")], dtype=dtype)
                expected = torch.copysign(a, b)
                compiled = torch.compile(
                    forward, backend="tpu", fullgraph=True, dynamic=False
                )
                for _ in range(2):
                    for result in compiled(a.to("tpu"), b.to("tpu")):
                        got = result.cpu()
                        torch.testing.assert_close(
                            got, expected, atol=0, rtol=0, equal_nan=True
                        )
                        torch.testing.assert_close(
                            torch.signbit(got),
                            torch.signbit(expected),
                            atol=0,
                            rtol=0,
                        )

    def test_copysign_out_promotion(self):
        # Promote before abs: abs(min_int32) would overflow in its input dtype.
        a = torch.tensor([torch.iinfo(torch.int32).min, 0, 1], dtype=torch.int32)
        b = torch.tensor([-0.0, 1.0, -1.0], dtype=torch.float32)
        expected = torch.copysign(a, b)
        out = torch.empty(0, dtype=torch.float32, device="tpu")
        result = torch.copysign(a.to("tpu"), b.to("tpu"), out=out)
        self.assertIs(result, out)
        torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
