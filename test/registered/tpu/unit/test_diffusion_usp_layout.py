# SPDX-License-Identifier: Apache-2.0
"""On-device checks for the device-neutral Ulysses output relayout."""

import importlib.util
import math
import unittest

import torch

from sglang.kernels.ops.diffusion import can_use_usp_merge_heads, usp_merge_heads
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)


class TestUspLayoutTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        # Do not turn initialization/lowering failures into a successful skip.
        torch.zeros(1, device="tpu").cpu()

    def _check(self, merge):
        for dtype in (torch.bfloat16, torch.float32):
            for shape in ((2, 3, 1, 2, 8), (4, 3, 2, 2, 8)):
                with self.subTest(dtype=dtype, shape=shape):
                    host = (
                        (torch.arange(math.prod(shape)) % 97).reshape(shape).to(dtype)
                    )
                    device = host.to("tpu")
                    for x, reference in (
                        (device, host),
                        (device.transpose(1, 3), host.transpose(1, 3)),
                    ):
                        self.assertFalse(can_use_usp_merge_heads(x))
                        actual = merge(x)
                        self.assertEqual(actual.device.type, "tpu")
                        torch.testing.assert_close(
                            actual.cpu(),
                            reference.permute(2, 1, 0, 3, 4).contiguous(),
                            rtol=0,
                            atol=0,
                        )

    def test_eager_matches_cpu_exactly(self):
        self._check(usp_merge_heads)

    def test_compiled_matches_cpu_exactly(self):
        compiled = torch.compile(
            usp_merge_heads, backend="tpu", fullgraph=True, dynamic=False
        )
        self._check(compiled)


if __name__ == "__main__":
    unittest.main()
