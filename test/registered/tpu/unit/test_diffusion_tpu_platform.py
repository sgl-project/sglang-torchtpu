"""TPU CI entry for the sglang-diffusion (multimodal_gen) TPU platform tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit")

import unittest

import torch

from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.test.unit.test_tpu_platform import (  # noqa: F401
    TestTorchTpuIntegration,
    TestTpuPlatformPlugin,
    TestTpuPlatformSurface,
)
from sglang.test.test_utils import CustomTestCase


class TestWorkerBootstrap(CustomTestCase):
    """Real-hardware worker initialization smoke test."""

    def test_set_device_brings_up_the_runtime(self):
        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        self.assertTrue(torch.tpu.is_initialized())

        x = torch.ones((8, 128), device="tpu", dtype=torch.bfloat16)
        y = (x + x).sum()
        torch.tpu.synchronize()
        self.assertEqual(y.item(), 2 * 8 * 128)

        self.assertGreater(TpuPlatform.get_device_total_memory(), 0)
        self.assertGreater(TpuPlatform.get_current_memory_usage(), 0.0)


if __name__ == "__main__":
    unittest.main()
