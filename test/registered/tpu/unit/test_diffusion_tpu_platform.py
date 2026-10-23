"""TPU CI entry for the sglang-diffusion (multimodal_gen) TPU platform tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.test.unit.test_server_args import (  # noqa: F401
    TestTpuDistributedArgs,
)
from sglang.multimodal_gen.test.unit.test_tpu_platform import (  # noqa: F401
    TestSplashBackendSelection,
    TestTorchTpuIntegration,
    TestTpuPlatformPlugin,
    TestTpuPlatformSurface,
    TestWorkerInitOnce,
)
from sglang.test.test_utils import CustomTestCase


class TestWorkerBootstrap(CustomTestCase):
    """Real-hardware worker initialization smoke test."""

    def test_set_device_brings_up_the_runtime(self):
        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        self.assertTrue(torch.tpu.is_initialized())

        x = torch.ones((8, 128), device="tpu", dtype=torch.bfloat16)
        y = (x + x).sum()
        torch.accelerator.synchronize()
        self.assertEqual(y.item(), 2 * 8 * 128)

        self.assertGreater(TpuPlatform.get_device_total_memory(), 0)
        self.assertGreater(TpuPlatform.get_current_memory_usage(), 0.0)

    def test_device_tensors_survive_a_pickle_round_trip(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        contiguous = torch.arange(12, dtype=torch.float32, device="tpu") * 2.0
        strided = contiguous.view(3, 4).transpose(0, 1)
        for original in (contiguous, strided):
            restored = pickle.loads(dist_utils._dumps([original]))[0]
            self.assertEqual(restored.device.type, "tpu")
            self.assertEqual(restored.shape, original.shape)
            self.assertEqual(restored.stride(), original.stride())
            self.assertEqual(restored.is_contiguous(), original.is_contiguous())
            self.assertTrue(torch.equal(restored.cpu(), original.cpu()))

    def test_requires_grad_survives_the_round_trip(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        for requires_grad in (False, True):
            original = torch.ones(4, dtype=torch.float32, device="tpu")
            original.requires_grad_(requires_grad)
            restored = pickle.loads(dist_utils._dumps([original]))[0]
            self.assertEqual(restored.requires_grad, requires_grad)


class TestFloat64IsRealOnDevice(CustomTestCase):
    """The measurements behind `TpuPlatform.is_float64_supported() -> True`."""

    def test_elementwise_float64_matches_the_host(self):
        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        self.assertTrue(TpuPlatform.is_float64_supported())
        inv = 1.0 / (
            10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64, device="tpu") / 64)
        )
        pos = torch.arange(128, dtype=torch.float64, device="tpu")
        got = torch.outer(pos, inv).cos().cpu()

        ref_inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        ref = torch.outer(torch.arange(128, dtype=torch.float64), ref_inv).cos()
        self.assertEqual(got.dtype, torch.float64)
        self.assertLess((got - ref).abs().max().item(), 1e-12)

    def test_float64_matmul_still_downcasts(self):
        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        a = torch.linspace(0, 1, 64, device="tpu").double().reshape(8, 8)
        got = (a @ a.t()).cpu()
        ref_a = torch.linspace(0, 1, 64).double().reshape(8, 8)
        err = (got - ref_a @ ref_a.t()).abs().max().item()
        self.assertGreater(
            err,
            1e-12,
            "float64 matmul now matches the host: widen "
            "TpuPlatform.is_float64_supported's comment and re-check minimax_h3",
        )


if __name__ == "__main__":
    unittest.main()
