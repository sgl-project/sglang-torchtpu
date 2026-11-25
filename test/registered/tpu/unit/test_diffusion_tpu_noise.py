# SPDX-License-Identifier: Apache-2.0
"""Random normals on TPU workers (sglang_tpu.multimodal_gen.noise)."""

import importlib.util
import unittest

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import noise

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

# A FLUX.2 1024x1024 latent.
SHAPE = (1, 128, 64, 64)


def _generator(device, seed=0):
    return torch.Generator(device).manual_seed(seed)


class TestRandnTpu(unittest.TestCase):
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

    def setUp(self):
        self.stock = getattr(torch.randn, "stock", torch.randn)
        noise.install()
        self.addCleanup(setattr, torch, "randn", self.stock)

    def _float32_draw_rounded(self, generator, dtype):
        return (
            self.stock(SHAPE, generator=generator, device="tpu", dtype=torch.float32)
            .to(dtype)
            .cpu()
        )

    def test_low_precision_tpu_draw_is_the_float32_draw_rounded(self):
        """torch_tpu draws a bfloat16 normal from 7 random bits (128 values, all
        within [-2.89, 2.52]) and a float16 normal from 10 (1024 values)."""
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                got = torch.randn(
                    SHAPE, generator=_generator("tpu"), device="tpu", dtype=dtype
                )
                self.assertEqual(got.dtype, dtype)
                got = got.cpu()
                want = self._float32_draw_rounded(_generator("tpu"), dtype)
                self.assertTrue(torch.equal(got, want))
                self.assertGreater(got.unique().numel(), 2000)
                self.assertGreater(got.float().abs().max().item(), 4.0)

    def test_latent_stage_draws_round_device_draws_and_keep_host_draws(self):
        # The latent stages call randn_tensor with the request's generators: TPU
        # generators under the HTTP default, CPU ones for generator_device "cpu".
        latents = randn_tensor(
            SHAPE, generator=[_generator("tpu")], device="tpu", dtype=torch.bfloat16
        )
        want = self._float32_draw_rounded(_generator("tpu"), torch.bfloat16)
        self.assertTrue(torch.equal(latents.cpu(), want))
        latents = randn_tensor(
            SHAPE, generator=_generator("cpu"), device="tpu", dtype=torch.bfloat16
        )
        want = self.stock(SHAPE, generator=_generator("cpu"), dtype=torch.bfloat16)
        self.assertTrue(torch.equal(latents.cpu(), want))

    def test_install_wraps_once_and_keeps_the_other_call_forms(self):
        randn = torch.randn
        self.assertIs(randn.stock, self.stock)
        noise.install()
        self.assertIs(torch.randn, randn)
        out = torch.empty(SHAPE, dtype=torch.bfloat16, device="tpu")
        got = torch.randn(SHAPE, out=out, dtype=torch.bfloat16, device="tpu")
        self.assertIs(got, out)
        leaf = torch.randn(4, dtype=torch.bfloat16, device="tpu", requires_grad=True)
        self.assertTrue(leaf.is_leaf)
        # Model init under ``with torch.device("meta")`` still lands on meta.
        with torch.device("meta"):
            self.assertEqual(torch.randn(4).device.type, "meta")


if __name__ == "__main__":
    unittest.main()
