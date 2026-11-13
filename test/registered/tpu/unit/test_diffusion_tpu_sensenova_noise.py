# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1's starting noise on TPU workers (sglang_tpu.multimodal_gen.sensenova_u1)."""

import importlib.util
import unittest

import torch

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify import (
    modeling_neo_chat,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import sensenova_u1

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

SHAPE = (1, 3, 256, 256)


def _host_draw(seed, shape=SHAPE):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


class TestSenseNovaNoiseTpu(unittest.TestCase):
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
        self._stock = modeling_neo_chat._randn_with_seed
        sensenova_u1.install()
        self.addCleanup(setattr, modeling_neo_chat, "_randn_with_seed", self._stock)

    def test_tpu_noise_is_the_host_float32_draw_rounded(self):
        """torch_tpu draws a bfloat16 normal from 7 random bits (128 values, all
        within [-2.89, 2.52]); SenseNova rendered near-black images from it."""
        noise = modeling_neo_chat._randn_with_seed(
            SHAPE, device="tpu", dtype=torch.bfloat16, seed=42
        )
        self.assertEqual(noise.device.type, "tpu")
        self.assertEqual(noise.dtype, torch.bfloat16)
        noise = noise.cpu()
        self.assertTrue(torch.equal(noise, _host_draw(42).bfloat16()))
        self.assertGreater(noise.unique().numel(), 1000)
        self.assertGreater(noise.float().abs().max().item(), 3.5)

    def test_list_seeds_draw_one_row_per_seed(self):
        shape = (2, *SHAPE[1:])
        noise = modeling_neo_chat._randn_with_seed(
            shape, device="tpu", dtype=torch.bfloat16, seed=[42, 7]
        ).cpu()
        expected = torch.cat([_host_draw(42), _host_draw(7)]).bfloat16()
        self.assertTrue(torch.equal(noise, expected))

    def test_cpu_keeps_the_stock_draw(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                patched = modeling_neo_chat._randn_with_seed(
                    SHAPE, device="cpu", dtype=dtype, seed=42
                )
                stock = self._stock(SHAPE, device="cpu", dtype=dtype, seed=42)
                self.assertTrue(torch.equal(patched, stock))


if __name__ == "__main__":
    unittest.main()
