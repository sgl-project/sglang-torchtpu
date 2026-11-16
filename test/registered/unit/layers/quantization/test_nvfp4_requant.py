"""NVFP4 expert requantization to block-256 scales (SGLANG_TPU_MUSE_MK_QBLOCK)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.moe.nvfp4_requant import requant_nvfp4_blocks

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _decode(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], -1
    )
    v = E2M1[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    return v * scale.float().repeat_interleave(packed.shape[1] // scale.shape[1], dim=1)


def _reference(packed: torch.Tensor, scale: torch.Tensor, block: int):
    """Nearest e2m1 by argmin over the magnitudes, ties to the even code."""
    w = _decode(packed, scale)
    e, k, n = w.shape
    wb = w.reshape(e, k // block, block, n)
    sb = (wb.abs().amax(dim=2) / 6.0).to(torch.bfloat16)
    q = (wb / torch.where(sb.float() > 0, sb.float(), 1.0)[:, :, None]).clamp(-6, 6)
    d = (q.abs()[..., None] - E2M1).abs()
    near = d == d.amin(dim=-1, keepdim=True)
    idx = torch.where(near, torch.arange(8) % 2, 2).argmin(dim=-1)
    codes = (idx | torch.where(q < 0, 8, 0)).to(torch.uint8).reshape(e, k, n // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), sb


class TestNvfp4Requant(CustomTestCase):
    def _random(self, e=3, k=512, n=256, seed=0):
        g = torch.Generator().manual_seed(seed)
        packed = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, generator=g)
        scale = (torch.rand(e, k // 16, n, generator=g) + 0.25).to(torch.bfloat16)
        return packed, scale

    def test_matches_argmin_reference(self):
        for seed in range(3):
            packed, scale = self._random(seed=seed)
            codes, sb = requant_nvfp4_blocks(packed, scale, 256)
            ref_codes, ref_sb = _reference(packed, scale, 256)
            self.assertEqual(sb.shape, (3, 2, 256))
            self.assertTrue(torch.equal(sb, ref_sb))
            self.assertTrue(torch.equal(codes, ref_codes))

    def test_one_scale_per_block_is_lossless(self):
        """Blocks whose 16 scales already agree (and reach the e2m1 max) requantize to
        themselves."""
        packed, _ = self._random(seed=7)
        packed[:, ::256, :] = (
            0x77  # +6 in both nibbles: every column's block amax is 6 s
        )
        scale = torch.full((3, 512 // 16, 256), 0.375, dtype=torch.bfloat16)
        codes, sb = requant_nvfp4_blocks(packed, scale, 256)
        self.assertTrue(torch.equal(_decode(codes, sb), _decode(packed, scale)))

    def test_ties_round_to_even(self):
        packed = torch.zeros(1, 256, 1, dtype=torch.uint8)
        packed[0, 0, 0] = (
            0x77  # column 0 and 1 amax 6 -> block scale = the stored scale
        )
        scale = torch.ones(1, 16, 2, dtype=torch.bfloat16)
        scale[0, 1] = 0.5  # rows 16..31: 0.5 x 1.5 = 0.75 and 0.5 x 5 is not a code
        packed[0, 16, 0] = 0x33  # 1.5 x 0.5 = 0.75: a tie between 0.5 and 1.0 -> 1.0
        packed[0, 17, 0] = 0x11  # 0.5 x 0.5 = 0.25: a tie between 0 and 0.5 -> 0
        codes, _ = requant_nvfp4_blocks(packed, scale, 256)
        self.assertEqual(int(codes[0, 16, 0]), 0x22)
        self.assertEqual(int(codes[0, 17, 0]), 0x00)


if __name__ == "__main__":
    unittest.main()
