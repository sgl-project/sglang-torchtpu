import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    _eager_temb_table_slices,
    _wan_temb_table_slices,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


class TestWanTembTableSlices(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and current_platform.is_cuda(),
        "requires NVIDIA CUDA fusion",
    )
    def test_fused_matches_eager_and_is_contiguous(self):
        torch.manual_seed(0)
        for batch, seq, hidden in [(1, 517, 3072), (2, 64, 1536)]:
            temb = torch.randn(
                batch, seq, 6, hidden, device="cuda", dtype=torch.bfloat16
            )
            table = torch.randn(1, 6, hidden, device="cuda", dtype=torch.float32)
            reference = _eager_temb_table_slices(table, temb)
            fused = _wan_temb_table_slices(table, temb)
            self.assertEqual(len(fused), 6)
            for ref, out in zip(reference, fused):
                self.assertTrue(torch.equal(ref, out))
                self.assertTrue(out.is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp32_temb_also_supported(self):
        torch.manual_seed(1)
        temb = torch.randn(1, 33, 6, 512, device="cuda", dtype=torch.float32)
        table = torch.randn(1, 6, 512, device="cuda", dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        fused = _wan_temb_table_slices(table, temb)
        for ref, out in zip(reference, fused):
            self.assertTrue(torch.equal(ref, out))

    def test_eager_slices_are_contiguous_and_match_chunks(self):
        torch.manual_seed(3)
        temb = torch.randn(2, 9, 6, 64, dtype=torch.bfloat16)
        table = torch.randn(1, 6, 64, dtype=torch.float32)
        chunks = (table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        slices = _eager_temb_table_slices(table, temb)
        self.assertEqual(len(slices), 6)
        for chunk, out in zip(chunks, slices):
            self.assertTrue(torch.equal(chunk.squeeze(2), out))
            self.assertTrue(out.is_contiguous())

    def test_eager_slices_of_sp_views_and_fp32_temb(self):
        torch.manual_seed(4)
        batch, sp, local, hidden = 2, 3, 5, 64
        table = torch.randn(1, 6, hidden, dtype=torch.float32)
        for dtype in (torch.bfloat16, torch.float32):
            base = torch.randn(batch, sp * local, 6, hidden, dtype=dtype)
            # The DiT's per-rank shard: a non-contiguous view when batch > 1.
            temb = base.view(batch, sp, local, 6, hidden)[:, 1]
            with self.subTest(dtype=dtype):
                self.assertFalse(temb.is_contiguous())
                chunks = (table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
                for chunk, out in zip(chunks, _eager_temb_table_slices(table, temb)):
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertTrue(out.is_contiguous())
                    self.assertTrue(torch.equal(chunk.squeeze(2), out))

    def test_eager_slices_are_fp32_for_a_bf16_table(self):
        # The loader casts scale_shift_table to the DiT dtype, so serving sees bf16.
        torch.manual_seed(5)
        temb = torch.randn(2, 9, 6, 64, dtype=torch.bfloat16)
        table = torch.randn(1, 6, 64, dtype=torch.bfloat16)
        chunks = (table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        for chunk, out in zip(chunks, _eager_temb_table_slices(table, temb)):
            self.assertEqual(out.dtype, torch.float32)
            self.assertTrue(out.is_contiguous())
            self.assertTrue(torch.equal(chunk.squeeze(2), out))

    def test_cpu_falls_back_to_eager(self):
        torch.manual_seed(2)
        temb = torch.randn(1, 9, 6, 64, dtype=torch.bfloat16)
        table = torch.randn(1, 6, 64, dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        out = _wan_temb_table_slices(table, temb)
        for ref, got in zip(reference, out):
            self.assertTrue(torch.equal(ref, got))


if __name__ == "__main__":
    unittest.main()
