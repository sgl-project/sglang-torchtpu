# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 native KV cache: fp8 codes plus scales read back to the bf16 the reference's in-place
quantizers write, and the loc-indexed layout that lets a radix hit read its donor's rows."""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

from sglang_tpu.dsv41_native.kernel_ref import act_quant, fp4_act_quant
from sglang_tpu.dsv41_native.kv_layout import (
    check_page_layout,
    group_tail_positions,
    num_locs,
    prefix_loc_writes,
)
from sglang_tpu.dsv41_native.kv_quant import (
    CODE_DTYPE,
    dequant,
    quantize_fp4_e4m3,
    quantize_fp4_e8m0,
    quantize_fp8_ue8m0,
)


def _inputs(dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(16, 512) * torch.logspace(-6, 3, 16).unsqueeze(-1)
    x[0] = 0  # amax floor
    x[1, :32] = 448.0  # fp8 saturation edge
    x[2, :32] = -1e4  # clamped past fp8 max
    x[3] = 1e-7  # tiny blocks
    x[4, ::2] = -x[4, ::2].abs()
    return x.to(dtype)


class TestKvQuant(unittest.TestCase):
    def _check(self, quantize, inplace, block, scale_dtype):
        for dtype in (torch.bfloat16, torch.float32):
            x = _inputs(dtype)
            q, s = quantize(x, block)
            self.assertEqual(q.dtype, CODE_DTYPE)
            self.assertEqual(s.dtype, scale_dtype)
            self.assertEqual(tuple(s.shape), (16, 512 // block))
            want = inplace(x.clone(), block)
            got = dequant(q, s, dtype)
            # A block whose amax / 6 exceeds the e4m3 max gets a NaN E4M3 scale in both forms
            # (the cast does not saturate). The two forms can produce different NaN bit patterns
            # (seen on the x86 pod, not on an ARM host), so NaN positions must match and the rest
            # must match bit for bit, which still catches -0.0 vs 0.0.
            self.assertTrue(
                torch.equal(got.isnan(), want.isnan()), f"{quantize.__name__} {dtype}"
            )
            bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
            keep = ~want.isnan()
            self.assertTrue(
                torch.equal(got.view(bits)[keep], want.view(bits)[keep]),
                f"{quantize.__name__} {dtype}",
            )

    def test_window_kv_matches_act_quant(self):
        self._check(
            quantize_fp8_ue8m0,
            lambda x, b: act_quant(x, b, "ue8m0", torch.float32, True),
            32,
            torch.uint8,
        )

    def test_index_k_matches_fp4_e8m0(self):
        self._check(
            quantize_fp4_e8m0, lambda x, b: fp4_act_quant(x, b, True), 32, torch.uint8
        )

    def test_compressed_kv_matches_fp4_e4m3(self):
        self._check(
            quantize_fp4_e4m3,
            lambda x, b: fp4_act_quant(x, b, True, scale_dtype=torch.float8_e4m3fn),
            16,
            torch.float8_e4m3fn,
        )


class TestKvLayout(unittest.TestCase):
    def test_num_locs_adds_padding_page(self):
        self.assertEqual(num_locs(max_total_tokens=4096, page_size=64), 4096 + 64)
        # a partial page rounds up: 4000 -> 63 pages (4032), then the padding page
        self.assertEqual(num_locs(max_total_tokens=4000, page_size=64), 4032 + 64)

    def test_page_must_hold_whole_groups(self):
        check_page_layout(page_size=64, ratios=[0, 1, 2])
        with self.assertRaises(AssertionError):
            check_page_layout(page_size=64, ratios=[0, 3])

    def test_group_rows_are_distinct_within_a_page(self):
        # positions 0..63 of one request on page 5: groups of ratio 2 map to 32 distinct rows
        locs = torch.arange(5 * 64, 6 * 64)
        tails = group_tail_positions(torch.arange(32), 2)
        rows = locs[tails] // 2
        self.assertEqual(rows.unique().numel(), 32)
        self.assertTrue((rows >= 5 * 32).all() and (rows < 6 * 32).all())

    def test_prefix_loc_writes_copy_req_to_token(self):
        req_to_token = torch.zeros(4, 256, dtype=torch.int32)
        req_to_token[2, :128] = torch.arange(640, 768)  # donor prefix on pages 10..11
        req_to_token[3, :64] = torch.arange(1024, 1088)
        rows, cols, locs = prefix_loc_writes(
            req_to_token=req_to_token, slots=[2, 1, 3], prefix_lens=[128, 0, 64]
        )
        self.assertEqual(rows.numel(), 1024)  # bucketed
        n = 128 + 64
        self.assertTrue((rows[:128] == 2).all() and (rows[128:n] == 3).all())
        self.assertTrue(torch.equal(cols[:128], torch.arange(128, dtype=torch.int32)))
        self.assertTrue(
            torch.equal(locs[:128], torch.arange(640, 768, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(locs[128:n], torch.arange(1024, 1088, dtype=torch.int32))
        )
        # padding writes land on (slot 0, position 0) with location 0
        self.assertTrue(
            (rows[n:] == 0).all() and (cols[n:] == 0).all() and (locs[n:] == 0).all()
        )


if __name__ == "__main__":
    unittest.main()
