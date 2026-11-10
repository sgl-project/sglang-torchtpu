# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 native KV cache: fp8 codes plus scales, and the loc-indexed layout that lets a radix hit
read its donor's rows.

Window KV must read back the reference's bf16 bit for bit. Index K and CSA KV follow the production
TPU quantizers (fp8; the reference uses fp4), so they are checked against those JAX references."""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import numpy as np

from sglang_tpu.dsv41_native.kernel_ref import act_quant
from sglang_tpu.dsv41_native.kv_layout import (
    check_page_layout,
    group_tail_positions,
    num_locs,
    prefix_loc_writes,
)
from sglang_tpu.dsv41_native.kv_quant import (
    CODE_DTYPE,
    CSA_ROPE_DIM,
    CSA_SCALE_PERIOD,
    INDEX_K_BLOCK,
    dequant,
    dequant_lane_periodic,
    quantize_fp8_lane_periodic,
    quantize_fp8_ue8m0,
)


def _bytes(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint8).numpy()


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

    def test_index_k_matches_production_quantizer(self):
        """One UE8M0 scale per 128-dim vector, byte for byte as the production index cache."""
        from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
            quantize_index_rows,
        )

        x = _inputs(torch.float32).reshape(64, 128)
        q, s = quantize_fp8_ue8m0(x, INDEX_K_BLOCK)
        self.assertEqual(tuple(s.shape), (64, 1))
        want_q, _, want_s = quantize_index_rows(x.numpy())
        np.testing.assert_array_equal(_bytes(q), np.asarray(want_q).view(np.uint8))
        np.testing.assert_array_equal(s[:, 0].numpy(), np.asarray(want_s))

    def test_csa_kv_matches_production_lane_periodic(self):
        """448 fp8 nope dims, scale j over dims d % 64 == j, byte for byte as the production CSA
        store; the 64 RoPE dims pass through in bf16."""
        # Load the JAX reference by path: the sglang_tpu.dsv4 package __init__ imports the
        # attention backend, which this pure quantizer check does not need.
        import importlib.util
        from pathlib import Path

        import sglang_tpu

        path = (
            Path(sglang_tpu.__file__).parent
            / "dsv4/compress_and_store/compress_store_ref.py"
        )
        spec = importlib.util.spec_from_file_location("_csa_store_ref", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.quantize_fp8_lane_periodic

        x = _inputs(torch.float32)
        nope = 512 - CSA_ROPE_DIM
        q, s, rope = quantize_fp8_lane_periodic(x)
        want_q, want_s = reference(x.numpy(), CSA_SCALE_PERIOD, nope)
        np.testing.assert_array_equal(
            _bytes(q), np.asarray(want_q).view(np.uint8)[:, :nope]
        )
        np.testing.assert_array_equal(_bytes(s), np.asarray(want_s).view(np.uint8))
        self.assertTrue(torch.equal(rope, x[:, nope:].to(torch.bfloat16)))

    def test_csa_dequant_tiles_scales_by_lane(self):
        """A large value in one lane must not coarsen the other lanes: dequant has to apply scale
        d % 64, not a contiguous block of 7, or small lanes come back far off."""
        x = torch.full((1, 512), 0.01)
        x[0, 5] = x[0, 5 + 64] = 400.0  # lane 5 only
        q, s, rope = quantize_fp8_lane_periodic(x)
        got = dequant_lane_periodic(q, s, rope, torch.float32)
        rel = ((got - x).abs() / x.abs())[0, : 512 - CSA_ROPE_DIM]
        self.assertLess(rel.max().item(), 2**-3)  # e4m3 keeps 3 mantissa bits


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
