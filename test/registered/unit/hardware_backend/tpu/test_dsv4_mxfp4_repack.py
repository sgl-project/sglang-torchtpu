# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4 MXFP4 -> TPU FP4 expert repack.

The repack runs on the device that holds the experts, gathers the FP4 LUT
through a flattened index, and fills preallocated outputs chunk by chunk.
Guards that it is bit-identical to the reference conversion.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import dsv4_mxfp4_moe as m


def _ref_dequant(packed, scale):
    lut = torch.tensor(m._FP4_E2M1_LUT)
    lo, hi = (packed & 0xF).long(), ((packed >> 4) & 0xF).long()
    w = torch.stack((lut[lo], lut[hi]), -1).flatten(-2)
    s = torch.exp2(scale.int().float() - 127)
    return (w.unflatten(-1, (-1, m._MXFP4_BLOCK_SIZE)) * s.unsqueeze(-1)).flatten(-2)


def _ref_quant(w):
    e, n, k = w.shape
    th = torch.tensor(m._FP4_E2M1_THRESHOLDS)
    wb = w.reshape(e, n, k // m._TPU_FP4_BLOCK_SIZE, m._TPU_FP4_BLOCK_SIZE)
    bs = wb.abs().amax(-1, keepdim=True) / m._FP4_MAX
    inv = torch.where((bs == 0) | ~torch.isfinite(bs), torch.zeros_like(bs), 1.0 / bs)
    ws = (wb * inv).clamp(-m._FP4_MAX, m._FP4_MAX)
    mag = torch.bucketize(ws.abs(), th, out_int32=True, right=False).to(torch.uint8)
    idx = (mag | (((ws < 0) & (mag > 0)).to(torch.uint8) << 3)).reshape(e, n, k)
    packed = (idx[..., 0::2] | (idx[..., 1::2] << 4)).permute(0, 2, 1).contiguous()
    return packed, bs.squeeze(-1).permute(0, 2, 1).unsqueeze(2)


class TestMxfp4Repack(CustomTestCase):
    def test_dequant_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        packed = torch.randint(0, 256, (3, 8, 512), dtype=torch.uint8, generator=g)
        scale = torch.randint(110, 135, (3, 8, 32), dtype=torch.uint8, generator=g)
        self.assertTrue(
            torch.equal(
                m._dequantize_dsv4_mxfp4_block32(packed, scale),
                _ref_dequant(packed, scale),
            )
        )


class TestConvertExperts(CustomTestCase):
    def test_chunked_preallocated_output_matches_reference(self):
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            _pad_gate_up_halves_axis1,
            _pad_last_axis,
        )

        g = torch.Generator().manual_seed(2)
        experts = m._MXFP4_EXPERTS_PER_CHUNK + 3  # a partial last chunk
        inter, hidden = 512, 512
        w13 = torch.randint(
            0, 256, (experts, 2 * inter, hidden // 2), dtype=torch.uint8, generator=g
        )
        w13_s = torch.randint(
            118, 130, (experts, 2 * inter, hidden // 32), dtype=torch.uint8, generator=g
        )
        w2 = torch.randint(
            0, 256, (experts, hidden, inter // 2), dtype=torch.uint8, generator=g
        )
        w2_s = torch.randint(
            118, 130, (experts, hidden, inter // 32), dtype=torch.uint8, generator=g
        )
        got = m._convert_mxfp4_experts(
            w13, w13_s, w2, w2_s, _pad_gate_up_halves_axis1, _pad_last_axis
        )
        want13 = _ref_quant(
            _pad_gate_up_halves_axis1(_ref_dequant(w13, w13_s), inter, 0)
        )
        want2 = _ref_quant(_pad_last_axis(_ref_dequant(w2, w2_s), 0))
        for a, b in zip(got, (*want13, *want2)):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
