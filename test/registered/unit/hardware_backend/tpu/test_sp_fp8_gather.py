"""Unit tests for the FP8 all-gather of the sequence-parallel stage input.

hooks/sp_fp8_gather.py replaces the bf16 all-gather at the end of
layer_boundary.prepare._tp_reduce_scatter_update_read_gather with 1-byte FP8 rows
plus an fp32 per-row scale. These CPU tests pin the rank-block layout of the
gathered rows, the quantization error bound, that FP8 consumers re-quantizing
the dequantized rows get back (nearly) the same bytes, and the env gate.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.fp8_linear import _quantize_activation_per_row
from sglang_tpu.hooks.sp_fp8_gather import (
    fp8_gather_rows,
    quantize_rows_fp8,
    sp_gather_stage_input,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


def _fake_group(shards: list[torch.Tensor]):
    """An all_gather_into_tensor over ``len(shards)`` ranks whose inputs are the
    FP8 bytes / scales of ``shards``, in rank order."""
    quantized = [quantize_rows_fp8(s) for s in shards]

    def all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor) -> None:
        if input.dtype == torch.uint8:
            parts = [q.view(torch.uint8) for q, _ in quantized]
        else:
            parts = [scale for _, scale in quantized]
        assert parts[0].shape == input.shape
        output.copy_(torch.cat(parts, dim=0))

    return all_gather_into_tensor


class TestSpFp8Gather(CustomTestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        # Rows spanning very different magnitudes, as normed stage inputs do not,
        # but a per-tensor scale would break on.
        self.shards = [
            (torch.randn(4, 256, generator=generator) * (10.0 ** (r - 1))).to(
                torch.bfloat16
            )
            for r in range(4)
        ]

    def test_quantizer_matches_fp8_linear_activation_quantizer(self):
        x = self.shards[2]
        q, scale = quantize_rows_fp8(x)
        q_ref, scale_ref = _quantize_activation_per_row(x, FP8)
        torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
        self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))

    def test_gathers_rank_blocks_in_order_within_fp8_error(self):
        out = torch.empty(16, 256, dtype=torch.bfloat16)
        fp8_gather_rows(out, self.shards[1], _fake_group(self.shards))
        expected = torch.cat(self.shards, dim=0).float()
        # e4m3 keeps 3 mantissa bits: |x - deq| <= amax(row) / 16 (half an ulp at
        # the top binade, plus bf16 rounding of the dequantized value).
        row_amax = expected.abs().amax(dim=1, keepdim=True)
        err = (out.float() - expected).abs()
        self.assertTrue(bool((err <= row_amax / 16 + 1e-30).all()))
        # Each row block came from the matching rank's shard.
        for r, shard in enumerate(self.shards):
            block = out[4 * r : 4 * (r + 1)].float()
            rel = (block - shard.float()).norm() / shard.float().norm()
            self.assertLess(rel.item(), 0.05)

    def test_requantizing_dequantized_rows_recovers_bytes(self):
        out = torch.empty(16, 256, dtype=torch.bfloat16)
        fp8_gather_rows(out, self.shards[0], _fake_group(self.shards))
        q_sent = torch.cat([quantize_rows_fp8(s)[0] for s in self.shards], dim=0)
        q_again, _ = _quantize_activation_per_row(out, FP8)
        same = (q_again.view(torch.uint8) == q_sent.view(torch.uint8)).float().mean()
        # bf16 rounding of q * scale can move a value across an fp8 rounding
        # boundary on re-quantization; that must stay rare.
        self.assertGreater(same.item(), 0.99)

    def test_flag_off_calls_original(self):
        original = mock.Mock()
        out = torch.empty(4, 256, dtype=torch.bfloat16)
        with envs.SGLANG_TPU_SP_FP8_ALL_GATHER.override(False):
            sp_gather_stage_input(original, out, self.shards[0])
        original.assert_called_once_with(out, self.shards[0])

    def test_flag_on_empty_shard_calls_original(self):
        original = mock.Mock()
        empty = torch.empty(0, 256, dtype=torch.bfloat16)
        with envs.SGLANG_TPU_SP_FP8_ALL_GATHER.override(True):
            sp_gather_stage_input(original, empty, empty)
        original.assert_called_once()

    def test_flag_on_uses_fp8_gather(self):
        original = mock.Mock()
        out = torch.empty(4, 256, dtype=torch.bfloat16)
        with (
            envs.SGLANG_TPU_SP_FP8_ALL_GATHER.override(True),
            mock.patch(
                "sglang.srt.layers.dp_attention.attn_tp_all_gather_into_tensor",
                _fake_group([self.shards[3]]),
            ),
        ):
            sp_gather_stage_input(original, out, self.shards[3])
        original.assert_not_called()
        rel = (out.float() - self.shards[3].float()).norm() / self.shards[
            3
        ].float().norm()
        self.assertLess(rel.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
