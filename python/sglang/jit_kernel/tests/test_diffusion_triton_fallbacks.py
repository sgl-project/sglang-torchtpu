"""CPU tests for the torch-native fallbacks of the diffusion triton kernels.

The fallbacks serve backends without triton (TPU, MPS, CPU); they must match
the fused-kernel math exactly.
"""

import unittest

import torch

from sglang.jit_kernel.diffusion.triton.rotary import apply_rotary_embedding
from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel


def _reference_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x1 = x[..., 0::2].double()
    x2 = x[..., 1::2].double()
    cos = cos.double()[:, None, :]
    sin = sin.double()[:, None, :]
    out = torch.empty_like(x, dtype=torch.double)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(x.dtype)


class TestRotaryFallback(unittest.TestCase):
    def test_matches_reference_3d(self):
        torch.manual_seed(0)
        num_tokens, num_heads, head_size = 6, 2, 8
        x = torch.randn(num_tokens, num_heads, head_size)
        cos = torch.randn(num_tokens, head_size // 2)
        sin = torch.randn(num_tokens, head_size // 2)
        torch.testing.assert_close(
            apply_rotary_embedding(x, cos, sin), _reference_rotary(x, cos, sin)
        )

    def test_matches_reference_4d(self):
        torch.manual_seed(1)
        bsz, num_tokens, num_heads, head_size = 2, 5, 3, 4
        x = torch.randn(bsz, num_tokens, num_heads, head_size)
        cos = torch.randn(num_tokens, head_size // 2)
        sin = torch.randn(num_tokens, head_size // 2)
        torch.testing.assert_close(
            apply_rotary_embedding(x, cos, sin), _reference_rotary(x, cos, sin)
        )

    def test_interleaved_full_width_mask(self):
        # With interleaved=True and full-head-size cos/sin, the kernel uses the
        # even-indexed columns; the fallback must do the same.
        torch.manual_seed(2)
        num_tokens, num_heads, head_size = 4, 1, 6
        x = torch.randn(num_tokens, num_heads, head_size)
        cos_full = torch.randn(num_tokens, head_size)
        sin_full = torch.randn(num_tokens, head_size)
        expected = _reference_rotary(x, cos_full[:, ::2], sin_full[:, ::2])
        torch.testing.assert_close(
            apply_rotary_embedding(x, cos_full, sin_full, interleaved=True), expected
        )


class TestScaleShiftFallback(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.B, self.F, self.C = 2, 3, 4
        self.L = self.F * 5  # tokens = frames * tokens_per_frame
        self.x = torch.randn(self.B, self.L, self.C)

    def test_per_batch_scale_shift(self):
        scale = torch.randn(self.B, self.C)
        shift = torch.randn(self.B, self.C)
        result = fuse_scale_shift_kernel(self.x, scale, shift, scale_constant=1.0)
        expected = self.x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        torch.testing.assert_close(result, expected)

    def test_per_frame_scale_shift(self):
        scale = torch.randn(self.B, self.F, 1, self.C)
        shift = torch.randn(self.B, self.F, 1, self.C)
        result = fuse_scale_shift_kernel(self.x, scale, shift, scale_constant=0.0)
        per_token = self.L // self.F
        scale_tok = scale.expand(self.B, self.F, per_token, self.C).reshape(
            self.B, self.L, self.C
        )
        shift_tok = shift.expand(self.B, self.F, per_token, self.C).reshape(
            self.B, self.L, self.C
        )
        torch.testing.assert_close(result, self.x * scale_tok + shift_tok)

    def test_full_tensor_scale_shift(self):
        scale = torch.randn(self.B, self.L, self.C)
        shift = torch.randn(self.B, self.L, self.C)
        result = fuse_scale_shift_kernel(self.x, scale, shift, scale_constant=1.0)
        torch.testing.assert_close(result, self.x * (1.0 + scale) + shift)


if __name__ == "__main__":
    unittest.main()
