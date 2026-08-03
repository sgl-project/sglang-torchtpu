"""Equivalence check for Kimi-K2.5's real-arithmetic MoonViT RoPE path.

The TPU compile path feeds MoonViT's RoPE as a real ``(cos, sin)`` tuple instead
of the eager complex64 ``freqs_cis`` (the TPU backend cannot lower
``torch.view_as_complex`` / complex multiply). This test pins the real path
(``apply_rope`` tuple branch, via ``_rotate_interleaved``) to be identical to the
complex multiply it replaces.

CPU-only. Run:
    python test/manual/tpu/test_kimi_rope_real.py
"""

import unittest

import torch

from sglang.srt.models.kimi_k25 import apply_rope, freqs_cis_to_cos_sin


def _reference_complex(xq, xk, freqs_cis):
    """The original complex-multiply RoPE (eager path)."""
    fc = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_ * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class TestKimiRealRope(unittest.TestCase):
    def _run(self, dtype):
        torch.manual_seed(0)
        seq, heads, head_dim = 300, 8, 64
        xq = torch.randn(seq, heads, head_dim, dtype=dtype)
        xk = torch.randn(seq, heads, head_dim, dtype=dtype)
        angles = torch.randn(seq, head_dim // 2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64

        ref_q, ref_k = _reference_complex(xq, xk, freqs_cis)
        cos, sin = freqs_cis_to_cos_sin(freqs_cis)
        got_q, got_k = apply_rope(xq, xk, (cos, sin))

        self.assertEqual(got_q.dtype, dtype)
        torch.testing.assert_close(got_q.float(), ref_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(got_k.float(), ref_k.float(), rtol=0, atol=0)

    def test_float32(self):
        self._run(torch.float32)

    def test_bfloat16(self):
        self._run(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
