"""FusedRMSNormGated torch fallback on TPU."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla import fused_norm_gate
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference(x, g, weight, eps, activation):
    x_f = x.float()
    y = x_f * torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    y = y * weight.float()
    gate = torch.sigmoid(g.float())
    if activation in ("swish", "silu"):
        gate = g.float() * gate
    return (y * gate).to(x.dtype)


def _norm(activation, dtype=torch.float32):
    torch.manual_seed(0)
    norm = FusedRMSNormGated(64, eps=1e-5, activation=activation)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    return norm.to(dtype)


class TestFusedRMSNormGatedFallback(CustomTestCase):
    def test_tpu_uses_torch_path(self):
        """TPU has no Triton; it must not reach the Triton launcher."""
        x, g = torch.randn(3, 64), torch.randn(3, 64)
        for activation in ("swish", "silu", "sigmoid"):
            norm = _norm(activation)
            with patch.object(fused_norm_gate, "_is_tpu", True):
                out = norm(x, g)
            torch.testing.assert_close(
                out, _reference(x, g, norm.weight, norm.eps, activation)
            )

    def test_prenorm_residual_out_follows_triton(self):
        """residual_out is the pre-norm stream, in layer_norm_gated_fwd's dtype."""
        x, g, residual = (torch.randn(3, 64, dtype=torch.bfloat16) for _ in range(3))
        norm = _norm("silu", torch.bfloat16)
        with patch.object(fused_norm_gate, "_is_tpu", True):
            _, with_residual = norm(
                x, g, residual=residual, prenorm=True, residual_in_fp32=True
            )
            _, fp32_no_residual = norm(x, g, prenorm=True, residual_in_fp32=True)
            _, plain = norm(x, g, prenorm=True)
        self.assertEqual(with_residual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            with_residual, (x.float() + residual.float()).bfloat16()
        )
        self.assertEqual(fp32_no_residual.dtype, torch.float32)
        torch.testing.assert_close(fp32_no_residual, x.float())
        torch.testing.assert_close(plain, x)


if __name__ == "__main__":
    unittest.main()
