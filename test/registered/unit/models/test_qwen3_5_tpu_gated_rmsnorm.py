"""Unit tests for the TPU pure-torch gated RMSNorm used by Qwen3.5 GDN.

The pure-torch ``RMSNormGated`` replaces FLA's Triton ``RMSNormGated`` on TPU
(the Triton kernel queries CUDA device properties). It is pure torch, so these
run on CPU and pin the gating math + the activation guard that a silent-identity
bug would otherwise slip through.

Imported directly from its module (not via ``qwen3_5``): on a CPU/CI host
``_is_tpu()`` is False, so ``qwen3_5`` would bind ``RMSNormGated`` to FLA's
CUDA-only class instead of the pure-torch one under test.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.attention.native_torch.layernorm_gated import RMSNormGated
from sglang.test.test_utils import CustomTestCase


def _ref(x, weight, z, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.float()
    if z is not None:
        out = out * torch.nn.functional.silu(z.float())
    return out.to(x.dtype)


class TestTpuGatedRMSNorm(CustomTestCase):
    H = 128
    EPS = 1e-6

    def _norm(self, activation="silu"):
        norm = RMSNormGated(self.H, eps=self.EPS, activation=activation)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(self.H))
        return norm

    def test_silu_gated_matches_reference(self):
        torch.manual_seed(0)
        norm = self._norm("silu")
        x = torch.randn(4, self.H)
        z = torch.randn(4, self.H)
        out = norm(x, z)
        exp = _ref(x, norm.weight, z, self.EPS)
        torch.testing.assert_close(out, exp, rtol=1e-5, atol=1e-5)
        self.assertEqual(out.dtype, x.dtype)

    def test_no_gate_when_z_none(self):
        norm = self._norm("silu")
        x = torch.randn(2, self.H)
        out = norm(x, None)
        exp = _ref(x, norm.weight, None, self.EPS)
        torch.testing.assert_close(out, exp, rtol=1e-5, atol=1e-5)

    def test_unsupported_activation_raises(self):
        # A non-silu gate must fail loudly, not silently drop the gate.
        with self.assertRaises(NotImplementedError):
            RMSNormGated(self.H, activation="gelu")

    def test_bf16_dtype_preserved(self):
        norm = RMSNormGated(self.H, eps=self.EPS, dtype=torch.bfloat16)
        x = torch.randn(2, self.H, dtype=torch.bfloat16)
        z = torch.randn(2, self.H, dtype=torch.bfloat16)
        out = norm(x, z)
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
