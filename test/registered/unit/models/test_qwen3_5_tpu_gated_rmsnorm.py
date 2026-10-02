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
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_qwen38_swish_gate_matches_reference(self):
        """The original Qwen3.8-27B config names its SiLU gate 'swish'."""
        torch.manual_seed(0)
        norm = self._norm("swish").to(torch.bfloat16)
        x = torch.randn(4, self.H, dtype=torch.bfloat16)
        z = torch.randn_like(x)
        expected = _ref(x, norm.weight, z, self.EPS)
        torch.testing.assert_close(norm(x, z), expected)

    def test_unsupported_activation_raises(self):
        # A non-silu gate must fail loudly, not silently drop the gate.
        with self.assertRaises(NotImplementedError):
            RMSNormGated(self.H, activation="gelu")

    def test_tpu_attention_gate_without_npu_kernel(self):
        """TPU full attention must not call the NPU-only sigmoid kernel."""
        from sglang.srt.models import qwen3_5

        x = torch.tensor([[2.0, -3.0, 4.0, 1.0]], dtype=torch.bfloat16)
        for gate_shape in ((1, 4), (1, 2, 2)):
            with self.subTest(gate_shape=gate_shape):
                gate = torch.tensor([-2.0, 0.0, 1.0, 3.0]).to(x.dtype)
                gate = gate.reshape(gate_shape)
                layer = SimpleNamespace(
                    attn_output_gate=True,
                    _prepare_qkv_gate=lambda **kwargs: (None, None, None, gate),
                    attn=lambda *args: x,
                    o_proj=lambda value: (value, None),
                )
                with (
                    patch.object(qwen3_5, "_is_tpu", True),
                    patch.object(qwen3_5, "_is_npu", False),
                ):
                    out = qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention(
                        layer,
                        positions=torch.tensor([0]),
                        hidden_states=x,
                        forward_batch=None,
                    )
                torch.testing.assert_close(out, x * torch.sigmoid(gate.reshape_as(x)))

    def test_bf16_dtype_preserved(self):
        norm = RMSNormGated(self.H, eps=self.EPS, dtype=torch.bfloat16)
        x = torch.randn(2, self.H, dtype=torch.bfloat16)
        z = torch.randn(2, self.H, dtype=torch.bfloat16)
        out = norm(x, z)
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
