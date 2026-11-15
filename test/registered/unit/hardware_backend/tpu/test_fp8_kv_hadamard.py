"""CPU guards for the fp8 KV Hadamard rotation (SGLANG_TPU_FP8_KV_HADAMARD).

The RPA backends rotate q and k of every head by an orthonormal
Hadamard matrix (built in-graph from iotas) before K is quantized to fp8, so attention logits are
unchanged while fp8 rounding stops concentrating on K's outlier channels.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import (
    _fp8_k_hadamard_dim,
    _hadamard_matrix,
    _hadamard_signs,
    _quantize_kv_to_fp8,
    _rotate_heads,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


class TestHadamardMatrix(CustomTestCase):
    def test_orthonormal_and_symmetric(self):
        for dim in (64, 128, 256):
            with self.subTest(dim=dim):
                h = _hadamard_matrix(dim)
                self.assertEqual(h.dtype, torch.float32)
                torch.testing.assert_close(h @ h.T, torch.eye(dim), atol=1e-6, rtol=0)
                torch.testing.assert_close(h, h.T)

    def test_refuses_non_power_of_two(self):
        for dim in (0, 96, 192):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                _hadamard_matrix(dim)


class TestHadamardSigns(CustomTestCase):
    def test_matches_sylvester_construction(self):
        for dim in (1, 2, 64, 128, 256):
            with self.subTest(dim=dim):
                torch.testing.assert_close(
                    _hadamard_signs(dim, device="cpu", dtype=torch.float32),
                    _hadamard_matrix(dim) * dim**0.5,
                    atol=1e-5,
                    rtol=0,
                )


class TestHadamardGating(CustomTestCase):
    def _dim(self, **overrides):
        kwargs = dict(fp8_kv=True, enabled=True, head_dim=128, external_k_writer=False)
        kwargs.update(overrides)
        return _fp8_k_hadamard_dim(**kwargs)

    def test_on_for_fp8_power_of_two(self):
        self.assertEqual(self._dim(), 128)
        self.assertEqual(self._dim(head_dim=64), 64)  # gpt-oss

    def test_off_for_bf16_cache_or_env_off(self):
        self.assertIsNone(self._dim(fp8_kv=False))
        self.assertIsNone(self._dim(enabled=False))

    def test_off_for_non_power_of_two_or_unknown_head_dim(self):
        for head_dim in (None, 0, 80, 96, 192):
            with self.subTest(head_dim=head_dim):
                self.assertIsNone(self._dim(head_dim=head_dim))

    def test_off_with_external_k_writer(self):
        # DFlash/DSpark drafts and the Muse megakernel write K outside _forward.
        self.assertIsNone(self._dim(external_k_writer=True))


class TestRotateHeads(CustomTestCase):
    def test_matches_hadamard_matmul(self):
        torch.manual_seed(0)
        for dim in (1, 2, 64, 128, 256):
            with self.subTest(dim=dim):
                x = torch.randn(5, 3, dim)
                torch.testing.assert_close(
                    _rotate_heads(x), x @ _hadamard_matrix(dim), atol=1e-5, rtol=1e-5
                )

    def test_bf16_matches_float32_reference(self):
        torch.manual_seed(0)
        x = torch.randn(6, 4, 128)
        ref = x @ _hadamard_matrix(128)
        out = _rotate_heads(x.to(torch.bfloat16)).float()
        torch.testing.assert_close(out, ref, atol=0.05, rtol=0.02)

    def test_is_its_own_inverse(self):
        x = torch.randn(4, 2, 128)
        torch.testing.assert_close(
            _rotate_heads(_rotate_heads(x)), x, atol=1e-5, rtol=0
        )

    def test_preserves_logits(self):
        torch.manual_seed(0)
        q = torch.randn(7, 8, 128)
        k = torch.randn(7, 8, 128)
        torch.testing.assert_close(
            (_rotate_heads(q) * _rotate_heads(k)).sum(-1),
            (q * k).sum(-1),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_keeps_dtype(self):
        x = torch.randn(3, 2, 128).to(torch.bfloat16)
        self.assertEqual(_rotate_heads(x).dtype, torch.bfloat16)

    def test_reduces_fp8_logit_error_on_outlier_keys(self):
        # Keys shaped like Qwen3's (RMS-normalized, times a k_norm weight with
        # a few large channels shared with the query): fp8 error on the
        # rotated keys must be well below the error on the raw keys.
        torch.manual_seed(0)
        d = 128
        w = torch.full((d,), 1.5)
        w[[3, 67]] = 34.5
        k = torch.randn(2000, 1, d)
        k = k / k.pow(2).mean(-1, keepdim=True).sqrt() * w
        q = torch.randn(1, 1, d)
        q = q / q.pow(2).mean(-1, keepdim=True).sqrt() * (w / 3)
        ref = (q * k).sum(-1)
        raw = (q * _quantize_kv_to_fp8(k, 1.0, FP8).float()).sum(-1)
        rot = (
            _rotate_heads(q) * _quantize_kv_to_fp8(_rotate_heads(k), 1.0, FP8).float()
        ).sum(-1)
        err_raw = (raw - ref).pow(2).mean().sqrt()
        err_rot = (rot - ref).pow(2).mean().sqrt()
        self.assertLess(err_rot, err_raw / 2)


if __name__ == "__main__":
    unittest.main()
