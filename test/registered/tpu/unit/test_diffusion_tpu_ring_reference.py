"""CPU checks of the float32 reference behind TPU ring steps and LSE attention.

Ring steps and LSE attention the Pallas kernel does not take (here: short
shapes) run a chunked float32 reference. These tests use CPU tensors; they sit
in the TPU lane because importing sglang_tpu needs the TPU stack.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest.mock import patch

import torch

from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.tpu_splash_attn import TpuSplashAttentionImpl


class TestRingKernelFallback(unittest.TestCase):
    """Ring steps the Pallas kernel does not take (here: short shapes) run a
    chunked float32 reference."""

    def setUp(self):
        self.tsa = tpu_splash_attn
        torch.manual_seed(0)
        self.batch, self.heads, self.kv_heads, self.dim = 2, 4, 2, 64
        self.scale = self.dim**-0.5

    def _rows_per_chunk(self, rows, kv_len):
        # Bytes for `rows` query rows of float32 scores: forces several chunks.
        return patch.object(
            self.tsa,
            "_REFERENCE_SCORE_BYTES",
            4 * self.batch * self.heads * kv_len * rows,
        )

    def test_chained_steps_match_attention_over_the_real_keys(self):
        b, h, hkv, d = self.batch, self.heads, self.kv_heads, self.dim
        q_len, shard, pad = 48, 40, 7
        q = torch.randn(b, h, q_len, d)
        keys = torch.randn(2, b, hkv, shard, d)
        values = torch.randn(2, b, hkv, shard, d)
        # Padding past the real keys may hold anything, NaN included.
        keys[1, :, :, shard - pad :] = float("nan")
        values[1, :, :, shard - pad :] = float("nan")
        real = 2 * shard - pad
        carry = None
        with self._rows_per_chunk(5, shard):
            for i in range(2):
                info = torch.tensor(
                    [i * b * hkv, min(max(real - i * shard, 0), shard)],
                    dtype=torch.int32,
                )
                out, carry = self.tsa.tpu_ring_attention_step(
                    q, keys.flatten(0, 1), values.flatten(0, 1), info, carry, self.scale
                )
        k_all = keys.permute(1, 2, 0, 3, 4).reshape(b, hkv, 2 * shard, d)[:, :, :real]
        v_all = values.permute(1, 2, 0, 3, 4).reshape(b, hkv, 2 * shard, d)[:, :, :real]
        k_all = k_all.repeat_interleave(h // hkv, 1)
        v_all = v_all.repeat_interleave(h // hkv, 1)
        want = torch.softmax(q @ k_all.transpose(-1, -2) * self.scale, -1) @ v_all
        self.assertFalse(out.isnan().any())
        torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5)

    def test_an_empty_shard_keeps_the_carry(self):
        b, hkv, d = self.batch, self.kv_heads, self.dim
        q = torch.randn(b, self.heads, 24, d)
        keys, values = torch.randn(b, hkv, 32, d), torch.randn(b, hkv, 32, d)
        full = torch.tensor([0, 32], dtype=torch.int32)
        out, carry = self.tsa.tpu_ring_attention_step(
            q, keys, values, full, None, self.scale
        )
        empty = torch.tensor([0, 0], dtype=torch.int32)
        again, _ = self.tsa.tpu_ring_attention_step(
            q, keys, values, empty, carry, self.scale
        )
        torch.testing.assert_close(again, out)

    def test_chunked_lse_attention_matches_full_scores(self):
        b, h, d = self.batch, self.heads, self.dim
        q, k, v = (torch.randn(b, h, 30, d) for _ in range(3))
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                scores = q @ k.transpose(-1, -2) * self.scale
                if is_causal:
                    scores = scores.masked_fill(
                        torch.ones(30, 30, dtype=torch.bool).triu(1), float("-inf")
                    )
                with self._rows_per_chunk(4, 30):
                    out, lse = self.tsa.tpu_attention_with_lse(
                        q, k, v, self.scale, is_causal=is_causal
                    )
                torch.testing.assert_close(lse, torch.logsumexp(scores, -1))
                torch.testing.assert_close(out, torch.softmax(scores, -1) @ v)

    def test_rejects_a_malformed_kv_info(self):
        q = torch.randn(1, 2, 8, self.dim)
        k = torch.randn(1, 2, 8, self.dim)
        with self.assertRaisesRegex(AssertionError, "int32"):
            self.tsa.tpu_ring_attention_step(
                q, k, k, torch.tensor([0, 8]), None, self.scale
            )

    def test_lse_forward_off_tpu_uses_the_reference(self):
        impl = TpuSplashAttentionImpl(
            num_heads=self.heads,
            head_size=self.dim,
            causal=False,
            softmax_scale=self.scale,
        )
        q, k, v = (torch.randn(1, 12, self.heads, self.dim) for _ in range(3))
        with patch.object(
            self.tsa, "_FLASH_RING_OP", side_effect=AssertionError("kernel")
        ):
            out, lse = impl.forward(q, k, v, None, return_softmax_lse=True)
        scores = torch.einsum("bshd,bthd->bhst", q, k) * self.scale
        torch.testing.assert_close(lse, torch.logsumexp(scores, -1))
        torch.testing.assert_close(
            out, torch.einsum("bhst,bthd->bshd", torch.softmax(scores, -1), v)
        )


if __name__ == "__main__":
    unittest.main()
