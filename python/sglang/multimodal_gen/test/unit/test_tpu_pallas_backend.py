"""Tests for the TPU pallas attention backend.

The interface tests run everywhere; the numeric parity test runs only on a
machine with a TPU and torch_tpu installed.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_pallas import (
    TpuPallasAttentionBackend,
    TpuPallasAttentionImpl,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


def _tpu_available() -> bool:
    try:
        return hasattr(torch, "tpu") and torch.tpu.device_count() > 0
    except Exception:
        return False


class TestBackendInterface(unittest.TestCase):
    def test_enum_and_impl(self):
        self.assertIs(
            TpuPallasAttentionBackend.get_enum(),
            AttentionBackendEnum.TPU_PALLAS_FLASH,
        )
        self.assertIs(
            TpuPallasAttentionBackend.get_impl_cls(), TpuPallasAttentionImpl
        )

    def test_causal_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            TpuPallasAttentionImpl(
                num_heads=8, head_size=64, causal=True, softmax_scale=0.125
            )

    def test_gqa_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            TpuPallasAttentionImpl(
                num_heads=8,
                head_size=64,
                causal=False,
                softmax_scale=0.125,
                num_kv_heads=2,
            )


@unittest.skipUnless(_tpu_available(), "requires a TPU with torch_tpu")
class TestNumericParity(unittest.TestCase):
    def test_matches_sdpa_reference_with_padding(self):
        torch.manual_seed(0)
        batch, heads, seq, head_dim = 1, 2, 100, 64  # seq forces segment padding
        scale = head_dim**-0.5
        impl = TpuPallasAttentionImpl(
            num_heads=heads, head_size=head_dim, causal=False, softmax_scale=scale
        )
        q = torch.randn(batch, seq, heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, heads, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, seq, heads, head_dim, dtype=torch.bfloat16)

        out = impl.forward(
            q.to("tpu"), k.to("tpu"), v.to("tpu"), attn_metadata=None
        )
        torch.tpu.synchronize()

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)
        torch.testing.assert_close(
            out.cpu().float(), expected, atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
