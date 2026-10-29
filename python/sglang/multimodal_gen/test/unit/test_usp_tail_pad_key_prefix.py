"""Ulysses + tail-pad meta on a non-FA backend: attention over the real key prefix."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class _SdpaImpl:
    def __init__(self):
        self.kv_lens = []

    def forward(self, q, k, v, attn_metadata):
        self.kv_lens.append(k.shape[1])
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2)


class TestTailPadKeyPrefix(unittest.TestCase):
    real, pad = 13, 3

    def _forward(self, backend):
        attn = USPAttention.__new__(USPAttention)
        attn.skip_sequence_parallel = False
        attn.sp_attention_mode = "ulysses"
        attn.sp_attention_mode_is_auto = False
        attn.is_cross_attention = False
        attn.causal = False
        # Read by the gathered-mask SDPA fallback.
        attn.allow_cudnn_sdp = False
        attn.softmax_scale = None
        attn.backend = backend
        attn.attn_impl = _SdpaImpl()

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, self.real + self.pad, 2, 8) for _ in range(3))
        meta = {
            "pad_start": self.real,
            "pad_end": self.real + self.pad,
            "local_pad": self.pad,
        }
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(f"{_LAYER}._ipc_input_a2a_qkv", return_value=None),
            patch(
                f"{_LAYER}._usp_input_all_to_all_qkv",
                side_effect=lambda q, k, v: (q, k, v),
            ),
            patch(
                f"{_LAYER}._usp_output_all_to_all", side_effect=lambda t, head_dim: t
            ),
        ):
            out = attn.forward(q, k, v, attn_mask_meta=meta)

        want = F.scaled_dot_product_attention(
            q[:, : self.real].transpose(1, 2),
            k[:, : self.real].transpose(1, 2),
            v[:, : self.real].transpose(1, 2),
        ).transpose(1, 2)
        torch.testing.assert_close(out[:, : self.real], want)
        return attn.attn_impl.kv_lens, out

    def test_selected_backend_attends_over_the_real_key_prefix(self):
        kv_lens, out = self._forward(AttentionBackendEnum.TORCH_SDPA)
        self.assertEqual(kv_lens, [self.real])
        self.assertTrue(torch.all(out[:, self.real :] == 0))

    def test_sparse_backend_keeps_the_masked_fallback(self):
        # Sparse kernels need K/V as long as Q: the gathered-mask SDPA serves them.
        kv_lens, _ = self._forward(AttentionBackendEnum.SLIDING_TILE_ATTN)
        self.assertEqual(kv_lens, [])


if __name__ == "__main__":
    unittest.main()
