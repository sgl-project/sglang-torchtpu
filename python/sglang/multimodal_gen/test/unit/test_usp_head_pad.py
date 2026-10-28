"""USPAttention pads heads to a multiple of the Ulysses degree only where it applies."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


def _pad(world_size, heads=30, kv_heads=None, **kwargs):
    q = torch.empty(1, 4, heads, 8)
    k = torch.empty(1, 4, kv_heads or heads, 8)
    args = dict(attn_mask=None, skip_sp=False, qkv_pre_all_to_all=False, q_prefix=None)
    args.update(kwargs)
    with (
        patch.object(layer, "get_sequence_parallel_world_size", lambda: world_size * 2),
        patch.object(layer, "get_ring_parallel_world_size", lambda: 2),
    ):
        return USPAttention._ulysses_head_pad(q, k, **args)


class TestUlyssesHeadPad(unittest.TestCase):
    def test_pads_to_a_multiple_of_the_ulysses_degree(self):
        self.assertEqual(_pad(8), 2)  # Z-Image: 30 heads -> 32
        self.assertEqual(_pad(4), 2)
        self.assertEqual(_pad(2), 0)
        self.assertEqual(_pad(8, heads=24), 0)

    def test_no_padding_where_it_does_not_apply(self):
        self.assertEqual(_pad(1), 0)
        self.assertEqual(_pad(8, skip_sp=True), 0)
        self.assertEqual(_pad(8, qkv_pre_all_to_all=True), 0)
        self.assertEqual(_pad(8, q_prefix=torch.empty(1)), 0)
        self.assertEqual(_pad(8, kv_heads=6), 0)  # GQA keeps the old contract
        per_head_mask = torch.empty(1, 30, 4, 4)
        self.assertEqual(_pad(8, attn_mask=per_head_mask), 0)
        self.assertEqual(_pad(8, attn_mask=torch.empty(1, 1, 4, 4)), 2)


if __name__ == "__main__":
    unittest.main()
