"""USPAttention under Ulysses: tail-pad meta attends over the real key prefix,
and heads pad to a multiple of the Ulysses degree only where it applies."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.unit.test_usp_attention_kv_gather import (
    _RankInvariantMeta,
)

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


def _attention(backend=AttentionBackendEnum.TORCH_SDPA):
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
    return attn


@contextlib.contextmanager
def _ulysses(degree, exchanged_heads):
    """Ulysses ``degree`` x ring 1 with identity exchanges; the input exchange
    records the head count it receives."""

    def input_all_to_all(q, k, v):
        exchanged_heads.append(q.shape[2])
        return q, k, v

    with (
        patch(
            f"{_LAYER}.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=degree),
        patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=degree),
        patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
        patch(f"{_LAYER}._ipc_input_a2a_qkv", return_value=None),
        patch(f"{_LAYER}._usp_input_all_to_all_qkv", side_effect=input_all_to_all),
        patch(f"{_LAYER}._usp_output_all_to_all", side_effect=lambda t, head_dim: t),
    ):
        yield


def _head_pad(world_size, heads=30, kv_heads=None, mode="ulysses", **kwargs):
    q = torch.empty(1, 4, heads, 8)
    k = torch.empty(1, 4, kv_heads or heads, 8)
    attn = _attention()
    attn.sp_attention_mode = mode
    args = dict(
        attn_mask=None,
        skip_sp=False,
        qkv_pre_all_to_all=False,
        q_prefix=None,
        seq_lens=None,
        num_replicated_prefix=0,
        num_replicated_suffix=0,
        num_replicated_kv_prefix=0,
    )
    args.update(kwargs)
    with (
        patch.object(layer, "get_sequence_parallel_world_size", lambda: world_size * 2),
        patch.object(layer, "get_ring_parallel_world_size", lambda: 2),
    ):
        return attn._ulysses_head_pad(q, k, **args)


class TestUlyssesPadding(unittest.TestCase):
    real, pad = 13, 3

    def _tail_pad_forward(self, backend, heads=2, degree=2):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, self.real + self.pad, heads, 8) for _ in range(3))
        meta = _RankInvariantMeta(
            pad_start=self.real, pad_end=self.real + self.pad, local_pad=self.pad
        )
        attn, exchanged_heads = _attention(backend), []
        with _ulysses(degree, exchanged_heads):
            out = attn.forward(q, k, v, attn_mask_meta=meta)

        want = F.scaled_dot_product_attention(
            q[:, : self.real].transpose(1, 2),
            k[:, : self.real].transpose(1, 2),
            v[:, : self.real].transpose(1, 2),
        ).transpose(1, 2)
        self.assertEqual(out.shape, q.shape)
        torch.testing.assert_close(out[:, : self.real], want)
        return attn.attn_impl.kv_lens, exchanged_heads

    def test_selected_backend_attends_over_the_real_key_prefix(self):
        kv_lens, _ = self._tail_pad_forward(AttentionBackendEnum.TORCH_SDPA)
        self.assertEqual(kv_lens, [self.real])

    def test_sparse_backend_keeps_the_masked_fallback(self):
        # Sparse kernels need K/V as long as Q: the gathered-mask SDPA serves them.
        kv_lens, _ = self._tail_pad_forward(AttentionBackendEnum.SLIDING_TILE_ATTN)
        self.assertEqual(kv_lens, [])

    def test_padded_heads_cross_the_exchange_and_are_cut_after_it(self):
        # Z-Image's 30 heads at Ulysses 8, scaled down: 3 heads at Ulysses 4
        # enter the exchange as 4 and every real head comes back.
        kv_lens, exchanged_heads = self._tail_pad_forward(
            AttentionBackendEnum.TORCH_SDPA, heads=3, degree=4
        )
        self.assertEqual(exchanged_heads, [4])
        self.assertEqual(kv_lens, [self.real])

    def test_padded_heads_keep_the_replicated_suffix(self):
        # Z-Image's caption tokens: the padded call still keeps them out of
        # the exchange, or every rank would contribute its copy as sequence rows.
        suffix_calls = []

        def suffix_path(q, k, v, attn_metadata, num_rep):
            suffix_calls.append((q.shape[2], num_rep))
            return q

        q = torch.randn(1, 6, 3, 8)
        with (
            _ulysses(4, []),
            patch.object(
                USPAttention, "_forward_with_replicated_suffix", side_effect=suffix_path
            ),
        ):
            out = _attention().forward(q, q, q, num_replicated_suffix=2)
        self.assertEqual(suffix_calls, [(4, 2)])
        self.assertEqual(out.shape, q.shape)

    def test_kv_gather_keeps_every_head(self):
        # The gather exchanges K/V along the sequence; no head is split.
        gathered_heads = []

        def kv_gather(q, *args):
            gathered_heads.append(q.shape[2])
            return q

        attn = _attention()
        attn.sp_attention_mode = "kv_gather"
        q = torch.randn(1, 4, 3, 8)
        with (
            _ulysses(4, []),
            patch.object(
                USPAttention, "_forward_with_kv_gather", side_effect=kv_gather
            ),
        ):
            attn.forward(q, q, q)
        self.assertEqual(gathered_heads, [3])
        # The calls it leaves to the Ulysses exchange still pad: varlen, and
        # auto mode's fallback for a float mask.
        self.assertEqual(_head_pad(8, mode="kv_gather", seq_lens=[4]), 2)
        float_mask = torch.ones(1, 4)
        self.assertEqual(_head_pad(8, mode="kv_gather", attn_mask=float_mask), 2)

    def test_pads_to_a_multiple_of_the_ulysses_degree(self):
        self.assertEqual(_head_pad(8), 2)  # Z-Image: 30 heads -> 32
        self.assertEqual(_head_pad(4), 2)
        self.assertEqual(_head_pad(2), 0)
        self.assertEqual(_head_pad(8, heads=24), 0)

    def test_no_padding_where_it_does_not_apply(self):
        self.assertEqual(_head_pad(1), 0)
        self.assertEqual(_head_pad(8, skip_sp=True), 0)
        self.assertEqual(_head_pad(8, qkv_pre_all_to_all=True), 0)
        self.assertEqual(_head_pad(8, q_prefix=torch.empty(1)), 0)
        self.assertEqual(_head_pad(8, kv_heads=6), 0)  # GQA keeps the old contract
        per_head_mask = torch.empty(1, 30, 4, 4)
        self.assertEqual(_head_pad(8, attn_mask=per_head_mask), 0)
        self.assertEqual(_head_pad(8, attn_mask=torch.empty(1, 1, 4, 4)), 2)


if __name__ == "__main__":
    unittest.main()
