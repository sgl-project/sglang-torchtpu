"""Kimi-K3 post-load parameter layout (sglang_tpu/kimi_k3/param_layout.py), without a TPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kimi_k3.param_layout import store_decode_params

HEADS, HEAD_DIM = 4, 128


def _kda_layer(weight):
    attn = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
    nn.Module.__init__(attn)
    attn.head_dim, attn.local_num_heads = HEAD_DIM, HEADS
    attn.o_norm = nn.Module()
    attn.o_norm.weight = nn.Parameter(weight, requires_grad=False)
    layer = nn.Module()
    layer.self_attn = attn
    return layer


class TestK3ParamLayout(CustomTestCase):
    def test_kda_onorm_weight_is_stored_f32_per_head(self):
        w = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
        layer = _kda_layer(w.clone())
        mla = nn.Module()
        mla.self_attn = nn.Linear(2, 2)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[layer, mla, PPMissingLayer()])
        )
        store_decode_params(model)
        got = layer.self_attn.o_norm.weight
        self.assertEqual(
            (got.dtype, tuple(got.shape)), (torch.float32, (HEADS, HEAD_DIM))
        )
        # Exact: the same values the forward broadcast before.
        torch.testing.assert_close(
            got, w.float().expand(HEADS, HEAD_DIM), atol=0, rtol=0
        )
        store_decode_params(model)  # idempotent
        self.assertEqual(tuple(layer.self_attn.o_norm.weight.shape), (HEADS, HEAD_DIM))


if __name__ == "__main__":
    unittest.main()
