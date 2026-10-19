"""Packed KV heads (SGLANG_TPU_KV_PACK_HEADS): two 64-wide heads in one 128-lane row.

The RPA backend turns per-head GQA attention into one 128-wide kv head: q placed in its
head's lanes, k/v rows side by side, output lanes selected back. Exact against the
per-head reference on CPU.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.layers.attention.attention_pallas_rpa_base import (
    _pack_qkv_heads,
    _unpack_out_heads,
)
from sglang_tpu.mem_cache.kv_pool import kv_pack_heads

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _attend(q, k, v, scale):
    """q [T, Hq, d], k/v [T, Hkv, d]: causal GQA attention -> [T, Hq, d]."""
    T, hq, _ = q.shape
    g = hq // k.shape[1]
    k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    s = torch.einsum("thd,shd->hts", q, k) * scale
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    return torch.einsum("hts,shd->thd", s.softmax(-1), v)


class TestKvPackHeads(unittest.TestCase):
    def test_packed_attention_matches_per_head(self):
        torch.manual_seed(0)
        T, hq, hkv, d = 11, 16, 2, 64
        q, k, v = (torch.randn(T, h, d, dtype=torch.float64) for h in (hq, hkv, hkv))
        ref = _attend(q, k, v, d**-0.5)
        qp, kp, vp = _pack_qkv_heads(q, k, v, hkv)
        self.assertEqual(
            (qp.shape, kp.shape, vp.shape), ((T, hq, 128), (T, 1, 128), (T, 1, 128))
        )
        out = _unpack_out_heads(_attend(qp, kp, vp, d**-0.5), hkv, d)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)

    def test_pack_rule(self):
        with tpu_envs.SGLANG_TPU_KV_PACK_HEADS.override(True):
            self.assertEqual(kv_pack_heads(2, 64, 128), 2)
            self.assertEqual(kv_pack_heads(1, 128, 128), 1)  # nothing to pack
            self.assertEqual(kv_pack_heads(4, 64, 128), 1)  # does not fit one row
        with tpu_envs.SGLANG_TPU_KV_PACK_HEADS.override(False):
            self.assertEqual(kv_pack_heads(2, 64, 128), 1)


if __name__ == "__main__":
    unittest.main()
