"""SDPA with query-broadcast masks on TPU matches CPU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _inputs(q_len: int = 1024, kv_len: int = 1024):
    torch.manual_seed(0)
    q = torch.randn(2, 8, q_len, 128).bfloat16()
    k = torch.randn(2, 8, kv_len, 128).bfloat16()
    v = torch.randn(2, 8, kv_len, 128).bfloat16()
    return q, k, v


def _key_padding_mask():
    keep = torch.ones(2, 1, 1, 1024, dtype=torch.bool)
    keep[:, :, :, :100] = False
    return keep


def _init_worker():
    from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

    TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuSdpaQueryBroadcastMask(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_worker()

    def _check(self, mask, *, q_len: int = 1024, kv_len: int = 1024):
        q, k, v = _inputs(q_len=q_len, kv_len=kv_len)
        ref_mask = mask if mask.dtype == torch.bool else mask.float()
        want = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=ref_mask
        )
        got = F.scaled_dot_product_attention(
            q.to("tpu"), k.to("tpu"), v.to("tpu"), attn_mask=mask.to("tpu")
        ).to("cpu")
        self.assertTrue(torch.isfinite(got.float()).all().item())
        self.assertLess((got.float() - want).abs().max().item(), 2e-2)

    def test_additive_key_padding_mask(self):
        mask = torch.zeros(2, 1, 1, 1024)
        mask[:, :, :, :100] = torch.finfo(torch.bfloat16).min
        self._check(mask.bfloat16())

    def test_cross_attention_key_padding_mask(self):
        # LTX-2 video-to-text cross-attention: [B, H, S_video, D] x [B, H, S_text, D]
        # with a [B, 1, 1, S_text] additive bfloat16 padding mask.
        mask = torch.zeros(2, 1, 1, 1024)
        mask[:, :, :, -100:] = torch.finfo(torch.bfloat16).min
        self._check(mask.bfloat16(), q_len=4096, kv_len=1024)

    def test_bool_key_padding_mask(self):
        self._check(_key_padding_mask())

    def test_head_broadcast_mask(self):
        self._check(torch.zeros(1, 1, 1, 1024).bfloat16())


if __name__ == "__main__":
    unittest.main()
