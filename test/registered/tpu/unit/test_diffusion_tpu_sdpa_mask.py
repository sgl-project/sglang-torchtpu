"""SDPA with query-broadcast masks on TPU matches CPU after the plugin's patch."""

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


def _inputs():
    torch.manual_seed(0)
    q = torch.randn(2, 8, 1024, 128).bfloat16()
    return q, torch.randn_like(q), torch.randn_like(q)


def _key_padding_mask():
    keep = torch.ones(2, 1, 1, 1024, dtype=torch.bool)
    keep[:, :, :, :100] = False
    return keep


def _init_worker():
    from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

    # Worker bring-up installs the plugin's runtime patches.
    TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuSdpaQueryBroadcastMask(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_worker()
        assert getattr(F.scaled_dot_product_attention, "_sglang_tpu_mask_patch", False)

    def _check(self, mask):
        q, k, v = _inputs()
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

    def test_bool_key_padding_mask(self):
        self._check(_key_padding_mask())

    def test_head_broadcast_mask(self):
        self._check(torch.zeros(1, 1, 1, 1024).bfloat16())


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTorchTpuStillNeedsTheMaskPatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_worker()

    def test_unpatched_sdpa_returns_nan(self):
        # Fails once torch_tpu handles query-broadcast masks; then remove
        # sglang_tpu/multimodal_gen/sdpa_mask.py and this test.
        sdpa = getattr(
            F.scaled_dot_product_attention,
            "__wrapped__",
            F.scaled_dot_product_attention,
        )
        q, k, v = _inputs()
        got = sdpa(
            q.to("tpu"),
            k.to("tpu"),
            v.to("tpu"),
            attn_mask=_key_padding_mask().to("tpu"),
        ).to("cpu")
        self.assertTrue(
            torch.isnan(got.float()).any().item(),
            "torch_tpu now handles query-broadcast SDPA masks at this shape: "
            "check the production shapes (LTX-2 text masks), then remove "
            "sglang_tpu/multimodal_gen/sdpa_mask.py and this test.",
        )


if __name__ == "__main__":
    unittest.main()
