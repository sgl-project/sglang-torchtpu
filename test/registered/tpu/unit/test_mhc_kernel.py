# SPDX-License-Identifier: Apache-2.0
"""The fused TPU mHC site kernel (sglang_tpu.mhc) vs the shared hc_pre / hc_post
torch path on CPU.

The reference runs in fp32 on CPU: torch_tpu evaluates its fp32 projection at
bf16 precision (gate errors ~3e-3), while the kernel's bf16 hi + lo split of
hc_fn tracks the fp32 projection.
"""

import unittest
from contextlib import ExitStack

import torch

from sglang.kernels.ops.layernorm.mhc import hc_post, hc_pre
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci

try:
    import torch_tpu  # noqa: F401

    import sglang_tpu.mhc as kernel
    from sglang_tpu.mhc import supports_mhc_site
except ImportError:
    torch_tpu = None

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

HC, HIDDEN = 4, 4096
RMS_EPS, HC_EPS, ITERS = 1e-5, 1e-6, 20
SITE = dict(hc_mult=HC, rms_eps=RMS_EPS, hc_eps=HC_EPS, sinkhorn_iters=ITERS)
KERNEL_SITE = dict(SITE, post_mult_value=2.0)


def tpu(t):
    return t.to("tpu")


@unittest.skipUnless(torch_tpu is not None, "Requires torch_tpu")
class TestMhcKernel(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        for env in (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE,
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST,
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE,
        ):
            self.stack.enter_context(env.override(False))
        self.addCleanup(self.stack.close)
        kernel.register_ops()

        self.gen = torch.Generator().manual_seed(7)
        mix = (2 + HC) * HC
        self.fn = (
            (torch.randn(mix, HC * HIDDEN, generator=self.gen) * 0.02)
            .bfloat16()
            .float()
        )
        self.scale = torch.tensor([0.7, 1.3, 2.0])
        self.base = torch.randn(mix, generator=self.gen) * 0.5
        # The next sublayer's input RMSNorm, applied in-kernel to the collapse.
        self.norm_weight = (
            1 + 0.1 * torch.randn(HIDDEN, generator=self.gen)
        ).bfloat16()

    def params(self, to=lambda t: t):
        return to(self.fn), to(self.scale), to(self.base)

    def kernel_params(self):
        return dict(
            hc_fn=tpu(self.fn), hc_scale=tpu(self.scale), hc_base=tpu(self.base)
        )

    def ref_norm(self, layer_in):
        x = layer_in.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (x * self.norm_weight.float()).bfloat16()

    def assertClose(self, actual, expected, rtol):
        actual, expected = actual.cpu().double(), expected.cpu().double()
        rel = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.assertLess(rel.item(), rtol, f"relative error {rel.item():.3e}")

    def test_pre_post_pre_post_match_torch_path(self):
        norm = dict(out_norm_weight=tpu(self.norm_weight), out_norm_eps=RMS_EPS)
        # 200 tokens exercises the multi-block prefill path with a partial block.
        for tokens in (1, 16, 200):
            with self.subTest(tokens=tokens):
                res = torch.randn(tokens, HC * HIDDEN, generator=self.gen).bfloat16()
                out = torch.randn(tokens, HIDDEN, generator=self.gen).bfloat16()
                self.assertTrue(supports_mhc_site(tpu(res), HC, x=tpu(out)))

                ref_in, ref_comb, ref_post, ref_fused = hc_pre(
                    res, *self.params(), **SITE
                )
                self.assertFalse(ref_fused)
                layer_in, comb, post, fused = kernel.hc_pre(
                    x=tpu(res),
                    **self.kernel_params(),
                    **KERNEL_SITE,
                    hc_norm_weight=None,
                    **norm,
                )
                self.assertTrue(fused)
                # Normalized collapsed inputs are bf16: one rounding step of error.
                self.assertClose(layer_in, self.ref_norm(ref_in), 4e-3)
                self.assertClose(post, ref_post, 1e-5)
                self.assertClose(comb, ref_comb, 1e-5)

                gates = (tpu(ref_post), tpu(ref_comb))
                ref_res = hc_post(out, res, ref_post, ref_comb, hc_mult=HC)
                ref_in2, ref_comb2, ref_post2, _ = hc_pre(
                    ref_res, *self.params(), **SITE
                )
                seam = kernel.hc_post_pre(
                    x=tpu(out),
                    residual=tpu(res),
                    h_post=gates[0],
                    h_res=gates[1],
                    **self.kernel_params(),
                    **KERNEL_SITE,
                    **norm,
                )
                self.assertIsNotNone(seam)
                layer_in2, new_res, comb2, post2, fused2 = seam
                self.assertTrue(fused2)
                self.assertClose(new_res, ref_res, 4e-3)
                self.assertClose(layer_in2, self.ref_norm(ref_in2), 4e-3)
                self.assertClose(post2, ref_post2, 1e-4)
                self.assertClose(comb2, ref_comb2, 1e-4)

                last = kernel.hc_post(
                    x=tpu(out),
                    residual=tpu(res),
                    h_post=gates[0],
                    h_res=gates[1],
                    hc_mult=HC,
                )
                self.assertClose(last, ref_res, 4e-3)

    def test_pre_tracks_an_fp32_hc_fn(self):
        # Not bf16-representable: one bf16 pass would round the weight.
        mix = (2 + HC) * HC
        self.fn = torch.randn(mix, HC * HIDDEN, generator=self.gen) * 0.02
        self.assertFalse(torch.equal(self.fn, self.fn.bfloat16().float()))
        res = torch.randn(16, HC * HIDDEN, generator=self.gen).bfloat16()
        _, ref_comb, ref_post, _ = hc_pre(res, *self.params(), **SITE)
        _, comb, post, _ = kernel.hc_pre(
            x=tpu(res),
            **self.kernel_params(),
            **KERNEL_SITE,
            hc_norm_weight=None,
            out_norm_weight=None,
            out_norm_eps=None,
        )
        self.assertClose(post, ref_post, 1e-5)
        self.assertClose(comb, ref_comb, 1e-5)

    def test_unsupported_sites_take_the_torch_path(self):
        res = tpu(torch.randn(4, HC * HIDDEN, generator=self.gen).bfloat16())
        self.assertFalse(supports_mhc_site(res.cpu(), HC))
        self.assertFalse(supports_mhc_site(res.float(), HC))
        self.assertFalse(supports_mhc_site(res[:, : HC * 96], HC))
        self.assertFalse(supports_mhc_site(res, HC, post_mult_value=1.0))
        self.assertFalse(
            supports_mhc_site(res, HC, hc_norm_weight=tpu(self.norm_weight))
        )
        # A declined site returns None and takes the torch path.
        self.assertIsNone(
            kernel.hc_pre(
                x=res,
                **self.kernel_params(),
                **dict(KERNEL_SITE, post_mult_value=1.0),
                hc_norm_weight=None,
                out_norm_weight=tpu(self.norm_weight),
                out_norm_eps=RMS_EPS,
            )
        )


if __name__ == "__main__":
    unittest.main()
