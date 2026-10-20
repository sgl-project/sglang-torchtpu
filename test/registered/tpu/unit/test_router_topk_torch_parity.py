"""Registered router top-k op (through the biased_topk hook) vs biased_topk_impl
on torch_tpu, for bf16 and f32 router logits, with random and tied scores."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import torch
    import torch_tpu  # noqa: F401

    _HAS_TPU = torch.tpu.is_available()
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

E, K, SCALE = 896, 16, 2.5


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestRouterTopkTorchParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.moe import router_topk_tpu as rt

        rt.register_router_topk_op()
        cls.rt = rt

    def _compare(self, logits, bias, apply_on_output, msg):
        from sglang.srt.layers.moe.topk import biased_topk_impl

        lt, bt = logits.to("tpu"), bias.to("tpu")
        self.assertTrue(self.rt._eligible(lt, bt, K, 0), msg)
        args = (lt, lt, bt, K, True, "sigmoid", 0, SCALE, None, None, apply_on_output)
        ref_w, ref_ids = (t.cpu() for t in biased_topk_impl(*args))
        got_w, got_ids = (
            t.cpu() for t in self.rt.biased_topk_around(biased_topk_impl, *args)
        )
        for t in range(logits.shape[0]):
            want = dict(zip(ref_ids[t].tolist(), ref_w[t].tolist()))
            got = dict(zip(got_ids[t].tolist(), got_w[t].tolist()))
            self.assertEqual(set(got), set(want), f"{msg} row {t}")
            for i, x in want.items():
                self.assertAlmostEqual(got[i], x, delta=1e-6 * max(1.0, abs(x)))

    def test_matches_biased_topk_impl(self):
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float32):
            for case in ("random", "ties"):
                for tokens in (1, 8, 64):
                    for apply_on_output in (False, True):
                        logits = torch.randn(tokens, E, generator=g) * 2.0
                        if case == "ties":
                            logits = torch.round(logits * 2) / 2  # ~17 values
                        bias = (torch.randn(E, generator=g) * 0.05).float()
                        msg = f"{dtype} {case} {tokens=} {apply_on_output=}"
                        with self.subTest(msg=msg):
                            self._compare(logits.to(dtype), bias, apply_on_output, msg)


if __name__ == "__main__":
    unittest.main()
