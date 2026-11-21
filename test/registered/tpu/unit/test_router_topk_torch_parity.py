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

SCALE = 2.5


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestRouterTopkTorchParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.moe import router_topk_tpu as rt

        rt.register_router_topk_op()
        cls.rt = rt

    def _compare(self, logits, bias, k, apply_on_output, grouped, msg, exact=False):
        from sglang.srt.layers.moe import topk as srt_topk

        lt, bt = logits.to("tpu"), bias.to("tpu")
        tokens = logits.shape[0]
        self.assertTrue(
            self.rt._eligible(lt, bt, k, 0)
            or self.rt._large_batch_eligible(lt, bt, k, 0),
            msg,
        )
        if grouped:
            impl, hook = (
                srt_topk.biased_grouped_topk_impl,
                self.rt.biased_grouped_topk_around,
            )
            args = (lt, lt, bt, k, True, 1, 1, 0, SCALE, apply_on_output)
        else:
            impl, hook = srt_topk.biased_topk_impl, self.rt.biased_topk_around
            args = (
                lt,
                lt,
                bt,
                k,
                True,
                "sigmoid",
                0,
                SCALE,
                None,
                None,
                apply_on_output,
            )
        # The eager, unhooked function, as TPU runs it.
        impl = getattr(impl, "_torchdynamo_orig_callable", impl)
        self.assertTrue(impl.__code__.co_filename.endswith("moe/topk.py"), impl)
        ref_w, ref_ids = (t.cpu() for t in impl(*args))
        got_w, got_ids = (t.cpu() for t in hook(impl, *args))
        if exact:
            # Same ids in the same order and bit-identical weights as the sort.
            self.assertTrue(
                torch.equal(got_ids.to(torch.int64), ref_ids.to(torch.int64)), msg
            )
            self.assertTrue(torch.equal(got_w.float(), ref_w.float()), msg)
            return
        for t in range(tokens):
            want = dict(zip(ref_ids[t].tolist(), ref_w[t].tolist()))
            got = dict(zip(got_ids[t].tolist(), got_w[t].tolist()))
            self.assertEqual(set(got), set(want), f"{msg} row {t}")
            for i, x in want.items():
                self.assertAlmostEqual(got[i], x, delta=1e-6 * max(1.0, abs(x)))

    def test_matches_biased_topk_impl(self):
        """Kimi-K3: top-16 of 896 through biased_topk_impl."""
        g = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float32):
            for case in ("random", "ties"):
                for tokens in (1, 8, 64):
                    for apply_on_output in (False, True):
                        logits = torch.randn(tokens, 896, generator=g) * 2.0
                        if case == "ties":
                            logits = torch.round(logits * 2) / 2  # ~17 values
                        bias = (torch.randn(896, generator=g) * 0.05).float()
                        msg = f"{dtype} {case} {tokens=} {apply_on_output=}"
                        with self.subTest(msg=msg):
                            self._compare(
                                logits.to(dtype), bias, 16, apply_on_output, False, msg
                            )

    def _grouped(self, logits, bias):
        from sglang.srt.layers.moe import topk as srt_topk

        args = (logits, logits, bias, 8, True, 1, 1, 0, SCALE, True)
        impl = srt_topk.biased_grouped_topk_impl
        impl = getattr(impl, "_torchdynamo_orig_callable", impl)
        return self.rt.biased_grouped_topk_around(impl, *args)

    def test_grouped_top8_of_288_matches_the_sort(self):
        """GLM-5.3-Flash: top-8 of 288, one group, bf16 logits, f32 bias. The
        kernel serves up to 128 tokens; without exact ties its ids (in order)
        and weights equal biased_grouped_topk_impl's bit for bit."""
        g = torch.Generator().manual_seed(1)
        for tokens in (1, 16, 64, 128):
            logits = (torch.randn(tokens, 288, generator=g) * 2.0).to(torch.bfloat16)
            bias = (torch.randn(288, generator=g) * 0.05).float()
            msg = f"{tokens=}"
            with self.subTest(msg=msg):
                self._compare(logits, bias, 8, True, True, msg, exact=True)

    def test_prefill_kernel_matches_the_argmax_rounds(self):
        """SGLANG_TPU_ROUTER_TOPK_KERNEL_PREFILL: prefill batches (whole 256-row
        blocks) through the gridded kernel give the argmax rounds' ids and
        weights bit for bit, with random and tied scores."""
        from sglang_tpu.envs import envs as tpu_envs

        g = torch.Generator().manual_seed(3)
        for tokens in (256, 1024, 4096):
            for case in ("random", "ties"):
                logits = torch.randn(tokens, 288, generator=g) * 2.0
                if case == "ties":
                    logits = torch.round(logits * 2) / 2
                lt = logits.to(torch.bfloat16).to("tpu")
                bt = (torch.randn(288, generator=g) * 0.05).float().to("tpu")
                want_w, want_ids = (
                    t.cpu() for t in self.rt._large_batch_topk(lt, bt, 8, True, SCALE, True)
                )
                with tpu_envs.SGLANG_TPU_ROUTER_TOPK_KERNEL_PREFILL.override(True):
                    self.assertTrue(self.rt._prefill_kernel_eligible(lt, bt, 8, 0))
                    got_w, got_ids = (t.cpu() for t in self._grouped(lt, bt))
                with self.subTest(tokens=tokens, case=case):
                    self.assertTrue(torch.equal(got_ids.long(), want_ids.long()))
                    self.assertTrue(torch.equal(got_w.float(), want_w.float()))

    def test_grouped_top8_ties_pick_the_lowest_ids(self):
        """Exact choice ties (a few logit values, a bias on a 0.25 grid): the
        kernel (up to 128 tokens) and the argmax rounds past it both take the
        lowest tied expert ids. torch.topk on TPU breaks ties in another order,
        so a stable CPU sort is the reference."""
        from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

        g = torch.Generator().manual_seed(2)
        for tokens in (1, 16, 64, 128, 136):
            logits = (torch.randint(-6, 7, (tokens, 288), generator=g) / 2).to(
                torch.bfloat16
            )
            bias = torch.randint(0, 4, (288,), generator=g).float() * 0.25
            scores = torch.sigmoid(logits.float()).to(torch.bfloat16).float()
            choice = scores + bias[None, :]
            want = torch.sort(choice, dim=-1, descending=True, stable=True).indices[
                :, :8
            ]
            want_w = scores.gather(1, want)
            want_w = want_w / (want_w.sum(-1, keepdim=True) + _RENORMALIZE_SUM_EPSILON)
            w, ids = (t.cpu() for t in self._grouped(logits.to("tpu"), bias.to("tpu")))
            with self.subTest(tokens=tokens):
                self.assertTrue(torch.equal(ids.long(), want), f"{tokens=}")
                torch.testing.assert_close(w.float(), want_w * SCALE, rtol=1e-6, atol=0)


if __name__ == "__main__":
    unittest.main()
