"""CPU checks of which router top-k calls the TPU hooks hand to the kernel."""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.moe import router_topk_tpu as rt

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, K = 896, 16


def _logits(tokens=1, experts=E, dtype=torch.bfloat16):
    return torch.randn(tokens, experts).to(dtype)


def _bias(experts=E, dtype=torch.float32):
    return torch.randn(experts).to(dtype)


class TestRouterTopkHooks(CustomTestCase):
    def setUp(self):
        self.kernel_out = ("kernel_w", "kernel_ids")
        self.original = mock.Mock(return_value=("sort_w", "sort_ids"))
        patches = [
            mock.patch.object(rt, "_OP", [lambda *a: self.kernel_out]),
            mock.patch.object(rt, "_OP_READY", True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _grouped(self, logits=None, bias=None, num_expert_group=1, topk_group=1):
        logits = _logits() if logits is None else logits
        bias = _bias() if bias is None else bias
        return rt.biased_grouped_topk_around(
            self.original, logits, logits, bias, K, True, num_expert_group, topk_group
        )

    def test_kimi_k3_shape_takes_the_kernel(self):
        self.assertEqual(self._grouped(), self.kernel_out)
        self.assertEqual(
            rt.biased_topk_around(self.original, None, _logits(), _bias(), K, True),
            self.kernel_out,
        )
        self.original.assert_not_called()

    def test_everything_else_keeps_the_sort(self):
        cases = {
            "unvalidated expert count (Kimi-K2.x)": dict(
                logits=_logits(experts=384), bias=_bias(384)
            ),
            "bf16 bias": dict(bias=_bias(dtype=torch.bfloat16)),
            "per-token bias rows": dict(bias=torch.randn(1, E)),
            "several groups": dict(num_expert_group=8, topk_group=4),
            "more than MAX_TOKENS": dict(logits=_logits(tokens=rt.MAX_TOKENS + 1)),
        }
        for name, case in cases.items():
            with self.subTest(name):
                self.original.reset_mock()
                self.assertEqual(self._grouped(**case), ("sort_w", "sort_ids"))
                self.original.assert_called_once()

    def test_large_batch_matches_the_sort(self):
        # GLM-5.3-Flash prefill: top-8 of 288, one group, past the kernel's MAX_TOKENS.
        from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

        logits, bias = _logits(tokens=rt.MAX_TOKENS + 7, experts=288), _bias(288)
        args = (logits, logits, bias, 8, True, 1, 1, 0, 2.5, True)
        w, ids = rt.biased_grouped_topk_around(self.original, *args)
        self.original.assert_not_called()
        # The eager function, as TPU runs it: on CPU the module-level
        # torch.compile (Inductor) keeps the bf16 sigmoid in f32 and can flip
        # near-ties.
        eager = getattr(
            biased_grouped_topk_impl,
            "_torchdynamo_orig_callable",
            biased_grouped_topk_impl,
        )
        ref_w, ref_ids = eager(*args)
        order, ref_order = ids.argsort(dim=-1), ref_ids.argsort(dim=-1)
        torch.testing.assert_close(ids.gather(1, order), ref_ids.gather(1, ref_order))
        torch.testing.assert_close(w.gather(1, order), ref_w.gather(1, ref_order))

    def test_softmax_scoring_keeps_the_sort(self):
        out = rt.biased_topk_around(
            self.original, None, _logits(), _bias(), K, True, scoring_func="softmax"
        )
        self.assertEqual(out, ("sort_w", "sort_ids"))

    def test_unregistered_op_keeps_the_sort(self):
        with mock.patch.object(rt, "_OP_READY", False):
            self.assertEqual(self._grouped(), ("sort_w", "sort_ids"))

    def test_switch_off_leaves_the_op_unregistered(self):
        with (
            tpu_envs.SGLANG_TPU_ENABLE_ROUTER_TOPK_KERNEL.override(False),
            mock.patch.object(rt, "_OP_READY", False),
            mock.patch.object(rt, "_build_op") as build,
        ):
            rt.register_router_topk_op()
            self.assertFalse(rt._OP_READY)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
