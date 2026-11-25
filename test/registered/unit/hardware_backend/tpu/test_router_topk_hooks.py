"""CPU checks of which router top-k calls the TPU hooks hand to the kernel."""

import types
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

    def test_prefill_batches_take_the_kernel(self):
        """The gridded kernel has no token cap: GLM-5.3-Flash and Kimi-K3
        prefill chunks route through it as decode does."""
        for tokens, experts in ((4096, 288), (65, E)):
            with self.subTest(tokens=tokens, experts=experts):
                logits = _logits(tokens=tokens, experts=experts)
                self.assertEqual(
                    self._grouped(logits=logits, bias=_bias(experts)), self.kernel_out
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
            "no tokens (empty grid)": dict(logits=_logits(tokens=0)),
        }
        for name, case in cases.items():
            with self.subTest(name):
                self.original.reset_mock()
                self.assertEqual(self._grouped(**case), ("sort_w", "sort_ids"))
                self.original.assert_called_once()

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


class TestMuseSparkRouteGating(CustomTestCase):
    def test_switch_off_restores_the_sort_at_every_batch_size(self):
        """SGLANG_TPU_ENABLE_ROUTER_TOPK_KERNEL=0 must take Muse back to the sort
        for prefill as well as decode; on, every non-empty batch takes the kernel."""
        from sglang_tpu.models import muse_spark as ms

        cases = (
            (False, 8, "sort"),
            (False, 1024, "sort"),
            (True, 0, "sort"),
            (True, 8, "kernel"),
            (True, 65, "kernel"),
            (True, 4096, "kernel"),
        )
        for enabled, tokens, want in cases:
            with self.subTest(enabled=enabled, tokens=tokens):
                moe = types.SimpleNamespace(
                    gate=lambda _x, t=tokens: (torch.randn(t, 256), None),
                    router_multiplier=1.0,
                    e_score_correction_bias=torch.zeros(256),
                    top_k=8,
                    route_eps=ms._ROUTE_EPS,
                    route_scale=1.0,
                )
                ids = torch.zeros(tokens, 8, dtype=torch.int32)
                kernel = mock.Mock(return_value=(torch.zeros(tokens, 8), ids))
                with (
                    mock.patch.object(rt, "_OP_READY", enabled),
                    mock.patch.object(ms, "sigmoid_router_topk", kernel),
                ):
                    ms.MuseSparkMoE._route(moe, None)
                self.assertEqual("kernel" if kernel.called else "sort", want)


if __name__ == "__main__":
    unittest.main()
