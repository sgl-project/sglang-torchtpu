"""Backend routing must retain fallback, bias overrides and shared postprocessing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as tk
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _router(*, router_logits, correction_bias, topk_config, **kwargs):
    if topk_config.scoring_func != "sigmoid":
        return None
    scores = router_logits.sigmoid()
    ids = (scores + correction_bias).topk(topk_config.top_k).indices.flip(-1)
    weights = scores.gather(1, ids)
    if topk_config.renormalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights.float(), ids.int()


class TestBiasedTopkRegistry(CustomTestCase):
    def setUp(self):
        self.enterContext(patch.object(tk, "_BIASED_TOPK_IMPLS", {}))
        self.enterContext(get_context().override_server_args(model_path="dummy"))
        self.enterContext(patch.object(tk, "_use_aiter", False))
        self.enterContext(patch.object(tk, "_is_cuda", False))
        self.enterContext(patch.object(tk, "_is_hip", False))
        self.enterContext(patch.object(tk, "_is_tpu", False))
        self.enterContext(
            patch(
                "sglang.srt.platforms.current_platform",
                SimpleNamespace(get_dispatch_key_name=lambda: "test_plugin"),
            )
        )
        self.logits = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
        self.bias = torch.zeros(8)
        self.config = tk.TopKConfig(
            top_k=2,
            correction_bias=self.bias,
            scoring_func="sigmoid",
            torch_native=True,
            allow_routed_experts_capture=False,
        )

    def _select(self, **kwargs):
        return tk.select_experts(
            self.logits, self.logits, self.config, layer_id=0, **kwargs
        )

    def test_backend_receives_effective_bias_and_retains_padding(self):
        """A backend must not bypass padding or read the stale config bias."""
        tk.register_biased_topk_impl("test_plugin", _router)
        dynamic_bias = torch.tensor([10.0, 9.0, 0, 0, 0, 0, 0, 0])
        with patch.object(tk, "_is_tpu", True):
            result = self._select(
                dynamic_expert_bias=dynamic_bias,
                num_token_non_padded=torch.tensor(2),
            )
        weights, ids = _router(
            router_logits=self.logits,
            correction_bias=dynamic_bias,
            topk_config=self.config,
        )
        torch.testing.assert_close(result.topk_ids[:2], ids[:2])
        torch.testing.assert_close(result.topk_weights[:2], weights[:2])
        self.assertTrue(torch.all(result.topk_ids[2:] == -1))
        self.assertTrue(torch.all(result.topk_weights[2:] == 0))
        self.assertIs(result.router_logits, self.logits)

    def test_missing_or_declining_backend_preserves_builtin_result(self):
        self.config.scoring_func = "softmax"
        expected = self._select()
        tk.register_biased_topk_impl("other_plugin", _router)
        unmatched = self._select()
        tk.register_biased_topk_impl("test_plugin", _router)
        declined = self._select()
        for result in (unmatched, declined):
            torch.testing.assert_close(result.topk_ids, expected.topk_ids)
            torch.testing.assert_close(result.topk_weights, expected.topk_weights)

    def test_custom_router_takes_precedence(self):
        def custom(**kwargs):
            return torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int32)

        self.config.custom_routing_function = custom
        self.config.torch_native = False
        tk.register_biased_topk_impl("test_plugin", _router)
        result = self._select()
        self.assertTrue(torch.all(result.topk_ids == 0))
        self.assertTrue(torch.all(result.topk_weights == 1))

    def test_backend_dispatch_traces_without_graph_breaks(self):
        tk.register_biased_topk_impl("test_plugin", _router)

        def route(logits):
            return tk.select_experts(logits, logits, self.config, layer_id=0).topk_ids

        expected = route(self.logits)
        compiled = torch.compile(route, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(self.logits), expected)

    def test_registration_rejects_conflicts_without_replacing_backend(self):
        tk.register_biased_topk_impl("test_plugin", _router)
        tk.register_biased_topk_impl("test_plugin", _router)
        with self.assertRaisesRegex(ValueError, "already registered"):
            tk.register_biased_topk_impl("test_plugin", lambda **kw: None)
        result = self._select()
        self.assertTrue(torch.equal(result.topk_ids[0], torch.tensor([6, 7])))


if __name__ == "__main__":
    unittest.main()
