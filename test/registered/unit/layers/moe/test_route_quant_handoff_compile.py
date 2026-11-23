import unittest

import torch

from sglang.srt.layers.moe import route_quant_handoff
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRouteQuantHandoffCompile(CustomTestCase):
    def tearDown(self):
        route_quant_handoff.clear()
        torch._dynamo.reset()

    def test_unstaged_route_traces_in_one_graph(self):
        # select_experts runs these on every grouped-topk route; TPU compiles that
        # route with torch.compile, so a miss must not store to the handoff.
        def route(x):
            route_quant_handoff.bind_select_experts(None, fused_routing_ok=True)
            masked = route_quant_handoff.take_padded_rows_masked()
            return x if masked else x * 2

        torch._dynamo.reset()
        out = torch.compile(route, fullgraph=True, backend="eager")(torch.ones(2))
        self.assertTrue(torch.equal(out, torch.full((2,), 2.0)))

    def test_a_hit_is_taken_once(self):
        route_quant_handoff._handoff.padded_rows_masked = True
        self.assertTrue(route_quant_handoff.take_padded_rows_masked())
        self.assertFalse(route_quant_handoff.take_padded_rows_masked())


if __name__ == "__main__":
    unittest.main()
