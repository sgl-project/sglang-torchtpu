"""TPU CI entry for the diffusion platform fusion barrier."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.test.unit.test_tpu_platform import (  # noqa: F401
    TestFusionBarrier,
)
from sglang.test.test_utils import CustomTestCase


class TestFusionBarrierOnDevice(CustomTestCase):
    def test_values_pass_through_eager_and_a_compiled_graph(self):
        import torch_tpu._internal.compile  # noqa: F401
        from torch._dynamo.backends.registry import lookup_backend

        TpuPlatform.set_device(TpuPlatform.get_local_torch_device())
        x = (torch.randn(256, 512) * 3).bfloat16().to("tpu")
        # Eager runs op by op: there is no fusion to block, so no copy either.
        self.assertIs(TpuPlatform.fusion_barrier(x), x)

        targets = []
        tpu_backend = lookup_backend("tpu")

        def recording_backend(gm, example_inputs):
            targets.extend(str(node.target) for node in gm.graph.nodes)
            return tpu_backend(gm, example_inputs)

        def mlp_act(t):
            h = torch.nn.functional.gelu(t, approximate="tanh")
            return TpuPlatform.fusion_barrier(h) * 2

        compiled = torch.compile(
            mlp_act, backend=recording_backend, dynamic=False, fullgraph=True
        )
        torch.testing.assert_close(
            compiled(x).cpu(), mlp_act(x).cpu(), rtol=1.6e-2, atol=1e-2
        )
        self.assertTrue(any("optimization_barrier" in t for t in targets), targets)


if __name__ == "__main__":
    unittest.main()
