"""TPU CI entry for the diffusion platform fusion barrier."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.test.unit.test_spmd_safe_regions import (  # noqa: F401
    TestFusionBarrier,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform


class TestTpuFusionBarrier(unittest.TestCase):
    # Only .device.type is read, so a stand-in needs no TPU device.
    _TPU_TENSOR = SimpleNamespace(device=SimpleNamespace(type="tpu"))

    def test_host_tensors_pass_through(self):
        x = torch.arange(4.0)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertIs(TpuOOTPlatform.fusion_barrier(x), x)

    def test_eager_tpu_tensors_pass_through(self):
        x = self._TPU_TENSOR
        self.assertIs(TpuOOTPlatform.fusion_barrier(x), x)

    def test_compiled_tpu_tensors_take_the_torch_tpu_barrier(self):
        x = self._TPU_TENSOR
        public = MagicMock(return_value=["public"])
        raw = MagicMock(return_value=["raw"])
        for tpu_module, want in (
            (
                SimpleNamespace(ops=SimpleNamespace(optimization_barrier=public)),
                "public",
            ),
            (SimpleNamespace(), "raw"),
        ):
            with (
                self.subTest(want=want),
                patch.object(torch, "tpu", tpu_module, create=True),
                patch.object(
                    torch.ops, "tpu", SimpleNamespace(optimization_barrier=raw)
                ),
                patch.object(torch.compiler, "is_compiling", return_value=True),
            ):
                self.assertEqual(TpuOOTPlatform.fusion_barrier(x), want)
        public.assert_called_once_with([x])
        raw.assert_called_once_with([x])


class TestFusionBarrierOnDevice(CustomTestCase):
    def test_values_pass_through_eager_and_a_compiled_graph(self):
        import torch_tpu._internal.compile  # noqa: F401
        from torch._dynamo.backends.registry import lookup_backend

        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        x = (torch.randn(256, 512) * 3).bfloat16().to("tpu")
        # Eager runs op by op: there is no fusion to block, so no copy either.
        self.assertIs(TpuOOTPlatform.fusion_barrier(x), x)

        targets = []
        tpu_backend = lookup_backend("tpu")

        def recording_backend(gm, example_inputs):
            targets.extend(str(node.target) for node in gm.graph.nodes)
            return tpu_backend(gm, example_inputs)

        def mlp_act(t):
            h = torch.nn.functional.gelu(t, approximate="tanh")
            return TpuOOTPlatform.fusion_barrier(h) * 2

        compiled = torch.compile(
            mlp_act, backend=recording_backend, dynamic=False, fullgraph=True
        )
        torch.testing.assert_close(
            compiled(x).cpu(), mlp_act(x).cpu(), rtol=1.6e-2, atol=1e-2
        )
        self.assertTrue(any("optimization_barrier" in t for t in targets), targets)


if __name__ == "__main__":
    unittest.main()
