"""TPU CI entry for the diffusion torch.compile arguments, graph-break and
recompile tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.utils import torch_compile
from sglang.multimodal_gen.test.unit.test_compile_graph_breaks import (  # noqa: F401
    TestCompileGraphBreaks,
    TestDenoisingStepTimestep,
    TestPlatformCompileGates,
    TestWanStepCounter,
)
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform


class TestTpuCompileKwargs(unittest.TestCase):
    """What the TPU platform compiles with: its own backend, no mode."""

    def test_tpu_compiles_with_the_platform_backend(self):
        with patch.object(torch_compile, "current_platform", TpuOOTPlatform()):
            compile_kwargs = torch_compile.build_torch_compile_kwargs(
                mode="max-autotune-no-cudagraphs",
                module=torch.nn.Module(),
                enable_inductor_compute_comm_overlap=True,
            )
        self.assertEqual(
            compile_kwargs, {"backend": "tpu", "dynamic": None, "fullgraph": False}
        )


if __name__ == "__main__":
    unittest.main()
