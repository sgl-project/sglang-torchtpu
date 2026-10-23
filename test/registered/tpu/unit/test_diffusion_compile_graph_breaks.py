"""TPU CI entry for the diffusion torch.compile arguments, graph-break and
recompile tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_compile_graph_breaks import (  # noqa: F401
    TestCompileGraphBreaks,
    TestDenoisingStepTimestep,
    TestTpuCompileKwargs,
    TestWanStepCounter,
)

if __name__ == "__main__":
    unittest.main()
