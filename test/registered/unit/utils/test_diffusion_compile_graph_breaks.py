"""CPU CI entry for the diffusion torch.compile graph-break and recompile tests
(backend="eager"); the TPU suite also runs them."""

import unittest

from sglang.multimodal_gen.test.unit.test_compile_graph_breaks import (  # noqa: F401
    TestCompileGraphBreaks,
    TestDenoisingStepTimestep,
    TestPlatformCompileGates,
    TestWanStepCounter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

if __name__ == "__main__":
    unittest.main()
