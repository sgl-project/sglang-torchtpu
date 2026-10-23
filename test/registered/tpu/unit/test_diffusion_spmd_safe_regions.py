"""TPU CI entry for the SPMD-safe regional-compile tests.

Mostly host-only checks, registered here because the CPU CI image does not
install the diffusion runtime; the torch_tpu wrapper check needs torch_tpu.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_spmd_safe_regions import (  # noqa: F401
    TestSpmdSafeRegions,
    TestWanBlockInputsAreRankInvariant,
)

if __name__ == "__main__":
    unittest.main()
