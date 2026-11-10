"""TPU CI entry for the core tests of the regional-compile hooks: SPMD-safe
regions, the fusion barrier and the per-request drain.

Host-only checks, registered here because the CPU CI image does not install
the diffusion runtime. TpuOOTPlatform's side is in
test_diffusion_tpu_oot_platform.py.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_spmd_safe_regions import (  # noqa: F401
    TestDrainAfterRequest,
    TestFusionBarrier,
    TestSpmdSafeRegions,
    TestWanBlockCompilesAsOneGraph,
    TestWanBlockInputsAreRankInvariant,
)

if __name__ == "__main__":
    unittest.main()
