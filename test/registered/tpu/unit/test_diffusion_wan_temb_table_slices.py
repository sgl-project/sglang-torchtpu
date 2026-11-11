"""TPU CI entry for the Wan adaLN temb-table slice tests.

Host-only checks (the CUDA cases skip), registered here because the CPU CI image
does not install the diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_wan_temb_table_slices import (  # noqa: F401
    TestWanTembTableSlices,
)

if __name__ == "__main__":
    unittest.main()
