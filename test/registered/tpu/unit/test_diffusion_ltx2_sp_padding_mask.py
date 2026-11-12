"""TPU CI entry for the LTX-2 sequence-parallel padding-mask tests.

Host-only checks, registered here because the CPU CI image does not install the
diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_ltx2_sp_padding_mask import (  # noqa: F401
    TestLtx2SpPaddingMask,
)

if __name__ == "__main__":
    unittest.main()
