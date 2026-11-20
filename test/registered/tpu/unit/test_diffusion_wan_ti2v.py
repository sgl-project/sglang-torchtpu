"""TPU CI entry for the multimodal_gen Wan TI2V helper tests."""

import unittest

from sglang.multimodal_gen.test.unit.test_wan_ti2v_helpers import (  # noqa: F401
    TestWanTI2VHelpers,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

if __name__ == "__main__":
    unittest.main()
