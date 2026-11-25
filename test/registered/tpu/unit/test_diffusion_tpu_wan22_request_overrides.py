"""TPU CI entry for the Wan2.2 per-request boundary_ratio and flow_shift tests."""

import unittest

from sglang.multimodal_gen.test.unit.test_wan22_request_overrides import (  # noqa: F401
    TestWan22RequestOverrides,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

if __name__ == "__main__":
    unittest.main()
