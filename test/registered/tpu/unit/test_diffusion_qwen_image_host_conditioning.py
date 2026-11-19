"""TPU CI entry for Qwen-Image's host-side conditioning tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_qwen_image_host_conditioning import (  # noqa: F401
    TestQwenImageHostConditioning,
)

if __name__ == "__main__":
    unittest.main()
