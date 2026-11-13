"""TPU CI entry for the multimodal_gen SANA VAE loader test."""

import unittest

from sglang.multimodal_gen.test.unit.test_vae_loader import (  # noqa: F401
    TestSanaVAELoad,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

if __name__ == "__main__":
    unittest.main()
