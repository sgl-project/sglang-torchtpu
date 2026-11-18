"""TPU CI entry for the Wan VAE compiled decoder calls."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="nightly-tpu-4-v7", nightly=True)

import unittest

from sglang.multimodal_gen.test.unit.test_wan_vae_decode_calls import (  # noqa: F401
    TestWanVaeDecodeCalls,
)

if __name__ == "__main__":
    unittest.main()
