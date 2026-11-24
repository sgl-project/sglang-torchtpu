"""TPU CI entry for the Wan VAE compiled decoder calls.

The sharded cases run 2 and 3 gloo processes on the host (halos by all-gather,
as on TPU) and take most of the time.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="nightly-tpu-4-v7", nightly=True)

import unittest

from sglang.multimodal_gen.test.unit.test_wan_vae_decode_calls import (  # noqa: F401
    TestPinnedSpatialParallelDecode,
    TestWanVaeDecodeCalls,
    TestWanVaeShardedDecodeCalls,
)

if __name__ == "__main__":
    unittest.main()
