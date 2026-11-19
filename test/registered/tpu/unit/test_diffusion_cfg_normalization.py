"""TPU CI entry for the core CFG combine tests: CFG normalization over the
whole latents when SP ranks each denoise a shard (Z-Image), without a host
read of the norms.

Host-only checks, registered here because the CPU CI image does not install
the diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_cfg_policy import (  # noqa: F401
    TestCfgNormalization,
    TestCFGPolicyCombine,
)

if __name__ == "__main__":
    unittest.main()
