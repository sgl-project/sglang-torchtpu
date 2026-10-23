"""TPU CI entry for the diffusion torch.compile cache-key tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_compile_cache_key import (  # noqa: F401
    TestCompileCacheHook,
    TestTpuCompileCacheKey,
)

if __name__ == "__main__":
    unittest.main()
