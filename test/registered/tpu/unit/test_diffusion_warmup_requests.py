"""TPU CI entry for the diffusion warmup request tests: warmup requests for
cfg-parallel and sequence-shard pipelines, and the cfg-parallel branch guards.

Host-only checks, registered here because the CPU CI image does not install the
diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_cfg_parallel_warmup import (  # noqa: F401
    TestCfgParallelServesOneBranch,
    TestFlux2FinetunedVaeEncodePreprocess,
    TestImageVaeEncodingLatentRetrieval,
    TestInputValidationCfgParallelSingleBranch,
    TestWarmupReqCfgParallel,
    TestWarmupSequenceShard,
)

if __name__ == "__main__":
    unittest.main()
