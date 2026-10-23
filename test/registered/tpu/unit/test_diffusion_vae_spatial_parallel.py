"""TPU CI entry for the multimodal_gen spatial-parallel VAE tests.

The multi-rank cases run each rank on a thread over an in-process decode group
(collectives only, no process group or ports).
"""

import unittest

from sglang.multimodal_gen.test.unit.test_decoding_stage_parallelism import (  # noqa: F401
    TestDecodeUsesEveryRank,
    TestMovaDecodingStageParallelism,
)
from sglang.multimodal_gen.test.unit.test_vae_spatial_parallel_decode import (  # noqa: F401
    TestVAESpatialParallelDecode,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

if __name__ == "__main__":
    unittest.main()
