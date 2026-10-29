"""TPU CI entry for the Wan sequence-parallel padding tests.

Host-only checks, registered here because the CPU CI image does not install the
diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_usp_attention_kv_gather import (  # noqa: F401
    TestUSPAttentionKVGather,
)
from sglang.multimodal_gen.test.unit.test_usp_tail_pad_key_prefix import (  # noqa: F401
    TestTailPadKeyPrefix,
)
from sglang.multimodal_gen.test.unit.test_wan_sp_padding import (  # noqa: F401
    TestWanSequenceShardPadding,
)

if __name__ == "__main__":
    unittest.main()
