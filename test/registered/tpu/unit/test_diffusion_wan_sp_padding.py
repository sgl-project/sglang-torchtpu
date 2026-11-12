"""TPU CI entry for the Wan sequence-parallel padding tests and the
USPAttention dispatch tests they extend.

Host-only checks, registered here because the CPU CI image does not install the
diffusion runtime.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.multimodal_gen.test.unit.test_usp_all_valid_key_mask import (  # noqa: F401
    TestAllValidKeyMask,
)
from sglang.multimodal_gen.test.unit.test_usp_attention_kv_gather import (  # noqa: F401
    TestKVGatherCallSupport,
    TestReplicatedModeCountCompile,
    TestSpAttentionModeResolution,
    TestUlyssesAttentionKVGather,
    TestUSPAttentionKVGather,
)
from sglang.multimodal_gen.test.unit.test_usp_attention_replicated_prefix import (  # noqa: F401
    TestUSPAttentionMaskedReplicatedGuard,
    TestUSPAttentionMaskedReplicatedPrefix,
    TestUSPAttentionReplicatedPrefix,
)
from sglang.multimodal_gen.test.unit.test_usp_ulysses_padding import (  # noqa: F401
    TestUlyssesPadding,
)
from sglang.multimodal_gen.test.unit.test_wan_sp_padding import (  # noqa: F401
    TestWanSequenceShardPadding,
)

if __name__ == "__main__":
    unittest.main()
