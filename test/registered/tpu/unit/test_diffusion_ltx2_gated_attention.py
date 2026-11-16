"""TPU CI entry for the LTX-2 gated attention layout test (LTX-2.3, JoyAI-Echo).

Host-only check, registered here because the CPU CI image does not install the
diffusion runtime. Whole node, not chips=1: the parallel-state setup runs the TPU
worker bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS
pin.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit")

import unittest

from sglang.multimodal_gen.test.unit.test_ltx2_gated_attention_layout import (  # noqa: F401
    TestLtx2GatedAttentionLayout,
)

if __name__ == "__main__":
    unittest.main()
