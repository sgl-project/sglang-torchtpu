"""Mode B (SGLANG_PLATFORM=tpu_oot) run of test_tpu_qwen3_0_6b_singlehost_smoke.py.

Covers dense MHA through the OOT KV pool and allocator arms.
"""

import os
import runpy

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=80,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "smoke", "tpu-oot"],
)

if __name__ == "__main__":
    # Overrides the jobset's SGLANG_PLATFORM=tpu; the launched server inherits it.
    os.environ["SGLANG_PLATFORM"] = "tpu_oot"
    runpy.run_path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_tpu_qwen3_0_6b_singlehost_smoke.py",
        ),
        run_name="__main__",
    )
