"""Mode B (SGLANG_PLATFORM=tpu_oot) run of test_tpu_gpt_oss_20b_singlehost_spec.py.

Covers hybrid-SWA pools, MoE, EAGLE3 and the speculative platform checks
across ranks.
"""

import os
import runpy

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=650,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "spec", "tpu-oot"],
)

if __name__ == "__main__":
    # Overrides the jobset's SGLANG_PLATFORM=tpu; the launched server inherits it.
    os.environ["SGLANG_PLATFORM"] = "tpu_oot"
    runpy.run_path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_tpu_gpt_oss_20b_singlehost_spec.py",
        ),
        run_name="__main__",
    )
