"""Mode B (SGLANG_PLATFORM=tpu_oot) run of test_tpu_kimi_k3_tiny_singlehost_compile.py.

Covers MLA attention, a linear-attention pool, the worker-side backend init
at TP4/EP4, and compiled serving.
"""

import os
import runpy

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=240,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "compile", "tpu-oot"],
)

if __name__ == "__main__":
    # Overrides the jobset's SGLANG_PLATFORM=tpu; the launched server inherits it.
    os.environ["SGLANG_PLATFORM"] = "tpu_oot"
    runpy.run_path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_tpu_kimi_k3_tiny_singlehost_compile.py",
        ),
        run_name="__main__",
    )
