"""GPT-OSS-120B EAGLE3 single-host GSM8K quality gate on the experimental spec worker.

Reuses test_tpu_gpt_oss_120b_singlehost_spec.py with
SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1, which routes EAGLE3 to TpuEagle3Worker.
"""

import unittest
from typing import ClassVar

# Module import, not the class: unittest would also collect a bare base class.
import test_tpu_gpt_oss_120b_singlehost_spec as base

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "spec", "experimental-spec"],
)

MODEL_CONFIG = {
    **base.MODEL_CONFIG,
    "test_name": base.MODEL_CONFIG["test_name"] + "-experimental",
}


class TestGptOssSingleHostSpecExperimental(base.TestGptOssSingleHostSpec):
    tag = "gpt-oss-spec-gsm8k-experimental-e2e"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        **base.TestGptOssSingleHostSpec.runtime_envs,
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
    }

    def test_accuracy(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
