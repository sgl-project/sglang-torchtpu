"""GPT-OSS-20B single-host EAGLE3 PR gate on the experimental spec worker.

Reuses test_tpu_gpt_oss_20b_singlehost_spec.py with
SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1, which routes EAGLE3 to TpuEagle3Worker.
"""

import copy

import test_tpu_gpt_oss_20b_singlehost_spec as base

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=175,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "spec", "experimental-spec"],
)

MODEL_CONFIG = copy.deepcopy(base.MODEL_CONFIG)
MODEL_CONFIG["test_name"] += "-experimental"
MODEL_CONFIG["launch"]["tag"] = "gpt-oss-20b-spec-experimental-e2e"
MODEL_CONFIG["launch"]["env"]["SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER"] = "1"


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=base.BASE_URL)
