"""gpt-oss-120b + EAGLE3 c64 throughput sweep on the experimental spec worker.

Reuses test_tpu_perf_gpt_oss_120b_eagle3_c64.py with
SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1, which routes EAGLE3 to TpuEagle3Worker.
Read the two side by side to compare the workers; the default-worker file keeps
its own trend and its pairing with the non-speculative c64 sweep.
"""

import copy

import test_tpu_perf_gpt_oss_120b_eagle3_c64 as base

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-gpt-oss-120b-exp-c64",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "perf",
        "spec",
        "eagle3",
        "experimental-spec",
    ],
)

MODEL_CONFIG = copy.deepcopy(base.MODEL_CONFIG)
MODEL_CONFIG.update(
    test_name="gpt-oss-120b-eagle3-exp-c64",
    id="perf-gpt-oss-120b-eagle3-exp-c64",
    display_name="GPT-OSS 120B EAGLE3 experimental worker (c64)",
    variant="TP8+EP8+EAGLE3-exp-c64",
)
MODEL_CONFIG["launch"]["tag"] = "perf-gpt-oss-eagle3-exp-c64"
MODEL_CONFIG["launch"]["env"]["SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER"] = "1"


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=base.BASE_URL)
