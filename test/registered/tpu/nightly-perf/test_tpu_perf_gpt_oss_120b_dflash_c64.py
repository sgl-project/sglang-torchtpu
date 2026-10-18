"""Throughput sweep for gpt-oss-120b + DFlash on the 8k1k workload at c64.

The third leg of the c64 set: read against test_tpu_perf_gpt_oss_120b_c64.py for
the cost of speculation at all, and against the EAGLE3 sibling for which drafter
wins at 64-wide batching. All three serve the same model on the same workload at
the same concurrency, so the only free variable is the draft path.

See test_tpu_perf_gpt_oss_120b.py for the ADDING A MODEL guide.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args

# Longer than the non-speculative c64 file: bring-up also loads the draft model
# and compiles the draft forward. Drives the JobSet poll budget.
register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-gpt-oss-120b-dflash-c64",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf", "spec", "dflash"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

WORKLOAD = "8k1k"
CONFIG_MODULE = "recipe/configs/gpt_oss_120b/perf_dflash_c64.py"
ISL, OSL = 8192, 1024
# The server is sized for exactly this concurrency, so drive what it was
# sized for: the recipe's profile is the one source of the number.
CONCUR = recipe_server_args(CONFIG_MODULE)["max_running_requests"]

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-dflash-c64",
    "id": "perf-gpt-oss-120b-dflash-c64",
    "display_name": "GPT-OSS 120B DFlash (c64)",
    "variant": "TP8+EP8+DFLASH-c64",
    "launch": {
        # Deltas over the c64 baseline: the draft model and its block size. As
        # with EAGLE3, the draft weights share MEM_FRACTION_STATIC with the
        # target's KV pool -- tune that fraction on an OOM, not the pool size,
        # which has to match the baseline for the numbers to compare.
        "config": CONFIG_MODULE,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss-dflash-c64",
    },
    "gates": [
        {
            "dataset": "bench_serving",
            "label": WORKLOAD,
            "sweep": [
                {
                    "workload": WORKLOAD,
                    "isl": ISL,
                    "osl": OSL,
                    "concurrency": CONCUR,
                }
            ],
            "bench_env": {
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
