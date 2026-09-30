"""Throughput sweep for DeepSeek-V3.1 on the 8k1k workload, tp8 / one host.

Same backbone, same recipe and same shape as its R1-0528 sibling
(test_tpu_perf_deepseek_r1_0528.py) -- read that file for why tp8, why fp8 KV
and why CONCUR=16 is the KV pool's ceiling rather than a client choice. Only the
checkpoint differs, so the two series are comparable point for point.

WHY ITS OWN FILE, not a second sweep point on R1: one file per model is what
makes them separate nightly jobs on separate slices. Sharing a suite would put
them in one sequential job, where a slow model delays the other and a failing
one costs it its measurement.

The weights live on their own PVC (deepseek-v3-1-rox), not the shared
model-weights claim -- the launcher mounts exactly one PVC at /weights, so the
matrix entry in perf-test-tpu.yml names it.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-perf-deepseek-v3-1",
    nightly=True,
    labels=["singlehost", "deepseek-v3-1", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2400"))

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "deepseek-ai/DeepSeek-V3.1"
WEIGHTS = "/weights/DeepSeek-V3.1"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "deepseek-v3-1",
    "id": "perf-deepseek-v3-1",
    "display_name": "DeepSeek V3.1",
    "variant": "TP8+EP8-FP8KV",
    "launch": {
        # The R1-0528 recipe: V3.1 is the same 671B blockwise-fp8 architecture,
        # and the only per-checkpoint delta is MODEL.
        "recipe": "recipe/deepseek_v3_perf_tp8.sh",
        "env": {
            "MODEL": WEIGHTS,
            "EXTRA_ARGS": f"--served-model-name {MODEL}",
            "ISL": str(ISL),
            "OSL": str(OSL),
            "CONCUR": str(CONCUR),
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-v3-1",
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
