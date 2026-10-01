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
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
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
        # Imports the R1-0528 config: same 671B blockwise-fp8 architecture, so
        # the only per-checkpoint deltas are the path and the served name.
        "config": "recipe/configs/deepseek_v3_1/perf.py",
        "env": {"MODEL": WEIGHTS},
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
