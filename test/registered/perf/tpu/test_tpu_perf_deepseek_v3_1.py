"""Throughput sweep for DeepSeek-V3.1 on the 8k1k workload, tp8 / one host.

Same backbone, same config and same shape as its R1-0528 sibling
(test_tpu_perf_deepseek_r1_0528.py) -- read that file for why tp8, why fp8 KV
and why CONCURRENCY=16 is the KV pool's ceiling rather than a client choice.
recipe/configs/deepseek_v3_1/perf.py imports R1's and changes the checkpoint,
so the two series are comparable point for point.

WHY ITS OWN FILE, not a second sweep point on R1: one file per model is what
makes them separate nightly jobs on separate slices. Sharing a suite would put
them in one sequential job, where a slow model delays the other and a failing
one costs it its measurement.

The weights live on their own PVC (deepseek-v3-1-rox), not the shared
model-weights claim -- the launcher mounts exactly one PVC at /weights, so the
matrix entry in perf-test-tpu.yml names it.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=1690,
    suite="nightly-tpu-perf-deepseek-v3-1",
    nightly=True,
    labels=["singlehost", "deepseek-v3-1", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/deepseek_v3_1/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2400"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
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
                # random-ids + exact input_ids, matching the InferenceX harness
                # the published TPU numbers come from; a tokenizer round-trip
                # would otherwise move the real ISL.
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                # Tokenizer from the local checkpoint (no HF fetch in the pod);
                # requests still carry the name the server serves under.
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


class TestDeepSeekV31PerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-v3-1"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        # Drop an inherited MODEL so the config's own checkpoint path wins.
        "MODEL": None,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_throughput(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
