"""Throughput sweep for Gemma-4-26B-A4B-it on the 8k1k workload, tp8+ep8.

The dense sibling (test_tpu_perf_gemma4_31b.py) carries the Gemma 4 notes --
bounded hybrid KV pools, two sets of graph captures, text-only measurement --
and this is the same sweep against the MoE checkpoint, so the two trends are
comparable point for point.

WHY ITS OWN FILE, not a second point on the 31B job: one file per model is what
makes them separate nightly jobs on separate slices. Sharing a suite would put
them in one sequential job, where a slow model delays the other and a failing
one costs it its measurement.

What the MoE checkpoint changes: EP = TP, because 704-wide expert intermediates
are not lane-splittable 8 ways, so each rank owns whole experts (128/8).

Serving config lives in recipe/configs/gemma4_26b_a4b/perf.py, which imports the
31B perf config.

DEPENDS ON #944 (Gemma 4 TPU support) AND on the 26B MoE padding work, which is
not yet in a PR. Until both land this job has nothing to boot.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=960,
    suite="nightly-tpu-perf-gemma4-26b-a4b",
    nightly=True,
    labels=["singlehost", "gemma4-26b-a4b", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/gemma4_26b_a4b/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Weight load plus the EXTEND/decode captures for BOTH hybrid KV pools.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "google/gemma-4-26B-A4B-it"
WEIGHTS = "/weights/gemma-4-26B-A4B-it"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gemma4-26b-a4b",
    "id": "perf-gemma4-26b-a4b",
    "display_name": "Gemma 4 26B-A4B",
    "variant": "TP8+EP8",
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


class TestGemma426BA4BPerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-gemma4-26b-a4b"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        # The config serves the HF id; CI mounts the checkpoint at /weights.
        "MODEL": WEIGHTS,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
    }

    def test_throughput(self):
        run_model_config(MODEL_CONFIG, base_url=self.base_url)


if __name__ == "__main__":
    unittest.main()
