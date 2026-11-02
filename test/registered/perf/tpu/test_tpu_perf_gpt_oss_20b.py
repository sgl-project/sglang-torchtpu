"""Throughput sweep for GPT-OSS-20B on the 8k1k workload, tp8/ep8, one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide
and the reason a perf run serves from its own config; read it first. The
serving config lives in recipe/configs/gpt_oss_20b/perf.py, which derives from
the accuracy recipe next to it, so a serving change there moves this trend
with it.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=480,
    suite="nightly-tpu-perf-gpt-oss-20b",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_20b/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Weight load plus one prefill and one decode capture.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "openai/gpt-oss-20b"
WEIGHTS = "/weights/gpt-oss-20b"

MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-20b",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-gpt-oss-20b",
    "display_name": "GPT-OSS 20B",
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
                # random-ids + exact input_ids, as the InferenceX harness sends them;
                # a tokenizer round-trip would move the real ISL.
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


class TestGptOss20BPerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-gpt-oss-20b"
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
