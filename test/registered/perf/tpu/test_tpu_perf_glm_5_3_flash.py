"""Throughput sweep for GLM-5.3-Flash on the 8k1k workload, tp8 / one host.

Measured at c1 and c16: the c1 point sits next to the decode-megakernel series
(test_tpu_perf_glm_5_3_flash_mk.py), which serves one request at a time.

Serving config lives in recipe/configs/glm_5_3_flash/perf.py, which derives from
server.py next to it, so a tuning change to the serve moves this trend with it.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-glm-5-3-flash",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/glm_5_3_flash/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Weight load measured 905-1113 s in CI, then the graph compiles.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
ISL, OSL = 8192, 1024
# The top point must match CONCURRENCY in the launch config: the server is sized
# for it, and a client driving more would measure a pool-bound shape.
CONCURRENCIES = (1, 16)

MODEL = "zai-org/GLM-5.3-Flash"
WEIGHTS = "/weights/GLM-5.3-Flash"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "glm-5-3-flash",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-glm-5-3-flash",
    "display_name": "GLM-5.3-Flash",
    "variant": "TP8-DSA-FP8KV",
    "gates": [
        {
            "dataset": "bench_serving",
            "label": WORKLOAD,
            "sweep": [
                {"workload": WORKLOAD, "isl": ISL, "osl": OSL, "concurrency": c}
                for c in CONCURRENCIES
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


class TestGlm53FlashPerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-glm-5-3-flash"
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
