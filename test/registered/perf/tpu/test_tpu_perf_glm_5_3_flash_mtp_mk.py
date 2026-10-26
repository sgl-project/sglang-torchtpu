"""Throughput for GLM-5.3-Flash MTP with the decode megakernel, 8k1k at c1, tp8 / one host.

The megakernel decodes one request at a time, so this series has a single c1 point.

Serving config lives in recipe/configs/glm_5_3_flash/perf_mtp_mk.py, which derives
from mtp_mk.py next to it.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

# Healthy 371 s on tpu7x with weights in page cache (CI loads them in 905-1113 s),
# then the sweep takes 19 s.
register_tpu_ci(
    est_time=1800,
    suite="nightly-tpu-perf-glm-5-3-flash-mtp-mk",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "spec", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/glm_5_3_flash/perf_mtp_mk.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Weight load measured 905-1113 s in CI, then the graph compiles.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 1

MODEL = "zai-org/GLM-5.3-Flash"
WEIGHTS = "/weights/GLM-5.3-Flash"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "glm-5-3-flash-mtp-mk",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-glm-5-3-flash-mtp-mk",
    "display_name": "GLM-5.3-Flash",
    "variant": "TP8-DSA-FP8KV-MTP3-Megakernel",
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


class TestGlm53FlashMtpMegakernelPerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-glm-5-3-flash-mtp-mk"
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
