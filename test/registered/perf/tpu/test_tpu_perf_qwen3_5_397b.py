"""Throughput sweep for Qwen3.5-397B-A17B-FP8 on the 8k1k workload, tp8.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide.
What is specific to this model:

  * hybrid: 45 of 60 layers are GDN linear attention with a fixed-size recurrent
    state, 15 are full attention. The recurrent slots bill against the same token
    budget as KV, so the config sizes the pool for both and keeps four spare
    slots -- without them a replacement request waits on one specific request to
    finish (p99 TTFT 10797 ms vs 1108 ms, measured at c4).
  * TEXT ONLY, which is what makes the RoPE cache slice and skipping the ViT
    precompile safe.

Serving config lives in recipe/configs/qwen3_5_397b/perf.py, ported from a
measured serving recipe at the same 8k1k shape.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=1670,
    suite="nightly-tpu-perf-qwen3-5-397b",
    nightly=True,
    labels=["singlehost", "qwen3.5-397b", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/qwen3_5_397b/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
WEIGHTS = "/weights/Qwen3.5-397B-A17B-FP8"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "qwen3-5-397b",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-qwen3-5-397b",
    "display_name": "Qwen3.5 397B-A17B",
    "variant": "TP8+EP8-FP8",
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


class TestQwen35397BPerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-qwen3-5-397b"
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
