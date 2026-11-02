"""Throughput sweep for Kimi-K2.6 on the 8k1k workload, tp8 / one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide;
the DeepSeek siblings here carry the reasoning behind a single-host tp8 perf
serve of a frontier-size MoE. What is specific to this model:

  * native int4 (compressed-tensors) experts, so the config keeps quantization
    "unquant" -- asking for fp8 would silently re-quantize them.
  * TEXT ONLY. MoonViT is part of the checkpoint but no vision prefill shape is
    compiled and the server warmup is skipped; this measures the text serve.
  * the tightest HBM of the three: ~82 GB of int4 experts per rank at tp8
    (40.9 GB measured at tp16), leaving the KV pool and the 8192-token extend
    graph to share the rest.

Serving config lives in recipe/configs/kimi_k26/perf.py, which CI runs verbatim.

FIRST RUN IS A VALIDATION, NOT A DATAPOINT. Every other number about this model
was measured at tp16 across two hosts. If the job dies in bring-up, lower
CONCURRENCY (that is the pool) before mem_fraction_static (the weights still
have to fit), and re-anchor est_time from the run that works.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

# Measured in CI: 1734 s to healthy, then 160 requests at ~13.6 s.
register_tpu_ci(
    est_time=4410,
    suite="nightly-tpu-perf-kimi-k26",
    nightly=True,
    labels=["singlehost", "kimi-k2.6", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/kimi_k26/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# 555 GB of int4 experts stream from Filestore in 630-1160 s, then the precompile.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

# No public HF id for this checkpoint, so the mount is also the served name.
# The dashboard joins the series on `id` regardless.
WEIGHTS = "/weights/Kimi-K2.6"


MODEL_CONFIG = {
    "model": WEIGHTS,
    "test_name": "kimi-k26",
    "id": "perf-kimi-k26",
    "display_name": "Kimi K2.6",
    "variant": "TP8+EP8-INT4-FP8KV",
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
                # Tokenizer from the local checkpoint (no HF fetch in the pod).
                "MODEL": WEIGHTS,
            },
        }
    ],
}


class TestKimiK26PerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-kimi-k26"
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
