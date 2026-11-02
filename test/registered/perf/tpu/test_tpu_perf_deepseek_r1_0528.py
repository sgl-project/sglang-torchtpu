"""Throughput sweep for DeepSeek-R1-0528 on the 8k1k workload, tp8 / one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide
and the reason a perf run serves from its own config rather than the accuracy
nightly's; read it first. The deltas here are the ones a 671B MLA model forces:

  * tp8 on ONE host, not the accuracy nightly's 2-host tp16. A perf job takes a
    single-host slice, and this is the largest model that fits one: ~80 GB of
    fp8 weights per rank, of the 101.7 GB a rank owns.
  * CONCURRENCY=16 is the POOL's ceiling, not a client choice. What the weights
    leave is a ~175k-token fp8 KV pool (measured 2026-09-04), and 16 x the
    9280-token context is 148k of it. c64 -- the shape gpt-oss also measures --
    would need 594k, so there is no c64 sibling for this model.
  * fp8 MLA KV. bf16 KV halves the pool, which would put even c16 out of reach.

Serving config lives in recipe/configs/deepseek_r1_0528/perf.py, which CI runs
verbatim; V3.1 imports it (test_tpu_perf_deepseek_v3_1.py), so the two trends
are comparable point for point.

est_time is a first estimate, not a measurement: ~5 min to stream 642 GB off the
weights PVC, a cold compile of the single prefill and decode bucket, then the
sweep leg. It drives the JobSet poll budget, so re-anchor it once a night has
run rather than trimming it.
"""

import os
import unittest
from typing import ClassVar
from urllib.parse import urlsplit

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import run_model_config
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=2410,
    suite="nightly-tpu-perf-deepseek-r1-0528",
    nightly=True,
    labels=["singlehost", "deepseek-r1-0528", "perf"],
)

LAUNCH_CONFIG_MODULE = "recipe/configs/deepseek_r1_0528/perf.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# 642 GB of weights: the health wait covers the load, not just the compile.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2400"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown -- the same
# shape gpt-oss-120b is measured at, so the models sit on one axis.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

# The canonical HF id is what the config serves under and what the dashboard
# keys on; the checkpoint itself is the mount the config points at, since a
# mount path would split the trend the day the weights move.
MODEL = "deepseek-ai/DeepSeek-R1-0528"
WEIGHTS = "/weights/DeepSeek-R1-0528"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "deepseek-r1-0528",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-deepseek-r1-0528",
    "display_name": "DeepSeek R1-0528",
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


class TestDeepSeekR1PerfSingleHost(TPULauncherTestCase):
    config_module = LAUNCH_CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "perf-r1-0528"
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
