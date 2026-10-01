"""Throughput sweep for DeepSeek-R1-0528 on the 8k1k workload, tp8 / one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide
and the reason a perf run serves from its own recipe rather than the accuracy
nightly's; read it first. The deltas here are the ones a 671B MLA model forces:

  * tp8 on ONE host, not the accuracy nightly's 2-host tp16. A perf job takes a
    single-host slice, and this is the largest model that fits one: ~80 GB of
    fp8 weights per rank, of the 101.7 GB a rank owns.
  * CONCUR=16 is the POOL's ceiling, not a client choice. What the weights leave
    is a ~175k-token fp8 KV pool (measured 2026-09-04), and 16 x CONTEXT_LEN
    9280 = 148k of it. c64 -- the shape gpt-oss also measures -- would need
    594k, so there is no c64 sibling for this model.
  * fp8 MLA KV. bf16 KV halves the pool, which would put even c16 out of reach.

V3.1 runs the same recipe against its own checkpoint
(test_tpu_perf_deepseek_v3_1.py); the two trends are comparable point for point.

est_time is a first estimate, not a measurement: ~5 min to stream 642 GB off the
weights PVC, a cold compile of the single prefill + decode bucket, then the
sweep leg. It drives the JobSet poll budget, so re-anchor it once a night has
run rather than trimming it.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-perf-deepseek-r1-0528",
    nightly=True,
    labels=["singlehost", "deepseek-r1-0528", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# 642 GB of weights: the health wait covers the load, not just the compile.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2400"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown -- the same
# shape gpt-oss-120b is measured at, so the models sit on one axis.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

# Served from the shared weights PVC at /weights; the canonical HF id is what the
# server is told to serve as and what the dashboard keys on, since a mount path
# would split the trend the day the weights move.
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
    "launch": {
        # Serving config, launched through benchmark.tpu.launch_server. It
        # owns the shape; a `config` launch reads only MODEL/HOST/PORT from env.
        "config": "recipe/configs/deepseek_r1_0528/perf.py",
        "env": {"MODEL": WEIGHTS},
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-r1-0528",
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
                # random-ids + exact input_ids, matching the InferenceX harness
                # the published TPU numbers come from; a tokenizer round-trip
                # would otherwise move the real ISL.
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                # Tokenizer from the local checkpoint (no HF fetch in the pod);
                # requests still carry the HF id served above.
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
