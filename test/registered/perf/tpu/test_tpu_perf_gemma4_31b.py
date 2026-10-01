"""Throughput sweep for Gemma-4-31B-it on the 8k1k workload, tp8 / one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide
and the reason a perf run serves from its own recipe; read it first. What is
specific to Gemma 4:

  * bf16 weights and bf16 KV. ~62 GB of weights is ~8 GB per rank, so unlike the
    frontier MoEs on this lane the KV pool is not what the weights left over --
    it is deliberately BOUNDED (max_total_tokens), because compile materializes
    the pool as graph args and outputs.
  * hybrid attention: 10 global layers at head_dim 512 with K == V, 50 sliding
    at 256, which is two KV pools and two sets of graph captures. That is what
    the long health timeout is for.
  * TEXT ONLY as measured here. The checkpoint serves images, but a random-ids
    sweep sends none, so the ViT never runs and no vision prefill shape is
    compiled.

CONCURRENCY and the pool live in recipe/configs/gemma4_31b/perf.py, which
derives from the serving config next to it -- a tuning change there moves this
trend with it, which is the point of not restating the settings here.

DEPENDS ON #944 (Gemma-4-31B TPU support and its serving config). Until that
lands this job has nothing to boot; the file is here so the wiring lands with
it.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-perf-gemma4-31b",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Weight load plus the EXTEND/decode captures for BOTH hybrid KV pools.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape gpt-oss-120b and the DeepSeek pair are
# measured at, so the models sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "google/gemma-4-31B-it"
WEIGHTS = "/weights/gemma-4-31B-it"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gemma4-31b",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-gemma4-31b",
    "display_name": "Gemma 4 31B",
    "variant": "TP8",
    "launch": {
        # Thin shell entry point over the Python config; MODEL is the only
        # delta the measurement supplies, since the config owns the shape.
        "recipe": "recipe/gemma4_31b_perf_tp8.sh",
        "env": {"MODEL": WEIGHTS},
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gemma4-31b",
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
                # requests still carry the name the server serves under.
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
