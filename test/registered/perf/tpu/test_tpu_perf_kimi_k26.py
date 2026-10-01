"""Throughput sweep for Kimi-K2.6 on the 8k1k workload, tp8 / one host.

tpu/nightly-perf/test_tpu_perf_gpt_oss_120b.py carries the ADDING A MODEL guide;
the DeepSeek siblings here carry the reasoning behind a single-host tp8 perf serve of a
frontier-size MoE. What is specific to this model:

  * native int4 (compressed-tensors) experts, so QUANTIZATION stays "unquant" --
    asking for fp8 would silently re-quantize them. The recipe owns that.
  * TEXT ONLY. MoonViT is part of the checkpoint but no vision prefill shape is
    compiled and the server warmup is skipped; this measures the text serve.
  * the tightest HBM of the three: ~82 GB of int4 experts per rank at tp8
    (40.9 GB measured at tp16), leaving the KV pool and the 8192-token extend
    graph to share the rest.

FIRST RUN IS A VALIDATION, NOT A DATAPOINT. Every other number about this model
was measured at tp16 across two hosts. If the job dies in bring-up, lower CONCUR
(that is the pool) before touching MEM_FRACTION_STATIC (the weights still have
to fit), and re-anchor est_time from the run that works.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-perf-kimi-k26",
    nightly=True,
    labels=["singlehost", "kimi-k2.6", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2400"))

WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

# No public HF id for this checkpoint, so the mount path is the served name too.
# The dashboard joins the series on `id` regardless.
WEIGHTS = "/weights/Kimi-K2.6"


MODEL_CONFIG = {
    "model": WEIGHTS,
    "test_name": "kimi-k26",
    "id": "perf-kimi-k26",
    "display_name": "Kimi K2.6",
    "variant": "TP8+EP8-INT4-FP8KV",
    "launch": {
        "config": "recipe/configs/kimi_k26/perf.py",
        "env": {"MODEL": WEIGHTS},
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-kimi-k26",
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
                # random-ids + exact input_ids, matching the harness the
                # published TPU numbers come from; a tokenizer round-trip would
                # otherwise move the real ISL.
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                "MODEL": WEIGHTS,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
