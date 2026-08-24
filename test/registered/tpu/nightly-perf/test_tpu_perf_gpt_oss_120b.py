"""Throughput sweep for gpt-oss-120b on the 8k1k workload.

One file and one suite per model, so the models run as separate nightly jobs on
separate slices. See test_tpu_perf_qwen3_30b.py for the protocol on adding one.

This model differs from that one in two ways worth knowing before copying it: it
serves from a mounted checkpoint rather than an HF id, so the served name has to
be set explicitly in both the launch and the bench; and its recipe derives its
whole shape from ISL/OSL/CONCUR, so those go in `env` rather than a context
length being computed here.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

# One bring-up (weight load + per-shape compile, ~10 min with the single-bucket
# compile below) plus one sweep leg. Drives the JobSet poll budget, so an
# under-estimate kills the run. Not auto-refreshed: update_est_time.py omits tpu.
register_tpu_ci(est_time=1800, suite="nightly-tpu-perf-gpt-oss-120b", nightly=True)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown.
WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 16

# Serves from the shared ROX PVC (model-weights-rox at /weights), but the
# canonical HF id is what the server is told to serve as and what the dashboard
# keys the series on -- a mount path would be a poor key, since remounting
# elsewhere would silently split the trend in two.
MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b",
    "variant": "TP8+EP8",
    "launch": {
        # server.sh's own sample usage is `ISL=8192 OSL=1024 CONCUR=N ./server.sh`
        # -- it derives context, prefill chunk, KV pool and compile buckets from
        # exactly those three. Delta on top of the recipe:
        #   MODEL         the ROX PVC mount; server.sh defaults to the HF id.
        #   EXTRA_ARGS    serve under the canonical HF name (see MODEL).
        #   ISL/OSL/CONCUR  the shape. These give CONTEXT_LEN = 9280, an
        #                 8192-token prefill chunk (a cold 8k prompt is one
        #                 chunk), MAX_RUNNING/TORCH_COMPILE_MAX_BS = CONCUR, and
        #                 a 2*CONCUR*CONTEXT_LEN = 297k-token fp8 KV pool -- far
        #                 below the ~1.5M where its extend precompile OOMs.
        #   DISABLE_RADIX_CACHE  the recipe leaves prefix caching ON because it
        #                 is paired with a deliberate cache-hit sweep. This is
        #                 the opposite measurement: random-prompt throughput at
        #                 InferenceX parity, where a cache hit would let a
        #                 request skip a real prefill and drift the trend upward
        #                 for no serving reason.
        #   TPU_COMPILE_*  one prefill and one decode bucket, keeping bring-up
        #                 ~10 min; the derived default is powers of two up to
        #                 CONCUR. Same trade the accuracy nightly makes.
        #   ENABLE_PROFILER_ENDPOINT  a nightly captures no traces.
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": {
            "MODEL": WEIGHTS,
            "EXTRA_ARGS": f"--served-model-name {MODEL}",
            "ISL": str(ISL),
            "OSL": str(OSL),
            "CONCUR": str(CONCUR),
            "DISABLE_RADIX_CACHE": "1",
            "TPU_COMPILE_TOKENS": str(ISL),
            "TPU_COMPILE_BS": str(CONCUR),
            "ENABLE_PROFILER_ENDPOINT": "0",
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss",
    },
    # The throughput gate: `dataset: bench_serving` runs the sweep instead of
    # scoring an eval. `sweep` is the shapes to measure; `bench_env` is what
    # bench_serving.sh is handed.
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
                # the name requests carry, and the dashboard reports, is the HF
                # id served above.
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
