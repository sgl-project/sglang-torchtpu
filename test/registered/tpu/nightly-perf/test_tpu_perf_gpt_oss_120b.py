"""Throughput sweep for gpt-oss-120b on the 8k1k workload.

ADDING A MODEL: copy this file, and change six things.

  1. the filename and its `suite=` -- ONE FILE AND ONE SUITE PER MODEL. That is
     what makes the models run as separate nightly jobs on separate slices; a
     shared suite would put them back into one sequential job, where a slow model
     delays the others and a failing one costs them their measurement entirely.
     Add the new suite name to run_suite.py's TPU list and to the matrix in
     perf-test-tpu.yml (the nightly calls that workflow).
  2. `id` and `display_name` -- `id` is the dashboard's series join key, NEVER
     renamed after the first nightly (a forced rename goes through
     .github/tpu-ci/dashboard_aliases.json); `display_name` is cosmetic and free.
  3. `est_time` -- one bring-up (weight load + per-shape compile) plus the sweep.
     It drives the JobSet poll budget, so an under-estimate kills the run. Not
     auto-refreshed: update_est_time.py omits tpu.
  4. `MODEL` and `launch.recipe` -- point at the recipe that already knows how to
     serve this model, then list ONLY the deltas the measurement needs in `env`.
  5. the shape (ISL/OSL/CONCUR/WORKLOAD), if this model is measured at a
     different one. `WORKLOAD` is the name the dashboard's dropdown shows.
  6. `launch.tag` -- what the boot line is labelled with in the log.

Deliberately duplicated rather than shared across perf files: the point of one of
these is that you can read how a model is served without jumping between files,
and what would actually be shared is three integers.

WHY ITS OWN SERVER, not the accuracy nightly's: an accuracy recipe pins whatever
makes its score reproducible -- for gpt-oss that is bf16 KV, a 16384 context and
a 4096 prefill chunk so a GPQA prompt fits. A throughput run wants fp8 KV, a
context sized to the workload, and a chunk that swallows an 8k prompt whole. A
throughput number measured on an accuracy server is one nobody would quote, and
one that tuning work would not move in the expected direction.

NO THROUGHPUT FLOOR. The dashboard's trend is the regression detector; a
threshold invented before any nightly has run would either never fire or fire on
noise. What the gate does check -- that the measurement is COMPLETE -- lives in
bench_serving_runner, and is what separates "slower" from "the server served
nothing".

This model serves from a mounted checkpoint rather than an HF id, so the served
name has to be set explicitly in both the launch and the bench; and its recipe
derives its whole shape from ISL/OSL/CONCUR, so those go in `env` rather than a
context length being computed here.

(Qwen3-Coder-480B is measured elsewhere: its perf source is the multi-turn
agentic sweep in e2e/test_tpu_qwen3_coder_480b_multihost_agentic.py, which runs
in the multi-host nightly and dumps its own workload-labelled metrics. Its
predecessor here, test_tpu_perf_qwen3_30b.py, was retired with it.)
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

# One bring-up (weight load + per-shape compile, ~10 min with the single-bucket
# compile below) plus one sweep leg. Drives the JobSet poll budget, so an
# under-estimate kills the run. Not auto-refreshed: update_est_time.py omits tpu.
register_tpu_ci(
    est_time=550,
    suite="nightly-tpu-perf-gpt-oss-120b",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown.
WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 16

# Serves from the shared read-only Filestore PVC (model-weights-filestore at /weights), but the
# canonical HF id is what the server is told to serve as and what the dashboard
# keys the series on -- a mount path would be a poor key, since remounting
# elsewhere would silently split the trend in two.
MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-gpt-oss-120b",
    # Cosmetic, free to change at will -- the split from `id` is what makes a
    # display rename a non-event for the trend.
    "display_name": "GPT-OSS 120B",
    "variant": "TP8+EP8",
    "launch": {
        # server.sh's own sample usage is `ISL=8192 OSL=1024 CONCUR=N ./server.sh`
        # -- it derives context, prefill chunk, KV pool and compile buckets from
        # exactly those three. Delta on top of the recipe:
        #   MODEL         the Filestore PVC mount; server.sh defaults to the HF id.
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
