"""Throughput sweep for Qwen3-Coder-30B-A3B-FP8 on the 8k1k workload.

ADDING A MODEL: copy this file, and change five things.

  1. the filename and its `suite=` -- ONE FILE AND ONE SUITE PER MODEL. That is
     what makes the models run as separate nightly jobs on separate slices; a
     shared suite would put them back into one sequential job, where a slow model
     delays the others and a failing one costs them their measurement entirely.
     Add the new suite name to run_suite.py's TPU list and to the matrices in
     perf-test-tpu.yml and nightly-test-tpu.yml.
  2. `est_time` -- one bring-up (weight load + per-shape compile) plus the sweep.
     It drives the JobSet poll budget, so an under-estimate kills the run. Not
     auto-refreshed: update_est_time.py omits tpu.
  3. `MODEL` and `launch.recipe` -- point at the recipe that already knows how to
     serve this model, then list ONLY the deltas the measurement needs in `env`.
  4. the shape (ISL/OSL/CONCUR/WORKLOAD), if this model is measured at a
     different one. `WORKLOAD` is the name the dashboard's dropdown shows.
  5. `launch.tag` -- what the boot line is labelled with in the log.

Deliberately duplicated rather than shared with the other perf files: the point
of one of these is that you can read how a model is served without jumping
between files, and what would actually be shared is three integers.

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
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

# One bring-up plus one sweep leg at c=16. See point 2 above.
register_tpu_ci(est_time=1800, suite="nightly-tpu-perf-qwen3-30b", nightly=True)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown. One shape and
# one concurrency to start; add points once this one has a baseline to read them
# against.
WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "Qwen3-Coder-30B-A3B-Instruct-FP8",
    "variant": "TP8+EP8",
    "launch": {
        # The same script the (currently PR-suite, disabled) accuracy test and a
        # human repro run, so there is one copy of "how this model is served".
        # Delta on top of it; everything unlisted stays at the recipe's value:
        #   CONTEXT_LEN   the recipe's 8192 is sized for its lm-eval MMLU sweep
        #                 and is one token short of an 8k prompt, let alone its
        #                 1k completion. ISL+OSL+64 is launch_server.sh's own
        #                 default formula.
        #   MAX_RUNNING / TORCH_COMPILE_MAX_BS
        #                 the recipe's 64 is that sweep's ceiling; at c=16 the
        #                 extra buckets are compile time spent on shapes this run
        #                 never reaches. The KV pool also has to hold it:
        #                 16 x 9280 = 148k tokens fits the recipe's 262144-token
        #                 pool, where 64 x 9280 = 594k would not.
        #   ENABLE_PROFILER_ENDPOINT  a nightly captures no traces.
        "recipe": "recipe/qwen3_coder_30b_a3b_fp8.sh",
        "env": {
            "CONTEXT_LEN": str(ISL + OSL + 64),
            "MAX_RUNNING": str(CONCUR),
            "TORCH_COMPILE_MAX_BS": str(CONCUR),
            "ENABLE_PROFILER_ENDPOINT": "0",
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-qwen3-30b",
    },
    # The throughput gate: `dataset: bench_serving` runs the sweep instead of
    # scoring an eval. `sweep` is the shapes to measure; `bench_env` is what
    # bench_serving.sh is handed.
    #
    # MODEL defaults to the config's model, an HF id that is both the tokenizer
    # source and the served name, so this gate needs no id overrides.
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
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
