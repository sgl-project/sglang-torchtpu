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
  4. `MODEL` and `launch.config` -- point at the model's Python recipe. A perf
     recipe is its `server.py` sized by one of the profiles in
     benchmark/tpu/recipe/configs/profiles.py; it departs from the deployment
     only where the measurement needs it to.
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
name is set explicitly in the recipe and in the bench. The recipe sizes the
server from the `long_8k1k` profile, and the sweep below drives that same shape:
the published cookbook command, the server CI measures, and this number are one
configuration.

(Qwen3-Coder-480B is measured elsewhere: its perf source is the multi-turn
agentic sweep in e2e/test_tpu_qwen3_coder_480b_multihost_agentic.py, which runs
in the multi-host nightly and dumps its own workload-labelled metrics. Its
predecessor here, test_tpu_perf_qwen3_30b.py, was retired with it.)
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args

# One bring-up (weight load + per-shape compile, ~10 min with the single-bucket
# compile below) plus one sweep leg. Drives the JobSet poll budget, so an
# under-estimate kills the run. Not auto-refreshed: update_est_time.py omits tpu.
register_tpu_ci(
    est_time=1800,
    suite="nightly-tpu-perf-gpt-oss-120b",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# 8192 in + 1024 out, named for the dashboard's Workload dropdown.
WORKLOAD = "8k1k"
CONFIG_MODULE = "recipe/configs/gpt_oss_120b/perf.py"
ISL, OSL = 8192, 1024
# The server is sized for exactly this concurrency, so drive what it was sized
# for: the recipe's profile is the one source of the number.
CONCUR = recipe_server_args(CONFIG_MODULE)["max_running_requests"]

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
        "config": CONFIG_MODULE,
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
