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
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

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
ISL, OSL, CONCUR = 8192, 1024, 16

# Serves from the shared ROX PVC (model-weights-rox at /weights), but the
# canonical HF id is what the server is told to serve as and what the dashboard
# keys the series on -- a mount path would be a poor key, since remounting
# elsewhere would silently split the trend in two.
MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": int(ISL) + int(OSL) + 64,
        "max_running_requests": int(CONCUR),
        "torch_compile_max_bs": int(CONCUR),
        "chunked_prefill_size": min(max(int(ISL), 2048), 8192),
        "max_prefill_tokens": min(max(int(ISL), 2048), 8192),
        "max_total_tokens": 2 * (int(CONCUR)) * (int(ISL) + int(OSL) + 64),
        "model_path": WEIGHTS,
        "disable_radix_cache": True,
        "tpu_compile_tokens": [ISL],
        "tpu_compile_bs": [CONCUR],
        "served_model_name": MODEL,
    }
)

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
        **LAUNCH_CONFIG,
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
