"""GPT-OSS-120B cold-start precompile time -- nightly regression signal (#1041).

Precompile time regressed ~10% after #814 and was only caught by user reports:
CI checked correctness and throughput, never how long bring-up takes. This boots
gpt-oss-120b from its recipe with the torch_tpu compile cache OFF, so every
shape bucket compiles from scratch, and publishes /server_info's startup_time
once the server answers /health_generate (see evals/startup_timing.py):

  headline  precompile_s           scheduler bring-up minus weight load and KV
                                   allocation: the shape-bucket precompile,
                                   end to end
  info      load_weight, scheduler_e2e, tokenizer_e2e, per-phase capture time,
            and per-shape compile_steps

NO TIME FLOOR, like the throughput tests: the dashboard's trend is the
regression detector, and a threshold invented before any nightly has run would
either never fire or fire on noise. The gate checks the measurement is COMPLETE.

Same model, shape and serving config as test_tpu_perf_gpt_oss_120b.py, except:
  ENABLE_COMPILE_CACHE=0  no tier-2/tier-3 torch_tpu cache, so the number is
                          a cold compile even if a cache ever survives a pod.
  no TPU_COMPILE_*        the recipe's DERIVED bucket grid (powers of two up to
                          CONCUR) instead of the perf test's single bucket: that
                          grid is what a user's bring-up compiles, and more
                          shapes make a per-shape slowdown easier to see.
  no sweep                bring-up is the measurement.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.evals.startup_timing import startup_timing_runner

# One cold bring-up: weight load + the full derived bucket grid with no compile
# cache. Drives the JobSet poll budget, so an under-estimate kills the run.
# Measured on tpu7x: ~160s to ready with weights in /dev/shm (~50s load, ~88s
# precompile); 289s for the whole test in CI, where the PVC load takes ~150s.
# Not auto-refreshed: update_est_time.py omits tpu.
register_tpu_ci(
    est_time=900,
    suite="nightly-tpu-perf-gpt-oss-120b-precompile",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf", "compile-time"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1200"))

WORKLOAD = "cold-precompile"
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"
SERIES_ID = "perf-gpt-oss-120b-precompile"
DISPLAY_NAME = "GPT-OSS 120B (cold precompile)"

MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-precompile",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": SERIES_ID,
    "display_name": DISPLAY_NAME,
    "variant": "TP8+EP8",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": {
            "MODEL": WEIGHTS,
            "EXTRA_ARGS": f"--served-model-name {MODEL}",
            "ISL": str(ISL),
            "OSL": str(OSL),
            "CONCUR": str(CONCUR),
            "DISABLE_RADIX_CACHE": "1",
            "ENABLE_COMPILE_CACHE": "0",
            # ENABLE_COMPILE_CACHE=0 only stops the launcher from exporting
            # cache names; torch_tpu still falls back to an implicit tier-2
            # cache ("default") whenever WORLD_SIZE > 1. "disabled" is
            # torch_tpu's sentinel that really turns tier-2 off.
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
            "ENABLE_PROFILER_ENDPOINT": "0",
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "precompile-gpt-oss",
    },
    "gates": [
        {
            "dataset": "startup_timing",
            "label": WORKLOAD,
            "runner": startup_timing_runner(
                test_name="gpt-oss-120b-precompile",
                model=MODEL,
                workload=WORKLOAD,
                series_id=SERIES_ID,
                display_name=DISPLAY_NAME,
                extra_labels={"c": CONCUR},
            ),
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
