"""DeepSeek-R1-0528 (671B MoE, fp8) multi-host e2e accuracy: GPQA, AIME25, MMLU.

The JobSet launches the server (recipe/deepseek_r1_0528_mh.sh, 2 hosts, tp=16);
this file attaches to it — the same launch group as the GSM8K gate, so the two
files share one bring-up. Each dataset runs twice (repeat=2, scores averaged)
under the R1 card protocol (temperature 0.6, top_p 0.95, 32768-token budget).

Floors are wide on purpose: with one server config and 2 passes the job's role
is "serves the card protocol end-to-end and lands in the right accuracy band",
not a tight regression bound. Reference points: card GPQA-Diamond 0.715 and
AIME'25 0.875 are pass@1 averaged over 64 samples; a 2026-08-25 one-pass run
on the 2x2x2 dev slice measured GPQA 0.793.

The whole file runs under one `timeout` in jobset-tpu.yaml, so
MULTIHOST_LAUNCH declares the 10800s budget the six passes need (discovery
takes the max across the group; see discover_multihost.py).
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

# est_time is the expected wall clock (bookkeeping); the enforced per-file
# cap is MULTIHOST_LAUNCH.timeout_per_file below.
# Temporarily out of the nightly lane: on-demand suite, discovered by
# accuracy-test-tpu.yml (-f model=deepseek-r1-0528). Move the suite back to
# "nightly-tpu-multihost" to restore the nightly trend.
register_tpu_ci(est_time=7200, suite="nightly-tpu-multihost-ondemand", nightly=True)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/deepseek_r1_0528_mh.sh",
    "weights_pvc": "model-weights-rox",
    "timeout_per_file": 10800,
}

MODEL_PATH = os.environ.get("MODEL", "/weights/DeepSeek-R1-0528")
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))

CARD_TEMPERATURE = 0.6
CARD_TOP_P = 0.95
MAX_TOKENS = 32768

MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "DeepSeek-R1-0528-multihost",
    "variant": "2host-TP16+EP16",
    "launch": {"external": True, "health_timeout": HEALTH_TIMEOUT, "tag": "r1-e2e"},
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond",
            "baseline": 0.715,
            "atol": 0.10,
            "repeat": 2,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": MAX_TOKENS,
        },
        {
            "dataset": "aime25",
            "label": "aime25",
            # 30 questions/pass: even averaged over 2 passes, one flipped
            # answer moves the score 1.7pp, hence the wide band.
            "baseline": 0.875,
            "atol": 0.20,
            "repeat": 2,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": MAX_TOKENS,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu",
            "baseline": 0.90,
            "atol": 0.08,
            "repeat": 2,
            "num_examples": 250,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": 16384,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
