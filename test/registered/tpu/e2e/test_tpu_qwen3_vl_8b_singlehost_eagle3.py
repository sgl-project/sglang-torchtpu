"""Qwen3-VL-8B EAGLE3 single-host MMMU-Pro (vision) gate.

Serves the VL recipe with speculative decoding on and scores an image
benchmark, so the gate covers the combination the text-only spec tests cannot:
a ViT prefill feeding a draft/verify decode loop.

The score is report-only -- the enforced threshold is the accept length, which
is what goes flat when the draft stops tracking the target on image prompts.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3170,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "qwen3-vl-8b", "spec", "eagle3", "multimodal"],
)

SERVED_MODEL_NAME = "qwen3-vl-8b"
MODEL = "/weights/Qwen3-VL-8B-Instruct"
DRAFT_MODEL = "/weights/qwen3_8b_eagle3"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Budget: the ViT and verify-graph captures. 2400s timed out mid-capture.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "5400"))

# A floor on "spec is still accepting", not a tuned target: the draft is the
# text-only Qwen3-8B speculator, and no VL night has been measured yet. Ratchet
# it up once the trend exists.
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.30")
)

LAUNCH_ENV = {
    "MODEL": MODEL,
    # Concurrency has to come down with the compile cap: reconcile_tpu_batch_limits
    # raises torch_compile_max_bs back to max_running_requests, so CONC=32 would
    # restore the bs=32 grid. 8 gives bs buckets [1, 2, 4, 8].
    "CONC": "8",
    "TORCH_COMPILE_MAX_BS": "8",
    # The default PowerBucketer gives 8 token buckets (64..8192), and the VL
    # and spec EXTEND grids each multiply by them: 24 + 32 + 16 shapes, ~20 min
    # of the startup. Four buckets halve that; 8192 must stay to cover
    # MAX_PREFILL_TOKENS.
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    # bf16 checkpoint with no KV scales; launch_server.sh's fp8_e4m3 default
    # would score the gate through scaling factors of 1.0.
    "KV_CACHE_DTYPE": "bfloat16",
    # common_env.sh defaults this to +recompiles, and dynamo reprints the whole
    # cached guard list per recompile: 222k of the 235k log lines. Neither ""
    # nor "-recompiles" silences it -- the former loses to ${TORCH_LOGS:-...},
    # and a "-" on an artifact still enables it. Serving-time recompiles stay
    # fatal via the scheduler's fail_on_recompile stance.
    "TORCH_LOGS": "-dynamo",
    "EXTRA_ARGS": (
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm EAGLE3 "
        f"--speculative-draft-model-path {DRAFT_MODEL} "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-vl-8b-eagle3-mmmu-pro-vision",
    "variant": "TP8+EAGLE3",
    "launch": {
        "recipe": "recipe/qwen3_vl_8b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-vl-eagle3-mmmu-e2e",
    },
    "gates": [
        {
            "dataset": "mmmu_pro_vision",
            "label": "mmmu_pro_vision",
            "report_only": True,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
