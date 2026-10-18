"""GPT-OSS-120B single-host prefix caching + KV offload (raiden) + EAGLE3 gate."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner

# Expected runtime: 45 minutes; CI uses this for scheduling and timeouts.
register_tpu_ci(
    est_time=2700,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "prefix-cache", "kv-offload", "spec"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
CONFIG_MODULE = "recipe/configs/gpt_oss_120b/prefix_cache_kv_offload_spec.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# Accuracy should stay near the measured 0.83-0.84 score.
GSM8K_ACCURACY = 0.80  # measured 0.83-0.84 over 200 examples, accept length 2.77
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)

# The probe divides by the pool it is actually served, so read these from the
# recipe, which sizes the device pool small enough to evict quickly.
_SERVER_ARGS = recipe_server_args(CONFIG_MODULE)
DEVICE_POOL_TOKENS = _SERVER_ARGS["max_total_tokens"]
HICACHE_RATIO = _SERVER_ARGS["hicache_ratio"]
# A prefix match is floored to a page multiple, which the gate's "at least one
# page" checks are in terms of.
PAGE_SIZE = _SERVER_ARGS["page_size"]
PROBE_TOKENS = 2048


MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-spec",
    "variant": "TP8+EP8+HiCache-raiden+EAGLE3",
    "launch": {
        "config": CONFIG_MODULE,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-spec-e2e",
    },
    # Keep this specialized result off the standard accuracy dashboard.
    "gates": [
        # Run first while the probe owns an empty cache.
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload-spec",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                # Require verify forwards on every read, and the host reload to
                # cost the same forwards as the device hit.
                spec_decoding=True,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
