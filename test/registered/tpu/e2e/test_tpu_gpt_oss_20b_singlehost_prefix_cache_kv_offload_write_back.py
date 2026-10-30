"""GPT-OSS-20B single-host cache/offload gate under the write_back policy.

The write_through gate backs every prefix up at insert; this one backs a prefix
up only near eviction, through the copy-ahead window or, past it, at eviction.
The probe's host reload therefore covers write_back's copies of prompt and
decode KV, and GSM8K the quality after it.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner

# Expected runtime includes model download, compilation, and the eviction probe.
register_tpu_ci(
    est_time=280,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "prefix-cache", "kv-offload"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
CONFIG_MODULE = "recipe/configs/gpt_oss_20b/prefix_cache_kv_offload_write_back.py"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# The probe divides by the pool it is actually served, so read it from the
# recipe, which sizes the device pool small enough to evict quickly.
_SERVER_ARGS = recipe_server_args(CONFIG_MODULE)
DEVICE_POOL_TOKENS = _SERVER_ARGS["max_total_tokens"]
HICACHE_RATIO = _SERVER_ARGS["hicache_ratio"]
PAGE_SIZE = _SERVER_ARGS["page_size"]
PROBE_TOKENS = 1024
GSM8K_ACCURACY = 0.60

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-prefix-cache-kv-offload-write-back",
    "variant": "TP8+EP8+HiCache-raiden-write-back",
    "launch": {
        "config": CONFIG_MODULE,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-prefix-cache-kv-offload-write-back-e2e",
    },
    "gates": [
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload-write-back",
            "runner": kv_offload_runner(
                test_name="gpt-oss-20b-prefix-cache-kv-offload-write-back",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                decode_tails=True,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k-smoke",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 40,
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
