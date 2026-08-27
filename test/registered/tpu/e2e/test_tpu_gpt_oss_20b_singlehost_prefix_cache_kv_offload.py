"""GPT-OSS-20B single-host cache/offload and quality gate.

The PR gate verifies device-cache reuse, Raiden host offload, and basic GSM8K
quality. The 120B variant provides broader nightly coverage.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

# Expected runtime includes model download, compilation, and the eviction probe.
register_tpu_ci(est_time=1800, suite="stage-b-test-tpu-4-v7")

SERVED_MODEL_NAME = "gpt-oss-20b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# Keep the device pool small so the probe can force eviction quickly.
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0
PROBE_TOKENS = 1024
PAGE_SIZE = 64
GSM8K_ACCURACY = 0.60

LAUNCH_ENV = {
    "MODEL": "openai/gpt-oss-20b",
    "ISL": "2048",
    "OSL": "512",
    "CONCUR": "8",
    "CONTEXT_LEN": "4096",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    "PAGE_SIZE": str(PAGE_SIZE),
    "TPU_COMPILE_TOKENS": "512 1024 2048",
    "TPU_COMPILE_BS": "1 2 4 8",
    # Raiden reloads require RPA v3 and the fused host-pool layout.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    # Prefix caching is the substrate HiCache is built on, and launch_server.sh
    # defaults it OFF.
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    # Back up each prefix before device eviction.
    "HICACHE_WRITE_POLICY": "write_through",
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss " f"--served-model-name {SERVED_MODEL_NAME}"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-prefix-cache-kv-offload",
    "variant": "TP8+EP8+HiCache-raiden",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-prefix-cache-kv-offload-e2e",
    },
    "gates": [
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload",
            "runner": kv_offload_runner(
                test_name="gpt-oss-20b-prefix-cache-kv-offload",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                require_counting=False,
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
