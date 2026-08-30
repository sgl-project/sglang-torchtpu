"""GPT-OSS-120B single-host prefix-cache + Raiden KV-offload + full DPA gate."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

register_tpu_ci(est_time=5400, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70
DP_SIZE = 8
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
PAGE_SIZE = 64

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    "PAGE_SIZE": str(PAGE_SIZE),
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2",
    # TP8 / DP8 gives one unsharded attention replica per DP rank.
    "DP_SIZE": str(DP_SIZE),
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    "HICACHE_WRITE_POLICY": "write_through",
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--tpu-compile-tracer dynamo"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-dpa",
    "variant": "TP8+EP8+DP8(attn-tp1)+HiCache-raiden",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-dpa-e2e",
    },
    "gates": [
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload-dpa",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload-dpa",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                # Full DPA exposed occasional instruction-following variance on a
                # cold read; compare the device-hit and host-reload text directly.
                require_counting=False,
                dp_size=DP_SIZE,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
