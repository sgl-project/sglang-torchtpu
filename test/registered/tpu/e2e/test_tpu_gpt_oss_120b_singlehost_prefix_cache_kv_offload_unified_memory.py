"""GPT-OSS-120B single-host prefix-cache + Raiden KV-offload gate with
--enable-unified-memory, where the full-attention and sliding-window KV share one
device pool and one raiden host pool."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner

register_tpu_ci(
    est_time=4200,
    suite="external-nightly-1host",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "prefix-cache",
        "kv-offload",
        "unified-memory",
    ],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70
DEVICE_POOL_TOKENS = 16384
# --max-total-tokens 16384 at the default 0.8 sliding-window ratio is 16384 + 13056
# tokens of pages, 29184 after the 8-page floor, all of which a full-attention
# prefix can take, so the fillers must exceed that.
UNIFIED_POOL_TOKENS = 29184
# The one host pool stores a full-attention and a sliding-window copy of every
# backed-up prefix, so it must hold twice what the probe and its fillers back up:
# 23 x 2048 x 2 = 94208 of its 116736 tokens.
HICACHE_RATIO = 4.0
PROBE_TOKENS = 2048
PAGE_SIZE = 64

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "64",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    "PAGE_SIZE": str(PAGE_SIZE),
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 8 16 32 64",
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
        f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME} "
        "--enable-unified-memory"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-unified-memory",
    "variant": "TP8+EP8+HiCache-raiden+unified-memory",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-unified-memory-e2e",
    },
    "gates": [
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload-unified-memory",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload-unified-memory",
                device_pool_tokens=UNIFIED_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 64,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
