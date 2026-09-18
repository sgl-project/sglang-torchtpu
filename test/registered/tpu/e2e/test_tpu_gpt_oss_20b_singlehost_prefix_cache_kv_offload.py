"""GPT-OSS-20B single-host cache/offload and quality gate.

The PR gate verifies device-cache reuse, Raiden host offload, and basic GSM8K
quality. The 120B variant provides broader nightly coverage.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

# Expected runtime includes model download, compilation, and the eviction probe.
register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "prefix-cache", "kv-offload"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# Keep the device pool small so the probe can force eviction quickly.
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0
PROBE_TOKENS = 1024
PAGE_SIZE = 64
GSM8K_ACCURACY = 0.60


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 4096,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "chunked_prefill_size": min(max(2048, 2048), 8192),
        "max_prefill_tokens": min(max(2048, 2048), 8192),
        "max_total_tokens": int(DEVICE_POOL_TOKENS),
        "model_path": "openai/gpt-oss-20b",
        "page_size": int(PAGE_SIZE),
        "tpu_compile_tokens": [512, 1024, 2048],
        "tpu_compile_bs": [1, 2, 4, 8],
        "attention_backend": "pallas_tpu_rpa_v3",
        "hicache_mem_layout": "page_first_fused",
        "disable_radix_cache": False,
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_ratio": float(HICACHE_RATIO),
        "hicache_write_policy": "write_through",
        "enable_cache_report": True,
        "reasoning_parser": "gpt-oss",
        "served_model_name": SERVED_MODEL_NAME,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-prefix-cache-kv-offload",
    "variant": "TP8+EP8+HiCache-raiden",
    "launch": {
        **LAUNCH_CONFIG,
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
