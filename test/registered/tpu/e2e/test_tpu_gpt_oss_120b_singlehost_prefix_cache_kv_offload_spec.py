"""GPT-OSS-120B single-host prefix caching + KV offload (raiden) + EAGLE3 gate."""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

# Expected runtime: 45 minutes; CI uses this for scheduling and timeouts.
register_tpu_ci(
    est_time=2700,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "prefix-cache", "kv-offload", "spec"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# Accuracy should stay near the measured 0.83-0.84 score.
GSM8K_ACCURACY = 0.80  # measured 0.83-0.84 over 200 examples, accept length 2.77
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)

# Permits probe eviction while supporting eight concurrent requests.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
PAGE_SIZE = 64


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 8192,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "chunked_prefill_size": min(max(4096, 2048), 8192),
        "max_prefill_tokens": min(max(4096, 2048), 8192),
        "max_total_tokens": int(DEVICE_POOL_TOKENS),
        "model_path": "/weights/gpt-oss-120b",
        "page_size": int(PAGE_SIZE),
        "tpu_compile_tokens": [512, 2048, 4096],
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
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-spec",
    "variant": "TP8+EP8+HiCache-raiden+EAGLE3",
    "launch": {
        **LAUNCH_CONFIG,
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
