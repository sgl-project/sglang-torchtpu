"""GPT-OSS-120B single-host prefix caching + KV offload (raiden) + EAGLE3 gate."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

# Expected runtime: 45 minutes; CI uses this for scheduling and timeouts.
register_tpu_ci(est_time=2700, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# Accuracy should stay near the measured 0.83-0.84 score.
GSM8K_ACCURACY = 0.80  # measured 0.83-0.84 over 200 examples, accept length 2.77

# Permits probe eviction while supporting eight concurrent requests.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
PAGE_SIZE = 64

# Values above 1.0 prove that draft tokens were accepted.
MIN_ACCEPT_LENGTH = 1.0  # measured 2.67 on the probe reads, 2.77 over GSM8K

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    # Keep concurrency within the small device pool.
    "CONCUR": "8",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    "PAGE_SIZE": str(PAGE_SIZE),
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 8",
    # Raiden requires RPA v3 and the fused layout.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    # Enable the radix cache used by HiCache.
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    # Copy the probe to host before device eviction.
    "HICACHE_WRITE_POLICY": "write_through",
    # Include cache-hit details in OpenAI API responses.
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TODO(tpu): remove after distributed top-k precompiles the 2048-token bucket.
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm EAGLE3 "
        f"--speculative-draft-model-path {DRAFT_MODEL_PATH} "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-spec",
    "variant": "TP8+EP8+HiCache-raiden+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
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
                # Require verify forwards on every read and mean accept length above 1.0.
                min_accept_length=MIN_ACCEPT_LENGTH,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
