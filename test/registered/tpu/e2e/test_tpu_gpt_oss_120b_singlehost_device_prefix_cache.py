"""GPT-OSS-120B single-host device prefix-cache diagnostic.

This diagnostic keeps the dependency stack and direct-compile mode from the
TorchTPU/Raiden upgrade, but disables HiCache and Raiden. It checks whether the
same prompt still produces a correct continuation when its prefix is read directly
from the device radix cache.

The device-cache probe runs on a cache it flushes and fully controls. The test ends
after that focused A/B result so the diagnostic releases its TPU quickly.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import device_cache_runner

# Bring-up plus one cold and one warm 2k-token prefill.
register_tpu_ci(est_time=3600, suite="stage-b-test-tpu-4-v7")
register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# Preserve the exact device-pool size from the failing HiCache test.
DEVICE_POOL_TOKENS = 16384
PROBE_TOKENS = 2048
# The recipe's default, restated here because the gate's "at least one page" checks
# are in terms of it -- a prefix match is floored to a page multiple.
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
    # Preserve the attention backend from the failing HiCache test while removing
    # only the hierarchical host-cache path.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    # Prefix caching stays enabled so the immediate re-send must read device KV.
    "DISABLE_RADIX_CACHE": "0",
    # Explicitly disable HiCache. launch_server.sh only adds the hierarchical-cache,
    # host-layout, and Raiden flags when this value is 1.
    "KV_OFFLOAD": "0",
    # Preserve the cache-report setting from the failing test. The focused
    # /generate probe reads its tier breakdown independently of this OpenAI-usage
    # reporting flag.
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss " f"--served-model-name {SERVED_MODEL_NAME}"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-device-prefix-cache",
    "variant": "TP8+EP8+device-cache-only",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-device-prefix-cache-e2e",
    },
    # No dashboard key: this is a focused diagnostic, not a persistent benchmark.
    "gates": [
        # Run first, on a cache it flushes and fully owns.
        {
            "dataset": "device_prefix_cache",
            "label": "device-prefix-cache",
            "runner": device_cache_runner(
                test_name="gpt-oss-120b-device-prefix-cache",
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
            ),
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
