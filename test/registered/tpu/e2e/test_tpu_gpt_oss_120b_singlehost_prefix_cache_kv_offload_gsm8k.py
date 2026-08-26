"""GPT-OSS-120B single-host prefix caching + KV offload (raiden) gate.

Covers the P1 `prefix_caching_kv_offload` requirement (issue #372): one server with
the radix cache on and HiCache mirroring it into host RAM over the raiden DMA
backend must show a cache hit, show offloading actually happening, and still clear
the GSM8K quality bar.

GPT-OSS is hybrid-SWA, which is what makes raiden the backend under test rather
than torch_tpu: raiden gives each sub-pool its own host pool, while torch_tpu
mirrors only the full-attention pool and server_args rejects it outright for a
sliding-window model.

The prefix-cache + KV-offload gate runs first, on a cache it flushes and fully
controls. GSM8K then scores the same boot, with prefix caching still on.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

# Bring-up plus the offload probe's ~26k tokens of filler prefill plus 200 GSM8K
# examples. Above the 3600 the spec-decoding lane uses, by the probe's cost.
register_tpu_ci(est_time=4200, suite="stage-b-test-tpu-4-v7")
register_tpu_ci(est_time=4200, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

# Same floor as the spec-decoding lane: offloading changes where KV lives, not what
# the model computes, so a drop here is a corrupted host round-trip.
GSM8K_ACCURACY = 0.70

# The device pool has to be small enough that a handful of filler prefills evict the
# probe to host, and large enough that GSM8K at CONCUR=8 still fits (short prompts
# plus 512 output tokens each). Both hold at 16k.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0
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
    # raiden requires the RPA v3 backend (the only TPU backend with the HiCache
    # reload-commit hook) and the fused page-first layout, and server_args rejects any
    # other combination. Both are already the recipe's default -- pinned here so a
    # recipe change cannot silently take the offload path away from this gate.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    # Prefix caching is the substrate HiCache is built on, and launch_server.sh
    # defaults it OFF.
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    # Write through on every prefill so the probe is already on host by the time the
    # fillers evict it. write_back would race that eviction and drop it instead.
    "HICACHE_WRITE_POLICY": "write_through",
    # cached_tokens_details in meta_info is how the gate tells a device hit from a
    # host reload. Without this flag both read 0.
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss " f"--served-model-name {SERVED_MODEL_NAME}"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload",
    "variant": "TP8+EP8+HiCache-raiden",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-e2e",
    },
    # No dashboard key: this is feature validation, and its GSM8K score is gated at a
    # config the accuracy lane does not serve, so plotting it next to that lane's
    # would compare two different servers.
    "gates": [
        # First, on a cache it flushes and fully owns -- GSM8K would otherwise leave
        # the pool full and the eviction arithmetic would no longer hold.
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload",
                device_pool_tokens=DEVICE_POOL_TOKENS,
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
            # Matches CONCUR above: the small device pool cannot hold more requests
            # in flight without retracting them.
            "num_threads": 64,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
