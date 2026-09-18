"""Qwen3-Coder-480B single-host prefix caching + KV offload (raiden) gate.

Covers the P1 `prefix_caching_kv_offload` requirement (issue #372): one server with
the radix cache on and HiCache mirroring it into host RAM over the raiden DMA
backend must show a cache hit, show offloading actually happening, and still clear
the GSM8K quality bar.

Qwen3-Coder is the uniform-attention half of that requirement, and the larger one:
it is where the host pool has to absorb a 480B model's KV, and it is the only row
that exercises raiden's single-pool controller, since GPT-OSS being hybrid-SWA
routes to the two-pool one instead.

The prefix-cache + KV-offload gate runs first, on a cache it flushes and fully
controls. GSM8K then scores the same boot, with prefix caching still on.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.qwen3_coder_480b.server import CONFIG

# Bring-up plus the offload probe's ~26k tokens of filler prefill plus 200 GSM8K
# examples. Above the 3600 the spec-decoding lane uses, by the probe's cost.
register_tpu_ci(
    est_time=4200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "qwen3-coder-480b", "prefix-cache", "kv-offload"],
)

SERVED_MODEL_NAME = "qwen3-coder-480b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# Same floor as the spec-decoding lane: offloading changes where KV lives, not what
# the model computes, so a drop here is a corrupted host round-trip.
GSM8K_ACCURACY = 0.65

# The device pool has to be small enough that a handful of filler prefills evict the
# probe to host, and large enough that GSM8K at CONCUR=8 still fits (short prompts
# plus 512 output tokens each). Both hold at 16k.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
# recipe/configs/qwen3_coder_480b/server.py exports PAGE_SIZE unconditionally, so this is
# not overridable from `env` -- it is restated because the gate's "at least one
# page" checks are in terms of it, and a prefix match is floored to a page multiple.
PAGE_SIZE = 256


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 8192,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "chunked_prefill_size": 512,
        "max_prefill_tokens": 512,
        "max_total_tokens": int(DEVICE_POOL_TOKENS),
        "model_path": "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "tpu_compile_tokens": [512],
        "tpu_compile_bs": [1, 2, 4, 8],
        "attention_backend": "pallas_tpu_rpa_v3",
        "hicache_mem_layout": "page_first_fused",
        "disable_radix_cache": False,
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_ratio": float(HICACHE_RATIO),
        "hicache_write_policy": "write_through",
        "enable_cache_report": True,
        "served_model_name": SERVED_MODEL_NAME,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-coder-480b-prefix-cache-kv-offload",
    "variant": "TP8+EP8+HiCache-raiden",
    "launch": {
        **LAUNCH_CONFIG,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-coder-480b-prefix-cache-kv-offload-e2e",
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
                test_name="qwen3-coder-480b-prefix-cache-kv-offload",
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
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
