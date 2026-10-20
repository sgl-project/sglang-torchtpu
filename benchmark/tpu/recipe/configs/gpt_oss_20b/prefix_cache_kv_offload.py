"""GPT-OSS-20B sized for the prefix-cache / Raiden KV-offload PR gate.

server.py owns how the model is served; this file sizes the pools small enough
that the gate's probe can force a device eviction in a few requests.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/prefix_cache_kv_offload.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 8
CONTEXT_LENGTH = 4096
PREFILL_TOKENS = 2048
# The probe reads these back: a device pool this small holds only a couple of
# the gate's prompts, so the next one must evict.
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 2, 4, CONCURRENCY],
        "tpu_compile_tokens": [512, 1024, PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": DEVICE_POOL_TOKENS,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "enable_hierarchical_cache": True,
        # Raiden reloads require RPA v3 (the base backend) and the fused host
        # pool layout.
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "hicache_ratio": HICACHE_RATIO,
        # Back up each prefix before eviction, so the probe measures a reload
        # rather than a recompute.
        "hicache_write_policy": "write_through",
        # Per-request cached_tokens in the usage block; the probe counts hits
        # from it.
        "enable_cache_report": True,
        "reasoning_parser": "gpt-oss",
    }
)
