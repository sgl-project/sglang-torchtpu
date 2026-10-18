"""GPT-OSS-120B KV offload with EAGLE3, single-host gate.

server.py owns how this model is served; this file changes only what the
gate needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/prefix_cache_kv_offload_spec.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 8
CONTEXT_LENGTH = 8192
PREFILL_TOKENS = 4096
# The probe reads this back: a device pool this small evicts within a few
# requests, which is what puts the prefix on the host.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 2, 4, CONCURRENCY],
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
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
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "lmsys/EAGLE3-gpt-oss-120b-bf16",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
    }
)
