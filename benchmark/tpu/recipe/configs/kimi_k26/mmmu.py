"""Kimi-K2.6 (W4A16 MoE + MoonViT) on one 2x2x2 tpu7x slice: tp16/ep16, fp8 KV,
no DP attention. Each MMMU-Pro shard job boots one of these.

    python3 -m benchmark.tpu.launch_server --config recipe/configs/kimi_k26/mmmu.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 128
# 98k thinking budget + prompt and images.
CONTEXT_LENGTH = 106496
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Kimi-K2.6",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        # The checkpoint ships its own int4 config; fp8 would re-quantize it.
        "quantization": None,
        "attention_backend": "pallas_tpu_mla",
        # Page 128 keeps the page table under the SMEM cliff at 106k context.
        "page_size": 128,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "tpu_compile_bs": [1, 16, 32, 64, CONCURRENCY],
        "mem_fraction_static": 0.88,
        "trust_remote_code": True,
        "skip_server_warmup": True,
        "scheduler_recv_interval": 8,
        "weight_loader_drop_cache_after_load": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_MM_EMBED_ON_DEVICE": "1",
        "SGLANG_ENABLE_EMBED_REPLICATION": "1",
        # Square patch-count buckets warmed at startup.
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": (
            "256,400,576,784,1024,1600,2304,3136,4096,6400,9216,12544,16900"
        ),
    }
)
