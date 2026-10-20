"""Kimi-K3 (MXFP4 MoE, KDA + MLA) on four tpu7x hosts, 4 chips x 2 cores each = tp32.

Sized by a profile (recipe/configs/profiles.py), as the cookbook page renders it:

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/server.py
    # every host runs the same command; node_rank / dist_init_addr come from
    # TPU_WORKER_ID / TPU_WORKER_HOSTNAMES. Add --dry-run to print the command.

Low-concurrency decode: at tp32 the weights leave room for a few requests, and
the K3 decode kernels below are tuned for batch 1-4.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 4
# Prefill is chunked at 1024 and only that token bucket is compiled: each extra
# EXTEND bucket adds a compiled program per host, and eight of them up to 8192
# OOM-killed all four hosts.
PREFILL_TOKENS = 1024
CONTEXT_LENGTH = 8192 + 1024 + 64

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "moonshotai/Kimi-K3",
        "nnodes": 4,
        "tp_size": 32,
        # ep8: each expert is split over 4 ranks, which evens out the slowest
        # rank in the decode MoE all-reduce.
        "ep_size": 8,
        # The checkpoint ships its own MXFP4 config; fp8 would re-quantize it.
        "quantization": None,
        "kv_cache_dtype": None,
        "attention_backend": "pallas_tpu_mla",
        "page_size": 128,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "tpu_compile_bs": [1, CONCURRENCY],
        "mem_fraction_static": 0.75,
        "trust_remote_code": True,
        "reasoning_parser": "kimi_k3",
        "tool_call_parser": "kimi_k3",
        "mamba_ssm_dtype": "bfloat16",
        "scheduler_recv_interval": 16,
        # Each rank reads only its own experts instead of mapping every shard.
        "weight_loader_disable_mmap": True,
        # The four-host load and first compile outlast the default watchdog.
        "watchdog_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_EAGER_MODE": "DEFER_AND_FUSE",
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "512",
        # The K3 decode kernels; each is off by default and validated together.
        # Static-scale fp8 GEMVs for the RMSNorm-fed projections.
        "SGLANG_TPU_K3_STATIC_FP8": "moe_front,latent_up,kda_qkvg",
        # Replicated single-token MoE GEMVs split across the 8 cores of a host.
        "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV": "1",
        # Pallas all-reduce, within a host and then across hosts; and two of
        # the MoE all-reduces merged into one.
        "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE": "1",
        "SGLANG_K3_MERGE_MOE_ALLREDUCE": "1",
        "SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL": "1",
        "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER": "1",
    }
)
