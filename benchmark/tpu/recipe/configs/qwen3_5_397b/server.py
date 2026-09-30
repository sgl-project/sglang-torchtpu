"""Qwen3.5-397B-A17B (bf16 hybrid GDN MoE + ViT) on one 2x2x2 tpu7x slice:
2 hosts x 4 chips x 2 cores = tp16/ep16, vision on.

Sample usage (from the repo root, on BOTH hosts of the slice):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_5_397b/server.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

# Largest concurrency the eval drives; also the recurrent-state slot count,
# which (not KV) bounds in-flight requests on this hybrid model.
CONCURRENCY = 128
# 32k thinking budget (model-card recommendation) plus prompt and images.
CONTEXT_LENGTH = 32768 + 8192
PREFILL_TOKENS = 8192

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.5-397B-A17B",
        # The launcher derives node_rank and dist_init_addr from
        # TPU_WORKER_ID / TPU_WORKER_HOSTNAMES on each host.
        "nnodes": 2,
        "tp_size": 16,
        # 512 experts at moe_intermediate 1024: whole experts per rank.
        "ep_size": 16,
        # bf16 checkpoint. "unquant" is the explicit opt-out, so the model
        # overrides cannot re-detect a quant method.
        "quantization": "unquant",
        # Explicit: the TPU platform resolves "auto" to fp8_e4m3 for every
        # RPA backend.
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        # 1.31M tokens = 18.8 GB/rank of bf16 KV, leaving ~46 GB at "Capture
        # target decode TPU graph begin". Uncapped, the pool takes all
        # post-weight HBM (~46.5 GB/rank of weights) and decode capture OOMs.
        "max_total_tokens": 1310720,
        "mem_fraction_static": 0.85,
        "trust_remote_code": True,
        # The decoder + ViT graph captures outlast the 300 s default.
        "watchdog_timeout": 3600,
        "reasoning_parser": "qwen3",
        # One image capped at 1280*28*28 px (~980 LLM tokens, 3920 ViT
        # patches); uncapped, a single MMMU-Pro page exceeds the 8k chunk.
        "mm_process_config": {"image": {"min_pixels": 65536, "max_pixels": 1003520}},
    }
)
CONFIG["runtime_env"].update(
    {
        # A request runs all its images through one ViT call, so warm square
        # buckets up to 128^2 patches (~16 capped images); larger totals
        # recompile once per size.
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "1024,2304,4096,6400,9216,12544,16384",
    }
)
