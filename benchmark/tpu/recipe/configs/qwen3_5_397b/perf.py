"""Qwen3.5-397B-A17B throughput config: tp16/ep16 across a 2x2x2 slice, bf16.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_5_397b/perf.py

TWO HOSTS, unlike the rest of the perf lane: 397B of bf16 weights is ~99 GB per
rank at tp8, of the 101.7 GB a rank owns, which leaves nothing for the KV pool.
The JobSet runs this config on both hosts and node_rank comes from the topology.

Text only: the checkpoint serves images, but the sweep sends none, so the ViT
precompile ladder and the image budget the accuracy config carries are left out.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 16
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.5-397B-A17B",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 2,
        "tp_size": 16,
        # 512 experts at moe_intermediate 1024: whole experts per rank.
        "ep_size": 16,
        # bf16 checkpoint; "unquant" also blocks quant-method re-detection.
        "quantization": "unquant",
        # tpu_platform resolves "auto" to fp8_e4m3 for every RPA backend.
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        # CI compiles cold every run: one prefill and one decode bucket.
        "tpu_compile_tokens": [INPUT_TOKENS],
        "tpu_compile_bs": [CONCURRENCY],
        # Uncapped the pool takes all post-weight HBM and the decode capture OOMs.
        "max_total_tokens": 1310720,
        "mem_fraction_static": 0.85,
        "trust_remote_code": True,
        # Graph captures outlast the 300 s default.
        "watchdog_timeout": 3600,
    }
)
