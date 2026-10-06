"""DeepSeek-R1-0528 (blockwise-fp8) on one 2x2x2 tpu7x slice: tp16/ep16, fp8 KV,
no DP attention. Serves the GPQA, AIME25 and MMLU-Redux nightly.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_r1_0528/accuracy.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 128
# 64k thinking budget + prompt.
CONTEXT_LENGTH = 73728
PREFILL_TOKENS = 8192

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-R1-0528",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "attention_backend": "pallas_tpu_mla",
        "page_size": 64,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "tpu_compile_bs": [64, CONCURRENCY],
        "mem_fraction_static": 0.85,
        "reasoning_parser": "deepseek-r1",
        # 642 GB of checkpoint pages would otherwise stay in the host page cache.
        "weight_loader_drop_cache_after_load": True,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FP8_MOE_FP32_SCALES": "1"})
