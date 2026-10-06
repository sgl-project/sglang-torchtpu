"""Gemma-4-31B-it (bf16 dense, hybrid sliding/global attention + ViT) on one
tpu7x host, 4 chips x 2 cores = tp8.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gemma4_31b/server.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
# max_running_requests and torch_compile_max_bs follow it, so one boot covers
# c=1..CONCURRENCY.
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "google/gemma-4-31B-it",
        "tp_size": 8,
        # "unquant", not None: it also sets _quantization_explicitly_unset, so
        # the model overrides cannot re-detect a quant method.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        # Compile materializes the KV pool as graph args + outputs, so an
        # unbounded pool doubles the precompile footprint.
        "max_total_tokens": 131072,
        "mem_fraction_static": 0.35,
        # The EXTEND/decode captures outlast the 300s default and trip the
        # watchdog into repeated stack dumps mid-compile.
        "watchdog_timeout": 3600,
    }
)
# The wrapper runs eagerly around the compiled text model, so the position bump
# must not mutate the batch's tensor in place.
CONFIG["runtime_env"].update({"SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION": "1"})
