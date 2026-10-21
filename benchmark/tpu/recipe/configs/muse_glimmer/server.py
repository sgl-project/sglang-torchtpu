"""Muse Glimmer 30B (bf16 dense, hybrid sliding-window/full attention) text
serving on one tpu7x host, 4 chips x 2 cores = tp8.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_glimmer/server.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "meta-models/Muse-Glimmer-30B",
        "tp_size": 8,
        # "unquant", not None: it also sets _quantization_explicitly_unset, so
        # the model overrides cannot re-detect a quant method.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        # The vision tower is not served on TPU.
        "language_model_only": True,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "disable_radix_cache": False,
        # Compile materializes the KV pool as graph args + outputs, so an
        # unbounded pool doubles the precompile footprint.
        "max_total_tokens": 131072,
        "mem_fraction_static": 0.5,
        # The EXTEND/decode captures outlast the 300s default and trip the
        # watchdog into repeated stack dumps mid-compile.
        "watchdog_timeout": 3600,
    }
)
