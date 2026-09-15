"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 1024
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64
PREFILL_TOKENS = min(max(INPUT_TOKENS, 2048), 8192)

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-120b",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tp_size": 8,
        "ep_size": 8,
        "quantization": None,
        "page_size": 64,
        "disable_radix_cache": False,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_total_tokens": 2 * CONCURRENCY * CONTEXT_LENGTH,
        "mem_fraction_static": 0.6,
        "tpu_compile_tracer": "direct",
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
