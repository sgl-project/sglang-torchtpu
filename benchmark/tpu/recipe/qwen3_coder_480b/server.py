"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 0
PREFILL_TOKENS = INPUT_TOKENS

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tp_size": 8,
        "ep_size": 8,
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        "mem_fraction_static": 0.8,
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 1 * CONCURRENCY * CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tracer": "direct",
        "tpu_fp8_runtime": "channelwise",
        "sampling_defaults": "openai",
        "enable_fused_qk_norm_rope": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "2048",
        "TORCH_TPU_INTERNAL_XLA_OPTIONS": "xla_tpu_use_dynamic_smem_negotiation=true",
    }
)
