"""GPT-OSS-20B: one TP8/EP8 instance with high-reasoning evaluation gates."""

ENV = {
    "MODEL": "openai/gpt-oss-20b",
    "FORCE_SINGLE_HOST": "1",
    "TP_SIZE": "8",
    "EP_SIZE": "8",
    "QUANTIZATION": "",
    "KV_CACHE_DTYPE": "fp8_e4m3",
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "MEM_FRACTION_STATIC": "0.8",
    "PAGE_SIZE": "128",
    "DISABLE_RADIX_CACHE": "1",
    "MAX_RUNNING": "128",
    "TORCH_COMPILE_MAX_BS": "128",
    "CONTEXT_LEN": "131072",
    "MAX_TOTAL_TOKENS": "8388608",
    "CHUNKED_PREFILL_SIZE": "4096",
    "MAX_PREFILL_TOKENS": "4096",
    "TPU_COMPILE_TOKENS": "4096",
    "TPU_COMPILE_BS": "1 8 32 64 128",
    "DISABLE_OVERLAP_SCHEDULE": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
    # Match the validated C128 run. Cap generation to the remaining context;
    # the evaluator requests the full 131072-token budget before prompt tokens.
    "EXTRA_ARGS": "--served-model-name gpt-oss-20b --reasoning-parser gpt-oss --tpu-compile-tracer dynamo --allow-auto-truncate --cuda-graph-bs 1 8 32 64 128",
}
