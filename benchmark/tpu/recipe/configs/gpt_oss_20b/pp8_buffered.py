"""GPT-OSS-20B (MXFP4 MoE) on one tpu7x host as eight single-device pipeline
stages with the buffered point-to-point hand-off.

The pool is profiled, not capped: the first stage (embedding) and the last
(logits) have less free memory than the middle stages, so each stage sizes its
KV budget on its own memory and the stages agree on the smallest token count.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/pp8_buffered.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 32
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "pp_size": 8,
        "tp_size": 1,
        "pp_max_micro_batch_size": 4,
        # As the other single-device PP recipes (gpt_oss_120b/pd_singlehost_pp_spec.py).
        "disable_overlap_schedule": True,
        # The checkpoint's own mxfp4 config; fp8 would re-quantize it.
        "quantization": None,
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        "kv_cache_dtype": "bfloat16",
        "context_length": 16384,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_tokens": [512, 1024, 2048, PREFILL_TOKENS],
        "tpu_compile_bs": [8, 16, CONCURRENCY],
        "mem_fraction_static": 0.8,
        "tpu_compile_tracer": "dynamo",
        "sampling_backend": "pytorch",
        # A stage waiting for the previous one looks idle to the watchdog.
        "watchdog_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        # SparseCore ragged-gather MoE cannot be lowered for GPT-OSS (see gpt_oss_120b/server.py).
        "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
        "SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF": "1",
    }
)
