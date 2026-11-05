"""Inkling-2-Small on one tpu7x host (4 chips, 8 TensorCores), tp8 / ep8.

NVFP4 routed experts on the fused reduce-scatter kernel, MXFP8 layers as
channelwise FP8, Pallas ragged paged attention with an FP8 KV cache, prefix
cache on, DSpark speculative decoding on the bundled MTP head, and the full
1M-token window.

Sample usage (from the repo root):

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/inkling2/server.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import CONFIG as _BASE_CONFIG

MODEL_PATH = "thinkingmachines/Inkling-2-Small"

CONFIG = deepcopy(_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": MODEL_PATH,
        "host": "0.0.0.0",
        "tp_size": 8,
        "ep_size": 8,
        "attention_backend": "pallas_tpu_rpa_v3",
        "moe_runner_backend": "pallas_tpu_fused_rs",
        "kv_cache_dtype": "fp8_e4m3",
        # 512-token pages keep the attention page table of 64 requests at 1M
        # tokens inside the kernel's scalar memory.
        "page_size": 512,
        "context_length": 1048576,
        "chunked_prefill_size": 8192,
        "max_running_requests": 64,
        "mem_fraction_static": 0.85,
        # Prefix cache: the conv-state pool sized for 64 running requests, and
        # the track interval a multiple of the page size.
        "mamba_track_interval": 512,
        "max_mamba_cache_size": 352,
        "swa_full_tokens_ratio": 0.05,
        "enable_torch_compile": True,
        "tpu_compile_tracer": "direct",
        "tpu_compile_bs": [1, 8, 16, 32, 64],
        "tpu_compile_tokens": [1024, 2048, 4096, 8192],
        "speculative_algorithm": "DSPARK",
        "speculative_draft_model_path": MODEL_PATH,
        "speculative_num_draft_tokens": 8,
        "grammar_backend": "xgrammar",
        "enable_multimodal": True,
        "reasoning_parser": "inkling",
        "tool_call_parser": "inkling",
    }
)
CONFIG["runtime_env"].update(
    {
        "LIBTPU_INIT_ARGS": "--xla_tpu_user_reserved_hbm_bytes=8589934592",
        "SGLANG_TPU_INKLING2_NVFP4_EXPERTS": "1",
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "1024",
        "SGLANG_TPU_FUSED_MOE_SC_GATHER_MIN_ROWS": "1024",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "SGLANG_SPEC_TP_SYNC": "rng",
        # The draft head's config caps positions at 262144; the target serves 1M.
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "sglang_tpu.multimodal",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH": "Inkling2ForConditionalGeneration",
    }
)
