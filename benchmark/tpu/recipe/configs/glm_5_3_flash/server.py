"""GLM-5.3-Flash text serving on one tpu7x host (tp8), DeepSeek Sparse Attention.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server.py

GLM_FLASH_DENSE_MAX_CONTEXT=<tokens> serves dense MLA instead (no indexer), at
that context length; it matches the model only up to index_topk tokens.
"""

import json
import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 1
DENSE_MAX_CONTEXT = os.environ.get("GLM_FLASH_DENSE_MAX_CONTEXT")
CONTEXT_LENGTH = (
    int(DENSE_MAX_CONTEXT) if DENSE_MAX_CONTEXT else INPUT_TOKENS + OUTPUT_TOKENS + 1024
)
MAX_RUNNING = CONCURRENCY if CONCURRENCY == 1 else CONCURRENCY + 2
COMPILE_BS = [b for b in (1, 2, 4, 8, 16) if b <= CONCURRENCY] or [1]

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/GLM-5.3-Flash",
        "tp_size": 8,
        # At small batch EP8 waits on the busiest rank; EP=1 splits every expert.
        "ep_size": 1,
        "language_model_only": True,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        # Smaller buckets so short prompts and tail chunks skip a full 8k prefill.
        "tpu_compile_tokens": [1024, 2048, 4096, 8192],
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": COMPILE_BS,
        "max_total_tokens": max(2, MAX_RUNNING) * CONTEXT_LENGTH,
        "max_mamba_cache_size": MAX_RUNNING + 2,
        "attention_backend": (
            "pallas_tpu_mla" if DENSE_MAX_CONTEXT else "pallas_tpu_mla_sparse"
        ),
        "kv_cache_dtype": "fp8_e4m3",
        "page_size": 128,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.9,
        "disable_shared_experts_fusion": True,
        "watchdog_timeout": 1800,
    }
)
if DENSE_MAX_CONTEXT:
    # is_deepseek_dsa reads index_topk from both config levels.
    CONFIG["server_args"]["json_model_override_args"] = json.dumps(
        {"index_topk": None, "text_config": {"index_topk": None}}
    )
CONFIG["runtime_env"].update(
    {
        "SGLANG_MAMBA_SSM_DTYPE": "float32",
        # The TileLang mHC kernels and the fusion built on them are CUDA-only.
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
        "SGLANG_OPT_FUSE_MHC_POST_PRE": "0",
        "SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC": "1",
        "SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL": "1",
        "SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL": "1",
        # f32 MoE scales skip a per-step scale relayout in gmm_v2.
        "SGLANG_TPU_FP8_MOE_FP32_SCALES": "1",
        # Route-meta permute (single-token SMEM counting sort) for small batches.
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "512",
    }
)
# libtpu >= 0.0.46 bug (#1137, fixed in 0.0.51): with large KV pools the fast-merge
# fallback corrupts SparseCore buffers (SparseCore halt or NaN routing).
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] += (
    " --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false"
)
