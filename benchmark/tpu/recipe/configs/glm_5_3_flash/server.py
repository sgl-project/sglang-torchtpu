"""GLM-5.3-Flash text serving on one tpu7x host (2x2x1 = 4 chips x 2 cores = tp8).

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/server.py
    # add --dry-run to print the resolved sglang.launch_server command.

The 11 full-attention layers run the model's DeepSeek Sparse Attention: the
pooled (kpool) indexer picks index_topk tokens per query, and absorbed MLA attends
over them on pallas_tpu_mla_sparse, with an FP8 latent cache. The 34 KDA layers
run on the fused GDN v3 kernel in KDA mode. SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT
switches to dense attention, which matches the model only up to index_topk tokens.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 1
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 1024
MAX_RUNNING = CONCURRENCY if CONCURRENCY == 1 else CONCURRENCY + 2
COMPILE_BS = [b for b in (1, 2, 4, 8, 16) if b <= CONCURRENCY] or [1]

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/GLM-5.3-Flash",
        "tp_size": 8,
        # Tensor-parallel experts: at small batch EP8 waits on the busiest rank
        # (~2.6 of a token's 8 experts); EP=1 gives every rank 1/8 of each expert.
        "ep_size": 1,
        "language_model_only": True,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": INPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        # Every extend pads to the next bucket: with 8192 alone, a short prompt
        # or the tail chunk of a longer one costs a full 8k prefill.
        "tpu_compile_tokens": [1024, 2048, 4096, 8192],
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": COMPILE_BS,
        "max_total_tokens": max(2, MAX_RUNNING) * CONTEXT_LENGTH,
        # KDA state slots: one per running request, plus spare.
        "max_mamba_cache_size": MAX_RUNNING + 2,
        "attention_backend": "pallas_tpu_mla_sparse",
        "kv_cache_dtype": "fp8_e4m3",
        "page_size": 128,
        "disable_radix_cache": True,
        "mem_fraction_static": 0.9,
        "disable_shared_experts_fusion": True,
        "watchdog_timeout": 1800,
    }
)
CONFIG["runtime_env"].update({"SGLANG_MAMBA_SSM_DTYPE": "float32"})
# libtpu >= 0.0.46 (sglang-torchtpu#1137, fixed in 0.0.51): the fast-merge buffer
# assignment fallback misplaces SparseCore kernel buffers in this model's forward
# once the KV pool grows, halting the SparseCore or feeding the MoE router NaN.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] += (
    " --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false"
)
