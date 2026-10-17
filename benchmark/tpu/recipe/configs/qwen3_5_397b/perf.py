"""Qwen3.5-397B-A17B-FP8 throughput config: tp8/ep8 on one tpu7x host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_5_397b/perf.py

Ported from the measured recipe on branch haolin/qwen35-recipe-dp-attention
(benchmark/tpu/recipe/qwen3_5_397b/server.sh), which published this model's
numbers at this exact 8k1k shape. Every value below that departs from the base
config came from a measurement there; the comments say which.

A hybrid model: 45 of 60 layers are GDN linear attention with a fixed-size
recurrent state, 15 are full attention. That shapes the slot and pool math.

The recipe's DP-attention path (dp_size 8, the prefill delayer) is how this
model is served at high concurrency. It is left out here: at CONCURRENCY=16 the
delayer has nothing to align, and the whole lane measures the TP path.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG, LIBTPU_FLAGS

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
CONCURRENCY = 16
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 20
PREFILL_TOKENS = 4096

# max_mamba_cache_size follows max_running_requests, so running == concurrency
# leaves one GDN slot per in-flight request and none spare: a replacement waits
# for one specific request to finish. Measured at CONCUR=4: p99 TTFT 10797 ms
# with no spare slots, 1108 ms with four.
MAX_RUNNING = CONCURRENCY + 4
# A mamba slot bills against the SAME token budget as KV (~2048 token
# equivalents), so the pool is the in-flight contexts plus every slot's charge.
# Sizing it at CONCURRENCY * CONTEXT_LENGTH over-commits and only 15 fit.
MAX_TOTAL_TOKENS = CONCURRENCY * CONTEXT_LENGTH + MAX_RUNNING * 2048

# One departure from the base set, measured here: the SparseCore GATHER offload
# stays ON (off, the s32[40960] inverse-permutation gather costs ~24 ms/step).
# The all-gather collective offload takes its slot.
_LIBTPU_FLAGS = [
    (
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
        if flag == "--xla_tpu_offload_gather_to_sparsecore=false"
        else flag
    )
    for flag in LIBTPU_FLAGS
]

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.5-397B-A17B-FP8",
        "served_model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
        "tp_size": 8,
        "ep_size": 8,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_v3",
        "page_size": 256,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        # CI compiles cold every run: one prefill and one decode bucket.
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "tpu_compile_bs": [MAX_RUNNING],
        "mem_fraction_static": 0.85,
        # Text only. The ViT never executes, and without this its 29 patch
        # buckets still precompile: 88 s warm, 296 s cold, of pure waste.
        "language_only": True,
        "tpu_fp8_runtime": "channelwise",
    }
)
CONFIG["runtime_env"].update(
    {
        "LIBTPU_INIT_ARGS": " ".join(_LIBTPU_FLAGS),
        # Decode takes the dense MoE permute below this, a prefill chunk the
        # SparseCore gather above it. Must clear MAX_RUNNING * topk (10 here):
        # at 1024 a batch over 102 drops the unpermute onto the gather path,
        # measured as ITL +7.9%. 1024 covers MAX_RUNNING=20 and stays <= CHUNK.
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "1024",
        # Build the RoPE cache at the served context, not the checkpoint's
        # 262144 positions. Safe only because this serves text.
        "SGLANG_TPU_ENABLE_ROPE_CACHE_SLICE": "1",
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "",
    }
)
