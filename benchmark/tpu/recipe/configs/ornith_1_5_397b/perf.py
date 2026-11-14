"""Ornith-1.5-397B-FP8 throughput config: tp8/ep8 on one tpu7x host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/ornith_1_5_397b/perf.py

Ornith-1.5-397B is a Qwen3.5-397B-A17B fine-tune (same qwen3_5_moe dims), so this
is recipe/configs/qwen3_5_397b/perf.py with two differences:

  * the checkpoint is compressed-tensors W8A8-FP8, channelwise weights and dynamic
    per-token activations, so quantization is left to auto-detection (passing
    "fp8" fails _verify_quantization) and tpu_fp8_runtime has nothing to convert.
    Its GDN projections stay bf16, where Qwen's FP8 checkpoint quantizes them.
  * one server covers both 8k1k up to c64 and 256k1k up to c4, so the context is
    the 256k shape's and the pool the larger of the two.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

SHORT_CONTEXT = 8192 + 1024 + 20
SHORT_CONCURRENCY = 64
LONG_CONTEXT = 262144 + 1024 + 20
LONG_CONCURRENCY = 4
PREFILL_TOKENS = 4096

# Four spare GDN slots, as in qwen3_5_397b/perf.py: without them a replacement
# request waits on one specific request to finish.
MAX_RUNNING = SHORT_CONCURRENCY + 4
# A mamba slot bills ~2048 token equivalents against the same pool as KV.
MAX_TOTAL_TOKENS = (
    max(SHORT_CONCURRENCY * SHORT_CONTEXT, LONG_CONCURRENCY * LONG_CONTEXT)
    + MAX_RUNNING * 2048
)

CONFIG = deepcopy(_FP8_BASE_CONFIG)
del CONFIG["server_args"]["quantization"]
CONFIG["server_args"].update(
    {
        "model_path": "/dev/shm/Ornith-1.5-397B-FP8",
        "served_model_name": "ornith-ai/Ornith-1.5-397B-FP8",
        "tp_size": 8,
        "ep_size": 8,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_v3",
        "page_size": 256,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "context_length": LONG_CONTEXT,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        # c1/c4 pad to 8, c16 to 20, c64 to 68.
        "tpu_compile_bs": [8, 20, MAX_RUNNING],
        "mem_fraction_static": 0.85,
        "language_only": True,
        "trust_remote_code": True,
        "watchdog_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        # Same departures from the base set as qwen3_5_397b/perf.py, measured there.
        "LIBTPU_INIT_ARGS": " ".join(
            [
                "--xla_tpu_enable_async_collective_fusion=true",
                "--xla_tpu_overlap_compute_collective_tc=true",
                "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
                "--xla_tpu_enable_latency_hiding_scheduler=true",
                "--xla_latency_hiding_scheduler_rerun=10",
                "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
                "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
                "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            ]
        ),
        # Must clear MAX_RUNNING * topk (680) and stay <= PREFILL_TOKENS.
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "1024",
        "SGLANG_TPU_ENABLE_ROPE_CACHE_SLICE": "1",
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "",
    }
)
