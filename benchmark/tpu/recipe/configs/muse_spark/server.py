"""Muse Spark (816B-A42B NVFP4) on one tpu7x host, 4 chips x 2 cores = tp8.

Experts are tensor-parallel across all 8 ranks (ep 1): each rank holds a slice of
every expert's intermediate dim, which balances decode. NVFP4 experts run
through the Pallas fused-RS kernel; attention is RPA v3 over the hybrid SWA pool
(46 of 62 layers use a 2048-token window).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/server.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG, LIBTPU_FLAGS

CONTEXT_LENGTH = 262144

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
        "tp_size": 8,
        "ep_size": 1,
        "quantization": "modelopt_fp4",
        "moe_runner_backend": "pallas_tpu_fused_rs",
        "kv_cache_dtype": "bfloat16",
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "tpu_compile_bs": [1, 2, 4, 8],
        "tpu_compile_tokens": [1024, 2048, 3072, 4096],
        "swa_full_tokens_ratio": 0.125,
        "mem_fraction_static": 0.92,
        "disable_radix_cache": False,
        "reasoning_parser": "muse",
        "tool_call_parser": "muse",
        "watchdog_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        "FORCE_SINGLE_HOST": "1",
        # Pad prefill token tensors on the host; padding on the TPU compiles a
        # program per distinct prompt length.
        "SGLANG_TPU_HOST_PADDING": "1",
        # The measured configuration also keeps the SparseCore all-gather offload off.
        "LIBTPU_INIT_ARGS": " ".join(
            LIBTPU_FLAGS
            + ("--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",)
        ),
    }
)
