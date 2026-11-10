"""Qwen3.5-397B-A17B-FP8 Single-Turn (8k in / 1k out) Recipe on Cloud TPU v7x.

Targets the single-turn 8,192 input / 1,024 output token FP8 workload on a
single Cloud TPU v7x 2x2x1 host (4 chips, 8 TensorCores). Incorporates all
validated host-overhead, memory-budget, kernel, collective, and scheduler
optimizations:
- page-size 256 with Pallas RPA v3 attention and channelwise FP8 runtime
- SparseCore gather offload enabled with all-gather collective offload disabled
- Scoped VMEM limit of 32 MiB (32768 KiB)
- Dynamic dense MoE permute threshold tuned for batch size and top-k = 10
- TorchDynamo graph attribute lifting (level 2), guard skip on decode, lean AOT
  runtime wrapper, split-exec direct calls, and decode alloc fast path
- Scheduler thread core pinning with busy-core avoidance and synchronous decode
  step dispatch (SGLANG_TPU_ASYNC_COPY_WORKER=0)
- RoPE cache sliced to served context length (10240 tokens)
- Text-only execution (language_model_only=True) skipping vision compile/weights
- Concurrency-aware hybrid parallelism across C=4 through C=256:
    * Below C=64: TP8/EP8 with 20480 prefill tokens, five prefill buckets
      [4096, 8192, 10240, 16384, 20480], prefill_max_requests=3,
      scheduler_recv_interval=2, hierarchical SparseCore all-reduce, and
      two-shot Pallas decode all-reduce.
    * At or above C=64: synchronized DP8/TP8/EP8 (attn_dp_size=8) with 32768
      global prefill tokens (4096 per rank), compile buckets [2048, 4096],
      scheduler_recv_interval=2, prefill_drain_interval=256, scaled per-rank
      batch size (32/48/64), max_mamba_cache_size (320/448/576),
      max_total_tokens (786432/851968/983040), multi-chunk 128 MiB Pallas RDMA
      all-gather, SparseCore reduce-scatter offload, chunked fused MoE decode
      up to 512 tokens, and busy-core-aware scheduler thread pinning.
    * Override layout explicitly via dp_attention=False/True or
      SGLANG_RECIPE_DP_ATTENTION.

Usage:
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_5_397b/single_turn_8k1k.py --concurrency 16
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG, LIBTPU_FLAGS


def _env_str(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is None:
        return default
    stripped = val.strip()
    return stripped if stripped != "" else default


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = _env_str(name, "")
    if raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = _env_str(name, "")
    if raw == "":
        return default
    return float(raw)


INPUT_TOKENS = 8192
OUTPUT_TOKENS = 1024
DEFAULT_CONCURRENCY = 16
# Page-aligned context length matching 40 * 256 = 10240 tokens.
CONTEXT_LENGTH: int = _env_int("SGLANG_RECIPE_CONTEXT_LENGTH", 10240)  # type: ignore[assignment]
PREFILL_TOKENS: int = _env_int("SGLANG_RECIPE_PREFILL_TOKENS", 20480)  # type: ignore[assignment]


def resolve_model_path() -> str:
    """Resolve local weights directory or fallback model path."""
    explicit_path = os.environ.get("QWEN35_397B_WEIGHTS_PATH", "")
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    candidate_paths = [
        "/weights/Qwen3.5-397B-A17B-FP8",
        "/weights",
        "/workspace/weights/Qwen3.5-397B-A17B-FP8",
    ]
    for path in candidate_paths:
        if path and os.path.exists(os.path.join(path, "config.json")):
            return path

    return "/weights/Qwen3.5-397B-A17B-FP8"


def _build_libtpu_flags(attn_dp_size: int = 1) -> list[str]:
    """Build LIBTPU_INIT_ARGS flags for Cloud TPU v7x."""
    flags = [
        (
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
            if flag == "--xla_tpu_offload_gather_to_sparsecore=false"
            else flag
        )
        for flag in LIBTPU_FLAGS
    ]

    # Scoped VMEM limit (32 MiB is optimal for 397B decode step ITL)
    raw_vmem = os.environ.get("SGLANG_RECIPE_VMEM_KIB")
    vmem_str = raw_vmem.strip() if raw_vmem is not None else ""
    if vmem_str == "":
        vmem_str = "32768"
    if vmem_str.lower() not in ("0", "none", "off", "false"):
        flags.append(f"--xla_tpu_scoped_vmem_limit_kib={vmem_str}")

    # Hierarchical SparseCore all-reduce offload
    if _env_int("SGLANG_RECIPE_TPU_COLLECTIVE_V2", 0) == 1:
        flags.append(
            "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608"
        )
        flags.append("--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true")
        flags.append(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
        )

    # Verified TPU v7x collective offloads
    if _env_int("SGLANG_RECIPE_AIPERF_COLLECTIVES", 1) == 1:
        flags = [
            f
            for f in flags
            if not f.startswith(
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce="
            )
            and not f.startswith(
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather="
            )
            and not f.startswith(
                "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter="
            )
        ]
        flags.extend(
            [
                "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
                "--xla_tpu_ars_combiner_threshold_in_bytes=0",
                "--xla_tpu_enable_async_collective_merger=false",
                "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
                "--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true",
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608",
            ]
        )
        if _env_int("SGLANG_RECIPE_OFFLOAD_AG", 0) == 1:
            flags.extend(
                [
                    "--xla_enable_async_all_gather=true",
                    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
                    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144",
                    "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
                    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false",
                ]
            )
        else:
            flags.append(
                "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
            )

        if attn_dp_size > 1 or _env_int("SGLANG_RECIPE_OFFLOAD_RS", 0) == 1:
            flags.extend(
                [
                    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
                    "--xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=1048576",
                ]
            )
        else:
            flags.append(
                "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
            )

    extra_flags = _env_str("SGLANG_EXTRA_LIBTPU_ARGS", "")
    if extra_flags:
        flags.extend(extra_flags.split())

    return flags


def get_config(
    concurrency: int = DEFAULT_CONCURRENCY,
    dp_attention: bool | None = None,
) -> dict[str, Any]:
    """Generate server configuration tailored to target concurrency."""
    context_length: int = _env_int("SGLANG_RECIPE_CONTEXT_LENGTH", 10240) or 10240
    compile_tokens_str = _env_str("SGLANG_RECIPE_COMPILE_TOKENS", "")

    # Precedence for attention DP size: explicit env, dp_attention arg/env,
    # then hybrid default (DP8 at concurrency 64 and above).
    explicit_attn_dp_size = _env_int("SGLANG_RECIPE_ATTN_DP_SIZE", None)
    if explicit_attn_dp_size is not None:
        attn_dp_size = explicit_attn_dp_size
    elif dp_attention is not None:
        attn_dp_size = 8 if dp_attention else 1
    else:
        explicit_dp_attention = _env_str("SGLANG_RECIPE_DP_ATTENTION", "")
        if explicit_dp_attention != "":
            attn_dp_size = (
                8 if explicit_dp_attention.lower() in ("1", "true", "yes") else 1
            )
        else:
            attn_dp_size = 8 if concurrency >= 64 else 1

    default_prefill_tokens = 20480 if attn_dp_size == 1 else 32768
    prefill_tokens: int = (
        _env_int("SGLANG_RECIPE_PREFILL_TOKENS", default_prefill_tokens)
        or default_prefill_tokens
    )

    override_max_running = _env_int("SGLANG_RECIPE_MAX_RUNNING", None)
    enable_mixed_chunk = _env_int("SGLANG_RECIPE_ENABLE_MIXED_CHUNK", 0) == 1
    prefill_max_requests = _env_int(
        "SGLANG_RECIPE_PREFILL_MAX_REQUESTS",
        3
        if (attn_dp_size == 1 and prefill_tokens >= 16384 and not enable_mixed_chunk)
        else None,
    )
    if prefill_max_requests is not None and prefill_max_requests <= 0:
        prefill_max_requests = None

    sched_recv_interval = _env_int("SGLANG_RECIPE_SCHED_RECV_INTERVAL", 2)
    if sched_recv_interval is not None and sched_recv_interval <= 0:
        sched_recv_interval = None

    mem_fraction = _env_float("SGLANG_RECIPE_MEM_FRACTION", None)

    # Determine parallel architecture and token budgets.
    if attn_dp_size == 1:
        # Pure TP8/EP8 mode
        if compile_tokens_str:
            compile_tokens = sorted(
                {int(x.strip()) for x in compile_tokens_str.split(",") if x.strip()}
            )
        elif prefill_tokens >= 20480:
            compile_tokens = sorted({4096, 8192, 10240, 16384, prefill_tokens})
        elif prefill_tokens >= 16384:
            # Quad buckets eliminate the 8193..9830 to 16384 padding cliff.
            compile_tokens = [4096, 8192, 10240, prefill_tokens]
        else:
            compile_tokens = [prefill_tokens]

        if mem_fraction is None:
            mem_fraction = 0.94 if concurrency >= 256 else 0.85

        if override_max_running is not None:
            max_running = override_max_running
        elif concurrency == 1:
            max_running = 1
        elif concurrency <= 64:
            # Round up to multiple of 8 so decode buckets reach sort-free rowmax_topk.
            max_running = ((concurrency + 4 + 7) // 8) * 8
        elif concurrency == 256:
            max_running = 256
        else:
            max_running = ((concurrency + 8 + 7) // 8) * 8

        mamba_cache_size = _env_int("SGLANG_RECIPE_MAMBA_CACHE_SIZE", max_running + 8)

        override_max_total_tokens = _env_int("SGLANG_RECIPE_MAX_TOTAL_TOKENS", None)
        if override_max_total_tokens is not None:
            max_total_tokens = override_max_total_tokens
        elif concurrency >= 256:
            max_total_tokens = 2490368
        else:
            raw_tokens = concurrency * context_length + (max_running + 2) * 3584
            max_total_tokens = ((raw_tokens + 255) // 256) * 256

        compile_bs = sorted(
            {b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256] if b <= max_running}
            | {max_running}
        )
        global_decode_rows = max_running

        parallel_server_args = {
            "tp_size": 8,
            "ep_size": 8,
            "max_running_requests": max_running,
            "max_mamba_cache_size": mamba_cache_size,
            "torch_compile_max_bs": max_running,
            "tpu_compile_bs": compile_bs,
            "tpu_compile_tokens": compile_tokens,
            "max_total_tokens": max_total_tokens,
            "chunked_prefill_size": prefill_tokens,
            "max_prefill_tokens": prefill_tokens,
        }
        if prefill_max_requests is not None:
            parallel_server_args["prefill_max_requests"] = prefill_max_requests
        if sched_recv_interval is not None:
            parallel_server_args["scheduler_recv_interval"] = sched_recv_interval
    else:
        # DP attention mode (for example DP8/TP8/EP8 at attn_dp_size=8).
        dp_ranks = attn_dp_size
        per_dp_prefill = prefill_tokens // dp_ranks
        if compile_tokens_str:
            compile_tokens = sorted(
                {int(x.strip()) for x in compile_tokens_str.split(",") if x.strip()}
            )
        elif per_dp_prefill >= 8192:
            compile_tokens = sorted({2048, 4096, 8192, per_dp_prefill})
        elif per_dp_prefill >= 4096:
            compile_tokens = sorted({2048, 4096, per_dp_prefill})
        else:
            compile_tokens = [per_dp_prefill]

        if mem_fraction is None:
            mem_fraction = 0.92 if concurrency >= 256 else 0.90

        if override_max_running is not None:
            per_rank_bs = max(
                8, ((math.ceil(override_max_running / dp_ranks) + 7) // 8) * 8
            )
            max_running = per_rank_bs * dp_ranks
        elif concurrency >= 256:
            per_rank_bs = 64
            max_running = per_rank_bs * dp_ranks
        elif concurrency >= 128:
            per_rank_bs = 48
            max_running = per_rank_bs * dp_ranks
        elif dp_ranks >= 8:
            per_rank_bs = 32
            max_running = per_rank_bs * dp_ranks
        else:
            per_rank_bs = min(
                32, max(16, ((math.ceil(concurrency / dp_ranks) + 7) // 8) * 8)
            )
            max_running = per_rank_bs * dp_ranks

        per_rank_conc = math.ceil(concurrency / dp_ranks)
        per_rank_mamba_slots = _env_int(
            "SGLANG_RECIPE_PER_RANK_MAMBA_SLOTS", per_rank_bs + 8
        )
        mamba_cache_size = _env_int(
            "SGLANG_RECIPE_MAMBA_CACHE_SIZE", per_rank_mamba_slots * dp_ranks
        )

        override_max_total_tokens = _env_int("SGLANG_RECIPE_MAX_TOTAL_TOKENS", None)
        if override_max_total_tokens is not None:
            max_total_tokens = override_max_total_tokens
        elif dp_ranks >= 8:
            if concurrency >= 256:
                max_total_tokens = 983040
            elif concurrency >= 128:
                max_total_tokens = 851968
            else:
                max_total_tokens = 786432
        else:
            raw_tokens = max(per_rank_conc, per_rank_bs - 8) * context_length + 8 * max(
                per_dp_prefill, 2048
            )
            max_total_tokens = ((raw_tokens + 255) // 256) * 256

        compile_bs = sorted(
            {b for b in [1, 2, 4, 8, 16, 32, 64] if b <= per_rank_bs} | {per_rank_bs}
        )
        global_decode_rows = per_rank_bs * dp_ranks

        parallel_server_args = {
            "tp_size": 8,
            "attn_dp_size": dp_ranks,
            "ep_size": 8,
            "enable_dp_lm_head": True,
            "max_running_requests": max_running,
            "max_mamba_cache_size": mamba_cache_size,
            "torch_compile_max_bs": per_rank_bs,
            "tpu_compile_bs": compile_bs,
            "tpu_compile_tokens": compile_tokens,
            "max_total_tokens": max_total_tokens,
            # parallel_hook automatically divides chunked_prefill_size by attn_dp_size.
            "chunked_prefill_size": prefill_tokens,
            "max_prefill_tokens": per_dp_prefill,
        }
        if prefill_max_requests is not None:
            parallel_server_args["prefill_max_requests"] = prefill_max_requests
        if sched_recv_interval is not None:
            parallel_server_args["scheduler_recv_interval"] = sched_recv_interval

    moe_runner_backend = _env_str(
        "SGLANG_RECIPE_MOE_RUNNER_BACKEND", "pallas_tpu_fused_decode"
    )
    fmd_max_tokens = _env_int("SGLANG_RECIPE_FUSED_MOE_DECODE_MAX_TOKENS", None)
    if fmd_max_tokens is not None:
        fmd_limit = fmd_max_tokens
    else:
        fmd_limit = max(256, global_decode_rows)
    uses_fmd_for_all_decode = (
        moe_runner_backend.lower() == "pallas_tpu_fused_decode"
        and global_decode_rows <= fmd_limit
    )

    # Dynamic Dense MoE Permute Threshold (topk=10 for Qwen3.5-397B).
    override_dense_thresh = _env_int("SGLANG_RECIPE_DENSE_PERMUTE_THRESHOLD", None)
    if override_dense_thresh is not None:
        dense_thresh = override_dense_thresh
    elif _env_int("SGLANG_RECIPE_EP_COUNTING_SORT", 1) == 1:
        dense_thresh = 512
    elif uses_fmd_for_all_decode:
        # Decode buckets run via fused_moe_decode; keep threshold at 1024 so
        # prefill buckets at 1024 and above hit SparseCore counting sort.
        dense_thresh = 1024
    else:
        min_global_prefill_bucket = min(compile_tokens) * attn_dp_size
        dense_thresh = 1024
        while (
            dense_thresh < global_decode_rows * 10 + 1
            and dense_thresh * 2 <= min_global_prefill_bucket
        ):
            dense_thresh *= 2

    config = deepcopy(_FP8_BASE_CONFIG)

    server_args_extra: dict[str, Any] = {}
    if moe_runner_backend.lower() not in ("", "auto", "none"):
        server_args_extra["moe_runner_backend"] = moe_runner_backend
    if enable_mixed_chunk:
        server_args_extra["enable_mixed_chunk"] = True

    enable_delayer_env = _env_str("SGLANG_RECIPE_ENABLE_DELAYER", "")
    if enable_delayer_env != "":
        enable_delayer = enable_delayer_env.lower() in ("1", "true", "yes")
    else:
        enable_delayer = False

    if enable_delayer:
        server_args_extra["enable_prefill_delayer"] = True
        server_args_extra["prefill_delayer_max_delay_passes"] = _env_int(
            "SGLANG_RECIPE_DELAYER_PASSES", 64 if attn_dp_size == 1 else 128
        )
        server_args_extra["prefill_delayer_max_delay_ms"] = _env_int(
            "SGLANG_RECIPE_DELAYER_MAX_MS", 500 if attn_dp_size == 1 else 2500
        )
        delayer_queue_ratio = _env_float("SGLANG_RECIPE_DELAYER_QUEUE_RATIO", None)
        if delayer_queue_ratio is not None:
            server_args_extra["prefill_delayer_queue_min_ratio"] = delayer_queue_ratio

    prefill_decode_interval = _env_int("SGLANG_RECIPE_PREFILL_DECODE_INTERVAL", None)
    if prefill_decode_interval is not None and prefill_decode_interval > 0:
        server_args_extra["prefill_decode_interval"] = prefill_decode_interval

    attention_backend = _env_str("SGLANG_RECIPE_ATTENTION_BACKEND", "pallas_tpu_rpa_v3")

    runtime_env_extra: dict[str, str] = {}
    if _env_int("SGLANG_RECIPE_SMALL_M_FP8", 0) == 1:
        runtime_env_extra["SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL"] = "1"

    if _env_int("SGLANG_RECIPE_SEQ_ALONG_LANE_KV", 0) == 1:
        runtime_env_extra["SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"] = "1"

    mamba_ssm_dtype = _env_str("SGLANG_RECIPE_MAMBA_SSM_DTYPE", "")
    if mamba_ssm_dtype:
        runtime_env_extra["SGLANG_MAMBA_SSM_DTYPE"] = mamba_ssm_dtype

    if _env_int("SGLANG_RECIPE_PREMAPPED_16G", 1) == 1:
        runtime_env_extra["TPU_PREMAPPED_BUFFER_SIZE"] = "17179869184"
        runtime_env_extra["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = (
            "17179869184"
        )

    if fmd_max_tokens is not None or fmd_limit > 256:
        runtime_env_extra["SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS"] = str(fmd_limit)

    if _env_int("SGLANG_RECIPE_EP_COUNTING_SORT", 1) == 1:
        runtime_env_extra["SGLANG_TPU_FUSED_MOE_EP_COUNTING_SORT"] = "1"

    ragged_ver = _env_int("SGLANG_RECIPE_RAGGED_GATHER_REDUCE_VERSION", 3)
    if ragged_ver is not None:
        runtime_env_extra["SGLANG_TPU_RAGGED_GATHER_REDUCE_VERSION"] = str(ragged_ver)

    if _env_int("SGLANG_RECIPE_STRICT_TOKEN_RESERVE", 0) == 1:
        runtime_env_extra["SGLANG_INIT_NEW_TOKEN_RATIO"] = "1.0"
        runtime_env_extra["SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR"] = "1.0"

    drain_interval = _env_int(
        "SGLANG_RECIPE_DRAIN_PREFILL_INTERVAL",
        256 if (attn_dp_size > 1 and not enable_delayer) else 0,
    )
    if drain_interval > 0:
        runtime_env_extra["SGLANG_TPU_PREFILL_DRAIN_INTERVAL"] = str(drain_interval)

    default_twoshot_allreduce = 1 if attn_dp_size == 1 else 0
    if _env_int("SGLANG_RECIPE_TWOSHOT_ALLREDUCE", default_twoshot_allreduce) == 1:
        runtime_env_extra["SGLANG_TPU_ENABLE_TWOSHOT_ALLREDUCE"] = "1"

    default_rdma_all_gather = 1 if attn_dp_size > 1 else 0
    if _env_int("SGLANG_RECIPE_RDMA_ALL_GATHER", default_rdma_all_gather) == 1:
        runtime_env_extra["SGLANG_TPU_ENABLE_RDMA_ALL_GATHER"] = "1"

    config["server_args"].update(
        {
            "model_path": resolve_model_path(),
            "served_model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e4m3",
            "tpu_fp8_runtime": "channelwise",
            "attention_backend": attention_backend,
            "page_size": 256,
            "context_length": context_length,
            "mem_fraction_static": mem_fraction,
            "language_model_only": True,
            "trust_remote_code": True,
            "watchdog_timeout": 3600,
            "disable_radix_cache": True,
            "enable_metrics": True,
            **parallel_server_args,
            **server_args_extra,
        }
    )

    config["runtime_env"].update(
        {
            "FORCE_SINGLE_HOST": "1",
            "LIBTPU_INIT_ARGS": " ".join(_build_libtpu_flags(attn_dp_size)),
            "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": str(dense_thresh),
            "SGLANG_TPU_ENABLE_ROPE_CACHE_SLICE": "1",
            "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "",
            "SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS": "2",
            "SGLANG_TPU_DECODE_SKIP_GUARD_EVAL": "1",
            "SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER": "1",
            "SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL": "1",
            "SGLANG_TPU_ALLOC_DECODE_FAST_PATH": "1",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
            "SGLANG_TPU_ASYNC_COPY_WORKER": "0",
            "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER": "1",
            "SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE": "1",
            **runtime_env_extra,
        }
    )

    return config


CONFIG = get_config(DEFAULT_CONCURRENCY)
