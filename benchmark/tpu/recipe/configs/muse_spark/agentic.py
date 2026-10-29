"""Muse Spark AgentX on one tpu7x 2x2x1 host (4 chips, TP8), one server per client concurrency.

Every concurrency starts from _BASE (decode megakernel, 8-token DFlash, HiCache) and
applies its _Patch: KV layout (bf16 at c4/c8, fp8 1M-context elsewhere), running
requests, prefill/decode interval and megakernel row cap.

The fp8 pool assumes a windowed draft: a DFlash2 config declaring is_hybrid_swa and
an all-SWA hybrid_layer_pattern keeps the draft's KV in the target's SWA slots.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentic.py --concurrency 16
"""

from __future__ import annotations

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

# 7 proposals + the anchor; SIMULATED_ACC_LEN is InferenceX's reference for 8-token verify.
DRAFT_TOKENS = 8
SIMULATED_ACC_LEN = 4.05

_BASE = deepcopy(SERVER_CONFIG)
_BASE["server_args"].update(
    {
        "context_length": 1048576,
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "RadixArk/Muse-Spark-Open-DFlash2",
        "speculative_num_draft_tokens": DRAFT_TOKENS,
        "scheduler_recv_interval": 8,
        "enable_cache_report": True,
        "enable_metrics": True,
        # HiCache: host pool = 2x device pool, written back on eviction over raiden.
        "enable_hierarchical_cache": True,
        "hicache_ratio": 2.0,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "hicache_write_policy": "write_back",
    }
)
_BASE["runtime_env"].update(
    {
        # Acceptance is simulated so the curve compares speed, not draft quality.
        "SGLANG_SIMULATE_ACC_LEN": str(SIMULATED_ACC_LEN),
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        # Core DFlash worker: the experimental one only simulates the fixed token mode.
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "0",
        "SGLANG_SPEC_TP_SYNC": "rng",
        "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
        "SGLANG_TPU_HOST_PADDING_NO_D2H": "0",
        # Packed KV heads halve the per-token KV bytes.
        "SGLANG_TPU_KV_PACK_HEADS": "1",
        # Decode megakernel (whole decoder in one Pallas kernel) and its attention tuning.
        "SGLANG_TPU_MUSE_MEGAKERNEL": "1",
        "SGLANG_TPU_MUSE_MK_ATTN_FIXED_MAX": "1",
        "SGLANG_TPU_MUSE_MK_ATTN_SKIP_IDLE": "1",
        "SGLANG_TPU_MUSE_MK_ATTN_COL_MASK": "1",
        "SGLANG_TPU_MK_FOLD_ATTN_SCALE": "1",
        # Megakernel MoE: overlap the reduce-scatter sends with the remaining experts.
        "SGLANG_TPU_MK_Y_EARLY": "1",
        "SGLANG_TPU_MK_Y_BLOCK_SEND": "1",
        # Fused reduce-scatter expert TP (prefill MoE).
        "SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS": "1",
        "SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE": "1",
        "SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO": "1",
        "SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE": "1",
        "SGLANG_TPU_FUSED_RS_STAGING_SLOTS": "8",
        # DFlash2 draft: TP-sharded context projection, matmul conv, fp8 weight-only linears.
        "SGLANG_DFLASH_ROW_PARALLEL_FC": "1",
        "SGLANG_DFLASH_CONV_MATMUL_EXPAND": "1",
        "SGLANG_TPU_DFLASH_FP8_DRAFT": "1",
        # HiCache transfers: one segment per run of contiguous pages.
        "SGLANG_TPU_RAIDEN_COALESCE_PAGES": "1",
        # Scheduling: short prefills ride along a long one; no idle deferral when nothing decodes.
        "SGLANG_PREFILL_CHUNK_SHARE_MIN_TOKENS": "1024",
        "SGLANG_PREFILL_DECODE_INTERVAL_SKIP_IDLE": "1",
        # Checkpoint a long prefill into the radix tree every 128K tokens, not every chunk.
        "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
        # Reserve each request's full max_new_tokens at admission: queue, never retract.
        "SGLANG_INIT_NEW_TOKEN_RATIO": "1.0",
        "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR": "1.0",
    }
)


# bf16 token-major KV under the fixed token cap the agentic runs were tuned with.
_BF16_KV: dict[str, dict] = {
    "server_args": {"kv_cache_dtype": "bfloat16", "max_total_tokens": 917504},
    "runtime_env": {},
}

# fp8 sequence-along-lane KV (256 B/token), pool sized by HBM, batched RPA prefill.
_FP8_1M_KV: dict[str, dict] = {
    "server_args": {
        "kv_cache_dtype": "fp8_e4m3",
        "page_size": 256,
        "attention_backend": "pallas_tpu_rpa_batched",
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
        "schedule_policy": "fcfs",
        "tpu_compile_tokens": [1024, 2048, 4096],
        # The windowed draft frees its full-context KV for the target's pool.
        "mem_fraction_static": 0.95,
        "swa_full_tokens_ratio": 0.0625,
    },
    "runtime_env": {
        "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1",
        # Megakernel attention over SEQ_ALONG_LANE pools.
        "SGLANG_TPU_MUSE_MK_SAL_CHUNK_TOKENS": "2048",
        "SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS": "512",
        "SGLANG_TPU_MK_SAL_FP8_DOT": "1",
        # Batched RPA prefill attention (SEQ_ALONG_LANE only).
        "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1",
        "SGLANG_TPU_RPA_EXP2_SOFTMAX": "1",
        "SGLANG_TPU_RPA_ONES_V": "1",
        "SGLANG_TPU_RPA_TRANSPOSED": "1",
        "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    },
}


def _patch(
    max_running_requests: int,
    kv: dict[str, dict] = _FP8_1M_KV,
    # Decode rounds between prefill batches: protects ITL, costs TTFT.
    prefill_decode_interval: int = 128,
    # A step is max_running_requests x DRAFT_TOKENS rows; past 32 the megakernel needs
    # its large-step program, and larger steps run the slow native decoder.
    mk_max_rows: int = 32,
) -> dict[str, dict]:
    rows = max_running_requests * DRAFT_TOKENS
    if rows > mk_max_rows:
        raise ValueError(f"{rows} rows per step > mk_max_rows {mk_max_rows}")
    return {
        "server_args": {
            **kv["server_args"],
            "max_running_requests": max_running_requests,
            "torch_compile_max_bs": max_running_requests,
            "tpu_compile_bs": list(range(1, max_running_requests + 1)),
            "prefill_decode_interval": prefill_decode_interval,
        },
        "runtime_env": {
            **kv["runtime_env"],
            "SGLANG_TPU_MUSE_MK_MAX_ROWS": str(mk_max_rows),
        },
    }


# Haolin's per-concurrency settings. Admission stays far below the concurrency because
# the megakernel caps a step at 48 rows (6 requests); the rest queue.
_PATCHES_BY_CONCURRENCY: dict[int, dict[str, dict]] = {
    1: _patch(2),
    4: _patch(1, kv=_BF16_KV),
    8: _patch(2, kv=_BF16_KV),
    16: _patch(2),
    24: _patch(6, mk_max_rows=48),
    28: _patch(6, mk_max_rows=48),
    32: _patch(6, prefill_decode_interval=16, mk_max_rows=48),
}


def _apply(patch: dict[str, dict]) -> dict:
    config = deepcopy(_BASE)
    for section, overrides in patch.items():
        config[section].update(deepcopy(overrides))
    return config


def build_config(max_running_requests: int, **patch_kwargs) -> dict:
    """_BASE with the fp8 1M-context KV unless patch_kwargs overrides _patch's defaults."""
    return _apply(_patch(max_running_requests, **patch_kwargs))


def get_config(concurrency: int) -> dict:
    if concurrency not in _PATCHES_BY_CONCURRENCY:
        raise ValueError(
            f"no patch for concurrency {concurrency}; defined: {sorted(_PATCHES_BY_CONCURRENCY)}"
        )
    return _apply(_PATCHES_BY_CONCURRENCY[concurrency])
