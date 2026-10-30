"""Muse Spark AgentX on tpu7x 2x2x1 hosts (4 chips), one engine per host and client concurrency.

Every concurrency starts from _BASE (decode megakernel, 8-token DFlash, HiCache) and
applies its _Patch: KV layout (bf16 at c4/c8, fp8 1M-context elsewhere), running
requests, prefill/decode interval and megakernel row cap. The mode picks the engine:

- mixed: one TP8 server, prefill and decode together.
- prefill / decode: one Raiden P/D engine each, a PP2 x TP4 prefill host feeding a TP8
  megakernel decode host, both on the experimental spec worker.

The fp8 pool assumes a windowed draft: a DFlash2 config declaring is_hybrid_swa and
an all-SWA hybrid_layer_pattern keeps the draft's KV in the target's SWA slots;
main() writes it next to the checkpoint.

    bash benchmark/tpu/recipe/muse_spark_agentx_launcher.sh <mixed|prefill|decode> 16
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentic.py --concurrency 16   # mixed only

A P/D pair still needs its router, e.g. on the prefill host:

    python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \\
        --prefill http://<prefill>:30000 8998 --decode http://<decode>:30000 \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

# 7 proposals + the anchor; SIMULATED_ACC_LEN is InferenceX's reference for 8-token verify.
DRAFT_TOKENS = 8
SIMULATED_ACC_LEN = 4.05

# Local weights main() serves; MODEL_ROOT holds both checkpoints.
MODEL_ROOT = Path("/models")
MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFT = "RadixArk/Muse-Spark-Open-DFlash2"
PORT = 8000
MODES = ("mixed", "prefill", "decode")

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


# Both P/D engines: the experimental spec worker is the only one serving DFlash on a PP prefill.
_PD: dict[str, dict] = {
    "server_args": {
        "disaggregation_transfer_backend": "raiden",
        "host": "0.0.0.0",
        "watchdog_timeout": 3600,
        "dist_timeout": 3600,
    },
    "runtime_env": {
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        # Pipeline transfers keep full mirrors, which outgrow the default at this context.
        "SGLANG_RAIDEN_HOST_POOL_GB": "128",
        # The experimental worker only simulates acceptance with fixed draft tokens.
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed",
        "SGLANG_WARMUP_TIMEOUT": "3600",
        # These JobSet variables describe the public router, not its engines.
        "HOST": None,
        "PORT": None,
        # Each engine is rank 0 of its own one-host group, whatever TPU_WORKER_ID its pod has.
        "SGLANG_LAUNCHER_RANK": "0",
    },
}

_ROLES: dict[str, dict[str, dict]] = {
    "prefill": {
        "server_args": {
            "disaggregation_mode": "prefill",
            # Raiden's pipeline producer needs attention TP=1 (DP attention), while NVFP4
            # MoE needs fused-RS expert TP > 1; PP rejects the overlap schedule.
            "tp_size": 4,
            "dp_size": 4,
            "enable_dp_attention": True,
            # The DFlash draft shares the target head, sharded over attention TP under DP.
            "enable_dp_lm_head": True,
            "ep_size": 1,
            "pp_size": 2,
            "disable_overlap_schedule": True,
        },
        # The megakernel runs the whole decoder in one kernel, so it cannot be staged.
        "runtime_env": {"SGLANG_TPU_MUSE_MEGAKERNEL": "0"},
    },
    "decode": {
        "server_args": {
            "disaggregation_mode": "decode",
            # The decode HiCache (pd_decode_swa_hicache) rejects host_pool retraction.
            "disaggregation_decode_retraction_backup": "cpu_tensor",
        },
        # A non-PP decode names its layer pools to pull from the PP producer.
        "runtime_env": {"SGLANG_RAIDEN_PIPELINE_TRANSFER": "1"},
    },
}


def get_config(concurrency: int, mode: str = "mixed") -> dict:
    if concurrency not in _PATCHES_BY_CONCURRENCY:
        raise ValueError(
            f"no patch for concurrency {concurrency}; defined: {sorted(_PATCHES_BY_CONCURRENCY)}"
        )
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    config = _apply(_PATCHES_BY_CONCURRENCY[concurrency])
    if mode != "mixed":
        for patch in (_PD, _ROLES[mode]):
            for section, overrides in patch.items():
                config[section].update(deepcopy(overrides))
        # One engine on this host; launch_server starts no router for it.
        config["group"] = {"hosts": 1}
    return config


def _write_swa_draft(model_root: Path) -> Path:
    """The windowed draft: DFLASH weights linked under a config declaring every layer SWA."""
    source, target = model_root / DRAFT, model_root / f"{DRAFT}-swa"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "config.json":
            continue
        link = target / path.name
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(path)
    draft_config = json.loads((source / "config.json").read_text())
    draft_config["is_hybrid_swa"] = True
    draft_config["hybrid_layer_pattern"] = [1] * draft_config["num_hidden_layers"]
    (target / "config.json").write_text(json.dumps(draft_config, indent=2))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve one Muse Spark AgentX engine.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true", help="print the config only")
    args = parser.parse_args(argv)
    config = get_config(args.concurrency, mode=args.mode)
    config["server_args"].update(
        model_path=str(MODEL_ROOT / MODEL),
        served_model_name=MODEL,
        speculative_draft_model_path=str(MODEL_ROOT / f"{DRAFT}-swa"),
    )
    if args.mode == "mixed":
        config["server_args"].update(host="0.0.0.0", port=PORT)
    print(json.dumps(config, indent=1))
    if args.dry_run:
        return 0
    _write_swa_draft(MODEL_ROOT)
    path = Path(f"/tmp/muse_agentx_{args.mode}_c{args.concurrency}.json")
    path.write_text(json.dumps(config))
    from benchmark.tpu.launch_server import main as launch

    return launch(["--config", f"@{path}"])


if __name__ == "__main__":
    raise SystemExit(main())
