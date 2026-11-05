"""Muse Spark AgentX on one tpu7x host (tp8, 4 chips) at the model's native 1M context.

The per-concurrency server configs of the AgentX sweep at concurrency 1, 7, 9, 10,
16 and 22, as one profile table on top of agentx_base.py. agentx_c<N>.py is the config
of one swept point; README.md shows how to boot a point and replay the AgentX
traces against it.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentx_c22.py

Every point runs fp8 KV in the SEQ_ALONG_LANE layout (batched RPA prefill,
megakernel decode, 256-token pages) with host KV offload through HiCache over
raiden, FCFS admission, and DFlash verifying 8 tokens, narrowing to 4 once
enough requests decode together. The draft is served through a windowed copy of
its config (windowed_draft.py), so its KV lives in the target's sliding-window
slots. Acceptance is pinned to the AgentX golden acceptance length.

Print any profile, swept or not, as a launcher JSON config:

    python3 -m benchmark.tpu.recipe.configs.muse_spark.agentx --conc 22
"""

import argparse
import json
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.agentx_base import CONFIG as AGENTIC_CONFIG
from benchmark.tpu.recipe.configs.muse_spark.windowed_draft import default_dir

SWEPT = (1, 7, 9, 10, 16, 22)

# conc: (running requests, verify tokens, decode steps between prefill batches,
#        adaptive verify "MIN_BS:WIDTH" or None). MIN_BS:WIDTH verifies WIDTH
# tokens once MIN_BS or more requests decode.
PROFILES = {
    # A session's subagent turns run beside it instead of queueing.
    1: (2, 8, 128, "4:4"),
    # Running requests step up with concurrency, so interactivity falls and
    # throughput rises from conc 1 to conc 10.
    7: (3, 8, 128, "4:4"),
    9: (4, 8, 128, "4:4"),
    10: (6, 8, 64, "4:4"),
    16: (16, 8, 64, "3:4"),
    22: (16, 8, 64, "3:4"),
    # Not swept; kept for reruns.
    2: (2, 8, 128, "4:4"),
    4: (2, 8, 128, "4:4"),
    3: (1, 8, 128, None),
    6: (6, 8, 32, "3:4"),
    8: (8, 4, 32, None),
    20: (8, 4, 128, None),
    24: (16, 4, 64, None),
    28: (16, 8, 64, "3:4"),
    32: (16, 8, 40, "3:4"),
    40: (16, 8, 64, "3:4"),
}
# Any other concurrency.
DEFAULT_PROFILE = (16, 8, 64, "3:4")

# The AgentX golden acceptance length of Muse Spark's DFlash2 draft (thinking on),
# by draft tokens per verify (the verify width minus the current token).
GOLDEN_AL = {
    1: "1.85",
    2: "2.54",
    3: "3.06",
    4: "3.44",
    5: "3.70",
    6: "3.93",
    7: "4.05",
}


# --mk-decode: only the decode megakernel's Waves 1-4 and expert grouping (PR #1909). No KPERM:
# it rewrites the fp4 experts for a dequant path the recipe's SGLANG_TPU_FP4_DEQUANT_VIA_FP8=1 skips.
MK_DECODE_ENV = {
    "SGLANG_TPU_MUSE_MK_EXPERT_GROUP": "15",
    "SGLANG_TPU_MUSE_MK_OPT": "1",
}

# --mk-prefill: extend steps run the megakernel's prefill program (PR #1909), with
# the decode Waves 1-4, expert grouping and the fused-RS/RPA settings it was measured on.
MK_PREFILL_ENV = {
    "SGLANG_TPU_MUSE_MK_EXPERT_GROUP": "15",
    "SGLANG_TPU_MUSE_MK_OPT": "1",
    "SGLANG_TPU_MUSE_MK_KPERM": "1",
    "SGLANG_TPU_MK_SAL_FP8_DOT": "2",
    "SGLANG_TPU_MUSE_MK_PREFILL": "1",
    "SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER": "1",
    "MUSE_PF_STAGE_GAP": "1",
    "MUSE_PF_2D_RS": "1",
    "MUSE_PF_2D_AG": "1",
    "MUSE_FUSED_RS_FAST_ISSUE": "1",
    "MUSE_FUSED_RS_FWD_PSUM": "1",
    "MUSE_FUSED_RS_DIRECT_STAGE": "2",
    "MUSE_PF_FUSED_SCATTER_COMBINE": "1",
    "MUSE_PF_MOE_KPERM": "1",
    "SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG": "1",
    "SGLANG_TPU_FP4_DEQUANT_VIA_FP8": "0",
    "SGLANG_TPU_RPA_NO_MAX": "1",
    "SGLANG_TPU_RPA_TRANSPOSED": "1",
}


def compile_bs(running, verify, adaptive):
    """Batch buckets: 1, the request cap, and every count whose verify step fills
    the megakernel's 8-row tiles exactly (above 8 requests only multiples of 4);
    other counts pad up to the next bucket."""
    if running == 16 and adaptive:
        # 9-11 decoding requests get their own programs instead of padding to 12.
        return [*range(1, 13), 16]
    buckets = [1] + [
        b for b in range(2, running) if b * verify % 8 == 0 and (b <= 8 or b % 4 == 0)
    ]
    return buckets + [running] if running > 1 else buckets


def megakernel_rows(running, verify, adaptive):
    """The largest verify step (requests x width, padded to 8 rows)."""
    rows = running * verify
    if adaptive:
        # Full-width steps stop below MIN_BS requests; narrow ones run up to the cap.
        min_bs, width = map(int, adaptive.split(":"))
        rows = max((min_bs - 1) * verify, running * width)
    return -(-rows // 8) * 8


def config_for(
    conc,
    *,
    draft_model_path=None,
    simulate_acceptance=True,
    mk_prefill=False,
    mk_decode=False,
):
    """The server config for one AgentX concurrency.

    simulate_acceptance=False verifies for real (evals); throughput runs commit
    drafts at the golden acceptance length regardless of the target logits.
    """
    running, verify, interval, adaptive = PROFILES.get(conc, DEFAULT_PROFILE)
    # From conc 16: one 4096-token prefill graph beside a 56 GB write-through
    # host pool per rank, 30% of it for sliding-window KV, and sessions.
    large = conc >= 16
    config = deepcopy(AGENTIC_CONFIG)
    args, env = config["server_args"], config["runtime_env"]
    # The pool follows the HBM budget; the cc=1 recipe pins a smaller one.
    args.pop("max_total_tokens")
    args.update(
        {
            "served_model_name": args["model_path"],
            # The 1M AgentX corpus replays prompts up to ~990K tokens: keep the
            # model's native context.
            "context_length": None,
            "max_running_requests": running,
            "torch_compile_max_bs": running,
            "tpu_compile_bs": compile_bs(running, verify, adaptive),
            # Each prefill graph at 1M context holds its own page-table-sized
            # metadata in host RAM: three fit beside the ratio-2.0 host pool, so
            # a typical turn runs a smaller graph than a padded 4096 one.
            "tpu_compile_tokens": [4096] if large else [1024, 2048, 4096],
            "speculative_draft_model_path": draft_model_path or default_dir(),
            "speculative_num_draft_tokens": verify,
            "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
            "schedule_policy": "fcfs",
            "prefill_decode_interval": interval,
            "page_size": 256,
            "kv_cache_dtype": "fp8_e4m3",
            "attention_backend": "pallas_tpu_rpa_batched",
            # The windowed draft frees its full-context KV for the target's pool.
            "mem_fraction_static": 0.95,
            "swa_full_tokens_ratio": 0.0625,
            "enable_hierarchical_cache": True,
            "hicache_io_backend": "raiden",
            "hicache_mem_layout": "page_first_fused",
            # Host pool = ratio x device pool per rank; an explicit size (GB per
            # rank) overrides the ratio.
            "hicache_ratio": 3.0 if large else 2.0,
            "hicache_size": 56 if large else None,
            # Write-through copies KV to the host once cached, so a device
            # eviction never waits for a write-back.
            "hicache_write_policy": "write_through" if large else "write_back",
        }
    )
    if large:
        # The client's X-Correlation-ID (one per conversation and per subagent)
        # keys one radix session per conversation. A session covers the prompt
        # end of its latest turn; covered KV leaves the host pools last, while
        # device eviction stays LRU.
        args.update(
            {
                "enable_session_radix_cache": True,
                "session_id_header": "X-Correlation-ID",
                "session_leaf_at_prompt_end": True,
                "session_coverage_host_only": True,
            }
        )
    env.update(
        {
            # Reserve each request's full max_new_tokens at admission: long
            # requests queue instead of retracting.
            "SGLANG_INIT_NEW_TOKEN_RATIO": "1.0",
            "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR": "1.0",
            "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1",
            # Radix-tree checkpoints of a long prefill: every 16K tokens from
            # conc 16 keeps a sliding window close below where a resumed turn
            # leaves the cached prompt.
            "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "16384"
            if large
            else "131072",
            # Megakernel attention over the fp8 SEQ_ALONG_LANE pool (all exact):
            # 512-token chunks on sliding-window layers, idle softmax streams
            # skipped, pipelined KV writes, the scale folded into q, fp8 MXU dots,
            # and the pipelined attention loop on 33-64-row steps too.
            "SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS": "512",
            "SGLANG_TPU_MUSE_MK_ATTN_SKIP_IDLE": "1",
            "SGLANG_TPU_MUSE_MK_KV_WRITE_PIPELINED": "1",
            "SGLANG_TPU_MK_FOLD_ATTN_SCALE": "1",
            "SGLANG_TPU_MK_SAL_FP8_DOT": "2",
            "SGLANG_TPU_MUSE_MK_LARGE_STEP_ATTN_PIPE": "1",
            # Batched-RPA prefill: partial query blocks compute only their real
            # 256-query groups, no running max (q and k are RMS-normalized),
            # 512 x 512 sliding-window tiles, and the kernel stores each head's
            # output RMS-normalized (the eps must equal the model's rms_norm_eps).
            "SGLANG_TPU_RPA_SKIP_PAD_Q": "1",
            "SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP": "256",
            "SGLANG_TPU_RPA_NO_MAX": "1",
            "SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS": "512:512",
            "SGLANG_TPU_RPA_OUT_HEAD_NORM_EPS": "1e-05",
            # Prefill MoE (expert-TP fused-RS, exact): local row blocks, tiled
            # reduce, no zeroing of a fully written output, 8 rows per DMA loop
            # iteration, one-pass combine, 10 staging slots, on-chip pair exchange
            # with its run table and window/run adds, fp4 dequant through fp8,
            # weight ping-pong, and the second matmul storing rows directly.
            "SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS": "1",
            "SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE": "1",
            "SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO": "1",
            "SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL": "8",
            "SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE": "1",
            "SGLANG_TPU_FUSED_RS_STAGING_SLOTS": "10",
            "SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE": "1",
            "SGLANG_TPU_FUSED_RS_PAIR_RUN_TABLE": "1",
            "SGLANG_TPU_FUSED_RS_PAIR_WINDOW_ADD": "1",
            "SGLANG_TPU_FUSED_RS_PAIR_RUN_ADD": "1",
            "SGLANG_TPU_FUSED_RS_GMM2_ROW_OUT": "1",
            "SGLANG_TPU_FUSED_RS_PAIR_LOCAL_RANKS": "1",
            "SGLANG_TPU_FP4_DEQUANT_VIA_FP8": "1",
            "SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG": "1",
            # Prefill attention and layers: full key tiles skip the mask, fp8
            # tiles computed transposed, the residual stream token-sharded over
            # the TP group, expert column shards gathered along the column dim.
            "SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK": "1",
            "SGLANG_TPU_RPA_TRANSPOSED": "1",
            "SGLANG_TPU_MUSE_PREFILL_SP": "1",
            "SGLANG_TPU_MUSE_GATHER_COLUMNS_DIM": "1",
            # Decode: each row's expert outputs go to their reduce-scatter owner
            # as soon as the row is done, one DMA per owner block.
            "SGLANG_TPU_MK_Y_EARLY": "1",
            "SGLANG_TPU_MK_Y_BLOCK_SEND": "1",
            # Scheduling: a lone chunked prefill skips the interval while nothing
            # decodes; waiting turns that fit share a long prompt's prefill batches.
            "SGLANG_PREFILL_DECODE_INTERVAL_SKIP_IDLE": "1",
            "SGLANG_PREFILL_CHUNK_SHARE_MIN_TOKENS": "1024",
            # Host KV: raiden moves runs of consecutive pages as one transfer, and
            # the host heap is released before the pools are built.
            "SGLANG_TPU_RAIDEN_COALESCE_PAGES": "1",
            "SGLANG_TPU_HICACHE_TRIM_BEFORE_ALLOC": "1",
            "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
            # Outlast AIPerf's pooled keep-alive connections (300 s).
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
        }
    )
    if large:
        env.update(
            {
                "SGLANG_HICACHE_SWA_HOST_FRACTION": "0.3",
                # Without sessions, the sliding-window host pool evicts first the
                # windows a later turn's match walked past (off when sessions are on).
                "SGLANG_HICACHE_SWA_HOST_EVICT_PASSED_FIRST": "1",
            }
        )
    if running >= 16:
        # Steps above 24 rows run experts with few live rows in their own loop.
        env["SGLANG_TPU_MUSE_MK_EXPERT_SPLIT"] = "4"
        if conc > 16:
            # While fewer than 6 requests decode and prefill waits, the interval
            # ends after 8 decode steps, so admission fills the decode batch.
            env["SGLANG_PREFILL_DECODE_INTERVAL_ADAPTIVE"] = "6:8"
    if adaptive:
        # Narrow steps carry the golden AL of their own width.
        width = int(adaptive.split(":")[1])
        env["SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY"] = f"{adaptive}:{GOLDEN_AL[width - 1]}"
    rows = megakernel_rows(running, verify, adaptive)
    if rows > 32:
        # The megakernel takes steps of up to 32 rows by default.
        env["SGLANG_TPU_MUSE_MK_MAX_ROWS"] = str(rows)
    if simulate_acceptance:
        env.update(
            {
                "SGLANG_SIMULATE_ACC_LEN": GOLDEN_AL[verify - 1],
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
            }
        )
    if mk_decode:
        env.update(MK_DECODE_ENV)
    if mk_prefill:
        env.update(MK_PREFILL_ENV)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conc", type=int, required=True)
    parser.add_argument(
        "--draft-model-path", help="default: windowed_draft.py's directory"
    )
    parser.add_argument(
        "--eval", action="store_true", help="verify for real (no simulated acceptance)"
    )
    parser.add_argument(
        "--mk-prefill",
        action="store_true",
        help="megakernel prefill program and its settings (MK_PREFILL_ENV)",
    )
    parser.add_argument(
        "--mk-decode",
        action="store_true",
        help="decode megakernel Waves 1-4 and expert grouping only (MK_DECODE_ENV)",
    )
    args = parser.parse_args(argv)
    config = config_for(
        args.conc,
        draft_model_path=args.draft_model_path,
        simulate_acceptance=not args.eval,
        mk_prefill=args.mk_prefill,
        mk_decode=args.mk_decode,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
