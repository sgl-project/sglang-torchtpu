"""Muse Spark AgentX on one tpu7x host (tp8, 4 chips) at the model's native 1M context.

The per-concurrency server configs of the AgentX sweep at concurrency 1, 7, 9, 10,
16 and 22, as one profile table on top of agentx_base.py. agentx_c<N>.py is the config
of one swept point; README.md shows how to boot a point and replay the AgentX
traces against it.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentx_c22.py

Every point runs fp8 KV in the SEQ_ALONG_LANE layout (batched RPA prefill,
megakernel decode, 256-token pages) with host KV offload through HiCache over
raiden, FCFS admission, and DFlash on the experimental spec worker verifying a
fixed 8 tokens up to conc 10 and 4 from conc 16. The draft is served through a windowed copy of
its config (windowed_draft.py), so its KV lives in the target's sliding-window
slots. Acceptance is pinned to the AgentX golden acceptance length.

Print any profile, swept or not, as a launcher JSON config:

    python3 -m benchmark.tpu.recipe.configs.muse_spark.agentx --conc 22

MOE_MODE=ep: the same profile on the EP megakernel (sglang_tpu/muse_megakernel_v2, PR #2231):
EP8 experts, extend on prefill/spep, decode and verify on decode/spep, 8-token verify.
EP_ARM picks gather / a2a / mixed (default mixed).

    MOE_MODE=ep python3 -m benchmark.tpu.recipe.configs.muse_spark.agentx --conc 22 --mk-prefill
"""

import argparse
import json
import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.agentx_base import CONFIG as AGENTIC_CONFIG
from benchmark.tpu.recipe.configs.muse_spark.windowed_draft import default_dir

SWEPT = (1, 8, 9, 10, 16, 22)

# conc: (running requests, verify tokens, decode steps between prefill batches,
#        adaptive verify "MIN_BS:WIDTH" or None). MIN_BS:WIDTH verifies WIDTH
# tokens once MIN_BS or more requests decode.
PROFILES = {
    # The swept points run the experimental worker at a fixed width. Up to conc 10
    # 1-3 requests decode most steps, where an 8-token block pads to the same 8- or
    # 16-row step as 4 tokens but accepts 4.05 instead of 3.06 (j-2e0cc62e: q4 at
    # c7 -23% per-request decode at 1 running); from conc 16 q4 wins.
    # A session's subagent turns run beside it instead of queueing.
    1: (2, 8, 128, None),
    # Running requests step up with concurrency, so interactivity falls and
    # throughput rises from conc 1 to conc 10.
    # c4: j-db36468a's 1 running request, verifying 4 tokens (golden AL 3.06).
    4: (1, 4, 128, None),
    7: (3, 8, 128, None),
    # c8-c10 verify 4 tokens (golden AL 3.06); c8 keeps j-03bad578's 2 running requests.
    8: (2, 4, 128, None),
    9: (4, 4, 128, None),
    # c10 at 8 running: +2.3% tok/s/chip over 6 (j-83d7512f vs j-fe083895); decode rarely exceeds 6.
    10: (8, 4, 64, None),
    16: (16, 4, 64, None),
    22: (16, 4, 64, None),
    # Not swept; kept for reruns.
    # c22 with ~20% more running requests and clients (76 verify rows pad to an 80-row step).
    26: (19, 4, 64, None),
    # c32 at 20 running requests (80 verify rows, the same step as c26).
    32: (20, 4, 64, None),
    # Not swept; kept for reruns.
    2: (2, 8, 128, "4:4"),
    3: (1, 8, 128, None),
    6: (6, 8, 32, "3:4"),
    20: (8, 4, 128, None),
    24: (16, 4, 64, None),
    28: (16, 8, 64, "3:4"),
    40: (16, 8, 64, "3:4"),
}
# Any other concurrency.
DEFAULT_PROFILE = (16, 8, 64, "3:4")

# conc: host pool GB per rank (None: hicache_ratio), for points served by the
# experimental spec worker (fixed simulated draft tokens, no adaptive width).
EXP_WORKER_HICACHE_GB = {
    1: None,
    4: None,
    7: None,
    8: None,
    9: None,
    10: None,
    16: 72,
    22: 72,
    26: 72,
    32: 72,
}

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
# No KPERM: main's decode megakernel serves only unpermuted experts.
MK_PREFILL_ENV = {
    "SGLANG_TPU_MUSE_MK_EXPERT_GROUP": "15",
    "SGLANG_TPU_MUSE_MK_OPT": "1",
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
    "SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG": "1",
    "SGLANG_TPU_FP4_DEQUANT_VIA_FP8": "0",
    "SGLANG_TPU_RPA_NO_MAX": "1",
    "SGLANG_TPU_RPA_TRANSPOSED": "1",
}


# MOE_MODE=ep: the env PR #2231's _validate_ep_serving requires; KPERM and the e4m3 prefill
# gathers are TP-layout knobs the EP programs refuse. EP keeps the profile's stacked expert
# groups (SGLANG_TPU_MUSE_MK_EXPERT_GROUP=15).
EP_ENV = {
    "SGLANG_TPU_MUSE_MOE_MODE": "ep",
    "SGLANG_TPU_MUSE_MK_QBLOCK": "256",
    "SGLANG_TPU_MUSE_MK_PREFILL": "1",
    "SGLANG_TPU_MUSE_MK_PREFILL_SP": "1",
    "SGLANG_TPU_MUSE_MK_SP_EP_LAYOUT": "1",
}
EP_DROPPED_ENV = ("SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER", "SGLANG_TPU_MUSE_MK_KPERM")
EP_VERIFY = 8


def apply_ep(config, *, running, adaptive, simulate_acceptance, ep_arm):
    """The profile on the EP megakernel: only the EP layout, an 8-token verify and its rows."""
    if adaptive:
        raise ValueError("MOE_MODE=ep: no adaptive verify width")
    args, env = config["server_args"], config["runtime_env"]
    args["ep_size"] = 8
    # The replicated wo (64 MiB/layer/rank) eats the 0.95 headroom: the logits program
    # failed to load 60 MB short at extend precompile.
    args["mem_fraction_static"] = 0.93
    args["speculative_num_draft_tokens"] = EP_VERIFY
    for name in EP_DROPPED_ENV:
        env.pop(name, None)
    env.update(EP_ENV)
    env["SGLANG_TPU_MUSE_MK_EP_ARM"] = ep_arm
    env["SGLANG_TPU_MUSE_MK_MAX_ROWS"] = str(-(-running * EP_VERIFY // 16) * 16)
    if simulate_acceptance:
        env["SGLANG_SIMULATE_ACC_LEN"] = GOLDEN_AL[EP_VERIFY - 1]
    return config


def compile_bs(running, verify, adaptive):
    """Batch buckets: 1, the request cap, and every count whose verify step fills
    the megakernel's 8-row tiles exactly (above 8 requests only multiples of 4);
    other counts pad up to the next bucket."""
    if running == 20 and verify == 4:
        # c32: a program for every request count, so no step pads a request.
        return list(range(1, running + 1))
    if running in (16, 19) and (adaptive or verify == 4):
        # 9-11 decoding requests get their own programs instead of padding to 12.
        return sorted({*range(1, 13), 16, running})
    if verify == 4 and running <= 12:
        # Every count, as the adaptive points compiled: odd counts would pad a request.
        return list(range(1, running + 1))
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
    mk_qblock=0,
    moe_mode="tp",
    ep_arm="mixed",
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
            "hicache_size": (EXP_WORKER_HICACHE_GB.get(conc) or 56) if large else None,
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
            # The per-step TP all-reduce of ready HiCache acks costs ~0.9 ms of host
            # time; while no rank has one, poll every 50th step.
            "SGLANG_HICACHE_QUIET_SYNC_EVERY": "50",
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
    exp_worker = conc in EXP_WORKER_HICACHE_GB
    if exp_worker:
        if adaptive:
            raise ValueError(
                "the experimental spec worker has no adaptive verify width"
            )
        env.update(
            {
                "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
                # Host launch path at low batch: Dynamo guards only the forward
                # inputs, and since the worker never compiles after warmup, the
                # frozen draft/verify forwards evaluate only the guards that pick
                # a precompiled graph (cc=1: 6.05 -> 5.60 ms per step).
                "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER": "1",
                "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "0",
                "SGLANG_TPU_DECODE_SKIP_GUARD_EVAL": "1",
            }
        )
    if simulate_acceptance:
        env.update(
            {
                "SGLANG_SIMULATE_ACC_LEN": GOLDEN_AL[verify - 1],
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                # The experimental worker only simulates with fixed draft tokens.
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed"
                if exp_worker
                else "real-draft-token",
            }
        )
    if mk_decode:
        env.update(MK_DECODE_ENV)
    if mk_prefill:
        env.update(MK_PREFILL_ENV)
    if mk_qblock:
        # Block-256 requantized fp4 experts (main #2131) and the key-column mask.
        env["SGLANG_TPU_MUSE_MK_QBLOCK"] = str(mk_qblock)
        env["SGLANG_TPU_MUSE_MK_ATTN_COL_MASK"] = "1"
    if moe_mode == "ep":
        apply_ep(
            config,
            running=running,
            adaptive=adaptive,
            simulate_acceptance=simulate_acceptance,
            ep_arm=ep_arm,
        )
    elif moe_mode != "tp":
        raise ValueError(f"MOE_MODE={moe_mode!r}: tp or ep")
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
    parser.add_argument(
        "--mk-qblock",
        type=int,
        default=0,
        help="fp4 experts requantized to one scale per this many K rows (0 off, 256)",
    )
    args = parser.parse_args(argv)
    config = config_for(
        args.conc,
        draft_model_path=args.draft_model_path,
        simulate_acceptance=not args.eval,
        mk_prefill=args.mk_prefill,
        mk_decode=args.mk_decode,
        mk_qblock=args.mk_qblock,
        moe_mode=os.environ.get("MOE_MODE", "tp"),
        ep_arm=os.environ.get("EP_ARM", "mixed"),
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
