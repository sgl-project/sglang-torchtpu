#!/usr/bin/env bash
# Single source of truth for the Qwen3-Coder-480B FP8 TPU agentic benchmark.
# Sourced by serve.sh / bench_leg.sh / run_nightly.sh, so every leg of every run
# uses identical settings. Anything the team has agreed on is FIXED below and
# guarded; everything else is a documented default that can be overridden from
# the environment.
#
#   source config.sh          # then use the resolved vars
#   PROFILE=smoke source config.sh
#
# The workload is our own synthetic agentic trace (gen_agentic_workload.py): 4k
# new input + 1k output per round, accumulating to 16k, with a configurable
# number of sessions accumulating to 256k instead.
#
# How that compares to a real public trace, for calibration when reading a
# result. Measured over the Mooncake FAST'25 `toolagent` corpus (23,608
# requests): total prompt p50 6.3k / p90 16.8k / max 126k; NEW tokens per round
# p50 689 / p90 9,990; output per round p50 30 / mean 182; prefix-hit p50 82%.
# So our 4k/round sits between that corpus's p50 and p90 for new input, and our
# 1k output/round is ~5x its mean.
#
# ============================================================================
# FIXED — the benchmark contract. Changing one of these makes results
# incomparable with the team's other runs, so each is asserted, not defaulted.
# ============================================================================
#   model                Qwen3-Coder-480B-A35B-Instruct-FP8
#   quantization         fp8 weights + fp8_e4m3 KV
#   parallelism          colocated: tp=16, dp=4  -> attn_tp=4
#                        P/D:       tp=8,  dp=2  -> attn_tp=4  (per engine)
#   server context len   1048576 ALWAYS
#   workload             multi-turn agentic, 4k new input tokens + 1k output
#                        tokens per round, accumulating to the context target
#   context targets      16384 normal / 262144 for the "long" sessions
#   concurrency sweep    1, 16, 32
#   long-request sweep   0, 1, 2, 4 concurrent long-context sessions
#   chunked prefill      4096 tokens per DP rank (= one round's new input)
#   features             prefix caching ON, KV offload (HiCache) ON
#
# The parallelism, precision, features and server context length are asserted.
# The workload shape (the four length knobs + chunked prefill) has the contract
# values as *defaults* and can be overridden per run -- doing so is recorded in
# config.log and in every leg's meta, so it can never pass unnoticed.
#
# ============================================================================
# FLEXIBLE — chosen by heuristic, documented at each site. Tune these.
# ============================================================================
set -euo pipefail

NIGHTLY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TPU_BENCH_DIR=$(cd -- "$NIGHTLY_DIR/../../.." && pwd)   # benchmark/tpu
REPO_ROOT=$(cd -- "$TPU_BENCH_DIR/../.." && pwd)

# ---------------------------------------------------------------------------
# Profile. Only affects how far the workload is time-boxed; never the shape.
#   full    (default) exactly the contract above: the long sessions ramp at
#           TURN_INPUT_TOKENS, i.e. ~51 rounds each to reach 256k. At the 4k/1k
#           default this is far cheaper than it was at 8k/1M (~113 rounds), so
#           `full` is plausibly nightly-viable now -- time one leg before
#           committing the whole sweep to it.
#   nightly the long sessions ramp in 16k rounds (~15 rounds) to bound the ramp's
#           share of wall clock. The normal sessions stay at TURN_INPUT_TOKENS, so
#           the agentic granularity under test is unchanged. Use for a manual
#           overnight run on a slice you own.
#   ci      what the GKE nightly runs. Same shape as `nightly`, but the sweep and
#           the context targets are cut to fit the JobSet's budget: the TPU CI
#           template gives a live-server test file `timeout 3600` and the whole
#           JobSet activeDeadlineSeconds 9000, most of which the 480B weight load
#           + tp=16 compile consumes. See MAX_LEG_SECONDS below.
#   smoke   plumbing check: 2 concurrency legs, small targets. Minutes.
#
# NOTE: unrelated to bench_serving.sh's PROFILE=0|1 (torch profiler capture).
# Scripts here pass PROFILE=0 when they call bench_serving.sh, and record this
# value as `run_profile` in the leg metadata, so the two never collide.
# ---------------------------------------------------------------------------
PROFILE=${PROFILE:-full}

# ===== FIXED: model + precision =====
MODEL=${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8_e4m3

# ===== FIXED: server context length =====
# ALWAYS 1M, for every profile -- including smoke, which shrinks the workload but
# not the server, and independent of LONG_TARGET_CONTEXT. One server config for
# every leg, so no two results are separated by a server difference. Not
# overridable; that is the point.
SERVER_CONTEXT_LEN=1048576

# ===== CONTRACT DEFAULTS: workload shape =====
# These four are the team's agreed shape. They are overridable so a one-off can
# explore a different shape, but a run that changes one is NOT comparable with
# the team's other runs -- every value is echoed into config.log and recorded in
# each leg's *.meta.json, so a changed shape is always visible in the result.
#
# Public reference for the first two (Mooncake toolagent corpus): new
# tokens/round p50 689 / p90 9,990; output/round p50 30 / mean 182. 4k+1k sits
# between that corpus's p50 and p90 for new input, and ~5x its mean output.
#
# TARGET_CONTEXT=16384 is the same corpus's p90 total prompt (16.8k): a normal
# session is the heaviest ordinary agentic request, not an outlier. At 4k+1k that
# is 3 rounds ending on a ~14.4k prompt (15,390 tok of context, 6% under the
# target -- see _turns_for_context). Only the long sessions carry long-context
# work. Consequence to know when reading a result: with 3 rounds/session the best
# achievable prefix-cache hit rate is ~56%, not the ~94% a 25-round session
# gives, because round 1 is always a miss. The CI gate derives its floor from the
# round count for exactly this reason.
TURN_INPUT_TOKENS=${TURN_INPUT_TOKENS:-4096}       # new input tokens per round
TURN_OUTPUT_TOKENS=${TURN_OUTPUT_TOKENS:-1024}     # output tokens per round
TARGET_CONTEXT=${TARGET_CONTEXT:-16384}            # normal session ctx target
LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT:-262144} # long session ctx target

# ===== FIXED: parallelism per mode =====
COLOCATED_TP=16
COLOCATED_DP=4                  # -> attn_tp = 16/4 = 4
PD_TP=8
PD_DP=2                         # -> attn_tp = 8/2 = 4

# ===== CONTRACT DEFAULT: prefill chunking =====
# 4096 = TURN_INPUT_TOKENS, so one agentic round's new input is exactly one
# prefill chunk: a cache-hit round prefills in a single pass, and only a
# cache-miss round (or a session's first round) chunks.
# sglang divides --chunked-prefill-size by dp_size when DP attention is on
# (server_args.py: "The chunked prefill size is adjusted to ..."), so the value
# passed to the server must be pre-multiplied to land at this many per DP rank.
CHUNKED_PREFILL_PER_RANK=${CHUNKED_PREFILL_PER_RANK:-4096}

# ===== FIXED: features under test =====
PREFIX_CACHE=1                  # -> --disable-radix-cache is NOT passed
KV_OFFLOAD=1                    # -> --enable-hierarchical-cache (torch_tpu io)

# ===== FIXED: sweeps =====
# Concurrency (client-side max in-flight sessions).
SWEEP_BS=${SWEEP_BS:-"1 16 32"}
# How many of those sessions accumulate to LONG_TARGET_CONTEXT instead of
# TARGET_CONTEXT. Orthogonal to SWEEP_BS -- run_nightly.sh crosses the two and
# skips legs where long > bs.
SWEEP_LONG=${SWEEP_LONG:-"0 1 2 4"}

# ===== FLEXIBLE =====
# Expert parallelism. EP=TP puts one expert shard per chip, which is what the
# existing 480B TPU recipes use; 160 experts divides by both 8 and 16.
EP_SIZE_COLOCATED=${EP_SIZE_COLOCATED:-$COLOCATED_TP}
EP_SIZE_PD=${EP_SIZE_PD:-$PD_TP}

# Attention backend. pallas_tpu_rpa_v3 is not a free choice here: it is the only
# TPU backend with the HiCache reload-commit hook and the fused KV pool, so
# --hicache-io-backend torch_tpu rejects anything else (server_args.py
# _resolve_layout_io_compatibility). The faster pallas_tpu_rpa_batched kernel is
# therefore unavailable while KV offload is part of the contract -- revisit if
# the reload hook lands there.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}

# Page size. 128 is the rpa_v3 default in these scripts. Even a 1M-token request
# is only 8192 pages, so the page table stays small; larger pages waste KV on the
# short early rounds of a session.
PAGE_SIZE=${PAGE_SIZE:-128}

# HBM split. 480B FP8 weights are ~480 GB, so at tp=16 each rank holds ~30 GB of
# weights and the KV pool wants everything that is left; 0.85 leaves headroom
# for activations at an 8k chunk. Lower this first if the server OOMs at launch.
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

# Host KV pool = HICACHE_RATIO x device KV pool. With the 16k normal target the
# offload pressure comes from the long sessions and the replayed prefixes, not
# from the normal batch: a bs=32 x 4-long leg holds ~1M long-session tokens plus
# ~0.5M of normal context, and 3x the device pool covers that once the device
# pool overflows -- the case KV offload exists for. Costs host RAM: ratio x
# device pool bytes, so check the pod's memory limit before raising it.
HICACHE_RATIO=${HICACHE_RATIO:-3}

# Normal-session replay depth. During a leg with long sessions, the short
# sessions finish sooner than the long ones; without replays the leg spends
# most of its wall clock at (bs - long) idle slots and the aggregate numbers
# describe a batch that was never full. Replaying costs no extra wall clock (the
# long sessions set the leg's length) -- only trace size and generation time,
# hence the cap.
NORMAL_REPLAYS=${NORMAL_REPLAYS:-auto}
MAX_NORMAL_REPLAYS=${MAX_NORMAL_REPLAYS:-8}

# How the long sessions ramp to LONG_TARGET_CONTEXT. Contract value is
# TURN_INPUT_TOKENS; the nightly profile raises it purely to bound wall clock
# (see PROFILE above).
LONG_RAMP_TOKENS=${LONG_RAMP_TOKENS:-$TURN_INPUT_TOKENS}

# Where generated workload traces live. Deliberately OUTSIDE the timestamped run
# directory: the generator is deterministic given (shape, seed), so a trace is
# reusable forever and regenerating it every night is pure waste (~5 min per run
# for the default sweep). One trace per SWEEP_LONG value, sized for the largest
# concurrency, so the whole sweep needs 4 files (~90 MB) per (shape, seed) --
# a second PROFILE with a different long ramp adds another set.
# Point this at a scratch path to keep the cache out of $HOME.
TRACE_CACHE_DIR=${TRACE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sglang_tpu_agentic_traces}

# Per-leg wall-clock deadline. A hung leg must not consume the whole nightly;
# run_nightly.sh kills the bench client at this point and records the leg as
# timed out (with its log) instead of failing the run.
MAX_LEG_SECONDS=${MAX_LEG_SECONDS:-7200}
# Server readiness deadline. 480B weight load + torch.compile warmup is slow,
# and a cold compile cache makes it much slower.
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-5400}

# Observability. Both are required, not optional: they are how a run proves that
# prefix caching and KV offload actually fired rather than being merely enabled.
ENABLE_METRICS=1
ENABLE_CACHE_REPORT=1

# Ports.
PORT=${PORT:-30000}
PD_PREFILL_PORT=${PD_PREFILL_PORT:-30000}
PD_DECODE_PORT=${PD_DECODE_PORT:-30001}
PD_BOOTSTRAP_PORT=${PD_BOOTSTRAP_PORT:-8998}
PD_ROUTER_PORT=${PD_ROUTER_PORT:-8000}

PYTHON=${PYTHON:-python3}

# ---------------------------------------------------------------------------
# Profile overrides. Applied last so an explicit env var still wins on the
# knobs that are not part of the fixed contract.
# ---------------------------------------------------------------------------
case "$PROFILE" in
    full)
        ;;
    nightly)
        LONG_RAMP_TOKENS=${LONG_RAMP_TOKENS_OVERRIDE:-16384}
        ;;
    ci)
        # Budget: the JobSet gives rank-0's test file `timeout 3600`, of which the
        # health recheck can take the first 600, so the whole sweep must land
        # inside ~45 min with margin. The four legs at the extremes of the
        # contract (lowest and highest concurrency, with and without a
        # long-context session) fit inside that at MAX_LEG_SECONDS below; the
        # numbers are for regression tracking, not a full pareto curve. Raise
        # SWEEP_* on a slice you own.
        LONG_RAMP_TOKENS=${LONG_RAMP_TOKENS_OVERRIDE:-16384}
        SWEEP_BS=${SWEEP_BS_OVERRIDE:-"1 16"}
        SWEEP_LONG=${SWEEP_LONG_OVERRIDE:-"0 1"}
        # TARGET_CONTEXT is left at the contract's 16384: a normal session is
        # already only 3 rounds, so there is nothing to time-box, and shrinking
        # it further would flatten the CI legs toward single-turn.
        LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT_OVERRIDE:-65536}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS_OVERRIDE:-420}
        ;;
    smoke)
        SWEEP_BS=${SWEEP_BS_OVERRIDE:-"1 2"}
        SWEEP_LONG=${SWEEP_LONG_OVERRIDE:-"0 1"}
        LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT_OVERRIDE:-65536}
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS_OVERRIDE:-1800}
        # NOTE: SERVER_CONTEXT_LEN stays 1M even here -- the smoke run must
        # exercise the real server config, only a smaller workload.
        ;;
    *)
        echo "ERROR: unknown PROFILE='$PROFILE' (want full|nightly|ci|smoke)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Derived values.
# ---------------------------------------------------------------------------
# Admission cap, derived from the FINAL sweep -- i.e. after the profile block, so a
# profile (or a SWEEP_BS_OVERRIDE) that shrinks or grows the sweep resizes the
# server with it. Deriving this earlier was a bug: `PROFILE=smoke
# SWEEP_BS_OVERRIDE=4` left MAX_RUNNING=2, so a bs=4 leg was admission-capped at 2
# per DP rank and silently measured a concurrency it did not report.
#
# One server serves every leg (reloading 480 GB per leg would dominate the run), so
# it is sized for the largest leg and the client's semaphore is what varies. Under
# DP attention sglang treats this per DP rank (the scheduler reports
# effective_max_running_requests_per_dp), so the global cap is MAX_RUNNING x dp --
# deliberately above the client's. Sizing above the sweep only buys extra
# torch.compile buckets, which is the most expensive thing in a CI bring-up.
_max_bs=1
for _b in $SWEEP_BS; do (( _b > _max_bs )) && _max_bs=$_b; done
MAX_CONCURRENCY=$_max_bs
MAX_RUNNING=${MAX_RUNNING:-$MAX_CONCURRENCY}
TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_CONCURRENCY}

# What serve.sh passes as --context-length. Always SERVER_CONTEXT_LEN (1M), even
# when LONG_TARGET_CONTEXT is lower: every leg and every profile shares one
# server config, so nothing but the workload differs between two results.
CONTEXT_LEN=$SERVER_CONTEXT_LEN
# Round counts are rounded to the nearest whole round, so a session can end up to
# half a round PAST its target. Keep that much headroom, or the last round of the
# longest session is rejected by the server for a target that looks legal here.
_round_headroom=$(( (LONG_RAMP_TOKENS + TURN_OUTPUT_TOKENS) / 2 ))
if (( LONG_TARGET_CONTEXT + _round_headroom > CONTEXT_LEN )); then
    echo "ERROR: LONG_TARGET_CONTEXT=$LONG_TARGET_CONTEXT (+$_round_headroom of" \
         "rounding headroom) exceeds the server's context length $CONTEXT_LEN;" \
         "the last rounds would be rejected." >&2
    exit 2
fi
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

# Qwen3-Coder-480B is trained at 262144 positions, so a >262k context length
# needs YaRN extrapolation. It is on whenever CONTEXT_LEN exceeds that, which the
# fixed 1M server context always does -- even though at the 256k
# LONG_TARGET_CONTEXT default no request actually goes past the native window.
# Throughput is unaffected either way; set ENABLE_LONG_ROPE=0 to serve at the
# native rope when no leg needs the extrapolation.
ROPE_SCALING_JSON=${ROPE_SCALING_JSON:-'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'}
ENABLE_LONG_ROPE=${ENABLE_LONG_ROPE:-1}
if (( CONTEXT_LEN > 262144 )) && [[ "$ENABLE_LONG_ROPE" == "1" ]]; then
    JSON_MODEL_OVERRIDE=${JSON_MODEL_OVERRIDE:-$ROPE_SCALING_JSON}
else
    JSON_MODEL_OVERRIDE=${JSON_MODEL_OVERRIDE:-}
fi

# Sanity: a leg cannot hold more long sessions than it has slots.
for _b in $SWEEP_BS; do
    case "$_b" in ''|*[!0-9]*) echo "ERROR: SWEEP_BS has non-numeric '$_b'" >&2; exit 2;; esac
done
for _l in $SWEEP_LONG; do
    case "$_l" in ''|*[!0-9]*) echo "ERROR: SWEEP_LONG has non-numeric '$_l'" >&2; exit 2;; esac
done

# ---------------------------------------------------------------------------
# log_config <mode> — dumped into every run directory and every server log so a
# result can never be read without the settings that produced it.
# ---------------------------------------------------------------------------
log_config() {
    local mode="${1:-colocated}"
    # A per-engine layout for the engine roles; for the driver-level modes
    # (pd / both) report the layout of the engines that mode will start.
    local tp dp ep
    case "$mode" in
        colocated|both) tp=$COLOCATED_TP; dp=$COLOCATED_DP; ep=$EP_SIZE_COLOCATED ;;
        prefill|decode|pd) tp=$PD_TP; dp=$PD_DP; ep=$EP_SIZE_PD ;;
        *) tp=$COLOCATED_TP; dp=$COLOCATED_DP; ep=$EP_SIZE_COLOCATED ;;
    esac
    cat <<EOF
==== qwen3-coder-480b tpu agentic benchmark config (mode=$mode) ====
  profile                = $PROFILE
  model                  = $MODEL
  quantization           = $QUANTIZATION   kv_cache_dtype = $KV_CACHE_DTYPE
  tp / dp / attn_tp / ep = $tp / $dp / $((tp / dp)) / $ep  (per engine)
  layouts                = colocated tp=$COLOCATED_TP dp=$COLOCATED_DP attn_tp=$((COLOCATED_TP / COLOCATED_DP)) | pd tp=$PD_TP dp=$PD_DP attn_tp=$((PD_TP / PD_DP)) per engine
  context_len            = $CONTEXT_LEN  (fixed for every test and profile)
  long_rope (yarn)       = $([[ -n "$JSON_MODEL_OVERRIDE" ]] && echo "on: $JSON_MODEL_OVERRIDE" || echo off)
  page_size              = $PAGE_SIZE
  attention_backend      = $ATTENTION_BACKEND
  chunked_prefill        = $CHUNKED_PREFILL_PER_RANK per DP rank -> $((CHUNKED_PREFILL_PER_RANK * dp)) passed to server
  mem_fraction_static    = $MEM_FRACTION_STATIC
  max_running_requests   = $MAX_RUNNING (per DP rank under DP attention)
  torch_compile_max_bs   = $TORCH_COMPILE_MAX_BS
  --- features ---
  prefix_caching         = $([[ "$PREFIX_CACHE" == "1" ]] && echo ENABLED || echo disabled)
  kv_offload (hicache)   = $([[ "$KV_OFFLOAD" == "1" ]] && echo "ENABLED (io=torch_tpu, layout=page_first_fused, host/device ratio=$HICACHE_RATIO)" || echo disabled)
  chunked_prefill        = ENABLED
  metrics endpoint       = $([[ "$ENABLE_METRICS" == "1" ]] && echo ENABLED || echo disabled)
  cache report           = $([[ "$ENABLE_CACHE_REPORT" == "1" ]] && echo ENABLED || echo disabled)
  --- workload ---
  turn input / output    = $TURN_INPUT_TOKENS / $TURN_OUTPUT_TOKENS tokens
  target context         = $TARGET_CONTEXT (normal) / $LONG_TARGET_CONTEXT (long sessions)
  concurrency sweep      = $SWEEP_BS
  long-session sweep     = $SWEEP_LONG
  long ramp tokens/round = $LONG_RAMP_TOKENS $([[ "$LONG_RAMP_TOKENS" != "$TURN_INPUT_TOKENS" ]] && echo "(TIME-BOXED: differs from the $TURN_INPUT_TOKENS-token contract)")
  normal replays         = $NORMAL_REPLAYS (cap $MAX_NORMAL_REPLAYS)
  --- limits ---
  max_leg_seconds        = $MAX_LEG_SECONDS
  health_timeout         = $HEALTH_TIMEOUT
  trace_cache            = $TRACE_CACHE_DIR ($(du -sh "$TRACE_CACHE_DIR" 2>/dev/null | cut -f1 || echo "empty"))
====================================================================
EOF
}
