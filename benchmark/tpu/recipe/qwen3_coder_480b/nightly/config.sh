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
#   parallelism          colocated: tp=16, dp=2  -> attn_tp=8
#                        P/D:       tp=8,  dp=2  -> attn_tp=4  (per engine)
#   server context len   1048576 ALWAYS
#   workload             multi-turn agentic, 4k new input tokens + 1k output
#                        tokens per round, accumulating to the context target
#   context targets      16384 normal / 262144 for the "long" sessions
#   concurrency sweep    1, 16, 32
#   long-request sweep   0, 1, 2, 4 concurrent long-context sessions
#   chunked prefill      4096 tokens per DP rank (= one round's new input)
#   features             prefix caching ON, KV offload (HiCache) ON
#   CI-only feature      the spec multi-host launch shim supplies the matching
#                        Qwen3 draft (3 steps / topk 1 / 4 tokens)
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
# DP size is not just a throughput knob here: each DP group owns its own KV pool
# and radix tree, and dispatch is round_robin (the only non-PD option sglang
# offers -- none of round_robin / total_requests / total_tokens is cache-aware).
# A multi-turn session sends its rounds sequentially, so round r lands on the
# rank that last saw round r-dp, and only that rank's history is a usable prefix.
# The reachable fraction of the single-cache ideal is therefore
#     (R-dp)(R-dp+1) / (R(R-1))
# which is exactly ZERO whenever R <= dp. At the old dp=4 with a 3-round session
# that is 0, and the measured hit rate was 0.0% on every c1 leg -- the prefix
# cache could not hit even once. dp=2 with the 6-round session set below reaches
# 2/3 of the ideal; measured 50.02% against a predicted 50.5%.
# Overridable so a sweep can vary it without editing this file.
#
# TODO(revisit): dp=2 is for cache reachability, not speed -- attn_tp=2 (dp=8)
# is typically the faster TPU v7 layout. It needs cache-aware routing first: by
# the formula above, dp=8 alone would want R >= 34 (~174k context per session).
COLOCATED_DP=${COLOCATED_DP:-2} # -> attn_tp = 16/2 = 8
PD_TP=8
PD_DP=2                         # -> attn_tp = 8/2 = 4

# Pin the KV pool. Left unset it is derived from MEM_FRACTION_STATIC and sized
# per DP group, so it scales inversely with DP: at dp=4 that landed on 877440
# tokens (12.97 GB of K and of V per rank), and simply halving DP doubled it to
# 1864192 / 27.56 GB. With ~30 GB of FP8 weights per chip that leaves nothing for
# compile temporaries and the extend graph dies before the server is ever
# healthy -- "ran out of memory on HBM, HLO temporaries (110.23G) exceeds
# available HBM (94.74G)". Pinning makes the pool independent of DP, so this
# cannot silently return the next time the parallelism changes.
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-877440}
export MAX_TOTAL_TOKENS

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

# Optional EAGLE3 draft. The generic benchmark recipe remains non-speculative
# unless its caller supplies DRAFT_MODEL; the spec multi-host CI shim does,
# because speculative decoding is part of that test's contract. Keeping this switch
# avoids changing the recipe's P/D and hand-run defaults just because the CI
# launch group gained another feature.
DRAFT_MODEL=${DRAFT_MODEL:-}
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_EAGLE_TOPK=1
SPECULATIVE_NUM_DRAFT_TOKENS=4
# The synthetic vocabulary-sampled workload has intentionally low semantic
# predictability, so its first calibrated gate is the non-fallback boundary:
# strictly above 1.0 proves that the draft contributed at least one token.
# Override after enough nightly data exists to ratchet this into a quality floor.
MIN_SPEC_ACCEPT_LENGTH=${MIN_SPEC_ACCEPT_LENGTH:-1.0}

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

# Attention backend. These runs were validated on pallas_tpu_rpa_v3 with
# --hicache-io-backend torch_tpu. The faster pallas_tpu_rpa_batched also supports
# KV offload on the fused KV pool (not with SGLANG_TPU_USE_SEQ_ALONG_LANE_KV; see
# hicache_hook.py resolve_layout_io_compatibility) but has no baseline here yet.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}

# The 1M context plus the request pool's extra slots needs 8193 pages at size
# 128. At 32 rows that scalar-prefetched page table exceeds the 1 MiB SMEM
# budget. Size 512 also covers the full profile's 64-row compile ceiling
# without reducing context or concurrency.
PAGE_SIZE=${PAGE_SIZE:-512}

# HBM split. MEASURED on a 2x2x2 tpu7x slice (94.74 GB HBM/rank, weights 30.90 GB
# at tp=16), not estimated:
#   0.85 -> XLA compile OOM, 99.26 GB of HLO temporaries vs 94.74 GB available
#   0.70 -> OOM during precompile, a 585 MB alloc with 406 MB free
#   0.60 -> starts; KV pool 877,440 tokens/rank, ~37.9 GB left for temporaries
# 0.60 is the largest value observed to start. The old 0.85 was a reasoned
# estimate that had never been run and cannot boot this model. Sized for the
# smoke sweep; a larger sweep changes the activation profile, so re-measure
# before trusting it for PROFILE=ci/full.
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.60}

# Host KV pool = HICACHE_RATIO x device KV pool. With the 16k normal target the
# offload pressure comes from the long sessions and the replayed prefixes, not
# from the normal batch: a bs=32 x 4-long leg holds ~1M long-session tokens plus
# ~0.5M of normal context, and 3x the device pool covers that once the device
# pool overflows -- the case KV offload exists for. Costs host RAM: ratio x
# device pool bytes, so check the pod's memory limit before raising it.
# MEASURED: raiden pins its host pool eagerly, and at ratio 3 that is 8 ranks x
# 3 x the ~26 GB device pool ~= 624 GB, which OOMKilled the pod (node has 963 GB
# allocatable and jobset-tpu.yaml requests only 32Gi). Ratio 1 (~208 GB) starts.
# Raising this back toward 3 needs the pod's memory request raised with it --
# and note the unmerged upstream fix that derives raiden's host staging from a
# memory budget rather than from the context length.
HICACHE_RATIO=${HICACHE_RATIO:-1}
# TPU KV-offload io backend: raiden (default) or torch_tpu. See serve.sh for why
# raiden -- torch_tpu's precompile_kv_offload stages a pow2-rounded whole-pool
# D2H needing ~2x the KV pool in HBM.
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-raiden}

# How many FRESH normal sessions each non-long concurrency slot plays
# back-to-back. During a leg with long sessions, the short sessions finish
# sooner than the long ones; without the extra sessions the leg spends most
# of its wall clock at (bs - long) idle slots and the aggregate numbers
# describe a batch that was never full. The extra sessions cost no wall clock
# (the long sessions set the leg's length) -- only trace size and generation
# time, hence the cap. Formerly named NORMAL_REPLAYS, which wrongly suggested
# re-sending the same session (each is distinct -- no artificial prefix-cache
# hits); the old env names still work.
# In CI, auto also supplies at least three sessions per normal slot when
# concurrency > 1 (subject to the cap), to replace sessions that finish early.
# This increases measured work; c1 legs still use the round-count calculation.
SESSIONS_PER_SLOT=${SESSIONS_PER_SLOT:-${NORMAL_REPLAYS:-auto}}
MAX_SESSIONS_PER_SLOT=${MAX_SESSIONS_PER_SLOT:-${MAX_NORMAL_REPLAYS:-8}}

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
        # Share a small compile grid across baseline and EAGLE. Compiling every
        # derived shape for the 480B model can exhaust host RAM during startup.
        export TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-4096}"
        export TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 8 16}"
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
        # 6 rounds, not the contract's 3. A session must outlast the DP count for
        # any round to find its prefix (see COLOCATED_DP), and at dp=2 three
        # rounds reach only a third of the single-cache ideal -- under the 60%
        # gate, so the leg would report throughput measured with a cache that
        # barely engages. Six rounds reach two thirds and leave real margin.
        # This does move the normal session off the Mooncake toolagent p90
        # (16.8k) the 16384 default was anchored to; 31k is still inside the
        # corpus's range (max 126k).
        TARGET_CONTEXT=${TARGET_CONTEXT_OVERRIDE:-30720}   # 30720/5130 -> 6 rounds
        LONG_TARGET_CONTEXT=${LONG_TARGET_CONTEXT_OVERRIDE:-65536}
        # Twice the rounds, and prefill grows with history, so a leg costs about
        # 1.9x what it did: c1-long0 went 80.8s -> 150.1s measured. The largest
        # leg was 275.9s, which scales past the old 420s cap and would be
        # time-boxed -- and a time-boxed leg measures something else. 900s keeps
        # every leg intact; four legs at ~1200s total still fit SWEEP_TIMEOUT.
        MAX_LEG_SECONDS=${MAX_LEG_SECONDS_OVERRIDE:-900}
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
# deliberately above the client's. serve.sh pre-multiplies by DP_SIZE to get
# there, the same way it does for CHUNKED_PREFILL_PER_RANK; this value is
# per-rank. Sizing above the sweep only buys extra
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
# rope_theta MUST be restated here. transformers 5.x folds rope_scaling into
# rope_parameters, and sglang's get_rope_config() then reads
# rope_params["rope_theta"] unconditionally -- so an override that omits it
# REPLACES the checkpoint's rope_parameters with a dict that has no theta, and
# every scheduler dies at model init with `KeyError: 'rope_theta'` before a
# single weight is loaded. Verified against transformers 5.8.1 on the real
# checkpoint: the override without rope_theta fails, with it resolves to
# {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings':
# 262144, 'rope_theta': 10000000}.
# Must match the checkpoint's own config.json rope_theta (10000000 here).
ROPE_THETA=${ROPE_THETA:-10000000}
ROPE_SCALING_JSON=${ROPE_SCALING_JSON:-"{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":262144,\"rope_theta\":$ROPE_THETA}}"}
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
  kv_offload (hicache)   = $([[ "$KV_OFFLOAD" == "1" ]] && echo "ENABLED (io=$HICACHE_IO_BACKEND, layout=page_first_fused, host/device ratio=$HICACHE_RATIO)" || echo disabled)
  speculative decoding   = $([[ -n "$DRAFT_MODEL" ]] && echo "EAGLE3 (draft=$DRAFT_MODEL, steps/topk/draft_tokens=$SPECULATIVE_NUM_STEPS/$SPECULATIVE_EAGLE_TOPK/$SPECULATIVE_NUM_DRAFT_TOKENS)" || echo disabled)
  min spec accept length = > $MIN_SPEC_ACCEPT_LENGTH (every scheduler)
  chunked_prefill        = ENABLED
  metrics endpoint       = $([[ "$ENABLE_METRICS" == "1" ]] && echo ENABLED || echo disabled)
  cache report           = $([[ "$ENABLE_CACHE_REPORT" == "1" ]] && echo ENABLED || echo disabled)
  --- workload ---
  turn input / output    = $TURN_INPUT_TOKENS / $TURN_OUTPUT_TOKENS tokens
  target context         = $TARGET_CONTEXT (normal) / $LONG_TARGET_CONTEXT (long sessions)
  concurrency sweep      = $SWEEP_BS
  long-session sweep     = $SWEEP_LONG
  long ramp tokens/round = $LONG_RAMP_TOKENS $([[ "$LONG_RAMP_TOKENS" != "$TURN_INPUT_TOKENS" ]] && echo "(TIME-BOXED: differs from the $TURN_INPUT_TOKENS-token contract)")
  sessions per slot      = $SESSIONS_PER_SLOT (cap $MAX_SESSIONS_PER_SLOT)
  --- limits ---
  max_leg_seconds        = $MAX_LEG_SECONDS
  health_timeout         = $HEALTH_TIMEOUT
  trace_cache            = $TRACE_CACHE_DIR ($(du -sh "$TRACE_CACHE_DIR" 2>/dev/null | cut -f1 || echo "empty"))
====================================================================
EOF
}
