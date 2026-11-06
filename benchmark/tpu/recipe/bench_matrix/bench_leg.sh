#!/usr/bin/env bash
# Run ONE leg of the TPU benchmark matrix: (concurrency BS) x (NUM_LONG
# long requests/sessions) against an already-running endpoint. Adapted from
# recipe/qwen3_coder_480b/nightly/bench_leg.sh; the tools it drives
# (gen_agentic_workload.py, metrics_probe.py, dump_meta.py) are reused from
# there, not copied.
#
# Emits into $OUTPUT_DIR (all tagged c<BS>-long<NUM_LONG>):
#   <tag>.jsonl / <tag>.meta.json / <tag>.bench.log / <tag>.status
#   <tag>.metrics.{pre,post,delta}.json           (from PROBE_URL)
#   <tag>.decode.metrics.{pre,post,delta}.json    (P/D only, from PROBE_URL_2)
#
# Env (beyond config.sh):
#   BS=            REQUIRED  client concurrency
#   NUM_LONG=0
#   BASE_URL=http://127.0.0.1:30000     endpoint to bench (mini-LB for P/D)
#   PROBE_URL=$BASE_URL                 where /metrics lives (P/D: the prefill
#                                       engine -- the LB serves no /metrics)
#   PROBE_URL_2=                        optional second /metrics snapshot
#   OUTPUT_DIR=    REQUIRED
#   TRACE_DIR=$TRACE_CACHE_DIR
#   FLUSH_CACHE=1
#   SEED=1
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"

BS=${BS:?set BS (client concurrency)}
NUM_LONG=${NUM_LONG:-0}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
PROBE_URL=${PROBE_URL:-$BASE_URL}
PROBE_URL_2=${PROBE_URL_2:-}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
TRACE_DIR=${TRACE_DIR:-$TRACE_CACHE_DIR}
FLUSH_CACHE=${FLUSH_CACHE:-1}
SEED=${SEED:-1}

if (( NUM_LONG > BS )); then
    echo "ERROR: NUM_LONG=$NUM_LONG > BS=$BS" >&2
    exit 2
fi
TAG="c${BS}-long${NUM_LONG}"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"
BENCH_LOG="$OUTPUT_DIR/${TAG}.bench.log"
STATUS_FILE="$OUTPUT_DIR/${TAG}.status"

# Rounds per session, mirroring gen_agentic_workload.py's nearest-whole-round
# arithmetic (the generator stays the source of truth and records its resolved
# counts in the trace metadata).
_long_growth=$(( LONG_RAMP_TOKENS + TURN_OUTPUT_TOKENS ))
_normal_growth=$(( TURN_INPUT_TOKENS + TURN_OUTPUT_TOKENS ))
long_rounds=$(( (LONG_TARGET_CONTEXT + _long_growth / 2) / _long_growth ))
normal_rounds=$(( (TARGET_CONTEXT + _normal_growth / 2) / _normal_growth ))
(( long_rounds < 1 )) && long_rounds=1
(( normal_rounds < 1 )) && normal_rounds=1

# Fresh sessions per non-long slot, so those slots stay busy while the long
# sessions run (see SESSIONS_PER_SLOT in config.sh; the 480B nightly's
# NORMAL_REPLAYS is the same policy).
sessions_per_slot=$SESSIONS_PER_SLOT
if [[ "$sessions_per_slot" == "auto" ]]; then
    if (( NUM_LONG > 0 )); then
        sessions_per_slot=$(( (long_rounds + normal_rounds - 1) / normal_rounds ))
        (( sessions_per_slot > MAX_SESSIONS_PER_SLOT )) && sessions_per_slot=$MAX_SESSIONS_PER_SLOT
        (( sessions_per_slot < 1 )) && sessions_per_slot=1
    else
        sessions_per_slot=1
    fi
fi
sessions_for() {
    local bs="$1" n
    n=$(( NUM_LONG + (bs - NUM_LONG) * sessions_per_slot ))
    (( n < bs )) && n=$bs
    echo "$n"
}
num_sessions=$(sessions_for "$BS")

# One trace serves every concurrency at this NUM_LONG (long sessions come
# first; --num-prompts takes a prefix).
trace_bs=$MAX_CONCURRENCY
(( BS > trace_bs )) && trace_bs=$BS
trace_sessions=$(sessions_for "$trace_bs")

TRACE=${TRACE:-$TRACE_DIR/${WORKLOAD}-${BENCH_MODEL}-s${trace_sessions}-long${NUM_LONG}-in${TURN_INPUT_TOKENS}-out${TURN_OUTPUT_TOKENS}-ctx${TARGET_CONTEXT}-lctx${LONG_TARGET_CONTEXT}-ramp${LONG_RAMP_TOKENS}-seed${SEED}.json}
if [[ -s "$TRACE" ]]; then
    echo "[leg $TAG] reusing trace $TRACE"
else
    echo "[leg $TAG] generating trace $TRACE ($trace_sessions sessions)"
    gen_rc=0
    trace_tmp="${TRACE}.tmp.$$"
    trap 'rm -f "$trace_tmp"' EXIT
    "$PYTHON" "$NIGHTLY_TOOLS_DIR/gen_agentic_workload.py" \
        --out "$trace_tmp" \
        --tokenizer "${TOKENIZER:-$MODEL}" \
        --num-sessions "$trace_sessions" \
        --num-long "$NUM_LONG" \
        --turn-input-tokens "$TURN_INPUT_TOKENS" \
        --long-turn-input-tokens "$LONG_RAMP_TOKENS" \
        --turn-output-tokens "$TURN_OUTPUT_TOKENS" \
        --target-context "$TARGET_CONTEXT" \
        --long-target-context "$LONG_TARGET_CONTEXT" \
        --seed "$SEED" \
        > >(tee "$OUTPUT_DIR/${TAG}.trace.log") 2>&1 || gen_rc=$?
    if (( gen_rc != 0 )) || [[ ! -s "$trace_tmp" ]]; then
        echo "failed:trace-gen" >"$STATUS_FILE"
        echo "[leg $TAG] trace generation FAILED (rc=$gen_rc)" >&2
        rm -f "$trace_tmp"
        exit 0
    fi
    mv -f "$trace_tmp" "$TRACE"
    trap - EXIT
fi

echo "[leg $TAG] waiting up to ${HEALTH_TIMEOUT}s for $BASE_URL/health"
health_start=$SECONDS
while :; do
    code=$(curl -sS -o /dev/null -m "$CURL_MAX_TIME" -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")
    [[ "$code" == "200" ]] && break
    if (( SECONDS - health_start >= HEALTH_TIMEOUT )); then
        echo "failed:health" >"$STATUS_FILE"
        echo "ERROR: $BASE_URL/health never reached 200 (last=$code)" >&2
        exit 1
    fi
    sleep "$HEALTH_POLL_INTERVAL"
done

if [[ "$FLUSH_CACHE" == "1" ]]; then
    # Legs must not inherit the previous leg's radix tree (the mini-LB
    # forwards /flush_cache to both engines).
    curl -sS -X POST -m 60 "$BASE_URL/flush_cache" >/dev/null 2>&1 \
        && echo "[leg $TAG] flushed prefix cache" \
        || echo "[leg $TAG] WARN: /flush_cache failed; cache carries over" >&2
    sleep 2
fi

"$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" snapshot --url "$PROBE_URL" \
    --out "$OUTPUT_DIR/${TAG}.metrics.pre.json" || true
[[ -n "$PROBE_URL_2" ]] && "$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" snapshot \
    --url "$PROBE_URL_2" --out "$OUTPUT_DIR/${TAG}.decode.metrics.pre.json" || true

# --warmup-requests 0: bench_serving would replay input_requests[0] -- an
# entire long conversation -- before the timer starts. TPU compile buckets are
# precompiled at server start instead.
bench_args=(
    -m sglang.benchmark.serving
    --backend sglang-oai-chat
    --base-url "$BASE_URL"
    --model "$MODEL"
    --dataset-name agentic-trace
    --dataset-path "$TRACE"
    --sharegpt-output-len "$TURN_OUTPUT_TOKENS"
    --num-prompts "$num_sessions"
    --max-concurrency "$BS"
    --warmup-requests 0
    --seed "$SEED"
    --output-file "$OUTPUT_DIR/${TAG}.jsonl"
    --output-details
    --cache-report
)
[[ -n "${EXTRA_BENCH_ARGS:-}" ]] && bench_args+=(${EXTRA_BENCH_ARGS})

{
    echo "==== leg $TAG ($BENCH_MODEL/$BENCH_RECIPE/$WORKLOAD) ===="
    echo "  base_url        = $BASE_URL  (metrics: $PROBE_URL)"
    echo "  concurrency     = $BS"
    echo "  long requests   = $NUM_LONG  (target ctx $LONG_TARGET_CONTEXT, ${long_rounds} round(s) each)"
    echo "  normal sessions = $(( num_sessions - NUM_LONG ))  (target ctx $TARGET_CONTEXT, ${normal_rounds} round(s), ${sessions_per_slot} session(s)/slot)"
    echo "  per round       = $TURN_INPUT_TOKENS in / $TURN_OUTPUT_TOKENS out"
    echo "  deadline        = ${MAX_LEG_SECONDS}s"
    echo "=================="
} | tee "$BENCH_LOG"

set +e
timeout -k "$LEG_KILL_GRACE" "$MAX_LEG_SECONDS" "$PYTHON" "${bench_args[@]}" 2>&1 | tee -a "$BENCH_LOG"
bench_rc=${PIPESTATUS[0]}
set -e

if [[ "$bench_rc" == "0" ]]; then
    echo "ok" >"$STATUS_FILE"
elif [[ "$bench_rc" == "124" || "$bench_rc" == "137" ]]; then
    echo "timeout" >"$STATUS_FILE"
    echo "[leg $TAG] TIMED OUT after ${MAX_LEG_SECONDS}s" >&2
else
    echo "failed:$bench_rc" >"$STATUS_FILE"
    echo "[leg $TAG] FAILED rc=$bench_rc -- see $BENCH_LOG" >&2
fi

# Post-leg observability; runs even on failure (the counters are often the
# whole diagnosis).
"$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" snapshot --url "$PROBE_URL" \
    --out "$OUTPUT_DIR/${TAG}.metrics.post.json" || true
[[ -n "$PROBE_URL_2" ]] && "$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" snapshot \
    --url "$PROBE_URL_2" --out "$OUTPUT_DIR/${TAG}.decode.metrics.post.json" || true
if (( NUM_LONG > 0 )); then
    expect_context=$LONG_TARGET_CONTEXT
else
    expect_context=$TARGET_CONTEXT
fi
"$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" diff \
    --pre "$OUTPUT_DIR/${TAG}.metrics.pre.json" \
    --post "$OUTPUT_DIR/${TAG}.metrics.post.json" \
    --out "$OUTPUT_DIR/${TAG}.metrics.delta.json" \
    --expect-context "$expect_context" \
    --expect-concurrency "$BS" 2>&1 | tee -a "$BENCH_LOG" || true
if [[ -n "$PROBE_URL_2" ]]; then
    # Decode-side stages (bootstrap, transfer queue, waiting) and failures; the
    # prefill-side snapshot above has the KV transfer histograms.
    "$PYTHON" "$NIGHTLY_TOOLS_DIR/metrics_probe.py" diff --role decode \
        --pre "$OUTPUT_DIR/${TAG}.decode.metrics.pre.json" \
        --post "$OUTPUT_DIR/${TAG}.decode.metrics.post.json" \
        --out "$OUTPUT_DIR/${TAG}.decode.metrics.delta.json" 2>&1 \
        | sed 's/^\[metrics_probe\]/[metrics_probe decode]/' | tee -a "$BENCH_LOG" || true
fi

bench_argv_json=$("$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${bench_args[@]}")
"$PYTHON" "$TPU_BENCH_DIR/dump_meta.py" --kind bench \
    --out "$OUTPUT_DIR/${TAG}.meta.json" \
    --json bench_argv="$bench_argv_json" \
    --set model="$MODEL" \
    --set model_slug="$MODEL_SLUG" \
    --set bench_model="$BENCH_MODEL" \
    --set bench_recipe="$BENCH_RECIPE" \
    --set workload="$WORKLOAD" \
    --set base_url="$BASE_URL" \
    --set backend=sglang-oai-chat \
    --set dataset_name=agentic-trace \
    --set dataset_path="$TRACE" \
    --set run_profile="$PROFILE" \
    --set leg_tag="$TAG" \
    --set concurrency="$BS" \
    --set num_long_sessions="$NUM_LONG" \
    --set num_sessions="$num_sessions" \
    --set trace_sessions="$trace_sessions" \
    --set sessions_per_slot="$sessions_per_slot" \
    --set rounds_per_long_session="$long_rounds" \
    --set rounds_per_normal_session="$normal_rounds" \
    --set turn_input_tokens="$TURN_INPUT_TOKENS" \
    --set turn_output_tokens="$TURN_OUTPUT_TOKENS" \
    --set long_ramp_tokens="$LONG_RAMP_TOKENS" \
    --set target_context="$TARGET_CONTEXT" \
    --set long_target_context="$LONG_TARGET_CONTEXT" \
    --set server_context_len="$CONTEXT_LEN" \
    --set chunked_prefill_per_rank="$CHUNKED_PREFILL_SIZE" \
    --set prefix_cache="$PREFIX_CACHE" \
    --set kv_offload="$KV_OFFLOAD" \
    --set spec_decode="$SPEC_DECODE" \
    --set pd_disagg="$PD_DISAGG" \
    --set pd_prefill_engines="$PD_PREFILL_ENGINES" \
    --set hicache_ratio="$HICACHE_RATIO" \
    --set seed="$SEED" \
    --set max_leg_seconds="$MAX_LEG_SECONDS" \
    --set status="$(cat "$STATUS_FILE")" \
    --set summary_file="${TAG}.jsonl" \
    --server-info-url "$PROBE_URL" \
    || echo "[leg $TAG] WARN: dump_meta failed" >&2

echo "[leg $TAG] status=$(cat "$STATUS_FILE")  artifacts in $OUTPUT_DIR"
exit 0
