#!/usr/bin/env bash
# Run ONE leg of the agentic benchmark: (concurrency BS) x (NUM_LONG sessions
# accumulating to LONG_TARGET_CONTEXT), against an already-running server.
#
#   BS=32 NUM_LONG=2 BASE_URL=http://127.0.0.1:30000 \
#   OUTPUT_DIR=./out/colocated ./bench_leg.sh
#
# Follows benchmark/tpu/bench_serving.sh's contract (health poll, one JSONL of
# metrics per leg + a dump_meta.py repro record beside it) but drives the
# multi-turn `agentic-trace` dataset, which needs a chat backend, a generated
# trace file and a per-round output length -- none of which bench_serving.sh
# exposes.
#
# What it emits into $OUTPUT_DIR (all tagged c<BS>-long<NUM_LONG>):
#   <tag>.jsonl          bench_serving metrics summary (the numbers)
#   <tag>.meta.json      dump_meta record: git, versions, argv, live /server_info
#   <tag>.metrics.pre/post.json + <tag>.metrics.delta.json
#                        Prometheus counters around the leg: prefix-cache hits,
#                        KV-offload traffic, retractions, P/D transfer failures
#   <tag>.bench.log      full client stdout/stderr
#   <tag>.status         one line: ok | timeout | failed:<exit code>
#
# Env (beyond config.sh):
#   BS=                REQUIRED  client concurrency (max in-flight sessions)
#   NUM_LONG=0                   sessions that reach LONG_TARGET_CONTEXT
#   BASE_URL=http://127.0.0.1:30000
#   OUTPUT_DIR=        REQUIRED
#   TRACE=                       trace JSON; generated into TRACE_DIR if unset
#   TRACE_DIR=$TRACE_CACHE_DIR   shared, persistent trace cache (see config.sh).
#                                One trace serves every concurrency at a given
#                                NUM_LONG: it is built for the largest leg in
#                                SWEEP_BS and --num-prompts takes a prefix.
#   FLUSH_CACHE=1                POST /flush_cache first so legs are independent
#   SEED=1
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"

BS=${BS:?set BS (client concurrency)}
NUM_LONG=${NUM_LONG:-0}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
TRACE_DIR=${TRACE_DIR:-$TRACE_CACHE_DIR}
FLUSH_CACHE=${FLUSH_CACHE:-1}
SEED=${SEED:-1}

if (( NUM_LONG > BS )); then
    echo "ERROR: NUM_LONG=$NUM_LONG > BS=$BS -- a leg cannot hold more long" \
         "sessions than it has concurrency slots." >&2
    exit 2
fi
TAG="c${BS}-long${NUM_LONG}"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"
BENCH_LOG="$OUTPUT_DIR/${TAG}.bench.log"
STATUS_FILE="$OUTPUT_DIR/${TAG}.status"

# -----------------------------------------------------------------------
# Rounds per session, mirroring gen_agentic_workload.py's arithmetic. Used
# to size the normal-session pool; the generator remains the source of truth
# and records its own resolved counts in the trace metadata.
# -----------------------------------------------------------------------
# Nearest whole round, matching _turns_for_context() -- truncating would cost a
# whole round, which at the 8k normal target is the difference between a 2-round
# session and a single-turn request.
_long_growth=$(( LONG_RAMP_TOKENS + TURN_OUTPUT_TOKENS ))
_normal_growth=$(( TURN_INPUT_TOKENS + TURN_OUTPUT_TOKENS ))
long_rounds=$(( (LONG_TARGET_CONTEXT + _long_growth / 2) / _long_growth ))
normal_rounds=$(( (TARGET_CONTEXT + _normal_growth / 2) / _normal_growth ))
(( long_rounds < 1 )) && long_rounds=1
(( normal_rounds < 1 )) && normal_rounds=1

# Normal sessions are replayed so the non-long slots stay busy for as long as
# the long sessions run; see config.sh (NORMAL_REPLAYS). `replays` depends only on
# the round counts, so it is the same for every concurrency.
replays=$NORMAL_REPLAYS
if [[ "$replays" == "auto" ]]; then
    if (( NUM_LONG > 0 )); then
        replays=$(( (long_rounds + normal_rounds - 1) / normal_rounds ))
        (( replays > MAX_NORMAL_REPLAYS )) && replays=$MAX_NORMAL_REPLAYS
        (( replays < 1 )) && replays=1
    else
        replays=1
    fi
fi
sessions_for() {   # sessions a leg of concurrency $1 sends
    local bs="$1" n
    n=$(( NUM_LONG + (bs - NUM_LONG) * replays ))
    (( n < bs )) && n=$bs
    echo "$n"
}
num_sessions=$(sessions_for "$BS")

# One trace serves every concurrency at this NUM_LONG: build it for the LARGEST
# leg in the sweep and let --num-prompts take a prefix. That is exact, not an
# approximation -- the generator emits the long sessions first and the loader
# takes the first N conversations, so a prefix of the bs=32 trace is byte-for-byte
# the trace a bs=16 leg would have generated for itself. Turns ~1 trace per leg
# into 1 per SWEEP_LONG value.
trace_bs=$MAX_CONCURRENCY
(( BS > trace_bs )) && trace_bs=$BS       # a hand-run leg above the sweep's max
trace_sessions=$(sessions_for "$trace_bs")

# Trace path keyed by every knob that changes its content, so a changed knob can
# never silently reuse a stale file.
TRACE=${TRACE:-$TRACE_DIR/agentic-s${trace_sessions}-long${NUM_LONG}-in${TURN_INPUT_TOKENS}-out${TURN_OUTPUT_TOKENS}-ctx${TARGET_CONTEXT}-lctx${LONG_TARGET_CONTEXT}-ramp${LONG_RAMP_TOKENS}-seed${SEED}.json}
if [[ -s "$TRACE" ]]; then
    echo "[leg $TAG] reusing trace $TRACE" \
         "(built for $trace_sessions sessions; this leg sends $num_sessions)"
else
    echo "[leg $TAG] generating trace $TRACE ($trace_sessions sessions," \
         "shared by every concurrency at long=$NUM_LONG)"
    gen_rc=0
    # Generate to a per-process temp path and rename into place. The cache is
    # shared across runs, so a half-written file must never be visible: `-s`
    # would happily accept a truncated trace and poison every later run, and two
    # concurrent runs would otherwise interleave writes to the same path. rename
    # within one directory is atomic.
    trace_tmp="${TRACE}.tmp.$$"
    # The cache is persistent, so an orphaned temp from a killed run would sit
    # there forever. Clear it on any exit; the rename below makes this a no-op on
    # the success path.
    trap 'rm -f "$trace_tmp"' EXIT
    "$PYTHON" "$HERE/gen_agentic_workload.py" \
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
        # Report as a leg failure rather than aborting the sweep: a missing
        # tokenizer or an out-of-disk shows up as one bad row, not a dead run.
        echo "failed:trace-gen" >"$STATUS_FILE"
        echo "[leg $TAG] trace generation FAILED (rc=$gen_rc); see" \
             "$OUTPUT_DIR/${TAG}.trace.log" >&2
        rm -f "$trace_tmp"
        exit 0
    fi
    mv -f "$trace_tmp" "$TRACE"
    trap - EXIT
fi

# ---------------------------------------------------------------------------
# Wait for the server. Same poll as bench_serving.sh: /health answers 503 while
# compile buckets warm up, so `curl -f` would fail-fast on a healthy server.
# ---------------------------------------------------------------------------
echo "[leg $TAG] waiting up to ${HEALTH_TIMEOUT}s for $BASE_URL/health"
health_start=$SECONDS
while :; do
    code=$(curl -sS -o /dev/null -m 10 -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")
    [[ "$code" == "200" ]] && { echo "[leg $TAG]   healthy after $((SECONDS - health_start))s"; break; }
    if (( SECONDS - health_start >= HEALTH_TIMEOUT )); then
        echo "ERROR: $BASE_URL/health never reached 200 (last=$code)" >&2
        echo "failed:health" >"$STATUS_FILE"
        exit 1
    fi
    sleep 10
done

if [[ "$FLUSH_CACHE" == "1" ]]; then
    # Legs must not inherit the previous leg's radix tree, or a later leg reports
    # cache hits it did not earn.
    curl -sS -X POST -m 60 "$BASE_URL/flush_cache" >/dev/null 2>&1 \
        && echo "[leg $TAG] flushed prefix cache" \
        || echo "[leg $TAG] WARN: /flush_cache failed; cache carries over" >&2
    sleep 2
fi

"$PYTHON" "$HERE/metrics_probe.py" snapshot --url "$BASE_URL" \
    --out "$OUTPUT_DIR/${TAG}.metrics.pre.json" || true

# ---------------------------------------------------------------------------
# The measured run.
# ---------------------------------------------------------------------------
# --warmup-requests 0 is load-bearing for the synthetic test: bench_serving warms
# up with input_requests[0], and with long sessions emitted first that would
# replay an entire long-context conversation before the timer even starts.
# Compile-bucket warmup is done once per server by run_nightly.sh instead.
bench_args=(
    -m sglang.benchmark.serving
    --backend sglang-oai-chat            # round-by-round replay needs a chat API
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
    # Adds a cache_report block to the JSONL with the device / host / storage
    # split of cached prompt tokens -- the client-side proof that prefix caching
    # hit and that reloads came from the host pool rather than being recomputed.
    --cache-report
)
[[ -n "${EXTRA_BENCH_ARGS:-}" ]] && bench_args+=(${EXTRA_BENCH_ARGS})

{
    echo "==== leg $TAG ===="
    echo "  base_url        = $BASE_URL"
    echo "  concurrency     = $BS"
    echo "  long sessions   = $NUM_LONG  (target ctx $LONG_TARGET_CONTEXT, ${long_rounds} rounds each)"
    echo "  normal sessions = $(( num_sessions - NUM_LONG ))  (target ctx $TARGET_CONTEXT, ${normal_rounds} rounds each, ${replays} replay(s) per slot)"
    echo "  total sessions  = $num_sessions   total rounds = $(( NUM_LONG * long_rounds + (num_sessions - NUM_LONG) * normal_rounds ))"
    echo "  per round       = $TURN_INPUT_TOKENS in / $TURN_OUTPUT_TOKENS out"
    echo "  trace           = $TRACE (holds $trace_sessions sessions; first $num_sessions used)"
    echo "  deadline        = ${MAX_LEG_SECONDS}s"
    echo "=================="
} | tee "$BENCH_LOG"

set +e
# `timeout -k`: SIGTERM at the deadline, SIGKILL 60s later if the client ignores
# it (a stuck aiohttp session does). Exit 124 is timeout's signal for "deadline".
timeout -k 60 "$MAX_LEG_SECONDS" "$PYTHON" "${bench_args[@]}" 2>&1 | tee -a "$BENCH_LOG"
bench_rc=${PIPESTATUS[0]}
set -e

if [[ "$bench_rc" == "0" ]]; then
    echo "ok" >"$STATUS_FILE"
elif [[ "$bench_rc" == "124" || "$bench_rc" == "137" ]]; then
    echo "timeout" >"$STATUS_FILE"
    echo "[leg $TAG] TIMED OUT after ${MAX_LEG_SECONDS}s -- see $BENCH_LOG" >&2
else
    echo "failed:$bench_rc" >"$STATUS_FILE"
    echo "[leg $TAG] FAILED rc=$bench_rc -- see $BENCH_LOG" >&2
fi

# ---------------------------------------------------------------------------
# Post-leg observability. Runs even on failure: a timed-out leg's counters are
# often the whole diagnosis (e.g. retractions, or zero cache hits).
# ---------------------------------------------------------------------------
"$PYTHON" "$HERE/metrics_probe.py" snapshot --url "$BASE_URL" \
    --out "$OUTPUT_DIR/${TAG}.metrics.post.json" || true
# The longest context this leg is expected to reach, for the capacity check: the
# long target when the leg has long sessions, else the normal target.
if (( NUM_LONG > 0 )); then
    expect_context=$LONG_TARGET_CONTEXT
else
    expect_context=$TARGET_CONTEXT
fi
"$PYTHON" "$HERE/metrics_probe.py" diff \
    --pre "$OUTPUT_DIR/${TAG}.metrics.pre.json" \
    --post "$OUTPUT_DIR/${TAG}.metrics.post.json" \
    --out "$OUTPUT_DIR/${TAG}.metrics.delta.json" \
    --expect-context "$expect_context" \
    --expect-concurrency "$BS" 2>&1 | tee -a "$BENCH_LOG" || true

# Repro record, same shape as bench_serving.sh's (kind=bench) so the existing
# dashboard ingest works unchanged.
bench_argv_json=$("$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${bench_args[@]}")
"$PYTHON" "$TPU_BENCH_DIR/dump_meta.py" --kind bench \
    --out "$OUTPUT_DIR/${TAG}.meta.json" \
    --json bench_argv="$bench_argv_json" \
    --set model="$MODEL" \
    --set model_slug="$MODEL_SLUG" \
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
    --set normal_replays="$replays" \
    --set rounds_per_long_session="$long_rounds" \
    --set rounds_per_normal_session="$normal_rounds" \
    --set turn_input_tokens="$TURN_INPUT_TOKENS" \
    --set turn_output_tokens="$TURN_OUTPUT_TOKENS" \
    --set long_ramp_tokens="$LONG_RAMP_TOKENS" \
    --set target_context="$TARGET_CONTEXT" \
    --set long_target_context="$LONG_TARGET_CONTEXT" \
    --set server_context_len="$SERVER_CONTEXT_LEN" \
    --set chunked_prefill_per_rank="$CHUNKED_PREFILL_PER_RANK" \
    --set prefix_cache="$PREFIX_CACHE" \
    --set kv_offload="$KV_OFFLOAD" \
    --set hicache_ratio="$HICACHE_RATIO" \
    --set seed="$SEED" \
    --set max_leg_seconds="$MAX_LEG_SECONDS" \
    --set status="$(cat "$STATUS_FILE")" \
    --set summary_file="${TAG}.jsonl" \
    --server-info-url "$BASE_URL" \
    || echo "[leg $TAG] WARN: dump_meta failed" >&2

echo "[leg $TAG] status=$(cat "$STATUS_FILE")  artifacts in $OUTPUT_DIR"
# The leg's own failure must not abort the sweep; run_nightly.sh reads .status.
exit 0
