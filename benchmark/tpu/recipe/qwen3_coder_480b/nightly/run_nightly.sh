#!/usr/bin/env bash
# End-to-end driver for the Qwen3-Coder-480B FP8 TPU agentic benchmark.
# Brings up the engine(s), warms compile buckets once, sweeps
# (concurrency x number of long-context sessions), tears down, and writes a single
# results JSON + markdown summary.
#
#   ./run_nightly.sh                             # colocated, full contract
#   PROFILE=nightly ./run_nightly.sh
#   PROFILE=smoke SWEEP_BS=4 SWEEP_LONG=1 ./run_nightly.sh   # plumbing check
#   MODE=external BASE_URL=http://host:30000 ./run_nightly.sh # sweep a live server
#
#
# Workload traces are NOT here: they go to $TRACE_CACHE_DIR (default
# ~/.cache/sglang_tpu_agentic_traces) and are reused across legs AND across runs.
#
# A failing leg never stops the sweep: its status is recorded and the run moves
# on. The run's exit code is non-zero only if no leg produced metrics, or if
# STRICT=1 and any leg did not end 'ok'.
#
# MODE=colocated needs 2 TPU hosts (tp=16). Easiest is Ray: bring up a cluster
# over both hosts and run this on the head with SGLANG_USE_RAY=1 NNODES=2.
#
# Env (beyond config.sh):
#   MODE=colocated|external      (pd requires ALLOW_PD=1; both is disabled)
#   BASE_URL=                    MODE=external: the server to sweep
#   EXTERNAL_MODE_LABEL=colocated  MODE=external: results subdir / summary label
#   RUN_ROOT=./results/nightly/<ts>
#   STRICT=0                 1 -> exit non-zero unless every leg is 'ok'
#   SKIP_WARMUP=0            1 -> skip the compile-bucket warmup pass
#   PD_PREFILL_SSH= PD_DECODE_SSH=      ssh targets for the P/D engines
#   PREFILL_BASE_URL= DECODE_BASE_URL=  pre-started P/D engines (skips launch)
#   ROUTER_BASE_URL=                    pre-started router (skips launching one)
#   ROUTER_HOST=127.0.0.1
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"

MODE=${MODE:-colocated}
STRICT=${STRICT:-0}
SKIP_WARMUP=${SKIP_WARMUP:-0}
ROUTER_HOST=${ROUTER_HOST:-127.0.0.1}
TS=$(date -u +%Y%m%d_%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-$HERE/results/nightly/$TS}
mkdir -p "$RUN_ROOT"

# Absolute so the ssh-launched engines and the summarizer agree on paths.
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd)
# Traces live in the shared, persistent cache -- NOT under the timestamped run
# dir. They are deterministic given (shape, seed), so every run after the first
# reuses them instead of spending ~5 min regenerating identical content.
TRACE_DIR="$TRACE_CACHE_DIR"
mkdir -p "$TRACE_DIR"

log_config "$MODE" | tee "$RUN_ROOT/config.log"
{
    echo "run_root  = $RUN_ROOT"
    echo "mode      = $MODE"
    echo "host      = $(hostname)"
    echo "started   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git       = $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} | tee -a "$RUN_ROOT/config.log"

# Every child process we start, so the EXIT trap can always clean up. A leaked
# 480B engine holds every chiplet on the host and blocks the next run.
declare -a CHILD_PIDS=()
cleanup() {
    local rc=$?
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "[run] stopping pid $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # Give the engines time to release the TPU before SIGKILL: a half-dead
    # process keeps its chiplets claimed and the next launch fails on init.
    for _ in $(seq 1 30); do
        local alive=0
        for pid in "${CHILD_PIDS[@]:-}"; do
            [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
        done
        (( alive )) || break
        sleep 2
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
    done
    if [[ -n "${PD_PREFILL_SSH:-}" ]]; then
        ssh -o BatchMode=yes "$PD_PREFILL_SSH" "pkill -f 'sglang.launch_server' || true" 2>/dev/null || true
    fi
    if [[ -n "${PD_DECODE_SSH:-}" ]]; then
        ssh -o BatchMode=yes "$PD_DECODE_SSH" "pkill -f 'sglang.launch_server' || true" 2>/dev/null || true
    fi
    return $rc
}
trap cleanup EXIT
# On a signal, clean up and *stop*. A bare `trap cleanup INT` would return into
# the sweep with the engines already killed, and every remaining leg would fail
# against a dead server. 130 is the conventional SIGINT exit status; the EXIT
# trap fires again and is a no-op by then.
trap 'cleanup; exit 130' INT TERM

# ---------------------------------------------------------------------------
wait_healthy() {
    local url="$1" label="$2" timeout="$3" watch_pid="${4:-}"
    local start=$SECONDS code
    echo "[run] waiting up to ${timeout}s for $label at $url"
    while :; do
        code=$(curl -sS -o /dev/null -m 10 -w "%{http_code}" "$url/health" 2>/dev/null || echo "000")
        if [[ "$code" == "200" ]]; then
            echo "[run] $label healthy after $((SECONDS - start))s"
            return 0
        fi
        # Distinguish "still compiling" from "died": a dead engine will never
        # answer, and waiting the full 90 min for that is a wasted nightly.
        if [[ -n "$watch_pid" ]] && ! kill -0 "$watch_pid" 2>/dev/null; then
            echo "ERROR: $label exited before becoming healthy (pid $watch_pid gone)" >&2
            return 1
        fi
        if (( SECONDS - start >= timeout )); then
            echo "ERROR: $label never became healthy in ${timeout}s (last=$code)" >&2
            return 1
        fi
        sleep 15
    done
}

# Compile-bucket warmup, once per server. The agentic legs run with
# --warmup-requests 0 (bench_serving would otherwise replay a whole long session
# as its warmup), so without this the first measured leg pays for every
# torch.compile bucket it touches.
#
# Shape matches what the legs actually hit: a full prefill chunk (the contract's
# chunked-prefill size) and a decode batch at the
# sweep's largest concurrency. PROFILE=0 is the *torch profiler* switch in
# bench_serving.sh, unrelated to this benchmark's PROFILE.
warmup_server() {
    local url="$1" log="$2"
    [[ "$SKIP_WARMUP" == "1" ]] && { echo "[run] warmup skipped"; return 0; }
    echo "[run] warmup: covering the 8k-prefill and bs<=$MAX_CONCURRENCY decode buckets"
    MODEL="$MODEL" BASE_URL="$url" \
    DATASET_NAME=random-ids TOKENIZE_PROMPT=1 RANDOM_RANGE_RATIO=1.0 \
    ISL="$CHUNKED_PREFILL_PER_RANK" OSL=32 \
    BATCH_SIZE="$MAX_CONCURRENCY" NUM_PROMPTS="$((MAX_CONCURRENCY * 2))" \
    WARMUP_REQUESTS=0 DUMP_META=0 PROFILE=0 OUTPUT_DIR="$RUN_ROOT/warmup" \
        bash "$TPU_BENCH_DIR/bench_serving.sh" >>"$log" 2>&1 \
        || echo "[run] WARN: warmup pass failed (see $log); continuing" >&2
}

# One (bs x long) sweep over a live endpoint.
run_sweep() {
    local mode="$1" url="$2" out_dir="$3"
    mkdir -p "$out_dir"
    for bs in $SWEEP_BS; do
        for long in $SWEEP_LONG; do
            if (( long > bs )); then
                echo "[run] skip $mode c${bs}-long${long}: more long sessions than slots"
                printf 'skipped:long>bs\n' >"$out_dir/c${bs}-long${long}.status"
                continue
            fi
            echo
            echo "[run] ===== $mode leg c${bs}-long${long} ====="
            BS="$bs" NUM_LONG="$long" BASE_URL="$url" \
            OUTPUT_DIR="$out_dir" TRACE_DIR="$TRACE_DIR" \
                bash "$HERE/bench_leg.sh" || true
        done
    done
}

# ---------------------------------------------------------------------------
# MODE=colocated: one engine, tp=16 dp=4.
# ---------------------------------------------------------------------------
run_colocated() {
    local out_dir="$RUN_ROOT/colocated"
    local log="$out_dir/server.log"
    mkdir -p "$out_dir"
    local url="http://127.0.0.1:$PORT"

    echo "[run] launching colocated engine (tp=$COLOCATED_TP dp=$COLOCATED_DP)"
    ROLE=colocated PORT="$PORT" RUN_DIR="$out_dir" LOG_FILE="$log" \
        bash "$HERE/serve.sh" &
    local pid=$!
    CHILD_PIDS+=("$pid")

    if ! wait_healthy "$url" "colocated engine" "$HEALTH_TIMEOUT" "$pid"; then
        echo "[run] colocated engine failed to start; last 80 log lines:" >&2
        tail -n 80 "$log" >&2 || true
        echo "failed:server" >"$out_dir/MODE_STATUS"
        return 1
    fi
    warmup_server "$url" "$log"
    run_sweep colocated "$url" "$out_dir"
    echo "ok" >"$out_dir/MODE_STATUS"

    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# MODE=pd: prefill engine + decode engine + router, tp=8 dp=2 each.
# ---------------------------------------------------------------------------
# Sets LAUNCHED_PID rather than echoing it: a $(...) capture would run the
# CHILD_PIDS append in a subshell, and the cleanup trap would never learn about
# the engine it started.
LAUNCHED_PID=""
launch_pd_engine() {
    local role="$1" ssh_target="$2" port="$3" log="$4"
    if [[ -z "$ssh_target" ]]; then
        echo "[run] launching $role engine locally on :$port"
        ROLE="$role" RUN_DIR="$(dirname "$log")" LOG_FILE="$log" \
            bash "$HERE/serve.sh" &
    else
        echo "[run] launching $role engine on $ssh_target:$port over ssh"
        # -tt so the remote process gets a controlling tty and dies with the ssh
        # session; without it a SIGTERM here leaves a 480B engine on the remote
        # host holding all 8 chiplets.
        ssh -tt -o BatchMode=yes "$ssh_target" \
            "cd $HERE && PROFILE='$PROFILE' MODEL='$MODEL' ROLE=$role \
             PD_PREFILL_PORT=$PD_PREFILL_PORT PD_DECODE_PORT=$PD_DECODE_PORT \
             PD_BOOTSTRAP_PORT=$PD_BOOTSTRAP_PORT PYTHON='$PYTHON' \
             bash ./serve.sh" >"$log" 2>&1 &
    fi
    LAUNCHED_PID=$!
    CHILD_PIDS+=("$LAUNCHED_PID")
}

run_pd() {
    local out_dir="$RUN_ROOT/pd"
    mkdir -p "$out_dir"
    local prefill_url decode_url prefill_pid="" decode_pid=""

    if [[ -n "${PREFILL_BASE_URL:-}" && -n "${DECODE_BASE_URL:-}" ]]; then
        echo "[run] using pre-started P/D engines"
        prefill_url="$PREFILL_BASE_URL"
        decode_url="$DECODE_BASE_URL"
    else
        local prefill_host="${PD_PREFILL_SSH:-}" decode_host="${PD_DECODE_SSH:-}"
        if [[ -z "$prefill_host" && -z "$decode_host" ]]; then
            # Both engines on one host would need 8 chiplets each on a 8-chiplet
            # host. Refuse rather than fail obscurely inside PjRt init.
            echo "ERROR: MODE=pd needs one TPU host per engine. Set" \
                 "PD_PREFILL_SSH + PD_DECODE_SSH, or PREFILL_BASE_URL +" \
                 "DECODE_BASE_URL for engines you started yourself." >&2
            echo "failed:no-hosts" >"$out_dir/MODE_STATUS"
            return 1
        fi
        launch_pd_engine prefill "$prefill_host" "$PD_PREFILL_PORT" "$out_dir/server-prefill.log"
        prefill_pid=$LAUNCHED_PID
        launch_pd_engine decode "$decode_host" "$PD_DECODE_PORT" "$out_dir/server-decode.log"
        decode_pid=$LAUNCHED_PID
        # Strip any user@ prefix from the ssh target to get the hostname.
        prefill_url="http://${prefill_host:+${prefill_host##*@}}:$PD_PREFILL_PORT"
        decode_url="http://${decode_host:+${decode_host##*@}}:$PD_DECODE_PORT"
        [[ -z "$prefill_host" ]] && prefill_url="http://127.0.0.1:$PD_PREFILL_PORT"
        [[ -z "$decode_host" ]] && decode_url="http://127.0.0.1:$PD_DECODE_PORT"
    fi

    if ! wait_healthy "$prefill_url" "prefill engine" "$HEALTH_TIMEOUT" "$prefill_pid"; then
        tail -n 80 "$out_dir/server-prefill.log" >&2 || true
        echo "failed:prefill" >"$out_dir/MODE_STATUS"; return 1
    fi
    if ! wait_healthy "$decode_url" "decode engine" "$HEALTH_TIMEOUT" "$decode_pid"; then
        tail -n 80 "$out_dir/server-decode.log" >&2 || true
        echo "failed:decode" >"$out_dir/MODE_STATUS"; return 1
    fi

    # Router. The bootstrap port is passed positionally after the prefill URL so
    # the router can tell the decode engine where to pull KV from.
    local router_pid="" router_url
    if [[ -n "${ROUTER_BASE_URL:-}" ]]; then
        echo "[run] using pre-started router at $ROUTER_BASE_URL"
        router_url="$ROUTER_BASE_URL"
    else
        echo "[run] launching P/D router on $ROUTER_HOST:$PD_ROUTER_PORT"
        "$PYTHON" -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill "$prefill_url" "$PD_BOOTSTRAP_PORT" \
            --decode "$decode_url" \
            --host "$ROUTER_HOST" --port "$PD_ROUTER_PORT" \
            >"$out_dir/router.log" 2>&1 &
        router_pid=$!
        CHILD_PIDS+=("$router_pid")
        router_url="http://$ROUTER_HOST:$PD_ROUTER_PORT"
    fi
    if ! wait_healthy "$router_url" "router" 300 "$router_pid"; then
        tail -n 60 "$out_dir/router.log" 2>/dev/null >&2 || true
        echo "failed:router" >"$out_dir/MODE_STATUS"; return 1
    fi

    warmup_server "$router_url" "$out_dir/router.log"
    run_sweep pd "$router_url" "$out_dir"
    echo "ok" >"$out_dir/MODE_STATUS"

    for pid in "$router_pid" "$decode_pid" "$prefill_pid"; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done
}

# Sweep an endpoint somebody else brought up. Used by the CI test (the JobSet owns
# the server, so the driver must not launch anything) and by anyone benchmarking a
# server they started by hand.
run_external() {
    local out_dir="$RUN_ROOT/${EXTERNAL_MODE_LABEL:-colocated}"
    mkdir -p "$out_dir"
    local url="${BASE_URL:?MODE=external needs BASE_URL=<server>}"
    if ! wait_healthy "$url" "external server" "$HEALTH_TIMEOUT" ""; then
        echo "failed:server" >"$out_dir/MODE_STATUS"
        return 1
    fi
    warmup_server "$url" "$out_dir/warmup.log"
    run_sweep "${EXTERNAL_MODE_LABEL:-colocated}" "$url" "$out_dir"
    echo "ok" >"$out_dir/MODE_STATUS"
}

# ---------------------------------------------------------------------------
mode_rc=0
case "$MODE" in
    colocated) run_colocated || mode_rc=1 ;;
    external)  run_external || mode_rc=1 ;;
    pd)
        # P/D is disabled: the TPU CI's multi-host primitive runs ONE recipe on every
        # host (a single global TP group), so it cannot bring up a prefill engine and
        # a decode engine as separate roles -- jobset-tpu.yaml says as much ("true
        # per-group gating for P/D is future work... which MULTIHOST_LAUNCH does not
        # model yet"). The engines and the router below still work when you launch
        # them yourself, so this is a guard, not a removal.
        if [[ "${ALLOW_PD:-0}" != "1" ]]; then
            echo "ERROR: MODE=pd is disabled. The TPU CI cannot launch prefill and" >&2
            echo "       decode as separate roles yet (see the note in this script)." >&2
            echo "       Set ALLOW_PD=1 to run it by hand on a slice you own, with" >&2
            echo "       PD_PREFILL_SSH/PD_DECODE_SSH or pre-started engine URLs." >&2
            exit 2
        fi
        echo "[run] WARNING: MODE=pd via ALLOW_PD=1 -- not a CI-supported path" >&2
        run_pd || mode_rc=1
        ;;
    both)
        echo "ERROR: MODE=both is disabled while P/D is (see MODE=pd below)." >&2
        echo "       Use MODE=colocated." >&2
        exit 2
        ;;
    *)
        echo "ERROR: MODE must be colocated|external (pd needs ALLOW_PD=1)," \
             "got '$MODE'" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
echo
echo "[run] summarizing $RUN_ROOT"
summarize_rc=0
"$PYTHON" "$HERE/summarize.py" \
    --run-root "$RUN_ROOT" \
    --json-out "$RUN_ROOT/summary.json" \
    --md-out "$RUN_ROOT/summary.md" \
    --profile "$PROFILE" \
    --mode "$MODE" || summarize_rc=$?

echo
[[ -f "$RUN_ROOT/summary.md" ]] && cat "$RUN_ROOT/summary.md"
echo
echo "[run] results: $RUN_ROOT/summary.json"

if (( summarize_rc != 0 )); then
    echo "[run] FAILED: no leg produced metrics" >&2
    exit 1
fi
if [[ "$STRICT" == "1" ]] && (( mode_rc != 0 )); then
    echo "[run] FAILED: STRICT=1 and at least one mode did not complete" >&2
    exit 1
fi
exit 0
