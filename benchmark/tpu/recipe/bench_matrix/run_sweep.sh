#!/usr/bin/env bash
# Sweep driver for the TPU benchmark matrix. Live-server only: the
# CI JobSet owns the server (serve_group.sh), so this sweeps BASE_URL
# across (SWEEP_BS x SWEEP_LONG), then writes one summary.json + summary.md
# via the 480B nightly's summarize.py.
#
#   BENCH_MODEL=gpt_oss_120b BENCH_RECIPE=1host_pc WORKLOAD=8k1k \
#   BASE_URL=http://127.0.0.1:30000 RUN_ROOT=/tmp/out ./run_sweep.sh
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"

BASE_URL=${BASE_URL:?set BASE_URL (the live endpoint to sweep)}
TS=$(date -u +%Y%m%d_%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-$HERE/results/$TS}
mkdir -p "$RUN_ROOT"
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd)
mkdir -p "$TRACE_CACHE_DIR"

# summarize.py collects legs from the "colocated" and "pd" subdirs.
if [[ "$PD_DISAGG" == "1" ]]; then
    MODE_LABEL=pd
    # The mini-LB serves no /metrics; the cache/offload counters live on the
    # prefill engine, the spec/abort side on the decode engine. Engine leaders
    # follow PHYSICAL host order (serve_group.sh's rendezvous writes the map
    # for this pod), not pod-index order.
    if [[ -s /tmp/bench_matrix_phys_hosts ]]; then
        mapfile -t _slice_hosts < /tmp/bench_matrix_phys_hosts
    else
        IFS=',' read -r -a _slice_hosts <<< "${TPU_WORKER_HOSTNAMES:-}"
    fi
    if [[ -z "${PROBE_URL:-}" && ${#_slice_hosts[@]} -ge 1 ]]; then
        PROBE_URL="http://${_slice_hosts[0]}:$PD_PREFILL_PORT"
    fi
    PROBE_URL=${PROBE_URL:-http://127.0.0.1:$PD_PREFILL_PORT}
    # The mini-LB pairs each request with a random prefill engine, so a leg's
    # cache counters are the sum over all of them; probing only the first lets
    # a low-concurrency leg read 0% while its session hits on another engine.
    if [[ -z "${PROBE_URLS:-}" ]]; then
        PROBE_URLS=$PROBE_URL
        for (( i = 1; i < PD_PREFILL_ENGINES && i < ${#_slice_hosts[@]}; i++ )); do
            PROBE_URLS+=" http://${_slice_hosts[$i]}:$PD_PREFILL_PORT"
        done
    fi
    if [[ -z "${PROBE_URL_2:-}" && ${#_slice_hosts[@]} -ge 3 ]]; then
        PROBE_URL_2="http://${_slice_hosts[2]}:$PD_DECODE_PORT"
    fi
else
    MODE_LABEL=colocated
    PROBE_URL=${PROBE_URL:-$BASE_URL}
fi
OUT_DIR="$RUN_ROOT/$MODE_LABEL"
mkdir -p "$OUT_DIR"

log_config | tee "$RUN_ROOT/config.log"
{
    echo "run_root  = $RUN_ROOT"
    echo "base_url  = $BASE_URL   probe_url = ${PROBE_URLS:-$PROBE_URL}"
    echo "host      = $(hostname)"
    echo "started   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git       = $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} | tee -a "$RUN_ROOT/config.log"

echo "[sweep] waiting for $BASE_URL"
start=$SECONDS
while :; do
    code=$(curl -sS -o /dev/null -m "$CURL_MAX_TIME" -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo 000)
    [[ "$code" == "200" ]] && break
    if (( SECONDS - start >= HEALTH_TIMEOUT )); then
        echo "[sweep] ERROR: $BASE_URL never healthy (last=$code)" >&2
        echo "failed:server" >"$OUT_DIR/MODE_STATUS"
        exit 1
    fi
    sleep "$HEALTH_POLL_INTERVAL"
done

for bs in $SWEEP_BS; do
    for long in $SWEEP_LONG; do
        if (( long > bs )); then
            echo "[sweep] skip c${bs}-long${long}: more long requests than slots"
            printf 'skipped:long>bs\n' >"$OUT_DIR/c${bs}-long${long}.status"
            continue
        fi
        echo
        echo "[sweep] ===== leg c${bs}-long${long} ====="
        BS="$bs" NUM_LONG="$long" BASE_URL="$BASE_URL" \
        PROBE_URL="$PROBE_URL" PROBE_URLS="${PROBE_URLS:-$PROBE_URL}" \
        PROBE_URL_2="${PROBE_URL_2:-}" \
        OUTPUT_DIR="$OUT_DIR" TRACE_DIR="$TRACE_CACHE_DIR" \
            bash "$HERE/bench_leg.sh" || true
    done
done
echo "ok" >"$OUT_DIR/MODE_STATUS"

echo
echo "[sweep] summarizing $RUN_ROOT"
summarize_rc=0
"$PYTHON" "$NIGHTLY_TOOLS_DIR/summarize.py" \
    --run-root "$RUN_ROOT" \
    --json-out "$RUN_ROOT/summary.json" \
    --md-out "$RUN_ROOT/summary.md" \
    --profile "$PROFILE" \
    --mode external \
    --benchmark-label "${BENCH_MODEL}-${BENCH_RECIPE}-${WORKLOAD}" \
    --title "$BENCH_MODEL — TPU benchmark matrix ($BENCH_RECIPE, $WORKLOAD)" \
    || summarize_rc=$?

[[ -f "$RUN_ROOT/summary.md" ]] && cat "$RUN_ROOT/summary.md"
echo "[sweep] results: $RUN_ROOT/summary.json"

if (( summarize_rc != 0 )); then
    echo "[sweep] FAILED: no leg produced metrics" >&2
    exit 1
fi
exit 0
