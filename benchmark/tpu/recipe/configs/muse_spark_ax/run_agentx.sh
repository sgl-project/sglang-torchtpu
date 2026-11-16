#!/usr/bin/env bash
# AgentX sweep for Muse Spark on one tpu7x host. For each concurrency: boot the
# agentx.py profile, replay the AgentX traces against it with AIPerf, summarize,
# stop the server. README.md in this directory explains the setup and the output.
#
#   benchmark/tpu/recipe/configs/muse_spark_ax/run_agentx.sh          # conc 1 7 9 10 16 22
#   CONCS="16 22" benchmark/tpu/recipe/configs/muse_spark_ax/run_agentx.sh
#   PRINT_ONLY=1 benchmark/tpu/recipe/configs/muse_spark_ax/run_agentx.sh   # render only
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
CONCS=${CONCS:-"1 7 9 10 16 22"}
PORT=${PORT:-30000}
# Seconds of profiling per point; AgentX requires at least 900. FAST=1 is AIPerf's
# short mode: 20 minutes, one warmup request per lane.
DURATION=${DURATION:-3600}
WARMUP_REQUESTS_PER_LANE=${WARMUP_REQUESTS_PER_LANE:-10}
if [[ "${FAST:-0}" == 1 ]]; then
    DURATION=1200
    WARMUP_REQUESTS_PER_LANE=1
fi
RUN_DIR=${RUN_DIR:-$REPO/agentx_runs/$(date -u +%Y%m%d_%H%M%SZ)}
# Weights and tokenizer: the Hub id (from the HF cache) or a local snapshot directory.
MODEL=${MODEL:-meta-models/Muse-Spark-Open-816B-A42B-NVFP4}
SERVED_MODEL_NAME=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
export MUSE_WINDOWED_DRAFT_DIR=${MUSE_WINDOWED_DRAFT_DIR:-$HOME/.cache/sglang/muse-spark-open-dflash2-windowed}
# The server reads weights from the local cache only, as the submitted runs did.
SERVER_HF_OFFLINE=${SERVER_HF_OFFLINE:-1}
CHIPS=${CHIPS:-4}
# MK_PREFILL=1: the megakernel prefill program (agentx.py --mk-prefill).
MK_PREFILL_FLAG=$([[ "${MK_PREFILL:-0}" == 1 ]] && echo --mk-prefill || true)
# MK_DECODE=1: only the decode megakernel's Waves 1-4 and expert grouping (agentx.py --mk-decode).
MK_DECODE_FLAG=$([[ "${MK_DECODE:-0}" == 1 ]] && echo --mk-decode || true)
# AIPerf at the revision the submission's harness pinned, in its own venv: it must
# not share site-packages with the server.
AIPERF_REPO=https://github.com/SemiAnalysisAI/aiperf.git
AIPERF_COMMIT=754356e9a39acc6cc6afb242d123bb57c3fb6f75
AIPERF_HOME=${AIPERF_HOME:-$HOME/.cache/agentx-aiperf}
AIPERF_VENV=$AIPERF_HOME/venv-${AIPERF_COMMIT:0:12}
TRACES=semianalysis_cc_traces_weka_062126
TRACES_DATASET=semianalysisai/cc-traces-weka-062126
PYTHON=${PYTHON:-python3}

render() {  # conc out_dir: the launcher config and its resolved command
    local conc=$1 out=$2
    mkdir -p "$out"
    (cd "$REPO" && HF_HUB_OFFLINE=$SERVER_HF_OFFLINE TRANSFORMERS_OFFLINE=$SERVER_HF_OFFLINE \
        "$PYTHON" -m benchmark.tpu.recipe.configs.muse_spark_ax.agentx --conc "$conc" \
        --draft-model-path "$MUSE_WINDOWED_DRAFT_DIR" ${MK_PREFILL_FLAG} ${MK_DECODE_FLAG} > "$out/server_config.json")
    # SERVER_ARGS_PATCH: JSON merged into server_args, e.g. '{"tp_size": 16, "nnodes": 2}'.
    if [ -n "${SERVER_ARGS_PATCH:-}" ]; then
        "$PYTHON" - "$out/server_config.json" "$SERVER_ARGS_PATCH" <<'PY'
import json, sys
path, patch = sys.argv[1], json.loads(sys.argv[2])
cfg = json.load(open(path))
cfg["server_args"].update(patch)
json.dump(cfg, open(path, "w"), indent=2)
PY
    fi
    # SERVER_ARGS_DROP: server_args keys to remove, e.g. the HiCache and session-cache flags.
    if [ -n "${SERVER_ARGS_DROP:-}" ]; then
        "$PYTHON" - "$out/server_config.json" $SERVER_ARGS_DROP <<'PY'
import json, sys
path, keys = sys.argv[1], sys.argv[2:]
cfg = json.load(open(path))
for k in keys:
    cfg["server_args"].pop(k, None)
json.dump(cfg, open(path, "w"), indent=2)
PY
    fi
    # RUNTIME_ENV_PATCH: JSON merged into runtime_env, e.g. '{"SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed"}'.
    if [ -n "${RUNTIME_ENV_PATCH:-}" ]; then
        "$PYTHON" - "$out/server_config.json" "$RUNTIME_ENV_PATCH" <<'PY'
import json, sys
path, patch = sys.argv[1], json.loads(sys.argv[2])
cfg = json.load(open(path))
cfg["runtime_env"].update(patch)
json.dump(cfg, open(path, "w"), indent=2)
PY
    fi
    # RUNTIME_ENV_DROP: runtime_env keys to remove, e.g. "FORCE_SINGLE_HOST LOCAL_SIZE" for a
    # multi-host engine (the one-host base config pins both).
    if [ -n "${RUNTIME_ENV_DROP:-}" ]; then
        "$PYTHON" - "$out/server_config.json" $RUNTIME_ENV_DROP <<'PY'
import json, sys
path, keys = sys.argv[1], sys.argv[2:]
cfg = json.load(open(path))
for k in keys:
    cfg["runtime_env"].pop(k, None)
json.dump(cfg, open(path, "w"), indent=2)
PY
    fi
    (cd "$REPO" && HF_HUB_OFFLINE=$SERVER_HF_OFFLINE TRANSFORMERS_OFFLINE=$SERVER_HF_OFFLINE \
        MODEL="$MODEL" PORT="$PORT" \
        "$PYTHON" -m benchmark.tpu.launch_server --config "@$out/server_config.json" --dry-run \
        > "$out/launch_plan.json")
}

aiperf_command() {  # conc out_dir: the replay, as the AgentX harness builds it
    local conc=$1 out=$2
    AIPERF_CMD=(
        "$AIPERF_VENV/bin/aiperf" profile --scenario inferencex-agentx-mvp
        --url "http://localhost:$PORT" --endpoint /v1/chat/completions --endpoint-type chat
        --streaming --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL"
        --concurrency "$conc" --benchmark-duration "$DURATION" --stats-interval 30
        --random-seed 42 --failed-request-threshold 0.10
        --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
        --warmup-requests-per-lane "$WARMUP_REQUESTS_PER_LANE"
        --trace-idle-gap-cap-seconds 300 --warmup-grace-period 3600
        --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code
        --num-dataset-entries 393 --slice-duration 1.0
        --output-artifact-dir "$out/aiperf_artifacts"
    )
    # Shorter than the scenario's minimum: allowed, but flagged submission_valid=false.
    if [ "$DURATION" -lt 900 ]; then AIPERF_CMD+=(--unsafe-override); fi
    AIPERF_CMD+=(--public-dataset "$TRACES" --server-metrics "http://localhost:$PORT/metrics")
}

# Recorded assistant turns build later prompts (live responses are discarded);
# dataset setup can take many minutes; the rolling metrics print every 30 s; and
# warmup can leave bytes unacknowledged past the 30 s TCP default.
AIPERF_ENV=(
    AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
    AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
    AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
    AIPERF_UI_REALTIME_METRICS_ENABLED=true
    AIPERF_HTTP_TCP_USER_TIMEOUT=900000
)

install_aiperf() {
    if [ -x "$AIPERF_VENV/bin/aiperf" ]; then return; fi
    command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
    local src=$AIPERF_HOME/aiperf-${AIPERF_COMMIT:0:12}
    if [ ! -f "$src/pyproject.toml" ]; then
        git init -q "$src"
        git -C "$src" fetch -q --depth 1 "$AIPERF_REPO" "$AIPERF_COMMIT"
        git -C "$src" checkout -q FETCH_HEAD
    fi
    uv venv -q --python 3.11 "$AIPERF_VENV"
    # The harness's client packages, unpinned as there.
    uv pip install -q --python "$AIPERF_VENV/bin/python" -e "$src" \
        "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests \
        "numpy>=1.24" "pandas>=2.0.0" "aiohttp>=3.10" "transformers>=4.46" \
        "xlsxwriter>=3.2.1" "tqdm>=4.66" tiktoken matplotlib
}

SERVER_PID=
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID"
        wait "$SERVER_PID" || true
    fi
    SERVER_PID=
}
trap stop_server EXIT

# SYNC_DIR (multi-host): every host marks itself ready for a point and waits for SYNC_HOSTS
# marks before launching, so no host's TPU session start times out on a slower peer.
sync_hosts() {  # conc
    [ -n "${SYNC_DIR:-}" ] || return 0
    mkdir -p "$SYNC_DIR"
    touch "$SYNC_DIR/ready.c$1.h${TPU_WORKER_ID:-0}"
    echo "[agentx] waiting for ${SYNC_HOSTS:-2} hosts at c$1"
    until [ "$(ls "$SYNC_DIR" | grep -c "^ready\.c$1\.h")" -ge "${SYNC_HOSTS:-2}" ]; do sleep 5; done
    echo "[agentx] all hosts ready at c$1"
}

boot() {  # out_dir: start the server and wait for /health
    local out=$1
    (cd "$REPO" && exec env HF_HUB_OFFLINE=$SERVER_HF_OFFLINE TRANSFORMERS_OFFLINE=$SERVER_HF_OFFLINE \
        MODEL="$MODEL" PORT="$PORT" HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-7200}" \
        "$PYTHON" -m benchmark.tpu.launch_server --config "@$out/server_config.json") \
        > "$out/server.log" 2>&1 &
    SERVER_PID=$!
    echo "[agentx] booting (log: $out/server.log)"
    local deadline=$(( $(date +%s) + ${BOOT_TIMEOUT:-3600} ))
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[agentx] server exited; see $out/server.log" >&2; exit 1; }
        # a multi-host peer whose server died will never let this one come up
        if [ -n "${SYNC_DIR:-}" ] && ls "$SYNC_DIR" 2>/dev/null | grep -q "^failed\."; then
            echo "[agentx] a peer host's server exited; stopping" >&2; exit 1
        fi
        [ "$(date +%s)" -lt "$deadline" ] || { echo "[agentx] no /health after ${BOOT_TIMEOUT:-3600}s" >&2; exit 1; }
        sleep 10
    done
}

mkdir -p "$RUN_DIR"
echo "[agentx] run dir $RUN_DIR"
if [[ "${PRINT_ONLY:-0}" == 1 ]]; then
    for conc in $CONCS; do
        out=$RUN_DIR/c$conc
        render "$conc" "$out"
        aiperf_command "$conc" "$out"
        printf '%s ' "${AIPERF_ENV[@]}" "${AIPERF_CMD[@]}" > "$out/benchmark_command.txt"
        echo "[agentx] conc $conc: $out/launch_plan.json, $out/benchmark_command.txt"
    done
    exit 0
fi

(cd "$REPO" && "$PYTHON" -m benchmark.tpu.recipe.configs.muse_spark_ax.windowed_draft \
    --draft "${DRAFT_SRC:-RadixArk/Muse-Spark-Open-DFlash2}" --out "$MUSE_WINDOWED_DRAFT_DIR")

# ROLE=follower (a nonzero host of a multi-host engine): serve each point's config and
# leave the replay to host 0; a point ends when host 0's /health goes away for good.
if [[ "${ROLE:-leader}" == follower ]]; then
    : "${LEADER_HOST:?ROLE=follower needs LEADER_HOST}"
    for conc in $CONCS; do
        out=$RUN_DIR/c$conc
        render "$conc" "$out"
        sync_hosts "$conc"
        (cd "$REPO" && exec env HF_HUB_OFFLINE=$SERVER_HF_OFFLINE TRANSFORMERS_OFFLINE=$SERVER_HF_OFFLINE \
            MODEL="$MODEL" PORT="$PORT" HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-7200}" \
            "$PYTHON" -m benchmark.tpu.launch_server --config "@$out/server_config.json") \
            > "$out/server.log" 2>&1 &
        SERVER_PID=$!
        seen=0; gone=0
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            if curl -sf "http://$LEADER_HOST:$PORT/health" >/dev/null; then
                seen=1; gone=0
            elif [ "$seen" = 1 ]; then
                gone=$((gone + 1))
                [ "$gone" -ge 30 ] && break
            fi
            sleep 10
        done
        if [ "$seen" = 0 ] && [ -n "${SYNC_DIR:-}" ]; then
            touch "$SYNC_DIR/failed.c$conc.h${TPU_WORKER_ID:-0}"
        fi
        echo "[agentx] follower conc $conc: leader done (seen=$seen); stopping"
        stop_server
    done
    exit 0
fi
install_aiperf
# Fetch the traces once into the HF cache; AIPerf reads them from there.
"$AIPERF_VENV/bin/hf" download --repo-type dataset "$TRACES_DATASET" >/dev/null

for conc in $CONCS; do
    out=$RUN_DIR/c$conc
    render "$conc" "$out"
    sync_hosts "$conc"
    boot "$out"
    aiperf_command "$conc" "$out"
    printf '%s ' "${AIPERF_ENV[@]}" "${AIPERF_CMD[@]}" > "$out/benchmark_command.txt"
    echo "[agentx] conc $conc: replaying for ${DURATION}s after warmup"
    set +e
    env "${AIPERF_ENV[@]}" "${AIPERF_CMD[@]}" > "$out/benchmark.log" 2>&1 &
    AIPERF_PID=$!
    # a replay against a dead server only burns the slot: stop it when the server exits
    while kill -0 "$AIPERF_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[agentx] conc $conc: server exited during the replay; stopping AIPerf" >&2
            kill -TERM "$AIPERF_PID" 2>/dev/null
            sleep 30
            kill -KILL "$AIPERF_PID" 2>/dev/null
            break
        fi
        sleep 10
    done
    wait "$AIPERF_PID"
    rc=$?
    set -e
    stop_server
    [ "$rc" -eq 0 ] || echo "[agentx] conc $conc: AIPerf exited $rc; see $out/benchmark.log" >&2
    "$PYTHON" "$HERE/agentx_summary.py" "$out/aiperf_artifacts" --chips "$CHIPS" \
        --concurrency "$conc" | tee "$out/summary.txt" || true
done
