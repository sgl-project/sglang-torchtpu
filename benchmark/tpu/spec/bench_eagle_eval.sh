#!/bin/bash
# End-to-end accuracy + speedup for EAGLE3 speculative decoding on TPU.
#
# Launches a baseline server (no spec decode) and/or an EAGLE3 server, runs an
# accuracy benchmark against each, and reports accuracy (and, for GSM8K, output
# throughput so the spec-decode speedup is visible). Greedy EAGLE3 spec decode
# is lossless, so the accuracy of the two phases should match.
#
# GSM8K (default) is the better spec-decode workload: its long chain-of-thought
# generations actually exercise the draft/verify loop. MMLU emits ~1-token
# answers, so it validates accuracy but barely engages spec decode.
#
# Quick smoke:        NUM_QUESTIONS=20 ./bench_eagle_eval.sh
# Full GSM8K:         NUM_QUESTIONS=1319 ./bench_eagle_eval.sh
# MMLU instead:       BENCHMARK=mmlu ./bench_eagle_eval.sh
# EAGLE3 only:        SKIP_BASELINE=1 ./bench_eagle_eval.sh
# With torch.compile: COMPILE=1 COMPILE_BS="1 8" ./bench_eagle_eval.sh
#
# Environment variables:
#   BENCHMARK         gsm8k | mmlu (default: gsm8k)
#   MODEL             Target model (default: Qwen/Qwen3-1.7B)
#   DRAFT_MODEL       Draft model  (default: AngelSlim/Qwen3-1.7B_eagle3)
#   TP_SIZE           Tensor parallel size (default: 4)
#   PORT              Server port (default: 30000)
#   PARALLEL          Concurrent requests (default: 8; >1 exercises bs>1)
#   CONTEXT_LEN       Context length (default: 4096)
#   CHUNKED_PREFILL   Split prefill so >bucket prompts fit (default: 2048)
#   COMPILE           1 to enable torch.compile (default: 0 = eager)
#   COMPILE_BS        Compile batch buckets (default: "1 8")
#   COMPILE_TOKENS    Compile token buckets (default: "1024 2048")
#   PAGE_SIZE         Override page_size (default: platform default; spec forces 1)
#   SPEC_STEPS/SPEC_TOPK/SPEC_DRAFT_TOKENS   EAGLE3 params (default 3/1/4)
#   SKIP_BASELINE     1 to skip the no-spec phase
#   SKIP_EAGLE        1 to skip the EAGLE3 phase
#   SERVER_TIMEOUT    Max seconds to wait for readiness (default: 1800)
#  GSM8K-specific:
#   NUM_QUESTIONS     (default: 200)   NUM_SHOTS (default: 5)
#   MAX_NEW_TOKENS    (default: 512)
#  MMLU-specific:
#   NSUB              subjects 1..60 (default: 60)   NTRAIN few-shot (default: 5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BENCHMARK="${BENCHMARK:-gsm8k}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-30000}"
PARALLEL="${PARALLEL:-8}"
CONTEXT_LEN="${CONTEXT_LEN:-4096}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
COMPILE="${COMPILE:-0}"
COMPILE_BS="${COMPILE_BS:-1 8}"
COMPILE_TOKENS="${COMPILE_TOKENS:-1024 2048}"
PAGE_SIZE="${PAGE_SIZE:-}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_EAGLE="${SKIP_EAGLE:-0}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
# gsm8k
NUM_QUESTIONS="${NUM_QUESTIONS:-200}"
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# mmlu
NSUB="${NSUB:-60}"
NTRAIN="${NTRAIN:-5}"

case "$BENCHMARK" in
    gsm8k|mmlu) ;;
    *) echo "ERROR: BENCHMARK must be gsm8k or mmlu, got '$BENCHMARK'"; exit 2 ;;
esac

export SGLANG_PLATFORM=tpu
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS 2>/dev/null || true

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=32768 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_enable_latency_hiding_scheduler=true}"

RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cleanup() {
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

wait_for_server() {
    echo "Waiting for server (max ${SERVER_TIMEOUT}s)..."
    for i in $(seq 1 $((SERVER_TIMEOUT / 10))); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process exited during startup"
            return 1
        fi
        code=$(curl -sS -o /dev/null -m 5 -w "%{http_code}" \
            "http://127.0.0.1:${PORT}/health_generate" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "Server ready after ~$((i * 10))s"
            return 0
        fi
        sleep 10
    done
    echo "ERROR: server not ready after ${SERVER_TIMEOUT}s"
    return 1
}

# Runs the selected benchmark and writes "<accuracy>\t<metric>" to a .acc file.
run_eval() {
    local label="$1"
    local log="$RESULT_DIR/${TIMESTAMP}_${BENCHMARK}_${label}.log"
    local acc metric
    echo ""
    if [ "$BENCHMARK" = "gsm8k" ]; then
        echo "Running GSM8K ($label): num_questions=$NUM_QUESTIONS shots=$NUM_SHOTS max_new=$MAX_NEW_TOKENS parallel=$PARALLEL"
        python3 "$REPO_ROOT/benchmark/gsm8k/bench_sglang.py" \
            --num-questions "$NUM_QUESTIONS" --num-shots "$NUM_SHOTS" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --host 127.0.0.1 --port "$PORT" --parallel "$PARALLEL" \
            --backend srt 2>&1 | tee "$log" || true
        acc=$(grep -oE "Accuracy: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || true)
        metric=$(grep -oE "Output throughput: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+" | tail -1 || true)
    else
        echo "Running MMLU ($label): nsub=$NSUB ntrain=$NTRAIN parallel=$PARALLEL"
        python3 "$REPO_ROOT/benchmark/mmlu/bench_sglang.py" \
            --ntrain "$NTRAIN" --nsub "$NSUB" \
            --host 127.0.0.1 --port "$PORT" --parallel "$PARALLEL" \
            --backend srt 2>&1 | tee "$log" || true
        acc=$(grep -oE "Average accuracy: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || true)
        metric=$(grep -oE "Total latency: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+" | tail -1 || true)
    fi
    printf '%s\t%s\n' "${acc:-NA}" "${metric:-NA}" > "$RESULT_DIR/${TIMESTAMP}_${label}.acc"
    echo "  -> $label accuracy: ${acc:-FAILED}  metric: ${metric:-NA}"
}

# Common server args
BASE_ARGS=(
    --model-path "$MODEL"
    --context-length "$CONTEXT_LEN"
    --device tpu
    --tp-size "$TP_SIZE"
    --dtype bfloat16
    --disable-overlap-schedule
    --decode-log-interval 20
    --host 127.0.0.1
    --port "$PORT"
    --watchdog-timeout 3600
)
[ -n "$PAGE_SIZE" ] && BASE_ARGS+=(--page-size "$PAGE_SIZE")

if [ "$COMPILE" = "1" ]; then
    # Under torch.compile, prefill only runs at the --tpu-compile-tokens buckets,
    # so a prompt longer than the largest bucket has no compiled shape. Chunk the
    # prefill to <= the max bucket so long prompts (e.g. MMLU 5-shot ~2.3k tokens)
    # fit. Eager needs no chunking -- it prefills at the prompt's real length.
    BASE_ARGS+=(
        --enable-torch-compile
        --tpu-compile-bs $COMPILE_BS
        --tpu-compile-tokens $COMPILE_TOKENS
        --chunked-prefill-size "$CHUNKED_PREFILL"
    )
else
    BASE_ARGS+=(--skip-server-warmup)
fi

if [ "$BENCHMARK" = "gsm8k" ]; then
    metric_label="throughput (tok/s)"
else
    metric_label="latency (s)"
fi

echo "============================================"
echo "  EAGLE3 ${BENCHMARK^^} e2e"
echo "============================================"
echo "Model:        $MODEL"
echo "Draft:        $DRAFT_MODEL"
echo "TP / Port:    $TP_SIZE / $PORT"
if [ "$BENCHMARK" = "gsm8k" ]; then
    echo "GSM8K:        num_questions=$NUM_QUESTIONS shots=$NUM_SHOTS max_new=$MAX_NEW_TOKENS parallel=$PARALLEL"
else
    echo "MMLU:         nsub=$NSUB ntrain=$NTRAIN parallel=$PARALLEL"
fi
echo "Context len:  $CONTEXT_LEN"
if [ "$COMPILE" = "1" ]; then
    echo "Compile:      1  (bs='$COMPILE_BS' tokens='$COMPILE_TOKENS' chunked-prefill=$CHUNKED_PREFILL)"
else
    echo "Compile:      0  (eager)"
fi
echo "============================================"

# ---- Baseline (no spec decode) ----
if [ "$SKIP_BASELINE" != "1" ]; then
    echo ""
    echo "==== Phase 1: Baseline (no speculative decoding) ===="
    cleanup
    python -u -m sglang.launch_server "${BASE_ARGS[@]}" &
    SERVER_PID=$!
    if wait_for_server; then
        run_eval "baseline"
    else
        echo "Baseline server failed to start"
    fi
    cleanup
fi

# ---- EAGLE3 ----
if [ "$SKIP_EAGLE" != "1" ]; then
    echo ""
    echo "==== Phase 2: EAGLE3 speculative decoding ===="
    cleanup
    EAGLE_ARGS=(
        "${BASE_ARGS[@]}"
        --speculative-algorithm EAGLE3
        --speculative-draft-model-path "$DRAFT_MODEL"
        --speculative-num-steps "$SPEC_STEPS"
        --speculative-eagle-topk "$SPEC_TOPK"
        --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"
    )
    python -u -m sglang.launch_server "${EAGLE_ARGS[@]}" &
    SERVER_PID=$!
    if wait_for_server; then
        run_eval "eagle3"
    else
        echo "EAGLE3 server failed to start"
    fi
    cleanup
fi

echo ""
echo "============================================"
echo "  ${BENCHMARK^^} Summary"
echo "============================================"
b_acc=NA; b_m=NA; e_acc=NA; e_m=NA
if [ -f "$RESULT_DIR/${TIMESTAMP}_baseline.acc" ]; then
    b_acc=$(cut -f1 "$RESULT_DIR/${TIMESTAMP}_baseline.acc"); b_m=$(cut -f2 "$RESULT_DIR/${TIMESTAMP}_baseline.acc")
fi
if [ -f "$RESULT_DIR/${TIMESTAMP}_eagle3.acc" ]; then
    e_acc=$(cut -f1 "$RESULT_DIR/${TIMESTAMP}_eagle3.acc"); e_m=$(cut -f2 "$RESULT_DIR/${TIMESTAMP}_eagle3.acc")
fi
printf '  %-20s acc=%-7s %s=%s\n' "Baseline (no spec):" "$b_acc" "$metric_label" "$b_m"
printf '  %-20s acc=%-7s %s=%s\n' "EAGLE3 spec:" "$e_acc" "$metric_label" "$e_m"
if [ "$b_acc" != "NA" ] && [ "$e_acc" != "NA" ]; then
    [ "$b_acc" = "$e_acc" ] && echo "  -> LOSSLESS: identical accuracy" \
        || echo "  -> DIFF: $b_acc vs $e_acc (expected identical for greedy spec decode)"
    if [ "$BENCHMARK" = "gsm8k" ] && [ "$b_m" != "NA" ] && [ "$e_m" != "NA" ]; then
        awk -v b="$b_m" -v e="$e_m" 'BEGIN{ if (b>0) printf "  -> EAGLE3 throughput speedup: %.2fx\n", e/b }'
    fi
fi
echo "  logs: $RESULT_DIR/${TIMESTAMP}_${BENCHMARK}_*.log"
echo "============================================"
