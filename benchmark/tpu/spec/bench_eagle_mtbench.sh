#!/bin/bash
# Benchmark EAGLE3 speculative decoding on TPU using MT-Bench.
#
# Usage:
#   # 1. Launch the server first (see launch_server.sh or use:)
#   SGLANG_PLATFORM=tpu python -m sglang.launch_server \
#       --model-path meta-llama/Llama-3.1-8B-Instruct \
#       --speculative-algorithm EAGLE3 \
#       --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge \
#       --speculative-num-steps 3 --speculative-eagle-topk 1 \
#       --speculative-num-draft-tokens 4 \
#       --device tpu --tp-size 4 --dtype bfloat16 \
#       --disable-overlap-schedule --decode-log-interval 1 \
#       --skip-server-warmup \
#       --host 127.0.0.1 --port 30000 --watchdog-timeout 3600
#
#   # 2. Run this script:
#   ./bench_eagle_mtbench.sh
#
# Environment variables:
#   HOST            Server host (default: 127.0.0.1)
#   PORT            Server port (default: 30000)
#   NUM_QUESTIONS   Number of MT-Bench questions (default: 80)
#   PARALLEL        Concurrent requests (default: 1)
#   RESULT_FILE     Output file for results (default: bench_eagle_mtbench_result.jsonl)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
NUM_QUESTIONS="${NUM_QUESTIONS:-10}"
PARALLEL="${PARALLEL:-1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
RESULT_FILE="${RESULT_FILE:-$SCRIPT_DIR/results/bench_eagle_mtbench_result.jsonl}"

mkdir -p "$(dirname "$RESULT_FILE")"

echo "=== EAGLE3 MT-Bench Benchmark ==="
echo "Server:         http://${HOST}:${PORT}"
echo "Questions:      ${NUM_QUESTIONS}"
echo "Parallel:       ${PARALLEL}"
echo "Result file:    ${RESULT_FILE}"
echo ""

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    code=$(curl -sS -o /dev/null -m 5 -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Server not ready after 600s"
        exit 1
    fi
    sleep 10
done

# Warmup rounds: trigger XLA compilation for all shapes, discard results.
for round in $(seq 1 "$WARMUP_ROUNDS"); do
    echo ""
    echo "Warmup round $round/$WARMUP_ROUNDS..."
    python3 "$REPO_ROOT/benchmark/mtbench/bench_sglang_eagle.py" \
        --num-questions "$NUM_QUESTIONS" \
        --parallel "$PARALLEL" \
        --host "$HOST" \
        --port "$PORT" \
        --backend srt \
        --result-file "/dev/null"
done

echo ""
echo "Benchmark run..."
python3 "$REPO_ROOT/benchmark/mtbench/bench_sglang_eagle.py" \
    --num-questions "$NUM_QUESTIONS" \
    --parallel "$PARALLEL" \
    --host "$HOST" \
    --port "$PORT" \
    --backend srt \
    --result-file "$RESULT_FILE"

echo ""
echo "Results written to: $RESULT_FILE"
echo "=== Done ==="
