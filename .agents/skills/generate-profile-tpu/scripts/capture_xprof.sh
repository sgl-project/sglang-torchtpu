#!/usr/bin/env bash
# Take two xprof captures of a running SGLang TPU server: one while a wave of
# CONCURRENCY requests is being prefilled, and one while it decodes.
#
# The server must already be up, writing its log to SERVER_LOG. The script:
#   1. sends a warmup wave of the same shape, untraced, so every graph the
#      captures see is already compiled;
#   2. sends a second wave with a new seed, so no prompt is in the prefix cache;
#   3. starts the prefill capture at the first "Prefill batch" log line of
#      that wave;
#   4. starts the decode capture at the first "Decode batch" line after it.
#      While a waiting request still fits, the scheduler runs prefill first,
#      so by then every request that fits is decoding. This may be fewer than
#      CONCURRENCY; the line it starts at is printed.
# Each capture is one POST /start_profile. The server stops it by itself
# after the given number of forward steps.
#
# Env (defaults shown):
#   MODEL=                  # required: the served checkpoint (for the tokenizer)
#   SERVER_LOG=             # required: the server's log file
#   OUTPUT_DIR=             # required: a new or empty directory
#   BASE_URL=http://127.0.0.1:30000
#   CONCURRENCY=64          # requests in each wave, all sent at once
#   ISL=8192 OSL=1024       # OSL must be well above 40 + DECODE_STEPS
#   PREFILL_STEPS=1
#   DECODE_STEPS=5
#   PYTHON=python3          # a Python that has sglang
set -uo pipefail

: "${MODEL:?set MODEL}" "${SERVER_LOG:?set SERVER_LOG}" "${OUTPUT_DIR:?set OUTPUT_DIR}"
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
CONCURRENCY=${CONCURRENCY:-64}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
PREFILL_STEPS=${PREFILL_STEPS:-1}
DECODE_STEPS=${DECODE_STEPS:-5}
PYTHON=${PYTHON:-python3}

# Sends CONCURRENCY requests at once and waits for all of them.
send_wave() {
    local log=$1 seed=$2
    "$PYTHON" -m sglang.bench_serving --backend sglang --base-url "$BASE_URL" \
        --model "$MODEL" --dataset-name random-ids --tokenize-prompt \
        --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0.8 \
        --num-prompts "$CONCURRENCY" --max-concurrency "$CONCURRENCY" \
        --warmup-requests 0 --request-rate inf --seed "$seed" > "$log" 2>&1
}

# Waits for a line of SERVER_LOG after line $1 that matches the regex $2,
# and prints that line's number. Gives up after $3 seconds.
wait_for_log() {
    local after=$1 pattern=$2 timeout_s=$3 start=$SECONDS n
    while (( SECONDS - start < timeout_s )); do
        n=$(tail -n +"$(( after + 1 ))" "$SERVER_LOG" | grep -n -m1 -E "$pattern" | cut -d: -f1)
        if [[ -n "$n" ]]; then
            echo $(( after + n ))
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Starts a capture of the next $2 forward steps, in $OUTPUT_DIR/$1-<id>-TP-<r>...
# No "activities": on TPU the server then traces the device and the host.
start_capture() {
    local phase=$1 steps=$2
    curl -sS -f -m 120 "$BASE_URL/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"$OUTPUT_DIR\", \"num_steps\": $steps, \"profile_prefix\": \"$phase\"}" \
        > /dev/null || { echo "$phase capture: /start_profile failed"; return 1; }
    echo "$phase capture: started for $steps steps"
}

if [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
    echo "refusing: $OUTPUT_DIR is not empty"
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

echo "warmup wave: $CONCURRENCY requests, untraced"
send_wave "$OUTPUT_DIR/warmup_wave.log" 1 || { echo "warmup wave failed, see $OUTPUT_DIR/warmup_wave.log"; exit 1; }
mark=$(wc -l < "$SERVER_LOG")
echo "profiled wave: $CONCURRENCY requests"
send_wave "$OUTPUT_DIR/profiled_wave.log" 2 &
wave=$!
status=0
if line=$(wait_for_log "$mark" "Prefill batch" 600); then
    start_capture prefill "$PREFILL_STEPS" || status=1
else
    echo "the wave never reached prefill"
    status=1
    line=$mark
fi
if line=$(wait_for_log "$line" "Decode batch" 1800); then
    echo "decode starts at: $(sed -n "${line}p" "$SERVER_LOG" | grep -o -E '#running-req: [0-9]+')"
    start_capture decode "$DECODE_STEPS" || status=1
else
    echo "the wave never reached decode"
    status=1
fi
wait "$wave" || { echo "profiled wave failed, see $OUTPUT_DIR/profiled_wave.log"; status=1; }

echo "profiler lines in the server log (count over ranks, line):"
tail -n +"$(( mark + 1 ))" "$SERVER_LOG" \
    | grep -E "Profiling (starts|done)|Stop profiling|already in progress" \
    | sed -E 's/^\[[^]]*\] //; s/ \(with profile id: [^)]*\)//' | sort | uniq -c
# The TPU recipes set TORCH_LOGS=+recompiles, so a compile during the wave
# shows up here. Without it, this count is always 0.
echo "recompile lines during the profiled wave: $(tail -n +"$(( mark + 1 ))" "$SERVER_LOG" | grep -ci 'recompil')"

echo "captures:"
while read -r file; do
    rank=${file#"$OUTPUT_DIR"/}
    rank=${rank%%/*}
    note=""
    grep -q -a -F '/device:TPU:' "$file" || note="host only"
    if grep -q -a -F 'Trace Buffers Dropped' "$file"; then
        note="DROPPED EVENTS: the trace hit its size limit; take fewer steps"
        status=1
    fi
    printf '  %-48s %6s  %s\n' "$rank" "$(du -h "$file" | cut -f1)" "$note"
done < <(find "$OUTPUT_DIR" -name '*.xplane.pb' | sort)
exit "$status"
