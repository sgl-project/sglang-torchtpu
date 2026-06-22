#!/usr/bin/env bash
# run_eval driver for an sglang torch_tpu server.
#
# Example — Qwen3-32B MMLU evaluation:
#   MODEL=Qwen/Qwen3-32B EVAL_NAME=mmlu ./run_eval.sh
#
# Prereq: ./launch_server.sh is serving $MODEL at $BASE_URL.
#
# Overridable env vars (defaults shown):
#   MODEL=                          # REQUIRED; must match the served model
#   BASE_URL=http://127.0.0.1:30000
#   EVAL_NAME=mmlu                  # mmlu, gsm8k, gpqa, humaneval, math, etc.
#   NUM_EXAMPLES=                   # empty = full eval; set N for smoke
#   NUM_THREADS=4                   # concurrency
#   API=chat                        # chat or completion
#   OUTPUT_DIR=                     # defaults under ./results/
#   HEALTH_TIMEOUT=600
#   PYTHON=python3
#   JAX_BAZEL=                      # only for Bazel-built torch_tpu
#   EXTRA_ARGS=                     # appended to run_eval; word-split

set -euo pipefail

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL is required and must match the served model, e.g. MODEL=Qwen/Qwen3-30B-A3B ./run_eval.sh" >&2
    exit 2
fi
# Slug for the default results dir: lowercase, non-alnum → '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
EVAL_NAME=${EVAL_NAME:-mmlu}
NUM_EXAMPLES=${NUM_EXAMPLES:-}
NUM_THREADS=${NUM_THREADS:-4}
API=${API:-chat}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}
PYTHON=${PYTHON:-python3}
JAX_BAZEL=${JAX_BAZEL:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-./results/${MODEL_SLUG}-tpu/$TS}
mkdir -p "$OUTPUT_DIR"

if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Wait for /health to return 200.
echo "waiting up to ${HEALTH_TIMEOUT}s for $BASE_URL/health to return 200..."
__health_start=$SECONDS
while :; do
    __code=$(curl -sS -o /dev/null -m 10 -w "%{http_code}" "$BASE_URL/health" 2>/dev/null || echo "000")
    if [[ "$__code" == "200" ]]; then
        echo "  /health -> 200 after $((SECONDS - __health_start))s"
        break
    fi
    if (( SECONDS - __health_start >= HEALTH_TIMEOUT )); then
        echo "ERROR: $BASE_URL/health never reached 200 (last code=$__code, waited ${HEALTH_TIMEOUT}s)" >&2
        echo "Start it first with ./launch_server.sh, then rerun." >&2
        exit 1
    fi
    sleep 5
done

# Run evaluation.
args=(
    -m sglang.test.run_eval
    --base-url "$BASE_URL"
    --model "$MODEL"
    --eval-name "$EVAL_NAME"
    --num-threads "$NUM_THREADS"
    --api "$API"
)

if [[ -n "$NUM_EXAMPLES" ]]; then
    args+=(--num-examples "$NUM_EXAMPLES")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # Word-split intentionally — lets EXTRA_ARGS carry multi-token sglang
    # flags (e.g. "--temperature 0.5 --max-tokens 1024") from a single env var.
    args+=($EXTRA_ARGS)
fi

echo "==== run_eval config ===="
echo "  base_url    = $BASE_URL"
echo "  model       = $MODEL"
echo "  eval_name   = $EVAL_NAME"
echo "  num_threads = $NUM_THREADS"
echo "  api         = $API"
if [[ -n "$NUM_EXAMPLES" ]]; then
    echo "  num_examples= $NUM_EXAMPLES"
fi
echo "  output_dir  = $OUTPUT_DIR"
echo "========================="

"$PYTHON" "${args[@]}"

# run_eval.py dumps reports to /tmp/
# Let's copy them to the output directory.
file_stem="${EVAL_NAME}_$(printf '%s' "$MODEL" | tr '/' '_')"
tmp_html="/tmp/${file_stem}.html"
tmp_json="/tmp/${file_stem}.json"

if [[ -f "$tmp_html" ]]; then
    mv "$tmp_html" "$OUTPUT_DIR/"
    echo "→ HTML report moved to $OUTPUT_DIR/${file_stem}.html"
fi

if [[ -f "$tmp_json" ]]; then
    mv "$tmp_json" "$OUTPUT_DIR/"
    echo "→ JSON results moved to $OUTPUT_DIR/${file_stem}.json"
fi
