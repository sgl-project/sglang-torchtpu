#!/usr/bin/env bash
# Run lm-eval against a live sglang torch_tpu server and check the baselines in
# an lm_eval_configs YAML file.
#
# Example — Qwen3-Coder-30B FP8 TP8/EP8 MMLU:
#   ./run_lm_eval.sh
#
# Prereq: ./launch_server.sh is serving the config's model at $BASE_URL.
#
# Overridable env vars:
#   EVAL_CONFIG=../../test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml
#   MODEL=                          # optional; overrides config model_name
#   BASE_URL=http://127.0.0.1:30000
#   OUTPUT_DIR=./results/<config-name>/<ts>
#   HEALTH_TIMEOUT=600
#   PYTHON=python3
#   JAX_BAZEL=                      # only for Bazel-built torch_tpu
#   SKIP_BASELINE_CHECK=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$SCRIPT_DIR/../../test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml"

EVAL_CONFIG=${EVAL_CONFIG:-$DEFAULT_CONFIG}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}
PYTHON=${PYTHON:-python3}
JAX_BAZEL=${JAX_BAZEL:-}
SKIP_BASELINE_CHECK=${SKIP_BASELINE_CHECK:-0}

CONFIG_SLUG=$(basename "$EVAL_CONFIG" .yaml | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/results/$CONFIG_SLUG/$TS}

if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

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
        exit 1
    fi
    sleep 5
done

args=(
    "$SCRIPT_DIR/run_lm_eval.py"
    --config "$EVAL_CONFIG"
    --base-url "$BASE_URL"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "${MODEL:-}" ]]; then
    args+=(--model "$MODEL")
fi

if [[ "$SKIP_BASELINE_CHECK" == "1" ]]; then
    args+=(--skip-baseline-check)
fi

echo "==== lm-eval config ===="
echo "  config     = $EVAL_CONFIG"
echo "  base_url   = $BASE_URL"
echo "  output_dir = $OUTPUT_DIR"
echo "========================"

"$PYTHON" "${args[@]}"
