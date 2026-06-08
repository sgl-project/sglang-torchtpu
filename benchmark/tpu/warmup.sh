#!/usr/bin/env bash
# Standalone compile-bucket pre-warmup for an sglang torch_tpu server.
#
# STOPGAP — only run this while the in-server torch.compile warmup does not yet
# cover all compile-shape buckets. sglang's server-startup kernel_warmup covers
# decode-mode model.forward for the {16,32,64} token buckets, but the
# extend-mode prefill path (buckets {128..4096}) and the LP K-bucket grid stay
# cold; they would otherwise compile during the first measured requests of a
# benchmark and inflate their TTFT. This forces those compiles up front by
# sending one synthetic /generate per remaining bucket (see warmup_buckets.py).
#
# Once the server's own warmup covers these buckets, you DON'T need this — just
# run ./bench_serving.sh directly.
#
# Run AFTER ./launch_server.sh and BEFORE ./bench_serving.sh:
#   MODEL=Qwen/Qwen3-32B ./launch_server.sh        # shell 1 (blocks)
#   MODEL=Qwen/Qwen3-32B ./warmup.sh               # shell 2, once server is up
#   MODEL=Qwen/Qwen3-32B ./bench_serving.sh        # shell 2
#
# Overridable env vars (defaults shown):
#   MODEL=                          # REQUIRED — must match the served model
#   BASE_URL=http://127.0.0.1:30000
#   HEALTH_TIMEOUT=600              # seconds to wait for /health -> 200
#   OUTPUT_DIR=                     # optional dir to tee the warmup log into
#   PYTHON=python3                  # interpreter with torch_tpu + sglang installed
#   JAX_BAZEL=                      # opt-in; only for a Bazel-built torch_tpu

set -euo pipefail

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-32B ./warmup.sh" >&2
    exit 2
fi

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}
OUTPUT_DIR=${OUTPUT_DIR:-}

# A torch_tpu built from source with Bazel keeps jax/jaxlib/ml_dtypes/
# opt_einsum/scipy in its Bazel external repo. If you built that way, point
# JAX_BAZEL at <bazel-output-user-root>/<hash>/external. Pip-installed
# torch_tpu does not need this — leave JAX_BAZEL unset. (See ./README.md.)
JAX_BAZEL=${JAX_BAZEL:-}
if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# /health returns 503 while sglang is still warming up, then flips to 200 once
# the scheduler accepts requests. Poll until 200 or HEALTH_TIMEOUT.
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

PYTHON=${PYTHON:-python3}
WARMUP_SCRIPT="$(dirname "$0")/warmup_buckets.py"
if [[ ! -f "$WARMUP_SCRIPT" ]]; then
    echo "ERROR: $WARMUP_SCRIPT not found next to this script" >&2
    exit 1
fi

echo "==== TPU extend/LP-K bucket pre-warmup ===="
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    WARMUP_LOG="$OUTPUT_DIR/warmup_buckets.log"
    "$PYTHON" "$WARMUP_SCRIPT" "$BASE_URL" "$MODEL" 2>&1 | tee "$WARMUP_LOG"
    echo "==== bucket pre-warmup done (log: $WARMUP_LOG) ===="
else
    "$PYTHON" "$WARMUP_SCRIPT" "$BASE_URL" "$MODEL"
    echo "==== bucket pre-warmup done ===="
fi
