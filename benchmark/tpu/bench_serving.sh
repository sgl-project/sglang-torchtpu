#!/usr/bin/env bash
# Model-agnostic bench_serving driver for an sglang torch_tpu server. Supports
# either a single run (default) or a sweep over (concurrency, isl, osl) tuples
# (set SWEEP). Set MODEL to match whatever ./launch_server.sh is serving.
#
# Example — Qwen3-32B, default single run (bs=64, isl=1024, osl=1024):
#   MODEL=Qwen/Qwen3-32B ./bench_serving.sh
#
# Example — 1k/1k workload used for TPU FP8 EP/RPA throughput runs:
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 NUM_PROMPTS=640 ./bench_serving.sh
#
# Example — exact 1k/1k c=64 for Qwen3-Coder-480B FP8:
#   MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
#   DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
#   BATCH_SIZE=64 NUM_PROMPTS=64 ISL=1024 OSL=1024 \
#   ./bench_serving.sh
#
# Example — Qwen3-32B concurrency/shape sweep:
#   MODEL=Qwen/Qwen3-32B SWEEP="16:8192:512 64:8192:512" ./bench_serving.sh
# Each SWEEP token is "concurrency:isl:osl". The server's max-running-requests
# must be >= the largest concurrency. Per-tuple num_prompts defaults to
# concurrency x PROMPTS_PER_CONCURRENCY (so each in-flight slot processes
# that many requests in the run).
#
# Prereqs: an sglang server is up at $BASE_URL (default http://127.0.0.1:30000)
# serving $MODEL. Start it with ./launch_server.sh. For torch profiler traces,
# launch with ENABLE_PROFILER_ENDPOINT=1 (sets SGLANG_TORCH_PROFILER_DIR).
# See ./README.md for the shared env contract.
#
# If the in-server torch.compile warmup does not yet cover all compile-shape
# buckets, run ./warmup.sh once between launch and bench (stopgap; see README).
#
# Overridable env vars (defaults shown):
#   MODEL=                          # REQUIRED — must match the served model
#   BASE_URL=http://127.0.0.1:30000
#   BACKEND=sglang
#   DATASET_NAME=random
#   TOKENIZE_PROMPT=0              # 1 sends exact input_ids; requires BACKEND=sglang
#   ISL=8192                        # single-run input length
#   OSL=1024                        # single-run output length
#   BATCH_SIZE=64                   # single-run max-concurrency
#   NUM_PROMPTS=                    # single-run total prompts (defaults
#                                   # to BATCH_SIZE * PROMPTS_PER_CONCURRENCY)
#   PROMPTS_PER_CONCURRENCY=10      # used to derive NUM_PROMPTS when unset
#                                   # (10x concurrency matches InferenceX)
#   WARMUP_REQUESTS=auto            # "auto" → 2x the leg's concurrency
#                                   # (matches InferenceX's --num-warmups
#                                   # 2*CONC). Set an explicit count, or 0
#                                   # to skip and rely on ./warmup.sh.
#   SEED=1
#   RANDOM_RANGE_RATIO=0.8          # lens uniform in [0.8*L, L]. Matches the
#                                   # InferenceX harness (and torchtpu-vllm)
#   PROFILE=0                       # 1 → torch profiler /start_profile capture
#   PROFILE_NUM_STEPS=5
#   PROFILE_OUTPUT_DIR=             # auto-generated when PROFILE=1
#   OUTPUT_DIR=./results/<model-slug>-tpu/<ts>
#                                   # JSONL summaries (one per sweep config) land here
#   SWEEP=                          # space-separated "c:isl:osl" tuples
#                                   # (overrides ISL/OSL/BATCH_SIZE/NUM_PROMPTS)
#   HEALTH_TIMEOUT=600              # seconds to wait for /health -> 200
#   DUMP_META=1                     # 1 → write <tag>.meta.json next to each
#                                   # <tag>.jsonl (repro record for the CI
#                                   # dashboard; pulls live /server_info); 0 skips
#   PYTHON=python3                  # interpreter with torch_tpu + sglang installed
#   JAX_BAZEL=                      # opt-in; only for a Bazel-built torch_tpu

set -euo pipefail

# Resolve this script's dir so we can find dump_meta.py regardless of CWD.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 1 (default) → write <tag>.meta.json next to each <tag>.jsonl (reproducibility
# record for the CI dashboard). Set 0 to skip.
DUMP_META=${DUMP_META:-1}

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-32B ./bench_serving.sh" >&2
    exit 2
fi
# Slug for the default results dir / profile prefix: lowercase, non-alnum → '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
BACKEND=${BACKEND:-sglang}
DATASET_NAME=${DATASET_NAME:-random}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-0}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
BATCH_SIZE=${BATCH_SIZE:-64}
PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY:-10}
NUM_PROMPTS=${NUM_PROMPTS:-}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-auto}
SEED=${SEED:-1}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
PROFILE=${PROFILE:-0}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-5}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-}
SWEEP=${SWEEP:-}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-./results/${MODEL_SLUG}-tpu/$TS}
mkdir -p "$OUTPUT_DIR"

# A torch_tpu built from source with Bazel keeps jax/jaxlib/ml_dtypes/
# opt_einsum/scipy in its Bazel external repo. If you built that way, point
# JAX_BAZEL at <bazel-output-user-root>/<hash>/external so `python -m
# sglang.bench_serving` can import them. Pip-installed torch_tpu does not
# need this — leave JAX_BAZEL unset. (See ./launch_server.sh / ./README.md.)
JAX_BAZEL=${JAX_BAZEL:-}
if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# /health returns 503 while sglang is still warming up compile-buckets,
# then flips to 200 once the scheduler accepts requests. `curl -fsS` would
# fail-fast on 503, so we poll: re-issue every 5 s until we either see
# 200 or exhaust HEALTH_TIMEOUT seconds.
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

# Single-config runner; the sweep loop calls this once per tuple. Echoes
# the path of the JSONL summary it produced.
run_one() {
    local bs="$1"
    local isl="$2"
    local osl="$3"
    local num_prompts="$4"

    local tag="c${bs}-isl${isl}-osl${osl}"
    local output_file="$OUTPUT_DIR/${tag}.jsonl"
    local profile_prefix="${MODEL_SLUG}-${tag}"

    # Resolve "auto" → 2x concurrency (InferenceX parity) so the warmup
    # saturates the measurement's batch bucket before timing starts.
    local warmup_n="$WARMUP_REQUESTS"
    [[ "$warmup_n" == "auto" ]] && warmup_n=$((2 * bs))

    local args=(
        -m sglang.bench_serving
        --backend "$BACKEND"
        --base-url "$BASE_URL"
        --model "$MODEL"
        --dataset-name "$DATASET_NAME"
        --random-input-len "$isl"
        --random-output-len "$osl"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts "$num_prompts"
        --max-concurrency "$bs"
        --warmup-requests "$warmup_n"
        --seed "$SEED"
        --output-file "$output_file"
    )
    [[ "$TOKENIZE_PROMPT" == "1" ]] && args+=(--tokenize-prompt)

    if [[ "$PROFILE" == "1" ]]; then
        local pdir="${PROFILE_OUTPUT_DIR:-$OUTPUT_DIR/profile-${tag}}"
        mkdir -p "$pdir"
        args+=(
            --profile
            --profile-activities CPU GPU
            --profile-num-steps "$PROFILE_NUM_STEPS"
            --profile-prefix "$profile_prefix"
            --profile-output-dir "$pdir"
        )
    fi

    echo "==== bench_serving config ($tag) ===="
    echo "  base_url        = $BASE_URL"
    echo "  model           = $MODEL"
    echo "  backend         = $BACKEND"
    echo "  dataset         = $DATASET_NAME"
    echo "  tokenize_prompt = $TOKENIZE_PROMPT"
    echo "  isl/osl         = $isl / $osl"
    echo "  concurrency     = $bs"
    echo "  num_prompts     = $num_prompts  (= $bs x $PROMPTS_PER_CONCURRENCY)"
    echo "  warmup_requests = $warmup_n  (input: $WARMUP_REQUESTS)"
    echo "  range_ratio     = $RANDOM_RANGE_RATIO"
    echo "  profile         = $PROFILE"
    echo "  output_file     = $output_file"
    echo "======================================="

    "$PYTHON" "${args[@]}"
    echo
    echo "→ summary appended to $output_file"

    # ===== reproducibility metadata dump =====
    # One <tag>.meta.json per config, next to the metrics <tag>.jsonl. Records
    # the workload/bench args + git + versions, and pulls the LIVE server config
    # from $BASE_URL/server_info so the record reflects what the server is
    # actually running (not a stale launch-side file). Never fatal.
    if [[ "$DUMP_META" == "1" ]]; then
        local meta_file="$OUTPUT_DIR/${tag}.meta.json"
        local bench_argv_json
        bench_argv_json=$("$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${args[@]}")
        "$PYTHON" "$SCRIPT_DIR/dump_meta.py" --kind bench --out "$meta_file" \
            --json bench_argv="$bench_argv_json" \
            --set model="$MODEL" \
            --set model_slug="$MODEL_SLUG" \
            --set base_url="$BASE_URL" \
            --set backend="$BACKEND" \
            --set dataset_name="$DATASET_NAME" \
            --set tokenize_prompt="$TOKENIZE_PROMPT" \
            --set isl="$isl" \
            --set osl="$osl" \
            --set concurrency="$bs" \
            --set num_prompts="$num_prompts" \
            --set prompts_per_concurrency="$PROMPTS_PER_CONCURRENCY" \
            --set warmup_requests="$warmup_n" \
            --set seed="$SEED" \
            --set random_range_ratio="$RANDOM_RANGE_RATIO" \
            --set profile="$PROFILE" \
            --set summary_file="${tag}.jsonl" \
            --server-info-url "$BASE_URL" \
            || echo "WARN: dump_meta failed for $tag; continuing" >&2
        echo "→ metadata written to $meta_file"
    fi
}

if [[ -n "$SWEEP" ]]; then
    echo "==== sweep over ${SWEEP} ===="
    echo "  output_dir = $OUTPUT_DIR"
    echo "============================="
    for tuple in $SWEEP; do
        IFS=':' read -r sw_bs sw_isl sw_osl <<<"$tuple"
        if [[ -z "$sw_bs" || -z "$sw_isl" || -z "$sw_osl" ]]; then
            echo "ERROR: malformed SWEEP token '$tuple' (expected c:isl:osl)" >&2
            exit 2
        fi
        sw_num_prompts=$(( sw_bs * PROMPTS_PER_CONCURRENCY ))
        run_one "$sw_bs" "$sw_isl" "$sw_osl" "$sw_num_prompts"
    done
    echo "==== sweep done — JSONL summaries in $OUTPUT_DIR ===="
    ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || true
else
    [[ -z "$NUM_PROMPTS" ]] && NUM_PROMPTS=$(( BATCH_SIZE * PROMPTS_PER_CONCURRENCY ))
    run_one "$BATCH_SIZE" "$ISL" "$OSL" "$NUM_PROMPTS"
fi
