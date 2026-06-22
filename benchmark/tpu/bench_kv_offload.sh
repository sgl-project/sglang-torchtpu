#!/usr/bin/env bash
# KV-offload (HiCache --hicache-io-backend pallas_tpu) reload benchmark, built on
# top of ./bench_serving.sh. It reuses bench_serving's metrics/JSONL and just
# orchestrates the workload so that a real host->HBM reload is exercised.
#
# WHY A WRAPPER: a plain bench_serving run with --dataset-name random sends fresh
# prompts -> no prefix reuse -> no cache hit -> that's the *recompute* (offload
# OFF) case, never a reload. A reload only fires when a prompt was seen before,
# written through to the host pool, evicted from the device pool, and then
# re-requested. Two facts make that reproducible on top of bench_serving:
#   * bench_serving re-seeds RNGs at startup, so --dataset-name random-ids
#     --seed S yields the SAME prompts on every invocation;
#   * random-ids (--tokenize-prompt) sends exact input_ids -> guaranteed radix
#     match across runs.
# So we run the same seed twice: pass 1 populates + evicts to host, pass 2 is the
# measured reload. A different seed = fresh prompts = recompute (offload OFF).
#
# THE SIZING (the whole game): the working set must straddle the two pools ::
#       device pool ( --max-total-tokens )  <  NUM_PROMPTS*ISL  <  host pool
#                                                              ( hicache-ratio * device )
#   below the band -> everything fits device -> device hit, no reload
#   above the band -> host pool thrashes too -> recompute, no reload
#   inside it      -> device evicts, host keeps -> reload on the re-run
# This script aims the working set at ~1.9x the device pool by default.
#
# PREREQ: a server up at $BASE_URL with offload enabled, e.g.
#   TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false \
#   python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --device tpu \
#     --tp-size 8 --ep-size 8 --enable-torch-compile --page-size 128 \
#     --attention-backend pallas_tpu_rpa_v3 --mem-fraction-static 0.30 \
#     --enable-hierarchical-cache --hicache-io-backend pallas_tpu --hicache-ratio 3 \
#     --max-total-tokens 16384 --enable-cache-report --port 30033
#
#   # then (DEVICE_TOKENS/HOST_RATIO must match the launch flags):
#   MODEL=Qwen/Qwen3-30B-A3B BASE_URL=http://127.0.0.1:30033 \
#   DEVICE_TOKENS=16384 HOST_RATIO=3 ./bench_kv_offload.sh
#
# Confirm a reload actually fired: the reload pass throughput should be far above
# the recompute pass, and the server log should show "#cached-token" > 0 on the
# reload pass (launch with --enable-cache-report).
#
# Overridable env (defaults shown):
#   MODEL=                     # REQUIRED, must match the served model
#   BASE_URL=http://127.0.0.1:30000
#   DEVICE_TOKENS=16384        # MUST match server --max-total-tokens
#   HOST_RATIO=3               # MUST match server --hicache-ratio
#   ISL=1300                   # input length (tokens) per prompt
#   OSL=8                      # output length (small: keep the step prefill/reload-bound)
#   CONCURRENCY=12             # max in-flight (<= server --max-running-requests)
#   NUM_PROMPTS=               # default: WS_TOKENS/ISL (WS_TOKENS ~= 1.9*device)
#   WS_TOKENS=                 # default: 1.9 * DEVICE_TOKENS
#   SAMPLES=3                  # measured reload passes (per mode)
#   SEED=1234
#   AB=0                       # 1 -> A/B async vs serial reload (needs the dev
#                              #      /cache/kvoffload_serial toggle in
#                              #      load_to_device_layerwise)
#   PROFILE=0                  # forwarded to bench_serving.sh
#   OUTPUT_DIR=./results/kvoffload-<model-slug>-<ts>
#   PYTHON=python3
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL=<served model>, e.g. MODEL=Qwen/Qwen3-30B-A3B ./bench_kv_offload.sh" >&2
    exit 2
fi
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
DEVICE_TOKENS=${DEVICE_TOKENS:-16384}
HOST_RATIO=${HOST_RATIO:-3}
ISL=${ISL:-1300}
OSL=${OSL:-8}
CONCURRENCY=${CONCURRENCY:-12}
SAMPLES=${SAMPLES:-3}
SEED=${SEED:-1234}
AB=${AB:-0}
PROFILE=${PROFILE:-0}
SERIAL_FLAG=/cache/kvoffload_serial

HOST_TOKENS=$(( DEVICE_TOKENS * HOST_RATIO ))
WS_TOKENS=${WS_TOKENS:-$(( DEVICE_TOKENS * 19 / 10 ))}
NUM_PROMPTS=${NUM_PROMPTS:-$(( WS_TOKENS / ISL ))}
WS=$(( NUM_PROMPTS * ISL ))

MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
TS=$(date +%Y%m%d_%H%M%S)
OUT_BASE=${OUTPUT_DIR:-./results/kvoffload-${MODEL_SLUG}-tpu/$TS}
mkdir -p "$OUT_BASE"

echo "==== kv-offload reload benchmark ===="
echo "  model         = $MODEL"
echo "  base_url      = $BASE_URL"
echo "  device pool   = $DEVICE_TOKENS tok    host pool = $HOST_TOKENS tok (ratio $HOST_RATIO)"
echo "  working set   = $NUM_PROMPTS prompts x $ISL = $WS tok"
echo "  band check    = device($DEVICE_TOKENS) < ws($WS) < host($HOST_TOKENS) ?"
if (( WS <= DEVICE_TOKENS )); then
    echo "  !! WARNING: working set <= device pool => device-hit, NO reload. Raise NUM_PROMPTS/ISL." >&2
fi
if (( WS >= HOST_TOKENS )); then
    echo "  !! WARNING: working set >= host pool => host thrash => recompute, NO reload. Lower NUM_PROMPTS or raise hicache-ratio." >&2
fi
echo "  concurrency   = $CONCURRENCY   isl/osl = $ISL/$OSL   samples = $SAMPLES   ab = $AB"
echo "  output_dir    = $OUT_BASE"
echo "====================================="

# One bench_serving.sh run = one pass over the working set.
pass() {
    local tag="$1" seed="$2"
    echo
    echo ">>> pass: $tag (seed=$seed)"
    MODEL="$MODEL" BASE_URL="$BASE_URL" \
    DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
    ISL="$ISL" OSL="$OSL" BATCH_SIZE="$CONCURRENCY" NUM_PROMPTS="$NUM_PROMPTS" \
    SEED="$seed" WARMUP_REQUESTS=0 PROFILE="$PROFILE" \
    OUTPUT_DIR="$OUT_BASE/$tag" \
    bash "$HERE/bench_serving.sh"
}

# Pass 1: populate the cache (cold prompts -> recompute + write-through to host,
# device LRU-evicts the overflow). Not a measurement.
pass populate "$SEED"

# Measured reload passes. Because WS > device, every subsequent same-seed pass
# re-evicts and re-reloads, so each pass is a genuine reload.
for k in $(seq 1 "$SAMPLES"); do
    if [[ "$AB" == "1" ]]; then
        rm -f "$SERIAL_FLAG";  pass "reload-async-$k"  "$SEED"
        touch "$SERIAL_FLAG";  pass "reload-serial-$k" "$SEED"
        rm -f "$SERIAL_FLAG"
    else
        pass "reload-$k" "$SEED"
    fi
done

# Recompute baseline (offload OFF): fresh seed => never-seen prompts => full prefill.
for k in $(seq 1 "$SAMPLES"); do
    pass "recompute-$k" "$(( SEED + 1000 + k ))"
done

echo
echo "==== done. JSONL summaries under $OUT_BASE ===="
echo "compare reload-* vs recompute-* throughput; check server log #cached-token>0 on reload passes."
find "$OUT_BASE" -name '*.jsonl' | sort | sed 's/^/  /'
