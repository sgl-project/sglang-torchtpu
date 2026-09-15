#!/usr/bin/env bash
# Sample usage: ISL=8192 OSL=1024 CONCUR=64 ./bench.sh
set -uo pipefail

export ISL=${ISL:-8192}
export OSL=${OSL:-1024}
# Workload name shown in the dashboard's Workload dropdown for this test's shape.
export WORKLOAD=${WORKLOAD:-8k1k}
CONCUR=${CONCUR:-64}
# 10 prompts per concurrency slot, matching the InferenceX benchmark client.
export NUM_PROMPTS=${NUM_PROMPTS:-$((CONCUR * 10))}

export MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
export DATASET_NAME=random-ids
export TOKENIZE_PROMPT=1
export BATCH_SIZE=$CONCUR

cd "$(dirname "$0")/../.."
exec ./bench_serving.sh "$@"
