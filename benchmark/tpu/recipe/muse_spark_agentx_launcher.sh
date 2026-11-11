#!/usr/bin/env bash
# Muse Spark AgentX: one engine on this tpu7x 2x2x1 host for one client concurrency.
#
#   bash benchmark/tpu/recipe/muse_spark_agentx_launcher.sh mixed 16             # one TP8 server
#   bash benchmark/tpu/recipe/muse_spark_agentx_launcher.sh prefill 16           # P/D: PP8 prefill
#   bash benchmark/tpu/recipe/muse_spark_agentx_launcher.sh decode 16            # P/D: TP8 decode
#   bash benchmark/tpu/recipe/muse_spark_agentx_launcher.sh decode 16 --dry-run  # print config only
#
# Everything else (weights, parallelism, envs) lives in configs/muse_spark/agentic.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # recipe/ -> repo root

usage="usage: $0 <mixed|prefill|decode> <concurrency> [--dry-run]"
MODE="${1:?$usage}"
CONCURRENCY="${2:?$usage}"
shift 2
exec python3 -m benchmark.tpu.recipe.configs.muse_spark.agentic "$MODE" "$CONCURRENCY" "$@"
