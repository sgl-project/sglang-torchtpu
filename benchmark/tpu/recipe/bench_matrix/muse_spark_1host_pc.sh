#!/usr/bin/env bash
# CI launch group: muse_spark, recipe 1host_pc. Thin wrapper so the
# JobSet has a distinct launch_script per (model, recipe) group; all config
# lives in config.sh, dispatch in serve_group.sh.
set -euo pipefail
exec env BENCH_MODEL=muse_spark BENCH_RECIPE=1host_pc bash "$(dirname "$0")/serve_group.sh"
