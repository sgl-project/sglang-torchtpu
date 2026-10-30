#!/usr/bin/env bash
# CI launch group: gpt_oss_120b, recipe 1host_pc_spec_kvoff. Thin wrapper so the
# JobSet has a distinct launch_script per (model, recipe) group; all config
# lives in config.sh, dispatch in serve_group.sh.
set -euo pipefail
exec env BENCH_MODEL=gpt_oss_120b BENCH_RECIPE=1host_pc_spec_kvoff bash "$(dirname "$0")/serve_group.sh"
