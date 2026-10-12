#!/usr/bin/env bash
# CI launch group: gpt_oss_120b, recipe pd_1p1d_pc. Thin wrapper so the
# JobSet has a distinct launch_script per (model, recipe) group; all config
# lives in config.sh, dispatch in serve_group.sh.
exec env BENCH_MODEL=gpt_oss_120b BENCH_RECIPE=pd_1p1d_pc bash "$(dirname "$0")/serve_group.sh"
