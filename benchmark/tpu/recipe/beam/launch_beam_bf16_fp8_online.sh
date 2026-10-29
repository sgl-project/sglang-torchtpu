#!/usr/bin/env bash
# SINGLE-HOST Reflection Beam-1.0-501B-A23B (BF16 checkpoint) served as FP8 on
# 1x tpu7x-8 (2x2x1, 8 devices): online quantization at load, then exactly the
# launch_beam_fp8.sh configuration (tp8 / ep8, pure TP attention, bf16 KV).
#
# The official FP8_DYNAMIC quantization_config (fp8_dynamic_quantization_config.json:
# per-output-channel weight scales, dynamic per-token activations; lm_head,
# embed_tokens and the router mlp.gate stay BF16) is injected with
# --json-model-override-args. Beam's loader then quantizes each BF16 linear on
# the host as it streams, before it reaches HBM (1.0 TB of BF16 would not fit in
# 8 x 101.7 GB), reproducing llm-compressor bit for bit: the weights served are
# byte-identical to Beam-1.0-501B-A23B-FP8 (check_online_fp8_parity.py).
#
#   ./launch_beam_bf16_fp8_online.sh       # wait for "The server is fired up and ready to roll!"
#
# Storage: the BF16 checkpoint (1002.66 GB) does not fit in /dev/shm (~919 GB),
# so it streams from the gcsfuse mount (default MODEL=/gcs/Beam-1.0-501B-A23B).
# EP-local reads keep each rank to its 32 experts, so the host reads the
# checkpoint about once (~1.1 TB with the replicated non-expert tensors) and
# holds at most a few shards in RAM. For repeated restarts of a BF16-only
# derivative, quantize once offline instead (llm-compressor FP8_DYNAMIC) and
# serve that with launch_beam_fp8.sh from /dev/shm.
#
# Every launch_beam_fp8.sh knob (CONTEXT_LEN, MAX_RUNNING, KV_CACHE_DTYPE, ...)
# passes through.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

QUANT_CONFIG="$(tr -d '\n' < "$HERE/fp8_dynamic_quantization_config.json")"
unset BEAM_LOCAL_MODEL   # launch_beam_fp8.sh would prefer it over MODEL
export MODEL="${MODEL:-/gcs/Beam-1.0-501B-A23B}"
export JSON_MODEL_OVERRIDE="{\"quantization_config\": ${QUANT_CONFIG}}"
exec "$HERE/launch_beam_fp8.sh" "$@"
