#!/usr/bin/env bash
# SINGLE-HOST Reflection Beam-1.0-501B-A23B-NVFP4 on 1x tpu7x-8 (2x2x1, 8 devices).
#   tp=8, ep=8, pure TP attention (48 q / 8 kv heads -> 6 q / 1 kv per rank).
#   ModelOpt NVFP4, routed experts only (packed E2M1, e4m3 block-16 scales);
#   attention, router, dense MLPs (layers 0-2) and lm_head stay BF16.
#   Weights ~294 GB -> ~41 GB/rank of ~101.7 GB HBM (FP8 checkpoint: ~63 GB/rank).
#
#   ./launch_beam_nvfp4.sh     # wait for "The server is fired up and ready to roll!"
#
# Checkpoint location, first match wins:
#   BEAM_LOCAL_MODEL=/mnt/disks/local/Beam-1.0-501B-A23B-NVFP4   # local-disk copy (preferred)
#   MODEL=/gcs/Beam-1.0-501B-A23B-NVFP4                          # gcsfuse mount (default)
#
# How the experts run (sglang_tpu/modelopt_fp4.py, sglang_tpu/fp4_moe.py):
#   W4A16 on the Pallas fused reduce-scatter MoE kernel. FP4 codes stay packed
#   in HBM (0.5 B/weight + bf16 block-16 scales with weight_scale_2 folded in),
#   dequantized to bf16 in VMEM; activations stay bf16 (the checkpoint's static
#   FP4 activation scales are not applied). Group-16 NVFP4 has no other TPU MoE
#   kernel, so the fused-RS backend is required: EP == TP == 8, no DP attention,
#   and no speculative decoding (fused-RS does not support it yet).
#
# Sizing vs launch_beam_fp8.sh (same 0.85 static fraction):
#   * ~22 GB/rank of weights freed -> KV pool ~40 GB/rank instead of ~17 GB.
#     KV per token per rank (bf16, 1 kv head): 4.5 KB over the 9 full layers,
#     21.5 KB over the 43 SWA layers. SWA_FULL_TOKENS_RATIO=0.1 (SWA pool =
#     10% of full-pool tokens; at 64 running requests the SWA pool needs only
#     64 x 512 + one chunk) puts ~5.8M tokens in the full-attention pool, vs
#     ~1.8M at the 0.8 default: long contexts and a 64-request batch fit together.
#   * MAX_RUNNING 64 (FP8 recipe: 32): decode is HBM-bound and a 64-batch
#     touches ~28 of each rank's 32 experts per layer, so a step reads ~35 GB
#     instead of ~27 GB at 32: ~1.5x the tokens per step for ~1.3x the bytes.
#   * CHUNKED_PREFILL_SIZE 4096: W4A16 runs the MXU in bf16, so the expert
#     GEMMs turn compute-bound above ~3K tokens per chunk (FP8: ~10K). At 4096
#     prefill matches the FP8 checkpoint on a roofline; at 8192 it would be ~1.7x
#     slower. Keep it at or below 4096.
#
# Override any knob via env, e.g.
#   CONTEXT_LEN=262144 MAX_RUNNING=32 ./launch_beam_nvfp4.sh
# For a weightless plumbing/compile smoke, add EXTRA_ARGS="--load-format dummy".
# Shared env (PYTHON, LIBTPU flags, ...) comes from ../../common_env.sh, sourced
# by launch_server.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> benchmark/tpu (launch_server.sh + common_env.sh)

if [[ -n "${BEAM_LOCAL_MODEL:-}" && -f "${BEAM_LOCAL_MODEL}/config.json" ]]; then
    MODEL="$BEAM_LOCAL_MODEL"
else
    MODEL="${MODEL:-/gcs/Beam-1.0-501B-A23B-NVFP4}"
fi

CHUNK="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING="${MAX_RUNNING:-64}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.1}"

# --weight-loader-disable-mmap: gcsfuse mmap SIGBUSes on large shards, and it
#   enables EP-local expert reads (each rank reads only its 32 of 256 experts:
#   ~35 GB of experts instead of all ~277 GB).
# QUANTIZATION is empty on purpose: the method (modelopt -> modelopt_fp4) comes
#   from the checkpoint's quantization_config; launch_server.sh would otherwise
#   pass --quantization fp8.
# Distributed top-k stays off for this recipe: with the fused-RS MoE backend
# its logits module halts a SparseCore at the 2048/4096-token prefill buckets
# (libtpu 0.0.50); see _DISTRIBUTED_TOP_K_SPARSE_CORE_OVERRIDES in
# sglang_tpu/env_override.py. FULL_VOCAB logits pass every bucket.
SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE="${SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE:-100000}" \
MODEL="$MODEL" \
TP_SIZE=8 EP_SIZE=8 \
QUANTIZATION="" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}" \
ATTENTION_BACKEND="${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}" \
PAGE_SIZE="${PAGE_SIZE:-128}" \
CONTEXT_LEN="${CONTEXT_LEN:-131072}" \
MAX_RUNNING="$MAX_RUNNING" \
TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}" \
CHUNKED_PREFILL_SIZE="$CHUNK" MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$CHUNK}" \
TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-64 128 256 512 1024 2048 4096}" \
TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2 4 8 16 32 64}" \
# Compile only the EXTEND bucket shapes proven correct on libtpu 0.0.50 with
# the SparseCore all-gather offload ON (base default): 256 wedges during
# warmup, 512/2048 return corrupt logits at run time, and with the offload
# OFF 2048/4096 halt (SparseCoreSequencer:0:0x1b). 64/128/1024/4096 are
# correct; 129-4096-token prefills pad up to the next bucket. Keep the
# offload ON here (opt out of the fused-RS override, which would turn it
# OFF and make 4096 unbootable); the offloads stay off by default for
# fused-RS everywhere else.
export SGLANG_TPU_FUSED_RS_SPARSE_CORE_OVERRIDE="${SGLANG_TPU_FUSED_RS_SPARSE_CORE_OVERRIDE:-0}"
# Keep the inductor compile cache (~6-9 GB per warmup) off container-local
# storage: pods get evicted for ephemeral-storage pressure around 73 GB
# written. Next to the checkpoint it is on NFS and survives pod restarts.
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$(dirname "$MODEL")/.sglang_cache_nvfp4}"
EXTRA_ARGS="--moe-runner-backend pallas_tpu_fused_rs --swa-full-tokens-ratio $SWA_FULL_TOKENS_RATIO --weight-loader-disable-mmap --weight-loader-drop-cache-after-load --tpu-compile-tokens 64 128 1024 4096 ${EXTRA_ARGS:-}" \
exec ./launch_server.sh "$@"
