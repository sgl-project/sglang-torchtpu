#!/usr/bin/env bash
# 1P1D deployments used by the P/D stress reports (TPU v7x 2x2x1 per engine, TP8).
#
#   launch_pd.sh PROFILE prefill            # on the prefill host
#   launch_pd.sh PROFILE decode             # on the decode host
#   launch_pd.sh PROFILE router P_HOST D_HOST [mini|rust] [PORT]
#
# PROFILE:
#   gptoss         GPT-OSS-120B, production-sized pools (1M tokens/rank, HiCache ratio 3)
#   qwen3vl        Qwen3-VL-8B, production-sized pools (64K tokens)
#   qwen3vl-tight  Qwen3-VL-8B, tiny pools (4K tokens, 256-token chunks, strict KV checks)
#
# Extra server flags can be appended with EXTRA_ARGS (later flags win). For the
# GPT-OSS profile, MAX_TOTAL_TOKENS=auto drops --max-total-tokens (auto-sized pool)
# and HICACHE_RATIO overrides the prefill --hicache-ratio.
#
# Qwen3-VL on current main needs qwen3vl_moe_flags_workaround.patch to start with
# --enable-torch-compile. Set SGLANG_HOST_IP to an address the peer engine and the
# router can reach (defaults to the first address of this host).
set -euo pipefail

PROFILE=${1:?profile}
ROLE=${2:?role}

if [[ $ROLE == router ]]; then
  P=${3:?prefill host} D=${4:?decode host} KIND=${5:-mini} PORT=${6:-8000}
  args=(--pd-disaggregation --prefill "http://$P:30000" 8998 --decode "http://$D:30000" --host 0.0.0.0 --port "$PORT")
  [[ $KIND == mini ]] && args=(--mini-lb "${args[@]}")
  exec python3 -m sglang_router.launch_router "${args[@]}"
fi

export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}
export SGLANG_TPU_ALLOW_SPEC_RECOMPILE=1
export TORCH_TPU_INTERNAL_XLA_OPTIONS=${TORCH_TPU_INTERNAL_XLA_OPTIONS:-xla_tpu_use_dynamic_smem_negotiation=true}

COMMON=(--dtype bfloat16 --device tpu --attention-backend pallas_tpu_rpa_v3 --tp-size 8
  --grammar-backend none --host 0.0.0.0 --port 30000 --page-size 64 --kv-cache-dtype bfloat16
  --enable-torch-compile --enable-metrics --enable-cache-report --sampling-backend pytorch
  --skip-server-warmup --watchdog-timeout 7200 --soft-watchdog-timeout 600 --dist-timeout 3600
  --random-seed 42 --speculative-algorithm EAGLE3 --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --enable-priority-scheduling --default-priority-value 0 --priority-scheduling-preemption-threshold 10)
PREFILL_EXTRA=()
DECODE_EXTRA=()

case $PROFILE in
  gptoss)
    export SGLANG_TPU_FUSED_MOE_SC_GATHER=0
    COMMON+=(--model-path openai/gpt-oss-120b --served-model-name gpt-oss-120b --ep-size 8
      --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --reasoning-parser gpt-oss
      --max-running-requests 32 --chunked-prefill-size 4096 --max-prefill-tokens 4096
      --tpu-compile-tokens 512 1024 2048 4096 --tpu-compile-bs 1 2 4 8 16 32 --torch-compile-max-bs 32
      --mem-fraction-static 0.6 --context-length 16384)
    [[ ${MAX_TOTAL_TOKENS:-1048576} == auto ]] || COMMON+=(--max-total-tokens "${MAX_TOTAL_TOKENS:-1048576}")
    DEFAULT_HICACHE_RATIO=3 ;;
  qwen3vl | qwen3vl-tight)
    COMMON+=(--model-path Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b --ep-size 1
      --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --quantization unquant
      --trust-remote-code --mm-process-config '{"image":{"min_pixels":65536,"max_pixels":313600}}'
      --max-running-requests 16 --tpu-compile-bs 1 2 4 8 16 --torch-compile-max-bs 16
      --mem-fraction-static 0.3)
    DEFAULT_HICACHE_RATIO=4
    if [[ $PROFILE == qwen3vl ]]; then
      COMMON+=(--chunked-prefill-size 4096 --max-prefill-tokens 4096
        --tpu-compile-tokens 512 1024 2048 4096 --context-length 9216 --max-total-tokens 65536)
    else
      export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1 SGLANG_CHECK_KV_PAGE_INVARIANTS=1
      COMMON+=(--chunked-prefill-size 256 --max-prefill-tokens 256
        --tpu-compile-tokens 64 128 256 --context-length 4096 --max-total-tokens 4096)
      PREFILL_EXTRA+=(--radix-eviction-policy priority)
      [[ $ROLE == decode ]] && export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=128
    fi ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

case $ROLE in
  prefill)
    exec python3 -u -m sglang.launch_server "${COMMON[@]}" "${PREFILL_EXTRA[@]}" \
      --enable-hierarchical-cache --hicache-io-backend raiden --hicache-mem-layout page_first_fused \
      --hicache-ratio "${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}" \
      --disaggregation-mode prefill --disaggregation-transfer-backend raiden --disaggregation-bootstrap-port 8998 \
      ${EXTRA_ARGS:-} ;;
  decode)
    exec python3 -u -m sglang.launch_server "${COMMON[@]}" "${DECODE_EXTRA[@]}" --retraction-policy priority \
      --disaggregation-mode decode --disaggregation-transfer-backend raiden \
      --disaggregation-decode-retraction-backup cpu_tensor ${EXTRA_ARGS:-} ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac
