#!/usr/bin/env bash
# GPT-OSS-120B (MXFP4 MoE) on one tpu7x host — 4 chips x 2 cores = tp8.
#
# Sample usage:
#   ISL=1024 OSL=1024 CONCUR=64 ./server.sh      # sized for the 1k/1k sweep
#   ISL=8192 OSL=1024 CONCUR=64 ./server.sh      # sized for the 8k/1k sweep
#
# Pair with ./bench.sh (shared-prefix / prefix-cache-hit concurrency sweep).
# CONCUR is the LARGEST concurrency the sweep will drive: max-running-requests
# and torch_compile_max_bs are set from it, and the derived bs buckets are
# powers of two up to it, so one server covers c=1..CONCUR without a relaunch.
set -uo pipefail

ISL=${ISL:-1024}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-64}

export MODEL=${MODEL:-openai/gpt-oss-120b}
export TP_SIZE=8
# 128 experts / 8 = 16 local experts per rank. EP (not TP) for the experts:
# the TPU MXFP4 path dequantizes each expert to bf16 in gmm_v2's (E, K, N)
# layout, and TP-sharding intermediate_size=2880 across 8 ranks (360/rank)
# neither matches the MXFP4 32-element block nor the MXU's 128 lane tile.
export EP_SIZE=8

# The SparseCore ragged-gather MoE path (default ON) cannot be lowered for
# GPT-OSS: ragged_gather_v2's pltpu.async_copy indexer needs a static row count,
# and the EXTEND graph reaches it with a symbolic token dim, so the very first
# extend bucket dies with "The indexer shape in scatter/gather via
# `pltpu.async_copy` does not match the expected shape. Want: (16, <dynamic>)".
# Turning it off also matches the faster configuration measured on the
# Qwen3.5-397B A/B (decode TPOT -11.7% with the gather off; it only helps TTFT).
export SGLANG_TPU_FUSED_MOE_SC_GATHER=${SGLANG_TPU_FUSED_MOE_SC_GATHER:-0}

# GPT-OSS ships its own quantization_config (quant_method=mxfp4), so leave
# --quantization unset: forcing fp8 here would try to re-quantize MXFP4 weights.
export QUANTIZATION=
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}

# Only pallas_tpu_rpa_v3 implements attention sinks, which GPT-OSS needs
# (per-head learned sink logits); pallas_tpu_rpa_batched declares
# supports_attention_sink=False and would raise. GPT-OSS is also hybrid-SWA
# (sliding_window=128 on every other layer) — the RPA base handles the
# full->swa index mapping.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}

# 64 is the TPU platform's own default page size. It also sets the granularity
# of a radix-cache prefix match (the match length is floored to a page
# multiple), which is what makes a ~95%-hit workload reachable at ISL=1024:
# 960/1024 = 93.75% at page 64, vs a 896/1024 = 87.5% ceiling at page 128.
export PAGE_SIZE=${PAGE_SIZE:-64}

# Prefix caching ON. launch_server.sh defaults this to 1 (InferenceX parity for
# random-prompt throughput runs); a cache-hit workload needs the radix tree.
export DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}

export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 64))}
# Prefer one cold prefill of a full-ISL prompt per chunk; floor at 2048 so
# several cache-hit extends (~5% of ISL each) still batch together. Capped at
# 8192: the EXTEND graph's HBM temporaries scale with the chunk, and past ~8k
# the precompile OOMs (a 129k single-chunk prefill needs >94 GB). Above the cap
# a cold prefill is simply chunked, which costs nothing measurable here — the
# ~5% uncached remainder of a cache-hit request still lands in one chunk for
# any ISL up to ~160k.
CHUNK=${CHUNK:-$(( ISL > 2048 ? (ISL < 8192 ? ISL : 8192) : 2048 ))}
export CHUNKED_PREFILL_SIZE=$CHUNK
export MAX_PREFILL_TOKENS=$CHUNK

export MAX_RUNNING=$CONCUR
export TORCH_COMPILE_MAX_BS=$CONCUR
# 2x the in-flight requirement: the extra half is what the radix cache gets to
# retain shared prefixes with instead of evicting them between requests.
#
# BUT this default only holds at short context. KV costs ~15.2 KB per token per
# rank here (measured across the 1k/8k/128k runs) — the fp8 pool has KV head_dim
# 128, and hybrid-SWA allocates a *second* pool on top at swa_full_tokens_ratio.
# So the pool competes with the EXTEND graph's HBM temporaries, and past roughly
# 1.5M tokens the extend precompile dies with
#   "ran out of memory on HBM, the total memory required for HLO temporaries
#    (97.23G) exceeds available HBM (94.75G)"
# Keep `avail mem` at "Capture target decode TPU graph begin" >= ~45 GB. At
# CONTEXT_LEN=131072 that means overriding this to ~10 x CONTEXT_LEN (1310720),
# not the 2x16=2.1M this formula would give. A cache hit does not add to the
# requirement: the shared prefix *is* part of the hitting sequence's KV, so
# CONCUR full-length sequences already covers every resident prefix.
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$(( 2 * CONCUR * CONTEXT_LEN ))}
# MXFP4 MoE weights are dequantized to bf16 at load (~233 GB total, ~29 GB per
# chip of the 101.7 GB HBM), so 0.6 leaves a ~32 GB/chip KV budget — far more
# than MAX_TOTAL_TOKENS needs — plus ~40 GB/chip of headroom for the
# torch.compile EXTEND-graph transient.
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
