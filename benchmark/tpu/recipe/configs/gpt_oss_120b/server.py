"""GPT-OSS-120B (MXFP4 MoE) on one tpu7x host, 4 chips x 2 cores = tp8.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/server.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 1024
# Largest concurrency the sweep drives: max_running_requests and
# torch_compile_max_bs are set from it so one server covers c=1..CONCURRENCY
# without a relaunch.
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64
# Floor 2048 so cache-hit extends still batch; cap 8192 because the EXTEND
# graph's HBM temporaries scale with the chunk and the precompile OOMs past ~8k.
PREFILL_TOKENS = min(max(INPUT_TOKENS, 2048), 8192)

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-120b",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tp_size": 8,
        # EP, not TP, for the 128 experts: the TPU MXFP4 path dequantizes each
        # expert to bf16 in gmm_v2's (E, K, N) layout, and TP-sharding
        # intermediate_size=2880 across 8 ranks matches neither the MXFP4
        # 32-element block nor the MXU's 128 lane tile.
        "ep_size": 8,
        # GPT-OSS ships its own quantization_config (mxfp4); forcing fp8 here
        # would try to re-quantize MXFP4 weights.
        "quantization": None,
        # Page 64 sets the radix-cache prefix-match granularity, making a
        # ~95%-hit workload reachable at ISL=1024 (960/1024 at page 64 vs an
        # 896/1024 ceiling at page 128).
        "page_size": 64,
        "disable_radix_cache": False,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        # 2x in-flight: the extra half lets the radix cache retain shared
        # prefixes between requests. Only holds at short context; KV costs
        # ~15.2 KB/token/rank (fp8, plus a second hybrid-SWA pool), so past
        # ~1.5M tokens the EXTEND precompile OOMs. At CONTEXT_LENGTH=131072
        # override to ~10x CONTEXT_LENGTH, and keep the "Capture target decode
        # TPU graph begin" avail-mem line >= ~45 GB.
        "max_total_tokens": 2 * CONCURRENCY * CONTEXT_LENGTH,
        # MXFP4 weights dequantize to ~233 GB bf16 (~29 GB/chip), so 0.6 leaves
        # ~32 GB/chip KV plus headroom for the EXTEND-graph transient.
        "mem_fraction_static": 0.6,
        "tpu_compile_tracer": "direct",
    }
)
# The SparseCore ragged-gather MoE path (default ON) cannot be lowered for
# GPT-OSS: ragged_gather_v2's pltpu.async_copy indexer needs a static row count
# and EXTEND reaches it with a symbolic token dim. Off also matched the faster
# config measured on the Qwen3.5-397B A/B (decode TPOT -11.7%).
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
