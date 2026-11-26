"""DeepSeek-V4.1-Flash + DSpark served for AgentX on one tpu7x-8 host (DP8/EP8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c32.py

The AgentX sizing the canonical harness has applied on top of
deepseek_v41_flash_dspark.py (combined-1 canonical: c8 7,314 tok/s/chip @ p90
interactivity 40.9, c32 24,140 @ 30.0). It is the base of the per-concurrency
recipes deepseek_v41_flash_dspark_c{4,8,16,32,64,128,256}.py, which override
only what their concurrency needs.

Global values are divided by attn_dp_size (8) per rank.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark import (
    CONFIG as DSPARK_CONFIG,
)

CONFIG = deepcopy(DSPARK_CONFIG)
CONFIG["server_args"].update(
    {
        # AgentX prompts reach ~1M tokens (avg ~140k).
        "context_length": 1048576,
        # Radix cache + encoder SWA bounded replay: most of each prompt repeats
        # the previous turn. Needs mixed chunk off and >= 384 prefill tokens
        # per rank.
        "disable_radix_cache": False,
        "enable_encoder_swa_bounded_replay": True,
        # 1024 prefill tokens per rank and step.
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        # 16 per rank. The per-concurrency recipes raise it at c >= 64.
        "max_running_requests": 128,
        "schedule_policy": "fcfs",
    }
)
# K512 experts, as measured for the per-concurrency recipes (the base DSpark recipe
# defaults to K32Z); each per-concurrency recipe picks its MoE dispatch.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_EXPERT_FORMAT"] = "k512"
# Draft MoE packed-FP4 down projection on gmm_v2's calculate_tiling instead of
# the tuned-table fallback; bit-identical output. AgentX c32: paired p90
# interactivity ratio 1.019 (95% CI 1.006-1.035), tok/s/chip +1.3%; c256
# neutral, +0.7% tok/s.
CONFIG["runtime_env"]["SGLANG_TPU_FP4_GMM2_CALC_TILING"] = "1"
# fused_moe_v2 combine: fold the routing scales into the combine kernel;
# bit-identical on TPU, faster per call. Engages where the combine kernel runs
# (t_local >= 256: every extend step and c256 verify).
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_EP_V2_COMBINE_FUSE_SCALES"] = "1"
# G-B-lite: the mHC boundary plus the next layer's wqkv_a and q/kv RMSNorm as one
# epilogue on single-tile seams of <= SGLANG_TPU_DSV41_MHC_GBLITE_MAX_ROWS (128)
# rows; larger seams take the shifted mHC kernel. c4: -0.41 ms per step; AgentX c32
# paired p90 interactivity ratio 1.039; GSM8K 0.919.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_MHC_QA_EPILOGUE"] = "1"
# T48/T96 target verify: consume the next attention's KV RoPE in this epilogue.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_MHC_QA_KV_ROPE"] = "1"
# DSpark Markov drafts split over the vocab across the attention-DP ranks (needs
# the vocab-parallel head). AgentX real-draft pairs: c4 paired p90 interactivity
# ratio 1.047 (95% CI 1.014-1.082), c32 tok/s/chip +2.8% at flat p90, c256 p90
# 1.009 (1.001-1.017) and +2.1% tok/s; GSM8K 0.907.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_DSPARK_MARKOV_SHARD"] = "1"
# Unified-cache radix insert walks with a key offset instead of slicing each
# node's key: tree-identical (fuzz-tested), cheaper on long sessions; AgentX c256
# product (tok/s x p90 interactivity) ~1.26.
CONFIG["runtime_env"]["SGLANG_OPT_RADIX_INSERT_KEY_OFFSET"] = "1"
# Lift the packed metadata views out of the per-layer path: numerics-identical,
# neutral at c4/c32, AgentX c256 paired p90 interactivity +2.1%.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_LIFT_PACKED_VIEWS"] = "1"


def with_max_running_requests(n: int) -> dict:
    """This config with a different global max_running_requests."""
    config = deepcopy(CONFIG)
    config["server_args"]["max_running_requests"] = n
    return config
