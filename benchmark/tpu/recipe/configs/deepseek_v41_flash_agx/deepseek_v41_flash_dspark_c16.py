"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 16 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c16.py

deepseek_v41_flash_dspark_agentx.py (128 running requests, 16 per rank) with
compile batch sizes from 1 and 2048-token prefill chunks per rank.

Why: at c16 a rank holds ~2 requests on average, so the inherited [8, 16, 32,
64] buckets padded most verify/draft steps to bs 8. AgentX canonical c16 on the
fused-EP tree, only tpu_compile_bs differs (j-04766568-bda9-4168-a3b9 vs
j-1805fb3a-fbc1-4bb7-8108): 17,873 -> 18,529 tok/s/chip, p90 interactivity
62.8 -> 67.0, TTFT p90 2,836 -> 2,538 ms, TPOT p50 12.11 -> 10.93 ms; paired
per-request interactivity 1.106 (95% CI 1.098-1.113), TTFT 0.947
(0.938-0.955). GSM8K with this bucket set: 0.912 / 0.906
(j-516f79b6-4fb7-4e75-9ba1). KV capacity unchanged; ~2 min more compile at boot.

Chunked prefill 2048 tokens per rank (16384 global, max_prefill_tokens to match;
the 2048 token bucket is already compiled), measured together with the buckets
(j-2995b11f-1e80-4f19-9eda): 19,008 tok/s/chip, p90 interactivity 66.1, TTFT
p90 1,749 ms; paired vs the buckets alone TTFT 0.783 (0.771-0.791),
interactivity 1.009 (1.005-1.018); vs the default recipe TTFT 0.737,
interactivity 1.131.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
CONFIG["server_args"].update(
    {
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64],
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
        # 16 decode rounds after each prefill step (the DSpark base runs 4): the
        # slowest users' decode is interrupted less often. AgentX c16 FAST, same
        # tree and image, only this differs (j-1241559f-f38a-471b-b2d4 vs
        # j-ee096c09-bf08-4a65-8ae9): p90 interactivity 84.3 -> 114.2, p90 TTFT
        # 1.6 -> 3.7 s, tok/s/chip flat (16,498 -> 16,540); paired per-request
        # interactivity 1.027 (95% CI 1.003-1.046), paired p90 ratio 1.357
        # (95% CI 1.280-1.445), TTFT 1.52. Policy: tok/s and p90 interactivity
        # decide, TTFT is reported only.
        # Raised to 24 by the product rule (tok/s x p90 interactivity): AgentX
        # c16 real-draft canonical, pdi 24 (j-794d1c25-81da-45ab-95ac) vs 16
        # (j-d5b8421b-6cc3-4ef0-8f1c): paired p90 ratio 1.083 (95% CI
        # 1.071-1.093), tok/s/chip 21,517 -> 21,479; product 1.081.
        "prefill_decode_interval": 24,
    }
)
# Pin each rank's scheduler process to its own CPUs. AgentX canonical c16 on the
# fused candidate, only this differs (j-854a9645-4110-4669-9b1b vs
# j-684a79db-2c0a-408d-86d3): paired p90 interactivity ratio 1.048 (95% CI
# 1.003-1.087), tok/s/chip 21,733 -> 21,907. Still off at c4 (neutral there).
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
# fused_moe_v2 plan kernel: engages at t_local 6/12/24/48, self-gates above.
# AgentX c16 fixed-mode, only this differs (j-1c6b06f9-3440-492a-bc7c vs
# j-b88c5e84-2ada-4a11-9a0f): paired p90 interactivity ratio 1.025 (95% CI
# 1.015-1.032), median 1.022.
# (plan kernel: superseded by push dispatch below, which turns it off)
# Shifted mHC row floor 1 (same basis as c4/c64; c16 itself not measured separately).
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS"] = "1"
# Eager-mode defer-and-fuse: AgentX c16 (j-7fb46912 vs sweep c16): p90 interactivity
# 1.031 (significant), tok/s +0.8%; product 1.040. GSM8K passed at c4.
CONFIG["runtime_env"]["SGLANG_TPU_EAGER_MODE"] = "DEFER_AND_FUSE"
# libtpu flags (as at c4/c256): XLA sorts on SparseCore,
# async all-to-all, no deduplicated calls. Replaces the inherited string: the
# flags are part of the compile-cache key. AgentX c16, only this differs
# (j-5ec851c8-0016-40a6-9975 vs j-ac7604f3-1b27-4b51-8632): paired p90
# interactivity ratio 1.071 (95% CI 1.063-1.079), median 1.081 (1.076-1.086),
# p90 TTFT 0.852, tok/s/chip 22,198 -> 23,279; product 1.123. No SparseCore halt.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_dvfs_p_state=7"
    " --xla_tpu_enable_offloading_sort_to_sparsecore=true"
    " --xla_tpu_enable_async_all_to_all=true"
    " --xla_tpu_enable_deduplicated_calls=false"
)

# Fused EP push dispatch (K512 push build, K512 dispatch_ahead 25)
# with the all-gather plan kernel off and push's in-kernel remote DMAs allowed.
# Adopted at every c: same-slot A/B at c4 (no loss: p90 0.995 n.s., median +2.2%) and c256 (merged tip
# 9e490d50b8, j-3838320f-fb7d-4db8-a5c0 vs j-0afca8ae-f063-4e6a-8ff2: p90
# interactivity 1.044 [1.040, 1.047], tok/s +0.8%, product ~1.052); in-between c by
# mechanism (device: t48 ~neutral, t96-512 0.81-0.96x all-gather time). GSM8K 0.903.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] += (
    " --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
)
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_DISPATCH"] = "push"
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_EP_V2_PLAN_KERNEL"] = "0"
# Extend SWA window plan sized to this rank's request slots instead of the fixed
# extend batch of 64 (exact; no-op at 64+ slots per rank, so not set at c256+).
# adopted on the mechanism (exact; extends plan fewer padded window rows). GSM8K 0.904 vs 0.903.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SWA_PLAN_TRIM"] = "1"
# Fused attention output projection (FP8 wo_a + wo_b, one Pallas kernel), by mechanism:
# it removes the BF16 wo_a stream (~0.4 ms per cycle), measured on at c4 (device
# 42.4 -> 24.7 us per layer, paired median interactivity 1.023) and c64-c256. The
# next full sweep re-measures c16.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
